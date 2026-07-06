"""Read-only proxy to the Ultralytics platform dataset API.

The platform exposes public dataset metadata and images at:

    GET /api/datasets/{slug}?username={username}
    GET /api/datasets/{slug}/images?username={username}&split=&limit=&cursor=

We proxy those two endpoints server-side so the browser never has to reach
platform.ultralytics.com directly (avoids CORS) and so the caller only ever
supplies a ``username``/``slug`` pair — the upstream host is fixed here, which
keeps this from turning into an open proxy.

Routes are sync ``def`` on purpose: FastAPI runs them in a threadpool, so the
blocking ``urllib`` fetch never stalls the event loop. This mirrors the bench
router and the existing ``download_dataset_images`` script (both stdlib urllib),
so no runtime HTTP dependency is added.
"""
from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from yolorag.providers.base import LLMRequest, ProviderError
from yolorag.providers.factory import (
    default_describe_model,
    get_llm_provider,
    list_vision_providers,
)


router = APIRouter()

PLATFORM_BASE = "https://platform.ultralytics.com"
USER_AGENT = "yolorag-dataset-explorer/1.0"
FETCH_TIMEOUT_SECONDS = 20.0
# How many images to page through (with per-image labels) when picking the most
# label-diverse ones. Bounded so a page load stays responsive; if not every
# class has been seen yet, the scan extends up to _HARD_SCAN_CAP.
DEFAULT_SCAN = 400
_HARD_SCAN_CAP = 2500
_PAGE_SIZE = 100
# Platform slugs/usernames are lowercase slugs; keep this strict so a parsed
# ref can only ever produce a well-formed upstream URL.
_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Images to describe are fetched server-side. Restrict to the platform CDN so
# this can't be turned into an SSRF proxy for arbitrary hosts.
_ALLOWED_IMAGE_HOST_SUFFIXES = (".ul.run", ".ultralytics.com")
_ALLOWED_IMAGE_HOSTS = {"ul.run", "cdn.ul.run"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_DESCRIBE_IMAGES = 8
_EXT_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}

# Mosaic mode: instead of sending N images, letterbox each into an equal square
# cell and tile them into a near-square grid (2x2 for four), separated by a thin
# gap so the model still reads them as distinct sample panels. Sending one image
# is more compatible than many (some models 500 on multi-image requests).
_MOSAIC_CELL_PX = 512
_MOSAIC_GAP_PX = 6
_MOSAIC_GAP_COLOR = (32, 32, 32)  # BGR — dark separator between panels


def parse_dataset_ref(ref: str) -> tuple[str, str]:
    """Resolve a user-supplied reference into ``(username, slug)``.

    Accepts any of:
      * a full platform URL ``https://platform.ultralytics.com/ddxy/datasets/dogs-cats``
      * ``ddxy/datasets/dogs-cats``
      * ``ddxy/dogs-cats``
    """
    raw = (ref or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="A dataset URL or username/slug is required.")

    # If it looks like a URL, keep only the path so query strings/regions drop off.
    if "://" in raw or raw.startswith("platform.ultralytics.com"):
        parsed = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
        path = parsed.path
    else:
        path = raw

    parts = [segment for segment in path.split("/") if segment]
    # Drop the "datasets" marker if present: ["ddxy", "datasets", "dogs-cats"].
    parts = [segment for segment in parts if segment != "datasets"]

    if len(parts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Could not read username/slug. Use a full platform dataset URL "
            "or 'username/slug'.",
        )

    username, slug = parts[0], parts[-1]
    if not _SLUG_RE.match(username) or not _SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="Invalid username or dataset slug.")
    return username, slug


def _fetch_platform_json(path: str, params: dict[str, str]) -> dict:
    """GET a platform API path and return parsed JSON, surfacing upstream errors."""
    query = urllib.parse.urlencode(params)
    url = f"{PLATFORM_BASE}{path}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _upstream_error_detail(exc)
        # 401/403 from a private dataset is a client-side "not accessible", not a 5xx.
        status = exc.code if exc.code in {400, 401, 403, 404} else 502
        raise HTTPException(status_code=status, detail=detail) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not reach the Ultralytics platform ({exc})."
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Platform returned a non-JSON response.") from exc

    return payload


def _upstream_error_detail(exc: urllib.error.HTTPError) -> str:
    """Pull a human-readable message out of the platform's JSON error body."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
        message = body.get("error") or body.get("detail")
        if message:
            if exc.code in {401, 403}:
                return f"{message} — this dataset may be private."
            return str(message)
    except Exception:  # noqa: BLE001 - fall back to a generic message
        pass
    return f"Platform request failed ({exc.code})."


@router.get("/dataset/meta")
def dataset_meta(ref: str = Query(..., description="Platform dataset URL or username/slug")) -> dict:
    """Return metadata (and a few sample images) for a public platform dataset."""
    username, slug = parse_dataset_ref(ref)
    payload = _fetch_platform_json(f"/api/datasets/{slug}", {"username": username})
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=404, detail=f"No dataset found for {username}/{slug}.")

    dataset["platformUrl"] = f"{PLATFORM_BASE}/{username}/datasets/{slug}"
    return {"dataset": dataset}


def _image_class_ids(image: dict) -> set[int]:
    """The set of distinct label class ids present in one image."""
    return {
        label.get("classId")
        for label in (image.get("labels") or [])
        if label.get("classId") is not None
    }


def _select_diverse_images(images: list[dict], count: int) -> tuple[list[dict], set[int]]:
    """Pick ``count`` images that together convey as much of the dataset as possible.

    Greedy: at each step take the image that (1) introduces the most not-yet-
    covered classes, then prefers (2) images that actually have labels over blank
    ones, (3) whose rarest present class is least represented so far (balance),
    (4) with more distinct classes, then (5) more labels. So a cats+dogs dataset
    yields a balanced spread (e.g. 2 cats + 2 dogs), and once every class is
    covered the remaining slots still go to the most informative labeled images —
    never to unlabeled images while labeled ones remain.
    """
    remaining = list(images)
    selected: list[dict] = []
    covered: set[int] = set()
    selected_class_counts: Counter[int] = Counter()

    while remaining and len(selected) < count:
        def score(image: dict) -> tuple[float, ...]:
            ids = _image_class_ids(image)
            label_count = len(image.get("labels") or [])
            new_classes = len(ids - covered)
            # Rarest present class's current count. Empty images have no classes,
            # so give them the worst value here instead of the best (default=0).
            least_represented = min(
                (selected_class_counts[cid] for cid in ids), default=math.inf
            )
            return (
                new_classes,
                1 if label_count else 0,
                -least_represented,
                len(ids),
                label_count,
            )

        best = max(remaining, key=score)
        selected.append(best)
        best_ids = _image_class_ids(best)
        covered |= best_ids
        selected_class_counts.update(best_ids)
        remaining.remove(best)

    return selected, covered


@router.get("/dataset/highlights")
def dataset_highlights(
    ref: str = Query(..., description="Platform dataset URL or username/slug"),
    count: int = Query(default=4, ge=1, le=24),
) -> dict:
    """Return the ``count`` images that together cover the most distinct labels.

    Scans the dataset's images (with per-image labels), then greedily selects a
    small set spanning as many label classes as possible — so for a cats+dogs
    dataset the result surfaces both, even when no single image contains both.
    """
    username, slug = parse_dataset_ref(ref)

    scanned: list[dict] = []
    covered: set[int] = set()
    classes: list[str] = []
    total_classes: int | None = None
    cursor: str | None = None
    seen = 0

    while seen < _HARD_SCAN_CAP:
        params = {"username": username, "limit": str(_PAGE_SIZE), "includeLabels": "true"}
        if cursor:
            params["cursor"] = cursor
        page = _fetch_platform_json(f"/api/datasets/{slug}/images", params)

        classes = page.get("classes") or classes
        if total_classes is None:
            total_classes = len(classes) if classes else None

        page_images = page.get("images") or []
        for image in page_images:
            scanned.append(image)
            covered |= _image_class_ids(image)
        seen += len(page_images)
        cursor = page.get("nextCursor")

        all_covered = total_classes is not None and len(covered) >= total_classes
        # Stop once we've seen enough for a good pool, unless a class is still
        # missing — then keep paging (up to the hard cap) to try to include it.
        if seen >= DEFAULT_SCAN and all_covered:
            break
        if not page.get("hasMore") or not cursor or not page_images:
            break

    selected, selected_covered = _select_diverse_images(scanned, count)

    def class_name(class_id: int | None) -> str:
        if class_id is not None and 0 <= class_id < len(classes):
            return classes[class_id]
        return str(class_id)

    for image in selected:
        for label in image.get("labels") or []:
            label["className"] = class_name(label.get("classId"))
        image["classNames"] = sorted(class_name(cid) for cid in _image_class_ids(image))

    max_distinct = max((len(_image_class_ids(img)) for img in selected), default=0)

    return {
        "images": selected,
        "classes": classes,
        "coveredClasses": sorted(class_name(cid) for cid in selected_covered),
        "maxDistinctLabelsInImage": max_distinct,
        "scannedCount": seen,
        "requestedCount": count,
        "returnedCount": len(selected),
    }


# --- Describe: send the images + a prompt to a vision LLM ------------------


class BoxAnnotation(BaseModel):
    """One object box in YOLO-normalized [cx, cy, w, h] — matches the UI overlay."""

    bbox: list[float] = Field(default_factory=list)
    className: str | None = None
    classId: int | None = None


class DescribeImage(BaseModel):
    url: str
    boxes: list[BoxAnnotation] = Field(default_factory=list)


class DescribeBody(BaseModel):
    prompt: str
    provider: str = "gemini"
    model: str | None = None
    images: list[DescribeImage] = Field(default_factory=list)
    # When true (and >1 image), combine the images into a single 2x2-style
    # mosaic and send that one image instead of many.
    mosaic: bool = False

    @field_validator("images", mode="before")
    @classmethod
    def _coerce_images(cls, value: object) -> object:
        """Accept bare URL strings as a shorthand for ``{"url": ...}`` (no boxes)."""
        if isinstance(value, list):
            return [{"url": item} if isinstance(item, str) else item for item in value]
        return value


def _mime_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return _EXT_MIME.get(ext, "image/jpeg")


def _fetch_image_bytes(url: str) -> tuple[bytes, str]:
    """Download one image from an allowlisted host, returning ``(data, mime)``."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Image URLs must be http(s).")
    host = (parsed.hostname or "").lower()
    allowed = host in _ALLOWED_IMAGE_HOSTS or host.endswith(_ALLOWED_IMAGE_HOST_SUFFIXES)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Refusing to fetch image from host {host!r}.")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            # Read one byte past the cap so oversize images are detectable.
            data = response.read(_MAX_IMAGE_BYTES + 1)
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch image ({exc}).") from exc

    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds the size limit.")
    mime = content_type if content_type.startswith("image/") else _mime_from_url(url)
    return data, mime


def _to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# Box colors mirror the frontend CLASS_COLORS (indexed by classId), converted to
# cv2's BGR order, so burned-in boxes match what the UI overlays show.
_CLASS_COLORS_BGR = [
    (238, 211, 34),   # #22d3ee
    (182, 114, 244),  # #f472b6
    (47, 255, 215),   # #d7ff2f
    (255, 107, 47),   # #2f6bff
    (60, 146, 251),   # #fb923c
    (250, 139, 167),  # #a78bfa
]
_DEFAULT_BOX_COLOR_BGR = (170, 161, 161)  # #a1a1aa


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _class_color_bgr(class_id: int | None) -> tuple[int, int, int]:
    if class_id is None or class_id < 0:
        return _DEFAULT_BOX_COLOR_BGR
    return _CLASS_COLORS_BGR[class_id % len(_CLASS_COLORS_BGR)]


def _draw_boxes(image: np.ndarray, boxes: list[BoxAnnotation]) -> None:
    """Draw YOLO-normalized [cx,cy,w,h] boxes + class labels onto ``image`` (in place)."""
    height, width = image.shape[:2]
    thickness = max(2, round(min(height, width) / 240))
    for box in boxes:
        if not box.bbox or len(box.bbox) < 4:
            continue
        cx, cy, bw, bh = box.bbox[:4]
        x1 = _clamp(round((cx - bw / 2) * width), 0, width - 1)
        y1 = _clamp(round((cy - bh / 2) * height), 0, height - 1)
        x2 = _clamp(round((cx + bw / 2) * width), 0, width - 1)
        y2 = _clamp(round((cy + bh / 2) * height), 0, height - 1)
        color = _class_color_bgr(box.classId)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = box.className or (str(box.classId) if box.classId is not None else "")
        if label:
            _draw_box_label(image, label, x1, y1, color)


def _draw_box_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw ``text`` in a filled tag pinned to the top-left of a box."""
    height, width = image.shape[:2]
    scale = max(0.4, round(min(height, width) / 900, 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    tag_h = text_h + baseline + 6
    top = y - tag_h if y - tag_h >= 0 else y  # above the box, or inside if clipped
    x2 = _clamp(x + text_w + 6, 0, width - 1)
    y2 = _clamp(top + tag_h, 0, height - 1)
    cv2.rectangle(image, (x, top), (x2, y2), color, -1)
    cv2.putText(image, text, (x + 3, top + text_h + 3), font, scale, (20, 20, 20), 1, cv2.LINE_AA)


def _annotated_data_url(data: bytes, mime: str, boxes: list[BoxAnnotation]) -> str:
    """A ``data:`` URL for one image, with object boxes burned in when present."""
    if not boxes:
        return _to_data_url(data, mime)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return _to_data_url(data, mime)
    _draw_boxes(image, boxes)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return _to_data_url(encoded.tobytes(), "image/jpeg") if ok else _to_data_url(data, mime)


def _fit_into_cell(image: np.ndarray, cell: int) -> np.ndarray:
    """Resize ``image`` to fit a ``cell``x``cell`` square, letterboxed on black."""
    height, width = image.shape[:2]
    scale = min(cell / width, cell / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell, cell, 3), dtype=np.uint8)
    y0, x0 = (cell - new_h) // 2, (cell - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _compose_mosaic(
    images: list[tuple[bytes, str]],
    boxes_per: list[list[BoxAnnotation]] | None = None,
) -> bytes:
    """Tile images into a near-square grid (2x2 for four) as one JPEG.

    When ``boxes_per`` is given (aligned to ``images``), each image's object
    boxes are drawn in at full resolution before it is scaled into its cell.
    """
    cells: list[np.ndarray] = []
    for index, (data, _mime) in enumerate(images):
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        if boxes_per is not None and boxes_per[index]:
            _draw_boxes(decoded, boxes_per[index])
        cells.append(_fit_into_cell(decoded, _MOSAIC_CELL_PX))
    if not cells:
        raise HTTPException(status_code=502, detail="Could not decode images to build a mosaic.")

    cols = math.ceil(math.sqrt(len(cells)))
    rows = math.ceil(len(cells) / cols)
    cell, gap = _MOSAIC_CELL_PX, _MOSAIC_GAP_PX
    canvas = np.full(
        (rows * cell + (rows + 1) * gap, cols * cell + (cols + 1) * gap, 3),
        _MOSAIC_GAP_COLOR,
        dtype=np.uint8,
    )
    for index, tile in enumerate(cells):
        row, col = divmod(index, cols)
        y = gap + row * (cell + gap)
        x = gap + col * (cell + gap)
        canvas[y : y + cell, x : x + cell] = tile

    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=502, detail="Could not encode the mosaic image.")
    return encoded.tobytes()


def _mosaic_data_url(
    images: list[tuple[bytes, str]],
    boxes_per: list[list[BoxAnnotation]] | None = None,
) -> str:
    return _to_data_url(_compose_mosaic(images, boxes_per), "image/jpeg")


def _describe_messages(prompt: str, image_urls: list[str]) -> list[dict]:
    """One user turn — images first, then the instruction (OpenAI vision shape)."""
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@router.get("/dataset/describe/providers")
def dataset_describe_providers() -> dict:
    """List the vision providers + models available for description."""
    return {"providers": list_vision_providers()}


@router.post("/dataset/describe")
async def dataset_describe(body: DescribeBody) -> dict:
    """Send the given images + prompt to a vision LLM and return its description."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="A prompt is required.")
    if not body.images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(body.images) > _MAX_DESCRIBE_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"At most {_MAX_DESCRIBE_IMAGES} images can be described at once."
        )

    model = body.model or default_describe_model(body.provider)
    if not model:
        raise HTTPException(status_code=400, detail="A model is required for this provider.")

    # Downloads (and cv2 box drawing / mosaic compositing) are blocking; run off-loop.
    fetched = [await run_in_threadpool(_fetch_image_bytes, image.url) for image in body.images]
    boxes_per = [image.boxes for image in body.images]
    mosaic = body.mosaic and len(fetched) > 1
    if mosaic:
        image_urls = [await run_in_threadpool(_mosaic_data_url, fetched, boxes_per)]
    else:
        image_urls = [
            await run_in_threadpool(_annotated_data_url, data, mime, boxes)
            for (data, mime), boxes in zip(fetched, boxes_per)
        ]
    messages = _describe_messages(prompt, image_urls)

    try:
        provider = get_llm_provider(body.provider)
        result = await provider.complete(
            LLMRequest(messages=messages, model=model, temperature=0.4, max_tokens=640)
        )
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        # Unknown provider, or a missing/unset API key.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: dict = {
        "description": result.content,
        "provider": result.provider,
        "model": result.model,
        "imageCount": len(image_urls),
        "sourceImageCount": len(fetched),
        "mosaic": mosaic,
        "latencyMs": result.latency_ms,
    }
    usage = result.usage
    if usage is not None and usage.normalized_total():
        response["usage"] = {
            "inputTokens": usage.input_tokens,
            "outputTokens": usage.output_tokens,
            "totalTokens": usage.normalized_total(),
        }
        # Cost is best-effort: an unpriced model still returns a description.
        cost = result.cost
        if cost is not None and cost.unavailable_reason is None:
            response["costUsd"] = float(cost.total_usd)
        else:
            response["costUsd"] = None
            if cost is not None:
                response["costUnavailableReason"] = cost.unavailable_reason

    return response
