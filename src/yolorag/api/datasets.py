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

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from fastapi import APIRouter, HTTPException, Query


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
    """Pick ``count`` images that together span the most distinct label classes.

    Greedy set-cover: at each step take the image that introduces the most
    not-yet-covered classes. Ties break toward balance — an image whose classes
    are the least represented in the selection so far wins — then toward images
    with more distinct classes and more labels. So a cats+dogs dataset yields a
    balanced spread (e.g. 2 cats + 2 dogs) rather than covering both once and
    then piling onto whichever class happens to be richest.
    """
    remaining = list(images)
    selected: list[dict] = []
    covered: set[int] = set()
    selected_class_counts: Counter[int] = Counter()

    while remaining and len(selected) < count:
        def score(image: dict) -> tuple[int, int, int, int]:
            ids = _image_class_ids(image)
            new_classes = len(ids - covered)
            # Prefer images whose rarest class is under-represented so far.
            least_represented = min((selected_class_counts[cid] for cid in ids), default=0)
            return (new_classes, -least_represented, len(ids), len(image.get("labels") or []))

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
