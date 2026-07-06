const META_URL = import.meta.env.VITE_YOLORAG_DATASET_META_URL || "/api/dataset/meta";
const HIGHLIGHTS_URL =
  import.meta.env.VITE_YOLORAG_DATASET_HIGHLIGHTS_URL || "/api/dataset/highlights";

async function getJson(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function fetchDatasetMeta(ref, signal) {
  const url = `${META_URL}?ref=${encodeURIComponent(ref)}`;
  const payload = await getJson(url, signal);
  return payload.dataset;
}

export async function fetchDatasetHighlights({ ref, count = 4 }, signal) {
  const params = new URLSearchParams({ ref, count: String(count) });
  return getJson(`${HIGHLIGHTS_URL}?${params.toString()}`, signal);
}
