const META_URL = import.meta.env.VITE_YOLORAG_DATASET_META_URL || "/api/dataset/meta";
const HIGHLIGHTS_URL =
  import.meta.env.VITE_YOLORAG_DATASET_HIGHLIGHTS_URL || "/api/dataset/highlights";
const DESCRIBE_URL =
  import.meta.env.VITE_YOLORAG_DATASET_DESCRIBE_URL || "/api/dataset/describe";
const DESCRIBE_PROVIDERS_URL =
  import.meta.env.VITE_YOLORAG_DATASET_DESCRIBE_PROVIDERS_URL || "/api/dataset/describe/providers";

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

export async function fetchDescribeProviders(signal) {
  const payload = await getJson(DESCRIBE_PROVIDERS_URL, signal);
  return payload.providers || [];
}

export async function describeDataset({ prompt, provider, model, images, mosaic }, signal) {
  const response = await fetch(DESCRIBE_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, provider, model, images, mosaic }),
    signal,
  });
  if (!response.ok) {
    let detail = `Describe request failed (${response.status}).`;
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
