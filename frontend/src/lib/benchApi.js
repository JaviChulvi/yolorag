const META_URL = import.meta.env.VITE_YOLORAG_BENCH_META_URL || "/api/benchmark/meta";
const RUN_URL = import.meta.env.VITE_YOLORAG_BENCH_URL || "/api/benchmark";

export async function fetchBenchmarkMeta(signal) {
  const response = await fetch(META_URL, { signal });
  if (!response.ok) {
    throw new Error(`Failed to load benchmark meta (${response.status}).`);
  }
  return response.json();
}

export async function runBenchmark(params, signal) {
  const response = await fetch(RUN_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  if (!response.ok) {
    let detail = `Benchmark request failed (${response.status}).`;
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
