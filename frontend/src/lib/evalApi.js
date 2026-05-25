import profileQuestions from "../../../evals/profile_questions.json";
import { DEFAULT_RUNTIME_SELECTION, config, runtimeUrl } from "./config.js";

const DEFAULT_BATCH_SIZE = 5;
const PROVIDERS = ["openai", "deepseek"];
const KNOWLEDGE_PROVIDERS = ["mongodb", "postgresql"];

export async function runFastEvals({
  signal,
  batchSize = DEFAULT_BATCH_SIZE,
  runtimeSelection = DEFAULT_RUNTIME_SELECTION,
  evalScope = "selected",
  onProgress,
} = {}) {
  const questions = profileQuestions.map((item, index) => ({
    id: String(item.id || `profile-${String(index + 1).padStart(3, "0")}`),
    question: item.question,
    tags: item.tags || [],
  }));
  const dataset = {
    name: "profile_questions.json",
    path: "evals/profile_questions.json",
    question_count: questions.length,
  };
  const runId = randomRunId();
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const combos = evalCombos(runtimeSelection, evalScope);
  const run = {
    id: runId,
    mode: "fast",
    endpoint: "/api/chat/fast",
    eval_scope: evalScope,
    combo_count: combos.length,
    started_at: startedAt,
    completed_at: null,
    duration_ms: 0,
    batch_size: batchSize,
  };
  const results = [];
  emitProgress();

  for (const combo of combos) {
    for (let index = 0; index < questions.length; index += batchSize) {
      const batch = questions.slice(index, index + batchSize);
      await Promise.all(
        batch.map((item, batchIndex) =>
          runFastEvalQuestion({
            item,
            combo,
            runId,
            userMessageIndex: index + batchIndex,
            signal,
          }).then((result) => {
            results.push(result);
            results.sort(
              (first, second) =>
                first.combo_order - second.combo_order ||
                first.user_message_index - second.user_message_index,
            );
            emitProgress();
            return result;
          }),
        ),
      );
    }
  }

  run.completed_at = new Date().toISOString();
  run.duration_ms = elapsedMs(started);
  const report = buildReport({
    dataset,
    run,
    combos,
    results,
    requestedCount: questions.length * combos.length,
  });
  onProgress?.(report);
  return report;

  function emitProgress() {
    run.duration_ms = elapsedMs(started);
    onProgress?.(
      buildReport({
        dataset,
        run,
        combos,
        results,
        requestedCount: questions.length * combos.length,
      }),
    );
  }
}

async function runFastEvalQuestion({ item, combo, runId, userMessageIndex, signal }) {
  const started = performance.now();
  let clientTtft = 0;
  let answer = "";
  let metrics = null;

  try {
    const response = await fetch(runtimeUrl(config.chatApiUrl, combo), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        session_id: `eval-fast-${runId}-${combo.id}-${item.id}`,
        messages: [{ role: "user", content: item.question }],
        analytics: false,
        include_metrics: true,
      }),
      signal,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `Eval request failed with ${response.status}`);
    }
    if (!response.body) {
      throw new Error("The browser did not expose a response stream.");
    }

    await readServerSentEvents(response.body, (event) => {
      if (event.error) throw new Error(event.error);
      if (event.type === "metrics") {
        metrics = event.metrics;
        return;
      }
      if (!event.content) return;
      if (!clientTtft) clientTtft = elapsedMs(started);
      answer += event.content;
    });

    if (!metrics) throw new Error("Fast chat did not return metrics.");
    return successResult({
      item,
      combo,
      metrics,
      answer,
      started,
      clientTtft,
      userMessageIndex,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    return errorResult(item, combo, error, started, userMessageIndex);
  }
}

async function readServerSentEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      parseFrame(frame, onEvent);
    }

    if (done) break;
  }

  if (buffer.trim()) parseFrame(buffer, onEvent);
}

function parseFrame(frame, onEvent) {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");

  if (!data || data === "[DONE]") return;

  let event;
  try {
    event = JSON.parse(data);
  } catch {
    event = { content: data };
  }
  onEvent(event);
}

function successResult({
  item,
  combo,
  metrics,
  answer,
  started,
  clientTtft,
  userMessageIndex,
}) {
  const timings = metrics.timings_ms || {};
  return {
    id: item.id,
    question: item.question,
    tags: item.tags,
    status: "ok",
    combo_id: combo.id,
    combo_label: combo.label,
    combo_order: combo.order,
    provider: metrics.provider || combo.provider,
    knowledge_provider: combo.knowledgeProvider,
    model: metrics.model,
    timings_ms: {
      total: numberValue(timings.total),
      retrieval: numberValue(timings.retrieval),
      query_embedding: numberValue(timings.query_embedding),
      vector_search: numberValue(timings.vector_search),
      rerank: numberValue(timings.rerank),
      llm: numberValue(timings.llm),
      ttft: numberValue(timings.ttft),
      llm_ttft: numberValue(timings.llm_ttft),
      orchestration_overhead: numberValue(timings.orchestration_overhead),
      wall: elapsedMs(started),
      client_ttft: clientTtft,
    },
    retrieval: metrics.retrieval || emptyRetrieval(),
    usage: metrics.usage || emptyUsage(),
    route_reason: metrics.route_reason || "",
    answer_chars: answer.length,
    user_message_index: userMessageIndex,
  };
}

function errorResult(item, combo, error, started, userMessageIndex) {
  const errorName = error?.name || "Error";
  const errorMessage = error?.message || String(error);
  return {
    id: item.id,
    question: item.question,
    tags: item.tags,
    status: "error",
    combo_id: combo.id,
    combo_label: combo.label,
    combo_order: combo.order,
    provider: combo.provider,
    knowledge_provider: combo.knowledgeProvider,
    timings_ms: {
      total: elapsedMs(started),
      retrieval: 0,
      query_embedding: 0,
      vector_search: 0,
      rerank: 0,
      llm: 0,
      ttft: 0,
      llm_ttft: 0,
      orchestration_overhead: 0,
      wall: elapsedMs(started),
      client_ttft: 0,
    },
    retrieval: emptyRetrieval(),
    usage: emptyUsage(),
    error: `${errorName}: ${errorMessage}`,
    user_message_index: userMessageIndex,
  };
}

function buildReport({ dataset, run, combos, results, requestedCount }) {
  return {
    dataset,
    run: { ...run },
    combos: [...combos],
    summary: summarize(results, requestedCount),
    results: [...results],
  };
}

function evalCombos(runtimeSelection, evalScope) {
  const selected = normalizeSelection(runtimeSelection);
  if (evalScope === "databases") {
    return KNOWLEDGE_PROVIDERS.map((knowledgeProvider, order) =>
      combo({ ...selected, knowledgeProvider }, order),
    );
  }
  if (evalScope === "providers") {
    return PROVIDERS.map((provider, order) => combo({ ...selected, provider }, order));
  }
  if (evalScope === "matrix") {
    return PROVIDERS.flatMap((provider) =>
      KNOWLEDGE_PROVIDERS.map((knowledgeProvider) => ({ provider, knowledgeProvider })),
    ).map((selection, order) => combo(selection, order));
  }
  return [combo(selected, 0)];
}

function normalizeSelection(selection = {}) {
  return {
    provider: selection.provider || DEFAULT_RUNTIME_SELECTION.provider,
    knowledgeProvider:
      selection.knowledgeProvider || DEFAULT_RUNTIME_SELECTION.knowledgeProvider,
  };
}

function combo(selection, order) {
  const normalized = normalizeSelection(selection);
  return {
    ...normalized,
    id: `${normalized.provider}-${normalized.knowledgeProvider}`,
    label: `${providerLabel(normalized.provider)} / ${knowledgeLabel(normalized.knowledgeProvider)}`,
    order,
  };
}

function providerLabel(provider) {
  return {
    openai: "OpenAI",
    deepseek: "DeepSeek",
  }[provider] || provider;
}

function knowledgeLabel(provider) {
  return {
    mongodb: "MongoDB",
    postgresql: "PostgreSQL",
  }[provider] || provider;
}

function summarize(results, requestedCount = results.length) {
  const completed = results.filter((result) => result.status === "ok");
  const failed = results.filter((result) => result.status === "error");
  const timingKeys = [
    "total",
    "retrieval",
    "query_embedding",
    "vector_search",
    "rerank",
    "llm",
    "ttft",
    "llm_ttft",
    "orchestration_overhead",
    "wall",
    "client_ttft",
  ];

  return {
    requested_count: requestedCount,
    completed_count: completed.length,
    failed_count: failed.length,
    pending_count: Math.max(requestedCount - results.length, 0),
    retrieval_used_count: completed.filter((result) => result.retrieval?.used).length,
    retrieval_error_count: completed.filter((result) => result.retrieval?.error).length,
    averages_ms: Object.fromEntries(
      timingKeys.map((key) => [
        key,
        average(completed.map((result) => numberValue(result.timings_ms?.[key]))),
      ]),
    ),
    totals_ms: Object.fromEntries(
      timingKeys.map((key) => [
        key,
        completed.reduce((total, result) => total + numberValue(result.timings_ms?.[key]), 0),
      ]),
    ),
    average_input_tokens: average(
      completed.map((result) => numberValue(result.usage?.input_tokens)),
    ),
    average_output_tokens: average(
      completed.map((result) => numberValue(result.usage?.output_tokens)),
    ),
    total_estimated_cost_usd: round(
      completed.reduce(
        (total, result) => total + numberValue(result.usage?.estimated_cost_usd),
        0,
      ),
      8,
    ),
  };
}

function emptyRetrieval() {
  return {
    used: false,
    reranked: false,
    candidate_count: 0,
    returned_count: 0,
    document_ids: [],
    error: null,
  };
}

function emptyUsage() {
  return {
    input_tokens: 0,
    output_tokens: 0,
    estimated_cost_usd: 0,
    pricing_source: "",
  };
}

function average(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (!clean.length) return 0;
  return round(clean.reduce((total, value) => total + value, 0) / clean.length, 2);
}

function numberValue(value) {
  const parsed = Number(value || 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function elapsedMs(started) {
  return Math.max(Math.round(performance.now() - started), 0);
}

function round(value, decimals) {
  const multiplier = 10 ** decimals;
  return Math.round(value * multiplier) / multiplier;
}

function randomRunId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
