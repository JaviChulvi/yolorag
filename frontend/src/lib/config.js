const chatApiUrl = import.meta.env.VITE_YOLORAG_API_URL || "/api/chat/fast";
const deepAgentEventsApiUrl =
  import.meta.env.VITE_YOLORAG_DEEP_EVENTS_API_URL || "/api/chat/deep/events";

export const DEFAULT_RUNTIME_SELECTION = {
  provider: "deepseek",
  knowledgeProvider: "mongodb",
};

export const config = {
  chatApiUrl,
  deepAgentEventsApiUrl,
  widgetScriptSrc: import.meta.env.VITE_LLM_WIDGET_SRC || "/vendor/ultralytics-chat.js",
};

export function runtimeUrl(baseUrl, selection = DEFAULT_RUNTIME_SELECTION) {
  const url = new URL(baseUrl, window.location.origin);
  url.searchParams.set("provider", selection.provider || DEFAULT_RUNTIME_SELECTION.provider);
  url.searchParams.set(
    "knowledge_provider",
    selection.knowledgeProvider || DEFAULT_RUNTIME_SELECTION.knowledgeProvider,
  );
  if (url.origin === window.location.origin) {
    return `${url.pathname}${url.search}${url.hash}`;
  }
  return url.toString();
}
