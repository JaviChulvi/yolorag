export const config = {
  chatApiUrl: import.meta.env.VITE_YOLORAG_API_URL || "/api/chat/fast",
  widgetScriptSrc: import.meta.env.VITE_LLM_WIDGET_SRC || "/vendor/ultralytics-chat.js",
};

export function apiBaseFromChatUrl(chatApiUrl = config.chatApiUrl) {
  return chatApiUrl.replace(/\/chat(?:\/(?:fast|deep))?\/?$/, "");
}
