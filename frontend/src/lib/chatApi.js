import { config } from "./config.js";

export async function streamDeepAgentChat({
  messages,
  sessionId,
  instructions,
  signal,
  onEvent,
}) {
  const response = await fetch(config.deepAgentEventsApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      session_id: sessionId || undefined,
      instructions,
      tools: ["docs_search", "mcp"],
      analytics: true,
    }),
    signal,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  const nextSessionId = response.headers.get("X-Session-ID") || sessionId || null;
  if (!response.body) {
    throw new Error("The browser did not expose a response stream.");
  }

  await readServerSentEvents(response.body, onEvent);
  return { sessionId: nextSessionId };
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

  try {
    onEvent(JSON.parse(data));
  } catch {
    onEvent({ type: "content", content: data });
  }
}
