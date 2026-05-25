import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  BrainCircuit,
  Cpu,
  Database,
  Layers3,
  Play,
  Sparkles,
  Zap,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import LLMWidget from "./components/LLMWidget.jsx";
import { streamDeepAgentChat } from "./lib/chatApi.js";
import { DEFAULT_RUNTIME_SELECTION, config, runtimeUrl } from "./lib/config.js";
import { runFastEvals } from "./lib/evalApi.js";

const STORAGE_KEY = "yolorag.deepAgentConversations.v1";
const AGENT_INSTRUCTIONS =
  "You are the YoloRAG deep agent inside the local Ultralytics testing console. Use tools when they materially improve the answer and keep responses direct.";
const STARTER_MESSAGES = [
  "Trace the deep route for a YOLO export question",
  "Find the docs path for training a custom dataset",
  "Check what the agent would inspect for a GitHub issue",
];
const PROVIDER_OPTIONS = [
  { id: "openai", label: "OpenAI", Icon: Sparkles },
  { id: "deepseek", label: "DeepSeek", Icon: Cpu },
];
const KNOWLEDGE_OPTIONS = [
  { id: "mongodb", label: "MongoDB", Icon: Database },
  { id: "postgresql", label: "Postgres", Icon: Layers3 },
];
const EVAL_SCOPE_OPTIONS = [
  { id: "selected", label: "Selected", Icon: Play },
  { id: "databases", label: "Both DBs", Icon: Database },
  { id: "providers", label: "Both LLMs", Icon: Bot },
  { id: "matrix", label: "Matrix", Icon: Layers3 },
];

export default function App() {
  const initialConversations = useRef(null);
  if (initialConversations.current === null) {
    initialConversations.current = loadConversations();
  }

  const [conversations, setConversations] = useState(initialConversations.current);
  const [activeConversationId, setActiveConversationId] = useState(
    initialConversations.current[0]?.id || null,
  );
  const [activePage, setActivePage] = useState("chat");
  const [input, setInput] = useState("");
  const [now, setNow] = useState(Date.now());
  const [runtimeSelection, setRuntimeSelection] = useState(DEFAULT_RUNTIME_SELECTION);
  const requestRef = useRef(null);
  const messagesRef = useRef(null);
  const autoFollowRef = useRef(true);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );
  const sortedConversations = useMemo(
    () => [...conversations].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)),
    [conversations],
  );
  const activeAssistant = activeConversation?.messages.find(
    (message) => message.role === "assistant" && message.status === "working",
  );
  const isWorking = Boolean(activeAssistant);
  const deepEventsUrl = runtimeUrl(config.deepAgentEventsApiUrl, runtimeSelection);

  useEffect(() => {
    const root = document.documentElement;
    const previousTheme = root.dataset.theme;
    root.dataset.theme = "dark";
    return () => {
      if (previousTheme) root.dataset.theme = previousTheme;
      else delete root.dataset.theme;
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.yoloragPage = activePage;
    return () => {
      delete document.documentElement.dataset.yoloragPage;
    };
  }, [activePage]);

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    if (!activeConversationId && conversations[0]) {
      setActiveConversationId(conversations[0].id);
    }
  }, [activeConversationId, conversations]);

  useEffect(() => {
    if (!isWorking) return undefined;
    const interval = window.setInterval(() => setNow(Date.now()), 400);
    return () => window.clearInterval(interval);
  }, [isWorking]);

  useEffect(() => {
    autoFollowRef.current = true;
    window.requestAnimationFrame(() => scrollMessagesToBottom("auto"));
  }, [activeConversationId]);

  useEffect(() => {
    if (!autoFollowRef.current) return;
    window.requestAnimationFrame(() => scrollMessagesToBottom(isWorking ? "auto" : "smooth"));
  }, [activeConversation?.updatedAt, activeConversation?.messages.length, isWorking]);

  async function handleSubmit(event) {
    event.preventDefault();
    const content = input.trim();
    if (!content || !activeConversation || isWorking) return;

    const startedAt = new Date().toISOString();
    const userMessage = createMessage("user", content, startedAt);
    const assistantMessage = {
      ...createMessage("assistant", "", startedAt),
      startedAt,
      status: "working",
      phase: "Starting deep agent",
      events: [],
    };
    const nextTitle =
      activeConversation.messages.length === 0 ? titleFromMessage(content) : activeConversation.title;
    const requestMessages = [...activeConversation.messages, userMessage]
      .filter((message) => message.content.trim())
      .map(({ role, content: messageContent }) => ({ role, content: messageContent }));

    autoFollowRef.current = true;
    setInput("");
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === activeConversation.id
          ? {
              ...conversation,
              title: nextTitle,
              messages: [...conversation.messages, userMessage, assistantMessage],
              updatedAt: startedAt,
            }
          : conversation,
      ),
    );

    const controller = new AbortController();
    requestRef.current = controller;
    let sawDone = false;
    const localStart = Date.now();

    try {
      const result = await streamDeepAgentChat({
        messages: requestMessages,
        sessionId: activeConversation.sessionId,
        instructions: AGENT_INSTRUCTIONS,
        runtimeSelection,
        signal: controller.signal,
        onEvent: (agentEvent) => {
          if (agentEvent.type === "done") sawDone = true;
          applyAgentEvent(activeConversation.id, assistantMessage.id, agentEvent, localStart);
        },
      });

      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === activeConversation.id
            ? {
                ...conversation,
                sessionId: result.sessionId || conversation.sessionId,
              }
            : conversation,
        ),
      );

      if (!sawDone) {
        finishAssistant(activeConversation.id, assistantMessage.id, {
          durationMs: Date.now() - localStart,
          status: "done",
        });
      }
    } catch (error) {
      const aborted = error.name === "AbortError";
      finishAssistant(activeConversation.id, assistantMessage.id, {
        durationMs: Date.now() - localStart,
        status: aborted ? "stopped" : "error",
        phase: aborted ? "Stopped" : "Request failed",
        fallbackContent: aborted ? "Stopped before the agent finished." : error.message,
      });
    } finally {
      requestRef.current = null;
    }
  }

  function applyAgentEvent(conversationId, messageId, agentEvent, localStart) {
    setConversations((current) =>
      current.map((conversation) => {
        if (conversation.id !== conversationId) return conversation;
        return {
          ...conversation,
          updatedAt: new Date().toISOString(),
          messages: conversation.messages.map((message) =>
            message.id === messageId
              ? reduceAssistantMessage(message, agentEvent, localStart)
              : message,
          ),
        };
      }),
    );
  }

  function finishAssistant(conversationId, messageId, patch) {
    setConversations((current) =>
      current.map((conversation) => {
        if (conversation.id !== conversationId) return conversation;
        return {
          ...conversation,
          updatedAt: new Date().toISOString(),
          messages: conversation.messages.map((message) => {
            if (message.id !== messageId) return message;
            return {
              ...message,
              content: message.content || patch.fallbackContent || "",
              status: patch.status,
              phase: patch.phase || message.phase,
              completedAt: new Date().toISOString(),
              durationMs: patch.durationMs,
            };
          }),
        };
      }),
    );
  }

  function handleNewChat() {
    const conversation = createConversation();
    setConversations((current) => [conversation, ...current]);
    setActiveConversationId(conversation.id);
    setActivePage("chat");
    setInput("");
  }

  function handleStop() {
    requestRef.current?.abort();
  }

  function handleMessagesScroll() {
    const container = messagesRef.current;
    if (!container) return;

    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const nextIsAtBottom = distanceFromBottom < 96;
    autoFollowRef.current = nextIsAtBottom;
  }

  function scrollMessagesToBottom(behavior = "smooth") {
    const container = messagesRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior,
    });
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <main className={`console-shell ${activePage === "eval" ? "eval-active" : ""}`}>
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">YO</div>
          <div>
            <p className="brand-kicker">Console</p>
            <h1>YoloRAG</h1>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={handleNewChat}>
          <span aria-hidden="true">+</span>
          New chat
        </button>

        <nav className="conversation-list" aria-label="Conversations">
          {sortedConversations.map((conversation) => (
            <button
              className={`conversation-item ${
                conversation.id === activeConversationId ? "is-active" : ""
              }`}
              key={conversation.id}
              type="button"
              onClick={() => {
                setActiveConversationId(conversation.id);
                setActivePage("chat");
              }}
            >
              <span className="conversation-title">{conversation.title}</span>
              <span className="conversation-meta">
                {conversation.messages.length
                  ? lastMessagePreview(conversation.messages)
                  : "Empty thread"}
              </span>
              <time dateTime={conversation.updatedAt}>{formatShortTime(conversation.updatedAt)}</time>
            </button>
          ))}
        </nav>

        <button
          className={`eval-page-button ${activePage === "eval" ? "is-active" : ""}`}
          type="button"
          onClick={() => setActivePage("eval")}
        >
          <span>Eval</span>
          <small>Fast timing lab</small>
        </button>

        <div className="fast-widget-status">
          <div className="fast-widget-heading">
            <Zap size={15} aria-hidden="true" />
            <span>Fast chat</span>
          </div>
          <LLMWidget runtimeSelection={runtimeSelection} />
        </div>
      </aside>

      {activePage === "eval" ? (
        <EvalPage
          runtimeSelection={runtimeSelection}
          setRuntimeSelection={setRuntimeSelection}
        />
      ) : (
        <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="route-label">Ultralytics / YoloRAG</p>
            <h2>{activeConversation?.title || "New deep agent chat"}</h2>
          </div>
          <div className="header-actions">
            <RuntimeSelector
              runtimeSelection={runtimeSelection}
              setRuntimeSelection={setRuntimeSelection}
            />
            <div className="route-pill icon-pill">
              <BrainCircuit size={14} aria-hidden="true" />
              Deep events
            </div>
            <div className="route-pill muted">{deepEventsUrl}</div>
          </div>
        </header>

        {activeAssistant ? (
          <AgentRunStatus message={activeAssistant} now={now} onStop={handleStop} />
        ) : null}

        <section
          className="messages"
          ref={messagesRef}
          aria-live="polite"
          onScroll={handleMessagesScroll}
        >
          {activeConversation?.messages.length ? (
            activeConversation.messages.map((message) => (
              <MessageRow key={message.id} message={message} now={now} />
            ))
          ) : (
            <div className="empty-state">
              <div className="empty-mark">YO</div>
              <div className="starter-grid">
                {STARTER_MESSAGES.map((starter) => (
                  <button
                    className="starter"
                    key={starter}
                    type="button"
                    onClick={() => setInput(starter)}
                  >
                    {starter}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div />
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Message"
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask the deep agent"
            rows={1}
            value={input}
          />
          {isWorking ? (
            <button
              aria-label="Stop response"
              className="composer-action stop"
              onClick={handleStop}
              title="Stop response"
              type="button"
            >
              <span aria-hidden="true">■</span>
            </button>
          ) : (
            <button
              aria-label="Send message"
              className="composer-action"
              disabled={!input.trim()}
              title="Send message"
              type="submit"
            >
              <span aria-hidden="true">↑</span>
            </button>
          )}
        </form>
        </section>
      )}
    </main>
  );
}

function RuntimeSelector({ runtimeSelection, setRuntimeSelection }) {
  return (
    <div className="runtime-selector">
      <SegmentedControl
        ariaLabel="LLM provider"
        onChange={(provider) =>
          setRuntimeSelection((current) => ({
            ...current,
            provider,
          }))
        }
        options={PROVIDER_OPTIONS}
        value={runtimeSelection.provider}
      />
      <SegmentedControl
        ariaLabel="Knowledge database"
        onChange={(knowledgeProvider) =>
          setRuntimeSelection((current) => ({
            ...current,
            knowledgeProvider,
          }))
        }
        options={KNOWLEDGE_OPTIONS}
        value={runtimeSelection.knowledgeProvider}
      />
    </div>
  );
}

function SegmentedControl({
  ariaLabel,
  className = "",
  disabled = false,
  onChange,
  options,
  value,
}) {
  return (
    <div className={`segmented-control ${className}`} aria-label={ariaLabel} role="group">
      {options.map(({ id, label, Icon }) => (
        <button
          aria-pressed={value === id}
          className={value === id ? "is-active" : ""}
          disabled={disabled}
          key={id}
          onClick={() => onChange(id)}
          title={label}
          type="button"
        >
          <Icon size={14} aria-hidden="true" />
          <span>{label}</span>
        </button>
      ))}
    </div>
  );
}

function EvalPage({ runtimeSelection, setRuntimeSelection }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [startedAt, setStartedAt] = useState(null);
  const [evalScope, setEvalScope] = useState("selected");
  const [now, setNow] = useState(Date.now());
  const requestRef = useRef(null);
  const isRunning = Boolean(startedAt);
  const selectedEvalUrl = runtimeUrl(config.chatApiUrl, runtimeSelection);

  useEffect(() => {
    if (!isRunning) return undefined;
    const interval = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(interval);
  }, [isRunning]);

  async function handleRun() {
    const controller = new AbortController();
    requestRef.current = controller;
    setError("");
    setResult(null);
    setStartedAt(Date.now());
    setNow(Date.now());

    try {
      const nextResult = await runFastEvals({
        signal: controller.signal,
        runtimeSelection,
        evalScope,
        onProgress: setResult,
      });
      setResult(nextResult);
    } catch (evalError) {
      setError(evalError.name === "AbortError" ? "Eval run stopped." : evalError.message);
    } finally {
      requestRef.current = null;
      setStartedAt(null);
    }
  }

  function handleStop() {
    requestRef.current?.abort();
  }

  const summary = result?.summary;
  const runDuration = startedAt ? now - startedAt : result?.run?.duration_ms || 0;
  const metricCards = [
    ["Avg total", summary?.averages_ms?.total],
    ["Avg Total TTFT", summary?.averages_ms?.ttft],
    ["Avg LLM", summary?.averages_ms?.llm],
    ["Avg retrieval", summary?.averages_ms?.retrieval],
    ["Avg embedding", summary?.averages_ms?.query_embedding],
    ["Avg vector DB", summary?.averages_ms?.vector_search],
    ["Avg rerank", summary?.averages_ms?.rerank],
    ["Avg overhead", summary?.averages_ms?.orchestration_overhead],
  ];

  return (
    <section className="eval-panel">
      <header className="eval-header">
        <div>
          <p className="route-label">Ultralytics / Eval</p>
          <h2>Fast endpoint timing</h2>
        </div>
        <div className="header-actions">
          <RuntimeSelector
            runtimeSelection={runtimeSelection}
            setRuntimeSelection={setRuntimeSelection}
          />
          <div className="route-pill">profile_questions.json</div>
          <div className="route-pill muted">{selectedEvalUrl}</div>
        </div>
      </header>

      <section className="eval-hero">
        <div className="eval-hero-copy">
          <p className="eval-eyebrow">Real LLM + retrieval run</p>
          <h3>Run the 30 profile questions through the selected stack.</h3>
          <p>
            Timing only: total wall time, time to first token, LLM completion,
            retrieval, query embedding, vector database search, reranking, and
            orchestration overhead.
          </p>
          <SegmentedControl
            ariaLabel="Eval matrix"
            className="eval-scope-control"
            disabled={isRunning}
            onChange={setEvalScope}
            options={EVAL_SCOPE_OPTIONS}
            value={evalScope}
          />
        </div>
        <div className="eval-run-box">
          <div>
            <span>{isRunning ? "Running" : result ? "Last run" : "Ready"}</span>
            <strong>{formatDuration(runDuration)}</strong>
          </div>
          {isRunning ? (
            <button className="eval-run-button stop" type="button" onClick={handleStop}>
              Stop
            </button>
          ) : (
            <button className="eval-run-button" type="button" onClick={handleRun}>
              Run eval
            </button>
          )}
        </div>
      </section>

      {error ? <div className="eval-error">{error}</div> : null}

      <section className="eval-summary-grid" aria-label="Timing summary">
        {metricCards.map(([label, value]) => (
          <div className="eval-metric" key={label}>
            <span>{label}</span>
            <strong>{formatDurationValue(value)}</strong>
          </div>
        ))}
      </section>

      {summary ? (
        <section className="eval-run-meta">
          <div>
            <span>Completed</span>
            <strong>
              {summary.completed_count}/{summary.requested_count}
            </strong>
          </div>
          <div>
            <span>Retrieval used</span>
            <strong>{summary.retrieval_used_count}</strong>
          </div>
          <div>
            <span>Retrieval errors</span>
            <strong>{summary.retrieval_error_count}</strong>
          </div>
          <div>
            <span>Stacks</span>
            <strong>{result?.run?.combo_count || 1}</strong>
          </div>
          <div>
            <span>Estimated cost</span>
            <strong>${summary.total_estimated_cost_usd.toFixed(6)}</strong>
          </div>
          <div>
            <span>Avg tokens</span>
            <strong>
              {Math.round(summary.average_input_tokens)} in /{" "}
              {Math.round(summary.average_output_tokens)} out
            </strong>
          </div>
        </section>
      ) : null}

      <EvalComparison result={result} />

      <section className="eval-results">
        <div className="eval-results-header">
          <h3>Question timings</h3>
          <span>{result?.dataset?.question_count || 30} questions</span>
        </div>
        <div className="eval-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Stack</th>
                <th>Question</th>
                <th>Total</th>
                <th>TTFT</th>
                <th>LLM</th>
                <th>Retrieval</th>
                <th>Embedding</th>
                <th>Vector DB</th>
                <th>Rerank</th>
                <th>Overhead</th>
                <th>Docs</th>
              </tr>
            </thead>
            <tbody>
              {result?.results?.length ? (
                result.results.map((item) => (
                  <EvalResultRow item={item} key={`${item.combo_id || "selected"}-${item.id}`} />
                ))
              ) : (
                <tr>
                  <td colSpan="11">
                    {isRunning
                      ? "Question results will appear as each request finishes."
                      : "Run eval to populate timing data."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}

function EvalResultRow({ item }) {
  const timings = item.timings_ms || {};
  return (
    <tr className={item.status === "error" ? "is-error" : ""}>
      <td>
        <div className="eval-stack-cell">
          <strong>{item.provider || "-"}</strong>
          <span>{item.knowledge_provider || "-"}</span>
        </div>
      </td>
      <td>
        <div className="eval-question-cell">
          <span>{item.id}</span>
          <strong>{item.question}</strong>
          {item.error ? <small>{item.error}</small> : null}
        </div>
      </td>
      <td>{formatDurationValue(timings.total)}</td>
      <td>{formatDurationValue(timings.ttft)}</td>
      <td>{formatDurationValue(timings.llm)}</td>
      <td>{formatDurationValue(timings.retrieval)}</td>
      <td>{formatDurationValue(timings.query_embedding)}</td>
      <td>{formatDurationValue(timings.vector_search)}</td>
      <td>{formatDurationValue(timings.rerank)}</td>
      <td>{formatDurationValue(timings.orchestration_overhead)}</td>
      <td>{item.retrieval?.returned_count ?? 0}</td>
    </tr>
  );
}

function EvalComparison({ result }) {
  const comparison = useMemo(() => buildEvalComparisons(result), [result]);
  if (!comparison) return null;

  return (
    <section className="eval-comparison" aria-label="Eval comparison">
      <div className="eval-comparison-header">
        <div>
          <p className="eval-eyebrow">Stack comparison</p>
          <h3>Average latency by runtime</h3>
        </div>
        <span>{comparison.completedCount} successful calls</span>
      </div>
      <div className="eval-comparison-grid">
        {comparison.stack.length > 1 ? (
          <EvalComparisonGroup title="Provider + DB" rows={comparison.stack} />
        ) : null}
        {comparison.providers.length > 1 ? (
          <EvalComparisonGroup title="Provider" rows={comparison.providers} />
        ) : null}
        {comparison.databases.length > 1 ? (
          <EvalComparisonGroup title="Database" rows={comparison.databases} />
        ) : null}
      </div>
    </section>
  );
}

function EvalComparisonGroup({ title, rows }) {
  const maxTotal = Math.max(...rows.map((row) => row.averages.total), 1);
  return (
    <div className="comparison-group">
      <div className="comparison-group-title">
        <h4>{title}</h4>
        <span>Avg total</span>
      </div>
      <div className="comparison-list">
        {rows.map((row) => (
          <div className={`comparison-row ${row.rank === 1 ? "is-best" : ""}`} key={row.key}>
            <div className="comparison-row-top">
              <div>
                <strong>{row.label}</strong>
                <span>
                  {row.completedCount}/{row.count} ok
                </span>
              </div>
              <time>{formatDurationValue(row.averages.total)}</time>
            </div>
            <div className="comparison-bar" aria-hidden="true">
              <span style={{ width: `${Math.max((row.averages.total / maxTotal) * 100, 4)}%` }} />
            </div>
            <div className="comparison-breakdown">
              <div>
                <span>TTFT</span>
                <strong>{formatDurationValue(row.averages.ttft)}</strong>
              </div>
              <div>
                <span>LLM</span>
                <strong>{formatDurationValue(row.averages.llm)}</strong>
              </div>
              <div>
                <span>Retrieval</span>
                <strong>{formatDurationValue(row.averages.retrieval)}</strong>
              </div>
              <div>
                <span>Vector</span>
                <strong>{formatDurationValue(row.averages.vector_search)}</strong>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentRunStatus({ message, now, onStop }) {
  const duration = message.startedAt ? now - Date.parse(message.startedAt) : 0;
  const latestEvent = message.events?.at(-1)?.label || message.phase || "Starting deep agent";

  return (
    <div className="run-status" role="status" aria-live="polite">
      <div className="run-status-orbit" aria-hidden="true">
        <span />
      </div>
      <div className="run-status-copy">
        <span>Agent is working</span>
        <strong>{latestEvent}</strong>
      </div>
      <time>{formatDuration(duration)}</time>
      <button type="button" onClick={onStop}>
        Stop
      </button>
    </div>
  );
}

function MessageRow({ message, now }) {
  const isAssistant = message.role === "assistant";
  const isWorkingMessage = message.status === "working";
  const duration =
    isAssistant && message.startedAt
      ? message.durationMs || (message.status === "working" ? now - Date.parse(message.startedAt) : null)
      : null;

  return (
    <article className={`message-row ${isAssistant ? "assistant" : "user"} ${message.status || ""}`}>
      <div className="message-avatar">{isAssistant ? "YO" : "You"}</div>
      <div className="message-body">
        <div className="message-meta">
          <span>{isAssistant ? "YoloRAG" : "You"}</span>
          <time dateTime={message.createdAt}>{formatMessageTime(message.createdAt)}</time>
          {duration !== null ? <span>{agentDurationLabel(message.status, duration)}</span> : null}
        </div>
        {isAssistant && message.events?.length ? <AgentTimeline events={message.events} /> : null}
        <div
          className={`message-content ${message.status === "error" ? "is-error" : ""}`}
          aria-busy={isWorkingMessage}
        >
          {message.content ? (
            <MarkdownMessage content={message.content} />
          ) : isWorkingMessage ? (
            <WorkingMessage phase={message.phase} />
          ) : (
            ""
          )}
        </div>
      </div>
    </article>
  );
}

function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: MarkdownLink,
        table: MarkdownTable,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function MarkdownLink({ node, ...props }) {
  return <a {...props} rel="noreferrer" target="_blank" />;
}

function MarkdownTable({ node, ...props }) {
  return (
    <div className="markdown-table-wrap">
      <table {...props} />
    </div>
  );
}

function WorkingMessage({ phase }) {
  return (
    <div className="working-message">
      <span className="working-dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <span>{phase || "Working"}</span>
    </div>
  );
}

function AgentTimeline({ events }) {
  return (
    <ol className="agent-timeline" aria-label="Agent events">
      {events.slice(-5).map((event) => (
        <li className={event.type} key={event.id}>
          <span>{event.label}</span>
          <time dateTime={event.createdAt}>{formatShortTime(event.createdAt)}</time>
        </li>
      ))}
    </ol>
  );
}

function reduceAssistantMessage(message, event, localStart) {
  const timestamp = new Date().toISOString();

  if (event.type === "content") {
    return {
      ...message,
      content: `${message.content}${event.content || ""}`,
      phase: "Responding",
    };
  }

  if (event.type === "done") {
    return {
      ...message,
      status: "done",
      phase: "Complete",
      completedAt: timestamp,
      durationMs: Number.isFinite(event.latency_ms) ? event.latency_ms : Date.now() - localStart,
      stepCount: event.step_count,
      toolCallCount: event.tool_call_count,
      events: appendTimelineEvent(message.events, event, timestamp),
    };
  }

  if (event.type === "error") {
    return {
      ...message,
      content: message.content || event.error || "The agent request failed.",
      status: "error",
      phase: "Request failed",
      completedAt: timestamp,
      durationMs: Date.now() - localStart,
      events: appendTimelineEvent(message.events, event, timestamp),
    };
  }

  return {
    ...message,
    phase: event.message || event.type || message.phase,
    events: appendTimelineEvent(message.events, event, timestamp),
  };
}

function appendTimelineEvent(events = [], event, createdAt) {
  const label = timelineLabel(event);
  if (!label) return events;
  return [
    ...events,
    {
      id: createId(),
      type: event.type || "status",
      label,
      createdAt,
    },
  ];
}

function timelineLabel(event) {
  if (event.type === "status") return event.message || "Working";
  if (event.type === "tool_call") return `Calling ${event.tool || "tool"}`;
  if (event.type === "tool_result") {
    const suffix = event.error || event.summary || "completed";
    return `${event.tool || "tool"}: ${suffix}`;
  }
  if (event.type === "done") {
    const count = Number(event.tool_call_count || 0);
    return count === 1 ? "Finished with 1 tool call" : `Finished with ${count} tool calls`;
  }
  if (event.type === "error") return event.error || "Request failed";
  return null;
}

function createConversation() {
  const now = new Date().toISOString();
  return {
    id: createId(),
    title: "New deep agent chat",
    sessionId: null,
    updatedAt: now,
    messages: [],
  };
}

function createMessage(role, content, createdAt = new Date().toISOString()) {
  return {
    id: createId(),
    role,
    content,
    createdAt,
  };
}

function loadConversations() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(saved) && saved.length) return saved;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
  }
  return [createConversation()];
}

function saveConversations(conversations) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.slice(0, 24)));
  } catch {
    return undefined;
  }
  return undefined;
}

function createId() {
  return window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function titleFromMessage(message) {
  const clean = message.replace(/\s+/g, " ").trim();
  return clean.length > 52 ? `${clean.slice(0, 49)}...` : clean;
}

function lastMessagePreview(messages) {
  const last = [...messages].reverse().find((message) => message.content.trim());
  if (!last) return "Working";
  const content = last.content.replace(/\s+/g, " ").trim();
  return content.length > 42 ? `${content.slice(0, 39)}...` : content;
}

function buildEvalComparisons(result) {
  const rows = result?.results || [];
  const completed = rows.filter((row) => row.status === "ok");
  if (!completed.length || Number(result?.run?.combo_count || 1) <= 1) return null;

  const stack = comparisonRows(rows, (row) => ({
    key: row.combo_id || `${row.provider}-${row.knowledge_provider}`,
    label: row.combo_label || `${providerLabel(row.provider)} / ${databaseLabel(row.knowledge_provider)}`,
  }));
  const providers = comparisonRows(rows, (row) => ({
    key: row.provider || "unknown",
    label: providerLabel(row.provider),
  }));
  const databases = comparisonRows(rows, (row) => ({
    key: row.knowledge_provider || "unknown",
    label: databaseLabel(row.knowledge_provider),
  }));

  return {
    completedCount: completed.length,
    stack,
    providers,
    databases,
  };
}

function comparisonRows(rows, groupForRow) {
  const groups = new Map();
  rows.forEach((row) => {
    const group = groupForRow(row);
    if (!groups.has(group.key)) {
      groups.set(group.key, {
        key: group.key,
        label: group.label,
        rows: [],
      });
    }
    groups.get(group.key).rows.push(row);
  });

  return [...groups.values()]
    .map((group) => {
      const completed = group.rows.filter((row) => row.status === "ok");
      return {
        key: group.key,
        label: group.label,
        count: group.rows.length,
        completedCount: completed.length,
        averages: {
          total: averageMetric(completed, "total"),
          ttft: averageMetric(completed, "ttft"),
          llm: averageMetric(completed, "llm"),
          retrieval: averageMetric(completed, "retrieval"),
          vector_search: averageMetric(completed, "vector_search"),
        },
      };
    })
    .filter((group) => group.completedCount > 0)
    .sort((first, second) => first.averages.total - second.averages.total)
    .map((group, index) => ({ ...group, rank: index + 1 }));
}

function averageMetric(rows, metric) {
  if (!rows.length) return 0;
  const values = rows.map((row) => numberValue(row.timings_ms?.[metric]));
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function numberValue(value) {
  const parsed = Number(value || 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function providerLabel(provider) {
  return {
    openai: "OpenAI",
    deepseek: "DeepSeek",
  }[provider] || provider || "Unknown";
}

function databaseLabel(database) {
  return {
    mongodb: "MongoDB",
    postgresql: "Postgres",
  }[database] || database || "Unknown";
}

function formatShortTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatMessageTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatDuration(ms) {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

function formatDurationValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return formatDuration(Number(value));
}

function agentDurationLabel(status, duration) {
  if (status === "working") return `Working ${formatDuration(duration)}`;
  if (status === "stopped") return `Stopped after ${formatDuration(duration)}`;
  if (status === "error") return `Failed after ${formatDuration(duration)}`;
  return `Agent worked ${formatDuration(duration)}`;
}
