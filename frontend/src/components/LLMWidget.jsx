import { useEffect, useMemo, useRef, useState } from "react";

import { config } from "../lib/config.js";

export default function LLMWidget() {
  const chatRef = useRef(null);
  const [status, setStatus] = useState("Loading widget");

  const widgetConfig = useMemo(
    () => ({
      apiUrl: config.chatApiUrl,
      analytics: true,
      pageContent: true,
      instructions:
        "You are Ultralytics AI, an assistant for Ultralytics Docs, YOLO training, dataset preparation, model export, and deployment questions.",
      branding: {
        name: "Ultralytics AI",
        tagline: "Ask anything about Ultralytics, YOLO, and more",
        pillText: "Ask AI",
      },
      welcome: {
        title: "Ultralytics AI",
        message:
          "Your assistant for Ultralytics Docs. Ask about YOLO26 training, dataset preparation, model export, or deployment.",
        chatExamples: [
          "How do I train YOLO26 on a custom dataset?",
          "How do I export a model to ONNX?",
          "How do I run inference on a video?",
        ],
        searchExamples: [
          "YOLO26 training",
          "custom dataset YAML",
          "ONNX export",
          "video inference",
        ],
      },
      ui: {
        placeholder: "Ask anything about Ultralytics...",
      },
      tools: [
        { id: "search", name: "Search", icon: "globe" },
        { id: "github", name: "GitHub", icon: "github" },
        { id: "trace", name: "Trace", icon: "sparkles" },
      ],
      theme: {
        primary: "#2f6bff",
        dark: "#07080d",
        accent: "#d7ff2f",
        text: "#f8fafc",
      },
    }),
    [],
  );

  useEffect(() => {
    let cancelled = false;

    loadWidgetScript(config.widgetScriptSrc)
      .then(() => {
        if (cancelled) return;
        if (!window.UltralyticsChat) throw new Error("UltralyticsChat was not registered.");
        chatRef.current = new window.UltralyticsChat(widgetConfig);
        window.yoloragChat = chatRef.current;
        setStatus("Widget ready");
      })
      .catch((error) => {
        console.error(error);
        if (!cancelled) setStatus("Widget failed to load");
      });

    return () => {
      cancelled = true;
      chatRef.current?.destroy?.();
      if (window.yoloragChat === chatRef.current) delete window.yoloragChat;
      chatRef.current = null;
    };
  }, [widgetConfig]);

  return (
    <p className="widget-status-text" aria-live="polite">
      {status}
    </p>
  );
}

function loadWidgetScript(src) {
  if (window.UltralyticsChat) return Promise.resolve();
  const existing = document.querySelector(`script[data-yolorag-widget="${src}"]`);
  if (existing) {
    return new Promise((resolve, reject) => {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.yoloragWidget = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load widget script: ${src}`));
    document.head.appendChild(script);
  });
}
