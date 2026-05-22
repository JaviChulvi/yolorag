import LLMWidget from "./components/LLMWidget.jsx";
import { config } from "./lib/config.js";

export default function App() {
  return (
    <main className="min-h-screen bg-white px-6 py-10 text-neutral-950">
      <section className="mx-auto max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-widest text-neutral-500">YoloRAG</p>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight md:text-6xl">LLM widget test page</h1>
        <p className="mt-6 text-base leading-7 text-neutral-600">
          API endpoint: <code className="rounded bg-neutral-100 px-2 py-1">{config.chatApiUrl}</code>
        </p>
        <LLMWidget />
      </section>
    </main>
  );
}
