import { useRef, useState } from "react";
import {
  Boxes,
  Database,
  ExternalLink,
  Image as ImageIcon,
  Layers3,
  Sparkles,
  Tag,
  Wand2,
} from "lucide-react";

import { fetchDatasetHighlights, fetchDatasetMeta } from "../lib/datasetApi.js";

const EXAMPLE_REF = "https://platform.ultralytics.com/ddxy/datasets/dogs-cats";
const HIGHLIGHT_COUNT = 4;
const DEFAULT_PROMPT =
  "You are a computer-vision dataset analyst. Using the attached sample images " +
  "(with their bounding-box labels) and the dataset metadata, write a clear, concise " +
  "description of this dataset: what it depicts, its label classes, typical image " +
  "characteristics, and likely use cases. Keep it to a short paragraph.";
// Distinct colors for label classes, indexed by classId.
const CLASS_COLORS = ["#22d3ee", "#f472b6", "#d7ff2f", "#2f6bff", "#fb923c", "#a78bfa"];

function classColor(classId) {
  if (classId == null || classId < 0) return "#a1a1aa";
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}

export default function DatasetPage() {
  const [refInput, setRefInput] = useState(EXAMPLE_REF);
  const [dataset, setDataset] = useState(null);
  const [loadingMeta, setLoadingMeta] = useState(false);
  const [error, setError] = useState("");

  const [highlights, setHighlights] = useState(null);
  const [loadingHighlights, setLoadingHighlights] = useState(false);
  const [highlightsError, setHighlightsError] = useState("");

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [sending, setSending] = useState(false);
  const [sendResult, setSendResult] = useState(null);

  const requestRef = useRef(null);
  const sendTimerRef = useRef(null);

  async function loadHighlights(ref, controller) {
    setHighlightsError("");
    setLoadingHighlights(true);
    try {
      const result = await fetchDatasetHighlights(
        { ref, count: HIGHLIGHT_COUNT },
        controller.signal,
      );
      setHighlights(result);
    } catch (err) {
      if (err.name !== "AbortError") setHighlightsError(err.message);
    } finally {
      setLoadingHighlights(false);
    }
  }

  async function handleLoadDataset(event) {
    event?.preventDefault();
    const ref = refInput.trim();
    if (!ref || loadingMeta) return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setError("");
    setHighlightsError("");
    setLoadingMeta(true);
    setDataset(null);
    setHighlights(null);
    window.clearTimeout(sendTimerRef.current);
    setSending(false);
    setSendResult(null);

    try {
      const meta = await fetchDatasetMeta(ref, controller.signal);
      setDataset(meta);
      // Metadata first, then the label-diverse images.
      loadHighlights(ref, controller);
    } catch (metaErr) {
      if (metaErr.name !== "AbortError") setError(metaErr.message);
    } finally {
      setLoadingMeta(false);
    }
  }

  // Dummy for now: no LLM call is made. Simulate a short round-trip and show a
  // preview of the request that would be sent (prompt + images + metadata).
  // TODO: wire to a backend endpoint that forwards this to the LLM.
  function handleSend() {
    const images = highlights?.images || [];
    if (!prompt.trim() || sending || !images.length) return;
    setSendResult(null);
    setSending(true);
    window.clearTimeout(sendTimerRef.current);
    sendTimerRef.current = window.setTimeout(() => {
      setSendResult({
        prompt,
        imageNames: images.map((image) => image.name || image.hash),
        classes: highlights?.classes || dataset?.classNames || [],
      });
      setSending(false);
    }, 600);
  }

  const classNames = dataset?.classNames || [];
  const availableSplits = ["train", "val", "test"].filter(
    (name) => Number(dataset?.splits?.[name]) > 0,
  );

  return (
    <section className="eval-panel dataset-panel">
      <header className="eval-header">
        <div>
          <p className="route-label">Ultralytics / Dataset</p>
          <h2>Platform dataset explorer</h2>
        </div>
        <div className="header-actions">
          {dataset ? (
            <div className="route-pill">
              <Database size={14} aria-hidden="true" />
              {dataset.username}/{dataset.slug}
            </div>
          ) : null}
          {dataset ? (
            <a
              className="route-pill muted dataset-platform-link"
              href={dataset.platformUrl}
              target="_blank"
              rel="noreferrer"
            >
              platform.ultralytics.com
              <ExternalLink size={12} aria-hidden="true" />
            </a>
          ) : null}
        </div>
      </header>

      <div className="dataset-body">
        <div className="dataset-controls">
          <form className="dataset-form" onSubmit={handleLoadDataset}>
            <label>
              <span>Dataset URL or username/slug</span>
              <input
                type="text"
                value={refInput}
                placeholder="https://platform.ultralytics.com/ddxy/datasets/dogs-cats"
                onChange={(event) => setRefInput(event.target.value)}
                disabled={loadingMeta}
                spellCheck={false}
              />
            </label>
            <button
              type="submit"
              className="eval-run-button"
              disabled={loadingMeta || !refInput.trim()}
            >
              {loadingMeta ? "Loading…" : "Load dataset"}
            </button>
            <p className="dataset-hint">
              Pulls public metadata from the Ultralytics platform, then shows the{" "}
              {HIGHLIGHT_COUNT} images with the most label diversity.
            </p>
          </form>
        </div>

        <div className="dataset-info">
          {error ? <div className="eval-error">{error}</div> : null}

          {!dataset && !error ? (
            <div className="dataset-empty">
              <Boxes size={30} aria-hidden="true" />
              <p>Enter a dataset reference and load it to see metadata and images here.</p>
            </div>
          ) : null}

          {dataset ? (
            <>
              <DatasetMetaCard
                dataset={dataset}
                classNames={classNames}
                availableSplits={availableSplits}
              />
              <HighlightsSection
                highlights={highlights}
                loading={loadingHighlights}
                error={highlightsError}
              />
              <GenerateSection
                prompt={prompt}
                setPrompt={setPrompt}
                sending={sending}
                sendResult={sendResult}
                onSend={handleSend}
                imageCount={highlights?.images?.length || 0}
                classes={highlights?.classes || classNames}
              />
            </>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function HighlightsSection({ highlights, loading, error }) {
  const classes = highlights?.classes || [];
  const covered = highlights?.coveredClasses || [];
  const maxDistinct = highlights?.maxDistinctLabelsInImage ?? 0;
  const images = highlights?.images || [];

  return (
    <section className="dataset-images">
      <div className="dataset-images-header">
        <h3>
          <Sparkles size={16} aria-hidden="true" />
          {HIGHLIGHT_COUNT} most label-diverse images
        </h3>
        {covered.length ? (
          <span>
            covering {covered.join(", ")}
          </span>
        ) : null}
      </div>

      {covered.length ? (
        <div className="dataset-legend" aria-label="Label classes">
          {classes.map((name, index) => (
            <span className="dataset-legend-item" key={name}>
              <i style={{ background: classColor(index) }} aria-hidden="true" />
              {name}
            </span>
          ))}
        </div>
      ) : null}

      {maxDistinct > 0 && classes.length > 1 ? (
        <p className="dataset-hint">
          {maxDistinct > 1
            ? `Each image below carries up to ${maxDistinct} distinct labels.`
            : `No single image in this dataset contains more than one class, so these ${HIGHLIGHT_COUNT} together cover all ${classes.length} (${classes.join(", ")}).`}
        </p>
      ) : null}

      {error ? <div className="eval-error">{error}</div> : null}

      <div className="dataset-image-grid highlight-grid">
        {images.map((image) => (
          <HighlightCard key={image.id} image={image} />
        ))}
        {loading && !images.length
          ? Array.from({ length: HIGHLIGHT_COUNT }).map((_, index) => (
              <div className="dataset-image-card is-skeleton" key={`skeleton-${index}`} />
            ))
          : null}
      </div>

      {!loading && !images.length && !error ? (
        <p className="dataset-hint">No labeled images found for this dataset.</p>
      ) : null}
    </section>
  );
}

function GenerateSection({ prompt, setPrompt, sending, sendResult, onSend, imageCount, classes }) {
  const ready = imageCount > 0;
  return (
    <section className="dataset-generate">
      <div className="dataset-generate-head">
        <h3>
          <Wand2 size={16} aria-hidden="true" />
          Generate dataset description
        </h3>
        <span>{imageCount} images + metadata → LLM</span>
      </div>
      <p className="dataset-hint">
        Sends the prompt below along with the {imageCount} sample images and the dataset
        metadata to an LLM to auto-write a description.
      </p>

      <label className="dataset-prompt">
        <span>Prompt</span>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          rows={5}
          placeholder="Describe how the model should summarize this dataset…"
        />
      </label>

      <div className="dataset-attach-row" aria-label="Attached to the request">
        <span className="dataset-attach-chip">
          <ImageIcon size={13} aria-hidden="true" />
          {imageCount} images
        </span>
        {classes.length ? (
          <span className="dataset-attach-chip">
            <Tag size={13} aria-hidden="true" />
            {classes.join(", ")}
          </span>
        ) : null}
        <span className="dataset-attach-chip">
          <Database size={13} aria-hidden="true" />
          metadata
        </span>
      </div>

      <div className="dataset-generate-actions">
        <button
          type="button"
          className="eval-run-button"
          onClick={onSend}
          disabled={sending || !prompt.trim() || !ready}
        >
          {sending ? "Sending…" : "Send"}
        </button>
        <span className="dataset-dummy-note">Dummy — not connected to an LLM yet</span>
      </div>

      {sendResult ? (
        <div className="dataset-generate-result">
          <div className="dataset-result-badge">Preview · no LLM call made</div>
          <p>
            This is where the generated description will appear once <strong>Send</strong> is
            wired to a model. The request would include:
          </p>
          <ul>
            <li>
              <strong>{sendResult.imageNames.length} images:</strong>{" "}
              {sendResult.imageNames.join(", ")}
            </li>
            {sendResult.classes.length ? (
              <li>
                <strong>Classes:</strong> {sendResult.classes.join(", ")}
              </li>
            ) : null}
            <li>
              <strong>Prompt:</strong> {sendResult.prompt}
            </li>
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function HighlightCard({ image }) {
  const labels = image.labels || [];
  // Match the frame to the image's aspect ratio so it isn't cropped — the
  // normalized bbox overlays only line up when the whole image is shown.
  const ratio = image.width && image.height ? `${image.width} / ${image.height}` : "1 / 1";
  return (
    <figure className="dataset-image-card">
      <div className="dataset-image-frame" style={{ aspectRatio: ratio }}>
        {image.thumbnailUrl ? (
          <img src={image.thumbnailUrl} alt={image.name || "dataset image"} loading="lazy" />
        ) : null}
        {labels.map((label, index) => (
          <BBox key={index} label={label} />
        ))}
      </div>
      <figcaption>
        <span className="dataset-image-name" title={image.name}>
          {image.name || image.hash}
        </span>
        <span className="dataset-image-tags">
          {(image.classNames || []).map((name) => {
            const classId = image.classNames.indexOf(name);
            const color = classColor(labels.find((l) => l.className === name)?.classId ?? classId);
            return (
              <span className="dataset-image-tag" key={name} style={{ borderColor: color, color }}>
                {name}
              </span>
            );
          })}
        </span>
      </figcaption>
    </figure>
  );
}

function BBox({ label }) {
  const bbox = label.bbox;
  if (!Array.isArray(bbox) || bbox.length < 4) return null;
  // YOLO normalized [cx, cy, w, h] -> CSS percentages.
  const [cx, cy, w, h] = bbox;
  const color = classColor(label.classId);
  const style = {
    left: `${(cx - w / 2) * 100}%`,
    top: `${(cy - h / 2) * 100}%`,
    width: `${w * 100}%`,
    height: `${h * 100}%`,
    borderColor: color,
  };
  return (
    <span className="dataset-bbox" style={style}>
      <span className="dataset-bbox-label" style={{ background: color }}>
        {label.className ?? label.classId}
      </span>
    </span>
  );
}

function DatasetMetaCard({ dataset, classNames, availableSplits }) {
  const stats = [
    ["Images", Number(dataset.imageCount || 0).toLocaleString()],
    ["Classes", dataset.classCount ?? classNames.length],
    ["Annotations", Number(dataset.annotationCount || 0).toLocaleString()],
    ["Size", formatBytes(dataset.totalBytes)],
    ["Format", (dataset.format || "-").toUpperCase()],
    ["Region", (dataset.region || "-").toUpperCase()],
  ];

  return (
    <section className="dataset-meta-card">
      <div className="dataset-meta-top">
        <div>
          <h3>{dataset.name || dataset.slug}</h3>
          <p className="dataset-meta-sub">
            by {dataset.username} · updated {formatDate(dataset.updatedAt)}
          </p>
        </div>
        <div className="dataset-badges">
          {dataset.task ? <span className="dataset-badge accent">{dataset.task}</span> : null}
          {dataset.visibility ? <span className="dataset-badge">{dataset.visibility}</span> : null}
          {dataset.status ? <span className="dataset-badge">{dataset.status}</span> : null}
        </div>
      </div>

      {dataset.description ? <p className="dataset-description">{dataset.description}</p> : null}

      <div className="dataset-meta-grid">
        {stats.map(([label, value]) => (
          <div className="dataset-stat" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      {classNames.length ? (
        <div className="dataset-tag-row">
          <span className="dataset-tag-label">
            <Tag size={13} aria-hidden="true" />
            Classes
          </span>
          <div className="dataset-tags">
            {classNames.map((name) => (
              <span className="dataset-tag" key={name}>
                {name}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {availableSplits.length ? (
        <div className="dataset-tag-row">
          <span className="dataset-tag-label">
            <Layers3 size={13} aria-hidden="true" />
            Splits
          </span>
          <div className="dataset-tags">
            {availableSplits.map((name) => (
              <span className="dataset-tag" key={name}>
                {name}
                <em>{Number(dataset.splits[name]).toLocaleString()}</em>
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(value) {
  if (!value) return "-";
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(new Date(value));
  } catch {
    return "-";
  }
}
