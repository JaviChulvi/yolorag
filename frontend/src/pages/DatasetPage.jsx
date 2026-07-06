import { useEffect, useRef, useState } from "react";
import {
  Boxes,
  Database,
  ExternalLink,
  Grid2x2,
  Image as ImageIcon,
  Images,
  Layers3,
  Sparkles,
  SquareDashed,
  Tag,
  Wand2,
} from "lucide-react";

import {
  describeDataset,
  fetchDatasetHighlights,
  fetchDatasetMeta,
  fetchDescribeProviders,
} from "../lib/datasetApi.js";

const EXAMPLE_REF = "https://platform.ultralytics.com/ddxy/datasets/dogs-cats";
const HIGHLIGHT_COUNT = 4;

const TASK_LABELS = {
  detect: "object detection",
  segment: "instance segmentation",
  classify: "image classification",
  pose: "pose estimation",
  obb: "oriented bounding-box detection",
};

// Seeds the editable prompt with the dataset's real metadata so the model
// describes the DATASET (not just the sample images) and knows its actual task.
// The user can edit it before sending.
function buildDefaultPrompt(dataset, sampleCount = HIGHLIGHT_COUNT) {
  const classes = dataset?.classNames || [];
  const classList = classes.length ? classes.map((name) => `'${name}'`).join(", ") : "unspecified";
  const task = dataset?.task ? TASK_LABELS[dataset.task] || dataset.task : "computer-vision";

  const facts = [];
  if (dataset?.name) facts.push(`Name: ${dataset.name}`);
  facts.push(`Task: ${task}`);
  facts.push(`Classes (${classes.length || "?"}): ${classList}`);
  if (dataset?.imageCount) {
    const splits = dataset.splits || {};
    const splitBits = ["train", "val", "test"].filter((key) => splits[key]).map((key) => `${key} ${splits[key]}`);
    facts.push(`Total images: ${dataset.imageCount}${splitBits.length ? ` (${splitBits.join(", ")})` : ""}`);
  }
  if (dataset?.annotationCount) facts.push(`Total annotations: ${dataset.annotationCount}`);

  return (
    `You are writing an SEO-friendly catalog description for a computer-vision dataset, so ` +
    `people can discover it through text search.\n\n` +
    `Dataset facts:\n- ${facts.join("\n- ")}\n\n` +
    `Attached are ${sampleCount} representative sample images from the dataset (a small sample ` +
    `of the full set), each drawn with its bounding-box label. Use them to understand the kind ` +
    `of imagery the dataset contains.\n\n` +
    `Write a search-optimized description of the dataset (2–4 sentences, roughly 300–500 ` +
    `characters). Requirements:\n` +
    `- Describe the actual visual content concretely: typical subjects, notable varieties/breeds ` +
    `or object types, and the settings, composition and image style seen in the imagery.\n` +
    `- Naturally weave in relevant search keywords a user might type — the ${task} task, the ` +
    `class names, the visual domain, and common use cases.\n` +
    `- Describe the dataset as a whole; do NOT count or list the individual sample images or ` +
    `narrate them one by one.\n` +
    `- Return plain prose only — no markdown, bold, headings, or bullet points.`
  );
}
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

  const [prompt, setPrompt] = useState(() => buildDefaultPrompt(null));
  const [sending, setSending] = useState(false);
  const [sendResult, setSendResult] = useState(null);
  const [sendError, setSendError] = useState("");
  // Optional: combine the samples into one 2x2 mosaic instead of sending them
  // separately. Off by default (send the images as-is); handy for models that
  // error on multi-image requests.
  const [mosaic, setMosaic] = useState(false);
  // Optional: burn each object's label box + class name into the image. On by
  // default (matches the UI overlays and the prompt); off sends raw imagery.
  const [withBoxes, setWithBoxes] = useState(true);

  const [providers, setProviders] = useState([]);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");

  const requestRef = useRef(null);
  const sendAbortRef = useRef(null);

  // Load the available vision providers + models once.
  useEffect(() => {
    const controller = new AbortController();
    fetchDescribeProviders(controller.signal)
      .then((loaded) => {
        setProviders(loaded);
        const preferred = loaded.find((item) => item.available) || loaded[0];
        if (preferred) {
          setProvider(preferred.name);
          setModel(preferred.default_model || preferred.models?.[0] || "");
        }
      })
      .catch(() => {
        /* provider list is best-effort; describe will surface errors on send */
      });
    return () => controller.abort();
  }, []);

  async function loadHighlights(ref, controller, meta) {
    setHighlightsError("");
    setLoadingHighlights(true);
    try {
      const result = await fetchDatasetHighlights(
        { ref, count: HIGHLIGHT_COUNT },
        controller.signal,
      );
      setHighlights(result);
      // Seed the prompt with this dataset's real metadata.
      setPrompt(buildDefaultPrompt(meta, result.images?.length || HIGHLIGHT_COUNT));
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
    sendAbortRef.current?.abort();
    setSending(false);
    setSendResult(null);
    setSendError("");

    try {
      const meta = await fetchDatasetMeta(ref, controller.signal);
      setDataset(meta);
      // Metadata first, then the label-diverse images.
      loadHighlights(ref, controller, meta);
    } catch (metaErr) {
      if (metaErr.name !== "AbortError") setError(metaErr.message);
    } finally {
      setLoadingMeta(false);
    }
  }

  // Send the highlight images (with their label boxes) + prompt to the VLM.
  // Boxes ride along so the backend can burn them into the pixels the model
  // sees — matching the overlays shown in the UI.
  async function handleSend() {
    const images = highlights?.images || [];
    const payload = images
      .filter((image) => image.thumbnailUrl)
      .map((image) => ({
        url: image.thumbnailUrl,
        boxes: withBoxes
          ? (image.labels || [])
              .filter((label) => Array.isArray(label.bbox) && label.bbox.length >= 4)
              .map((label) => ({
                bbox: label.bbox,
                className: label.className ?? null,
                classId: label.classId ?? null,
              }))
          : [],
      }));
    if (!prompt.trim() || sending || !payload.length) return;

    sendAbortRef.current?.abort();
    const controller = new AbortController();
    sendAbortRef.current = controller;
    setSendResult(null);
    setSendError("");
    setSending(true);
    try {
      const result = await describeDataset(
        { prompt, provider, model, images: payload, mosaic },
        controller.signal,
      );
      setSendResult(result);
    } catch (err) {
      if (err.name !== "AbortError") setSendError(err.message);
    } finally {
      setSending(false);
    }
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
                sendError={sendError}
                onSend={handleSend}
                imageCount={highlights?.images?.length || 0}
                thumbnails={(highlights?.images || []).map((image) => image.thumbnailUrl).filter(Boolean)}
                mosaic={mosaic}
                setMosaic={setMosaic}
                withBoxes={withBoxes}
                setWithBoxes={setWithBoxes}
                hasBoxes={(highlights?.images || []).some((image) => (image.labels || []).length > 0)}
                classes={highlights?.classes || classNames}
                providers={providers}
                provider={provider}
                setProvider={setProvider}
                model={model}
                setModel={setModel}
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

function GenerateSection({
  prompt,
  setPrompt,
  sending,
  sendResult,
  sendError,
  onSend,
  imageCount,
  thumbnails,
  mosaic,
  setMosaic,
  withBoxes,
  setWithBoxes,
  hasBoxes,
  classes,
  providers,
  provider,
  setProvider,
  model,
  setModel,
}) {
  const willMosaic = mosaic && imageCount > 1;
  const selectedProvider = providers.find((item) => item.name === provider) || null;
  const models = selectedProvider?.models || [];
  const providerUnavailable = selectedProvider && !selectedProvider.available;
  const ready = imageCount > 0 && Boolean(provider) && !providerUnavailable;

  function handleProviderChange(name) {
    setProvider(name);
    const next = providers.find((item) => item.name === name);
    setModel(next?.default_model || next?.models?.[0] || "");
  }

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
        metadata to a vision model to auto-write a description.
      </p>

      <div className="dataset-model-row">
        <label>
          <span>Provider</span>
          <select value={provider} onChange={(event) => handleProviderChange(event.target.value)}>
            {providers.length ? (
              providers.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.label}
                  {item.available ? "" : " (no key)"}
                </option>
              ))
            ) : (
              <option value="">Loading…</option>
            )}
          </select>
        </label>
        <label>
          <span>Model</span>
          <select value={model} onChange={(event) => setModel(event.target.value)} disabled={!models.length}>
            {models.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

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
          {willMosaic ? `1 mosaic image (from ${imageCount})` : `${imageCount} images`}
          {hasBoxes && withBoxes ? " · boxed" : ""}
        </span>
        {classes.length ? (
          <span className="dataset-attach-chip">
            <Tag size={13} aria-hidden="true" />
            {classes.join(", ")}
          </span>
        ) : null}
      </div>

      {imageCount > 1 ? (
        <div className="dataset-mode" role="radiogroup" aria-label="How to attach the sample images">
          <span className="dataset-mode-title">Send the {imageCount} samples as…</span>
          <div className="dataset-mode-cards">
            <ModeCard
              active={!mosaic}
              onClick={() => setMosaic(false)}
              preview={<ThumbPreview thumbnails={thumbnails} variant="separate" />}
              icon={<Images size={14} aria-hidden="true" />}
              title={`${imageCount} separate images`}
              hint="Full detail — best for capable models"
            />
            <ModeCard
              active={mosaic}
              onClick={() => setMosaic(true)}
              preview={<ThumbPreview thumbnails={thumbnails} variant="mosaic" />}
              icon={<Grid2x2 size={14} aria-hidden="true" />}
              title="One 2×2 mosaic image"
              hint="Combines them into one — most compatible"
            />
          </div>
        </div>
      ) : null}

      {hasBoxes ? (
        <div className="dataset-mode" role="radiogroup" aria-label="Whether to draw label boxes">
          <span className="dataset-mode-title">Label boxes</span>
          <div className="dataset-mode-cards">
            <ModeCard
              active={withBoxes}
              onClick={() => setWithBoxes(true)}
              preview={<BoxPreview thumbnail={thumbnails[0]} withBox />}
              icon={<SquareDashed size={14} aria-hidden="true" />}
              title="With label boxes"
              hint="Draw each object's box + class into the image"
            />
            <ModeCard
              active={!withBoxes}
              onClick={() => setWithBoxes(false)}
              preview={<BoxPreview thumbnail={thumbnails[0]} />}
              icon={<ImageIcon size={14} aria-hidden="true" />}
              title="Without boxes"
              hint="Send the raw images only"
            />
          </div>
        </div>
      ) : null}

      <div className="dataset-generate-actions">
        <button
          type="button"
          className="eval-run-button"
          onClick={onSend}
          disabled={sending || !prompt.trim() || !ready}
        >
          {sending ? "Generating…" : "Send to model"}
        </button>
        {providerUnavailable ? (
          <span className="dataset-dummy-note">
            {selectedProvider.label} needs {(selectedProvider.env_keys || ["an API key"]).join(" or ")} on the backend.
          </span>
        ) : null}
      </div>

      {sendError ? <div className="eval-error">{sendError}</div> : null}

      {sendResult ? (
        <div className="dataset-generate-result">
          <div className="dataset-result-head">
            <span className="dataset-result-badge">
              {sendResult.provider} · {sendResult.model}
            </span>
            <span className="dataset-result-meta">
              {formatCost(sendResult.costUsd, sendResult.usage)}
              {sendResult.latencyMs ? ` · ${formatMs(sendResult.latencyMs)}` : ""}
              {sendResult.usage
                ? ` · ${sendResult.usage.inputTokens.toLocaleString()} in / ${sendResult.usage.outputTokens.toLocaleString()} out`
                : ""}
            </span>
          </div>
          <p className="dataset-description-out">{sendResult.description}</p>
        </div>
      ) : null}
    </section>
  );
}

function ModeCard({ active, onClick, preview, icon, title, hint }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      className={`dataset-mode-card ${active ? "is-active" : ""}`}
      onClick={onClick}
    >
      {preview}
      <span className="dataset-mode-text">
        <strong>
          {icon}
          {title}
        </strong>
        <span>{hint}</span>
      </span>
    </button>
  );
}

// Up-to-4 thumbnails, either spaced apart (separate) or tight (mosaic).
function ThumbPreview({ thumbnails, variant }) {
  const tiles = (thumbnails || []).slice(0, 4);
  return (
    <span className={`dataset-mode-preview ${variant}`} aria-hidden="true">
      {tiles.map((url, index) => (
        <img key={index} src={url} alt="" loading="lazy" />
      ))}
    </span>
  );
}

// One thumbnail, optionally with a box outline overlaid to illustrate "boxed".
function BoxPreview({ thumbnail, withBox }) {
  return (
    <span className="dataset-mode-preview single" aria-hidden="true">
      {thumbnail ? <img src={thumbnail} alt="" loading="lazy" /> : null}
      {withBox ? <span className="dataset-mode-box" /> : null}
    </span>
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

function formatCost(usd, usage) {
  if (usd == null) return usage ? "cost n/a" : "";
  if (usd === 0) return "$0";
  if (usd < 0.00001) return "<$0.00001";
  return `$${usd.toFixed(5)}`;
}

function formatMs(ms) {
  if (!ms) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
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
