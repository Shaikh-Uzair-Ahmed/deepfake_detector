import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const STEPS = [
  { id: "queued", label: "Queued" },
  { id: "extracting_frames", label: "Extracting" },
  { id: "detecting", label: "Detecting" },
  { id: "done", label: "Complete" },
];

const ACCEPT = "video/mp4,video/webm,video/quicktime,video/x-msvideo,image/jpeg,image/png,image/webp";

function pct(value) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function stepIndex(status) {
  if (status === "error") return -1;
  const i = STEPS.findIndex((s) => s.id === status);
  if (status === "done") return STEPS.length - 1;
  return i;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const pollRef = useRef(null);

  const loadHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth({ loaded: false, error: "Cannot reach API on port 8000." });
    }
  }, []);

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [previewUrl]);

  const chooseFile = (next) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(next);
    setJob(null);
    setError(null);
    if (!next) {
      setPreviewUrl(null);
      return;
    }
    if (next.type.startsWith("image/") || next.type.startsWith("video/")) {
      setPreviewUrl(URL.createObjectURL(next));
    } else {
      setPreviewUrl(null);
    }
  };

  const pollJob = async (jobId) => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error("Could not read job status.");
    const data = await res.json();
    setJob(data);
    if (data.status === "done" || data.status === "error") {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      setBusy(false);
      if (data.status === "error") setError(data.error || "Analysis failed.");
    }
  };

  const analyze = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    setJob(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/detect", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed.");
      setJob(data);
      pollRef.current = setInterval(() => {
        pollJob(data.job_id || data.id).catch((err) => {
          setError(err.message);
          setBusy(false);
          if (pollRef.current) clearInterval(pollRef.current);
        });
      }, 600);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  const reset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    chooseFile(null);
    setBusy(false);
  };

  const activeStep = stepIndex(job?.status);
  const verdict = job?.status === "done" ? job.label : null;
  const isVideo = file?.type.startsWith("video/");
  const healthOk = Boolean(health?.loaded);

  const modelLine = useMemo(() => {
    if (!health) return "Checking model…";
    if (!health.loaded) return health.error || "Checkpoint not loaded";
    const device = health.device || "cpu";
    return `${health.model} · ${health.backbone} · ${device}`;
  }, [health]);

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="kicker">Detection lab</p>
          <h1>Deepfake Detector</h1>
        </div>
        <div className={`health ${healthOk ? "ok" : "bad"}`}>
          <span className="dot" />
          <span>{modelLine}</span>
        </div>
      </header>

      <main className="grid">
        <section className="panel upload">
          <h2>Source media</h2>
          <p className="lede">
            Drop a video or still. VideoViT v2 samples face frames and classifies the clip as REAL or FAKE.
          </p>

          <div
            className={`drop ${dragOver ? "over" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const next = e.dataTransfer.files?.[0];
              if (next) chooseFile(next);
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              hidden
              onChange={(e) => chooseFile(e.target.files?.[0] || null)}
            />
            {previewUrl && isVideo && (
              <video className="media-preview" src={previewUrl} controls muted />
            )}
            {previewUrl && !isVideo && (
              <img className="media-preview" src={previewUrl} alt="Selected media preview" />
            )}
            {!previewUrl && (
              <div className="drop-copy">
                <strong>Upload source video or image</strong>
                <span>MP4, WebM, MOV, JPG, PNG · max 200 MB</span>
              </div>
            )}
          </div>

          {file && (
            <p className="file-meta">
              {file.name} · {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          )}

          <div className="actions">
            <button
              className="primary"
              disabled={!file || busy || !healthOk}
              onClick={analyze}
              type="button"
            >
              {busy ? "Analyzing…" : "Analyze authenticity"}
            </button>
            <button className="ghost" type="button" onClick={reset} disabled={busy && !job}>
              Reset
            </button>
          </div>
        </section>

        <section className="panel result">
          <h2>Analysis</h2>

          <ol className={`steps ${job?.status === "error" ? "failed" : ""}`}>
            {STEPS.map((step, i) => (
              <li
                key={step.id}
                className={
                  job?.status === "error" && i === Math.max(activeStep, 0)
                    ? "err"
                    : job?.status === "done" || i < activeStep
                      ? "done"
                      : i === activeStep
                        ? "active"
                        : ""
                }
              >
                {step.label}
              </li>
            ))}
          </ol>

          <p className="status-msg">
            {error || job?.message || "Waiting for an upload."}
          </p>

          {verdict && (
            <div className={`verdict ${verdict.toLowerCase()}`}>
              <p className="verdict-label">{verdict}</p>
              <p className="verdict-conf">{pct(job.confidence)} confidence</p>
              <div className="bars">
                <div>
                  <span>Real {pct(job.real_probability)}</span>
                  <div className="bar">
                    <i style={{ width: `${(job.real_probability || 0) * 100}%` }} />
                  </div>
                </div>
                <div>
                  <span>Fake {pct(job.fake_probability)}</span>
                  <div className="bar fake">
                    <i style={{ width: `${(job.fake_probability || 0) * 100}%` }} />
                  </div>
                </div>
              </div>
              <p className="crop-note">
                {job.used_face_crop
                  ? "Face crops (MTCNN ≥ 0.95) — same gate as training."
                  : "No high-confidence face found; full frames were used."}
              </p>
            </div>
          )}

          {job?.frames?.length > 0 && (
            <div className="frames">
              <h3>Frames the model saw</h3>
              <div className="strip">
                {job.frames.map((frame) => (
                  <figure key={frame.index}>
                    <img src={frame.url} alt={`Analyzed frame ${frame.index + 1}`} />
                    <figcaption>{String(frame.index + 1).padStart(2, "0")}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
