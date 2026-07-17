import "./style.css";
import { CHUNK_SAMPLES, decodeAudioFile, logMelSpectrogram, SAMPLE_RATE, type MelSpectrogram } from "./audio";
import { clearModelCache, enableModelCache, hasCompleteModelCache, modelCacheInfo } from "./model-cache";
import { NemotronClient } from "./nemotron-client";

const byId = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
};

const language = byId<HTMLSelectElement>("language");
const fileInput = byId<HTMLInputElement>("audio-file");
const microphoneButton = byId<HTMLButtonElement>("microphone");
const microphoneLabel = byId<HTMLSpanElement>("microphone-label");
const runButton = byId<HTMLButtonElement>("transcribe");
const runLabel = byId<HTMLSpanElement>("transcribe-label");
const clearCacheButton = byId<HTMLButtonElement>("clear-cache");
const status = byId<HTMLElement>("status");
const transcript = byId<HTMLParagraphElement>("transcript");
const errorMessage = byId<HTMLParagraphElement>("error-message");
const loadTime = byId<HTMLElement>("load-time");
const durationMetric = byId<HTMLElement>("audio-time");
const chunkCount = byId<HTMLElement>("chunk-count");
const chunkTime = byId<HTMLElement>("chunk-time");
const cacheStatus = byId<HTMLElement>("cache-status");
const audioElement = byId<HTMLAudioElement>("native-audio");
const playbackButton = byId<HTMLButtonElement>("playback");
const playbackTime = byId<HTMLElement>("playback-time");
const playbackDuration = byId<HTMLElement>("transport-duration");
const transportRange = byId<HTMLInputElement>("transport-range");
const signalFrame = document.querySelector<HTMLElement>(".signal-frame")!;
const playhead = byId<HTMLElement>("playhead");
const waveformCanvas = byId<HTMLCanvasElement>("waveform");
const waveformContext = waveformCanvas.getContext("2d", { desynchronized: true })!;
const waveformBufferCanvas = byId<HTMLCanvasElement>("waveform-buffer");
const waveformBufferContext = waveformBufferCanvas.getContext("2d", { desynchronized: true })!;
const waveformFront = { canvas: waveformCanvas, context: waveformContext };
const waveformBack = { canvas: waveformBufferCanvas, context: waveformBufferContext };

const model = new NemotronClient();
const cacheReady = enableModelCache().catch(() => false);
let loaded = false;
let busy = false;
let transcriptionAbort: AbortController | null = null;
let recording = false;
let preparedMel: MelSpectrogram | null = null;
let visualAudio: Float32Array<ArrayBufferLike> | null = null;
let audioUrl: string | null = null;
let microphone: MicrophoneSession | null = null;
let microphoneChunks = 0;
let microphoneStarted = 0;
let microphoneStopping = false;
let visualizationFrame: number | null = null;
// Live waveform scrolled by a wall-clock transform over cached, content-locked bins.
const VIEW_SAMPLES = SAMPLE_RATE * 2; // 2s visible window
const BIN_SAMPLES = 32; // ~2ms per bin, finer than a pixel
const VIEW_BINS = VIEW_SAMPLES / BIN_SAMPLES;
const LIVE_LATENCY_SAMPLES = SAMPLE_RATE * 0.12; // draw 120ms behind real-time
const LIVE_REPAINT_MS = 100;
const RING_BINS = VIEW_BINS + Math.ceil(LIVE_LATENCY_SAMPLES / BIN_SAMPLES) + 16;
let binMin: Float32Array | null = null;
let binMax: Float32Array | null = null;
let liveSamples = 0; // total samples fed
let liveBin = -1; // absolute index of the bin currently filling
let liveCursor = 0; // smooth display position (right edge), in samples
let livePaintCursor = 0;
let livePaintAt = -Infinity;
let visibleWaveform = waveformFront;
let hiddenWaveform = waveformBack;
let pendingPaintCursor: number | null = null;
let cacheRefreshing = false;
let cacheProgressDeadline = 0;
let transcriptTarget = "";
let transcriptTimer: number | null = null;
let transcriptVisibleLength = transcript.textContent?.length ?? 0;
let transcriptHeight = transcript.scrollHeight;
let transcriptDelay = 200;

interface MicrophoneSession {
  context: AudioContext;
  stream: MediaStream;
  source: MediaStreamAudioSourceNode;
  worklet: AudioWorkletNode;
}

interface CacheProgressMessage {
  type: "model-cache-progress";
  url: string;
  loaded: number;
  total: number;
  done: boolean;
  error: boolean;
}

void initializePage();
window.addEventListener("resize", () => {
  if (recording) drawLive();
  else drawWaveform();
  updateTranscriptSize();
});
navigator.serviceWorker?.addEventListener("message", (event: MessageEvent<CacheProgressMessage>) => {
  const progress = event.data;
  if (progress?.type !== "model-cache-progress") return;
  const file = decodeURIComponent(new URL(progress.url).pathname.split("/").pop() ?? "model");
  if (progress.done) {
    if (progress.error) {
      cacheProgressDeadline = performance.now() + 3000;
      cacheStatus.textContent = `${file} · cache failed`;
      return;
    }
    cacheProgressDeadline = 0;
    void refreshCacheStatus();
    return;
  }
  cacheProgressDeadline = performance.now() + 1500;
  cacheStatus.textContent = progress.total
    ? `${file} · ${formatBytes(progress.loaded)}/${formatBytes(progress.total)}`
    : `${file} · ${formatBytes(progress.loaded)}`;
});

async function initializePage(): Promise<void> {
  drawWaveform();
  busy = true;
  updateControls();
  try {
    setStatus("checking model cache");
    const cacheEnabled = await cacheReady;
    await refreshCacheStatus();
    if (!cacheEnabled || !(await hasCompleteModelCache().catch(() => false))) {
      setStatus("standing by");
      return;
    }
    setStatus("restoring cached model");
    await ensureModelLoaded();
    setStatus("model ready");
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateControls();
  }
}

async function ensureModelLoaded(): Promise<void> {
  if (loaded) return;
  let cacheEnabled = await cacheReady;
  if (!cacheEnabled && "serviceWorker" in navigator) {
    cacheEnabled = await enableModelCache().catch(() => false);
  }
  if ("serviceWorker" in navigator && !cacheEnabled) {
    throw new Error("The cache worker could not control this page. Refresh and retry; no model download was started.");
  }
  const started = performance.now();
  await model.load(setStatus);
  loaded = true;
  loadTime.textContent = formatDuration(performance.now() - started);
  await refreshCacheStatus();
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file || busy || recording) return;
  busy = true;
  clearError();
  updateControls();
  preparedMel = null;
  visualAudio = null;
  clearPlayback();
  drawWaveform();
  try {
    setStatus("decoding audio");
    const audio = await decodeAudioFile(file);
    setStatus("extracting mel features");
    const paddedSamples = Math.ceil(audio.length / CHUNK_SAMPLES) * CHUNK_SAMPLES;
    preparedMel = logMelSpectrogram(audio, paddedSamples);
    visualAudio = audio;
    setPlayback(URL.createObjectURL(file), audio.length / SAMPLE_RATE);
    setTranscript(`Signal ready: ${file.name}`, true);
    chunkCount.textContent = "0";
    chunkTime.textContent = "—";
    setStatus("signal locked");
    drawWaveform();
  } catch (error) {
    showError(error);
  } finally {
    fileInput.value = "";
    busy = false;
    updateControls();
  }
});

runButton.addEventListener("click", async () => {
  if (transcriptionAbort) {
    transcriptionAbort.abort();
    setStatus("stopping");
    updateControls();
    return;
  }
  if (!preparedMel || busy || recording) return;
  const abort = new AbortController();
  transcriptionAbort = abort;
  busy = true;
  clearError();
  updateControls();
  setTranscript("Preparing model…", true);
  try {
    await ensureModelLoaded();
    abort.signal.throwIfAborted();
    setStatus("running streaming encoder");
    const text = await model.transcribe(preparedMel, Number(language.value), (progress) => {
      setTranscript(progress.text || "(no non-blank token yet)", !progress.text, !!progress.text);
      chunkCount.textContent = `${progress.chunk}/${progress.totalChunks}`;
      chunkTime.textContent = formatDuration(progress.elapsedMs);
      setStatus(`decoding chunk`);
    }, abort.signal);
    setTranscript(text || "(no speech decoded)", !text, !!text);
    setStatus("transcript resolved");
  } catch (error) {
    if (abort.signal.aborted) setStatus("stopped");
    else showError(error);
  } finally {
    transcriptionAbort = null;
    busy = false;
    updateControls();
  }
});

microphoneButton.addEventListener("click", async () => {
  if (recording) {
    await stopMicrophone();
    return;
  }
  if (busy) return;
  busy = true;
  clearError();
  updateControls();
  try {
    setTranscript("Preparing model…", true);
    await ensureModelLoaded();
    microphone = await createMicrophone();
    model.startStream(Number(language.value));
    clearPlayback();
    preparedMel = null;
    visualAudio = null;
    binMin = new Float32Array(RING_BINS);
    binMax = new Float32Array(RING_BINS);
    liveSamples = 0;
    liveBin = -1;
    liveCursor = 0;
    resetLiveWaveforms();
    microphoneChunks = 0;
    microphoneStarted = performance.now();
    microphone.worklet.port.onmessage = ({ data }: MessageEvent<{ samples?: ArrayBuffer }>) => {
      if (!data.samples) return;
      const samples = new Float32Array(data.samples);
      queueVisualization(samples);
      void processMicrophoneSamples(samples).catch(handleMicrophoneError);
    };
    microphone.worklet.addEventListener(
      "processorerror",
      () => void handleMicrophoneError(new Error("Microphone audio processor stopped")),
      { once: true },
    );
    microphone.stream.getAudioTracks()[0]?.addEventListener(
      "ended",
      () => void handleMicrophoneError(new Error("Microphone input ended")),
      { once: true },
    );
    recording = true;
    startVisualization();
    setTranscript("Listening…", true);
    chunkCount.textContent = "0";
    chunkTime.textContent = "—";
    setStatus("recording");
    drawLive();
  } catch (error) {
    showError(error);
    await closeMicrophone();
  } finally {
    busy = false;
    updateControls();
  }
});

async function processMicrophoneSamples(samples: Float32Array): Promise<void> {
  const progress = await model.pushAudio(samples, false, showPartialTranscript);
  if (progress.chunks === 0) return;
  microphoneChunks += progress.chunks;
  setTranscript(progress.text || "(no non-blank token yet)", !progress.text, !!progress.text);
  chunkCount.textContent = String(microphoneChunks);
  chunkTime.textContent = formatDuration(progress.elapsedMs);
}

function showPartialTranscript(text: string): void {
  setTranscript(text, false, true);
}

async function handleMicrophoneError(error: unknown): Promise<void> {
  showError(error);
  recording = false;
  stopVisualization();
  if (microphoneStopping) return;
  busy = true;
  updateControls();
  if (microphone) microphone.worklet.port.onmessage = null;
  try {
    await closeMicrophone();
  } finally {
    busy = false;
    updateControls();
  }
}

async function stopMicrophone(): Promise<void> {
  if (!microphone || busy) return;
  microphoneStopping = true;
  busy = true;
  recording = false;
  stopVisualization();
  updateControls();
  setStatus("flushing audio");
  try {
    microphone.source.disconnect();
    microphone.worklet.port.postMessage("flush");
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error("Microphone flush timed out")), 2000);
      const previous = microphone!.worklet.port.onmessage;
      microphone!.worklet.port.onmessage = (event) => {
        if (event.data?.flushed) {
          clearTimeout(timeout);
          resolve();
        }
        else previous?.call(microphone!.worklet.port, event);
      };
    });
    // AudioWorklet messages and worker requests are ordered; this final request
    // runs after every sample message sent before `flushed`.
    const final = await model.pushAudio(new Float32Array(), true, showPartialTranscript);
    microphoneChunks += final.chunks;
    setTranscript(final.text || "(no speech decoded)", !final.text, !!final.text);
    chunkCount.textContent = String(microphoneChunks);
    if (final.chunks) chunkTime.textContent = formatDuration(final.elapsedMs);
    liveCursor = liveSamples;
    drawLive();
    setStatus("transcript resolved");
  } catch (error) {
    showError(error);
  } finally {
    try {
      await closeMicrophone();
    } finally {
      microphoneStopping = false;
      busy = false;
      updateControls();
    }
  }
}

clearCacheButton.addEventListener("click", async () => {
  if (busy || recording || !confirm("Clear the cached model files? The current loaded session will keep working.")) return;
  busy = true;
  clearError();
  updateControls();
  try {
    await clearModelCache();
    cacheProgressDeadline = 0;
    await refreshCacheStatus();
    setStatus("model cache cleared");
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateControls();
  }
});

playbackButton.addEventListener("click", async () => {
  if (!audioElement.src) return;
  try {
    if (audioElement.paused || audioElement.ended) await audioElement.play();
    else audioElement.pause();
  } catch (error) {
    showError(error);
  }
});
audioElement.addEventListener("play", updateTransport);
audioElement.addEventListener("pause", updateTransport);
audioElement.addEventListener("ended", updateTransport);
audioElement.addEventListener("timeupdate", updateTransport);
transportRange.addEventListener("input", () => {
  audioElement.currentTime = Number(transportRange.value);
  updateTransport();
});
signalFrame.addEventListener("pointerdown", (event) => {
  if (!audioElement.src || !audioElement.duration) return;
  const rect = signalFrame.getBoundingClientRect();
  audioElement.currentTime = ((event.clientX - rect.left) / rect.width) * audioElement.duration;
  updateTransport();
});

function setPlayback(url: string, duration: number): void {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = url;
  audioElement.src = url;
  playbackButton.disabled = false;
  transportRange.disabled = false;
  transportRange.max = String(duration);
  durationMetric.textContent = formatTime(duration);
  playbackDuration.textContent = formatTime(duration);
  playhead.hidden = false;
  updateTransport();
}

function clearPlayback(): void {
  audioElement.pause();
  audioElement.removeAttribute("src");
  audioElement.load();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  playbackButton.disabled = true;
  transportRange.disabled = true;
  transportRange.value = "0";
  playbackTime.textContent = "0:00";
  playbackDuration.textContent = "0:00";
  durationMetric.textContent = "0:00";
  playhead.hidden = true;
}

function updateTransport(): void {
  const current = Number.isFinite(audioElement.currentTime) ? audioElement.currentTime : 0;
  const duration = Number.isFinite(audioElement.duration) ? audioElement.duration : Number(transportRange.max);
  transportRange.value = String(current);
  transportRange.style.setProperty("--progress", `${duration ? (100 * current) / duration : 0}%`);
  playbackTime.textContent = formatTime(current);
  playbackButton.classList.toggle("is-playing", !audioElement.paused);
  playhead.style.left = `${duration ? (100 * current) / duration : 0}%`;
}

function updateControls(): void {
  fileInput.disabled = busy || recording;
  runButton.disabled = recording || !preparedMel || (busy && !transcriptionAbort) || !!transcriptionAbort?.signal.aborted;
  runButton.classList.toggle("transcribing", !!transcriptionAbort);
  runLabel.textContent = transcriptionAbort?.signal.aborted ? "Stopping" : transcriptionAbort ? "Stop" : "Run";
  microphoneButton.disabled = busy;
  microphoneButton.classList.toggle("recording", recording);
  microphoneLabel.textContent = recording ? "Stop" : "Record";
  language.disabled = busy || recording;
  clearCacheButton.disabled = busy || recording;
}

async function refreshCacheStatus(): Promise<void> {
  if (cacheRefreshing || performance.now() < cacheProgressDeadline) return;
  cacheRefreshing = true;
  try {
    cacheStatus.textContent = await modelCacheInfo().catch(() => "unavailable");
  } finally {
    cacheRefreshing = false;
  }
}

function setStatus(message: string): void {
  status.textContent = message;
  status.classList.remove("status-error");
}

function setTranscript(message: string, empty = false, animated = false): void {
  if (animated) {
    const wasEmpty = transcript.classList.contains("empty");
    transcript.classList.remove("empty");
    transcriptTarget = message;
    const current = wasEmpty ? "" : transcript.textContent ?? "";
    if (wasEmpty) {
      transcript.textContent = "";
      transcriptVisibleLength = 0;
    }
    if (!message.startsWith(current)) {
      transcript.textContent = message;
      transcriptVisibleLength = message.length;
      updateTranscriptSize();
    } else if (transcriptTimer === null && current !== message) {
      transcriptVisibleLength = current.length;
      transcriptDelay = 200;
      advanceTranscript();
    }
    return;
  }
  if (transcriptTimer !== null) clearTimeout(transcriptTimer);
  transcriptTimer = null;
  transcriptTarget = message;
  transcript.textContent = message;
  transcriptVisibleLength = message.length;
  transcript.classList.toggle("empty", empty);
  updateTranscriptSize();
}

function advanceTranscript(): void {
  const codePoint = transcriptTarget.codePointAt(transcriptVisibleLength);
  if (codePoint === undefined) {
    transcriptTimer = null;
    updateTranscriptSize();
    return;
  }
  const nextLength = transcriptVisibleLength + (codePoint > 0xffff ? 2 : 1);
  const nextCharacter = transcriptTarget.slice(transcriptVisibleLength, nextLength);
  const textNode = transcript.firstChild;
  if (textNode instanceof Text) textNode.appendData(nextCharacter);
  else transcript.append(nextCharacter);
  transcriptVisibleLength = nextLength;
  if (transcript.scrollHeight !== transcriptHeight) updateTranscriptSize();
  if (transcriptVisibleLength < transcriptTarget.length) scheduleNextTranscriptCharacter();
  else {
    transcriptTimer = null;
    updateTranscriptSize();
  }
}

function scheduleNextTranscriptCharacter(): void {
  const backlog = transcriptTarget.length - transcriptVisibleLength;
  const target = Math.max(8, Math.min(200, 350 / backlog));
  transcriptDelay += (target - transcriptDelay) * 0.25;
  transcriptTimer = window.setTimeout(advanceTranscript, transcriptDelay);
}

function updateTranscriptSize(): void {
  transcript.classList.remove("compact", "dense", "tight", "packed");
  const active = !transcript.classList.contains("empty");
  const lineHeight = Number.parseFloat(getComputedStyle(transcript).lineHeight);
  transcript.classList.toggle("compact", active && transcript.scrollHeight > lineHeight * 4);
  const compactLineHeight = Number.parseFloat(getComputedStyle(transcript).lineHeight);
  transcript.classList.toggle("dense", active && transcript.scrollHeight > compactLineHeight * 8);
  const denseLineHeight = Number.parseFloat(getComputedStyle(transcript).lineHeight);
  transcript.classList.toggle("tight", active && transcript.scrollHeight > denseLineHeight * 12);
  const tightLineHeight = Number.parseFloat(getComputedStyle(transcript).lineHeight);
  transcript.classList.toggle("packed", active && transcript.scrollHeight > tightLineHeight * 16);
  transcriptHeight = transcript.scrollHeight;
}

function clearError(): void {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  status.textContent = "fault";
  status.classList.add("status-error");
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  console.error(error);
}

function formatDuration(milliseconds: number): string {
  return milliseconds < 1000 ? `${milliseconds.toFixed(0)} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function formatBytes(bytes: number): string {
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
    : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function queueVisualization(samples: Float32Array): void {
  if (!binMin || !binMax) return;
  // Fold samples into fixed, content-locked bins: bin b always covers the same absolute
  // sample range, so its min/max never re-aliases as the view scrolls (no shimmer).
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const absBin = Math.floor(liveSamples / BIN_SAMPLES);
    const ring = absBin % RING_BINS;
    if (absBin !== liveBin) {
      binMin[ring] = s;
      binMax[ring] = s;
      liveBin = absBin;
    } else {
      if (s < binMin[ring]) binMin[ring] = s;
      if (s > binMax[ring]) binMax[ring] = s;
    }
    liveSamples++;
  }
}

function startVisualization(): void {
  stopVisualization();
  const tick = () => {
    if (!recording) {
      visualizationFrame = null;
      return;
    }
    drawLive();
    const elapsed = formatTime((performance.now() - microphoneStarted) / 1000);
    if (durationMetric.textContent !== elapsed) durationMetric.textContent = elapsed;
    visualizationFrame = requestAnimationFrame(tick);
  };
  visualizationFrame = requestAnimationFrame(tick);
}

function stopVisualization(): void {
  if (visualizationFrame !== null) cancelAnimationFrame(visualizationFrame);
  visualizationFrame = null;
}

function strokeBaseline(context: CanvasRenderingContext2D, width: number, mid: number): void {
  context.strokeStyle = "#55f0ff";
  context.globalAlpha = .55;
  context.beginPath();
  context.moveTo(0, mid);
  context.lineTo(width, mid);
  context.stroke();
  context.globalAlpha = 1;
}

function drawLive(): void {
  if (!binMin || !binMax) return;
  const now = performance.now();
  // Absolute time keeps velocity constant; following 320-sample arrivals would add a
  // small correction step every 20ms. Latency keeps this clock behind received audio.
  if (recording) {
    liveCursor = Math.max(0, ((now - microphoneStarted) / 1000) * SAMPLE_RATE - LIVE_LATENCY_SAMPLES);
  }

  const viewportWidth = Math.max(1, signalFrame.clientWidth);
  const cacheWidth = Math.ceil(viewportWidth * (1 + LIVE_LATENCY_SAMPLES / VIEW_SAMPLES));
  const height = Math.max(1, waveformCanvas.clientHeight);

  if (!recording || livePaintAt === -Infinity) {
    pendingPaintCursor = null;
    livePaintCursor = liveCursor;
    livePaintAt = now;
    visibleWaveform.canvas.style.transform = "translate3d(0, 0, 0)";
    paintLiveWaveform(visibleWaveform, livePaintCursor, viewportWidth, cacheWidth, height);
    if (!recording) return;
  } else if (pendingPaintCursor !== null) {
    [visibleWaveform, hiddenWaveform] = [hiddenWaveform, visibleWaveform];
    livePaintCursor = pendingPaintCursor;
    pendingPaintCursor = null;
    const offset = Math.max(0, (liveCursor - livePaintCursor) * viewportWidth / VIEW_SAMPLES);
    visibleWaveform.canvas.style.transform = `translate3d(${-offset}px, 0, 0)`;
    visibleWaveform.canvas.style.opacity = "1";
    hiddenWaveform.canvas.style.opacity = "0";
  }

  const offset = Math.max(0, (liveCursor - livePaintCursor) * viewportWidth / VIEW_SAMPLES);
  visibleWaveform.canvas.style.transform = `translate3d(${-offset}px, 0, 0)`;

  const repaint = now - livePaintAt >= LIVE_REPAINT_MS
    || visibleWaveform.canvas.width !== cacheWidth || visibleWaveform.canvas.height !== height;
  if (repaint && pendingPaintCursor === null) {
    pendingPaintCursor = liveCursor;
    livePaintAt = now;
    hiddenWaveform.canvas.style.transform = "translate3d(0, 0, 0)";
    paintLiveWaveform(hiddenWaveform, pendingPaintCursor, viewportWidth, cacheWidth, height);
  }
}

function paintLiveWaveform(
  surface: typeof waveformFront,
  paintCursor: number,
  viewportWidth: number,
  cacheWidth: number,
  height: number,
): void {
  const { canvas, context } = surface;
  if (canvas.width !== cacheWidth || canvas.height !== height) {
    canvas.width = cacheWidth;
    canvas.height = height;
    canvas.style.width = `${cacheWidth}px`;
  }
  context.clearRect(0, 0, cacheWidth, height);
  const mid = height / 2;
  const anchorBin = paintCursor / BIN_SAMPLES;
  const pixelsPerBin = viewportWidth / VIEW_BINS;
  const firstBin = Math.max(0, Math.ceil(anchorBin - VIEW_BINS));
  const rightBin = Math.min(
    Math.floor(liveSamples / BIN_SAMPLES),
    Math.floor((paintCursor + LIVE_LATENCY_SAMPLES) / BIN_SAMPLES),
  );
  context.strokeStyle = "#d8f23d";
  context.lineWidth = 1.4;
  context.beginPath();
  for (let b = firstBin; b <= rightBin; b++) {
    const ring = b % RING_BINS;
    const x = viewportWidth - (anchorBin - b) * pixelsPerBin;
    context.moveTo(x, mid + binMin![ring] * mid * 1.42);
    context.lineTo(x, mid + binMax![ring] * mid * 1.42);
  }
  context.stroke();
  strokeBaseline(context, cacheWidth, mid);
}

function resetLiveWaveforms(): void {
  visibleWaveform = waveformFront;
  hiddenWaveform = waveformBack;
  pendingPaintCursor = null;
  livePaintCursor = 0;
  livePaintAt = -Infinity;
  waveformCanvas.style.opacity = "1";
  waveformBufferCanvas.style.opacity = "0";
  waveformCanvas.style.transform = "";
  waveformBufferCanvas.style.transform = "";
}

function setupCanvas() {
  // 1x backing store (rough envelope): less per-frame GPU raster competing with inference.
  resetLiveWaveforms();
  waveformCanvas.style.width = "";
  const width = Math.max(1, waveformCanvas.clientWidth);
  const height = Math.max(1, waveformCanvas.clientHeight);
  if (waveformCanvas.width !== width || waveformCanvas.height !== height) {
    waveformCanvas.width = width;
    waveformCanvas.height = height;
  }
  return { context: waveformContext, width, height };
}

function drawWaveform(): void {
  const { context, width, height } = setupCanvas();
  context.clearRect(0, 0, width, height);
  if (!visualAudio?.length) return;

  const mid = height / 2;
  context.strokeStyle = "#d8f23d";
  context.lineWidth = 1.4;
  context.beginPath();
  for (let x = 0; x < width; x++) {
    const start = Math.floor((x / width) * visualAudio.length);
    const end = Math.max(start + 1, Math.floor(((x + 1) / width) * visualAudio.length));
    let min = 1;
    let max = -1;
    for (let i = start; i < end; i++) {
      min = Math.min(min, visualAudio[i] ?? 0);
      max = Math.max(max, visualAudio[i] ?? 0);
    }
    context.moveTo(x, mid + min * mid * 1.42);
    context.lineTo(x, mid + max * mid * 1.42);
  }
  context.stroke();
  strokeBaseline(context, width, mid);
}

async function createMicrophone(): Promise<MicrophoneSession> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: SAMPLE_RATE, echoCancellation: true, noiseSuppression: true },
  });
  let context: AudioContext | null = null;
  try {
    context = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (context.sampleRate !== SAMPLE_RATE) throw new Error(`Browser returned ${context.sampleRate}Hz; 16000Hz is required`);
    const moduleUrl = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
    try { await context.audioWorklet.addModule(moduleUrl); }
    finally { URL.revokeObjectURL(moduleUrl); }
    const source = context.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(context, "pcm-capture", { numberOfOutputs: 1, outputChannelCount: [1] });
    source.connect(worklet);
    worklet.connect(context.destination);
    await context.resume();
    return { context, stream, source, worklet };
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    if (context && context.state !== "closed") await context.close();
    throw error;
  }
}

async function closeMicrophone(): Promise<void> {
  if (!microphone) return;
  const session = microphone;
  microphone = null;
  session.worklet.port.onmessage = null;
  for (const track of session.stream.getTracks()) {
    try { track.stop(); } catch { /* already stopped */ }
  }
  try { session.worklet.disconnect(); } catch { /* already disconnected */ }
  if (session.context.state !== "closed") await session.context.close().catch(() => undefined);
}

const workletSource = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(320);
    this.length = 0;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        this.send();
        this.port.postMessage({ flushed: true });
      }
    };
  }
  send() {
    if (!this.length) return;
    const samples = this.buffer.slice(0, this.length);
    this.port.postMessage({ samples: samples.buffer }, [samples.buffer]);
    this.length = 0;
  }
  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    let offset = 0;
    while (offset < input.length) {
      const count = Math.min(input.length - offset, this.buffer.length - this.length);
      this.buffer.set(input.subarray(offset, offset + count), this.length);
      this.length += count;
      offset += count;
      if (this.length === this.buffer.length) this.send();
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;
