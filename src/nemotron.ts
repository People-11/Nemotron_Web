import * as ort from "onnxruntime-web/webgpu";
import { CHUNK_FRAMES, CHUNK_SAMPLES, HOP_LENGTH, LOG_ZERO_GUARD, logMelSpectrogram, N_FFT, N_MELS, type MelSpectrogram } from "./audio";

const MODEL_ROOT =
  "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/8364d9e2dd9da23789b480bdbba9e423717e42ee";
const PRE_ENCODE_CACHE = 9;
const HIDDEN_SIZE = 1024;
const ENCODER_LAYERS = 24;
const LEFT_CONTEXT = 56;
const CONV_CONTEXT = 8;
const DECODER_LAYERS = 2;
const DECODER_HIDDEN = 640;
const BLANK_ID = 13_087;
const MAX_SYMBOLS_PER_STEP = 10;

export interface ChunkProgress {
  chunk: number;
  totalChunks: number;
  text: string;
  elapsedMs: number;
}

export interface StreamProgress {
  chunks: number;
  text: string;
  elapsedMs: number;
}

type Sessions = {
  encoder: ort.InferenceSession;
  decoder: ort.InferenceSession;
  joint: ort.InferenceSession;
};

export class NemotronBrowserASR {
  private sessions: Sessions | null = null;
  private vocabulary: string[] = [];
  private transcriptText = "";
  private cacheChannel!: ort.Tensor;
  private cacheTime!: ort.Tensor;
  private cacheLength!: ort.Tensor;
  private hidden!: ort.Tensor;
  private cell!: ort.Tensor;
  private lastToken = BLANK_ID;
  private streamAudio: Float32Array<ArrayBufferLike> = new Float32Array();
  private streamProcessedSamples = 0;
  private streamLanguageId = 101;

  async load(onStatus: (message: string) => void): Promise<void> {
    if (this.sessions) return;
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;
    const hasWebGpu = "gpu" in navigator;

    onStatus("loading tokenizer");
    const tokenizer = await fetchJson(`${MODEL_ROOT}/tokenizer.json`);
    this.vocabulary = tokenizer.model.vocab.map((entry: [string, number]) => entry[0]);

    if (hasWebGpu) {
      try {
        this.sessions = await createSessions(["webgpu"], true, onStatus);
      } catch (error) {
        console.warn("WebGPU initialization failed; retrying with WASM", error);
        onStatus("WebGPU failed; retrying with WASM");
        this.sessions = await createSessions(["wasm"], false, onStatus);
      }
    } else {
      onStatus("WebGPU unavailable; using WASM");
      this.sessions = await createSessions(["wasm"], false, onStatus);
    }
    this.reset();
  }

  async transcribe(
    mel: MelSpectrogram,
    languageId: number,
    onChunk: (progress: ChunkProgress) => void,
    signal?: AbortSignal,
  ): Promise<string> {
    if (!this.sessions) throw new Error("Model is not loaded");
    this.reset();
    const totalChunks = Math.ceil(mel.samples / CHUNK_SAMPLES);

    for (let chunk = 0; chunk < totalChunks; chunk++) {
      signal?.throwIfAborted();
      const started = performance.now();
      const frameStart = chunk * CHUNK_FRAMES;
      const chunkSamples = Math.min(CHUNK_SAMPLES, mel.samples - chunk * CHUNK_SAMPLES);
      const mainFrames = Math.min(CHUNK_FRAMES, Math.floor(chunkSamples / HOP_LENGTH) + 1);
      await this.runFeatureChunk(mel, frameStart, mainFrames, languageId);
      signal?.throwIfAborted();

      onChunk({
        chunk: chunk + 1,
        totalChunks,
        text: this.transcriptText,
        elapsedMs: performance.now() - started,
      });
    }

    return this.transcriptText;
  }

  startStream(languageId: number): void {
    if (!this.sessions) throw new Error("Model is not loaded");
    this.reset();
    this.streamAudio = new Float32Array();
    this.streamProcessedSamples = 0;
    this.streamLanguageId = languageId;
  }

  async pushAudio(
    samples: Float32Array,
    flush = false,
    onPartial?: (text: string) => void,
  ): Promise<StreamProgress> {
    if (!this.sessions) throw new Error("Model is not loaded");
    this.streamAudio = append(this.streamAudio, samples);
    const pendingSamples = this.streamAudio.length - this.streamProcessedSamples;
    const chunksToRun = flush
      ? Math.ceil(pendingSamples / CHUNK_SAMPLES)
      : Math.floor(pendingSamples / CHUNK_SAMPLES);
    if (chunksToRun === 0) {
      return { chunks: 0, text: this.transcriptText, elapsedMs: 0 };
    }

    const started = performance.now();
    const paddedSamples = flush
      ? this.streamProcessedSamples + chunksToRun * CHUNK_SAMPLES
      : this.streamAudio.length;
    const mel = logMelSpectrogram(this.streamAudio, paddedSamples);
    let partialSent = false;
    for (let chunk = 0; chunk < chunksToRun; chunk++) {
      const frameStart = Math.floor(this.streamProcessedSamples / HOP_LENGTH);
      const chunkSamples = Math.min(CHUNK_SAMPLES, mel.samples - this.streamProcessedSamples);
      const mainFrames = Math.min(CHUNK_FRAMES, Math.floor(chunkSamples / HOP_LENGTH) + 1);
      await this.runFeatureChunk(mel, frameStart, mainFrames, this.streamLanguageId, () => {
        if (!partialSent) {
          partialSent = true;
          onPartial?.(this.transcriptText);
        }
      });
      this.streamProcessedSamples += chunkSamples;
    }

    this.trimStreamBuffer();
    return { chunks: chunksToRun, text: this.transcriptText, elapsedMs: performance.now() - started };
  }

  private async runFeatureChunk(
    mel: MelSpectrogram,
    frameStart: number,
    mainFrames: number,
    languageId: number,
    onToken?: () => void,
  ): Promise<void> {
    if (!this.sessions) throw new Error("Model is not loaded");
    const features = buildFeatureChunk(mel, frameStart, mainFrames);
    const audioSignal = new ort.Tensor("float32", features, [1, PRE_ENCODE_CACHE + CHUNK_FRAMES, N_MELS]);
    const length = int64Tensor(PRE_ENCODE_CACHE + CHUNK_FRAMES);
    const langId = int64Tensor(languageId);
    let result: ort.InferenceSession.OnnxValueMapType;
    try {
      result = await this.sessions.encoder.run({
        audio_signal: audioSignal,
        length,
        cache_last_channel: this.cacheChannel,
        cache_last_time: this.cacheTime,
        cache_last_channel_len: this.cacheLength,
        lang_id: langId,
      });
    } finally {
      dispose(audioSignal, length, langId);
    }

    dispose(this.cacheChannel, this.cacheTime, this.cacheLength);
    this.cacheChannel = result.cache_last_channel_next;
    this.cacheTime = result.cache_last_time_next;
    this.cacheLength = result.cache_last_channel_len_next;
    try {
      const encodedLength = Number((result.encoded_lengths.data as BigInt64Array)[0]);
      await this.decodeEncoderOutput(result.outputs, encodedLength, onToken);
    } finally {
      dispose(result.encoded_lengths, result.outputs);
    }
  }

  private trimStreamBuffer(): void {
    // Keep only enough history to rebuild the nine mel-cache frames. The extra
    // half FFT window and one sample preserve centered STFT and pre-emphasis.
    const requiredHistory = PRE_ENCODE_CACHE * HOP_LENGTH + N_FFT / 2 + 1;
    const keepSamples = Math.ceil(requiredHistory / HOP_LENGTH) * HOP_LENGTH;
    const removable = Math.max(0, this.streamProcessedSamples - keepSamples);
    const aligned = Math.floor(removable / HOP_LENGTH) * HOP_LENGTH;
    if (aligned === 0) return;
    this.streamAudio = this.streamAudio.slice(aligned);
    this.streamProcessedSamples -= aligned;
  }

  private reset(): void {
    dispose(this.cacheChannel, this.cacheTime, this.cacheLength, this.hidden, this.cell);
    this.transcriptText = "";
    this.lastToken = BLANK_ID;
    this.cacheChannel = zeros([1, ENCODER_LAYERS, LEFT_CONTEXT, HIDDEN_SIZE]);
    this.cacheTime = zeros([1, ENCODER_LAYERS, HIDDEN_SIZE, CONV_CONTEXT]);
    this.cacheLength = int64Tensor(0);
    this.hidden = zeros([DECODER_LAYERS, 1, DECODER_HIDDEN]);
    this.cell = zeros([DECODER_LAYERS, 1, DECODER_HIDDEN]);
  }

  private async decodeEncoderOutput(encoded: ort.Tensor, frames: number, onToken?: () => void): Promise<void> {
    if (!this.sessions) return;
    const data = encoded.data as Float32Array;
    const dims = encoded.dims.map(Number);
    const hiddenFirst = dims[1] === HIDDEN_SIZE;
    const timeSize = hiddenFirst ? dims[2] : dims[1];
    const frameCount = Math.min(frames, timeSize);
    let decoderOutput: ort.Tensor | undefined;
    let decoderForJoint: ort.Tensor | undefined;
    let nextHidden: ort.Tensor | undefined;
    let nextCell: ort.Tensor | undefined;
    const frameData = new Float32Array(frameCount * HIDDEN_SIZE);
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
      for (let hiddenIndex = 0; hiddenIndex < HIDDEN_SIZE; hiddenIndex++) {
        frameData[frameIndex * HIDDEN_SIZE + hiddenIndex] = hiddenFirst
          ? data[hiddenIndex * timeSize + frameIndex]
          : data[frameIndex * HIDDEN_SIZE + hiddenIndex];
      }
    }
    const singleFrameData = new Float32Array(HIDDEN_SIZE);
    const encoderFrame = new ort.Tensor("float32", singleFrameData, [1, 1, HIDDEN_SIZE]);
    const targetData = new BigInt64Array(1);
    const target = new ort.Tensor("int64", targetData, [1, 1]);

    try {
      frameLoop: for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        for (let symbol = 0; symbol < MAX_SYMBOLS_PER_STEP; symbol++) {
          if (!decoderForJoint) {
            targetData[0] = BigInt(this.lastToken);
            const decoderResult = await this.sessions.decoder.run({
              targets: target,
              h_in: this.hidden,
              c_in: this.cell,
            });
            decoderOutput = decoderResult.decoder_output;
            nextHidden = decoderResult.h_out;
            nextCell = decoderResult.c_out;
            // The decoder exports [batch, 640, target_len], while joint consumes
            // [batch, target_len, 640]. GPU outputs need a non-owning buffer view.
            decoderForJoint = decoderOutput.location === "gpu-buffer"
              ? ort.Tensor.fromGpuBuffer(decoderOutput.gpuBuffer, {
                  dataType: "float32",
                  dims: [1, 1, DECODER_HIDDEN],
                })
              : decoderOutput.reshape([1, 1, DECODER_HIDDEN]);
          }

          const encoderBatch = symbol === 0
            ? new ort.Tensor("float32", frameData.subarray(frameIndex * HIDDEN_SIZE), [
                1,
                frameCount - frameIndex,
                HIDDEN_SIZE,
              ])
            : undefined;
          let jointResult: ort.InferenceSession.OnnxValueMapType | undefined;
          let token = BLANK_ID;
          try {
            jointResult = await this.sessions.joint.run({
              encoder_output: encoderBatch ?? encoderFrame,
              decoder_output: decoderForJoint,
            });
            const tokens = jointResult.token.data as BigInt64Array;
            if (symbol === 0) {
              const firstNonBlank = tokens.findIndex((value) => Number(value) !== BLANK_ID);
              if (firstNonBlank < 0) break frameLoop;
              frameIndex += firstNonBlank;
              singleFrameData.set(
                frameData.subarray(frameIndex * HIDDEN_SIZE, (frameIndex + 1) * HIDDEN_SIZE),
              );
              token = Number(tokens[firstNonBlank]);
            } else {
              token = Number(tokens[0]);
            }
          } finally {
            dispose(encoderBatch, jointResult?.token);
          }

          if (token === BLANK_ID) break;

          dispose(this.hidden, this.cell, decoderForJoint, decoderOutput);
          this.hidden = nextHidden!;
          this.cell = nextCell!;
          decoderForJoint = decoderOutput = nextHidden = nextCell = undefined;
          this.lastToken = token;
          const previousLength = this.transcriptText.length;
          this.appendToken(token);
          if (this.transcriptText.length > previousLength) onToken?.();
        }
      }
    } finally {
      dispose(encoderFrame, target, decoderForJoint, decoderOutput, nextHidden, nextCell);
    }
  }

  private appendToken(token: number): void {
    const piece = this.vocabulary[token] ?? "";
    if (piece === "<blank>" || piece === "<unk>" || /^<[a-z]{2}(?:-[A-Z]{2})?>$/.test(piece)) return;
    const text = piece.replaceAll("▁", " ");
    this.transcriptText += this.transcriptText ? text : text.trimStart();
  }
}

function buildFeatureChunk(
  mel: MelSpectrogram,
  frameStart: number,
  mainFrames: number,
): Float32Array {
  const width = PRE_ENCODE_CACHE + CHUNK_FRAMES;
  const chunk = new Float32Array(N_MELS * width);
  chunk.fill(Math.log(LOG_ZERO_GUARD), PRE_ENCODE_CACHE * N_MELS);
  const cacheStart = Math.max(0, frameStart - PRE_ENCODE_CACHE);
  const cacheFrames = frameStart - cacheStart;
  const cacheOffset = PRE_ENCODE_CACHE - cacheFrames;

  for (let f = 0; f < cacheFrames; f++) {
    for (let m = 0; m < N_MELS; m++) {
      chunk[(cacheOffset + f) * N_MELS + m] = mel.data[m * mel.frames + cacheStart + f];
    }
  }
  for (let f = 0; f < mainFrames; f++) {
    for (let m = 0; m < N_MELS; m++) {
      chunk[(PRE_ENCODE_CACHE + f) * N_MELS + m] = mel.data[m * mel.frames + frameStart + f];
    }
  }
  return chunk;
}

async function createSession(
  name: "encoder" | "decoder" | "joint",
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
  keepEncoderCacheOnGpu = false,
): Promise<ort.InferenceSession> {
  const modelUrl = name === "joint" ? "/joint-argmax.onnx" : `${MODEL_ROOT}/${name}.onnx`;
  let preferredOutputLocation: ort.InferenceSession.SessionOptions["preferredOutputLocation"];
  if (keepEncoderCacheOnGpu && name === "encoder") {
    preferredOutputLocation = {
      cache_last_channel_next: "gpu-buffer",
      cache_last_time_next: "gpu-buffer",
      cache_last_channel_len_next: "gpu-buffer",
    };
  } else if (keepEncoderCacheOnGpu && name === "decoder") {
    preferredOutputLocation = { decoder_output: "gpu-buffer", h_out: "gpu-buffer", c_out: "gpu-buffer" };
  }
  return ort.InferenceSession.create(modelUrl, {
    executionProviders,
    graphOptimizationLevel: "all",
    preferredOutputLocation,
    externalData: [{ path: `${name}.onnx.data`, data: `${MODEL_ROOT}/${name}.onnx.data` }],
  });
}

async function createSessions(
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
  keepEncoderCacheOnGpu: boolean,
  onStatus: (message: string) => void,
): Promise<Sessions> {
  let encoder: ort.InferenceSession | undefined;
  let decoder: ort.InferenceSession | undefined;
  let joint: ort.InferenceSession | undefined;
  try {
    onStatus("loading encoder");
    encoder = await createSession("encoder", executionProviders, keepEncoderCacheOnGpu);
    onStatus("loading decoder");
    decoder = await createSession("decoder", executionProviders, keepEncoderCacheOnGpu);
    onStatus("loading joint");
    joint = await createSession("joint", executionProviders);
    return { encoder, decoder, joint };
  } catch (error) {
    await Promise.allSettled([encoder?.release(), decoder?.release(), joint?.release()]);
    throw error;
  }
}

function zeros(dims: number[]): ort.Tensor {
  return new ort.Tensor("float32", new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
}

function int64Tensor(value: number, dims = [1]): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.of(BigInt(value)), dims);
}

function dispose(...tensors: Array<ort.Tensor | undefined>): void {
  for (const tensor of tensors) tensor?.dispose();
}

function append(left: Float32Array, right: Float32Array): Float32Array {
  if (right.length === 0) return left;
  const joined = new Float32Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

async function fetchJson(url: string): Promise<any> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  return response.json();
}
