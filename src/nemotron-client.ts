import type { MelSpectrogram } from "./audio";
import type { ChunkProgress, StreamProgress } from "./nemotron";

// Same API surface as NemotronBrowserASR, but the model runs in a worker so heavy
// inference (mel + WebGPU readbacks) never blocks the main thread / starves rAF.

export type WorkerRequest =
  | { id: number; type: "load" }
  | { id: number; type: "transcribe"; mel: MelSpectrogram; langId: number }
  | { id: number; type: "startStream"; langId: number }
  | { id: number; type: "pushAudio"; samples: ArrayBuffer; flush: boolean }
  | { type: "abort" };

export type WorkerResponse =
  | { id: number; type: "status"; message: string }
  | { id: number; type: "chunk"; progress: ChunkProgress }
  | { id: number; type: "partial"; text: string }
  | { id: number; type: "result"; value: unknown }
  | { id: number; type: "error"; message: string };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onStatus?: (message: string) => void;
  onChunk?: (progress: ChunkProgress) => void;
  onPartial?: (text: string) => void;
};

export class NemotronClient {
  private worker = new Worker(new URL("./nemotron.worker.ts", import.meta.url), { type: "module" });
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor() {
    this.worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      const entry = this.pending.get(data.id);
      if (!entry) return;
      if (data.type === "status") entry.onStatus?.(data.message);
      else if (data.type === "chunk") entry.onChunk?.(data.progress);
      else if (data.type === "partial") entry.onPartial?.(data.text);
      else if (data.type === "result") {
        this.pending.delete(data.id);
        entry.resolve(data.value);
      } else {
        this.pending.delete(data.id);
        entry.reject(new Error(data.message));
      }
    };
  }

  private request<T>(
    message: Extract<WorkerRequest, { id: number }>,
    transfer: Transferable[] = [],
    handlers: Pick<Pending, "onStatus" | "onChunk" | "onPartial"> = {},
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.id, { resolve: resolve as Pending["resolve"], reject, ...handlers });
      this.worker.postMessage(message, transfer);
    });
  }

  load(onStatus: (message: string) => void): Promise<void> {
    return this.request<void>({ id: this.nextId++, type: "load" }, [], { onStatus });
  }

  transcribe(
    mel: MelSpectrogram,
    langId: number,
    onChunk: (progress: ChunkProgress) => void,
    signal?: AbortSignal,
  ): Promise<string> {
    // Clone mel (no transfer) so the caller can re-run the same file.
    signal?.addEventListener("abort", () => this.worker.postMessage({ type: "abort" }), { once: true });
    return this.request<string>({ id: this.nextId++, type: "transcribe", mel, langId }, [], { onChunk });
  }

  startStream(langId: number): void {
    // Fire-and-forget: the worker handles requests in postMessage order, so this
    // resets stream state before any pushAudio that follows it.
    void this.request<void>({ id: this.nextId++, type: "startStream", langId }).catch(() => {});
  }

  pushAudio(samples: Float32Array, flush = false, onPartial?: (text: string) => void): Promise<StreamProgress> {
    const buffer = samples.buffer as ArrayBuffer;
    return this.request<StreamProgress>(
      { id: this.nextId++, type: "pushAudio", samples: buffer, flush },
      [buffer],
      { onPartial },
    );
  }
}
