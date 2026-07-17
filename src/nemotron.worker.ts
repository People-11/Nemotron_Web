import { NemotronBrowserASR } from "./nemotron";
import type { WorkerRequest, WorkerResponse } from "./nemotron-client";

// tsconfig uses the DOM lib, not WebWorker, so cast the worker global to the bits we use.
const ctx = self as unknown as {
  postMessage(message: WorkerResponse): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};

const model = new NemotronBrowserASR();
let queue: Promise<unknown> = Promise.resolve();
let abort: AbortController | null = null;

ctx.onmessage = ({ data }) => {
  if (data.type === "abort") {
    abort?.abort();
    return;
  }
  // Serialize: model state is order-sensitive, so each request finishes before the next.
  queue = queue.then(() => handle(data)).catch(() => {});
};

async function handle(request: Extract<WorkerRequest, { id: number }>): Promise<void> {
  const { id } = request;
  try {
    let value: unknown;
    if (request.type === "load") {
      await model.load((message) => ctx.postMessage({ id, type: "status", message }));
    } else if (request.type === "transcribe") {
      abort = new AbortController();
      value = await model.transcribe(
        request.mel,
        request.langId,
        (progress) => ctx.postMessage({ id, type: "chunk", progress }),
        abort.signal,
      );
      abort = null;
    } else if (request.type === "startStream") {
      model.startStream(request.langId);
    } else {
      value = await model.pushAudio(
        new Float32Array(request.samples),
        request.flush,
        (text) => ctx.postMessage({ id, type: "partial", text }),
      );
    }
    ctx.postMessage({ id, type: "result", value });
  } catch (error) {
    ctx.postMessage({ id, type: "error", message: error instanceof Error ? error.message : String(error) });
  }
}
