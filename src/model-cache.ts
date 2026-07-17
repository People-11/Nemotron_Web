const CACHE_NAME = "nemotron-8364d9e2";
const MODEL_ROOT =
  "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/8364d9e2dd9da23789b480bdbba9e423717e42ee";
const MODEL_FILES = [
  "tokenizer.json",
  "encoder.onnx",
  "encoder.onnx.data",
  "decoder.onnx",
  "decoder.onnx.data",
  "joint.onnx.data",
];

export async function enableModelCache(): Promise<boolean> {
  if (!("serviceWorker" in navigator) || !("caches" in globalThis)) return false;
  await navigator.serviceWorker.register("/model-cache-sw.js", { updateViaCache: "none" });
  const registration = await navigator.serviceWorker.ready;
  if (!navigator.serviceWorker.controller) {
    const controlled = new Promise<void>((resolve) => {
      navigator.serviceWorker.addEventListener("controllerchange", () => resolve(), { once: true });
    });
    registration.active?.postMessage({ type: "claim-model-cache-clients" });
    await Promise.race([controlled, new Promise<void>((resolve) => setTimeout(resolve, 5000))]);
  }
  await navigator.storage?.persist?.().catch(() => false);
  return navigator.serviceWorker.controller !== null;
}

export async function modelCacheInfo(): Promise<string> {
  if (!("caches" in globalThis)) return "Cache Storage unavailable";
  const cache = await caches.open(CACHE_NAME);
  const [files, estimate, persisted] = await Promise.all([
    cache.keys(),
    navigator.storage?.estimate?.(),
    navigator.storage?.persisted?.(),
  ]);
  const usage = formatBytes(estimate?.usage ?? 0);
  return `${files.length} files · ${usage}${persisted ? " · persistent" : ""}`;
}

export async function hasCompleteModelCache(): Promise<boolean> {
  if (!("caches" in globalThis)) return false;
  const cache = await caches.open(CACHE_NAME);
  const matches = await Promise.all(
    MODEL_FILES.map((file) => cache.match(`${MODEL_ROOT}/${file}`, { ignoreVary: true })),
  );
  return matches.every(Boolean);
}

export async function clearModelCache(): Promise<void> {
  if (!("caches" in globalThis)) return;
  await caches.delete(CACHE_NAME);
}

function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  const mb = bytes / 1024 ** 2;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${mb.toFixed(mb < 100 ? 1 : 0)} MB`;
}
