const CACHE_NAME = "nemotron-8364d9e2";
const MODEL_ROOT =
  "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/8364d9e2dd9da23789b480bdbba9e423717e42ee/";

self.addEventListener("install", (event) => event.waitUntil(self.skipWaiting()));

self.addEventListener("message", (event) => {
  if (event.data?.type === "claim-model-cache-clients") {
    event.waitUntil(self.clients.claim());
  }
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then((names) =>
        Promise.all(
          names
            .filter((name) => name.startsWith("nemotron-") && name !== CACHE_NAME)
            .map((name) => caches.delete(name)),
        ),
      ),
    ]),
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET" || !event.request.url.startsWith(MODEL_ROOT)) return;
  event.respondWith(cacheFirst(event));
});

async function cacheFirst(event) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(event.request);
  if (cached) return cached;

  const response = await fetch(event.request);
  if (response.ok && response.body) {
    const total = Number(response.headers.get("content-length")) || 0;
    let loaded = 0;
    let lastUpdate = 0;
    const stream = response.clone().body.pipeThrough(
      new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          const now = Date.now();
          if (now - lastUpdate >= 250) {
            lastUpdate = now;
            void broadcastProgress(event.request.url, loaded, total, false);
          }
          controller.enqueue(chunk);
        },
      }),
    );
    const cacheResponse = new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
    event.waitUntil(
      cache
        .put(event.request, cacheResponse)
        .then(() => broadcastProgress(event.request.url, loaded, total, true))
        .catch(() => broadcastProgress(event.request.url, loaded, total, true, true)),
    );
  }
  return response;
}

async function broadcastProgress(url, loaded, total, done, error = false) {
  const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  for (const client of windows) {
    client.postMessage({ type: "model-cache-progress", url, loaded, total, done, error });
  }
}
