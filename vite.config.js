const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default {
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
};
