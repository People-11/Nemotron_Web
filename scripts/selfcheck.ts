import assert from "node:assert/strict";
import {
  createMelFilterbank,
  downmixToMono,
  HOP_LENGTH,
  LOG_ZERO_GUARD,
  logMelSpectrogram,
  N_FFT,
  N_MELS,
  SAMPLE_RATE,
  WIN_LENGTH,
} from "../src/audio.ts";

assert.deepEqual(
  downmixToMono([Float32Array.of(1, -1), Float32Array.of(-1, 1)]),
  Float32Array.of(0, 0),
);
assert.ok(Math.abs(downmixToMono([
  Float32Array.of(1),
  Float32Array.of(1),
  Float32Array.of(1),
  Float32Array.of(100),
  Float32Array.of(1),
  Float32Array.of(1),
])[0] - (2 * Math.SQRT1_2 + 2)) < 1e-6);

const oneSecond = Float32Array.from(
  { length: SAMPLE_RATE },
  (_, index) => Math.sin((2 * Math.PI * 440 * index) / SAMPLE_RATE),
);
const filterbank = createMelFilterbank();
const mel = logMelSpectrogram(oneSecond, oneSecond.length, filterbank);

assert.equal(filterbank.length, N_MELS * (N_FFT / 2 + 1));
assert.equal(mel.frames, Math.floor(oneSecond.length / HOP_LENGTH) + 1);
assert.equal(mel.samples, oneSecond.length);
assert.equal(mel.data.length, N_MELS * mel.frames);
assert.ok(mel.data.every(Number.isFinite));

const firstFrame = Array.from({ length: N_MELS }, (_, index) => mel.data[index * mel.frames]);
assert.ok(Math.max(...firstFrame) > Math.min(...firstFrame), "Mel bands should not be constant");

// Independent DFT check for the centered 400-sample window used by torch.stft.
const checkedFrame = 10;
const windowOffset = (N_FFT - WIN_LENGTH) / 2;
const frame = new Float64Array(N_FFT);
for (let i = 0; i < WIN_LENGTH; i++) {
  const paddedIndex = checkedFrame * HOP_LENGTH + windowOffset + i;
  const audioIndex = paddedIndex - N_FFT / 2;
  const sample = audioIndex < 0 || audioIndex >= oneSecond.length
    ? 0
    : audioIndex === 0
      ? oneSecond[0]
      : oneSecond[audioIndex] - 0.97 * oneSecond[audioIndex - 1];
  frame[windowOffset + i] = sample * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN_LENGTH - 1)));
}
const power = new Float64Array(N_FFT / 2 + 1);
for (let bin = 0; bin < power.length; bin++) {
  let real = 0;
  let imag = 0;
  for (let i = 0; i < N_FFT; i++) {
    const angle = (-2 * Math.PI * bin * i) / N_FFT;
    real += frame[i] * Math.cos(angle);
    imag += frame[i] * Math.sin(angle);
  }
  power[bin] = real * real + imag * imag;
}
let maxError = 0;
for (let melIndex = 0; melIndex < N_MELS; melIndex++) {
  let energy = 0;
  for (let bin = 0; bin < power.length; bin++) {
    energy += filterbank[melIndex * power.length + bin] * power[bin];
  }
  const expected = Math.log(energy + LOG_ZERO_GUARD);
  maxError = Math.max(maxError, Math.abs(expected - mel.data[melIndex * mel.frames + checkedFrame]));
}
assert.ok(maxError < 2e-4, `FFT differs from centered-window DFT by ${maxError}`);

const shortAudio = oneSecond.subarray(0, 8_000);
const paddedMel = logMelSpectrogram(shortAudio, 8_960, filterbank);
assert.equal(paddedMel.samples, 8_960);
assert.equal(paddedMel.frames, Math.floor(8_960 / HOP_LENGTH) + 1);
assert.ok(
  Array.from({ length: N_MELS }, (_, index) => paddedMel.data[index * paddedMel.frames + 50])
    .some((value) => Math.abs(value - Math.log(LOG_ZERO_GUARD)) > 1e-3),
  "The first padded tail frame should still overlap real audio",
);

// Trimming old microphone samples must not change the cached/current frames.
const streamingAudio = Float32Array.from(
  { length: 8_960 * 2 + 256 },
  (_, index) => Math.sin((2 * Math.PI * 440 * index) / SAMPLE_RATE),
);
const globalMel = logMelSpectrogram(streamingAudio, streamingAudio.length, filterbank);
const trimmedSamples = 7_200;
const trimmedMel = logMelSpectrogram(streamingAudio.subarray(trimmedSamples), streamingAudio.length - trimmedSamples, filterbank);
let trimError = 0;
for (let frame = 0; frame < 65; frame++) {
  for (let melIndex = 0; melIndex < N_MELS; melIndex++) {
    const globalValue = globalMel.data[melIndex * globalMel.frames + 47 + frame];
    const trimmedValue = trimmedMel.data[melIndex * trimmedMel.frames + 2 + frame];
    trimError = Math.max(trimError, Math.abs(globalValue - trimmedValue));
  }
}
assert.ok(trimError < 1e-5, `Streaming history trim changed mel frames by ${trimError}`);

// Moving a cached waveform must preserve the direct sample-to-screen mapping.
const waveformWidth = 1_000;
const viewSamples = SAMPLE_RATE * 2;
const paintCursor = SAMPLE_RATE * 10;
const currentCursor = paintCursor + SAMPLE_RATE * 0.08;
const sample = paintCursor - SAMPLE_RATE * 0.5;
const cachedX = waveformWidth - (paintCursor - sample) * waveformWidth / viewSamples;
const translatedX = cachedX - (currentCursor - paintCursor) * waveformWidth / viewSamples;
const directX = waveformWidth - (currentCursor - sample) * waveformWidth / viewSamples;
assert.ok(Math.abs(translatedX - directX) < 1e-9, "Cached waveform translation changed its time mapping");

console.log(`selfcheck ok: ${mel.frames} frames, ${mel.data.length} finite values, DFT error ${maxError}`);
