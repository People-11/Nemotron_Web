export const SAMPLE_RATE = 16_000;
export const N_FFT = 512;
export const WIN_LENGTH = 400;
export const HOP_LENGTH = 160;
export const N_MELS = 128;
export const LOG_ZERO_GUARD = 2 ** -24;
export const CHUNK_FRAMES = 56;
export const CHUNK_SAMPLES = CHUNK_FRAMES * HOP_LENGTH;

const PREEMPHASIS = 0.97;
const F_SP = 200 / 3;
const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
const LOG_STEP = 0.06875177742094912;

export interface MelSpectrogram {
  data: Float32Array;
  frames: number;
  samples: number;
}

export async function decodeAudioFile(file: File): Promise<Float32Array> {
  const context = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  return downmixToMono(
    Array.from({ length: decoded.numberOfChannels }, (_, channel) => decoded.getChannelData(channel)),
  );
}

export function downmixToMono(channels: readonly Float32Array[]): Float32Array {
  if (channels.length === 1) return channels[0];
  const mono = new Float32Array(channels[0].length);
  if (channels.length === 6) {
    const [left, right, center, , surroundLeft, surroundRight] = channels;
    for (let i = 0; i < mono.length; i++) {
      mono[i] = Math.SQRT1_2 * (left[i] + right[i]) + center[i] + 0.5 * (surroundLeft[i] + surroundRight[i]);
    }
  } else {
    const gain = 1 / channels.length;
    for (const channel of channels) {
      for (let i = 0; i < mono.length; i++) mono[i] += channel[i] * gain;
    }
  }
  return mono;
}

export function createMelFilterbank(): Float32Array {
  const bins = N_FFT / 2 + 1;
  const result = new Float32Array(N_MELS * bins);
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const points = new Float64Array(N_MELS + 2);

  for (let i = 0; i < points.length; i++) {
    points[i] = melToHz((melMax * i) / (N_MELS + 1));
  }

  for (let mel = 0; mel < N_MELS; mel++) {
    const left = points[mel];
    const center = points[mel + 1];
    const right = points[mel + 2];
    const norm = 2 / (right - left);
    for (let bin = 0; bin < bins; bin++) {
      const frequency = (bin * SAMPLE_RATE) / N_FFT;
      const lower = (frequency - left) / (center - left);
      const upper = (right - frequency) / (right - center);
      result[mel * bins + bin] = Math.max(0, Math.min(lower, upper)) * norm;
    }
  }
  return result;
}

const DEFAULT_FILTERBANK = createMelFilterbank();
const DEFAULT_FILTER_RANGES = createFilterRanges(DEFAULT_FILTERBANK);
const DEFAULT_WINDOW = symmetricHann(WIN_LENGTH);
const WINDOW_OFFSET = (N_FFT - WIN_LENGTH) / 2;

export function logMelSpectrogram(
  audio: Float32Array,
  paddedSamples = audio.length,
  filterbank = DEFAULT_FILTERBANK,
): MelSpectrogram {
  if (audio.length === 0) return { data: new Float32Array(), frames: 0, samples: 0 };
  if (paddedSamples < audio.length) throw new Error("Mel padding cannot truncate audio");
  const filterRanges = filterbank === DEFAULT_FILTERBANK ? DEFAULT_FILTER_RANGES : createFilterRanges(filterbank);

  // Pad after pre-emphasis, matching the official processor's masked audio padding.
  const padded = new Float32Array(paddedSamples + N_FFT);
  const audioOffset = N_FFT / 2;
  padded[audioOffset] = audio[0];
  for (let i = 1; i < audio.length; i++) {
    padded[audioOffset + i] = audio[i] - PREEMPHASIS * audio[i - 1];
  }
  const frames = Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1;
  const bins = N_FFT / 2 + 1;
  const real = new Float32Array(N_FFT);
  const imag = new Float32Array(N_FFT);
  const power = new Float32Array(bins);
  const output = new Float32Array(N_MELS * frames);

  for (let frame = 0; frame < frames; frame++) {
    real.fill(0);
    imag.fill(0);
    const start = frame * HOP_LENGTH;
    for (let i = 0; i < WIN_LENGTH; i++) {
      real[WINDOW_OFFSET + i] = padded[start + WINDOW_OFFSET + i] * DEFAULT_WINDOW[i];
    }
    fftInPlace(real, imag);
    for (let bin = 0; bin < bins; bin++) {
      power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
    }

    for (let mel = 0; mel < N_MELS; mel++) {
      let energy = 0;
      const offset = mel * bins;
      const rangeOffset = mel * 2;
      for (let bin = filterRanges[rangeOffset]; bin < filterRanges[rangeOffset + 1]; bin++) {
        energy += filterbank[offset + bin] * power[bin];
      }
      output[mel * frames + frame] = Math.log(energy + LOG_ZERO_GUARD);
    }
  }

  return { data: output, frames, samples: paddedSamples };
}

function createFilterRanges(filterbank: Float32Array): Uint16Array {
  const bins = N_FFT / 2 + 1;
  const ranges = new Uint16Array(N_MELS * 2);
  for (let mel = 0; mel < N_MELS; mel++) {
    const offset = mel * bins;
    let start = 0;
    let end = bins;
    while (start < bins && filterbank[offset + start] === 0) start++;
    while (end > start && filterbank[offset + end - 1] === 0) end--;
    ranges[mel * 2] = start;
    ranges[mel * 2 + 1] = end;
  }
  return ranges;
}

function symmetricHann(length: number): Float32Array {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return window;
}

function fftInPlace(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const angle = (-2 * Math.PI) / size;
    const stepReal = Math.cos(angle);
    const stepImag = Math.sin(angle);
    for (let start = 0; start < n; start += size) {
      let twiddleReal = 1;
      let twiddleImag = 0;
      for (let offset = 0; offset < size / 2; offset++) {
        const even = start + offset;
        const odd = even + size / 2;
        const oddReal = real[odd] * twiddleReal - imag[odd] * twiddleImag;
        const oddImag = real[odd] * twiddleImag + imag[odd] * twiddleReal;
        real[odd] = real[even] - oddReal;
        imag[odd] = imag[even] - oddImag;
        real[even] += oddReal;
        imag[even] += oddImag;
        const nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
        twiddleImag = twiddleReal * stepImag + twiddleImag * stepReal;
        twiddleReal = nextReal;
      }
    }
  }
}

function hzToMel(hz: number): number {
  return hz < MIN_LOG_HZ ? hz / F_SP : MIN_LOG_MEL + Math.log(hz / MIN_LOG_HZ) / LOG_STEP;
}

function melToHz(mel: number): number {
  return mel < MIN_LOG_MEL ? mel * F_SP : MIN_LOG_HZ * Math.exp((mel - MIN_LOG_MEL) * LOG_STEP);
}
