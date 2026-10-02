import { BOARD_HEIGHT, BOARD_WIDTH } from "../../domain/board/board";
import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";
import { unknownRecognitionEngine } from "./mock";
import type { RecognitionResult, RecognizedCell } from "./types";
import { AUTOMATIC_BOARD_CONFIG as C } from "./automatic-config";

export type AutomaticBoardResult =
  | { ok: true; region: PixelRegion; result: RecognitionResult }
  | { ok: false; reason: "NOT_FOUND" | "AMBIGUOUS" | "INVALID_FRAME" };
const valid = (frame: CapturedFrame) =>
  frame &&
  Number.isInteger(frame.width) &&
  Number.isInteger(frame.height) &&
  frame.width > 0 &&
  frame.height > 0 &&
  frame.width * frame.height <= C.maxPixels &&
  Number.isFinite(frame.timestamp) &&
  frame.pixels instanceof Uint8ClampedArray &&
  frame.pixels.length === frame.width * frame.height * 4;
const maskPixel = (frame: CapturedFrame, x: number, y: number) => {
  const i = (y * frame.width + x) * 4;
  const [r, g, b, a] = frame.pixels.subarray(i, i + 4);
  return (
    a === 255 &&
    g >= 65 &&
    b >= 70 &&
    r < g * 0.72 &&
    g / b > 0.58 &&
    g / b < 1.55
  );
};
function components(frame: CapturedFrame): PixelRegion[] {
  const scale = Math.min(
    1,
    C.detectionEdge / Math.max(frame.width, frame.height),
  );
  const width = Math.floor(frame.width * scale),
    height = Math.floor(frame.height * scale);
  const mask = new Uint8Array(width * height),
    queue = new Int32Array(width * height);
  const found: PixelRegion[] = [];
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      mask[y * width + x] = Number(
        maskPixel(frame, Math.floor(x / scale), Math.floor(y / scale)),
      );
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] !== 1) continue;
    let head = 0,
      tail = 1,
      minX = start % width,
      maxX = minX,
      minY = Math.floor(start / width),
      maxY = minY;
    queue[0] = start;
    mask[start] = 0;
    while (head < tail) {
      const i = queue[head++],
        x = i % width,
        y = Math.floor(i / width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      for (const n of [
        x > 0 ? i - 1 : -1,
        x + 1 < width ? i + 1 : -1,
        y > 0 ? i - width : -1,
        y + 1 < height ? i + width : -1,
      ]) {
        if (n >= 0 && mask[n] === 1) {
          mask[n] = 0;
          queue[tail++] = n;
        }
      }
    }
    const w = maxX - minX + 1,
      h = maxY - minY + 1;
    if (
      w / scale / BOARD_WIDTH < C.minCellPixels ||
      h / scale / BOARD_HEIGHT < C.minCellPixels ||
      w / h < C.minComponentAspect ||
      w / h > C.maxComponentAspect ||
      tail / (w * h) < C.minMaskCoverage
    )
      continue;
    found.push({
      x: minX / scale,
      y: minY / scale,
      width: w / scale,
      height: h / scale,
    });
    if (found.length > C.maxCandidates) {
      mask.fill(0);
      queue.fill(0);
      return [];
    }
  }
  mask.fill(0);
  queue.fill(0);
  return found;
}
function contrast(
  frame: CapturedFrame,
  x: number,
  y: number,
  dx: number,
  dy: number,
) {
  const x1 = Math.round(x),
    y1 = Math.round(y),
    x2 = x1 + dx,
    y2 = y1 + dy;
  if (
    x1 < 0 ||
    y1 < 0 ||
    x2 < 0 ||
    y2 < 0 ||
    x1 >= frame.width ||
    x2 >= frame.width ||
    y1 >= frame.height ||
    y2 >= frame.height
  )
    return 0;
  const a = (y1 * frame.width + x1) * 4,
    b = (y2 * frame.width + x2) * 4;
  if (frame.pixels[a + 3] !== 255 || frame.pixels[b + 3] !== 255) return 0;
  return Math.hypot(
    frame.pixels[a] - frame.pixels[b],
    frame.pixels[a + 1] - frame.pixels[b + 1],
    frame.pixels[a + 2] - frame.pixels[b + 2],
  );
}
function fit(frame: CapturedFrame, box: PixelRegion): PixelRegion | null {
  const projection = (vertical: boolean) => {
    const origin = Math.floor(vertical ? box.x : box.y),
      length = Math.ceil(vertical ? box.width : box.height);
    const values = new Float64Array(length);
    for (let i = 0; i < length; i++) {
      const contrasts: number[] = [];
      for (let j = 0; j < C.projectionSamples; j++) {
        const across = (j + 0.5) / C.projectionSamples;
        contrasts.push(
          contrast(
            frame,
            vertical ? origin + i : box.x + box.width * across,
            vertical ? box.y + box.height * across : origin + i,
            vertical ? 1 : 0,
            vertical ? 0 : 1,
          ),
        );
      }
      contrasts.sort((a, b) => a - b);
      values[i] = contrasts[Math.floor(contrasts.length * 0.45)];
    }
    return { origin, values };
  };
  const xs = projection(true),
    ys = projection(false);
  const maxPitch = Math.min(box.width / BOARD_WIDTH, box.height / BOARD_HEIGHT);
  let best = -Infinity,
    winner: PixelRegion | null = null;
  for (
    let p =
      Math.ceil(
        Math.max(C.minCellPixels, maxPitch * C.minPitchFraction) / C.pitchStep,
      ) * C.pitchStep;
    p <= maxPitch;
    p += C.pitchStep
  ) {
    const search = (axis: ReturnType<typeof projection>, count: number) => {
      const candidates: { score: number; start: number }[] = [];
      for (
        let offset = 0;
        offset <= axis.values.length - count * p + Math.ceil(p * 0.08);
        offset++
      ) {
        const lines: number[] = [];
        for (let k = 1; k < count; k++) {
          const i = Math.round(offset + k * p);
          lines.push(axis.values[i] ?? 0);
        }
        const score =
          lines.reduce((sum, value) => sum + value, 0) / lines.length;
        if (score >= C.minGridContrast)
          candidates.push({ score, start: offset });
      }
      candidates.sort((a, b) => b.score - a.score || a.start - b.start);
      const kept: typeof candidates = [];
      for (const candidate of candidates) {
        if (
          kept.every(
            (c) => Math.abs(c.start - candidate.start) >= p * C.axisSeparation,
          )
        )
          kept.push(candidate);
        if (kept.length === C.axisCandidates) break;
      }
      return kept;
    };
    const x = search(xs, BOARD_WIDTH),
      y = search(ys, BOARD_HEIGHT);
    for (const a of x)
      for (const b of y) {
        const candidate = {
          x: xs.origin + a.start,
          y: ys.origin + b.start,
          width: p * BOARD_WIDTH,
          height: p * BOARD_HEIGHT,
        };
        if (
          candidate.x + candidate.width > frame.width ||
          candidate.y + candidate.height > frame.height
        )
          continue;
        const seamScore = Math.min(a.score, b.score, 99) / 100;
        if (BOARD_WIDTH * BOARD_HEIGHT + seamScore <= best) continue;
        if (localGridScore(frame, candidate) < C.minGridContrast) continue;
        let resolved = 0;
        for (let row = 0; row < BOARD_HEIGHT; row++)
          for (let col = 0; col < BOARD_WIDTH; col++)
            if (classify(frame, candidate, row, col).occupied !== null)
              resolved++;
        if (resolved < BOARD_WIDTH * BOARD_HEIGHT * C.minResolvedCells)
          continue;
        // Prefer a full board over a shifted crop that contains UI cards/buttons.
        const score = resolved + seamScore;
        if (score <= best) continue;
        best = score;
        winner = candidate;
      }
  }
  return best >= C.minGridContrast &&
    winner &&
    winner.x >= 0 &&
    winner.y >= 0 &&
    winner.x + winner.width <= frame.width &&
    winner.y + winner.height <= frame.height
    ? winner
    : null;
}
/** Check every seam inside this crop, rather than unrelated strong panel edges. */
function localGridScore(frame: CapturedFrame, region: PixelRegion): number {
  const pitch = region.width / BOARD_WIDTH;
  let weakest = Infinity;
  for (const vertical of [true, false]) {
    const count = vertical ? BOARD_WIDTH : BOARD_HEIGHT;
    const seamPitch = vertical ? pitch : region.height / BOARD_HEIGHT;
    for (let k = 1; k < count; k++) {
      let strongest = 0;
      for (
        let shift = -Math.ceil(pitch * 0.08);
        shift <= Math.ceil(pitch * 0.08);
        shift++
      ) {
        const values: number[] = [];
        for (let j = 0; j < C.projectionSamples; j++) {
          const across = (j + 0.5) / C.projectionSamples;
          values.push(
            contrast(
              frame,
              vertical
                ? region.x + k * pitch + shift
                : region.x + region.width * across,
              vertical
                ? region.y + region.height * across
                : region.y + k * seamPitch + shift,
              vertical ? 1 : 0,
              vertical ? 0 : 1,
            ),
          );
        }
        values.sort((a, b) => a - b);
        strongest = Math.max(
          strongest,
          values[Math.floor(values.length * 0.45)],
        );
      }
      weakest = Math.min(weakest, strongest);
      if (weakest < C.minGridContrast) return weakest;
    }
  }
  return weakest;
}
function classify(
  frame: CapturedFrame,
  region: PixelRegion,
  row: number,
  col: number,
): RecognizedCell {
  const points: { x: number; y: number; v: number[] }[] = [];
  const pitch = region.width / BOARD_WIDTH;
  for (let y = 0; y < 7; y++)
    for (let x = 0; x < 7; x++) {
      const i =
        (Math.floor(
          region.y + (row + 0.2 + y * 0.1) * (region.height / BOARD_HEIGHT),
        ) *
          frame.width +
          Math.floor(region.x + (col + 0.2 + x * 0.1) * pitch)) *
        4;
      if (frame.pixels[i + 3] !== 255)
        return { row, col, occupied: null, status: "unknown" };
      points.push({
        x: x - 3,
        y: y - 3,
        v: Array.from(frame.pixels.subarray(i, i + 3)),
      });
    }
  const mean = [0, 1, 2].map(
    (c) => points.reduce((sum, p) => sum + p.v[c], 0) / 49,
  );
  const slopeX = [0, 1, 2].map(
    (c) => points.reduce((sum, p) => sum + p.x * (p.v[c] - mean[c]), 0) / 196,
  );
  const slopeY = [0, 1, 2].map(
    (c) => points.reduce((sum, p) => sum + p.y * (p.v[c] - mean[c]), 0) / 196,
  );
  const variation = Math.sqrt(
    points.reduce(
      (sum, p) => sum + p.v.reduce((s, v, c) => s + (v - mean[c]) ** 2, 0),
      0,
    ) / 49,
  );
  const residual = Math.sqrt(
    points.reduce(
      (sum, p) =>
        sum +
        p.v.reduce(
          (s, v, c) =>
            s + (v - mean[c] - slopeX[c] * p.x - slopeY[c] * p.y) ** 2,
          0,
        ),
      0,
    ) / 49,
  );
  const slope = -[0, 1, 2].reduce((s, c) => s + slopeX[c] + slopeY[c], 0);
  if (
    variation <= C.emptyVariation &&
    points.every(
      (p) =>
        p.v[1] >= 65 &&
        p.v[2] >= 70 &&
        p.v[0] < p.v[1] * 0.72 &&
        p.v[1] / p.v[2] > 0.58 &&
        p.v[1] / p.v[2] < 1.55,
    )
  )
    return { row, col, occupied: false, status: "recognized" };
  if (
    variation >= C.tileVariation &&
    slope >= C.minTileSlope &&
    residual / variation <= C.maxTileResidualRatio
  )
    return { row, col, occupied: true, status: "recognized" };
  return { row, col, occupied: null, status: "uncertain" };
}
/** Deterministic local UI profile. Never retains/mutates pixels or bypasses review. */
export function recognizeAutomaticBoard(
  frame: CapturedFrame,
): AutomaticBoardResult {
  if (!valid(frame)) return { ok: false, reason: "INVALID_FRAME" };
  const regions = components(frame)
    .map((box) => fit(frame, box))
    .filter((r): r is PixelRegion => r !== null);
  if (regions.length !== 1)
    return { ok: false, reason: regions.length ? "AMBIGUOUS" : "NOT_FOUND" };
  const region = regions[0],
    result = unknownRecognitionEngine.recognize({ kind: "frame", frame });
  result.board = Array.from({ length: BOARD_HEIGHT }, (_, row) =>
    Array.from({ length: BOARD_WIDTH }, (_, col) =>
      classify(frame, region, row, col),
    ),
  );
  result.source.kind = "automatic-board";
  result.summary.engine = "grid-tiles";
  return { ok: true, region, result };
}

/** Explicit/session ROI: inspect current pixels, never silently relocate it. */
export function recognizeBoardRegion(
  frame: CapturedFrame,
  region: PixelRegion,
): AutomaticBoardResult {
  if (!valid(frame)) return { ok: false, reason: "INVALID_FRAME" };
  if (
    !region ||
    ![region.x, region.y, region.width, region.height].every(Number.isFinite) ||
    region.x < 0 ||
    region.y < 0 ||
    region.width / BOARD_WIDTH < C.minCellPixels ||
    region.height / BOARD_HEIGHT < C.minCellPixels ||
    Math.abs(region.width / BOARD_WIDTH / (region.height / BOARD_HEIGHT) - 1) >
      0.05 ||
    region.x + region.width > frame.width ||
    region.y + region.height > frame.height ||
    localGridScore(frame, region) < C.minGridContrast
  )
    return { ok: false, reason: "NOT_FOUND" };
  const result = unknownRecognitionEngine.recognize({ kind: "frame", frame });
  result.board = Array.from({ length: BOARD_HEIGHT }, (_, row) =>
    Array.from({ length: BOARD_WIDTH }, (_, col) =>
      classify(frame, region, row, col),
    ),
  );
  result.source.kind = "automatic-board";
  result.summary.engine = "grid-tiles";
  return { ok: true, region: { ...region }, result };
}
