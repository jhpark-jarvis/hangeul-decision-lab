import { BOARD_HEIGHT, BOARD_WIDTH } from "../../domain/board/board";
import type { CapturedFrame } from "../capture/capture";
import { unknownRecognitionEngine } from "./mock";
import type { RecognitionEngine, RecognizedCell } from "./types";

export type RGB = readonly [number, number, number];
export type BoardRecognitionConfig = {
  /** Frame-pixel coordinates of the full 10-column / 16-row board. */
  region: { x: number; y: number; width: number; height: number };
  emptyColors: readonly RGB[];
  occupiedColors: readonly RGB[];
  maxDistance: number;
  /** Required distance advantage over the opposite class; never probability. */
  minDistanceGap: number;
  minMatchedFraction: number;
  /** Central fraction of each cell, sampled with a bounded regular grid. */
  sampleFraction: number;
  samplesPerAxis: number;
};

const MAX_RGB_DISTANCE = Math.sqrt(3 * 255 ** 2);
const MAX_FRAME_PIXELS = 16_777_216;
const MAX_COLORS = 16;
const MAX_SAMPLES_PER_AXIS = 9;
const finite = (n: number) => typeof n === "number" && Number.isFinite(n);
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const bounded = (n: number, min: number, max: number) =>
  finite(n) && n >= min && n <= max;

function colorsValid(colors: readonly RGB[]): boolean {
  return (
    Array.isArray(colors) &&
    colors.length > 0 &&
    colors.length <= MAX_COLORS &&
    Array.from(colors).every(
      (color) =>
        Array.isArray(color) &&
        color.length === 3 &&
        Array.from(color).every(
          (channel) => Number.isInteger(channel) && bounded(channel, 0, 255),
        ),
    )
  );
}

function validateConfig(config: BoardRecognitionConfig): void {
  if (
    !object(config) ||
    !object(config.region) ||
    !finite(config.region.x) ||
    !finite(config.region.y) ||
    config.region.x < 0 ||
    config.region.y < 0 ||
    !finite(config.region.width) ||
    !finite(config.region.height) ||
    config.region.width <= 0 ||
    config.region.height <= 0 ||
    !colorsValid(config.emptyColors) ||
    !colorsValid(config.occupiedColors) ||
    !bounded(config.maxDistance, 0, MAX_RGB_DISTANCE) ||
    !bounded(config.minDistanceGap, Number.EPSILON, MAX_RGB_DISTANCE) ||
    !bounded(config.minMatchedFraction, 0.5 + Number.EPSILON, 1) ||
    !bounded(config.sampleFraction, Number.EPSILON, 1) ||
    !Number.isInteger(config.samplesPerAxis) ||
    !bounded(config.samplesPerAxis, 1, MAX_SAMPLES_PER_AXIS) ||
    (config.region.width / BOARD_WIDTH) * config.sampleFraction <
      config.samplesPerAxis ||
    (config.region.height / BOARD_HEIGHT) * config.sampleFraction <
      config.samplesPerAxis
  )
    throw new Error("보드 영역·색상 표본·인식 설정이 올바르지 않습니다.");
}

function validateFrame(frame: CapturedFrame, config: BoardRecognitionConfig) {
  if (
    !object(frame) ||
    !Number.isInteger(frame.width) ||
    !Number.isInteger(frame.height) ||
    frame.width <= 0 ||
    frame.height <= 0 ||
    frame.width * frame.height > MAX_FRAME_PIXELS ||
    !finite(frame.timestamp) ||
    !(frame.pixels instanceof Uint8ClampedArray) ||
    frame.pixels.length !== frame.width * frame.height * 4 ||
    config.region.x + config.region.width > frame.width ||
    config.region.y + config.region.height > frame.height
  )
    throw new Error("프레임 형식 또는 보드 영역 범위가 올바르지 않습니다.");
}

function distance(r: number, g: number, b: number, colors: readonly RGB[]) {
  return Math.min(
    ...colors.map((c) => Math.hypot(r - c[0], g - c[1], b - c[2])),
  );
}

function recognizeCell(
  frame: CapturedFrame,
  config: BoardRecognitionConfig,
  row: number,
  col: number,
): RecognizedCell {
  const { region, sampleFraction, samplesPerAxis } = config;
  const cellWidth = region.width / BOARD_WIDTH;
  const cellHeight = region.height / BOARD_HEIGHT;
  let empty = 0;
  let occupied = 0;
  for (let sy = 0; sy < samplesPerAxis; sy++) {
    for (let sx = 0; sx < samplesPerAxis; sx++) {
      const fraction = (n: number) =>
        (1 - sampleFraction) / 2 +
        ((n + 0.5) / samplesPerAxis) * sampleFraction;
      const x = Math.floor(region.x + (col + fraction(sx)) * cellWidth);
      const y = Math.floor(region.y + (row + fraction(sy)) * cellHeight);
      const index = (y * frame.width + x) * 4;
      // Transparent pixels are not evidence of an empty board cell.
      if (frame.pixels[index + 3] !== 255) continue;
      const args = [
        frame.pixels[index],
        frame.pixels[index + 1],
        frame.pixels[index + 2],
      ] as const;
      const e = distance(...args, config.emptyColors);
      const o = distance(...args, config.occupiedColors);
      if (e <= config.maxDistance && o - e >= config.minDistanceGap) empty++;
      else if (o <= config.maxDistance && e - o >= config.minDistanceGap)
        occupied++;
    }
  }
  const total = samplesPerAxis ** 2;
  // Any opposite-class evidence makes this cell uncertain, even with a majority.
  if (empty === 0 && occupied / total >= config.minMatchedFraction)
    return { row, col, occupied: true, status: "recognized" };
  if (occupied === 0 && empty / total >= config.minMatchedFraction)
    return { row, col, occupied: false, status: "recognized" };
  return {
    row,
    col,
    occupied: null,
    status: empty + occupied ? "uncertain" : "unknown",
  };
}

/** Local calibrated prototype. No game-specific palette, ROI detection or I/O. */
export function createCalibratedBoardEngine(
  config: BoardRecognitionConfig,
): RecognitionEngine {
  validateConfig(config);
  // A caller changing its calibration must not silently alter an existing engine.
  const settings = structuredClone(config);
  return {
    recognize(input) {
      if (!object(input) || input.kind !== "frame")
        throw new Error("인식 입력은 명시적인 프레임이어야 합니다.");
      const { frame } = input;
      validateFrame(frame, settings);
      const result = unknownRecognitionEngine.recognize({
        kind: "frame",
        frame,
      });
      result.board = Array.from({ length: BOARD_HEIGHT }, (_, row) =>
        Array.from({ length: BOARD_WIDTH }, (_, col) =>
          recognizeCell(frame, settings, row, col),
        ),
      );
      result.source.kind = "calibrated-board";
      result.summary.engine = "rgb-samples";
      return result;
    },
  };
}
