import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";

/** Ratios of the supplied game UI. Unsupported layouts remain unresolved. */
export const PANEL_PROFILE = {
  maxPixels: 16_777_216,
  minBoardPitch: 12,
  cardLeft: 10.5,
  cardTop: 0.9,
  cardWidth: 4.7,
  cardHeight: 2.7,
  cardStep: 2.9,
  miniaturePitchMin: 0.052,
  miniaturePitchMax: 0.088,
  pitchMismatch: 0.24,
  tileFraction: 0.68,
  emptyFraction: 0.16,
  textMatch: 0.87,
  textGap: 0.08,
  numberGap: 0.06,
  numberHoleShift: 0.1,
} as const;

export function validPanelFrame(frame: CapturedFrame): boolean {
  return (
    !!frame &&
    Number.isInteger(frame.width) &&
    Number.isInteger(frame.height) &&
    frame.width > 0 &&
    frame.height > 0 &&
    frame.width * frame.height <= PANEL_PROFILE.maxPixels &&
    Number.isFinite(frame.timestamp) &&
    frame.pixels instanceof Uint8ClampedArray &&
    frame.pixels.length === frame.width * frame.height * 4
  );
}
export function contained(frame: CapturedFrame, r: PixelRegion): boolean {
  return (
    !!r &&
    [r.x, r.y, r.width, r.height].every(Number.isFinite) &&
    r.x >= 0 &&
    r.y >= 0 &&
    r.width > 0 &&
    r.height > 0 &&
    r.x + r.width <= frame.width &&
    r.y + r.height <= frame.height
  );
}
export function relative(
  r: PixelRegion,
  x: number,
  y: number,
  w: number,
  h: number,
): PixelRegion {
  return {
    x: r.x + r.width * x,
    y: r.y + r.height * y,
    width: r.width * w,
    height: r.height * h,
  };
}
export function panelCards(
  frame: CapturedFrame,
  board: PixelRegion,
): PixelRegion[] | null {
  if (!validPanelFrame(frame) || !contained(frame, board)) return null;
  const p = board.width / 10;
  if (
    p < PANEL_PROFILE.minBoardPitch ||
    Math.abs(board.height / 16 / p - 1) > 0.05
  )
    return null;
  return Array.from({ length: 3 }, (_, slot) => ({
    x: board.x + p * PANEL_PROFILE.cardLeft,
    y: board.y + p * (PANEL_PROFILE.cardTop + slot * PANEL_PROFILE.cardStep),
    width: p * PANEL_PROFILE.cardWidth,
    height: p * PANEL_PROFILE.cardHeight,
  }));
}
