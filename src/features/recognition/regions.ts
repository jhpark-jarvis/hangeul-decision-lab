import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";
import { recognizeBoardRegion, type AutomaticBoardResult } from "./automatic";
import { getInitialCatalog } from "../../domain/pieces/catalog";
import { contained, relative } from "./panel-profile";
import { recognizePieceCards } from "./panel-pieces";
import { recognizePanelAbilities } from "./panel-abilities";
import type { TextTemplate } from "./panel-mask";

export type CaptureRegions = {
  dimensions: { width: number; height: number };
  board: PixelRegion;
  pieces: PixelRegion;
  abilities: PixelRegion;
};
export const regionLabels = {
  board: "보드",
  pieces: "보유 조각",
  abilities: "능력",
} as const;
export type RegionKey = keyof typeof regionLabels;
export function suggestedRegions(
  frame: Pick<CapturedFrame, "width" | "height">,
  board: PixelRegion,
): CaptureRegions {
  const p = board.width / 10;
  return {
    dimensions: { width: frame.width, height: frame.height },
    board: { ...board },
    pieces: {
      x: board.x + 10.5 * p,
      y: board.y + 0.9 * p,
      width: 4.7 * p,
      height: 8.5 * p,
    },
    abilities: {
      x: board.x + 10.5 * p,
      y: board.y + 10 * p,
      width: 4.7 * p,
      height: 6 * p,
    },
  };
}
export const pieceCardRegions = (group: PixelRegion) =>
  Array.from({ length: 3 }, (_, slot) =>
    relative(group, 0, (slot * 2.9) / 8.5, 1, 2.7 / 8.5),
  );
export const countRegions = (group: PixelRegion) => ({
  singleCell: relative(group, 3.45 / 4.7, 3.95 / 6, 0.65 / 4.7, 0.6 / 6),
  reroll: relative(group, 3.45 / 4.7, 5.35 / 6, 0.65 / 4.7, 0.6 / 6),
  total: relative(group, 1.9 / 4.7, 2.8 / 6, 0.4 / 4.7, 0.65 / 6),
});
export function regionsFit(
  frame: CapturedFrame,
  regions: CaptureRegions,
): boolean {
  return (
    !!regions &&
    regions.dimensions.width === frame.width &&
    regions.dimensions.height === frame.height &&
    (Object.keys(regionLabels) as RegionKey[]).every((key) =>
      contained(frame, regions[key]),
    )
  );
}
export function recognizeConfiguredGame(
  frame: CapturedFrame,
  regions: CaptureRegions,
  templates: readonly TextTemplate[],
): AutomaticBoardResult {
  if (!regionsFit(frame, regions)) return { ok: false, reason: "NOT_FOUND" };
  const board = recognizeBoardRegion(frame, regions.board);
  if (!board.ok) return board;
  const pieces = recognizePieceCards(
    frame,
    pieceCardRegions(regions.pieces),
    getInitialCatalog(),
    templates,
  );
  const counts = recognizePanelAbilities(
    frame,
    countRegions(regions.abilities),
    templates,
  );
  return {
    ...board,
    result: {
      ...board.result,
      pieces: pieces.pieces,
      abilities: counts.abilities,
      source: { ...board.result.source, kind: "automatic-game" },
      summary: { engine: "grid-panel" },
    },
  };
}
