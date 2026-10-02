import { getInitialCatalog } from "../../domain/pieces/catalog";
import type { Piece } from "../../domain/pieces/types";
import type { CapturedFrame } from "../capture/capture";
import {
  recognizeAutomaticBoard,
  type AutomaticBoardResult,
} from "./automatic";
import { abilityRegions, recognizePanelAbilities } from "./panel-abilities";
import type { TextTemplate } from "./panel-mask";
import { recognizePanelPieces } from "./panel-pieces";

export function recognizeAutomaticGame(
  frame: CapturedFrame,
  templates: readonly TextTemplate[],
  catalog: readonly Piece[] = getInitialCatalog(),
): AutomaticBoardResult {
  const board = recognizeAutomaticBoard(frame);
  if (!board.ok) return board;
  const pieces = recognizePanelPieces(frame, board.region, catalog, templates);
  const counts = recognizePanelAbilities(
    frame,
    abilityRegions(board.region),
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
