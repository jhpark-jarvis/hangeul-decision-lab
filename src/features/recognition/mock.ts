import { BOARD_HEIGHT, BOARD_WIDTH } from "../../domain/board/board";
import type { GameState } from "../../domain/game/types";
import type { RecognitionEngine, RecognitionResult } from "./types";

export const unknownRecognitionEngine: RecognitionEngine = {
  recognize({ frame }) {
    return {
      board: Array.from({ length: BOARD_HEIGHT }, (_, row) =>
        Array.from({ length: BOARD_WIDTH }, (_, col) => ({
          row,
          col,
          occupied: null,
          status: "unknown",
        })),
      ),
      pieces: ([0, 1, 2] as const).map((slot) => ({
        slot,
        pieceId: null,
        empty: null,
        status: "unknown",
      })),
      hiddenItems: [],
      hiddenItemsStatus: "unknown",
      abilities: {
        reroll: { value: null, status: "unknown" },
        singleCell: { value: null, status: "unknown" },
      },
      source: {
        kind: "capture-stub",
        timestamp: frame.timestamp,
        dimensions: { width: frame.width, height: frame.height },
      },
      summary: { engine: "unimplemented" },
    };
  },
};

/** Development-only copy of manual input. Does not recognize image pixels. */
export function recognizeManualState(state: GameState): RecognitionResult {
  return {
    board: state.board.map((cells, row) =>
      cells.map((occupied, col) => ({
        row,
        col,
        occupied,
        status: "recognized",
      })),
    ),
    pieces: ([0, 1, 2] as const).map((slot) => {
      const piece = state.remainingPieces.find(
        (entry) => entry.pieceIndex === slot,
      );
      return {
        slot,
        pieceId: piece?.piece.id ?? null,
        empty: !piece,
        status: "recognized",
      };
    }),
    hiddenItems: state.hiddenItems.map((item) => ({
      ...item,
      status: "recognized",
    })),
    hiddenItemsStatus: "recognized",
    abilities: {
      reroll: { value: state.abilities.reroll, status: "recognized" },
      singleCell: { value: state.abilities.singleCell, status: "recognized" },
    },
    source: { kind: "manual-mock", timestamp: Date.now() },
    summary: { engine: "manual-copy" },
  };
}
