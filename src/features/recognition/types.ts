import type { CapturedFrame } from "../capture/capture";

export type RecognitionStatus = "recognized" | "uncertain" | "unknown";
export type Observation = { status: RecognitionStatus; confidence?: number };
export type RecognizedCell = Observation & {
  row: number;
  col: number;
  occupied: boolean | null;
};
export type RecognizedPieceSlot = Observation & {
  slot: 0 | 1 | 2;
  pieceId: string | null;
  empty: boolean | null;
};
export type RecognizedHiddenItem = Observation & {
  row: number | null;
  col: number | null;
  type: "reroll" | "single-cell" | null;
};
export type RecognizedCount = Observation & { value: number | null };
export type RecognitionSource = {
  kind: "capture-stub" | "manual-mock" | "calibrated-board" | "automatic-board";
  timestamp: number;
  dimensions?: { width: number; height: number };
};
export type RecognitionResult = {
  board: RecognizedCell[][];
  pieces: RecognizedPieceSlot[];
  hiddenItems: RecognizedHiddenItem[];
  hiddenItemsStatus: RecognitionStatus;
  abilities: { reroll: RecognizedCount; singleCell: RecognizedCount };
  source: RecognitionSource;
  summary: {
    engine: "unimplemented" | "manual-copy" | "rgb-samples" | "grid-tiles";
  };
};
export type RecognitionInput = { kind: "frame"; frame: CapturedFrame };
export interface RecognitionEngine {
  recognize(input: RecognitionInput): RecognitionResult;
}
export type ReviewState = {
  original: RecognitionResult;
  draft: RecognitionResult;
  revision: number;
  confirmed: boolean;
};
export type FieldIssue = {
  path: string;
  code: "INVALID" | "UNRESOLVED" | "UNCONFIRMED" | "STALE" | "PENDING";
  message: string;
};
