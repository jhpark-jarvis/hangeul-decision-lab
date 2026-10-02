import { describe, expect, it } from "vitest";
import {
  createCalibratedBoardEngine,
  type BoardRecognitionConfig,
  type RGB,
} from "../../src/features/recognition/board";
import type { CapturedFrame } from "../../src/features/capture/capture";
import {
  createReviewState,
  updateReview,
  validateRecognitionResult,
  validateReviewedState,
} from "../../src/features/recognition/review";
import { recognizeManualState } from "../../src/features/recognition/mock";
import { applyReviewedState } from "../../src/features/recognition/adapter";
import {
  applyStep,
  createSession,
  installAnalysis,
  loadNextPieces,
} from "../../src/features/puzzle/session";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";

// Synthetic palette and coordinates, deliberately unrelated to the actual game.
const empty: RGB = [20, 30, 40];
const filled: RGB = [180, 190, 200];
const other: RGB = [200, 0, 200];
const config = (): BoardRecognitionConfig => ({
  region: { x: 4, y: 6, width: 100, height: 160 },
  emptyColors: [empty],
  occupiedColors: [filled],
  maxDistance: 12,
  minDistanceGap: 20,
  minMatchedFraction: 0.8,
  sampleFraction: 0.6,
  samplesPerAxis: 3,
});
const board = (test: (r: number, c: number) => boolean) =>
  Array.from({ length: 16 }, (_, r) =>
    Array.from({ length: 10 }, (_, c) => test(r, c)),
  );
function pixel(frame: CapturedFrame, x: number, y: number, rgb: RGB, a = 255) {
  frame.pixels.set([...rgb, a], (y * frame.width + x) * 4);
}
function fixture(
  expected = board((r, c) => (r + c) % 2 === 0),
  settings = config(),
): CapturedFrame {
  const frame: CapturedFrame = {
    width: 128,
    height: 192,
    timestamp: 1234,
    pixels: new Uint8ClampedArray(128 * 192 * 4),
  };
  const { x, y, width, height } = settings.region;
  for (let py = 0; py < frame.height; py++) {
    for (let px = 0; px < frame.width; px++) {
      const r = Math.floor(((py + 0.5 - y) / height) * 16);
      const c = Math.floor(((px + 0.5 - x) / width) * 10);
      const color =
        r >= 0 && r < 16 && c >= 0 && c < 10
          ? expected[r][c]
            ? filled
            : empty
          : other;
      pixel(frame, px, py, color);
    }
  }
  return frame;
}
const recognize = (frame: CapturedFrame, settings = config()) =>
  createCalibratedBoardEngine(settings).recognize({ kind: "frame", frame });
function firstCell(
  frame: CapturedFrame,
  paint: (x: number, y: number) => void,
) {
  for (let y = 6; y < 16; y++) for (let x = 4; x < 14; x++) paint(x, y);
}

describe("calibrated board recognition prototype", () => {
  it.each(["empty", "filled", "checker", "edges"])(
    "recognizes all 160 %s cells in an explicit offset region without mutating pixels",
    (pattern) => {
      const expected = board((r, c) =>
        pattern === "filled"
          ? true
          : pattern === "checker"
            ? (r + c) % 2 === 0
            : pattern === "edges"
              ? r === 0 || r === 15 || c === 0 || c === 9
              : false,
      );
      const frame = fixture(expected);
      const copy = frame.pixels.slice();
      const result = recognize(frame);
      expect(result.board.map((row) => row.map((c) => c.occupied))).toEqual(
        expected,
      );
      expect(result.board.flat().every((c) => c.status === "recognized")).toBe(
        true,
      );
      expect(validateRecognitionResult(result)).toEqual([]);
      expect(frame.pixels).toEqual(copy);
      expect(result.source).toEqual({
        kind: "calibrated-board",
        timestamp: 1234,
        dimensions: { width: 128, height: 192 },
      });
      expect(result.summary.engine).toBe("rgb-samples");
      expect(JSON.stringify(result)).not.toMatch(/pixels|Colors|confidence/);
      expect(result.pieces.every((p) => p.status === "unknown")).toBe(true);
      expect(result.hiddenItemsStatus).toBe("unknown");
      expect(result.abilities.reroll.value).toBeNull();
      expect(result.abilities.singleCell.value).toBeNull();
    },
  );
  it("handles fractional regions/cell dimensions without a game-specific origin", () => {
    const settings = config();
    settings.region = { x: 7.25, y: 9.5, width: 103, height: 165 };
    const expected = board((r, c) => r === c || r + c === 15);
    const result = recognize(fixture(expected, settings), settings);
    expect(result.board.map((row) => row.map((c) => c.occupied))).toEqual(
      expected,
    );
  });
  it("accepts explicit multi-color occupied calibration, including bounded noise", () => {
    const settings = config();
    const alternate: RGB = [80, 100, 220];
    settings.occupiedColors = [filled, alternate];
    const frame = fixture();
    firstCell(frame, (x, y) => pixel(frame, x, y, [82, 101, 219]));
    expect(recognize(frame, settings).board[0][0]).toEqual({
      row: 0,
      col: 0,
      occupied: true,
      status: "recognized",
    });
  });
  it.each(["foreign", "transparent", "nearby-palettes"])(
    "keeps %s unknown instead of assuming empty",
    (mode) => {
      const settings = config();
      const frame = fixture();
      if (mode === "nearby-palettes") {
        settings.occupiedColors = [[21, 30, 40]];
        firstCell(frame, (x, y) => pixel(frame, x, y, empty));
      } else
        firstCell(frame, (x, y) =>
          pixel(
            frame,
            x,
            y,
            mode === "foreign" ? other : empty,
            mode === "transparent" ? 0 : 255,
          ),
        );
      expect(recognize(frame, settings).board[0][0]).toEqual({
        row: 0,
        col: 0,
        occupied: null,
        status: "unknown",
      });
    },
  );
  it("keeps mixed opposite-class samples uncertain even above a majority threshold", () => {
    const frame = fixture();
    // 8 of 9 sampled pixels are filled; one contradicts that classification.
    pixel(frame, 7, 9, empty);
    expect(recognize(frame).board[0][0].status).toBe("uncertain");
    expect(recognize(frame).board[0][0].occupied).toBeNull();
  });
  it("applies the explicit coverage boundary without treating it as probability", () => {
    const frame = fixture();
    pixel(frame, 7, 9, other);
    const settings = config();
    expect(recognize(frame, settings).board[0][0].occupied).toBe(true);
    settings.minMatchedFraction = 0.9;
    expect(recognize(frame, settings).board[0][0]).toMatchObject({
      occupied: null,
      status: "uncertain",
    });
  });
  it("ignores border decorations only within the caller's central sampling window", () => {
    const frame = fixture(board(() => false));
    firstCell(frame, (x, y) => {
      if (x === 4 || x === 13 || y === 6 || y === 15) pixel(frame, x, y, other);
    });
    expect(recognize(frame).board[0][0].occupied).toBe(false);
  });
  it("detaches calibration at construction and emits independent results", () => {
    const settings = config();
    const editableColor: [number, number, number] = [...empty];
    settings.emptyColors = [editableColor];
    const engine = createCalibratedBoardEngine(settings);
    const frame = fixture();
    const first = engine.recognize({ kind: "frame", frame });
    settings.region.x = 1000;
    editableColor[0] = 255;
    first.board[0][0].occupied = false;
    const second = engine.recognize({ kind: "frame", frame });
    expect(second.board[0][0].occupied).toBe(true);
  });
  it.each([
    { region: { x: -1, y: 0, width: 100, height: 160 } },
    { region: { x: 0, y: NaN, width: 100, height: 160 } },
    { region: { x: 0, y: 0, width: 10, height: 16 } },
    { emptyColors: [] },
    { occupiedColors: [[256, 0, 0]] },
    { occupiedColors: Array(1) },
    { emptyColors: [Array(3)] },
    { emptyColors: Array(17).fill(empty) },
    { maxDistance: Infinity },
    { maxDistance: -1 },
    { minDistanceGap: 0 },
    { minMatchedFraction: 0.5 },
    { minMatchedFraction: 1.1 },
    { sampleFraction: 0 },
    { sampleFraction: 1.1 },
    { samplesPerAxis: 1.5 },
    { samplesPerAxis: 10 },
  ])("rejects malformed calibration %j", (patch) => {
    expect(() =>
      createCalibratedBoardEngine({
        ...config(),
        ...patch,
      } as BoardRecognitionConfig),
    ).toThrow("인식 설정");
  });
  it.each([
    { width: 0 },
    { height: 1.5 },
    { timestamp: NaN },
    { pixels: new Uint8ClampedArray(2) },
    { pixels: new Uint8Array(128 * 192 * 4) },
    { width: 128, height: 192, pixels: [] },
    { width: 16_777_217, height: 1 },
  ])("rejects malformed frame %j and allows a later valid call", (patch) => {
    const engine = createCalibratedBoardEngine(config());
    const frame = fixture();
    expect(() =>
      engine.recognize({
        kind: "frame",
        frame: { ...frame, ...patch } as CapturedFrame,
      }),
    ).toThrow("프레임 형식");
    expect(
      engine.recognize({ kind: "frame", frame }).board[0][0].occupied,
    ).toBe(true);
  });
  it("rejects a region outside the frame instead of clamping/rescaling", () => {
    const settings = config();
    settings.region.x = 29;
    expect(() => recognize(fixture(), settings)).toThrow("보드 영역 범위");
  });
  it("cannot enter the session until non-board fields are reviewed and confirmed", () => {
    const input = loadNextPieces(createSession(), ["DOT", "DOT", "DOT"]);
    const expected = board((r, c) => r === 0 && c < 9);
    const recognition = recognize(fixture(expected));
    const original = structuredClone(recognition);
    let review = { ...createReviewState(recognition), confirmed: true };
    expect(validateReviewedState(review).length).toBe(6);
    expect(
      applyReviewedState(input, review, JSON.stringify(input.game)).ok,
    ).toBe(false);
    review = updateReview(review, (draft) => {
      const manual = recognizeManualState(input.game);
      draft.pieces = manual.pieces;
      draft.hiddenItemsStatus = "recognized";
      draft.abilities = manual.abilities;
    });
    expect(review.confirmed).toBe(false);
    expect(
      applyReviewedState(input, review, JSON.stringify(input.game)).ok,
    ).toBe(false);
    review = { ...review, confirmed: true };
    const result = applyReviewedState(
      input,
      review,
      JSON.stringify(input.game),
    );
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.game.board).toEqual(expected);
    expect(input.game.board.flat().every((c) => !c)).toBe(true);
    expect(recognition).toEqual(original);
    expect(review.original).toEqual(original);
    const analyzed = installAnalysis(
      result.session,
      result.session.version,
      solveTurn(result.session.game, getInitialCatalog()),
      1,
    );
    const applied = applyStep(analyzed, analyzed.version, 0);
    expect(applied.error).toBeNull();
    expect(applied.game).not.toEqual(result.session.game);
  });
});
