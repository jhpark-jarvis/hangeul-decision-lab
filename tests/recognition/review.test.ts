import { describe, expect, it } from "vitest";
import {
  createSession,
  loadNextPieces,
  installAnalysis,
  applyStep,
} from "../../src/features/puzzle/session";
import {
  recognizeManualState,
  unknownRecognitionEngine,
} from "../../src/features/recognition/mock";
import {
  createReviewState,
  updateReview,
  validateRecognitionResult,
  validateReviewedState,
} from "../../src/features/recognition/review";
import { applyReviewedState } from "../../src/features/recognition/adapter";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";
import { validateGameState } from "../../src/domain/game/game-state";

const session = () => loadNextPieces(createSession(), ["DOT", "DOT", "DOT"]);
const review = () => ({
  ...createReviewState(recognizeManualState(session().game)),
  confirmed: true,
});
describe("recognition review boundary", () => {
  it("rejects sparse board and piece arrays rather than silently skipping unknowns", () => {
    const r = review().draft;
    r.board[0] = Array(10);
    r.pieces = Array(3);
    expect(
      validateRecognitionResult(r).filter((issue) =>
        issue.path.startsWith("board.0"),
      ).length,
    ).toBe(10);
    expect(
      validateRecognitionResult(r).filter((issue) =>
        issue.path.startsWith("pieces"),
      ).length,
    ).toBe(3);
  });
  it("preserves the manual GameState contract and existing Analyze/Apply", () => {
    const input = session();
    const snapshot = structuredClone(input);
    const result = applyReviewedState(
      input,
      review(),
      JSON.stringify(input.game),
    );
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.game).toEqual(input.game);
    expect(result.session.game).not.toBe(input.game);
    expect(validateGameState(result.session.game).ok).toBe(true);
    const solution = solveTurn(result.session.game, getInitialCatalog());
    const installed = installAnalysis(
      result.session,
      result.session.version,
      solution,
      1,
    );
    const applied = applyStep(installed, installed.version, 0);
    expect(applied.game.remainingPieces).toHaveLength(2);
    expect(input).toEqual(snapshot);
  });
  it("detaches original, draft and subsequent edits; edits remove confirmation", () => {
    const recognition = recognizeManualState(session().game);
    const source = structuredClone(recognition);
    const initial = { ...createReviewState(recognition), confirmed: true };
    const edited = updateReview(initial, (draft) => {
      draft.board[0][0].occupied = true;
      draft.abilities.reroll.value = 2;
    });
    expect(recognition).toEqual(source);
    expect(initial.original).toEqual(source);
    expect(initial.draft).toEqual(source);
    expect(edited.draft.board[0][0].occupied).toBe(true);
    expect(edited.confirmed).toBe(false);
    expect(edited.revision).toBe(1);
    recognition.board[0][0].occupied = true;
    expect(initial.original.board[0][0].occupied).toBe(false);
  });
  it.each(["unknown", "uncertain"] as const)(
    "blocks %s cells even when their value is present",
    (status) => {
      const r = review();
      r.draft.board[2][3].status = status;
      expect(validateReviewedState(r)).toContainEqual(
        expect.objectContaining({ path: "board.2.3", code: "UNRESOLVED" }),
      );
      expect(
        applyReviewedState(session(), r, JSON.stringify(session().game)).ok,
      ).toBe(false);
    },
  );
  it("blocks unresolved pieces and rejects unknown catalog IDs", () => {
    const r = review();
    r.draft.pieces[1].pieceId = null;
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ path: "pieces.1", code: "UNRESOLVED" }),
    );
    r.draft.pieces[1].pieceId = "not-a-piece";
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ path: "pieces.1", code: "INVALID" }),
    );
  });
  it("distinguishes confirmed consumed slots from unknown slots", () => {
    const r = review();
    r.draft.pieces[1] = {
      slot: 1,
      empty: true,
      pieceId: null,
      status: "recognized",
    };
    const result = applyReviewedState(
      session(),
      r,
      JSON.stringify(session().game),
    );
    expect(result.ok).toBe(true);
    if (result.ok)
      expect(
        result.session.game.remainingPieces.map((p) => p.pieceIndex),
      ).toEqual([0, 2]);
    r.draft.pieces[1].empty = null;
    expect(validateReviewedState(r).length).toBeGreaterThan(0);
  });
  it.each([-1, 8, 1.2, NaN])(
    "rejects invalid ability %s at its field",
    (value) => {
      const r = review();
      r.draft.abilities.reroll.value = value;
      expect(validateReviewedState(r)).toContainEqual(
        expect.objectContaining({ path: "abilities.reroll", code: "INVALID" }),
      );
    },
  );
  it("rejects combined abilities over 7 and unknown counts", () => {
    const r = review();
    r.draft.abilities.reroll.value = 6;
    r.draft.abilities.singleCell.value = 2;
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ path: "abilities", code: "INVALID" }),
    );
    r.draft.abilities.reroll.value = null;
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ path: "abilities.reroll", code: "UNRESOLVED" }),
    );
  });
  it.each([
    { row: 16, col: 0 },
    { row: -1, col: 0 },
    { row: 0, col: 10 },
    { row: 0, col: 0.5 },
  ])("rejects item coordinates %j", (position) => {
    const r = review();
    r.draft.hiddenItems = [
      { ...position, type: "reroll", status: "recognized" },
    ];
    expect(
      validateReviewedState(r).some((issue) =>
        issue.path.startsWith("hiddenItems.0"),
      ),
    ).toBe(true);
  });
  it("requires complete item list and rejects duplicates/over3", () => {
    const r = review();
    r.draft.hiddenItemsStatus = "unknown";
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({
        path: "hiddenItemsStatus",
        code: "UNRESOLVED",
      }),
    );
    r.draft.hiddenItemsStatus = "recognized";
    r.draft.hiddenItems = Array.from({ length: 2 }, () => ({
      row: 0,
      col: 0,
      type: "reroll",
      status: "recognized",
    }));
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ path: "hiddenItems.1", code: "INVALID" }),
    );
    r.draft.hiddenItems.push(...structuredClone(r.draft.hiddenItems));
    expect(
      validateRecognitionResult(r.draft).some((i) => i.path === "hiddenItems"),
    ).toBe(true);
  });
  it("requires explicit confirmation regardless of recognized status", () => {
    const r = review();
    r.confirmed = false;
    expect(validateReviewedState(r)).toContainEqual(
      expect.objectContaining({ code: "UNCONFIRMED" }),
    );
    expect(
      applyReviewedState(session(), r, JSON.stringify(session().game)).ok,
    ).toBe(false);
  });
  it("rejects stale manual state without changing the session", () => {
    const input = session();
    const expected = JSON.stringify(input.game);
    input.game.board[0][0] = true;
    const copy = structuredClone(input);
    const result = applyReviewedState(input, review(), expected);
    expect(result).toEqual({
      ok: false,
      issues: [expect.objectContaining({ code: "STALE" })],
    });
    expect(input).toEqual(copy);
  });
  it("cannot bypass pending reroll", () => {
    const input = session();
    const target = input.game.remainingPieces[0];
    input.game.pendingReroll = { pieceIndex: 0, instanceId: target.instanceId };
    expect(
      applyReviewedState(input, review(), JSON.stringify(input.game)),
    ).toEqual({
      ok: false,
      issues: [expect.objectContaining({ code: "PENDING" })],
    });
  });
  it("accepts a legal corrected board/piece/item/ability and rejects old Apply tokens", () => {
    const input = session();
    const analyzed = installAnalysis(
      input,
      input.version,
      solveTurn(input.game, getInitialCatalog()),
      1,
    );
    const r = review();
    r.draft.board[0][0].occupied = true;
    r.draft.pieces[2].pieceId = "C_5";
    r.draft.abilities.singleCell.value = 1;
    r.draft.hiddenItems = [
      { row: 1, col: 2, type: "reroll", status: "recognized" },
    ];
    const result = applyReviewedState(
      analyzed,
      r,
      JSON.stringify(analyzed.game),
    );
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.analysis).toBeNull();
    expect(result.session.version).toBeGreaterThan(analyzed.version);
    const rejected = applyStep(result.session, analyzed.version, 0);
    expect(rejected.game).toEqual(result.session.game);
    expect(rejected.error).toBeTruthy();
  });
  it("stub consumes only frame metadata, returns unknowns without fake confidence/pixels", () => {
    const pixels = new Uint8ClampedArray(8);
    const result = unknownRecognitionEngine.recognize({
      kind: "frame",
      frame: { width: 2, height: 1, timestamp: 1, pixels },
    });
    expect(validateRecognitionResult(result)).toEqual([]);
    expect(
      validateReviewedState({ ...createReviewState(result), confirmed: true }),
    ).toHaveLength(166);
    expect(JSON.stringify(result)).not.toContain("confidence");
    expect(JSON.stringify(result)).not.toContain("pixels");
    expect(result.source.dimensions).toEqual({ width: 2, height: 1 });
    expect(pixels).toEqual(new Uint8ClampedArray(8));
  });
  it.each([null, {}, { board: [] }])(
    "rejects malformed recognition %j",
    (input) =>
      expect(validateRecognitionResult(input).length).toBeGreaterThan(0),
  );
  it("rejects bad confidence, coordinates, dimensions and duplicate slot order", () => {
    const r = review().draft;
    r.board[0][0].confidence = 1.1;
    r.board[0][1].row = 4;
    r.pieces[1].slot = 0;
    r.source.dimensions = { width: 0, height: 1 };
    expect(validateRecognitionResult(r).map((i) => i.path)).toEqual([
      "board.0.0",
      "board.0.1",
      "pieces.1",
      "source.dimensions",
    ]);
  });
});
