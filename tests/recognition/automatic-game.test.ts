import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { recognizeAutomaticGame } from "../../src/features/recognition/automatic-game";
import { abilityRegions } from "../../src/features/recognition/panel-abilities";
import {
  createReviewState,
  validateRecognitionResult,
  validateReviewedState,
} from "../../src/features/recognition/review";
import { automaticFixture, referenceRows } from "./automatic-fixture";
import {
  panelFixture,
  drawPiece,
  drawUsed,
  drawGlyph,
  paint,
  rowMask,
  syntheticUsedRows,
} from "./panel-fixture";
const zero = ["01110", "11011", "11011", "11011", "11011", "11011", "01110"];
const templates = [
  { value: "0", mask: rowMask(zero) },
  { value: "사용 완료", mask: rowMask(syntheticUsedRows) },
];
function fixture() {
  const full = panelFixture(),
    board = automaticFixture();
  for (let y = 0; y < board.height; y++)
    full.frame.pixels.set(
      board.pixels.subarray(y * board.width * 4, (y + 1) * board.width * 4),
      y * full.frame.width * 4,
    );
  drawPiece(full.frame, full.cards[0], ["11", "10", "11"]);
  drawUsed(full.frame, full.cards[1]);
  drawUsed(full.frame, full.cards[2]);
  for (const [key, r] of Object.entries(abilityRegions(full.board))) {
    paint(
      full.frame,
      { x: r.x - 4, y: r.y - 4, width: r.width + 8, height: r.height + 8 },
      key === "total" ? [230, 251, 252] : [20, 102, 190],
    );
    drawGlyph(
      full.frame,
      r,
      zero,
      key === "total" ? [20, 130, 160] : [255, 255, 255],
      key === "total" ? 0.16 : 0.11,
    );
  }
  return full;
}
describe("same-frame automatic game contract", () => {
  it("merges board, pieces and counts without confirming or creating hidden items", () => {
    const { frame } = fixture(),
      before = createHash("sha256").update(frame.pixels).digest("hex");
    const found = recognizeAutomaticGame(frame, templates);
    expect(found.ok).toBe(true);
    if (!found.ok) return;
    expect(
      found.result.board.map((r) =>
        r.map((c) => (c.occupied ? "1" : "0")).join(""),
      ),
    ).toEqual(referenceRows);
    expect(
      found.result.pieces.map((p) => (p.empty ? "empty" : p.pieceId)),
    ).toEqual(["C_5", "empty", "empty"]);
    expect(
      found.result.abilities.singleCell.value,
      JSON.stringify({
        region: found.region,
        abilities: found.result.abilities,
      }),
    ).toBe(0);
    expect(found.result.abilities.reroll.value).toBe(0);
    expect(found.result.source).toEqual({
      kind: "automatic-game",
      timestamp: 123,
      dimensions: { width: frame.width, height: frame.height },
    });
    expect(found.result.summary.engine).toBe("grid-panel");
    expect(validateRecognitionResult(found.result)).toEqual([]);
    const review = createReviewState(found.result);
    expect(review.confirmed).toBe(false);
    expect(validateReviewedState(review)).not.toEqual([]);
    expect(found.result.hiddenItems).toEqual([]);
    expect(found.result.hiddenItemsStatus).toBe("unknown");
    expect(createHash("sha256").update(frame.pixels).digest("hex")).toBe(
      before,
    );
  });
  it("keeps board and a known slot when another slot or fonts are unavailable", () => {
    const { frame, cards } = fixture();
    paint(frame, cards[2], [30, 30, 30]);
    const found = recognizeAutomaticGame(frame, []);
    expect(found.ok).toBe(true);
    if (!found.ok) return;
    expect(
      found.result.board.flat().filter((c) => c.occupied === null),
    ).toHaveLength(0);
    expect(found.result.pieces[0].pieceId).toBe("C_5");
    expect(found.result.pieces[1].empty).toBeNull();
    expect(found.result.pieces[2].empty).toBeNull();
    expect(found.result.abilities.singleCell.value).toBeNull();
    expect(found.result.abilities.reroll.value).toBeNull();
  });
  it("returns the existing board failure and never reads a replacement frame", () => {
    const f = fixture().frame;
    f.pixels.fill(0);
    expect(recognizeAutomaticGame(f, templates)).toEqual({
      ok: false,
      reason: "NOT_FOUND",
    });
    expect(recognizeAutomaticGame({ ...f, timestamp: NaN }, templates)).toEqual(
      { ok: false, reason: "INVALID_FRAME" },
    );
  });
});
