import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import reference from "../fixtures/pieces/catalog-v2.json";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import {
  recognizePanelPieces,
  recognizePieceCards,
} from "../../src/features/recognition/panel-pieces";
import {
  panelFixture,
  drawPiece,
  drawUsed,
  drawGlyph,
  rowMask,
  syntheticUsedRows,
  paint,
} from "./panel-fixture";

// Coordinate oracle independent of production matrix transforms.
function variants(rows: string[]) {
  const cells = rows.flatMap((line, r) =>
    [...line].flatMap((v, c) => (v === "1" ? [[r, c]] : [])),
  );
  const result = new Map<string, string[]>();
  for (const reflected of [false, true]) {
    let coords = cells.map(([r, c]) => [r, reflected ? -c : c]);
    for (let n = 0; n < 4; n++) {
      const minR = Math.min(...coords.map(([r]) => r)),
        minC = Math.min(...coords.map(([, c]) => c));
      const normalized = coords.map(([r, c]) => [r - minR, c - minC]);
      const h = Math.max(...normalized.map(([r]) => r)) + 1,
        w = Math.max(...normalized.map(([, c]) => c)) + 1;
      const shape = Array.from({ length: h }, (_, r) =>
        Array.from({ length: w }, (_, c) =>
          normalized.some(([x, y]) => x === r && y === c) ? "1" : "0",
        ).join(""),
      );
      result.set(shape.join("/"), shape);
      coords = coords.map(([r, c]) => [c, -r]);
    }
  }
  return [...result.values()];
}
const catalog = getInitialCatalog();
describe("local panel piece recognition", () => {
  it.each(reference.pieces)(
    "recognizes $name in every independent rotation/reflection at multiple scales",
    (ref) => {
      for (const scale of [0.75, 1, 1.5])
        for (const rows of variants(ref.rows)) {
          const { frame, board, cards } = panelFixture(scale, 9);
          drawPiece(frame, cards[0], rows);
          const before = createHash("sha256")
            .update(frame.pixels)
            .digest("hex");
          const result = recognizePanelPieces(frame, board, catalog);
          expect(
            result.pieces[0],
            `${ref.id}/${rows.join("/")}/scale${scale}`,
          ).toEqual({
            slot: 0,
            pieceId: ref.id,
            empty: false,
            status: "recognized",
          });
          expect(result.pieces[1].empty).toBe(null);
          expect(createHash("sha256").update(frame.pixels).digest("hex")).toBe(
            before,
          );
        }
    },
  );
  it("keeps duplicate instances and requires explicit used text rather than absence", () => {
    const { frame, board, cards } = panelFixture();
    drawPiece(frame, cards[0], ["1"]);
    drawPiece(frame, cards[1], ["1"]);
    drawUsed(frame, cards[2]);
    expect(recognizePanelPieces(frame, board, catalog).pieces[2].empty).toBe(
      null,
    );
    const templates = [
      { value: "사용 완료", mask: rowMask(syntheticUsedRows) },
    ];
    expect(
      recognizePanelPieces(frame, board, catalog, templates).pieces,
    ).toEqual([
      { slot: 0, pieceId: "DOT", empty: false, status: "recognized" },
      { slot: 1, pieceId: "DOT", empty: false, status: "recognized" },
      { slot: 2, pieceId: null, empty: true, status: "recognized" },
    ]);
    expect(templates[0].mask.data.some(Boolean)).toBe(true);
  });
  it("reads a dimmed explicit usage glyph on cyan without guessing an empty card", () => {
    const { frame, board, cards } = panelFixture();
    paint(frame, cards[1], [20, 170, 190]);
    drawGlyph(frame, cards[1], syntheticUsedRows, [170, 220, 230], 0.037);
    paint(frame, cards[2], [20, 170, 190]);
    const templates = [
      { value: "사용 완료", mask: rowMask(syntheticUsedRows) },
    ];
    const result = recognizePanelPieces(frame, board, catalog, templates);
    expect(result.pieces[1]).toMatchObject({
      empty: true,
      status: "recognized",
    });
    expect(result.pieces[2].empty).toBeNull();
  });
  it("keeps other slots when a shape is unsupported, masked or ambiguous", () => {
    const { frame, board, cards } = panelFixture();
    drawPiece(frame, cards[0], ["1"]);
    drawPiece(frame, cards[1], ["1111"]);
    const result = recognizePanelPieces(frame, board, catalog);
    expect(result.pieces[0].pieceId).toBe("DOT");
    expect(result.pieces[1].pieceId).toBe(null);
    const cloned = catalog.concat([
      { id: "other-dot", name: "Other", shape: [[true]] },
    ]);
    expect(recognizePanelPieces(frame, board, cloned).pieces[0]).toMatchObject({
      status: "uncertain",
      pieceId: null,
      empty: null,
    });
    paint(frame, { ...cards[2], width: cards[2].width / 2 }, [20, 20, 20]);
    expect(recognizePanelPieces(frame, board, catalog).pieces[2].empty).toBe(
      null,
    );
  });
  it("rejects invalid input, out-of-frame cards and duplicate IDs safely", () => {
    const { frame, board, cards } = panelFixture();
    expect(
      recognizePanelPieces(
        { ...frame, pixels: new Uint8ClampedArray(0) },
        board,
        catalog,
      ).pieces.every((p) => p.status === "unknown"),
    ).toBe(true);
    expect(
      recognizePanelPieces(
        frame,
        { ...board, width: NaN },
        catalog,
      ).pieces.every((p) => p.empty === null),
    ).toBe(true);
    expect(
      recognizePieceCards(
        frame,
        cards.map((c) => ({ ...c, x: frame.width })),
        catalog,
      ).pieces.every((p) => p.empty === null),
    ).toBe(true);
    expect(
      recognizePanelPieces(
        frame,
        board,
        catalog.concat(catalog[0]),
      ).pieces.every((p) => p.status === "unknown"),
    ).toBe(true);
  });
});
