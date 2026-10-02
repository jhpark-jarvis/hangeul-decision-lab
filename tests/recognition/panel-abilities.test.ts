import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import {
  abilityRegions,
  recognizePanelAbilities,
} from "../../src/features/recognition/panel-abilities";
import { panelFixture, rowMask, paint, drawGlyph } from "./panel-fixture";

const digits = [
  ["01110", "11011", "11011", "11011", "11011", "11011", "01110"],
  ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
  ["01110", "11011", "00011", "00110", "01100", "11000", "11111"],
  ["11110", "00011", "00011", "01110", "00011", "00011", "11110"],
  ["00011", "00111", "01111", "11011", "11111", "00011", "00011"],
  ["11111", "11000", "11000", "11110", "00011", "00011", "11110"],
  ["01110", "11000", "11000", "11110", "11011", "11011", "01110"],
  ["11111", "00011", "00110", "00110", "01100", "01100", "01100"],
  ["01110", "11011", "11011", "01110", "11011", "11011", "01110"],
  ["01110", "11011", "11011", "01111", "00011", "00011", "01110"],
];
const templates = digits.map((rows, value) => ({
  value: String(value),
  mask: rowMask(rows),
}));
function fixture(a: number, b: number, total: number, scale = 1) {
  const { frame, board } = panelFixture(scale, 6),
    regions = abilityRegions(board);
  for (const key of ["singleCell", "reroll", "total"] as const) {
    paint(
      frame,
      regions[key],
      key === "total" ? [230, 251, 252] : [20, 102, 190],
    );
    const n = key === "singleCell" ? a : key === "reroll" ? b : total;
    if (n >= 0)
      drawGlyph(
        frame,
        regions[key],
        digits[n],
        key === "total" ? [20, 130, 160] : [255, 255, 255],
        key === "total" ? 0.16 : 0.11,
      );
  }
  return { frame, regions };
}
describe("local ability digit recognition", () => {
  it.each(Array.from({ length: 8 }, (_, a) => ({ a })))(
    "reads all valid pairs beginning with $a at three scales",
    ({ a }) => {
      for (let b = 0; b <= 7 - a; b++)
        for (const scale of [0.75, 1, 1.5]) {
          const { frame, regions } = fixture(a, b, a + b, scale);
          const hash = createHash("sha256").update(frame.pixels).digest("hex");
          expect(
            recognizePanelAbilities(frame, regions, templates).abilities,
            `${a}/${b}/${scale}`,
          ).toEqual({
            singleCell: { value: a, status: "recognized" },
            reroll: { value: b, status: "recognized" },
          });
          expect(createHash("sha256").update(frame.pixels).digest("hex")).toBe(
            hash,
          );
        }
    },
  );
  it("rejects inconsistent totals and never guesses blank as zero", () => {
    const bad = fixture(1, 2, 4);
    expect(
      recognizePanelAbilities(bad.frame, bad.regions, templates).abilities,
    ).toEqual({
      singleCell: { value: null, status: "uncertain" },
      reroll: { value: null, status: "uncertain" },
    });
    const partial = fixture(1, -1, 3);
    expect(
      recognizePanelAbilities(partial.frame, partial.regions, templates)
        .abilities,
    ).toEqual({
      singleCell: { value: 1, status: "recognized" },
      reroll: { value: null, status: "unknown" },
    });
    const noTotal = fixture(0, 0, -1);
    expect(
      recognizePanelAbilities(noTotal.frame, noTotal.regions, templates)
        .abilities.singleCell.value,
    ).toBe(null);
  });
  it("rejects out-of-range numbers, duplicate template values and missing pixels", () => {
    const f = fixture(8, 0, 8);
    expect(
      recognizePanelAbilities(f.frame, f.regions, templates).abilities
        .singleCell.value,
    ).toBe(null);
    const good = fixture(1, 2, 3);
    const ambiguous = templates.concat({ value: "5", mask: templates[1].mask });
    expect(
      recognizePanelAbilities(good.frame, good.regions, ambiguous).abilities
        .singleCell.value,
    ).toBe(null);
    const missing = { ...good.frame, pixels: new Uint8ClampedArray(0) };
    expect(
      recognizePanelAbilities(missing, good.regions, templates).abilities.reroll
        .value,
    ).toBe(null);
    good.frame.pixels[
      Math.ceil(good.regions.reroll.y) * good.frame.width * 4 +
        Math.ceil(good.regions.reroll.x) * 4 +
        3
    ] = 0;
    expect(
      recognizePanelAbilities(good.frame, good.regions, templates).abilities
        .reroll,
    ).toEqual({ value: null, status: "unknown" });
    expect(templates[0].mask.data.some(Boolean)).toBe(true);
  });
  it("preserves a readable sibling and rejects counts exceeding the displayed total", () => {
    const { frame, regions } = fixture(2, -1, 1);
    expect(
      recognizePanelAbilities(frame, regions, templates).abilities.singleCell
        .value,
    ).toBe(null);
    const partial = fixture(2, 1, 3);
    const outside = {
      ...partial.regions,
      reroll: { ...partial.regions.reroll, x: partial.frame.width },
    };
    expect(
      recognizePanelAbilities(partial.frame, outside, templates).abilities,
    ).toEqual({
      singleCell: { value: 2, status: "recognized" },
      reroll: { value: null, status: "unknown" },
    });
  });
});
