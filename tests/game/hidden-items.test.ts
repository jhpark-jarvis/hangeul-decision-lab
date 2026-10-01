import { describe, expect, it } from "vitest";
import { collectHiddenItems } from "../../src/domain/game/hidden-items";
import type { HiddenItem } from "../../src/domain/game/types";
import { freezeDeep, success } from "./fixtures";

const items: HiddenItem[] = [
  { row: 7, col: 2, type: "reroll" },
  { row: 2, col: 8, type: "reroll" },
  { row: 2, col: 1, type: "single-cell" },
  { row: 5, col: 0, type: "single-cell" },
];
describe("cleared-row reward policy", () => {
  it("acquires in row/col order regardless of item/cleared-row input ordering", () => {
    const result = success(
      collectHiddenItems(
        freezeDeep(items),
        { reroll: 3, singleCell: 2 },
        [7, 2],
      ),
    );
    expect(result.abilities).toEqual({ reroll: 4, singleCell: 3 });
    expect(result.acquiredItems).toEqual([items[2], items[1]]);
    expect(result.retainedItems).toEqual([items[0]]);
    expect(result.hiddenItems).toEqual([items[3], items[0]]);
    expect(
      success(
        collectHiddenItems(
          [...items].reverse(),
          { reroll: 3, singleCell: 2 },
          [2, 7],
        ),
      ),
    ).toEqual(result);
  });
  it("retains all unacquired icons at capacity, including icons on cleared rows", () => {
    const result = success(
      collectHiddenItems(items, { reroll: 7, singleCell: 0 }, [2, 7]),
    );
    expect(result.acquiredItems).toEqual([]);
    expect(result.retainedItems).toEqual([items[2], items[1], items[0]]);
    expect(result.hiddenItems).toEqual([
      items[2],
      items[1],
      items[3],
      items[0],
    ]);
    expect(result.abilities).toEqual({ reroll: 7, singleCell: 0 });
  });
  it("acquires a retained icon after capacity is freed and its row clears again", () => {
    const item: HiddenItem = { row: 2, col: 1, type: "single-cell" };
    const held = success(
      collectHiddenItems([item], { reroll: 7, singleCell: 0 }, [2]),
    );
    expect(held.acquiredItems).toEqual([]);
    expect(held.retainedItems).toEqual([item]);
    held.retainedItems[0].col = 9;
    expect(held.hiddenItems).toEqual([item]); // info cannot mutate the next state.
    const collected = success(
      collectHiddenItems(held.hiddenItems, { reroll: 6, singleCell: 0 }, [2]),
    );
    expect(collected.abilities).toEqual({ reroll: 6, singleCell: 1 });
    expect(collected.acquiredItems).toEqual([item]);
    expect(collected.retainedItems).toEqual([]);
    expect(collected.hiddenItems).toEqual([]);
    expect(held.hiddenItems).toEqual([item]);
  });
  it("returns detached objects on no-clear and reward paths", () => {
    const counts = freezeDeep({ reroll: 0, singleCell: 0 });
    const result = success(collectHiddenItems(items, counts, []));
    result.hiddenItems[0].row = 0;
    result.abilities.reroll = 1;
    expect(items[0].row).toBe(7);
    expect(counts.reroll).toBe(0);
    const collected = success(collectHiddenItems(items, counts, [7]));
    collected.acquiredItems[0].col = 9;
    expect(items[0].col).toBe(2);
  });
  it.each(
    [null, [16], [-1], [0.5], [NaN], ["1"], [2, 2], Array(1)].map((rows) => ({
      rows,
    })),
  )("rejects malformed cleared rows %#", ({ rows }) => {
    expect(
      collectHiddenItems(items, { reroll: 0, singleCell: 0 }, rows),
    ).toMatchObject({ ok: false, error: { code: "INVALID_CLEARED_ROWS" } });
  });
  it("validates item and capacity ingress rather than silently repairing it", () => {
    expect(
      collectHiddenItems(null, { reroll: 0, singleCell: 0 }, []),
    ).toMatchObject({ ok: false, error: { code: "INVALID_HIDDEN_ITEMS" } });
    expect(
      collectHiddenItems([], { reroll: 8, singleCell: 0 }, []),
    ).toMatchObject({ ok: false, error: { code: "INVALID_ABILITIES" } });
  });
});
