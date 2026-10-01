import { BOARD_HEIGHT, BOARD_WIDTH } from "../board/board";
import { validateAbilities } from "./abilities";
import {
  gameError,
  MAX_ABILITY_COUNT,
  type AbilityCounts,
  type GameResult,
  type HiddenItem,
} from "./types";

export function validateHiddenItems(
  input: unknown,
): GameResult<{ hiddenItems: HiddenItem[] }> {
  if (!Array.isArray(input))
    return gameError("INVALID_HIDDEN_ITEMS", "아이템 배열이 필요합니다.");
  const hiddenItems: HiddenItem[] = [];
  const positions = new Set<string>();
  for (const item of input) {
    if (!item || typeof item !== "object")
      return gameError(
        "INVALID_HIDDEN_ITEMS",
        "아이템 좌표와 종류가 필요합니다.",
      );
    const { row, col, type } = item as Record<string, unknown>;
    if (
      typeof row !== "number" ||
      typeof col !== "number" ||
      !Number.isInteger(row) ||
      !Number.isInteger(col) ||
      row < 0 ||
      row >= BOARD_HEIGHT ||
      col < 0 ||
      col >= BOARD_WIDTH ||
      (type !== "reroll" && type !== "single-cell") ||
      positions.has(`${row},${col}`)
    ) {
      return gameError(
        "INVALID_HIDDEN_ITEMS",
        "아이템은 보드 안의 서로 다른 좌표와 지원 종류를 가져야 합니다.",
      );
    }
    positions.add(`${row},${col}`);
    hiddenItems.push({ row, col, type });
  }
  return { ok: true, hiddenItems };
}

/** ADR-0005: spend first, then acquire cleared-row items in row/col order. */
export function collectHiddenItems(
  inputItems: unknown,
  inputAbilities: unknown,
  inputRows: unknown,
): GameResult<{
  hiddenItems: HiddenItem[];
  abilities: AbilityCounts;
  acquiredItems: HiddenItem[];
  discardedItems: HiddenItem[];
}> {
  const items = validateHiddenItems(inputItems);
  if (!items.ok) return items;
  const counts = validateAbilities(inputAbilities);
  if (!counts.ok) return counts;
  if (!Array.isArray(inputRows))
    return gameError("INVALID_CLEARED_ROWS", "삭제 행 배열이 필요합니다.");
  const rows = new Set<number>();
  for (const row of inputRows) {
    if (
      typeof row !== "number" ||
      !Number.isInteger(row) ||
      row < 0 ||
      row >= BOARD_HEIGHT ||
      rows.has(row)
    ) {
      return gameError(
        "INVALID_CLEARED_ROWS",
        "삭제 행은 보드 안의 중복 없는 정수여야 합니다.",
      );
    }
    rows.add(row);
  }
  const clearedItems = items.hiddenItems
    .filter((item) => rows.has(item.row))
    .sort((a, b) => a.row - b.row || a.col - b.col);
  const abilities = counts.abilities;
  const acquiredItems: HiddenItem[] = [];
  const discardedItems: HiddenItem[] = [];
  for (const item of clearedItems) {
    if (abilities.reroll + abilities.singleCell < MAX_ABILITY_COUNT) {
      abilities[item.type === "reroll" ? "reroll" : "singleCell"]++;
      acquiredItems.push(item);
    } else discardedItems.push(item);
  }
  return {
    ok: true,
    abilities,
    acquiredItems,
    discardedItems,
    hiddenItems: items.hiddenItems.filter((item) => !rows.has(item.row)),
  };
}
