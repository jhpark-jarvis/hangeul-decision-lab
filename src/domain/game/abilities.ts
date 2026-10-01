import {
  gameError,
  MAX_ABILITY_COUNT,
  type AbilityCounts,
  type GameResult,
} from "./types";

export function validateAbilities(
  input: unknown,
): GameResult<{ abilities: AbilityCounts }> {
  if (!input || typeof input !== "object") {
    return gameError("INVALID_ABILITIES", "능력 수 입력이 필요합니다.");
  }
  const { reroll, singleCell } = input as Record<string, unknown>;
  if (
    typeof reroll !== "number" ||
    typeof singleCell !== "number" ||
    !Number.isInteger(reroll) ||
    !Number.isInteger(singleCell) ||
    reroll < 0 ||
    singleCell < 0 ||
    reroll + singleCell > MAX_ABILITY_COUNT
  ) {
    return gameError(
      "INVALID_ABILITIES",
      `능력은 비음수 정수이며 합계 ${MAX_ABILITY_COUNT}개 이하여야 합니다.`,
    );
  }
  return { ok: true, abilities: { reroll, singleCell } };
}
