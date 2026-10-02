import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";
import type { RecognizedCount, RecognitionResult } from "./types";
import { validPanelFrame, contained, PANEL_PROFILE } from "./panel-profile";
import { readMask, matchText, type TextTemplate } from "./panel-mask";

export type AbilityRegions = {
  singleCell: PixelRegion;
  reroll: PixelRegion;
  total: PixelRegion;
};
export type PanelAbilitiesResult = {
  abilities: RecognitionResult["abilities"];
  total: RecognizedCount;
  reasons: string[];
};
/** Fixed UI profile coordinates relative to the detected board cell pitch. */
export function abilityRegions(board: PixelRegion): AbilityRegions {
  const p = board.width / 10;
  const at = (x: number, y: number, w: number, h: number) => ({
    x: board.x + x * p,
    y: board.y + y * p,
    width: w * p,
    height: h * p,
  });
  return {
    singleCell: at(13.95, 13.95, 0.65, 0.6),
    reroll: at(13.95, 15.35, 0.65, 0.6),
    total: at(12.4, 12.8, 0.4, 0.65),
  };
}
export function recognizePanelAbilities(
  frame: CapturedFrame,
  regions: AbilityRegions,
  templates: readonly TextTemplate[],
): PanelAbilitiesResult {
  const unknown = (): RecognizedCount => ({ value: null, status: "unknown" });
  const result: PanelAbilitiesResult = {
    abilities: { singleCell: unknown(), reroll: unknown() },
    total: unknown(),
    reasons: [],
  };
  if (!validPanelFrame(frame) || !regions) return result;
  const read = (key: keyof AbilityRegions): RecognizedCount => {
    if (!contained(frame, regions[key])) return unknown();
    const test =
      key === "total"
        ? (r: number, g: number, b: number) => r < 110 && g < 195 && b < 220
        : (r: number, g: number, b: number) =>
            Math.min(r, g, b) >= PANEL_PROFILE.glyphMinChannel;
    const mask = readMask(frame, regions[key], test);
    if (!mask) return unknown();
    try {
      const value = matchText(
        mask,
        templates.filter((t) => /^\d$/.test(t.value)),
        "number",
      );
      if (value.status !== "recognized" || value.value === null)
        return { value: null, status: value.status };
      const count = Number(value.value);
      return count <= 7
        ? { value: count, status: "recognized" }
        : { value: null, status: "uncertain" };
    } finally {
      mask.data.fill(0);
    }
  };
  result.abilities.singleCell = read("singleCell");
  result.abilities.reroll = read("reroll");
  result.total = read("total");
  const a = result.abilities.singleCell,
    b = result.abilities.reroll,
    total = result.total;
  if (total.value === null) {
    for (const key of ["singleCell", "reroll"] as const)
      if (result.abilities[key].value !== null)
        result.abilities[key] = { value: null, status: "uncertain" };
    result.reasons.push(
      "보유 합계 숫자를 읽지 못해 능력 수 확인이 필요합니다.",
    );
  } else if (
    a.value !== null &&
    b.value !== null &&
    a.value + b.value !== total.value
  ) {
    result.abilities.singleCell = { value: null, status: "uncertain" };
    result.abilities.reroll = { value: null, status: "uncertain" };
    result.reasons.push("두 능력 수와 보유 합계가 일치하지 않습니다.");
  } else {
    for (const key of ["singleCell", "reroll"] as const) {
      const count = result.abilities[key];
      if (count.value !== null && count.value > total.value) {
        result.abilities[key] = { value: null, status: "uncertain" };
        result.reasons.push("능력 수가 보유 합계보다 큽니다.");
      }
    }
  }
  if (!result.reasons.length)
    result.reasons.push(
      a.value !== null && b.value !== null
        ? "두 능력 수와 합계가 일치합니다."
        : "읽지 못한 능력 수를 직접 확인하세요.",
    );
  return result;
}
