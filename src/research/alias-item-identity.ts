import type { HiddenItem } from "../domain/game/types";

/** Item identity ignores normalization order; the caller separately checks input bytes. */
export function aliasItemIdentity(items: HiddenItem[]): string {
  return JSON.stringify(
    items
      .map(({ row, col, type }) => [row, col, type] as const)
      .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2].localeCompare(b[2])),
  );
}
