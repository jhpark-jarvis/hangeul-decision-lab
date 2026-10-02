export const pieceLabels = [
  "첫 번째 보유 조각",
  "두 번째 보유 조각",
  "세 번째 보유 조각",
] as const;
export const piecePositions = ["맨 위", "가운데", "맨 아래"] as const;
export const abilityLabels = {
  singleCell: "점 찍기",
  reroll: "바꿔 뽑기",
} as const;
export const itemLabels = {
  "single-cell": "점 찍기",
  reroll: "바꿔 뽑기",
} as const;
export const itemConfirmation =
  "보드의 아이템을 모두 확인했습니다 (없으면 그대로 체크)";

export function reviewFieldLabel(path: string): string {
  const [field, index, coordinate] = path.split(".");
  if (field === "board" && index !== undefined && coordinate !== undefined)
    return `보드 위에서 ${Number(index) + 1}번째 줄 · 왼쪽에서 ${Number(coordinate) + 1}번째 칸`;
  if (field === "board") return "보드 칸 상태";
  if (field === "pieces") return pieceLabels[Number(index)] ?? "보유 조각";
  if (field === "abilities")
    return (
      abilityLabels[index as keyof typeof abilityLabels] ?? "보유 능력 합계"
    );
  if (field === "hiddenItemsStatus") return "보드 아이템 전체 확인";
  if (field === "hiddenItems")
    return index === undefined
      ? "보드 아이템"
      : `아이템 ${Number(index) + 1}${coordinate === "row" ? " 행" : coordinate === "col" ? " 열" : ""}`;
  if (field === "confirmation") return "전체 상태 확인";
  return "입력 상태";
}
