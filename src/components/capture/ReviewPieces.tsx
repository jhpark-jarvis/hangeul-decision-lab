import { getInitialCatalog } from "@/domain/pieces/catalog";
import type { RecognizedPieceSlot } from "@/features/recognition/types";
import { ShapeGrid } from "../puzzle/ShapeGrid";
import { pieceLabels, piecePositions } from "./review-labels";

const catalog = getInitialCatalog();
export function ReviewPieces({
  pieces,
  invalid,
  onSelect,
}: {
  pieces: RecognizedPieceSlot[];
  invalid: (path: string) => boolean;
  onSelect: (slot: 0 | 1 | 2, value: string) => void;
}) {
  return (
    <section className="space-y-3" aria-label="보유 조각 확인">
      <h3 className="font-semibold">보유 조각 · 게임 오른쪽의 세 칸</h3>
      <p className="help">
        게임의 ‘보유 조각’을 위에서 아래 순서로 맞추세요. ‘사용 완료’라고 보이는
        칸은 ‘사용 완료 / 조각 없음’을 선택하세요. 이름이 어려우면 ‘모양으로
        고르기’를 펼치세요. 회전·반전된 모양도 같은 조각입니다.
      </p>
      <div className="grid grid-cols-3 gap-3">
        {pieces.map((piece) => {
          const value =
            piece.status !== "recognized"
              ? "unknown"
              : piece.empty
                ? "empty"
                : (piece.pieceId ?? "unknown");
          const chosen = catalog.find((entry) => entry.id === value);
          return (
            <div
              className="rounded-lg border border-stone-200 p-3 space-y-2"
              key={piece.slot}
            >
              <label>
                <span className="block font-semibold text-sm">
                  {pieceLabels[piece.slot]}
                </span>
                <span className="help">
                  게임 보유 조각의 {piecePositions[piece.slot]} 칸
                </span>
                <select
                  aria-label={pieceLabels[piece.slot]}
                  aria-invalid={invalid(`pieces.${piece.slot}`)}
                  value={value}
                  onChange={(event) => onSelect(piece.slot, event.target.value)}
                >
                  <option value="unknown">아직 선택하지 않음</option>
                  <option value="empty">사용 완료 / 조각 없음</option>
                  {catalog.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.name}
                    </option>
                  ))}
                </select>
              </label>
              <div
                className="min-h-16 flex items-center justify-center"
                aria-label={`${pieceLabels[piece.slot]} 선택 모양`}
              >
                {chosen ? (
                  <ShapeGrid shape={chosen.shape} />
                ) : (
                  <span className="help">
                    {value === "empty" ? "사용 완료" : "조각을 선택하세요"}
                  </span>
                )}
              </div>
              <details>
                <summary>{pieceLabels[piece.slot]} 모양으로 고르기</summary>
                <div className="grid grid-cols-2 gap-1 mt-2">
                  {catalog.map((entry) => (
                    <button
                      type="button"
                      className="small-button flex flex-col items-center justify-center gap-2 min-h-20"
                      aria-label={`${pieceLabels[piece.slot]} ${entry.name} 선택`}
                      aria-pressed={value === entry.id}
                      key={entry.id}
                      onClick={() => onSelect(piece.slot, entry.id)}
                    >
                      <ShapeGrid shape={entry.shape} />
                      <span>{entry.name}</span>
                    </button>
                  ))}
                </div>
              </details>
            </div>
          );
        })}
      </div>
    </section>
  );
}
