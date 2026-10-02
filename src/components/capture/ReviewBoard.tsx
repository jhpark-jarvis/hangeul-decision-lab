"use client";

import { useEffect, useId, useRef, useState } from "react";
import { BOARD_HEIGHT, BOARD_WIDTH } from "@/domain/board/board";
import type { RecognizedCell } from "@/features/recognition/types";

export function ReviewBoard({
  board,
  preview,
  invalid,
  onLabel,
}: {
  board: RecognizedCell[][];
  preview: HTMLCanvasElement | null;
  invalid: (path: string) => boolean;
  onLabel: (row: number, col: number, occupied: boolean | null) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const cells = useRef<(HTMLButtonElement | null)[]>([]);
  const editor = useRef<HTMLDivElement>(null);
  const id = useId();
  const [selected, setSelected] = useState<number | null>(null);
  const [imageVisible, setImageVisible] = useState(true);
  const flat = board.flat();
  const unresolved = flat.filter(
    (cell) => cell.status !== "recognized" || cell.occupied === null,
  );
  useEffect(() => {
    if (!preview || !host.current) return;
    host.current.replaceChildren(preview);
    return () => {
      preview.remove();
    };
  }, [preview]);
  useEffect(() => {
    if (selected !== null)
      editor.current
        ?.querySelector<HTMLButtonElement>("button")
        ?.focus({ preventScroll: true });
  }, [selected]);
  function select(index: number) {
    setSelected(index);
    cells.current[index]?.scrollIntoView({
      block: "center",
      behavior: "smooth",
    });
  }
  function next() {
    const indices = flat.flatMap((cell, i) =>
      cell.status !== "recognized" || cell.occupied === null ? [i] : [],
    );
    const index = indices.find((i) => i > (selected ?? -1)) ?? indices[0];
    if (index !== undefined) {
      select(index);
      cells.current[index]?.focus({ preventScroll: true });
    }
  }
  const target = selected === null ? null : flat[selected];
  const sourceVisible = !!preview && imageVisible;
  return (
    <div className="space-y-3" aria-label="보드 시각 검토">
      <div className="flex flex-wrap items-center gap-2">
        <p
          role="status"
          aria-label="미확정 칸 수"
          className="font-semibold text-amber-900"
        >
          {unresolved.length
            ? `확인할 미확정 칸 ${unresolved.length}개`
            : "미확정 칸 없음"}
        </p>
        <button
          type="button"
          className="small-button"
          disabled={!unresolved.length}
          onClick={next}
        >
          다음 미확정 칸
        </button>
        {preview && (
          <>
            <button
              type="button"
              className="small-button"
              aria-pressed={imageVisible}
              onClick={() => setImageVisible(true)}
            >
              게임 캡처 보기
            </button>
            <button
              type="button"
              className="small-button"
              aria-pressed={!imageVisible}
              onClick={() => setImageVisible(false)}
            >
              점유 표시 보기
            </button>
          </>
        )}
      </div>
      <p className="help">
        노란 테두리의 ?·! 칸을 눌러 바로 빈칸 또는 점유로 지정하세요. 다른 칸도
        눌러 수정할 수 있습니다.
      </p>
      <div
        className="review-board relative w-full max-w-lg"
        style={{
          aspectRatio: preview
            ? `${preview.width} / ${preview.height}`
            : `${BOARD_WIDTH} / ${BOARD_HEIGHT}`,
        }}
      >
        <div
          ref={host}
          className={`absolute inset-0 ${sourceVisible ? "" : "hidden"}`}
        />
        <div
          className="absolute inset-0 grid"
          style={{
            gridTemplateColumns: `repeat(${BOARD_WIDTH}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${BOARD_HEIGHT}, minmax(0, 1fr))`,
          }}
          aria-label="검토 보드"
        >
          {flat.map((cell, index) => {
            const unknown =
              cell.status !== "recognized" || cell.occupied === null;
            return (
              <button
                ref={(element) => {
                  cells.current[index] = element;
                }}
                key={`${cell.row},${cell.col}`}
                type="button"
                data-review-row={cell.row}
                data-review-col={cell.col}
                data-review-invalid={invalid(`board.${cell.row}.${cell.col}`)}
                data-review-unresolved={unknown}
                aria-label={`검토 row ${cell.row} col ${cell.col}: ${unknown ? "미확정" : cell.occupied ? "점유" : "빈칸"}`}
                aria-pressed={selected === index}
                aria-controls={selected === index ? id : undefined}
                className={`review-cell ${unknown ? "unresolved" : ""} ${sourceVisible ? "over-image" : cell.occupied ? "filled" : "empty"}`}
                onClick={() => select(index)}
              >
                <span>
                  {cell.status === "uncertain"
                    ? "!"
                    : unknown
                      ? "?"
                      : cell.occupied
                        ? "●"
                        : "·"}
                </span>
              </button>
            );
          })}
        </div>
        {target && (
          <div
            ref={editor}
            id={id}
            role="group"
            aria-label="선택한 칸 레이블링"
            className="cell-label-editor"
            style={{
              left: `clamp(124px, ${((target.col + 0.5) / BOARD_WIDTH) * 100}%, calc(100% - 124px))`,
              top: `${((target.row + (target.row < BOARD_HEIGHT - 4 ? 1 : 0)) / BOARD_HEIGHT) * 100}%`,
              transform: `translate(-50%, ${target.row < BOARD_HEIGHT - 4 ? "0" : "-100%"})`,
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.stopPropagation();
                setSelected(null);
                cells.current[selected!]?.focus({ preventScroll: true });
              }
            }}
          >
            <p className="text-xs font-semibold">
              선택한 칸 ·{" "}
              {target.status !== "recognized" || target.occupied === null
                ? "확인 필요"
                : target.occupied
                  ? "점유"
                  : "빈칸"}
            </p>
            <div className="flex gap-2 mt-2">
              <button
                type="button"
                className="small-button"
                aria-pressed={
                  target.status === "recognized" && target.occupied === false
                }
                onClick={() => onLabel(target.row, target.col, false)}
              >
                빈칸으로 표시
              </button>
              <button
                type="button"
                className="small-button"
                aria-pressed={
                  target.status === "recognized" && target.occupied === true
                }
                onClick={() => onLabel(target.row, target.col, true)}
              >
                점유로 표시
              </button>
            </div>
            <div className="flex gap-2 mt-2">
              <button
                type="button"
                className="small-button"
                onClick={() => onLabel(target.row, target.col, null)}
              >
                미확정으로 되돌리기
              </button>
              <button
                type="button"
                className="small-button"
                onClick={() => {
                  setSelected(null);
                  cells.current[selected!]?.focus({ preventScroll: true });
                }}
              >
                닫기
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
