"use client";

import { useEffect, useId, useRef, useState } from "react";
import { BOARD_HEIGHT, BOARD_WIDTH } from "@/domain/board/board";
import type {
  RecognizedCell,
  RecognizedHiddenItem,
} from "@/features/recognition/types";
import { itemLabels } from "./review-labels";

export function ReviewBoard({
  board,
  preview,
  invalid,
  onLabel,
  items,
  onItem,
  differences = [],
  inherited = [],
}: {
  board: RecognizedCell[][];
  preview: HTMLCanvasElement | null;
  invalid: (path: string) => boolean;
  onLabel: (row: number, col: number, occupied: boolean | null) => void;
  items: RecognizedHiddenItem[];
  onItem: (
    row: number,
    col: number,
    type: RecognizedHiddenItem["type"],
  ) => string | null;
  differences?: { row: number; col: number }[];
  inherited?: { row: number; col: number }[];
}) {
  const host = useRef<HTMLDivElement>(null);
  const cells = useRef<(HTMLButtonElement | null)[]>([]);
  const editor = useRef<HTMLDivElement>(null);
  const id = useId();
  const [selected, setSelected] = useState<number | null>(null);
  const [imageVisible, setImageVisible] = useState(true);
  const [itemError, setItemError] = useState<string | null>(null);
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
    setItemError(null);
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
  const targetItem = target
    ? items.find((item) => item.row === target.row && item.col === target.col)
    : undefined;
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
        노란 ?·! 칸을 눌러 빈칸 또는 점유로 지정하세요. 아이콘이 보이는 칸은
        같은 메뉴의 ‘이 칸의 아이템’에서 종류도 선택하세요. 아이템을 지정해도
        블록이 있는지는 따로 확인해야 합니다. 점=점 찍기, 뽑=바꿔 뽑기
        아이템입니다.
      </p>
      {!!(differences.length || inherited.length) && (
        <p className="help">
          빨간 테두리 Δ는 직전 예상과 다른 칸, 파란 ~는 미판독 상태를 이전 Apply
          결과로 채운 칸입니다. 게임과 비교해 달라진 곳만 수정하세요.
        </p>
      )}
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
            const different = differences.some(
              (c) => c.row === cell.row && c.col === cell.col,
            );
            const carried = inherited.some(
              (c) => c.row === cell.row && c.col === cell.col,
            );
            const unknown =
              cell.status !== "recognized" || cell.occupied === null;
            const item = items.find(
              (entry) => entry.row === cell.row && entry.col === cell.col,
            );
            const itemName = item
              ? item.type
                ? itemLabels[item.type]
                : "종류 미확정"
              : null;
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
                data-review-difference={different}
                data-review-inherited={carried}
                style={
                  different
                    ? { boxShadow: "inset 0 0 0 3px #e11d48" }
                    : carried
                      ? { boxShadow: "inset 0 0 0 2px #2563eb" }
                      : undefined
                }
                aria-label={`검토 row ${cell.row} col ${cell.col}: ${unknown ? "미확정" : cell.occupied ? "점유" : "빈칸"}${itemName ? ` · ${itemName} 아이템` : ""}${different ? " · 캡처 당시 예상과 다름" : carried ? " · 이전 Apply에서 이어받음" : ""}`}
                aria-pressed={selected === index}
                aria-controls={selected === index ? id : undefined}
                className={`review-cell ${unknown ? "unresolved" : ""} ${sourceVisible ? "over-image" : cell.occupied ? "filled" : "empty"}`}
                onClick={() => select(index)}
              >
                <span>
                  {different
                    ? "Δ"
                    : carried
                      ? "~"
                      : cell.status === "uncertain"
                        ? "!"
                        : unknown
                          ? "?"
                          : cell.occupied
                            ? "●"
                            : "·"}
                </span>
                {item && (
                  <small
                    className="review-item-badge"
                    data-review-item={item.type ?? "unknown"}
                    title={`${itemName} 아이템`}
                    aria-hidden="true"
                  >
                    {item.type === "single-cell"
                      ? "점"
                      : item.type === "reroll"
                        ? "뽑"
                        : "?"}
                  </small>
                )}
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
              top: `${((target.row + (target.row < BOARD_HEIGHT / 2 ? 1 : 0)) / BOARD_HEIGHT) * 100}%`,
              transform: `translate(-50%, ${target.row < BOARD_HEIGHT / 2 ? "0" : "-100%"})`,
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
            <p className="help mt-2">
              칸 상태 · 블록이 있으면 점유, 없으면 빈칸
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
            <div className="mt-3 border-t border-stone-200 pt-2 space-y-2">
              <p className="text-xs font-semibold">
                이 칸의 아이템 ·{" "}
                {targetItem
                  ? targetItem.type
                    ? itemLabels[targetItem.type]
                    : "종류 확인 필요"
                  : "표시한 아이템 없음"}
              </p>
              <div className="flex flex-wrap gap-1">
                {(["single-cell", "reroll"] as const).map((type) => (
                  <button
                    type="button"
                    className="small-button"
                    key={type}
                    aria-pressed={targetItem?.type === type}
                    onClick={() =>
                      setItemError(onItem(target.row, target.col, type))
                    }
                  >
                    {itemLabels[type]} 아이템
                  </button>
                ))}
                <button
                  type="button"
                  className="small-button"
                  onClick={() =>
                    setItemError(onItem(target.row, target.col, null))
                  }
                >
                  이 칸에 아이템 없음
                </button>
              </div>
              <p className="help">
                아이템은 칸 상태와 별개입니다. 종류 선택 후 블록 점유도
                확인하세요.
              </p>
              {itemError && (
                <p role="alert" className="text-xs text-red-700">
                  {itemError}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
