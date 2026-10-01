"use client";

import { Fragment } from "react";
import { BOARD_WIDTH, type Board } from "@/domain/board/board";
import type { HiddenItem } from "@/domain/game/types";
import type { BoardTool } from "@/features/puzzle/session";

const TOOLS: { id: BoardTool; label: string }[] = [
  { id: "toggle", label: "토글" },
  { id: "filled", label: "Filled" },
  { id: "empty", label: "Empty" },
  { id: "reroll", label: "Hidden: Reroll" },
  { id: "single-cell", label: "Hidden: Single Cell" },
  { id: "remove-item", label: "아이템 지우기" },
];

export function BoardEditor({
  board,
  items,
  tool,
  onTool,
  onCell,
  onClear,
  overlay,
  clearedRows,
  disabled,
}: {
  board: Board;
  items: HiddenItem[];
  tool: BoardTool;
  onTool: (tool: BoardTool) => void;
  onCell: (row: number, col: number) => void;
  onClear: () => void;
  overlay: { row: number; col: number }[];
  clearedRows: number[];
  disabled: boolean;
}) {
  const occupied = board.flat().filter(Boolean).length;
  return (
    <section className="panel" aria-labelledby="board-heading">
      <div className="flex items-center justify-between gap-4">
        <h2 id="board-heading">현재 보드</h2>
        <output aria-label="점유 칸 수">{occupied} / 160</output>
      </div>
      <p id="coordinate-help" className="help mt-2">
        16행 × 10열 · 좌상단 (0, 0) · row / col은 0부터 셉니다.
      </p>
      <div
        className="my-4 flex flex-wrap gap-2"
        role="group"
        aria-label="보드 입력 도구"
      >
        {TOOLS.map(({ id, label }) => (
          <button
            key={id}
            className="small-button"
            aria-pressed={tool === id}
            disabled={disabled}
            onClick={() => onTool(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        role="group"
        aria-label="16행 10열 보드"
        aria-describedby="coordinate-help keyboard-help"
        className="board-grid"
        style={{
          gridTemplateColumns: `1.25rem repeat(${BOARD_WIDTH}, minmax(0, 1fr))`,
        }}
      >
        <span />
        {Array.from({ length: BOARD_WIDTH }, (_, col) => (
          <span key={col} className="coordinate" aria-hidden="true">
            {col}
          </span>
        ))}
        {board.map((row, r) => (
          <Fragment key={r}>
            <span className="coordinate" aria-hidden="true">
              {r}
            </span>
            {row.map((filled, c) => {
              const item = items.find(
                (item) => item.row === r && item.col === c,
              );
              const suggested = overlay.some(
                (cell) => cell.row === r && cell.col === c,
              );
              const clearing = clearedRows.includes(r);
              return (
                <button
                  key={c}
                  type="button"
                  disabled={disabled}
                  aria-label={`row ${r}, col ${c}${item ? `, hidden ${item.type}` : ""}${suggested ? ", 추천" : ""}${clearing ? ", 삭제 예상" : ""}`}
                  aria-pressed={filled}
                  data-row={r}
                  data-col={c}
                  data-overlay={suggested}
                  data-clearing={clearing}
                  onClick={() => onCell(r, c)}
                  className={`board-cell ${filled ? "filled" : ""} ${suggested ? "suggested" : ""} ${clearing ? "clearing" : ""}`}
                >
                  <span aria-hidden="true">
                    {item
                      ? item.type === "reroll"
                        ? "R"
                        : "S"
                      : suggested
                        ? "+"
                        : filled
                          ? "·"
                          : ""}
                  </span>
                </button>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="help mt-4">
        주황: 점유 · 청록 +: 첫 추천 · 보라 테두리: 삭제 예상 행<br />
        R: hidden reroll · S: hidden single-cell (가로줄 삭제 때 획득) · 보드
        아이콘 최대 3개
      </p>
      <p id="keyboard-help" className="help mt-2">
        Tab으로 이동하고 Enter / Space로 입력하세요.
      </p>
      <button
        className="small-button mt-4"
        disabled={disabled || (occupied === 0 && items.length === 0)}
        onClick={onClear}
      >
        보드·아이템 비우기
      </button>
    </section>
  );
}
