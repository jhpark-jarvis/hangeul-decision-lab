"use client";

import { Fragment, useState } from "react";
import {
  BOARD_CELL_COUNT,
  BOARD_HEIGHT,
  BOARD_WIDTH,
  createEmptyBoard,
  toggleCell,
  type Board,
} from "@/domain/board/board";

type EditorState = { board: Board; notice: string; error: string | null };

function initialState(): EditorState {
  return {
    board: createEmptyBoard(),
    notice: "빈 보드에서 시작합니다.",
    error: null,
  };
}

export function BoardEditor() {
  const [state, setState] = useState(initialState);
  const occupied = state.board.flat().filter(Boolean).length;

  function editCell(row: number, col: number) {
    setState((current) => {
      const result = toggleCell(current.board, row, col);
      if (!result.ok) return { ...current, error: result.error.message };
      return {
        board: result.board,
        notice: `row ${row}, col ${col}을 ${result.board[row][col] ? "점유" : "빈칸"}로 변경했습니다.`,
        error: null,
      };
    });
  }

  return (
    <div className="grid items-start gap-6 md:grid-cols-[minmax(0,1.3fr)_minmax(240px,0.9fr)]">
      <section
        aria-labelledby="board-heading"
        className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm sm:p-6"
      >
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 id="board-heading" className="text-base font-semibold">
            현재 보드
          </h2>
          <span className="text-xs tabular-nums text-stone-500">
            {BOARD_HEIGHT}행 × {BOARD_WIDTH}열
          </span>
        </div>
        <p
          id="coordinate-help"
          className="mb-4 text-xs leading-5 text-stone-500"
        >
          좌상단은 (0, 0)입니다. 위 숫자는 col, 왼쪽 숫자는 row입니다.
        </p>
        <div
          role="group"
          aria-label="16행 10열 보드"
          aria-describedby="coordinate-help keyboard-help"
          className="mx-auto grid max-w-[400px] gap-1 rounded-xl bg-stone-50 p-2 sm:gap-1.5 sm:p-3"
          style={{
            gridTemplateColumns: `1.25rem repeat(${BOARD_WIDTH}, minmax(0, 1fr))`,
          }}
        >
          <span aria-hidden="true" />
          {Array.from({ length: BOARD_WIDTH }, (_, col) => (
            <span
              key={`col-${col}`}
              aria-hidden="true"
              className="text-center font-mono text-[11px] leading-5 text-stone-500"
            >
              {col}
            </span>
          ))}
          {state.board.map((cells, row) => (
            <Fragment key={row}>
              <span
                aria-hidden="true"
                className="flex items-center justify-center font-mono text-[11px] text-stone-500"
              >
                {row}
              </span>
              {cells.map((filled, col) => (
                <button
                  key={col}
                  type="button"
                  aria-label={`row ${row}, col ${col}`}
                  aria-pressed={filled}
                  onClick={() => editCell(row, col)}
                  className={`aspect-square min-w-0 cursor-pointer rounded border transition-colors motion-reduce:transition-none ${
                    filled
                      ? "border-orange-700 bg-orange-600 hover:bg-orange-700"
                      : "border-stone-300 bg-white hover:border-stone-400 hover:bg-stone-100"
                  }`}
                >
                  <span
                    aria-hidden="true"
                    className="block text-xs font-bold text-white"
                  >
                    {filled ? "·" : ""}
                  </span>
                </button>
              ))}
            </Fragment>
          ))}
        </div>
        <div
          className="mt-5 flex items-center gap-5 text-xs text-stone-600"
          aria-label="보드 범례"
        >
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-sm border border-stone-300 bg-white"
              aria-hidden="true"
            />
            빈칸
          </span>
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-sm bg-orange-600"
              aria-hidden="true"
            />
            점유
          </span>
        </div>
      </section>

      <aside className="space-y-4">
        <section
          aria-labelledby="input-heading"
          className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm"
        >
          <h2 id="input-heading" className="text-sm font-semibold">
            입력 상태
          </h2>
          <p className="mt-5 text-xs text-stone-500">점유된 칸</p>
          <output
            aria-label="점유 칸 수"
            className="mt-1 block text-4xl font-semibold tabular-nums tracking-tight"
          >
            {occupied}
            <span className="ml-2 text-base font-normal text-stone-400">
              / {BOARD_CELL_COUNT}
            </span>
          </output>
          <p className="mt-2 text-sm text-stone-600">
            빈칸 {BOARD_CELL_COUNT - occupied}개
          </p>
          <p
            role="status"
            className="mt-5 min-h-10 border-t border-stone-100 pt-4 text-xs leading-5 text-stone-600"
          >
            {state.notice}
          </p>
          {state.error && (
            <p
              role="alert"
              className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-800"
            >
              {state.error}
            </p>
          )}
          <button
            type="button"
            disabled={occupied === 0}
            onClick={() =>
              setState({ ...initialState(), notice: "보드를 비웠습니다." })
            }
            className="mt-4 w-full cursor-pointer rounded-lg border border-stone-300 px-4 py-2.5 text-sm font-medium hover:bg-stone-50 disabled:cursor-default disabled:opacity-40"
          >
            보드 비우기
          </button>
        </section>
        <section
          className="rounded-2xl border border-stone-200 p-5 text-sm leading-6 text-stone-600"
          aria-labelledby="help-heading"
        >
          <h2 id="help-heading" className="mb-2 font-semibold text-stone-800">
            입력 방법
          </h2>
          <p>칸을 클릭하면 점유 상태가 바뀝니다.</p>
          <p id="keyboard-help" className="mt-2">
            키보드로는 Tab으로 이동한 뒤 Enter 또는 Space를 누르세요.
          </p>
        </section>
        <p className="px-1 text-xs leading-5 text-stone-500">
          입력은 이 탭에서만 유지됩니다. 새로고침하거나 탭을 닫으면 보드가
          초기화됩니다.
        </p>
      </aside>
    </div>
  );
}
