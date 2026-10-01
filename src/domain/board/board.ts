export const BOARD_HEIGHT = 16;
export const BOARD_WIDTH = 10;
export const BOARD_CELL_COUNT = BOARD_HEIGHT * BOARD_WIDTH;

export type Board = boolean[][];

export type BoardError = {
  code: "INVALID_BOARD" | "INVALID_COORDINATE";
  message: string;
};

export type BoardResult =
  { ok: true; board: Board } | { ok: false; error: BoardError };

export function createEmptyBoard(): Board {
  return Array.from({ length: BOARD_HEIGHT }, () =>
    Array<boolean>(BOARD_WIDTH).fill(false),
  );
}

/** Validate untrusted input and return a detached board, never the input arrays. */
export function validateBoard(input: unknown): BoardResult {
  const invalid: BoardResult = {
    ok: false,
    error: {
      code: "INVALID_BOARD",
      message: `${BOARD_HEIGHT}행 × ${BOARD_WIDTH}열의 boolean 보드가 필요합니다.`,
    },
  };
  if (!Array.isArray(input) || input.length !== BOARD_HEIGHT) return invalid;

  const board: Board = [];
  for (let row = 0; row < BOARD_HEIGHT; row++) {
    const source: unknown = input[row];
    if (!Array.isArray(source) || source.length !== BOARD_WIDTH) return invalid;
    const cells: boolean[] = [];
    for (let col = 0; col < BOARD_WIDTH; col++) {
      const cell: unknown = source[col];
      if (typeof cell !== "boolean") return invalid;
      cells.push(cell);
    }
    board.push(cells);
  }
  return { ok: true, board };
}

export function toggleCell(
  input: unknown,
  row: number,
  col: number,
): BoardResult {
  const result = validateBoard(input);
  if (!result.ok) return result;
  if (
    !Number.isInteger(row) ||
    !Number.isInteger(col) ||
    row < 0 ||
    row >= BOARD_HEIGHT ||
    col < 0 ||
    col >= BOARD_WIDTH
  ) {
    return {
      ok: false,
      error: {
        code: "INVALID_COORDINATE",
        message: `row는 0~${BOARD_HEIGHT - 1}, col은 0~${BOARD_WIDTH - 1}의 정수여야 합니다.`,
      },
    };
  }
  result.board[row][col] = !result.board[row][col];
  return result;
}
