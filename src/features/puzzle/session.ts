import { createEmptyBoard, toggleCell } from "../../domain/board/board";
import { applyAction } from "../../domain/game/actions";
import {
  enterNextPieces,
  resolveReroll,
  validateGameState,
} from "../../domain/game/game-state";
import type {
  GameState,
  HiddenItem,
  PieceIndex,
  TransitionInfo,
} from "../../domain/game/types";
import { getInitialCatalog } from "../../domain/pieces/catalog";
import type {
  SolverCandidate,
  SolverResult,
  TurnSolution,
} from "../../domain/solver/types";

export type BoardTool =
  "toggle" | "filled" | "empty" | "reroll" | "single-cell" | "remove-item";
export type ReplayPlan = {
  candidate: SolverCandidate;
  snapshots: GameState[];
  info: TransitionInfo[];
};
export type Analysis = {
  solution: TurnSolution;
  plans: ReplayPlan[];
  selected: number;
  step: number;
  version: number;
  elapsedMs: number;
};
export type PuzzleSession = {
  game: GameState;
  version: number;
  analysis: Analysis | null;
  notice: string;
  error: string | null;
};

export function createSession(
  input: unknown = {
    board: createEmptyBoard(),
    remainingPieces: [],
    hiddenItems: [],
    abilities: { reroll: 0, singleCell: 0 },
    pendingReroll: null,
  },
): PuzzleSession {
  const valid = validateGameState(input);
  if (!valid.ok) throw new Error(valid.error.message);
  return {
    game: valid.state,
    version: 0,
    analysis: null,
    notice: "보드와 현재 블록 3개를 입력하세요.",
    error: null,
  };
}

function failure(session: PuzzleSession, message: string): PuzzleSession {
  return {
    ...session,
    version: session.version + 1,
    analysis: null,
    error: message,
    notice: "추천을 폐기했습니다. 입력을 확인하고 다시 분석하세요.",
  };
}

export const rejectSession = failure;

/** Invalid edits keep the last valid game, but discard recommendations. */
export function replaceGame(
  session: PuzzleSession,
  input: unknown,
  notice: string,
): PuzzleSession {
  const valid = validateGameState(input);
  if (!valid.ok) return failure(session, valid.error.message);
  return {
    game: valid.state,
    version: session.version + 1,
    analysis: null,
    notice,
    error: null,
  };
}

export function editCell(
  session: PuzzleSession,
  row: number,
  col: number,
  tool: BoardTool,
): PuzzleSession {
  const toggled = toggleCell(session.game.board, row, col);
  if (!toggled.ok) return failure(session, toggled.error.message);
  const game = {
    ...session.game,
    board: toggled.board,
    hiddenItems: session.game.hiddenItems.map((item) => ({ ...item })),
  };
  if (tool === "filled" || tool === "empty")
    game.board[row][col] = tool === "filled";
  if (["reroll", "single-cell", "remove-item"].includes(tool)) {
    game.board[row][col] = session.game.board[row][col];
    game.hiddenItems = game.hiddenItems.filter(
      (item) => item.row !== row || item.col !== col,
    );
    if (tool !== "remove-item")
      game.hiddenItems.push({ row, col, type: tool as HiddenItem["type"] });
  }
  return replaceGame(
    session,
    game,
    `(${row}, ${col}) 입력 변경. 이전 추천을 지웠습니다.`,
  );
}

export function editPiece(
  session: PuzzleSession,
  slot: PieceIndex,
  pieceId: string,
): PuzzleSession {
  if (session.game.pendingReroll)
    return failure(session, "실제 reroll 결과 입력을 완료하세요.");
  if (![0, 1, 2].includes(slot))
    return failure(session, "블록 slot은 0~2입니다.");
  const piece = getInitialCatalog().find((entry) => entry.id === pieceId);
  if (pieceId && !piece)
    return failure(session, "선택한 블록 종류가 없습니다.");
  const remainingPieces = session.game.remainingPieces.filter(
    (entry) => entry.pieceIndex !== slot,
  );
  if (piece)
    remainingPieces.push({
      pieceIndex: slot,
      instanceId: `input-${session.version + 1}-${slot}`,
      piece,
    });
  remainingPieces.sort((a, b) => a.pieceIndex - b.pieceIndex);
  return replaceGame(
    session,
    { ...session.game, remainingPieces },
    `slot ${slot} 변경. 다시 분석하세요.`,
  );
}

export function loadNextPieces(
  session: PuzzleSession,
  ids: string[],
): PuzzleSession {
  const pieces = ids.map((id, slot) => ({
    piece: getInitialCatalog().find((piece) => piece.id === id),
    pieceIndex: slot,
    instanceId: `set-${session.version + 1}-${slot}`,
  }));
  const result = enterNextPieces(session.game, pieces);
  return result.ok
    ? replaceGame(
        session,
        result.state,
        "새 블록 3개를 입력했습니다. 분석하세요.",
      )
    : failure(session, result.error.message);
}

export function inputReroll(session: PuzzleSession, id: string): PuzzleSession {
  const result = resolveReroll(
    session.game,
    getInitialCatalog().find((piece) => piece.id === id),
  );
  return result.ok
    ? replaceGame(
        session,
        result.state,
        "실제 reroll 결과를 반영했습니다. 다시 분석하세요.",
      )
    : failure(session, result.error.message);
}

export function installAnalysis(
  session: PuzzleSession,
  expectedVersion: number,
  result: SolverResult<{ result: TurnSolution }>,
  elapsedMs: number,
): PuzzleSession {
  if (session.version !== expectedVersion)
    return failure(session, "분석 중 입력이 바뀌었습니다. 다시 분석하세요.");
  if (!result.ok) return failure(session, result.error.message);
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0)
    return failure(session, "분석 시간 측정이 올바르지 않습니다.");
  // Detach results and replay every candidate through the same domain transition.
  const solution: TurnSolution = structuredClone(result.result);
  const plans: ReplayPlan[] = [];
  for (const candidate of [solution, ...solution.alternatives]) {
    const snapshots = [session.game];
    const info: TransitionInfo[] = [];
    for (const action of candidate.actions) {
      const applied = applyAction(snapshots[snapshots.length - 1], action);
      if (!applied.ok)
        return failure(session, `추천 재생 오류: ${applied.error.message}`);
      snapshots.push(applied.state);
      info.push(applied.info);
    }
    if (
      JSON.stringify(snapshots[snapshots.length - 1]) !==
      JSON.stringify(candidate.finalState)
    )
      return failure(session, "추천 최종 상태가 공통 전이와 다릅니다.");
    plans.push({ candidate, snapshots: structuredClone(snapshots), info });
  }
  return {
    ...session,
    error: null,
    notice: "분석 완료. 첫 추천과 삭제 예상 행을 확인하세요.",
    analysis: {
      solution,
      plans,
      selected: 0,
      step: 0,
      version: session.version,
      elapsedMs,
    },
  };
}

export function selectPlan(
  session: PuzzleSession,
  selected: number,
): PuzzleSession {
  const analysis = session.analysis;
  if (
    !analysis ||
    analysis.step !== 0 ||
    !Number.isInteger(selected) ||
    !analysis.plans[selected]
  )
    return failure(
      session,
      "적용한 뒤에는 대안을 바꿀 수 없습니다. 다시 분석하세요.",
    );
  return { ...session, analysis: { ...analysis, selected }, error: null };
}

/** Compare both revision and step tokens, then revalidate the actual action. */
export function applyStep(
  session: PuzzleSession,
  expectedVersion: number,
  expectedStep: number,
): PuzzleSession {
  const analysis = session.analysis;
  const plan = analysis?.plans[analysis.selected];
  if (
    !analysis ||
    !plan ||
    session.version !== expectedVersion ||
    analysis.version !== expectedVersion ||
    analysis.step !== expectedStep ||
    JSON.stringify(session.game) !==
      JSON.stringify(plan.snapshots[analysis.step])
  )
    return failure(
      session,
      "이전 추천 또는 이미 적용한 단계입니다. 다시 분석하세요.",
    );
  const action = plan.candidate.actions[analysis.step];
  if (!action)
    return failure(
      session,
      "적용할 추천이 없습니다. 다시 분석하거나 다음 입력을 완료하세요.",
    );
  const result = applyAction(session.game, action);
  if (!result.ok) return failure(session, `적용 실패: ${result.error.message}`);
  if (
    JSON.stringify(result.state) !==
    JSON.stringify(plan.snapshots[analysis.step + 1])
  )
    return failure(
      session,
      "적용 결과가 예상 상태와 다릅니다. 다시 분석하세요.",
    );
  return {
    game: result.state,
    version: session.version + 1,
    analysis:
      action.type === "reroll"
        ? null
        : {
            ...analysis,
            step: analysis.step + 1,
            version: session.version + 1,
          },
    error: null,
    notice: `단계 ${analysis.step + 1} 적용. 삭제 행: ${result.info.clearedRows.join(", ") || "없음"}, 아이템 획득 ${result.info.acquiredItems.length}개·상한으로 남김 ${result.info.retainedItems.length}개.${action.type === "reroll" ? " 실제 reroll 결과를 입력하세요." : ""}`,
  };
}

export function nextPreview(session: PuzzleSession) {
  const analysis = session.analysis;
  const plan = analysis?.plans[analysis.selected];
  const action = plan?.candidate.actions[analysis?.step ?? 0];
  const cells: { row: number; col: number }[] = [];
  if (action?.type === "single-cell")
    cells.push({ row: action.row, col: action.col });
  if (action?.type === "place-piece")
    action.variant.forEach((row, r) =>
      row.forEach((filled, c) => {
        if (filled) cells.push({ row: action.row + r, col: action.col + c });
      }),
    );
  return {
    action,
    cells,
    clearedRows: plan?.info[analysis?.step ?? 0]?.clearedRows ?? [],
  };
}
