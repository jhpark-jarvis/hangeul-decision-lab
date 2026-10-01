import { BOARD_HEIGHT, BOARD_WIDTH } from "../board/board";
import type { GameState } from "../game/types";
import { getValidPlacements } from "../pieces/placement";
import { calculateMobility } from "./mobility";
import type {
  OrdinarySolverAction,
  RerollReason,
  SolverAction,
  SolverResult,
} from "./types";

/** Internal helpers take an already validated GameState. */
export function getOrdinaryCandidates(
  state: GameState,
): SolverResult<{ actions: OrdinarySolverAction[] }> {
  if (state.pendingReroll || !state.remainingPieces.length)
    return { ok: true, actions: [] };
  const actions: OrdinarySolverAction[] = [];
  for (const instance of [...state.remainingPieces].sort(
    (a, b) => a.pieceIndex - b.pieceIndex,
  )) {
    const placements = getValidPlacements(state.board, instance.piece);
    if (!placements.ok) return placements;
    actions.push(
      ...placements.placements.map((placement) => ({
        type: "place-piece" as const,
        ...placement,
        instanceId: instance.instanceId,
        pieceIndex: instance.pieceIndex,
      })),
    );
  }
  return { ok: true, actions };
}

export function getAbilityCandidates(
  state: GameState,
  catalog: unknown,
  initialProbeFoundNoCompletePath: boolean,
): SolverResult<{ actions: SolverAction[]; excludedRerollTargets: number }> {
  if (state.pendingReroll || !state.remainingPieces.length)
    return { ok: true, actions: [], excludedRerollTargets: 0 };
  const ordinary = getOrdinaryCandidates(state);
  if (!ordinary.ok) return ordinary;
  const singles: Extract<SolverAction, { type: "single-cell" }>[] = [];
  const vacancies = state.board.map(
    (row) => row.filter((cell) => !cell).length,
  );
  if (state.abilities.singleCell > 0)
    for (let row = 0; row < BOARD_HEIGHT; row++)
      for (let col = 0; col < BOARD_WIDTH; col++)
        if (!state.board[row][col])
          singles.push({ type: "single-cell", row, col });
  singles.sort(
    (a, b) =>
      vacancies[a.row] - vacancies[b.row] || a.row - b.row || a.col - b.col,
  );
  const rerolls: Extract<SolverAction, { type: "reroll" }>[] = [];
  let excludedRerollTargets = 0;
  let zeroMobility = false;
  if (state.abilities.reroll > 0) {
    const mobility = calculateMobility(state.board, catalog);
    if (!mobility.ok) return mobility;
    zeroMobility =
      mobility.mobility.totalPieceTypes > 0 &&
      mobility.mobility.playablePieceTypes === 0;
    for (const instance of [...state.remainingPieces].sort(
      (a, b) => a.pieceIndex - b.pieceIndex,
    )) {
      const blocked = !ordinary.actions.some(
        (action) => action.instanceId === instance.instanceId,
      );
      const reason: RerollReason | null = blocked
        ? "blocked-piece"
        : initialProbeFoundNoCompletePath
          ? "initial-ordinary-probe-no-complete-path"
          : zeroMobility
            ? "zero-catalog-mobility"
            : null;
      if (reason)
        rerolls.push({
          type: "reroll",
          instanceId: instance.instanceId,
          pieceIndex: instance.pieceIndex,
          reason,
        });
      else excludedRerollTargets++;
    }
  }
  return {
    ok: true,
    actions: [
      ...singles.filter((action) => vacancies[action.row] === 1),
      ...ordinary.actions,
      ...rerolls,
      ...singles.filter((action) => vacancies[action.row] !== 1),
    ],
    excludedRerollTargets,
  };
}
