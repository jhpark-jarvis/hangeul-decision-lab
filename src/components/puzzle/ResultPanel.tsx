import type { SolverAction } from "@/domain/solver/types";
import type { Analysis } from "@/features/puzzle/session";
import { ShapeGrid } from "./ShapeGrid";

const REASONS = {
  "blocked-piece": "이 블록의 일반 배치가 막힘",
  "initial-ordinary-probe-no-complete-path":
    "일반 사전 탐색에서 소진 경로를 찾지 못함 (미완료일 수 있음)",
  "zero-catalog-mobility": "catalog 배치 가능 종류 0종",
};
export function describeAction(action: SolverAction) {
  if (action.type === "place-piece")
    return `slot ${action.pieceIndex} · ${action.pieceId} → (${action.row}, ${action.col}) · 회전 ${action.rotation}° · ${action.flipped ? "좌우 반전" : "반전 없음"}`;
  if (action.type === "single-cell")
    return `Single Cell → (${action.row}, ${action.col})`;
  return `Reroll → slot ${action.pieceIndex} · ${REASONS[action.reason]}`;
}

export function ResultPanel({
  analysis,
  disabled,
  onSelect,
  onApply,
}: {
  analysis: Analysis;
  disabled: boolean;
  onSelect: (index: number) => void;
  onApply: () => void;
}) {
  const plan = analysis.plans[analysis.selected];
  const { candidate } = plan;
  const evaluation = candidate.evaluation;
  const search = analysis.solution.search;
  const probe = search.abilitySearch?.ordinaryProbe;
  const next = candidate.actions[analysis.step];
  return (
    <section className="panel" aria-labelledby="result-heading">
      <h2 id="result-heading">추천 결과</h2>
      <p className="help mt-2">
        {search.searchComplete
          ? "설정 후보 범위 탐색 완료"
          : "노드 한도 도달 · 탐색 미완료 · 최적성 미보장"}
        . 전체 게임 최적성·미래 생존 보장은 아닙니다.
      </p>
      <p className="help mt-2">
        Reroll은 조건부 후보만 비교합니다. 미래 결과·성공 확률은 평가하지
        않습니다.
      </p>
      <div
        className="my-4 flex flex-wrap gap-2"
        role="group"
        aria-label="추천 대안"
      >
        {analysis.plans.map((entry, index) => (
          <button
            className="small-button"
            key={index}
            disabled={disabled || analysis.step > 0}
            aria-pressed={analysis.selected === index}
            onClick={() => onSelect(index)}
          >
            {index === 0 ? "최선" : `대안 ${index}`} ·{" "}
            {entry.candidate.actions.length}단계
          </button>
        ))}
      </div>
      <p className="text-sm">
        {next
          ? `다음: ${describeAction(next)}`
          : "추천 단계 완료. 다음 입력 또는 재분석이 필요합니다."}
      </p>
      <p className="help mt-2">
        다음 단계 삭제 예상 row:{" "}
        {plan.info[analysis.step]?.clearedRows.join(", ") || "없음"}
      </p>
      <button
        className="primary-button mt-4 w-full"
        disabled={disabled || !next}
        onClick={onApply}
      >
        Apply Step
      </button>
      <ol className="action-list mt-4" aria-label="추천 순서">
        {candidate.actions.map((action, index) => (
          <li
            key={index}
            className={
              index < analysis.step
                ? "applied"
                : index === analysis.step
                  ? "next-action"
                  : ""
            }
          >
            {index + 1}. {describeAction(action)}{" "}
            {action.type === "place-piece" && (
              <ShapeGrid shape={action.variant} />
            )}
            <span className="help block">
              {index < analysis.step ? "적용 완료 · " : ""}삭제 row{" "}
              {plan.info[index].clearedRows.join(", ") || "없음"} · 획득{" "}
              {plan.info[index].acquiredItems.length} · 상한으로 남김{" "}
              {plan.info[index].retainedItems.length}
            </span>
          </li>
        ))}
      </ol>
      <div
        className="mt-4 border-t border-stone-200 pt-4 text-sm"
        aria-label="경로 평가"
      >
        <p className="font-semibold">분석 당시 선택 경로의 최종 평가</p>
        <p>
          현재 블록 전부 소진:{" "}
          {evaluation.allCurrentPiecesPlaced ? "가능 경로 발견" : "미완료"}
        </p>
        <p>
          경로 끝 상태:{" "}
          {evaluation.phase === "gameover"
            ? "합법 행동 없음"
            : evaluation.phase === "await-reroll-result"
              ? "실제 reroll 결과 입력 대기"
              : evaluation.phase === "await-next-pieces"
                ? "다음 3개 입력 대기"
                : "진행 가능한 행동 있음"}
        </p>
        <p>
          Mobility {evaluation.mobility.playablePieceTypes}/
          {evaluation.mobility.totalPieceTypes}종 · 배치{" "}
          {evaluation.mobility.totalPlacements}개
        </p>
        <p>
          삭제 {evaluation.clearedRows}행 · 아이템 획득{" "}
          {evaluation.acquiredItems}개
        </p>
        <p>
          능력 사용 R {candidate.usedAbilities.reroll} / S{" "}
          {candidate.usedAbilities.singleCell} · 최종 보유 R{" "}
          {evaluation.remainingAbilities.reroll} / S{" "}
          {evaluation.remainingAbilities.singleCell}
        </p>
        <p className="help">
          제공 이미지의 19종 기준입니다. 출현 확률은 줄 삭제 누적 단계에 따라
          달라지며, 다음 조각·새 능력 아이콘은 예측하지 않습니다.
        </p>
      </div>
      <details className="mt-4">
        <summary>탐색 Debug</summary>
        <dl className="debug-grid mt-3">
          <dt>호출 시간</dt>
          <dd>
            <output aria-label="분석 호출 시간">
              {analysis.elapsedMs.toFixed(2)} ms
            </output>
          </dd>
          <dt>전체 방문 (probe 포함)</dt>
          <dd>
            {search.visitedNodes} / {search.maxNodes}
          </dd>
          <dt>최종 DFS memo 가지치기 / 보관</dt>
          <dd>
            {search.memoPrunedNodes} / {search.memoEntries}
          </dd>
          <dt>최종 DFS 발견 소진 경로</dt>
          <dd>{search.discoveredCompletePaths} (전체 경로 수 아님)</dd>
          <dt>능력 행동 실제 탐색</dt>
          <dd>{search.specialAbilitiesSearched ? "예" : "아니오"}</dd>
          <dt>동일 상태의 다른 경로 생략 가능</dt>
          <dd>{search.alternativesMayOmitEquivalentPaths ? "예" : "아니오"}</dd>
          <dt>reroll 제외 대상 관측 합계</dt>
          <dd>{search.abilitySearch?.excludedRerollTargets ?? 0}</dd>
          <dt>probe 방문 / 발견 경로</dt>
          <dd>
            {probe?.visitedNodes ?? 0} / {probe?.discoveredCompletePaths ?? 0}
          </dd>
          <dt>probe 완료 / 일반 소진 불가 증명</dt>
          <dd>
            {probe?.searchComplete ? "완료" : "미완료"} /{" "}
            {probe?.noCompletePathProven ? "예" : "아니오"}
          </dd>
        </dl>
      </details>
    </section>
  );
}
