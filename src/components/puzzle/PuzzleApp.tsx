"use client";

import { useEffect, useRef, useState } from "react";
import { createEmptyBoard } from "@/domain/board/board";
import { getAvailableActions } from "@/domain/game/actions";
import { MAX_ABILITY_COUNT, type PieceIndex } from "@/domain/game/types";
import { getInitialCatalog } from "@/domain/pieces/catalog";
import { solveTurn } from "@/domain/solver/solver";
import {
  applyStep,
  createSession,
  editCell,
  editPiece,
  inputReroll,
  installAnalysis,
  loadNextPieces,
  nextPreview,
  rejectSession,
  replaceGame,
  selectPlan,
  type BoardTool,
} from "@/features/puzzle/session";
import { BoardEditor } from "../board/BoardEditor";
import { ShapeGrid } from "./ShapeGrid";
import { ResultPanel } from "./ResultPanel";

const CATALOG = getInitialCatalog();
const SLOTS: PieceIndex[] = [0, 1, 2];
const ANALYSIS_PAINT_DELAY_MS = 32;
export function PuzzleApp() {
  const [session, setSession] = useState(createSession);
  const [tool, setTool] = useState<BoardTool>("toggle");
  const [nextIds, setNextIds] = useState(["", "", ""]);
  const [rerollId, setRerollId] = useState("");
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  const running = useRef(false);
  const timer = useRef<number | null>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, []);
  const available = getAvailableActions(session.game);
  const phase = available.ok ? available.phase : "gameover";
  const preview = nextPreview(session);
  const awaitingNext = phase === "await-next-pieces";
  const pending = session.game.pendingReroll;

  function analyze() {
    if (running.current) return;
    running.current = true;
    const snapshot = structuredClone(session.game);
    const version = session.version;
    setSession((current) => ({ ...current, analysis: null, error: null }));
    setBusy(true);
    // Paint the busy state before bounded synchronous work. No worker or network.
    timer.current = window.setTimeout(() => {
      timer.current = null;
      if (!alive.current) return;
      try {
        const started = performance.now();
        const result = solveTurn(snapshot, CATALOG);
        const elapsed = performance.now() - started;
        setSession((current) =>
          installAnalysis(current, version, result, elapsed),
        );
      } catch {
        setSession((current) =>
          rejectSession(
            current,
            "분석에 실패했습니다. 입력을 보존했습니다. 다시 시도하세요.",
          ),
        );
      } finally {
        running.current = false;
        if (alive.current) setBusy(false);
      }
    }, ANALYSIS_PAINT_DELAY_MS);
  }

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(420px,1.1fr)_minmax(420px,1fr)]">
      <div className="space-y-4 lg:sticky lg:top-4">
        <BoardEditor
          board={session.game.board}
          items={session.game.hiddenItems}
          tool={tool}
          onTool={setTool}
          disabled={busy}
          overlay={preview.cells}
          clearedRows={preview.clearedRows}
          onCell={(row, col) =>
            setSession((current) => editCell(current, row, col, tool))
          }
          onClear={() =>
            setSession((current) =>
              replaceGame(
                current,
                { ...current.game, board: createEmptyBoard(), hiddenItems: [] },
                "보드와 아이템을 비웠습니다. 다시 분석하세요.",
              ),
            )
          }
        />
        <section className="panel" aria-label="입력 및 적용 상태">
          <p role="status" className="text-sm leading-6">
            {busy ? "분석 중… 입력을 잠시 기다려 주세요." : session.notice}
          </p>
          {session.error && (
            <p
              role="alert"
              className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-800"
            >
              {session.error}
            </p>
          )}
          <p className="help mt-3">
            입력은 이 탭 메모리에만 유지됩니다. 새로고침하면 초기화됩니다.
          </p>
        </section>
      </div>
      <aside className="space-y-4">
        <section className="panel" aria-labelledby="pieces-heading">
          <h2 id="pieces-heading">
            {awaitingNext ? "블록 3개 입력" : "현재 블록 (고정 slot)"}
          </h2>
          <p className="help mt-2">
            {awaitingNext
              ? "현재/다음 세트의 실제 블록 3개를 선택하세요. 보드와 능력은 유지됩니다."
              : "이미 사용한 slot은 없음으로 표시됩니다. 입력을 보정하면 추천이 지워집니다."}
          </p>
          <div className="mt-4 grid grid-cols-3 gap-3">
            {SLOTS.map((slot) => {
              const current = session.game.remainingPieces.find(
                (entry) => entry.pieceIndex === slot,
              )?.piece;
              const id = awaitingNext ? nextIds[slot] : (current?.id ?? "");
              const piece = CATALOG.find((piece) => piece.id === id);
              return (
                <label
                  key={slot}
                  className={`piece-slot ${preview.action?.type === "reroll" && preview.action.pieceIndex === slot ? "reroll-target" : ""}`}
                >
                  <span>slot {slot}</span>
                  <span className="my-3 flex h-10 items-center justify-center">
                    {piece ? (
                      <ShapeGrid shape={piece.shape} />
                    ) : (
                      <span className="help">없음</span>
                    )}
                  </span>
                  <select
                    aria-label={`slot ${slot} 블록`}
                    disabled={busy || !!pending}
                    value={id}
                    onChange={(event) => {
                      const value = event.target.value;
                      if (awaitingNext) {
                        setNextIds((ids) =>
                          ids.map((id, index) => (index === slot ? value : id)),
                        );
                        setSession((current) => ({
                          ...current,
                          analysis: null,
                        }));
                      } else
                        setSession((current) =>
                          editPiece(current, slot, value),
                        );
                    }}
                  >
                    <option value="">
                      {awaitingNext ? "선택하세요" : "이미 사용 / 없음"}
                    </option>
                    {CATALOG.map((piece) => (
                      <option value={piece.id} key={piece.id}>
                        {piece.name}
                      </option>
                    ))}
                  </select>
                </label>
              );
            })}
          </div>
          {awaitingNext && (
            <button
              className="primary-button mt-4 w-full"
              disabled={busy || nextIds.some((id) => !id)}
              onClick={() => {
                setSession((current) => loadNextPieces(current, nextIds));
                setNextIds(["", "", ""]);
              }}
            >
              3개 블록 입력
            </button>
          )}
          <div
            className="mt-4 flex flex-wrap gap-4"
            aria-label="이벤트 블록 목록"
          >
            {CATALOG.map((piece) => (
              <span
                key={piece.id}
                data-piece-id={piece.id}
                className="flex items-center gap-2 text-xs"
              >
                <ShapeGrid shape={piece.shape} />
                {piece.name}
              </span>
            ))}
          </div>
          <p className="help mt-3">
            2026 한글 모아모아 · 제공 이미지의 블록 19종입니다. 이름은 모양을
            구분하기 위한 표기입니다. 실제 블록·아이콘이 바뀌면 현재 입력을
            보정하세요.
          </p>
        </section>
        <section className="panel" aria-labelledby="abilities-heading">
          <h2 id="abilities-heading">능력 보유 수</h2>
          <div className="mt-4 grid grid-cols-2 gap-4">
            {(["reroll", "singleCell"] as const).map((ability) => (
              <label key={ability} className="text-sm">
                {ability === "reroll" ? "Reroll" : "Single Cell"}
                <input
                  aria-label={`${ability === "reroll" ? "Reroll" : "Single Cell"} 보유 수`}
                  type="number"
                  min={0}
                  max={MAX_ABILITY_COUNT}
                  step={1}
                  disabled={busy || !!pending}
                  value={session.game.abilities[ability]}
                  onChange={(event) => {
                    const raw = event.target.value;
                    setSession((current) =>
                      raw === ""
                        ? rejectSession(
                            current,
                            "능력 수는 0~7 정수이며 합계는 7 이하여야 합니다.",
                          )
                        : replaceGame(
                            current,
                            {
                              ...current.game,
                              abilities: {
                                ...current.game.abilities,
                                [ability]: Number(raw),
                              },
                            },
                            "능력 수를 바꿨습니다. 다시 분석하세요.",
                          ),
                    );
                  }}
                />
              </label>
            ))}
          </div>
          <p className="help mt-2">
            합계{" "}
            {session.game.abilities.reroll + session.game.abilities.singleCell}{" "}
            / 7 · 소모와 획득은 Apply에서 자동 반영됩니다.
          </p>
        </section>
        {pending && (
          <section className="panel" aria-labelledby="reroll-heading">
            <h2 id="reroll-heading">
              slot {pending.pieceIndex} 실제 reroll 결과 입력
            </h2>
            <p className="help mt-2">
              능력은 이미 1개 소모했습니다. 게임에 나온 다른 종류를 선택하고
              반영하세요.
            </p>
            <select
              aria-label="실제 reroll 결과"
              value={rerollId}
              disabled={busy}
              onChange={(event) => setRerollId(event.target.value)}
            >
              <option value="">선택하세요</option>
              {CATALOG.map((piece) => (
                <option key={piece.id} value={piece.id}>
                  {piece.name}
                </option>
              ))}
            </select>
            <button
              className="primary-button mt-3"
              disabled={busy || !rerollId}
              onClick={() => {
                setSession((current) => inputReroll(current, rerollId));
                setRerollId("");
              }}
            >
              reroll 결과 반영
            </button>
          </section>
        )}
        <section className="panel">
          <h2>분석</h2>
          <p className="help mt-2">
            {phase === "gameover"
              ? "현재 합법 행동이 없습니다. 실제 입력을 확인하세요."
              : pending
                ? "실제 reroll 결과 입력 후 재분석하세요."
                : awaitingNext
                  ? "블록 3개를 입력한 뒤 분석하세요."
                  : "보드·블록·능력을 확인하고 추천을 계산하세요."}
          </p>
          <button
            className="primary-button mt-4 w-full"
            disabled={busy || awaitingNext || !!pending}
            onClick={analyze}
          >
            Analyze
          </button>
          <p className="help mt-2">
            분석 중에는 입력이 잠시 멈춥니다. 기본 512노드 제한이며 모든 경로의
            최적성을 보장하지 않습니다.
          </p>
        </section>
        {session.analysis && (
          <ResultPanel
            analysis={session.analysis}
            disabled={busy || !!session.error}
            onSelect={(index) =>
              setSession((current) => selectPlan(current, index))
            }
            onApply={() => {
              const version = session.version;
              const step = session.analysis?.step ?? -1;
              setSession((current) => applyStep(current, version, step));
            }}
          />
        )}
      </aside>
    </div>
  );
}
