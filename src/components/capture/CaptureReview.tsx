"use client";

import { useEffect, useRef, useState } from "react";
import type { GameState } from "@/domain/game/types";
import { getInitialCatalog } from "@/domain/pieces/catalog";
import {
  captureCurrentFrame,
  createCaptureController,
  requestDisplayMedia,
  type CaptureStatus,
} from "@/features/capture/capture";
import {
  recognizeManualState,
  unknownRecognitionEngine,
} from "@/features/recognition/mock";
import {
  createReviewState,
  updateReview,
  validateReviewedState,
} from "@/features/recognition/review";
import type {
  FieldIssue,
  RecognitionResult,
  ReviewState,
} from "@/features/recognition/types";

const catalog = getInitialCatalog();
export function CaptureReview({
  game,
  disabled,
  onInvalidate,
  onUse,
}: {
  game: GameState;
  disabled: boolean;
  onInvalidate: () => void;
  onUse: (review: ReviewState, snapshot: string) => FieldIssue[];
}) {
  const video = useRef<HTMLVideoElement>(null);
  const controller = useRef<ReturnType<typeof createCaptureController> | null>(
    null,
  );
  const [capture, setCapture] = useState<CaptureStatus>({
    phase: "idle",
    message: "공유는 직접 시작하며 프레임을 저장하거나 전송하지 않습니다.",
  });
  const [review, setReview] = useState<ReviewState | null>(null);
  const [snapshot, setSnapshot] = useState("");
  const [errors, setErrors] = useState<FieldIssue[]>([]);
  const [frameMessage, setFrameMessage] = useState("");
  useEffect(() => {
    const element = video.current;
    const owned = createCaptureController(requestDisplayMedia, (status) => {
      setCapture(status);
      if (status.phase !== "active" && video.current)
        video.current.srcObject = null;
    });
    controller.current = owned;
    return () => {
      owned.dispose();
      if (element) element.srcObject = null;
      controller.current = null;
    };
  }, []);

  function openReview(result: RecognitionResult) {
    onInvalidate();
    setReview(createReviewState(result));
    setSnapshot(JSON.stringify(game));
    setErrors([]);
  }
  function change(edit: (draft: RecognitionResult) => void) {
    if (!review) return;
    onInvalidate();
    setReview(updateReview(review, edit));
    setErrors([]);
  }
  const issues = review ? validateReviewedState(review, false) : [];
  const stale = !!review && snapshot !== JSON.stringify(game);
  const fieldIssues = [...issues, ...errors];
  const invalid = (path: string) =>
    fieldIssues.some((issue) => issue.path === path);

  return (
    <section className="panel space-y-4" aria-label="화면 캡처 및 인식 검토">
      <h2>화면 캡처·인식 검토</h2>
      <p className="help">
        화면 공유는 미리보기와 프레임 추출만 제공합니다. 이미지 자동 인식은 아직
        구현하지 않았습니다. 아래 mock은 현재 수동 입력의 복사입니다.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          className="small-button"
          disabled={
            disabled ||
            capture.phase === "requesting" ||
            capture.phase === "active"
          }
          onClick={() => {
            onInvalidate();
            setReview(null);
            setErrors([]);
            setFrameMessage("");
            void controller.current?.start().then(async (stream) => {
              if (!stream || !video.current) return;
              const element = video.current;
              element.srcObject = stream;
              try {
                await element.play();
              } catch {
                // A late play rejection after stop/unmount is not a new capture error.
                if (element.srcObject === stream) {
                  controller.current?.stop();
                  setFrameMessage(
                    "미리보기를 재생하지 못했습니다. 공유를 다시 시작하세요.",
                  );
                }
              }
            });
          }}
        >
          Start Screen Capture
        </button>
        <button
          className="small-button"
          disabled={
            capture.phase !== "active" && capture.phase !== "requesting"
          }
          onClick={() => controller.current?.stop()}
        >
          Stop Capture
        </button>
        <button
          className="small-button"
          disabled={disabled || capture.phase !== "active"}
          onClick={() => {
            onInvalidate();
            setReview(null);
            setErrors([]);
            try {
              if (!video.current) return;
              const frame = captureCurrentFrame(video.current);
              setFrameMessage(
                `프레임 ${frame.width}×${frame.height} · 로컬 추출 완료`,
              );
              openReview(
                unknownRecognitionEngine.recognize({ kind: "frame", frame }),
              );
              // Payload is not retained in React state, logs or recognition result.
            } catch (error) {
              setFrameMessage(
                error instanceof Error && error.message.startsWith("프레임")
                  ? error.message
                  : "프레임을 추출하지 못했습니다. 공유 화면과 재생 상태를 확인하세요.",
              );
            }
          }}
        >
          Capture Frame
        </button>
        <button
          className="small-button"
          disabled={disabled || !!game.pendingReroll}
          onClick={() => {
            setFrameMessage("");
            openReview(recognizeManualState(game));
          }}
        >
          현재 수동 입력으로 Mock 검토
        </button>
      </div>
      <p role="status" aria-label="캡처 상태">
        {capture.message}
      </p>
      <video
        ref={video}
        muted
        autoPlay
        playsInline
        aria-label="공유 화면 미리보기"
        className={`max-h-64 w-full rounded-lg bg-black ${capture.phase === "active" ? "" : "hidden"}`}
      />
      {frameMessage && (
        <p role="status" aria-label="프레임 상태">
          {frameMessage}
        </p>
      )}
      {review && (
        <fieldset
          disabled={disabled}
          className="space-y-4"
          aria-label="인식 결과 검토"
        >
          <legend className="font-semibold">인식 결과 검토·수정</legend>
          <p className="help">
            {review.draft.source.kind === "capture-stub"
              ? "프레임 인식기 미구현: 모든 값은 미확정입니다."
              : "현재 수동 입력의 개발용 복사입니다. 이미지 인식 결과가 아닙니다."}{" "}
            미확정(?)·불확실(!) 값은 직접 확인하세요. confidence는 생성하지
            않습니다.
          </p>
          {stale && (
            <p role="alert">
              검토 중 수동 상태가 바뀌었습니다. 현재 입력으로 검토를 다시
              시작하세요.
            </p>
          )}
          <button
            className="small-button"
            onClick={() =>
              change((draft) =>
                draft.board.forEach((row) =>
                  row.forEach((cell) => {
                    if (
                      cell.status !== "recognized" ||
                      cell.occupied === null
                    ) {
                      cell.occupied = false;
                      cell.status = "recognized";
                      delete cell.confidence;
                    }
                  }),
                ),
              )
            }
          >
            미확정 보드 칸을 빈칸으로 확인
          </button>
          <div
            className="grid grid-cols-10 gap-1 max-w-lg"
            aria-label="검토 보드"
          >
            {review.draft.board.flatMap((row, r) =>
              row.map((cell, c) => (
                <button
                  key={`${r},${c}`}
                  type="button"
                  data-review-row={r}
                  data-review-col={c}
                  data-review-invalid={invalid(`board.${r}.${c}`)}
                  aria-label={`검토 row ${r} col ${c}: ${cell.status !== "recognized" ? "미확정" : cell.occupied ? "점유" : "빈칸"}`}
                  className={`h-7 rounded border text-xs ${cell.occupied ? "bg-orange-500" : "bg-white"}`}
                  onClick={() =>
                    change((draft) => {
                      const target = draft.board[r][c];
                      target.occupied =
                        target.occupied === null
                          ? false
                          : target.occupied
                            ? null
                            : true;
                      target.status =
                        target.occupied === null ? "unknown" : "recognized";
                      delete target.confidence;
                    })
                  }
                >
                  {cell.status === "uncertain"
                    ? "!"
                    : cell.status === "unknown" || cell.occupied === null
                      ? "?"
                      : cell.occupied
                        ? "●"
                        : "·"}
                </button>
              )),
            )}
          </div>
          <div className="grid grid-cols-3 gap-3">
            {review.draft.pieces.map((piece) => (
              <label key={piece.slot}>
                검토 slot {piece.slot}
                <select
                  aria-label={`검토 slot ${piece.slot} 블록`}
                  aria-invalid={invalid(`pieces.${piece.slot}`)}
                  value={
                    piece.status !== "recognized"
                      ? "unknown"
                      : piece.empty
                        ? "empty"
                        : (piece.pieceId ?? "unknown")
                  }
                  onChange={(event) =>
                    change((draft) => {
                      const value = event.target.value;
                      draft.pieces[piece.slot] = {
                        slot: piece.slot,
                        pieceId:
                          value === "unknown" || value === "empty"
                            ? null
                            : value,
                        empty: value === "unknown" ? null : value === "empty",
                        status: value === "unknown" ? "unknown" : "recognized",
                      };
                    })
                  }
                >
                  <option value="unknown">미확정 / 불확실</option>
                  <option value="empty">이미 사용 / 없음</option>
                  {catalog.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.name}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-3">
            {(["reroll", "singleCell"] as const).map((key) => (
              <label key={key}>
                검토 {key}{" "}
                {review.draft.abilities[key].status !== "recognized" &&
                  "(미확정 / 불확실)"}
                <input
                  aria-label={`검토 ${key} 보유 수`}
                  aria-invalid={
                    invalid(`abilities.${key}`) || invalid("abilities")
                  }
                  type="number"
                  min={0}
                  max={7}
                  step={1}
                  value={review.draft.abilities[key].value ?? ""}
                  onChange={(event) =>
                    change((draft) => {
                      draft.abilities[key] = {
                        value:
                          event.target.value === ""
                            ? null
                            : Number(event.target.value),
                        status:
                          event.target.value === "" ? "unknown" : "recognized",
                      };
                    })
                  }
                />
              </label>
            ))}
          </div>
          <div className="space-y-2" aria-label="검토 아이템">
            <p>
              아이템 {review.draft.hiddenItems.length}/3 ·{" "}
              {review.draft.hiddenItemsStatus === "recognized"
                ? "목록 확인됨"
                : "목록 미확정 / 불확실"}
            </p>
            {review.draft.hiddenItems.map((item, index) => (
              <div className="flex flex-wrap gap-2" key={index}>
                <select
                  aria-label={`검토 아이템 ${index} 종류`}
                  value={item.type ?? "unknown"}
                  onChange={(event) =>
                    change((draft) => {
                      const value = event.target.value;
                      draft.hiddenItems[index].type =
                        value === "unknown"
                          ? null
                          : (value as "reroll" | "single-cell");
                      draft.hiddenItems[index].status = "recognized";
                      delete draft.hiddenItems[index].confidence;
                    })
                  }
                >
                  <option value="unknown">미확정</option>
                  <option value="reroll">Reroll</option>
                  <option value="single-cell">Single Cell</option>
                </select>
                {(["row", "col"] as const).map((key) => (
                  <label key={key}>
                    {key}
                    <input
                      className="max-w-20"
                      aria-label={`검토 아이템 ${index} ${key}`}
                      aria-invalid={
                        invalid(`hiddenItems.${index}`) ||
                        invalid(`hiddenItems.${index}.${key}`)
                      }
                      type="number"
                      value={item[key] ?? ""}
                      onChange={(event) =>
                        change((draft) => {
                          draft.hiddenItems[index][key] =
                            event.target.value === ""
                              ? null
                              : Number(event.target.value);
                          draft.hiddenItems[index].status = "recognized";
                          delete draft.hiddenItems[index].confidence;
                        })
                      }
                    />
                  </label>
                ))}
                <button
                  className="small-button"
                  onClick={() =>
                    change((draft) => {
                      draft.hiddenItems.splice(index, 1);
                    })
                  }
                >
                  검토 아이템 {index} 제거
                </button>
              </div>
            ))}
            <button
              className="small-button"
              disabled={review.draft.hiddenItems.length >= 3}
              onClick={() =>
                change((draft) => {
                  draft.hiddenItems.push({
                    row: null,
                    col: null,
                    type: null,
                    status: "unknown",
                  });
                })
              }
            >
              검토 아이템 추가
            </button>
            <label className="block">
              <input
                type="checkbox"
                style={{
                  width: "auto",
                  display: "inline-block",
                  marginRight: 8,
                }}
                checked={review.draft.hiddenItemsStatus === "recognized"}
                onChange={(event) =>
                  change((draft) => {
                    draft.hiddenItemsStatus = event.target.checked
                      ? "recognized"
                      : "unknown";
                  })
                }
              />{" "}
              아이템 목록 전체 확인 (없음 포함)
            </label>
          </div>
          {!!fieldIssues.length && (
            <div role="alert" aria-label="검토 오류">
              <p>확인할 필드 {fieldIssues.length}개</p>
              <ul>
                {fieldIssues.slice(0, 12).map((issue, index) => (
                  <li key={index}>
                    {issue.path}: {issue.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <label className="block">
            <input
              type="checkbox"
              style={{ width: "auto", display: "inline-block", marginRight: 8 }}
              aria-label="검토한 전체 상태 확인"
              checked={review.confirmed}
              onChange={(event) => {
                onInvalidate();
                setReview({ ...review, confirmed: event.target.checked });
                setErrors([]);
              }}
            />{" "}
            보드·블록·아이템·능력을 확인했습니다.
          </label>
          <div className="flex gap-2">
            <button
              className="primary-button"
              disabled={
                !!issues.length ||
                !review.confirmed ||
                stale ||
                !!game.pendingReroll
              }
              onClick={() => {
                const failures = onUse(review, snapshot);
                setErrors(failures);
                if (!failures.length) setReview(null);
              }}
            >
              Use This State
            </button>
            <button
              className="small-button"
              onClick={() => {
                onInvalidate();
                setReview(null);
                setErrors([]);
                setFrameMessage("");
              }}
            >
              검토 버리기
            </button>
          </div>
        </fieldset>
      )}
    </section>
  );
}
