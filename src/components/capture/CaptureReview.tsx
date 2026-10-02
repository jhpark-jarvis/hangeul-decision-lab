"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState } from "@/domain/game/types";
import { getInitialCatalog } from "@/domain/pieces/catalog";
import {
  captureCurrentFrame,
  frameCaptureFailureMessage,
  createCaptureController,
  requestDisplayMedia,
  type CaptureStatus,
  type CapturedFrame,
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
import { releaseFrame } from "@/features/recognition/calibration";
import { recognizeAutomaticBoard } from "@/features/recognition/automatic";
import { FrameCalibration } from "./FrameCalibration";

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
  const ownedFrame = useRef<CapturedFrame | null>(null);
  const frameSerial = useRef(0);
  const [calibration, setCalibration] = useState<{
    id: number;
    frame: CapturedFrame;
    snapshot: string;
  } | null>(null);
  const clearCalibration = useCallback(() => {
    releaseFrame(ownedFrame.current);
    ownedFrame.current = null;
    setCalibration(null);
  }, []);
  const gameSnapshot = JSON.stringify(game);
  const [observedGame, setObservedGame] = useState(gameSnapshot);
  // Reset frame selection in the same render as a changed game; the ownership
  // effect cleans its pixels after commit. No stale selector can reappear.
  if (observedGame !== gameSnapshot) {
    setObservedGame(gameSnapshot);
    if (calibration) {
      setCalibration(null);
      setFrameMessage(
        "수동 상태가 바뀌어 선택 중인 프레임을 지웠습니다. 다시 캡처하세요.",
      );
    }
  }
  useEffect(() => {
    const frame = calibration?.frame ?? null;
    return () => {
      releaseFrame(frame);
      if (ownedFrame.current === frame) ownedFrame.current = null;
    };
  }, [calibration]);
  useEffect(() => {
    const element = video.current;
    const owned = createCaptureController(requestDisplayMedia, (status) => {
      setCapture(status);
      if (status.phase !== "active") {
        clearCalibration();
        if (video.current) video.current.srcObject = null;
      }
    });
    controller.current = owned;
    return () => {
      owned.dispose();
      if (element) element.srcObject = null;
      controller.current = null;
      releaseFrame(ownedFrame.current);
      ownedFrame.current = null;
    };
  }, [clearCalibration]);

  function openReview(
    result: RecognitionResult,
    expectedSnapshot = JSON.stringify(game),
  ) {
    onInvalidate();
    setReview(createReviewState(result));
    setSnapshot(expectedSnapshot);
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
        화면을 공유한 뒤 Capture Frame을 누르면 보드 영역과 점유 칸을 자동으로
        찾습니다. 미확정 칸과 블록·아이템·능력은 직접 검토해야 합니다. 자동
        인식이 실패하면 수동 영역·색상 선택을 사용하세요. 아래 mock은 현재 수동
        입력의 복사입니다.
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
            clearCalibration();
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
            clearCalibration();
            setReview(null);
            setErrors([]);
            let frame: CapturedFrame | null = null;
            try {
              if (!video.current) return;
              frame = captureCurrentFrame(video.current);
              const automatic = recognizeAutomaticBoard(frame);
              if (automatic.ok) {
                openReview(automatic.result);
                const unresolved = automatic.result.board
                  .flat()
                  .filter((cell) => cell.occupied === null).length;
                setFrameMessage(
                  `자동 보드 판별 완료 · 미확정 ${unresolved}칸 · 프레임을 지웠습니다.`,
                );
              } else {
                openReview(
                  unknownRecognitionEngine.recognize({ kind: "frame", frame }),
                );
                setFrameMessage(
                  `프레임 ${frame.width}×${frame.height} · ${automatic.reason === "AMBIGUOUS" ? "보드 후보가 여러 개입니다" : "보드를 확정하지 못했습니다"}. 게임 보드가 크게 보이도록 공유한 뒤 다시 Capture Frame을 누르거나 수동 영역·색상 선택을 사용하세요. 프레임을 지웠습니다.`,
                );
              }
            } catch (error) {
              setFrameMessage(frameCaptureFailureMessage(error));
            } finally {
              releaseFrame(frame);
            }
          }}
        >
          Capture Frame
        </button>
        <button
          className="small-button"
          disabled={disabled || capture.phase !== "active"}
          onClick={() => {
            onInvalidate();
            clearCalibration();
            setReview(null);
            setErrors([]);
            try {
              if (!video.current) return;
              const frame = captureCurrentFrame(video.current);
              ownedFrame.current = frame;
              setCalibration({
                id: ++frameSerial.current,
                frame,
                snapshot: JSON.stringify(game),
              });
              setFrameMessage(
                `프레임 ${frame.width}×${frame.height} · 로컬 추출 완료`,
              );
              openReview(
                unknownRecognitionEngine.recognize({ kind: "frame", frame }),
              );
              // One owned frame survives only until calibration completes or is cleared.
            } catch (error) {
              setFrameMessage(frameCaptureFailureMessage(error));
            }
          }}
        >
          수동 영역·색상 선택
        </button>
        <button
          className="small-button"
          disabled={disabled || !!game.pendingReroll}
          onClick={() => {
            clearCalibration();
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
      {calibration && (
        <FrameCalibration
          key={calibration.id}
          frame={calibration.frame}
          disabled={disabled}
          onChange={() => {
            onInvalidate();
            setReview(null);
            setErrors([]);
          }}
          onResult={(result) => {
            const expected = calibration.snapshot;
            clearCalibration();
            if (expected !== JSON.stringify(game)) {
              setFrameMessage("수동 상태가 바뀌었습니다. 다시 캡처하세요.");
              return;
            }
            openReview(result, expected);
            const unresolved = result.board
              .flat()
              .filter((cell) => cell.status !== "recognized").length;
            setFrameMessage(
              `보드 판별 완료 · 미확정 ${unresolved}칸 · 프레임을 지웠습니다.`,
            );
          }}
          onCancel={() => {
            onInvalidate();
            clearCalibration();
            setReview(null);
            setErrors([]);
            setFrameMessage("프레임 선택을 취소하고 픽셀을 지웠습니다.");
          }}
        />
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
              ? "영역·색상 선택 전: 모든 값은 미확정입니다."
              : review.draft.source.kind === "calibrated-board"
                ? "지정 영역·색상 표본의 보드 판별 결과입니다. 블록·아이템·능력과 각 칸을 확인하세요."
                : review.draft.source.kind === "automatic-board"
                  ? "자동으로 찾은 보드 판별 결과입니다. 미확정 칸과 보드 전체, 블록·아이템·능력을 확인하세요."
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
                if (!failures.length) {
                  clearCalibration();
                  setReview(null);
                }
              }}
            >
              Use This State
            </button>
            <button
              className="small-button"
              onClick={() => {
                onInvalidate();
                clearCalibration();
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
