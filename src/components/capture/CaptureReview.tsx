"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState } from "@/domain/game/types";
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
import type { PixelRegion } from "@/features/recognition/calibration";
import { ReviewBoard } from "./ReviewBoard";
import { paintReviewPreview, releaseReviewPreview } from "./review-preview";
import { releaseFrame } from "@/features/recognition/calibration";
import { recognizeAutomaticGame } from "@/features/recognition/automatic-game";
import {
  createPanelTextTemplates,
  releasePanelTextTemplates,
} from "@/features/recognition/panel-fonts";
import { FrameCalibration } from "./FrameCalibration";

import { labelReviewItem } from "@/features/recognition/item-label";
import { ReviewPieces } from "./ReviewPieces";
import {
  abilityLabels,
  itemLabels,
  itemConfirmation,
  reviewFieldLabel,
} from "./review-labels";
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
  const ownedPreview = useRef<HTMLCanvasElement | null>(null);
  const [preview, setPreview] = useState<HTMLCanvasElement | null>(null);
  const clearPreview = useCallback(() => {
    releaseReviewPreview(ownedPreview.current);
    ownedPreview.current = null;
    setPreview(null);
  }, []);
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
    if (preview) setPreview(null);
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
    const image = preview;
    return () => {
      releaseReviewPreview(image);
      if (ownedPreview.current === image) ownedPreview.current = null;
    };
  }, [preview]);
  useEffect(() => {
    const element = video.current;
    const owned = createCaptureController(requestDisplayMedia, (status) => {
      setCapture(status);
      if (status.phase !== "active") {
        clearCalibration();
        clearPreview();
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
      releaseReviewPreview(ownedPreview.current);
      ownedPreview.current = null;
    };
  }, [clearCalibration, clearPreview]);

  function openReview(
    result: RecognitionResult,
    expectedSnapshot = JSON.stringify(game),
    image: HTMLCanvasElement | null = null,
  ) {
    onInvalidate();
    clearPreview();
    ownedPreview.current = image;
    setPreview(image);
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
        화면을 공유한 뒤 Capture Frame을 누르면 보드·보유 조각·능력 횟수를
        자동으로 채웁니다. 미확정 항목만 수정하고, 보드 위 아이템과 전체 상태를
        확인하세요. 자동 인식이 실패하면 수동 영역·색상 선택을 사용하세요. 아래
        mock은 현재 수동 입력의 복사입니다.
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
            clearPreview();
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
            clearPreview();
            setReview(null);
            setErrors([]);
            let frame: CapturedFrame | null = null;
            try {
              if (!video.current) return;
              frame = captureCurrentFrame(video.current);
              const templates = createPanelTextTemplates();
              let automatic;
              try {
                automatic = recognizeAutomaticGame(frame, templates);
              } finally {
                releasePanelTextTemplates(templates);
              }
              if (automatic.ok) {
                openReview(
                  automatic.result,
                  JSON.stringify(game),
                  paintReviewPreview(frame, automatic.region),
                );
                const unresolved = automatic.result.board
                  .flat()
                  .filter((cell) => cell.occupied === null).length;
                const pieces = automatic.result.pieces.filter(
                  (p) => p.status === "recognized",
                ).length;
                const counts = Object.values(automatic.result.abilities).filter(
                  (a) => a.status === "recognized",
                ).length;
                setFrameMessage(
                  `자동 보드 판별 완료 · 미확정 ${unresolved}칸 · 보유 조각 ${pieces}/3 · 능력 횟수 ${counts}/2 판독. 미확정 항목은 아래에서 수정하세요. 원본 프레임을 지웠습니다. 보드 캡처는 검토 종료 시 지웁니다.`,
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
            clearPreview();
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
            clearPreview();
            setReview(null);
            setErrors([]);
          }}
          onResult={(result, region: PixelRegion) => {
            const expected = calibration.snapshot;
            const image =
              expected === JSON.stringify(game)
                ? paintReviewPreview(calibration.frame, region)
                : null;
            clearCalibration();
            if (expected !== JSON.stringify(game)) {
              setFrameMessage("수동 상태가 바뀌었습니다. 다시 캡처하세요.");
              return;
            }
            openReview(result, expected, image);
            const unresolved = result.board
              .flat()
              .filter((cell) => cell.status !== "recognized").length;
            setFrameMessage(
              `보드 판별 완료 · 미확정 ${unresolved}칸 · 원본 프레임을 지웠습니다. 보드 캡처는 검토 종료 시 지웁니다.`,
            );
          }}
          onCancel={() => {
            onInvalidate();
            clearCalibration();
            clearPreview();
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
                : review.draft.source.kind === "automatic-game"
                  ? "자동으로 찾은 보드와 같은 화면의 보유 조각·능력 판독 결과입니다. 채워진 값은 게임과 비교하고 미확정 항목만 수정하세요. 아이템은 보드에서 직접 표시하세요."
                  : review.draft.source.kind === "automatic-board"
                    ? "자동으로 찾은 보드 판별 결과입니다. 미확정 칸과 보드 전체, 블록·아이템·능력을 확인하세요."
                    : "현재 수동 입력의 개발용 복사입니다. 이미지 인식 결과가 아닙니다."}{" "}
            아래 순서대로 보드·아이템, 보유 조각, 보유 능력을 확인하세요.
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
          <h3 className="font-semibold">보드 칸과 보드 위 아이템 확인</h3>
          <ReviewBoard
            key={`${review.draft.source.timestamp}:${snapshot}`}
            board={review.draft.board}
            items={review.draft.hiddenItems}
            onItem={(row, col, type) => {
              const result = labelReviewItem(review, row, col, type);
              if (!result.ok) return result.message;
              onInvalidate();
              setReview(result.review);
              setErrors([]);
              return null;
            }}
            preview={preview}
            invalid={invalid}
            onLabel={(row, col, occupied) =>
              change((draft) => {
                const target = draft.board[row][col];
                target.occupied = occupied;
                target.status = occupied === null ? "unknown" : "recognized";
                delete target.confidence;
              })
            }
          />
          <section
            className="rounded-xl border border-blue-200 bg-blue-50 p-4 space-y-3"
            aria-label="보드 아이템 확인"
          >
            <h3 className="font-semibold">
              보드 위 아이템 · 아이콘이 있는 칸에 표시
            </h3>
            <p className="text-sm">
              보드에서 아이콘이 보이는 칸을 클릭하고 ‘점 찍기 아이템’ 또는 ‘바꿔
              뽑기 아이템’을 선택하세요. 좌표를 직접 적을 필요는 없습니다.
            </p>
            <p
              className="text-sm"
              role="status"
              aria-label="표시한 보드 아이템"
            >
              표시한 아이템 {review.draft.hiddenItems.length}개 (보드에 최대
              3개)
            </p>
            {!review.draft.hiddenItems.length && (
              <p className="help">
                아직 표시한 아이템이 없습니다. 게임 보드에도 아이콘이 없다면
                추가하지 말고 아래 확인만 체크하세요.
              </p>
            )}
            {review.draft.hiddenItems.length > 0 && (
              <ul className="text-sm space-y-1">
                {review.draft.hiddenItems.map((item, index) => (
                  <li key={index}>
                    {item.type ? itemLabels[item.type] : "종류 선택 필요"}{" "}
                    아이템 ·{" "}
                    {item.row !== null && item.col !== null
                      ? `위에서 ${item.row + 1}번째 줄, 왼쪽에서 ${item.col + 1}번째 칸`
                      : "위치 선택 필요"}
                  </li>
                ))}
              </ul>
            )}
            <p className="help">
              게임 보드 전체에서 빠진 아이콘이 없는지 확인한 뒤 체크하세요.
              아이템 하나를 표시하는 것만으로 전체 확인이 끝나지는 않습니다.
            </p>
            <label className="block">
              <input
                type="checkbox"
                aria-label={itemConfirmation}
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
              {itemConfirmation}
            </label>
            <details>
              <summary>좌표로 직접 입력 (선택 사항)</summary>
              <p className="help mt-2">
                보드 클릭 대신 좌표를 입력할 때만 사용하세요. 왼쪽 위는
                행0·열0입니다. 목록을 수정하면 전체 아이템 확인을 다시 해야
                합니다.
              </p>
              <div className="space-y-2 mt-2">
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
                          draft.hiddenItemsStatus = "unknown";
                          delete draft.hiddenItems[index].confidence;
                        })
                      }
                    >
                      <option value="unknown">미확정</option>
                      <option value="reroll">바꿔 뽑기</option>
                      <option value="single-cell">점 찍기</option>
                    </select>
                    {(["row", "col"] as const).map((key) => (
                      <label key={key}>
                        {key === "row" ? "행 (0~15)" : "열 (0~9)"}
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
                              draft.hiddenItemsStatus = "unknown";
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
                          draft.hiddenItemsStatus = "unknown";
                        })
                      }
                    >
                      아이템 {index + 1} 제거
                    </button>
                  </div>
                ))}
                <button
                  className="small-button"
                  disabled={review.draft.hiddenItems.length >= 3}
                  onClick={() =>
                    change((draft) => {
                      draft.hiddenItemsStatus = "unknown";
                      draft.hiddenItems.push({
                        row: null,
                        col: null,
                        type: null,
                        status: "unknown",
                      });
                    })
                  }
                >
                  좌표로 아이템 추가
                </button>
              </div>
            </details>
          </section>
          <ReviewPieces
            pieces={review.draft.pieces}
            invalid={invalid}
            onSelect={(slot, value) =>
              change((draft) => {
                draft.pieces[slot] = {
                  slot,
                  pieceId:
                    value === "unknown" || value === "empty" ? null : value,
                  empty: value === "unknown" ? null : value === "empty",
                  status: value === "unknown" ? "unknown" : "recognized",
                };
              })
            }
          />
          <h3 className="font-semibold">보유 능력 · 게임 버튼 옆 남은 횟수</h3>
          <p className="help">
            채워진 횟수가 게임 오른쪽 아래 ‘점 찍기’와 ‘바꿔 뽑기’ 버튼 옆
            숫자와 맞는지 확인하세요. 비어 있거나 다르면 그 숫자로 수정하세요.
            보드 위 아이템 개수와는 다릅니다. 남은 능력이 없으면 0을 입력하세요.
            비워두면 아직 확인하지 않은 상태입니다.
          </p>
          <div className="grid grid-cols-2 gap-3">
            {(["singleCell", "reroll"] as const).map((key) => (
              <label key={key}>
                {abilityLabels[key]} 남은 횟수{" "}
                {review.draft.abilities[key].status !== "recognized" &&
                  "(숫자를 입력하세요)"}
                <input
                  aria-label={`${abilityLabels[key]} 남은 횟수`}
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
          {!!fieldIssues.length && (
            <div role="alert" aria-label="검토 오류">
              <p className="font-semibold">아직 확인할 항목이 있습니다.</p>
              {fieldIssues.some((issue) => issue.path.startsWith("board.")) && (
                <p>보드의 노란 ?·! 칸을 눌러 칸 상태를 확인하세요.</p>
              )}
              <ul>
                {fieldIssues
                  .filter((issue) => !issue.path.startsWith("board."))
                  .slice(0, 12)
                  .map((issue, index) => (
                    <li key={index}>
                      {reviewFieldLabel(issue.path)}: {issue.message}
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
                  clearPreview();
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
                clearPreview();
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
