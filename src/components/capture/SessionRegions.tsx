"use client";
import { useEffect, useRef, useState } from "react";
import type { CapturedFrame } from "@/features/capture/capture";
import {
  pointerToPixel,
  previewFrame,
  regionFromCorners,
  releaseFrame,
  type PixelPoint,
} from "@/features/recognition/calibration";
import {
  regionLabels,
  regionsFit,
  pieceCardRegions,
  countRegions,
  type CaptureRegions,
  type RegionKey,
} from "@/features/recognition/regions";

export function SessionRegions({
  frame,
  initial,
  disabled,
  onUse,
  onCancel,
}: {
  frame: CapturedFrame;
  initial: CaptureRegions | null;
  disabled: boolean;
  onUse: (regions: CaptureRegions) => void;
  onCancel: () => void;
}) {
  const [regions, setRegions] = useState<CaptureRegions>(
    () =>
      initial ?? {
        dimensions: { width: frame.width, height: frame.height },
        board: { x: 0, y: 0, width: 0, height: 0 },
        pieces: { x: 0, y: 0, width: 0, height: 0 },
        abilities: { x: 0, y: 0, width: 0, height: 0 },
      },
  );
  const [target, setTarget] = useState<RegionKey>("board"),
    [corner, setCorner] = useState<PixelPoint | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const copy = previewFrame(frame);
    element.width = copy.width;
    element.height = copy.height;
    const ctx = element.getContext("2d");
    try {
      if (!ctx) {
        element.width = 0;
        element.height = 0;
        return;
      }
      const image = ctx.createImageData(copy.width, copy.height);
      image.data.set(copy.pixels);
      try {
        ctx.putImageData(image, 0, 0);
      } finally {
        image.data.fill(0);
      }
      const sx = copy.width / frame.width,
        sy = copy.height / frame.height;
      for (const key of Object.keys(regionLabels) as RegionKey[]) {
        const r = regions[key];
        ctx.strokeStyle = key === target ? "#ffdf32" : "#ffffff";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x * sx, r.y * sy, r.width * sx, r.height * sy);
        ctx.fillStyle = "#161616";
        ctx.fillRect(r.x * sx, r.y * sy, 76, 18);
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px sans-serif";
        ctx.fillText(regionLabels[key], r.x * sx + 3, r.y * sy + 13);
        if (key === "board" && r.width > 0) {
          ctx.beginPath();
          for (let c = 1; c < 10; c++) {
            ctx.moveTo((r.x + (r.width * c) / 10) * sx, r.y * sy);
            ctx.lineTo((r.x + (r.width * c) / 10) * sx, (r.y + r.height) * sy);
          }
          for (let row = 1; row < 16; row++) {
            ctx.moveTo(r.x * sx, (r.y + (r.height * row) / 16) * sy);
            ctx.lineTo(
              (r.x + r.width) * sx,
              (r.y + (r.height * row) / 16) * sy,
            );
          }
          ctx.stroke();
        } else if (r.width > 0) {
          const sub =
            key === "pieces"
              ? pieceCardRegions(r)
              : Object.values(countRegions(r));
          for (const guide of sub)
            ctx.strokeRect(
              guide.x * sx,
              guide.y * sy,
              guide.width * sx,
              guide.height * sy,
            );
        }
      }
    } finally {
      releaseFrame(copy);
    }
    return () => {
      element.width = 0;
      element.height = 0;
    };
  }, [frame, regions, target]);
  return (
    <fieldset
      disabled={disabled}
      className="space-y-3"
      aria-label="공유 세션 영역 설정"
    >
      <legend className="font-semibold">처음 한 번 영역 설정</legend>
      <p className="help">
        영역을 고른 뒤 좌상단·우하단 모서리를 차례로 누르세요. 보드는 칸 안쪽
        전체, 보유 조각은 세 카드 전체, 능력은 ‘보유 능력’ 제목부터 두 버튼
        끝까지 잡으세요. 내부 가이드가 맞는지 확인하세요. 색상 선택은 필요
        없습니다.
      </p>
      <div className="flex gap-2">
        {(Object.keys(regionLabels) as RegionKey[]).map((key) => (
          <button
            type="button"
            className="small-button"
            key={key}
            aria-pressed={key === target}
            onClick={() => {
              setTarget(key);
              setCorner(null);
            }}
          >
            {regionLabels[key]} 영역 지정
          </button>
        ))}
      </div>
      <p role="status">
        {regionLabels[target]} ·{" "}
        {corner ? "반대 모서리를 누르세요" : "첫 모서리를 누르세요"}
      </p>
      <canvas
        ref={canvas}
        aria-label="세 영역 선택 프레임"
        className="max-h-[640px] max-w-full border"
        onPointerDown={(event) => {
          const point = pointerToPixel(
            frame,
            event.currentTarget.getBoundingClientRect(),
            event,
          );
          if (!point) return;
          if (!corner) {
            setCorner(point);
            return;
          }
          setRegions({
            ...regions,
            [target]: regionFromCorners(corner, point),
          });
          setCorner(null);
        }}
      />
      <div className="grid grid-cols-2 gap-2">
        {(["x", "y", "width", "height"] as const).map((key) => (
          <label key={key}>
            {regionLabels[target]} {key}
            <input
              type="number"
              aria-label={`공유 영역 ${target} ${key}`}
              value={regions[target][key]}
              min={0}
              onChange={(e) =>
                setRegions({
                  ...regions,
                  [target]: {
                    ...regions[target],
                    [key]: Number(e.target.value),
                  },
                })
              }
            />
          </label>
        ))}
      </div>
      <p className="help">
        이 공유에서 위치·크기가 유지되면 다음 턴에도 재사용합니다. 창 이동·배율
        변경은 다시 지정하고, 공유 종료 시 설정은 지워집니다.
      </p>
      <div className="flex gap-2">
        <button
          className="primary-button"
          type="button"
          disabled={!regionsFit(frame, regions) || !!corner}
          onClick={() => onUse(regions)}
        >
          이 영역으로 인식
        </button>
        <button className="small-button" type="button" onClick={onCancel}>
          영역 설정 취소
        </button>
      </div>
    </fieldset>
  );
}
