"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CapturedFrame } from "@/features/capture/capture";
import {
  createCalibratedBoardEngine,
  type RGB,
} from "@/features/recognition/board";
import {
  CALIBRATION_DEFAULTS,
  pointerToPixel,
  previewFrame,
  regionFromCorners,
  releaseFrame,
  sampleFrameColor,
  type PixelPoint,
  type PixelRegion,
} from "@/features/recognition/calibration";
import type { RecognitionResult } from "@/features/recognition/types";
import { BOARD_HEIGHT, BOARD_WIDTH } from "@/domain/board/board";

type Mode = "start" | "end" | "empty" | "occupied";
const modes: { value: Mode; label: string }[] = [
  { value: "start", label: "영역 첫 모서리 선택" },
  { value: "end", label: "영역 반대 모서리 선택" },
  { value: "empty", label: "빈칸 색상 선택" },
  { value: "occupied", label: "점유 색상 선택" },
];
const regionKeys = ["x", "y", "width", "height"] as const;
const settingLabels = {
  maxDistance: "색상 허용 거리",
  minDistanceGap: "반대 색상과 최소 거리 차이",
  minMatchedFraction: "최소 일치 표본 비율",
  sampleFraction: "칸 중앙 표본 영역 비율",
  samplesPerAxis: "표본 한 축의 점 수",
};
const numeric = (value: string): number | null =>
  value.trim() && Number.isFinite(Number(value)) ? Number(value) : null;

function paint(
  canvas: HTMLCanvasElement,
  frame: CapturedFrame,
  region: PixelRegion | null,
) {
  const preview = previewFrame(frame);
  canvas.width = preview.width;
  canvas.height = preview.height;
  const ctx = canvas.getContext("2d");
  try {
    if (!ctx) return;
    const image = ctx.createImageData(preview.width, preview.height);
    image.data.set(preview.pixels);
    try {
      ctx.putImageData(image, 0, 0);
    } finally {
      image.data.fill(0);
    }
    if (!region) return;
    const sx = preview.width / frame.width;
    const sy = preview.height / frame.height;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let col = 0; col <= BOARD_WIDTH; col++) {
      const x = (region.x + (col / BOARD_WIDTH) * region.width) * sx;
      ctx.moveTo(x, region.y * sy);
      ctx.lineTo(x, (region.y + region.height) * sy);
    }
    for (let row = 0; row <= BOARD_HEIGHT; row++) {
      const y = (region.y + (row / BOARD_HEIGHT) * region.height) * sy;
      ctx.moveTo(region.x * sx, y);
      ctx.lineTo((region.x + region.width) * sx, y);
    }
    ctx.stroke();
  } finally {
    releaseFrame(preview);
  }
}

export function FrameCalibration({
  frame,
  disabled,
  onChange,
  onResult,
  onCancel,
}: {
  frame: CapturedFrame;
  disabled: boolean;
  onChange: () => void;
  onResult: (result: RecognitionResult) => void;
  onCancel: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>("start");
  const [corner, setCorner] = useState<PixelPoint | null>(null);
  const [regionDraft, setRegionDraft] = useState({
    x: "",
    y: "",
    width: "",
    height: "",
  });
  const [point, setPoint] = useState({ x: "", y: "" });
  const [colors, setColors] = useState<{ empty: RGB[]; occupied: RGB[] }>({
    empty: [],
    occupied: [],
  });
  const [settings, setSettings] = useState(
    () =>
      Object.fromEntries(
        Object.entries(CALIBRATION_DEFAULTS).map(([key, value]) => [
          key,
          String(value),
        ]),
      ) as Record<keyof typeof CALIBRATION_DEFAULTS, string>,
  );
  const [message, setMessage] = useState("");
  const region = useMemo(() => {
    const values = regionKeys.map((key) => numeric(regionDraft[key]));
    if (values.some((v) => v === null)) return null;
    const [x, y, width, height] = values as number[];
    if (
      x < 0 ||
      y < 0 ||
      width <= 0 ||
      height <= 0 ||
      x + width > frame.width ||
      y + height > frame.height
    )
      return null;
    return { x, y, width, height };
  }, [regionDraft, frame.width, frame.height]);
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    paint(element, frame, region);
    return () => {
      element.width = element.height = 0;
    };
  }, [frame, region]);

  function edit() {
    onChange();
    setMessage("");
  }
  function addColor(kind: "empty" | "occupied", selected: PixelPoint) {
    edit();
    try {
      const color = sampleFrameColor(frame, selected);
      if (colors[kind].some((c) => c.every((v, i) => v === color[i]))) {
        setMessage("같은 색상 표본이 이미 있습니다.");
        return;
      }
      if (colors[kind].length >= 16) {
        setMessage("각 종류의 표본은 최대 16개입니다.");
        return;
      }
      setColors({ ...colors, [kind]: [...colors[kind], color] });
    } catch {
      setMessage("표본 좌표와 불투명한 픽셀인지 확인하세요.");
    }
  }
  function choose(selected: PixelPoint) {
    setPoint({ x: String(selected.x), y: String(selected.y) });
    if (mode === "empty" || mode === "occupied") {
      addColor(mode, selected);
      return;
    }
    edit();
    if (mode === "start") {
      setRegionDraft({ x: "", y: "", width: "", height: "" });
      setCorner(selected);
      setMode("end");
      setMessage(
        "반대 모서리를 선택하세요. 선택한 픽셀까지 영역에 포함됩니다.",
      );
      return;
    }
    if (!corner) {
      setMessage("첫 모서리를 먼저 선택하세요.");
      return;
    }
    const selectedRegion = regionFromCorners(corner, selected);
    setRegionDraft(
      Object.fromEntries(
        regionKeys.map((key) => [key, String(selectedRegion[key])]),
      ) as typeof regionDraft,
    );
    setCorner(null);
    setMode("empty");
  }
  function addAtCoordinates(kind: "empty" | "occupied") {
    const x = numeric(point.x);
    const y = numeric(point.y);
    if (x === null || y === null) {
      setMessage("표본 x·y 좌표를 입력하세요.");
      return;
    }
    addColor(kind, { x, y });
  }

  return (
    <fieldset
      disabled={disabled}
      className="space-y-3"
      aria-label="보드 영역·색상 표본 설정"
    >
      <legend className="font-semibold">
        정지 프레임에서 보드 영역·색상 선택
      </legend>
      <p className="help">
        프레임 {frame.width}×{frame.height}. 보드의 두 모서리와 빈칸·점유 칸의
        중앙 색상을 직접 선택하세요. 여러 점유 색상을 추가할 수 있습니다. 선택
        중 한 장만 이 탭 메모리에 유지하며 인식·취소·공유 종료 때 지웁니다.
      </p>
      <div className="flex flex-wrap gap-2">
        {modes.map((entry) => (
          <button
            className="small-button"
            key={entry.value}
            aria-pressed={mode === entry.value}
            onClick={() => setMode(entry.value)}
          >
            {entry.label}
          </button>
        ))}
      </div>
      <canvas
        ref={canvas}
        aria-label="정지 프레임 영역·표본 선택"
        style={{
          width: "auto",
          maxWidth: "100%",
          maxHeight: "70vh",
          height: "auto",
          cursor: "crosshair",
        }}
        onClick={(event) => {
          if (disabled) return;
          const selected = pointerToPixel(
            frame,
            event.currentTarget.getBoundingClientRect(),
            event,
          );
          if (selected) choose(selected);
        }}
      />
      <p className="help">
        마우스 대신 아래 원본 픽셀 좌표로 영역과 표본을 지정할 수 있습니다.
        보드는 10열×16행이며 흰 격자가 각 칸을 따라야 합니다.
      </p>
      <div className="grid grid-cols-4 gap-2">
        {regionKeys.map((key) => (
          <label key={key}>
            보드 영역 {key}
            <input
              type="number"
              aria-label={`보드 영역 ${key}`}
              value={regionDraft[key]}
              onChange={(event) => {
                edit();
                setCorner(null);
                setRegionDraft({ ...regionDraft, [key]: event.target.value });
              }}
            />
          </label>
        ))}
      </div>
      {!region && (
        <p className="help">프레임 안의 보드 영역을 먼저 지정하세요.</p>
      )}
      <div className="grid grid-cols-2 gap-2">
        {(["x", "y"] as const).map((key) => (
          <label key={key}>
            표본 {key}
            <input
              type="number"
              aria-label={`색상 표본 ${key}`}
              value={point[key]}
              onChange={(event) =>
                setPoint({ ...point, [key]: event.target.value })
              }
            />
          </label>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          className="small-button"
          onClick={() => addAtCoordinates("empty")}
        >
          좌표에서 빈칸 표본 추가
        </button>
        <button
          className="small-button"
          onClick={() => addAtCoordinates("occupied")}
        >
          좌표에서 점유 표본 추가
        </button>
      </div>
      {(["empty", "occupied"] as const).map((kind) => (
        <div
          key={kind}
          aria-label={kind === "empty" ? "빈칸 색상 표본" : "점유 색상 표본"}
        >
          <p>
            {kind === "empty" ? "빈칸" : "점유"} 표본 {colors[kind].length}개
          </p>
          <div className="flex flex-wrap gap-2">
            {colors[kind].map((color, index) => (
              <button
                key={index}
                className="small-button"
                aria-label={`${kind === "empty" ? "빈칸" : "점유"} 표본 ${index} 제거`}
                onClick={() => {
                  edit();
                  setColors({
                    ...colors,
                    [kind]: colors[kind].filter((_, i) => i !== index),
                  });
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: 16,
                    height: 16,
                    backgroundColor: `rgb(${color.join(",")})`,
                    marginRight: 4,
                  }}
                />
                {color.join(",")} ×
              </button>
            ))}
          </div>
        </div>
      ))}
      <details>
        <summary>판별 설정 조정</summary>
        <p className="help">
          기본값은 일반 표본 판별 설정이며 실제 게임 정확도나 확률을 뜻하지
          않습니다.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(settingLabels) as (keyof typeof settingLabels)[]).map(
            (key) => (
              <label key={key}>
                {settingLabels[key]}
                <input
                  type="number"
                  step="any"
                  aria-label={settingLabels[key]}
                  value={settings[key]}
                  onChange={(event) => {
                    edit();
                    setSettings({ ...settings, [key]: event.target.value });
                  }}
                />
              </label>
            ),
          )}
        </div>
      </details>
      {message && (
        <p role="alert" aria-label="보드 인식 설정 안내">
          {message}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <button
          className="primary-button"
          disabled={!region || !colors.empty.length || !colors.occupied.length}
          onClick={() => {
            edit();
            try {
              if (!region) return;
              const parsed = Object.fromEntries(
                Object.entries(settings).map(([key, value]) => [
                  key,
                  numeric(value) ?? NaN,
                ]),
              ) as typeof CALIBRATION_DEFAULTS;
              const engine = createCalibratedBoardEngine({
                region,
                emptyColors: colors.empty,
                occupiedColors: colors.occupied,
                ...parsed,
              });
              onResult(engine.recognize({ kind: "frame", frame }));
            } catch {
              setMessage(
                "보드 영역·표본·판별 설정을 확인하세요. 더 작은 표본 점 수나 더 큰 영역이 필요할 수 있습니다.",
              );
            }
          }}
        >
          선택한 영역의 보드 인식
        </button>
        <button className="small-button" onClick={onCancel}>
          프레임 선택 취소
        </button>
      </div>
    </fieldset>
  );
}
