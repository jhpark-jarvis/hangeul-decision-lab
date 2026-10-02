import type { CapturedFrame } from "@/features/capture/capture";
import {
  releaseFrame,
  type PixelRegion,
} from "@/features/recognition/calibration";
import { cropReviewPreview } from "@/features/recognition/review-preview";

/** Caller owns the one visible canvas; scratch pixels never survive this call. */
export function paintReviewPreview(
  frame: CapturedFrame,
  region: PixelRegion,
): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "검토할 게임 보드 캡처");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  let preview: CapturedFrame | null = null;
  let image: ImageData | null = null;
  let painted = false;
  try {
    preview = cropReviewPreview(frame, region);
    canvas.width = preview.width;
    canvas.height = preview.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    image = ctx.createImageData(preview.width, preview.height);
    image.data.set(preview.pixels);
    ctx.putImageData(image, 0, 0);
    painted = true;
    return canvas;
  } catch {
    return null;
  } finally {
    image?.data.fill(0);
    releaseFrame(preview);
    if (!painted) releaseReviewPreview(canvas);
  }
}

export function releaseReviewPreview(canvas: HTMLCanvasElement | null): void {
  if (!canvas) return;
  canvas.width = 0;
  canvas.height = 0;
  canvas.remove();
}
