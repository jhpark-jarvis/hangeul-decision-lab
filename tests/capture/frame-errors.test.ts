import { afterEach, describe, expect, it, vi } from "vitest";
import {
  captureCurrentFrame,
  FrameCaptureError,
  frameCaptureFailureMessage,
} from "../../src/features/capture/capture";

afterEach(() => vi.unstubAllGlobals());
const video = (values: Partial<HTMLVideoElement> = {}) =>
  ({
    videoWidth: 1,
    videoHeight: 1,
    readyState: 2,
    ...values,
  }) as HTMLVideoElement;
describe("safe frame failure boundary", () => {
  it("explains readiness without allocating Canvas", () => {
    const createElement = vi.fn();
    vi.stubGlobal("document", { createElement });
    let failure: unknown;
    try {
      captureCurrentFrame(video({ readyState: 0 }));
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(FrameCaptureError);
    expect(frameCaptureFailureMessage(failure)).toContain("미리보기 재생 후");
    expect(createElement).not.toHaveBeenCalled();
  });
  it("preserves the smaller-window recovery instruction before any allocation", () => {
    const createElement = vi.fn();
    vi.stubGlobal("document", { createElement });
    let failure: unknown;
    try {
      captureCurrentFrame(video({ videoWidth: 8192, videoHeight: 8192 }));
    } catch (error) {
      failure = error;
    }
    expect(frameCaptureFailureMessage(failure)).toBe(
      "공유 화면이 너무 큽니다. 더 작은 창을 공유하세요.",
    );
    expect(createElement).not.toHaveBeenCalled();
  });
  it("explains missing Canvas and frees its allocation", () => {
    const canvas = { width: 0, height: 0, getContext: () => null };
    vi.stubGlobal("document", { createElement: () => canvas });
    let failure: unknown;
    try {
      captureCurrentFrame(video());
    } catch (error) {
      failure = error;
    }
    expect(frameCaptureFailureMessage(failure)).toContain(
      "Canvas를 사용할 수 없습니다",
    );
    expect([canvas.width, canvas.height]).toEqual([0, 0]);
  });
  it.each([
    new Error("private-window-info"),
    new Error("프레임 private-window-info"),
    new DOMException("private-window-info", "SecurityError"),
    { message: "프레임 private-window-info" },
    null,
  ])("never exposes arbitrary or native error text %j", (failure) => {
    expect(frameCaptureFailureMessage(failure)).toBe(
      "프레임을 추출하지 못했습니다. 공유 화면과 재생 상태를 확인하세요.",
    );
  });
  it("uses its fixed code even when an error message has been changed", () => {
    const error = new FrameCaptureError("TOO_LARGE");
    error.message = "private-window-info";
    expect(frameCaptureFailureMessage(error)).toContain("더 작은 창");
    Object.defineProperty(error, "code", { value: "constructor" });
    expect(frameCaptureFailureMessage(error)).not.toContain(
      "private-window-info",
    );
    expect(typeof frameCaptureFailureMessage(error)).toBe("string");
  });
  it("clears native pixel-extraction failure before showing a safe fallback", () => {
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage: () => {},
        getImageData: () => {
          throw new DOMException("private-window-info", "SecurityError");
        },
      }),
    };
    vi.stubGlobal("document", { createElement: () => canvas });
    let failure: unknown;
    try {
      captureCurrentFrame(video());
    } catch (error) {
      failure = error;
    }
    expect([canvas.width, canvas.height]).toEqual([0, 0]);
    expect(frameCaptureFailureMessage(failure)).not.toMatch(
      /SecurityError|private-window-info/,
    );
  });
});
