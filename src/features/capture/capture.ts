export type CaptureStatus = {
  phase: "idle" | "requesting" | "active" | "error";
  message: string;
};
export type CapturedFrame = {
  width: number;
  height: number;
  timestamp: number;
  pixels: Uint8ClampedArray;
};

export function requestDisplayMedia(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getDisplayMedia)
    return Promise.reject(new DOMException("", "NotSupportedError"));
  // Called directly from the user's click, before any asynchronous work.
  return navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
}

export function createCaptureController(
  request: () => Promise<MediaStream>,
  notify: (status: CaptureStatus) => void,
) {
  let stream: MediaStream | null = null;
  let revision = 0;
  let disposed = false;
  let removeListeners = () => {};
  function release() {
    removeListeners();
    removeListeners = () => {};
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  function stop() {
    revision++;
    release();
    if (!disposed)
      notify({ phase: "idle", message: "화면 공유를 종료했습니다." });
  }
  return {
    async start(): Promise<MediaStream | null> {
      if (disposed) return null;
      const token = ++revision;
      release();
      notify({
        phase: "requesting",
        message: "공유할 화면이나 창을 선택하세요.",
      });
      try {
        const received = await request();
        if (disposed || token !== revision) {
          received.getTracks().forEach((track) => track.stop());
          return null;
        }
        if (
          !received
            .getVideoTracks()
            .some((track) => track.readyState === "live")
        ) {
          received.getTracks().forEach((track) => track.stop());
          throw new DOMException("", "NotReadableError");
        }
        stream = received;
        const ended = () => stop();
        const tracks = received.getVideoTracks();
        tracks.forEach((track) => track.addEventListener("ended", ended));
        removeListeners = () =>
          tracks.forEach((track) => track.removeEventListener("ended", ended));
        notify({
          phase: "active",
          message: "화면 공유 중 · 프레임은 이 탭에서만 처리합니다.",
        });
        return received;
      } catch (error) {
        if (disposed || token !== revision) return null;
        const name = error instanceof Error ? error.name : "UnknownError";
        const message =
          name === "NotAllowedError" || name === "AbortError"
            ? "화면 선택을 취소했거나 권한이 거절되었습니다. 다시 시작할 수 있습니다."
            : name === "NotSupportedError"
              ? "이 환경은 화면 공유를 지원하지 않습니다. Chrome/Edge의 로컬 페이지를 사용하세요."
              : "화면 공유를 시작하지 못했습니다. 다시 시도하세요.";
        notify({ phase: "error", message });
        return null;
      }
    },
    stop,
    dispose() {
      disposed = true;
      revision++;
      release();
    },
  };
}

// Explicit action only; no recognition/render loop, Blob URL, log or persistence.
export function captureCurrentFrame(video: HTMLVideoElement): CapturedFrame {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (video.readyState < 2 || width <= 0 || height <= 0)
    throw new Error(
      "프레임이 아직 준비되지 않았습니다. 미리보기 재생 후 다시 시도하세요.",
    );
  // Bound a single RGBA allocation to 64MiB. This is not a CV performance SLA.
  if (width * height > 16_777_216)
    throw new Error("공유 화면이 너무 큽니다. 더 작은 창을 공유하세요.");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  try {
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas를 사용할 수 없습니다.");
    context.drawImage(video, 0, 0, width, height);
    return {
      width,
      height,
      timestamp: Date.now(),
      pixels: context.getImageData(0, 0, width, height).data,
    };
  } finally {
    canvas.width = canvas.height = 0;
  }
}
