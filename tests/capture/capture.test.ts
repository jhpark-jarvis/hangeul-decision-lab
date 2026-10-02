import { afterEach, describe, expect, it, vi } from "vitest";
import {
  captureCurrentFrame,
  createCaptureController,
  type CaptureStatus,
} from "../../src/features/capture/capture";

class Track extends EventTarget {
  readyState = "live";
  stops = 0;
  stop() {
    this.readyState = "ended";
    this.stops++;
  }
}
function stream() {
  const track = new Track();
  const audio = new Track();
  const value = {
    getTracks: () => [track, audio],
    getVideoTracks: () => [track],
  } as unknown as MediaStream;
  return { track, audio, value };
}
afterEach(() => vi.unstubAllGlobals());
describe("capture ownership", () => {
  it("requests synchronously from start, stops all tracks without needing ended event", async () => {
    const s = stream();
    const states: CaptureStatus[] = [];
    const request = vi.fn(() => Promise.resolve(s.value));
    const owned = createCaptureController(request, (state) =>
      states.push(state),
    );
    const pending = owned.start();
    expect(request).toHaveBeenCalledOnce();
    expect(states[0].phase).toBe("requesting");
    expect(await pending).toBe(s.value);
    expect(states.at(-1)?.phase).toBe("active");
    owned.stop();
    expect(s.track.stops).toBe(1);
    expect(s.audio.stops).toBe(1);
    expect(states.at(-1)?.phase).toBe("idle");
    s.track.dispatchEvent(new Event("ended"));
    expect(s.track.stops).toBe(1);
    owned.dispose();
  });
  it.each([
    "AbortError",
    "NotAllowedError",
    "NotSupportedError",
    "NotReadableError",
  ])("recovers from %s without logging exception details", async (name) => {
    const states: CaptureStatus[] = [];
    let fail = true;
    const s = stream();
    const owned = createCaptureController(
      () =>
        fail
          ? Promise.reject(new DOMException("private desktop name", name))
          : Promise.resolve(s.value),
      (state) => states.push(state),
    );
    expect(await owned.start()).toBeNull();
    expect(states.at(-1)?.phase).toBe("error");
    expect(JSON.stringify(states)).not.toContain("private");
    fail = false;
    expect(await owned.start()).toBe(s.value);
    owned.dispose();
    expect(s.track.stops).toBe(1);
  });
  it("external track ended releases stream and enables restart", async () => {
    const first = stream();
    const second = stream();
    let count = 0;
    const states: CaptureStatus[] = [];
    const owned = createCaptureController(
      () => Promise.resolve(count++ ? second.value : first.value),
      (state) => states.push(state),
    );
    await owned.start();
    first.track.dispatchEvent(new Event("ended"));
    expect(states.at(-1)?.phase).toBe("idle");
    expect(first.audio.stops).toBe(1);
    await owned.start();
    expect(second.track.readyState).toBe("live");
    owned.dispose();
    expect(second.track.stops).toBe(1);
  });
  it.each(["stop", "dispose"] as const)(
    "%s while chooser is pending stops late stream",
    async (method) => {
      let resolve!: (value: MediaStream) => void;
      const s = stream();
      const states: CaptureStatus[] = [];
      const owned = createCaptureController(
        () =>
          new Promise((done) => {
            resolve = done;
          }),
        (state) => states.push(state),
      );
      const pending = owned.start();
      owned[method]();
      const count = states.length;
      resolve(s.value);
      expect(await pending).toBeNull();
      expect(s.track.stops).toBe(1);
      expect(states).toHaveLength(count);
      expect(
        await (method === "dispose" ? owned.start() : Promise.resolve(null)),
      ).toBeNull();
    },
  );
  it("new request supersedes old request and disposal cleans active capture", async () => {
    let resolve!: (value: MediaStream) => void;
    const old = stream();
    const active = stream();
    let count = 0;
    const owned = createCaptureController(
      () =>
        count++
          ? Promise.resolve(active.value)
          : new Promise((done) => {
              resolve = done;
            }),
      () => {},
    );
    const pending = owned.start();
    await owned.start();
    resolve(old.value);
    expect(await pending).toBeNull();
    expect(old.track.stops).toBe(1);
    owned.dispose();
    expect(active.track.stops).toBe(1);
  });
  it("rejects already ended capture and cleans all tracks", async () => {
    const s = stream();
    s.track.readyState = "ended";
    const states: CaptureStatus[] = [];
    const owned = createCaptureController(
      () => Promise.resolve(s.value),
      (state) => states.push(state),
    );
    expect(await owned.start()).toBeNull();
    expect(s.audio.stops).toBe(1);
    expect(states.at(-1)?.phase).toBe("error");
  });
  it("extracts actual RGBA payload and releases temporary Canvas", () => {
    const pixels = new Uint8ClampedArray([1, 2, 3, 255]);
    const drawImage = vi.fn();
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({ drawImage, getImageData: () => ({ data: pixels }) }),
    };
    vi.stubGlobal("document", { createElement: () => canvas });
    const video = {
      videoWidth: 1,
      videoHeight: 1,
      readyState: 2,
    } as HTMLVideoElement;
    const frame = captureCurrentFrame(video);
    expect(frame.pixels).toBe(pixels);
    expect(frame.width).toBe(1);
    expect(frame.timestamp).toBeGreaterThan(0);
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 1, 1);
    expect(canvas.width).toBe(0);
  });
  it("rejects unready and oversized frame before allocating canvas", () => {
    for (const values of [
      { videoWidth: 0, videoHeight: 1, readyState: 0 },
      { videoWidth: 8192, videoHeight: 8192, readyState: 2 },
    ])
      expect(() => captureCurrentFrame(values as HTMLVideoElement)).toThrow();
  });
  it("cleans canvas even when pixel extraction fails", () => {
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage: () => {},
        getImageData: () => {
          throw new DOMException("private", "SecurityError");
        },
      }),
    };
    vi.stubGlobal("document", { createElement: () => canvas });
    expect(() =>
      captureCurrentFrame({
        videoWidth: 1,
        videoHeight: 1,
        readyState: 2,
      } as HTMLVideoElement),
    ).toThrow();
    expect(canvas.width).toBe(0);
  });
});
