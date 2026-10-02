import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createPanelTextTemplates,
  releasePanelTextTemplates,
} from "../../src/features/recognition/panel-fonts";
afterEach(() => vi.unstubAllGlobals());
describe("synthetic local font ownership", () => {
  it("zeros raster buffers and destroys the canvas before returning glyphs", () => {
    const buffers: Uint8ClampedArray[] = [];
    const ctx = {
      fillStyle: "",
      font: "",
      textBaseline: "",
      fillRect: vi.fn(),
      fillText: vi.fn(),
      getImageData: () => {
        const data = new Uint8ClampedArray(140 * 32 * 4).fill(255);
        buffers.push(data);
        return { data };
      },
    };
    const canvas = { width: 0, height: 0, getContext: () => ctx };
    vi.stubGlobal("document", { createElement: () => canvas });
    const templates = createPanelTextTemplates();
    expect(templates).toHaveLength(130);
    expect(canvas.width).toBe(0);
    expect(canvas.height).toBe(0);
    expect(buffers.every((b) => b.every((v) => v === 0))).toBe(true);
    expect(templates.every((t) => t.mask.data.some(Boolean))).toBe(true);
    releasePanelTextTemplates(templates);
    expect(templates.every((t) => t.mask.data.every((v) => v === 0))).toBe(
      true,
    );
  });
  it.each(["context", "pixels"])(
    "falls back without an owned canvas after %s failure",
    (mode) => {
      const canvas = {
        width: 0,
        height: 0,
        getContext: () =>
          mode === "context"
            ? null
            : {
                fillRect: () => {},
                fillText: () => {},
                getImageData: () => {
                  throw Error("unavailable");
                },
              },
      };
      vi.stubGlobal("document", { createElement: () => canvas });
      expect(createPanelTextTemplates()).toEqual([]);
      expect(canvas.width).toBe(0);
      expect(canvas.height).toBe(0);
    },
  );
});
