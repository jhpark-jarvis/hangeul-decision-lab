import type { TextTemplate } from "./panel-mask";

/** Only synthetic system-font glyphs; never extracts or retains captured pixels. */
export function createPanelTextTemplates(): TextTemplate[] {
  const canvas = document.createElement("canvas");
  const templates: TextTemplate[] = [];
  try {
    canvas.width = 140;
    canvas.height = 32;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return templates;
    for (const value of [
      "사용 완료",
      ...Array.from({ length: 10 }, (_, i) => String(i)),
    ])
      for (const family of [
        "Malgun Gothic",
        "Arial",
        "Segoe UI",
        "Impact",
        "sans-serif",
      ])
        for (const weight of [400, 700])
          for (const size of value === "사용 완료" ? [12, 14, 16] : [16]) {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.font = `${weight} ${size}px ${family === "sans-serif" ? family : `"${family}"`}`;
            ctx.textBaseline = "alphabetic";
            ctx.fillStyle = "white";
            ctx.fillText(value, 4, 24);
            const pixels = ctx.getImageData(
              0,
              0,
              canvas.width,
              canvas.height,
            ).data;
            try {
              templates.push({
                value,
                mask: {
                  width: canvas.width,
                  height: canvas.height,
                  data: Uint8Array.from(
                    { length: canvas.width * canvas.height },
                    (_, i) => Number(pixels[i * 4] > 170),
                  ),
                },
              });
            } finally {
              pixels.fill(0);
            }
          }
    return templates;
  } catch {
    templates.forEach((t) => t.mask.data.fill(0));
    return [];
  } finally {
    canvas.width = 0;
    canvas.height = 0;
  }
}

export function releasePanelTextTemplates(templates: readonly TextTemplate[]) {
  templates.forEach((t) => t.mask.data.fill(0));
}
