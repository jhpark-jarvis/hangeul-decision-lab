import { readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

const buildId = readFileSync(".next/BUILD_ID", "utf8").trim();
const reports = new Map();
const button = (page, name) => page.getByRole("button", { name, exact: true });
const number = (page, name) =>
  page.getByRole("spinbutton", { name, exact: true });
const group = (page) =>
  page.getByRole("group", { name: "보드 영역·색상 표본 설정", exact: true });
const canvas = (page) =>
  page.getByLabel("정지 프레임 영역·표본 선택", { exact: true });
async function capture(page) {
  if (await button(page, "Start Screen Capture").isEnabled()) {
    await button(page, "Start Screen Capture").click();
    await page.waitForFunction(
      () =>
        document.querySelector("video").videoWidth === 960 &&
        document.querySelector("video").readyState >= 2,
    );
  }
  await button(page, "수동 영역·색상 선택").click();
  await expect(group(page)).toBeVisible();
  await page.evaluate(() =>
    window.__calibration.rememberCanvas(
      document.querySelector('canvas[aria-label="정지 프레임 영역·표본 선택"]'),
    ),
  );
}
async function region(page) {
  for (const [key, value] of Object.entries({
    x: 80,
    y: 80,
    width: 800,
    height: 1280,
  }))
    await number(page, `보드 영역 ${key}`).fill(String(value));
}
async function color(page, kind, x, y) {
  await number(page, "색상 표본 x").fill(String(x));
  await number(page, "색상 표본 y").fill(String(y));
  await button(page, `좌표에서 ${kind} 표본 추가`).click();
}
async function point(page, x, y) {
  const rect = await canvas(page).boundingBox();
  expect(rect.width).toBeLessThan(960);
  expect(rect.height).toBeLessThan(1440);
  await canvas(page).click({
    position: {
      x: ((x + 0.5) / 960) * rect.width,
      y: ((y + 0.5) / 1440) * rect.height,
    },
  });
}
async function cleared(page) {
  await expect(group(page)).toHaveCount(0);
  await page.waitForFunction(() => {
    const summary = window.__calibration.summary();
    return (
      summary.buffers.every(Boolean) &&
      summary.scratch.every(Boolean) &&
      summary.canvases.every(Boolean)
    );
  });
}
async function fillReview(page) {
  for (let slot = 0; slot < 3; slot++)
    await page
      .getByRole("combobox", {
        name: ["첫 번째 보유 조각", "두 번째 보유 조각", "세 번째 보유 조각"][
          slot
        ],
        exact: true,
      })
      .selectOption("DOT");
  for (const kind of ["바꿔 뽑기", "점 찍기"])
    await number(page, `${kind} 남은 횟수`).fill("0");
  await page
    .getByRole("checkbox", {
      name: "보드의 아이템을 모두 확인했습니다 (없으면 그대로 체크)",
      exact: true,
    })
    .check();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
}

test.beforeEach(async ({ page, context }, info) => {
  const entry = {
    buildId,
    measuredAt: new Date().toISOString(),
    browser: info.project.name,
    fixture: info.title,
    source: /^(panel|connected)/.test(info.title)
      ? "Synthetic 432x466 board and right-panel Canvas stream. Native picker and actual desktop never used."
      : /^(automatic|visual)/.test(info.title)
        ? "Synthetic 300x466 gradient Canvas stream / 10x16 board / local crop. Native picker and actual desktop never used."
        : "Synthetic 960x1440 Canvas stream / arbitrary colors / 10x16 board. Native picker and actual desktop never used.",
    requests: [],
    errors: [],
    sockets: [],
  };
  reports.set(info.testId, entry);
  context.on("request", (r) =>
    entry.requests.push({
      url: r.url(),
      method: r.method(),
      body: r.postData(),
      type: r.resourceType(),
    }),
  );
  page.on("pageerror", (error) => entry.errors.push(String(error)));
  page.on("console", (m) => {
    if (["error", "warning"].includes(m.type()))
      entry.errors.push(`${m.type()}: ${m.text()}`);
  });
  page.on("websocket", (socket) => entry.sockets.push(socket.url()));
  await context.route("**/*", (route) =>
    new URL(route.request().url()).origin === "http://127.0.0.1:45001"
      ? route.continue()
      : route.abort("blockedbyclient"),
  );
  await page.addInitScript(() => {
    const tracks = [],
      buffers = [],
      scratch = [],
      canvases = [];
    const created = [],
      fontCanvases = [];
    const createElement = document.createElement.bind(document);
    document.createElement = (...args) => {
      const element = createElement(...args);
      if (args[0] === "canvas") created.push(element);
      return element;
    };
    let canvasFailure = "";
    let automaticMode = "";
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (...args) {
      return canvasFailure === "context" ? null : getContext.apply(this, args);
    };
    const getImageData = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function (...args) {
      if (canvasFailure === "pixels")
        throw new DOMException("private-window-info", "SecurityError");
      const image = getImageData.apply(this, args);
      if (
        this.canvas.width === 140 &&
        this.canvas.height === 32 &&
        !fontCanvases.includes(this.canvas)
      )
        fontCanvases.push(this.canvas);
      buffers.push(image.data);
      return image;
    };
    const createImageData = CanvasRenderingContext2D.prototype.createImageData;
    CanvasRenderingContext2D.prototype.createImageData = function (...args) {
      const image = createImageData.apply(this, args);
      scratch.push(image.data);
      return image;
    };
    Object.defineProperty(navigator.mediaDevices, "getDisplayMedia", {
      configurable: true,
      value: () => {
        const element = document.createElement("canvas");
        if (automaticMode) {
          const panelMode = automaticMode.startsWith("panel");
          const connectedMode = automaticMode.startsWith("panel-connected");
          const frameWidth = panelMode ? 432 : 300;
          element.width = frameWidth;
          element.height = 466;
          const ctx = element.getContext("2d");
          const image = new ImageData(frameWidth, 466);
          const rows = [
            "0010000000",
            "0101000000",
            "0010000000",
            "1000000000",
            "1100000111",
            "1101000011",
            "1101111111",
            "1001101000",
            "0001100000",
            "0001000000",
            "0001000000",
            "0111110000",
            "0010101000",
            "0001111110",
            "0000010100",
            "0000001000",
          ];
          for (let y = 0; y < 466; y++)
            for (let x = 0; x < frameWidth; x++) {
              const row = Math.floor((y - 30) / 26),
                col = Math.floor((x - 20) / 26);
              let color = [30, 30, 30];
              if (connectedMode && x >= 20 && y >= 30 && y < 450) {
                color = [50, 175, 190];
                if (x >= 280 && (x - 20) % 26 < 2) color = [20, 95, 160];
              }
              if (
                automaticMode !== "missing" &&
                row >= 0 &&
                row < 16 &&
                col >= 0 &&
                col < 10
              ) {
                const fx = (x - 20) / 26 - col,
                  fy = (y - 30) / 26 - row;
                color = [50, 165 + row * 0.7, 185 + row * 0.2];
                if (fx < 0.06 || fy < 0.06) color = color.map((c) => c - 12);
                if (
                  automaticMode !== "panel-connected-empty" &&
                  rows[row][col] === "1" &&
                  fx > 0.08 &&
                  fx < 0.92 &&
                  fy > 0.08 &&
                  fy < 0.92
                ) {
                  const base = [
                    [60, 160, 220],
                    [130, 190, 40],
                    [210, 100, 180],
                    [220, 150, 30],
                  ][(row + col) % 4];
                  color = base.map((c) =>
                    Math.min(255, c + 100 * (1 - (fx + fy) / 2)),
                  );
                }
                if (
                  ["occluded", "items"].includes(automaticMode) &&
                  row === 1 &&
                  col === 0
                )
                  color = [255, 255, 255];
                if (automaticMode === "items" && row === 12 && col === 7)
                  color =
                    Math.hypot(fx - 0.5, fy - 0.5) < 0.28
                      ? [30, 90, 240]
                      : [255, 255, 255];
              }
              image.data.set(
                [...color.map(Math.round), 255],
                (y * frameWidth + x) * 4,
              );
            }
          ctx.putImageData(image, 0, 0);
          image.data.fill(0);
          if (panelMode) {
            for (let slot = 0; slot < 3; slot++) {
              const x = 293,
                y = 30 + 26 * (0.9 + 2.9 * slot),
                w = 122.2,
                h = 70.2;
              ctx.fillStyle = slot === 0 ? "#f4fafc" : "#16c4d6";
              ctx.fillRect(x, y, w, h);
              if (slot === 0) {
                const rows = ["11", "10", "11"],
                  p = w / 15,
                  left = x + w * 0.25 - p,
                  top = y + h / 2 - p * 1.5;
                ctx.fillStyle = "#e545af";
                rows.forEach((row, r) =>
                  [...row].forEach((cell, c) => {
                    if (cell === "1")
                      ctx.fillRect(left + c * p, top + r * p, p - 1, p - 1);
                  }),
                );
              } else {
                ctx.font = '700 14px "Malgun Gothic"';
                ctx.textAlign = "center";
                ctx.fillStyle = "white";
                ctx.fillText(
                  automaticMode === "panel-bad" && slot === 2
                    ? "사용 중"
                    : "사용 완료",
                  x + w / 2,
                  y + h * 0.6,
                );
              }
            }
            for (const [key, x, y, w, h] of [
              ["singleCell", 382.7, 392.7, 16.9, 15.6],
              ["reroll", 382.7, 429.1, 16.9, 15.6],
              ["total", 342.4, 362.8, 10.4, 16.9],
            ]) {
              ctx.fillStyle = key === "total" ? "#e6fbfc" : "#1466be";
              ctx.fillRect(x - 4, y - 4, w + 8, h + 8);
              ctx.font = `700 ${key === "total" ? 14 : 12}px ${automaticMode === "panel-narrow" ? "Impact" : "Arial"}`;
              ctx.textAlign = "center";
              ctx.fillStyle = key === "total" ? "#1482a0" : "#ffffff";
              const value =
                automaticMode === "panel-bad" && key === "total" ? "1" : "0";
              ctx.fillText(value, x + w / 2, y + h * 0.78);
            }
          }
          const stream = element.captureStream(1);
          tracks.push(...stream.getTracks());
          return Promise.resolve(stream);
        }
        element.width = 960;
        element.height = 1440;
        const ctx = element.getContext("2d");
        ctx.fillStyle = "rgb(200,0,200)";
        ctx.fillRect(0, 0, 960, 1440);
        for (let row = 0; row < 16; row++)
          for (let col = 0; col < 10; col++) {
            ctx.fillStyle =
              row === 0 && col < 9 ? "rgb(180,190,200)" : "rgb(20,30,40)";
            if (row === 0 && col === 1) ctx.fillStyle = "rgb(80,100,220)";
            if (row === 1 && col === 0) ctx.fillStyle = "rgb(200,0,200)";
            ctx.fillRect(80 + col * 80, 80 + row * 80, 80, 80);
          }
        const stream = element.captureStream(1);
        tracks.push(...stream.getTracks());
        return Promise.resolve(stream);
      },
    });
    window.__calibration = {
      automaticMode: (mode) => {
        automaticMode = mode;
      },
      canvasFailure: (mode) => {
        canvasFailure = mode;
      },
      rememberCanvas: (element) => canvases.push(element),
      end: () => tracks.at(-1).dispatchEvent(new Event("ended")),
      summary: () => ({
        buffers: buffers.map((data) => data.every((v) => v === 0)),
        scratch: scratch.map((data) => data.every((v) => v === 0)),
        canvases: canvases.map((c) => c.width === 0 && c.height === 0),
        previews: created
          .filter(
            (c) => c.getAttribute("aria-label") === "검토할 게임 보드 캡처",
          )
          .map((c) => c.width === 0 && c.height === 0),
        tracks: tracks.map((t) => t.readyState),
        fontCanvases: fontCanvases.map((c) => c.width === 0 && c.height === 0),
      }),
    };
  });
  await page.goto("/");
  for (let slot = 0; slot < 3; slot++)
    await page
      .getByRole("combobox", { name: `slot ${slot} 블록`, exact: true })
      .selectOption("DOT");
  await button(page, "3개 블록 입력").click();
  await button(page, "화면 캡처·검토 열기").click();
});
test.afterEach(async ({ page, context, browser }, info) => {
  if (await button(page, "화면 입력 닫기").count())
    await button(page, "화면 입력 닫기").click();
  const entry = reports.get(info.testId);
  entry.version = browser.version();
  entry.result = info.status;
  entry.cleanup = await page.evaluate(() => window.__calibration.summary());
  entry.storage = await page.evaluate(async () => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
    indexedDB: (await indexedDB.databases()).map((d) => d.name),
    caches: await caches.keys(),
  }));
  entry.cookies = (await context.cookies()).map((c) => c.name);
  writeFileSync(
    info.outputPath("calibration.json"),
    JSON.stringify(entry, null, 2) + "\n",
  );
  expect(entry.errors).toEqual([]);
  expect(entry.sockets).toEqual([]);
  expect(entry.cookies).toEqual([]);
  expect(entry.storage).toEqual({
    local: [],
    session: [],
    indexedDB: [],
    caches: [],
  });
  expect(entry.cleanup.buffers.every(Boolean)).toBe(true);
  expect(entry.cleanup.scratch.every(Boolean)).toBe(true);
  expect(entry.cleanup.canvases.every(Boolean)).toBe(true);
  expect(entry.cleanup.previews.every(Boolean)).toBe(true);
  expect(entry.cleanup.fontCanvases.every(Boolean)).toBe(true);
  expect(entry.cleanup.tracks.every((state) => state === "ended")).toBe(true);
  expect(
    entry.requests.filter(
      (r) =>
        !r.url.startsWith("http://127.0.0.1:45001/") ||
        r.method !== "GET" ||
        r.body !== null ||
        ["fetch", "xhr"].includes(r.type),
    ),
  ).toEqual([]);
});

for (const empty of [false, true])
  test(`connected panel is excluded from the ${empty ? "empty" : "occupied"} board preview`, async ({
    page,
  }) => {
    await page.evaluate(
      (empty) =>
        window.__calibration.automaticMode(
          empty ? "panel-connected-empty" : "panel-connected",
        ),
      empty,
    );
    await button(page, "Start Screen Capture").click();
    await page.waitForFunction(
      () =>
        document.querySelector("video").videoWidth === 432 &&
        document.querySelector("video").readyState >= 2,
    );
    await button(page, "Capture Frame").click();
    await expect(
      page.getByLabel("검토할 게임 보드 캡처", { exact: true }),
    ).toBeVisible();
    await expect(
      page.locator('button[data-review-unresolved="true"]'),
    ).toHaveCount(0);
    await expect(page.locator("button[data-review-row]")).toHaveCount(160);
    // Compare against the generated game frame's exact board crop, including the
    // horizontal phase. Geometry alone would miss a same-size crop shifted right.
    const matched = await page.evaluate(() => {
      const video = document.querySelector("video"),
        preview = document.querySelector(
          'canvas[aria-label="검토할 게임 보드 캡처"]',
        ),
        expected = document.createElement("canvas");
      expected.width = preview.width;
      expected.height = preview.height;
      const ctx = expected.getContext("2d");
      ctx.drawImage(
        video,
        20,
        30,
        260,
        416,
        0,
        0,
        expected.width,
        expected.height,
      );
      const a = ctx.getImageData(0, 0, expected.width, expected.height).data,
        b = preview
          .getContext("2d")
          .getImageData(0, 0, preview.width, preview.height).data;
      // Accept the documented grid-edge tolerance but reject UI white panels.
      let different = 0;
      for (let i = 0; i < a.length; i += 4)
        if (
          Math.abs(a[i] - b[i]) +
            Math.abs(a[i + 1] - b[i + 1]) +
            Math.abs(a[i + 2] - b[i + 2]) >
          60
        )
          different++;
      const ratio = different / (a.length / 4);
      a.fill(0);
      b.fill(0);
      expected.width = 0;
      expected.height = 0;
      return ratio;
    });
    expect(matched).toBeLessThan(0.08);
    await page.locator(".review-board").screenshot({
      path: test
        .info()
        .outputPath(`synthetic-connected-${empty ? "empty" : "occupied"}.png`),
    });
    await expect(
      page.getByRole("status", { name: "점유 칸 수", exact: true }),
    ).toHaveText("0 / 160");
    await expect(button(page, "Use This State")).toBeDisabled();
    await fillReview(page);
    await button(page, "Use This State").click();
    await expect(
      page.getByRole("status", { name: "점유 칸 수", exact: true }),
    ).toHaveText(`${empty ? 0 : 49} / 160`);
    await expect(
      page.getByLabel("검토할 게임 보드 캡처", { exact: true }),
    ).toHaveCount(0);
    await button(page, "Stop Capture").click();
  });

test("panel same-frame capture fills held pieces and abilities before explicit Use/Analyze/Apply", async ({
  page,
}) => {
  await page.evaluate(() => window.__calibration.automaticMode("panel"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 432 &&
      document.querySelector("video").readyState >= 2,
  );
  const start = performance.now();
  await button(page, "Capture Frame").click();
  const review = page.getByRole("group", {
    name: "인식 결과 검토",
    exact: true,
  });
  await expect(review).toContainText("같은 화면의 보유 조각·능력");
  const labels = [
    "첫 번째 보유 조각",
    "두 번째 보유 조각",
    "세 번째 보유 조각",
  ];
  for (const [index, value] of ["C_5", "empty", "empty"].entries())
    await expect(
      page.getByRole("combobox", { name: labels[index], exact: true }),
    ).toHaveValue(value);
  for (const label of ["점 찍기", "바꿔 뽑기"])
    await expect(number(page, `${label} 남은 횟수`)).toHaveValue("0");
  const entry = reports.get(test.info().testId);
  entry.frameToReviewMs = performance.now() - start;
  entry.source =
    "Synthetic 432x466 board and right panel, Canvas font glyphs; native picker not used.";
  await expect(
    page.getByRole("status", { name: "프레임 상태", exact: true }),
  ).toContainText("보유 조각 3/3 · 능력 횟수 2/2");
  const itemCheck = page.getByRole("checkbox", {
    name: "보드의 아이템을 모두 확인했습니다 (없으면 그대로 체크)",
    exact: true,
  });
  const allCheck = page.getByRole("checkbox", {
    name: "검토한 전체 상태 확인",
    exact: true,
  });
  await expect(itemCheck).not.toBeChecked();
  await expect(allCheck).not.toBeChecked();
  await expect(button(page, "Use This State")).toBeDisabled();
  await expect(number(page, "Single Cell 보유 수")).toHaveValue("0");
  await expect(
    page.getByRole("combobox", { name: "slot 0 블록", exact: true }),
  ).toHaveValue("DOT");
  await itemCheck.check();
  await allCheck.check();
  await expect(button(page, "Use This State")).toBeEnabled();
  await page
    .getByRole("combobox", { name: labels[0], exact: true })
    .selectOption("HOOK_4");
  await expect(allCheck).not.toBeChecked();
  await page
    .getByRole("combobox", { name: labels[0], exact: true })
    .selectOption("C_5");
  await allCheck.check();
  await review.screenshot({
    path: test.info().outputPath("synthetic-panel-review.png"),
  });
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("combobox", { name: "slot 0 블록", exact: true }),
  ).toHaveValue("C_5");
  for (const slot of [1, 2])
    await expect(
      page.getByRole("combobox", { name: `slot ${slot} 블록`, exact: true }),
    ).toHaveValue("");
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("49 / 160");
  await button(page, "Analyze").click();
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Apply Step").click();
  await expect(
    page.getByRole("region", { name: "입력 및 적용 상태", exact: true }),
  ).toContainText("단계 1");
  await button(page, "Capture Frame").click();
  await button(page, "Stop Capture").click();
  await cleared(page);
});

test("panel partial reads and inconsistent totals remain editable and stale review cannot apply", async ({
  page,
}) => {
  await page.evaluate(() => window.__calibration.automaticMode("panel-bad"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 432 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  const labels = [
    "첫 번째 보유 조각",
    "두 번째 보유 조각",
    "세 번째 보유 조각",
  ];
  for (const [index, value] of ["C_5", "empty", "unknown"].entries())
    await expect(
      page.getByRole("combobox", { name: labels[index], exact: true }),
    ).toHaveValue(value);
  for (const label of ["점 찍기", "바꿔 뽑기"])
    await expect(number(page, `${label} 남은 횟수`)).toHaveValue("");
  await expect(
    page.locator('button[data-review-unresolved="true"]'),
  ).toHaveCount(0);
  await expect(button(page, "Use This State")).toBeDisabled();
  await page
    .getByRole("combobox", { name: labels[2], exact: true })
    .selectOption("empty");
  for (const label of ["점 찍기", "바꿔 뽑기"])
    await number(page, `${label} 남은 횟수`).fill("0");
  await page
    .getByRole("checkbox", {
      name: "보드의 아이템을 모두 확인했습니다 (없으면 그대로 체크)",
      exact: true,
    })
    .check();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await expect(button(page, "Use This State")).toBeEnabled();
  await number(page, "Reroll 보유 수").fill("1");
  await expect(button(page, "Use This State")).toBeDisabled();
  await expect(
    page.getByRole("group", { name: "인식 결과 검토", exact: true }),
  ).toContainText("검토 중 수동 상태가 바뀌었습니다");
  await button(page, "현재 수동 입력으로 Mock 검토").click();
  await expect(
    page.getByLabel("검토할 게임 보드 캡처", { exact: true }),
  ).toHaveCount(0);
  await button(page, "Stop Capture").click();
});

test("panel compressed narrow digits stay unresolved rather than guessed as zero", async ({
  page,
}) => {
  await page.evaluate(() => window.__calibration.automaticMode("panel-narrow"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 432 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  for (const label of ["점 찍기", "바꿔 뽑기"])
    await expect(number(page, `${label} 남은 횟수`)).toHaveValue("");
  await expect(
    page.getByRole("combobox", { name: "첫 번째 보유 조각", exact: true }),
  ).toHaveValue("C_5");
  await expect(button(page, "Use This State")).toBeDisabled();
  await button(page, "검토 버리기").click();
  await button(page, "Stop Capture").click();
});

test("coordinate calibration errors recover into review and existing Analyze/Apply", async ({
  page,
}) => {
  await capture(page);
  await expect(button(page, "선택한 영역의 보드 인식")).toBeDisabled();
  await region(page);
  await color(page, "빈칸", -1, 200);
  await expect(
    page.getByRole("alert", { name: "보드 인식 설정 안내", exact: true }),
  ).toContainText("표본 좌표");
  await color(page, "빈칸", 200, 200);
  await color(page, "빈칸", 200, 200);
  await expect(group(page)).toContainText("같은 색상");
  await color(page, "점유", 120, 120);
  await color(page, "점유", 200, 120);
  await expect(group(page)).toContainText("점유 표본 2개");
  await number(page, "보드 영역 x").fill("500");
  await expect(button(page, "선택한 영역의 보드 인식")).toBeDisabled();
  await number(page, "보드 영역 x").fill("80");
  await group(page).getByText("판별 설정 조정", { exact: true }).click();
  await number(page, "색상 허용 거리").fill("-1");
  await button(page, "선택한 영역의 보드 인식").click();
  await expect(group(page)).toContainText("판별 설정을 확인");
  await number(page, "색상 허용 거리").fill("12");
  await button(page, "선택한 영역의 보드 인식").click();
  await cleared(page);
  await expect(
    page.getByRole("group", { name: "인식 결과 검토", exact: true }),
  ).toContainText("지정 영역·색상 표본");
  await expect(
    page.locator('button[data-review-row="0"][data-review-col="1"]'),
  ).toHaveAttribute("aria-label", /점유/);
  await expect(
    page.locator('button[data-review-row="1"][data-review-col="0"]'),
  ).toHaveText("?");
  await fillReview(page);
  await expect(button(page, "Use This State")).toBeDisabled();
  await page
    .locator('button[data-review-row="1"][data-review-col="0"]')
    .click();
  await button(page, "빈칸으로 표시").click();
  await expect(
    page.getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true }),
  ).not.toBeChecked();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("9 / 160");
  await button(page, "Analyze").click();
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Apply Step").click();
  await expect(
    page.getByRole("region", { name: "입력 및 적용 상태", exact: true }),
  ).toContainText("단계 1");
});

test("frame ownership clears on cancellation, replacement, stop, ended, manual edit, mock and unmount", async ({
  page,
}) => {
  await capture(page);
  await button(page, "프레임 선택 취소").click();
  await cleared(page);
  await capture(page);
  await region(page);
  await color(page, "빈칸", 200, 200);
  await capture(page);
  await expect(number(page, "보드 영역 x")).toHaveValue("");
  await expect(group(page)).toContainText("빈칸 표본 0개");
  expect(
    (await page.evaluate(() => window.__calibration.summary())).buffers
      .slice(0, -1)
      .every(Boolean),
  ).toBe(true);
  await button(page, "Stop Capture").click();
  await cleared(page);
  await capture(page);
  await page.evaluate(() => window.__calibration.end());
  await cleared(page);
  await capture(page);
  await number(page, "Reroll 보유 수").fill("1");
  await cleared(page);
  await expect(
    page.getByRole("status", { name: "프레임 상태", exact: true }),
  ).toContainText("수동 상태가 바뀌어");
  await number(page, "Reroll 보유 수").fill("0");
  await expect(group(page)).toHaveCount(0);
  await capture(page);
  await button(page, "현재 수동 입력으로 Mock 검토").click();
  await cleared(page);
  await capture(page);
  await button(page, "화면 입력 닫기").click();
  await cleared(page);
});

test("scaled preview pointer selection feeds original pixels and explicit grid", async ({
  page,
}) => {
  await capture(page);
  await point(page, 80, 80);
  await point(page, 879, 1359);
  for (const [key, expected] of Object.entries({
    x: 80,
    y: 80,
    width: 800,
    height: 1280,
  }))
    expect(
      Math.abs(
        Number(await number(page, `보드 영역 ${key}`).inputValue()) - expected,
      ),
    ).toBeLessThanOrEqual(4);
  await region(page); // exact numeric alternative compensates for pointer subpixel rounding.
  await button(page, "빈칸 색상 선택").click();
  await point(page, 200, 200);
  await button(page, "점유 색상 선택").click();
  await point(page, 120, 120);
  await point(page, 200, 120);
  await expect(group(page)).toContainText("점유 표본 2개");
  await group(page).scrollIntoViewIfNeeded();
  await page.screenshot({
    path: test.info().outputPath("synthetic-calibration.png"),
  });
  await button(page, "선택한 영역의 보드 인식").click();
  await cleared(page);
  await expect(
    page.locator('button[data-review-row="0"][data-review-col="8"]'),
  ).toHaveText("●");
  await expect(
    page.locator('button[data-review-row="0"][data-review-col="9"]'),
  ).toHaveText("·");
  await expect(button(page, "Use This State")).toBeDisabled();
});

test("safe capture failures release prior selection and recover without native text", async ({
  page,
}) => {
  const frameStatus = page.getByRole("status", {
    name: "프레임 상태",
    exact: true,
  });
  for (const failure of ["unready", "oversized", "context", "pixels"]) {
    await capture(page);
    await page.evaluate((mode) => {
      const video = document.querySelector("video");
      if (mode === "unready")
        Object.defineProperty(video, "readyState", {
          configurable: true,
          value: 0,
        });
      if (mode === "oversized")
        for (const key of ["videoWidth", "videoHeight"])
          Object.defineProperty(video, key, {
            configurable: true,
            value: 8192,
          });
      window.__calibration.canvasFailure(mode);
    }, failure);
    await button(page, "수동 영역·색상 선택").click();
    await cleared(page);
    await expect(frameStatus).toContainText(
      failure === "oversized"
        ? "더 작은 창을 공유하세요"
        : failure === "unready"
          ? "미리보기 재생 후"
          : failure === "context"
            ? "Canvas를 사용할 수 없습니다"
            : "공유 화면과 재생 상태",
    );
    await expect(page.locator("body")).not.toContainText("private-window-info");
    await expect(page.locator("body")).not.toContainText("SecurityError");
    await expect(
      page.getByRole("status", { name: "점유 칸 수", exact: true }),
    ).toHaveText("0 / 160");
    await page.evaluate(() => {
      const video = document.querySelector("video");
      for (const key of ["readyState", "videoWidth", "videoHeight"])
        delete video[key];
      window.__calibration.canvasFailure("");
    });
  }
  await capture(page);
  await expect(
    page.getByRole("group", { name: "인식 결과 검토", exact: true }),
  ).toContainText("영역·색상 선택 전");
  await region(page);
  await color(page, "빈칸", 200, 200);
  await color(page, "점유", 120, 120);
  await color(page, "점유", 200, 120);
  await button(page, "선택한 영역의 보드 인식").click();
  await cleared(page);
  await expect(frameStatus).toContainText("보드 판별 완료");
});

test("automatic Capture Frame needs no calibration and review still gates Analyze/Apply", async ({
  page,
}) => {
  await page.evaluate(() => window.__calibration.automaticMode("occluded"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 300 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  await cleared(page);
  const review = page.getByRole("group", {
    name: "인식 결과 검토",
    exact: true,
  });
  await expect(review).toContainText("자동으로 찾은 보드");
  await expect(
    review.locator("button[data-review-row]").filter({ hasText: "●" }),
  ).toHaveCount(49);
  const unknown = page.locator(
    'button[data-review-row="1"][data-review-col="0"]',
  );
  await expect(unknown).toHaveText("!");
  await expect(button(page, "Use This State")).toBeDisabled();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("0 / 160");
  await fillReview(page);
  await expect(button(page, "Use This State")).toBeDisabled();
  await unknown.click();
  await button(page, "빈칸으로 표시").click();
  await expect(
    page.getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true }),
  ).not.toBeChecked();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("49 / 160");
  await button(page, "Analyze").click();
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Apply Step").click();
  await expect(
    page.getByRole("region", { name: "입력 및 적용 상태", exact: true }),
  ).toContainText("단계 1");
  await button(page, "Capture Frame").click();
  await number(page, "Reroll 보유 수").fill("1");
  await expect(review).toContainText("검토 중 수동 상태가 바뀌었습니다");
  await expect(button(page, "Use This State")).toBeDisabled();
  await button(page, "Stop Capture").click();
  await cleared(page);
});

test("automatic missing board releases pixels and explicit manual fallback recovers", async ({
  page,
}) => {
  await page.evaluate(() => window.__calibration.automaticMode("missing"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 300 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  await cleared(page);
  await expect(
    page.getByRole("status", { name: "프레임 상태", exact: true }),
  ).toContainText("보드를 확정하지 못했습니다");
  await expect(button(page, "Use This State")).toBeDisabled();
  await button(page, "Stop Capture").click();
  await page.evaluate(() => window.__calibration.automaticMode(""));
  await capture(page);
  await region(page);
  await color(page, "빈칸", 200, 200);
  await color(page, "점유", 120, 120);
  await color(page, "점유", 200, 120);
  await button(page, "선택한 영역의 보드 인식").click();
  await cleared(page);
  await expect(
    page.getByRole("group", { name: "인식 결과 검토", exact: true }),
  ).toContainText("지정 영역·색상 표본");
});

test("visual unresolved cells label in place with keyboard, crop alignment and lifecycle", async ({
  page,
}) => {
  const preview = page.getByLabel("검토할 게임 보드 캡처", { exact: true });
  const summary = () => page.evaluate(() => window.__calibration.summary());
  const start = async () => {
    if (await button(page, "Start Screen Capture").isEnabled()) {
      await button(page, "Start Screen Capture").click();
      await page.waitForFunction(
        () =>
          document.querySelector("video").videoWidth === 300 &&
          document.querySelector("video").readyState >= 2,
      );
    }
    await button(page, "Capture Frame").click();
    await expect(preview).toBeVisible();
  };
  await page.evaluate(() => window.__calibration.automaticMode("occluded"));
  await start();
  await expect(
    page.locator('button[data-review-unresolved="true"]'),
  ).toHaveCount(1);
  const unknown = page.locator(
    'button[data-review-row="1"][data-review-col="0"]',
  );
  await expect(unknown).toHaveCSS("border-top-width", "3px");
  await expect(
    page.getByRole("status", { name: "미확정 칸 수", exact: true }),
  ).toContainText("1개");
  expect((await summary()).previews).toEqual([false]);
  expect((await summary()).buffers.every(Boolean)).toBe(true);
  expect((await summary()).scratch.every(Boolean)).toBe(true);
  const imageRect = await preview.boundingBox(),
    cellRect = await unknown.boundingBox();
  expect(Math.abs(cellRect.x - imageRect.x)).toBeLessThan(1);
  expect(
    Math.abs(cellRect.y - imageRect.y - imageRect.height / 16),
  ).toBeLessThan(1);
  await button(page, "다음 미확정 칸").click();
  const editor = page.getByRole("group", {
    name: "선택한 칸 레이블링",
    exact: true,
  });
  await expect(editor).toBeVisible();
  await expect(unknown).toHaveText("!"); // selecting never labels silently
  await expect(unknown).toHaveAttribute("aria-pressed", "true");
  const editRect = await editor.boundingBox();
  expect(editRect.x).toBeGreaterThanOrEqual(imageRect.x - 1);
  expect(editRect.x + editRect.width).toBeLessThanOrEqual(
    imageRect.x + imageRect.width + 1,
  );
  await page.locator(".review-board").screenshot({
    path: test.info().outputPath("synthetic-cell-labeling.png"),
  });
  await button(page, "빈칸으로 표시").press("Enter");
  await expect(unknown).toHaveText("·");
  await expect(button(page, "다음 미확정 칸")).toBeDisabled();
  await button(page, "점유로 표시").click();
  await expect(unknown).toHaveText("●");
  await button(page, "미확정으로 되돌리기").click();
  await expect(unknown).toHaveText("?");
  await button(page, "빈칸으로 표시").click();
  await button(page, "빈칸으로 표시").press("Escape");
  await expect(editor).toHaveCount(0);
  await expect(unknown).toBeFocused();
  await unknown.press("Space");
  await expect(editor).toBeVisible();
  await button(page, "닫기").click();
  await button(page, "점유 표시 보기").click();
  await expect(preview).toBeHidden();
  await button(page, "게임 캡처 보기").click();
  await expect(preview).toBeVisible();
  await page.screenshot({
    path: test.info().outputPath("synthetic-visual-review.png"),
  });
  for (const [row, col] of [
    [0, 9],
    [15, 0],
    [15, 9],
  ]) {
    await page
      .locator(`button[data-review-row="${row}"][data-review-col="${col}"]`)
      .click();
    const a = await editor.boundingBox(),
      b = await preview.boundingBox();
    expect(a.x).toBeGreaterThanOrEqual(b.x - 1);
    expect(a.x + a.width).toBeLessThanOrEqual(b.x + b.width + 1);
    expect(a.y).toBeGreaterThanOrEqual(b.y - 1);
    expect(a.y + a.height).toBeLessThanOrEqual(b.y + b.height + 1);
    await button(page, "닫기").click();
  }
  await button(page, "검토 버리기").click();
  expect((await summary()).previews.every(Boolean)).toBe(true);
  await start();
  await start(); // replacement disposes only the previous crop
  expect((await summary()).previews.slice(0, -1).every(Boolean)).toBe(true);
  await button(page, "현재 수동 입력으로 Mock 검토").click();
  expect((await summary()).previews.every(Boolean)).toBe(true);
  await start();
  await number(page, "Reroll 보유 수").fill("1");
  await expect(preview).toHaveCount(0);
  expect((await summary()).previews.every(Boolean)).toBe(true);
  await number(page, "Reroll 보유 수").fill("0");
  await start();
  await button(page, "Stop Capture").click();
  expect((await summary()).previews.every(Boolean)).toBe(true);
  await start();
  await page.evaluate(() => window.__calibration.end());
  await expect(preview).toHaveCount(0);
  expect((await summary()).previews.every(Boolean)).toBe(true);
  await start();
  await button(page, "화면 입력 닫기").click();
  expect((await summary()).previews.every(Boolean)).toBe(true);
});

test("visual game labels and board item marking preserve occupancy and confirmation boundaries", async ({
  page,
}) => {
  const itemCheck = page.getByRole("checkbox", {
    name: "보드의 아이템을 모두 확인했습니다 (없으면 그대로 체크)",
    exact: true,
  });
  const allCheck = page.getByRole("checkbox", {
    name: "검토한 전체 상태 확인",
    exact: true,
  });
  const cell = (row, col) =>
    page.locator(`button[data-review-row="${row}"][data-review-col="${col}"]`);
  const editor = page.getByRole("group", {
    name: "선택한 칸 레이블링",
    exact: true,
  });
  const items = page.getByRole("region", {
    name: "보드 아이템 확인",
    exact: true,
  });
  const assertMenuBounds = async () => {
    const menu = await editor.boundingBox(),
      board = await page.locator(".review-board").boundingBox();
    expect(menu.x).toBeGreaterThanOrEqual(board.x - 1);
    expect(menu.y).toBeGreaterThanOrEqual(board.y - 1);
    expect(menu.x + menu.width).toBeLessThanOrEqual(board.x + board.width + 1);
    expect(menu.y + menu.height).toBeLessThanOrEqual(
      board.y + board.height + 1,
    );
  };
  await page.evaluate(() => window.__calibration.automaticMode("items"));
  await button(page, "Start Screen Capture").click();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 300 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  await expect(cell(12, 7)).toHaveAttribute("data-review-unresolved", "true");
  await expect(items).toContainText("표시한 아이템 0개 (보드에 최대 3개)");
  await expect(items).toContainText(
    "게임 보드에도 아이콘이 없다면 추가하지 말고 아래 확인만 체크",
  );
  await expect(itemCheck).not.toBeChecked();
  await expect(
    page.getByRole("combobox", { name: "첫 번째 보유 조각", exact: true }),
  ).toHaveValue("unknown");
  await expect(number(page, "점 찍기 남은 횟수")).toHaveValue("");
  await expect(
    page.getByRole("group", { name: "인식 결과 검토", exact: true }),
  ).toContainText("게임 버튼 옆 남은 횟수");
  await fillReview(page);
  await cell(12, 7).click();
  await button(page, "점 찍기 아이템").click();
  await expect(cell(12, 7)).toHaveAttribute("data-review-unresolved", "true");
  await expect(cell(12, 7)).toHaveAttribute(
    "aria-label",
    /미확정 · 점 찍기 아이템/,
  );
  await expect(cell(12, 7).locator("[data-review-item]")).toHaveText("점");
  await expect(itemCheck).not.toBeChecked();
  await expect(allCheck).not.toBeChecked();
  await expect(number(page, "점 찍기 남은 횟수")).toHaveValue("0");
  await expect(items).toContainText(
    "점 찍기 아이템 · 위에서 13번째 줄, 왼쪽에서 8번째 칸",
  );
  await assertMenuBounds();
  await page
    .locator(".review-board")
    .screenshot({ path: test.info().outputPath("synthetic-item-review.png") });
  await button(page, "점유 표시 보기").click();
  await expect(cell(12, 7).locator("[data-review-item]")).toBeVisible();
  await button(page, "게임 캡처 보기").click();
  await button(page, "빈칸으로 표시").click();
  await button(page, "닫기").click();
  await cell(1, 0).click();
  await button(page, "빈칸으로 표시").click();
  await button(page, "닫기").click();
  for (const [row, col] of [
    [0, 9],
    [15, 0],
  ]) {
    await cell(row, col).click();
    await button(page, "바꿔 뽑기 아이템").click();
    await assertMenuBounds();
    await button(page, "닫기").click();
  }
  await itemCheck.check();
  await allCheck.check();
  await expect(button(page, "Use This State")).toBeEnabled();
  await cell(10, 9).click();
  await button(page, "점 찍기 아이템").click();
  await expect(editor.getByRole("alert")).toContainText("최대 3개");
  await expect(items).toContainText("표시한 아이템 3개");
  await expect(cell(10, 9).locator("[data-review-item]")).toHaveCount(0);
  await expect(itemCheck).toBeChecked();
  await expect(allCheck).toBeChecked();
  await expect(button(page, "Use This State")).toBeEnabled();
  await assertMenuBounds();
  await button(page, "닫기").click();
  await cell(12, 7).click();
  await button(page, "바꿔 뽑기 아이템").click(); // replacement is permitted at capacity
  await expect(cell(12, 7).locator("[data-review-item]")).toHaveText("뽑");
  await expect(itemCheck).not.toBeChecked();
  await button(page, "점 찍기 아이템").click();
  await button(page, "닫기").click();
  for (const [row, col] of [
    [0, 9],
    [15, 0],
  ]) {
    await cell(row, col).click();
    await button(page, "이 칸에 아이템 없음").click();
    await expect(cell(row, col).locator("[data-review-item]")).toHaveCount(0);
    await button(page, "닫기").click();
  }
  await expect(items).toContainText("표시한 아이템 1개");
  await page
    .getByText("첫 번째 보유 조각 모양으로 고르기", { exact: true })
    .click();
  await button(page, "첫 번째 보유 조각 ㄷ 선택").click();
  await expect(
    page
      .getByLabel("첫 번째 보유 조각 선택 모양", { exact: true })
      .locator(".shape-filled"),
  ).toHaveCount(5);
  await page
    .getByText("첫 번째 보유 조각 모양으로 고르기", { exact: true })
    .click();
  for (const label of ["두 번째 보유 조각", "세 번째 보유 조각"])
    await page
      .getByRole("combobox", { name: label, exact: true })
      .selectOption("empty");
  await number(page, "점 찍기 남은 횟수").fill("");
  await itemCheck.check();
  await allCheck.check();
  await expect(button(page, "Use This State")).toBeDisabled();
  await expect(
    page.getByRole("alert", { name: "검토 오류", exact: true }),
  ).toContainText("점 찍기");
  await number(page, "점 찍기 남은 횟수").fill("0");
  await allCheck.check();
  await items.screenshot({
    path: test.info().outputPath("synthetic-item-instructions.png"),
  });
  await button(page, "Use This State").click();
  await expect(
    page.locator('button[data-row="12"][data-col="7"]'),
  ).toHaveAttribute("aria-label", /hidden single-cell/);
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("49 / 160");
  await expect(number(page, "Single Cell 보유 수")).toHaveValue("0");
  await expect(number(page, "Reroll 보유 수")).toHaveValue("0");
  await expect(
    page.getByRole("combobox", { name: "slot 0 블록", exact: true }),
  ).toHaveValue("C_5");
  for (const slot of [1, 2])
    await expect(
      page.getByRole("combobox", { name: `slot ${slot} 블록`, exact: true }),
    ).toHaveValue("");
  await button(page, "Analyze").click();
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Apply Step").click();
  await button(page, "Stop Capture").click();
});
