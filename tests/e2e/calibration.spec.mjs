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
  await button(page, "Capture Frame").click();
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
      .getByRole("combobox", { name: `검토 slot ${slot} 블록`, exact: true })
      .selectOption("DOT");
  for (const kind of ["reroll", "singleCell"])
    await number(page, `검토 ${kind} 보유 수`).fill("0");
  await page
    .getByRole("checkbox", {
      name: "아이템 목록 전체 확인 (없음 포함)",
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
    source:
      "Synthetic 960x1440 Canvas stream / arbitrary colors / 10x16 board. Native picker and actual desktop never used.",
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
    const getImageData = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function (...args) {
      const image = getImageData.apply(this, args);
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
      rememberCanvas: (element) => canvases.push(element),
      end: () => tracks.at(-1).dispatchEvent(new Event("ended")),
      summary: () => ({
        buffers: buffers.map((data) => data.every((v) => v === 0)),
        scratch: scratch.map((data) => data.every((v) => v === 0)),
        canvases: canvases.map((c) => c.width === 0 && c.height === 0),
        tracks: tracks.map((t) => t.readyState),
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
