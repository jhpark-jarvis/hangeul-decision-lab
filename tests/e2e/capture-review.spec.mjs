import { readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

const builds = readFileSync(".next/BUILD_ID", "utf8").trim();
const traffic = new Map();
const button = (page, name) => page.getByRole("button", { name, exact: true });
const input = (page, name) =>
  page.getByRole("spinbutton", { name, exact: true });
const select = (page, name) =>
  page.getByRole("combobox", { name, exact: true });
const review = (page) =>
  page.getByRole("group", { name: "인식 결과 검토", exact: true });
async function seed(page) {
  await page.goto("/");
  for (let slot = 0; slot < 3; slot++)
    await select(page, `slot ${slot} 블록`).selectOption("DOT");
  await button(page, "3개 블록 입력").click();
  await button(page, "Filled").click();
  for (let col = 0; col < 9; col++)
    await page.locator(`button[data-row="0"][data-col="${col}"]`).click();
}
async function analyze(page) {
  await button(page, "Analyze").click();
  await expect(
    page.getByRole("heading", { name: "추천 결과", exact: true }),
  ).toBeVisible();
}
async function openMock(page) {
  await button(page, "화면 캡처·검토 열기").click();
  await button(page, "현재 수동 입력으로 Mock 검토").click();
}
test.beforeEach(async ({ page, context }, testInfo) => {
  const entry = {
    browser: testInfo.project.name,
    fixture: testInfo.title,
    buildId: builds,
    measuredAt: new Date().toISOString(),
    source:
      "Synthetic canvas stream + manual fixture; no actual display chooser or desktop data",
    requests: [],
    errors: [],
    sockets: [],
  };
  traffic.set(testInfo.testId, entry);
  context.on("request", (request) =>
    entry.requests.push({
      url: request.url(),
      method: request.method(),
      body: request.postData(),
      type: request.resourceType(),
    }),
  );
  page.on("pageerror", (error) => entry.errors.push(String(error)));
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type()))
      entry.errors.push(`${message.type()}: ${message.text()}`);
  });
  page.on("websocket", (socket) => entry.sockets.push(socket.url()));
  await context.route("**/*", (route) =>
    new URL(route.request().url()).origin === "http://127.0.0.1:45001"
      ? route.continue()
      : route.abort("blockedbyclient"),
  );
});
test.afterEach(async ({ page, context, browser }, testInfo) => {
  const entry = traffic.get(testInfo.testId);
  entry.version = browser.version();
  entry.storage = await page.evaluate(async () => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
    indexedDB: (await indexedDB.databases()).map((db) => db.name),
    caches: await caches.keys(),
  }));
  entry.cookies = (await context.cookies()).map((cookie) => cookie.name);
  entry.captureFixture = await page.evaluate(
    () => window.__captureFixture?.summary() ?? null,
  );
  entry.result = testInfo.status;
  writeFileSync(
    testInfo.outputPath("phase2.json"),
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

test("review correction, unknown rejection and existing Analyze/Apply", async ({
  page,
}) => {
  await seed(page);
  await analyze(page);
  await openMock(page);
  await expect(button(page, "Apply Step")).toHaveCount(0);
  await expect(button(page, "Use This State")).toBeDisabled();
  const cell = page.locator('button[data-review-row="0"][data-review-col="0"]');
  await cell.click(); // occupied -> unknown
  await expect(cell).toHaveText("?");
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await expect(button(page, "Use This State")).toBeDisabled();
  await cell.click(); // unknown -> empty; resets confirmation
  await select(page, "검토 slot 2 블록").selectOption("C_5");
  await input(page, "검토 reroll 보유 수").fill("6");
  await input(page, "검토 singleCell 보유 수").fill("2");
  await expect(
    page.getByRole("alert", { name: "검토 오류", exact: true }),
  ).toContainText("능력 보유 합계는 7");
  await input(page, "검토 singleCell 보유 수").fill("1");
  await select(page, "검토 slot 1 블록").selectOption("unknown");
  await expect(button(page, "Use This State")).toBeDisabled();
  await select(page, "검토 slot 1 블록").selectOption("DOT");
  await button(page, "검토 아이템 추가").click();
  await select(page, "검토 아이템 0 종류").selectOption("reroll");
  await input(page, "검토 아이템 0 row").fill("16");
  await input(page, "검토 아이템 0 col").fill("0");
  await expect(input(page, "검토 아이템 0 row")).toHaveAttribute(
    "aria-invalid",
    "true",
  );
  await input(page, "검토 아이템 0 row").fill("0");
  await expect(
    review(page).getByRole("checkbox", {
      name: "검토한 전체 상태 확인",
      exact: true,
    }),
  ).not.toBeChecked();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await expect(button(page, "Use This State")).toBeEnabled();
  await page.screenshot({
    path: test.info().outputPath("synthetic-review.png"),
  });
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("8 / 160");
  await expect(select(page, "slot 2 블록")).toHaveValue("C_5");
  await expect(input(page, "Reroll 보유 수")).toHaveValue("6");
  await expect(input(page, "Single Cell 보유 수")).toHaveValue("1");
  await expect(
    page.locator('button[data-row="0"][data-col="0"]'),
  ).toHaveAttribute("aria-label", /hidden reroll/);
  await analyze(page);
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Apply Step").click();
  await expect(
    page.getByRole("region", { name: "입력 및 적용 상태", exact: true }),
  ).toContainText("단계 1");
});

test("stale review preserves manual input and can be replaced", async ({
  page,
}) => {
  await seed(page);
  await openMock(page);
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await button(page, "Empty").click();
  await page.locator('button[data-row="0"][data-col="0"]').click();
  await expect(review(page)).toContainText("검토 중 수동 상태가 바뀌었습니다");
  await expect(button(page, "Use This State")).toBeDisabled();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("8 / 160");
  await button(page, "현재 수동 입력으로 Mock 검토").click();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("8 / 160");
});

test("capture cancel, permission, late stream, preview/frame, ended and unmount", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const tracks = [];
    const calls = [];
    let mode = "cancel";
    let release;
    function makeStream() {
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 32;
      const context = canvas.getContext("2d");
      context.fillStyle = "#13579b";
      context.fillRect(0, 0, 64, 32);
      const stream = canvas.captureStream(1);
      tracks.push(...stream.getTracks());
      return stream;
    }
    Object.defineProperty(navigator.mediaDevices, "getDisplayMedia", {
      configurable: true,
      value: (options) => {
        calls.push(options);
        if (mode === "cancel")
          return Promise.reject(new DOMException("", "AbortError"));
        if (mode === "permission")
          return Promise.reject(new DOMException("", "NotAllowedError"));
        if (mode === "pending")
          return new Promise((resolve) => {
            release = () => resolve(makeStream());
          });
        return Promise.resolve(makeStream());
      },
    });
    window.__captureFixture = {
      mode: (value) => {
        mode = value;
      },
      release: () => release(),
      end: () => tracks.at(-1).dispatchEvent(new Event("ended")),
      summary: () => ({
        calls,
        tracks: tracks.map((track) => track.readyState),
      }),
    };
  });
  await seed(page);
  await analyze(page);
  await button(page, "화면 캡처·검토 열기").click();
  await button(page, "Start Screen Capture").click();
  await expect(
    page.getByRole("status", { name: "캡처 상태", exact: true }),
  ).toContainText("취소했거나");
  await expect(button(page, "Apply Step")).toHaveCount(0);
  await page.evaluate(() => window.__captureFixture.mode("permission"));
  await button(page, "Start Screen Capture").click();
  await expect(
    page.getByRole("status", { name: "캡처 상태", exact: true }),
  ).toContainText("권한이 거절");
  await page.evaluate(() => window.__captureFixture.mode("pending"));
  await button(page, "Start Screen Capture").click();
  await expect(button(page, "Stop Capture")).toBeEnabled();
  await button(page, "Stop Capture").click();
  await page.evaluate(() => window.__captureFixture.release());
  await page.waitForFunction(() =>
    window.__captureFixture
      .summary()
      .tracks.every((state) => state === "ended"),
  );
  await page.evaluate(() => window.__captureFixture.mode("success"));
  await button(page, "Start Screen Capture").click();
  await expect(
    page.getByLabel("공유 화면 미리보기", { exact: true }),
  ).toBeVisible();
  await page.waitForFunction(
    () =>
      document.querySelector("video").videoWidth === 64 &&
      document.querySelector("video").readyState >= 2,
  );
  await button(page, "Capture Frame").click();
  await expect(
    page.getByRole("status", { name: "프레임 상태", exact: true }),
  ).toContainText("64×32");
  await expect(review(page).locator("button[data-review-row]")).toHaveCount(
    160,
  );
  await expect(review(page)).toContainText("모든 값은 미확정");
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await expect(button(page, "Use This State")).toBeDisabled();
  // Resolve the captured stub explicitly; a frame must never silently become
  // an empty board or a ready-to-use game state.
  await button(page, "미확정 보드 칸을 빈칸으로 확인").click();
  for (let slot = 0; slot < 3; slot++)
    await select(page, `검토 slot ${slot} 블록`).selectOption("DOT");
  await input(page, "검토 reroll 보유 수").fill("0");
  await input(page, "검토 singleCell 보유 수").fill("0");
  await review(page)
    .getByRole("checkbox", {
      name: "아이템 목록 전체 확인 (없음 포함)",
      exact: true,
    })
    .check();
  await page
    .getByRole("checkbox", { name: "검토한 전체 상태 확인", exact: true })
    .check();
  await expect(button(page, "Use This State")).toBeEnabled();
  await button(page, "Use This State").click();
  await expect(
    page.getByRole("status", { name: "점유 칸 수", exact: true }),
  ).toHaveText("0 / 160");
  await analyze(page);
  await expect(button(page, "Apply Step")).toBeEnabled();
  await button(page, "Stop Capture").click();
  await expect(
    page.getByLabel("공유 화면 미리보기", { exact: true }),
  ).toBeHidden();
  await page.waitForFunction(
    () =>
      document.querySelector("video").srcObject === null &&
      window.__captureFixture
        .summary()
        .tracks.every((state) => state === "ended"),
  );
  await button(page, "Start Screen Capture").click();
  await page.evaluate(() => window.__captureFixture.end());
  await expect(button(page, "Start Screen Capture")).toBeEnabled();
  await button(page, "Start Screen Capture").click();
  await expect(button(page, "Stop Capture")).toBeEnabled();
  await button(page, "화면 입력 닫기").click();
  await page.waitForFunction(() =>
    window.__captureFixture
      .summary()
      .tracks.every((state) => state === "ended"),
  );
});
