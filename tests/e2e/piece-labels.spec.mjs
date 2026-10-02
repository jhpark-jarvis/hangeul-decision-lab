import { readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

const buildId = readFileSync(".next/BUILD_ID", "utf8").trim();
test("user label page requires explicit unique choices, supports completion/edit and clipboard recovery", async ({
  page,
  context,
  browser,
}, info) => {
  const record = {
    buildId,
    measuredAt: new Date().toISOString(),
    browser: info.project.name,
    version: browser.version(),
    source:
      "Arbitrary synthetic letter permutation; not a real shape/name mapping",
    requests: [],
    errors: [],
    sockets: [],
  };
  context.on("request", (r) =>
    record.requests.push({
      url: r.url(),
      method: r.method(),
      body: r.postData(),
      type: r.resourceType(),
    }),
  );
  page.on("pageerror", (e) => record.errors.push(String(e)));
  page.on("websocket", (s) => record.sockets.push(s.url()));
  page.on("console", (m) => {
    if (["warning", "error"].includes(m.type())) record.errors.push(m.text());
  });
  await context.route("**/*", (route) =>
    new URL(route.request().url()).origin === "http://127.0.0.1:45001"
      ? route.continue()
      : route.abort("blockedbyclient"),
  );
  await page.addInitScript(() => {
    window.__labelClipboard = { calls: 0, text: "", fail: true };
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text) => {
          window.__labelClipboard.calls++;
          if (window.__labelClipboard.fail) throw new Error("test denied");
          window.__labelClipboard.text = text;
        },
      },
    });
  });
  try {
    await page.goto("/piece-labels");
    const choose = (n) =>
      page.getByRole("combobox", { name: `도형 ${n} 한글 이름`, exact: true });
    const finish = page.getByRole("button", { name: "매칭 완료", exact: true });
    const result = page.getByRole("textbox", {
      name: "한글 도형 매칭 JSON",
      exact: true,
    });
    await expect(page.getByRole("article")).toHaveCount(19);
    await expect(finish).toBeDisabled();
    for (let n = 1; n <= 19; n++) await expect(choose(n)).toHaveValue("");
    await choose(1).selectOption("점");
    await expect(choose(2).locator('option[value="점"]')).toHaveAttribute(
      "disabled",
      "",
    );
    await expect(choose(1).locator('option[value="ㅣ"]')).toHaveAttribute(
      "disabled",
      "",
    );
    // Synthetic names intentionally don't claim the semantic glyph of each shape.
    const remaining = [
      "ㄱ",
      "ㄴ",
      "ㄷ",
      "ㄹ",
      "ㅁ",
      "ㅂ",
      "ㅅ",
      "ㅇ",
      "ㅈ",
      "ㅊ",
      "ㅋ",
      "ㅌ",
      "ㅍ",
      "ㅎ",
      "ㅏ",
      "ㅑ",
    ];
    for (let n = 2; n <= 19; n++)
      await choose(n).selectOption(
        n === 2 ? "ㅡ" : n === 10 ? "ㅣ" : remaining.shift(),
      );
    await expect(
      page.getByRole("status", { name: "이름 매칭 진행", exact: true }),
    ).toContainText("19 / 19");
    await expect(finish).toBeEnabled();
    await page.screenshot({
      path: info.outputPath("synthetic-label-page.png"),
    });
    await finish.press("Enter");
    const data = JSON.parse(await result.inputValue());
    expect(data.mapping).toHaveLength(19);
    expect(data.mapping[0]).toEqual({ pieceId: "DOT", label: "점", cells: 1 });
    for (let n = 1; n <= 19; n++) await expect(choose(n)).toBeDisabled();
    expect(await page.evaluate(() => window.__labelClipboard.calls)).toBe(0);
    await page.getByRole("button", { name: "결과 복사", exact: true }).click();
    await expect(
      page.getByRole("status", { name: "매칭 결과 안내", exact: true }),
    ).toContainText("Ctrl+C");
    await page.evaluate(() => {
      window.__labelClipboard.fail = false;
    });
    await page.getByRole("button", { name: "결과 복사", exact: true }).click();
    expect(
      JSON.parse(await page.evaluate(() => window.__labelClipboard.text)),
    ).toEqual(data);
    await page.getByRole("button", { name: "매칭 수정", exact: true }).click();
    await expect(result).toHaveCount(0);
    await choose(19).selectOption("");
    await expect(finish).toBeDisabled();
    await choose(19).selectOption(data.mapping[18].label);
    await finish.click();
    expect(JSON.parse(await result.inputValue())).toEqual(data);
    await page.reload();
    await expect(choose(1)).toHaveValue("");
    await expect(finish).toBeDisabled();
    record.result = "PASS";
  } finally {
    record.storage = await page.evaluate(async () => ({
      local: Object.keys(localStorage),
      session: Object.keys(sessionStorage),
      indexedDB: await indexedDB.databases(),
      caches: await caches.keys(),
    }));
    record.cookies = (await context.cookies()).map((cookie) => cookie.name);
    writeFileSync(
      info.outputPath("labels.json"),
      JSON.stringify(record, null, 2) + "\n",
    );
  }
  expect(record.errors).toEqual([]);
  expect(record.sockets).toEqual([]);
  expect(record.cookies).toEqual([]);
  expect(record.storage).toEqual({
    local: [],
    session: [],
    indexedDB: [],
    caches: [],
  });
  expect(
    record.requests.filter(
      (r) =>
        !r.url.startsWith("http://127.0.0.1:45001/") ||
        r.method !== "GET" ||
        r.body !== null ||
        ["fetch", "xhr", "eventsource"].includes(r.type),
    ),
  ).toEqual([]);
});
