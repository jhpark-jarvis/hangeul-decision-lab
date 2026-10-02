import { readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";

const buildId = readFileSync(".next/BUILD_ID", "utf8").trim();
test("restored ㄹ and ㅌ render five complete rows and retain user labels", async ({
  page,
}, info) => {
  const mapping = JSON.parse(
    readFileSync("tests/fixtures/pieces/labels.json", "utf8"),
  ).mapping;
  await page.goto("/piece-labels");
  const shapes = [];
  for (const [n, label, rows] of [
    [18, "ㄹ", "1101111011"],
    [19, "ㅌ", "1110111011"],
  ]) {
    const card = page.getByRole("article", { name: `도형 ${n}`, exact: true });
    await card.scrollIntoViewIfNeeded();
    await expect(card).toContainText("8칸");
    const actual = await card
      .locator('[aria-hidden="true"]')
      .evaluate((grid) => {
        const wrapper = grid.parentElement.getBoundingClientRect();
        const rect = grid.getBoundingClientRect();
        return {
          mask: [...grid.children]
            .map((child) =>
              child.classList.contains("bg-pink-400") ? "1" : "0",
            )
            .join(""),
          gridHeight: rect.height,
          wrapperHeight: wrapper.height,
          fits:
            rect.top >= wrapper.top &&
            rect.bottom <= wrapper.bottom &&
            rect.left >= wrapper.left &&
            rect.right <= wrapper.right,
        };
      });
    expect(actual.mask).toBe(rows);
    expect(actual.fits).toBe(true);
    expect(actual.gridHeight).toBe(136);
    shapes.push({ n, label, ...actual });
  }
  for (const [index, entry] of mapping.entries())
    await page
      .getByRole("combobox", {
        name: `도형 ${index + 1} 한글 이름`,
        exact: true,
      })
      .selectOption(entry.label);
  await page.getByRole("button", { name: "매칭 완료", exact: true }).click();
  const result = JSON.parse(
    await page
      .getByRole("textbox", { name: "한글 도형 매칭 JSON" })
      .inputValue(),
  );
  expect(result.catalogVersion).toBe("2026-10-02-user-labels-v2");
  expect(
    result.mapping.map(({ pieceId, label }) => ({ pieceId, label })),
  ).toEqual(mapping.map(({ pieceId, label }) => ({ pieceId, label })));
  expect(result.mapping.slice(17).map(({ cells }) => cells)).toEqual([8, 8]);
  await page.screenshot({
    path: info.outputPath("corrected-label-page.png"),
    fullPage: true,
  });
  writeFileSync(
    info.outputPath("corrected-shapes.json"),
    JSON.stringify(
      {
        at: new Date().toISOString(),
        buildId,
        browser: info.project.name,
        source: "Canonical Hangul labels with corrected bottom rows",
        shapes,
        result: "PASS",
      },
      null,
      2,
    ) + "\n",
  );
});
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
