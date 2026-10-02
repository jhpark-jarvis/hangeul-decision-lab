import { readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { createJourney } from "../../scripts/acceptance/browser-journey.mjs";

const inputs = JSON.parse(
  readFileSync(
    new URL("../../scripts/acceptance/fixtures.json", import.meta.url),
    "utf8",
  ),
);
const buildId = readFileSync(".next/BUILD_ID", "utf8").trim();
// Preserve ADR-0010's original three representative boards. Added event
// journeys verify behavior separately and do not silently redefine the sample.
const measuredFixtureIds = new Set([
  "ordinary-line-item",
  "blocked-reroll",
  "single-cell-item-rescue",
]);
const catalogReference = JSON.parse(
  readFileSync("tests/fixtures/pieces/catalog-v2.json", "utf8"),
);

async function measureAnalyze(page) {
  const heapBefore = await page.evaluate(
    () => performance.memory?.usedJSHeapSize ?? null,
  );
  // Test-only DOM timing. Never inject private application state.
  await page.evaluate(() => {
    window.__acceptanceTiming = { start: null, elapsedMs: null };
    const button = [...document.querySelectorAll("button")].find(
      (entry) => entry.textContent.trim() === "Analyze",
    );
    button.addEventListener(
      "click",
      () => {
        window.__acceptanceTiming.start = performance.now();
        let sawPending = !document.querySelector("#result-heading");
        const observer = new MutationObserver(() => {
          if (!document.querySelector("#result-heading")) sawPending = true;
          if (sawPending && document.querySelector("#result-heading")) {
            observer.disconnect();
            requestAnimationFrame(() =>
              requestAnimationFrame(() => {
                window.__acceptanceTiming.elapsedMs =
                  performance.now() - window.__acceptanceTiming.start;
              }),
            );
          }
        });
        observer.observe(document.body, {
          childList: true,
          characterData: true,
          subtree: true,
        });
      },
      { once: true, capture: true },
    );
  });
  await page.getByRole("button", { name: "Analyze", exact: true }).click();
  await page.waitForFunction(
    () => window.__acceptanceTiming?.elapsedMs !== null,
  );
  const observed = await page.evaluate(() => ({
    elapsedMs: window.__acceptanceTiming.elapsedMs,
    heapAfter: performance.memory?.usedJSHeapSize ?? null,
  }));
  const text = await page
    .getByLabel("분석 호출 시간", { exact: true })
    .textContent();
  return { ...observed, heapBefore, solverMs: Number.parseFloat(text) };
}

for (const fixture of inputs.fixtures) {
  test(fixture.id, async ({ page, context, browser }, testInfo) => {
    const requests = [],
      errors = [],
      sockets = [],
      failures = [],
      responses = [];
    let journey;
    const report = {
      measuredAtUtc: new Date().toISOString(),
      executor: "Codex / Playwright",
      browser: testInfo.project.name,
      runtimeVersion: browser.version(),
      buildId,
      viewport: page.viewportSize(),
      fixtureId: fixture.id,
      fixtureVersion: inputs.version,
      requests,
      errors,
      sockets,
      failures,
      responses,
      samples: [],
      result: "IN_PROGRESS",
      isolation:
        "Fresh context; installed browser channel; loopback-only request guard; service workers blocked",
      timingDefinition:
        "Click capture to result DOM followed by two animation frames. Heap before/after observations, not peak or retained-only memory.",
    };
    context.on("request", (request) =>
      requests.push({
        url: request.url(),
        method: request.method(),
        type: request.resourceType(),
        body: request.postData(),
      }),
    );
    context.on("requestfailed", (request) =>
      failures.push({
        url: request.url(),
        error: request.failure()?.errorText,
      }),
    );
    context.on("response", (response) =>
      responses.push({ url: response.url(), status: response.status() }),
    );
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
      if (["error", "warning"].includes(message.type()))
        errors.push(`${message.type()}: ${message.text()}`);
    });
    page.on("websocket", (socket) => sockets.push(socket.url()));
    await context.route("**/*", (route) =>
      new URL(route.request().url()).origin === "http://127.0.0.1:45001"
        ? route.continue()
        : route.abort("blockedbyclient"),
    );
    try {
      await page.goto("/");
      if (fixture.journey === "ordinary") {
        const catalog = page.getByLabel("이벤트 블록 목록", { exact: true });
        report.catalogChecks = [];
        for (const piece of catalogReference.pieces) {
          const count = await catalog
            .locator(`[data-piece-id="${piece.id}"] .shape-filled`)
            .count();
          expect(count).toBe(piece.cells);
          for (let slot = 0; slot < 3; slot++) {
            const select = page.getByRole("combobox", {
              name: `slot ${slot} 블록`,
              exact: true,
            });
            await select.selectOption(piece.id);
            await expect(select).toHaveValue(piece.id);
          }
          report.catalogChecks.push({
            id: piece.id,
            cells: count,
            slots: 3,
            result: "PASS",
          });
        }
        expect(await catalog.locator("[data-piece-id]").count()).toBe(19);
        report.catalogScreenshot = testInfo.outputPath("catalog.png");
        await catalog.screenshot({ path: report.catalogScreenshot });
      }
      journey = createJourney(
        {
          playwright: page,
          reload: () => page.reload(),
          capture: async (name) => {
            report.capacityScreenshot = testInfo.outputPath(`${name}.png`);
            await page.screenshot({ path: report.capacityScreenshot });
          },
        },
        fixture,
      );
      await journey.begin();
      let input;
      do {
        input = await journey.inputBatch(32);
      } while (!input.complete);
      await journey.finishInput();
      if (measuredFixtureIds.has(fixture.id)) {
        report.warmup = await measureAnalyze(page);
        for (let sample = 0; sample < 3; sample++)
          report.samples.push(await measureAnalyze(page));
        for (const sample of report.samples) {
          expect(sample.solverMs).toBeLessThanOrEqual(3000);
          expect(sample.elapsedMs).toBeLessThanOrEqual(3000);
          expect(Number.isFinite(sample.heapBefore)).toBe(true);
          expect(Number.isFinite(sample.heapAfter)).toBe(true);
          expect(sample.heapBefore).toBeLessThanOrEqual(64 * 1024 * 1024);
          expect(sample.heapAfter).toBeLessThanOrEqual(64 * 1024 * 1024);
        }
        report.acceptance = {
          fullUiMs: 3000,
          beforeAfterJsHeapBytes: 64 * 1024 * 1024,
          source: "Selected UI response and pre/post JavaScript heap limits",
        };
        await page
          .getByRole("heading", { name: "추천 결과", exact: true })
          .scrollIntoViewIfNeeded();
        report.screenshot = testInfo.outputPath("recommendation.png");
        await page.screenshot({ path: report.screenshot });
      }
      report.journey = await journey.run();
      report.storage = await page.evaluate(async () => ({
        localKeys: Object.keys(localStorage),
        sessionKeys: Object.keys(sessionStorage),
        databases: (await indexedDB.databases()).map((entry) => entry.name),
        caches: await caches.keys(),
        serviceWorkers: (await navigator.serviceWorker.getRegistrations())
          .length,
      }));
      report.cookies = (await context.cookies()).map(({ name, domain }) => ({
        name,
        domain,
      }));
      expect(requests.length).toBeGreaterThan(0);
      expect(
        requests.filter(
          (entry) => new URL(entry.url).origin !== "http://127.0.0.1:45001",
        ),
      ).toEqual([]);
      expect(
        requests.filter(
          (entry) => entry.method !== "GET" || entry.body !== null,
        ),
      ).toEqual([]);
      expect(
        requests.filter((entry) =>
          ["fetch", "xhr", "eventsource"].includes(entry.type),
        ),
      ).toEqual([]);
      expect(failures).toEqual([]);
      expect(responses.filter((entry) => entry.status >= 400)).toEqual([]);
      expect(sockets).toEqual([]);
      expect(errors).toEqual([]);
      expect(report.storage).toEqual({
        localKeys: [],
        sessionKeys: [],
        databases: [],
        caches: [],
        serviceWorkers: 0,
      });
      expect(report.cookies).toEqual([]);
      report.result = "PASS";
    } catch (error) {
      report.result = "FAIL";
      report.failure = String(error);
      report.partialChecks = journey?.checks ?? [];
      throw error;
    } finally {
      writeFileSync(
        testInfo.outputPath("journey.json"),
        JSON.stringify(report, null, 2) + "\n",
      );
    }
  });
}
