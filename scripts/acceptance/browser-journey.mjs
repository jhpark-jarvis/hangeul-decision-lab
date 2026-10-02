// Uses only the documented locator API of a caller-supplied browser tab.
// No browser launch, private app state injection, network capture, or persistence.
export function createJourney(tab, fixture) {
  const checks = [];
  const button = (name) =>
    tab.playwright.getByRole("button", { name, exact: true });
  const input = (name) =>
    tab.playwright.getByRole("spinbutton", { name, exact: true });
  const slot = (index) =>
    tab.playwright.getByRole("combobox", {
      name: `slot ${index} 블록`,
      exact: true,
    });
  const cell = (row, col) =>
    tab.playwright.locator(`button[data-row="${row}"][data-col="${col}"]`);
  const occupied = () =>
    tab.playwright
      .getByRole("status", { name: "점유 칸 수", exact: true })
      .innerText();
  const overlay = () =>
    tab.playwright.locator('button[data-overlay="true"]').count();
  const clearing = () =>
    tab.playwright.locator('button[data-clearing="true"]').count();
  const filledCells = fixture.board.flatMap((row, r) =>
    row.flatMap((filled, c) => (filled ? [{ row: r, col: c }] : [])),
  );
  let enteredCells = 0;

  async function check(name, actual, expected) {
    const passed = JSON.stringify(actual) === JSON.stringify(expected);
    checks.push({ name, actual, expected, passed });
    if (!passed)
      throw new Error(
        `${fixture.id}: ${name}; actual ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`,
      );
  }
  async function analyze() {
    await button("Analyze").click();
    await tab.playwright
      .getByRole("heading", { name: "추천 결과", exact: true })
      .waitFor({ state: "visible" });
    await check(
      "analysis result shown",
      await tab.playwright
        .getByRole("heading", { name: "추천 결과", exact: true })
        .count(),
      1,
    );
  }
  async function apply() {
    await button("Apply Step").click();
  }
  async function nextSet() {
    await check(
      "awaiting exactly three new pieces",
      await button("3개 블록 입력").count(),
      1,
    );
    await check(
      "Analyze disabled while awaiting input",
      await button("Analyze").isEnabled(),
      false,
    );
    await check(
      "no enabled Apply while awaiting input",
      (await button("Apply Step").count()) > 0
        ? await button("Apply Step").isEnabled()
        : false,
      false,
    );
  }
  async function errorText() {
    return tab.playwright
      .getByRole("region", { name: "입력 및 적용 상태", exact: true })
      .getByRole("alert")
      .innerText();
  }

  return {
    fixtureId: fixture.id,
    checks,
    inputCount: filledCells.length,
    async begin() {
      await tab.reload();
      await check(
        "160 board cells",
        await tab.playwright.locator("button[data-row][data-col]").count(),
        160,
      );
      await check("reload starts empty", await occupied(), "0 / 160");
      for (let index = 0; index < 3; index++)
        await slot(index).selectOption(fixture.slotIds[index] || "DOT");
      await button("3개 블록 입력").click();
      for (let index = 0; index < 3; index++)
        if (!fixture.slotIds[index]) await slot(index).selectOption("");
      await input("Reroll 보유 수").fill(String(fixture.abilities.reroll));
      await input("Single Cell 보유 수").fill(
        String(fixture.abilities.singleCell),
      );
      await button("Filled").click();
      await check(
        "Filled tool selected",
        await button("Filled").getAttribute("aria-pressed"),
        "true",
      );
    },
    // Caller runs small batches so tool deadlines cannot silently truncate fixtures.
    async inputBatch(maxCells = 16) {
      const end = Math.min(filledCells.length, enteredCells + maxCells);
      while (enteredCells < end) {
        const { row, col } = filledCells[enteredCells];
        await cell(row, col).click();
        enteredCells++;
      }
      await check(
        `input count after batch ${enteredCells}`,
        await occupied(),
        `${enteredCells} / 160`,
      );
      return {
        entered: enteredCells,
        total: filledCells.length,
        complete: enteredCells === filledCells.length,
      };
    },
    async finishInput() {
      await check(
        "all requested occupied cells entered",
        enteredCells,
        filledCells.length,
      );
      for (const item of fixture.items) {
        await button(
          item.type === "reroll" ? "Hidden: Reroll" : "Hidden: Single Cell",
        ).click();
        await cell(item.row, item.col).click();
        await check(
          `hidden ${item.type} at ${item.row},${item.col}`,
          (await cell(item.row, item.col).getAttribute("aria-label")).includes(
            `hidden ${item.type}`,
          ),
          true,
        );
      }
      await check(
        "hidden input preserves occupancy",
        await occupied(),
        `${filledCells.length} / 160`,
      );
    },
    async run() {
      if (fixture.journey === "event-single") {
        const expected = fixture.expected;
        await analyze();
        await check(
          "event single gap",
          await cell(expected.singleRow, 9).getAttribute("data-overlay"),
          "true",
        );
        await check("event single clear preview", await clearing(), 10);
        await apply();
        await check(
          "event single spends charge",
          await input("Single Cell 보유 수").getAttribute("value"),
          "0",
        );
        await check(
          "event single preserves reroll",
          await input("Reroll 보유 수").getAttribute("value"),
          String(fixture.abilities.reroll),
        );
        await check(
          "event follow-up shape",
          await overlay(),
          expected.pieceCells,
        );
        await check(
          "event follow-up row preview",
          await clearing(),
          expected.clearCells,
        );
        await apply();
        await check(
          "event final occupancy",
          await occupied(),
          `${expected.occupied} / 160`,
        );
        await check(
          "event final single count",
          await input("Single Cell 보유 수").getAttribute("value"),
          String(expected.singleCell),
        );
        await check(
          "event final reroll count",
          await input("Reroll 보유 수").getAttribute("value"),
          String(expected.reroll),
        );
        const acquired = expected.acquiredItem;
        await check(
          "event acquired icon removed",
          (
            await cell(acquired.row, acquired.col).getAttribute("aria-label")
          ).includes(`hidden ${acquired.type}`),
          false,
        );
        if (expected.retainedItem) {
          const retained = expected.retainedItem;
          await check(
            "event capacity icon retained",
            (
              await cell(retained.row, retained.col).getAttribute("aria-label")
            ).includes(`hidden ${retained.type}`),
            true,
          );
          await check(
            "event retention notice",
            (
              await tab.playwright
                .getByRole("region", { name: "입력 및 적용 상태", exact: true })
                .innerText()
            ).includes("상한으로 남김 1개"),
            true,
          );
        }
        await tab.capture?.("event-single-result");
        await nextSet();
      } else if (fixture.journey === "capacity") {
        await analyze();
        await check("capacity clear overlay", await clearing(), 10);
        await apply();
        await check(
          "capacity row occupancy clears",
          await occupied(),
          "0 / 160",
        );
        await check(
          "unacquired icon stays on cleared empty coordinate",
          (await cell(0, 0).getAttribute("aria-label")).includes(
            "hidden single-cell",
          ),
          true,
        );
        await check(
          "capacity count stays7",
          await input("Reroll 보유 수").getAttribute("value"),
          "7",
        );
        await check(
          "retention message",
          (
            await tab.playwright
              .getByRole("region", { name: "입력 및 적용 상태", exact: true })
              .innerText()
          ).includes("상한으로 남김 1개"),
          true,
        );
        await tab.capture?.("capacity-retained");
        await button("Hidden: Reroll").click();
        await cell(15, 0).click();
        await cell(15, 1).click();
        await check(
          "third icon accepted",
          (await cell(15, 1).getAttribute("aria-label")).includes(
            "hidden reroll",
          ),
          true,
        );
        await cell(15, 2).click();
        await check(
          "fourth icon rejected",
          (await cell(15, 2).getAttribute("aria-label")).includes("hidden"),
          false,
        );
        await check(
          "fourth icon error",
          (
            await tab.playwright
              .getByRole("region", { name: "입력 및 적용 상태", exact: true })
              .innerText()
          ).includes("최대 3개"),
          true,
        );
        await check(
          "existing retained icon preserved on error",
          (await cell(0, 0).getAttribute("aria-label")).includes(
            "hidden single-cell",
          ),
          true,
        );
        await check(
          "icon error preserves occupancy",
          await occupied(),
          "0 / 160",
        );
        await tab.capture?.("icon-limit-error");
        await button("Hidden: Single Cell").click();
        await cell(15, 0).click();
        await check(
          "replace at limit allowed",
          (await cell(15, 0).getAttribute("aria-label")).includes(
            "hidden single-cell",
          ),
          true,
        );
        await button("아이템 지우기").click();
        await cell(15, 1).click();
        await button("Hidden: Reroll").click();
        await cell(15, 2).click();
        await check(
          "remove then add recovers",
          (await cell(15, 2).getAttribute("aria-label")).includes(
            "hidden reroll",
          ),
          true,
        );
        await button("아이템 지우기").click();
        await cell(15, 0).click();
        await cell(15, 2).click();
        await nextSet();
        for (let index = 0; index < 3; index++)
          await slot(index).selectOption("DOT");
        await button("3개 블록 입력").click();
        await input("Reroll 보유 수").fill("6"); // Re-enter actual state after an external ability use.
        await button("Filled").click();
        for (let col = 0; col < 9; col++) await cell(0, col).click();
        await check(
          "refill preserves retained icon",
          (await cell(0, 0).getAttribute("aria-label")).includes(
            "hidden single-cell",
          ),
          true,
        );
        await analyze();
        await apply();
        await check(
          "recleared row acquires retained single",
          await input("Single Cell 보유 수").getAttribute("value"),
          "1",
        );
        await check(
          "acquired icon removed",
          (await cell(0, 0).getAttribute("aria-label")).includes(
            "hidden single-cell",
          ),
          false,
        );
      } else if (fixture.journey === "ordinary") {
        await button("토글").click();
        await cell(15, 9).press("Enter");
        await check(
          "keyboard Enter toggles once",
          await occupied(),
          "10 / 160",
        );
        await cell(15, 9).press("Space");
        await check("keyboard Space toggles once", await occupied(), "9 / 160");
        await analyze();
        await check("first-action shape overlay", await overlay(), 1);
        await check("first-action row clear overlay", await clearing(), 10);
        await button("대안 1 · 3단계").click();
        await check(
          "alternative selected",
          await button("대안 1 · 3단계").getAttribute("aria-pressed"),
          "true",
        );
        await button("최선 · 3단계").click();
        await cell(15, 9).click();
        await check(
          "board edit discards recommendation",
          await button("Apply Step").count(),
          0,
        );
        await check("board edit clears overlay", await overlay(), 0);
        await cell(15, 9).click();
        await input("Reroll 보유 수").fill("8");
        await check(
          "invalid count preserves zero",
          await input("Reroll 보유 수").getAttribute("value"),
          "0",
        );
        await check(
          "invalid count displays product error",
          (await errorText()).length > 0,
          true,
        );
        await input("Reroll 보유 수").fill("0");
        await analyze();
        await apply();
        await check(
          "first apply clears completed row",
          await occupied(),
          "0 / 160",
        );
        await check(
          "first apply acquires single-cell",
          await input("Single Cell 보유 수").getAttribute("value"),
          "1",
        );
        await apply();
        await apply();
        await nextSet();
        const before = await occupied();
        for (let index = 0; index < 3; index++)
          await slot(index).selectOption("DOT");
        await button("3개 블록 입력").click();
        await check("next set preserves board", await occupied(), before);
        await check(
          "next set preserves abilities",
          await input("Single Cell 보유 수").getAttribute("value"),
          "1",
        );
        await analyze();
        await button("보드·아이템 비우기").click();
        await check(
          "clear board resets occupancy",
          await occupied(),
          "0 / 160",
        );
        await check(
          "clear board preserves pieces",
          await button("Analyze").isEnabled(),
          true,
        );
        await check(
          "clear board preserves abilities",
          await input("Single Cell 보유 수").getAttribute("value"),
          "1",
        );
        await check(
          "clear board discards recommendation",
          await button("Apply Step").count(),
          0,
        );
      } else if (fixture.journey === "reroll") {
        await analyze();
        await check(
          "reroll target highlight",
          await tab.playwright.locator(".piece-slot.reroll-target").count(),
          1,
        );
        await apply();
        await check(
          "reroll spends one charge",
          await input("Reroll 보유 수").getAttribute("value"),
          "0",
        );
        await check(
          "pending reroll blocks Analyze",
          await button("Analyze").isEnabled(),
          false,
        );
        const actual = tab.playwright.getByRole("combobox", {
          name: "실제 reroll 결과",
          exact: true,
        });
        await actual.selectOption(fixture.sameRerollId ?? "MIEUM");
        await button("reroll 결과 반영").click();
        await check("same type rejected", (await errorText()).length > 0, true);
        await check("same type keeps pending", await actual.count(), 1);
        await actual.selectOption(fixture.actualRerollId ?? "DOT");
        await button("reroll 결과 반영").click();
        await check(
          "different actual result resolves pending",
          await actual.count(),
          0,
        );
        if (fixture.actualRerollId) {
          await check(
            "actual event piece in fixed slot",
            await slot(2).inputValue(),
            fixture.actualRerollId,
          );
          await check(
            "actual event piece shape",
            await tab.playwright
              .locator(".piece-slot")
              .nth(2)
              .locator(".shape-filled")
              .count(),
            fixture.actualPieceCells,
          );
        }
        await analyze();
        await apply();
        await nextSet();
      } else if (fixture.journey === "single") {
        await analyze();
        await check(
          "single-cell recommendation at expected gap",
          await cell(4, 9).getAttribute("data-overlay"),
          "true",
        );
        await check("single-cell clear preview", await clearing(), 10);
        await apply();
        await check(
          "single-cell spends charge",
          await input("Single Cell 보유 수").getAttribute("value"),
          "0",
        );
        await check(
          "clear acquires reroll",
          await input("Reroll 보유 수").getAttribute("value"),
          "1",
        );
        await check("general follow-up clear preview", await clearing(), 20);
        await apply();
        await nextSet();
      } else if (fixture.journey === "gameover") {
        await analyze();
        await check(
          "gameover Apply disabled",
          await button("Apply Step").isEnabled(),
          false,
        );
        await button("Empty").click();
        await cell(0, 0).click();
        await slot(2).selectOption("DOT");
        await analyze();
        await check(
          "corrected input restores Apply",
          await button("Apply Step").isEnabled(),
          true,
        );
        await apply();
        await nextSet();
      } else throw new Error(`Unknown journey ${fixture.journey}`);
      return {
        fixtureId: fixture.id,
        checks,
        finalOccupied: await occupied(),
        result: "PASS",
      };
    },
  };
}
