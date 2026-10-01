import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "**/*.spec.mjs",
  workers: 1,
  retries: 0,
  timeout: 120_000,
  outputDir: ".next/acceptance-artifacts",
  reporter: [
    ["list"],
    ["json", { outputFile: ".next/acceptance-report.json" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:45001",
    viewport: { width: 1280, height: 900 },
    headless: true,
    serviceWorkers: "block",
    actionTimeout: 10_000,
  },
  projects: [
    { name: "chrome", use: { channel: "chrome" } },
    { name: "edge", use: { channel: "msedge" } },
  ],
});
