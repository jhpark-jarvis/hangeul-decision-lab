import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Bound solver-heavy files to reduce CPU contention during local checks.
    maxWorkers: 2,
  },
});
