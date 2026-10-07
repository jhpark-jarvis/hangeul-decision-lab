import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    files: ["src/domain/**/*.ts", "src/research/**/*.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            "react",
            "react/*",
            "react-dom",
            "react-dom/*",
            "next",
            "next/*",
            "@/components/*",
            "@/app/*",
          ],
        },
      ],
      "no-restricted-globals": [
        "error",
        "window",
        "document",
        "localStorage",
        "sessionStorage",
        "fetch",
      ],
    },
  },
  globalIgnores([
    ".next/**",
    ".research-output/**",
    "node_modules/**",
    "out/**",
    "coverage/**",
    "next-env.d.ts",
  ]),
]);
