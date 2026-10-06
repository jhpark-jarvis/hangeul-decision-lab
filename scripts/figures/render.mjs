import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { format } from "prettier";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const output = resolve(root, "assets/figures");
mkdirSync(output, { recursive: true });
const escape = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll('"', "&quot;");
const text = (x, y, value, css = "body") =>
  `<text x="${x}" y="${y}" class="${css}">${escape(value)}</text>`;
const line = (x1, y1, x2, y2, dashed = false) =>
  `<path d="M${x1} ${y1} L${x2} ${y2}" fill="none" stroke="#64748b" stroke-width="2" ${dashed ? 'stroke-dasharray="6 5"' : ""} marker-end="url(#arrow)"/>`;
const svg = (
  width,
  height,
  title,
  content,
) => `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}">
<title>${escape(title)}</title><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#64748b"/></marker></defs>
<style>text{font-family:Arial,sans-serif;fill:#172638}.title{font-size:29px;font-weight:700}.heading{font-size:21px;font-weight:700}.body{font-size:17px}.small{font-size:14px}.label{font-size:12px;fill:#64748b}.pill{font-size:13px;font-weight:700;fill:#166534}</style>
<rect width="100%" height="100%" fill="#fff"/>${content}</svg>\n`;
const box = (x, y, w, h, title, lines, future = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${future ? "#f8fafc" : "#f0fdfa"}" stroke="${future ? "#94a3b8" : "#0f766e"}" stroke-width="2" ${future ? 'stroke-dasharray="7 5"' : ""}/>${text(x + 20, y + 36, title, "heading")}${lines.map((value, i) => text(x + 20, y + 70 + i * 27, value)).join("")}`;

const before = Array.from({ length: 16 }, () => Array(10).fill(false));
for (const [r, cols] of [
  [2, [2, 3]],
  [3, [2]],
  [9, [0, 1, 2]],
  [10, [1]],
  [14, [0, 1]],
  [15, [0, 1, 2, 3, 4, 5, 6]],
]) {
  for (const c of cols) before[r][c] = true;
}
const placed = [
  [15, 7],
  [15, 8],
  [15, 9],
];
const filled = before.map((row) => [...row]);
for (const [r, c] of placed) filled[r][c] = true;
const clearedRows = filled.flatMap((row, r) => (row.every(Boolean) ? [r] : []));
const after = filled.map((row) =>
  row.every(Boolean) ? row.map(() => false) : [...row],
);
function board(x, y, state, overlay = []) {
  let result = "";
  for (let r = 0; r < 16; r++) {
    result += text(x - 25, y + r * 23 + 16, r + 1, "label");
    for (let c = 0; c < 10; c++) {
      const highlighted = overlay.some(([a, b]) => a === r && b === c);
      result += `<rect x="${x + c * 23}" y="${y + r * 23}" width="21" height="21" rx="2" fill="${highlighted ? "#f59e0b" : state[r][c] ? "#0f766e" : "#f1f5f9"}" stroke="${highlighted ? "#b45309" : "#cbd5e1"}"/>`;
    }
  }
  for (let c = 0; c < 10; c++)
    result += text(x + c * 23 + 6, y - 12, c + 1, "label");
  return result;
}
const lineClear = svg(
  950,
  570,
  "Synthetic 16 by 10 line-clear rule illustration",
  [
    text(38, 43, "A horizontal clear, without gravity", "title"),
    text(
      38,
      75,
      "Synthetic rule illustration. No game screenshot or search result.",
    ),
    text(82, 120, "Place LINE_3", "heading"),
    text(578, 120, "After clearing row 16", "heading"),
    board(88, 160, before, placed),
    board(590, 160, after),
    line(363, 339, 525, 339),
    text(378, 304, "place + clear"),
    text(371, 377, "Other rows stay"),
    text(371, 402, "at the same index."),
    `<rect x="88" y="150" width="225" height="378" fill="none" stroke="#cbd5e1"/><rect x="588" y="503" width="231" height="27" fill="none" stroke="#b45309" stroke-dasharray="5 4"/>`,
    text(
      38,
      554,
      "Teal: occupied    Amber: proposed placement    Dashed: cleared row    Coordinates shown 1-based",
      "small",
    ),
  ].join(""),
);
const pipeline = svg(
  1120,
  565,
  "Implemented observation review search and replay boundaries",
  [
    text(38, 46, "Observation to a verified decision", "title"),
    text(
      38,
      79,
      "All boxes describe implemented responsibilities. Game control remains manual.",
    ),
    box(38, 128, 310, 155, "1. Local observation", [
      "Manual input or shared frame",
      "Board / pieces / abilities",
    ]),
    box(405, 128, 310, 155, "2. Explicit review", [
      "Resolve uncertain cells / items",
      "Validate the confirmed state",
    ]),
    box(772, 128, 310, 155, "3. Bounded DFS", [
      "Shared 512-node default budget",
      "Rank replayable candidates",
    ]),
    line(350, 205, 400, 205),
    line(717, 205, 767, 205),
    line(926, 287, 926, 363),
    box(772, 368, 310, 147, "4. App state replay", [
      "Apply Step / snapshot guards",
      "Does not operate the game",
    ]),
    box(405, 368, 310, 147, "5. User plays game", [
      "Place / use ability manually",
      "Observe the actual outcome",
    ]),
    box(38, 368, 310, 147, "6. Read next state", [
      "Next pieces / reroll observation",
      "Return to explicit review",
    ]),
    line(768, 438, 722, 438),
    line(401, 438, 356, 438),
    line(193, 365, 193, 290),
    text(
      38,
      547,
      "Recognition is a draft, not state approval. Awaiting input is not a game-over result.",
      "small",
    ),
  ].join(""),
);
const roadmap = svg(
  1230,
  335,
  "Current DFS and conditional future research stages",
  [
    text(32, 45, "Search before learning", "title"),
    text(
      32,
      77,
      "Solid = implemented baseline. Dashed = research proposal, not a measured result.",
    ),
    box(32, 130, 205, 140, "DFS baseline", [
      "Fix revision / config",
      "Replay current rules",
    ]),
    box(
      271,
      130,
      205,
      140,
      "Benchmark first",
      ["Seeds / simulator", "Budget / evaluation"],
      true,
    ),
    box(
      510,
      130,
      205,
      140,
      "Beam Search",
      ["Width / horizon", "Same time budget"],
      true,
    ),
    box(
      749,
      130,
      205,
      140,
      "MCTS",
      ["Chance / rollout", "Validated dynamics"],
      true,
    ),
    box(
      988,
      130,
      205,
      140,
      "ML / RL",
      ["Policy / value hybrid", "Only if justified"],
      true,
    ),
    ...[237, 476, 715, 954].map((x) => line(x + 2, 200, x + 30, 200, true)),
    text(
      32,
      308,
      "Advance only after reproducible limitations, fair comparisons, and a separately approved experiment.",
      "small",
    ),
  ].join(""),
);
for (const [name, content] of [
  ["line-clear.svg", lineClear],
  ["decision-pipeline.svg", pipeline],
  ["research-roadmap.svg", roadmap],
]) {
  writeFileSync(resolve(output, name), content);
}
writeFileSync(
  resolve(output, "figure-data.json"),
  await format(
    JSON.stringify(
      {
        schemaVersion: 1,
        method:
          "Deterministic project-generated SVG; English labels for portable fonts.",
        source:
          "Synthetic board and conceptual responsibility diagrams; no game pixels.",
        kind: "illustration, not experiment result",
        figures: [
          {
            id: "figure-1",
            file: "line-clear.svg",
            purpose: "16x10 placement and no-gravity clear illustration",
            before,
            placed,
            clearedRows,
            after,
          },
          {
            id: "figure-2",
            file: "decision-pipeline.svg",
            purpose:
              "Implemented observation/review/search/replay responsibility boundary",
          },
          {
            id: "figure-3",
            file: "research-roadmap.svg",
            purpose: "Current DFS versus conditional future proposals",
          },
        ],
        regenerate: "node scripts/figures/render.mjs",
      },
      null,
      2,
    ),
    { parser: "json" },
  ),
);
console.log(
  "Generated 3 SVG illustrations and figure-data.json. No screenshot, solver, benchmark, or training executed.",
);
