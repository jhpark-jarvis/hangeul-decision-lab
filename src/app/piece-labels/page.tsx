"use client";

import Link from "next/link";
import { useState } from "react";
import {
  getInitialCatalog,
  EVENT_CATALOG_VERSION,
} from "@/domain/pieces/catalog";
import {
  completePieceLabels,
  HANGEUL_PIECE_LABELS,
  LABEL_CELL_COUNTS,
  type PieceLabelMapping,
} from "@/features/pieces/labels";

const catalog = getInitialCatalog();
export default function PieceLabelsPage() {
  const [labels, setLabels] = useState<string[]>(() => catalog.map(() => ""));
  const [completed, setCompleted] = useState<PieceLabelMapping | null>(null);
  const [message, setMessage] = useState("");
  const result = completePieceLabels(catalog, labels);
  const selectedCount = labels.filter(Boolean).length;
  const json = completed
    ? JSON.stringify(
        { catalogVersion: EVENT_CATALOG_VERSION, mapping: completed },
        null,
        2,
      )
    : "";

  return (
    <main
      className="mx-auto max-w-7xl p-6 space-y-5"
      aria-label="도형 한글 이름 매칭"
    >
      <header className="space-y-2">
        <h1 className="text-2xl font-bold">19개 도형에 한글 이름 붙이기</h1>
        <p>
          도형을 보고 아래에서 이름을 하나씩 골라 주세요. 이름은 미리 넣지
          않았습니다.
        </p>
        <p className="help">
          회전·반전 모양도 함께 참고하세요. 같은 이름은 한 번만 선택할 수
          있습니다. 모두 고르면 ‘매칭 완료’를 누르고 채팅에 완료했다고 알려
          주세요.
        </p>
        <p className="help">
          선택은 이 탭에서 유지됩니다. 새로고침하면 초기화되므로 완료 후 ‘결과
          복사’로 보관할 수 있습니다.
        </p>
        <Link
          href="/"
          prefetch={false}
          className="text-sm text-blue-700 underline"
        >
          솔버 화면으로 이동
        </Link>
      </header>
      <div className="sticky top-0 z-30 flex flex-wrap items-center gap-3 rounded-xl border border-blue-200 bg-white p-4 shadow-sm">
        <p role="status" aria-label="이름 매칭 진행" className="font-semibold">
          선택한 이름 {selectedCount} / 19
        </p>
        <button
          type="button"
          className="primary-button"
          disabled={!result.ok || !!completed}
          onClick={() => {
            if (!result.ok) return;
            setCompleted(result.mapping);
            setMessage(
              "19개 매칭이 완료되었습니다. 이 탭을 둔 채 채팅에 ‘완료’라고 알려 주세요.",
            );
          }}
        >
          매칭 완료
        </button>
        {completed && (
          <button
            type="button"
            className="small-button"
            onClick={() => {
              setCompleted(null);
              setMessage("");
            }}
          >
            매칭 수정
          </button>
        )}
        {!result.ok && selectedCount === 19 && (
          <p role="alert">{result.message}</p>
        )}
      </div>
      {message && (
        <p
          role="status"
          aria-label="매칭 결과 안내"
          className="rounded-xl bg-blue-50 p-4"
        >
          {message}
        </p>
      )}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {catalog.map((piece, index) => {
          const count = piece.shape.flat().filter(Boolean).length;
          return (
            <article
              key={piece.id}
              className="rounded-xl border border-stone-200 bg-white p-4"
              aria-label={`도형 ${index + 1}`}
            >
              <h2 className="font-semibold">
                도형 {index + 1}{" "}
                <span className="text-sm font-normal text-stone-500">
                  · {count}칸
                </span>
              </h2>
              <div
                className="flex h-40 items-center justify-center"
                aria-label={`도형 ${index + 1} 모양`}
              >
                <div
                  className="grid gap-1"
                  aria-hidden="true"
                  style={{
                    gridTemplateColumns: `repeat(${piece.shape[0].length}, 24px)`,
                  }}
                >
                  {piece.shape.flat().map((filled, cell) => (
                    <span
                      key={cell}
                      className={`h-6 w-6 rounded border ${filled ? "border-pink-700 bg-pink-400" : "border-transparent"}`}
                    />
                  ))}
                </div>
              </div>
              <label className="text-sm font-semibold">
                이 도형의 이름
                <select
                  aria-label={`도형 ${index + 1} 한글 이름`}
                  disabled={!!completed}
                  value={labels[index]}
                  onChange={(event) => {
                    const value = event.target.value;
                    setLabels((current) =>
                      current.map((label, position) =>
                        position === index ? value : label,
                      ),
                    );
                    setMessage("");
                  }}
                >
                  <option value="">이름을 골라 주세요</option>
                  {HANGEUL_PIECE_LABELS.map((label) => (
                    <option
                      key={label}
                      value={label}
                      disabled={
                        (LABEL_CELL_COUNTS[label] !== undefined &&
                          LABEL_CELL_COUNTS[label] !== count) ||
                        labels.some(
                          (selected, position) =>
                            selected === label && position !== index,
                        )
                      }
                    >
                      {label}
                      {label === "ㅡ"
                        ? " (3칸)"
                        : label === "ㅣ"
                          ? " (5칸)"
                          : label === "점"
                            ? " (1칸)"
                            : ""}
                    </option>
                  ))}
                </select>
              </label>
            </article>
          );
        })}
      </div>
      {completed && (
        <section aria-label="완료한 한글 매칭" className="panel space-y-3">
          <h2>완료한 매칭 결과</h2>
          <p className="help">
            완료 결과를 확인할 수 있도록 이 탭을 열어 두세요. 별도로 전달하려면
            아래 버튼으로 복사하세요.
          </p>
          <button
            type="button"
            className="small-button"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(json);
                setMessage(
                  "매칭 결과를 복사했습니다. 채팅에 붙여넣을 수 있습니다.",
                );
              } catch {
                setMessage(
                  "복사 권한이 없어 자동 복사하지 못했습니다. 아래 결과를 선택해 Ctrl+C로 복사하세요.",
                );
              }
            }}
          >
            결과 복사
          </button>
          <textarea
            className="w-full rounded-lg border border-stone-300 p-3 font-mono text-xs"
            rows={12}
            readOnly
            aria-label="한글 도형 매칭 JSON"
            value={json}
          />
        </section>
      )}
    </main>
  );
}
