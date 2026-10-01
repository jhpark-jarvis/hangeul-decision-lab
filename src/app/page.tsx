import { PuzzleApp } from "@/components/puzzle/PuzzleApp";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-10">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4 border-b border-stone-200 pb-6">
        <div>
          <p className="mb-2 text-xs font-semibold tracking-[0.18em] text-stone-500">
            MAPLE HANGEUL SOLVER
          </p>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            한글날 퍼즐 배치 도우미
          </h1>
          <p className="mt-3 text-sm leading-6 text-stone-600">
            현재 보드와 블록을 입력하고 배치 순서를 확인하세요. 추천을 한 단계씩
            적용할 수 있습니다.
          </p>
        </div>
        <span className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1.5 text-xs font-semibold text-orange-800">
          수동 입력
        </span>
      </header>
      <PuzzleApp />
      <footer className="mt-8 text-xs leading-5 text-stone-500">
        데스크톱 Chrome·Edge, 화면 폭 1280px 이상 우선 지원 목표. 개발 catalog와
        합성 규칙 기준이며 실제 이벤트 정확성·전체 제품 수락은 검증 중입니다.
      </footer>
    </main>
  );
}
