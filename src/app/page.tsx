import { BoardEditor } from "@/components/board/BoardEditor";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-5xl px-5 py-8 sm:px-8 sm:py-12">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4 border-b border-stone-200 pb-6">
        <div>
          <p className="mb-2 text-xs font-semibold tracking-[0.18em] text-stone-500">
            MAPLE HANGEUL SOLVER
          </p>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            퍼즐 보드 입력
          </h1>
          <p className="mt-3 text-sm leading-6 text-stone-600">
            게임 화면을 보며 점유된 칸을 표시하세요. 다시 누르면 빈칸으로
            돌아갑니다.
          </p>
        </div>
        <span className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1.5 text-xs font-semibold text-orange-800">
          수동 입력
        </span>
      </header>
      <BoardEditor />
      <footer className="mt-8 text-xs leading-5 text-stone-500">
        현재는 보드 입력 단계입니다. 블록 선택과 배치 추천은 다음 단계에서
        추가합니다.
      </footer>
    </main>
  );
}
