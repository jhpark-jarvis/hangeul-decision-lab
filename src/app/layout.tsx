import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "메이플 한글날 퍼즐 · 보드 입력",
  description: "16행 × 10열 퍼즐 보드를 직접 입력하는 로컬 도구",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
