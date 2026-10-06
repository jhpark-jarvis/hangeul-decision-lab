import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hangeul Decision Lab · 한글날 퍼즐 배치 도우미",
  description:
    "16행 × 10열 보드와 블록을 직접 입력하고 배치 순서를 분석하는 도구",
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
