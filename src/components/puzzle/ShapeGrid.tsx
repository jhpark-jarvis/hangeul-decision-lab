import type { Shape } from "@/domain/pieces/types";

export function ShapeGrid({ shape }: { shape: Shape }) {
  return (
    <span
      className="shape-grid"
      aria-hidden="true"
      style={{ gridTemplateColumns: `repeat(${shape[0].length}, 10px)` }}
    >
      {shape.flat().map((filled, index) => (
        <span key={index} className={filled ? "shape-filled" : ""} />
      ))}
    </span>
  );
}
