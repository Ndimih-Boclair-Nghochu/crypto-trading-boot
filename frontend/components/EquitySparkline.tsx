"use client";

type Props = {
  values: number[];
  width?: number;
  height?: number;
  positive: boolean;
};

/**
 * Minimal inline sparkline for the equity curve. Rendered as a hand-built
 * SVG path so the ticker strip stays dependency-free.
 */
export function EquitySparkline({ values, width = 120, height = 28, positive }: Props) {
  if (values.length < 2) {
    return (
      <svg width={width} height={height} aria-hidden="true">
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="var(--border)"
          strokeWidth={1.5}
          strokeDasharray="2 3"
        />
      </svg>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return [x, y];
  });

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
  const stroke = positive ? "var(--accent-green)" : "var(--accent-red)";

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
