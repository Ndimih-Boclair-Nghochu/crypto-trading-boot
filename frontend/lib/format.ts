export function num(value: number | string | null | undefined, digits = 2): number {
  if (value === null || value === undefined) return 0;
  const n = typeof value === "string" ? parseFloat(value) : value;
  return Number.isFinite(n) ? n : 0;
}

export function fmt(value: number | string | null | undefined, digits = 2): string {
  return num(value, digits).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtSigned(value: number | string | null | undefined, digits = 2): string {
  const n = num(value, digits);
  const sign = n > 0 ? "+" : "";
  return `${sign}${fmt(n, digits)}`;
}

export function fmtTime(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
