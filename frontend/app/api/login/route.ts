import { NextResponse } from "next/server";
import crypto from "crypto";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let body: { email?: string; password?: string } = {};
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  const email = (body.email || "").trim().toLowerCase();
  const password = body.password || "";
  const expectedEmail = (process.env.DASHBOARD_EMAIL || "").trim().toLowerCase();
  const hash = crypto.createHash("sha256").update("nbn:" + password).digest("hex");

  const emailOk = expectedEmail.length > 0 && email === expectedEmail;
  const passOk =
    (process.env.DASHBOARD_PWHASH || "").length > 0 && hash === process.env.DASHBOARD_PWHASH;

  // Constant-ish response; single generic error so nothing is hinted.
  if (!emailOk || !passOk) {
    return NextResponse.json({ error: "Invalid email or password." }, { status: 401 });
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.set("nbn_session", process.env.SESSION_TOKEN || "", {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });
  return res;
}
