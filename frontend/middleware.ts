import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Server-side gate. The session token is a long random secret set in .env.local;
// the httpOnly cookie is never readable from browser JS, and pages/API are
// blocked until a valid session cookie is present.
const SESSION = process.env.SESSION_TOKEN || "";

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Always-allowed paths: the login screen, its auth endpoints, and assets.
  if (
    pathname === "/login" ||
    pathname.startsWith("/api/login") ||
    pathname.startsWith("/api/logout") ||
    pathname.startsWith("/_next") ||
    pathname === "/icon.svg" ||
    pathname === "/favicon.ico" ||
    pathname === "/robots.txt"
  ) {
    return NextResponse.next();
  }

  const token = req.cookies.get("nbn_session")?.value;
  if (SESSION && token === SESSION) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/")) {
    return new NextResponse(JSON.stringify({ error: "unauthorized" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });
  }

  const url = req.nextUrl.clone();
  url.pathname = "/login";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
