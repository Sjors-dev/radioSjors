import { NextRequest, NextResponse } from "next/server";

import { COOKIE, isUnlocked } from "./lib/auth";

const OPEN_PATHS = ["/login", "/api/login"];

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (OPEN_PATHS.some((path) => pathname.startsWith(path))) {
    return NextResponse.next();
  }
  if (await isUnlocked(request.cookies.get(COOKIE)?.value)) {
    return NextResponse.next();
  }

  // The data route answers with JSON, so redirecting it would hand a fetch()
  // an HTML login page and a confusing parse error.
  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "locked" }, { status: 401 });
  }

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|icon.svg|.*\\.(?:svg|png|jpg|jpeg|gif|webp|woff2?)$).*)",
  ],
};
