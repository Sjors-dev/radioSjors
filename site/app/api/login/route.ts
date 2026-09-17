import { NextRequest, NextResponse } from "next/server";

import { COOKIE, passwordIsConfigured, tokenFor } from "@/lib/auth";

export async function POST(request: NextRequest) {
  if (!passwordIsConfigured()) {
    return NextResponse.json(
      { error: "SITE_PASSWORD is not set on this deployment." },
      { status: 503 },
    );
  }

  let password = "";
  try {
    password = String(((await request.json()) as { password?: string }).password ?? "");
  } catch {
    return NextResponse.json({ error: "Malformed request." }, { status: 400 });
  }

  if (password !== process.env.SITE_PASSWORD) {
    return NextResponse.json({ error: "Wrong password." }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: COOKIE,
    value: await tokenFor(password),
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 365,
  });
  return response;
}
