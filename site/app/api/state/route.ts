import { NextResponse } from "next/server";

import { readStation } from "@/lib/station";

// What is on the radio right now is the one thing that must never be cached.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  const result = await readStation();
  if (!result.ok) {
    return NextResponse.json({ error: result.error }, { status: 502 });
  }
  return NextResponse.json(result.state, {
    headers: { "Cache-Control": "no-store, max-age=0" },
  });
}
