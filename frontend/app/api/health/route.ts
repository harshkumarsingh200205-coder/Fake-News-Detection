import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/backend";

export async function GET() {
  try {
    const response = await fetchBackend("/health", { method: "GET" });
    const data = await response.json().catch(() => null);

    if (!response.ok)
      return NextResponse.json(
        data ?? { status: "unhealthy", backend: "down" },
        { status: 503 },
      );
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: "unhealthy", backend: "unreachable" },
      { status: 503 },
    );
  }
}
