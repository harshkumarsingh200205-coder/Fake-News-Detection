import { NextRequest, NextResponse } from "next/server";
import { fetchBackend } from "@/lib/backend";

export async function GET(request: NextRequest) {
  try {
    const limit = request.nextUrl.searchParams.get("limit") || "20";
    const response = await fetchBackend(`/history?limit=${limit}`, {
      method: "GET",
    });
    const data = await response.json().catch(() => null);

    if (!response.ok)
      return NextResponse.json(
        data ?? { error: "Backend error" },
        { status: response.status },
      );
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to connect to prediction service" },
      { status: 500 },
    );
  }
}
