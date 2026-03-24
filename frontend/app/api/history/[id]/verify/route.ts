import { NextRequest, NextResponse } from "next/server";
import { fetchBackend } from "@/lib/backend";

type RouteContext = {
  params: Promise<{
    id: string;
  }>;
};

export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { id } = await context.params;
    const body = await request.json();
    const response = await fetchBackend(`/history/${id}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json(data ?? { error: "Backend error" }, {
        status: response.status,
      });
    }

    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to connect to prediction service" },
      { status: 500 },
    );
  }
}
