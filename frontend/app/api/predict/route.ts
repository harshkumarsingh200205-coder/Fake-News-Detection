import { NextRequest, NextResponse } from "next/server";
import { fetchBackend } from "@/lib/backend";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const response = await fetchBackend("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: body.text,
        return_keywords: true,
        top_keywords: 10,
      }),
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
