import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/backend";

export async function GET() {
  try {
    const response = await fetchBackend("/training/stats", {
      method: "GET",
    });
    const data = await response.json().catch(() => null);

    if (!response.ok)
      return NextResponse.json(data ?? { error: "Backend error" }, {
        status: response.status,
      });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to connect to prediction service" },
      { status: 500 },
    );
  }
}

export async function POST() {
  try {
    const response = await fetchBackend("/retrain", {
      method: "POST",
    });
    const data = await response.json().catch(() => null);

    if (!response.ok)
      return NextResponse.json(data ?? { error: "Backend error" }, {
        status: response.status,
      });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to connect to prediction service" },
      { status: 500 },
    );
  }
}
