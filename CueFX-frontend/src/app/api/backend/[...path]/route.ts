import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ path: string[] }> };

const requestHeaderNames = ["accept", "content-type", "range", "if-range"] as const;
const responseHeaderNames = [
  "accept-ranges", "cache-control", "content-disposition", "content-length", "content-range",
  "content-type", "x-cuefx-effect-range", "x-cuefx-job-id", "x-cuefx-warnings",
] as const;

function backendOrigin(): URL | null {
  const raw = process.env.CUEFX_BACKEND_URL;
  if (!raw) return null;
  try {
    const url = new URL(raw);
    if (!(["http:", "https:"].includes(url.protocol)) || url.username || url.password || url.search || url.hash) return null;
    if (process.env.NODE_ENV === "production" && url.protocol !== "https:") return null;
    return url;
  } catch {
    return null;
  }
}

async function proxy(request: NextRequest, context: RouteContext): Promise<Response> {
  if (request.method !== "GET" && request.method !== "POST" && request.method !== "HEAD") {
    return NextResponse.json({ code: "method_not_allowed", message: "This API method is not available." }, { status: 405 });
  }

  const origin = backendOrigin();
  if (!origin) {
    return NextResponse.json({ code: "backend_not_configured", message: "The backend connection is not configured." }, { status: 503 });
  }

  const { path } = await context.params;
  if (!path.length || path.some((segment) => !/^[A-Za-z0-9._-]+$/.test(segment) || segment === "." || segment === "..")) {
    return NextResponse.json({ code: "invalid_path", message: "The backend API path is invalid." }, { status: 400 });
  }

  // The catch-all path already includes the leading `api` segment because
  // callers use /api/backend/api/.... Do not prepend a second /api here.
  const target = new URL(`/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`, origin);
  const headers = new Headers();
  for (const name of requestHeaderNames) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    cache: "no-store",
    signal: request.signal,
  };
  if (request.method === "POST") {
    init.body = request.body;
    init.duplex = "half";
  }

  try {
    const upstream = await fetch(target, init);
    const responseHeaders = new Headers();
    for (const name of responseHeaderNames) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set("x-content-type-options", "nosniff");
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return NextResponse.json({ code: "backend_unavailable", message: "The backend could not be reached. Please try again." }, { status: 502 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const HEAD = proxy;
