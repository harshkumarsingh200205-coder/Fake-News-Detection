const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

export const BACKEND_URL = (
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  DEFAULT_BACKEND_URL
).replace(/\/$/, "");

export const buildBackendUrl = (path: string) => {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${BACKEND_URL}${normalizedPath}`;
};

export const fetchBackend = (path: string, init?: RequestInit) =>
  fetch(buildBackendUrl(path), {
    cache: "no-store",
    ...init,
  });
