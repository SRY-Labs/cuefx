/** Browser-side client for the same-origin API proxy. */
const PROXY_ROOT = "/api/backend";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export type BackendClip = {
  clip_id: string;
  clip: string;
  fps: number;
  frames: number;
  plate_size: [number, number];
};

export type BackendCue = Record<string, unknown> & {
  clip: string;
  fps: number;
  frames: number;
  plate_size: [number, number];
  track: Record<string, unknown>[];
};

export type BackendHealth = {
  status: string;
  features?: { ltx_overlap_generate?: boolean };
};

export type BackendJob = {
  id: string;
  status: "queued" | "waiting" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  stage_name: string;
  elapsed_s: number;
  error: { code: string; message: string } | null;
  files?: { final_preview?: string; final_download?: string };
};

export type BackendStandinStatus = {
  standin_id: string;
  status: "queued" | "running" | "done" | "failed";
  error: { code: string; message: string } | null;
  result: { files: { hardcomp_preview: string; hardcomp_download: string } } | null;
};

export function apiUrl(path: string): string {
  const clean = path.replace(/^\/+/, "");
  if (!clean.startsWith("api/")) throw new Error("Backend paths must start with /api/.");
  return `${PROXY_ROOT}/${clean}`;
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    credentials: "same-origin",
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `Backend request failed (${response.status}).`;
    let code: string | undefined;
    try {
      const error = await response.json() as { code?: string; message?: string; detail?: string };
      message = error.message ?? error.detail ?? message;
      code = error.code;
    } catch {
      // Keep the safe status-based message when the response isn't JSON.
    }
    throw new ApiError(message, response.status, code);
  }

  return await response.json() as T;
}

export async function apiUpload<T>(path: string, form: FormData): Promise<T> {
  return apiRequest<T>(path, { method: "POST", body: form });
}

/** Stream a clip through the same-origin proxy and report actual browser upload progress. */
export function uploadClip(file: File, onProgress: (progress: number) => void): Promise<BackendClip> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("api/clips"));
    xhr.responseType = "json";
    xhr.timeout = 15 * 60 * 1000;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.min(99, Math.round(event.loaded / event.total * 100)));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
        onProgress(100);
        resolve(xhr.response as BackendClip);
        return;
      }
      const body = xhr.response as { message?: string; detail?: string; code?: string } | null;
      reject(new ApiError(body?.message ?? body?.detail ?? `Clip upload failed (${xhr.status || "unknown"}).`, xhr.status, body?.code));
    };
    xhr.onerror = () => reject(new ApiError("Could not reach the backend. Check the connection and try again.", 0));
    xhr.ontimeout = () => reject(new ApiError("The upload took too long. Check the backend and try again.", 0, "timeout"));
    xhr.onabort = () => reject(new ApiError("Clip upload was cancelled.", 0, "cancelled"));
    xhr.send(form);
  });
}
