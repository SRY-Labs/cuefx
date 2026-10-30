# CueFX Full-Stack Engineering Rules

These rules govern changes that connect or extend the CueFX Next.js frontend and the existing Python FastAPI backend. The backend repository and its files are read-only for frontend integration work unless the user explicitly changes that scope.

## 1. Goals and source of truth

- Preserve the current five-step studio flow, visual system, and existing browser-only editing behavior while adding real backend operations.
- Treat the FastAPI routes and `contract/` schemas in CueFX-backend as the source of truth. Do not invent endpoints, fields, enum values, response shapes, or success semantics.
- Keep UI entry points and existing controls working. Backend failures must show a recoverable message and must never be presented as successful analysis, rendering, or generation.
- Mark any feature that remains local, a stand-in, manual, experimental, or unavailable accurately in the UI.
- Read this file, `AGENTS.md`, and `.agents/skills/cuefx-nextjs-tailwind/SKILL.md` before making frontend changes.

## 2. Repository boundaries

- Do not edit, reformat, regenerate, or commit files in CueFX-backend during frontend integration.
- Keep API transport, request/response types, and response parsing in focused `src/lib` modules. Keep the App Router route handler small and focused on the backend proxy.
- Keep screen components responsible for presentation and user interaction. Put backend orchestration and API mapping in typed helpers or hooks when it becomes substantial.
- Do not add dependencies unless the existing platform APIs cannot safely meet a concrete requirement. Keep `package.json` and its lockfile synchronized if a dependency is necessary.

## 3. Backend connection and ports

- In Azure Container Apps, set the frontend's server-only environment variable `CUEFX_BACKEND_URL` to the backend origin, for example `https://cuefx-backend.graypond-7635469c.southindia.azurecontainerapps.io`.
- Do not prefix the backend origin with `NEXT_PUBLIC_`. The browser calls the frontend's same-origin `/api/backend/...` route; the Next.js server forwards it to FastAPI.
- Do not bake deployment URLs, API tokens, passwords, or credentials into the Docker image, frontend bundle, source code, or build arguments. The backend origin is public configuration, not a secret; still read it from the runtime environment so deployment changes do not require a frontend rebuild.
- The frontend container listens on `0.0.0.0:3000`; set frontend ingress target port to `3000`. The backend container listens on `0.0.0.0:8000`; set backend ingress target port to `8000`.
- Server-to-server proxying avoids browser CORS. If browser code is ever changed to call the backend origin directly, configure exact allowed origins; never use wildcard origins with credentials.
- Validate the configured backend URL at runtime: allow only HTTP(S), require HTTPS in production, and reject user information, query strings, and fragments. Do not permit a request path to replace the configured host.

## 4. Secrets and security

- `CUEFX_BACKEND_URL` is not a credential. It must not contain embedded username/password or tokens.
- Store backend credentials, ComfyUI tokens, signing keys, or other secrets as Azure Container App secrets and reference them only in backend server-side environment variables. Never expose secrets with `NEXT_PUBLIC_*`, HTML, client JavaScript, URLs, logs, browser storage, source control, screenshots, or error messages.
- Never forward browser `Authorization`, `Cookie`, or arbitrary headers through the backend proxy. Forward only the headers required by the CueFX contract (such as content type, accept, and byte ranges), and return only required response headers.
- Restrict the proxy to the configured backend host, `/api/` routes, and supported HTTP methods. Do not create an open proxy or accept client-supplied upstream URLs.
- A same-origin proxy is a transport boundary, not user authentication. The current backend contract has no authentication or authorization. If the apps are publicly reachable, anyone can call the backend directly or use the proxy; protect public production access at the Azure ingress/identity layer and prevent direct backend bypass before exposing costly generation.
- Validate uploads on the client for usability and rely on backend validation as the security boundary. Do not trust client MIME types, filenames, IDs, sizes, timing, or cue values.
- Use same-origin credentials policy for the frontend proxy; do not introduce cross-origin cookies or credentials for an unauthenticated API.
- Avoid logging request bodies, uploaded media, prompts with sensitive content, or secret-bearing headers. Return safe actionable errors and keep internal traces server-side.
- Preserve streaming for large uploads and video responses. Do not buffer 500 MB uploads or full video responses in memory.
- Add request timeouts/cancellation and cleanup for polling, object URLs, timers, and media elements. Poll job/stand-in status at a bounded interval and stop after terminal status or user cancellation.

## 5. API contract and workflow mapping

- Use only endpoints defined in the backend's current `contract/HTTP.md`, schemas, and live `/api/health` feature flags.
- Upload clips through `POST /api/clips` as multipart field `file`; retain the returned `clip_id` and use backend video/frame routes for backend-derived media.
- Analyze through `POST /api/analyze` or `POST /api/cue/new` using the required clip ID, body point, and seed units/frames exactly as documented. Convert UI-relative positions into conformed plate pixels using the returned clip transform/plate dimensions.
- Build and normalize cues through `/api/cue/new` and `/api/cue/normalize`; send cue and track to `/api/validate` and `/api/preview` as their schemas require. Do not send unknown fields.
- Render approval composites through `POST /api/standin`, poll `GET /api/standin/{standin_id}` until done/failed, and use the returned result/file contract. Do not treat HTTP 202 as completion.
- Call `/api/generate` only when `/api/health` says the feature is available, and only after the matching stand-in has completed and the approved cue/prompt is valid. Poll job status; handle `queued`, `running`, `done`, `failed`, and `cancelled` distinctly. Do not simulate completion for cloud jobs.
- The backend documents reference-image upload as unsupported. Do not claim the image is used by backend generation unless the contract gains an upload route and the integration implements it.
- If automatic generation is unavailable, clearly explain that configuration requirement or provide a contract-supported manual route. Do not label a stand-in or local preview as an AI-generated final render.
- Respect backend limits: supported video extensions, upload limit, 1440 conformed frames, 8-second effect window, 24 fps plate coordinates, and 202 asynchronous status responses. Surface backend error messages safely.

## 6. Next.js, React, TypeScript, and Tailwind

- Use Next.js App Router conventions. Keep client components for browser state and APIs; read `CUEFX_BACKEND_URL` only in server code.
- Keep strict TypeScript enabled. Define explicit request and response types from backend schemas; avoid `any`, unsafe assertions, and duplicated implicit API shapes.
- Use `fetch` with same-origin paths, no-store for job state, and `AbortSignal` for cancellation. Parse non-2xx error responses safely and distinguish network errors from backend errors.
- Preserve the installed Tailwind CSS v4 setup and the existing CueFX design tokens. Use native controls and semantic accessible labels, keyboard support, focus indication, and reduced-motion behavior.
- Keep progress, busy, retry, cancellation, and error states visible and accurate. Prevent duplicate submits while an operation is active.

## 7. Deployment and configuration

- The frontend Docker image must continue to bind `0.0.0.0:3000`, expose port `3000`, and run as the existing unprivileged user.
- Configure `CUEFX_BACKEND_URL` as a runtime environment variable in the frontend Container App. This server-only variable does not require embedding the backend URL into the client bundle.
- Configure the backend's `PORT=8000` and ingress target port `8000`. Keep backend data/replica assumptions aligned with the backend README; frontend changes must not alter them.
- Review the Azure deployment workflow before changing build arguments. Do not put secrets in workflow YAML; use GitHub Actions secrets/OIDC as currently configured.
- Keep `.env.example` limited to non-secret local configuration. Never commit real `.env` files.

## 8. Verification and release readiness

- Run `npm exec tsc -- --noEmit` and `npm run build` after implementation changes when the environment permits; report infrastructure or permission failures separately from code errors.
- Check the actual frontend and backend health endpoints, including `/api/health` and the reported feature flags, before claiming deployment readiness.
- Exercise the end-to-end path: upload → analyze → cue validation/preview → stand-in render → conditional generation → result playback/download. Test cancellation and backend error paths as well as success.
- Check browser network requests to confirm they use same-origin `/api/backend/` and disclose no secret values.
- Verify both desktop and narrow screen layouts after UI changes. Do not claim the apps are fully connected based only on successful URL configuration or a successful health check.
