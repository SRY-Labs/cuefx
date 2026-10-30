---
name: cuefx-nextjs-tailwind
description: Build or change the CueFX studio frontend with Next.js App Router, React, TypeScript, and Tailwind CSS while preserving the native five-step editing workflow and its visual system.
---

# CueFX Next.js and Tailwind workflow

Use this skill when implementing or modifying the CueFX frontend in this repository. Read the repository's `AGENTS.md` for the project-wide rules; this skill adds the workflow-specific details.

## Source of truth

- Treat `CueFX Screens.html` as a visual/design-system reference only. Inspect it for visual hierarchy, spacing, colors, control density, and screen sequence. Never embed or serve its page markup as the app UI.
- Build the experience from native Next.js routes, React components, CSS, and ordinary image/font assets. A screenshot or iframe is not a functional implementation.
- Preserve the visible five-step project flow: Footage (1) → Attach (2) → Direct (3) → Look (4) → Review (5). Creating is a transient render state, not a numbered navigation step.
- Match the shared visual tokens in `AGENTS.md`: Geist, a 56px header, numbered 12.5px step navigation, #09090B canvas, #FF7A45 accent, and the reference screen’s exact density and spacing. Footage uses a 720px content column, 48px page inset, 28px vertical rhythm, and a 200px upload drop zone.

## Implementation approach

1. Inspect the current app structure, package versions, local conventions, and relevant screen components before editing.
2. Identify the user-visible state and transitions first. Define typed data and handlers for each control, then implement the UI around those behaviors.
3. Keep route entry points small. Use client components for stateful studio interactions and isolate large screens, dialogs, and reusable controls into focused components.
4. Extend shared visual tokens and primitives before adding local styles. Use Tailwind for common layout and `globals.css` for shared design tokens, complex editor visuals, and responsive behavior.
5. Keep all browser-only APIs on the client. Validate local media before using it, present understandable errors, and clean up object URLs and timers.
6. Ensure navigation does not discard edits unexpectedly. Invalidate approval when its underlying Direct settings change; gate later actions until their prerequisites are met.
7. Make every enabled button, tab, slider, menu, keyboard shortcut, upload area, and action meaningful. Add accessible names, focus states, keyboard support, and accurate disabled states.
8. Be explicit when a workflow is simulated or local-only. Do not present a preview as a generated deliverable or imply cloud upload/persistence without a configured backend.

## CueFX behavior checklist

- Footage: valid MP4/MOV upload, drag/drop and browse, size/duration validation, progress and errors, selectable sample clips, and Continue to Attach.
- Attach: six tracking targets (Palm, Fist, Foot, Knee, Shoulder, Head), Left/Right where applicable, palm click-to-seed, find/track progress, uncertainty markers, retry, and gated continuation to Direct.
- Direct: preview playback, timeline and frame movement, effect geometry/color/motion/timing controls, path editing, toggles, undo/redo, shortcut help, render list, and continue navigation.
- Look: wait/retry/stop controls, composite playback, approval state, prompt editing, surprise mode, reference image browse/drop/replace/remove and validation, and correctly gated Generate.
- Creating: clear progress and stage, stop/retry/back actions, and a completion transition to Review.
- Review: preview, export with a clear format/capability disclosure, edit navigation, and try-another-look navigation.
- Responsive: verify editor controls, scroll regions, dialogs, upload, and footer actions at desktop and narrow viewport sizes.

## Verification

- Run `npm exec tsc -- --noEmit` and `npm run build` when available; distinguish sandbox/tooling errors from application errors.
- For visual changes, inspect the rendered app and check desktop and narrow layouts. For workflow changes, move through each step and exercise the changed controls.
- Do not add tests or new dependencies without a task need. Preserve the lockfile when changing dependencies.
