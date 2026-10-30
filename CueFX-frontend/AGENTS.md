# CueFX frontend project rules

## Product and design reference

- Build CueFX as a real, responsive web application using Next.js App Router, React, TypeScript, and Tailwind CSS.
- `CueFX Screens.html` is a visual reference only. Do not embed it, use it in an iframe, copy its screen markup into the application, or make the UI behave like a static screenshot. Recreate its visual language with native components, layout, styles, and browser behavior.
- Preserve the visible five-step navigation as Footage (1) → Attach (2) → Direct (3) → Look (4) → Review (5). The HTML reference contains a Creating screen, but it is a transient render state and is not a numbered navigation step.
- Match the reference design system: Geist at 14px/1.4, #09090B canvas, #0C0C0E 56px header, #FF7A45 accent, #8C8C96 muted text, 720px Footage column, 48px page inset, 28px section rhythm, and 200px upload drop zone. The header logo is the two-arc orange/white mark with 15px bold CueFX wordmark; navigation remains numbered and uses 12.5px labels. Reuse these shared tokens and primitives across screens.
- Keep the dark studio appearance, restrained warm orange accent, compact labels, rounded controls, and high-density editor layout. Extract tokens into shared styles rather than duplicating one-off values.
- Keep sample footage and fonts as ordinary static assets in `public/`; do not ship copied HTML screens as routes or assets.

## Application architecture

- Use the App Router under `src/app` and keep route entry points small. Put interactive product features in client components only when they need browser APIs, event handlers, or local state.
- Type components, props, event handlers, state, and domain data. Avoid `any`, non-null assertions without a proven invariant, and untyped DOM queries.
- Split substantial workflows into focused components and reusable controls. Keep domain state and transitions explicit; avoid hiding application logic inside styling or a monolithic page component.
- Use semantic HTML and native controls where possible. Every interactive control needs an accessible name, keyboard operation, visible focus, and a meaningful disabled state.
- Treat uploaded media as local browser data unless a real API is added. Validate format, size, and duration; report validation errors where the user can see them; revoke object URLs when replaced or no longer needed.
- Do not imply that cloud processing, persistence, audio export, or AI generation exists unless implemented by a configured service. Label local previews and simulated progress accurately.

## Next.js and TypeScript

- Follow App Router conventions, server/client component boundaries, metadata placement, and Next.js supported APIs for the installed version.
- Keep browser-only work inside client components and effects or event handlers. Avoid reading `window`, `document`, `File`, or `URL.createObjectURL` during server rendering.
- Keep strict TypeScript enabled. Resolve type errors at the source; do not disable checks to make a build pass.
- Keep dependencies and lockfile in sync. Prefer existing dependencies and platform APIs over adding packages for small interactions.

## Tailwind CSS and styling

- Tailwind CSS v4 is configured through `@import "tailwindcss"` and `@tailwindcss/postcss`; keep that setup consistent unless the framework version is deliberately changed.
- Use Tailwind utilities for straightforward layout and component styling. Use `globals.css` for design tokens, shared primitives, complex editor visuals, animations, and responsive rules that are clearer as CSS.
- Reuse shared tokens for color, spacing, radius, borders, and typography. Avoid arbitrary values when an established token or component class already exists.
- Verify that long editor panels, dialogs, toolbars, and upload controls remain usable on narrow screens and at reduced viewport heights. Do not let fixed-height layouts clip essential actions.
- Respect reduced-motion preferences and maintain sufficient contrast for labels, focus rings, status text, and disabled controls.

## CueFX interaction rules

- The initial screen is Footage. Choosing a valid upload or sample enables Continue to Attach; completing tracking there enables Continue to Direct; Direct continues to Look; Look approval and valid prompt/mode unlock Generate; generation completion leads to Review.
- Buttons, toggles, sliders, menus, uploads, keyboard shortcuts, undo/redo, timeline scrubbing, and navigation must perform their visible action. Do not leave decorative controls looking enabled when they do nothing.
- Preserve edits when navigating backward. Reset approval when Direct settings change because the approved composite is stale.
- Prompt and reference-image controls must validate, reflect their current state, and gate generation correctly. File inputs should support both browse and drag/drop when presented that way.
- Make loading, success, stopped, error, empty, and disabled states visible and recoverable.
- Avoid hard-coding misleading status claims. A simulated workflow must say it is a local preview or stand-in until connected to a real processing service.

## Change verification

- For implementation changes, run `npm exec tsc -- --noEmit` and `npm run build` when the environment permits. Report environment failures distinctly from source failures.
- Perform a visual pass at desktop and narrow viewport sizes for layout changes. Check the end-to-end screen progression and key controls when interactions change.
- Do not add test frameworks or tests unless requested; use existing checks and targeted manual verification.
