"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type CSSProperties,
} from "react";
import { apiRequest, apiUrl, uploadClip, type BackendClip, type BackendCue, type BackendHealth, type BackendJob, type BackendStandinStatus } from "../lib/api";

type Screen = "footage" | "attach" | "direct" | "look" | "creating" | "review";
type BodyPoint = "palm" | "fist" | "foot" | "knee" | "shoulder" | "head";
type Motion = "Hold" | "Hold + throw" | "Charge-up" | "Orbit";
type Shape = "Sphere" | "Ring" | "Crystal" | "Cube";
type Ending = "Fade" | "Burst" | "Shatter";
type Clip = { name: string; duration: number; url: string; isVideo: boolean; file?: File; backend?: BackendClip };
type Point = { x: number; y: number; easing?: "linear" | "in" | "out" | "inout"; size?: number | null };
type Fix = { id: number; time: number; x: number; y: number };
type BehindSegment = { id: number; side: "left" | "right"; start: number; end: number };
type DirectSettings = {
  shape: Shape;
  color: string;
  glow: number;
  size: number;
  motion: Motion;
  points: Point[];
  distance: number;
  speed: number;
  tilt: number;
  ending: Ending;
  endingLength: number;
  spin: number;
  offsetX: number;
  offsetY: number;
  fadeIn: number;
  appearAt: number;
  releaseAt: number;
  holdEnd: number;
  fixes: Fix[];
  behind: BehindSegment[];
};

const EDITOR_WIDTH = 1280;
const EDITOR_HEIGHT = 720;

function coverScale(width: number, height: number): number {
  return Math.max(EDITOR_WIDTH / Math.max(1, width), EDITOR_HEIGHT / Math.max(1, height));
}

/** Map source-frame pixels into the visible 16:9 editor, matching object-fit: cover. */
function frameToEditorPercent(x: number, y: number, width: number, height: number): Point {
  const scale = coverScale(width, height);
  const renderedWidth = width * scale;
  const renderedHeight = height * scale;
  return {
    x: ((x * scale + (EDITOR_WIDTH - renderedWidth) / 2) / EDITOR_WIDTH) * 100,
    y: ((y * scale + (EDITOR_HEIGHT - renderedHeight) / 2) / EDITOR_HEIGHT) * 100,
  };
}

/** Map an editor position back into source-frame pixels, excluding the cropped area. */
function editorPercentToFrame(x: number, y: number, width: number, height: number): Point {
  const scale = coverScale(width, height);
  const renderedWidth = width * scale;
  const renderedHeight = height * scale;
  return {
    x: (x * EDITOR_WIDTH / 100 - (EDITOR_WIDTH - renderedWidth) / 2) / scale,
    y: (y * EDITOR_HEIGHT / 100 - (EDITOR_HEIGHT - renderedHeight) / 2) / scale,
  };
}

/** Interpolate nearby valid backend samples; do not teleport to a distant sample across a lost section. */
function trackPointAt(track: Record<string, unknown>[] | null, frame: number, fps: number): Point | null {
  if (!track?.length) return null;
  let before: Record<string, unknown> | null = null;
  let after: Record<string, unknown> | null = null;
  for (const sample of track) {
    if (sample.lost === true || typeof sample.f !== "number" || typeof sample.x !== "number" || typeof sample.y !== "number") continue;
    if (sample.f <= frame) before = sample;
    if (sample.f >= frame) { after = sample; break; }
  }
  const maxGap = Math.max(2, fps * 0.25);
  if (before && after) {
    const gap = (after.f as number) - (before.f as number);
    if (gap <= maxGap) {
      const t = gap <= 0 ? 0 : Math.max(0, Math.min(1, (frame - (before.f as number)) / gap));
      return { x: (before.x as number) + ((after.x as number) - (before.x as number)) * t, y: (before.y as number) + ((after.y as number) - (before.y as number)) * t };
    }
  }
  const nearest = !before ? after : !after ? before : Math.abs((before.f as number) - frame) <= Math.abs((after.f as number) - frame) ? before : after;
  if (!nearest || Math.abs((nearest.f as number) - frame) > maxGap / 2) return null;
  return { x: nearest.x as number, y: nearest.y as number };
}

const initialDirect: DirectSettings = {
  shape: "Sphere",
  color: "#9FD4FF",
  glow: 60,
  size: 99,
  motion: "Hold + throw",
  points: [
    { x: 89, y: 17 }, { x: 93, y: 26 }, { x: 95, y: 44 }, { x: 93, y: 58 },
    { x: 80, y: 70 }, { x: 65, y: 74 }, { x: 47, y: 62 }, { x: 31, y: 65 }, { x: 25, y: 74 },
  ],
  distance: 1.5,
  speed: 180,
  tilt: 60,
  ending: "Fade",
  endingLength: 0.25,
  spin: 288,
  offsetX: 0,
  offsetY: -64,
  fadeIn: 0.5,
  appearAt: 1.875,
  releaseAt: 3.3333,
  holdEnd: 3.6667,
  fixes: [],
  behind: [],
};

function buildRenderCue(base: BackendCue, settings: DirectSettings, prompt: string): BackendCue {
  const fps = base.fps;
  const frames = base.frames;
  const [width, height] = base.plate_size;
  const scale = coverScale(width, height);
  const frameAt = (seconds: number, allowEnd = false) => Math.max(0, Math.min(allowEnd ? frames : frames - 1, Math.round(seconds * fps)));
  const isThrow = settings.motion === "Hold + throw" || settings.motion === "Charge-up";
  const behaviour = settings.motion === "Orbit" ? "orbit" : settings.motion === "Hold" ? "hold" : settings.motion === "Charge-up" ? "charge_up" : "hold_throw";
  const releaseAt = frameAt(settings.releaseAt);
  const diameter = settings.size / scale;
  const [endStartSeconds, endSeconds] = settings.motion === "Hold + throw"
    ? [settings.releaseAt + Math.max(0, settings.points.length - 1) / fps, settings.releaseAt + Math.max(0, settings.points.length - 1) / fps + settings.endingLength]
    : [settings.holdEnd, settings.holdEnd + settings.endingLength];
  const easingMap = { linear: "linear", in: "ease_in", out: "ease_out", inout: "ease_in_out" } as const;
  const route = isThrow ? settings.points.map((point, index) => ({
    f: Math.min(frames - 1, releaseAt + index),
    x: index === 0 ? null : editorPercentToFrame(point.x, point.y, width, height).x,
    y: index === 0 ? null : editorPercentToFrame(point.x, point.y, width, height).y,
    diameter_px: (point.size ?? settings.size) / scale,
    easing_in: easingMap[point.easing ?? "linear"],
  })) : [];
  const trackAt = (frame: number) => base.track.reduce<Record<string, unknown> | null>((nearest, item) => {
    if (typeof item.f !== "number") return nearest;
    if (!nearest || Math.abs((nearest.f as number) - frame) > Math.abs(item.f - frame)) return item;
    return nearest;
  }, null);
  const seenFixFrames = new Set<number>();
  const corrections = settings.fixes.flatMap((fix) => {
    const f = frameAt(fix.time);
    if (seenFixFrames.has(f)) return [];
    seenFixFrames.add(f);
    const tracked = trackAt(f);
    const x = typeof tracked?.x === "number" ? tracked.x : 0;
    const y = typeof tracked?.y === "number" ? tracked.y : 0;
    return [{ f, x: x + fix.x / scale, y: y + fix.y / scale }];
  });
  const behind = settings.behind.map((item) => ({
    start: frameAt(item.start), end: frameAt(item.end, true),
    occluder: item.side === "left" ? "left_hand" : "right_hand",
  }));
  const baseGrade = typeof base.grade === "object" && base.grade !== null ? base.grade as Record<string, unknown> : {};
  const cue: BackendCue = {
    ...base,
    behaviour,
    shape: settings.shape.toLowerCase(),
    spin_deg_per_s: settings.spin,
    ending: settings.ending.toLowerCase(),
    appear_frame: frameAt(settings.appearAt),
    release_frame: behaviour === "hold" || behaviour === "orbit" ? null : releaseAt,
    route,
    vanish: { start: frameAt(endStartSeconds), end: frameAt(endSeconds, true) },
    diameter_px: diameter,
    anchor_offset: [settings.offsetX / scale, settings.offsetY / scale],
    corrections,
    behind,
    prompt: prompt.trim() || null,
    grade: { ...baseGrade, tint: settings.color, glow: settings.glow / 100, fade_in_frames: Math.max(0, Math.round(settings.fadeIn * fps)) },
    orbit: behaviour === "orbit" ? { radius_sizes: settings.distance, speed_deg_per_s: settings.speed, tilt_deg: settings.tilt } : base.orbit,
  };
  if (behaviour !== "orbit") delete cue.orbit;
  return cue;
}

function directSettingsFromCue(cue: BackendCue): DirectSettings {
  const [width, height] = cue.plate_size;
  const scale = coverScale(width, height);
  const anchorOffset = Array.isArray(cue.anchor_offset) && cue.anchor_offset.length === 2
    ? cue.anchor_offset.map((value) => typeof value === "number" ? value : 0)
    : [0, 0];
  const releaseFrame = typeof cue.release_frame === "number" ? cue.release_frame : Math.round(cue.frames * 0.6);
  const target = trackPointAt(cue.track, releaseFrame, cue.fps);
  const held = target ? frameToEditorPercent(target.x + anchorOffset[0], target.y + anchorOffset[1], width, height) : { x: 89, y: 23 };
  const heldX = held.x;
  const heldY = held.y;
  const dx = heldX - initialDirect.points[0].x;
  const dy = heldY - initialDirect.points[0].y;
  const grade = typeof cue.grade === "object" && cue.grade !== null ? cue.grade as Record<string, unknown> : {};
  const orbit = typeof cue.orbit === "object" && cue.orbit !== null ? cue.orbit as Record<string, unknown> : {};
  const vanish = typeof cue.vanish === "object" && cue.vanish !== null ? cue.vanish as Record<string, unknown> : {};
  const timing = (value: unknown, fallback: number) => typeof value === "number" ? value / cue.fps : fallback;
  const motion: Motion = cue.behaviour === "orbit" ? "Orbit" : cue.behaviour === "charge_up" ? "Charge-up" : cue.behaviour === "hold" ? "Hold" : "Hold + throw";
  const shape: Shape = cue.shape === "ring" ? "Ring" : cue.shape === "crystal" ? "Crystal" : cue.shape === "cube" ? "Cube" : "Sphere";
  const ending: Ending = cue.ending === "burst" ? "Burst" : cue.ending === "shatter" ? "Shatter" : "Fade";
  const point = (key: string, fallback: number) => typeof orbit[key] === "number" ? orbit[key] as number : fallback;
  return {
    ...initialDirect,
    motion,
    shape,
    ending,
    color: typeof grade.tint === "string" ? grade.tint : initialDirect.color,
    glow: typeof grade.glow === "number" ? Math.max(0, Math.min(100, grade.glow * 100)) : initialDirect.glow,
    size: typeof cue.diameter_px === "number" ? cue.diameter_px * scale : initialDirect.size,
    spin: typeof cue.spin_deg_per_s === "number" ? cue.spin_deg_per_s : initialDirect.spin,
    offsetX: anchorOffset[0] * scale,
    offsetY: anchorOffset[1] * scale,
    appearAt: timing(cue.appear_frame, initialDirect.appearAt),
    releaseAt: timing(cue.release_frame, initialDirect.releaseAt),
    holdEnd: timing(vanish.start, initialDirect.holdEnd),
    endingLength: typeof vanish.start === "number" && typeof vanish.end === "number" ? (vanish.end - vanish.start) / cue.fps : initialDirect.endingLength,
    distance: point("radius_sizes", initialDirect.distance),
    speed: point("speed_deg_per_s", initialDirect.speed),
    tilt: point("tilt_deg", initialDirect.tilt),
    points: initialDirect.points.map((item) => ({ ...item, x: Math.max(-20, Math.min(120, item.x + dx)), y: Math.max(-20, Math.min(120, item.y + dy)) })),
  };
}

const flow = [
  { id: "footage" as const, step: "1", label: "Footage" },
  { id: "attach" as const, step: "2", label: "Attach" },
  { id: "direct" as const, step: "3", label: "Direct" },
  { id: "look" as const, step: "4", label: "Look" },
  { id: "review" as const, step: "5", label: "Review" },
];

const bodyPoints: { id: BodyPoint; label: string }[] = [
  { id: "palm", label: "Palm" },
  { id: "fist", label: "Fist" },
  { id: "foot", label: "Foot" },
  { id: "knee", label: "Knee" },
  { id: "shoulder", label: "Shoulder" },
  { id: "head", label: "Head" },
];

const colors = [
  { name: "Ice blue", hex: "#9FD4FF" },
  { name: "Ember", hex: "#FF8A4C" },
  { name: "Violet", hex: "#B79BFF" },
  { name: "Emerald", hex: "#6FE3A6" },
  { name: "White", hex: "#F4F4F5" },
];

const sampleClips = [
  { name: "Bundled hand sample", meta: "4s test video · still image, no motion", duration: 4 },
];

/** Turn the bundled still into a short browser-recorded clip for backend upload testing. */
async function createBundledSampleVideo(durationSeconds: number): Promise<File> {
  if (typeof MediaRecorder === "undefined" || !("captureStream" in HTMLCanvasElement.prototype)) {
    throw new Error("This browser cannot create the test video. Please upload an MP4, MOV, MKV, or WebM file instead.");
  }

  const image = new Image();
  image.src = "/assets/sample-footage.jpg";
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("The bundled test video could not be prepared in this browser.");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const stream = canvas.captureStream(24);
  const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
    .find((type) => MediaRecorder.isTypeSupported(type));
  if (!mimeType) {
    stream.getTracks().forEach((track) => track.stop());
    throw new Error("This browser cannot encode the bundled test video. Please upload a video file instead.");
  }

  const recorder = new MediaRecorder(stream, { mimeType });
  const chunks: BlobPart[] = [];
  const frameTimer = window.setInterval(() => context.drawImage(image, 0, 0, canvas.width, canvas.height), 1000 / 24);
  const recorded = new Promise<Blob>((resolve, reject) => {
    recorder.ondataavailable = (event) => { if (event.data.size > 0) chunks.push(event.data); };
    recorder.onerror = () => { window.clearInterval(frameTimer); reject(new Error("The bundled test video could not be recorded.")); };
    recorder.onstop = () => { window.clearInterval(frameTimer); resolve(new Blob(chunks, { type: recorder.mimeType })); };
  });
  recorder.start();
  await new Promise<void>((resolve) => window.setTimeout(resolve, durationSeconds * 1000));
  recorder.stop();
  const blob = await recorded;
  stream.getTracks().forEach((track) => track.stop());
  if (!blob.size) throw new Error("The browser created an empty test video. Please upload a video file instead.");
  return new File([blob], "cuefx-bundled-sample.webm", { type: blob.type || "video/webm" });
}

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException("Operation cancelled", "AbortError")); return; }
    const timer = window.setTimeout(() => { signal.removeEventListener("abort", cancel); resolve(); }, milliseconds);
    const cancel = () => { window.clearTimeout(timer); reject(new DOMException("Operation cancelled", "AbortError")); };
    signal.addEventListener("abort", cancel, { once: true });
  });
}

async function waitForStandin(initial: BackendStandinStatus, signal: AbortSignal): Promise<BackendStandinStatus> {
  let current = initial;
  const deadline = Date.now() + 15 * 60 * 1000;
  while (current.status === "queued" || current.status === "running") {
    if (Date.now() >= deadline) throw new Error("The backend render is taking longer than expected. You can retry the check.");
    await wait(1200, signal);
    current = await apiRequest<BackendStandinStatus>(`api/standin/${current.standin_id}`, { signal });
  }
  if (current.status === "failed") throw new Error(current.error?.message ?? "The backend could not render this composite.");
  if (!current.result) throw new Error("The backend finished without returning a composite video.");
  return current;
}

async function waitForJob(jobId: string, signal: AbortSignal, onUpdate: (job: BackendJob) => void): Promise<BackendJob> {
  const deadline = Date.now() + 2 * 60 * 60 * 1000;
  let current = await apiRequest<BackendJob>(`api/jobs/${encodeURIComponent(jobId)}`, { signal });
  while (["queued", "running", "waiting"].includes(current.status)) {
    onUpdate(current);
    if (Date.now() >= deadline) throw new Error("Generation is taking longer than expected. You can return and check the job again.");
    await wait(1500, signal);
    current = await apiRequest<BackendJob>(`api/jobs/${encodeURIComponent(jobId)}`, { signal });
  }
  onUpdate(current);
  if (current.status !== "done") throw new Error(current.error?.message ?? `Generation ${current.status}.`);
  return current;
}
function lostTrackRanges(cue: BackendCue | null): { start: number; end: number }[] {
  if (!cue) return [];
  const ranges: { start: number; end: number }[] = [];
  let first: number | null = null;
  let last: number | null = null;
  const flush = () => {
    if (first !== null && last !== null) ranges.push({ start: first / cue.fps, end: (last + 1) / cue.fps });
    first = null;
    last = null;
  };
  for (const point of cue.track) {
    if (typeof point.f !== "number") continue;
    if (point.lost === true) {
      if (first === null) first = point.f;
      last = point.f;
    } else flush();
  }
  flush();
  return ranges;
}

const hexToRgba = (hex: string, alpha: number) => {
  const raw = hex.replace("#", "");
  const full = raw.length === 3 ? raw.split("").map((char) => char + char).join("") : raw;
  const r = parseInt(full.slice(0, 2), 16), g = parseInt(full.slice(2, 4), 16), b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

type EffectFrame = { visible: boolean; x: number; y: number; scale: number; opacity: number; ending: boolean };

function effectFrameAt(settings: DirectSettings, time: number, tracked = { x: 89, y: 23 }): EffectFrame {
  const isThrow = settings.motion === "Hold + throw";
  const flight = Math.max(0, settings.points.length - 1) / 24;
  const endStart = isThrow ? settings.releaseAt + flight : settings.holdEnd;
  const gone = endStart + settings.endingLength;
  const fadeStart = settings.appearAt - settings.fadeIn;
  const visible = time >= Math.max(0, fadeStart) && time <= gone;
  let opacity = time < settings.appearAt ? Math.min(1, Math.max(0, (time - fadeStart) / Math.max(0.01, settings.fadeIn))) : 1;
  if (time >= endStart) opacity *= Math.max(0, 1 - (time - endStart) / Math.max(0.01, settings.endingLength));
  const anchor = { x: tracked.x + settings.offsetX * 100 / 1280, y: tracked.y + settings.offsetY * 100 / 720 };
  let x = anchor.x, y = anchor.y, scale = 1;
  if (settings.motion === "Orbit") {
    const angle = (time * settings.speed * Math.PI) / 180;
    const radius = 4 * settings.distance;
    x = anchor.x + Math.cos(angle) * radius;
    y = anchor.y + Math.sin(angle) * radius * (1 - settings.tilt / 120);
  } else if (settings.motion === "Charge-up") {
    scale = .6 + Math.min(1, time / Math.max(.1, settings.appearAt)) * .7;
  } else if (isThrow && time >= settings.releaseAt && settings.points.length >= 2) {
    const u = Math.min((time - settings.releaseAt) * 24, settings.points.length - 1);
    const index = Math.min(settings.points.length - 2, Math.floor(u));
    let blend = u - index;
    const easing = settings.points[index + 1].easing ?? "linear";
    if (easing === "in") blend *= blend;
    else if (easing === "out") blend = 1 - (1 - blend) * (1 - blend);
    else if (easing === "inout") blend = blend < .5 ? 2 * blend * blend : 1 - Math.pow(-2 * blend + 2, 2) / 2;
    const from = settings.points[index], to = settings.points[index + 1];
    x = from.x + (to.x - from.x) * blend;
    y = from.y + (to.y - from.y) * blend;
    scale = ((from.size ?? settings.size) + ((to.size ?? settings.size) - (from.size ?? settings.size)) * blend) / Math.max(1, settings.size);
  }
  return { visible, x, y, scale, opacity, ending: time >= endStart };
}

function MovingEffect({ settings, time, tracked }: { settings: DirectSettings; time: number; tracked?: { x: number; y: number } | null }) {
  if (tracked === null && !(settings.motion === "Hold + throw" && time >= settings.releaseAt)) return null;
  const frame = effectFrameAt(settings, time, tracked ?? undefined);
  if (!frame.visible) return null;
  const throwScale = settings.motion === "Hold + throw" ? frame.scale : 1;
  const chargeScale = settings.motion === "Charge-up" ? frame.scale : 1;
  return <div className={`effect-position ${settings.ending === "Burst" && frame.ending ? "effect-burst" : ""}`} style={{ left: `${frame.x}%`, top: `${frame.y}%`, transform: `translate(-50%,-50%) scale(${chargeScale})`, opacity: frame.opacity }}><EffectOrb settings={settings} scale={throwScale} /></div>;
}

function usePlaybackClock(videoRef: React.RefObject<HTMLVideoElement | null>, playing: boolean, duration: number, looping: boolean, setTime: (value: number | ((previous: number) => number)) => void, setPlaying: (value: boolean) => void, enabled = true) {
  useEffect(() => {
    if (!playing || !enabled) return;
    let frame = 0;
    let last = performance.now();
    const step = (now: number) => {
      const video = videoRef.current;
      if (video && video.readyState >= 1) setTime(video.currentTime);
      else {
        const delta = (now - last) / 1000;
        setTime((value) => {
          const next = value + delta;
          if (next < duration) return next;
          if (looping) return next % duration;
          setPlaying(false);
          return duration;
        });
      }
      last = now;
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [duration, enabled, looping, playing, setPlaying, setTime, videoRef]);
}

function Icon({ name, size = 16, strokeWidth = 1.8 }: { name: string; size?: number; strokeWidth?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  const paths: Record<string, React.ReactNode> = {
    upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>,
    play: <path d="m8 5 12 7-12 7z" fill="currentColor" stroke="none" />,
    pause: <><path d="M8 5h3v14H8z" fill="currentColor" stroke="none" /><path d="M15 5h3v14h-3z" fill="currentColor" stroke="none" /></>,
    back: <><path d="m14 5-7 7 7 7" /><path d="M7 12h13" /></>,
    next: <><path d="m10 5 7 7-7 7" /><path d="M4 12h13" /></>,
    undo: <><path d="M9 14 4 9l5-5" /><path d="M4 9h9a6 6 0 0 1 0 12h-2" /></>,
    redo: <><path d="m15 14 5-5-5-5" /><path d="M20 9h-9a6 6 0 0 0 0 12h2" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    close: <><path d="m6 6 12 12" /><path d="M18 6 6 18" /></>,
    plus: <><path d="M12 5v14" /><path d="M5 12h14" /></>,
    minus: <path d="M5 12h14" />,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" /><path d="M19 16v5M16.5 18.5h5" /></>,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor" /><circle cx="12" cy="12" r="1" fill="currentColor" /><circle cx="19" cy="12" r="1" fill="currentColor" /></>,
    keyboard: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 9h.01M11 9h.01M15 9h.01M7 13h.01M11 13h.01M15 13h.01M8 16h8" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    refresh: <><path d="M20 7v5h-5" /><path d="M4 17v-5h5" /><path d="M5.6 9a7 7 0 0 1 11.6-2L20 12M4 12l2.8 5a7 7 0 0 0 11.6-2" /></>,
    download: <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M4 20h16" /></>,
    trash: <><path d="M4 7h16" /><path d="M10 11v6M14 11v6" /><path d="m5 7 1 14h12l1-14M9 7V4h6v3" /></>,
    image: <><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.5" cy="9" r="1.5" /><path d="m21 15-5-5L5 20" /></>,
    move: <><path d="M12 3v18M3 12h18" /><path d="m8 7 4-4 4 4M8 17l4 4 4-4M7 8l-4 4 4 4M17 8l4 4-4 4" /></>,
    eye: <><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></>,
    person: <><circle cx="12" cy="6" r="3" /><path d="M6 21v-2a6 6 0 0 1 12 0v2M12 12v5M6 14l-2 3M18 14l2 3" /></>,
    loop: <><path d="M17 2l4 4-4 4" /><path d="M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4" /><path d="M21 13v2a3 3 0 0 1-3 3H3" /></>,
    warning: <><path d="M12 3 2.8 19a1.3 1.3 0 0 0 1.1 2h16.2a1.3 1.3 0 0 0 1.1-2z" /><path d="M12 9v4M12 17h.01" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5M12 16.5h.01" /></>,
    sliders: <><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></>,
    path: <path d="M4 18c3-9 7 2 10-6s4-4 6-6" />,
    body: <><circle cx="12" cy="5" r="2.2" /><path d="M12 8v7M8 21l4-6 4 6M6 11h12" /></>,
    frame: <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />,
    hold: <><circle cx="12" cy="9" r="4" /><path d="M6 20c1.5-3 3.6-4.5 6-4.5s4.5 1.5 6 4.5" /></>,
    throw: <><circle cx="18" cy="6" r="3" /><path d="M3 19c4-1 8-4 12.5-10.5" strokeDasharray="2 3" /></>,
    charge: <><circle cx="12" cy="12" r="2.5" /><circle cx="12" cy="12" r="6" strokeOpacity=".6" /><circle cx="12" cy="12" r="9.5" strokeOpacity=".3" /></>,
    orbit: <><ellipse cx="12" cy="12" rx="9.5" ry="4" transform="rotate(-20 12 12)" /><circle cx="12" cy="12" r="2.2" /><circle cx="20" cy="9" r="1.6" fill="currentColor" stroke="none" /></>,
    sphere: <><circle cx="12" cy="12" r="8" /><ellipse cx="12" cy="12" rx="8" ry="3" strokeOpacity=".5" /></>,
    ring: <><ellipse cx="12" cy="12" rx="9" ry="4.5" /><ellipse cx="12" cy="12" rx="5" ry="2" /></>,
    crystal: <><path d="M12 2.5l5.5 7L12 21.5 6.5 9.5z" /><path d="M6.5 9.5h11M12 2.5v19" strokeOpacity=".5" /></>,
    cube: <><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" /><path d="M4 7.5l8 4.5 8-4.5M12 12v9" strokeOpacity=".5" /></>,
    zoom: <><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4.2-4.2M8.5 11h5" /></>,
    lock: <><rect x="4" y="10" width="16" height="10" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></>,
    layers: <><path d="M12 20v-6M12 14l-9-5 9-5 9 5z" /><path d="M3 14v3l9 5 9-5v-3" /></>,
  };
  return <svg {...common}>{paths[name] ?? <circle cx="12" cy="12" r="8" />}</svg>;
}

function Button({ children, className = "", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`transition-colors ${className}`} {...props}>{children}</button>;
}

function IconButton({ label, icon, onClick, disabled = false }: { label: string; icon: string; onClick: () => void; disabled?: boolean }) {
  return <Button type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick} className="icon-button"><Icon name={icon} /></Button>;
}

function Brand() {
  return <div className="brand-lockup"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4.2" fill="#FF7A45" /><path d="M3.5 12a8.5 8.5 0 0 1 14.2-6.3" stroke="#FF7A45" strokeWidth="1.8" strokeLinecap="round" /><path d="M20.5 12a8.5 8.5 0 0 1-14.2 6.3" stroke="#EDEDEF" strokeOpacity=".55" strokeWidth="1.8" strokeLinecap="round" /></svg><span>CueFX</span></div>;
}

function StudioHeader({ screen, clip, onNavigate, canUndo = false, canRedo = false, onUndo, onRedo, onDrawer }: { screen: Screen; clip: Clip | null; onNavigate: (screen: Screen) => void; canUndo?: boolean; canRedo?: boolean; onUndo?: () => void; onRedo?: () => void; onDrawer?: (drawer: "shortcuts" | "renders") => void }) {
  const active = screen === "footage" ? "footage" : screen === "attach" ? "attach" : screen === "direct" ? "direct" : screen === "review" ? "review" : "look";
  const activeIndex = flow.findIndex((item) => item.id === active);
  const allowed = (id: (typeof flow)[number]["id"]) => id === "footage" || (!!clip && (id === "attach" || id === "direct" && ["direct", "look", "creating", "review"].includes(screen) || id === "look" && ["look", "creating", "review"].includes(screen) || id === "review" && screen === "review"));
  return (
    <header className="studio-header">
      <div className="header-brand-group"><Brand />{screen === "attach" ? <Button className="header-back-link" onClick={() => onNavigate("footage")}><Icon name="back" size={13} />Back to Footage</Button> : clip && (screen === "direct" || screen === "look") && <span className="header-clip-title header-clip-title-direct">{clip.name}<span> · {clip.duration.toFixed(2)} s shot</span></span>}</div>
      <nav aria-label="Project steps" className="step-nav">
        {flow.map((item, index) => {
          const selected = item.id === active;
          const done = index < activeIndex;
          const disabled = !allowed(item.id);
          return <Button key={item.id} type="button" disabled={disabled} onClick={() => onNavigate(item.id)} className={`step-chip ${selected ? "step-active" : ""} ${done ? "step-done" : ""} ${disabled ? "step-disabled" : ""}`}>
            {done ? <span className="step-check"><Icon name="check" size={14} strokeWidth={2.2} /></span> : <span className="step-number">{item.step}</span>}{item.label}
          </Button>;
        })}
      </nav>
      {screen === "direct" && onUndo && onRedo && onDrawer ? <div className="direct-header-actions"><IconButton label="Undo" icon="undo" disabled={!canUndo} onClick={onUndo} /><IconButton label="Redo" icon="redo" disabled={!canRedo} onClick={onRedo} /><span className="header-action-divider" /><IconButton label="Keyboard shortcuts" icon="keyboard" onClick={() => onDrawer("shortcuts")} /><Button className="header-renders-button" onClick={() => onDrawer("renders")}><Icon name="image" size={14} />Renders</Button></div> : screen === "look" ? <div className="direct-header-actions"><Button className="look-header-back" onClick={() => onNavigate("direct")}><Icon name="back" size={14} />Back to Direct</Button></div> : <div className="header-spacer" aria-hidden="true" />}
    </header>
  );
}

function UploadPage({ clip, onUpload, onSample, progress, uploadName, error, onClearError, onContinue }: { clip: Clip | null; onUpload: (file: File) => void; onSample: (sample: typeof sampleClips[number]) => void; progress: number | null; uploadName: string; error: string; onClearError: () => void; onContinue: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const fileChanged = (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) onUpload(file); event.target.value = ""; };
  const dropped = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files?.[0]; if (file) onUpload(file); };
  return (
    <main className="upload-page">
      <section className="upload-content">
        <div className="upload-intro">
          <h1>Start a new shot</h1>
          <p>Upload a clip for backend processing, or try the bundled test sample.</p>
        </div>

        <input ref={inputRef} type="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.mkv,.webm" className="hidden" onChange={fileChanged} aria-label="Upload a video" />
        {error ? <div role="alert" className="upload-error-card"><Icon name="warning" size={18} /><div><p>{error}</p><Button className="ghost-button" onClick={() => { onClearError(); inputRef.current?.click(); }}>Try another file</Button></div></div> : progress !== null ? <div className="upload-busy">
          <div className="upload-busy-content"><div><span>Preparing or uploading {uploadName}</span><span className="mono">{progress}%</span></div><div className="upload-progress"><span style={{ width: `${progress}%` }} /></div></div>
        </div> : clip ? <div className="upload-ready">
          <div className="upload-ready-thumb">{clip.isVideo ? <video src={clip.url} muted playsInline /> : <img src="/assets/sample-footage.jpg" alt="Selected shot preview" />}</div>
          <div className="upload-ready-copy"><strong>{clip.name}</strong><span className="muted">{clip.duration.toFixed(1)}s ready</span></div>
          <Button className="ghost-button" onClick={() => inputRef.current?.click()}>Change</Button>
        </div> : <div role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} onClick={() => inputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={dropped} className={`dropzone ${dragging ? "dropzone-active" : ""}`}>
          <div className="upload-icon"><Icon name="upload" size={30} strokeWidth={1.5} /></div>
          <p>{dragging ? "Drop it here" : "Drop a video here, or click to browse"}</p>
          <p className="muted">MP4, MOV, MKV or WebM · up to 500 MB · up to 60 seconds</p>
        </div>}

        {clip && !error && progress === null && <div className="upload-continue-row"><Button className="primary-button" onClick={onContinue}>Continue to Attach <Icon name="next" size={16} /></Button></div>}

        <div className="uploaded-section">
          <p className="section-eyebrow">Bundled test sample · uploaded to backend</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {sampleClips.map((sample) => <button type="button" onClick={() => onSample(sample)} key={sample.name} className="sample-clip group text-left">
              <div className="sample-image"><img src="/assets/sample-footage.jpg" alt="A person extending their hand in a sample clip" /></div>
              <div className="sample-copy"><p className="truncate">{sample.name}</p><p>{sample.meta}</p></div>
            </button>)}
          </div>
        </div>
      </section>
    </main>
  );
}

function AttachPage({ clip, onContinue, onSelectBody, onTrack, track }: {
  clip: Clip;
  onContinue: (body: string) => void;
  onSelectBody: (body: string) => void;
  onTrack: (input: { point: BodyPoint; side: "left" | "right"; seed: Point | null; frame: number }) => Promise<void>;
  track: Record<string, unknown>[] | null;
}) {
  const [point, setPoint] = useState<BodyPoint>("palm");
  const [side, setSide] = useState<"left" | "right">("right");
  const [seed, setSeed] = useState<Point | null>(null);
  const [seedFrame, setSeedFrame] = useState(0);
  const [trackError, setTrackError] = useState("");
  const [status, setStatus] = useState<"idle" | "analyzing" | "done">("idle");
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const pointLabel = bodyPoints.find((item) => item.id === point)?.label.toLowerCase() ?? "palm";
  const needsSeed = point === "palm";
  const seedHelp = point === "head" ? "guide the target frame" : "help choose the correct side";

  useEffect(() => () => {
    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
  }, []);

  const setPointAndReset = (next: BodyPoint) => {
    setPoint(next);
    setSeed(null);
    setSeedFrame(0);
    setTrackError("");
    setStatus("idle");
    setProgress(0);
  };

  const seedVideo = (event: React.MouseEvent<HTMLDivElement>) => {
    if (status === "analyzing") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const [width, height] = clip.backend?.plate_size ?? [EDITOR_WIDTH, EDITOR_HEIGHT];
    const editorX = ((event.clientX - bounds.left) / bounds.width) * 100;
    const editorY = ((event.clientY - bounds.top) / bounds.height) * 100;
    const framePoint = editorPercentToFrame(editorX, editorY, width, height);
    setSeed({ x: framePoint.x * 100 / width, y: framePoint.y * 100 / height });
    const video = event.currentTarget.querySelector("video");
    setSeedFrame(Math.max(0, Math.round((video?.currentTime ?? 0) * (clip.backend?.fps ?? 24))));
    setTrackError("");
    setStatus("idle");
  };

  const findTarget = async () => {
    setStatus("analyzing");
    setProgress(0);
    setTrackError("");
    try {
      await onTrack({ point, side, seed, frame: seedFrame });
      setProgress(100);
      setStatus("done");
    } catch (error) {
      setStatus("idle");
      setTrackError(error instanceof Error ? error.message : "The backend could not track this target.");
    }
  };

  const retry = () => {
    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    intervalRef.current = null;
    timeoutRef.current = null;
    setStatus("idle"); setProgress(0); setSeed(null); setSeedFrame(0); setTrackError("");
  };

  const statusHint = status === "done"
    ? `Tracked. Click the video again to re-pick the ${pointLabel}.`
    : !seed && needsSeed
      ? `Click the ${pointLabel} in the video`
      : !seed
        ? `Click the ${pointLabel} to ${seedHelp}, or use Find the ${pointLabel}`
        : `Target selected — use Find the ${pointLabel} on the right`;
  const seedOnStage = (() => {
    if (!seed || !clip.backend) return seed;
    const [width, height] = clip.backend.plate_size;
    return frameToEditorPercent(seed.x * width / 100, seed.y * height / 100, width, height);
  })();
  const trackPath = (() => {
    if (!track || !clip.backend) return "";
    const [width, height] = clip.backend.plate_size;
    const parts: string[] = [];
    let segment: Point[] = [];
    const flush = () => {
      if (segment.length) {
        // Median-filter only the display path to suppress single-frame detector jitter;
        // the backend samples sent to Direct and rendering remain untouched.
        const smooth = segment.map((point, index) => {
          const values = segment.slice(Math.max(0, index - 1), Math.min(segment.length, index + 2));
          const median = (items: number[]) => items.sort((a, b) => a - b)[Math.floor(items.length / 2)];
          return { x: median(values.map((item) => item.x)), y: median(values.map((item) => item.y)) };
        });
        parts.push(`M ${smooth.map((point) => `${(point.x * 12.8).toFixed(1)} ${(point.y * 7.2).toFixed(1)}`).join(" L ")}`);
      }
      segment = [];
    };
    for (const sample of track) {
      if (sample.lost === true || typeof sample.x !== "number" || typeof sample.y !== "number") {
        flush();
        continue;
      }
      segment.push(frameToEditorPercent(sample.x, sample.y, width, height));
    }
    flush();
    return parts.join(" ");
  })();

  return <main className="attach-page">
    <section className="attach-preview-column">
      <div className={`attach-video-stage ${status !== "analyzing" ? "attach-stage-pick" : ""}`} onClick={seedVideo} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSeed({ x: 50, y: 50 }); const video = event.currentTarget.querySelector("video"); setSeedFrame(Math.max(0, Math.round((video?.currentTime ?? 0) * (clip.backend?.fps ?? 24)))); setTrackError(""); setStatus("idle"); } }} aria-label={`Choose the ${pointLabel} in the video`}>
        {clip.isVideo ? <video src={clip.url} muted autoPlay loop playsInline preload="metadata" className="stage-media" /> : <img src="/assets/sample-footage.jpg" alt="Sample footage to choose a body point to track" className="stage-media" />}
        {status === "done" && trackPath && <svg className="attach-skeleton" viewBox="0 0 1280 720" preserveAspectRatio="none" aria-hidden="true"><path d={trackPath} /></svg>}
        {seedOnStage && <span className="attach-seed" style={{ left: `${seedOnStage.x}%`, top: `${seedOnStage.y}%` }}><i /></span>}
        {status !== "analyzing" && <span className="attach-hint">{statusHint}</span>}
        {status === "analyzing" && <div className="attach-analyzing"><div className="loading-spinner" /><strong>Finding the {pointLabel}…</strong><div className="upload-progress"><span className="upload-progress-indeterminate" /></div></div>}
      </div>
    </section>

    <aside className="attach-panel">
      <div className="attach-panel-scroll">
        <section className="attach-panel-section">
          <p className="section-eyebrow">ATTACH TO</p>
          <div className="attach-point-grid">
            {bodyPoints.map((item) => <button type="button" key={item.id} aria-pressed={point === item.id} className={`attach-point ${point === item.id ? "attach-point-active" : ""}`} onClick={() => setPointAndReset(item.id)}>{item.label}</button>)}
          </div>
          {point !== "head" && <div className="segmented attach-side-select"><button className={side === "left" ? "segment-active" : ""} aria-pressed={side === "left"} onClick={() => setSide("left")}>Left</button><button className={side === "right" ? "segment-active" : ""} aria-pressed={side === "right"} onClick={() => setSide("right")}>Right</button></div>}
        </section>
        <div className="attach-divider" />
        <section className="attach-panel-section attach-tracking-section">
          <p className="section-eyebrow">TRACKING</p>
          {status === "done" ? <div className="attach-done-card"><strong><Icon name="check" size={15} />Tracked</strong><p className="muted">Tracking completed by the CueFX backend. Review the path in Direct before rendering.</p><Button className="ghost-button w-fit" onClick={retry}>Track again</Button></div>
            : status === "analyzing" ? <div className="attach-progress-copy"><p className="muted">The backend is analyzing the clip. Longer clips can take a little while.</p><span className="mono">Working…</span></div>
                : needsSeed && !seed ? <div className="attach-instruction"><Icon name="move" size={16} /><p className="muted">Click the palm in the video so CueFX knows which one to follow.</p></div>
                  : !needsSeed && !seed ? <div className="attach-instruction"><Icon name="move" size={16} /><p className="muted">Click the {pointLabel} in the video to {seedHelp}, or find it automatically.</p></div>
                : <Button className="primary-button w-full justify-center attach-find-button" onClick={findTarget}>Find the {pointLabel}</Button>}
          {trackError && <p role="alert" className="upload-error-copy">{trackError}</p>}
        </section>
      </div>
      <div className="attach-panel-footer"><Button disabled={status !== "done"} className="primary-button w-full justify-center" onClick={() => { const selected = point === "head" ? "Head" : `${side === "left" ? "Left" : "Right"} ${pointLabel}`; onSelectBody(selected); onContinue(selected); }}>Continue to Direct <Icon name="next" /></Button></div>
    </aside>
  </main>;
}

function EffectOrb({ settings, scale = 1 }: { settings: DirectSettings; scale?: number }) {
  const width = Math.max(32, settings.size * 0.82 * scale);
  const style: CSSProperties = { width, height: width, color: settings.color, filter: `drop-shadow(0 0 ${settings.glow / 3}px ${settings.color}90)` };
  const shapeClass = `effect-orb orb-${settings.shape.toLowerCase().replace(" ", "-")}`;
  return <div className={shapeClass} style={style} aria-label={`${settings.color} ${settings.shape.toLowerCase()} effect`}><span /></div>;
}

function Range({ label, value, min, max, step = 1, suffix = "", fill, decimals, hideValue = false, onChange }: { label: string; value: number; min: number; max: number; step?: number; suffix?: string; fill?: string; decimals?: number; hideValue?: boolean; onChange: (value: number) => void }) {
  const span = Math.max(0.0001, max - min);
  const pct = Math.min(100, Math.max(0, ((value - min) / span) * 100));
  const display = decimals != null ? value.toFixed(decimals) : String(Number(value.toFixed(2)));
  return <label className="range-field">{!hideValue && <span className="frow"><span>{label}</span><span className="val">{display}{suffix}</span></span>}<span className="sl"><span className="sl-tr" /><span className="sl-fill" style={{ width: `${pct}%`, background: fill }} /><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /></span></label>;
}

function DirectPage({ clip, settings, patch, currentTime, setCurrentTime, selectedBody, onBodyChange, onBack, onContinue, onReview, canUndo, canRedo, onUndo, onRedo, drawer, setDrawer, trackingWarnings, resultUrl, track }: {
  clip: Clip; settings: DirectSettings; patch: (value: Partial<DirectSettings>, recordUndo?: boolean) => void;
  currentTime: number; setCurrentTime: (value: number | ((previous: number) => number)) => void; selectedBody: string; onBodyChange: (body: string) => Promise<void>; onBack: () => void; onContinue: () => void; onReview: () => void;
  canUndo: boolean; canRedo: boolean; onUndo: () => void; onRedo: () => void;
  drawer: "renders" | "shortcuts" | null; setDrawer: (drawer: "renders" | "shortcuts" | null) => void;
  trackingWarnings: { start: number; end: number }[];
  resultUrl: string;
  track: Record<string, unknown>[] | null;
}) {
  const [playing, setPlaying] = useState(false);
  const [looping, setLooping] = useState(true);
  const [pathOn, setPathOn] = useState(true);
  const [bodyOn, setBodyOn] = useState(false);
  const [safeOn, setSafeOn] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [warningIndex, setWarningIndex] = useState<number | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [draggingPoint, setDraggingPoint] = useState<number | null>(null);
  const [fineTune, setFineTune] = useState(false);
  const [moreTracks, setMoreTracks] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [zoomed, setZoomed] = useState(false);
  const [speedMenu, setSpeedMenu] = useState(false);
  const [attachMenu, setAttachMenu] = useState(false);
  const [attachPoint, setAttachPoint] = useState<BodyPoint>("palm");
  const [attachSide, setAttachSide] = useState<"left" | "right">("right");
  const [retracking, setRetracking] = useState(false);
  const [toast, setToast] = useState("");
  const stageRef = useRef<HTMLDivElement>(null);
  const lanesRef = useRef<HTMLDivElement>(null);
  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const dragKind = useRef<"ph" | "appear" | "release" | "hold" | "end" | null>(null);
  const pathDragMoved = useRef(false);
  const duration = clip.duration || 4.04;
  const anchorAt = (time: number): Point | null => {
    if (!clip.backend) return { x: 89, y: 23 };
    const [width, height] = clip.backend.plate_size;
    const point = trackPointAt(track, time * clip.backend.fps, clip.backend.fps);
    return point ? frameToEditorPercent(point.x, point.y, width, height) : null;
  };
  const trackedAnchor = anchorAt(currentTime);
  const releaseAnchor = anchorAt(settings.releaseAt);
  const pathAnchor = settings.motion === "Hold + throw" ? releaseAnchor : trackedAnchor;
  const displayPoints = settings.points.map((point, index) => index === 0 && pathAnchor
    ? { ...point, x: pathAnchor.x + settings.offsetX * 100 / EDITOR_WIDTH, y: pathAnchor.y + settings.offsetY * 100 / EDITOR_HEIGHT }
    : point);
  const previewSettings = { ...settings, points: displayPoints };

  const say = (message: string) => { setToast(message); window.setTimeout(() => setToast(""), 2400); };
  useEffect(() => {
    if (!playing) return;
    const flight = Math.max(0, settings.points.length - 1) / 24;
    const visibleStart = settings.motion === "Hold + throw" ? settings.releaseAt + flight : settings.holdEnd;
    const gone = visibleStart + settings.endingLength;
    const fadeStart = settings.appearAt - settings.fadeIn;
    const timer = window.setInterval(() => setCurrentTime((time) => {
      const next = time + (1 / 24) * speed;
      const loopEnd = looping ? Math.min(duration, gone + 0.25) : duration;
      const loopStart = looping ? Math.max(0, fadeStart - 0.25) : 0;
      if (next >= loopEnd) {
        if (looping) return loopStart;
        setPlaying(false);
        return duration;
      }
      return next;
    }), 1000 / 24);
    return () => window.clearInterval(timer);
  }, [duration, looping, playing, setCurrentTime, settings.appearAt, settings.endingLength, settings.fadeIn, settings.holdEnd, settings.motion, settings.points.length, settings.releaseAt, speed]);

  useEffect(() => {
    const video = sourceVideoRef.current;
    if (!video || !clip.isVideo) return;
    video.loop = looping;
    video.playbackRate = speed;
    if (!playing && Math.abs(video.currentTime - currentTime) > 0.2) video.currentTime = Math.min(currentTime, Math.max(0, duration - 0.05));
    if (playing) void video.play().catch(() => undefined); else video.pause();
  }, [clip.isVideo, currentTime, duration, looping, playing, speed]);

  useEffect(() => {
    const shortcuts = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.code === "Space") { event.preventDefault(); setPlaying((value) => !value); }
      else if (event.key === "ArrowLeft") setCurrentTime((time) => Math.max(0, time - (event.shiftKey ? 10 : 1) / 24));
      else if (event.key === "ArrowRight") setCurrentTime((time) => Math.min(duration, time + (event.shiftKey ? 10 : 1) / 24));
      else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && event.shiftKey) { event.preventDefault(); onRedo(); }
      else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); onUndo(); }
      else if (event.key === "Delete" && selectedPoint !== null && selectedPoint > 0 && settings.points.length > 2) { patch({ points: settings.points.filter((_, index) => index !== selectedPoint) }); setSelectedPoint(null); }
      else if (event.key === "Escape") { setDrawing(false); setSelectedPoint(null); }
      else if (event.key.toLowerCase() === "z") setZoomed((value) => !value);
    };
    window.addEventListener("keydown", shortcuts);
    return () => window.removeEventListener("keydown", shortcuts);
  }, [duration, onRedo, onUndo, patch, selectedPoint, settings.points, setCurrentTime]);

  const addCanvasPoint = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect();
    patch({ points: [...settings.points, { x: ((event.clientX - rect.left) / rect.width) * 100, y: ((event.clientY - rect.top) / rect.height) * 100 }] });
  };
  const frame = (delta: number) => { setPlaying(false); setCurrentTime((time) => Math.min(duration, Math.max(0, time + delta / 24))); };
  const timecode = (value: number) => `00:${value.toFixed(2).padStart(5, "0")}`;
  const canvasClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest(".path-point-handle")) return;
    if (drawing || settings.motion === "Hold + throw" && pathOn) {
      setSelectedPoint(null);
      addCanvasPoint(event);
      return;
    }
    if (!stageRef.current || !trackedAnchor) return;
    const rect = stageRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    patch({ offsetX: Math.round((x - trackedAnchor.x) * EDITOR_WIDTH / 100), offsetY: Math.round((y - trackedAnchor.y) * EDITOR_HEIGHT / 100) });
  };
  const dragPathPoint = (event: React.PointerEvent<HTMLDivElement>) => {
    if (draggingPoint === null || !stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100));
    const y = Math.max(0, Math.min(100, (event.clientY - rect.top) / rect.height * 100));
    let update: Partial<DirectSettings>;
    if (draggingPoint === 0) {
      if (!pathAnchor) return;
      update = { offsetX: Math.round((x - pathAnchor.x) * EDITOR_WIDTH / 100), offsetY: Math.round((y - pathAnchor.y) * EDITOR_HEIGHT / 100) };
    } else {
      update = { points: settings.points.map((point, index) => index === draggingPoint ? { ...point, x, y } : point) };
    }
    const recordUndo = !pathDragMoved.current;
    pathDragMoved.current = true;
    patch(update, recordUndo);
  };
  const selectPathPoint = (event: React.PointerEvent<HTMLButtonElement>, index: number) => {
    event.stopPropagation();
    pathDragMoved.current = false;
    setSelectedPoint(index);
    setDraggingPoint(index);
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const selectedPathPoint = selectedPoint === null ? null : settings.points[selectedPoint] ?? null;
  const selectedPathPointPosition = selectedPoint === null ? null : displayPoints[selectedPoint] ?? null;
  const removePathPoint = () => {
    if (selectedPoint === null || selectedPoint === 0 || settings.points.length <= 2) return;
    patch({ points: settings.points.filter((_, index) => index !== selectedPoint) });
    setSelectedPoint(null);
  };
  const addFix = () => patch({ fixes: [...settings.fixes, { id: Date.now(), time: Number(currentTime.toFixed(2)), x: 0, y: 0 }] });
  const removeFix = (id: number) => patch({ fixes: settings.fixes.filter((fix) => fix.id !== id) });
  const addBehind = (side: "left" | "right") => {
    const start = Number(currentTime.toFixed(2));
    const end = Math.min(duration, Number((currentTime + .5).toFixed(2)));
    patch({ behind: [...settings.behind, { id: Date.now(), side, start, end }] });
    setMoreTracks(true);
  };
  const removeBehind = (id: number) => patch({ behind: settings.behind.filter((entry) => entry.id !== id) });
  const bodyLabel = bodyPoints.find((item) => selectedBody.toLowerCase().endsWith(item.label.toLowerCase()))?.label.toLowerCase() ?? "palm";
  const openingAttachment = () => {
    setAttachPoint(bodyPoints.find((item) => item.label.toLowerCase() === bodyLabel)?.id ?? "palm");
    setAttachSide(selectedBody.toLowerCase().startsWith("left") ? "left" : "right");
    setAttachMenu((value) => !value);
  };
  const confirmAttachment = async () => {
    const nextBody = attachPoint === "head" ? "Head" : `${attachSide === "left" ? "Left" : "Right"} ${bodyPoints.find((item) => item.id === attachPoint)?.label.toLowerCase()}`;
    const current = nextBody.toLowerCase() === selectedBody.toLowerCase();
    if (current) return;
    setAttachMenu(false); setRetracking(true);
    try {
      await onBodyChange(nextBody);
      say(`Tracking updated to ${nextBody.toLowerCase()}`);
    } catch (error) {
      say(error instanceof Error ? error.message : "The backend could not update the tracking target.");
    } finally {
      setRetracking(false);
    }
  };
  const attachmentTarget = attachPoint === "head" ? "Head" : `${attachSide === "left" ? "Left" : "Right"} ${bodyPoints.find((item) => item.id === attachPoint)?.label.toLowerCase()}`;
  const attachmentUnchanged = attachmentTarget.toLowerCase() === selectedBody.toLowerCase();

  const throwFlight = Math.max(0, settings.points.length - 1) / 24;
  const endMin = settings.ending === "Fade" ? 0.1 : settings.ending === "Burst" ? 0.3 : 0.5;
  const isThrow = settings.motion === "Hold + throw";
  const heldUntil = isThrow ? settings.releaseAt : settings.holdEnd;
  const endVisibleStart = isThrow ? settings.releaseAt + throwFlight : settings.holdEnd;
  const endGone = endVisibleStart + settings.endingLength;
  const endingOverflow = endGone > duration + 1e-3;
  const endingDesc = settings.ending === "Fade" ? "Fades away where it stops." : settings.ending === "Burst" ? "Swells and bursts outward." : "Breaks into pieces.";
  const effectStart = settings.appearAt - settings.fadeIn;
  const effectLength = endGone - effectStart;
  const fitEnding = () => {
    const over = endGone - duration;
    if (over <= 0) return;
    const lo = settings.appearAt + 2 / 24;
    if (isThrow) {
      const nextRelease = Math.max(lo, Math.floor((settings.releaseAt - over) * 24) / 24);
      const room = duration - nextRelease - throwFlight;
      const nextLen = room < settings.endingLength ? Math.max(endMin, Math.floor(room * 24) / 24) : settings.endingLength;
      patch({ releaseAt: nextRelease, endingLength: nextLen });
      say("Moved the release earlier so the ending fits.");
    } else {
      const nextHold = Math.max(lo, Math.floor((settings.holdEnd - over) * 24) / 24);
      const room = duration - nextHold;
      const nextLen = room < settings.endingLength ? Math.max(endMin, Math.floor(room * 24) / 24) : settings.endingLength;
      patch({ holdEnd: nextHold, endingLength: nextLen });
      say("Started the ending earlier so it fits.");
    }
  };

  const errors: { msg: string; kind: string }[] = [];
  if (endingOverflow) errors.push({ msg: `The ending runs ${(endGone - duration).toFixed(2)} s past the end of the shot.`, kind: "over" });
  if (effectStart < 1 / 24 - 1e-6) errors.push({ msg: "The fade-in has to start after the first frame. Make it shorter or appear later.", kind: "start" });
  if (effectLength > 8 + 1e-6) errors.push({ msg: "The effect is longer than 8 seconds.", kind: "len" });
  if (isThrow && settings.points.length < 2) errors.push({ msg: "Place at least one point for the throw.", kind: "path" });
  const blocked = errors.length > 0;

  const tlMax = Math.max(duration, endGone);
  const tlMin = zoomed ? Math.max(0, Math.floor((effectStart - 0.3) * 4) / 4) : 0;
  const tlSpan = Math.max(0.001, tlMax - tlMin);
  const pc = (t: number) => ((t - tlMin) / tlSpan) * 100;
  const tickStep = zoomed ? 0.25 : 0.5;
  const ticks: number[] = [];
  for (let x = Math.ceil(tlMin / tickStep) * tickStep; x <= tlMax + 1e-6; x += tickStep) { if (pc(x) <= 97.5) ticks.push(Number(x.toFixed(2))); }
  const effTint = (alpha: number) => hexToRgba(settings.color, alpha);
  const endBg = settings.ending === "Fade" ? "linear-gradient(90deg, rgba(255,122,69,.32), rgba(255,122,69,.05))" : settings.ending === "Burst" ? "linear-gradient(90deg, rgba(255,122,69,.28), rgba(255,200,170,.38))" : "repeating-linear-gradient(135deg, rgba(255,122,69,.34) 0 4px, rgba(255,122,69,.12) 4px 8px)";
  const heldLabel = settings.motion === "Charge-up" ? `Charging on ${selectedBody.toLowerCase()}` : settings.motion === "Orbit" ? `Orbiting ${selectedBody.toLowerCase()}` : `Held on ${selectedBody.toLowerCase()}`;
  const endLabel = settings.ending;
  const timeAt = (clientX: number) => { const el = lanesRef.current; if (!el) return currentTime; const rect = el.getBoundingClientRect(); const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)); return tlMin + ratio * tlSpan; };
  const applyHandle = (name: "appear" | "release" | "hold" | "end", t: number) => {
    const time = Number(t.toFixed(4));
    if (name === "appear") patch({ appearAt: Math.min(heldUntil - 2 / 24, Math.max(settings.fadeIn + 1 / 24, time)) });
    else if (name === "release") patch({ releaseAt: Math.min(duration, Math.max(settings.appearAt + 2 / 24, time)) });
    else if (name === "hold") patch({ holdEnd: Math.min(duration, Math.max(settings.appearAt + 2 / 24, time)) });
    else patch({ endingLength: Math.min(1.5, Math.max(endMin, time - endVisibleStart)) });
  };
  const onLanesPointer = (event: React.PointerEvent<HTMLDivElement>, kind: "down" | "move" | "up") => {
    if (kind === "down") {
      const handle = (event.target as HTMLElement).closest("[data-handle]") as HTMLElement | null;
      dragKind.current = handle?.dataset.handle as typeof dragKind.current ?? "ph";
      event.currentTarget.setPointerCapture(event.pointerId);
      setPlaying(false);
      if (dragKind.current === "ph") setCurrentTime(Number(timeAt(event.clientX).toFixed(4)));
      else if (dragKind.current) applyHandle(dragKind.current, timeAt(event.clientX));
    } else if (kind === "move" && dragKind.current) {
      if (dragKind.current === "ph") setCurrentTime(Number(timeAt(event.clientX).toFixed(4)));
      else applyHandle(dragKind.current, timeAt(event.clientX));
    } else if (kind === "up") {
      dragKind.current = null;
      try { event.currentTarget.releasePointerCapture(event.pointerId); } catch { /* already released */ }
    }
  };
  const pathFull = displayPoints.length > 1 ? `M ${displayPoints.map((point) => `${point.x} ${point.y}`).join(" L ")}` : "";
  const throwProgress = isThrow && currentTime >= settings.releaseAt ? Math.min((currentTime - settings.releaseAt) * 24, Math.max(0, settings.points.length - 1)) : 0;
  const pathDone = (() => {
    if (!isThrow || throwProgress <= 0 || displayPoints.length < 2) return "";
    const end = throwProgress;
    let d = `M ${displayPoints[0].x} ${displayPoints[0].y}`;
    const last = Math.min(end, displayPoints.length - 1);
    for (let i = 1; i <= Math.floor(last); i += 1) d += ` L ${displayPoints[i].x} ${displayPoints[i].y}`;
    if (last % 1 > 1e-6) {
      const i = Math.min(Math.floor(last), displayPoints.length - 2);
      const f = last - i;
      const from = displayPoints[i], to = displayPoints[i + 1];
      d += ` L ${from.x + (to.x - from.x) * f} ${from.y + (to.y - from.y) * f}`;
    }
    return d;
  })();
  const stageHint = !isThrow ? (settings.motion === "Orbit" ? "Drag the start point to move the orbit centre" : "Drag the start point to move the effect") : !pathOn ? "Turn on Path to edit the throw" : drawing ? "Click the video to place the throw, point by point" : "Click to add a point  ·  Drag a point to reshape  ·  Click a point for options";
  const chooseEnding = (ending: Ending) => {
    const min = ending === "Fade" ? 0.1 : ending === "Burst" ? 0.3 : 0.5;
    patch({ ending, endingLength: Math.max(min, Math.min(1.5, settings.endingLength)) });
  };

  return <main className="editor-page">
    <div className="editor-workspace">
      <section className="video-workspace" aria-label="Video editor">
          <div className="stage-holder">
          <div ref={stageRef} onClick={canvasClick} onPointerMove={dragPathPoint} onPointerUp={() => { setDraggingPoint(null); pathDragMoved.current = false; }} onPointerCancel={() => { setDraggingPoint(null); pathDragMoved.current = false; }} className={`video-stage ${drawing ? "stage-drawing" : ""}`} style={{ transform: zoomed ? "scale(1.12)" : undefined }}>
            <div role="group" aria-label="Show on video" className="viewer-toggle-group" onClick={(event) => event.stopPropagation()}><button onClick={() => setPathOn((v) => !v)} aria-pressed={pathOn} className={`viewer-toggle ${pathOn ? "viewer-toggle-on" : ""}`}><Icon name="path" size={14} strokeWidth={1.9} />Path</button><button onClick={() => setBodyOn((v) => !v)} aria-pressed={bodyOn} className={`viewer-toggle ${bodyOn ? "viewer-toggle-on" : ""}`}><Icon name="body" size={14} strokeWidth={1.9} />Body</button><button onClick={() => setSafeOn((v) => !v)} aria-pressed={safeOn} className={`viewer-toggle ${safeOn ? "viewer-toggle-on" : ""}`}><Icon name="frame" size={14} strokeWidth={1.9} />Safe frame</button></div>
            {clip.isVideo ? <video ref={sourceVideoRef} src={clip.url} muted playsInline className="stage-media" /> : <img src="/assets/sample-footage.jpg" alt="Sample shot: a man extending his right hand" className="stage-media" />}
            <div className="stage-dim" />
            {safeOn && <><div className="safe-guide safe-guide-outer" /><div className="safe-guide" /></>}
            {bodyOn && <svg className="body-overlay" viewBox="0 0 1280 720" preserveAspectRatio="none" aria-hidden="true"><g stroke="#9FD4FF" strokeOpacity=".75" strokeWidth="2.5" strokeLinecap="round" fill="none"><path d="M624 112 L652 226 M652 226 L582 285 L881 242 L1094 164 L1140 165 M652 226 L781 328 L795 470 L809 627 M652 226 L640 660 M525 669 L752 669" /></g><g fill="#0B0B0D" stroke="#9FD4FF" strokeWidth="2"><circle cx="624" cy="100" r="12" /><circle cx="652" cy="226" r="5" /><circle cx="582" cy="285" r="5" /><circle cx="881" cy="242" r="5" /><circle cx="1094" cy="164" r="5" /><circle cx="781" cy="328" r="5" /><circle cx="795" cy="470" r="5" /><circle cx="809" cy="627" r="5" /></g><circle cx="1140" cy="165" r="8" fill="#FF7A45" stroke="#0B0B0D" strokeWidth="2" /></svg>}
            {pathOn && isThrow && releaseAnchor && settings.points.length > 1 && <svg className="path-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d={pathFull} fill="none" stroke="#000" strokeOpacity=".35" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /><path d={pathFull} fill="none" stroke="#fff" strokeOpacity=".85" strokeWidth=".45" strokeDasharray="0.4 1.6" strokeLinecap="round" vectorEffect="non-scaling-stroke" />{pathDone && <><path d={pathDone} fill="none" stroke="#000" strokeOpacity=".35" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /><path d={pathDone} fill="none" stroke="#FF7A45" strokeWidth=".7" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /></>}</svg>}
            {pathOn && (settings.motion !== "Hold + throw" || releaseAnchor) && (settings.motion === "Hold + throw" ? displayPoints : displayPoints.slice(0,1)).map((point, index) => <button type="button" key={index} aria-label={index === 0 ? "Start point. Drag to move the effect" : index === settings.points.length - 1 ? "End point" : `Path point ${index}`} title={index === 0 ? "Start point" : index === settings.points.length - 1 ? "End point" : `Path point ${index}`} onPointerDown={(event) => selectPathPoint(event, index)} onClick={(event) => event.stopPropagation()} className={`path-point-handle ${selectedPoint === index ? "path-point-selected" : ""} ${index === 0 ? "path-point-start" : index === settings.points.length - 1 ? "path-point-end" : ""}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} />)}
            {selectedPathPoint && selectedPathPointPosition && selectedPoint !== null && <div className="point-popover" style={{ left: `${Math.max(18, Math.min(82, selectedPathPointPosition.x))}%`, top: selectedPathPointPosition.y > 58 ? undefined : `${selectedPathPointPosition.y + 4}%`, bottom: selectedPathPointPosition.y > 58 ? `${100 - selectedPathPointPosition.y + 4}%` : undefined }} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => event.stopPropagation()}>
              <strong>{selectedPoint === 0 ? settings.motion === "Hold + throw" ? "Start point" : "Effect position" : selectedPoint === settings.points.length - 1 ? "End point" : `Point ${selectedPoint}`}</strong>
              {settings.motion === "Hold + throw" && <span className="mono text-[10px] text-zinc-400">{(settings.releaseAt + selectedPoint / 24).toFixed(2)} s</span>}
              {selectedPoint === 0 ? <><p className="muted">{settings.motion === "Hold + throw" ? "Where the throw begins: on the right palm at the moment of release. Drag it to move the whole effect." : "Drag it to move the effect relative to the right palm."}</p><span className="muted">Offset from palm <b className="mono text-zinc-200">{settings.offsetX}, {settings.offsetY} px</b></span></> : <>
                <span className="text-[10px] text-zinc-500">How it moves into this point</span>
                <div className="segmented point-easing">{[{ id: "linear", label: "Steady" }, { id: "in", label: "Speed up" }, { id: "out", label: "Slow down" }, { id: "inout", label: "Smooth" }].map((easing) => <button key={easing.id} className={(selectedPathPoint.easing ?? "linear") === easing.id ? "segment-active" : ""} onClick={() => patch({ points: settings.points.map((point, index) => index === selectedPoint ? { ...point, easing: easing.id as Point["easing"] } : point) })}>{easing.label}</button>)}</div>
                <label className="point-size-row">Size here {selectedPathPoint.size != null && <input aria-label="Size at this point" type="number" min="10" max="600" value={selectedPathPoint.size} onChange={(event) => patch({ points: settings.points.map((point, index) => index === selectedPoint ? { ...point, size: Math.max(10, Math.min(600, Number(event.target.value))) } : point) })} />}<Button className="ghost-button" onClick={() => patch({ points: settings.points.map((point, index) => index === selectedPoint ? { ...point, size: point.size == null ? settings.size : null } : point) })}>{selectedPathPoint.size == null ? "Same as effect" : "Use effect size"}</Button></label>
              </>}
              <div className="flex gap-2 border-t border-white/10 pt-2"><Button className="ghost-button flex-1 justify-center" onClick={() => { setPlaying(false); setCurrentTime(settings.motion === "Hold + throw" ? settings.releaseAt + selectedPoint / 24 : settings.appearAt); }}>Jump to this moment</Button>{selectedPoint > 0 && selectedPoint < settings.points.length - 1 && <Button className="ghost-button" onClick={removePathPoint}>Remove</Button>}</div>
            </div>}
            {!(settings.motion === "Hold + throw" && currentTime >= settings.releaseAt && !releaseAnchor) && <MovingEffect settings={previewSettings} time={currentTime} tracked={trackedAnchor} />}
            {trackedAnchor && <span aria-hidden="true" className="tracked-point" style={{ left: `${trackedAnchor.x}%`, top: `${trackedAnchor.y}%` }} />}
            {trackedAnchor && selectedPoint === null && <div className="palm-chip" style={{ left: `${trackedAnchor.x + settings.offsetX * 100 / EDITOR_WIDTH + 1.2}%`, top: `${trackedAnchor.y + settings.offsetY * 100 / EDITOR_HEIGHT + 4}%` }}><span /><span>{isThrow ? `${selectedBody} · released at ${settings.releaseAt.toFixed(2)} s` : selectedBody}</span></div>}
            <div className="preview-note" onClick={(event) => event.stopPropagation()}><span className="preview-note-dot" style={{ background: settings.color, boxShadow: `0 0 10px ${settings.color}` }} /><span className="preview-note-text"><span>Motion preview</span><small>A stand-in. The real look is made after approval.</small></span></div>
            <div className="stage-hint" onClick={(event) => event.stopPropagation()}><span>{stageHint}</span>{drawing && <button type="button" onClick={() => setDrawing(false)}>Done</button>}</div>
            {toast && <div role="status" className="toast-message">{toast}</div>}
          </div>
        </div>

        <div className="timeline-panel">
          <div className="timeline-controls">
            <div className="timeline-transport"><IconButton label="Back one frame" icon="back" onClick={() => frame(-1)} /><Button aria-label={playing ? "Pause" : "Play"} className="play-button" onClick={() => setPlaying((value) => !value)}><Icon name={playing ? "pause" : "play"} size={14} /></Button><IconButton label="Forward one frame" icon="next" onClick={() => frame(1)} /><Button type="button" aria-label="Loop around the effect" title="Loop around the effect" aria-pressed={looping} className="icon-button" style={{ color: looping ? "#ff9a6e" : "#a6a6af" }} onClick={() => setLooping((value) => !value)}><Icon name="loop" strokeWidth={1.9} /></Button><span className="mono ml-3 flex items-baseline gap-1.5"><span className="text-[14px] text-zinc-100">{timecode(currentTime)}</span><span className="text-[12px] text-zinc-500">/ {timecode(duration)}</span></span><div className="relative ml-2"><Button className="ghost-button mono" aria-label="Playback speed" onClick={() => setSpeedMenu((value) => !value)}>{speed === 1 ? "1×" : speed === .5 ? "½×" : "¼×"}</Button>{speedMenu && <div className="floating-menu speed-menu">{[{ value: 1, label: "Normal speed" }, { value: .5, label: "Half speed" }, { value: .25, label: "Quarter speed" }].map((option) => <button key={option.value} className={`menu-row ${speed === option.value ? "menu-row-selected" : ""}`} onClick={() => { setSpeed(option.value); setSpeedMenu(false); }}>{option.label}</button>)}</div>}</div></div>
            <div className="timeline-meta"><div className="effect-length"><span className="lbl">Effect length</span><div className="len-bar"><div style={{ width: `${Math.min(100, Math.max(0, effectLength / 8 * 100))}%`, background: effectLength > 8 ? "#f2a391" : "#ff7a45" }} /></div><span className="val">{effectLength.toFixed(1)} s <span>of 8 s</span></span></div><div className="meta-divider" /><Button className="zoom-effect-btn" onClick={() => setZoomed((value) => !value)}><Icon name="zoom" size={14} strokeWidth={1.9} />{zoomed ? "Show whole shot" : "Zoom to effect"}</Button></div>
          </div>
          <div className="timeline-grid">
            <div className="timeline-labels">
              <span className="lane-spacer" />
              <span className="lane-lbl" style={{ height: 36 }}>Shot</span>
              <span className="lane-lbl" style={{ height: 40 }}>Effect</span>
              <span className="lane-lbl" style={{ height: 24 }}>Path</span>
              <span className="lane-lbl" style={{ height: 20 }}>Tracking</span>
              {moreTracks && <><span className="lane-lbl" style={{ height: 20 }}>Behind hand</span><span className="lane-lbl" style={{ height: 20 }}>Position fixes</span></>}
              <button className="more-tracks-btn" onClick={() => setMoreTracks((value) => !value)}>{moreTracks ? "− Fewer tracks" : "+ 2 more tracks"}</button>
            </div>
            <div ref={lanesRef} className="timeline-lanes-area" onPointerDown={(event) => { if ((event.target as HTMLElement).closest("button:not([data-handle])")) return; onLanesPointer(event, "down"); }} onPointerMove={(event) => onLanesPointer(event, "move")} onPointerUp={(event) => onLanesPointer(event, "up")}>
              <input aria-label="Timeline playhead" className="timeline-scrubber" type="range" min={tlMin} max={tlMax} step={1 / 24} value={Math.min(tlMax, Math.max(tlMin, currentTime))} onChange={(event) => { setPlaying(false); setCurrentTime(Number(event.target.value)); }} />
              <div className="lane-ruler mono">{ticks.map((tick) => <span key={tick} style={{ left: `${pc(tick)}%`, transform: pc(tick) < 1 ? "translateX(0)" : "translateX(-50%)" }}>{tick === 0 ? "0s" : tickStep < 0.5 ? tick.toFixed(2) : tick.toFixed(1)}</span>)}</div>
              <div className="lane-row" style={{ height: 36 }}>
                <div className="tl-strip" style={{ width: `${pc(duration)}%`, backgroundImage: "url('/assets/clip-thumbnail.jpg')" }} />
                {endingOverflow && <div className="tl-past" style={{ left: `${pc(duration)}%` }}><span>Past the end</span></div>}
              </div>
              <div className="lane-row" style={{ height: 40 }}>
                <div className="tl-seg" style={{ left: `${pc(effectStart)}%`, width: `${Math.max(0, pc(settings.appearAt) - pc(effectStart))}%`, background: `linear-gradient(90deg, ${effTint(0.04)}, ${effTint(0.3)})`, borderRadius: "8px 0 0 8px", color: "#e6f3ff" }}>Fade in</div>
                <div className="tl-seg" style={{ left: `${pc(settings.appearAt)}%`, width: `${Math.max(0, pc(heldUntil) - pc(settings.appearAt))}%`, background: effTint(0.3), color: "#f0f7ff" }}>{heldLabel}</div>
                {isThrow && <div className="tl-seg" style={{ left: `${pc(settings.releaseAt)}%`, width: `${Math.max(0, pc(endVisibleStart) - pc(settings.releaseAt))}%`, background: "rgba(255,122,69,.38)", color: "#ffe6db", justifyContent: "center", paddingLeft: 0, fontWeight: 600 }}>Thrown</div>}
                <div className="tl-seg" style={{ left: `${pc(endVisibleStart)}%`, width: `${Math.max(0, pc(endGone) - pc(endVisibleStart))}%`, background: endBg, borderRadius: "0 8px 8px 0", color: "#ffd9c8" }}>{endLabel}</div>
                <button type="button" data-handle="appear" aria-label={`Fully visible at ${settings.appearAt.toFixed(2)} s`} title={`Fully visible at ${settings.appearAt.toFixed(2)} s`} className="tl-handle" style={{ left: `${pc(settings.appearAt)}%` }}><span className="tl-grip" style={{ background: "#cfe8ff" }} /></button>
                {isThrow ? <button type="button" data-handle="release" aria-label={`Released at ${settings.releaseAt.toFixed(2)} s`} title={`Released at ${settings.releaseAt.toFixed(2)} s`} className="tl-handle" style={{ left: `${pc(settings.releaseAt)}%` }}><span className="tl-grip" style={{ background: "#ff7a45" }} /></button> : <button type="button" data-handle="hold" aria-label={`Ending starts at ${settings.holdEnd.toFixed(2)} s`} title={`Ending starts at ${settings.holdEnd.toFixed(2)} s`} className="tl-handle" style={{ left: `${pc(settings.holdEnd)}%` }}><span className="tl-grip" style={{ background: "#ff9a6e" }} /></button>}
                <button type="button" data-handle="end" aria-label={`Gone at ${endGone.toFixed(2)} s`} title={`Gone at ${endGone.toFixed(2)} s`} className="tl-handle" style={{ left: `${pc(endGone)}%` }}><span className="tl-grip" style={{ background: "#f4c8b4" }} /></button>
              </div>
              <div className="lane-row" style={{ height: 24 }}>
                {isThrow ? <>
                  <div className="tl-pathline" style={{ left: `${pc(settings.releaseAt)}%`, width: `${Math.max(0, pc(endVisibleStart) - pc(settings.releaseAt))}%` }} />
                  <span className="tl-pathnote" style={{ right: `${100 - pc(settings.releaseAt) + 1.2}%` }}>{Math.max(0, settings.points.length - 1)} points · {throwFlight.toFixed(2)} s flight</span>
                  {settings.points.map((point, index) => { const first = index === 0, last = index === settings.points.length - 1, size = first ? 10 : last ? 11 : 7; return <button key={index} type="button" aria-label={`${first ? "Start point" : last ? "End point" : `Point ${index}`} at ${(settings.releaseAt + index / 24).toFixed(2)} s`} title={`${first ? "Start point" : last ? "End point" : `Point ${index}`} at ${(settings.releaseAt + index / 24).toFixed(2)} s`} onPointerDown={(event) => event.stopPropagation()} onClick={() => { setPlaying(false); setCurrentTime(Number((settings.releaseAt + index / 24).toFixed(4))); }} className="tl-pt" style={{ left: `${pc(settings.releaseAt + index / 24)}%`, width: size, height: size, borderRadius: first ? 2 : "50%", transform: `translate(-50%,-50%)${first ? " rotate(45deg)" : ""}`, background: last ? "#ff7a45" : "#ededef", border: last ? "2px solid #0e0e11" : "0" }} />; })}
                </> : <span className="tl-nopath" style={{ left: `${pc(settings.appearAt)}%` }}>No path. This motion stays on the palm.</span>}
              </div>
              <div className="lane-row" style={{ height: 20 }}>
                <div className="tl-trackbase" style={{ width: `${pc(duration)}%` }} />
                {trackingWarnings.map((warning, index) => <button key={warning.start} type="button" className="tl-warn" aria-label={`Hand hard to see, ${warning.start.toFixed(2)} to ${warning.end.toFixed(2)} s`} title={`${warning.start.toFixed(2)}–${warning.end.toFixed(2)} s · Hand hard to see`} onPointerDown={(event) => event.stopPropagation()} onClick={() => setWarningIndex(warningIndex === index ? null : index)} style={{ left: `${pc(warning.start)}%`, width: `${Math.max(1.2, pc(warning.end) - pc(warning.start))}%` }} />)}
                {warningIndex !== null && <div className="tracking-warning-popup" style={{ left: `${pc(trackingWarnings[warningIndex].start)}%` }}><strong>Hand hard to see · {trackingWarnings[warningIndex].start.toFixed(2)}–{trackingWarnings[warningIndex].end.toFixed(2)} s</strong><p>The palm is blurred here, so the effect may drift. Watch this moment, or add a position fix.</p><div><button onClick={() => { const warning = trackingWarnings[warningIndex]; setCurrentTime(Math.max(0, warning.start - .2)); setPlaying(true); setWarningIndex(null); window.setTimeout(() => setPlaying(false), (warning.end - warning.start + .5) * 1000); }}>Watch it</button><button onClick={() => { const warning = trackingWarnings[warningIndex]; patch({ fixes: [...settings.fixes, { id: Date.now(), time: warning.start, x: 0, y: 0 }] }); setCurrentTime(warning.start); setFineTune(true); setWarningIndex(null); }}>Add a position fix</button></div></div>}
              </div>
              {moreTracks && <>
                <div className="lane-row" style={{ height: 20 }}>{settings.behind.length === 0 ? <span className="tl-empty">None. Add one in Fine-tune.</span> : settings.behind.map((entry) => <button key={entry.id} type="button" title={`Behind ${entry.side} hand from ${entry.start.toFixed(2)} to ${entry.end.toFixed(2)} s`} onPointerDown={(event) => event.stopPropagation()} onClick={() => setCurrentTime(entry.start)} style={{ position: "absolute", top: 2, bottom: 2, left: `${pc(entry.start)}%`, width: `${pc(entry.end) - pc(entry.start)}%`, borderRadius: 5, background: "rgba(183,155,255,.28)", border: 0, padding: 0 }} />)}</div>
                <div className="lane-row" style={{ height: 20 }}>{settings.fixes.length === 0 ? <span className="tl-empty">None. The effect follows the tracked palm.</span> : settings.fixes.map((fix) => <button key={fix.id} type="button" title={`Position fix at ${fix.time.toFixed(2)} s`} onPointerDown={(event) => event.stopPropagation()} onClick={() => setCurrentTime(fix.time)} className="tl-fix" style={{ left: `${pc(fix.time)}%` }} />)}</div>
              </>}
              <div className="tl-playhead" style={{ left: `${pc(currentTime)}%` }}><div className="tl-playhead-line" /><div className="tl-playhead-flag" /></div>
            </div>
          </div>
        </div>
      </section>

      <aside className="inspector-panel">
        {!fineTune ? <div className="inspector-scroll">
          <div className="attachment-section">
            <div className="attachment-summary"><div className="attachment-icon"><Icon name="person" size={21} strokeWidth={1.6} /></div><div className="attachment-copy"><span>Attached to</span><strong>{selectedBody}</strong></div><Button className="ghost-button attachment-change" aria-expanded={attachMenu} onClick={openingAttachment}>Change</Button></div>
            {attachMenu && <div role="dialog" aria-label="Change attachment" className="attachment-dialog-panel"><p className="text-[13px] font-semibold text-zinc-100">Attach the effect to</p><div className="grid grid-cols-3 gap-1.5">{bodyPoints.map((body) => <button key={body.id} aria-pressed={attachPoint === body.id} onClick={() => setAttachPoint(body.id)} className={`attach-point ${attachPoint === body.id ? "attach-point-active" : ""}`}>{body.label}</button>)}</div>{attachPoint !== "head" && <div className="segmented"><button aria-pressed={attachSide === "left"} className={attachSide === "left" ? "segment-active" : ""} onClick={() => setAttachSide("left")}>Left</button><button aria-pressed={attachSide === "right"} className={attachSide === "right" ? "segment-active" : ""} onClick={() => setAttachSide("right")}>Right</button></div>}<p className="muted">{attachmentUnchanged ? `Currently attached to the ${selectedBody.toLowerCase()}.` : "The shot is tracked again for the new body point, which takes a moment. Your effect, look and timing are kept."}</p><div className="flex gap-2"><Button className="ghost-button flex-1 justify-center" onClick={() => setAttachMenu(false)}>Cancel</Button><Button disabled={retracking || attachmentUnchanged} className="primary-button flex-1 justify-center" onClick={confirmAttachment}>{retracking ? "Tracking…" : "Track again"}</Button></div></div>}
          </div>
          <div className="inspector-group effect-group">
            <div className="effect-heading"><p className="section-eyebrow">EFFECT</p><div className="shape-grid">{(["Sphere", "Ring", "Crystal", "Cube"] as Shape[]).map((shape) => <button key={shape} className={`shape-tile ${settings.shape === shape ? "shape-tile-active" : ""}`} onClick={() => patch({ shape, spin: shape === "Sphere" ? 288 : shape === "Ring" ? 90 : 60 })}><Icon name={shape.toLowerCase()} size={22} strokeWidth={1.5} /><span>{shape}</span></button>)}</div></div>
            <div className="effect-color-row"><span className="inspector-label">Colour</span><div className="color-row">{colors.map((color) => <button key={color.hex} aria-label={color.name} title={color.name} className={`color-swatch ${settings.color.toUpperCase() === color.hex ? "color-swatch-active" : ""}`} style={{ background: color.hex }} onClick={() => patch({ color: color.hex })} />)}<label className="custom-color" title="Custom colour"><Icon name="plus" size={11} strokeWidth={2.4} /><input type="color" aria-label="Custom colour" value={settings.color} onChange={(event) => patch({ color: event.target.value.toUpperCase() })} /></label></div></div>
            <Range label="Glow" value={settings.glow} min={0} max={100} suffix="%" fill={`linear-gradient(90deg, ${hexToRgba(settings.color, 0.25)}, ${settings.color})`} onChange={(glow) => patch({ glow })} />
            <div className="size-block"><div className="frow"><span className="inspector-label">Size</span><div className="size-actions"><Button className="ghost-button" onClick={() => { patch({ size: 99 }); say("Size matched to the palm in this shot."); }}>Match hand</Button><span className="val">{settings.size} px</span></div></div><Range label="Size" value={settings.size} min={20} max={400} hideValue onChange={(size) => patch({ size })} /></div>
          </div>
          <div className="inspector-group motion-group">
            <p className="section-eyebrow">MOTION</p>
            <div className="motion-grid">{([{ id: "Hold" as const, icon: "hold" }, { id: "Hold + throw" as const, icon: "throw" }, { id: "Charge-up" as const, icon: "charge" }, { id: "Orbit" as const, icon: "orbit" }]).map((motion) => <button key={motion.id} onClick={() => patch({ motion: motion.id })} className={`motion-option ${settings.motion === motion.id ? "motion-option-active" : ""}`}><Icon name={motion.icon} size={18} strokeWidth={1.6} />{motion.id}</button>)}</div>
            <p className="muted motion-desc">{settings.motion === "Hold" ? "Stays in the palm from start to finish." : settings.motion === "Hold + throw" ? "Held in the palm, then thrown along the path you draw." : settings.motion === "Charge-up" ? "Grows in the palm as it charges up." : "Circles around the palm."}</p>
            {settings.motion === "Hold + throw" && <div className="throw-path-heading"><span className="inspector-label">Throw path</span><div className="throw-path-value-group"><span className="val">{Math.max(0, settings.points.length - 1)} points · {throwFlight.toFixed(2)} s</span><Button className="ghost-button" onClick={() => { patch({ points: [initialDirect.points[0]] }); setDrawing(true); }}>Redraw</Button></div></div>}
            {settings.motion === "Orbit" && <div className="orbit-card"><Range label="Distance from palm" value={settings.distance} min={0.1} max={10} step={0.1} decimals={1} suffix="× hand width" onChange={(distance) => patch({ distance })} /><Range label="Speed" value={settings.speed} min={-3600} max={3600} step={10} suffix={settings.speed === 0 ? " · Still" : `°/s · ${settings.speed > 0 ? "clockwise" : "anticlockwise"}`} onChange={(speed) => patch({ speed })} /><Range label="Tilt" value={settings.tilt} min={0} max={90} suffix={settings.tilt === 0 ? "° · flat on screen" : settings.tilt >= 85 ? "° · edge-on" : "°"} onChange={(tilt) => patch({ tilt })} /></div>}
          </div>
          <div className="inspector-group ending-section"><div className="ending-heading"><p className="section-eyebrow">ENDING</p><Button className="ghost-button ending-preview-btn" onClick={() => { setCurrentTime(Math.max(0, endVisibleStart - 0.4)); setPlaying(true); }}><Icon name="play" size={10} />Preview ending</Button></div><div className="ending-tabs" role="group" aria-label="Ending">{(["Fade", "Burst", "Shatter"] as Ending[]).map((ending) => <button key={ending} className={settings.ending === ending ? "ending-active" : ""} onClick={() => chooseEnding(ending)}>{ending}</button>)}</div><div className="ending-length-block"><Range label="Ending length" value={settings.endingLength} min={endMin} max={1.5} step={1 / 24} decimals={2} suffix=" s" fill="#ff9a6e" onChange={(endingLength) => patch({ endingLength })} /><div className="ending-span-row"><span className="muted">{endingDesc}</span><span className="muted mono">{endVisibleStart.toFixed(2)} → {endGone.toFixed(2)} s</span></div></div>{endingOverflow && <div className="ending-overflow"><Icon name="info" size={15} strokeWidth={2} /><div className="ending-overflow-body"><span>This {settings.ending.toLowerCase()} needs {settings.endingLength.toFixed(2)} s, but the shot ends {Math.max(0, duration - endVisibleStart).toFixed(2)} s after it starts.</span><Button className="ghost-button ending-fit-btn" onClick={fitEnding}>Fit it in the shot</Button></div></div>}</div>
          <button type="button" className="fine-tune-link" aria-expanded={false} onClick={() => setFineTune(true)}><Icon name="sliders" size={18} strokeWidth={1.7} /><span><strong>Fine-tune</strong><small className="muted">Spin, position, exact timing, fixes, behind the hand{settings.fixes.length ? ` · ${settings.fixes.length} fix${settings.fixes.length > 1 ? "es" : ""}` : ""}{settings.behind.length ? ` · ${settings.behind.length} behind-hand` : ""}</small></span><Icon name="next" size={16} /></button>
        </div> : <div className="inspector-scroll"><div className="fine-tune-heading"><IconButton label="Back to Effect" icon="back" onClick={() => setFineTune(false)} /><strong>Fine-tune</strong></div>
          <div className="inspector-section"><p className="section-eyebrow">SPIN</p><Range label="Rotation" value={settings.spin} min={-1440} max={1440} suffix="°/s" onChange={(spin) => patch({ spin })} /><Button className="text-button mt-2" onClick={() => patch({ spin: 288 })}>Shape default</Button></div>
          <div className="inspector-section"><p className="section-eyebrow">POSITION</p><label className="number-field">Left / right <input type="number" min="-600" max="600" value={settings.offsetX} onChange={(event) => patch({ offsetX: Math.max(-600, Math.min(600, Number(event.target.value))) })} /> px</label><label className="number-field">Up / down <input type="number" min="-600" max="600" value={settings.offsetY} onChange={(event) => patch({ offsetY: Math.max(-600, Math.min(600, Number(event.target.value))) })} /> px</label><p className="muted mt-2">Distance from the {selectedBody.toLowerCase()}. You can also drag the start point on the video.</p></div>
          <div className="inspector-section"><p className="section-eyebrow">TIMING</p><label className="number-field"><span>Fade-in length <small>{Math.round(settings.fadeIn * 24)} frames</small></span><input type="number" min="0" max="3" step=".04" value={settings.fadeIn} onChange={(event) => patch({ fadeIn: Number(event.target.value) })} /> s</label><label className="number-field">Fully visible at <input type="number" min="0" max={duration} step=".04" value={settings.appearAt} onChange={(event) => patch({ appearAt: Number(event.target.value) })} /> s</label>{settings.motion === "Hold + throw" ? <label className="number-field">Released at <input type="number" min="0" max={duration} step=".04" value={settings.releaseAt} onChange={(event) => patch({ releaseAt: Number(event.target.value) })} /> s</label> : <label className="number-field">Ending starts at <input type="number" min="0" max={duration} step=".04" value={settings.holdEnd} onChange={(event) => patch({ holdEnd: Number(event.target.value) })} /> s</label>}<label className="number-field"><span>Ending length <small>{Math.round(settings.endingLength * 24)} frames</small></span><input type="number" min=".1" max="3" step=".04" value={settings.endingLength} onChange={(event) => patch({ endingLength: Math.max(.1, Math.min(3, Number(event.target.value))) })} /> s</label><label className="number-field"><span>Gone at</span><input type="number" aria-label="Gone at" readOnly value={(settings.motion === "Hold + throw" ? settings.releaseAt + Math.max(0, duration - settings.releaseAt - .35) : settings.holdEnd) + settings.endingLength} /> s</label></div>
          <div className="inspector-section"><div className="mb-3 flex items-center justify-between"><p className="section-eyebrow">POSITION FIXES</p><Button className="text-button" onClick={addFix}><Icon name="plus" size={13} /> Add at {currentTime.toFixed(2)}s</Button></div><p className="muted mb-3">If the effect slips, add a fix and nudge it back. Nearby frames blend smoothly.</p>{settings.fixes.map((fix) => <div key={fix.id} className="fix-row"><button onClick={() => setCurrentTime(fix.time)} className="mono text-[11px] text-zinc-300">{fix.time.toFixed(2)}s</button><input aria-label="Fix left right" type="number" value={fix.x} onChange={(event) => patch({ fixes: settings.fixes.map((item) => item.id === fix.id ? { ...item, x: Number(event.target.value) } : item) })} /><input aria-label="Fix up down" type="number" value={fix.y} onChange={(event) => patch({ fixes: settings.fixes.map((item) => item.id === fix.id ? { ...item, y: Number(event.target.value) } : item) })} /><IconButton label="Remove position fix" icon="trash" onClick={() => removeFix(fix.id)} /></div>)}</div>
          <div className="inspector-section"><p className="section-eyebrow">BEHIND THE HAND</p><p className="muted">Hide the effect behind a hand for part of the shot, for example as it passes behind the fingers.</p>{settings.behind.map((entry) => <div key={entry.id} className="behind-row"><span>Behind {entry.side} hand</span><span className="mono text-[10px] text-zinc-400">{entry.start.toFixed(2)}–{entry.end.toFixed(2)} s</span><IconButton label="Remove behind-hand range" icon="close" onClick={() => removeBehind(entry.id)} /></div>)}<div className="flex gap-2"><Button className="ghost-button flex-1 justify-center" onClick={() => addBehind("left")}>+ Left hand</Button><Button className="ghost-button flex-1 justify-center" onClick={() => addBehind("right")}>+ Right hand</Button></div><p className="muted text-[9px]">Added from the playhead for half a second. Adjust on the timeline.</p></div>
        </div>}

        <div className="inspector-footer">{blocked ? errors.map((error) => <p key={error.kind} className="footer-error"><Icon name="info" size={15} strokeWidth={2} /><span>{error.msg}</span></p>) : <p><Icon name="check" size={15} strokeWidth={2.2} />Ready. Next, you check the composite.</p>}<Button disabled={blocked} className="primary-button w-full justify-center" onClick={onContinue}>Continue to approval <Icon name="next" /></Button></div>
      </aside>

      {drawer && <div className="modal-backdrop" onClick={() => setDrawer(null)}><section className="modal-card" onClick={(event) => event.stopPropagation()}><div className="mb-5 flex items-center justify-between"><h2>{drawer === "renders" ? "Renders" : "Keyboard shortcuts"}</h2><IconButton label="Close" icon="close" onClick={() => setDrawer(null)} /></div>{drawer === "renders" ? resultUrl ? <div className="render-list-item"><video src={resultUrl} muted playsInline /><div><strong>Latest CueFX render</strong><span>Ready to review</span></div><Button className="ghost-button" onClick={onReview}>Review</Button></div> : <p className="muted">No completed backend renders for this clip yet. Finished renders appear here.</p> : <div className="space-y-3">{[["Play or pause", "Space"], ["Back / forward one frame", "← →"], ["Undo", "Ctrl Z"], ["Redo", "Ctrl Shift Z"], ["Toggle safe frame", "Z"]].map(([what, key]) => <div className="shortcut-row" key={what}><span>{what}</span><kbd>{key}</kbd></div>)}</div>}</section></div>}
    </div>
  </main>;
}

function LookPage({ clip, settings, approved, setApproved, ready, compositeUrl, compositeError, generationAvailable, prompt, setPrompt, mode, setMode, onRecheck, onBack, onGenerate }: {
  clip: Clip; settings: DirectSettings; approved: boolean; setApproved: (value: boolean) => void;
  ready: boolean; compositeUrl: string; compositeError: string; generationAvailable: boolean; prompt: string; setPrompt: (value: string) => void; mode: "describe" | "surprise";
  setMode: (value: "describe" | "surprise") => void;
  onRecheck: () => void; onBack: () => void; onGenerate: () => void;
}) {
  const [playing, setPlaying] = useState(true);
  const [looping, setLooping] = useState(true);
  const [displayTime, setDisplayTime] = useState(0);
  const [toast, setToast] = useState("");
  const compositeVideoRef = useRef<HTMLVideoElement>(null);
  const say = (message: string) => { setToast(message); window.setTimeout(() => setToast(""), 2400); };
  const canGenerate = approved && generationAvailable && (mode === "surprise" || prompt.trim().length > 0);
  const duration = clip.duration || 4.04;
  const clock = (value: number) => {
    const cs = Math.max(0, Math.round(value * 100));
    const m = Math.floor(cs / 6000);
    const sec = (cs % 6000) / 100;
    return `${m}:${sec < 10 ? "0" : ""}${sec.toFixed(2)}`;
  };
  const scrubPct = `${Math.min(100, Math.max(0, (displayTime / duration) * 100))}%`;
  const seekTo = (value: number) => {
    const next = Math.min(duration, Math.max(0, value));
    setDisplayTime(next);
    if (compositeVideoRef.current) compositeVideoRef.current.currentTime = next;
  };
  useEffect(() => {
    const video = compositeVideoRef.current;
    if (!video) return;
    video.loop = looping;
    if (playing) void video.play().catch(() => undefined); else video.pause();
  }, [looping, playing, ready]);
  usePlaybackClock(compositeVideoRef, playing, duration, looping, setDisplayTime, setPlaying, ready);

  return <main className="look-page">
    <section className="composite-column" aria-label="Hard composite">
      <div className="composite-stack">
        <div className="composite-heading">
          <div><h1>{ready ? "Review the composite" : "Building the composite"}</h1><p className="muted look-sub">{ready ? "This short video is what generation starts from. Watch it once before you approve." : "We’re rendering the effect into the shot as a stand-in video, so you can check it before generating."}</p></div>
        </div>
        <div className="composite-frame">
          {compositeError ? <div className="composite-loading"><div><h2>Could not prepare the composite</h2><p role="alert" className="muted">{compositeError}</p></div><div className="flex gap-2"><Button className="ghost-button" onClick={onBack}>Back to edit</Button><Button className="ghost-button look-retry" onClick={onRecheck}>Try again</Button></div></div> : !ready ? <div className="composite-loading"><div className="loading-spinner look-spinner" /><div><h2>Rendering in the backend</h2><p className="muted">CueFX is rendering a real stand-in composite from your tracked clip.</p></div><Button className="ghost-button" onClick={onBack}>Back to edit</Button></div> : <>
            {compositeUrl ? <video ref={compositeVideoRef} src={compositeUrl} muted loop={looping} playsInline onTimeUpdate={(event) => setDisplayTime(event.currentTarget.currentTime)} onEnded={() => { if (!looping) setPlaying(false); }} className="stage-media" /> : <img src="/assets/sample-footage.jpg" alt="The backend composite" className="stage-media" />}
            <div className="composite-tag"><span className="composite-tag-dot" />Backend stand-in · generation source</div>
            {approved && <div className="approved-badge"><Icon name="check" size={13} strokeWidth={2.4} />Approved</div>}
            <div className="overlay-player" onPointerDown={(event) => event.stopPropagation()}>
              <label className="overlay-scrub"><span className="overlay-track" /><span className="overlay-fill overlay-fill-look" style={{ width: scrubPct }} /><span className="overlay-thumb" style={{ left: scrubPct }} /><input aria-label="Composite position" type="range" min={0} max={duration} step={0.01} value={Math.min(displayTime, duration)} onChange={(event) => { setPlaying(false); seekTo(Number(event.target.value)); }} /></label>
              <div className="overlay-row"><Button aria-label={playing ? "Pause" : "Play"} className="play-button" onClick={() => setPlaying((value) => !value)}><Icon name={playing ? "pause" : "play"} size={13} /></Button><span className="mono overlay-time">{clock(displayTime)} <span>/ {clock(duration)}</span></span><span className="overlay-spacer" /><Button type="button" aria-label={looping ? "Loop on" : "Loop off"} aria-pressed={looping} className="icon-button overlay-loop" style={{ color: looping ? "#ff9a6e" : "#a6a6af" }} onClick={() => setLooping((value) => !value)}><Icon name="loop" size={15} strokeWidth={2} /></Button><span className="overlay-note">No audio in this stand-in</span></div>
            </div>
            {toast && <div role="status" className="toast-message">{toast}</div>}
          </>}
        </div>
        {ready && <div className="composite-actions"><Button className="text-back" onClick={onBack}><Icon name="back" size={14} />Back to edit</Button>{approved ? <span className="locked-source"><Icon name="check" size={14} strokeWidth={2.4} />Locked in as the source for generation. Editing Direct will ask you to approve again.</span> : <Button disabled={!ready} className="approve-button" onClick={() => { setApproved(true); say("Composite approved. Choose a look to continue."); }}><Icon name="check" size={15} strokeWidth={2.4} />Approve this composite</Button>}</div>}
      </div>
    </section>

    <aside className="look-panel" aria-label="Look">
      <div className="look-panel-scroll">
        {!approved && <div className="locked-note"><Icon name="lock" size={16} strokeWidth={1.8} /><span><strong>Locked for now.</strong> Approve the composite on the left to choose a look.</span></div>}
        <div className={`look-fields ${approved ? "" : "look-fields-locked"}`}>
          <div className="look-block"><p className="section-eyebrow">LOOK</p><div className="look-card"><div className="look-swatch" style={{ background: `radial-gradient(circle at 35% 30%, #e4f4ff, ${settings.color} 55%, #2c4a63 100%)` }} /><div><strong>Ice Energy</strong><span className="muted">The look this shot is configured with</span></div></div></div>
          <div className="look-block">
            <div className="segmented look-modes" role="group" aria-label="How to describe the look"><button className={mode === "describe" ? "segment-active" : ""} onClick={() => setMode("describe")}>Describe it</button><button className={mode === "surprise" ? "segment-active" : ""} onClick={() => setMode("surprise")}>Surprise me</button></div>
            {mode === "describe" ? <>
              <label className="sr-only" htmlFor="look-prompt">Describe the final look</label>
              <textarea id="look-prompt" disabled={!approved} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="e.g. a crackling shard of blue ice, cold mist drifting off it" maxLength={800} rows={5} />
              <p className="muted">Describes the final rendered style. The motion and timing you set in Direct stay exactly as they are.</p>
              <div className="reference-block"><span className="inspector-label">Reference image <span>· unavailable</span></span><p className="muted">The current backend API does not accept reference image uploads. Your prompt and surprise mode are supported.</p></div>
            </> : <div className="surprise-note"><Icon name="spark" size={16} strokeWidth={1.8} /><p className="muted">CueFX will pick a finished style for you. You’ll see it for the first time in Review.</p></div>}
          </div>
        </div>
      </div>
      <div className="look-footer">{!canGenerate && <p className="look-validation"><Icon name="info" size={14} strokeWidth={2} /><span>{!approved ? "Approve the composite above to unlock Generate." : !generationAvailable ? "Automatic generation is not enabled on the backend yet." : "Describe a look, or choose Surprise me."}</span></p>}<Button disabled={!canGenerate} className="primary-button generate-button" onClick={onGenerate}><Icon name="spark" size={16} strokeWidth={2.2} />Generate</Button></div>
    </aside>
  </main>;
}

function CreatingPage({ clip, settings, status, progress, elapsed, error, resultUrl, canCancel, onCancel, onRestart, onBackLook, onBackDirect, onViewResult }: {
  clip: Clip; settings: DirectSettings; status: "running" | "stopped" | "done" | "failed"; progress: number; elapsed: number; error: string; resultUrl: string; canCancel: boolean;
  onCancel: () => void; onRestart: () => void; onBackLook: () => void; onBackDirect: () => void; onViewResult: () => void;
}) {
  const stages = ["Preparing shot", "Building reference frame", "Rendering pass 1", "Refining pass 2", "Blending into footage"];
  const stageIndex = Math.min(stages.length - 1, Math.floor(progress / 20));
  const elapsedText = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`;
  return <main className="creating-page">
    <section className="creating-card">
      {status === "stopped" ? <div className="generation-result stopped-result"><div className="result-icon result-stopped"><Icon name="close" size={22} /></div><p className="section-eyebrow">GENERATION CANCELLED</p><h1>The backend job was cancelled</h1><p className="muted max-w-[390px]">You can retry generation when you’re ready.</p><div className="mt-6 flex flex-wrap justify-center gap-2"><Button className="ghost-button" onClick={onBackLook}>Back to Look</Button><Button className="primary-button" onClick={onRestart}><Icon name="refresh" />Generate again</Button></div></div> : status === "failed" ? <div className="generation-result stopped-result"><div className="result-icon result-stopped"><Icon name="warning" size={22} /></div><p className="section-eyebrow">GENERATION FAILED</p><h1>The backend could not finish</h1><p role="alert" className="muted max-w-[420px]">{error}</p><div className="mt-6 flex flex-wrap justify-center gap-2"><Button className="ghost-button" onClick={onBackLook}>Back to Look</Button><Button className="primary-button" onClick={onRestart}><Icon name="refresh" />Retry generation</Button></div></div> : status === "done" ? <div className="generation-result"><div className="result-icon"><Icon name="check" size={24} /></div><p className="section-eyebrow">GENERATION COMPLETE</p><h1>Your effect is ready</h1><p className="muted">Generated in {elapsedText}. Play it back, then download or keep iterating.</p><div className="result-preview"><video src={resultUrl} autoPlay muted loop playsInline /><span className="result-badge"><Icon name="check" size={12} />Backend render ready</span></div><Button className="primary-button mt-5" onClick={onViewResult}>View your result <Icon name="next" /></Button></div> : <>
        <div className="creating-heading"><div className="loading-spinner" /><p className="section-eyebrow">CUEFX STUDIO</p><h1>Creating your effect</h1><p className="muted">The CueFX backend is processing the render job.</p></div>
        <div className="generation-progress"><div className="flex items-center justify-between"><strong>{stages[stageIndex]}</strong><span className="mono text-[11px] text-zinc-400">{Math.round(progress)}%</span></div><div className="generation-track"><span style={{ width: `${progress}%` }} /></div><div className="generation-steps">{stages.map((stage, index) => <div key={stage} className={`generation-step ${index < stageIndex ? "generation-step-done" : index === stageIndex ? "generation-step-active" : ""}`}><span>{index < stageIndex ? <Icon name="check" size={10} /> : index + 1}</span>{stage}</div>)}</div></div>
        <div className="creating-bottom"><div><span className="section-eyebrow">TIME ELAPSED</span><p className="mono mt-1 text-[14px] text-zinc-200">{elapsedText}</p></div><div className="flex items-center gap-2"><Button className="ghost-button" onClick={onBackDirect}>Back to Direct</Button><Button className="ghost-button" disabled={!canCancel} onClick={onCancel}>{canCancel ? "Cancel generation" : "Starting backend job…"}</Button></div></div>
        <p className="muted mt-4 text-center">The render runs on the backend. Keep this page open to follow its status.</p>
      </>}
    </section>
  </main>;
}

async function exportPreviewVideo(clip: Clip, settings: DirectSettings) {
  if (!("MediaRecorder" in window) || !HTMLCanvasElement.prototype.captureStream) throw new Error("Video export is not supported in this browser.");
  const width = 1280, height = 720, duration = Math.max(2, Math.min(clip.duration || 4.04, 60));
  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("Could not prepare the render canvas.");
  let video: HTMLVideoElement | null = null; let image: HTMLImageElement | null = null;
  if (clip.isVideo) {
    video = document.createElement("video"); video.src = clip.url; video.muted = true; video.playsInline = true; video.preload = "auto";
    await new Promise<void>((resolve, reject) => { video!.onloadeddata = () => resolve(); video!.onerror = () => reject(new Error("This video could not be decoded for export.")); video!.load(); });
    video.currentTime = 0; await video.play();
  } else {
    image = new Image();
    await new Promise<void>((resolve, reject) => { image!.onload = () => resolve(); image!.onerror = () => reject(new Error("Could not load the preview image.")); image!.src = "/assets/sample-footage.jpg"; });
  }
  const stream = canvas.captureStream(24);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  const recorder = new MediaRecorder(stream, { mimeType: mime });
  const parts: BlobPart[] = [];
  const finished = new Promise<Blob>((resolve, reject) => { recorder.ondataavailable = (event) => { if (event.data.size) parts.push(event.data); }; recorder.onerror = () => reject(new Error("The preview could not be exported.")); recorder.onstop = () => resolve(new Blob(parts, { type: "video/webm" })); });
  recorder.start();
  const started = performance.now();
  await new Promise<void>((resolve) => {
    const draw = () => {
      const elapsed = (performance.now() - started) / 1000;
      ctx!.clearRect(0, 0, width, height);
      if (video) { if (video.readyState >= 2) ctx!.drawImage(video, 0, 0, width, height); }
      else if (image) ctx!.drawImage(image, 0, 0, width, height);
      const fx = effectFrameAt(settings, video ? video.currentTime : elapsed);
      if (fx.visible && fx.opacity > 0) {
        const x = width * fx.x / 100, y = height * fx.y / 100, r = settings.size * .68 * fx.scale;
        const gradient = ctx!.createRadialGradient(x - r * .2, y - r * .24, 2, x, y, r * 1.9);
        gradient.addColorStop(0, "#ffffff"); gradient.addColorStop(.28, settings.color); gradient.addColorStop(.7, `${settings.color}aa`); gradient.addColorStop(1, `${settings.color}00`);
        ctx!.save(); ctx!.globalAlpha = Math.min(1, .45 + settings.glow / 180) * fx.opacity; ctx!.fillStyle = gradient; ctx!.beginPath(); ctx!.arc(x, y, r * 1.9, 0, Math.PI * 2); ctx!.fill(); ctx!.restore();
      }
      if (elapsed >= duration || (video && video.ended)) { if (recorder.state !== "inactive") recorder.stop(); video?.pause(); resolve(); return; }
      requestAnimationFrame(draw);
    };
    draw();
  });
  const blob = await finished;
  stream.getTracks().forEach((track) => track.stop());
  return blob;
}

function ReviewPage({ duration, resultUrl, downloadUrl, elapsed, onEdit, onLook }: { duration: number; resultUrl: string; downloadUrl: string; elapsed: number; onEdit: () => void; onLook: () => void }) {
  const [toast, setToast] = useState("");
  const [playing, setPlaying] = useState(true);
  const [zoomed, setZoomed] = useState(false);
  const [time, setTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const clipDuration = duration || 4.04;
  const clock = (value: number) => {
    const cs = Math.max(0, Math.round(value * 100));
    const m = Math.floor(cs / 6000);
    const sec = (cs % 6000) / 100;
    return `${m}:${sec < 10 ? "0" : ""}${sec.toFixed(2)}`;
  };
  const scrubPct = `${Math.min(100, Math.max(0, (time / clipDuration) * 100))}%`;
  const elapsedText = `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s`;
  const lookName = "CueFX render";
  const seekTo = (value: number) => {
    const next = Math.min(clipDuration, Math.max(0, value));
    setTime(next);
    if (videoRef.current) videoRef.current.currentTime = next;
  };
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.loop = true;
    if (playing) void video.play().catch(() => setPlaying(false)); else video.pause();
  }, [playing]);
  usePlaybackClock(videoRef, playing, clipDuration, true, setTime, setPlaying);
  const downloadFinal = () => {
    if (!downloadUrl) { setToast("The backend did not provide a download link."); return; }
    window.location.assign(downloadUrl);
  };
  return <main className="review-page">
    <div className="review-stack">
      <div className="review-heading"><h1>Your effect is ready</h1><p className="muted review-sub">Generated in {elapsedText}. Play it back, then download or keep iterating.</p></div>
      <div className={`review-frame ${zoomed ? "review-frame-zoomed" : ""}`}>
        {resultUrl ? <video ref={videoRef} src={resultUrl} muted loop playsInline className="stage-media" onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)} /> : <div role="alert" className="composite-loading">The backend did not return a final video.</div>}
        <div className="review-badge"><span className="review-badge-dot" />Final render · {lookName}</div>
        <div className="overlay-player">
          <label className="overlay-scrub"><span className="overlay-track" /><span className="overlay-fill overlay-fill-review" style={{ width: scrubPct }} /><span className="overlay-thumb" style={{ left: scrubPct }} /><input aria-label="Preview position" type="range" min={0} max={clipDuration} step={0.01} value={Math.min(time, clipDuration)} onChange={(event) => { setPlaying(false); seekTo(Number(event.target.value)); }} /></label>
          <div className="overlay-row"><Button className="play-button" aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying((value) => !value)}><Icon name={playing ? "pause" : "play"} size={13} /></Button><span className="mono overlay-time">{clock(time)} <span>/ {clock(clipDuration)}</span></span><span className="overlay-spacer" /><Button className="ghost-button overlay-zoom" aria-pressed={zoomed} onClick={() => setZoomed((value) => !value)}><Icon name="zoom" size={13} strokeWidth={1.9} />{zoomed ? "Fit" : "Zoom"}</Button></div>
        </div>
      </div>
      <div className="review-actions">
        <div className="review-secondary"><Button className="review-ghost" onClick={onEdit}><Icon name="layers" size={15} strokeWidth={1.9} />Edit in Direct</Button><Button className="review-ghost" onClick={onLook}><Icon name="spark" size={15} strokeWidth={1.9} />Try another look</Button></div>
        <Button className="primary-button review-download" onClick={downloadFinal}><Icon name="download" size={16} strokeWidth={2.2} />Download final video</Button>
      </div>
      <p className="muted review-disclosure">This video was rendered by the CueFX backend. Audio is not included in the backend result.</p>
      {toast && <p role="status" className="export-status">{toast}</p>}
    </div>
  </main>;
}

export default function CueFXStudio() {
  const [screen, setScreen] = useState<Screen>("footage");
  const [clip, setClip] = useState<Clip | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [directSettings, setDirectSettings] = useState<DirectSettings>(initialDirect);
  const directRef = useRef<DirectSettings>(initialDirect);
  const undoStack = useRef<DirectSettings[]>([]);
  const redoStack = useRef<DirectSettings[]>([]);
  const [currentTime, setCurrentTime] = useState(1.5);
  const [compositeReady, setCompositeReady] = useState(false);
  const [compositeRetry, setCompositeRetry] = useState(0);
  const [compositeUrl, setCompositeUrl] = useState("");
  const [compositeError, setCompositeError] = useState("");
  const [standinId, setStandinId] = useState("");
  const [approvedCue, setApprovedCue] = useState<BackendCue | null>(null);
  const [generationAvailable, setGenerationAvailable] = useState(false);
  const [generationError, setGenerationError] = useState("");
  const [generationJobId, setGenerationJobId] = useState("");
  const [finalVideoUrl, setFinalVideoUrl] = useState("");
  const [finalDownloadUrl, setFinalDownloadUrl] = useState("");
  const [approved, setApproved] = useState(false);
  const [lookPrompt, setLookPrompt] = useState("");
  const [lookMode, setLookMode] = useState<"describe" | "surprise">("describe");
  const [generationStatus, setGenerationStatus] = useState<"running" | "stopped" | "done" | "failed">("running");
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generationElapsed, setGenerationElapsed] = useState(0);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [directDrawer, setDirectDrawer] = useState<"renders" | "shortcuts" | null>(null);
  const [selectedBody, setSelectedBody] = useState("Right palm");
  const [trackingComplete, setTrackingComplete] = useState(false);
  const [backendCue, setBackendCue] = useState<BackendCue | null>(null);
  const uploadObjectUrl = useRef<string | null>(null);
  const generationControllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => {
    if (uploadObjectUrl.current) URL.revokeObjectURL(uploadObjectUrl.current);
    generationControllerRef.current?.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void apiRequest<BackendHealth>("api/health", { signal: controller.signal })
      .then((health) => setGenerationAvailable(health.features?.ltx_overlap_generate === true))
      .catch(() => setGenerationAvailable(false));
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (screen !== "look") return;
    setCompositeReady(false);
    setCompositeUrl("");
    setCompositeError("");
    setApprovedCue(null);
    setStandinId("");
    if (!clip?.backend || !backendCue) {
      setCompositeError("Upload and track a video before rendering a backend composite. Built-in sample clips are local previews only.");
      return;
    }
    const backend = clip.backend;
    const controller = new AbortController();
    const render = async () => {
      try {
        // Prompt text is sent separately to generation; keeping it out of the approved cue makes
        // the stand-in cue and generation cue identical, as required by the backend.
        const draft = buildRenderCue(backendCue, directSettings, "");
        const cue = await apiRequest<BackendCue>("api/cue/normalize", {
          method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ cue: draft }), signal: controller.signal,
        });
        const warnings = await apiRequest<{ severity?: string; message?: string }[]>("api/validate", {
          method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ cue }), signal: controller.signal,
        });
        const invalid = warnings.find((warning) => warning.severity === "error");
        if (invalid) throw new Error(invalid.message ?? "Fix the cue validation errors before rendering.");
        const started = await apiRequest<BackendStandinStatus>("api/standin", {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ cue, clip_id: backend.clip_id }), signal: controller.signal,
        });
        const completed = await waitForStandin(started, controller.signal);
        if (controller.signal.aborted || !completed.result) return;
        setApprovedCue(cue);
        setStandinId(completed.standin_id);
        setCompositeUrl(apiUrl(completed.result.files.hardcomp_preview));
        setCompositeReady(true);
      } catch (error) {
        if (!controller.signal.aborted) setCompositeError(error instanceof Error ? error.message : "The backend could not render this composite.");
      }
    };
    void render();
    return () => controller.abort();
  }, [screen, clip, backendCue, directSettings, compositeRetry]);

  useEffect(() => {
    if (!clip) return;
    return () => { if (clip.isVideo && clip.url.startsWith("blob:")) URL.revokeObjectURL(clip.url); };
  }, [clip]);

  const patchDirect = useCallback((patch: Partial<DirectSettings>, recordUndo = true) => {
    if (recordUndo) {
      undoStack.current.push(structuredClone(directRef.current));
      if (undoStack.current.length > 100) undoStack.current.shift();
      redoStack.current = [];
    }
    const next = { ...directRef.current, ...patch };
    directRef.current = next;
    setDirectSettings(next);
    setFinalVideoUrl("");
    setFinalDownloadUrl("");
    if (recordUndo) { setCanUndo(true); setCanRedo(false); }
  }, []);
  const undo = () => {
    const previous = undoStack.current.pop(); if (!previous) return;
    redoStack.current.push(structuredClone(directRef.current)); directRef.current = previous; setDirectSettings(previous);
    setFinalVideoUrl(""); setFinalDownloadUrl("");
    setCanUndo(undoStack.current.length > 0); setCanRedo(true);
  };
  const redo = () => {
    const next = redoStack.current.pop(); if (!next) return;
    undoStack.current.push(structuredClone(directRef.current)); directRef.current = next; setDirectSettings(next);
    setFinalVideoUrl(""); setFinalDownloadUrl("");
    setCanUndo(true); setCanRedo(redoStack.current.length > 0);
  };

  const beginUpload = async (file: File) => {
    setUploadError("");
    const validType = /\.(mov|mp4|mkv|webm)$/i.test(file.name);
    if (!validType) { setUploadError("Choose an MP4, MOV, MKV, or WebM video."); return; }
    if (file.size > 500 * 1024 * 1024) { setUploadError("This video is larger than 500 MB."); return; }
    const url = URL.createObjectURL(file);
    if (uploadObjectUrl.current) URL.revokeObjectURL(uploadObjectUrl.current);
    uploadObjectUrl.current = url;
    let duration = 4.04;
    try {
      const probe = document.createElement("video"); probe.preload = "metadata"; probe.src = url;
      duration = await new Promise<number>((resolve) => {
        const timeout = window.setTimeout(() => resolve(4.04), 1800);
        probe.onloadedmetadata = () => { window.clearTimeout(timeout); resolve(Number.isFinite(probe.duration) ? probe.duration : 4.04); };
        probe.onerror = () => { window.clearTimeout(timeout); resolve(4.04); };
      });
    } catch { duration = 4.04; }
    if (duration > 60) { URL.revokeObjectURL(url); uploadObjectUrl.current = null; setUploadError("Use a clip that is 60 seconds or shorter."); return; }
    setTrackingComplete(false);
    setBackendCue(null);
    setDirectSettings(initialDirect);
    directRef.current = initialDirect;
    undoStack.current = [];
    redoStack.current = [];
    setFinalVideoUrl(""); setFinalDownloadUrl("");
    setSelectedBody("Right palm");
    setUploadName(file.name);
    setUploadProgress(0);
    try {
      const backend = await uploadClip(file, setUploadProgress);
      URL.revokeObjectURL(url);
      uploadObjectUrl.current = null;
      const normalizedDuration = backend.frames / backend.fps;
      setClip({
        name: backend.clip,
        duration: normalizedDuration,
        url: apiUrl(`api/clips/${backend.clip_id}/video`),
        isVideo: true,
        file,
        backend,
      });
      setCurrentTime(Math.min(1.5, normalizedDuration));
      window.setTimeout(() => setUploadProgress(null), 350);
    } catch (error) {
      URL.revokeObjectURL(url);
      uploadObjectUrl.current = null;
      setUploadProgress(null);
      setUploadError(error instanceof Error ? error.message : "The backend could not upload this clip.");
    }
  };
  const chooseSample = async (sample: typeof sampleClips[number]) => {
    setTrackingComplete(false);
    setBackendCue(null);
    setDirectSettings(initialDirect);
    directRef.current = initialDirect;
    undoStack.current = [];
    redoStack.current = [];
    setFinalVideoUrl(""); setFinalDownloadUrl("");
    setSelectedBody("Right palm");
    setClip(null);
    setCurrentTime(Math.min(1.5, sample.duration)); setUploadError("");
    setUploadName(`${sample.name}.webm`);
    setUploadProgress(0);
    try {
      const video = await createBundledSampleVideo(sample.duration);
      await beginUpload(video);
    } catch (error) {
      setUploadProgress(null);
      setUploadError(error instanceof Error ? error.message : "The bundled test video could not be prepared.");
    }
  };
  const goDirect = () => { if (clip) setScreen("direct"); };
  const goLook = () => { setApproved(false); setScreen("look"); };
  const startGeneration = async () => {
    setGenerationProgress(0);
    setGenerationElapsed(0);
    setGenerationError("");
    setGenerationJobId("");
    setFinalVideoUrl("");
    setFinalDownloadUrl("");
    setGenerationStatus("running");
    setScreen("creating");
    if (!clip?.backend || !approvedCue || !standinId || !generationAvailable) {
      setGenerationError("The backend is not ready for automatic generation. Check its health and generation feature configuration.");
      setGenerationStatus("failed");
      return;
    }
    const controller = new AbortController();
    generationControllerRef.current?.abort();
    generationControllerRef.current = controller;
    try {
      const cue = buildRenderCue(approvedCue, directSettings, "");
      const job = await apiRequest<BackendJob>("api/generate", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          cue,
          clip_id: clip.backend.clip_id,
          standin_id: standinId,
          prompt_mode: lookMode === "surprise" ? "surprise" : "user",
          prompt_text: lookMode === "surprise" ? null : lookPrompt.trim(),
          saved_look_identity: typeof cue.look === "string" ? cue.look : null,
          experimental_ltx_overlap: true,
        }),
        signal: controller.signal,
      });
      setGenerationJobId(job.id);
      const completed = await waitForJob(job.id, controller.signal, (update) => {
        setGenerationProgress(Math.max(0, Math.min(100, update.progress * 100)));
        setGenerationElapsed(update.elapsed_s);
      });
      if (controller.signal.aborted) return;
      if (!completed.files?.final_preview || !completed.files.final_download) throw new Error("The backend completed without returning both preview and download files.");
      setFinalVideoUrl(apiUrl(completed.files.final_preview));
      setFinalDownloadUrl(apiUrl(completed.files.final_download));
      setGenerationStatus("done");
      setGenerationProgress(100);
    } catch (error) {
      if (!controller.signal.aborted) {
        setGenerationError(error instanceof Error ? error.message : "The backend generation failed.");
        setGenerationStatus("failed");
      }
    }
  };
  const cancelGeneration = async () => {
    if (generationJobId) {
      try {
        const job = await apiRequest<BackendJob>(`api/generate/${encodeURIComponent(generationJobId)}/cancel`, { method: "POST" });
        if (job.status === "done" && job.files?.final_preview && job.files.final_download) {
          setFinalVideoUrl(apiUrl(job.files.final_preview));
          setFinalDownloadUrl(apiUrl(job.files.final_download));
          setGenerationProgress(100);
          setGenerationStatus("done");
          return;
        }
        if (job.status !== "cancelled") throw new Error(job.error?.message ?? "The backend did not confirm cancellation.");
      } catch (error) {
        setGenerationError(error instanceof Error ? error.message : "Could not confirm cancellation. The backend job may still be running.");
        setGenerationStatus("failed");
        generationControllerRef.current?.abort();
        return;
      }
    }
    generationControllerRef.current?.abort();
    setGenerationStatus("stopped");
  };
  const goScreen = (next: Screen) => {
    if (next === "footage" || next === "attach" && clip || next === "direct" && clip && trackingComplete || next === "look" && clip && trackingComplete || next === "review" && finalVideoUrl) setScreen(next);
  };

  const trackBackendTarget = async (input: { point: BodyPoint; side: "left" | "right"; seed: Point | null; frame: number }) => {
    if (!clip?.backend) throw new Error("Upload a video to connect to the backend. The built-in sample clips are local previews only.");
    const [plateWidth, plateHeight] = clip.backend.plate_size;
    const bodyPoint = { part: input.point, side: input.point === "head" ? null : input.side };
    const body: Record<string, unknown> = {
      clip_id: clip.backend.clip_id,
      hand: input.side,
      body_point: bodyPoint,
    };
    if (input.seed) {
      body.hand_seed = {
        frame: input.frame,
        x: input.seed.x / 100 * plateWidth,
        y: input.seed.y / 100 * plateHeight,
      };
    }
    const result = await apiRequest<BackendCue>("api/cue/new", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    setBackendCue(result);
    const direct = directSettingsFromCue(result);
    directRef.current = direct;
    undoStack.current = [];
    redoStack.current = [];
    setDirectSettings(direct);
    setCanUndo(false);
    setCanRedo(false);
    const actualBody = typeof result.body_point === "object" && result.body_point !== null ? result.body_point as { part?: string; side?: string | null } : null;
    const actualPart = actualBody?.part ?? input.point;
    const actualSide = actualPart === "head" ? "" : `${actualBody?.side ?? input.side} `;
    setSelectedBody(`${actualSide}${actualPart}`.replace(/^\w/, (value) => value.toUpperCase()));
    setCurrentTime(Math.min(result.frames / result.fps, typeof result.appear_frame === "number" ? result.appear_frame / result.fps : 0));
    setFinalVideoUrl("");
    setFinalDownloadUrl("");
  };
  const retrackBody = async (body: string) => {
    if (!clip?.backend || !backendCue) throw new Error("Upload and track a clip before changing the tracking target.");
    const lower = body.toLowerCase();
    const part = (lower.endsWith("palm") ? "palm" : lower.endsWith("fist") ? "fist" : lower.endsWith("foot") ? "foot" : lower.endsWith("knee") ? "knee" : lower.endsWith("shoulder") ? "shoulder" : "head") as BodyPoint;
    const side = lower.startsWith("left ") ? "left" : lower.startsWith("right ") ? "right" : (backendCue.hand === "left" ? "left" : "right");
    const request: Record<string, unknown> = {
      clip_id: clip.backend.clip_id,
      hand: side,
      body_point: { part, side: part === "head" ? null : side },
    };
    if (part === "palm") {
      const seed = backendCue.hand_seed;
      if (typeof seed !== "object" || seed === null || typeof (seed as Record<string, unknown>).f !== "number" || typeof (seed as Record<string, unknown>).x !== "number" || typeof (seed as Record<string, unknown>).y !== "number") {
        throw new Error("The backend cue is missing the seed needed to track a palm again.");
      }
      const cueSeed = seed as Record<string, number>;
      request.hand_seed = { frame: cueSeed.f, x: cueSeed.x, y: cueSeed.y };
    }
    const updated = await apiRequest<BackendCue>("api/cue/new", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(request) });
    setBackendCue(updated);
    const [width, height] = updated.plate_size;
    const releaseFrame = typeof directRef.current.releaseAt === "number" ? Math.round(directRef.current.releaseAt * updated.fps) : 0;
    const oldPoint = trackPointAt(backendCue.track, releaseFrame, updated.fps);
    const newPoint = trackPointAt(updated.track, releaseFrame, updated.fps);
    if (oldPoint && newPoint) {
      const oldAnchor = frameToEditorPercent(oldPoint.x, oldPoint.y, width, height);
      const newAnchor = frameToEditorPercent(newPoint.x, newPoint.y, width, height);
      const dx = newAnchor.x - oldAnchor.x;
      const dy = newAnchor.y - oldAnchor.y;
      const moved: DirectSettings = {
        ...directRef.current,
        points: directRef.current.points.map((point, index) => index === 0 ? point : ({ ...point, x: Math.max(-20, Math.min(120, point.x + dx)), y: Math.max(-20, Math.min(120, point.y + dy)) })),
      };
      directRef.current = moved;
      setDirectSettings(moved);
    }
    setSelectedBody(body);
    setFinalVideoUrl(""); setFinalDownloadUrl("");
  };

  return <div className="cuefx-app">
    <StudioHeader screen={screen} clip={clip} onNavigate={goScreen} canUndo={canUndo} canRedo={canRedo} onUndo={undo} onRedo={redo} onDrawer={setDirectDrawer} />
    {screen === "footage" && <UploadPage clip={clip} onUpload={beginUpload} onSample={chooseSample} progress={uploadProgress} uploadName={uploadName} error={uploadError} onClearError={() => setUploadError("")} onContinue={() => setScreen("attach")} />}
    {screen === "attach" && clip && <AttachPage clip={clip} track={backendCue?.track ?? null} onContinue={() => { setTrackingComplete(true); setScreen("direct"); }} onSelectBody={setSelectedBody} onTrack={trackBackendTarget} />}
    {screen === "direct" && clip && <DirectPage clip={clip} settings={directSettings} patch={patchDirect} currentTime={currentTime} setCurrentTime={setCurrentTime} selectedBody={selectedBody} onBodyChange={retrackBody} onBack={() => setScreen("attach")} onContinue={goLook} onReview={() => { if (finalVideoUrl) setScreen("review"); }} canUndo={canUndo} canRedo={canRedo} onUndo={undo} onRedo={redo} drawer={directDrawer} setDrawer={setDirectDrawer} trackingWarnings={lostTrackRanges(backendCue)} resultUrl={finalVideoUrl} track={backendCue?.track ?? null} />}
    {screen === "look" && clip && <LookPage clip={clip} settings={directSettings} approved={approved} setApproved={setApproved} ready={compositeReady} compositeUrl={compositeUrl} compositeError={compositeError} generationAvailable={generationAvailable} prompt={lookPrompt} setPrompt={setLookPrompt} mode={lookMode} setMode={setLookMode} onRecheck={() => { setApproved(false); setCompositeReady(false); setCompositeRetry((value) => value + 1); }} onBack={goDirect} onGenerate={startGeneration} />}
    {screen === "creating" && clip && <CreatingPage clip={clip} settings={directSettings} status={generationStatus} progress={generationProgress} elapsed={generationElapsed} error={generationError} resultUrl={finalVideoUrl} canCancel={Boolean(generationJobId)} onCancel={cancelGeneration} onRestart={startGeneration} onBackLook={() => setScreen("look")} onBackDirect={goDirect} onViewResult={() => setScreen("review")} />}
    {screen === "review" && clip && <ReviewPage duration={clip.duration} resultUrl={finalVideoUrl} downloadUrl={finalDownloadUrl} elapsed={generationElapsed} onEdit={goDirect} onLook={() => setScreen("look")} />}
  </div>;
}

