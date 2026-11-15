// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createCanvasFrameCapturer} from './webgpu-canvas-capture';

import {safeMod} from '@math.gl/core';

/**
 * Animation and video capture of the live WebGPU map canvas.
 *
 * This fallback uses MediaRecorder on a 2D canvas stream (WebM/MP4 as the browser provides).
 * native-webcodecs-capture handles offline encoding with the MP4/WebM muxers in video-muxers.
 * Frames are stepped deterministically (view state or animation time), then paced in real time because
 * MediaRecorder timestamps frames by wall clock. Capture therefore takes about as long as the video.
 */
export const MAXIMUM_VIDEO_FRAMES = 3600;
export const MAXIMUM_VIDEO_SECONDS = 120;
export const MAXIMUM_VIDEO_FPS = 60;
export const MAXIMUM_VIDEO_DIMENSION = 3840;
export const MAXIMUM_VIDEO_PIXELS = 8294400;

export type VideoViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  bearing: number;
  pitch: number;
};

export type VideoKeyframe = {time: number; viewState: VideoViewState};

export type VideoFramePlan = {
  frameCount: number;
  fps: number;
  /** Seconds of the plan for each frame (0 .. durationSeconds). */
  times: number[];
};

/** Live-map operations supplied by the host. The host owns the Kepler store and the canvas. */
export type MediaExportHost = {
  getCanvas(): HTMLCanvasElement | null;
  getViewState(): VideoViewState;
  setViewState(viewState: VideoViewState): void;
  /** Present when a time bar / layer animation exists. */
  getAnimationRange?(): [number, number] | null;
  getAnimationTime?(): number | null;
  setAnimationTime?(time: number): void;
  /** Resolve after the deck has presented a frame for the current state. */
  waitForRender(): Promise<void>;
};

let registeredHost: MediaExportHost | null = null;
export function registerMediaExportHost(host: MediaExportHost | null): void {
  registeredHost = host;
}
export function getMediaExportHost(): MediaExportHost | null {
  return registeredHost;
}

export type VideoSupport = {
  supported: boolean;
  mimeType: string | null;
  webCodecs: boolean;
  message: string;
};

export function detectVideoExportSupport(
  scope: {
    MediaRecorder?: {isTypeSupported(type: string): boolean};
    VideoEncoder?: unknown;
  } = globalThis as never
): VideoSupport {
  const webCodecs = typeof scope.VideoEncoder !== 'undefined';
  const recorder = scope.MediaRecorder;
  if (!recorder)
    return {
      supported: false,
      mimeType: null,
      webCodecs,
      message: 'Video export needs MediaRecorder, which this browser does not provide.'
    };
  const mimeType =
    ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(type =>
      recorder.isTypeSupported(type)
    ) ?? null;
  return mimeType
    ? {supported: true, mimeType, webCodecs, message: `Recording as ${mimeType}.`}
    : {
        supported: false,
        mimeType: null,
        webCodecs,
        message: 'This browser cannot record WebM or MP4 video from a canvas.'
      };
}

export function validateVideoPlan(
  durationSeconds: number,
  fps: number,
  width: number,
  height: number
): void {
  // Bounds protect memory and recording time: <=120 s, <=60 fps, <=3600 frames, <=4K, <=8.3 MP.
  if (
    !(durationSeconds > 0) ||
    durationSeconds > MAXIMUM_VIDEO_SECONDS ||
    !Number.isInteger(fps) ||
    fps < 1 ||
    fps > MAXIMUM_VIDEO_FPS ||
    Math.ceil(durationSeconds * fps) > MAXIMUM_VIDEO_FRAMES ||
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 2 ||
    height < 2 ||
    width > MAXIMUM_VIDEO_DIMENSION ||
    height > MAXIMUM_VIDEO_DIMENSION ||
    width * height > MAXIMUM_VIDEO_PIXELS
  )
    throw new Error(
      `Choose up to ${MAXIMUM_VIDEO_SECONDS} seconds, ${MAXIMUM_VIDEO_FPS} fps, ${MAXIMUM_VIDEO_FRAMES} frames and 3840x2160 pixels.`
    );
}

export function planVideoFrames(durationSeconds: number, fps: number): VideoFramePlan {
  const frameCount = Math.max(2, Math.ceil(durationSeconds * fps) + 1);
  return {
    frameCount,
    fps,
    times: Array.from(
      {length: frameCount},
      (_, index) => (index / (frameCount - 1)) * durationSeconds
    )
  };
}

function interpolateAngle(from: number, to: number, fraction: number): number {
  const delta = safeMod(to - from + 540, 360) - 180;
  return from + delta * fraction;
}

/** Piecewise-linear keyframe interpolation (shortest-path bearing). Keyframes must be sorted by time. */
export function interpolateKeyframes(
  keyframes: readonly VideoKeyframe[],
  time: number
): VideoViewState {
  const first = keyframes[0];
  const last = keyframes[keyframes.length - 1];
  if (time <= first.time) return first.viewState;
  if (time >= last.time) return last.viewState;
  const nextIndex = keyframes.findIndex(keyframe => keyframe.time > time);
  const from = keyframes[nextIndex - 1];
  const to = keyframes[nextIndex];
  const fraction = (time - from.time) / (to.time - from.time);
  const mix = (a: number, b: number) => a + (b - a) * fraction;
  return {
    longitude: mix(from.viewState.longitude, to.viewState.longitude),
    latitude: mix(from.viewState.latitude, to.viewState.latitude),
    zoom: mix(from.viewState.zoom, to.viewState.zoom),
    bearing: interpolateAngle(from.viewState.bearing, to.viewState.bearing, fraction),
    pitch: mix(from.viewState.pitch, to.viewState.pitch)
  };
}

export type VideoCaptureOptions = {
  host: MediaExportHost;
  support: VideoSupport;
  durationSeconds: number;
  fps: number;
  width: number;
  height: number;
  /** View-state keyframes (times in seconds) and/or the host animation range swept over the duration. */
  keyframes?: readonly VideoKeyframe[];
  sweepAnimation?: boolean;
  signal?: AbortSignal;
  onProgress?(completed: number, count: number): void;
};

/**
 * Record the map. Always restores the original view state and animation time, including on error or
 * cancellation. It dispatches only view/animation updates, so device, buffers and uploads are untouched
 * (saved config is also restored by the final setViewState/setAnimationTime).
 */
export async function captureVideo(options: VideoCaptureOptions): Promise<Blob> {
  const {host, support, signal} = options;
  // Unsupported browsers get the detection message, not a failed recorder.
  if (!support.supported || !support.mimeType) throw new Error(support.message);
  validateVideoPlan(options.durationSeconds, options.fps, options.width, options.height);
  const source = host.getCanvas();
  if (!source) throw new Error('The map canvas is unavailable.');
  const plan = planVideoFrames(options.durationSeconds, options.fps);
  const animationRange = options.sweepAnimation ? (host.getAnimationRange?.() ?? null) : null;
  if (options.sweepAnimation && !animationRange)
    throw new Error('Add a time-enabled layer or filter animation to capture the time bar.');
  const originalView = host.getViewState();
  const originalTime = host.getAnimationTime?.() ?? null;
  const target = document.createElement('canvas');
  target.width = options.width;
  target.height = options.height;
  const context = target.getContext('2d')!;
  const capturer = createCanvasFrameCapturer(host);
  const stream = target.captureStream(0);
  const track = stream.getVideoTracks()[0] as MediaStreamTrack & {requestFrame?(): void};
  const recorder = new MediaRecorder(stream, {
    mimeType: support.mimeType,
    videoBitsPerSecond: Math.min(40_000_000, options.width * options.height * options.fps * 0.15)
  });
  const chunks: Blob[] = [];
  recorder.ondataavailable = event => event.data.size && chunks.push(event.data);
  const finished = new Promise<void>((resolve, reject) => {
    recorder.onstop = () => resolve();
    recorder.onerror = () => reject(new Error('The browser video encoder failed.'));
  });
  const start = performance.now();
  try {
    recorder.start();
    for (let index = 0; index < plan.frameCount; index++) {
      if (signal?.aborted) throw new Error('Video export was cancelled.');
      const time = plan.times[index];
      if (options.keyframes?.length)
        host.setViewState(interpolateKeyframes(options.keyframes, time));
      if (animationRange)
        host.setAnimationTime?.(
          animationRange[0] +
            (animationRange[1] - animationRange[0]) * (time / options.durationSeconds)
        );
      await host.waitForRender();
      const canvas = host.getCanvas() ?? source;
      // Copy in the submit task (GPU readback when drawImage is blank) so WebGPU frames are not empty.
      context.drawImage(
        (await capturer.capture(canvas)).source,
        0,
        0,
        options.width,
        options.height
      );
      track.requestFrame?.();
      options.onProgress?.(index + 1, plan.frameCount);
      const wait = start + ((index + 1) * 1000) / plan.fps - performance.now();
      if (wait > 0) await new Promise(resolve => setTimeout(resolve, wait));
    }
    if (signal?.aborted) throw new Error('Video export was cancelled.');
    recorder.stop();
    await finished;
    return new Blob(chunks, {type: support.mimeType.split(';')[0]});
  } finally {
    if (recorder.state !== 'inactive') recorder.stop();
    for (const videoTrack of stream.getTracks()) videoTrack.stop();
    host.setViewState(originalView);
    if (originalTime !== null) host.setAnimationTime?.(originalTime);
  }
}
