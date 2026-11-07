// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  interpolateKeyframes,
  planVideoFrames,
  validateVideoPlan,
  type VideoCaptureOptions
} from './native-video-export';
import {
  createWebCodecsVideoEncoder,
  detectWebCodecsEncoder,
  resolveLatencyMode,
  type WebCodecsVideoFormat
} from './webcodecs-video-encoder';
import {createCanvasFrameCapturer} from './webgpu-canvas-capture';

export type WebCodecsCaptureOptions = Omit<VideoCaptureOptions, 'support'> & {
  format: WebCodecsVideoFormat | 'auto';
};

/**
 * Deterministic offline capture: each frame is rendered, copied after submit (see
 * webgpu-canvas-capture) and encoded immediately, so a 10 s clip does not need 10 s of wall time.
 * Restores view state and animation time on every exit path. Throws when WebCodecs cannot encode the
 * requested size so the caller can fall back to the MediaRecorder path.
 */
export async function captureVideoWithWebCodecs(options: WebCodecsCaptureOptions): Promise<Blob> {
  const {host, signal, width, height, fps} = options;
  validateVideoPlan(options.durationSeconds, fps, width, height);
  const choice = await detectWebCodecsEncoder(options.format, width, height);
  // WebCodecs cannot encode this size or format here; callers fall back to MediaRecorder.
  if (!choice)
    throw new Error('WebCodecs cannot encode this video size or format in this browser.');
  if (!host.getCanvas()) throw new Error('The map canvas is unavailable.');
  const plan = planVideoFrames(options.durationSeconds, fps);
  const animationRange = options.sweepAnimation ? (host.getAnimationRange?.() ?? null) : null;
  if (options.sweepAnimation && !animationRange)
    throw new Error('Add a time-enabled layer or filter animation to capture the time bar.');
  const originalView = host.getViewState();
  const originalTime = host.getAnimationTime?.() ?? null;
  const target = document.createElement('canvas');
  target.width = width;
  target.height = height;
  const context = target.getContext('2d')!;
  const capturer = createCanvasFrameCapturer(host);
  const bitrate = Math.min(40_000_000, Math.round(width * height * fps * 0.15));
  // Quality latency mode lets H.264 encoders use B-frames; the muxer writes ctts and an edit list.
  const latencyMode = await resolveLatencyMode(choice, width, height, bitrate, true);
  const encoder = createWebCodecsVideoEncoder({choice, width, height, fps, bitrate, latencyMode});
  try {
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
      const canvas = host.getCanvas();
      if (!canvas) throw new Error('The map canvas is unavailable.');
      const frame = await capturer.capture(canvas);
      context.drawImage(frame.source, 0, 0, width, height);
      await encoder.addFrame(target, index);
      options.onProgress?.(index + 1, plan.frameCount);
    }
    if (signal?.aborted) throw new Error('Video export was cancelled.');
    return await encoder.finish();
  } finally {
    encoder.cancel();
    host.setViewState(originalView);
    if (originalTime !== null) host.setAnimationTime?.(originalTime);
  }
}
