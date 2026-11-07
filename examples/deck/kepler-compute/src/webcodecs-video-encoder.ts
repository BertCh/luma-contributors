// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {buildMp4, buildWebm, type EncodedVideoSample} from './video-muxers';

/**
 * Offline video encoding with WebCodecs. Frames are encoded as fast as they are rendered (no
 * real-time pacing, unlike MediaRecorder) with exact timestamps, then muxed to MP4 (H.264) or WebM
 * (VP9/VP8) in memory. Feature detection is asynchronous because isConfigSupported is.
 */
export type WebCodecsVideoFormat = 'mp4' | 'webm';

export type EncoderCodecChoice = {
  format: WebCodecsVideoFormat;
  codec: string;
  muxCodec: 'avc' | 'vp8' | 'vp9';
};

export function chooseAvcCodecString(width: number, height: number): string {
  // High profile; level 4.0 covers 1080p30, 5.1 covers 4K.
  return width * height > 2_097_152 ? 'avc1.640033' : 'avc1.640028';
}

export function getEncoderCandidates(
  format: WebCodecsVideoFormat | 'auto',
  width: number,
  height: number
): EncoderCodecChoice[] {
  const mp4: EncoderCodecChoice = {
    format: 'mp4',
    codec: chooseAvcCodecString(width, height),
    muxCodec: 'avc'
  };
  const webm: EncoderCodecChoice[] = [
    {format: 'webm', codec: 'vp09.00.40.08', muxCodec: 'vp9'},
    {format: 'webm', codec: 'vp8', muxCodec: 'vp8'}
  ];
  if (format === 'mp4') return [mp4];
  if (format === 'webm') return webm;
  return [mp4, ...webm];
}

type VideoEncoderScope = {
  VideoEncoder?: {
    isConfigSupported(config: Record<string, unknown>): Promise<{supported?: boolean}>;
  };
};

export type VideoEncoderLatencyMode = 'quality' | 'realtime';

/**
 * Encoder configuration shared by the support probe and the real encoder. H.264 asks for the 'avc'
 * bitstream format (length-prefixed samples plus an avcC record). Quality latency mode lets encoders
 * that support it use B-frames; realtime mode is the fallback and never reorders frames.
 */
export function buildEncoderConfig(
  choice: EncoderCodecChoice,
  width: number,
  height: number,
  bitrate: number,
  latencyMode: VideoEncoderLatencyMode,
  fps?: number
): Record<string, unknown> {
  return {
    codec: choice.codec,
    width,
    height,
    bitrate,
    ...(fps ? {framerate: fps} : {}),
    latencyMode,
    ...(choice.muxCodec === 'avc' ? {avc: {format: 'avc'}} : {})
  };
}

/** Pick 'quality' when requested and supported, otherwise fall back to 'realtime'. */
export async function resolveLatencyMode(
  choice: EncoderCodecChoice,
  width: number,
  height: number,
  bitrate: number,
  requestBFrames: boolean,
  scope: VideoEncoderScope = globalThis as never
): Promise<VideoEncoderLatencyMode> {
  if (!requestBFrames || !scope.VideoEncoder) return 'realtime';
  try {
    const result = await scope.VideoEncoder.isConfigSupported(
      buildEncoderConfig(choice, width, height, bitrate, 'quality')
    );
    return result.supported ? 'quality' : 'realtime';
  } catch {
    return 'realtime';
  }
}

/** Resolve the first supported encoder, or null when WebCodecs video encoding is unavailable. */
export async function detectWebCodecsEncoder(
  format: WebCodecsVideoFormat | 'auto',
  width: number,
  height: number,
  scope: VideoEncoderScope = globalThis as never
): Promise<EncoderCodecChoice | null> {
  if (
    !scope.VideoEncoder ||
    typeof (globalThis as {VideoFrame?: unknown}).VideoFrame === 'undefined'
  )
    return null;
  for (const candidate of getEncoderCandidates(format, width, height)) {
    try {
      const result = await scope.VideoEncoder.isConfigSupported({
        codec: candidate.codec,
        width,
        height,
        bitrate: 1_000_000,
        ...(candidate.muxCodec === 'avc' ? {avc: {format: 'avc'}} : {})
      });
      if (result.supported) return candidate;
    } catch {
      // Treat a throwing probe as unsupported and try the next codec.
    }
  }
  return null;
}

export type WebCodecsVideoEncoder = {
  format: WebCodecsVideoFormat;
  addFrame(source: CanvasImageSource, frameIndex: number, forceKeyFrame?: boolean): Promise<void>;
  finish(): Promise<Blob>;
  cancel(): void;
  /** Encoded sample count and how many left the encoder out of presentation order (B-frames). */
  getStatistics(): {sampleCount: number; reorderedSampleCount: number};
};

export function createWebCodecsVideoEncoder(options: {
  choice: EncoderCodecChoice;
  width: number;
  height: number;
  fps: number;
  bitrate: number;
  /** Defaults to 'quality' (allows B-frames where supported); use resolveLatencyMode to probe. */
  latencyMode?: VideoEncoderLatencyMode;
}): WebCodecsVideoEncoder {
  const {choice, width, height, fps, bitrate, latencyMode = 'quality'} = options;
  let highestTimestamp = -Infinity;
  let reorderedSampleCount = 0;
  const samples: EncodedVideoSample[] = [];
  let avcConfiguration: Uint8Array | null = null;
  let failure: Error | null = null;
  const frameMicroseconds = Math.round(1_000_000 / fps);
  const encoder = new VideoEncoder({
    output: (chunk, metadata) => {
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      const description = metadata?.decoderConfig?.description;
      if (description && !avcConfiguration)
        avcConfiguration = new Uint8Array(description as ArrayBuffer);
      // Chunks arrive in decode order; chunk.timestamp is the presentation time (pts).
      if (chunk.timestamp < highestTimestamp) reorderedSampleCount++;
      highestTimestamp = Math.max(highestTimestamp, chunk.timestamp);
      samples.push({
        data,
        timestampMicroseconds: chunk.timestamp,
        durationMicroseconds: chunk.duration ?? frameMicroseconds,
        isKeyFrame: chunk.type === 'key'
      });
    },
    error: error => {
      failure = error;
    }
  });
  encoder.configure(buildEncoderConfig(choice, width, height, bitrate, latencyMode, fps) as never);
  return {
    format: choice.format,
    async addFrame(source, frameIndex, forceKeyFrame) {
      if (failure) throw failure;
      // Keep the encoder queue short so decoded frames do not pile up in memory.
      while (encoder.encodeQueueSize > 8) await new Promise(resolve => setTimeout(resolve, 1));
      const frame = new VideoFrame(source, {
        timestamp: frameIndex * frameMicroseconds,
        duration: frameMicroseconds
      });
      try {
        encoder.encode(frame, {keyFrame: forceKeyFrame ?? frameIndex % (fps * 2) === 0});
      } finally {
        frame.close();
      }
    },
    async finish() {
      await encoder.flush();
      encoder.close();
      if (failure) throw failure;
      if (!samples.length) throw new Error('The encoder produced no frames.');
      if (choice.muxCodec === 'avc') {
        // The encoder supplies the avcC record on its first keyframe output.
        if (!avcConfiguration)
          throw new Error('The H.264 encoder did not provide a decoder configuration.');
        return new Blob([buildMp4({width, height, avcConfiguration, samples}) as BlobPart], {
          type: 'video/mp4'
        });
      }
      return new Blob([buildWebm({width, height, codec: choice.muxCodec, samples}) as BlobPart], {
        type: 'video/webm'
      });
    },
    cancel() {
      if (encoder.state !== 'closed') encoder.close();
    },
    getStatistics() {
      return {sampleCount: samples.length, reorderedSampleCount};
    }
  };
}
