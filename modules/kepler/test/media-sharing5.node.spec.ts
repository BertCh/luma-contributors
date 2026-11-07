// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  buildMp4,
  buildWebm,
  computeMp4SampleTiming,
  type EncodedVideoSample
} from '../../../examples/deck/kepler-compute/src/video-muxers';
import {
  buildEncoderConfig,
  resolveLatencyMode
} from '../../../examples/deck/kepler-compute/src/webcodecs-video-encoder';
import {
  formatTiledProgress,
  getTiledProgressFraction,
  summarizeTiledImage
} from '../../../examples/deck/kepler-compute/src/native-tiled-capture-progress';
import {
  captureTiledImage,
  TiledCaptureCancelledError
} from '../../../examples/deck/kepler-compute/src/native-tiled-capture';

const FRAME = 40000;
// Decode order I P B B; presentation order is I B B P.
const B_FRAME_SAMPLES: EncodedVideoSample[] = [0, 3, 1, 2].map((slot, index) => ({
  data: new Uint8Array([index + 1, index + 1]),
  timestampMicroseconds: slot * FRAME,
  durationMicroseconds: FRAME,
  isKeyFrame: index === 0
}));

function findBoxPayload(bytes: Uint8Array, type: string): [number, number] | null {
  const needle = [...type].map(character => character.charCodeAt(0));
  for (let index = 4; index < bytes.length - 4; index++)
    if (needle.every((value, offset) => bytes[index + offset] === value)) {
      const size = new DataView(bytes.buffer, bytes.byteOffset).getUint32(index - 4);
      return [index + 4, index - 4 + size];
    }
  return null;
}

test('computeMp4SampleTiming derives monotonic dts and signed offsets for B-frames', () => {
  const timing = computeMp4SampleTiming(B_FRAME_SAMPLES);
  expect(timing.hasReordering).toBe(true);
  expect(timing.decodeDeltas).toEqual([FRAME, FRAME, FRAME, FRAME]);
  expect(timing.compositionOffsets).toEqual([0, 2 * FRAME, -FRAME, -FRAME]);
  expect(timing.mediaTime).toBe(0);
  expect(timing.presentationDuration).toBe(4 * FRAME);
});

test('buildMp4 writes a version 1 ctts box and an edit list for reordered samples', () => {
  const bytes = buildMp4({
    width: 16,
    height: 16,
    avcConfiguration: new Uint8Array([1, 100, 0, 40]),
    samples: B_FRAME_SAMPLES
  });
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const [cttsStart] = findBoxPayload(bytes, 'ctts')!;
  expect(view.getUint32(cttsStart)).toBe(0x01000000);
  expect(view.getUint32(cttsStart + 4)).toBe(3);
  expect(view.getUint32(cttsStart + 8)).toBe(1);
  expect(view.getInt32(cttsStart + 12)).toBe(0);
  expect(view.getInt32(cttsStart + 16)).toBe(1);
  expect(view.getInt32(cttsStart + 20)).toBe(2 * FRAME);
  expect(view.getUint32(cttsStart + 24)).toBe(2);
  expect(view.getInt32(cttsStart + 28)).toBe(-FRAME);
  const [elstStart] = findBoxPayload(bytes, 'elst')!;
  expect(view.getUint32(elstStart + 4)).toBe(1);
  expect(view.getUint32(elstStart + 8)).toBe(4 * FRAME);
  expect(view.getUint32(elstStart + 12)).toBe(0);
});

test('buildMp4 omits ctts when samples are already in presentation order', () => {
  const ordered = B_FRAME_SAMPLES.map((sample, index) => ({
    ...sample,
    timestampMicroseconds: index * FRAME
  }));
  const bytes = buildMp4({
    width: 16,
    height: 16,
    avcConfiguration: new Uint8Array(4),
    samples: ordered
  });
  expect(findBoxPayload(bytes, 'ctts')).toBeNull();
  expect(findBoxPayload(bytes, 'elst')).toBeNull();
});

test('buildMp4 uses an edit list when the first pts is not zero', () => {
  const shifted = B_FRAME_SAMPLES.slice(0, 1).map(sample => ({
    ...sample,
    timestampMicroseconds: FRAME
  }));
  const bytes = buildMp4({
    width: 16,
    height: 16,
    avcConfiguration: new Uint8Array(4),
    samples: shifted
  });
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const [elstStart] = findBoxPayload(bytes, 'elst')!;
  expect(view.getUint32(elstStart + 12)).toBe(FRAME);
});

test('buildEncoderConfig requests avc bitstream format and latency mode', () => {
  const choice = {format: 'mp4', codec: 'avc1.640028', muxCodec: 'avc'} as const;
  expect(buildEncoderConfig(choice, 640, 480, 1e6, 'quality', 30)).toMatchObject({
    latencyMode: 'quality',
    framerate: 30,
    avc: {format: 'avc'}
  });
  expect(
    buildEncoderConfig({...choice, muxCodec: 'vp8'}, 640, 480, 1e6, 'realtime')
  ).not.toHaveProperty('avc');
});

test('resolveLatencyMode falls back to realtime when quality is unsupported or throws', async () => {
  const choice = {format: 'mp4', codec: 'avc1.640028', muxCodec: 'avc'} as const;
  const make = (supported: boolean | 'throw') => ({
    VideoEncoder: {
      isConfigSupported: async () => {
        if (supported === 'throw') throw new Error('nope');
        return {supported};
      }
    }
  });
  expect(await resolveLatencyMode(choice, 64, 64, 1e6, true, make(true))).toBe('quality');
  expect(await resolveLatencyMode(choice, 64, 64, 1e6, true, make(false))).toBe('realtime');
  expect(await resolveLatencyMode(choice, 64, 64, 1e6, true, make('throw'))).toBe('realtime');
  expect(await resolveLatencyMode(choice, 64, 64, 1e6, false, make(true))).toBe('realtime');
});

// ---- WebM cues

function readVint(bytes: Uint8Array, offset: number, keepMarker: boolean): [number, number] {
  let length = 1;
  while (length <= 8 && !(bytes[offset] & (0x80 >> (length - 1)))) length++;
  let value = keepMarker ? bytes[offset] : bytes[offset] & (0xff >> length);
  for (let index = 1; index < length; index++) value = value * 256 + bytes[offset + index];
  return [value, length];
}

type Element = {id: number; start: number; end: number};

function readElements(bytes: Uint8Array, start: number, end: number): Element[] {
  const elements: Element[] = [];
  for (let offset = start; offset < end; ) {
    const [id, idLength] = readVint(bytes, offset, true);
    const [size, sizeLength] = readVint(bytes, offset + idLength, false);
    const dataStart = offset + idLength + sizeLength;
    elements.push({id, start: dataStart, end: dataStart + size});
    offset = dataStart + size;
  }
  return elements;
}

function readUnsigned(bytes: Uint8Array, element: Element): number {
  let value = 0;
  for (let index = element.start; index < element.end; index++) value = value * 256 + bytes[index];
  return value;
}

test('buildWebm writes SeekHead and Cues that point at keyframe clusters', () => {
  const samples: EncodedVideoSample[] = Array.from({length: 300}, (_, index) => ({
    data: new Uint8Array([index % 256]),
    timestampMicroseconds: index * 33333,
    durationMicroseconds: 33333,
    isKeyFrame: index % 60 === 0
  }));
  const bytes = buildWebm({width: 16, height: 16, codec: 'vp9', samples});
  const top = readElements(bytes, 0, bytes.length);
  const segment = top.find(element => element.id === 0x18538067)!;
  const children = readElements(bytes, segment.start, segment.end);
  expect(children[0].id).toBe(0x114d9b74);
  const cues = children.find(element => element.id === 0x1c53bb6b)!;
  expect(children[children.length - 1]).toBe(cues);
  const seekPositions = readElements(bytes, children[0].start, children[0].end).map(entry => {
    const [seekId, seekPosition] = readElements(bytes, entry.start, entry.end);
    return [readUnsigned(bytes, seekId), readUnsigned(bytes, seekPosition)];
  });
  const cuesSeek = seekPositions.find(([id]) => id === 0x1c53bb6b)!;
  expect(segment.start + cuesSeek[1]).toBe(cues.start - 12);
  const clusters = children.filter(element => element.id === 0x1f43b675);
  expect(clusters.length).toBeGreaterThan(1);
  const cuePoints = readElements(bytes, cues.start, cues.end);
  expect(cuePoints.length).toBe(clusters.length);
  cuePoints.forEach((point, index) => {
    const [time, positions] = readElements(bytes, point.start, point.end);
    const [, clusterPosition] = readElements(bytes, positions.start, positions.end);
    const cluster = clusters[index];
    // Cluster header is 4 id bytes + 8 size bytes before its data start.
    expect(segment.start + readUnsigned(bytes, clusterPosition)).toBe(cluster.start - 12);
    const firstChild = readElements(bytes, cluster.start, cluster.end)[0];
    expect(readUnsigned(bytes, time)).toBe(readUnsigned(bytes, firstChild));
  });
});

// ---- Tiled helpers

test('summarizeTiledImage reports tile counts and rejects out-of-range sizes', () => {
  const result = summarizeTiledImage(8192, 4096, 4096, 4096);
  expect(result).toMatchObject({summary: {tileCount: 2, columns: 2, rows: 1, megapixels: 33.55}});
  expect(summarizeTiledImage(40000, 100, 4096, 100)).toHaveProperty('error');
});

test('formatTiledProgress estimates the remaining time', () => {
  expect(formatTiledProgress(2, 8, 4000)).toBe('Rendered tile 2 of 8 (about 12 s left)');
  expect(formatTiledProgress(8, 8, 4000)).toBe('Rendered tile 8 of 8');
  expect(getTiledProgressFraction(3, 4)).toBe(0.75);
  expect(getTiledProgressFraction(1, 0)).toBe(0);
});

test('captureTiledImage rejects with a cancellation error when the signal aborts', async () => {
  const controller = new AbortController();
  controller.abort();
  const promise = captureTiledImage({
    width: 8192,
    height: 4096,
    tileWidth: 4096,
    tileHeight: 4096,
    signal: controller.signal,
    isCurrent: () => true,
    renderTile: async () => {
      throw new Error('should not render after abort');
    }
  });
  await expect(promise).rejects.toBeInstanceOf(TiledCaptureCancelledError);
});
