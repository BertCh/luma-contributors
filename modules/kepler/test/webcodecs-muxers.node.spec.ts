// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  buildMp4,
  buildWebm,
  type EncodedVideoSample
} from '../../../examples/deck/kepler-compute/src/video-muxers';
import {
  chooseAvcCodecString,
  detectWebCodecsEncoder,
  getEncoderCandidates
} from '../../../examples/deck/kepler-compute/src/webcodecs-video-encoder';
import {
  getAlignedBytesPerRow,
  isFrameBlank,
  unpackTextureRows
} from '../../../examples/deck/kepler-compute/src/webgpu-canvas-capture';
import {
  applyTileProjectionOffset,
  getAllowedTiledScales,
  getTileProjectionOffset
} from '../../../examples/deck/kepler-compute/src/tiled-projection-offset';

type Box = {type: string; start: number; end: number; payloadStart: number};

function readBoxes(bytes: Uint8Array, start = 0, end = bytes.length): Box[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const boxes: Box[] = [];
  for (let offset = start; offset < end; ) {
    const size = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    boxes.push({type, start: offset, end: offset + size, payloadStart: offset + 8});
    offset += size;
  }
  return boxes;
}

function findBox(bytes: Uint8Array, path: string[]): Box {
  let boxes = readBoxes(bytes);
  let found: Box | undefined;
  for (const [depth, type] of path.entries()) {
    found = boxes.find(box => box.type === type);
    if (!found) throw new Error(`missing ${type}`);
    if (depth < path.length - 1) {
      // Full boxes (stsd, dref) carry 8 extra bytes before children; none are traversed here.
      boxes = readBoxes(bytes, found.payloadStart, found.end);
    }
  }
  return found!;
}

const SAMPLES: EncodedVideoSample[] = [
  {
    data: new Uint8Array([1, 2, 3]),
    timestampMicroseconds: 0,
    durationMicroseconds: 33333,
    isKeyFrame: true
  },
  {
    data: new Uint8Array([4, 5]),
    timestampMicroseconds: 33333,
    durationMicroseconds: 33333,
    isKeyFrame: false
  },
  {
    data: new Uint8Array([6, 7, 8, 9]),
    timestampMicroseconds: 66666,
    durationMicroseconds: 33334,
    isKeyFrame: false
  }
];

test('MP4 muxer writes ftyp, moov before mdat, and correct chunk offset and sample tables', () => {
  const avcC = new Uint8Array([1, 0x64, 0, 0x28, 0xff, 0xe0, 0]);
  const mp4 = buildMp4({width: 320, height: 240, avcConfiguration: avcC, samples: SAMPLES});
  const top = readBoxes(mp4);
  expect(top.map(box => box.type)).toEqual(['ftyp', 'moov', 'mdat']);
  expect(top[top.length - 1].end).toBe(mp4.length);
  const mdat = top[2];
  expect(Array.from(mp4.subarray(mdat.payloadStart, mdat.end))).toEqual([
    1, 2, 3, 4, 5, 6, 7, 8, 9
  ]);
  const moov = findBox(mp4, ['moov']);
  const trak = readBoxes(mp4, moov.payloadStart, moov.end).find(box => box.type === 'trak')!;
  const mdia = readBoxes(mp4, trak.payloadStart, trak.end).find(box => box.type === 'mdia')!;
  const minf = readBoxes(mp4, mdia.payloadStart, mdia.end).find(box => box.type === 'minf')!;
  const stbl = readBoxes(mp4, minf.payloadStart, minf.end).find(box => box.type === 'stbl')!;
  const tables = Object.fromEntries(
    readBoxes(mp4, stbl.payloadStart, stbl.end).map(box => [box.type, box])
  );
  const view = new DataView(mp4.buffer, mp4.byteOffset, mp4.byteLength);
  // stco: version/flags, count, offset -> first byte of mdat payload.
  expect(view.getUint32(tables['stco'].payloadStart + 4)).toBe(1);
  expect(view.getUint32(tables['stco'].payloadStart + 8)).toBe(mdat.payloadStart);
  // stsz: version/flags, default size 0, count 3, then sizes.
  expect(view.getUint32(tables['stsz'].payloadStart + 8)).toBe(3);
  expect([12, 16, 20].map(offset => view.getUint32(tables['stsz'].payloadStart + offset))).toEqual([
    3, 2, 4
  ]);
  // stss lists only the keyframe (1-based).
  expect(view.getUint32(tables['stss'].payloadStart + 4)).toBe(1);
  expect(view.getUint32(tables['stss'].payloadStart + 8)).toBe(1);
  // stts run-length encodes the two equal 33333 deltas and the final 33334.
  expect(view.getUint32(tables['stts'].payloadStart + 4)).toBe(2);
  expect(view.getUint32(tables['stts'].payloadStart + 8)).toBe(2);
  expect(view.getUint32(tables['stts'].payloadStart + 12)).toBe(33333);
  // avcC is embedded verbatim.
  const text = String.fromCharCode(...mp4.subarray(0, moov.end));
  expect(text).toContain('avcC');
  expect(text).toContain('avc1');
  expect(Array.from(mp4.subarray(mp4.length - 9 - 0, mp4.length))).toEqual([
    1, 2, 3, 4, 5, 6, 7, 8, 9
  ]);
});

function readEbml(bytes: Uint8Array, start: number, end: number) {
  const elements: Array<{id: number; start: number; end: number}> = [];
  for (let offset = start; offset < end; ) {
    let id = bytes[offset];
    let idLength = 1;
    for (let mask = 0x80; !(id & mask) && idLength < 4; mask >>= 1) idLength++;
    id = 0;
    for (let index = 0; index < idLength; index++) id = id * 256 + bytes[offset + index];
    // Size is always 8 bytes with a 0x01 marker in this writer.
    expect(bytes[offset + idLength]).toBe(0x01);
    let size = 0;
    for (let index = 0; index < 7; index++)
      size = size * 256 + bytes[offset + idLength + 1 + index];
    const payloadStart = offset + idLength + 8;
    elements.push({id, start: payloadStart, end: payloadStart + size});
    offset = payloadStart + size;
  }
  return elements;
}

test('WebM muxer writes EBML header, segment, tracks and cluster blocks', () => {
  const webm = buildWebm({width: 64, height: 32, codec: 'vp9', samples: SAMPLES});
  const top = readEbml(webm, 0, webm.length);
  expect(top.map(element => element.id)).toEqual([0x1a45dfa3, 0x18538067]);
  expect(top[1].end).toBe(webm.length);
  const segment = readEbml(webm, top[1].start, top[1].end);
  expect(segment.map(element => element.id)).toEqual([
    0x114d9b74, 0x1549a966, 0x1654ae6b, 0x1f43b675, 0x1c53bb6b
  ]);
  const cluster = readEbml(webm, segment[3].start, segment[3].end);
  expect(cluster.map(element => element.id)).toEqual([0xe7, 0xa3, 0xa3, 0xa3]);
  const block = cluster[1];
  // track 1, relative timecode 0, keyframe flag, payload.
  expect(Array.from(webm.subarray(block.start, block.end))).toEqual([0x81, 0, 0, 0x80, 1, 2, 3]);
  const second = cluster[2];
  expect(Array.from(webm.subarray(second.start, second.start + 4))).toEqual([0x81, 0, 33, 0]);
  expect(String.fromCharCode(...webm.subarray(top[0].start, top[0].end))).toContain('webm');
  expect(String.fromCharCode(...webm.subarray(segment[2].start, segment[2].end))).toContain(
    'V_VP9'
  );
});

test('WebM splits clusters at the first keyframe after 5 seconds', () => {
  const samples: EncodedVideoSample[] = [0, 3_000_000, 6_000_000].map(
    (timestampMicroseconds, index) => ({
      data: new Uint8Array([index]),
      timestampMicroseconds,
      durationMicroseconds: 1_000_000,
      isKeyFrame: index !== 1
    })
  );
  const webm = buildWebm({width: 2, height: 2, codec: 'vp8', samples});
  const top = readEbml(webm, 0, webm.length);
  const segment = readEbml(webm, top[1].start, top[1].end);
  expect(segment.filter(element => element.id === 0x1f43b675)).toHaveLength(2);
});

test('encoder candidates and feature detection', async () => {
  expect(chooseAvcCodecString(1280, 720)).toBe('avc1.640028');
  expect(chooseAvcCodecString(3840, 2160)).toBe('avc1.640033');
  expect(getEncoderCandidates('mp4', 640, 480).map(candidate => candidate.format)).toEqual(['mp4']);
  expect(getEncoderCandidates('auto', 640, 480).map(candidate => candidate.format)).toEqual([
    'mp4',
    'webm',
    'webm'
  ]);
  expect(await detectWebCodecsEncoder('auto', 640, 480, {})).toBeNull();
  (globalThis as {VideoFrame?: unknown}).VideoFrame = class {};
  try {
    const choice = await detectWebCodecsEncoder('auto', 640, 480, {
      VideoEncoder: {
        isConfigSupported: async (config: Record<string, unknown>) => ({
          supported: String(config['codec']).startsWith('vp09')
        })
      }
    });
    expect(choice?.muxCodec).toBe('vp9');
  } finally {
    delete (globalThis as {VideoFrame?: unknown}).VideoFrame;
  }
});

test('canvas readback helpers: row alignment, swizzle, blank detection', () => {
  expect(getAlignedBytesPerRow(10)).toBe(256);
  expect(getAlignedBytesPerRow(64)).toBe(256);
  expect(getAlignedBytesPerRow(65)).toBe(512);
  const padded = new Uint8Array(256 * 2);
  padded.set([10, 20, 30, 255, 1, 2, 3, 4], 0);
  padded.set([50, 60, 70, 255, 5, 6, 7, 8], 256);
  expect(Array.from(unpackTextureRows(padded, 2, 2, 256, 'bgra8unorm'))).toEqual([
    30, 20, 10, 255, 3, 2, 1, 4, 70, 60, 50, 255, 7, 6, 5, 8
  ]);
  expect(Array.from(unpackTextureRows(padded, 1, 1, 256, 'rgba8unorm'))).toEqual([10, 20, 30, 255]);
  expect(isFrameBlank(new Uint8Array(64))).toBe(true);
  expect(isFrameBlank(new Uint8Array([0, 0, 0, 255]))).toBe(false);
});

test('tile projection offset maps the tile rectangle to full clip space', () => {
  const tile = {x: 600, y: 200, width: 400, height: 300};
  const offset = getTileProjectionOffset(tile, 1000, 800);
  // Identity projection: clip = (x, y, 0, 1). Full NDC of the tile corners must become -1/+1.
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const matrix = applyTileProjectionOffset(identity, offset);
  const transform = (x: number, y: number) => [
    matrix[0] * x + matrix[4] * y + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[13]
  ];
  const toNdc = (px: number, py: number) => [(px / 1000) * 2 - 1, 1 - (py / 800) * 2];
  const topLeft = transform(...(toNdc(600, 200) as [number, number]));
  const bottomRight = transform(...(toNdc(1000, 500) as [number, number]));
  expect(topLeft[0]).toBeCloseTo(-1, 9);
  expect(topLeft[1]).toBeCloseTo(1, 9);
  expect(bottomRight[0]).toBeCloseTo(1, 9);
  expect(bottomRight[1]).toBeCloseTo(-1, 9);
});

test('tiled scales respect the 32,768 px and 128 MP bounds', () => {
  expect(getAllowedTiledScales(1200, 900)).toEqual([2, 4, 8]);
  expect(getAllowedTiledScales(4000, 4000)).toEqual([2]);
  expect(getAllowedTiledScales(20000, 20000)).toEqual([]);
});
