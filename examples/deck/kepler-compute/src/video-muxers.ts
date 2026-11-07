// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Minimal in-memory muxers for WebCodecs output. Both are pure byte writers (no DOM) so their layout
 * is unit tested in node. MP4 writes a progressive file (ftyp, moov, mdat) for H.264 samples in AVCC
 * (length-prefixed) form with the encoder's avcC record. Samples are given in decode order with their
 * presentation timestamps; when those differ (B-frames) the muxer writes a version 1 ctts box with
 * signed composition offsets and an edit list that starts playback at the first presented frame.
 * WebM writes EBML with one VP8/VP9 video track, SeekHead, clusters that start at keyframes, and a
 * Cues element with one entry per keyframe cluster so players can seek.
 */

export type EncodedVideoSample = {
  data: Uint8Array;
  /** Presentation time (pts) of the sample in microseconds. Samples are listed in decode order. */
  timestampMicroseconds: number;
  durationMicroseconds: number;
  isKeyFrame: boolean;
};

function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
  let length = 0;
  for (const part of parts) length += part.length;
  const output = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function writeUint(value: number, byteLength: number): Uint8Array {
  const bytes = new Uint8Array(byteLength);
  let remaining = value;
  for (let index = byteLength - 1; index >= 0; index--) {
    bytes[index] = remaining % 256;
    remaining = Math.floor(remaining / 256);
  }
  return bytes;
}

// ---------------------------------------------------------------- MP4 (ISO-BMFF)

const MP4_TIMESCALE = 1_000_000;

function makeMp4Box(type: string, ...payloads: Uint8Array[]): Uint8Array {
  const body = concatBytes(payloads);
  const header = new Uint8Array(8);
  header.set(writeUint(body.length + 8, 4), 0);
  for (let index = 0; index < 4; index++) header[4 + index] = type.charCodeAt(index);
  return concatBytes([header, body]);
}

function makeFullBox(type: string, version: number, flags: number, ...payloads: Uint8Array[]) {
  return makeMp4Box(type, writeUint(version * 0x1000000 + flags, 4), ...payloads);
}

function repeatWords(...words: number[]): Uint8Array {
  return concatBytes(words.map(word => writeUint(word, 4)));
}

const IDENTITY_MATRIX = repeatWords(0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000);

export type Mp4SampleTiming = {
  /** stts deltas in decode order (monotonic decode timestamps starting at 0). */
  decodeDeltas: number[];
  /** Signed composition offsets (pts - dts) in decode order. */
  compositionOffsets: number[];
  /** Media time of the first presented frame, used by the edit list. */
  mediaTime: number;
  /** Duration of the presented timeline (last pts end minus first pts). */
  presentationDuration: number;
  hasReordering: boolean;
};

/** Derive monotonic decode timestamps and signed composition offsets from decode-order samples. */
export function computeMp4SampleTiming(samples: readonly EncodedVideoSample[]): Mp4SampleTiming {
  const decodeDeltas = samples.map(sample => Math.max(1, Math.round(sample.durationMicroseconds)));
  const compositionOffsets: number[] = [];
  let decodeTime = 0;
  let firstPresentation = Infinity;
  let lastPresentationEnd = 0;
  samples.forEach((sample, index) => {
    const presentation = Math.round(sample.timestampMicroseconds);
    compositionOffsets.push(presentation - decodeTime);
    firstPresentation = Math.min(firstPresentation, presentation);
    lastPresentationEnd = Math.max(lastPresentationEnd, presentation + decodeDeltas[index]);
    decodeTime += decodeDeltas[index];
  });
  const hasReordering = compositionOffsets.some(offset => offset !== compositionOffsets[0]);
  if (!samples.length) firstPresentation = 0;
  return {
    decodeDeltas,
    compositionOffsets,
    mediaTime: firstPresentation,
    presentationDuration: lastPresentationEnd - firstPresentation,
    hasReordering
  };
}

function writeInt32(value: number): Uint8Array {
  return writeUint(value >>> 0, 4);
}

export function buildMp4(options: {
  width: number;
  height: number;
  /** avcC decoder configuration record from VideoEncoder's decoderConfig.description. */
  avcConfiguration: Uint8Array;
  samples: readonly EncodedVideoSample[];
}): Uint8Array {
  const {width, height, avcConfiguration, samples} = options;
  const timing = computeMp4SampleTiming(samples);
  const durationUnits = timing.presentationDuration;
  const deltas = timing.decodeDeltas;
  const needsComposition = timing.hasReordering || timing.compositionOffsets[0] !== 0;
  const offsetRuns: Array<[number, number]> = [];
  for (const offset of timing.compositionOffsets) {
    const last = offsetRuns[offsetRuns.length - 1];
    if (last && last[1] === offset) last[0]++;
    else offsetRuns.push([1, offset]);
  }
  const runs: Array<[number, number]> = [];
  for (const delta of deltas) {
    const last = runs[runs.length - 1];
    if (last && last[1] === delta) last[0]++;
    else runs.push([1, delta]);
  }
  const mediaDuration = deltas.reduce((sum, delta) => sum + delta, 0);
  const syncIndices = samples.flatMap((sample, index) => (sample.isKeyFrame ? [index + 1] : []));
  const ftyp = makeMp4Box(
    'ftyp',
    new TextEncoder().encode('isom'),
    writeUint(512, 4),
    new TextEncoder().encode('isomiso2avc1mp41')
  );
  const buildMoov = (chunkOffset: number) => {
    const sampleEntry = makeMp4Box(
      'avc1',
      new Uint8Array(6),
      writeUint(1, 2),
      new Uint8Array(16),
      writeUint(width, 2),
      writeUint(height, 2),
      repeatWords(0x00480000, 0x00480000, 0),
      writeUint(1, 2),
      new Uint8Array(32),
      writeUint(0x18, 2),
      writeUint(0xffff, 2),
      makeMp4Box('avcC', avcConfiguration)
    );
    const sampleTable = makeMp4Box(
      'stbl',
      makeFullBox('stsd', 0, 0, writeUint(1, 4), sampleEntry),
      makeFullBox(
        'stts',
        0,
        0,
        writeUint(runs.length, 4),
        ...runs.map(([count, delta]) => repeatWords(count, delta))
      ),
      ...(needsComposition
        ? [
            makeFullBox(
              'ctts',
              1,
              0,
              writeUint(offsetRuns.length, 4),
              ...offsetRuns.map(([count, offset]) =>
                concatBytes([writeUint(count, 4), writeInt32(offset)])
              )
            )
          ]
        : []),
      makeFullBox(
        'stss',
        0,
        0,
        writeUint(syncIndices.length, 4),
        ...syncIndices.map(index => writeUint(index, 4))
      ),
      makeFullBox('stsc', 0, 0, repeatWords(1, 1, samples.length, 1)),
      makeFullBox(
        'stsz',
        0,
        0,
        repeatWords(0, samples.length),
        ...samples.map(sample => writeUint(sample.data.length, 4))
      ),
      makeFullBox('stco', 0, 0, repeatWords(1, chunkOffset))
    );
    const mediaInformation = makeMp4Box(
      'minf',
      makeFullBox('vmhd', 0, 1, new Uint8Array(8)),
      makeMp4Box('dinf', makeFullBox('dref', 0, 0, writeUint(1, 4), makeFullBox('url ', 0, 1))),
      sampleTable
    );
    const media = makeMp4Box(
      'mdia',
      makeFullBox(
        'mdhd',
        0,
        0,
        repeatWords(0, 0, MP4_TIMESCALE, mediaDuration),
        writeUint(0x55c4, 2),
        new Uint8Array(2)
      ),
      makeFullBox(
        'hdlr',
        0,
        0,
        new Uint8Array(4),
        new TextEncoder().encode('vide'),
        new Uint8Array(12),
        new TextEncoder().encode('VideoHandler\0')
      ),
      mediaInformation
    );
    const track = makeMp4Box(
      'trak',
      makeFullBox(
        'tkhd',
        0,
        3,
        repeatWords(0, 0, 1, 0, durationUnits),
        new Uint8Array(8),
        new Uint8Array(8),
        IDENTITY_MATRIX,
        repeatWords(width * 0x10000, height * 0x10000)
      ),
      ...(needsComposition
        ? [
            makeMp4Box(
              'edts',
              makeFullBox(
                'elst',
                0,
                0,
                writeUint(1, 4),
                repeatWords(durationUnits, timing.mediaTime, 0x00010000)
              )
            )
          ]
        : []),
      media
    );
    return makeMp4Box(
      'moov',
      makeFullBox(
        'mvhd',
        0,
        0,
        repeatWords(0, 0, MP4_TIMESCALE, durationUnits, 0x00010000),
        writeUint(0x0100, 2),
        new Uint8Array(10),
        IDENTITY_MATRIX,
        new Uint8Array(24),
        writeUint(2, 4)
      ),
      track
    );
  };
  const moovLength = buildMoov(0).length;
  const dataOffset = ftyp.length + moovLength + 8;
  const mediaData = concatBytes(samples.map(sample => sample.data));
  return concatBytes([ftyp, buildMoov(dataOffset), makeMp4Box('mdat', mediaData)]);
}

// ---------------------------------------------------------------- WebM (EBML)

function encodeEbmlSize(size: number): Uint8Array {
  const bytes = writeUint(size, 8);
  bytes[0] = 0x01;
  return bytes;
}

function encodeEbmlId(id: number): Uint8Array {
  const length = id > 0xffffff ? 4 : id > 0xffff ? 3 : id > 0xff ? 2 : 1;
  return writeUint(id, length);
}

function makeEbmlElement(id: number, ...payloads: Uint8Array[]): Uint8Array {
  const body = concatBytes(payloads);
  return concatBytes([encodeEbmlId(id), encodeEbmlSize(body.length), body]);
}

function makeEbmlUint(id: number, value: number): Uint8Array {
  let length = 1;
  while (value >= 2 ** (8 * length)) length++;
  return makeEbmlElement(id, writeUint(value, length));
}

function makeEbmlString(id: number, value: string): Uint8Array {
  return makeEbmlElement(id, new TextEncoder().encode(value));
}

function makeEbmlFloat64(id: number, value: number): Uint8Array {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value);
  return makeEbmlElement(id, bytes);
}

const WEBM_CLUSTER_MILLISECONDS = 5000;
const WEBM_CLUSTER_LIMIT_MILLISECONDS = 30000;

export function buildWebm(options: {
  width: number;
  height: number;
  codec: 'vp8' | 'vp9';
  samples: readonly EncodedVideoSample[];
}): Uint8Array {
  const {width, height, codec, samples} = options;
  const timestamps = samples.map(sample => Math.round(sample.timestampMicroseconds / 1000));
  const last = samples[samples.length - 1];
  const durationMilliseconds = last
    ? (last.timestampMicroseconds + last.durationMicroseconds) / 1000
    : 0;
  const header = makeEbmlElement(
    0x1a45dfa3,
    makeEbmlUint(0x4286, 1),
    makeEbmlUint(0x42f7, 1),
    makeEbmlUint(0x42f2, 4),
    makeEbmlUint(0x42f3, 8),
    makeEbmlString(0x4282, 'webm'),
    makeEbmlUint(0x4287, 4),
    makeEbmlUint(0x4285, 2)
  );
  const info = makeEbmlElement(
    0x1549a966,
    makeEbmlUint(0x2ad7b1, 1_000_000),
    makeEbmlString(0x4d80, 'luma.gl kepler-compute'),
    makeEbmlString(0x5741, 'luma.gl kepler-compute'),
    makeEbmlFloat64(0x4489, durationMilliseconds)
  );
  const tracks = makeEbmlElement(
    0x1654ae6b,
    makeEbmlElement(
      0xae,
      makeEbmlUint(0xd7, 1),
      makeEbmlUint(0x73c5, 1),
      makeEbmlUint(0x83, 1),
      makeEbmlString(0x86, codec === 'vp9' ? 'V_VP9' : 'V_VP8'),
      makeEbmlElement(0xe0, makeEbmlUint(0xb0, width), makeEbmlUint(0xba, height))
    )
  );
  const clusters: Uint8Array[] = [];
  const clusterCues: Array<{time: number; clusterIndex: number} | null> = [];
  let clusterStart = 0;
  let clusterStartsWithKey = false;
  let blocks: Uint8Array[] = [];
  const flush = () => {
    if (blocks.length) {
      clusterCues.push(
        clusterStartsWithKey ? {time: clusterStart, clusterIndex: clusters.length} : null
      );
      clusters.push(makeEbmlElement(0x1f43b675, makeEbmlUint(0xe7, clusterStart), ...blocks));
    }
    blocks = [];
  };
  samples.forEach((sample, index) => {
    const timestamp = timestamps[index];
    const elapsed = timestamp - clusterStart;
    // Prefer starting clusters on keyframes so every cue lands on a decodable frame; the hard limit
    // keeps the signed 16-bit block timestamp in range when keyframes are sparse.
    if (
      !blocks.length ||
      (elapsed >= WEBM_CLUSTER_MILLISECONDS && sample.isKeyFrame) ||
      elapsed >= WEBM_CLUSTER_LIMIT_MILLISECONDS
    ) {
      flush();
      clusterStart = timestamp;
      clusterStartsWithKey = sample.isKeyFrame;
    }
    const prefix = new Uint8Array(4);
    prefix[0] = 0x81;
    prefix.set(writeUint((timestamp - clusterStart) & 0xffff, 2), 1);
    prefix[3] = sample.isKeyFrame ? 0x80 : 0;
    blocks.push(makeEbmlElement(0xa3, prefix, sample.data));
  });
  flush();
  const makeSeekHead = (infoPosition: number, tracksPosition: number, cuesPosition: number) =>
    makeEbmlElement(
      0x114d9b74,
      ...[
        [0x1549a966, infoPosition],
        [0x1654ae6b, tracksPosition],
        [0x1c53bb6b, cuesPosition]
      ].map(([id, position]) =>
        makeEbmlElement(
          0x4dbb,
          makeEbmlElement(0x53ab, encodeEbmlId(id)),
          makeEbmlElement(0x53ac, writeUint(position, 4))
        )
      )
    );
  // Positions are relative to the first byte after the Segment header; the SeekHead has a fixed size.
  const seekHeadLength = makeSeekHead(0, 0, 0).length;
  const infoPosition = seekHeadLength;
  const tracksPosition = infoPosition + info.length;
  const firstClusterPosition = tracksPosition + tracks.length;
  const clusterPositions: number[] = [];
  let runningPosition = firstClusterPosition;
  for (const cluster of clusters) {
    clusterPositions.push(runningPosition);
    runningPosition += cluster.length;
  }
  const cuePoints = clusterCues.flatMap(cue =>
    cue
      ? [
          makeEbmlElement(
            0xbb,
            makeEbmlUint(0xb3, cue.time),
            makeEbmlElement(
              0xb7,
              makeEbmlUint(0xf7, 1),
              makeEbmlUint(0xf1, clusterPositions[cue.clusterIndex])
            )
          )
        ]
      : []
  );
  const cues = makeEbmlElement(0x1c53bb6b, ...cuePoints);
  const seekHead = makeSeekHead(infoPosition, tracksPosition, runningPosition);
  return concatBytes([
    header,
    makeEbmlElement(0x18538067, seekHead, info, tracks, ...clusters, cues)
  ]);
}
