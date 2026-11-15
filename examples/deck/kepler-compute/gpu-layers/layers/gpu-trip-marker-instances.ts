// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type CommandEncoder, type Device, type ShaderLayout} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {GPUVector, GPUVectorFormat, VertexList} from '@luma.gl/graph/gpu-data';
import {getGPUDataBuffer} from './gpu-vector-layer-utils';

type Chunk<Format extends GPUVectorFormat> = GPUVector<Format>['data'][number];

/** Words in one interleaved instance record; the record is read as Deck instance attributes. */
export const GPU_TRIP_MARKER_RECORD_WORDS = 24;
/** Bytes between consecutive instance records. */
export const GPU_TRIP_MARKER_RECORD_STRIDE = GPU_TRIP_MARKER_RECORD_WORDS * 4;
/**
 * Byte offsets of the Deck ScenegraphLayer instance attributes inside one record:
 * `instancePositions` (high then low float32x3), `instanceModelMatrix` (three columns and a
 * translation) and `instanceColors` (unorm8x4).
 */
export const GPU_TRIP_MARKER_RECORD_OFFSETS = {
  positions: 0,
  modelMatrix: 24,
  colors: 72
} as const;

/** Resident inputs of one physical trip chunk. All chunks are borrowed. */
export type GPUTripMarkerInstanceSource = {
  path: Chunk<VertexList<'float32x3'>>;
  timestamps: Chunk<VertexList<'float32'>>;
  selection?: Chunk<'uint32'>;
  /** One color per trip. */
  tripColors?: Chunk<'unorm8x4'>;
  /** One color per path vertex; wins over the per-trip color. */
  vertexColors?: Chunk<'unorm8x4'>;
  /** One flag per path vertex; unselected vertices take the nearest selected vertex's values. */
  vertexSelection?: Chunk<'uint32'>;
  vertexRoll?: Chunk<'float32'>;
  vertexPitch?: Chunk<'float32'>;
  vertexYaw?: Chunk<'float32'>;
};

/** Values that change per frame. */
export type GPUTripMarkerInstanceFrame = {
  currentTime: number;
  sizeScale: number;
  /** Column-major 3x3 applied after the per-trip orientation. */
  baseRotation: readonly number[];
  /** Packed RGBA when no per-trip or per-vertex color exists; white by default. */
  constantColor?: readonly number[];
  invertRoll?: boolean;
  invertPitch?: boolean;
  invertYaw?: boolean;
};

const WORKGROUP_SIZE = 64;

// Record words: 0-2 position, 3-5 position low part, 6-14 model matrix columns, 15-17 translation,
// 18 packed color, 19 visible flag, 20 yaw degrees, 21 roll degrees, 22 pitch degrees.
const RECORD_PRELUDE = /* wgsl */ `
const RECORD_WORDS: u32 = ${GPU_TRIP_MARKER_RECORD_WORDS}u;
@group(0) @binding(0) var<storage, read_write> instanceRecords: array<u32>;
fn readRecordFloat(word: u32) -> f32 { return bitcast<f32>(instanceRecords[word]); }
fn writeRecordFloat(word: u32, value: f32) { instanceRecords[word] = bitcast<u32>(value); }
`;

const SEGMENT_PRELUDE = /* wgsl */ `
${RECORD_PRELUDE}
@group(0) @binding(1) var<storage, read> tripTimes: array<f32>;
@group(0) @binding(2) var<storage, read> tripOffsets: array<u32>;

struct TripSegment {
  visible: bool,
  index: u32,
  advancement: f32,
};

fn readTripTime(vertexIndex: u32) -> f32 {
  return tripTimes[config.timeWordOffset + vertexIndex];
}

// Native _findDatumForFeatureByTime: bisectRight over timestamps, clamped to the last vertex, and
// a marker only while the time lies inside the trip.
fn findTripSegment(first: u32, last: u32) -> TripSegment {
  var segment: TripSegment;
  segment.visible = false;
  segment.index = first;
  segment.advancement = 0.0;
  if (last <= first) { return segment; }
  let currentTime = config.currentTime;
  if (currentTime < readTripTime(first) || currentTime > readTripTime(last - 1u)) { return segment; }
  var low = first;
  var high = last;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (readTripTime(middle) <= currentTime) { low = middle + 1u; } else { high = middle; }
  }
  let index = min(low, last - 1u);
  if (index > first) {
    let previousTime = readTripTime(index - 1u);
    let duration = readTripTime(index) - previousTime;
    segment.advancement = select(1.0, (currentTime - previousTime) / duration, duration > 0.0);
  }
  segment.visible = true;
  segment.index = index;
  return segment;
}

// Native interpolateAngle.
fn interpolateAngle(first: f32, second: f32, advancement: f32) -> f32 {
  var a = first;
  var b = second;
  if (a - b > 180.0) { b += 360.0; } else if (b - a > 180.0) { a += 360.0; }
  return a + (b - a) * advancement;
}
`;

const PLACE_SHADER = /* wgsl */ `
struct Config {
  currentTime: f32,
  instanceCount: u32,
  timeWordOffset: u32,
  pathWordOffset: u32,
  selectionEnabled: u32,
  selectionWordOffset: u32,
  selectionWordStride: u32,
  padding: u32,
};
@group(0) @binding(3) var<storage, read> tripPathValues: array<f32>;
@group(0) @binding(4) var<storage, read> tripSelection: array<u32>;
@group(0) @binding(5) var<uniform> config: Config;
${SEGMENT_PRELUDE}

fn readTripPosition(vertexIndex: u32) -> vec3<f32> {
  let wordIndex = config.pathWordOffset + vertexIndex * 3u;
  return vec3<f32>(tripPathValues[wordIndex], tripPathValues[wordIndex + 1u], tripPathValues[wordIndex + 2u]);
}

// Native getYawFromSpeedVector: heading of the Web Mercator world-space segment, in degrees.
fn getSegmentYawDegrees(previous: vec3<f32>, next: vec3<f32>) -> f32 {
  let previousWorld = vec2<f32>(radians(previous.x), log(tan(0.785398163 + radians(previous.y) * 0.5)));
  let nextWorld = vec2<f32>(radians(next.x), log(tan(0.785398163 + radians(next.y) * 0.5)));
  let delta = nextWorld - previousWorld;
  if (delta.x == 0.0 && delta.y == 0.0) { return 0.0; }
  return degrees(atan2(delta.y, delta.x));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn placeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tripIndex = globalId.x;
  if (tripIndex >= config.instanceCount) { return; }
  let base = tripIndex * RECORD_WORDS;
  for (var word = 0u; word < RECORD_WORDS; word++) { instanceRecords[base + word] = 0u; }
  if (config.selectionEnabled != 0u &&
      tripSelection[config.selectionWordOffset + tripIndex * config.selectionWordStride] == 0u) { return; }
  let first = tripOffsets[tripIndex];
  let last = tripOffsets[tripIndex + 1u];
  let segment = findTripSegment(first, last);
  if (!segment.visible) { return; }
  let coordinate = readTripPosition(segment.index);
  var position = coordinate;
  var yaw = 0.0;
  if (segment.index > first) {
    position = mix(readTripPosition(segment.index - 1u), coordinate, segment.advancement);
  }
  let localIndex = segment.index - first;
  if (localIndex > 1u) {
    let previousYaw = getSegmentYawDegrees(readTripPosition(segment.index - 2u), readTripPosition(segment.index - 1u));
    let nextYaw = getSegmentYawDegrees(readTripPosition(segment.index - 1u), coordinate);
    yaw = interpolateAngle(previousYaw, nextYaw, segment.advancement);
  } else if (localIndex > 0u) {
    yaw = getSegmentYawDegrees(readTripPosition(segment.index - 1u), coordinate);
  }
  writeRecordFloat(base, position.x);
  writeRecordFloat(base + 1u, position.y);
  writeRecordFloat(base + 2u, position.z);
  writeRecordFloat(base + 19u, 1.0);
  writeRecordFloat(base + 20u, yaw);
}`;

const ANCHOR_FUNCTION = /* wgsl */ `
// Unselected vertices repeat the previous selected vertex (leading ones the first selected), which
// is how the filtered paths drop them; per-vertex values follow the same rule.
fn getAnchorVertex(first: u32, last: u32, vertex: u32) -> u32 {
  if (config.selectionEnabled == 0u) { return vertex; }
  var candidate = vertex + 1u;
  while (candidate > first) {
    candidate = candidate - 1u;
    if (vertexSelection[config.selectionWordOffset + candidate] != 0u) { return candidate; }
  }
  for (var forward = vertex; forward < last; forward++) {
    if (vertexSelection[config.selectionWordOffset + forward] != 0u) { return forward; }
  }
  return vertex;
}
`;

const COLOR_SHADER = /* wgsl */ `
struct Config {
  currentTime: f32,
  instanceCount: u32,
  timeWordOffset: u32,
  mode: u32,
  colorWordOffset: u32,
  colorWordStride: u32,
  constantColor: u32,
  selectionEnabled: u32,
  selectionWordOffset: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
};
@group(0) @binding(3) var<storage, read> colorValues: array<u32>;
@group(0) @binding(4) var<storage, read> vertexSelection: array<u32>;
@group(0) @binding(5) var<uniform> config: Config;
${SEGMENT_PRELUDE}
${ANCHOR_FUNCTION}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn colorMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tripIndex = globalId.x;
  if (tripIndex >= config.instanceCount) { return; }
  let base = tripIndex * RECORD_WORDS;
  if (readRecordFloat(base + 19u) == 0.0) { return; }
  var color = config.constantColor;
  if (config.mode == 1u) {
    color = colorValues[config.colorWordOffset + tripIndex * config.colorWordStride];
  } else if (config.mode == 2u) {
    let first = tripOffsets[tripIndex];
    let last = tripOffsets[tripIndex + 1u];
    // Native table-mode trips color each marker by the datum at bisectRight(timestamps, time).
    let segment = findTripSegment(first, last);
    color = colorValues[config.colorWordOffset + getAnchorVertex(first, last, segment.index)];
  }
  instanceRecords[base + 18u] = color;
}`;

const ANGLE_SHADER = /* wgsl */ `
struct Config {
  currentTime: f32,
  instanceCount: u32,
  timeWordOffset: u32,
  channel: u32,
  valueWordOffset: u32,
  sign: f32,
  selectionEnabled: u32,
  selectionWordOffset: u32,
};
@group(0) @binding(3) var<storage, read> vertexValues: array<f32>;
@group(0) @binding(4) var<storage, read> vertexSelection: array<u32>;
@group(0) @binding(5) var<uniform> config: Config;
${SEGMENT_PRELUDE}
${ANCHOR_FUNCTION}

fn readDatum(first: u32, last: u32, vertex: u32) -> f32 {
  let value = vertexValues[config.valueWordOffset + getAnchorVertex(first, last, vertex)];
  // Null source values arrive as NaN; native treats a missing datum as zero.
  return select(value, 0.0, value != value);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn angleMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tripIndex = globalId.x;
  if (tripIndex >= config.instanceCount) { return; }
  let base = tripIndex * RECORD_WORDS;
  if (readRecordFloat(base + 19u) == 0.0) { return; }
  let first = tripOffsets[tripIndex];
  let last = tripOffsets[tripIndex + 1u];
  let segment = findTripSegment(first, last);
  let localIndex = segment.index - first;
  let datum = readDatum(first, last, segment.index);
  if (config.channel == 2u) {
    // Native _getYaw with a yaw field: the field is a compass bearing, 90 - bearing is the angle.
    let nextYaw = 90.0 - config.sign * datum;
    var yaw = 0.0;
    if (localIndex > 1u) {
      let previousYaw = 90.0 - config.sign * readDatum(first, last, segment.index - 1u);
      yaw = interpolateAngle(previousYaw, nextYaw, segment.advancement);
    } else if (localIndex > 0u) {
      yaw = nextYaw;
    }
    writeRecordFloat(base + 20u, yaw);
    return;
  }
  var value = datum;
  if (localIndex > 0u) {
    value = interpolateAngle(readDatum(first, last, segment.index - 1u), datum, segment.advancement);
  }
  writeRecordFloat(base + select(22u, 21u, config.channel == 0u), config.sign * value);
}`;

const MATRIX_SHADER = /* wgsl */ `
struct Config {
  instanceCount: u32,
  scale: f32,
  padding0: u32,
  padding1: u32,
  rotationColumn0: vec4<f32>,
  rotationColumn1: vec4<f32>,
  rotationColumn2: vec4<f32>,
};
${RECORD_PRELUDE}
@group(0) @binding(1) var<uniform> config: Config;

// Native: Matrix4.fromQuaternion(toQuaternion(roll, -pitch, yaw) * adjustAndModelRotation).scale(s).
@compute @workgroup_size(${WORKGROUP_SIZE})
fn matrixMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let tripIndex = globalId.x;
  if (tripIndex >= config.instanceCount) { return; }
  let base = tripIndex * RECORD_WORDS;
  if (readRecordFloat(base + 19u) == 0.0) { return; }
  let yaw = radians(readRecordFloat(base + 20u));
  let roll = radians(readRecordFloat(base + 21u));
  let pitch = -radians(readRecordFloat(base + 22u));
  let cy = cos(yaw);
  let sy = sin(yaw);
  let cp = cos(pitch);
  let sp = sin(pitch);
  let cr = cos(roll);
  let sr = sin(roll);
  let orientation = mat3x3<f32>(
    vec3<f32>(cy * cp, sy * cp, -sp),
    vec3<f32>(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
    vec3<f32>(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)
  );
  let baseRotation = mat3x3<f32>(
    config.rotationColumn0.xyz,
    config.rotationColumn1.xyz,
    config.rotationColumn2.xyz
  );
  let matrix = orientation * baseRotation * config.scale;
  for (var column = 0u; column < 3u; column++) {
    writeRecordFloat(base + 6u + column * 3u, matrix[column].x);
    writeRecordFloat(base + 7u + column * 3u, matrix[column].y);
    writeRecordFloat(base + 8u + column * 3u, matrix[column].z);
  }
}`;

type BindingType = 'storage' | 'read-only-storage' | 'uniform';
function makeShaderLayout(types: BindingType[], names: string[]): ShaderLayout {
  return {
    bindings: types.map(
      (type, location) =>
        ({name: names[location], type, group: 0, location}) as ShaderLayout['bindings'][number]
    ),
    attributes: []
  };
}

type Pass = {computation: Computation; configuration: Buffer};

/**
 * Per-chunk GPU producer of Deck ScenegraphLayer instance attributes for Trips markers. Compute
 * passes bisect each trip's resident timestamps at the current time and write the interpolated
 * position, the native speed-vector or datum-field orientation, the model matrix and the marker
 * color into one interleaved instance buffer, so no CPU interpolation happens per frame.
 */
export class GPUTripMarkerInstances {
  /** Interleaved instance records, `GPU_TRIP_MARKER_RECORD_STRIDE` bytes each. */
  readonly records: Buffer;
  readonly instanceCount: number;
  private readonly buffers: Buffer[] = [];
  private readonly computations: Computation[] = [];
  private readonly place: Pass;
  private readonly color: Pass;
  private readonly angles: (Pass | null)[];
  private readonly matrix: Pass;
  private readonly colorMode: 0 | 1 | 2;

  constructor(
    private readonly device: Device,
    private readonly id: string,
    private readonly source: GPUTripMarkerInstanceSource
  ) {
    const {path, timestamps, selection, tripColors, vertexColors, vertexSelection} = source;
    assert(path.valueOffsets && path.byteStride === 12);
    assert(timestamps.valueOffsets && timestamps.byteStride === 4);
    assert(path.byteOffset % 4 === 0 && timestamps.byteOffset % 4 === 0);
    this.instanceCount = path.length;
    this.records = this.makeBuffer(
      'records',
      this.instanceCount * GPU_TRIP_MARKER_RECORD_STRIDE,
      Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    );
    const offsets = this.device.createBuffer({
      id: `${id}-offsets`,
      data: new Uint32Array(path.valueOffsets),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.buffers.push(offsets);
    const fallback = this.device.createBuffer({
      id: `${id}-fallback`,
      data: new Uint32Array([1]),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.buffers.push(fallback);
    const vertexAddress = (chunk: Chunk<'float32'> | Chunk<'unorm8x4'> | Chunk<'uint32'>) => {
      // Per-vertex channels share the path's flattened vertex order, tightly packed.
      assert(chunk.byteStride === 4 && chunk.byteOffset % 4 === 0);
      assert(chunk.length >= path.valueLength);
      return getGPUDataBuffer(chunk);
    };
    const vertexSelectionBuffer = vertexSelection ? vertexAddress(vertexSelection) : fallback;
    this.colorMode = vertexColors ? 2 : tripColors ? 1 : 0;
    if (tripColors) assert(tripColors.byteOffset % 4 === 0 && tripColors.byteStride % 4 === 0);
    const base = {
      record: this.records,
      tripTimes: getGPUDataBuffer(timestamps),
      tripOffsets: offsets
    };
    const names = ['instanceRecords', 'tripTimes', 'tripOffsets'];
    const make = (
      name: string,
      shaderSource: string,
      types: BindingType[],
      bindingNames: string[],
      bindings: Record<string, Buffer>,
      configurationBytes: number
    ): Pass => {
      const configuration = this.makeBuffer(
        `${name}-configuration`,
        configurationBytes,
        Buffer.UNIFORM | Buffer.COPY_DST
      );
      const computation = new Computation(this.device, {
        id: `${id}-${name}`,
        source: shaderSource,
        shaderLayout: makeShaderLayout(types, bindingNames),
        bindings: {...bindings, config: configuration}
      });
      this.computations.push(computation);
      return {computation, configuration};
    };
    const storage: BindingType[] = ['storage', 'read-only-storage', 'read-only-storage'];
    const readOnly: BindingType = 'read-only-storage';
    const recordBindings = {
      instanceRecords: base.record,
      tripTimes: base.tripTimes,
      tripOffsets: base.tripOffsets
    };
    this.place = make(
      'place',
      PLACE_SHADER,
      [...storage, readOnly, readOnly, 'uniform'],
      [...names, 'tripPathValues', 'tripSelection', 'config'],
      {
        ...recordBindings,
        tripPathValues: getGPUDataBuffer(path),
        tripSelection: selection ? getGPUDataBuffer(selection) : fallback
      },
      32
    );
    this.color = make(
      'color',
      COLOR_SHADER,
      [...storage, readOnly, readOnly, 'uniform'],
      [...names, 'colorValues', 'vertexSelection', 'config'],
      {
        ...recordBindings,
        colorValues: vertexColors
          ? vertexAddress(vertexColors)
          : tripColors
            ? getGPUDataBuffer(tripColors)
            : fallback,
        vertexSelection: vertexSelectionBuffer
      },
      48
    );
    this.angles = [source.vertexRoll, source.vertexPitch, source.vertexYaw].map((channel, index) =>
      channel
        ? make(
            `angle-${index}`,
            ANGLE_SHADER,
            [...storage, readOnly, readOnly, 'uniform'],
            [...names, 'vertexValues', 'vertexSelection', 'config'],
            {
              ...recordBindings,
              vertexValues: vertexAddress(channel),
              vertexSelection: vertexSelectionBuffer
            },
            32
          )
        : null
    );
    this.matrix = make(
      'matrix',
      MATRIX_SHADER,
      ['storage', 'uniform'],
      ['instanceRecords', 'config'],
      {instanceRecords: base.record},
      64
    );
  }

  /**
   * Recomputes every instance for the frame. With a `commandEncoder` the passes are recorded into
   * it, so they run after the application's selection and style compute already encoded there;
   * without one they are submitted immediately on a private encoder, ahead of the render pass.
   */
  encode(frame: GPUTripMarkerInstanceFrame, commandEncoder?: CommandEncoder): void {
    if (this.instanceCount === 0) return;
    const {source} = this;
    const writeConfiguration = (
      pass: Pass,
      write: (floats: Float32Array, words: Uint32Array) => void
    ) => {
      const bytes = new ArrayBuffer(pass.configuration.byteLength);
      write(new Float32Array(bytes), new Uint32Array(bytes));
      pass.configuration.write(new Uint8Array(bytes));
    };
    const timeWordOffset = source.timestamps.byteOffset / 4;
    const vertexSelectionWordOffset = source.vertexSelection
      ? source.vertexSelection.byteOffset / 4
      : 0;
    writeConfiguration(this.place, (floats, words) => {
      floats[0] = frame.currentTime;
      words[1] = this.instanceCount;
      words[2] = timeWordOffset;
      words[3] = source.path.byteOffset / 4;
      words[4] = source.selection ? 1 : 0;
      words[5] = source.selection ? source.selection.byteOffset / 4 : 0;
      words[6] = source.selection ? source.selection.byteStride / 4 : 1;
    });
    const constantColor = frame.constantColor ?? [255, 255, 255, 255];
    writeConfiguration(this.color, (floats, words) => {
      floats[0] = frame.currentTime;
      words[1] = this.instanceCount;
      words[2] = timeWordOffset;
      words[3] = this.colorMode;
      const colors = source.vertexColors ?? source.tripColors;
      words[4] = colors ? colors.byteOffset / 4 : 0;
      words[5] = source.tripColors && !source.vertexColors ? source.tripColors.byteStride / 4 : 1;
      words[6] =
        (((constantColor[3] ?? 255) << 24) |
          ((constantColor[2] ?? 255) << 16) |
          ((constantColor[1] ?? 255) << 8) |
          (constantColor[0] ?? 255)) >>>
        0;
      words[7] = source.vertexSelection ? 1 : 0;
      words[8] = vertexSelectionWordOffset;
    });
    const channels = [source.vertexRoll, source.vertexPitch, source.vertexYaw];
    const inverted = [frame.invertRoll, frame.invertPitch, frame.invertYaw];
    this.angles.forEach((pass, channel) => {
      if (!pass) return;
      writeConfiguration(pass, (floats, words) => {
        floats[0] = frame.currentTime;
        words[1] = this.instanceCount;
        words[2] = timeWordOffset;
        words[3] = channel;
        words[4] = channels[channel]!.byteOffset / 4;
        floats[5] = inverted[channel] ? -1 : 1;
        words[6] = source.vertexSelection ? 1 : 0;
        words[7] = vertexSelectionWordOffset;
      });
    });
    writeConfiguration(this.matrix, (floats, words) => {
      words[0] = this.instanceCount;
      floats[1] = frame.sizeScale;
      for (let column = 0; column < 3; column++) {
        floats.set(frame.baseRotation.slice(column * 3, column * 3 + 3), 4 + column * 4);
      }
    });
    const workgroups = Math.ceil(this.instanceCount / WORKGROUP_SIZE);
    // One workgroup dimension holds 65535 groups, about four million trips.
    assert(workgroups <= 65535);
    // Without a caller encoder the render pass of the frame is still open on the device's default
    // encoder, so the work uses its own encoder, which queue order places before the frame.
    const encoder =
      commandEncoder ?? this.device.createCommandEncoder({id: `${this.id}-instances`});
    const computePass = encoder.beginComputePass({});
    for (const pass of [this.place, this.color, ...this.angles, this.matrix]) {
      pass?.computation.dispatch(computePass, workgroups);
    }
    computePass.end();
    if (!commandEncoder) this.device.submit(encoder.finish());
  }

  destroy(): void {
    for (const computation of this.computations) computation.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.computations.length = 0;
    this.buffers.length = 0;
  }

  private makeBuffer(name: string, byteLength: number, usage: number): Buffer {
    const buffer = this.device.createBuffer({
      id: `${this.id}-${name}`,
      byteLength: Math.max(16, byteLength),
      usage
    });
    this.buffers.push(buffer);
    return buffer;
  }
}
