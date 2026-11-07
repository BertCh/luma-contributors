// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type ComputePass, type Device, type ShaderLayout} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {_GlobeViewport} from '@deck.gl/core';
import {WebMercatorViewport, lngLatToWorld} from '@math.gl/web-mercator';
import type {LabelCollisionViewport} from './label-collision-runtime';

/**
 * GPU label collision (conservative cell-winner).
 *
 * Each row projects its anchor with the Web Mercator viewport matrix and builds the same label
 * rectangle as `computeLabelVisibilityWords`. Every screen cell the padded rectangle covers keeps
 * the maximum 32-bit key written with `atomicMax`: `priority (8 bits) << 24 | (0xFFFFFF - row)`.
 * A second pass accepts a row only when it owns the maximum in every covered cell and sets its bit
 * in the visibility mask (`atomicOr`, 1 bit per row, same shape as `packAcceptedLabelRows`).
 * Two overlapping rectangles share at least one cell (cell indices are clamped to the grid, which
 * keeps this true off-grid), so at most one of them can win: accepted labels never overlap.
 * Accepted rectangles block a 4x4 sub-cell mask per cell, so later rounds only drop rows whose
 * rectangle really overlaps (to one sub-cell) an accepted one; with enough rounds the result
 * approaches the greedy CPU placement. A row that loses in one cell waits for the next round.
 *
 * Convergence: `updateUntilStable` repeats claim/accept rounds (an accepted-count atomic is read
 * back between batches of rounds) until a batch accepts nothing new, so the result converges to the
 * priority-ordered greedy placement at cell granularity: the highest remaining key always wins all
 * of its unblocked cells. Globe viewports project with Deck's globe common space and hide the far
 * hemisphere (`dot(position, camera) > radius^2`).
 *
 * Limits: row index must stay below 0xFFFFFF; priority is quantized to 256 levels (ties break by
 * lower row index); rectangles that cover more than `MAXIMUM_CELLS_PER_LABEL` cells are hidden.
 */
export const GPU_LABEL_COLLISION_LIMITS = Object.freeze({
  maximumRows: 0xfffffe,
  maximumCellsPerLabel: 512,
  maximumGridCells: 1 << 20
});

export type GpuLabelCollisionOptions = {
  device: Device;
  /** Resident anchor longitude/latitude (chunks must align with `rowBounds`). */
  positions: GPUVector<'float32x2' | 'float32x3'>;
  /** Per-chunk [widestLineAdvance, paragraphHeight, 0, 0] in atlas pixels (label batch rowBounds). */
  rowBounds: readonly GPUData<'float32x4'>[];
  /** Optional per-row validity; zero hides the row. */
  validity?: GPUVector<'uint32'>;
  /** Optional per-row priority; larger wins. Quantized to 8 bits between `priorityRange`. */
  priorities?: GPUVector<'float32'>;
  priorityRange?: readonly [number, number];
  /** Screen cell size in pixels, split into 4x4 exact sub-cells (default 8). */
  cellSize?: number;
  /** Extra pixels kept clear between labels (default 2, as the CPU path). */
  padding?: number;
  /** Claim/accept rounds of `update`; later rounds recover rows that lost to a rejected winner (default 6). */
  rounds?: number;
  /** Upper bound of rounds for `updateUntilStable` (default 64). */
  maximumRounds?: number;
};

export type GpuLabelCollisionUpdate = {
  viewport: LabelCollisionViewport;
  size: number;
  fontSize: number;
  anchor: 'start' | 'middle' | 'end';
  alignment: 'top' | 'center' | 'bottom';
  radius: number;
  radiusPixelScale: number;
  radiusPadding: number;
  verticalPadding: number;
};

export type GpuLabelCollision = {
  readonly rowCount: number;
  readonly wordCount: number;
  /** Packed uint32 visibility words (1 bit per row), valid after `update` has been submitted. */
  readonly mask: Buffer;
  update(update: GpuLabelCollisionUpdate): void;
  /**
   * Repeats claim/accept rounds in batches until a batch accepts no new label (or the round cap),
   * approaching greedy placement. Resolves with the executed rounds and accepted label count.
   */
  updateUntilStable(update: GpuLabelCollisionUpdate): Promise<{rounds: number; accepted: number}>;
  readVisibilityWords(): Promise<Uint32Array>;
  destroy(): void;
};

const SCREEN_MARGIN = 256;
const PARAMETER_WORDS = 52;
const WORKGROUP_SIZE = 64;

const HEADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> positionValues : array<f32>;
@group(0) @binding(1) var<storage, read> boundsValues : array<f32>;
@group(0) @binding(2) var<storage, read> validityValues : array<u32>;
@group(0) @binding(3) var<storage, read> priorityValues : array<f32>;
@group(0) @binding(4) var<storage, read> params : array<u32>;
@group(0) @binding(5) var<storage, read_write> cells : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> maskWords : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> blockedCells : array<atomic<u32>>;
`;

// params layout (u32 words, floats via bitcast): 0-15 matrix columns with the reference point
// folded in, 16 centerLongitude, 17 centerMercatorY, 18 offsetX, 19 offsetY, 20 anchorShift,
// 21 baselineShift, 22 scale, 23 halfPadding, 24 originX/Y (-margin), 25 cellSize, 26 viewportWidth,
// 27 viewportHeight, 28 priorityMinimum, 29 priorityScale, 32 columns, 33 rows, 34 cellCount,
// 35 wordCount + 1 (the final mask word counts accepted labels), 36 hasValidity, 37 hasPriority, 38 rowBase, 39 chunkRowCount, 40 positionOffset,
// 41 positionStride, 42 boundsOffset, 43 validityOffset, 44 priorityOffset, 45 globe flag (matrix
// columns 0-15 then project the 256-radius globe position), 46-48 globe camera position.
const RECT_SOURCE = /* wgsl */ `
fn param(index : u32) -> f32 { return bitcast<f32>(params[index]); }

struct LabelRect {
  valid : bool,
  key : u32,
  firstColumn : u32,
  lastColumn : u32,
  firstRow : u32,
  lastRow : u32,
  left : f32,
  right : f32,
  top : f32,
  bottom : f32,
}

fn mercatorY(latitude : f32) -> f32 {
  let phi = radians(latitude);
  return 512.0 * (3.14159265358979 + log(tan(0.78539816339745 + phi * 0.5))) / 6.28318530717959;
}

fn getLabelRect(localRow : u32) -> LabelRect {
  var rect : LabelRect;
  rect.valid = false;
  if (localRow >= params[39]) { return rect; }
  if (params[36] != 0u && validityValues[params[43] + localRow] == 0u) { return rect; }
  let positionIndex = params[40] + localRow * params[41];
  let longitude = positionValues[positionIndex];
  let latitude = positionValues[positionIndex + 1u];
  let isGlobe = params[45] != 0u;
  if (!isGlobe && !(latitude > -89.5 && latitude < 89.5 && longitude > -540.0 && longitude < 540.0)) { return rect; }
  let boundsIndex = params[42] + localRow * 4u;
  let width = boundsValues[boundsIndex] * param(22u);
  let height = boundsValues[boundsIndex + 1u] * param(22u);
  if (!(width > 0.0 && height > 0.0)) { return rect; }
  var projected : vec4<f32>;
  if (isGlobe) {
    let lambda = radians(longitude);
    let phi = radians(latitude);
    let cosPhi = cos(phi);
    let world = vec3<f32>(sin(lambda) * cosPhi, -cos(lambda) * cosPhi, sin(phi)) * 256.0;
    // Points on the far hemisphere are hidden by the globe.
    if (!(dot(world, vec3<f32>(param(46u), param(47u), param(48u))) > 65536.0)) { return rect; }
    projected = vec4<f32>(param(0u), param(1u), param(2u), param(3u)) * world.x
      + vec4<f32>(param(4u), param(5u), param(6u), param(7u)) * world.y
      + vec4<f32>(param(8u), param(9u), param(10u), param(11u)) * world.z
      + vec4<f32>(param(12u), param(13u), param(14u), param(15u));
  } else {
    let deltaX = (longitude - param(16u)) * (512.0 / 360.0);
    let deltaY = mercatorY(latitude) - param(17u);
    projected = vec4<f32>(param(0u), param(1u), param(2u), param(3u)) * deltaX
      + vec4<f32>(param(4u), param(5u), param(6u), param(7u)) * deltaY
      + vec4<f32>(param(12u), param(13u), param(14u), param(15u));
  }
  if (!(projected.w > 0.0)) { return rect; }
  let screenX = projected.x / projected.w;
  let screenY = projected.y / projected.w;
  let margin = -param(24u);
  if (!(screenX >= -margin && screenY >= -margin &&
      screenX <= param(26u) + margin && screenY <= param(27u) + margin)) { return rect; }
  let x = screenX + param(18u) + width * param(20u);
  let y = screenY + param(19u) + height * param(21u);
  let halfPadding = param(23u);
  let cellSize = param(25u);
  let origin = param(24u);
  let columnLimit = f32(params[32] - 1u);
  let rowLimit = f32(params[33] - 1u);
  let firstColumn = clamp(floor((x - halfPadding - origin) / cellSize), 0.0, columnLimit);
  let lastColumn = clamp(floor((x + width + halfPadding - origin) / cellSize), 0.0, columnLimit);
  let firstRow = clamp(floor((y - halfPadding - origin) / cellSize), 0.0, rowLimit);
  let lastRow = clamp(floor((y + height + halfPadding - origin) / cellSize), 0.0, rowLimit);
  if ((lastColumn - firstColumn + 1.0) * (lastRow - firstRow + 1.0) > ${GPU_LABEL_COLLISION_LIMITS.maximumCellsPerLabel}.0) {
    return rect;
  }
  var priority = 0u;
  if (params[37] != 0u) {
    priority = u32(clamp((priorityValues[params[44] + localRow] - param(28u)) * param(29u), 0.0, 255.0));
  }
  rect.valid = true;
  rect.key = (priority << 24u) | (0xFFFFFFu - (params[38] + localRow));
  rect.firstColumn = u32(firstColumn);
  rect.lastColumn = u32(lastColumn);
  rect.firstRow = u32(firstRow);
  rect.lastRow = u32(lastRow);
  rect.left = x - halfPadding;
  rect.right = x + width + halfPadding;
  rect.top = y - halfPadding;
  rect.bottom = y + height + halfPadding;
  return rect;
}

// Each cell is split into 4x4 sub-cells (16 bits). The mask holds the sub-cells a rectangle covers
// (rounded outward), so accepted rectangles block only the pixels they use: placement is exact to
// one sub-cell instead of one cell. Cells on the grid border (where indices are clamped) are full.
fn getCellMask(rect : LabelRect, column : u32, row : u32) -> u32 {
  if (column == 0u || row == 0u || column + 1u == params[32] || row + 1u == params[33]) { return 0xFFFFu; }
  let cellSize = param(25u);
  let origin = param(24u);
  let subSize = cellSize / 4.0;
  let cellLeft = origin + f32(column) * cellSize;
  let cellTop = origin + f32(row) * cellSize;
  let firstX = u32(clamp(floor((rect.left - cellLeft) / subSize), 0.0, 4.0));
  let endX = u32(clamp(ceil((rect.right - cellLeft) / subSize), 0.0, 4.0));
  let firstY = u32(clamp(floor((rect.top - cellTop) / subSize), 0.0, 4.0));
  let endY = u32(clamp(ceil((rect.bottom - cellTop) / subSize), 0.0, 4.0));
  var mask = 0u;
  for (var subY = firstY; subY < endY; subY++) {
    for (var subX = firstX; subX < endX; subX++) {
      mask |= 1u << (subY * 4u + subX);
    }
  }
  return mask;
}
`;

const CLAIM_SOURCE = `${HEADER}${RECT_SOURCE}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let rect = getLabelRect(id.x);
  if (!rect.valid) { return; }
  let globalRow = params[38] + id.x;
  if ((atomicLoad(&maskWords[globalRow >> 5u]) & (1u << (globalRow & 31u))) != 0u) { return; }
  for (var row = rect.firstRow; row <= rect.lastRow; row++) {
    for (var column = rect.firstColumn; column <= rect.lastColumn; column++) {
      if ((atomicLoad(&blockedCells[row * params[32] + column]) & getCellMask(rect, column, row)) != 0u) { return; }
    }
  }
  for (var row = rect.firstRow; row <= rect.lastRow; row++) {
    for (var column = rect.firstColumn; column <= rect.lastColumn; column++) {
      atomicMax(&cells[row * params[32] + column], rect.key);
    }
  }
}
`;

const ACCEPT_SOURCE = `${HEADER}${RECT_SOURCE}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let rect = getLabelRect(id.x);
  if (!rect.valid) { return; }
  for (var row = rect.firstRow; row <= rect.lastRow; row++) {
    for (var column = rect.firstColumn; column <= rect.lastColumn; column++) {
      if (atomicLoad(&cells[row * params[32] + column]) != rect.key) { return; }
    }
  }
  let globalRow = params[38] + id.x;
  atomicOr(&maskWords[globalRow >> 5u], 1u << (globalRow & 31u));
  atomicAdd(&maskWords[params[35] - 1u], 1u);
  for (var row = rect.firstRow; row <= rect.lastRow; row++) {
    for (var column = rect.firstColumn; column <= rect.lastColumn; column++) {
      atomicOr(&blockedCells[row * params[32] + column], getCellMask(rect, column, row));
    }
  }
}
`;

const getClearSource = (full: boolean) => /* wgsl */ `
@group(0) @binding(4) var<storage, read> params : array<u32>;
@group(0) @binding(5) var<storage, read_write> cells : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> maskWords : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> blockedCells : array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < params[34]) {
    atomicStore(&cells[id.x], 0u);
    ${full ? 'atomicStore(&blockedCells[id.x], 0u);' : ''}
  }
  ${
    full
      ? 'if (id.x < params[35]) { atomicStore(&maskWords[id.x], 0u); }'
      : '// The claim-only clear still references every binding so it shares the pipeline layout.\n  _ = &maskWords;\n  _ = &blockedCells;'
  }
}
`;

const BINDING_TYPES = [
  'positionValues',
  'boundsValues',
  'validityValues',
  'priorityValues',
  'params'
] as const;

const WRITABLE_BINDINGS = ['cells', 'maskWords', 'blockedCells'] as const;

function makeShaderLayout(names: readonly string[]): ShaderLayout {
  const bindings: ShaderLayout['bindings'] = names.map((name, location) => ({
    name,
    type: location >= BINDING_TYPES.length ? 'storage' : 'read-only-storage',
    group: 0,
    location
  }));
  return {bindings, attributes: []};
}

/** Creates the reusable GPU collision pipeline. Row order is the chunk order of `positions`. */
export function createGpuLabelCollision(options: GpuLabelCollisionOptions): GpuLabelCollision {
  const {device, positions, rowBounds} = options;
  const rowCount = positions.length;
  // Rows above 0xFFFFFE do not fit the 24-bit inverted row index in the 32-bit key.
  if (rowCount > GPU_LABEL_COLLISION_LIMITS.maximumRows)
    throw new Error('GPU label collision row cap');
  const wordCount = Math.max(1, Math.ceil(rowCount / 32));
  const cellSize = options.cellSize ?? 8;
  const padding = options.padding ?? 2;
  const rounds = options.rounds ?? 6;
  const priorityRange = options.priorityRange ?? [0, 1];
  const created: Array<{destroy(): void}> = [];
  const track = <T extends {destroy(): void}>(resource: T): T => {
    created.push(resource);
    return resource;
  };
  const mask = track(
    device.createBuffer({
      id: 'gpu-label-collision-mask',
      // The trailing word counts accepted labels, which keeps the round kernels within the
      // default limit of eight storage buffers per shader stage.
      byteLength: (wordCount + 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  );
  const emptyBuffer = track(
    device.createBuffer({
      id: 'gpu-label-collision-empty',
      data: new Uint32Array(4),
      usage: Buffer.STORAGE
    })
  );
  let cells: Buffer | undefined;
  let blockedCells: Buffer | undefined;
  let cellCapacity = 0;
  let computations: Computation[] = [];
  let parameterBuffers: Buffer[] = [];
  let boundCells: Buffer | undefined;
  const chunkRowBases: number[] = [];
  let runningRows = 0;
  for (const chunk of positions.data) {
    chunkRowBases.push(runningRows);
    runningRows += chunk.length;
  }
  const parameterWords = new Uint32Array(PARAMETER_WORDS);
  const parameterFloats = new Float32Array(parameterWords.buffer);
  const roundLayout = makeShaderLayout([...BINDING_TYPES, ...WRITABLE_BINDINGS]);
  const clearLayout: ShaderLayout = {
    bindings: [
      {name: 'params', type: 'read-only-storage', group: 0, location: 4},
      {name: 'cells', type: 'storage', group: 0, location: 5},
      {name: 'maskWords', type: 'storage', group: 0, location: 6},
      {name: 'blockedCells', type: 'storage', group: 0, location: 7}
    ],
    attributes: []
  };

  const prepareChunkParameterBuffers = (): void => {
    if (parameterBuffers.length) return;
    parameterBuffers = positions.data.map((_chunk, chunkIndex) =>
      track(
        device.createBuffer({
          id: `gpu-label-collision-params-${chunkIndex}`,
          byteLength: PARAMETER_WORDS * 4,
          usage: Buffer.STORAGE | Buffer.COPY_DST
        })
      )
    );
  };

  const prepareCells = (cellCount: number): void => {
    if (cells && blockedCells && cellCapacity >= cellCount && boundCells === cells) return;
    if (!cells || !blockedCells || cellCapacity < cellCount) {
      for (const old of [cells, blockedCells]) {
        if (!old) continue;
        old.destroy();
        created.splice(created.indexOf(old), 1);
      }
      const makeCellBuffer = (id: string) =>
        track(
          device.createBuffer({
            id,
            byteLength: Math.max(4, cellCount * 4),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        );
      cells = makeCellBuffer('gpu-label-collision-cells');
      blockedCells = makeCellBuffer('gpu-label-collision-blocked');
      cellCapacity = cellCount;
    }
    for (const computation of computations) computation.destroy();
    computations = [];
    boundCells = cells;
    prepareChunkParameterBuffers();
    const cellBindings = {
      params: parameterBuffers[0] ?? emptyBuffer,
      cells,
      maskWords: mask,
      blockedCells
    };
    // Index 0 clears everything, 1 clears the claim cells; then claim and accept kernels per chunk.
    computations.push(
      new Computation(device, {
        id: 'gpu-label-collision-clear',
        source: getClearSource(true),
        shaderLayout: clearLayout,
        bindings: cellBindings
      }),
      new Computation(device, {
        id: 'gpu-label-collision-clear-claims',
        source: getClearSource(false),
        shaderLayout: clearLayout,
        bindings: cellBindings
      })
    );
    positions.data.forEach((positionChunk, chunkIndex) => {
      const common = {
        positionValues: positionChunk.buffer,
        boundsValues: rowBounds[chunkIndex].buffer,
        validityValues: options.validity?.data[chunkIndex]?.buffer ?? emptyBuffer,
        priorityValues: options.priorities?.data[chunkIndex]?.buffer ?? emptyBuffer,
        params: parameterBuffers[chunkIndex],
        cells: cells as Buffer,
        maskWords: mask,
        blockedCells: blockedCells as Buffer
      };
      computations.push(
        new Computation(device, {
          id: `gpu-label-collision-claim-${chunkIndex}`,
          source: CLAIM_SOURCE,
          shaderLayout: roundLayout,
          bindings: common
        }),
        new Computation(device, {
          id: `gpu-label-collision-accept-${chunkIndex}`,
          source: ACCEPT_SOURCE,
          shaderLayout: roundLayout,
          bindings: common
        })
      );
    });
  };

  const writeParameters = (update: GpuLabelCollisionUpdate): number => {
    const {viewport} = update;
    const globe = Boolean(viewport.globe);
    const projector = globe
      ? new _GlobeViewport({
          longitude: viewport.longitude,
          latitude: viewport.latitude,
          zoom: viewport.zoom,
          width: viewport.width,
          height: viewport.height,
          pitch: viewport.pitch ?? 0,
          bearing: viewport.bearing ?? 0
        })
      : new WebMercatorViewport({
          longitude: viewport.longitude,
          latitude: viewport.latitude,
          zoom: viewport.zoom,
          width: viewport.width,
          height: viewport.height,
          pitch: viewport.pitch ?? 0,
          bearing: viewport.bearing ?? 0
        });
    const matrix = projector.pixelProjectionMatrix;
    const [centerX, centerY] = lngLatToWorld([viewport.longitude, viewport.latitude]);
    // Fold the reference point into the translation column so the shader works on small offsets.
    // Globe positions are already small unit-sphere coordinates, so the translation stays as is.
    const translation = [0, 1, 2, 3].map(row =>
      globe
        ? matrix[12 + row]
        : matrix[row] * centerX + matrix[4 + row] * centerY + matrix[12 + row]
    );
    let columns = Math.ceil((viewport.width + 2 * SCREEN_MARGIN) / cellSize);
    let rows = Math.ceil((viewport.height + 2 * SCREEN_MARGIN) / cellSize);
    let effectiveCellSize = cellSize;
    while (columns * rows > GPU_LABEL_COLLISION_LIMITS.maximumGridCells) {
      effectiveCellSize *= 2;
      columns = Math.ceil((viewport.width + 2 * SCREEN_MARGIN) / effectiveCellSize);
      rows = Math.ceil((viewport.height + 2 * SCREEN_MARGIN) / effectiveCellSize);
    }
    const cellCount = columns * rows;
    const anchorShift = update.anchor === 'start' ? 0 : update.anchor === 'end' ? -1 : -0.5;
    const baselineShift =
      update.alignment === 'top' ? 0 : update.alignment === 'bottom' ? -1 : -0.5;
    const directionX = update.anchor === 'start' ? 1 : update.anchor === 'end' ? -1 : 0;
    const directionY = update.alignment === 'top' ? -1 : update.alignment === 'bottom' ? 1 : 0;
    const radiusOffset = update.radius * update.radiusPixelScale + update.radiusPadding;
    const [priorityMinimum, priorityMaximum] = priorityRange;
    prepareCells(cellCount);
    for (let index = 0; index < 4; index++) {
      parameterFloats[index] = matrix[index];
      parameterFloats[4 + index] = matrix[4 + index];
      parameterFloats[8 + index] = matrix[8 + index];
      parameterFloats[12 + index] = translation[index];
    }
    parameterFloats[16] = viewport.longitude;
    parameterFloats[17] =
      (512 * (Math.PI + Math.log(Math.tan(Math.PI / 4 + (viewport.latitude * Math.PI) / 360)))) /
      (2 * Math.PI);
    parameterFloats[18] = directionX * radiusOffset;
    parameterFloats[19] = directionY * (radiusOffset + update.verticalPadding);
    parameterFloats[20] = anchorShift;
    parameterFloats[21] = baselineShift;
    parameterFloats[22] = update.size / update.fontSize;
    parameterFloats[23] = padding / 2;
    parameterFloats[24] = -SCREEN_MARGIN;
    parameterFloats[25] = effectiveCellSize;
    parameterFloats[26] = viewport.width;
    parameterFloats[27] = viewport.height;
    parameterFloats[28] = priorityMinimum;
    parameterFloats[29] = 255 / Math.max(1e-30, priorityMaximum - priorityMinimum);
    parameterWords[32] = columns;
    parameterWords[33] = rows;
    parameterWords[34] = cellCount;
    parameterWords[35] = wordCount + 1;
    parameterWords[36] = options.validity ? 1 : 0;
    parameterWords[37] = options.priorities ? 1 : 0;
    parameterWords[45] = globe ? 1 : 0;
    if (globe) {
      const camera = (projector as unknown as {cameraPosition: number[]}).cameraPosition;
      parameterFloats[46] = camera[0];
      parameterFloats[47] = camera[1];
      parameterFloats[48] = camera[2];
    }
    positions.data.forEach((positionChunk, chunkIndex) => {
      parameterWords[38] = chunkRowBases[chunkIndex];
      parameterWords[39] = positionChunk.length;
      parameterWords[40] = positionChunk.byteOffset / 4;
      parameterWords[41] = positions.format === 'float32x3' ? 3 : 2;
      parameterWords[42] = rowBounds[chunkIndex].byteOffset / 4;
      parameterWords[43] = (options.validity?.data[chunkIndex]?.byteOffset ?? 0) / 4;
      parameterWords[44] = (options.priorities?.data[chunkIndex]?.byteOffset ?? 0) / 4;
      parameterBuffers[chunkIndex].write(parameterWords);
    });
    return Math.ceil(Math.max(cellCount, wordCount) / WORKGROUP_SIZE);
  };

  const runRounds = (clearGroups: number, firstRound: number, count: number): void => {
    const dispatchChunks = (pass: ComputePass, offset: number) =>
      positions.data.forEach((positionChunk, chunkIndex) => {
        if (positionChunk.length)
          computations[offset + chunkIndex * 2].dispatch(
            pass,
            Math.ceil(positionChunk.length / WORKGROUP_SIZE)
          );
      });
    // Each round lets the per-cell winners claim and accept; accepted rectangles then block their
    // cells for later rounds, so rows that lost only to a rejected winner get another chance.
    for (let round = 0; round < count; round++) {
      let pass = device.beginComputePass({});
      computations[firstRound + round === 0 ? 0 : 1].dispatch(pass, clearGroups);
      pass.end();
      pass = device.beginComputePass({});
      dispatchChunks(pass, 2);
      pass.end();
      pass = device.beginComputePass({});
      dispatchChunks(pass, 3);
      pass.end();
    }
  };

  return {
    rowCount,
    wordCount,
    mask,
    update(update) {
      const clearGroups = writeParameters(update);
      runRounds(clearGroups, 0, rounds);
      device.submit();
    },
    async updateUntilStable(update) {
      const clearGroups = writeParameters(update);
      const maximumRounds = options.maximumRounds ?? 64;
      let executed = 0;
      let accepted = 0;
      while (executed < maximumRounds) {
        const count = Math.min(rounds, maximumRounds - executed);
        runRounds(clearGroups, executed, count);
        device.submit();
        executed += count;
        const bytes = await mask.readAsync(wordCount * 4, 4);
        const total = new Uint32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 4)
        )[0];
        // A batch of rounds that accepts nothing new has converged.
        const stable = total === accepted;
        accepted = total;
        if (stable) break;
      }
      return {rounds: executed, accepted};
    },
    async readVisibilityWords() {
      const bytes = await mask.readAsync(0, wordCount * 4);
      return new Uint32Array(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + wordCount * 4)
      );
    },
    destroy() {
      for (const computation of computations) computation.destroy();
      computations = [];
      for (const resource of created) resource.destroy();
      created.length = 0;
    }
  };
}
