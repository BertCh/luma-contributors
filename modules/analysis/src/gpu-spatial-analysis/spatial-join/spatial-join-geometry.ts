// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateGPUFeatureGeometryPort} from '../../analysis-core/contracts/index';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateRequirement,
  validateCapacity,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSpatialJoinBoundsNode, SPATIAL_JOIN_WGSL_HELPERS} from './spatial-join-passes';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';
import type {
  GPUMultipartGeometryPort,
  GPUPackedGeometryPort
} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUSpatialPredicateJoin';

import {
  getPackedFloat32Values,
  getPackedUint32Values
} from '../../analysis-core/kernels/packed-readback';
import type {CPUSpatialJoinGeometry} from './cpu-feature-join';

/** Readback of allocated geometry; byte views are interpreted as float32/uint32 elements. */
export type GPUSpatialJoinGeometryReadback = {
  kind: 'points' | 'lines' | 'polygons';
  positions: Float32Array | Uint8Array | ArrayBuffer;
  featureOffsets?: Uint32Array | Uint8Array | ArrayBuffer;
  lineOffsets?: Uint32Array | Uint8Array | ArrayBuffer;
  polygonOffsets?: Uint32Array | Uint8Array | ArrayBuffer;
  ringOffsets?: Uint32Array | Uint8Array | ArrayBuffer;
  sourceIds?: Uint32Array | Uint8Array | ArrayBuffer;
};

/**
 * Decodes a live feature prefix, following its terminal offsets through parts, polygons, rings
 * and finally positions. Equal offsets inside that prefix retain genuine empty features.
 * Tails beyond the live prefix are unspecified and are omitted. No completeness is inferred.
 */
export function decodeGPUSpatialJoinGeometry(
  geometry: GPUSpatialJoinGeometryReadback,
  featureCount: number | Uint32Array | Uint8Array | ArrayBuffer
): CPUSpatialJoinGeometry {
  const id = 'decodeGPUSpatialJoinGeometry';
  validateRequirement(
    id,
    ['points', 'lines', 'polygons'].includes(geometry.kind),
    `kind must be points, lines or polygons; received ${geometry.kind}`
  );
  const count =
    typeof featureCount === 'number' ? featureCount : getPackedUint32Values(featureCount, 1)[0];
  validateCapacity(id, 'featureCount', count, 0);
  const positions = getPackedFloat32Values(geometry.positions);
  validateRequirement(id, positions.length % 2 === 0, 'positions needs whole float32x2 rows');
  let childCount = count;
  const result: CPUSpatialJoinGeometry = {kind: geometry.kind, positions};
  const readOffsets = (
    name: 'featureOffsets' | 'lineOffsets' | 'polygonOffsets' | 'ringOffsets'
  ) => {
    const input = geometry[name];
    validateRequirement(id, input !== undefined, `${name} is required for ${geometry.kind}`);
    const offsets = getPackedUint32Values(input!, childCount + 1).slice(0, childCount + 1);
    validateRequirement(id, offsets[0] === 0, `${name} must start at zero`);
    for (let row = 1; row < offsets.length; row++) {
      validateRequirement(
        id,
        offsets[row] >= offsets[row - 1],
        `${name} must be nondecreasing at row ${row}`
      );
    }
    childCount = offsets[offsets.length - 1];
    result[name] = offsets;
  };
  if (geometry.kind === 'polygons' || geometry.featureOffsets) readOffsets('featureOffsets');
  if (geometry.kind === 'polygons') {
    readOffsets('polygonOffsets');
    readOffsets('ringOffsets');
  } else if (geometry.kind === 'lines') readOffsets('lineOffsets');
  validateRequirement(
    id,
    childCount <= positions.length / 2,
    `positions needs ${childCount} rows, received ${positions.length / 2}`
  );
  result.positions = positions.slice(0, childCount * 2);
  if (geometry.sourceIds)
    result.sourceIds = getPackedUint32Values(geometry.sourceIds, count).slice(0, count);
  return result;
}

/** @internal Whether feature rows group point or linestring parts. */
export function isMultipartJoinGeometry(
  geometry: Readonly<{kind: string}>
): geometry is GPUPackedGeometryPort<GPUMultipartGeometryPort> {
  return (
    (geometry.kind === 'points' || geometry.kind === 'lines') &&
    'featureOffsets' in geometry &&
    geometry.featureOffsets !== undefined
  );
}

/** Refuse multipart layouts at boundaries that still require individual point/part rows. @internal */
export function validateSpatialJoinMultipartGeometry(
  id: string,
  name: string,
  geometry: Readonly<{kind: string}>
): void {
  if (geometry.kind === 'points' || geometry.kind === 'lines') {
    validateRequirement(
      id,
      !('featureOffsets' in geometry) || geometry.featureOffsets === undefined,
      `${name}.featureOffsets multipart ${geometry.kind} are unsupported; use explicit part rows with an owner map or a feature-level CPU join`
    );
  }
}

/** Returns the number of features in a predicate-join geometry. @internal */
export function getSpatialJoinFeatureCount(geometry: GPUSpatialJoinGeometry): number {
  if (isMultipartJoinGeometry(geometry)) return geometry.featureOffsets.length - 1;
  switch (geometry.kind) {
    case 'points':
      return geometry.positions.length;
    case 'lines':
      return geometry.lineOffsets.length - 1;
    case 'polygons':
      return geometry.featureOffsets.length - 1;
  }
}

/** Returns every input view of a predicate-join geometry. @internal */
export function getSpatialJoinGeometryViews(geometry: GPUSpatialJoinGeometry): GraphDataView[] {
  if (geometry.kind === 'lines') {
    return [
      geometry.positions,
      geometry.lineOffsets,
      ...(isMultipartJoinGeometry(geometry) ? [geometry.featureOffsets] : [])
    ];
  }
  if (geometry.kind === 'polygons') {
    return [
      geometry.positions,
      geometry.featureOffsets,
      geometry.polygonOffsets,
      geometry.ringOffsets
    ];
  }
  return [
    geometry.positions,
    ...(isMultipartJoinGeometry(geometry) ? [geometry.featureOffsets] : [])
  ];
}

/**
 * Throws unless the geometry views are packed, the port layout is valid and, unless `allowEmpty`,
 * the geometry has at least one feature. @internal
 */
export function validateSpatialJoinGeometry(
  id: string,
  name: string,
  geometry: GPUSpatialJoinGeometry,
  options: {allowEmpty?: boolean; allowMultipart?: boolean} = {}
): void {
  if (!options.allowMultipart) validateSpatialJoinMultipartGeometry(id, name, geometry);
  for (const view of [...getSpatialJoinGeometryViews(geometry), geometry.sourceIds]) {
    if (view) {
      validateSinglePackedView(view, ['float32x2', 'uint32'], `${id} ${name}`);
    }
  }
  validateGPUFeatureGeometryPort(`${id} ${name}`, geometry);
  if (!options.allowEmpty && getSpatialJoinFeatureCount(geometry) < 1) {
    throw new Error(`${id} ${name} needs at least one feature`);
  }
}

/** Throws unless every present status scalar is a packed uint32 view with one row. @internal */
export function validateStatusScalars(
  id: string,
  status: Readonly<Record<string, GraphDataView | undefined>>
): void {
  for (const [name, view] of Object.entries(status)) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} output.status.${name} needs one row`);
      }
    }
  }
}

/** Returns whether two geometries are the same kind over the same views. @internal */
export function isSameSpatialJoinGeometry(
  first: GPUSpatialJoinGeometry,
  second: GPUSpatialJoinGeometry
): boolean {
  if (first.kind !== second.kind) {
    return false;
  }
  const firstViews = getSpatialJoinGeometryViews(first);
  const secondViews = getSpatialJoinGeometryViews(second);
  return (
    firstViews.length === secondViews.length &&
    firstViews.every((view, index) => {
      const other = secondViews[index];
      const physical = view.buffer.defaultBuffer
        ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
        : view.buffer;
      const otherPhysical = other.buffer.defaultBuffer
        ? getCoreDefaultBuffer(other.buffer.defaultBuffer)
        : other.buffer;
      return (
        physical === otherPhysical &&
        view.format === other.format &&
        view.length === other.length &&
        view.byteOffset === other.byteOffset &&
        view.byteStride === other.byteStride
      );
    })
  );
}

/** Writes `(ringStart, ringEnd)` per polygon feature; malformed offsets give an empty range. */
export function createFeatureRingsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    featureCount: number;
    geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>;
    featureRings: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'feature-rings',
    bindings: [
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureRings', view: props.featureRings, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: `const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;`,
    body: `let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`
  });
}

/** Writes per-feature bounds for any geometry kind; empty features get inverted bounds. */
export function createBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    /** Operation name of the generated kernels; defaults to the spatial join. */
    operation?: string;
    featureCount: number;
    geometry: GPUSpatialJoinGeometry;
    featureRings?: GraphDataView<'uint32x2'>;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
    /** Reject the whole predicate feature rather than retaining its finite parts. */
    rejectNonFiniteFeature?: boolean;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  if (geometry.kind === 'points' && !isMultipartJoinGeometry(geometry)) {
    return createSpatialJoinBoundsNode<Parameters>(graph, {
      id,
      operation: props.operation ?? OPERATION,
      featureCount: props.featureCount,
      source: {kind: 'points', positions: geometry.positions},
      minima: props.minima,
      maxima: props.maxima
    });
  }
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'}
  ];
  let range: string;
  if (isMultipartJoinGeometry(geometry)) {
    bindings.push({
      name: 'featureOffsets',
      view: geometry.featureOffsets,
      type: 'u32',
      access: 'read'
    });
    const partCount =
      geometry.kind === 'points' ? geometry.positions.length : geometry.lineOffsets.length - 1;
    if (geometry.kind === 'lines')
      bindings.push({name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'});
    range = `let partStart = featureOffsets[featureOffsetsOffset + index];
  let partEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (partStart < partEnd && partEnd <= ${partCount}u) {
    vertexStart = ${geometry.kind === 'points' ? 'partStart' : 'lineOffsets[lineOffsetsOffset + partStart]'};
    vertexEnd = min(${geometry.kind === 'points' ? 'partEnd' : 'lineOffsets[lineOffsetsOffset + partEnd]'}, VERTEX_COUNT);
  }`;
  } else if (geometry.kind === 'lines') {
    bindings.push({name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'});
    range = `let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);`;
  } else if (geometry.kind === 'polygons') {
    bindings.push(
      {
        name: 'featureRings',
        view: props.featureRings as GraphDataView,
        type: 'u32',
        access: 'read'
      },
      {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'}
    );
    range = `let ringStart = featureRings[featureRingsOffset + index * 2u];
  let ringEnd = featureRings[featureRingsOffset + index * 2u + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (ringStart < ringEnd) {
    vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
    vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
  }`;
  } else {
    range = 'let vertexStart = index; let vertexEnd = index + 1u;';
  }
  bindings.push(
    {name: 'featureMinima', view: props.minima, type: 'f32', access: 'read_write'},
    {name: 'featureMaxima', view: props.maxima, type: 'f32', access: 'read_write'}
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: props.operation ?? OPERATION,
    variant: `bounds-${geometry.kind}`,
    bindings,
    invocationCount: props.featureCount,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const VERTEX_COUNT: u32 = ${geometry.positions.length}u;`,
    body: `${range}
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  ${props.rejectNonFiniteFeature ? 'var invalid = false;' : ''}
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    } ${props.rejectNonFiniteFeature ? 'else { invalid = true; }' : ''}
  }
  ${props.rejectNonFiniteFeature ? 'if (invalid) { minimum = vec2f(FLOAT32_MAXIMUM); maximum = vec2f(-FLOAT32_MAXIMUM); }' : ''}
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`
  });
}
