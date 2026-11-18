// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotContiguityOperationProps} from './weights-construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {GPUContiguityCriterion} from './gpu-contiguity-weights';
import {
  GPUContiguityWeightsOperation,
  type ContiguityWeightsOperationProps
} from './gpu-contiguity-weights-operation';

/** Four-level polygon rows; rings close implicitly. Offset ranges may leave unused padding. */
export type CPUContiguityWeightsInput = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
};
/** Capacity-bounded CSR, with scalar status arrays. Slots beyond the terminal offset are unused. */
export type CPUContiguityWeightsResult = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  overflow: Uint32Array;
  requiredCount: Uint32Array;
  candidateOverflow: Uint32Array;
};

function validatePolygonOffsets(name: string, offsets: Uint32Array, maximum: number): void {
  if (offsets.length < 2)
    throw new Error(`computeContiguityWeightsOnCPU ${name} must contain at least two offsets`);
  let previous = 0;
  for (const offset of offsets) {
    if (offset < previous || offset > maximum)
      throw new Error(
        `computeContiguityWeightsOnCPU ${name} must be non-decreasing and within [0, ${maximum}]`
      );
    previous = offset;
  }
}

/**
 * Constructs exact queen/rook adjacency using shared point/edge keys, including holes and parts.
 * Directed slots are sorted by (feature, neighbor), then truncated to capacity. As on the GPU,
 * offsets are clamped and the count is the terminal offset; there is no separate count port.
 * Snapping is anchored at ringOffsets[0], falling back to zero for empty or non-finite input.
 *
 * What differs from the GPU contributor:
 * - An explicit pairCapacity binds candidate emission before deduplication; candidateOverflow
 *   reports loss and requiredCount is then a lower bound. Without an explicit budget the CPU
 *   uses no bounded pair scratch. Under candidate loss the retained prefix can differ because
 *   GPU float32 key ordering and CPU point-key insertion order can differ. Final overflow is
 *   separate and depends only on retained distinct links exceeding final capacity.
 * - Positive-tolerance snap keys use float64 round-half-up, without the GPU's float32 rounding,
 *   2^23-step shortcut or signed-integer clamp. Cells near float32 rounding boundaries may differ.
 * - The GPU snap-cell range limit is dropped. Zero tolerance still compares exact input float32
 *   values, normalizes signed zero, and ignores non-finite vertices.
 * - Unused output slots are zero-filled; the GPU leaves them unspecified. Integer used rows and
 *   binary weights agree exactly when snap cell identities agree and GPU candidates fit.
 *
 * @throws If positions is not paired, an offset array is inconsistent, or options are invalid.
 */
export function computeContiguityWeightsOnCPU(
  input: CPUContiguityWeightsInput,
  options: {
    criterion: GPUContiguityCriterion;
    capacity: number;
    snapTolerance?: number;
    pairCapacity?: number;
  }
): CPUContiguityWeightsResult {
  const {positions, ringOffsets, polygonOffsets, featureOffsets} = input;
  const {criterion, capacity} = options;
  const tolerance = options.snapTolerance ?? 0;
  validateCapacity('computeContiguityWeightsOnCPU', 'capacity', capacity, 0);
  if (options.pairCapacity !== undefined)
    validateCapacity('computeContiguityWeightsOnCPU', 'pairCapacity', options.pairCapacity);
  validateRequirement(
    'computeContiguityWeightsOnCPU',
    criterion === 'queen' || criterion === 'rook',
    "criterion must be 'queen' or 'rook'"
  );
  validateRequirement(
    'computeContiguityWeightsOnCPU',
    Number.isFinite(tolerance) && tolerance >= 0,
    'snapTolerance must be finite and >= 0'
  );
  if (positions.length % 2 !== 0)
    throw new Error('computeContiguityWeightsOnCPU positions must contain float32x2 rows');
  validatePolygonOffsets('ringOffsets', ringOffsets, positions.length / 2);
  validatePolygonOffsets('polygonOffsets', polygonOffsets, ringOffsets.length - 1);
  validatePolygonOffsets('featureOffsets', featureOffsets, polygonOffsets.length - 1);
  const featureCount = featureOffsets.length - 1;
  const firstVertex = ringOffsets[0];
  const hasOrigin =
    firstVertex < ringOffsets[ringOffsets.length - 1] &&
    Number.isFinite(positions[2 * firstVertex]) &&
    Number.isFinite(positions[2 * firstVertex + 1]);
  const originX = hasOrigin ? positions[2 * firstVertex] : 0;
  const originY = hasOrigin ? positions[2 * firstVertex + 1] : 0;
  const pointIds = new Map<string, number>();
  const getPointId = (vertex: number): number | undefined => {
    const coordinateX = positions[vertex * 2];
    const coordinateY = positions[vertex * 2 + 1];
    if (!Number.isFinite(coordinateX) || !Number.isFinite(coordinateY)) return undefined;
    const keyX =
      tolerance > 0 ? Math.floor((coordinateX - originX) / tolerance + 0.5) : coordinateX;
    const keyY =
      tolerance > 0 ? Math.floor((coordinateY - originY) / tolerance + 0.5) : coordinateY;
    const key = `${keyX},${keyY}`;
    let pointId = pointIds.get(key);
    if (pointId === undefined) {
      pointId = pointIds.size;
      pointIds.set(key, pointId);
    }
    return pointId;
  };
  const owners = new Map<string, Set<number>>();
  for (let feature = 0; feature < featureCount; feature++) {
    for (let polygon = featureOffsets[feature]; polygon < featureOffsets[feature + 1]; polygon++) {
      for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
        const begin = ringOffsets[ring];
        const end = ringOffsets[ring + 1];
        for (let vertex = begin; vertex < end; vertex++) {
          const first = getPointId(vertex);
          if (first === undefined) continue;
          let key = String(first);
          if (criterion === 'rook') {
            const second = getPointId(vertex + 1 === end ? begin : vertex + 1);
            if (second === undefined || first === second) continue;
            key = `${Math.min(first, second)},${Math.max(first, second)}`;
          }
          let features = owners.get(key);
          if (!features) {
            features = new Set();
            owners.set(key, features);
          }
          features.add(feature);
        }
      }
    }
  }
  const adjacency = Array.from({length: featureCount}, () => new Set<number>());
  let candidateCount = 0;
  const pairCapacity = options.pairCapacity ?? Infinity;
  for (const features of owners.values()) {
    for (const first of features) {
      for (const second of features)
        if (first !== second) {
          if (candidateCount < pairCapacity) adjacency[first].add(second);
          candidateCount++;
        }
    }
  }
  const offsets = new Uint32Array(featureCount + 1);
  const neighbors = new Uint32Array(capacity);
  const weights = new Float32Array(capacity);
  let requiredCount = 0;
  adjacency.forEach((features, feature) => {
    for (const neighbor of [...features].sort((left, right) => left - right)) {
      if (requiredCount < capacity) {
        neighbors[requiredCount] = neighbor;
        weights[requiredCount] = 1;
      }
      requiredCount++;
    }
    offsets[feature + 1] = Math.min(requiredCount, capacity);
  });
  if (requiredCount > 0xffffffff)
    throw new Error('computeContiguityWeightsOnCPU requiredCount exceeds uint32 layout');
  return {
    offsets,
    neighbors,
    weights,
    requiredCount: Uint32Array.of(requiredCount),
    overflow: Uint32Array.of(Number(requiredCount > capacity)),
    candidateOverflow: Uint32Array.of(Number(candidateCount > pairCapacity))
  };
}

/** CPU twin. Explicit pairCapacity is honored; the GPU snap-range restriction is dropped. */
export class CPUContiguityWeightsOperation extends GPUCPUOperation {
  constructor(props: ContiguityWeightsOperationProps) {
    props = snapshotContiguityOperationProps(props);
    const id = props.id ?? 'contiguity-weights';
    // Validate names before duplicate object keys can erase a diagnostic in the CPU schema.
    const operation = new GPUContiguityWeightsOperation(props);
    const shape = {length: props.capacity};
    super({
      id,
      inputs: operation.inputs,
      outputs: {
        [props.output.offsets]: gpuGraph.vector('uint32', {length: props.featureCount + 1}),
        [props.output.neighbors]: gpuGraph.vector('uint32', shape),
        [props.output.weights]: gpuGraph.vector('float32', shape),
        [props.output.status.overflow]: gpuGraph.vector('uint32', {length: 1}),
        ...(props.output.status.requiredCount
          ? {[props.output.status.requiredCount]: gpuGraph.vector('uint32', {length: 1})}
          : {}),
        ...(props.output.status.candidateOverflow
          ? {[props.output.status.candidateOverflow]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const featureOffsets = getCPUOperationUint32Rows(id, inputs, 'featureOffsets');
        if (featureOffsets.length !== props.featureCount + 1)
          throw new Error(`${id} featureOffsets length must equal featureCount + 1`);
        const result = computeContiguityWeightsOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets'),
            polygonOffsets: getCPUOperationUint32Rows(id, inputs, 'polygonOffsets'),
            featureOffsets
          },
          props
        );
        return {
          [props.output.offsets]: result.offsets,
          [props.output.neighbors]: result.neighbors,
          [props.output.weights]: result.weights,
          [props.output.status.overflow]: result.overflow,
          ...(props.output.status.requiredCount
            ? {[props.output.status.requiredCount]: result.requiredCount}
            : {}),
          ...(props.output.status.candidateOverflow
            ? {[props.output.status.candidateOverflow]: result.candidateOverflow}
            : {})
        };
      }
    });
  }
}
