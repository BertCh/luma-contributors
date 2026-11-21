// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {
  GPUContributorOperation,
  GPUContributorOperationViewGetter
} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows,
  type CPUOperationInputs
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {FeatureJoinGeometryValues} from './feature-join-operation';
import type {CPUSpatialJoinGeometry} from './cpu-feature-join';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';
import {validateCPUHostOffsets} from '../../analysis-core/operations/compact-host-offsets';

const FIELDS = [
  'positions',
  'featureOffsets',
  'lineOffsets',
  'polygonOffsets',
  'ringOffsets',
  'sourceIds'
] as const;
/** Packed planar geometry declaration shared only by this stream's adapters. @internal */
export function snapshotPointJoinGeometry(
  id: string,
  geometry: FeatureJoinGeometryValues
): FeatureJoinGeometryValues {
  const fields = Object.fromEntries(
    FIELDS.flatMap(field => {
      const descriptor = geometry[field];
      if (!descriptor) return [];
      const {value, length} = descriptor;
      validateCapacity(
        id,
        `${field}.length`,
        length,
        field === 'positions' || field === 'sourceIds' ? 0 : 1
      );
      return [[field, Object.freeze({value, length})]];
    })
  );
  const result = Object.freeze({kind: geometry.kind, ...fields}) as FeatureJoinGeometryValues;
  validateRequirement(id, Boolean(result.positions), 'positions is required');
  validateRequirement(
    id,
    ['points', 'lines', 'polygons'].includes(result.kind),
    'kind must be points, lines or polygons'
  );
  validateRequirement(
    id,
    result.kind !== 'lines' || Boolean(result.lineOffsets),
    'lines require lineOffsets'
  );
  validateRequirement(
    id,
    result.kind !== 'polygons' ||
      Boolean(result.featureOffsets && result.polygonOffsets && result.ringOffsets),
    'polygons require featureOffsets, polygonOffsets and ringOffsets'
  );
  validateRequirement(
    id,
    !result.sourceIds || result.sourceIds.length === getPointJoinFeatureCount(result),
    'sourceIds needs one row per feature'
  );
  return result;
}
/** Number of logical features, independent of part rows. @internal */
export function getPointJoinFeatureCount(geometry: FeatureJoinGeometryValues): number {
  return geometry.featureOffsets
    ? geometry.featureOffsets.length - 1
    : geometry.kind === 'points'
      ? geometry.positions.length
      : geometry.lineOffsets!.length - 1;
}
/** Declares geometry values under a role prefix. @internal */
export function getPointJoinGeometryInputs(
  prefix: string,
  geometry: FeatureJoinGeometryValues
): Record<string, string> {
  return Object.fromEntries(
    FIELDS.filter(field => geometry[field]).map(field => [
      `${prefix}${field}`,
      geometry[field]!.value
    ])
  );
}
/** Resolves only packed views of the declared size. @internal */
export function resolvePointJoinGeometry(
  id: string,
  prefix: string,
  geometry: FeatureJoinGeometryValues,
  getView: GPUContributorOperationViewGetter
): GPUSpatialJoinGeometry {
  return {
    kind: geometry.kind,
    ...Object.fromEntries(
      FIELDS.filter(field => geometry[field]).map(field => {
        const view = getView(`${prefix}${field}`);
        validateRequirement(
          id,
          view.length === geometry[field]!.length,
          `${prefix}${field} must match its declared length`
        );
        return [field, view];
      })
    )
  } as GPUSpatialJoinGeometry;
}
/** Reads the same values in a CPU operation. @internal */
export function readPointJoinGeometry(
  id: string,
  prefix: string,
  geometry: FeatureJoinGeometryValues,
  inputs: CPUOperationInputs
): CPUSpatialJoinGeometry {
  return {
    kind: geometry.kind,
    ...Object.fromEntries(
      FIELDS.filter(field => geometry[field]).map(field => {
        const rows =
          field === 'positions'
            ? getCPUOperationFloat32Rows(id, inputs, `${prefix}${field}`, 'float32x2')
            : getCPUOperationUint32Rows(id, inputs, `${prefix}${field}`);
        validateRequirement(
          id,
          rows.length === geometry[field]!.length * (field === 'positions' ? 2 : 1),
          `${prefix}${field} must match its declared length`
        );
        return [field, rows];
      })
    )
  } as CPUSpatialJoinGeometry;
}
/** CPU callback with the GPU adapter's exact value names, formats and dimensions. @internal */
export function createPointJoinCPUOperation(
  operation: GPUContributorOperation,
  compute: (inputs: CPUOperationInputs) => Record<string, Float32Array | Uint32Array>
): GPUCPUOperation {
  return new GPUCPUOperation({
    id: operation.id,
    inputs: operation.inputs,
    outputs: Object.fromEntries(
      Object.values(operation.outputs).map(output => [
        output.value,
        gpuGraph.vector(output.format, {length: output.length})
      ])
    ),
    execute({inputs}) {
      const result = compute(inputs);
      return Object.fromEntries(
        Object.entries(operation.outputs).map(([port, output]) => [output.value, result[port]])
      );
    }
  });
}
/** Validates a typed offset hierarchy on the CPU; equal offsets represent empty features. @internal */
export function validatePointJoinOffsets(
  id: string,
  name: string,
  offsets: Uint32Array,
  childCount: number
): void {
  validateCPUHostOffsets(id, name, offsets, childCount);
}
/** Validates point layout and returns feature offsets, preserving empty feature rows. @internal */
export function getCPUPointJoinOffsets(id: string, points: CPUSpatialJoinGeometry): Uint32Array {
  validateRequirement(
    id,
    points.kind === 'points' && points.positions.length % 2 === 0,
    'points.positions needs complete XY rows'
  );
  const count = points.positions.length / 2;
  const offsets = points.featureOffsets ?? Uint32Array.from({length: count + 1}, (_, row) => row);
  validatePointJoinOffsets(id, 'points.featureOffsets', offsets, count);
  validateRequirement(
    id,
    !points.sourceIds || points.sourceIds.length === offsets.length - 1,
    'points.sourceIds needs one row per feature'
  );
  return offsets;
}
