// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUContributorOperation,
  type GPUContributorOperationViewGetter
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
import {GPUSpatialPredicateJoin} from './gpu-spatial-predicate-join';
import {GPUNearestFeatureJoin} from './gpu-nearest-feature-join';
import {
  computeSpatialPredicateJoinOnCPU,
  computeNearestFeatureJoinOnCPU,
  FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY,
  type CPUSpatialJoinGeometry,
  type MultipartJoinPredicate
} from './cpu-feature-join';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';

/** One packed graph value and its compile-time row count. */
export type FeatureJoinValue = Readonly<{value: string; length: number}>;
/** Named graph values with the shared point/line/polygon hierarchy and feature-row identities. */
export type FeatureJoinGeometryValues = Readonly<{
  kind: 'points' | 'lines' | 'polygons';
  positions: FeatureJoinValue;
  featureOffsets?: FeatureJoinValue;
  lineOffsets?: FeatureJoinValue;
  polygonOffsets?: FeatureJoinValue;
  ringOffsets?: FeatureJoinValue;
  sourceIds?: FeatureJoinValue;
}>;
/** Planar, unprepared predicate operation; both factories accept exactly this props type. */
export type SpatialPredicateJoinOperationProps = {
  id?: string;
  left: FeatureJoinGeometryValues;
  right: FeatureJoinGeometryValues;
  predicate: MultipartJoinPredicate;
  candidateCapacity: number;
  pairCapacity: number;
  output: {
    leftRows: string;
    rightRows: string;
    count: string;
    requiredCount: string;
    overflow: string;
    candidateOverflow: string;
    candidateTruncated?: string;
  };
};
/** Planar, unprepared one-neighbor operation; ties select the lowest target FEATURE row. */
export type NearestFeatureJoinOperationProps = {
  id?: string;
  queries: FeatureJoinGeometryValues;
  features: FeatureJoinGeometryValues;
  output: {
    neighborRows: string;
    neighborIds: string;
    neighborCounts: string;
    neighborDistances: string;
    overflow: string;
    candidateOverflow: string;
  };
};
const GEOMETRY_FIELDS = [
  'positions',
  'featureOffsets',
  'lineOffsets',
  'polygonOffsets',
  'ringOffsets',
  'sourceIds'
] as const;
function getPortName(prefix: string, field: string): string {
  return prefix + field[0].toUpperCase() + field.slice(1);
}
function snapshotGeometry(
  id: string,
  geometry: FeatureJoinGeometryValues,
  minimumFeatureCount = 1
): FeatureJoinGeometryValues {
  const result = {kind: geometry.kind} as {kind: FeatureJoinGeometryValues['kind']} & Partial<
    Record<(typeof GEOMETRY_FIELDS)[number], FeatureJoinValue>
  >;
  for (const field of GEOMETRY_FIELDS) {
    const value = geometry[field];
    if (value) {
      const {value: graphValue, length} = value;
      validateCapacity(
        id,
        `${field}.length`,
        length,
        field === 'positions' || field === 'sourceIds' ? 0 : 1
      );
      result[field] = Object.freeze({value: graphValue, length});
    }
  }
  validateRequirement(id, Boolean(result.positions), 'positions is required');
  validateRequirement(
    id,
    ['points', 'lines', 'polygons'].includes(result.kind),
    'geometry.kind must be points, lines or polygons'
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
  validateCapacity(
    id,
    'feature count',
    getFeatureCount(result as FeatureJoinGeometryValues),
    minimumFeatureCount
  );
  validateRequirement(
    id,
    !result.sourceIds ||
      result.sourceIds.length === getFeatureCount(result as FeatureJoinGeometryValues),
    'sourceIds length must equal feature count'
  );
  return Object.freeze(result) as FeatureJoinGeometryValues;
}
function getFeatureCount(geometry: FeatureJoinGeometryValues): number {
  return geometry.featureOffsets
    ? geometry.featureOffsets.length - 1
    : geometry.kind === 'points'
      ? geometry.positions.length
      : geometry.lineOffsets!.length - 1;
}
function getGeometryInputs(
  prefix: string,
  geometry: FeatureJoinGeometryValues
): Record<string, string> {
  return Object.fromEntries(
    GEOMETRY_FIELDS.filter(field => geometry[field]).map(field => [
      getPortName(prefix, field),
      geometry[field]!.value
    ])
  );
}
function resolveGeometry(
  id: string,
  prefix: string,
  geometry: FeatureJoinGeometryValues,
  getView: GPUContributorOperationViewGetter
): GPUSpatialJoinGeometry {
  return {
    kind: geometry.kind,
    ...Object.fromEntries(
      GEOMETRY_FIELDS.filter(field => geometry[field]).map(field => {
        const port = getPortName(prefix, field);
        const view = getView(port);
        validateRequirement(
          id,
          view.length === geometry[field]!.length,
          `${port} must match its declared length`
        );
        return [field, view];
      })
    )
  } as GPUSpatialJoinGeometry;
}
function readGeometry(
  id: string,
  prefix: string,
  geometry: FeatureJoinGeometryValues,
  inputs: CPUOperationInputs
): CPUSpatialJoinGeometry {
  return {
    kind: geometry.kind,
    ...Object.fromEntries(
      GEOMETRY_FIELDS.filter(field => geometry[field]).map(field => {
        const rows =
          field === 'positions'
            ? getCPUOperationFloat32Rows(id, inputs, getPortName(prefix, field), 'float32x2')
            : getCPUOperationUint32Rows(id, inputs, getPortName(prefix, field));
        validateRequirement(
          id,
          rows.length === geometry[field]!.length * (field === 'positions' ? 2 : 1),
          `${getPortName(prefix, field)} must match its declared length`
        );
        return [field, rows];
      })
    )
  } as CPUSpatialJoinGeometry;
}

/** Internal graph adapter using the existing GPUContributorOperation. @internal */
export function createSpatialPredicateJoinOperation(
  input: SpatialPredicateJoinOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'spatial-predicate-join-operation';
  const left = snapshotGeometry(id, input.left),
    right = snapshotGeometry(id, input.right);
  const {predicate, candidateCapacity, pairCapacity} = input;
  const output = Object.freeze({...input.output});
  validateCapacity(
    id,
    'candidateCapacity',
    candidateCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  validateCapacity(id, 'pairCapacity', pairCapacity, 1, FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY);
  validateRequirement(
    id,
    ['intersects', 'within', 'coveredBy'].includes(predicate),
    'predicate must be intersects, within or coveredBy'
  );
  validateRequirement(
    id,
    left.kind !== 'polygons' || right.kind !== 'polygons' || predicate === 'intersects',
    'polygon/polygon containment is unsupported by the feature-join operation forms'
  );
  return new GPUContributorOperation({
    id,
    inputs: {...getGeometryInputs('left', left), ...getGeometryInputs('right', right)},
    outputs: Object.fromEntries(
      Object.entries(output).map(([field, value]) => [
        field,
        {
          value,
          format: 'uint32' as const,
          length:
            field === 'leftRows' || field === 'rightRows'
              ? pairCapacity
              : field === 'candidateTruncated'
                ? getFeatureCount(left)
                : 1
        }
      ])
    ),
    emptyOutputPorts: ['leftRows', 'rightRows'],
    createContributor: getView =>
      new GPUSpatialPredicateJoin({
        id,
        left: resolveGeometry(id, 'left', left, getView),
        right: resolveGeometry(id, 'right', right, getView),
        predicate,
        engine: predicate === 'coveredBy' ? 'auto' : 'fast',
        candidateCapacity,
        output: {
          pairs: {
            leftRows: getView<'uint32'>('leftRows'),
            rightRows: getView<'uint32'>('rightRows'),
            status: {
              count: getView<'uint32'>('count'),
              requiredCount: getView<'uint32'>('requiredCount'),
              overflow: getView<'uint32'>('overflow'),
              candidateOverflow: getView<'uint32'>('candidateOverflow')
            }
          },
          ...(output.candidateTruncated
            ? {candidateTruncated: getView<'uint32'>('candidateTruncated')}
            : {})
        }
      })
  });
}
/** CPU twin of the internal predicate adapter, including the same candidate budget. @internal */
export function createCPUSpatialPredicateJoinOperation(
  input: SpatialPredicateJoinOperationProps
): GPUCPUOperation {
  const {
    id,
    left: originalLeft,
    right: originalRight,
    predicate,
    candidateCapacity,
    pairCapacity,
    output
  } = input;
  const left = snapshotGeometry(id ?? 'spatial-predicate-join-operation', originalLeft),
    right = snapshotGeometry(id ?? 'spatial-predicate-join-operation', originalRight);
  const operation = createSpatialPredicateJoinOperation({
    id,
    left,
    right,
    predicate,
    candidateCapacity,
    pairCapacity,
    output
  });
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
      const result = computeSpatialPredicateJoinOnCPU(
        readGeometry(operation.id, 'left', left, inputs),
        readGeometry(operation.id, 'right', right, inputs),
        {predicate, candidateCapacity, pairCapacity}
      );
      const arrays: Record<string, Uint32Array> = {
        leftRows: result.leftRows,
        rightRows: result.rightRows,
        candidateTruncated: result.candidateTruncated
      };
      for (const [name, value] of Object.entries(result.status))
        arrays[name] = Uint32Array.of(value);
      return Object.fromEntries(
        Object.entries(operation.outputs).map(([port, output]) => [output.value, arrays[port]])
      );
    }
  });
}
/** Internal GPU one-neighbor adapter, keeping feature rows and application IDs separate. @internal */
export function createNearestFeatureJoinOperation(
  input: NearestFeatureJoinOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'nearest-feature-join-operation';
  const queries = snapshotGeometry(id, input.queries, 0),
    features = snapshotGeometry(id, input.features, 0);
  const output = Object.freeze({...input.output});
  return new GPUContributorOperation({
    id,
    inputs: {...getGeometryInputs('queries', queries), ...getGeometryInputs('features', features)},
    outputs: Object.fromEntries(
      Object.entries(output).map(([field, value]) => [
        field,
        {
          value,
          format: field === 'neighborDistances' ? ('float32' as const) : ('uint32' as const),
          length:
            field === 'overflow' || field === 'candidateOverflow' ? 1 : getFeatureCount(queries)
        }
      ])
    ),
    emptyOutputPorts: ['neighborRows', 'neighborIds', 'neighborCounts', 'neighborDistances'],
    createContributor: getView =>
      new GPUNearestFeatureJoin({
        id,
        queries: resolveGeometry(id, 'queries', queries, getView),
        features: resolveGeometry(id, 'features', features, getView),
        spatialSort: false,
        querySort: false,
        output: {
          neighborRows: getView<'uint32'>('neighborRows'),
          neighborIds: getView<'uint32'>('neighborIds'),
          neighborCounts: getView<'uint32'>('neighborCounts'),
          neighborDistances: getView<'float32'>('neighborDistances'),
          status: {
            overflow: getView<'uint32'>('overflow'),
            candidateOverflow: getView<'uint32'>('candidateOverflow')
          }
        }
      })
  });
}
/** CPU twin of the internal planar nearest adapter; minimum over all parts on both sides. @internal */
export function createCPUNearestFeatureJoinOperation(
  input: NearestFeatureJoinOperationProps
): GPUCPUOperation {
  const {id, queries: originalQueries, features: originalFeatures, output} = input;
  const queries = snapshotGeometry(id ?? 'nearest-feature-join-operation', originalQueries, 0),
    features = snapshotGeometry(id ?? 'nearest-feature-join-operation', originalFeatures, 0);
  const operation = createNearestFeatureJoinOperation({id, queries, features, output});
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
      const result = computeNearestFeatureJoinOnCPU(
        readGeometry(operation.id, 'queries', queries, inputs),
        readGeometry(operation.id, 'features', features, inputs)
      );
      const arrays: Record<string, Float32Array | Uint32Array> = {
        neighborRows: result.neighborRows,
        neighborIds: result.neighborIds,
        neighborCounts: result.neighborCounts,
        neighborDistances: result.neighborDistances,
        overflow: Uint32Array.of(result.status.overflow),
        candidateOverflow: Uint32Array.of(result.status.candidateOverflow)
      };
      return Object.fromEntries(
        Object.entries(operation.outputs).map(([port, output]) => [output.value, arrays[port]])
      );
    }
  });
}
