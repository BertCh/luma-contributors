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
  validateFeatureJoinOptions,
  type CPUFeatureJoinOptions,
  type CPUSpatialJoinGeometry,
  type MultipartJoinPredicate
} from './cpu-feature-join';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {decodeGPUSpatialJoinGeometry} from './spatial-join-geometry';

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
  /** Named one-row uint32 live feature count, bounded by left capacity. */
  leftFeatureCount?: string;
  /** Named one-row uint32 producer loss/certification inputs. */
  leftStatus?: {overflow?: string; candidateOverflow?: string; complete?: string};
  right: FeatureJoinGeometryValues;
  predicate: MultipartJoinPredicate;
  pattern?: CPUFeatureJoinOptions['pattern'];
  distance?: number;
  engine?: CPUFeatureJoinOptions['engine'];
  how?: CPUFeatureJoinOptions['how'];
  candidateCapacity: number;
  pairCapacity: number;
  output: {
    leftRows?: string;
    rightRows?: string;
    /** Ascending left feature rows without a known match; pairCapacity slots, readable only through the unmatched count (count for anti). Tails are unspecified. */
    unmatchedRows?: string;
    /** Independent compact absence diagnostics for an inner join's unmatchedRows. */
    unmatchedCount?: string;
    unmatchedRequiredCount?: string;
    unmatchedOverflow?: string;
    unmatchedCandidateOverflow?: string;
    relate?: string;
    count: string;
    requiredCount: string;
    overflow: string;
    candidateOverflow: string;
    candidateTruncated?: string;
    /** Provisional GPU pairs with uncertified orientations; the exact CPU twin writes zero. */
    uncertainCount?: string;
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

function resolvePredicateScalar(
  id: string,
  port: string,
  getView: GPUContributorOperationViewGetter
) {
  const view = getView(port, 'uint32');
  validateRequirement(
    id,
    view.length === 1,
    `${port} needs one uint32 row; received ${view.length}`
  );
  return view;
}

/** Internal graph adapter using the existing GPUContributorOperation. @internal */
export function createSpatialPredicateJoinOperation(
  input: SpatialPredicateJoinOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'spatial-predicate-join-operation';
  const left = snapshotGeometry(id, input.left),
    right = snapshotGeometry(id, input.right);
  const leftFeatureCount = input.leftFeatureCount;
  const originalLeftStatus = input.leftStatus;
  const leftStatus =
    originalLeftStatus &&
    snapshotConstructionFields(originalLeftStatus, ['overflow', 'candidateOverflow', 'complete']);
  const {
    predicate,
    candidateCapacity,
    pairCapacity,
    distance,
    engine,
    how,
    pattern: originalPattern
  } = input;
  const pattern = Array.isArray(originalPattern)
    ? Object.freeze([...originalPattern])
    : originalPattern;
  const output = snapshotConstructionFields(input.output, [
    'leftRows',
    'rightRows',
    'unmatchedRows',
    'unmatchedCount',
    'unmatchedRequiredCount',
    'unmatchedOverflow',
    'unmatchedCandidateOverflow',
    'relate',
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'candidateTruncated',
    'uncertainCount'
  ]);
  validateRequirement(
    id,
    predicate !== 'dwithin' || !output.relate,
    'output.relate requires a topological predicate because dwithin measures proximity; use a separate relate join'
  );
  validateFeatureJoinOptions(
    id,
    {predicate, candidateCapacity, pairCapacity, distance, engine, how, pattern},
    Boolean(
      (left.kind !== 'polygons' && left.featureOffsets) ||
        (right.kind !== 'polygons' && right.featureOffsets)
    )
  );
  validateRequirement(
    id,
    how === 'anti'
      ? Boolean(output.unmatchedRows) && !output.relate && !output.leftRows && !output.rightRows
      : Boolean(output.leftRows && output.rightRows),
    'output requires unmatchedRows for how anti, or leftRows and rightRows for an inner join; relate requires an inner join'
  );
  validateRequirement(
    id,
    how === 'anti' ||
      !output.unmatchedRows ||
      Boolean(output.unmatchedCount && output.unmatchedOverflow),
    'inner output.unmatchedRows requires separate unmatchedCount and unmatchedOverflow destinations'
  );
  validateRequirement(
    id,
    ![
      'unmatchedCount',
      'unmatchedRequiredCount',
      'unmatchedOverflow',
      'unmatchedCandidateOverflow'
    ].some(field => output[field as keyof typeof output] !== undefined) ||
      (how !== 'anti' && Boolean(output.unmatchedRows)),
    'separate unmatched diagnostics require inner output.unmatchedRows; anti uses count, requiredCount, overflow and candidateOverflow'
  );
  for (const field of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const) {
    validateRequirement(
      id,
      typeof output[field] === 'string' && output[field].length > 0,
      how === 'anti' && field === 'candidateOverflow'
        ? "how 'anti' requires output.candidateOverflow. Add a non-empty graph value name for candidate loss."
        : `outputs.${field}.value must be a non-empty graph value name`
    );
  }
  return new GPUContributorOperation({
    id,
    inputs: {
      ...getGeometryInputs('left', left),
      ...getGeometryInputs('right', right),
      ...(leftFeatureCount !== undefined ? {leftFeatureCount} : {}),
      ...Object.fromEntries(
        Object.entries(leftStatus ?? {})
          .filter(([, value]) => value !== undefined)
          .map(([field, value]) => [getPortName('leftStatus', field), value!])
      )
    },
    outputs: Object.fromEntries(
      Object.entries(output)
        .filter(
          ([field, value]) =>
            value !== undefined ||
            ['count', 'requiredCount', 'overflow', 'candidateOverflow'].includes(field)
        )
        .map(([field, value]) => [
          field,
          {
            value,
            format: 'uint32' as const,
            length:
              field === 'leftRows' ||
              field === 'rightRows' ||
              field === 'unmatchedRows' ||
              field === 'relate'
                ? pairCapacity
                : field === 'candidateTruncated'
                  ? getFeatureCount(left)
                  : 1
          }
        ])
    ),
    emptyOutputPorts: ['unmatchedRows', 'leftRows', 'rightRows'],
    createContributor: getView =>
      new GPUSpatialPredicateJoin({
        id,
        left: resolveGeometry(id, 'left', left, getView),
        leftFeatureCount:
          leftFeatureCount !== undefined
            ? resolvePredicateScalar(id, 'leftFeatureCount', getView)
            : undefined,
        leftStatus:
          leftStatus &&
          Object.fromEntries(
            Object.entries(leftStatus)
              .filter(([, value]) => value !== undefined)
              .map(([field]) => [
                field,
                resolvePredicateScalar(id, getPortName('leftStatus', field), getView)
              ])
          ),
        right: resolveGeometry(id, 'right', right, getView),
        predicate,
        pattern,
        distance,
        how,
        engine,
        candidateCapacity,
        output: {
          ...(how === 'anti'
            ? {
                unmatched: {
                  ids: getView<'uint32'>('unmatchedRows'),
                  status: {
                    count: getView<'uint32'>('count'),
                    requiredCount: getView<'uint32'>('requiredCount'),
                    overflow: getView<'uint32'>('overflow'),
                    candidateOverflow: getView<'uint32'>('candidateOverflow')
                  }
                }
              }
            : {
                pairs: {
                  leftRows: getView<'uint32'>('leftRows'),
                  rightRows: getView<'uint32'>('rightRows'),
                  status: {
                    count: getView<'uint32'>('count'),
                    requiredCount: getView<'uint32'>('requiredCount'),
                    overflow: getView<'uint32'>('overflow'),
                    candidateOverflow: getView<'uint32'>('candidateOverflow')
                  }
                }
              }),
          ...(how !== 'anti' && output.unmatchedRows
            ? {
                unmatched: {
                  ids: getView<'uint32'>('unmatchedRows'),
                  status: {
                    count: getView<'uint32'>('unmatchedCount'),
                    overflow: getView<'uint32'>('unmatchedOverflow'),
                    ...(output.unmatchedRequiredCount
                      ? {requiredCount: getView<'uint32'>('unmatchedRequiredCount')}
                      : {}),
                    ...(output.unmatchedCandidateOverflow
                      ? {candidateOverflow: getView<'uint32'>('unmatchedCandidateOverflow')}
                      : {})
                  }
                }
              }
            : {}),
          ...(output.relate ? {relate: getView<'uint32'>('relate')} : {}),
          ...(output.candidateTruncated
            ? {candidateTruncated: getView<'uint32'>('candidateTruncated')}
            : {}),
          ...(output.uncertainCount
            ? {status: {uncertainCount: getView<'uint32'>('uncertainCount')}}
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
    leftFeatureCount,
    leftStatus: originalLeftStatus,
    right: originalRight,
    predicate,
    candidateCapacity,
    pairCapacity,
    pattern: originalPattern,
    distance,
    engine,
    how,
    output
  } = input;
  const pattern = Array.isArray(originalPattern)
    ? Object.freeze([...originalPattern])
    : originalPattern;
  const left = snapshotGeometry(id ?? 'spatial-predicate-join-operation', originalLeft),
    right = snapshotGeometry(id ?? 'spatial-predicate-join-operation', originalRight);
  const leftStatus =
    originalLeftStatus &&
    snapshotConstructionFields(originalLeftStatus, ['overflow', 'candidateOverflow', 'complete']);
  const operation = createSpatialPredicateJoinOperation({
    id,
    left,
    leftFeatureCount,
    leftStatus,
    right,
    predicate,
    candidateCapacity,
    pairCapacity,
    pattern,
    distance,
    engine,
    how,
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
      const readScalar = (port: string): number => {
        const values = getCPUOperationUint32Rows(operation.id, inputs, port);
        validateRequirement(
          operation.id,
          values.length === 1,
          `${port} needs one uint32 row; received ${values.length}`
        );
        return values[0];
      };
      const capacity = getFeatureCount(left);
      const count = leftFeatureCount !== undefined ? readScalar('leftFeatureCount') : capacity;
      const invalidCount = count > capacity;
      const status = Object.fromEntries(
        Object.entries(leftStatus ?? {})
          .filter(([, value]) => value !== undefined)
          .map(([field]) => [field, readScalar(getPortName('leftStatus', field))])
      );
      if (invalidCount) status['overflow'] = 1;
      const leftGeometry = readGeometry(operation.id, 'left', left, inputs);
      const result = computeSpatialPredicateJoinOnCPU(
        leftFeatureCount !== undefined
          ? decodeGPUSpatialJoinGeometry(leftGeometry, invalidCount ? 0 : count)
          : leftGeometry,
        readGeometry(operation.id, 'right', right, inputs),
        {
          predicate,
          candidateCapacity,
          pairCapacity,
          pattern,
          distance,
          engine,
          how,
          leftStatus: status,
          relate: Boolean(operation.outputs['relate'])
        }
      );
      const arrays: Record<string, Uint32Array> = {
        leftRows: result.leftRows,
        rightRows: result.rightRows,
        unmatchedRows: result.unmatchedRows,
        unmatchedCount: Uint32Array.of(result.unmatchedStatus.count),
        unmatchedRequiredCount: Uint32Array.of(result.unmatchedStatus.requiredCount),
        unmatchedOverflow: Uint32Array.of(result.unmatchedStatus.overflow),
        unmatchedCandidateOverflow: Uint32Array.of(result.unmatchedStatus.candidateOverflow),
        relate: result.relate,
        uncertainCount: Uint32Array.of(0)
      };
      if (operation.outputs['candidateTruncated']) {
        const truncated = result.candidateTruncated;
        if (truncated.length === capacity) arrays['candidateTruncated'] = truncated;
        else {
          const padded = new Uint32Array(capacity);
          padded.set(truncated);
          arrays['candidateTruncated'] = padded;
        }
      }
      for (const [name, value] of Object.entries(
        how === 'anti' ? result.unmatchedStatus : result.status
      ))
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
  const output = snapshotConstructionFields(input.output, [
    'neighborRows',
    'neighborIds',
    'neighborCounts',
    'neighborDistances',
    'overflow',
    'candidateOverflow'
  ]);
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
