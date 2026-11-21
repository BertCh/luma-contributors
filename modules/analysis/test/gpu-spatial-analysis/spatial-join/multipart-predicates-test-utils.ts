// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import type {FeatureJoinGeometryValues} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

export type MultipartFixtureGeometry = {
  kind: CPUSpatialJoinGeometry['kind'];
  positions: readonly number[];
  featureOffsets?: readonly number[];
  lineOffsets?: readonly number[];
  polygonOffsets?: readonly number[];
  ringOffsets?: readonly number[];
  sourceIds?: readonly number[];
};

// By hand: feature 0 has no candidates; feature 1 is within right row 1; feature 2 is
// within right row 0. Deliberately unrelated IDs catch accidental ID/row substitution.
export const MULTIPART_ROW_FIXTURE = {
  left: {
    kind: 'points',
    positions: [-2, 0, 0.25, 0, 0.75, 0, 3.25, 0, 3.75, 0],
    featureOffsets: [0, 1, 3, 5],
    sourceIds: [90, 91, 92]
  },
  right: {
    kind: 'lines',
    positions: [3, 0, 3.5, 0, 3.5, 0, 4, 0, 0, 0, 0.5, 0, 0.5, 0, 1, 0],
    featureOffsets: [0, 2, 4],
    lineOffsets: [0, 2, 4, 6, 8],
    sourceIds: [32, 31]
  }
} satisfies {left: MultipartFixtureGeometry; right: MultipartFixtureGeometry};

export const MULTIPART_BUDGET_CASES = [
  {
    candidateCapacity: 2,
    pairCapacity: 2,
    how: 'inner',
    leftRows: [1, 2],
    rightRows: [1, 0],
    count: 2,
    requiredCount: 2,
    overflow: 0,
    candidateOverflow: 0,
    candidateTruncated: [0, 0, 0]
  },
  {
    candidateCapacity: 2,
    pairCapacity: 1,
    how: 'inner',
    leftRows: [1],
    rightRows: [1],
    count: 1,
    requiredCount: 2,
    overflow: 1,
    candidateOverflow: 0,
    candidateTruncated: [0, 0, 0]
  },
  {
    candidateCapacity: 1,
    pairCapacity: 2,
    how: 'inner',
    leftRows: [1],
    rightRows: [1],
    count: 1,
    requiredCount: 1,
    overflow: 0,
    candidateOverflow: 1,
    candidateTruncated: [0, 0, 1]
  },
  {
    candidateCapacity: 2,
    pairCapacity: 2,
    how: 'anti',
    leftRows: [0],
    rightRows: [],
    count: 1,
    requiredCount: 1,
    overflow: 0,
    candidateOverflow: 0,
    candidateTruncated: [0, 0, 0]
  },
  {
    candidateCapacity: 1,
    pairCapacity: 2,
    how: 'anti',
    leftRows: [0],
    rightRows: [],
    count: 1,
    requiredCount: 1,
    overflow: 0,
    candidateOverflow: 1,
    candidateTruncated: [0, 0, 1]
  }
] as const;
export function createMultipartCPUGeometry(
  geometry: MultipartFixtureGeometry
): CPUSpatialJoinGeometry {
  return Object.fromEntries(
    Object.entries(geometry).map(([field, value]) => [
      field,
      field === 'kind'
        ? value
        : field === 'positions'
          ? Float32Array.from(value as readonly number[])
          : Uint32Array.from(value as readonly number[])
    ])
  ) as CPUSpatialJoinGeometry;
}
export function createMultipartOperationInputs(
  left: MultipartFixtureGeometry,
  right: MultipartFixtureGeometry
) {
  const inputs: Record<string, OperationGraphInput> = {};
  const describe = (
    prefix: string,
    geometry: MultipartFixtureGeometry
  ): FeatureJoinGeometryValues =>
    ({
      kind: geometry.kind,
      ...Object.fromEntries(
        Object.entries(createMultipartCPUGeometry(geometry))
          .filter(([field]) => field !== 'kind')
          .map(([field, values]) => {
            const value = `${prefix}-${field}`;
            const positions = field === 'positions';
            inputs[value] = {
              format: positions ? 'float32x2' : 'uint32',
              values: values as Float32Array | Uint32Array
            };
            return [
              field,
              {value, length: (values as Float32Array | Uint32Array).length / (positions ? 2 : 1)}
            ];
          })
      )
    }) as FeatureJoinGeometryValues;
  return {left: describe('left', left), right: describe('right', right), inputs};
}
