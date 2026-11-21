// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import type {CPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import type {
  GPUMultipartGeometryPort,
  GPUPackedGeometryPort
} from '../../../src/analysis-core/contracts/index';
import oracle from './multipart-nearest.json';

export const MULTIPART_NEAREST_CASES = oracle.cases;
export const NO_FEATURE = 0xffffffff;
type Format = 'float32' | 'float32x2' | 'uint32';
type Allocate = <ValueFormat extends Format>(
  name: string,
  format: ValueFormat,
  length: number,
  values?: Float32Array | Uint32Array
) => GraphDataView<ValueFormat>;

export function getMultipartNearestGeometry(
  side: 'query' | 'feature',
  kind: string
): CPUSpatialJoinGeometry {
  const packed = oracle.geometries[`${side}${kind}` as keyof typeof oracle.geometries];
  return {
    kind: packed.kind as 'points' | 'lines',
    positions: Float32Array.from(packed.positions),
    featureOffsets: Uint32Array.from(packed.featureOffsets),
    sourceIds: Uint32Array.from(packed.sourceIds),
    ...('lineOffsets' in packed ? {lineOffsets: Uint32Array.from(packed.lineOffsets)} : {})
  };
}

export function createMultipartNearestFixture(
  graph: GPUCommandGraph,
  allocate: Allocate,
  caseIndex: number,
  options: {
    mode?: 'nearest' | 'neighbors';
    neighborCount?: number;
    neighborCapacity?: number;
    ties?: 'lowest-row' | 'all';
    radius?: number;
    leafCapacity?: number;
    querySort?: boolean;
  } = {}
) {
  const definition = MULTIPART_NEAREST_CASES[caseIndex];
  const queryCount = definition.ranked.length;
  const featureCount = definition.ranked[0].length;
  const k = options.neighborCount ?? 4;
  const capacity = options.neighborCapacity ?? k;
  const createGeometry = (
    side: 'query' | 'feature',
    kind: string
  ): GPUPackedGeometryPort<GPUMultipartGeometryPort> => {
    const packed = getMultipartNearestGeometry(side, kind);
    const positions = allocate(
      `${side}-positions`,
      'float32x2',
      packed.positions.length / 2,
      packed.positions
    );
    const featureOffsets = allocate(
      `${side}-offsets`,
      'uint32',
      packed.featureOffsets!.length,
      packed.featureOffsets
    );
    const sourceIds = allocate(`${side}-ids`, 'uint32', packed.sourceIds!.length, packed.sourceIds);
    return packed.kind === 'points'
      ? {kind: 'points', positions, featureOffsets, sourceIds}
      : {
          kind: 'lines',
          positions,
          featureOffsets,
          sourceIds,
          lineOffsets: allocate(
            `${side}-line-offsets`,
            'uint32',
            packed.lineOffsets!.length,
            packed.lineOffsets
          )
        };
  };
  const queries = createGeometry('query', definition.queryKind);
  const features = createGeometry('feature', definition.featureKind);
  const results: Record<string, GraphDataView> = {};
  const output = <ValueFormat extends Format>(
    name: string,
    format: ValueFormat,
    length: number
  ) => {
    // Poison every output, including zero-valued status/count ports, so an unwritten port
    // cannot pass merely because storage starts at zero. Coordinate padding must overwrite
    // a FINITE sentinel with NaNs; scalar float outputs start at NaN instead.
    const words = length * (format === 'float32x2' ? 2 : 1);
    const initial =
      format === 'uint32'
        ? new Uint32Array(words).fill(0xdeadbeef)
        : new Float32Array(words).fill(format === 'float32x2' ? 123.5 : NaN);
    const result = allocate(name, format, length, initial);
    results[name] = result;
    return result;
  };
  const status = {
    overflow: output('overflow', 'uint32', 1),
    candidateOverflow: output('candidateOverflow', 'uint32', 1),
    candidateCount: output('candidateCount', 'uint32', 1)
  };
  const candidateTruncated = output('candidateTruncated', 'uint32', queryCount);
  const radiusValue = options.radius ?? (options.mode === 'nearest' ? 2.25 : undefined);
  const radius =
    radiusValue === undefined
      ? undefined
      : allocate('radius', 'float32', 1, Float32Array.of(radiusValue));
  const neighborOutput =
    options.mode === 'nearest'
      ? undefined
      : {
          neighborIds: output('neighborIds', 'uint32', queryCount * capacity),
          neighborRows: output('neighborRows', 'uint32', queryCount * capacity),
          neighborCounts: output('neighborCounts', 'uint32', queryCount),
          neighborDistances: output('neighborDistances', 'float32', queryCount * capacity),
          neighborFootPoints: output('neighborFootPoints', 'float32x2', queryCount * capacity),
          neighborQueryPoints: output('neighborQueryPoints', 'float32x2', queryCount * capacity),
          neighborSegmentIndices: output('neighborSegmentIndices', 'uint32', queryCount * capacity)
        };
  const join = new GPUNearestFeatureJoin({
    queries,
    features,
    radius,
    leafCapacity: options.leafCapacity,
    spatialSort: false,
    querySort: options.querySort ?? false,
    ...(neighborOutput ? {neighborCount: k, neighborCapacity: capacity, ties: options.ties} : {}),
    output: {
      ...(neighborOutput ?? {
        nearestFeatureIds: output('nearestFeatureIds', 'uint32', queryCount),
        nearestDistances: output('nearestDistances', 'float32', queryCount),
        featureCounts: output('featureCounts', 'uint32', featureCount),
        matches: {
          ids: output('matchIds', 'uint32', queryCount),
          status: {
            count: output('matchCount', 'uint32', 1),
            requiredCount: output('matchRequiredCount', 'uint32', 1),
            overflow: output('matchOverflow', 'uint32', 1),
            candidateOverflow: output('matchCandidateOverflow', 'uint32', 1)
          }
        }
      }),
      candidateTruncated,
      status
    }
  });
  const weights = neighborOutput
    ? new GPUNearestFeatureWeights({
        neighborRows: neighborOutput.neighborRows,
        neighborCounts: neighborOutput.neighborCounts,
        neighborDistances: neighborOutput.neighborDistances,
        neighborStatus: status,
        slotCapacity: capacity,
        weightType: 'inverse-distance',
        distanceFloor: 0.5,
        output: {
          offsets: output('weightOffsets', 'uint32', queryCount + 1),
          neighbors: output('weightNeighbors', 'uint32', queryCount * capacity),
          weights: output('weightValues', 'float32', queryCount * capacity),
          distances: output('weightDistances', 'float32', queryCount * capacity),
          status: {
            overflow: output('weightOverflow', 'uint32', 1),
            candidateOverflow: output('weightCandidateOverflow', 'uint32', 1)
          }
        }
      })
    : undefined;
  return {
    join,
    weights,
    results,
    definition,
    capacity,
    radiusValue,
    queryCount,
    featureCount,
    k,
    ties: options.ties
  };
}

export async function assertMultipartNearestFixture(
  fixture: ReturnType<typeof createMultipartNearestFixture>,
  read: (view: GraphDataView) => Promise<number[]>
): Promise<void> {
  const {results, definition, radiusValue, capacity, queryCount, featureCount} = fixture;
  const get = (name: string) => read(results[name]);
  const expectNumbers = (actual: number[], expected: number[]) => {
    expect(actual.length).toBe(expected.length);
    expected.forEach((value, index) => {
      if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
      else {
        expect(Number.isFinite(actual[index]), `finite row ${index}`).toBe(true);
        // Positions are <=1000, exact quarter-grid inputs; 2e-4 covers float32
        // projection, normalized distance ranking, sqrt and inverse distance rounding.
        expect(Math.abs(actual[index] - value), `row ${index}`).toBeLessThanOrEqual(2e-4);
      }
    });
  };
  const ranked = definition.ranked.map(row =>
    row.filter(value => radiusValue === undefined || value.distance <= radiusValue)
  );
  const wanted = ranked.map(row => {
    let count = Math.min(fixture.k, row.length);
    if (fixture.ties === 'all' && count) {
      while (count < row.length && row[count].distance === row[fixture.k - 1].distance) count++;
    }
    return row.slice(0, count);
  });
  const tieOverflow = Number(wanted.some(row => row.length > capacity));
  const featureIds = getMultipartNearestGeometry('feature', definition.featureKind).sourceIds!;
  expect(await get('overflow')).toEqual([tieOverflow]);
  for (const name of ['candidateOverflow', 'candidateCount']) expect(await get(name)).toEqual([0]);
  expect(await get('candidateTruncated')).toEqual(Array(queryCount).fill(0));
  if (fixture.join.mode === 'nearest-feature') {
    const counts = Array<number>(featureCount).fill(0);
    const matches: number[] = [];
    const queryIds = getMultipartNearestGeometry('query', definition.queryKind).sourceIds!;
    ranked.forEach((row, index) => {
      if (row.length) {
        counts[row[0].row]++;
        matches.push(queryIds[index]);
      }
    });
    expect(await get('nearestFeatureIds')).toEqual(
      ranked.map(row => (row.length ? featureIds[row[0].row] : NO_FEATURE))
    );
    expectNumbers(
      await get('nearestDistances'),
      ranked.map(row => row[0]?.distance ?? -1)
    );
    expect(await get('featureCounts')).toEqual(counts);
    expect((await get('matchIds')).slice(0, matches.length)).toEqual(matches);
    expect(await get('matchCount')).toEqual([matches.length]);
    expect(await get('matchRequiredCount')).toEqual([matches.length]);
    expect(await get('matchOverflow')).toEqual([0]);
    expect(await get('matchCandidateOverflow')).toEqual([0]);
    return;
  }
  const kept = wanted.map(row => row.slice(0, capacity));
  const slots = kept.flatMap(row => Array.from({length: capacity}, (_, slot) => row[slot]));
  expect(await get('neighborCounts')).toEqual(kept.map(row => row.length));
  expect(await get('neighborRows')).toEqual(slots.map(value => value?.row ?? NO_FEATURE));
  expect(await get('neighborIds')).toEqual(
    slots.map(value => (value ? featureIds[value.row] : NO_FEATURE))
  );
  expectNumbers(
    await get('neighborDistances'),
    slots.map(value => value?.distance ?? -1)
  );
  expectNumbers(
    await get('neighborFootPoints'),
    slots.flatMap(value => value?.featurePoint ?? [NaN, NaN])
  );
  expectNumbers(
    await get('neighborQueryPoints'),
    slots.flatMap(value => value?.queryPoint ?? [NaN, NaN])
  );
  expect(await get('neighborSegmentIndices')).toEqual(
    slots.map(value => value?.segment ?? NO_FEATURE)
  );
  const sortedRows = kept.map(row => [...row].sort((first, second) => first.row - second.row));
  const offsets = [0];
  for (const row of sortedRows) offsets.push(offsets[offsets.length - 1] + row.length);
  const entries = sortedRows.flat();
  expect(await get('weightOffsets')).toEqual(offsets);
  expect((await get('weightNeighbors')).slice(0, entries.length)).toEqual(
    entries.map(value => value.row)
  );
  expectNumbers(
    (await get('weightDistances')).slice(0, entries.length),
    entries.map(value => value.distance)
  );
  expectNumbers(
    (await get('weightValues')).slice(0, entries.length),
    entries.map(value => 1 / Math.max(value.distance, 0.5))
  );
  expect(await get('weightOverflow')).toEqual([0]);
  expect(await get('weightCandidateOverflow')).toEqual([tieOverflow]);
}
