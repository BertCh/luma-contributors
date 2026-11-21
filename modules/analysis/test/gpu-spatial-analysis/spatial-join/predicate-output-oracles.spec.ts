// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {createSpatialPredicateJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {
  getPredicateOracleOperationProps,
  getPredicateOracleOperationInputs
} from './predicate-output-oracles-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph,
  type OracleInput
} from '../spatial-clustering/output-oracle-test-utils';

function getInputs(): Record<string, OracleInput> {
  // Shapely oracle command and enumeration in predicate-output-oracles.oracle.py.
  // Two inclusive rectangles cover each of rows 0,1; rows 2,3 are outside both.
  return {
    points: {format: 'float32x2', length: 4, values: Float32Array.of(1, 1, 2, 1, 5, 5, 6, 6)},
    positions: {
      format: 'float32x2',
      length: 8,
      values: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 1, 0, 3, 0, 3, 2, 1, 2)
    },
    features: {format: 'uint32', length: 3, values: Uint32Array.of(0, 1, 2)},
    polygons: {format: 'uint32', length: 3, values: Uint32Array.of(0, 1, 2)},
    rings: {format: 'uint32', length: 3, values: Uint32Array.of(0, 4, 8)}
  };
}

it('predicate anti join publishes exact bounded unmatched diagnostics independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const candidateCapacity of [1, 4, 8]) {
    for (const optionalPort of ['requiredCount', 'absent']) {
      const publishCandidateOverflow = candidateCapacity < 8;
      const harness = createPoisonedOracleGraph(device, getInputs(), {
        ids: {format: 'uint32', length: 1},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        ...(optionalPort === 'requiredCount'
          ? {requiredCount: {format: 'uint32' as const, length: 1}}
          : {}),
        ...(publishCandidateOverflow
          ? {candidateOverflow: {format: 'uint32' as const, length: 1}}
          : {})
      });
      const view = harness.getView;
      harness.graph.add(
        new GPUSpatialPredicateJoin({
          left: {kind: 'points', positions: view('points')},
          right: {
            kind: 'polygons',
            positions: view('positions'),
            featureOffsets: view('features'),
            polygonOffsets: view('polygons'),
            ringOffsets: view('rings')
          },
          predicate: 'intersects',
          how: 'anti',
          candidateCapacity,
          output: {
            unmatched: {
              ids: view('ids'),
              status: {
                count: view('count'),
                overflow: view('overflow'),
                ...(optionalPort === 'requiredCount'
                  ? {requiredCount: view<'uint32'>('requiredCount')}
                  : {}),
                ...(publishCandidateOverflow
                  ? {candidateOverflow: view<'uint32'>('candidateOverflow')}
                  : {})
              }
            }
          }
        })
      );
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        // Candidate-limited rows 0,1 are not declared unmatched; rows 2,3 have zero
        // candidates and are certified unmatched even with a one-slot candidate table.
        expect(result['ids']).toEqual([2]);
        expect(result['count']).toEqual([1]);
        expect(result['overflow']).toEqual([1]);
        if (optionalPort === 'requiredCount') expect(result['requiredCount']).toEqual([2]);
        if (publishCandidateOverflow)
          expect(result['candidateOverflow']).toEqual([candidateCapacity === 1 ? 1 : 0]);
      }
    }
  }
});

it('predicate CSR uncertainty reports one uncertifiable line pair with other diagnostics omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Nonzero product exponents span over 200 bits: the exact orientation fallback
  // cannot certify this pair and reports one uncertain candidate. This is the
  // existing multipart identity-grouping fixture, routed to the CSR destination.
  const large = 2 ** 120,
    delta = 2 ** 98,
    small = 2 ** -100;
  for (const uncertaintyPresent of [true, false]) {
    const harness = createPoisonedOracleGraph(
      device,
      {
        left: {
          format: 'float32x2',
          length: 2,
          values: Float32Array.of(large / 4, large / 4 + delta, large / 2, large / 2 + delta)
        },
        right: {
          format: 'float32x2',
          length: 2,
          values: Float32Array.of(small, small, large, large)
        },
        leftLines: {format: 'uint32', length: 2, values: Uint32Array.of(0, 2)},
        rightLines: {format: 'uint32', length: 2, values: Uint32Array.of(0, 2)},
        leftFeatures: {format: 'uint32', length: 2, values: Uint32Array.of(0, 1)}
      },
      {
        offsets: {format: 'uint32', length: 2},
        neighbors: {format: 'uint32', length: 1},
        weights: {format: 'float32', length: 1},
        overflow: {format: 'uint32', length: 1},
        ...(uncertaintyPresent ? {uncertainCount: {format: 'uint32' as const, length: 1}} : {})
      }
    );
    const view = harness.getView;
    harness.graph.add(
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'lines',
          positions: view('left'),
          lineOffsets: view('leftLines'),
          featureOffsets: view('leftFeatures')
        },
        right: {kind: 'lines', positions: view('right'), lineOffsets: view('rightLines')},
        predicate: 'intersects',
        candidateCapacity: 8,
        output: {
          weights: {
            offsets: view('offsets'),
            neighbors: view('neighbors'),
            weights: view('weights'),
            ...(uncertaintyPresent
              ? {status: {uncertainCount: view<'uint32'>('uncertainCount')}}
              : {})
          },
          status: {overflow: view('overflow')}
        }
      })
    );
    for (const result of await runPoisonedOracleGraph(harness, device)) {
      expect(result['offsets']).toEqual([0, 1]);
      expect(result['neighbors']).toEqual([0]);
      expect(result['weights']).toEqual([1]);
      expect(result['overflow']).toEqual([0]);
      if (uncertaintyPresent) expect(result['uncertainCount']).toEqual([1]);
    }
  }
});

it('predicate weights preserve every supported diagnostic with unrelated destinations omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const field of [
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'absent'
  ] as const) {
    const harness = createPoisonedOracleGraph(device, getInputs(), {
      offsets: {format: 'uint32', length: 5},
      neighbors: {format: 'uint32', length: 4},
      weights: {format: 'float32', length: 4},
      overallOverflow: {format: 'uint32', length: 1},
      ...(field === 'absent' ? {} : {diagnostic: {format: 'uint32' as const, length: 1}})
    });
    const view = harness.getView;
    harness.graph.add(
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: view('points')},
        right: {
          kind: 'polygons',
          positions: view('positions'),
          featureOffsets: view('features'),
          polygonOffsets: view('polygons'),
          ringOffsets: view('rings')
        },
        predicate: 'intersects',
        candidateCapacity: 4,
        output: {
          weights: {
            offsets: view('offsets'),
            neighbors: view('neighbors'),
            weights: view('weights'),
            ...(field === 'absent' ? {} : {status: {[field]: view<'uint32'>('diagnostic')}})
          },
          status: {overflow: view('overallOverflow')}
        }
      })
    );
    for (const result of await runPoisonedOracleGraph(harness, device)) {
      // Shapely four-pair enumeration in the adjacent oracle: no capacity loss and
      // integer coordinates with certifiable signs. This also pins nonzero counts.
      expect(result['offsets']).toEqual([0, 2, 4, 4, 4]);
      expect(result['neighbors']).toEqual([0, 1, 0, 1]);
      expect(result['weights']).toEqual([1, 1, 1, 1]);
      expect(result['overallOverflow']).toEqual([0]);
      if (field !== 'absent')
        expect(result['diagnostic']).toEqual([
          field === 'count' || field === 'requiredCount' ? 4 : 0
        ]);
    }
  }
});

it('predicate GPU operation publishes the same independent Shapely pair and completeness literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPoisonedOracleGraph(device, getPredicateOracleOperationInputs(), {
    leftRows: {format: 'uint32', length: 4},
    rightRows: {format: 'uint32', length: 4},
    count: {format: 'uint32', length: 1},
    requiredCount: {format: 'uint32', length: 1},
    overflow: {format: 'uint32', length: 1},
    candidateOverflow: {format: 'uint32', length: 1},
    candidateTruncated: {format: 'uint32', length: 4}
  });
  harness.graph.add(
    createSpatialPredicateJoinOperation(getPredicateOracleOperationProps()).createContributor(
      harness.getView
    )
  );
  for (const result of await runPoisonedOracleGraph(harness, device)) {
    expect(result['leftRows']).toEqual([0, 0, 1, 1]);
    expect(result['rightRows']).toEqual([0, 1, 0, 1]);
    expect(result['count']).toEqual([4]);
    expect(result['requiredCount']).toEqual([4]);
    expect(result['overflow']).toEqual([0]);
    expect(result['candidateOverflow']).toEqual([0]);
    expect(result['candidateTruncated']).toEqual([0, 0, 0, 0]);
  }
});

it('predicate CSR uncertainty and candidate count are literal Shapely expectations with optional ports isolated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optionalPort of ['uncertainCount', 'candidateCount', 'absent']) {
    const harness = createPoisonedOracleGraph(device, getInputs(), {
      offsets: {format: 'uint32', length: 5},
      neighbors: {format: 'uint32', length: 4},
      weights: {format: 'float32', length: 4},
      overflow: {format: 'uint32', length: 1},
      ...(optionalPort === 'uncertainCount'
        ? {uncertainCount: {format: 'uint32' as const, length: 1}}
        : {}),
      ...(optionalPort === 'candidateCount'
        ? {candidateCount: {format: 'uint32' as const, length: 1}}
        : {})
    });
    const view = harness.getView;
    harness.graph.add(
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: view('points')},
        right: {
          kind: 'polygons',
          positions: view('positions'),
          featureOffsets: view('features'),
          polygonOffsets: view('polygons'),
          ringOffsets: view('rings')
        },
        predicate: 'intersects',
        candidateCapacity: 4,
        output: {
          weights: {
            offsets: view('offsets'),
            neighbors: view('neighbors'),
            weights: view('weights'),
            ...(optionalPort === 'uncertainCount'
              ? {status: {uncertainCount: view<'uint32'>('uncertainCount')}}
              : {})
          },
          status: {
            overflow: view('overflow'),
            ...(optionalPort === 'candidateCount'
              ? {candidateCount: view<'uint32'>('candidateCount')}
              : {})
          }
        }
      })
    );
    for (const result of await runPoisonedOracleGraph(harness, device)) {
      expect(result['offsets']).toEqual([0, 2, 4, 4, 4]);
      expect(result['neighbors']).toEqual([0, 1, 0, 1]);
      expect(result['weights']).toEqual([1, 1, 1, 1]);
      expect(result['overflow']).toEqual([0]);
      // Integer coordinates have exact orientation signs, including boundary row 1.
      if (optionalPort === 'uncertainCount') expect(result['uncertainCount']).toEqual([0]);
      if (optionalPort === 'candidateCount') expect(result['candidateCount']).toEqual([4]);
    }
  }
});
