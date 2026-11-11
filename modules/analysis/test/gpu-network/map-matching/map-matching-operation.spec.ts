// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {GPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createCutoffMatchingFixture,
  createMatchingFixture,
  getGeographicSnapComparison,
  getMatchingGraphInputs
} from './map-matching-operation-fixture';

const OUTPUTS = [
  'matchedEdges',
  'matchedFractions',
  'matchedOffsets',
  'snapDistances',
  'snappedPositions',
  'breaks',
  'trackLogLikelihoods',
  'matchedCount',
  'breakCount',
  'overflow',
  'routeOverflow',
  'traversedEdges',
  'traversedCounts',
  'traversedOverflow'
];

for (const scene of [
  'competing-roads',
  'filtered-overflow',
  'unmatched',
  'great-circle',
  'route-table',
  'table-overflow',
  'score-cutoff',
  'capacity-one',
  'unused-capacity'
] as const) {
  it(`map matching operations agree on every defined output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {input, props} =
      scene === 'score-cutoff' ? createCutoffMatchingFixture() : createMatchingFixture();
    if (scene === 'capacity-one') props.traversedEdgeCapacity = 1;
    if (scene === 'unused-capacity') {
      props.output = {matchedEdges: 'matchedEdges'};
      props.traversedEdgeCapacity = 0;
    }
    if (scene === 'filtered-overflow') {
      input.positions = new Float32Array([2, 0.5, 2.5, 0.5, 28, 0.5, 15, 50, 5, 0.5]);
      input.trackOffsets = new Uint32Array([0, 5, 5]);
      props.minimumFixSpacingSigmas = 1;
      props.traversedEdgeCapacity = 2;
    }
    if (scene === 'unmatched')
      input.positions = new Float32Array([2, 50, 5, 50, 28, 50, 2, 50, 28, 50]);
    if (scene === 'great-circle') {
      props.spatialContext = {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'kilometers'
      };
      props.nodeCount = 4;
      props.edgeCount = 3;
      props.pointCount = 4;
      props.trackCount = 2;
      props.cellSize = 0.02;
      props.bounds = {minimum: [179.95, -0.02], maximum: [180.05, 0.02]};
      props.candidateCount = 2;
      input.topology = {
        nodePositions: new Float32Array([179.97, 0, 179.99, 0, -179.99, 0, -179.97, 0]),
        offsets: new Uint32Array([0, 1, 2, 3, 3]),
        neighbors: new Uint32Array([1, 2, 3]),
        weights: new Float32Array([1, 100, 2])
      };
      input.positions = new Float32Array([179.975, 0.001, -179.975, 0.002, -180, 0, 180, 0]);
      input.trackOffsets = new Uint32Array([0, 2, 4]);
      input.parameters = getGPUMapMatchingParameterValues({
        sigma: 0.2,
        beta: 0.5,
        searchRadius: 0.4,
        routeFactor: 2,
        routeSlack: 1
      });
    }
    if (scene === 'route-table' || scene === 'table-overflow') {
      const offsets = [0];
      const targets: number[] = [];
      const costs: number[] = [];
      for (let source = 0; source < 8; source++) {
        for (let target = source; target < (source < 4 ? 4 : 8); target++) {
          targets.push(target);
          costs.push(10 * (target - source));
        }
        offsets.push(targets.length);
      }
      input.routeTable = {
        offsets: Uint32Array.from(offsets),
        targets: Uint32Array.from(targets),
        costs: Float32Array.from(costs),
        costLimit: new Float32Array([100]),
        overflow: new Uint32Array([scene === 'table-overflow' ? 1 : 0])
      };
      props.routeTable = {
        entryCount: targets.length,
        offsets: 'tableOffsets',
        targets: 'tableTargets',
        costs: 'tableCosts',
        costLimit: 'tableCostLimit',
        status: {overflow: 'tableOverflow'}
      };
    }
    const outputNames = scene === 'unused-capacity' ? ['matchedEdges'] : OUTPUTS;
    const results = [];
    for (const Operation of [GPUMapMatchingOperation, CPUMapMatchingOperation]) {
      // runOperationGraph owns/destroys operations: each invocation gets a fresh instance.
      results.push(
        await runOperationGraph(device, {
          inputs: getMatchingGraphInputs(input),
          operations: [new Operation(props)],
          outputs: outputNames
        })
      );
    }
    const [gpu, cpu] = results;
    for (const port of outputNames) {
      expect(gpu[port].length, port).toBe(cpu[port].length);
      if (port === 'traversedEdges') {
        // The contributor deliberately leaves unused capacity untouched, so compare every
        // defined slot, including separators, and the full count/status arrays below.
        const capacity = props.traversedEdgeCapacity!;
        for (let track = 0; track < props.trackCount; track++) {
          const begin = track * capacity;
          const end = begin + Math.min(cpu['traversedCounts'][track], capacity);
          expect(Array.from(gpu[port].subarray(begin, end)), port).toEqual(
            Array.from(cpu[port].subarray(begin, end))
          );
        }
      } else if (port === 'snappedPositions' && props.spatialContext?.metric === 'great-circle') {
        for (let point = 0; point < props.pointCount; point++) {
          const comparison = getGeographicSnapComparison(
            [gpu[port][point * 2], gpu[port][point * 2 + 1]],
            [cpu[port][point * 2], cpu[port][point * 2 + 1]],
            input.topology,
            props.spatialContext
          );
          expect(
            comparison.distance,
            `${port}[${point}] in context length units`
          ).toBeLessThanOrEqual(comparison.tolerance);
        }
      } else if (cpu[port] instanceof Uint32Array) {
        expect(Array.from(gpu[port]), port).toEqual(Array.from(cpu[port]));
      } else {
        for (let row = 0; row < cpu[port].length; row++) {
          const expected = cpu[port][row];
          // Distances/offsets/fractions use relative tolerance even when small; a zero twin
          // cannot pass a positive result. Coordinates and log likelihoods get an absolute
          // allowance near zero for float32 projection and transcendental arithmetic.
          const positiveQuantity = ['matchedFractions', 'matchedOffsets', 'snapDistances'].includes(
            port
          );
          const tolerance =
            positiveQuantity && expected !== 0
              ? 5e-5 * Math.abs(expected)
              : Math.max(2e-5, 5e-5 * Math.abs(expected));
          expect(Math.abs(gpu[port][row] - expected), `${port}[${row}]`).toBeLessThanOrEqual(
            tolerance
          );
        }
      }
    }
    if (scene === 'competing-roads' || scene === 'route-table' || scene === 'table-overflow') {
      expect(Array.from(cpu['matchedEdges'])).toEqual([0, 0, 2, 3, 5]);
      expect(Array.from(cpu['traversedCounts'])).toEqual([3, 3]);
    }
    if (scene === 'filtered-overflow') {
      expect(Array.from(cpu['matchedEdges'])).toEqual([
        0,
        GPU_MAP_MATCHING_NONE,
        2,
        GPU_MAP_MATCHING_NONE,
        0
      ]);
      expect(Array.from(cpu['traversedCounts'])).toEqual([5, 0]);
      expect(cpu['traversedOverflow'][0]).toBe(1);
    }
    if (scene === 'unmatched') {
      expect(cpu['matchedCount'][0]).toBe(0);
      expect(Array.from(cpu['breaks'])).toEqual([0, 1, 1, 0, 1]);
    }
    // Reviewer predictions are hypotheses until this runs on the device: assert the hand-derived
    // values independently for EACH operation, in addition to comparing their complete outputs.
    for (const rows of results) {
      if (scene === 'great-circle') {
        // Every road segment is equatorial, so the closest position has the raw fix longitude
        // and latitude zero. Check this closed form separately from GPU/CPU agreement.
        for (let point = 0; point < props.pointCount; point++) {
          const comparison = getGeographicSnapComparison(
            [rows['snappedPositions'][point * 2], rows['snappedPositions'][point * 2 + 1]],
            [input.positions[point * 2], 0],
            input.topology,
            props.spatialContext!
          );
          expect(comparison.distance).toBeLessThanOrEqual(comparison.tolerance);
        }
      }
      if (scene === 'score-cutoff') {
        expect(Array.from(rows['matchedEdges'])).toEqual([
          GPU_MAP_MATCHING_NONE,
          GPU_MAP_MATCHING_NONE
        ]);
        expect(Array.from(rows['breaks'])).toEqual([0, 1]);
        expect(Array.from(rows['trackLogLikelihoods'])).toEqual([0]);
      }
      if (scene === 'capacity-one') {
        expect(Array.from(rows['traversedEdges'])).toEqual([0, 3]);
        expect(Array.from(rows['traversedCounts'])).toEqual([3, 3]);
        expect(rows['traversedOverflow'][0]).toBe(1);
      }
      if (scene === 'unused-capacity')
        expect(Array.from(rows['matchedEdges'])).toEqual([0, 0, 2, 3, 5]);
    }
  });
}
