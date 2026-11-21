// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {GPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createDecreasingOffsetPairScene,
  createDecreasingOffsetGeometryScene,
  createPairScene,
  createPolygonScene,
  createRectangleScene,
  createReversedRangeOrderGeometryScene,
  createStaggeredPairScene,
  createWordScene,
  createZeroRowScene,
  getSceneInputs,
  getSceneProps,
  type RestrictionScene
} from './trajectory-restriction-operation-cases';

function createScenes(): [string, RestrictionScene][] {
  const polygon = createPolygonScene();
  const pairs = createPairScene();
  const staticPairs = createPairScene();
  staticPairs.input.pairs = undefined;
  staticPairs.options.pairwiseTracks = [
    [0, 1],
    [1, 1],
    [0, 2]
  ];
  const emptyTrack = createRectangleScene();
  emptyTrack.input.trackOffsets = Uint32Array.of(0, 0, 3, 5);
  return [
    ['closed rectangle, moving and stationary tracks', createRectangleScene()],
    ['polygon hole, collinear edge, stable per-track union', polygon],
    [
      'open polygon boundary',
      {...polygon, options: {...polygon.options, geometryBoundary: 'open'}}
    ],
    [
      'slot and compact overflow',
      {
        ...polygon,
        options: {
          ...polygon.options,
          intervalsPerSegment: 1,
          compactCapacity: 1,
          mergeIntervals: false
        }
      }
    ],
    ['dynamic pairs with self-overlap, inactive and invalid rows', pairs],
    ['static pairs', staticPairs],
    ['staggered multi-step pair merge in both directions', createStaggeredPairScene()],
    ['pair overflow', {...pairs, options: {...pairs.options, intervalsPerPair: 1}}],
    [
      'open distance and time boundaries',
      {...pairs, options: {...pairs.options, geometryBoundary: 'open', timeBoundary: 'open'}}
    ],
    ['Int64 origin and strict gap', createWordScene()],
    ['empty track among ordinary tracks', emptyTrack],
    ['decreasing offsets define an empty pair range', createDecreasingOffsetPairScene()],
    ['decreasing offsets define an empty geometry range', createDecreasingOffsetGeometryScene()],
    [
      'compaction follows source order with reversed track ranges',
      createReversedRangeOrderGeometryScene()
    ],
    [
      'compaction preserves the kernel track cursor for reversed ranges',
      createReversedRangeOrderGeometryScene(true)
    ],
    ['zero observation rows with one empty track', createZeroRowScene()],
    ['zero observation rows and zero tracks', createZeroRowScene(0)],
    [
      'zero compact capacity with required intervals',
      {...polygon, options: {...polygon.options, compactCapacity: 0}}
    ]
  ];
}

for (const [name, scene] of createScenes()) {
  it(`trajectory restriction GPU and CPU graph values agree: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getSceneProps(scene);
    const outputs = Object.values(props.output).filter(
      (value): value is string => value !== undefined
    );
    const results = [];
    for (const Operation of [
      GPUTrajectoryRestrictionOperation,
      CPUTrajectoryRestrictionOperation
    ]) {
      // runOperationGraph destroys operations; each execution owns a fresh instance.
      results.push(
        await runOperationGraph(device, {
          inputs: getSceneInputs(scene),
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    for (const value of outputs) {
      const actual = results[0][value];
      const expected = results[1][value];
      expect(actual.length, value).toBe(expected.length);
      if (expected instanceof Uint32Array) {
        expect(Array.from(actual), value).toEqual(Array.from(expected));
      } else {
        for (let index = 0; index < expected.length; index++) {
          if (Number.isNaN(expected[index])) expect(actual[index], `${value}[${index}]`).toBeNaN();
          // Allow 2e-6 relative error for GPU division/interpolation, and 2e-6 absolute at zero.
          // Numeric distance to zero also avoids comparing the sign bit of GPU zero.
          else if (expected[index] === 0)
            expect(Math.abs(actual[index]), `${value}[${index}]`).toBeLessThanOrEqual(2e-6);
          else
            expect(
              Math.abs(actual[index] - expected[index]),
              `${value}[${index}]`
            ).toBeLessThanOrEqual(2e-6 * Math.abs(expected[index]));
        }
      }
    }
    // Teeth: ordinary scenes produce intervals; empty scenes have no interval flags.
    const emptySupport =
      name.startsWith('zero observation') ||
      name === 'decreasing offsets define an empty pair range';
    expect(results[1]['status'].some(flags => (flags & 3) !== 0)).toBe(!emptySupport);
    if (name === 'decreasing offsets define an empty pair range') {
      expect(Array.from(results[0]['relationships'])).toEqual([0xffffffff, 0xffffffff, 0, 0]);
      expect(Array.from(results[0]['status'])).toEqual([0]);
      expect(results[0]['intervals'].every(Number.isNaN)).toBe(true);
      expect(results[0]['positions'].every(Number.isNaN)).toBe(true);
    }
    if (name.startsWith('zero observation')) expect(results[0]['intervals']).toHaveLength(0);
    if (name === 'zero compact capacity with required intervals') {
      expect(results[0]['compactIntervals']).toHaveLength(0);
      expect(results[0]['compactPositions']).toHaveLength(0);
      expect(results[0]['compactSources']).toHaveLength(0);
      expect(results[0]['compactStatus']).toHaveLength(0);
      expect(Array.from(results[0]['compactSummary'])).toEqual([3, 0, 3]);
    }
    if (name.includes('overflow'))
      expect(results[1]['status'].some(flags => (flags & 256) !== 0)).toBe(true);
  });
}
