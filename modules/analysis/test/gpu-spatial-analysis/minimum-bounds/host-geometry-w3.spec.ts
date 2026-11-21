// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {GPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {CPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('UGE-04 keeps source bounds beside a distant group and diagnoses the hull lattice on every form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const distant of [false, true]) {
    const positions = [0, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, ...(distant ? [10000000, 0] : [])];
    const labels = [0, 0, 0, 0, ...(distant ? [1] : [])];
    const groupCount = distant ? 2 : 1;
    const fixture = createOracleFixture(device);
    const sourceBounds = fixture.output('float32x4', groupCount);
    const hullSizes = fixture.output('uint32', groupCount);
    const diameters = fixture.output('float32', groupCount);
    const precisionLimited = fixture.output('uint32', groupCount);
    const overflow = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUMinimumBounds({
        positions: fixture.input('float32x2', positions),
        labels: fixture.input('uint32', labels),
        groupCount,
        output: {
          sourceBounds: sourceBounds.view,
          hullSizes: hullSizes.view,
          diameters: diameters.view,
          precisionLimited: precisionLimited.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const raw = {
        bounds: Float32Array.from(await sourceBounds.read()),
        hullSizes: Uint32Array.from(await hullSizes.read()),
        diameters: Float32Array.from(await diameters.read()),
        precisionLimited: Uint32Array.from(await precisionLimited.read()),
        overflow: Uint32Array.from(await overflow.read())
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [raw];
      for (const Operation of [GPUMinimumBoundsOperation, CPUMinimumBoundsOperation])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: new Float32Array(positions)},
              labels: {format: 'uint32', values: new Uint32Array(labels)}
            },
            operations: [
              new Operation({
                positions: 'positions',
                labels: 'labels',
                rowCount: labels.length,
                groupCount,
                output: {
                  sourceBounds: 'bounds',
                  hullSizes: 'hullSizes',
                  diameters: 'diameters',
                  precisionLimited: 'precisionLimited',
                  status: {overflow: 'overflow'}
                }
              })
            ],
            outputs: ['bounds', 'hullSizes', 'diameters', 'precisionLimited', 'overflow']
          })
        );
      for (const result of results) {
        // Componentwise min/max of every stored source corner, including the distant point.
        const expectedBounds = [
          0,
          0,
          Math.fround(0.01),
          Math.fround(0.01),
          ...(distant ? [10000000, 0, 10000000, 0] : [])
        ];
        expect(result['bounds']).toHaveLength(groupCount * 4);
        expect(Array.from(result['bounds'])).toEqual(expectedBounds);
        expect(Array.from(result['hullSizes'])).toEqual([distant ? 1 : 4, ...(distant ? [1] : [])]);
        expect(Array.from(result['diameters'])).toEqual([
          distant ? 0 : Math.fround(Math.SQRT2 * Math.fround(0.01)),
          ...(distant ? [0] : [])
        ]);
        expect(Array.from(result['precisionLimited'])).toEqual([
          distant ? 1 : 0,
          ...(distant ? [0] : [])
        ]);
        expect(Array.from(result['overflow'])).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('CPL-14 publishes supplied label completion with zero geometric overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const complete of [0, 1]) {
    for (const Operation of [GPUMinimumBoundsOperation, CPUMinimumBoundsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1])},
          labels: {format: 'uint32', values: new Uint32Array([1, 1, 3, 3])},
          labelsComplete: {format: 'uint32', values: new Uint32Array([complete])}
        },
        operations: [
          new Operation({
            positions: 'positions',
            labels: 'labels',
            labelsComplete: 'labelsComplete',
            rowCount: 4,
            groupCount: 4,
            output: {
              sourceBounds: 'bounds',
              labelsComplete: 'partitionComplete',
              status: {overflow: 'overflow'}
            }
          })
        ],
        outputs: ['bounds', 'partitionComplete', 'overflow']
      });
      expect(Array.from(result['partitionComplete'])).toEqual([complete]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['bounds'])).toEqual([
        NaN,
        NaN,
        NaN,
        NaN,
        0,
        0,
        1,
        0,
        NaN,
        NaN,
        NaN,
        NaN,
        0,
        1,
        1,
        1
      ]);
    }
  }
});

it('S1 preserves finite source bounds when the hull cap suppresses shapes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1])},
      labels: {format: 'uint32', values: new Uint32Array(4)}
    },
    operations: [
      new GPUMinimumBoundsOperation({
        positions: 'positions',
        labels: 'labels',
        rowCount: 4,
        groupCount: 1,
        maximumHullVertices: 3,
        output: {
          sourceBounds: 'bounds',
          diameters: 'diameters',
          circles: 'circles',
          hullSizes: 'hullSizes',
          status: {overflow: 'overflow', groupOverflow: 'groupOverflow'}
        }
      })
    ],
    outputs: ['bounds', 'diameters', 'circles', 'hullSizes', 'overflow', 'groupOverflow']
  });
  // All four square corners are extreme. The cap 3 drops the hull, never its source envelope.
  expect(Array.from(result['bounds'])).toEqual([0, 0, 1, 1]);
  expect(Array.from(result['hullSizes'])).toEqual([4]);
  expect(Array.from(result['diameters'])).toEqual([NaN]);
  expect(Array.from(result['circles'])).toEqual([NaN, NaN, NaN]);
  expect(Array.from(result['overflow'])).toEqual([1]);
  expect(Array.from(result['groupOverflow'])).toEqual([1]);
});
