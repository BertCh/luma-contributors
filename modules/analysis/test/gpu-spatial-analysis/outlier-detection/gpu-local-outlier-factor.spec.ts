// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  GPULocalOutlierFactor,
  getGPULocalOutlierFactorParameterValues
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {getGPUNeighborSearchParameterValues} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  DUPLICATES_FIXTURE,
  RANDOM_FIXTURE,
  type LocalOutlierFactorFixture
} from './outlier-detection-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function expectClose(actual: number[], expected: number[], label: string, relative: number): void {
  expectNumericArrayClose(actual, expected, {
    absoluteTolerance: 1e-6,
    relativeTolerance: relative,
    label
  });
}

async function runLof(
  device: Device,
  fixture: LocalOutlierFactorFixture,
  threshold: number | undefined,
  useNeighborSearch: boolean
) {
  const rows = fixture.points.length / 2;
  const rig = new WeightsRig(device);
  const parameters = rig.input(getGPULocalOutlierFactorParameterValues({threshold}), 'float32', 4);
  const kDistance = rig.output('float32', rows);
  const density = rig.output('float32', rows);
  const lof = rig.output('float32', rows);
  const outlier = rig.output('uint32', rows);
  const outlierCount = rig.output('uint32', 1);
  const producers = [];
  let neighbors;
  if (useNeighborSearch) {
    const table = rig.weightsOutput(rows, rows * fixture.k, true);
    const overflow = rig.output('uint32', 1);
    producers.push(
      new GPUNeighborSearch({
        mode: 'knn',
        neighborCount: fixture.k,
        positions: rig.input(new Float32Array(fixture.points), 'float32x2', rows),
        parameters: rig.input(
          getGPUNeighborSearchParameterValues({bounds: [-100, -100, 300, 300]}),
          'float32',
          12
        ),
        gridSize: [4, 4],
        output: {...table.spatialWeights, status: {overflow: overflow.view}}
      })
    );
    neighbors = table.spatialWeights;
  } else {
    neighbors = rig.uploadWeights({
      offsets: fixture.offsets,
      neighbors: fixture.neighbors,
      weights: fixture.neighbors.map(() => 1),
      distances: fixture.distances
    });
  }
  producers.push(
    new GPULocalOutlierFactor({
      neighbors,
      parameters,
      output: {
        kDistance: kDistance.view,
        localReachabilityDensity: density.view,
        lof: lof.view,
        outlier: outlier.view,
        outlierCount: outlierCount.view
      }
    })
  );
  rig.run(...producers);
  const result = {
    kDistance: await readFloat32(kDistance.buffer, rows),
    lof: await readFloat32(lof.buffer, rows),
    outlier: await readUint32(outlier.buffer, rows),
    count: (await readUint32(outlierCount.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPULocalOutlierFactor matches scikit-learn on tie-free random points', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const useNeighborSearch of [false, true]) {
    const label = useNeighborSearch ? 'neighbor-search' : 'uploaded';
    const start = performance.now();
    const result = await runLof(device, RANDOM_FIXTURE, 1.5, useNeighborSearch);
    console.log(`lof ${label}: ${(performance.now() - start).toFixed(1)} ms`);
    expectClose(result.kDistance, RANDOM_FIXTURE.kDistance, `${label} kDistance`, 2e-6);
    expectClose(result.lof, RANDOM_FIXTURE.lof, `${label} lof`, 1e-4);
    const expectedMask = RANDOM_FIXTURE.lof.map((value): number => (value > 1.5 ? 1 : 0));
    expect(expectedMask.some(flag => flag === 1)).toBe(true);
    expect(result.outlier, `${label} mask`).toEqual(expectedMask);
    expect(result.count).toBe(expectedMask.reduce((sum, flag) => sum + flag, 0));
  }
});

it('GPULocalOutlierFactor keeps coincident points finite through the density floor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runLof(device, DUPLICATES_FIXTURE, 2, false);
  expectClose(result.lof, DUPLICATES_FIXTURE.lof, 'duplicates lof', 1e-3);
  for (let row = 0; row < 4; row++) {
    expect(result.lof[row]).toBeCloseTo(1, 5);
  }
  expect(result.lof.every(Number.isFinite)).toBe(true);
});

it('GPULocalOutlierFactor flags nothing without a threshold (there is no default cut)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runLof(device, RANDOM_FIXTURE, undefined, false);
  expectClose(result.lof, RANDOM_FIXTURE.lof, 'no-threshold lof', 1e-4);
  expect(RANDOM_FIXTURE.lof.some(value => value > 1.5)).toBe(true);
  expect(result.outlier.every(flag => flag === 0)).toBe(true);
  expect(result.count).toBe(0);
});

it('WGS-13 bounds malformed offsets and preserves weights-loss suppression and clean recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const offsets = rig.input(Uint32Array.of(0, 3, 3), 'uint32', 3);
    const loss = rig.input(Uint32Array.of(1), 'uint32', 1);
    const neighbors = {
      offsets,
      neighbors: rig.input(Uint32Array.of(1), 'uint32', 1),
      weights: rig.input(Float32Array.of(1), 'float32', 1),
      distances: rig.input(Float32Array.of(2), 'float32', 1),
      status: {candidateOverflow: loss}
    };
    const kDistance = rig.output('float32', 2);
    const density = rig.output('float32', 2);
    const lof = rig.output('float32', 2);
    rig.graph.add(
      new GPULocalOutlierFactor({
        neighbors,
        parameters: rig.input(
          getGPULocalOutlierFactorParameterValues({densityFloor: 1}),
          'float32',
          4
        ),
        output: {kDistance: kDistance.view, localReachabilityDensity: density.view, lof: lof.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const output of [kDistance, density, lof])
        expect((await readFloat32(output.buffer, 2)).every(Number.isNaN)).toBe(true);
      rig.bufferOf(loss).write(Uint32Array.of(0));
      // Even before topology repair the allocated domain is just one slot.
      submitGraph(device, compiled, undefined);
      const malformed = await readFloat32(kDistance.buffer, 2);
      expect(malformed).toEqual([2, 0]);
      rig.bufferOf(offsets).write(Uint32Array.of(0, 1, 1));
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(kDistance.buffer, 2)).toEqual([2, 0]);
      // Row 0 reaches island 1 at distance 2: density=1/(2+1)=1/3; LOF=0.
      const actualDensity = await readFloat32(density.buffer, 2);
      expect(Number.isFinite(actualDensity[0])).toBe(true);
      expect(actualDensity[0]).toBeCloseTo(1 / 3, 6);
      expect(actualDensity[1]).toBe(0);
      expect(await readFloat32(lof.buffer, 2)).toEqual([0, 1]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
