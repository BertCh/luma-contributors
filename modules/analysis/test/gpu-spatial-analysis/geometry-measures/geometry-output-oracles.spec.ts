// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {CPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPULocalFrameProjectionOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection-operation';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const budget of [1, 2, 3]) {
  it(`both public geodesic operations honor the float32 inverse budget ${budget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
      const actual = await runOperationGraph(device, {
        inputs: {
          origins: {
            format: 'float32x2',
            values: new Float32Array(budget === 1 ? [0, 30] : [0, 20])
          },
          targets: {
            format: 'float32x2',
            values: new Float32Array(budget === 1 ? [100, -45] : [170, 60])
          }
        },
        operations: [
          new Operation({
            origins: 'origins',
            targets: 'targets',
            pairCount: 1,
            maximumIterations: budget,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
            output: {
              distances: 'distances',
              status: {
                converged: 'converged',
                approximation: 'approximation',
                iterationLimitReached: 'iterationLimitReached'
              }
            }
          })
        ],
        outputs: ['distances', 'converged', 'approximation', 'iterationLimitReached']
      });
      expect(
        ['converged', 'approximation', 'iterationLimitReached'].map(name => actual[name][0])
      ).toEqual(budget === 3 ? [1, 0, 0] : [0, 1, 1]);
      expect(Number.isFinite(actual['distances'][0])).toBe(true);
      // Published float32 inverse allowance 3e-5 relative; branches differ by 17,906 m.
      expect(
        Math.abs(
          actual['distances'][0] -
            (budget === 1 ? 13052084.96 : budget === 2 ? 11073359.15 : 11091265.03)
        )
      ).toBeLessThan(budget === 3 ? 350 : 30);
    }
  });
}

for (const optional of [
  undefined,
  'converged',
  'approximation',
  'iterationLimitReached'
] as const) {
  for (const includeDistances of optional ? [false, true] : [true]) {
    it(`writes PROJ-qualified ordinary geodesic statuses: ${optional ?? 'absent'}, distances ${includeDistances}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const selected = optional ? fixture.output('selected', 'uint32', 1) : undefined;
      const views = {
        origins: fixture.input('origins', 'float32x2', new Float32Array([-73.75, 40.5])),
        targets: fixture.input('targets', 'float32x2', new Float32Array([2.25, 48.75])),
        distances: includeDistances ? fixture.output('distances', 'float32', 1) : undefined,
        ...(optional ? {[optional]: selected} : {})
      };
      const operation = new GPUGeodesicPairsOperation({
        origins: 'origins',
        targets: 'targets',
        pairCount: 1,
        maximumIterations: 100,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
        output: {
          distances: includeDistances ? 'distance' : undefined,
          status: optional ? {[optional]: 'status'} : undefined
        }
      });
      fixture.graph.add(
        operation.createContributor(port => views[port as keyof typeof views] as never)
      );
      const compiled = fixture.graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          distance: views.distances ? await fixture.readRows([views.distances]) : [],
          status: selected ? await fixture.readRows([selected]) : []
        }));
        expect(actual.status).toEqual(optional ? [optional === 'converged' ? 1 : 0] : []);
        if (includeDistances) {
          expect(Number.isFinite(actual.distance[0])).toBe(true);
          expect(Math.abs(actual.distance[0] - 5846818.164066021)).toBeLessThan(180);
        }
        // 3e-5 relative f32 Vincenty allowance on the 5,846,818 m PROJ inverse path.
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
  it(`writes non-origin local frame coordinates from direct PROJ ${kind}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const views = {
      positions: fixture.input('positions', 'float32x2', new Float32Array([0.5, 0.25, 1, 1])),
      output: fixture.output('output', 'float32x2', 2)
    };
    const operation = new GPULocalFrameProjectionOperation({
      positions: 'positions',
      rowCount: 2,
      frame: createGPULocalFrame({origin: [0, 0], kind}),
      bounds: [-1, -1, 1, 1],
      output: 'projected'
    });
    fixture.graph.add(
      operation.createContributor(port => views[port as keyof typeof views] as never)
    );
    const compiled = fixture.graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, () => fixture.readRows([views.output]));
      const expected =
        kind === 'azimuthal-equidistant'
          ? [55597.187279835605, 27799.122892627045, 111183.78889285575, 111200.72530193203]
          : [55659.174944973725, 27643.811936381957, 111305.47485636259, 110577.19525609486];
      // 0.01 m patch error plus f32 output/arithmetic error within a 160 km frame.
      actual.forEach((value, index) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThan(0.05);
      });
    } finally {
      compiled.destroy();
      fixture.destroy();
      operation.destroy();
    }
  });
}

it('writes group lengths 2,15,0 from independent 3-4-5 and 6-8-10 segments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const lengths = fixture.output('group-lengths', 'float32', 3);
  fixture.graph.add(
    new GPUGeometryMeasures({
      geometryType: 'lines',
      positions: fixture.input(
        'positions',
        'float32x2',
        new Float32Array([0, 0, 3, 4, 10, 0, 16, 8, 0, 10, 0, 12])
      ),
      lineOffsets: fixture.input('lines', 'uint32', new Uint32Array([0, 2, 4, 6])),
      groupIds: fixture.input('groups', 'uint32', new Uint32Array([1, 1, 0])),
      groupCount: 3,
      output: {},
      groupOutput: {lengths}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, () => fixture.readRows([lengths]));
    expect(actual).toEqual([2, 15, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
