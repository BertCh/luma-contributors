// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getRasterFractionalClippingWGSL} from '../../../src/raster-analysis/fractional-coverage/raster-fractional-coverage-index';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalZonalDistribution} from '../../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {GPURasterPatchMetrics} from '../../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {AREA_PACKET_CASES, createAreaPackets} from './versioned-area-packets.test-utils';

it('W5-12 both coverage routes invert packed basis vectors and retain their half-cell triangle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const searchStructure of ['bounds', 'brute-force'] as const) {
    for (const affine of [
      [1, 1, 0, 1, 1 + 3 * 2 ** -25, 0],
      [1 + 2 ** -23, 1, 0, 1 + 2 ** -22, 1 + 2 ** -23, 0]
    ] as const) {
      const rig = createParityGraph(device);
      const grid = {width: 1, height: 1, affine, pixelInterpretation: 'area' as const};
      const packet = getGPURasterFractionalCoverageParameterValues(grid);
      const settings = rig.input('parameters', Array.from(packet), 'float32').view;
      const positions = rig.input(
        'positions',
        [0, 0, packet[2], packet[4], packet[3], packet[5]],
        'float32x2'
      ).view;
      const pixels = rig.output('pixels', 4, 'float32');
      rig.graph.add(
        createWGSLKernelNode(rig.graph, {
          id: 'packed-basis',
          operation: 'fractional-basis-probe',
          invocationCount: 2,
          bindings: [
            {name: 'settings', view: settings, type: 'f32', access: 'read'},
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'pixels', view: pixels.view, type: 'f32', access: 'read_write'}
          ],
          declarations: getRasterFractionalClippingWGSL('area'),
          body: 'let pixel=point(index+1u);pixels[pixelsOffset+index*2u]=pixel.x;pixels[pixelsOffset+index*2u+1u]=pixel.y;'
        })
      );
      const fractions = rig.output('fractions', 1, 'float32');
      const count = rig.output('count', 1, 'uint32');
      const validity = rig.output('validity', 1, 'uint32');
      rig.graph.add(
        new GPURasterFractionalCoverage({
          grid,
          searchStructure,
          parameters: settings,
          zones: {
            positions,
            featureOffsets: rig.input('features', [0, 1], 'uint32').view,
            polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
            ringOffsets: rig.input('rings', [0, 3], 'uint32').view
          },
          output: {
            zoneIds: rig.output('zones', 1, 'uint32').view,
            cellIds: rig.output('cells', 1, 'uint32').view,
            fractions: fractions.view,
            zoneValidity: validity.view,
            status: {
              count: count.view,
              requiredCount: rig.output('required', 1, 'uint32').view,
              overflow: rig.output('overflow', 1, 'uint32').view
            }
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // Unit basis round trip; triangle (0,0),(1,0),(0,1) covers exactly half a cell.
        expect(await readFloat32(pixels.buffer, 4)).toEqual([1, 0, 0, 1]);
        expect(await readUint32(count.buffer, 1)).toEqual([1]);
        expect(await readUint32(validity.buffer, 1)).toEqual([1]);
        const [fraction] = await readFloat32(fractions.buffer, 1);
        expect(Number.isFinite(fraction)).toBe(true);
        expect(fraction).toBe(0.5);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  }
});

it('review S3 normalizes a subnormal inverse determinant before GPU division', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const scale = 2 ** -67;
  // Axes scale*[[1,2],[3,4]] have determinant -2*scale^2 = -2^-133.
  const grid = {
    width: 1,
    height: 1,
    affine: [scale, 2 * scale, 0, 3 * scale, 4 * scale, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const settingsInput = rig.input(
    'settings',
    Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
    'float32'
  );
  const positionsInput = rig.input(
    'positions',
    [scale, 3 * scale, 2 * scale, 4 * scale],
    'float32x2'
  );
  const pixels = rig.output('pixels', 4, 'float32');
  rig.graph.add(
    createWGSLKernelNode(rig.graph, {
      id: 'subnormal-inverse',
      operation: 'fractional-basis-probe',
      invocationCount: 2,
      bindings: [
        {name: 'settings', view: settingsInput.view, type: 'f32', access: 'read'},
        {name: 'positions', view: positionsInput.view, type: 'f32', access: 'read'},
        {name: 'pixels', view: pixels.view, type: 'f32', access: 'read_write'}
      ],
      declarations: getRasterFractionalClippingWGSL('area'),
      body: 'let pixel=point(index);pixels[pixelsOffset+index*2u]=pixel.x;pixels[pixelsOffset+index*2u+1u]=pixel.y;'
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const basisPixels = await readFloat32(pixels.buffer, 4);
    for (const coordinate of basisPixels) expect(Number.isFinite(coordinate)).toBe(true);
    // A^-1 A is exactly the unit basis. Zero divided by the negative determinant may
    // retain a negative sign on Metal; pixel coordinates have no signed-zero contract.
    expect(basisPixels.map(coordinate => (coordinate === 0 ? 0 : coordinate))).toEqual([
      1, 0, 0, 1
    ]);
    settingsInput.buffer.write(Float32Array.of(0, 0, Infinity, 1, 2 ** 100, 1, 1, 0));
    positionsInput.buffer.write(Float32Array.of(1, 1, 2, 2));
    submitGraph(device, compiled, undefined);
    expect((await readFloat32(pixels.buffer, 4)).every(Number.isFinite)).toBe(false);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

for (const {cellSizeMode, originY, area} of AREA_PACKET_CASES) {
  it(`UNI-03 GPU distribution and patch metrics read padded and rebased ${cellSizeMode} packets`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {grid, packets} = createAreaPackets(originY);
    for (const packet of packets.filter(parameters => parameters.length >= 8)) {
      const rig = createParityGraph(device);
      const parameters = rig.input('parameters', Array.from(packet), 'float32').view;
      const distribution = {
        means: rig.output('means', 1, 'float32'),
        populationVariances: rig.output('population', 1, 'float32'),
        sampleVariances: rig.output('sample', 1, 'float32'),
        effectiveSampleSizes: rig.output('support', 1, 'float32'),
        weightSums: rig.output('weights', 1, 'float32'),
        squaredWeightSums: rig.output('squares', 1, 'float32'),
        coveredArea: rig.output('covered', 1, 'float32'),
        validArea: rig.output('valid-area', 1, 'float32'),
        quantiles: rig.output('quantiles', 1, 'float32'),
        quantileValidity: rig.output('quantile-validity', 1, 'uint32'),
        validity: rig.output('validity', 1, 'uint32'),
        status: rig.output('status', 1, 'uint32')
      };
      const distributionProps: ConstructorParameters<
        typeof GPURasterFractionalZonalDistribution
      >[0] = {
        grid,
        cellSizeMode,
        parameters,
        zoneCount: 1,
        input: {
          id: 'band',
          format: 'float32',
          storage: {kind: 'buffer', values: rig.input('values', [7], 'float32').view}
        },
        probabilities: rig.input('probabilities', [0.5], 'float32').view,
        coverage: {
          zoneIds: rig.input('zone-ids', [0], 'uint32').view,
          cellIds: rig.input('cell-ids', [0], 'uint32').view,
          fractions: rig.input('fractions', [0.5], 'float32').view,
          zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
          status: {
            count: rig.input('count', [1], 'uint32').view,
            requiredCount: rig.input('required', [1], 'uint32').view,
            overflow: rig.input('overflow', [0], 'uint32').view
          }
        },
        output: {
          means: distribution.means.view,
          populationVariances: distribution.populationVariances.view,
          sampleVariances: distribution.sampleVariances.view,
          effectiveSampleSizes: distribution.effectiveSampleSizes.view,
          weightSums: distribution.weightSums.view,
          squaredWeightSums: distribution.squaredWeightSums.view,
          coveredArea: distribution.coveredArea.view,
          validArea: distribution.validArea.view,
          quantiles: distribution.quantiles.view,
          quantileValidity: distribution.quantileValidity.view,
          validity: distribution.validity.view,
          status: distribution.status.view
        }
      };
      let means = distributionProps.output.means;
      let fractions = distributionProps.coverage.fractions;
      let count = distributionProps.coverage.status.count;
      const reads = {means: 0, fractions: 0, count: 0};
      Reflect.deleteProperty(distributionProps.output, 'means');
      Reflect.deleteProperty(distributionProps.coverage, 'fractions');
      Reflect.deleteProperty(distributionProps.coverage.status, 'count');
      Object.setPrototypeOf(distributionProps.output, {
        get means() {
          reads.means++;
          return means;
        }
      });
      Object.setPrototypeOf(distributionProps.coverage, {
        get fractions() {
          reads.fractions++;
          return fractions;
        }
      });
      Object.setPrototypeOf(distributionProps.coverage.status, {
        get count() {
          reads.count++;
          return count;
        }
      });
      const contributor = new GPURasterFractionalZonalDistribution(distributionProps);
      means = distribution.populationVariances.view;
      fractions = distribution.means.view;
      count = distribution.validity.view;
      rig.graph.add(contributor);
      expect(reads).toEqual({means: 1, fractions: 1, count: 1});
      const patchAreas = rig.output('patch-areas', 1, 'float32');
      rig.graph.add(
        new GPURasterPatchMetrics({
          grid,
          cellSizeMode,
          parameters,
          labels: rig.input('labels', [1], 'uint32').view,
          output: {areas: patchAreas.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(distribution.validity.buffer, 1)).toEqual([1]);
        expect(await readFloat32(distribution.means.buffer, 1)).toEqual([7]);
        for (const [buffer, expected] of [
          [patchAreas.buffer, area],
          [distribution.coveredArea.buffer, 0.5 * area],
          [distribution.validArea.buffer, 0.5 * area]
        ] as const) {
          const [actual] = await readFloat32(buffer, 1);
          expect(Number.isFinite(actual)).toBe(true);
          expect(Math.abs(actual / expected - 1)).toBeLessThan(2e-5);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  });
}
