// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUNetworkCostWeights} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const role of ['rows', 'columns'] as const)
  for (const method of ['2sfca', '3sfca'] as const) {
    it(`${method} suppresses ${role} handle-only loss and recovers on the same graph`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const costs = rig.input(Float32Array.of(0, Infinity), 'float32', 2);
      const converged = rig.input(Uint32Array.of(0), 'uint32', 1);
      const horizon = rig.input(Float32Array.of(1), 'float32', 1);
      const parameters = rig.input(Float32Array.of(2, 0, 0, 1), 'float32', 4);
      const converters = (['rows', 'columns'] as const).map(orientation => {
        const converter = new GPUNetworkCostWeights({
          id: `cost-weights-${orientation}`,
          costs,
          rowCount: 1,
          orientation,
          parameters,
          costsComplete: orientation === role ? converged : undefined,
          costLimit: orientation === role ? horizon : undefined,
          output: {
            offsets: rig.output('uint32', orientation === 'rows' ? 2 : 3).view,
            neighbors: rig.output('uint32', 2).view,
            weights: rig.output('float32', 2).view,
            status: {
              overflow: rig.output('uint32', 1).view,
              complete: orientation === role ? rig.output('uint32', 1).view : undefined
            }
          }
        });
        rig.graph.add(converter);
        return converter;
      });
      const demand = rig.input(Float32Array.of(1, 1), 'float32', 2);
      const accessibility = rig.output('float32', 2),
        ratios = rig.output('float32', 1),
        reachable = rig.output('uint32', 2),
        tradeArea = rig.output('uint32', 2),
        expectedDemand = rig.output('float32', 1);
      rig.graph.add(
        new GPUCatchmentAccessibility({
          method,
          supply: rig.input(Float32Array.of(2), 'float32', 1),
          demand,
          facilityWeights: converters[0].outputWeights,
          demandWeights: converters[1].outputWeights,
          output: {
            accessibility: accessibility.view,
            ratios: ratios.view,
            reachableFacilities: reachable.view
          }
        })
      );
      rig.graph.add(
        new GPUHuffTradeAreas({
          attractiveness: rig.input(Float32Array.of(1), 'float32', 1),
          demand,
          facilityWeights: converters[0].outputWeights,
          demandWeights: converters[1].outputWeights,
          output: {tradeArea: tradeArea.view, expectedDemand: expectedDemand.view}
        })
      );
      const compiled = await rig.graph.compileAsync();
      try {
        submitGraph(device, compiled, undefined);
        expect((await readFloat32(accessibility.buffer, 2)).every(Number.isNaN)).toBe(true);
        expect((await readFloat32(expectedDemand.buffer, 1)).every(Number.isNaN)).toBe(true);
        if (role === 'columns')
          expect(await readUint32(reachable.buffer, 2)).toEqual([0xffffffff, 0xffffffff]);
        rig.bufferOf(costs).write(Float32Array.of(0, 2));
        rig.bufferOf(converged).write(Uint32Array.of(1));
        rig.bufferOf(horizon).write(Float32Array.of(2));
        submitGraph(device, compiled, undefined);
        // Two unit demands share supply 2: each receives accessibility 1, expected demand is 2.
        expect(await readFloat32(accessibility.buffer, 2)).toEqual([1, 1]);
        expect(await readFloat32(ratios.buffer, 1)).toEqual([1]);
        expect(await readUint32(reachable.buffer, 2)).toEqual([1, 1]);
        expect(await readUint32(tradeArea.buffer, 2)).toEqual([0, 0]);
        const capturedDemand = (await readFloat32(expectedDemand.buffer, 1))[0];
        expect(Number.isFinite(capturedDemand)).toBe(true);
        expect(capturedDemand).toBeCloseTo(2, 6);
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy();
      }
    });
  }
