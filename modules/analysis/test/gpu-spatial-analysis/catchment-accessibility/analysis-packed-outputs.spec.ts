// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {getSpatialAnalysisPackedKernelNodes} from '../../../src/gpu-spatial-analysis/geographic-distribution/spatial-analysis-packed-kernel';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addFiniteScaleAnalysisFixture} from './analysis-finite-ratios-test-fixture';

it.each([
  2 ** 127,
  2 ** -126
])('accessibility ratios and geographic moments preserve finite scale %s', async scale => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'analysis-finite-ratios');
  try {
    const {scores, tradeArea, probability, probabilities, expectedDemand, geographic} =
      addFiniteScaleAnalysisFixture(fixture, scale);
    fixture.run();
    for (const {accessibility, ratios} of scores) {
      expect((await fixture.readFloat32(accessibility, 1))[0]).toBeCloseTo(1, 6);
      expect((await fixture.readFloat32(ratios, 1))[0]).toBeCloseTo(1, 6);
    }
    expect(await fixture.readUint32(tradeArea, 1)).toEqual([0]);
    expect((await fixture.readFloat32(probability, 1))[0]).toBeCloseTo(1, 6);
    expect((await fixture.readFloat32(probabilities, 1))[0]).toBeCloseTo(1, 6);
    expect((await fixture.readFloat32(expectedDemand, 1))[0]).toBeCloseTo(3, 6);
    for (const {mean, direction} of geographic) {
      const center = await fixture.readFloat32(mean, 2);
      expect(center[0]).toBeCloseTo(1, 6);
      expect(center[1]).toBeCloseTo(0, 6);
      const values = await fixture.readFloat32(direction, 3);
      expect(values[0]).toBeCloseTo(0, 6);
      expect(values[1]).toBeCloseTo(0, 6);
      expect(values[2]).toBeCloseTo(1, 6);
    }
  } finally {
    fixture.destroy();
  }
});

it('packed storage publication preserves input prefixes and initial output values through one imported handle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'packed-usage-prefixes');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const values = new Float32Array(256);
    values.set([2, 4, 3, 5], 1);
    values.set([7, 11], 65);
    const source = fixture.input('source', values, 'float32', values.length);
    // The published graph requires one active imported handle for a writable physical buffer.
    const result = fixture.graph.createDataView(source.buffer, {
      format: 'float32',
      byteOffset: 260,
      length: 2
    });
    const output = fixture.output('readback', 'float32', 2);
    fixture.graph.add(
      getSpatialAnalysisPackedKernelNodes(fixture.graph, {
        id: 'prefix-arithmetic',
        operation: 'PackedSpatialAnalysisRegression',
        invocationCount: 2,
        bindings: [
          {
            name: 'values',
            view: fixture.graph.createDataView(source.buffer, {
              format: 'float32',
              byteOffset: 4,
              length: 2
            }),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'weights',
            view: fixture.graph.createDataView(source.buffer, {
              format: 'float32',
              byteOffset: 12,
              length: 2
            }),
            type: 'f32',
            access: 'read'
          },
          {name: 'result', view: result, type: 'f32', access: 'read_write'}
        ],
        body: 'result[resultOffset+index]+=values[valuesOffset+index]*weights[weightsOffset+index];'
      })
    );
    fixture.graph.add(
      getSpatialAnalysisPackedKernelNodes(fixture.graph, {
        id: 'prefix-readback',
        operation: 'PackedSpatialAnalysisRegression',
        invocationCount: 2,
        bindings: [
          {name: 'values', view: result, type: 'f32', access: 'read'},
          {name: 'result', view: output.view, type: 'f32', access: 'read_write'}
        ],
        body: 'result[resultOffset+index]=values[valuesOffset+index];'
      })
    );
    compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await fixture.readFloat32(output, 2)).toEqual([13, 31]);
    submitGraph(device, compiled, undefined);
    expect(await fixture.readFloat32(output, 2)).toEqual([19, 51]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('catchment, Huff and geographic outputs work in disjoint packed ranges, including separate dispatches in one alignment window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'analysis-packed-outputs');
  try {
    const weights = {
      offsets: fixture.input('offsets', new Uint32Array([0, 1, 2]), 'uint32', 3),
      neighbors: fixture.input('neighbors', new Uint32Array([1, 0]), 'uint32', 2),
      weights: fixture.input('weights', new Float32Array([1, 1]), 'float32', 2)
    };
    const demand = fixture.input('demand', new Float32Array([2, 4]), 'float32', 2);
    const catchment = fixture.output('catchment', 'uint32', 384);
    const huff = fixture.output('huff', 'uint32', 384);
    const geographic = fixture.output('geographic', 'uint32', 384);
    const scalar = (buffer: typeof catchment, byteOffset: number, length: number) =>
      fixture.graph.createDataView(buffer.view.buffer, {format: 'float32', byteOffset, length});
    const words = (buffer: typeof catchment, byteOffset: number, length: number) =>
      fixture.graph.createDataView(buffer.view.buffer, {format: 'uint32', byteOffset, length});
    const points = (byteOffset: number) =>
      fixture.graph.createDataView(geographic.view.buffer, {
        format: 'float32x2',
        byteOffset,
        length: 1
      });
    fixture.graph.add(
      new GPUCatchmentAccessibility({
        method: '3sfca',
        facilityWeights: weights,
        supply: fixture.input('supply', new Float32Array([10, 20]), 'float32', 2),
        demand,
        output: {
          accessibility: scalar(catchment, 0, 2),
          ratios: scalar(catchment, 256, 2),
          demandValidity: words(catchment, 264, 2),
          reachableFacilities: words(catchment, 512, 2)
        }
      })
    );
    fixture.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: fixture.input('attractiveness', new Float32Array([1, 2]), 'float32', 2),
        demandWeights: weights,
        facilityWeights: weights,
        demand,
        output: {
          tradeArea: words(huff, 0, 2),
          tradeAreaProbability: scalar(huff, 256, 2),
          probabilities: scalar(huff, 264, 2),
          expectedDemand: scalar(huff, 272, 2),
          demandValidity: words(huff, 280, 2)
        }
      })
    );
    fixture.graph.add(
      new GPUGeographicDistribution({
        positions: fixture.input('positions', new Float32Array([0, 0, 2, 0]), 'float32x2', 2),
        parameters: fixture.input(
          'parameters',
          getGPUGeographicDistributionParameterValues(),
          'float32',
          8
        ),
        output: {
          meanCenters: points(0),
          standardDistances: scalar(geographic, 8, 1),
          medianCenters: points(256),
          medianConverged: words(geographic, 512, 1),
          counts: words(geographic, 768, 1),
          weightSums: scalar(geographic, 1024, 1)
        }
      })
    );
    fixture.run();
    const catchmentValues = await fixture.readFloat32(catchment, 384);
    const catchmentFlags = await fixture.readUint32(catchment, 384);
    expect(catchmentValues.slice(0, 2)).toEqual([10, 2.5]);
    expect(catchmentValues.slice(64, 66)).toEqual([2.5, 10]);
    expect(catchmentFlags.slice(66, 68)).toEqual([1, 1]);
    expect(catchmentFlags.slice(128, 130)).toEqual([1, 1]);
    const huffValues = await fixture.readFloat32(huff, 384);
    const huffFlags = await fixture.readUint32(huff, 384);
    expect(huffFlags.slice(0, 2)).toEqual([1, 0]);
    expect(huffValues.slice(64, 68)).toEqual([1, 1, 1, 1]);
    expect(huffValues.slice(68, 70)).toEqual([4, 2]);
    expect(huffFlags.slice(70, 72)).toEqual([1, 1]);
    const geographicValues = await fixture.readFloat32(geographic, 384);
    const geographicFlags = await fixture.readUint32(geographic, 384);
    expect(geographicValues.slice(0, 3)).toEqual([1, 0, 1]);
    expect(geographicValues.slice(64, 66)).toEqual([1, 0]);
    expect(geographicFlags[128]).toBe(1);
    expect(geographicFlags[192]).toBe(2);
    expect(geographicValues[256]).toBe(2);
  } finally {
    fixture.destroy();
  }
});
