// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPURasterFractionalZonalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import * as expansion from '../../src/raster-analysis/raster-expansion-utils';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const floats = (length = 1) =>
    createTransientView(graph, `floats-${serial++}`, 'float32', length);
  const integers = (length = 1) =>
    createTransientView(graph, `integers-${serial++}`, 'uint32', length);
  const grid = {
    width: 1,
    height: 1,
    affine: [2, 0, 0, 0, 3, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const texture = graph.createTransientTexture({
    id: 'input',
    width: 1,
    height: 1,
    format: 'r32uint',
    usage: Texture.SAMPLE
  });
  const input = {
    id: 'input',
    format: 'uint32' as const,
    storage: {kind: 'texture' as const, view: graph.createTextureView(texture)}
  };
  const coverage = {
    zoneIds: integers(),
    cellIds: integers(),
    fractions: floats(),
    zoneValidity: integers(),
    status: {count: integers(), requiredCount: integers(), overflow: integers()}
  };
  const scalar = {
    grid,
    input,
    coverage,
    zoneCount: 1,
    quantity: 'extensive' as const,
    output: {
      totals: floats(),
      means: floats(),
      coveredArea: floats(),
      validArea: floats(),
      validity: integers()
    }
  };
  const categorical = {
    grid,
    input,
    coverage,
    zoneCount: 1,
    categoryCapacity: 2,
    output: {
      coverageSums: floats(2),
      classAreas: floats(2),
      proportions: floats(2),
      coveredArea: floats(),
      validArea: floats(),
      missingArea: floats(),
      majority: integers(),
      minority: integers(),
      validity: integers()
    }
  };
  const distribution = {
    grid,
    input,
    coverage,
    zoneCount: 1,
    probabilities: floats(),
    output: {
      means: floats(),
      populationVariances: floats(),
      sampleVariances: floats(),
      effectiveSampleSizes: floats(),
      weightSums: floats(),
      squaredWeightSums: floats(),
      coveredArea: floats(),
      validArea: floats(),
      quantiles: floats(),
      quantileValidity: integers(),
      validity: integers(),
      status: integers()
    }
  };
  return {device, graph, grid, input, scalar, categorical, distribution};
}

it('B1 categorical texture construction never length-checks a missing buffer plane', () => {
  const context = fixture();
  const check = vi.spyOn(expansion, 'requireRasterLength');
  try {
    new GPURasterFractionalCategoricalStatistics(context.categorical);
    expect(check.mock.calls.filter(call => call[3] === 'input values')).toHaveLength(0);
    const buffer = createTransientView(context.graph, 'short-categories', 'uint32', 0);
    expect(
      () =>
        new GPURasterFractionalCategoricalStatistics({
          ...context.categorical,
          input: {id: 'short', format: 'uint32', storage: {kind: 'buffer', values: buffer}}
        })
    ).toThrow(/input values|grid/);
    expect(
      () =>
        new GPURasterFractionalCategoricalStatistics({
          ...context.categorical,
          grid: {...context.grid, width: 2}
        })
    ).toThrow(/extent|width|dimensions/);
  } finally {
    check.mockRestore();
    context.device.destroy();
  }
});

for (const representation of ['inherited', 'non-enumerable'] as const) {
  for (const operation of ['scalar', 'categorical', 'distribution'] as const) {
    it(`B4 ${operation} captures ${representation} policy fields`, () => {
      const context = fixture();
      const policy = <Props extends object, Key extends keyof Props>(
        props: Props,
        key: Key,
        value: Props[Key]
      ): Props => {
        const result = {...props};
        Reflect.deleteProperty(result, key);
        if (representation === 'inherited')
          Object.setPrototypeOf(result, {
            get [key]() {
              return value;
            }
          });
        else Object.defineProperty(result, key, {get: () => value, enumerable: false});
        return result;
      };
      try {
        if (operation === 'scalar') {
          const scalar = new GPURasterFractionalZonalStatistics(
            policy(context.scalar, 'quantity', 'extensive')
          );
          // One fully covered area-6 cell of value 5 has extensive total 5, not 30.
          expect(scalar.props.quantity).toBe('extensive');
          const source = [
            ...getParsedKernelSources(scalar.getCommandNodes(context.graph)).values()
          ].join('\n');
          expect(source).toContain('total+=value*fraction*1.0;');
        } else if (operation === 'categorical') {
          const props = {...context.categorical, cellSizeMode: 'geographic' as const};
          const categorical = new GPURasterFractionalCategoricalStatistics(
            policy(props, 'cellSizeMode', 'geographic')
          );
          expect(categorical.props.cellSizeMode).toBe('geographic');
        } else {
          const props = {...context.distribution, quantileMethod: 'inverse-cdf' as const};
          const distribution = new GPURasterFractionalZonalDistribution(
            policy(props, 'quantileMethod', 'inverse-cdf')
          );
          expect(distribution.props.quantileMethod).toBe('inverse-cdf');
        }
      } finally {
        context.device.destroy();
      }
    });
  }
}

it('S1 distribution rejects non-storage status before declaring any transient resource', () => {
  const context = fixture();
  const statusBuffer = context.graph.createTransientBuffer({
    id: 'copy-only-status',
    byteLength: 4,
    usage: Buffer.COPY_SRC
  });
  const status = context.graph.createDataView(statusBuffer, {format: 'uint32', length: 1});
  const contributor = new GPURasterFractionalZonalDistribution({
    ...context.distribution,
    output: {...context.distribution.output, status}
  });
  const allocations = vi.spyOn(context.graph, 'createTransientBuffer');
  try {
    expect(() => context.graph.add(contributor)).toThrow(/STORAGE|storage/);
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    allocations.mockRestore();
    context.device.destroy();
  }
});

it('S1 distribution rejects foreign-graph status before declaring transient resources', () => {
  const context = fixture();
  const foreign = new GPUCommandGraph(context.device);
  const status = createTransientView(foreign, 'foreign-status', 'uint32', 1);
  const contributor = new GPURasterFractionalZonalDistribution({
    ...context.distribution,
    output: {...context.distribution.output, status}
  });
  const allocations = vi.spyOn(context.graph, 'createTransientBuffer');
  try {
    expect(() => context.graph.add(contributor)).toThrow(/belong|graph/);
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    allocations.mockRestore();
    context.device.destroy();
  }
});
