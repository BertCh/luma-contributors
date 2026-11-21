// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import MULTIVARIATE_REFERENCE from '../../oracles/fixtures/multivariate-quantile-lisa-numpy.json';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUQuantileLISA,
  type GPUQuantileLISAProps
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  QUANTILE_LISA_LEGACY_INFERENCE,
  computeQuantileLISAReference,
  QUANTILE_LISA_REFERENCE_CASES
} from './quantile-lisa-reference';

it('quantile LISA matches pinned type-five and half-open class fixtures', () => {
  for (const fixture of QUANTILE_LISA_REFERENCE_CASES) {
    const result = computeQuantileLISAReference(
      Float32Array.from(fixture.values),
      fixture.quantileCount,
      fixture.selectedQuantile
    );
    expect(result.boundaries, fixture.name).toEqual(fixture.boundaries);
    expect(Array.from(result.membership), fixture.name).toEqual(fixture.membership);
  }
});

it('review2 R9 type-five integer numerator retains the exact boundary member', () => {
  // Exact ranks: 21*9/14 - 1/2 = 13; 21*10/14 - 1/2 = 14.5.
  const result = computeQuantileLISAReference(
    Float32Array.from({length: 21}, (_, row) => row),
    14,
    10
  );
  expect(result.boundaries).toEqual([13, 14.5]);
  expect(Array.from(result.membership).flatMap((member, row) => (member ? [row] : []))).toEqual([
    13, 14
  ]);
});

export function createQuantileLISAProps(
  graph: GPUCommandGraph,
  rows = 8
): GPUQuantileLISAProps & {weights: GPUSpatialWeights} {
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `quantile-lisa-${serial++}`, format, length);
  return {
    ...QUANTILE_LISA_LEGACY_INFERENCE,
    values: view('float32', rows),
    quantileParameters: view('uint32', 2),
    parameters: view('uint32', 8),
    maximumPermutations: 99,
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', 2 * rows),
      weights: view('float32', 2 * rows)
    },
    output: {
      membership: view('uint32', rows),
      boundaries: view('float32', 2),
      validCount: view('uint32', 1),
      classes: view('uint32', rows),
      invalidQuantile: view('uint32', 1),
      exceedances: view('uint32', rows),
      pseudoPValues: view('float32', rows),
      status: {overflow: view('uint32', 1)}
    }
  };
}

it('quantile LISA validates controls and ownership and emits only resident nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createQuantileLISAProps(graph);
    const nodes = new GPUQuantileLISA(props).getCommandNodes(graph);
    expect(nodes.length).toBeGreaterThan(20);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    expect(nodes.map(node => node.id)).toContain('quantile-lisa-membership');
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          quantileParameters: createTransientView(graph, 'wrong-control', 'uint32', 1)
        })
    ).toThrow();
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          output: {...props.output, membership: props.weights.neighbors}
        })
    ).toThrow();
    expect(() =>
      new GPUQuantileLISA({
        ...props,
        output: {...props.output, membership: props.values as never}
      }).getCommandNodes(graph)
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('multivariate quantile LISA validates all columns, limits and output ownership at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createQuantileLISAProps(graph);
    const variable = {values: props.values, quantileParameters: props.quantileParameters};
    const multivariate = {
      ...props,
      additionalVariables: [variable],
      output: {
        ...props.output,
        boundaries: createTransientView(graph, 'joint-boundaries', 'float32', 4),
        validCount: createTransientView(graph, 'joint-counts', 'uint32', 2)
      }
    };
    expect(() => new GPUQuantileLISA({...props, additionalVariables: [variable]})).toThrow(
      /quantile-lisa output.boundaries/
    );
    expect(
      () =>
        new GPUQuantileLISA({
          ...multivariate,
          additionalVariables: [
            {...variable, values: createTransientView(graph, 'short-variable', 'float32', 7)}
          ]
        })
    ).toThrow(/quantile-lisa variables\[1\].values/);
    expect(
      () =>
        new GPUQuantileLISA({
          ...multivariate,
          additionalVariables: [
            {
              ...variable,
              quantileParameters: createTransientView(graph, 'short-settings', 'uint32', 1)
            }
          ]
        })
    ).toThrow(/quantile-lisa variables\[1\].quantileParameters/);
    expect(() => new GPUQuantileLISA({...multivariate, maximumNeighbors: 65})).toThrow(
      /quantile-lisa maximumNeighbors/
    );
    expect(() => new GPUQuantileLISA({...multivariate, maximumPermutations: 0})).toThrow(
      /quantile-lisa maximumPermutations/
    );
    expect(() => new GPUQuantileLISA({...multivariate, maximumPermutations: 2 ** 20 + 1})).toThrow(
      /quantile-lisa maximumPermutations/
    );
    expect(
      () =>
        new GPUQuantileLISA({...multivariate, additionalVariables: new Array(64).fill(variable)})
    ).toThrow(/quantile-lisa variable count/);
    expect(
      () =>
        new GPUQuantileLISA({
          ...multivariate,
          output: {...multivariate.output, classes: props.output.membership}
        })
    ).toThrow(/outputs must not share a buffer/);
    expect(
      () =>
        new GPUQuantileLISA({
          ...multivariate,
          additionalVariables: [{...variable, values: props.output.membership as never}]
        })
    ).toThrow();
    // An additional variable cannot alias any terminal output, including optional status ports.
    expect(
      () =>
        new GPUQuantileLISA({
          ...multivariate,
          output: {...multivariate.output, observed: props.values}
        })
    ).toThrow();
    const contributor = new GPUQuantileLISA(multivariate);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.filter(node => /-inference-permute$/.test(node.id))).toHaveLength(1);
    expect(nodes.map(node => node.id)).toContain('quantile-lisa-variable-1-membership');
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    device.destroy();
  }
});

it('quantile LISA bounds command nodes at the largest variable count and enforces the budget', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createQuantileLISAProps(graph);
    const contributor = new GPUQuantileLISA({
      ...props,
      maximumNeighbors: 64,
      maximumPermutations: 2 ** 20,
      additionalVariables: new Array(63).fill({
        values: props.values,
        quantileParameters: props.quantileParameters
      }),
      output: {
        ...props.output,
        boundaries: createTransientView(graph, 'maximum-boundaries', 'float32', 128),
        validCount: createTransientView(graph, 'maximum-counts', 'uint32', 64)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const chunkWeights = {
      offsets: createTransientView(graph, 'chunk-offsets', 'uint32', 2),
      neighbors: createTransientView(graph, 'chunk-neighbors', 'uint32', 1),
      weights: createTransientView(graph, 'chunk-weights', 'float32', 1)
    };
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          weights: {
            rowCount: 2048,
            chunks: Array.from({length: 2048}, (_, rowStart) => ({
              rowStart,
              rowCount: 1,
              weights: chunkWeights
            }))
          }
        })
    ).toThrow(/quantile-lisa estimated command node count/);
    expect(
      () =>
        new GPUQuantileLISA({
          ...props,
          values: createTransientView(graph, 'over-limit-values', 'float32', 2 ** 24)
        })
    ).toThrow(/quantile-lisa values length/);
  } finally {
    device.destroy();
  }
});

it('NumPy fixtures distinguish marginal quantiles from joint complete-case quantiles', () => {
  for (const fixture of MULTIVARIATE_REFERENCE.cases) {
    const columns = fixture.columns.map(column =>
      Float32Array.from(column, value => (value === null ? NaN : value))
    );
    const references = columns.map((values, variableIndex) =>
      computeQuantileLISAReference(
        values,
        fixture.settings[variableIndex][0],
        fixture.settings[variableIndex][1],
        Uint32Array.from(fixture.mask)
      )
    );
    expect(
      references.map(reference => reference.validCount),
      fixture.name
    ).toEqual(fixture.validCounts);
    expect(
      references.flatMap(reference => reference.boundaries),
      fixture.name
    ).toEqual(
      fixture.boundaries.map(boundary =>
        typeof boundary === 'number' ? boundary : Number(boundary)
      )
    );
    expect(
      Array.from({length: 8}, (_, row) =>
        references.every(reference => reference.membership[row]) ? 1 : 0
      ),
      fixture.name
    ).toEqual(fixture.membership);
    expect(
      Array.from({length: 8}, (_, row) =>
        references.every(reference => reference.validMask[row]) ? 1 : 0
      ),
      fixture.name
    ).toEqual(fixture.validMask);
  }
  expect(MULTIVARIATE_REFERENCE.cases[0].membership).toEqual([0, 0, 0, 0, 1, 1, 0, 0]);
  expect(MULTIVARIATE_REFERENCE.cases[2].membership).toEqual(new Array(8).fill(0));
});

it('quantile LISA checks the split inference row-position binding before creating nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    // Split preparation uses one float per row per binding, not eight floats per row.
    // The former 8-column cutoff must remain usable: 513 * 32 > 16384, but all split bindings fit.
    Object.assign(device.limits, {maxStorageBufferBindingSize: 16384});
    const graph = new GPUCommandGraph(device);
    expect(() =>
      new GPUQuantileLISA(createQuantileLISAProps(graph, 513)).getCommandNodes(graph)
    ).not.toThrow();
    const oversizedGraph = new GPUCommandGraph(device);
    const rows = 4096;
    const props: GPUQuantileLISAProps = {
      ...createQuantileLISAProps(oversizedGraph, rows),
      weights: {
        rowCount: rows,
        chunks: [0, 2048].map(rowStart => ({
          rowStart,
          rowCount: 2048,
          weights: {
            offsets: createTransientView(
              oversizedGraph,
              `chunk-${rowStart}-offsets`,
              'uint32',
              2049
            ),
            neighbors: createTransientView(
              oversizedGraph,
              `chunk-${rowStart}-neighbors`,
              'uint32',
              2048
            ),
            weights: createTransientView(
              oversizedGraph,
              `chunk-${rowStart}-weights`,
              'float32',
              2048
            )
          }
        }))
      }
    };
    // All borrowed bindings fit 16384 bytes. The private row-position sentinel adds one word:
    // (4096 + 1) * 4 = 16388 bytes, exceeding the device limit independently of CSR chunking.
    const contributor = new GPUQuantileLISA(props);
    expect(() => contributor.getCommandNodes(oversizedGraph)).toThrow(
      /row positions scratch.*limit/
    );
  } finally {
    device.destroy();
  }
});
