// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computePopulationSupportAlignmentOnCPU,
  CPUPopulationSupportAlignmentOperation
} from '../../../src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation';
import {
  GPUPopulationSupportAlignmentOperation,
  type PopulationSupportAlignmentOperationProps
} from '../../../src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment-operation';
import {GPUPopulationSupportAlignment} from '../../../src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment';
import {addPopulationSupportAlignmentRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const PROPS: PopulationSupportAlignmentOperationProps = {
  populations: 'inputPopulations',
  sourceIds: 'sourceIds',
  targetIds: 'targetIds',
  targetCount: 3,
  output: {populations: 'populations', validity: 'validity', status: 'status'}
};

it('matches a hand-calculated reordered support, including nonfinite population values', () => {
  const input = {
    populations: Float32Array.of(6, 4, NaN, Infinity, -1, 5),
    sourceIds: Uint32Array.of(42, 0xffffffff, 0),
    targetIds: Uint32Array.of(0, 42, 0xffffffff)
  };
  const result = computePopulationSupportAlignmentOnCPU(input);
  expect(Array.from(result.populations)).toEqual([-1, 5, 6, 4, NaN, Infinity]);
  expect(Array.from(result.validity)).toEqual([0, 0, 0]);
  expect(Array.from(result.status)).toEqual([1, 0, 0, 0]);
  expect(Array.from(input.sourceIds)).toEqual([42, 0xffffffff, 0]);
});

it('preserves missing-before-duplicate precedence and counts excluded source rows individually', () => {
  const result = computePopulationSupportAlignmentOnCPU({
    populations: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    sourceIds: Uint32Array.of(1, 1, 2, 3, 3),
    targetIds: Uint32Array.of(1, 2, 2, 4, 4)
  });
  expect(Array.from(result.populations)).toEqual(Array(10).fill(NaN));
  expect(Array.from(result.validity)).toEqual([2, 2, 2, 1, 1]);
  expect(Array.from(result.status)).toEqual([0, 2, 3, 2]);
});

it('matches an exhaustive scan oracle for every pair of three-row supports over three identifiers', () => {
  const identifiers = [0, 7, 0xffffffff];
  const populations = Float32Array.of(1, 10, 2, 20, 3, 30);
  const supports = Array.from({length: 27}, (_, code) =>
    Uint32Array.from(
      {length: 3},
      (_, position) => identifiers[Math.floor(code / 3 ** position) % 3]
    )
  );
  for (const sourceIds of supports)
    for (const targetIds of supports) {
      const expectedPopulations = new Float32Array(6).fill(NaN);
      const expectedValidity = new Uint32Array(3);
      targetIds.forEach((identifier, row) => {
        const matchingRows = Array.from(sourceIds.keys()).filter(
          source => sourceIds[source] === identifier
        );
        const duplicates = Array.from(targetIds).filter(target => target === identifier).length > 1;
        if (matchingRows.length === 0) expectedValidity[row] = 1;
        else if (matchingRows.length !== 1 || duplicates) expectedValidity[row] = 2;
        else
          expectedPopulations.set(
            populations.subarray(2 * matchingRows[0], 2 * matchingRows[0] + 2),
            2 * row
          );
      });
      const missing = Array.from(expectedValidity).filter(value => value === 1).length;
      const ambiguous = Array.from(expectedValidity).filter(value => value === 2).length;
      const excluded = Array.from(sourceIds).filter(
        source => !Array.from(targetIds).includes(source)
      ).length;
      const actual = computePopulationSupportAlignmentOnCPU({populations, sourceIds, targetIds});
      expect(actual.populations).toEqual(expectedPopulations);
      expect(actual.validity).toEqual(expectedValidity);
      expect(Array.from(actual.status)).toEqual([
        Number(!missing && !ambiguous && !excluded),
        missing,
        ambiguous,
        excluded
      ]);
    }
});

it('aligns a large unique support without a GPU-sized unit cap', () => {
  const count = 100001;
  const sourceIds = Uint32Array.from({length: count}, (_, row) => row);
  const targetIds = Uint32Array.from(sourceIds).reverse();
  const populations = Float32Array.from({length: 2 * count}, (_, component) => component);
  const result = computePopulationSupportAlignmentOnCPU({populations, sourceIds, targetIds});
  expect(
    result.populations.every(
      (value, component) => value === 2 * (count - 1 - Math.floor(component / 2)) + (component % 2)
    )
  ).toBe(true);
  expect(result.validity.every(value => value === 0)).toBe(true);
  expect(Array.from(result.status)).toEqual([1, 0, 0, 0]);
});

it('checks array lengths and logical operation contracts without changing legacy output names', () => {
  expect(() =>
    computePopulationSupportAlignmentOnCPU({
      populations: Float32Array.of(1),
      sourceIds: Uint32Array.of(1),
      targetIds: Uint32Array.of(1)
    })
  ).toThrow(/populations.*2 values/);
  expect(() =>
    computePopulationSupportAlignmentOnCPU({
      populations: new Float32Array(0),
      sourceIds: new Uint32Array(0),
      targetIds: Uint32Array.of(1)
    })
  ).toThrow(/sourceIds.length/);
  expect(() =>
    computePopulationSupportAlignmentOnCPU({
      populations: Float32Array.of(1, 2),
      sourceIds: Uint32Array.of(1),
      targetIds: new Uint32Array(0)
    })
  ).toThrow(/targetIds.length/);
  expect(() => new CPUPopulationSupportAlignmentOperation({...PROPS, targetCount: 0})).toThrow(
    /targetCount/
  );
  expect(
    () =>
      new CPUPopulationSupportAlignmentOperation({
        ...PROPS,
        output: {...PROPS.output, populations: PROPS.populations}
      })
  ).toThrow();
  const cpu = new CPUPopulationSupportAlignmentOperation(PROPS);
  const gpu = new GPUPopulationSupportAlignmentOperation(PROPS);
  expect(cpu.inputs).toEqual(gpu.inputs);
  for (const output of Object.values(gpu.outputs)) {
    expect(cpu.outputs[output.value]).toMatchObject({format: output.format, length: output.length});
  }
});

it('rejects mismatched output lengths, output aliases and foreign graphs at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const foreign = new GPUCommandGraph(device);
    const props = {
      id: 'support-boundary',
      populations: createTransientView(graph, 'input', 'float32x2', 3),
      sourceIds: createTransientView(graph, 'sources', 'uint32', 3),
      targetIds: createTransientView(graph, 'targets', 'uint32', 3),
      output: {
        populations: createTransientView(graph, 'output', 'float32x2', 3),
        validity: createTransientView(graph, 'validity', 'uint32', 3),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    };
    expect(
      () =>
        new GPUPopulationSupportAlignment({
          ...props,
          output: {
            ...props.output,
            validity: createTransientView(graph, 'short-validity', 'uint32', 2)
          }
        })
    ).toThrow(/support-boundary output.validity must have 3 rows/);
    expect(
      () =>
        new GPUPopulationSupportAlignment({
          ...props,
          output: {...props.output, populations: props.populations}
        })
    ).toThrow(/support-boundary.*share buffers/);
    expect(
      () =>
        new GPUPopulationSupportAlignment({
          ...props,
          output: {
            ...props.output,
            status: createTransientView(foreign, 'foreign-status', 'uint32x4', 1)
          }
        })
    ).toThrow(/support-boundary.*one graph/);
  } finally {
    device.destroy();
  }
});

it('declares indexed GPU joins and parallel row kernels within core binding limits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUPopulationSupportAlignment({
      id: 'support',
      populations: createTransientView(graph, 'input', 'float32x2', 4097),
      sourceIds: createTransientView(graph, 'sources', 'uint32', 4097),
      targetIds: createTransientView(graph, 'targets', 'uint32', 257),
      output: {
        populations: createTransientView(graph, 'output', 'float32x2', 257),
        validity: createTransientView(graph, 'validity', 'uint32', 257),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.some(kernel => kernel.id.startsWith('support-sources-'))).toBe(true);
    expect(kernels.some(kernel => kernel.id.startsWith('support-targets-'))).toBe(true);
    const alignment = kernels.find(kernel => kernel.id === 'support-align')!;
    expect(alignment.source).not.toMatch(/\b(for|while|loop)\b/);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    // Storage checks run before any sort/scan nodes or scratch are created.
    const storageLimit = device.limits.maxStorageBufferBindingSize;
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 16});
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /support source key scratch.*reduce sourceIds/
      );
    } finally {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: storageLimit});
    }
  } finally {
    device.destroy();
  }
});

it('the legacy recipe delegates to the indexed contributor and retains its old result keys', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const recipe = addPopulationSupportAlignmentRecipe(graph, {
      populations: createTransientView(graph, 'input', 'float32x2', 3),
      sourceIds: createTransientView(graph, 'sources', 'uint32', 3),
      targetIds: createTransientView(graph, 'targets', 'uint32', 2)
    });
    expect(recipe.contributors).toHaveLength(1);
    expect(recipe.contributors[0]).toBeInstanceOf(GPUPopulationSupportAlignment);
    expect(Object.keys(recipe.outputs)).toEqual(['populations', 'validity']);
    expect(recipe.status.format).toBe('uint32x4');
    expect(recipe.provenance.alignment).toBe('exact-identifier-join');
  } finally {
    device.destroy();
  }
});
