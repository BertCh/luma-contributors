// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addSegregationCounterfactualRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {
  addIsovistsRecipe,
  getGPUIsovistRayCount,
  type GPUIsovistsOutput
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getNodeCount(graph: GPUCommandGraph): number {
  return (graph as unknown as {nodes: unknown[]}).nodes.length;
}

it('UCT-03 local measures use polar rays and keep the supplied destination', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const localMeasures = createTransientView(graph, 'local', 'float32x4', 1);
    const result = addIsovistsRecipe(graph, {
      positions: createTransientView(graph, 'origins', 'float32x2', 1),
      segments: createTransientView(graph, 'obstacles', 'float32x4', 1),
      rayCount: 4,
      maximumDistance: 0.1,
      output: {localMeasures}
    });
    expect(result.outputs.localMeasures).toBe(localMeasures);
    const measures = getGPUCommandNodeKernelSources(
      (graph as unknown as {nodes: unknown[]}).nodes
    ).find(kernel => kernel.id.endsWith('-measures'))!;
    expect(measures.source).toContain('cos(');
    expect(measures.source).not.toContain('endpoints[');
    // Float32 spacing at this origin is 0.5: all four drawing endpoints round to the origin.
    expect(Math.fround(5_000_000 + 0.1) - 5_000_000).toBe(0);
  } finally {
    device.destroy();
  }
});

it('UCT-03 retired isovist destinations reject before allocating recipe storage', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      positions: createTransientView(graph, 'origins', 'float32x2', 1),
      segments: createTransientView(graph, 'obstacles', 'float32x4', 1),
      rayCount: 4,
      maximumDistance: 0.1,
      output: {localMeasures: createTransientView(graph, 'new-measures', 'float32x4', 1)}
    };
    const state = graph as unknown as {buffers: Map<string, unknown>};
    const before = [...state.buffers.keys()];
    const removedMeasures = {
      ...props,
      output: {...props.output, measures: props.output.localMeasures}
    };
    expect(() => addIsovistsRecipe(graph, removedMeasures)).toThrow(
      /output.measures was removed; use output.localMeasures/
    );
    const removedOutputs = {...props, outputs: props.output};
    expect(() => addIsovistsRecipe(graph, removedOutputs)).toThrow(
      /outputs was removed; use output/
    );
    expect(getNodeCount(graph)).toBe(0);
    expect([...state.buffers.keys()]).toEqual(before);
  } finally {
    device.destroy();
  }
});

it.each([
  ['positions', 'strided'],
  ['segments', 'strided'],
  ['positions', 'wrong-format'],
  ['segments', 'wrong-format']
] as const)('spatial-4:F14 rejects %s %s before allocation and permits a packed retry', (name, invalidLayout) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const segments = createTransientView(graph, 'segments', 'float32x4', 2);
    const storage = graph.createTransientBuffer({
      id: 'invalid-input',
      byteLength: 64,
      usage: Buffer.STORAGE
    });
    const format = name === 'positions' ? 'float32x2' : 'float32x4';
    const invalid = graph.createDataView(storage, {
      format:
        invalidLayout === 'strided' ? format : name === 'positions' ? 'float32x4' : 'float32x2',
      length: 2,
      ...(invalidLayout === 'strided' ? {byteStride: name === 'positions' ? 16 : 32} : {})
    });
    const state = graph as unknown as {buffers: Map<string, unknown>};
    const bufferCount = state.buffers.size;
    const props = {id: 'iso', positions, segments, rayCount: 4, maximumDistance: 1};
    expect(() => addIsovistsRecipe(graph, {...props, [name]: invalid})).toThrow();
    expect(getNodeCount(graph)).toBe(0);
    const rejectedBufferCount = state.buffers.size;
    expect(() => addIsovistsRecipe(graph, props)).not.toThrow();
    expect(rejectedBufferCount).toBe(bufferCount);
    expect(getNodeCount(graph)).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R1 counterfactual rejects indices and decomposition in one physical buffer before adding nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const storage = graph.createTransientBuffer({
      id: 'results',
      byteLength: 272,
      usage: Buffer.STORAGE
    });
    const indices = graph.createDataView(storage, {format: 'float32', length: 4});
    const decomposition = graph.createDataView(storage, {
      format: 'float32x4',
      byteOffset: 256,
      length: 1
    });
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        baseline: createTransientView(graph, 'baseline', 'float32x2', 2),
        final: createTransientView(graph, 'final', 'float32x2', 2),
        supportId: 'two-units',
        method: 'composition',
        output: {indices, decomposition}
      })
    ).toThrow(/must not share buffers with inputs/);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('R2 disjoint imported scenarios require one handle (separateHandles=%s)', separateHandles => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    const first = importGraphBuffer(graph, 'first-import', buffer, 'float32x2', 4);
    const second = separateHandles
      ? importGraphBuffer(graph, 'second-import', buffer, 'float32x2', 4)
      : first;
    const compositionOnly = graph.createDataView(first.buffer, {format: 'float32x2', length: 2});
    const spatialOnly = graph.createDataView(second.buffer, {
      format: 'float32x2',
      byteOffset: 16,
      length: 2
    });
    const baseline = createTransientView(graph, 'baseline', 'float32x2', 2);
    const final = createTransientView(graph, 'final', 'float32x2', 2);
    const add = () =>
      addSegregationCounterfactualRecipe(graph, {
        baseline,
        final,
        supportId: 'two-units',
        method: 'composition',
        output: {compositionOnly, spatialOnly}
      });
    if (separateHandles) {
      expect(add).toThrow(/must use one graph buffer handle/);
      expect(getNodeCount(graph)).toBe(0);
    } else {
      expect(add).not.toThrow();
      expect(getNodeCount(graph)).toBeGreaterThan(0);
    }
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('R4 isovists reject aliased angle and measure outputs before adding nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const angles = createTransientView(graph, 'shared-results', 'float32', 4);
    const measures = graph.createDataView(angles.buffer, {format: 'float32x4', length: 1});
    expect(() =>
      addIsovistsRecipe(graph, {
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        segments: createTransientView(graph, 'wall', 'float32x4', 1),
        rayCount: 4,
        maximumDistance: 1,
        output: {angles, localMeasures: measures}
      })
    ).toThrow(/logical byte ranges must not overlap/);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    device.destroy();
  }
});

it.each([
  0, 8
])('counterfactual rejects overlapping scenario ranges at byte offset %s before adding nodes', byteOffset => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const baseline = createTransientView(graph, 'baseline', 'float32x2', 2);
    const final = createTransientView(graph, 'final', 'float32x2', 2);
    const storage = graph.createTransientBuffer({
      id: 'scenarios',
      byteLength: 64,
      usage: Buffer.STORAGE
    });
    const compositionOnly = graph.createDataView(storage, {format: 'float32x2', length: 2});
    const spatialOnly = graph.createDataView(storage, {format: 'float32x2', byteOffset, length: 2});
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        baseline,
        final,
        supportId: 'two-units',
        method: 'composition',
        output: {compositionOnly, spatialOnly}
      })
    ).toThrow(/compositionOnly and spatialOnly logical byte ranges must not overlap/);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    device.destroy();
  }
});

it('counterfactual accepts disjoint scenario subviews in one allocation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const baseline = createTransientView(graph, 'baseline', 'float32x2', 2);
    const final = createTransientView(graph, 'final', 'float32x2', 2);
    const storage = graph.createTransientBuffer({
      id: 'scenarios',
      byteLength: 32,
      usage: Buffer.STORAGE
    });
    const compositionOnly = graph.createDataView(storage, {format: 'float32x2', length: 2});
    const spatialOnly = graph.createDataView(storage, {
      format: 'float32x2',
      byteOffset: 16,
      length: 2
    });
    const result = addSegregationCounterfactualRecipe(graph, {
      baseline,
      final,
      supportId: 'two-units',
      method: 'composition',
      output: {compositionOnly, spatialOnly}
    });
    expect(result.counterfactuals).toEqual({compositionOnly, spatialOnly});
    expect(getNodeCount(graph)).toBeGreaterThan(2);
  } finally {
    device.destroy();
  }
});

it('counterfactual rejects scenario aliases imported through different graph handles', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    const compositionOnly = importGraphBuffer(graph, 'first-import', buffer, 'float32x2', 2);
    const spatialOnly = importGraphBuffer(graph, 'second-import', buffer, 'float32x2', 2);
    expect(compositionOnly.buffer).not.toBe(spatialOnly.buffer);
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        baseline: createTransientView(graph, 'baseline', 'float32x2', 2),
        final: createTransientView(graph, 'final', 'float32x2', 2),
        supportId: 'two-units',
        method: 'composition',
        output: {compositionOnly, spatialOnly}
      })
    ).toThrow(/compositionOnly and spatialOnly logical byte ranges must not overlap/);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('counterfactual rejects an input/output alias and a short decomposition before adding nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const baseline = createTransientView(graph, 'baseline', 'float32x2', 2);
    const final = createTransientView(graph, 'final', 'float32x2', 2);
    const props = {baseline, final, supportId: 'two-units', method: 'composition' as const};
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        ...props,
        output: {spatialOnly: baseline}
      })
    ).toThrow(/spatialOnly and baseline logical byte ranges must not overlap/);
    expect(getNodeCount(graph)).toBe(0);
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        ...props,
        output: {decomposition: createTransientView(graph, 'short', 'float32x4', 0)}
      })
    ).toThrow(/must have 1 rows/);
    expect(getNodeCount(graph)).toBe(0);
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        ...props,
        output: {decomposition: createTransientView(graph, 'corrected', 'float32x4', 1)}
      })
    ).not.toThrow();
    expect(getNodeCount(graph)).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it.each([-1, 0, 1, 2])('vertex isovists reject base rayCount %s before adding nodes', rayCount => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      addIsovistsRecipe(graph, {
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        segments: createTransientView(graph, 'obstacles', 'float32x4', 1),
        rayCount,
        sampling: 'vertices',
        maximumDistance: 10
      })
    ).toThrow(/rayCount must be an integer/);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    device.destroy();
  }
});

it('vertex isovists reject a short distance override before adding classification or BVH nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const segments = createTransientView(graph, 'obstacles', 'float32x4', 1);
    const distances = createTransientView(graph, 'distances', 'float32', 3);
    expect(() =>
      addIsovistsRecipe(graph, {
        positions,
        segments,
        rayCount: 3,
        sampling: 'vertices',
        maximumDistance: 10,
        obstacleIndex: 'bvh',
        output: {distances}
      })
    ).toThrow(/must have 9 rows/);
    expect(getNodeCount(graph)).toBe(0);
    expect(() =>
      addIsovistsRecipe(graph, {
        positions,
        segments,
        rayCount: 3,
        sampling: 'vertices',
        maximumDistance: 10,
        obstacleIndex: 'bvh',
        output: {distances: createTransientView(graph, 'corrected-distances', 'float32', 9)}
      })
    ).not.toThrow();
    expect(getNodeCount(graph)).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it.each([
  ['endpoints', 'float32x2', 3, 9],
  ['angles', 'float32', 3, 9],
  ['segments', 'uint32', 3, 9],
  ['validity', 'uint32', 3, 9],
  ['localMeasures', 'float32x4', 2, 1],
  ['viewpointClassifications', 'uint32', 2, 1]
] as const)('isovists preflight %s before graph work', (name, format, length, expectedLength) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const provided = createTransientView(graph, 'provided', format, length);
    expect(() =>
      addIsovistsRecipe(graph, {
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        segments: createTransientView(graph, 'obstacles', 'float32x4', 1),
        rayCount: 3,
        sampling: 'vertices',
        maximumDistance: 10,
        obstacleIndex: 'bvh',
        output: {[name]: provided} as Partial<GPUIsovistsOutput>
      })
    ).toThrow(`must have ${expectedLength} rows`);
    expect(getNodeCount(graph)).toBe(0);
  } finally {
    device.destroy();
  }
});

it('isovist sizing helper distinguishes base and expanded rays, including both limits', () => {
  expect(getGPUIsovistRayCount(3, 1)).toBe(3);
  expect(getGPUIsovistRayCount(3, 1, 'vertices')).toBe(9);
  expect(getGPUIsovistRayCount(4, 682, 'vertices')).toBe(4096);
  expect(() => getGPUIsovistRayCount(5, 682, 'vertices')).toThrow(/at most 4096/);
  expect(getGPUIsovistRayCount(65536, 16777216, 'uniform')).toBe(65536);
  expect(() => getGPUIsovistRayCount(65537, 1)).toThrow(/rayCount/);
  for (const baseRayCount of [-1, 0, 1, 2, 3.5, NaN, Infinity]) {
    expect(() => getGPUIsovistRayCount(baseRayCount, 1, 'vertices')).toThrow(/rayCount/);
  }
  for (const segmentCount of [-1, 0, 1.5, NaN, Infinity, 16777217]) {
    expect(() => getGPUIsovistRayCount(3, segmentCount)).toThrow(/segments/);
  }
});

it('isovists use the sizing helper count for all ray views and retain base spacing in provenance', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const segments = createTransientView(graph, 'obstacles', 'float32x4', 1);
    const result = addIsovistsRecipe(graph, {
      positions,
      segments,
      rayCount: 3,
      sampling: 'vertices',
      maximumDistance: 10
    });
    for (const name of ['endpoints', 'distances', 'segments', 'validity', 'angles'] as const) {
      expect(result.outputs[name].length).toBe(18);
    }
    expect(result.outputs.localMeasures.length).toBe(2);
    expect(result.outputs.viewpointClassifications.length).toBe(2);
    expect(result.provenance.baseRayCount).toBe(3);
    expect(result.provenance.rayCount).toBe(getGPUIsovistRayCount(3, 1, 'vertices'));
    expect(result.provenance.angularSpacing).toBe((2 * Math.PI) / 3);
  } finally {
    device.destroy();
  }
});
