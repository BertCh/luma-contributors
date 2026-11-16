// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getDotTriangleCumulativeAreaNodes} from '../../../src/gpu-spatial-analysis/dot-density/dot-density-triangle-cdf';
import {
  GPUDotDensity,
  type GPUDotDensityProps
} from '../../../src/gpu-spatial-analysis/dot-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(graph: GPUCommandGraph, vertexCount = 12): GPUDotDensityProps {
  return {
    id: 'dots',
    positions: createTransientView(graph, 'positions', 'float32x2', vertexCount),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 4),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 4),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 4),
    parameters: createTransientView(graph, 'parameters', 'uint32', 8),
    values: createTransientView(graph, 'values', 'float32', 3),
    output: {
      positions: createTransientView(graph, 'dot-positions', 'float32x2', 10),
      dots: {
        ids: createTransientView(graph, 'dot-ids', 'uint32', 10),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  };
}

it('uses the same CDF sources and pipelines as the feature count grows', () => {
  const pipelineKeys = new Set<string>();
  for (const featureCount of [1, 3, 1024]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = getDotTriangleCumulativeAreaNodes(graph, {
      id: 'cdf',
      operation: 'GPUDotDensity',
      triangleAreas: createTransientView(graph, 'areas', 'float32', 4096),
      featureTriangleOffsets: createTransientView(graph, 'offsets', 'uint32', featureCount + 1),
      featureValid: createTransientView(graph, 'validity', 'uint32', featureCount),
      featureAreas: createTransientView(graph, 'totals', 'float32', featureCount),
      cumulativeAreas: createTransientView(graph, 'cdf', 'uint32', 4096 + featureCount)
    });
    expect(() => graph.add(nodes)).not.toThrow();
    for (const key of getGPUCommandNodePipelineKeys(nodes)) pipelineKeys.add(key);
    const ownSources = getGPUCommandNodeKernelSources(nodes).filter(
      entry => entry.operation === 'GPUDotDensity'
    );
    expect(ownSources).toHaveLength(3);
    for (const {source, constants} of ownSources) {
      expect(() => new WgslReflect(source)).not.toThrow();
      expect(constants).toEqual({
        INVOCATION_COUNT: 4096,
        WGSL_KERNEL_DISPATCH_X: 16,
        WGSL_KERNEL_DISPATCH_Y: 1
      });
      const identifiers =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      expect(source).not.toMatch(/\b(?:nan|inf|Infinity|NaN)\b/);
    }
    device.destroy();
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = getDotTriangleCumulativeAreaNodes(graph, {
    id: 'cdf',
    operation: 'GPUDotDensity',
    triangleAreas: createTransientView(graph, 'areas', 'float32', 4096),
    featureTriangleOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    featureValid: createTransientView(graph, 'validity', 'uint32', 1),
    featureAreas: createTransientView(graph, 'totals', 'float32', 1),
    cumulativeAreas: createTransientView(graph, 'cdf', 'uint32', 4096)
  });
  expect(pipelineKeys.size).toBe(new Set(getGPUCommandNodePipelineKeys(nodes)).size);
  device.destroy();
});

it('rejects dot output aliases including aligned prefixes at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        output: {...props.output, categories: props.output.dots.ids}
      })
  ).toThrow(/must not overlap/);
  const backing = createTransientView(graph, 'status-backing', 'uint32', 65);
  const count = graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 0, length: 1});
  const overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        output: {...props.output, dots: {...props.output.dots, status: {count, overflow}}}
      })
  ).toThrow(/aligned binding.*must not overlap/);
  device.destroy();
});

it('rejects oversized triangulation and actual aligned inputs before graph declarations', () => {
  for (const failure of ['scratch', 'aligned input'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, failure === 'scratch' ? 3_000_000 : 12);
    if (failure === 'aligned input') {
      const backing = createTransientView(graph, 'large-parameters', 'uint32', 33_554_433);
      props.parameters = graph.createDataView(backing.buffer, {
        format: 'uint32',
        byteOffset: 4,
        length: 33_554_432
      });
    }
    const contributor = new GPUDotDensity(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      failure === 'scratch' ? /scratch nodes/ : /input 5.*needs 134217732 bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('uses an atomic shared count clamp flag for concurrent slots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const sources = getGPUCommandNodeKernelSources(
    new GPUDotDensity(createProps(graph)).getCommandNodes(graph)
  );
  const counts = sources.find(entry => entry.source.includes('const SLOT_LIMIT:'))!.source;
  expect(counts).toContain('clampFlag: array<atomic<u32>>');
  expect(counts).toContain('atomicStore(&clampFlag[clampFlagOffset], 1u)');
  device.destroy();
});

it('validates empty topology descriptors at construction before nested triangulation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        ringOffsets: createTransientView(graph, 'missing-terminal', 'uint32', 0)
      })
  ).toThrow(/dots ringOffsets rows must be an integer in/);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        polygonOffsets: createTransientView(graph, 'no-polygons', 'uint32', 1)
      })
  ).toThrow(/dots positions must be empty/);
  device.destroy();
});

it('accepts packed dot fields written by separate passes but rejects co-bound prefixes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const backing = createTransientView(graph, 'shared-dot-fields', 'uint32', 30);
  props.output.positions = graph.createDataView(backing.buffer, {format: 'float32x2', length: 10});
  props.output.dots.ids = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 80,
    length: 10
  });
  const contributor = new GPUDotDensity(props);
  expect(() => graph.add(contributor.getCommandNodes(graph))).not.toThrow();
  device.destroy();
});

it('rejects normalized masks whose cooperative dispatch exceeds uint32 at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 0);
  props.featureOffsets = createTransientView(graph, 'many-empty-features', 'uint32', 16_777_217);
  props.values = createTransientView(graph, 'many-values', 'float32', 16_777_216);
  props.mask = {
    weights: createTransientView(graph, 'one-cell-mask', 'float32', 1),
    width: 1,
    height: 1,
    normalizeFeatures: true
  };
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPUDotDensity(props)).toThrow(
    /dots normalized feature count must be at most 16777215/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('derives the mask barrier guard from the workgroup ID, never the invocation index', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  props.mask = {
    weights: createTransientView(graph, 'mask', 'float32', 1),
    width: 1,
    height: 1,
    normalizeFeatures: true
  };
  const sources = getGPUCommandNodeKernelSources(new GPUDotDensity(props).getCommandNodes(graph));
  const source = sources.find(entry =>
    entry.source.includes('var<workgroup> partialMinimum:')
  )!.source;
  // WGSL does not prove (workgroupIndex * 256 + lane) / 256 uniform. Merely checking
  // that the guard precedes a barrier missed the actual Chromium validation failure.
  const code = source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
  expect(code).toMatch(/let feature = workgroupIndex;/);
  expect(code).not.toMatch(/let feature = index\s*\//);
  const workgroupIndex = code.match(/let workgroupIndex\s*=([^;]+);/)![1];
  expect(workgroupIndex).toContain('workgroupId');
  expect(workgroupIndex).not.toMatch(/localInvocationIndex|\bindex\b/);
  expect(source).toContain(
    'if (feature >= arrayLength(&featureOffsets) - featureOffsetsOffset - 1u) { return; }'
  );
  expect(source.indexOf('if (feature >=')).toBeLessThan(source.indexOf('workgroupBarrier();'));
  for (const {source} of sources) {
    const code = source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
    expect(
      (code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(word => WGSL_RESERVED_WORDS.has(word))
    ).toEqual([]);
  }
  device.destroy();
});

it('keeps mask normalization sources fixed and dispatches the declared workgroup count', () => {
  const sources = new Set<string>();
  for (const featureCount of [1, 65535, 65536]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    props.featureOffsets = createTransientView(
      graph,
      'normalization-features',
      'uint32',
      featureCount + 1
    );
    props.values = createTransientView(graph, 'normalization-values', 'float32', featureCount);
    props.mask = {
      weights: createTransientView(graph, 'mask', 'float32', 1),
      width: 1,
      height: 1,
      normalizeFeatures: true
    };
    const nodes = new GPUDotDensity(props).getCommandNodes(graph);
    const kernel = getGPUCommandNodeKernelSources(nodes).find(entry =>
      entry.source.includes('var<workgroup> partialMinimum:')
    )!;
    sources.add(kernel.source);
    expect(kernel.constants['INVOCATION_COUNT']).toBe(featureCount * 256);
    // Dispatch padding may add workgroups, but every requested feature has one, with a
    // workgroup-uniform bounds guard for the extra groups in the bounded 2D layout.
    const workgroupCount =
      kernel.constants['WGSL_KERNEL_DISPATCH_X'] * kernel.constants['WGSL_KERNEL_DISPATCH_Y'];
    expect(workgroupCount).toBeGreaterThanOrEqual(featureCount);
    if (featureCount === 1 || featureCount === 65535) expect(workgroupCount).toBe(featureCount);
    device.destroy();
  }
  expect(sources.size).toBe(1);
});
