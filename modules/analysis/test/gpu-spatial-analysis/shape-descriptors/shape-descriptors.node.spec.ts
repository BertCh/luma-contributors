// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES,
  GPUShapeDescriptors,
  getGPUShapeDescriptorsParameterValues
} from '../../../src/gpu-spatial-analysis/shape-descriptors/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('review D-19: orientation guards atan2(0, 0) and keeps the kernel parseable', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors'});
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 3),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {orientation: createTransientView(graph, 'orientation', 'float32', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const source = sources
    .map(entry => entry.source)
    .find(text => text.includes('orientation[orientationOffset'));
  expect(source).toBeDefined();
  expect(() => new WgslReflect(source!)).not.toThrow();
  // The second argument is a difference: guardedAtan2 treats a zero of either sign as +0.0.
  expect(source).toContain(
    'select(0.0, 0.5 * guardedAtan2(2.0 * xy, xx - yy), (xy != 0.0) || (xx != yy))'
  );
});

it("convexityMethod 'auto' decides from the declared largest feature when given", () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors-auto'});
  let serial = 0;
  const create = (overrides: {maximumVerticesPerFeature?: number; convexityMethod?: never}) =>
    new GPUShapeDescriptors({
      // Ten features of eight vertices: the average is far below the average-size threshold.
      positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 80),
      ringOffsets: createTransientView(graph, `ring-offsets-${serial++}`, 'uint32', 11),
      polygonOffsets: createTransientView(graph, `polygon-offsets-${serial++}`, 'uint32', 11),
      featureOffsets: createTransientView(graph, `feature-offsets-${serial++}`, 'uint32', 11),
      parameters: createTransientView(graph, `parameters-${serial++}`, 'float32', 4),
      output: {convexity: createTransientView(graph, `convexity-${serial++}`, 'float32', 10)},
      ...overrides
    });
  // Without a declaration the average rule applies, as before.
  expect(create({}).useMonotoneChainConvexity).toBe(false);
  expect(create({maximumVerticesPerFeature: 8}).useMonotoneChainConvexity).toBe(false);
  expect(
    create({maximumVerticesPerFeature: GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES - 1})
      .useMonotoneChainConvexity
  ).toBe(false);
  expect(
    create({maximumVerticesPerFeature: GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES})
      .useMonotoneChainConvexity
  ).toBe(true);
  // An explicit method ignores the declaration.
  expect(
    create({maximumVerticesPerFeature: 200000, convexityMethod: 'gift-wrapping' as never})
      .useMonotoneChainConvexity
  ).toBe(false);
  expect(() => create({maximumVerticesPerFeature: 0})).toThrow(/maximumVerticesPerFeature/);
  expect(() => create({maximumVerticesPerFeature: 1.5})).toThrow(/maximumVerticesPerFeature/);
});

it('normalizes finite second moments before eigenvalue squares and sums', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {elongation: createTransientView(graph, 'elongation', 'float32', 1)}
  });
  const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(entry =>
    entry.source.includes('let momentXX =')
  )!.source;
  expect(source).toContain('let exponent = frexp(magnitude).exp;');
  expect(source).toContain('let xx = ldexp(momentXX, -exponent);');
  expect(source).toContain('let yy = ldexp(momentYY, -exponent);');
  expect(source).toContain('let xy = ldexp(momentXY, -exponent);');
  expect(source).not.toContain('momentXX /');
  expect(source.indexOf('let xx = ldexp(momentXX, -exponent);')).toBeLessThan(
    source.indexOf('let radius = sqrt(')
  );
});

it('rejects unrepresentable runtime thresholds and overlapping output binding prefixes', () => {
  for (const sliverThreshold of [NaN, Infinity, 1e40, -1, -1e-100]) {
    expect(() => getGPUShapeDescriptorsParameterValues({sliverThreshold})).toThrow(
      /sliverThreshold/
    );
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'outputs', 'float32', 2);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      areas: graph.createDataView(backing.buffer, {
        format: 'float32' as const,
        byteOffset: 0,
        length: 1
      }),
      perimeters: graph.createDataView(backing.buffer, {
        format: 'float32' as const,
        byteOffset: 4,
        length: 1
      })
    }
  };
  expect(() => new GPUShapeDescriptors(props)).toThrow(/aligned binding.*must not overlap/);
  device.destroy();
});

it('preflights aligned views and nested geometry-measure scratch before any declaration', () => {
  for (const failure of ['feature scratch', 'aligned positions'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const featureCount = failure === 'feature scratch' ? 2_097_153 : 1;
    const positions = createTransientView(
      graph,
      'positions',
      'float32x2',
      failure === 'aligned positions' ? 16_777_217 : 4
    );
    const contributor = new GPUShapeDescriptors({
      positions:
        failure === 'aligned positions'
          ? graph.createDataView(positions.buffer, {
              format: 'float32x2',
              byteOffset: 4,
              length: 16_777_216
            })
          : positions,
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', featureCount + 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {orientation: createTransientView(graph, 'orientation', 'float32', featureCount)}
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      failure === 'feature scratch'
        ? /geometry measure feature scratch needs 134217792 bytes/
        : /positions chunk 0 needs 134217732 bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('splits compactness exponents before any perimeter multiplication', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {polsbyPopper: createTransientView(graph, 'compactness', 'float32', 1)}
  });
  const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(entry =>
    entry.source.includes('getScaledQuotient(area, perimeter, perimeter, 0)')
  )!.source;
  expect(source).toContain(
    'getScaledQuotient(area, perimeter, perimeter, 0) * (4.0 * 3.14159265358979)'
  );
  expect(source).toContain('numerator.exp - firstDivisor.exp - secondDivisor.exp');
  expect(source).not.toContain('area / perimeter');
});

it('accepts areas and orientation in disjoint logical ranges of one alignment window', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'packed-fields', 'float32', 2);
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      areas: graph.createDataView(backing.buffer, {format: 'float32', length: 1}),
      orientation: graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 4,
        length: 1
      })
    }
  });
  expect(() => graph.add(contributor.getCommandNodes(graph))).not.toThrow();
  device.destroy();
});

it('generates a fixed number of moment pipelines as tile count grows and checks all WGSL identifiers', () => {
  const counts: number[] = [];
  for (const rowCount of [4096, 32768, 1_048_576]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUShapeDescriptors({
      momentMethod: 'tiled',
      positions: createTransientView(graph, 'positions', 'float32x2', rowCount),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {orientation: createTransientView(graph, 'orientation', 'float32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id.endsWith('-moments-tiles'))).toBe(true);
    expect(() => graph.add(nodes)).not.toThrow();
    counts.push(
      new Set(getGPUCommandNodePipelineKeys(nodes.filter(node => node.id.includes('-moments-'))))
        .size
    );
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      const code = source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
      expect(
        (code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(word => WGSL_RESERVED_WORDS.has(word))
      ).toEqual([]);
      expect(code).not.toMatch(/(?:^|[=,(])\s*(?:NaN|Infinity|inf)\b/m);
      expect(() => new WgslReflect(source)).not.toThrow();
    }
    device.destroy();
  }
  expect(new Set(counts).size).toBe(1);
});

it('preflights convexity-specific prefilter scratch before any allocation', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 65536});
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUShapeDescriptors({
    convexityMethod: 'monotone-chain',
    positions: createTransientView(graph, 'positions', 'float32x2', 8192),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {convexity: createTransientView(graph, 'convexity', 'float32', 1)}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /hull prefilter scratch needs 65544 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});
