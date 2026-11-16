// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import type {ShaderLayout} from '@luma.gl/core';
import {GPUScalarScale} from '../../../src/dataframe-analysis/column-classification/gpu-scalar-scale';
import {GPUColorScale} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {GPUBivariateClassification} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {GPUColumnQuantiles} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import type {GPUClassBreaksMethod} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {getGraphViewChunks} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  type RecordedGPUKernelSource
} from '../../utils/gpu-contributor-test-utils';

import {getReservedWGSLIdentifiers} from './wgsl-reserved-words';

function getParsedKernelSources(nodes: readonly unknown[]): Map<string, string> {
  const sources = new Map<string, string>();
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    validateKernelBindings(kernel);
    sources.set(kernel.id, kernel.source);
  }
  return sources;
}

function validateKernelBindings({id, source, shaderLayout}: RecordedGPUKernelSource): void {
  expect(getReservedWGSLIdentifiers(source), id).toEqual([]);
  const reflection = new WgslReflect(source);
  const accessed = reflection.entry.compute[0].resources.map(
    resource => `${resource.group}:${resource.binding}:${resource.name}`
  );
  const supplied = (shaderLayout as ShaderLayout).bindings!.map(
    binding => `${binding.group ?? 0}:${binding.location}:${binding.name}`
  );
  // WebGPU infers layouts from entry-point accesses, not every module-scope declaration.
  expect(accessed.sort(), id).toEqual(supplied.sort());
}

function createFixture(lengths = [3, 0, 4]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const packed = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const rows = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    chunkLengths = lengths
  ) =>
    createVectorView(
      name,
      format,
      chunkLengths.map((length, chunkIndex) => packed(`${name}-${chunkIndex}`, format, length))
    );
  const values = rows('values', 'float32');
  const mask = rows('mask', 'uint32');
  const domain = packed('domain', 'float32', 9);
  const palette = packed('palette', 'uint32', 64);
  const parameters = packed('parameters', 'float32', 32);
  return {device, graph, packed, rows, values, mask, domain, palette, parameters};
}

it.each<GPUClassBreaksMethod>([
  'equal-interval',
  'quantile',
  'box-plot',
  'custom',
  'standard-deviation',
  'head-tail',
  'natural-breaks',
  'maximum-breaks'
])('matches %s shader layouts to statically accessed entry-point resources', method => {
  const {device, graph, values, mask, parameters, packed} = createFixture([0, 2, 0, 4]);
  try {
    const nodes = new GPUClassBreaks({
      values: method === 'maximum-breaks' ? packed('sort-values', 'float32', 6) : values,
      mask: method === 'maximum-breaks' ? undefined : mask,
      parameters,
      maximumClassCount: 6,
      methods: [method],
      naturalBreaksBinCount: 16,
      output: {
        breaks: packed('breaks', 'float32', 7),
        classCount: packed('class-count', 'uint32', 1),
        classCounts: packed('counts', 'uint32', 6)
      }
    }).getCommandNodes(graph);
    getParsedKernelSources(nodes);
  } finally {
    device.destroy();
  }
});

it('dispatches apply contributors per nonempty chunk and resets shared counts once', () => {
  const fixture = createFixture();
  const {device, graph, rows, packed, values, mask, domain, palette, parameters} = fixture;
  try {
    const scalar = new GPUScalarScale({
      values,
      mask,
      domain,
      range: packed('range', 'float32', 2),
      parameters,
      maximumDomainCount: 8,
      output: {values: rows('scaled', 'float32'), validity: rows('validity', 'uint32')}
    });
    const color = new GPUColorScale({
      values,
      mask,
      domain,
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      output: {
        colors: rows('colors', 'uint32'),
        classIndices: rows('classes', 'uint32'),
        classCounts: packed('counts', 'uint32', 8)
      }
    });
    const bivariate = new GPUBivariateClassification({
      valuesX: values,
      valuesY: rows('values-y', 'float32'),
      mask,
      alphaValues: rows('alpha', 'float32'),
      breaksX: domain,
      breaksY: packed('domain-y', 'float32', 9),
      palette,
      parameters,
      maximumClassCount: 8,
      output: {
        colors: rows('joint-colors', 'uint32'),
        classIds: rows('joint-classes', 'uint32'),
        classCounts: packed('joint-counts', 'uint32', 64)
      }
    });
    for (const contributor of [scalar, color, bivariate]) {
      const nodes = contributor.getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      expect([...sources.keys()].some(key => key.includes('chunk-1'))).toBe(false);
      expect([...sources.keys()].some(key => key.includes('chunk-2'))).toBe(true);
      expect([...sources.keys()].filter(key => key.endsWith('-init')).length).toBe(
        contributor === scalar ? 0 : 1
      );
    }
    expect(
      () => new GPUScalarScale({...scalar.props, mask: rows('wrong-mask', 'uint32', [4, 0, 3])})
    ).toThrow(/scalar-scale mask must match the source topology/);
    expect(
      () =>
        new GPUColorScale({...color.props, output: {colors: rows('wrong-colors', 'uint32', [7])}})
    ).toThrow(/output.colors must match the source topology/);
    expect(
      () =>
        new GPUBivariateClassification({
          ...bivariate.props,
          valuesY: rows('wrong-y', 'float32', [2, 1, 4])
        })
    ).toThrow(/valuesY must match the source topology/);
    expect(() => scalar.getCommandNodes(new GPUCommandGraph(device))).toThrow(/target graph/);
    expect(() => new GPUScalarScale({...scalar.props, output: {values}})).toThrow(
      /share buffers|byte ranges must not overlap/
    );
    expect(
      () =>
        new GPUColorScale({
          ...color.props,
          output: {
            colors: rows('aliased', 'uint32', [3, 0, 4]),
            classIndices: color.props.output.colors
          }
        })
    ).not.toThrow();
    expect(
      () =>
        new GPUColorScale({
          ...color.props,
          output: {colors: color.props.output.colors, classIndices: color.props.output.colors}
        })
    ).toThrow(/share buffers|byte ranges must not overlap/);
  } finally {
    device.destroy();
  }
});

it('streams quantile chunks between shared select nodes without a full-column binding', () => {
  // Logical 40 million rows exceed the default 128 MiB single binding; no buffers are allocated.
  const {device, graph, values, mask, packed, parameters} = createFixture([
    20_000_000, 0, 20_000_000
  ]);
  try {
    const contributor = new GPUColumnQuantiles({
      values,
      mask,
      parameters,
      quantileCount: 3,
      rationalProbabilities: packed('ratios', 'uint32', 6),
      output: {
        quantiles: packed('quantiles', 'float32', 3),
        validCount: packed('valid-count', 'uint32', 1)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    const sources = getParsedKernelSources(nodes);
    expect([...sources.keys()].filter(key => key.includes('-histogram-')).length).toBe(8);
    expect([...sources.keys()].filter(key => key.includes('-select-')).length).toBe(4);
    const identifiers = nodes.map(node => node.id);
    expect(identifiers.indexOf('column-quantiles-select-0')).toBeGreaterThan(
      identifiers.indexOf('column-quantiles-histogram-0-chunk-2')
    );
    expect(
      () => new GPUColumnQuantiles({...contributor.props, rationalProbabilities: undefined})
    ).toThrow(/2\^24/);
    expect(
      () =>
        new GPUColumnQuantiles({
          ...contributor.props,
          output: {...contributor.props.output, filterBounds: packed('filter', 'float32', 2)}
        })
    ).toThrow(/2\^24/);
    expect(() =>
      new GPUColumnQuantiles({
        ...contributor.props,
        values: createVectorView('oversized', 'float32', [
          packed('oversized-chunk', 'float32', 40_000_000)
        ]),
        mask: undefined
      }).getCommandNodes(graph)
    ).toThrow(/values chunk.*one binding/);
  } finally {
    device.destroy();
  }
});

it('validates quantile filter topology, preserves empty chunks, and bounds chunk graph size', () => {
  const {device, graph, rows, values, mask, packed, parameters} = createFixture();
  try {
    const contributor = new GPUColumnQuantiles({
      values,
      mask,
      parameters,
      quantileCount: 3,
      output: {
        quantiles: packed('quantiles', 'float32', 3),
        validCount: packed('count', 'uint32', 1),
        outputMask: rows('filtered', 'uint32'),
        filterBounds: packed('bounds', 'float32', 2)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBe(nodes.length);
    expect(
      () =>
        new GPUColumnQuantiles({
          ...contributor.props,
          output: {...contributor.props.output, outputMask: rows('wrong', 'uint32', [7])}
        })
    ).toThrow(/output.outputMask must match the source topology/);
    const manyChunks = createVectorView(
      'many',
      'float32',
      Array.from({length: 8192}, () => getGraphViewChunks(values)[0])
    );
    expect(
      () =>
        new GPUColumnQuantiles({
          ...contributor.props,
          values: manyChunks,
          mask: undefined,
          output: {...contributor.props.output, outputMask: undefined}
        })
    ).toThrow(/estimated command node count/);
  } finally {
    device.destroy();
  }
});

it.each<GPUClassBreaksMethod>([
  'equal-interval',
  'quantile',
  'box-plot',
  'custom',
  'standard-deviation',
  'head-tail',
  'natural-breaks'
])('class breaks accepts chunks for %s with bounded nodes and parseable WGSL', method => {
  const {device, graph, values, mask, packed, parameters} = createFixture();
  try {
    const contributor = new GPUClassBreaks({
      values,
      mask,
      parameters,
      maximumClassCount: 6,
      methods: [method],
      naturalBreaksBinCount: 16,
      output: {
        breaks: packed('breaks', 'float32', 7),
        classCount: packed('class-count', 'uint32', 1),
        classCounts: packed('counts', 'uint32', 6)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(0);
    if (method === 'natural-breaks') {
      expect([...sources.keys()].filter(key => key.includes('histogram-chunk-')).length).toBe(4);
      expect([...sources.keys()].some(key => key.includes('chunk-1'))).toBe(false);
      const accumulated = [...sources.entries()].filter(([key]) => key.includes('accumulate'));
      expect(accumulated.length).toBe(32);
      expect(
        nodes
          .filter(node => node.id.includes('accumulate'))
          .every(node => node.condition?.source === 'gpu' && node.condition.mode === 'indirect')
      ).toBe(true);
    }
    if (method === 'standard-deviation' || method === 'head-tail')
      expect([...sources.keys()].some(key => key.includes('float-values-2'))).toBe(true);
    expect(() => new GPUClassBreaks({...contributor.props, methods: ['maximum-breaks']})).toThrow(
      /one packed view/
    );
    expect(
      () =>
        new GPUClassBreaks({
          ...contributor.props,
          methods: ['natural-breaks'],
          naturalBreaksMode: 'exact'
        })
    ).toThrow(/one packed view/);
    expect(
      () =>
        new GPUClassBreaks({
          ...contributor.props,
          methods: ['natural-breaks'],
          naturalBreaksMode: 'monotone'
        })
    ).toThrow(/one packed view/);
  } finally {
    device.destroy();
  }
});

it('checks per-chunk apply scratch against device limits before creating nodes', () => {
  const {device, graph, values, packed, palette, domain, parameters} = createFixture([34_000_000]);
  try {
    const contributor = new GPUColorScale({
      values,
      domain,
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      output: {classCounts: packed('counts', 'uint32', 8)}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/values chunk.*one binding/);
  } finally {
    device.destroy();
  }
});

it.each([
  'approximate',
  'exact',
  'monotone'
] as const)('bounds the actual class-break graph for maximum class capacity in %s mode', naturalBreaksMode => {
  const {device, graph, packed} = createFixture();
  const length =
    naturalBreaksMode === 'exact' ? 8192 : naturalBreaksMode === 'monotone' ? 262144 : 2 ** 24;
  try {
    const contributor = new GPUClassBreaks({
      values: packed('large-values', 'float32', length),
      parameters: packed('large-parameters', 'float32', 80),
      maximumClassCount: 64,
      naturalBreaksMode,
      naturalBreaksExactStorage: naturalBreaksMode === 'exact' ? 'recompute' : undefined,
      output: {breaks: packed('breaks', 'float32', 65), classCount: packed('count', 'uint32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('rejects class-count overflow across individually valid chunks', () => {
  const {device, packed, values, domain, palette, parameters} = createFixture([30_000_000]);
  try {
    const repeated = createVectorView(
      'huge',
      'float32',
      Array.from({length: 144}, () => values.data[0])
    );
    expect(
      () =>
        new GPUColorScale({
          values: repeated,
          domain,
          palette,
          parameters,
          maximumDomainCount: 9,
          maximumPaletteCount: 8,
          output: {classCounts: packed('counts', 'uint32', 8)}
        })
    ).toThrow(/color-scale values length with output.classCounts.*4294967295/);
    expect(
      () =>
        new GPUBivariateClassification({
          valuesX: repeated,
          valuesY: repeated,
          breaksX: domain,
          breaksY: domain,
          palette,
          parameters,
          maximumClassCount: 8,
          output: {classCounts: packed('joint-counts', 'uint32', 64)}
        })
    ).toThrow(/bivariate-classification values length with output.classCounts.*4294967295/);
  } finally {
    device.destroy();
  }
});

it('checks chunk-proportional natural-breaks dispatch scratch before emitting nodes', () => {
  const {device, graph, values, packed, parameters} = createFixture([1]);
  Object.assign(device.limits, {maxStorageBufferBindingSize: 1024});
  try {
    const source = createVectorView(
      'many-chunks',
      'float32',
      Array.from({length: 40}, () => values.data[0])
    );
    const contributor = new GPUClassBreaks({
      values: source,
      parameters,
      methods: ['natural-breaks'],
      maximumClassCount: 6,
      naturalBreaksBinCount: 16,
      output: {breaks: packed('breaks', 'float32', 7), classCount: packed('count', 'uint32', 1)}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /class-breaks natural-breaks refinement dispatch scratch.*one binding/
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'quantiles',
  'validCount',
  'filterBounds'
] as const)('rejects quantile filter chunks sharing the %s output buffer', outputName => {
  const {device, graph, values, packed, rows, parameters} = createFixture([2, 2]);
  try {
    const shared = packed('shared', 'uint32', 8);
    const output = {
      quantiles: packed('quantiles', 'float32', 1),
      validCount: packed('count', 'uint32', 1),
      filterBounds: packed('bounds', 'float32', 2),
      outputMask: rows('filtered', 'uint32')
    };
    output[outputName] = graph.createDataView(shared.buffer, {
      format: outputName === 'validCount' ? 'uint32' : 'float32',
      length: 2
    }) as never;
    output.outputMask = createVectorView('overlapping-mask', 'uint32', [
      graph.createDataView(shared.buffer, {format: 'uint32', length: 2}),
      packed('mask-tail', 'uint32', 2)
    ]);
    expect(() => new GPUColumnQuantiles({values, parameters, quantileCount: 1, output})).toThrow(
      /byte ranges must not overlap/
    );
    const packedValues = packed('packed-values', 'float32', 4);
    expect(
      () =>
        new GPUColumnQuantiles({
          values: packedValues,
          parameters,
          quantileCount: 1,
          output: {
            ...output,
            outputMask: graph.createDataView(shared.buffer, {format: 'uint32', length: 4})
          }
        })
    ).toThrow(/byte ranges must not overlap/);
  } finally {
    device.destroy();
  }
});

it('shares histogram modules and pipelines across sixteen uneven chunks, including class-break quantiles', () => {
  const {device, graph, values, mask, packed, parameters} = createFixture(
    Array.from({length: 16}, (_, index) => 1024 + index)
  );
  try {
    const contributor = new GPUColumnQuantiles({
      values,
      mask,
      parameters,
      quantileCount: 3,
      output: {
        quantiles: packed('quantiles', 'float32', 3),
        validCount: packed('count', 'uint32', 1)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const histograms = nodes.filter(node => node.id.includes('-histogram-'));
    expect(new Set(getParsedKernelSources(histograms).values()).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(histograms).size).toBe(1);
    const breaks = new GPUClassBreaks({
      values,
      mask,
      parameters,
      maximumClassCount: 6,
      methods: ['quantile'],
      output: {
        breaks: packed('breaks', 'float32', 7),
        classCount: packed('class-count', 'uint32', 1)
      }
    });
    const breakHistograms = breaks
      .getCommandNodes(graph)
      .filter(node => node.id.includes('quantiles-histogram-'));
    expect(new Set(getParsedKernelSources(breakHistograms).values()).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(breakHistograms).size).toBe(1);
  } finally {
    device.destroy();
  }
});

it.each([
  'values',
  'mask',
  'outputMask',
  'quantiles',
  'validCount',
  'filterBounds'
] as const)('preflights aligned quantile %s bindings without leaving scratch after failure', bindingName => {
  const {device, graph, packed, parameters} = createFixture();
  // A small limit exercises exactly the same four-byte alignment overflow as 128 MiB.
  Object.assign(device.limits, {maxStorageBufferBindingSize: 1024});
  try {
    const storage = packed('oversized-storage', 'uint32', 257);
    const format =
      bindingName === 'values' || bindingName === 'quantiles' || bindingName === 'filterBounds'
        ? 'float32'
        : 'uint32';
    const bad = graph.createDataView(storage.buffer, {format, length: 256, byteOffset: 4});
    const props = {
      values: packed('values', 'float32', 256),
      mask: packed('mask', 'uint32', 256),
      parameters,
      rationalProbabilities: packed('ratios', 'uint32', 2),
      quantileCount: 1,
      output: {
        quantiles: packed('quantiles', 'float32', 1),
        validCount: packed('count', 'uint32', 1),
        outputMask: packed('filtered', 'uint32', 256),
        filterBounds: packed('bounds', 'float32', 2)
      }
    };
    if (bindingName === 'values' || bindingName === 'mask') props[bindingName] = bad as never;
    else props.output[bindingName] = bad as never;
    const contributor = new GPUColumnQuantiles(props);
    for (let attempt = 0; attempt < 2; attempt++)
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /needs 1028 bytes in one binding \(limit 1024\)/
      );
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'scalar-mask',
  'scalar-output',
  'color-mask',
  'color-output',
  'bivariate-y',
  'bivariate-alpha',
  'bivariate-output',
  'breaks-mask'
] as const)('preflights aligned %s views before any scratch mutation', scenario => {
  const {device, graph, values, mask, domain, palette, parameters, packed, rows} = createFixture([
    256
  ]);
  Object.assign(device.limits, {maxStorageBufferBindingSize: 1024});
  try {
    const storage = packed('bad-storage', 'uint32', 257);
    const badFloat = createVectorView('bad-float', 'float32', [
      graph.createDataView(storage.buffer, {format: 'float32', length: 256, byteOffset: 4})
    ]);
    const badUint = createVectorView('bad-uint', 'uint32', [
      graph.createDataView(storage.buffer, {format: 'uint32', length: 256, byteOffset: 4})
    ]);
    const contributor = scenario.startsWith('scalar')
      ? new GPUScalarScale({
          values,
          mask: scenario === 'scalar-mask' ? badUint : mask,
          domain,
          range: packed('range', 'float32', 2),
          parameters,
          maximumDomainCount: 8,
          output: {values: scenario === 'scalar-output' ? badFloat : rows('mapped', 'float32')}
        })
      : scenario.startsWith('color')
        ? new GPUColorScale({
            values,
            mask: scenario === 'color-mask' ? badUint : mask,
            domain,
            palette,
            parameters,
            maximumDomainCount: 9,
            maximumPaletteCount: 8,
            output: {
              classIndices: scenario === 'color-output' ? badUint : rows('classes', 'uint32'),
              classCounts: packed('counts', 'uint32', 8)
            }
          })
        : scenario.startsWith('bivariate')
          ? new GPUBivariateClassification({
              valuesX: values,
              valuesY: scenario === 'bivariate-y' ? badFloat : values,
              mask,
              alphaValues: scenario === 'bivariate-alpha' ? badFloat : values,
              breaksX: domain,
              breaksY: domain,
              palette,
              parameters,
              maximumClassCount: 8,
              output: {colors: scenario === 'bivariate-output' ? badUint : rows('colors', 'uint32')}
            })
          : new GPUClassBreaks({
              values,
              mask: badUint,
              parameters,
              maximumClassCount: 6,
              methods: ['quantile'],
              output: {
                breaks: packed('breaks', 'float32', 7),
                classCount: packed('count', 'uint32', 1)
              }
            });
    for (let attempt = 0; attempt < 2; attempt++)
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /needs 1028 bytes in one binding \(limit 1024\)/
      );
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'float32',
  'uint32',
  'sint32'
] as const)('audits every generated %s mapping and reduction identifier against the full WGSL list', format => {
  const {device, graph, packed, rows, domain, palette, parameters} = createFixture([
    2047, 0, 2048, 2049
  ]);
  try {
    const values = createVectorView(
      'formatted',
      format,
      [2047, 0, 2048, 2049].map((length, chunkIndex) =>
        createTransientView(graph, `formatted-${chunkIndex}`, format, length)
      )
    );
    const color = new GPUColorScale({
      values,
      domain,
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      integerValues: 'numeric',
      output: {colors: rows('colors', 'uint32')}
    });
    getParsedKernelSources(color.getCommandNodes(graph));
    if (format === 'uint32')
      getParsedKernelSources(
        new GPUColorScale({...color.props, id: 'ordinal'}).getCommandNodes(graph)
      );
    if (format !== 'sint32')
      getParsedKernelSources(
        new GPUScalarScale({
          values:
            values as import('../../../src/dataframe-analysis/column-classification/column-classification-rows').ClassificationRows<
              'float32' | 'uint32'
            >,
          domain,
          range: packed('range', 'float32', 2),
          parameters,
          maximumDomainCount: 8,
          output: {values: rows('scaled', 'float32')}
        }).getCommandNodes(graph)
      );
    getParsedKernelSources(
      new GPUClassBreaks({
        values,
        parameters,
        maximumClassCount: 6,
        methods: ['quantile', 'standard-deviation', 'head-tail', 'natural-breaks'],
        naturalBreaksBinCount: 16,
        output: {breaks: packed('breaks', 'float32', 7), classCount: packed('count', 'uint32', 1)}
      }).getCommandNodes(graph)
    );
  } finally {
    device.destroy();
  }
});

it('rejects the 128 MiB rational subview alignment overflow and permits a corrected retry', () => {
  const {device, graph, packed, parameters} = createFixture();
  try {
    const storage = packed('large-storage', 'float32', 2 ** 25 + 1);
    const chunk = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 2 ** 25,
      byteOffset: 4
    });
    const props = {
      values: createVectorView('values', 'float32', [chunk]),
      parameters,
      rationalProbabilities: packed('ratios', 'uint32', 2),
      quantileCount: 1,
      output: {
        quantiles: packed('quantiles', 'float32', 1),
        validCount: packed('count', 'uint32', 1)
      }
    };
    const contributor = new GPUColumnQuantiles(props);
    for (let attempt = 0; attempt < 2; attempt++)
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /134217732 bytes in one binding \(limit 134217728\)/
      );
    const aligned = graph.createDataView(storage.buffer, {format: 'float32', length: 2 ** 25});
    expect(() =>
      new GPUColumnQuantiles({
        ...props,
        values: createVectorView('aligned-values', 'float32', [aligned])
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'scalar',
  'color',
  'bivariate',
  'quantile',
  'breaks'
] as const)('keeps %s shaders and pipelines identical as uneven chunks grow and dispatches each size independently', family => {
  const build = (chunkCount: number) => {
    const lengths = [3, 0, ...Array.from({length: chunkCount}, (_, index) => 2051 + index * 263)];
    const fixture = createFixture(lengths);
    const {graph, values, mask, domain, palette, parameters, packed, rows} = fixture;
    const contributor =
      family === 'scalar'
        ? new GPUScalarScale({
            values,
            mask,
            domain,
            range: packed('range', 'float32', 2),
            parameters,
            maximumDomainCount: 8,
            output: {values: rows('scaled', 'float32'), validity: rows('valid', 'uint32')}
          })
        : family === 'color'
          ? new GPUColorScale({
              values,
              mask,
              domain,
              palette,
              parameters,
              maximumDomainCount: 9,
              maximumPaletteCount: 8,
              output: {
                colors: rows('colors', 'uint32'),
                classIndices: rows('classes', 'uint32'),
                classCounts: packed('counts', 'uint32', 8)
              }
            })
          : family === 'bivariate'
            ? new GPUBivariateClassification({
                valuesX: values,
                valuesY: rows('y', 'float32'),
                mask,
                breaksX: domain,
                breaksY: domain,
                palette,
                parameters,
                maximumClassCount: 8,
                output: {
                  colors: rows('colors', 'uint32'),
                  classIds: rows('classes', 'uint32'),
                  classCounts: packed('counts', 'uint32', 64)
                }
              })
            : family === 'quantile'
              ? new GPUColumnQuantiles({
                  values,
                  mask,
                  parameters,
                  quantileCount: 3,
                  output: {
                    quantiles: packed('quantiles', 'float32', 3),
                    validCount: packed('count', 'uint32', 1),
                    outputMask: rows('filtered', 'uint32')
                  }
                })
              : new GPUClassBreaks({
                  values,
                  mask,
                  parameters,
                  maximumClassCount: 6,
                  methods: ['quantile', 'standard-deviation', 'head-tail', 'natural-breaks'],
                  naturalBreaksBinCount: 16,
                  output: {
                    breaks: packed('breaks', 'float32', 7),
                    classCount: packed('count', 'uint32', 1),
                    classCounts: packed('counts', 'uint32', 6)
                  }
                });
    const nodes = contributor.getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    for (const source of sources.values()) expect(source).not.toMatch(/\boverride\b/);
    if (family === 'scalar') {
      expect(nodes[0].workload?.maximumInvocationCount).toBe(3);
      expect(nodes[0].workload?.maximumWorkgroupCount).toBe(1);
      expect(nodes[1].workload?.maximumInvocationCount).toBe(2051);
      expect(nodes[1].workload?.maximumWorkgroupCount).toBe(9);
    }
    return {
      device: fixture.device,
      keys: [...getGPUCommandNodePipelineKeys(nodes)].sort(),
      sources: [...new Set(sources.values())].sort()
    };
  };
  const smaller = build(2);
  const larger = build(8);
  try {
    expect(larger.keys).toEqual(smaller.keys);
    expect(larger.sources).toEqual(smaller.sources);
  } finally {
    smaller.device.destroy();
    larger.device.destroy();
  }
});

it('allows consecutive output chunks in one buffer and rejects same-dispatch aligned overlap', () => {
  const {device, graph, values, domain, parameters, packed} = createFixture([2, 0, 3]);
  try {
    const storage = packed('storage', 'float32', 128);
    const scaled = createVectorView('scaled', 'float32', [
      graph.createDataView(storage.buffer, {format: 'float32', length: 2}),
      graph.createDataView(storage.buffer, {format: 'float32', length: 0}),
      graph.createDataView(storage.buffer, {format: 'float32', length: 3, byteOffset: 8})
    ]);
    const validity = (firstByteOffset: number) =>
      createVectorView(`validity-${firstByteOffset}`, 'uint32', [
        graph.createDataView(storage.buffer, {
          format: 'uint32',
          length: 2,
          byteOffset: firstByteOffset
        }),
        graph.createDataView(storage.buffer, {format: 'uint32', length: 0}),
        graph.createDataView(storage.buffer, {format: 'uint32', length: 3, byteOffset: 264})
      ]);
    const props = {
      values,
      domain,
      parameters,
      range: packed('range', 'float32', 2),
      maximumDomainCount: 8
    };
    expect(() => new GPUScalarScale({...props, output: {values: scaled}})).not.toThrow();
    expect(
      () => new GPUScalarScale({...props, output: {values: scaled, validity: validity(256)}})
    ).not.toThrow();
    expect(
      () => new GPUScalarScale({...props, output: {values: scaled, validity: validity(16)}})
    ).toThrow(/aligned binding byte ranges must not overlap/);
    const overlap = createVectorView('overlap', 'float32', [
      scaled.data[0],
      scaled.data[1],
      graph.createDataView(storage.buffer, {format: 'float32', length: 3, byteOffset: 4})
    ]);
    expect(() => new GPUScalarScale({...props, output: {values: overlap}})).toThrow(
      /logical byte ranges must not overlap/
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'quantile',
  'approximate',
  'exact',
  'monotone'
] as const)('preflights all derived %s scratch before allocations and permits retry', route => {
  const {device, graph, packed, parameters} = createFixture();
  try {
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: route === 'approximate' ? 65536 : 1024
    });
    const contributor =
      route === 'quantile'
        ? new GPUColumnQuantiles({
            values: packed('values', 'float32', 100),
            parameters,
            quantileCount: 3,
            output: {
              quantiles: packed('quantiles', 'float32', 3),
              validCount: packed('count', 'uint32', 1)
            }
          })
        : new GPUClassBreaks({
            values: packed('values', 'float32', 100),
            parameters,
            maximumClassCount: 6,
            methods: ['natural-breaks'],
            naturalBreaksMode: route,
            naturalBreaksBinCount: 16,
            output: {
              breaks: packed('breaks', 'float32', 7),
              classCount: packed('count', 'uint32', 1)
            }
          });
    for (let attempt = 0; attempt < 2; attempt++)
      expect(() => contributor.getCommandNodes(graph)).toThrow(/bytes in one binding/);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'exact',
  'monotone'
] as const)('keeps packed %s reference kernels identical as row and tree-level counts grow', route => {
  const build = (length: number) => {
    const {device, graph, packed, parameters} = createFixture();
    const nodes = new GPUClassBreaks({
      values: packed('reference-values', 'float32', length),
      parameters,
      methods: ['natural-breaks'],
      naturalBreaksMode: route,
      naturalBreaksExactStorage: route === 'exact' ? 'recompute' : undefined,
      maximumClassCount: 6,
      output: {
        breaks: packed('reference-breaks', 'float32', 7),
        classCount: packed('reference-count', 'uint32', 1)
      }
    }).getCommandNodes(graph);
    // GPUSort/GPUScan own their upstream pipelines. Audit the classification reference kernels.
    const referenceNodes = nodes.filter(node =>
      /natural-breaks-(initialize|keys|quantize|pack-prefix|layer|advance|dispatch|search|backtrack)(-|$)/.test(
        node.id
      )
    );
    const sources = [...new Set(getParsedKernelSources(referenceNodes).values())].sort();
    for (const source of sources) expect(source).not.toMatch(/\boverride\b/);
    return {device, sources, keys: [...getGPUCommandNodePipelineKeys(referenceNodes)].sort()};
  };
  const smaller = build(23);
  const larger = build(37);
  try {
    expect(smaller.sources.length).toBeGreaterThan(0);
    expect(larger.sources).toEqual(smaller.sources);
    expect(larger.keys).toEqual(smaller.keys);
  } finally {
    smaller.device.destroy();
    larger.device.destroy();
  }
});

it('reuses colour pipelines across different domain binding lengths', () => {
  const build = (domainLength: number) => {
    const {device, graph, values, packed, parameters, palette, rows} = createFixture();
    const nodes = new GPUColorScale({
      values,
      domain: packed('varying-domain', 'float32', domainLength),
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      output: {colors: rows('colors', 'uint32')}
    }).getCommandNodes(graph);
    return {
      device,
      sources: [...new Set(getParsedKernelSources(nodes).values())].sort(),
      keys: [...getGPUCommandNodePipelineKeys(nodes)].sort()
    };
  };
  const smaller = build(9);
  const larger = build(17);
  try {
    expect(larger.sources).toEqual(smaller.sources);
    expect(larger.keys).toEqual(smaller.keys);
  } finally {
    smaller.device.destroy();
    larger.device.destroy();
  }
});

it.each([
  Number.NaN,
  Number.POSITIVE_INFINITY,
  Number.NEGATIVE_INFINITY,
  1e100,
  2 ** 32,
  1.5
])('rejects capacities that cannot become valid WGSL integer literals (%s)', invalidCapacity => {
  const {device, values, domain, palette, parameters, packed, rows} = createFixture();
  try {
    expect(
      () =>
        new GPUScalarScale({
          values,
          domain,
          range: packed('range', 'float32', 2),
          parameters,
          maximumDomainCount: invalidCapacity,
          output: {values: rows('scaled', 'float32')}
        })
    ).toThrow(/scalar-scale maximumDomainCount must be an integer/);
    for (const property of ['maximumDomainCount', 'maximumPaletteCount'] as const) {
      expect(
        () =>
          new GPUColorScale({
            values,
            domain,
            palette,
            parameters,
            maximumDomainCount: 9,
            maximumPaletteCount: 8,
            [property]: invalidCapacity,
            output: {colors: rows(`colors-${property}`, 'uint32')}
          })
      ).toThrow(new RegExp(`color-scale ${property} must be an integer`));
    }
    expect(
      () =>
        new GPUBivariateClassification({
          valuesX: values,
          valuesY: values,
          breaksX: domain,
          breaksY: domain,
          palette,
          parameters,
          maximumClassCount: invalidCapacity,
          output: {classIds: rows('classes', 'uint32')}
        })
    ).toThrow(/bivariate-classification maximumClassCount must be an integer/);
    expect(
      () =>
        new GPUColumnQuantiles({
          values,
          parameters,
          quantileCount: invalidCapacity,
          output: {
            quantiles: packed('quantiles', 'float32', 1),
            validCount: packed('count', 'uint32', 1)
          }
        })
    ).toThrow(/column-quantiles quantileCount must be an integer/);
    for (const property of ['maximumClassCount', 'naturalBreaksBinCount'] as const) {
      expect(
        () =>
          new GPUClassBreaks({
            values,
            parameters,
            methods: ['natural-breaks'],
            maximumClassCount: 6,
            naturalBreaksBinCount: 16,
            [property]: invalidCapacity,
            output: {
              breaks: packed(`breaks-${property}`, 'float32', 7),
              classCount: packed(`count-${property}`, 'uint32', 1)
            }
          })
      ).toThrow(new RegExp(`class-breaks ${property} must be an integer`));
    }
  } finally {
    device.destroy();
  }
});

it('uses the canonical WGSL reserved list and permits sample', () => {
  expect(getReservedWGSLIdentifiers('let target = active + filter + module + meta;')).toEqual([
    'target',
    'active',
    'filter',
    'module',
    'meta'
  ]);
  expect(getReservedWGSLIdentifiers('let sample = 1u;')).toEqual([]);
});
