// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUColorScale} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import type {GPUColorScaleProps} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_CODES,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor,
  type GPUColorScaleResidentThresholdSettings,
  type GPUColorScaleSettings
} from '../../../src/dataframe-analysis/column-classification/color-scale-parameters';
import {computeColorScaleOnCPU} from './color-scale-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {CLASSIFICATION_OUTPUT_SUBSETS, COLOR_OPTIONAL_FIXTURE} from './optional-output-fixtures';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

let serial = 0;

it('R4 color-scale snapshots the caller count-output selection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const counts = props.output.classCounts!;
    props.output = {classCounts: counts};
    const contributor = new GPUColorScale(props);
    const mutableOutput: Partial<GPUColorScaleProps['output']> = props.output;
    delete mutableOutput.classCounts;
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toEqual([
      'color-scale-init',
      'color-scale-classify',
      'color-scale-colorize'
    ]);
    expect(contributor.props.output.classCounts).toBe(counts);
    expect(
      nodes
        .filter(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === counts &&
              resource.usage !== 'storage-read'
          )
        )
        .map(node => node.id)
    ).toEqual(['color-scale-init', 'color-scale-colorize']);
  } finally {
    device.destroy();
  }
});

it('R6 color-scale count fixture distinguishes reversing the active bins', () => {
  const counts = COLOR_OPTIONAL_FIXTURE.counts;
  expect([counts[2], counts[1], counts[0], counts[3]]).not.toEqual(counts);
});

it('color-scale CPU reference matches every hand-derived output including unused capacity', () => {
  const expected = COLOR_OPTIONAL_FIXTURE;
  const actual = computeColorScaleOnCPU({
    ...expected,
    values: Float32Array.from(expected.values),
    scale: 'threshold',
    activeDomainCount: 4,
    maximumDomainCount: 4,
    maximumPaletteCount: 4,
    paletteCount: 3,
    noDataColor: 7
  });
  expect(Array.from(actual.classIndices)).toEqual(expected.classes);
  expect(Array.from(actual.colors)).toEqual(expected.colors);
  expect(Array.from(actual.classCounts)).toEqual(expected.counts);
});

it.each(
  CLASSIFICATION_OUTPUT_SUBSETS
)('color-scale optional bindings follow output use: %j', subset => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    // createProps without overrides allocates all three terminal views.
    const colors = props.output.colors!;
    const classIndices = props.output.classIndices!;
    const classCounts = props.output.classCounts!;
    const selected = {
      colors: subset.colors ? colors : undefined,
      classIndices: subset.classes ? classIndices : undefined,
      classCounts: subset.counts ? classCounts : undefined
    };
    const output: GPUColorScaleProps['output'] = subset.colors
      ? {...selected, colors}
      : subset.classes
        ? {...selected, classIndices}
        : {...selected, classCounts};
    const kernels = getGPUCommandNodeKernelSources(
      new GPUColorScale({
        ...props,
        output
      }).getCommandNodes(graph)
    );
    expect(kernels.length).toBe(1 + Number(subset.colors || subset.counts) + Number(subset.counts));
    if (subset.colors || subset.counts) {
      const colorize = kernels.at(-1)!;
      expect(colorize.source.includes('var<storage, read> palette:')).toBe(subset.colors);
      expect(colorize.source.includes('var<storage, read> positions:')).toBe(subset.colors);
    }
  } finally {
    device.destroy();
  }
});

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUColorScaleProps> = {}
): GPUColorScaleProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    values: view('float32', 10),
    domain: view('float32', 8),
    palette: view('uint32', 6),
    parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
    maximumDomainCount: 8,
    maximumPaletteCount: 6,
    output: {
      colors: view('uint32', 10),
      classIndices: view('uint32', 10),
      classCounts: view('uint32', 6)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUColorScaleProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUColorScale(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const view = <Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `extra-${serial++}`, format, length);

it('GPUColorScale parameter helper packs the documented layout', () => {
  const values = getGPUColorScaleParameterValues({
    scale: 'log',
    domainCount: 5,
    paletteCount: 4,
    interpolation: 'linear',
    clamp: true,
    noDataColor: 0xdeadbeef,
    logFloor: 0.5,
    exponent: 3
  });
  expect(values.length).toBe(GPU_COLOR_SCALE_PARAMETER_LENGTH);
  expect(Array.from(values)).toEqual([
    GPU_COLOR_SCALE_CODES.log,
    5,
    4,
    1,
    1,
    0xbeef,
    0xdead,
    0.5,
    3
  ]);
  const defaults = getGPUColorScaleParameterValues({
    scale: 'ordinal',
    domainCount: 0,
    paletteCount: 1
  });
  expect(Array.from(defaults)).toEqual([8, 0, 1, 0, 0, 0, 0, Math.fround(1e-5), 1]);
  expect(() =>
    getGPUColorScaleParameterValues({
      scale: 'linear',
      domainCount: 1.5,
      paletteCount: 1
    })
  ).toThrow(/domainCount/);
  expect(() =>
    getGPUColorScaleParameterValues({
      scale: 'bogus' as 'linear',
      domainCount: 1,
      paletteCount: 1
    })
  ).toThrow(/unknown scale/);
  expect(() =>
    getGPUColorScaleParameterValues(
      {scale: 'linear', domainCount: 1, paletteCount: 1},
      new Float32Array(2)
    )
  ).toThrow(/hold/);
  expect(packGPUColor(1, 2, 3, 4)).toBe(0x04030201);
  expect(packGPUColor(255, 255, 255)).toBe(0xffffffff);
});

it('CMP-13 packs resident threshold settings without a numeric domain count', () => {
  const settings = {scale: 'threshold', paletteCount: 2, noDataColor: 0x12345678} as const;
  const target = new Float32Array(GPU_COLOR_SCALE_PARAMETER_LENGTH).fill(-1);
  expect(getGPUColorScaleParameterValues(settings, target)).toBe(target);
  expect(Array.from(target)).toEqual([6, 0, 2, 0, 0, 0x5678, 0x1234, Math.fround(1e-5), 1]);
  expect(getGPUColorScaleParameterValues({...settings, domainCount: 3})[1]).toBe(3);
  expect(() => getGPUColorScaleParameterValues({...settings, domainCount: -1})).toThrow(
    /domainCount must be a non-negative integer/
  );
  expect(() => getGPUColorScaleParameterValues({...settings, paletteCount: -1})).toThrow(
    /paletteCount must be a non-negative integer/
  );
  expect(() => {
    // @ts-expect-error Only threshold settings can omit domainCount.
    getGPUColorScaleParameterValues({scale: 'linear', paletteCount: 2});
  }).toThrow(/domainCount must be a non-negative integer/);
});

it('CMP-13 accepts settings typed as either supported form', () => {
  const pack = (settings: GPUColorScaleSettings | GPUColorScaleResidentThresholdSettings) =>
    getGPUColorScaleParameterValues(settings);
  expect(pack({scale: 'threshold', paletteCount: 2})[1]).toBe(0);
  expect(pack({scale: 'linear', domainCount: 3, paletteCount: 2})[1]).toBe(3);
});

it('CMP-13 rejects unknown color options and lists their accepted values', () => {
  const target = new Float32Array(GPU_COLOR_SCALE_PARAMETER_LENGTH).fill(-1);
  const numeric = {scale: 'linear', domainCount: 3, paletteCount: 2} as const;
  const resident = {scale: 'threshold', paletteCount: 2} as const;
  expect(() => {
    // @ts-expect-error Exercise an inherited property name from an untyped caller.
    getGPUColorScaleParameterValues({...numeric, scale: 'constructor'}, target);
  }).toThrow(
    /scale 'constructor'.*linear, sqrt, pow, log, symlog, quantize, threshold, quantile, ordinal/
  );
  expect(() => {
    // @ts-expect-error Exercise an unknown interpolation from an untyped caller.
    getGPUColorScaleParameterValues({...resident, interpolation: 'nearest'}, target);
  }).toThrow(/interpolation 'nearest'.*step, linear/);
  expect(() => {
    // @ts-expect-error Exercise an unknown log mode from an untyped caller.
    getGPUColorScaleParameterValues({...numeric, scale: 'log', logMode: 'signed'}, target);
  }).toThrow(/logMode 'signed'.*positive-floor, domain-sign/);
  expect(target.every(value => value === -1)).toBe(true);
});

it('CMP-13 classifies two resident classes from three edges', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const classIndices = view(graph, 'uint32', 2);
    const props = {
      values: view(graph, 'float32', 2),
      domain: view(graph, 'float32', 3),
      classCount: view(graph, 'uint32', 1),
      palette: view(graph, 'uint32', 2),
      parameters: view(graph, 'float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
      maximumDomainCount: 3,
      maximumPaletteCount: 2,
      output: {classIndices}
    };
    const [classification] = new GPUColorScale(props).getCommandNodes(graph);
    const kernel = getWGSLKernelNodeRecord(graph, classification.id)!;
    const memory = createCPUGraphMemory();
    memory.write(props.values, Float32Array.of(0.5, 1.5));
    memory.write(props.domain, Float32Array.of(0, 1, 2));
    memory.write(props.classCount, Uint32Array.of(2));
    memory.write(classIndices, Uint32Array.of(99, 99));
    memory.write(
      props.parameters,
      getGPUColorScaleParameterValues({scale: 'threshold', paletteCount: 2})
    );
    // Keep the constructor's bindings and arithmetic, using launch uniforms the interpreter supports.
    const replay = createWGSLKernelNode(graph, {
      id: `${kernel.id}-arithmetic`,
      operation: kernel.operation,
      bindings: [...kernel.bindings],
      invocationCount: kernel.invocationCount,
      declarations: kernel.declarations,
      body: kernel.body
    });
    executeCommandNodesOnCPU([replay], memory);
    expect(Array.from(memory.readUint32(classIndices))).toEqual([0, 1]);
  } finally {
    device.destroy();
  }
});

it('GPUColorScale accepts valid props and returns deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUColorScale(createProps(graph, {id: 'scale'}));
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['scale-init', 'scale-classify', 'scale-colorize']);
  const secondGraph = new GPUCommandGraph(device);
  const secondContributor = new GPUColorScale(createProps(secondGraph, {id: 'scale'}));
  expect(secondContributor.getCommandNodes(secondGraph).map(node => node.id)).toEqual(ids);
  // Classes only need no colorize node and no init.
  const classOnlyGraph = new GPUCommandGraph(device);
  const classOnly = new GPUColorScale(
    createProps(classOnlyGraph, {
      id: 'classes-only',
      output: {classIndices: view(classOnlyGraph, 'uint32', 10)}
    })
  );
  expect(classOnly.getCommandNodes(classOnlyGraph).map(node => node.id)).toEqual([
    'classes-only-classify'
  ]);
  device.destroy();
});

it('GPUColorScale accepts uint32 ordinal values, classCount views and masks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUColorScale(
    createProps(graph, {
      values: view(graph, 'uint32', 10),
      mask: view(graph, 'uint32', 10),
      classCount: view(graph, 'uint32', 1)
    })
  );
  expect(contributor.getCommandNodes(graph).length).toBe(3);
  device.destroy();
});

it('GPUColorScale rejects an untyped resident domainCount view', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {id: 'scale'});
    const legacy = {...props, domainCount: view(graph, 'uint32', 1)};
    expect(() => new GPUColorScale(legacy)).toThrow(
      /^scale domainCount was removed; use classCount/
    );
    const numeric = {...props, domainCount: 3};
    expect(() => new GPUColorScale(numeric)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('GPUColorScale validates props', () => {
  expectThrows(() => ({maximumDomainCount: 0}), /maximumDomainCount/);
  expectThrows(() => ({maximumPaletteCount: 1.5}), /maximumPaletteCount/);
  // @ts-expect-error JavaScript callers still receive the runtime empty-selection guard.
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(graph => ({values: view(graph, 'float32', 0)}), /at least one row/);
  expectThrows(graph => ({values: view(graph, 'float32x2' as 'float32', 10)}), /values/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => ({mask: view(graph, 'float32' as 'uint32', 10)}), /mask/);
  expectThrows(graph => ({domain: view(graph, 'float32', 7)}), /domain must hold/);
  expectThrows(graph => ({domain: view(graph, 'uint32' as 'float32', 8)}), /domain/);
  expectThrows(graph => ({palette: view(graph, 'uint32', 5)}), /palette must hold/);
  expectThrows(graph => ({palette: view(graph, 'float32' as 'uint32', 6)}), /palette/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 8)}), /parameters must hold/);
  expectThrows(graph => ({parameters: view(graph, 'uint32' as 'float32', 9)}), /parameters/);
  expectThrows(graph => ({classCount: view(graph, 'uint32', 0)}), /classCount/);
  expectThrows(graph => ({classCount: view(graph, 'float32' as 'uint32', 1)}), /classCount/);
  expectThrows(graph => ({output: {colors: view(graph, 'uint32', 9)}}), /output.colors/);
  expectThrows(
    graph => ({output: {classIndices: view(graph, 'uint32', 9)}}),
    /output.classIndices/
  );
  expectThrows(graph => ({output: {classCounts: view(graph, 'uint32', 5)}}), /output.classCounts/);
  expectThrows(
    graph => ({output: {colors: view(graph, 'float32' as 'uint32', 10)}}),
    /output.colors/
  );
});

it('GPUColorScale rejects aliased buffers and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {mask: view(graph, 'uint32', 10)});
  const aliased: GPUColorScaleProps = {
    ...props,
    output: {...props.output, colors: props.mask as GraphDataView<'uint32'>}
  };
  expect(() => new GPUColorScale(aliased)).toThrow(/share buffers/);
  const sharedColors = props.output.colors!;
  const sharedOutputs: GPUColorScaleProps = {
    ...props,
    output: {colors: sharedColors, classIndices: sharedColors}
  };
  expect(() => new GPUColorScale(sharedOutputs)).toThrow(
    /aligned binding byte ranges must not overlap/
  );
  const otherGraph = new GPUCommandGraph(device);
  const contributor = new GPUColorScale(createProps(graph));
  expect(() => contributor.getCommandNodes(otherGraph)).toThrow(/target graph/);
  device.destroy();
});

it('computeColorScaleOnCPU follows d3 semantics on a hand-checked case', () => {
  const palette = [10, 20, 30, 40].map(value => packGPUColor(value, 0, 0, 255));
  const base = {
    values: Float32Array.from([-1, 0, 2, 3, 8, 9, NaN, 4]),
    mask: [1, 1, 1, 1, 1, 1, 1, 0],
    domain: Float32Array.from([0, 8]),
    activeDomainCount: 2,
    palette,
    paletteCount: 4,
    noDataColor: 0x01020304,
    maximumDomainCount: 2,
    maximumPaletteCount: 4
  };
  const unclamped = computeColorScaleOnCPU({...base, scale: 'linear'});
  const none = 0xffffffff;
  expect(Array.from(unclamped.classIndices)).toEqual([none, 0, 1, 1, 3, none, none, none]);
  expect(unclamped.colors[0]).toBe(0x01020304);
  expect(Array.from(unclamped.classCounts)).toEqual([1, 2, 0, 1]);
  const clamped = computeColorScaleOnCPU({
    ...base,
    scale: 'linear',
    clamp: true
  });
  expect(Array.from(clamped.classIndices)).toEqual([0, 0, 1, 1, 3, 3, none, none]);
  const threshold = computeColorScaleOnCPU({
    ...base,
    scale: 'threshold',
    domain: Float32Array.from([-Infinity, 1, 1, 5, Infinity]),
    activeDomainCount: 5,
    maximumDomainCount: 5
  });
  // Inner edges 1, 1, 5: a value on edge 1 counts both, so class 1 is empty.
  expect(Array.from(threshold.classIndices)).toEqual([0, 0, 2, 2, 3, 3, none, none]);
  const logScale = computeColorScaleOnCPU({
    ...base,
    scale: 'log',
    values: Float32Array.from([0, 1, 10, 100]),
    mask: undefined,
    domain: Float32Array.from([1, 100]),
    clamp: true
  });
  // log floor sends 0 below the domain, which clamps to class 0.
  expect(Array.from(logScale.classIndices)).toEqual([0, 0, 2, 3]);
  const ordinal = computeColorScaleOnCPU({
    ...base,
    scale: 'ordinal',
    values: Uint32Array.from([0, 3, 4, 100])
  });
  expect(Array.from(ordinal.classIndices)).toEqual([0, 3, none, none]);
});

it('GPUColorScale native options preserve legacy controls and the nine-word layout', () => {
  const settings = {scale: 'log' as const, domainCount: 2, paletteCount: 2};
  expect(getGPUColorScaleParameterValues(settings)[0]).toBe(3);
  expect(getGPUColorScaleParameterValues(settings)[4]).toBe(0);
  const native = getGPUColorScaleParameterValues({
    ...settings,
    logMode: 'domain-sign',
    extrapolate: true
  });
  expect(native.length).toBe(9);
  expect(native[0]).toBe(9);
  expect(native[4]).toBe(2);
  expect(getGPUColorScaleParameterValues({...settings, extrapolate: true, clamp: true})[4]).toBe(1);
});
