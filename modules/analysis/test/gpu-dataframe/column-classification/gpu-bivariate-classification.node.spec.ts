// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {selectBivariateOutput} from './select-classification-outputs.test-utils';
import {GPUBivariateClassification} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import type {GPUBivariateClassificationProps} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {
  getGPUBivariateClassificationParameterValues,
  GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH
} from '../../../src/dataframe-analysis/column-classification/bivariate-classification-parameters';
import {computeBivariateClassificationOnCPU} from './bivariate-classification-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  BIVARIATE_OPTIONAL_FIXTURE,
  CLASSIFICATION_OUTPUT_SUBSETS
} from './optional-output-fixtures';

let serial = 0;

it('R4 bivariate snapshots the caller count-output selection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const counts = props.output.classCounts!;
    props.output = {classCounts: counts};
    const contributor = new GPUBivariateClassification(props);
    const callerOutput: Partial<GPUBivariateClassificationProps['output']> = props.output;
    delete callerOutput.classCounts;
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toEqual([
      'bivariate-classification-init',
      'bivariate-classification-classify',
      'bivariate-classification-colorize'
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
    ).toEqual(['bivariate-classification-init', 'bivariate-classification-colorize']);
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('bivariate CPU reference matches all hand-derived outputs (alpha=%s)', includeAlpha => {
  const expected = BIVARIATE_OPTIONAL_FIXTURE;
  const actual = computeBivariateClassificationOnCPU({
    ...expected,
    valuesX: Float32Array.from(expected.valuesX),
    valuesY: Float32Array.from(expected.valuesY),
    classCountX: 2,
    classCountY: 2,
    maximumClassCount: 3,
    alphaValues: includeAlpha ? expected.alphaValues : undefined,
    valueByAlpha: includeAlpha ? {domain: [0, 1], minimumAlpha: 0.5} : undefined,
    noDataColor: 7
  });
  expect(Array.from(actual.classIds)).toEqual(expected.classes);
  expect(Array.from(actual.colors)).toEqual(includeAlpha ? expected.fadedColors : expected.colors);
  expect(Array.from(actual.classCounts)).toEqual(expected.counts);
});

it.each(
  CLASSIFICATION_OUTPUT_SUBSETS
)('bivariate optional bindings follow output use: %j', subset => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const kernels = getGPUCommandNodeKernelSources(
      new GPUBivariateClassification({
        ...props,
        alphaValues: view(graph, 'float32', 10),
        output: selectBivariateOutput({
          colors: subset.colors ? props.output.colors : undefined,
          classIds: subset.classes ? props.output.classIds : undefined,
          classCounts: subset.counts ? props.output.classCounts : undefined
        })
      }).getCommandNodes(graph)
    );
    expect(kernels.length).toBe(1 + Number(subset.colors || subset.counts) + Number(subset.counts));
    const colorize = kernels.at(-1)!;
    if (subset.colors || subset.counts) {
      expect(colorize.source.includes('var<storage, read> palette:')).toBe(subset.colors);
      expect(colorize.source.includes('var<storage, read> params:')).toBe(subset.colors);
      expect(colorize.source.includes('var<storage, read> alphaValues:')).toBe(subset.colors);
    }
  } finally {
    device.destroy();
  }
});

const view = <Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `view-${serial++}`, format, length);

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUBivariateClassificationProps> = {}
): GPUBivariateClassificationProps {
  return {
    valuesX: view(graph, 'float32', 10),
    valuesY: view(graph, 'float32', 10),
    breaksX: view(graph, 'float32', 4),
    breaksY: view(graph, 'float32', 4),
    palette: view(graph, 'uint32', 9),
    parameters: view(graph, 'float32', GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH),
    maximumClassCount: 3,
    output: {
      classIds: view(graph, 'uint32', 10),
      colors: view(graph, 'uint32', 10),
      classCounts: view(graph, 'uint32', 9)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUBivariateClassificationProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUBivariateClassification(createProps(graph, overrides(graph)))).toThrow(
    message
  );
  device.destroy();
}

it('GPUBivariateClassification parameter helper packs the documented layout', () => {
  expect(
    Array.from(
      getGPUBivariateClassificationParameterValues({
        classCountX: 3,
        classCountY: 4,
        noDataColor: 0xdeadbeef,
        valueByAlpha: {domain: [0.5, 2], minimumAlpha: 0.25}
      })
    )
  ).toEqual([3, 4, 0xbeef, 0xdead, 1, 0.5, 2, 0.25]);
  expect(
    Array.from(
      getGPUBivariateClassificationParameterValues({
        classCountX: 2,
        classCountY: 2
      })
    )
  ).toEqual([2, 2, 0, 0, 0, 0, 1, 1]);
  expect(() =>
    getGPUBivariateClassificationParameterValues({
      classCountX: -1,
      classCountY: 2
    })
  ).toThrow(/classCountX/);
  expect(() =>
    getGPUBivariateClassificationParameterValues(
      {classCountX: 1, classCountY: 1},
      new Float32Array(3)
    )
  ).toThrow(/hold/);
});

it('GPUBivariateClassification returns deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUBivariateClassification(createProps(graph, {id: 'bi'}));
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['bi-init', 'bi-classify', 'bi-colorize']);
  const otherGraph = new GPUCommandGraph(device);
  const other = new GPUBivariateClassification(createProps(otherGraph, {id: 'bi'}));
  expect(other.getCommandNodes(otherGraph).map(node => node.id)).toEqual(ids);
  const withAlpha = new GPUBivariateClassification(
    createProps(otherGraph, {
      id: 'va',
      alphaValues: view(otherGraph, 'float32', 10),
      mask: view(otherGraph, 'uint32', 10)
    })
  );
  expect(withAlpha.getCommandNodes(otherGraph).length).toBe(3);
  device.destroy();
});

it('GPUBivariateClassification validates props', () => {
  expectThrows(() => ({maximumClassCount: 0}), /maximumClassCount/);
  expectThrows(() => ({maximumClassCount: 17}), /maximumClassCount/);
  expectThrows(() => ({maximumClassCount: 2.5}), /maximumClassCount/);
  const emptyOutput = {};
  expectTypeOf(emptyOutput).not.toExtend<GPUBivariateClassificationProps['output']>();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      Reflect.construct(GPUBivariateClassification, [{...createProps(graph), output: emptyOutput}])
    ).toThrow(/at least one output/);
  } finally {
    device.destroy();
  }
  expectThrows(
    graph => ({
      valuesX: view(graph, 'float32', 0),
      valuesY: view(graph, 'float32', 0)
    }),
    /at least one row/
  );
  expectThrows(graph => ({valuesX: view(graph, 'uint32' as 'float32', 10)}), /valuesX/);
  expectThrows(graph => ({valuesY: view(graph, 'float32', 9)}), /valuesY length/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => ({alphaValues: view(graph, 'float32', 9)}), /alphaValues length/);
  expectThrows(graph => ({breaksX: view(graph, 'float32', 3)}), /breaksX must hold/);
  expectThrows(graph => ({breaksY: view(graph, 'float32', 3)}), /breaksY must hold/);
  expectThrows(graph => ({palette: view(graph, 'uint32', 8)}), /palette must hold/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 7)}), /parameters must hold/);
  expectThrows(graph => ({output: {classIds: view(graph, 'uint32', 9)}}), /output.classIds/);
  expectThrows(graph => ({output: {colors: view(graph, 'uint32', 9)}}), /output.colors/);
  expectThrows(graph => ({output: {classCounts: view(graph, 'uint32', 8)}}), /output.classCounts/);
});

it('GPUBivariateClassification rejects aliased buffers and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const mask = view(graph, 'uint32', 10);
  const props = createProps(graph, {mask});
  const aliasedOutput = view(graph, 'uint32', 10);
  expect(
    () =>
      new GPUBivariateClassification({
        ...props,
        output: {...props.output, colors: mask}
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUBivariateClassification({
        ...props,
        output: {classIds: aliasedOutput, colors: aliasedOutput}
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
  const contributor = new GPUBivariateClassification(createProps(graph));
  expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(/target graph/);
  device.destroy();
});

it('computeBivariateClassificationOnCPU matches a hand-checked 2x2 grid', () => {
  const none = 0xffffffff;
  const result = computeBivariateClassificationOnCPU({
    valuesX: Float32Array.from([0, 5, 10, 5, NaN, 2, 100]),
    valuesY: Float32Array.from([0, 5, 10, 0, 1, NaN, -100]),
    mask: [1, 1, 1, 1, 1, 1, 0],
    breaksX: Float32Array.from([0, 5, 10]),
    breaksY: Float32Array.from([0, 5, 10]),
    classCountX: 2,
    classCountY: 2,
    maximumClassCount: 2,
    palette: [0xff000001, 0xff000002, 0xff000003, 0xff000004],
    noDataColor: 7
  });
  // Row 1 sits on both inner edges, row 2 clamps to the last class on both axes.
  expect(Array.from(result.classIds)).toEqual([0, 3, 3, 1, none, none, none]);
  expect(Array.from(result.classCounts)).toEqual([1, 1, 0, 2]);
  expect(result.colors[4]).toBe(7);
  const faded = computeBivariateClassificationOnCPU({
    valuesX: Float32Array.from([0, 0, 0]),
    valuesY: Float32Array.from([0, 0, 0]),
    breaksX: Float32Array.from([0, 5, 10]),
    breaksY: Float32Array.from([0, 5, 10]),
    classCountX: 2,
    classCountY: 2,
    maximumClassCount: 2,
    palette: [0xff0000aa, 0, 0, 0],
    alphaValues: Float32Array.from([0, 1, NaN]),
    valueByAlpha: {domain: [0, 1], minimumAlpha: 0.5}
  });
  expect(Array.from(faded.colors, color => color >>> 24)).toEqual([128, 255, 128]);
});

it('PCK-07 rejects collapsed ordered opacity domains before writing the target', () => {
  const target = new Float32Array(8).fill(9);
  expect(() =>
    getGPUBivariateClassificationParameterValues(
      {
        classCountX: 1,
        classCountY: 1,
        valueByAlpha: {domain: [99.999999, 100.000001], minimumAlpha: 0}
      },
      target
    )
  ).toThrow(/99\.999999.*100\.000001.*100.*100/);
  expect(Array.from(target)).toEqual(Array(8).fill(9));
  const equal = getGPUBivariateClassificationParameterValues({
    classCountX: 1,
    classCountY: 1,
    valueByAlpha: {domain: [100, 100], minimumAlpha: 0}
  });
  expect(Array.from(equal.slice(5, 7))).toEqual([100, 100]);
});
