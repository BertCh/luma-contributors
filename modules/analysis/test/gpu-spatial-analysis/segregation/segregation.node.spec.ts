// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSegregationLayout,
  GPUSegregation
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {computeSegregationOracle} from './segregation-oracle';

let serial = 0;

it.each([
  'complete',
  'converged',
  'iterationLimitReached',
  'requiredCount'
] as const)('refuses a scale carrying unqualifiable %s evidence', name => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `scale-certificate-${serial++}`, format, length);
  try {
    const weights = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const props = {
      groupCount: 2,
      groupCounts: view('float32', 4),
      output: {indices: view('float32', getGPUSegregationLayout(2).stride)}
    };
    // Overflow words have a runtime guard. These four other kinds of evidence have
    // no writer/guard, so admitting their views could publish an incomplete result.
    expect(() => new GPUSegregation({...props, scales: [weights]})).not.toThrow();
    expect(
      () =>
        new GPUSegregation({
          ...props,
          scales: [{...weights, status: {[name]: view('uint32', 1)}}]
        })
    ).toThrow(/scales\[0\].status/);
  } finally {
    device.destroy();
  }
});

it('UST-10 captures missing-count settings and rejects unknown spatial forms', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `missing-contract-${serial++}`, format, length);
  try {
    const props = {
      groupCount: 2,
      groupCounts: view('float32', 4),
      missingCountPolicy: 'exclude-unit' as 'zero' | 'exclude-unit',
      output: {
        indices: view('float32', getGPUSegregationLayout(2).stride),
        invalidUnitCount: view('uint32', 1)
      }
    };
    const contributor = new GPUSegregation(props),
      originalCount = props.output.invalidUnitCount;
    props.missingCountPolicy = 'zero';
    props.output.invalidUnitCount = view('uint32', 1);
    expect(contributor.props.missingCountPolicy).toBe('exclude-unit');
    expect(contributor.props.output.invalidUnitCount).toBe(originalCount);
    expect(() => new GPUSegregation({...props, spatialForm: 'unknown' as 'environment'})).toThrow(
      /environment.*smoothed-population/
    );
  } finally {
    device.destroy();
  }
});

it('GPUSegregation declares nodes and validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segregation-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const layout = getGPUSegregationLayout(3);
  expect(layout.stride).toBe(3 + 9 + 9);
  const weights = {
    offsets: view('uint32', 11),
    neighbors: view('uint32', 20),
    weights: view('float32', 20)
  };
  const props = {
    groupCount: 3,
    groupCounts: view('float32', 30),
    scales: [null, weights],
    output: {indices: view('float32', 2 * layout.stride)}
  };
  expect(new GPUSegregation(props).getCommandNodes(graph).length).toBeGreaterThan(8);
  expect(() => new GPUSegregation({...props, groupCount: 1})).toThrow(/groupCount/);
  expect(() => new GPUSegregation({...props, atkinsonInequalityAversion: 1})).toThrow(
    /atkinsonInequalityAversion/
  );
  expect(
    () => new GPUSegregation({...props, output: {indices: view('float32', layout.stride)}})
  ).toThrow(/indices/);
  expect(
    () => new GPUSegregation({...props, scales: [{...weights, offsets: view('uint32', 5)}]})
  ).toThrow(/one row per unit/);
});

it('segregation oracle writes NaN for undefined indices and local values', () => {
  const layout = getGPUSegregationLayout(2);
  const empty = computeSegregationOracle({
    unitCount: 2,
    groupCount: 2,
    counts: [0, 0, 0, 0],
    weights: null
  });
  expect(empty.indices.every(Number.isNaN)).toBe(true);
  expect(empty.environment.every(Number.isNaN)).toBe(true);
  // One group present: E = 0 is defined, H / D / D_g / Atkinson / absent-group exposures are not.
  const single = computeSegregationOracle({
    unitCount: 2,
    groupCount: 2,
    counts: [4, 0, 6, 0],
    weights: null
  });
  expect(single.indices[layout.diversity]).toBe(0);
  expect(single.indices[layout.entropy]).toBeNaN();
  expect(single.indices[layout.multiGroupDissimilarity]).toBeNaN();
  expect(single.indices[layout.dissimilarity]).toBeNaN();
  expect(single.indices[layout.atkinson + 1]).toBeNaN();
  expect(single.indices[layout.isolation]).toBeCloseTo(1, 12);
  expect(single.indices[layout.isolation + 1]).toBeNaN();
  // A populated, mixed scene is fully defined; an empty unit has NaN local values only.
  const mixed = computeSegregationOracle({
    unitCount: 3,
    groupCount: 2,
    counts: [8, 2, 0, 0, 1, 9],
    weights: null
  });
  expect(mixed.indices.every(Number.isFinite)).toBe(true);
  expect(mixed.entropy[1]).toBeNaN();
  expect(Number.isFinite(mixed.entropy[0])).toBe(true);
  // The empty unit contributes nothing: the finite local contributions add up to the globals.
  expect(mixed.theil[0] + mixed.theil[2]).toBeCloseTo(mixed.indices[layout.entropy], 12);
});

it('GPUSegregation scales share one pipeline per kernel (rule 12) and parse as WGSL', () => {
  const build = (
    spatialScaleCount: number,
    spatialForm: 'environment' | 'smoothed-population' | undefined,
    flagged: boolean
  ) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `scales-${serial++}`, format, length);
    const layout = getGPUSegregationLayout(3);
    const weights = {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 20),
      weights: view('float32', 20),
      ...(flagged
        ? {status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}}
        : {})
    };
    const scales = [null, ...Array.from({length: spatialScaleCount}, () => weights)];
    return new GPUSegregation({
      groupCount: 3,
      groupCounts: view('float32', 30),
      scales,
      spatialForm,
      output: {
        indices: view('float32', scales.length * layout.stride),
        localEnvironment: view('float32', scales.length * 30),
        localEntropy: view('float32', scales.length * 10),
        localDissimilarity: view('float32', scales.length * 30),
        localTheil: view('float32', scales.length * 10)
      }
    }).getCommandNodes(graph);
  };
  for (const spatialForm of [undefined, 'environment', 'smoothed-population'] as const) {
    for (const flagged of [false, true]) {
      expect(getGPUCommandNodePipelineKeys(build(3, spatialForm, flagged)).size, spatialForm).toBe(
        getGPUCommandNodePipelineKeys(build(2, spatialForm, flagged)).size
      );
      for (const {id, source} of getGPUCommandNodeKernelSources(build(2, spatialForm, flagged))) {
        expect(() => new WgslReflect(source), id).not.toThrow();
      }
    }
  }
});
