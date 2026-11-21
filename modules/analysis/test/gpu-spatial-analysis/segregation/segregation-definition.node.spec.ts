// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {Buffer} from '@luma.gl/core';
import {
  createExampleWeights,
  SEGREGATION_EXAMPLE_SOURCES
} from '../region-statistics/review-example-wiring';
import {GPUSegregation} from '../../../src/gpu-spatial-analysis/segregation/gpu-segregation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {computeSegregationOracle} from './segregation-oracle';
import {getGPUSegregationLayout} from '../../../src/gpu-spatial-analysis/segregation/segregation-layout';
import {getGPUSegregationParameterValues} from '../../../src/gpu-spatial-analysis/segregation/segregation-parameters';
import {
  DEFINITION_COUNTS,
  DEFINITION_SMOOTHED_COUNTS,
  DEFINITION_WEIGHTS,
  getSegregationDefinition
} from './segregation-definition-fixtures';

for (const {name, source} of SEGREGATION_EXAMPLE_SOURCES) {
  test(`R19: ${name} weights import the physical search overflow into the consuming graph`, () => {
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    const allocate = () =>
      device.createBuffer({
        byteLength: 16,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
    const buffers = {
      offsets: allocate(),
      neighbors: allocate(),
      weights: allocate(),
      distances: allocate(),
      overflow: allocate(),
      capacity: 2
    };
    try {
      const weights = createExampleWeights(graph, source, buffers);
      expect(weights.status?.overflow?.buffer.defaultBuffer === buffers.overflow).toBe(true);
      expect(weights.status?.overflow?.buffer.graph === graph).toBe(true);
      const nodes = new GPUSegregation({
        groupCount: 2,
        groupCounts: createTransientView(graph, 'example-counts', 'float32', 4),
        scales: [weights],
        output: {indices: createTransientView(graph, 'example-indices', 'float32', 13)}
      }).getCommandNodes(graph);
      expect(
        getGPUCommandNodeKernelSources(nodes).some(kernel => kernel.id.endsWith('-incomplete'))
      ).toBe(true);
    } finally {
      for (const buffer of [
        buffers.offsets,
        buffers.neighbors,
        buffers.weights,
        buffers.distances,
        buffers.overflow
      ])
        buffer.destroy();
      device.destroy();
    }
  });
}

test.each([
  'environment',
  'smoothed-population'
] as const)('R2/R18: loss suppression preserves only independent diversity (%s)', spatialForm => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `review-loss-${serial++}`, format, length);
  try {
    const overflow = view('uint32', 1),
      candidateOverflow = view('uint32', 1);
    const nodes = new GPUSegregation({
      groupCount: 2,
      spatialForm,
      groupCounts: view('float32', 4),
      scales: [
        {
          offsets: view('uint32', 3),
          neighbors: view('uint32', 2),
          weights: view('float32', 2),
          status: {count: view('uint32', 1), overflow, candidateOverflow}
        }
      ],
      output: {indices: view('float32', 13)}
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels.some(kernel => /overflow\[overflowOffset\]/.test(kernel.source))).toBe(true);
    expect(
      kernels.some(kernel => /candidateOverflow\[candidateOverflowOffset\]/.test(kernel.source))
    ).toBe(true);
    const kernel = kernels.find(value => value.id.endsWith('-incomplete'))!;
    const reflection = new WgslReflect(kernel.source);
    for (const flags of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      const indices = new Float32Array(13).fill(123);
      indices[2] = Math.fround(Math.log(2));
      const data: Record<string, ArrayBufferView> = {
        indices,
        scaleOrdinal: Uint32Array.of(0),
        overflow: Uint32Array.of(flags[0]),
        candidateOverflow: Uint32Array.of(flags[1])
      };
      const bindings: Record<number, Record<number, ArrayBuffer>> = {};
      for (const storage of reflection.storage) {
        bindings[storage.group] ??= {};
        bindings[storage.group][storage.binding] = data[storage.name].buffer as ArrayBuffer;
      }
      for (const uniform of reflection.uniforms) {
        bindings[uniform.group] ??= {};
        bindings[uniform.group][uniform.binding] = new ArrayBuffer(uniform.size);
      }
      // WgslExec leaks invocation-local state; isolate rows, as the nearest-kernel probe does.
      const source = `override REVIEW_ROW: u32;\n${kernel.source}`
        .replace(/^\s*_\s*=.*;\s*$/gm, '')
        .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
        .replace(
          /let index = workgroupIndex \* \d+u \+ localInvocationIndex;/,
          'let index = REVIEW_ROW;'
        );
      for (let row = 0; row < 13; row++) {
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          reflection.entry.compute[0].name,
          1,
          bindings,
          {constants: {...kernel.constants, REVIEW_ROW: row}}
        );
      }
      Array.from(indices).forEach((value, row) => {
        if (flags.some(Boolean) && !(spatialForm === 'environment' && row === 2))
          expect(value, `index ${row}`).toBeNaN();
        else expect(value).toBe(row === 2 ? Math.fround(Math.log(2)) : 123);
      });
    }
  } finally {
    device.destroy();
  }
});

test('R8: explicit diagonal slots do not duplicate selfWeight', () => {
  for (const diagonal of [false, true]) {
    const actual = computeSegregationOracle({
      unitCount: 2,
      groupCount: 2,
      counts: [10, 0, 0, 10],
      selfWeight: 1,
      weights: diagonal
        ? {offsets: [0, 2, 4], neighbors: [0, 1, 0, 1], weights: [1, 1, 1, 1]}
        : {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]}
    });
    expect(actual.indices.slice(0, 2)).toEqual([0, 0]);
    expect(actual.environment).toEqual([0.5, 0.5, 0.5, 0.5]);
    expect(actual.dissimilarity).toEqual([0, 0, 0, 0]);
    expect(actual.theil).toEqual([0, 0]);
  }
});

test('segregation captures group count, scales, CSR ports and local output identities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `segregation-capture-${serial++}`, format, length);
  try {
    const neighbors = view('uint32', 2),
      indices = view('float32', 26);
    const weights = {offsets: view('uint32', 9), neighbors, weights: view('float32', 2)};
    const scales = [null, weights];
    const props = {
      groupCount: 2,
      groupCounts: view('float32', 16),
      scales,
      selfWeight: 0,
      output: {indices, localTheil: view('float32', 16)}
    };
    const contributor = new GPUSegregation(props);
    props.groupCount = 3;
    props.selfWeight = 10;
    scales[1] = null;
    weights.neighbors = view('uint32', 3);
    props.output.indices = view('float32', 42);
    expect(contributor.props.groupCount).toBe(2);
    expect(contributor.props.selfWeight).toBe(0);
    expect(contributor.props.scales?.[1]?.neighbors).toBe(neighbors);
    expect(contributor.props.output.indices).toBe(indices);
    expect(Object.isFrozen(contributor.props.scales)).toBe(true);
  } finally {
    device.destroy();
  }
});

test('segregation test reference agrees with independent NumPy matrix definitions on every output', () => {
  const maximumDifferences = {
    indices: 0,
    localEnvironment: 0,
    localEntropy: 0,
    localDissimilarity: 0,
    localTheil: 0
  };
  for (const form of ['environment', 'smoothed-population'] as const) {
    const expected = getSegregationDefinition(form);
    const aspatial = computeSegregationOracle({
      unitCount: 8,
      groupCount: 3,
      counts: DEFINITION_COUNTS,
      weights: null,
      atkinsonB: 0.3
    });
    const spatial = computeSegregationOracle({
      unitCount: 8,
      groupCount: 3,
      counts: form === 'environment' ? DEFINITION_COUNTS : DEFINITION_SMOOTHED_COUNTS,
      weights: form === 'environment' ? DEFINITION_WEIGHTS : null,
      selfWeight: 0,
      atkinsonB: 0.3
    });
    const actual = {
      indices: [...aspatial.indices, ...spatial.indices],
      localEnvironment: [...aspatial.environment, ...spatial.environment],
      localEntropy: [...aspatial.entropy, ...spatial.entropy],
      localDissimilarity: [...aspatial.dissimilarity, ...spatial.dissimilarity],
      localTheil: [...aspatial.theil, ...spatial.theil]
    };
    for (const output of Object.keys(expected) as (keyof typeof expected)[]) {
      expect(actual[output].length).toBe(expected[output].length);
      expected[output].forEach((value, row) => {
        if (Number.isNaN(value)) {
          expect(actual[output][row]).toBeNaN();
        } else {
          expect(Number.isFinite(actual[output][row])).toBe(true);
          const difference = Math.abs(actual[output][row] - value);
          maximumDifferences[output] = Math.max(maximumDifferences[output], difference);
          expect(difference).toBeLessThan(2e-14);
        }
      });
    }
  }
  console.info('NumPy definition maximum absolute differences', maximumDifferences);
});

test('segregation layout and parameter helpers retain the documented literal layouts', () => {
  expect(getGPUSegregationLayout(3)).toEqual({
    entropy: 0,
    multiGroupDissimilarity: 1,
    diversity: 2,
    dissimilarity: 3,
    isolation: 6,
    atkinson: 9,
    interaction: 12,
    stride: 21
  });
  expect(Array.from(getGPUSegregationParameterValues())).toEqual([1, 0.5]);
  expect(
    Array.from(getGPUSegregationParameterValues({selfWeight: 0, atkinsonInequalityAversion: 0.3}))
  ).toEqual([0, 0.30000001192092896]);
});

test('populated islands contribute nothing to global H, matching the sum of defined local terms', () => {
  // Eight mixed units (1,1); only units 0 and 1 have neighbors. selfWeight=0.
  // Every defined environment is (1/2,1/2), equal to the global proportions:
  // each defined t_i(E-E_i)/(T E)=0; the remaining six terms are missing.
  const result = computeSegregationOracle({
    unitCount: 8,
    groupCount: 2,
    counts: new Float32Array(16).fill(1),
    weights: {offsets: [0, 1, 2, 2, 2, 2, 2, 2, 2], neighbors: [1, 0], weights: [1, 1]},
    selfWeight: 0
  });
  expect(result.theil.slice(0, 2)).toEqual([0, 0]);
  expect(result.theil.slice(2).every(Number.isNaN)).toBe(true);
  expect(result.indices[0]).toBe(0);
});

function probeSegregationFinalizer(
  totals = Float32Array.of(8, 8, 16),
  suppliedSums?: Float32Array
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `definition-${serial++}`, format, length);
  try {
    const nodes = new GPUSegregation({
      groupCount: 2,
      groupCounts: view('float32', 16),
      selfWeight: 0,
      scales: [
        {offsets: view('uint32', 9), neighbors: view('uint32', 2), weights: view('float32', 2)}
      ],
      output: {indices: view('float32', 13)}
    }).getCommandNodes(graph);
    const kernel = getGPUCommandNodeKernelSources(nodes).find(candidate =>
      candidate.id.endsWith('-finalize')
    )!;
    const reflection = new WgslReflect(kernel.source);
    const indices = new Float32Array(13);
    const sums = new Float32Array(11);
    sums[0] = Math.fround(4 * Math.log(2));
    // Term layout: entropy, absolute deviation, D[2], Atkinson[2], exposure[4], active population.
    sums[4] = 2;
    sums[5] = 2;
    sums.set([1, 1, 1, 1], 6);
    sums[10] = 4;
    if (suppliedSums) sums.set(suppliedSums);
    const bindings: Record<number, Record<number, ArrayBuffer>> = {0: {}};
    for (const storage of reflection.storage) {
      const data =
        storage.name === 'parameters'
          ? Float32Array.of(0, 0.5)
          : storage.name === 'allTotals'
            ? totals
            : storage.name === 'allSums'
              ? sums
              : indices;
      bindings[storage.group] ??= {};
      bindings[storage.group][storage.binding] = data.buffer;
    }
    for (const uniform of reflection.uniforms) {
      bindings[uniform.group] ??= {};
      bindings[uniform.group][uniform.binding] = new ArrayBuffer(uniform.size);
    }
    // This interpreter predates WGSL phony assignments emitted to retain bindings.
    // Removing only discard statements leaves all numerical statements intact.
    const executable = kernel.source.replace(/^\s*_\s*=.*;\s*$/gm, '');
    new WgslExec(new WgslParser().parse(executable)).dispatchWorkgroups(
      reflection.entry.compute[0].name,
      1,
      bindings,
      {constants: kernel.constants}
    );
    return Array.from(indices);
  } finally {
    device.destroy();
  }
}

test('production segregation finalizer executes the island entropy definition in the WGSL interpreter', () => {
  expect(probeSegregationFinalizer()[0]).toBeCloseTo(0, 6);
});

test('R7: Atkinson is zero for identical mixed rows at ordinary and tiny normal float32 scales', () => {
  for (const scale of [1, 1e-32]) {
    // Two rows (s,s): population=4s, group totals=2s, Atkinson terms=2s.
    const sums = new Float32Array(11);
    sums[0] = 4 * scale * Math.log(2);
    sums[4] = sums[5] = 2 * scale;
    sums.set([scale, scale, scale, scale], 6);
    sums[10] = 4 * scale;
    const indices = probeSegregationFinalizer(
      Float32Array.of(2 * scale, 2 * scale, 4 * scale),
      sums
    );
    expect(indices.slice(7, 9)).toEqual([0, 0]);
  }
});
