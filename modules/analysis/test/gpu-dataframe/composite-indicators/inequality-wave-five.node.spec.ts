// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCPUVector,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  CPUInequalityOperation,
  computeInequalityOnCPU
} from '../../../src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {
  GPUInequalityOperation,
  type InequalityOperationProps
} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';
import {GPUInequality} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {getInequalityOperationTestOutput} from './inequality-operation-test-output';
import {INEQUALITY_EXPECTED} from './inequality-output-oracles-fixture';

const base = {
  id: 'wave-five',
  values: 'income',
  zoneIds: 'zones',
  parameters: 'settings',
  zoneCount: 1
};

it('declares every all-present, each-alone and each-omitted selection with a named terminal', () => {
  const names = Object.keys(INEQUALITY_EXPECTED) as (keyof typeof INEQUALITY_EXPECTED)[];
  const selections = [
    names,
    ...names.map(name => [name]),
    ...names.map(name => names.filter(other => other !== name))
  ];
  expect(selections).toHaveLength(21);
  for (const selected of selections) {
    const output = getInequalityOperationTestOutput(selected);
    expect(Object.keys(output).sort()).toEqual([...selected].sort());
    const operation = new GPUInequalityOperation({...base, output});
    expect(Object.keys(operation.outputs).sort()).toEqual([...selected].sort());
  }
});

for (const Operation of [CPUInequalityOperation, GPUInequalityOperation]) {
  it.each([
    {zoneCount: 0},
    {lorenzKnotCount: 1}
  ])(`${Operation.name} rejects invalid summary-only dimensions: %j`, dimensions => {
    expect(
      () => new Operation({...base, ...dimensions, output: {globalSummary: 'summary'}})
    ).toThrow(/zoneCount|lorenzKnotCount/);
  });
  it.each([
    'weightedGini',
    'unrecognizedOutput'
  ])(`${Operation.name} rejects unsupported port %s`, name => {
    const output = {mean: 'm', [name]: 'extra'};
    expect(() => new Operation({...base, output})).toThrow(
      name === 'weightedGini' ? /host.*output.gini/ : /supported/
    );
  });
  it.each(['weights', 'mask'] as const)(`${Operation.name} rejects empty optional %s`, name => {
    expect(() => new Operation({...base, [name]: '', output: {mean: 'm'}})).toThrow(
      new RegExp(`inputs\\.${name}`)
    );
  });
  it.each([{}, {mean: undefined}])(`${Operation.name} rejects no defined terminal: %j`, output => {
    // Deliberate JavaScript-invalid selection, also checked by the lead's typecheck.
    // @ts-expect-error At least one defined terminal is required.
    const props: InequalityOperationProps = {...base, output};
    expect(() => new Operation(props)).toThrow(/at least one/);
  });
  it.each([
    {},
    {weights: 'population', mask: 'included'}
  ])(`${Operation.name} accepts omitted or valid optional inputs: %j`, optional => {
    expect(() => new Operation({...base, ...optional, output: {mean: 'm'}})).not.toThrow();
  });
}

it('captures GPU selection, optional input presence, mode and dimensions before lowering', () => {
  const props: InequalityOperationProps = {
    ...base,
    mode: 'reference',
    weights: 'population',
    mask: 'included',
    lorenzKnotCount: 3,
    output: {mean: 'm', lorenzKnots: 'knots'}
  };
  const operation = new GPUInequalityOperation(props);
  const mutableOutput: Partial<Record<keyof typeof props.output, string>> = props.output;
  delete mutableOutput.mean;
  props.output = {gini: 'replacement'};
  props.mode = 'parallel';
  props.weights = undefined;
  props.mask = undefined;
  props.zoneCount = 2;
  props.lorenzKnotCount = 5;
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = {
      values: createTransientView(graph, 'income', 'float32', 2),
      zoneIds: createTransientView(graph, 'zones', 'uint32', 2),
      parameters: createTransientView(graph, 'settings', 'float32', 4),
      weights: createTransientView(graph, 'population', 'float32', 2),
      mask: createTransientView(graph, 'included', 'uint32', 2),
      mean: createTransientView(graph, 'm', 'float32', 1),
      lorenzKnots: createTransientView(graph, 'knots', 'float32', 3)
    };
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> => {
        const view = views[port as keyof typeof views];
        expect(view, port).toBeDefined();
        return view as GraphDataView<Format>;
      }
    );
    expect(contributor).toBeInstanceOf(GPUInequality);
    if (!(contributor instanceof GPUInequality)) throw new Error('Expected inequality contributor');
    expect(contributor.mode).toBe('reference');
    expect(contributor.props.zoneCount).toBe(1);
    expect(contributor.lorenzKnotCount).toBe(3);
    expect(contributor.props.weights).toBe(views.weights);
    expect(contributor.props.mask).toBe(views.mask);
    expect(contributor.props.output).toEqual({mean: views.mean, lorenzKnots: views.lorenzKnots});
    // The publication kernels must bind both originally declared destinations for writing.
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id === 'wave-five-publish-moments')).toBe(true);
    for (const destination of [views.mean, views.lorenzKnots]) {
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === destination &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

it('captures CPU optional inputs and dimensions before executing', async () => {
  const props: InequalityOperationProps = {
    ...base,
    weights: 'population',
    mask: 'included',
    lorenzKnotCount: 3,
    output: {mean: 'm', lorenzKnots: 'knots'}
  };
  const operation = new CPUInequalityOperation(props);
  props.weights = undefined;
  props.mask = undefined;
  props.zoneCount = 2;
  props.lorenzKnotCount = 5;
  props.output = {gini: 'replacement'};
  const rows = (values: Float32Array | Uint32Array): GPUCPUVector => ({
    format: values instanceof Uint32Array ? 'uint32' : 'float32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await operation.execute({
    inputs: {
      values: rows(Float32Array.of(1, 3, 100)),
      zoneIds: rows(Uint32Array.of(0, 0, 0)),
      weights: rows(Float32Array.of(3, 1, 1)),
      mask: rows(Uint32Array.of(1, 1, 0)),
      parameters: rows(getGPUInequalityParameterValues())
    }
  });
  // Included population: 3 people at income 1 and 1 at income 3, mean 6/4.
  expect(result['m']).toEqual(Float32Array.of(1.5));
  expect(result['knots']).toEqual(Float32Array.of(0, 1 / 3, 1));
  expect(Object.keys(result)).toEqual(['m', 'knots']);
});

it.each([
  1,
  1 + 2 ** -23,
  2,
  32
])('NUM-09 extends Atkinson continuously at zero for epsilon %s', epsilon => {
  const result = computeInequalityOnCPU({
    values: Float32Array.of(0, 2, 0, 0, 2),
    weights: Float32Array.of(1, 1, 1, 0, 1),
    zoneIds: Uint32Array.of(0, 0, 1, 2, 2),
    zoneCount: 4,
    parameters: getGPUInequalityParameterValues({epsilon})
  });
  // Geometric/harmonic/power means tend to zero as a positive-weight income tends to zero.
  // Zone 1 has mean zero, zone 2 excludes its zero-weight zero, and zone 3 is empty.
  expect(result.atkinson).toHaveLength(4);
  expect(result.theilL).toHaveLength(4);
  expect(Array.from(result.atkinson, value => (value === 0 ? 0 : value))).toEqual([1, NaN, 0, NaN]);
  expect(Array.from(result.theilL, value => (value === 0 ? 0 : value))).toEqual([NaN, NaN, 0, NaN]);
});

it.each([
  'parallel',
  'reference'
] as const)('direct %s rejects writerless ports and omits unrequested numerical work', mode => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      mode,
      values: createTransientView(graph, 'values', 'float32', 3),
      zoneIds: createTransientView(graph, 'zones', 'uint32', 3),
      zoneCount: 1,
      parameters: createTransientView(graph, 'params', 'float32', 4)
    };
    const mean = createTransientView(graph, 'mean', 'float32', 1);
    const extra = createTransientView(graph, 'extra', 'float32', 1);
    for (const name of ['weightedGini', 'unrecognizedOutput']) {
      const output = {mean, [name]: extra};
      expect(() => new GPUInequality({...props, output})).toThrow(
        name === 'weightedGini' ? /host.*output.gini/ : /supported/
      );
    }
    const contributor = new GPUInequality({
      ...props,
      output: {counts: createTransientView(graph, 'counts', 'uint32', 1)}
    });
    const identifiers = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).map(
      kernel => kernel.id
    );
    expect(identifiers.filter(identifier => identifier.includes('-zone-gini'))).toEqual([]);
    expect(
      identifiers.filter(identifier => identifier.includes('-atkinson-zero-boundary'))
    ).toEqual([]);
  } finally {
    device.destroy();
  }
});
