// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {
  GPUSpatialWeightsAlgebraOperation,
  type SpatialWeightsAlgebraOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const SOURCE = {
  offsets: 'sourceOffsets',
  neighbors: 'sourceNeighbors',
  weights: 'sourceWeights',
  distances: 'sourceDistances'
};
const RIGHT = {
  offsets: 'rightOffsets',
  neighbors: 'rightNeighbors',
  weights: 'rightWeights',
  distances: 'rightDistances'
};
const INPUTS: Record<string, OperationGraphInput> = {
  sourceOffsets: {format: 'uint32', values: Uint32Array.of(0, 2, 4, 5, 6)},
  sourceNeighbors: {format: 'uint32', values: Uint32Array.of(1, 3, 0, 2, 3, 0)},
  sourceWeights: {format: 'float32', values: Float32Array.of(2, 4, 3, 5, 6, 7)},
  sourceDistances: {format: 'float32', values: Float32Array.of(10, 20, 30, 40, 50, 60)},
  rightOffsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 3, 4)},
  rightNeighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 3, 1)},
  rightWeights: {format: 'float32', values: Float32Array.of(0.5, 1.5, 2.5, 3.5)},
  rightDistances: {format: 'float32', values: Float32Array.of(1, 2, 3, 4)},
  selfWeights: {format: 'float32', values: Float32Array.of(0, 0.5, 1, 2)},
  mask: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1)},
  groupIds: {format: 'uint32', values: Uint32Array.of(1, 0, 1, 0)}
};
const OUTPUT = {
  offsets: 'offsets',
  neighbors: 'neighbors',
  weights: 'weights',
  status: {overflow: 'overflow', requiredCount: 'requiredCount'}
};
const BASE = {rowCount: 4, capacity: 32, output: OUTPUT};
const CASES: SpatialWeightsAlgebraOperationProps[] = [
  ...(['union', 'intersection', 'difference', 'symmetricDifference'] as const).map(operation => ({
    ...BASE,
    operation,
    left: SOURCE,
    right: RIGHT,
    weightRule: 'sum' as const,
    output: {...OUTPUT, distances: 'distances'}
  })),
  {...BASE, operation: 'higherOrder', weights: SOURCE, order: 3, workCapacity: 64},
  {
    ...BASE,
    operation: 'higherOrder',
    weights: SOURCE,
    order: 2,
    cumulative: true,
    workCapacity: 64
  },
  {
    ...BASE,
    operation: 'selfWeight',
    weights: SOURCE,
    selfWeight: 'selfWeights',
    output: {...OUTPUT, distances: 'distances'}
  },
  {
    ...BASE,
    operation: 'subgraph',
    weights: SOURCE,
    mask: 'mask',
    output: {...OUTPUT, distances: 'distances'}
  },
  {...BASE, operation: 'block', groupIds: 'groupIds', groupCount: 2},
  ...(['cell-major', 'period-major'] as const).map(layout => ({
    ...BASE,
    rowCount: 12,
    capacity: 80,
    operation: 'spaceTime' as const,
    weights: SOURCE,
    periodCount: 3,
    layout,
    lags: [
      {lag: -1, weight: 0.5, includeSelf: true},
      {lag: 0, weight: 2},
      {lag: 1, weight: 3, includeSelf: true}
    ]
  })),
  {...BASE, capacity: 1, operation: 'block', groupIds: 'groupIds', groupCount: 2},
  {...BASE, operation: 'subgraph', weights: SOURCE, mask: 'mask', id: 'empty'},
  {
    ...BASE,
    operation: 'selfWeight',
    weights: SOURCE,
    selfWeight: 'selfWeights',
    id: 'replace-self',
    output: {...OUTPUT, distances: 'distances'}
  },
  {
    ...BASE,
    rowCount: 6,
    capacity: 1,
    operation: 'higherOrder',
    weights: SOURCE,
    order: 2,
    workCapacity: 32,
    id: 'generation-prefix'
  }
];

for (const [index, props] of CASES.entries())
  it(`algebra CPU/GPU agree on every output: ${index} ${props.operation}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    let inputs =
      props.id === 'empty'
        ? {...INPUTS, mask: {format: 'uint32' as const, values: new Uint32Array(4)}}
        : INPUTS;
    if (props.id === 'replace-self')
      inputs = {
        ...inputs,
        sourceNeighbors: {format: 'uint32', values: Uint32Array.of(0, 3, 0, 2, 3, 0)}
      };
    if (props.id === 'generation-prefix')
      inputs = {
        ...inputs,
        sourceOffsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 5, 5, 5, 5)},
        sourceNeighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 5, 3, 4)},
        sourceWeights: {format: 'float32', values: new Float32Array(5).fill(1)},
        sourceDistances: {format: 'float32', values: new Float32Array(5)}
      };
    const outputs = [
      ...Object.values(props.output).filter((value): value is string => typeof value === 'string'),
      ...Object.values(props.output.status)
    ];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialWeightsAlgebraOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSpatialWeightsAlgebraOperation(props)],
      outputs
    });
    // Integer/copy outputs are exact. All weights here are dyadic, so their sums and products
    // are exactly representable too: tolerance zero, with numeric equality for signed zero.
    // The CSR contract leaves slots beyond offsets[rowCount] unspecified on the GPU.
    const slotOutputs = [props.output.neighbors, props.output.weights, props.output.distances];
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      const length = slotOutputs.includes(output)
        ? cpu[props.output.offsets][props.rowCount]
        : cpu[output].length;
      for (let slot = 0; slot < length; slot++)
        expect(gpu[output][slot] === cpu[output][slot], `${output}[${slot}]`).toBe(true);
    }
  });

const REVIEW_INFORMATION = {rowCount: 2, transform: 'raw', symmetric: true} as const;
const REVIEW_CONTEXT = {coordinateSpace: 'planar', metric: 'native', units: 'meters'} as const;
const REVIEW_PROPS = {
  id: 'review-algebra',
  operation: 'union',
  rowCount: 2,
  capacity: 2,
  left: {...SOURCE, information: REVIEW_INFORMATION, spatialContext: REVIEW_CONTEXT},
  right: {...RIGHT, information: REVIEW_INFORMATION, spatialContext: REVIEW_CONTEXT},
  output: {...OUTPUT, distances: 'distances'}
} satisfies SpatialWeightsAlgebraOperationProps;
const REVIEW_INPUTS: Record<string, OperationGraphInput> = {
  sourceOffsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
  sourceNeighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
  sourceWeights: {format: 'float32', values: Float32Array.of(1, 1)},
  sourceDistances: {format: 'float32', values: Float32Array.of(10, 20)},
  rightOffsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
  rightNeighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
  rightWeights: {format: 'float32', values: Float32Array.of(1, 1)},
  rightDistances: {format: 'float32', values: Float32Array.of(10, 20)}
};
const REVIEW_CASES: {name: string; props: SpatialWeightsAlgebraOperationProps; error?: RegExp}[] = [
  {name: 'compatible contexts and metadata', props: REVIEW_PROPS},
  {name: 'undeclared right context and metadata', props: {...REVIEW_PROPS, right: {...RIGHT}}},
  {
    name: 'incompatible contexts',
    props: {
      ...REVIEW_PROPS,
      right: {
        ...REVIEW_PROPS.right,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        }
      }
    },
    error: /^review-algebra incompatible spatial contexts/
  }
];
for (const operand of ['left', 'right'] as const)
  for (const [field, information] of [
    ['rowCount', {...REVIEW_INFORMATION, rowCount: 3}],
    ['transform', {...REVIEW_INFORMATION, transform: 'invalid' as 'raw'}],
    ['symmetric', {...REVIEW_INFORMATION, symmetric: 'invalid' as 'unknown'}]
  ] as const)
    REVIEW_CASES.push({
      name: `${operand} metadata ${field}`,
      props: {...REVIEW_PROPS, [operand]: {...REVIEW_PROPS[operand], information}},
      error: new RegExp(`^review-algebra ${operand}\\.information\\.${field}`)
    });

for (const scene of REVIEW_CASES)
  it(`algebra review regression: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const outputs = ['offsets', 'neighbors', 'weights', 'distances', 'overflow', 'requiredCount'];
    for (const createOperation of [
      () => new GPUSpatialWeightsAlgebraOperation(scene.props),
      () => new CPUSpatialWeightsAlgebraOperation(scene.props)
    ]) {
      // The rejection is a host contract: GPU lowering builds the contributor before dispatch.
      const run = async () =>
        runOperationGraph(device, {
          inputs: REVIEW_INPUTS,
          operations: [createOperation()],
          outputs
        });
      if (scene.error) await expect(run()).rejects.toThrow(scene.error);
      else {
        const result = await run();
        // Identical CSR sets have this exact union; distances are copied from the left.
        const expected: Record<string, number[]> = {
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [1, 1],
          distances: [10, 20],
          overflow: [0],
          requiredCount: [2]
        };
        for (const output of outputs) expect(Array.from(result[output])).toEqual(expected[output]);
      }
    }
  });
