// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUContiguityWeightsOperation,
  type ContiguityWeightsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';

const props: ContiguityWeightsOperationProps = {
  id: 'named-contiguity',
  criterion: 'rook',
  polygons: {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets'
  },
  featureCount: 2,
  capacity: 2,
  output: {
    offsets: 'offsets',
    neighbors: 'neighbors',
    weights: 'weights',
    status: {overflow: 'overflow', candidateOverflow: 'candidateLoss'}
  }
};

it('OUT-04 retains the supported named candidate-overflow destination', () => {
  const operation = new GPUContiguityWeightsOperation(props);
  expect(operation.outputs['candidateOverflow']).toEqual({
    value: 'candidateLoss',
    format: 'uint32',
    length: 1
  });
});

for (const inherited of [false, true]) {
  for (const name of ['neighborCounts', 'customDestination']) {
    it(`review B3 CPU rejects ${inherited ? 'inherited' : 'own'} output.${name}`, () => {
      const output = {...props.output, ...(!inherited ? {[name]: 'requestedCounts'} : {})};
      if (inherited) Object.setPrototypeOf(output, {[name]: 'requestedCounts'});
      expect(() => new CPUContiguityWeightsOperation({...props, output})).toThrow(`output.${name}`);
    });
    it(`review B3 rejects ${inherited ? 'inherited' : 'own'} output.${name}`, () => {
      const output = {...props.output, ...(!inherited ? {[name]: 'requestedCounts'} : {})};
      if (inherited) Object.setPrototypeOf(output, {[name]: 'requestedCounts'});
      expect(() => new GPUContiguityWeightsOperation({...props, output})).toThrow(`output.${name}`);
    });
  }
  for (const name of ['count', 'converged', 'complete', 'iterationLimitReached', 'unknown']) {
    it(`OUT-04 rejects ${inherited ? 'inherited' : 'own'} output.status.${name}`, () => {
      const status = {...props.output.status, ...(!inherited ? {[name]: 'unwritten'} : {})};
      if (inherited) Object.setPrototypeOf(status, {[name]: 'unwritten'});
      expect(
        () =>
          new GPUContiguityWeightsOperation({
            ...props,
            output: {...props.output, status}
          })
      ).toThrow(`output.status.${name}`);
    });
  }
  it(`OUT-04 rejects ${inherited ? 'inherited' : 'own'} output.distances`, () => {
    const output = {...props.output, ...(!inherited ? {distances: 'unwritten'} : {})};
    if (inherited) Object.setPrototypeOf(output, {distances: 'unwritten'});
    expect(() => new GPUContiguityWeightsOperation({...props, output})).toThrow('output.distances');
  });
}
