// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUSpatialLagImpactsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation';
import {
  GPUSpatialLagImpactsOperation,
  type SpatialLagImpactsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixture from './likelihood-series-audit-fixture.json';

function createProps(): SpatialLagImpactsOperationProps {
  return {
    coefficients: 'coefficients',
    coefficientStride: 1,
    rhoIndex: 3,
    predictorCount: 2,
    rowCount: 8,
    maximumPower: 8,
    traces: {powerTraces: 'powerTraces', powerSums: 'powerSums', infinityNorm: 'infinityNorm'},
    output: {impacts: 'impacts'}
  };
}

it('CPU impacts retains the validated series length, predictor count and output after caller mutation', async () => {
  const props = createProps();
  const operation = new CPUSpatialLagImpactsOperation(props);
  props.maximumPower = 1;
  props.predictorCount = 1;
  props.output.impacts = 'changed';
  const vector = (values: Float32Array) => ({
    format: 'float32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await operation.execute({
    inputs: {
      coefficients: vector(Float32Array.of(99, 2, -3, 0.3)),
      powerTraces: vector(Float32Array.from(fixture.traces.exact.powerTraces)),
      powerSums: vector(Float32Array.from(fixture.traces.exact.powerSums)),
      infinityNorm: vector(Float32Array.of(1))
    }
  });
  expect(Object.keys(result)).toEqual(['impacts']);
  expectRegressionRowsClose(result['impacts'] as Float32Array, fixture.impacts, 3e-7);
});

it('GPU impacts lowering and its contributor retain validated topology after caller mutation', () => {
  const props = createProps();
  const operation = new GPUSpatialLagImpactsOperation(props);
  props.maximumPower = 1;
  props.predictorCount = 1;
  props.traces.powerTraces = 'changed';
  const contributor = constructRegressionOperationContributor(operation, {
    coefficients: {format: 'float32', length: 4},
    powerTraces: {format: 'float32', length: 8},
    powerSums: {format: 'float32', length: 8},
    infinityNorm: {format: 'float32', length: 1}
  }) as import('../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts').GPUSpatialLagImpacts;
  expect(contributor.props.maximumPower).toBe(8);
  expect(contributor.props.predictorCount).toBe(2);
  expect(Object.isFrozen(contributor.props)).toBe(true);
  expect(Object.isFrozen(contributor.props.traces)).toBe(true);
  expect(Object.isFrozen(contributor.props.output)).toBe(true);
});
