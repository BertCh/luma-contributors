// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPURasterFractionalCoverage} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {createScalabilityProps} from './raster-fractional-scalability/raster-fractional-scalability-fixture';
import {computeRasterFractionalZonalStatisticsOnCPU} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {
  computeRasterFractionalZonalDistributionOnCPU,
  CPURasterFractionalZonalDistributionOperation
} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import {GPURasterFractionalZonalDistributionOperation} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import {
  createSingletonStatisticsInput,
  createSingletonStatisticsProps
} from './raster-fractional-zonal-statistics-operation-fixture';
import {
  createUnitDistributionInput,
  getDistributionGraphInputs,
  DISTRIBUTION_PROPS
} from './fractional-zonal-distribution/raster-fractional-zonal-distribution-operation-fixture';

it.each([
  0, 1
])('G5 distribution reads only the first logical status word: candidate=%s', async loss => {
  const input = createUnitDistributionInput(Float32Array.of(7), Float32Array.of(0.5));
  const padded = (first: number, spare: number) =>
    Uint32Array.of(99, first, spare, 99).subarray(1, 3);
  input.coverage.count = padded(1, 0xffffffff);
  input.coverage.requiredCount = padded(1, 0xffffffff);
  input.coverage.overflow = padded(0, 1);
  input.coverage.candidateOverflow = padded(loss, 1 - loss);
  const props = {
    ...DISTRIBUTION_PROPS,
    grid: createSingletonStatisticsProps().grid,
    probabilityCount: 1,
    coverage: {
      ...DISTRIBUTION_PROPS.coverage,
      status: {...DISTRIBUTION_PROPS.coverage.status, candidateOverflow: 'candidateOverflow'}
    }
  };
  const inputs = {
    ...getDistributionGraphInputs(input),
    candidateOverflow: {format: 'uint32' as const, values: input.coverage.candidateOverflow}
  };
  expect
    .soft(() =>
      constructRasterOperationContributor(
        new GPURasterFractionalZonalDistributionOperation(props),
        Object.fromEntries(
          Object.entries(inputs).map(([name, value]) => [
            name,
            {format: value.format, length: value.values.length}
          ])
        )
      )
    )
    .not.toThrow();
  const result = computeRasterFractionalZonalDistributionOnCPU(input, props);
  expect(result.validity[0]).toBe(1 - loss);
  // A singleton has insufficient sample-variance support (8), but usable mean/area.
  expect(result.status[0]).toBe(8 | loss);
  if (loss) {
    expect(result.means[0]).toBeNaN();
    expect(result.coveredArea[0]).toBeNaN();
  } else {
    expect(result.means[0]).toBe(7);
    expect(result.coveredArea[0]).toBe(1);
  }
  const operationResult = await new CPURasterFractionalZonalDistributionOperation(props).execute({
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, input]) => [
        name,
        {
          format: input.format,
          length: input.values.length,
          chunks: [{values: input.values, length: input.values.length}],
          pack: () => input.values
        }
      ])
    )
  });
  expect(operationResult['validity']).toEqual(result.validity);
  expect(operationResult['status']).toEqual(result.status);
  expect(operationResult['means']).toEqual(result.means);
  expect(operationResult['coveredArea']).toEqual(result.coveredArea);
});

it.each([
  'count',
  'requiredCount',
  'overflow',
  'candidateOverflow'
] as const)('G5 distribution still refuses an empty coverage status: %s', name => {
  const input = createUnitDistributionInput(Float32Array.of(7), Float32Array.of(0.5));
  input.coverage[name] = new Uint32Array(0);
  expect(() =>
    computeRasterFractionalZonalDistributionOnCPU(input, {
      ...DISTRIBUTION_PROPS,
      grid: createSingletonStatisticsProps().grid
    })
  ).toThrow(`coverage.${name} must have at least one row`);
});

it('F14 candidate exhaustion publishes a sentinel independently of final tuple overflow', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {coverage} = createScalabilityProps(graph, 1, 1);
  const kernels = getGPUCommandNodeKernelSources(
    new GPURasterFractionalCoverage({...coverage, candidateCapacity: 1}).getCommandNodes(graph)
  );
  const source = kernels
    .find(
      kernel =>
        kernel.id.endsWith('-status') &&
        kernel.source.includes('candidateTotal[candidateTotalOffset]')
    )!
    .source.replaceAll(/override (\w+): u32;/g, 'const $1: u32 = 1u;');
  for (const exhausted of [true, false]) {
    const count = new Uint32Array(1);
    const required = new Uint32Array(1);
    const overflow = new Uint32Array(1);
    const candidateOverflow = new Uint32Array(1);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
      0: {
        0: Uint32Array.of(0, exhausted ? 1 : 2, 0),
        1: Uint32Array.of(exhausted ? 2 : 0),
        2: count,
        3: required,
        4: overflow,
        5: candidateOverflow
      }
    });
    expect([count[0], required[0], overflow[0], candidateOverflow[0]]).toEqual(
      exhausted ? [1, 0xffffffff, 0, 1] : [1, 2, 1, 0]
    );
  }
});

it.each([
  'count-mismatch',
  'candidate-only'
])('F1 scalar and distribution suppress %s prepared coverage', scenario => {
  const scalar = createSingletonStatisticsInput();
  scalar.input.scale = 1;
  scalar.input.offset = 0;
  scalar.input.values[0] = 7;
  const distribution = createUnitDistributionInput(Float32Array.of(7), Float32Array.of(0.5));
  for (const input of [scalar, distribution]) {
    if (scenario === 'count-mismatch') input.coverage.requiredCount[0] = 2;
    else Object.assign(input.coverage, {candidateOverflow: Uint32Array.of(1)});
  }
  const scalarResult = computeRasterFractionalZonalStatisticsOnCPU(
    scalar,
    createSingletonStatisticsProps()
  );
  expect.soft(scalarResult.validity[0]).toBe(0);
  expect.soft(scalarResult.means[0]).toBeNaN();
  const distributionResult = computeRasterFractionalZonalDistributionOnCPU(distribution, {
    ...DISTRIBUTION_PROPS,
    grid: createSingletonStatisticsProps().grid
  });
  expect(distributionResult.validity[0]).toBe(0);
  expect(distributionResult.status[0] & 1).toBe(1);
  expect(distributionResult.means[0]).toBeNaN();
});

it('F2 distribution suppresses effective geographic cross-axis parameters', () => {
  const input = createUnitDistributionInput(Float32Array.of(7, 7), Float32Array.of(0.5));
  input.cellAreas = undefined;
  input.parameters = Float32Array.of(0, 70, 1, 0, 5, -1, -1, 0);
  const result = computeRasterFractionalZonalDistributionOnCPU(input, {
    ...DISTRIBUTION_PROPS,
    grid: {...DISTRIBUTION_PROPS.grid, width: 2, height: 1, affine: [1, 0, 0, 0, -1, 70]},
    cellSizeMode: 'geographic'
  });
  expect(result.validity[0]).toBe(0);
  expect(result.coveredArea[0]).toBeNaN();
});

it('F7 one observation with normal float32 coverage 1e-32 keeps mean 7', () => {
  const input = createSingletonStatisticsInput();
  input.input.scale = 1;
  input.input.offset = 0;
  input.input.values[0] = 7;
  input.coverage.fractions[0] = 1e-32;
  const result = computeRasterFractionalZonalStatisticsOnCPU(
    input,
    createSingletonStatisticsProps()
  );
  expect(result.validity[0]).toBe(1);
  expect(result.means[0]).toBe(7);
});
