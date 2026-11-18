// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPURasterFocalPairs,
  GPU_RASTER_FOCAL_PAIRS_STATUS,
  type GPURasterFocalPairsProps
} from '../../../src/raster-analysis/focal-pairs/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import reference from '../../oracles/fixtures/raster-focal-pairs.json';
import {FIXTURES, METRICS} from './focal-pairs-fixtures';

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 5, height: 4});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length = 20) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const predictorValues = view('float32');
  const props: GPURasterFocalPairsProps = {
    grid,
    radius: 8,
    borderMode: 'reflect',
    noDataPolicy: 'propagate',
    propagateCenterNoData: true,
    minimumCount: 289,
    covarianceNormalization: 'population',
    predictor: {
      id: 'predictor',
      format: 'float32',
      storage: {kind: 'buffer', values: predictorValues},
      validity: view('uint32'),
      scale: 2,
      offset: 3,
      noDataValue: -999
    },
    response: {
      id: 'response',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32')},
      validity: view('uint32')
    },
    output: {
      pairedCount: view('uint32'),
      status: view('uint32'),
      covariance: view('float32'),
      correlation: view('float32'),
      slope: view('float32'),
      intercept: view('float32'),
      rSquared: view('float32'),
      residualSumSquares: view('float32')
    }
  };
  return {device, graph, props, view, predictorValues};
}

it('RS05 maximal output shaders parse within default binding limits and declare the full halo', () => {
  const {device, graph, props} = setup();
  try {
    for (const borderMode of ['nodata', 'clamp', 'reflect'] as const) {
      const contributor = new GPURasterFocalPairs({
        ...props,
        id: `focal-${borderMode}`,
        borderMode
      });
      expect(contributor.requiredHalo).toBe(8);
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(sources.size).toBe(4);
      for (const {source} of sources.configurations)
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    }
  } finally {
    device.destroy();
  }
});

it('RS05 rejects unsupported radii, counts, output capacities, aliases and graph ownership', () => {
  const {device, graph, props, view, predictorValues} = setup();
  try {
    for (const radius of [-1, 0.5, 9, Infinity])
      expect(() => new GPURasterFocalPairs({...props, radius})).toThrow();
    for (const minimumCount of [0, 1, 2.5, 290])
      expect(() => new GPURasterFocalPairs({...props, minimumCount})).toThrow();
    expect(
      () =>
        new GPURasterFocalPairs({
          ...props,
          output: {pairedCount: props.output.pairedCount, status: props.output.status}
        })
    ).toThrow();
    expect(
      () =>
        new GPURasterFocalPairs({...props, output: {...props.output, slope: view('float32', 19)}})
    ).toThrow();
    expect(
      () =>
        new GPURasterFocalPairs({
          ...props,
          output: {...props.output, intercept: props.output.slope}
        })
    ).toThrow();
    expect(
      () =>
        new GPURasterFocalPairs({
          ...props,
          predictor: {
            id: 'short-predictor',
            format: 'float32',
            storage: {kind: 'buffer', values: view('float32', 19)}
          }
        })
    ).toThrow();
    const foreign = new GPUCommandGraph(device);
    const contributor = new GPURasterFocalPairs({
      ...props,
      output: {...props.output, covariance: createTransientView(foreign, 'foreign', 'float32', 20)}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow();
    expect(
      () =>
        new GPURasterFocalPairs({
          ...props,
          output: {...props.output, covariance: predictorValues}
        })
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('RS05 reference pins independent library provenance and retains all diagnostic/output rows', () => {
  expect(reference.provenance.versions.numpy).toBe('2.5.4');
  const generator = readFileSync(new URL('../../oracles/raster-focal-pairs.py', import.meta.url));
  expect(createHash('sha256').update(generator).digest('hex')).toBe(
    reference.provenance.generatorSha256
  );
  for (const fixture of FIXTURES) {
    for (const values of Object.values(fixture.expected))
      expect(values).toHaveLength(fixture.width * fixture.height);
    expect(
      fixture.expected.pairedCount.every(count => count <= (2 * fixture.radius + 1) ** 2)
    ).toBe(true);
  }
  const positive = FIXTURES.find(fixture => fixture.name === 'positive')!;
  expect(positive.expected.pairedCount[7]).toBe(9);
  expect(positive.expected.slope[7]).toBeCloseTo(2, 12);
  expect(positive.expected.intercept[7]).toBeCloseTo(3, 12);
  const singular = FIXTURES.find(fixture => fixture.name === 'constant-predictor')!;
  expect(new Set(singular.expected.status)).toEqual(
    new Set([GPU_RASTER_FOCAL_PAIRS_STATUS.constantPredictor])
  );
  expect(singular.expected.covariance.every(value => value === 0)).toBe(true);
  expect(singular.expected.slope.every(value => value === null)).toBe(true);
  const empty = FIXTURES.find(fixture => fixture.name === 'empty-pairs')!;
  for (const metric of METRICS)
    expect(empty.expected[metric].every(value => value === null)).toBe(true);
});
