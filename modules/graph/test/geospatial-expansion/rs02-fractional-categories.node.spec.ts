// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import fixture from '../oracles/fixtures/rs02-fractional-categories.json';
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterFractionalCategoricalStatistics,
  type GPURasterFractionalCategoricalStatisticsProps
} from '../../src/gpu-raster/gpu-raster-fractional-categorical-statistics';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

function makeProps(
  graph: GPUCommandGraph,
  categoryCapacity = 4,
  zoneCount = 3
): GPURasterFractionalCategoricalStatisticsProps {
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `rs02-${identifier++}`, format, length);
  return {
    grid: {width: 4, height: 2, affine: [2, 0, 0, 0, -3, 6], pixelInterpretation: 'area'},
    input: {
      id: 'categories',
      format: 'uint32',
      storage: {kind: 'buffer', values: view('uint32', 8)},
      validity: view('uint32', 8),
      noDataValue: 0xffffffff
    },
    zoneCount,
    categoryCapacity,
    coverage: {
      zoneIds: view('uint32', 16),
      cellIds: view('uint32', 16),
      fractions: view('float32', 16),
      zoneValidity: view('uint32', zoneCount),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    },
    output: {
      coverageSums: view('float32', zoneCount * categoryCapacity),
      classAreas: view('float32', zoneCount * categoryCapacity),
      proportions: view('float32', zoneCount * categoryCapacity),
      coveredArea: view('float32', zoneCount),
      validArea: view('float32', zoneCount),
      missingArea: view('float32', zoneCount),
      majority: view('uint32', zoneCount),
      minority: view('uint32', zoneCount),
      validity: view('uint32', zoneCount)
    }
  };
}

it('RS02 maximal-category kernels parse at the default storage binding limit', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const kernels = getParsedKernelSources(
    new GPURasterFractionalCategoricalStatistics(makeProps(graph, 65534, 1)).getCommandNodes(graph)
  );
  expect(kernels.size).toBeGreaterThan(7);
  for (const [name, source] of kernels)
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
});

it('RS02 rejects incompatible category shape, calibration, area and output ownership', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = makeProps(graph);
  for (const categoryCapacity of [0, 1.5, 65535])
    expect(
      () => new GPURasterFractionalCategoricalStatistics({...props, categoryCapacity})
    ).toThrow();
  for (const zoneCount of [0, 0.5, 13108])
    expect(() => new GPURasterFractionalCategoricalStatistics({...props, zoneCount})).toThrow();
  for (const input of [
    {...props.input, scale: 2},
    {...props.input, offset: 1},
    {...props.input, noDataValue: -1},
    {...props.input, noDataValue: 1.5}
  ])
    expect(() => new GPURasterFractionalCategoricalStatistics({...props, input})).toThrow();
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...props,
        grid: {...props.grid, affine: [1, 1, 0, 1, 1, 0]}
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...props,
        output: {...props.output, proportions: props.output.classAreas}
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...props,
        output: {...props.output, majority: props.input.storage.values}
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...props,
        output: {
          ...props.output,
          classAreas: createTransientView(graph, 'short-output', 'float32', 2)
        }
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...props,
        coverage: {
          ...props.coverage,
          fractions: createTransientView(graph, 'short-coverage', 'float32', 2)
        }
      })
  ).toThrow();
});

it('RS02 validates graph ownership and supports omitted category mask/nodata', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = makeProps(graph);
  const foreignGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() =>
    new GPURasterFractionalCategoricalStatistics(props).getCommandNodes(foreignGraph)
  ).toThrow();
  const kernels = getParsedKernelSources(
    new GPURasterFractionalCategoricalStatistics({
      ...props,
      input: {...props.input, validity: undefined, noDataValue: undefined}
    }).getCommandNodes(graph)
  );
  expect(kernels.size).toBeGreaterThan(7);
});

it('RS02 independent fixture pins fractional-support ties and missing-area conservation', () => {
  const generator = readFileSync(
    new URL('../oracles/rs02-fractional-categories.py', import.meta.url)
  );
  expect(createHash('sha256').update(generator).digest('hex')).toBe(fixture.generatorSha256);
  expect(fixture.versions.exactextract).toBe('0.3.0');
  const first = fixture.frames[0].expected;
  // Raw cell-count minority would choose category 0; fractional support chooses category 1.
  expect(first.coverageSums.slice(0, 6)).toEqual([1, 0.5, 1.25, 0.5, 0, 0]);
  expect(first.minority[0]).toBe(1);
  expect(first.majority[1]).toBe(2);
  expect(first.minority[1]).toBe(0);
  expect(first.coveredArea[2]).toBe(3);
  expect(first.missingArea[2]).toBe(3);
  expect(first.coveredArea[3]).toBe(0);
  for (const frame of fixture.frames) {
    for (let zone = 0; zone < fixture.zoneCount; zone++) {
      const start = zone * fixture.categoryCapacity;
      expect(
        frame.expected.classAreas
          .slice(start, start + fixture.categoryCapacity)
          .reduce((sum, area) => sum + area, 0)
      ).toBe(frame.expected.validArea[zone]);
      expect(frame.expected.validArea[zone] + frame.expected.missingArea[zone]).toBe(
        frame.expected.coveredArea[zone]
      );
    }
  }
});
