// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURasterSkeleton} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {GPURasterRankFilter} from '../../../src/raster-analysis/gpu-raster-rank-filter';
import {
  GPURasterFractionalCoverage,
  GPURasterFractionalZonalStatistics
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(format: Format, length = 4) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const coverage = {
    zoneIds: view('uint32'),
    cellIds: view('uint32'),
    fractions: view('float32'),
    zoneValidity: view('uint32', 1),
    status: {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  };
  const producer = {
    id: 'instrument 7',
    grid,
    zones: {
      positions: view('float32x2'),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2)
    },
    parameters: view('float32', 8),
    output: coverage
  };
  const consumer = {
    id: 'instrument 7',
    grid,
    input: makeGPURasterBand(view('float32'), grid),
    zoneCount: 1,
    coverage,
    quantity: 'intensive' as const,
    output: {
      totals: view('float32', 1),
      means: view('float32', 1),
      coveredArea: view('float32', 1),
      validArea: view('float32', 1),
      validity: view('uint32', 1)
    }
  };
  return {device, graph, grid, view, coverage, producer, consumer};
}

for (const route of ['producer', 'consumer'] as const) {
  for (const problem of ['format', 'strided', 'short', 'alias', 'foreign'] as const) {
    it(`B1 captures inherited status count before ${problem} validation on ${route}`, () => {
      const fixture = createFixture();
      const foreign = createFixture();
      try {
        const count =
          problem === 'format'
            ? fixture.view('float32', 1)
            : problem === 'strided'
              ? fixture.graph.createDataView(fixture.view('uint32', 2).buffer, {
                  format: 'uint32',
                  byteStride: 8,
                  length: 1
                })
              : problem === 'short'
                ? fixture.view('uint32', 0)
                : problem === 'foreign'
                  ? foreign.view('uint32', 1)
                  : route === 'producer'
                    ? fixture.coverage.zoneIds
                    : fixture.consumer.output.validity;
        Reflect.deleteProperty(fixture.coverage.status, 'count');
        Object.setPrototypeOf(fixture.coverage.status, {
          get count() {
            return count;
          }
        });
        const construct = () =>
          route === 'producer'
            ? new GPURasterFractionalCoverage(fixture.producer)
            : new GPURasterFractionalZonalStatistics(fixture.consumer);
        if (problem === 'foreign')
          expect(() => construct().getCommandNodes(fixture.graph)).toThrow(/graph/);
        else
          expect(construct).toThrow(
            problem === 'format'
              ? /status.count.*uint32/
              : problem === 'strided'
                ? /status.count.*packed/
                : problem === 'short'
                  ? /status.count.*at least 1/
                  : /overlap|distinct|alias/
          );
      } finally {
        fixture.device.destroy();
        foreign.device.destroy();
      }
    });
  }

  it(`B1 captures a valid inherited count once on ${route}`, () => {
    const fixture = createFixture();
    try {
      let count = fixture.coverage.status.count;
      const original = count;
      Reflect.deleteProperty(fixture.coverage.status, 'count');
      Object.setPrototypeOf(fixture.coverage.status, {
        get count() {
          return count;
        }
      });
      const contributor =
        route === 'producer'
          ? new GPURasterFractionalCoverage(fixture.producer)
          : new GPURasterFractionalZonalStatistics(fixture.consumer);
      count = fixture.view('uint32', 0);
      const captured =
        contributor instanceof GPURasterFractionalCoverage
          ? contributor.props.output.status.count
          : contributor.props.coverage.status.count;
      expect(captured).toBe(original);
      expect(() => contributor.getCommandNodes(fixture.graph)).not.toThrow();
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const path of ['input', 'output.skeleton', 'output.status.converged']) {
  it(`B2 rejects absent ${path} at skeleton construction`, () => {
    const fixture = createFixture();
    try {
      const props = {
        id: 'instrument 7',
        grid: fixture.grid,
        input: fixture.view('uint32'),
        output: {skeleton: fixture.view('uint32'), status: {converged: fixture.view('uint32', 1)}}
      };
      const parent =
        path === 'input' ? props : path === 'output.skeleton' ? props.output : props.output.status;
      Reflect.deleteProperty(parent, path.split('.').at(-1)!);
      expect(() => new GPURasterSkeleton(props)).toThrowError(
        expect.objectContaining({code: 'MISSING_PROP', id: 'instrument 7', path})
      );
    } finally {
      fixture.device.destroy();
    }
  });
}

it('B2 keeps genuinely optional iteration status leaves optional', () => {
  const fixture = createFixture();
  try {
    expect(
      () =>
        new GPURasterSkeleton({
          grid: fixture.grid,
          input: fixture.view('uint32'),
          output: {skeleton: fixture.view('uint32'), status: {converged: fixture.view('uint32', 1)}}
        })
    ).not.toThrow();
  } finally {
    fixture.device.destroy();
  }
});

it('S3 preserves instance identity and the named path on packed-layout failures', () => {
  const fixture = createFixture();
  try {
    const allocation = fixture.view('uint32', 8);
    const validity = fixture.graph.createDataView(allocation.buffer, {
      format: 'uint32',
      byteStride: 8,
      length: 4
    });
    const props = {
      id: 'instrument 7',
      grid: fixture.grid,
      input: fixture.consumer.input,
      radius: 1,
      output: {values: fixture.view('float32'), validity}
    };
    expect(() => new GPURasterRankFilter(props)).toThrowError(
      expect.objectContaining({id: 'instrument 7', path: 'output.validity', code: 'VIEW_FORMAT'})
    );
    expect(() =>
      makeGPURasterBand(fixture.view('float32'), fixture.grid, {id: 'instrument 7', validity})
    ).toThrowError(
      expect.objectContaining({id: 'instrument 7', path: 'validity', code: 'VIEW_FORMAT'})
    );
  } finally {
    fixture.device.destroy();
  }
});

it('N2 rejects unknown fractional quantity spellings at construction', () => {
  const fixture = createFixture();
  try {
    Reflect.set(fixture.consumer, 'quantity', 'total');
    expect(() => new GPURasterFractionalZonalStatistics(fixture.consumer)).toThrow(
      /quantity.*extensive.*density.*intensive/
    );
  } finally {
    fixture.device.destroy();
  }
});
