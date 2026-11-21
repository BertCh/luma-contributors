// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {findConstantNonFiniteBitcasts} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  getGPURasterZonalStatisticsParameterValues,
  GPU_RASTER_ZONAL_STATISTICS_NO_ZONE,
  GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH,
  GPURasterZonalStatistics,
  type GPURasterZonalStatisticsProps
} from '../../../src/raster-analysis/raster-zonal-statistics/index';
import {getRasterZonalCellArea} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {computeZonalStatistics} from './raster-zonal-statistics-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const WIDTH = 4;
const HEIGHT = 3;
const CAPACITY = 5;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURasterZonalStatisticsProps> = {}
): GPURasterZonalStatisticsProps {
  return {
    id: 'zonal',
    grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
    zones: createTransientView(graph, 'zones', 'uint32', WIDTH * HEIGHT),
    values: {
      id: 'values',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: createTransientView(graph, 'values', 'float32', WIDTH * HEIGHT)
      }
    },
    zoneCapacity: CAPACITY,
    output: {means: createTransientView(graph, 'means', 'float32', CAPACITY)},
    ...overrides
  };
}

function getNodeIds(props: GPURasterZonalStatisticsProps, graph: GPUCommandGraph): string[] {
  return new GPURasterZonalStatistics(props).getCommandNodes(graph).map(node => node.id);
}

it('GPURasterZonalStatistics prefixes IDs and schedules only requested columns', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(GPU_RASTER_ZONAL_STATISTICS_NO_ZONE).toBe(0xffffffff);
  const meanIds = getNodeIds(createProps(graph), graph);
  expect(meanIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(meanIds).toContain('zonal-mask');
  expect(meanIds.some(id => id.startsWith('zonal-means'))).toBe(true);
  expect(meanIds.some(id => id.startsWith('zonal-overflow'))).toBe(false);

  // cellCounts alone needs no band canonicalization and no value mask.
  const countGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const countIds = getNodeIds(
    createProps(countGraph, {
      output: {
        cellCounts: createTransientView(countGraph, 'cell-counts', 'uint32', CAPACITY)
      }
    }),
    countGraph
  );
  expect(countIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(countIds.some(id => id.includes('values-elevation'))).toBe(false);
  expect(countIds.some(id => id.startsWith('zonal-means'))).toBe(false);
  expect(countIds.length).toBeLessThan(meanIds.length);

  const fullGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const fullIds = getNodeIds(
    createProps(fullGraph, {
      output: {
        cellCounts: createTransientView(fullGraph, 'o0', 'uint32', CAPACITY),
        valueCounts: createTransientView(fullGraph, 'o1', 'uint32', CAPACITY),
        sums: createTransientView(fullGraph, 'o2', 'float32', CAPACITY),
        means: createTransientView(fullGraph, 'o3', 'float32', CAPACITY),
        minimums: createTransientView(fullGraph, 'o4', 'float32', CAPACITY),
        maximums: createTransientView(fullGraph, 'o5', 'float32', CAPACITY),
        status: {overflow: createTransientView(fullGraph, 'overflow', 'uint32', 1)}
      }
    }),
    fullGraph
  );
  expect(fullIds).toContain('zonal-overflow-reset');
  // Counts, minimums and maximums share one workgroup-private pass (no per-column aggregation).
  expect(fullIds).toContain('zonal-zone-reduce');
  expect(fullIds).toContain('zonal-zone-reduce-decode');
  for (const name of ['cellCounts', 'valueCounts', 'minimums', 'maximums']) {
    expect(fullIds.some(id => id.startsWith(`zonal-${name}`))).toBe(false);
  }
  // The default sorted order computes sums in the shared `zonal-sorted-*` nodes.
  expect(fullIds.some(id => id.startsWith('zonal-sorted-'))).toBe(true);
  expect(fullIds).toContain('zonal-means');
});

function expectConstructorToThrow(
  createOverrides: (graph: GPUCommandGraph) => Partial<GPURasterZonalStatisticsProps>,
  pattern?: RegExp
): void {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPURasterZonalStatistics(createProps(graph, createOverrides(graph)))).toThrow(
    pattern
  );
}

it('GPURasterZonalStatistics validates its props', () => {
  expectConstructorToThrow(
    () => ({grid: makeGPURasterMetadata({width: 0, height: HEIGHT})}),
    /dimensions/
  );
  expectConstructorToThrow(
    graph => ({
      zones: createTransientView(graph, 'short-zones', 'uint32', 3)
    }),
    /zones/
  );
  // @ts-expect-error Runtime rejection remains necessary for JavaScript callers.
  expectConstructorToThrow(() => ({output: {}}), /at least one output/);
  expectConstructorToThrow(() => ({zoneCapacity: 0}), /zoneCapacity/);
  expectConstructorToThrow(() => ({zoneCapacity: 2 ** 32}), /zoneCapacity/);
  expectConstructorToThrow(() => ({ignoredZone: -1}), /ignoredZone/);
  expectConstructorToThrow(() => ({ignoredZone: 1.5}), /ignoredZone/);
  expectConstructorToThrow(
    graph => ({
      output: {
        means: createTransientView(graph, 'wrong-length', 'float32', CAPACITY + 1)
      }
    }),
    /zoneCapacity rows/
  );
  expectConstructorToThrow(graph => ({
    output: {
      means: createTransientView(graph, 'wrong-format', 'uint32', CAPACITY) as never
    }
  }));
  expectConstructorToThrow(
    graph => ({
      output: {
        sums: createTransientView(graph, 'sums', 'float32', CAPACITY),
        status: {overflow: createTransientView(graph, 'overflow-two', 'uint32', 2)}
      }
    }),
    /overflow/
  );
});

it('GPURasterZonalStatistics rejects aliasing and foreign graphs', () => {
  expectConstructorToThrow(graph => {
    const shared = createTransientView(graph, 'shared', 'float32', CAPACITY);
    return {output: {sums: shared, means: shared}};
  }, /logical byte ranges must not overlap/);
  expectConstructorToThrow(graph => {
    const zones = createTransientView(graph, 'aliased-zones', 'uint32', WIDTH * HEIGHT);
    return {
      zones,
      zoneCapacity: WIDTH * HEIGHT,
      output: {cellCounts: zones}
    };
  }, /logical byte ranges must not overlap/);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() =>
    new GPURasterZonalStatistics(createProps(graph)).getCommandNodes(otherGraph)
  ).toThrow(/target graph/);
});

it('GPURasterZonalStatistics sumOrder defaults to sorted and atomic skips the sort nodes', () => {
  const outputs = (graph: GPUCommandGraph) => ({
    sums: createTransientView(graph, 'o-sums', 'float32', CAPACITY),
    means: createTransientView(graph, 'o-means', 'float32', CAPACITY)
  });
  const defaultGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const defaultIds = getNodeIds(
    createProps(defaultGraph, {output: outputs(defaultGraph)}),
    defaultGraph
  );
  const sortedGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const sortedIds = getNodeIds(
    createProps(sortedGraph, {
      output: outputs(sortedGraph),
      sumOrder: 'sorted'
    }),
    sortedGraph
  );
  expect(sortedIds).toEqual(defaultIds);
  expect(defaultIds.some(id => id.startsWith('zonal-sorted-sort'))).toBe(true);
  expect(defaultIds).toContain('zonal-sorted-gather-sums');
  expect(defaultIds).toContain('zonal-means');

  const atomicGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const atomicIds = getNodeIds(
    createProps(atomicGraph, {
      output: outputs(atomicGraph),
      sumOrder: 'atomic'
    }),
    atomicGraph
  );
  expect(atomicIds.some(id => id.includes('-sorted-'))).toBe(false);
  expect(atomicIds.length).toBeLessThan(defaultIds.length);

  // Minimum-only requests never need the sort even when sorted is requested.
  const minGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const minIds = getNodeIds(
    createProps(minGraph, {
      output: {
        minimums: createTransientView(minGraph, 'o-min', 'float32', CAPACITY)
      },
      sumOrder: 'sorted'
    }),
    minGraph
  );
  expect(minIds.some(id => id.includes('-sorted-'))).toBe(false);

  expectConstructorToThrow(
    () => ({sumOrder: 'fast' as never}),
    /sumOrder must be atomic or sorted/
  );
});

// Geographic fixture: 2 x 3 cells of 10 degrees longitude, rows 60-40, 40-20 and 20-0 N.
// Expected values from python3 -I (closed form, numpy not installed): cell area
// R^2 dLon |sin(top) - sin(bottom)| with the WGS84 authalic R = 6371007.180918475 m; weighted
// variance sum w (v - m)^2 / (V1 - V2 / V1), the numpy.cov(v, aweights=w, ddof=1) formula.
const GEOGRAPHIC_SETTINGS = [10, 20, 60, 0];
const GEOGRAPHIC_ZONES = [0, 0, 0, 1, 1, 1];
const GEOGRAPHIC_VALUES = [1, 2, 3, 4, 5, 6];
const GEOGRAPHIC_ROW_AREAS = [1581471169984.1543, 2130710343385.227, 2422954403438.46];

it('getRasterZonalCellArea matches closed-form spherical row areas', () => {
  for (const [row, area] of GEOGRAPHIC_ROW_AREAS.entries()) {
    expect(getRasterZonalCellArea(GEOGRAPHIC_SETTINGS, row, 3, 1) / area).toBeCloseTo(1, 12);
    // Rows advancing north give the same areas.
    expect(getRasterZonalCellArea([10, 20, 0, 60], 2 - row, 3, 1) / area).toBeCloseTo(1, 12);
  }
});

it('the zonal oracle reproduces python area-weighted means and reliability variances', () => {
  const cellWeights = GEOGRAPHIC_ZONES.map((_, cell) =>
    getRasterZonalCellArea(GEOGRAPHIC_SETTINGS, Math.floor(cell / 2), 3, 1)
  );
  const weighted = computeZonalStatistics(GEOGRAPHIC_ZONES, GEOGRAPHIC_VALUES, {
    zoneCapacity: 2,
    cellWeights
  });
  expect(weighted.valueCounts).toEqual([3, 3]);
  const expected = [
    {weightSum: 5293652683353.535, sum: 11136544540108.145, mean: 2.1037542895717767},
    {weightSum: 6976619150262.146, sum: 35175339811363.97, mean: 5.041889066001582}
  ];
  const variances = [1.047000628016599, 0.9781397620488637];
  const deviations = [1.0232304862623078, 0.9890094853179435];
  for (const zone of [0, 1]) {
    expect(weighted.weightSums[zone] / expected[zone].weightSum).toBeCloseTo(1, 12);
    expect(weighted.sums[zone] / expected[zone].sum).toBeCloseTo(1, 12);
    expect(weighted.means[zone]).toBeCloseTo(expected[zone].mean, 12);
    expect(weighted.variances[zone]).toBeCloseTo(variances[zone], 12);
    expect(weighted.standardDeviations[zone]).toBeCloseTo(deviations[zone], 12);
  }
  // Uniform cells: plain mean and sum (v - mean)^2 / (n - 1) = 1 for 1,2,3 and 4,5,6.
  const uniform = computeZonalStatistics(GEOGRAPHIC_ZONES, GEOGRAPHIC_VALUES, {zoneCapacity: 2});
  expect(uniform.means).toEqual([2, 5]);
  expect(uniform.variances).toEqual([1, 1]);
  expect(uniform.weightSums).toEqual([3, 3]);
});

it('GPURasterZonalStatistics schedules weighted moments only when requested', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const ids = getNodeIds(
    createProps(graph, {
      cellSizeMode: 'geographic',
      parameterLayoutVersion: 1,
      parameters,
      output: {
        means: createTransientView(graph, 'o-means', 'float32', CAPACITY),
        weightSums: createTransientView(graph, 'o-weights', 'float32', CAPACITY),
        variances: createTransientView(graph, 'o-variances', 'float32', CAPACITY),
        standardDeviations: createTransientView(graph, 'o-deviations', 'float32', CAPACITY)
      }
    }),
    graph
  );
  expect(ids).toContain('zonal-means');
  expect(ids).toContain('zonal-deviations');
  expect(ids).toContain('zonal-variances');
  expect(ids).toContain('zonal-sorted-gather-weights');
  expect(ids).toContain('zonal-variance-sorted-gather-weight-squares');

  // Atomic order aggregates the same contributions without the sort.
  const atomicGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const atomicIds = getNodeIds(
    createProps(atomicGraph, {
      sumOrder: 'atomic',
      output: {
        standardDeviations: createTransientView(atomicGraph, 'o-deviations', 'float32', CAPACITY)
      }
    }),
    atomicGraph
  );
  expect(atomicIds.some(id => id.includes('-sorted-'))).toBe(false);
  expect(atomicIds).toContain('zonal-variances');

  // Planar requests without moments keep the previous node set.
  const planarGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const planarIds = getNodeIds(createProps(planarGraph), planarGraph);
  expect(planarIds.some(id => id.includes('deviations') || id.includes('weight'))).toBe(false);
});

it('GPURasterZonalStatistics validates geographic cell weighting', () => {
  expectConstructorToThrow(() => ({cellSizeMode: 'geographic'}), /need parameters/);
  expectConstructorToThrow(
    () => ({cellSizeMode: 'web-mercator' as never}),
    /cellSizeMode must be uniform or geographic/
  );
  expectConstructorToThrow(
    graph => ({parameters: createTransientView(graph, 'parameters', 'float32', 4)}),
    /parameters need cellSizeMode geographic/
  );
  expectConstructorToThrow(
    graph => ({
      cellSizeMode: 'geographic',
      parameterLayoutVersion: 1,
      parameters: createTransientView(graph, 'short-parameters', 'float32', 3)
    }),
    /parameters must contain at least 4/
  );
  expectConstructorToThrow(
    graph => ({
      cellSizeMode: 'geographic',
      parameters: createTransientView(graph, 'legacy-parameters', 'float32', 4)
    }),
    /exactly 7 values .* parameterLayoutVersion: 1/
  );
  expectConstructorToThrow(
    graph => ({
      output: {variances: createTransientView(graph, 'short-variances', 'float32', CAPACITY - 1)}
    }),
    /variances must contain zoneCapacity rows/
  );
});

it('GPURasterZonalStatistics skips the identity copy for canonical float32 buffer bands', () => {
  const countElevationNodes = (
    overrides: (graph: GPUCommandGraph) => Partial<GPURasterZonalStatisticsProps>
  ) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    return getNodeIds(createProps(graph, overrides(graph)), graph).filter(id =>
      id.includes('values-elevation')
    ).length;
  };
  const band = (
    graph: GPUCommandGraph,
    extra: {validity?: boolean; scale?: number; offset?: number; noDataValue?: number}
  ) => ({
    id: 'canonical',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: createTransientView(graph, 'canonical-values', 'float32', WIDTH * HEIGHT)
    },
    ...(extra.validity
      ? {validity: createTransientView(graph, 'canonical-validity', 'uint32', WIDTH * HEIGHT)}
      : {}),
    ...(extra.scale === undefined ? {} : {scale: extra.scale}),
    ...(extra.offset === undefined ? {} : {offset: extra.offset}),
    ...(extra.noDataValue === undefined ? {} : {noDataValue: extra.noDataValue})
  });
  // Already canonical: float32 buffer, validity, no calibration or nodata.
  expect(countElevationNodes(graph => ({values: band(graph, {validity: true})}))).toBe(0);
  // Without a caller mask the reducer supplies an all-eligible mask; its readers still reject
  // non-finite values. Conversion/calibration/nodata below still need canonicalization.
  expect(countElevationNodes(graph => ({values: band(graph, {})}))).toBe(0);
  expect(countElevationNodes(graph => ({values: band(graph, {validity: true, scale: 2})}))).toBe(1);
  expect(countElevationNodes(graph => ({values: band(graph, {validity: true, offset: 1})}))).toBe(
    1
  );
  expect(
    countElevationNodes(graph => ({values: band(graph, {validity: true, noDataValue: -9999})}))
  ).toBe(1);
});

it('GPURasterZonalStatistics zone-reduce kernel passes its workgroup size', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, {
    output: {minimums: createTransientView(graph, 'minimums', 'float32', CAPACITY)}
  });
  const sources = getGPUCommandNodeKernelSources(
    new GPURasterZonalStatistics(props).getCommandNodes(graph)
  ).filter(({source}) => source.includes('var<workgroup> local_'));
  expect(sources.length).toBe(1);
  expect(sources[0].source).toContain('@workgroup_size(256)');
  expect(sources[0].source).toContain('slot += 256u');
});

it('getGPURasterZonalStatisticsParameterValues writes the grid words', () => {
  expect(GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH).toBe(7);
  const grid = makeGPURasterMetadata({width: 2, height: 3, cellSize: [10, 20], origin: [0, 60]});
  expect(Array.from(getGPURasterZonalStatisticsParameterValues(grid))).toEqual([
    10, 20, 60, 0, -20, 30, 1
  ]);
  expect(Array.from(getGPURasterZonalStatisticsParameterValues(grid, new Float32Array(4)))).toEqual(
    [10, 20, 60, 0]
  );
  expect(() => getGPURasterZonalStatisticsParameterValues(grid, new Float32Array(3))).toThrow(
    /4 values/
  );
});

it('RT-137 guards counts-only and all zonal reductions after their normal publication', () => {
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = createProps(graph, {
      sumOrder,
      producerOverflow: createTransientView(graph, 'producer-overflow', 'uint32', 1),
      output: {cellCounts: createTransientView(graph, 'counts', 'uint32', CAPACITY)}
    });
    expect(() => new GPURasterZonalStatistics(props)).toThrow(/incomplete/);
    props.output.status = {incomplete: createTransientView(graph, 'incomplete', 'uint32', 1)};
    const ids = getNodeIds(props, graph);
    expect(ids).toContain('zonal-producer-cellCounts');
    expect(ids.at(-1)).toBe('zonal-producer-incomplete');
  }
});

it('RT-137 guards weighted moments and sparse dictionary result columns on both reduction routes', () => {
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    for (const sparse of [false, true]) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const column = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length = CAPACITY
      ) => createTransientView(graph, name, format, length);
      const props = createProps(graph, {
        sumOrder,
        cellSizeMode: 'geographic',
        parameters: column('parameters', 'float32', 7),
        producerOverflow: column('producer-overflow', 'uint32', 1),
        ...(sparse ? {zoneDictionary: column('dictionary', 'uint32')} : {}),
        output: {
          cellCounts: column('cell-counts', 'uint32'),
          valueCounts: column('value-counts', 'uint32'),
          sums: column('sums', 'float32'),
          means: column('weighted-means', 'float32'),
          minimums: column('minimums', 'float32'),
          maximums: column('maximums', 'float32'),
          weightSums: column('weights', 'float32'),
          variances: column('variances', 'float32'),
          standardDeviations: column('deviations', 'float32'),
          ...(sparse
            ? {
                sourceZoneIds: column('source-ids', 'uint32'),
                denseZoneRows: column('dense-rows', 'uint32')
              }
            : {}),
          status: {
            incomplete: column('incomplete', 'uint32', 1),
            ...(sparse ? {invalidCount: column('invalid-count', 'uint32', 1)} : {})
          }
        }
      });
      const nodes = new GPURasterZonalStatistics(props).getCommandNodes(graph);
      const ids = nodes.map(node => node.id);
      for (const name of Object.keys(props.output).filter(name => name !== 'status')) {
        expect(ids).toContain(`zonal-producer-${name}`);
      }
      expect(ids.at(-1)).toBe('zonal-producer-incomplete');
      const sources = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
        kernel.id.includes('-producer-')
      );
      expect(sources.length).toBe(Object.keys(props.output).length);
      for (const {id, source} of sources) {
        expect(source).toContain('producerOverflow[producerOverflowOffset]');
        expect(findConstantNonFiniteBitcasts(source), id).toEqual([]);
      }
    }
  }
});

it('CMP-09 captures loss-source lists and nested output maps at construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const loss = createTransientView(graph, 'loss', 'uint32', 1);
  const sources = [loss];
  const status = {incomplete: createTransientView(graph, 'incomplete', 'uint32', 1)};
  const output = {means: createTransientView(graph, 'captured-means', 'float32', CAPACITY), status};
  const props = createProps(graph, {incompleteSources: sources, output});
  const contributor = new GPURasterZonalStatistics(props);
  sources.length = 0;
  output.means = createTransientView(graph, 'replacement', 'float32', CAPACITY);
  status.incomplete = createTransientView(graph, 'replacement-status', 'uint32', 1);
  expect(contributor.props.incompleteSources).toEqual([loss]);
  expect(contributor.props.output.means).not.toBe(output.means);
  expect(contributor.props.output.status?.incomplete).not.toBe(status.incomplete);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain('zonal-producer-means');
});

it('t1 retains the identity copy when an unmasked input shares disjoint output storage', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const cellCount = WIDTH * HEIGHT;
  const pool = createTransientView(graph, 'shared-pool', 'float32', cellCount + CAPACITY);
  const props = createProps(graph, {
    values: {
      id: 'values',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: graph.createDataView(pool.buffer, {format: 'float32', length: cellCount})
      }
    },
    output: {
      means: graph.createDataView(pool.buffer, {
        format: 'float32',
        byteOffset: cellCount * 4,
        length: CAPACITY
      })
    }
  });
  // Logical ranges are disjoint but their aligned storage bindings overlap. The old copy made
  // this call safe; retain it instead of imposing a new alias rejection on a valid caller.
  const nodes = new GPURasterZonalStatistics(props).getCommandNodes(graph);
  expect(nodes.some(node => node.id.includes('values-elevation'))).toBe(true);
});
