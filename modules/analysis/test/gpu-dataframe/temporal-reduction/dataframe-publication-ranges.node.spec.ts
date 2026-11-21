// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUForestInference} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {GPUCompositeScore} from '../../../src/dataframe-analysis/composite-indicators/gpu-composite-score';
import {GPUInequality} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {GPUCalendarBuckets} from '../../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {GPUTimeWindowFilter} from '../../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {GPUTemporalReduction} from '../../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction';
import {GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} from '../../../src/dataframe-analysis/composite-indicators/composite-score-parameters';
import {GPU_INEQUALITY_PARAMETER_LENGTH} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

const ROUTES = [
  'groups',
  'join',
  'forest',
  'composite',
  'inequality-parallel',
  'inequality-reference',
  'calendar',
  'filter-unsorted',
  'filter-sorted',
  'temporal-unsorted',
  'temporal-sorted'
] as const;

for (const route of ROUTES) {
  it(`DPC-125 ${route} validates logical writes and actual co-bound publication ranges`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const input = <Format extends 'uint32' | 'uint32x2' | 'sint32' | 'float32'>(
      format: Format,
      length = 1
    ) => createTransientView(graph, `input-${serial++}`, format, length);
    const backing = input('uint32', 128);
    const range = <Format extends 'uint32' | 'sint32' | 'float32'>(
      format: Format,
      byteOffset: number
    ) => graph.createDataView(backing.buffer, {format, length: 1, byteOffset});
    const create = (offset: number): GPUCommandNodeProducer => {
      switch (route) {
        case 'groups':
          return new GPUGroupStatistics({
            keys: input('uint32'),
            columns: [],
            output: {
              keys: range('uint32', 0),
              counts: range('uint32', offset),
              status: {count: input('uint32'), overflow: input('uint32')}
            }
          });
        case 'join':
          return new GPUKeyLookup({
            leftKeys: input('uint32'),
            rightKeys: input('uint32'),
            output: {matched: range('uint32', 0), rightRows: range('uint32', offset)}
          });
        case 'forest':
          return new GPUForestInference({
            model: {
              featureCount: 1,
              treeCount: 1,
              nodeCount: 3,
              outputCount: 1,
              maximumDepth: 1,
              task: 'regression',
              transform: 'identity'
            },
            rowCount: 1,
            features: input('float32'),
            nodes: input('uint32', 18),
            trees: input('uint32', 4),
            leafValues: input('float32', 3),
            parameters: input('float32', 2),
            output: {predictions: range('float32', 0), validity: range('uint32', offset)}
          });
        case 'composite':
          return new GPUCompositeScore({
            indicators: input('float32'),
            indicatorCount: 1,
            parameters: input('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
            output: {score: range('float32', 0), scaled: range('float32', offset)}
          });
        case 'inequality-parallel':
        case 'inequality-reference':
          return new GPUInequality({
            mode: route === 'inequality-parallel' ? 'parallel' : 'reference',
            values: input('float32'),
            zoneIds: input('uint32'),
            zoneCount: 1,
            parameters: input('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
            output: {gini: range('float32', 0), theilT: range('float32', offset)}
          });
        case 'calendar':
          return new GPUCalendarBuckets({
            timestamps: input('uint32x2'),
            parameters: input('sint32', 2),
            output: {year: range('sint32', 0), hour: range('uint32', offset)}
          });
        case 'filter-unsorted':
        case 'filter-sorted':
          return new GPUTimeWindowFilter({
            sorted: route === 'filter-sorted',
            timestamps: input('float32'),
            window: input('float32', 8),
            output: {
              rows: input('uint32'),
              status: {count: range('uint32', 0), overflow: range('uint32', offset)}
            }
          });
        case 'temporal-unsorted':
        case 'temporal-sorted':
          return new GPUTemporalReduction({
            sorted: route === 'temporal-sorted',
            cellIds: input('uint32'),
            timestamps: input('float32'),
            values: input('float32'),
            parameters: input('float32', 2),
            cellCount: 1,
            bucketCount: 1,
            output: {
              counts: range('uint32', 0),
              first: range('float32', offset),
              last: input('float32'),
              minimums: input('float32'),
              maximums: input('float32'),
              occupiedSlots: {
                ids: input('uint32'),
                status: {count: input('uint32'), overflow: input('uint32')}
              }
            }
          });
      }
    };
    try {
      // Logical [0,4) and [256,260) also have disjoint aligned bindings; sharing storage is legal.
      const producer = create(256);
      const nodes = producer.getCommandNodes(graph);
      expect(nodes.length).toBeGreaterThan(0);
      if (route === 'composite') {
        for (const suffix of ['tile-centered-sums', 'centered-means']) {
          const node = nodes.find(node => node.id.endsWith(`-${suffix}`))!;
          expect(node).toBeDefined();
          for (const source of getGPUCommandNodeKernelSources([node])) {
            expect(() => new WgslReflect(source.source)).not.toThrow();
            expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
          }
        }
      }
      // Logical [0,4) and [4,8) do not overlap, but their co-bound 256-byte prefixes do.
      // Rejection must happen in the constructor, before graph resources/nodes are declared.
      expect(() => create(4)).toThrow(/aligned binding.*must not overlap/);
    } finally {
      device.destroy();
    }
  });
}

it('DPC-125 group statistics preserves packed outputs beside chunked source-row writes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `mixed-${serial++}`, format, length);
  const backing = view('uint32', 64);
  const keys = createVectorView(
    'source-keys',
    'uint32',
    [1, 0, 2].map(length => view('uint32', length))
  );
  const values = createVectorView(
    'source-values',
    'float32',
    [1, 0, 2].map(length => view('float32', length))
  );
  const output = {
    keys: graph.createDataView(backing.buffer, {format: 'uint32', length: 1}),
    counts: view('uint32', 1),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  const create = (meanOffset: number) =>
    new GPUGroupStatistics({
      keys,
      columns: [
        {
          values,
          statistics: ['mean', 'zScore'],
          output: {
            means: graph.createDataView(backing.buffer, {
              format: 'float32',
              length: 1,
              byteOffset: meanOffset
            }),
            zScores: graph.createDataView(backing.buffer, {
              format: 'float32',
              length: 3,
              byteOffset: 8
            })
          }
        }
      ],
      output
    });
  try {
    // Keys [0,4), mean [4,8), z-scores [8,20) share an alignment window but dispatch separately.
    expect(create(4).getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(() => create(0)).toThrow(/output.keys.*columns\[0\].output.means.*logical.*overlap/);
  } finally {
    device.destroy();
  }
});

it('DPC-125 forest retains borrowed raster batches with disjoint shared output ranges', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const input = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `raster-${serial++}`, format, length);
  const backing = input('uint32', 128);
  const predictions = createVectorView(
    'raster-predictions',
    'float32',
    [0, 4].map(byteOffset =>
      graph.createDataView(backing.buffer, {format: 'float32', length: 1, byteOffset})
    )
  );
  const props = {
    model: {
      featureCount: 1,
      treeCount: 1,
      nodeCount: 3,
      outputCount: 1,
      maximumDepth: 1,
      task: 'regression' as const,
      transform: 'identity' as const
    },
    rowCount: 2,
    nodes: input('uint32', 18),
    trees: input('uint32', 4),
    leafValues: input('float32', 3),
    parameters: input('float32', 2),
    rasterStack: {
      bands: [
        {
          id: 'band',
          format: 'float32' as const,
          storage: {kind: 'buffer' as const, values: input('float32', 2)},
          validity: input('uint32', 2)
        }
      ],
      grid: {
        width: 2,
        height: 1,
        affine: [1, 0, 0, 0, -1, 0] as [number, number, number, number, number, number],
        pixelInterpretation: 'area' as const
      }
    }
  };
  const create = (offset: number) =>
    new GPUForestInference({
      ...props,
      output: {
        predictions,
        validity: createVectorView(
          'raster-validity',
          'uint32',
          [offset, offset + 4].map(byteOffset =>
            graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset})
          )
        )
      }
    });
  try {
    expect(create(256).getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(() => create(8)).toThrow(/aligned binding.*overlap/);
  } finally {
    device.destroy();
  }
});

it('B3 tiled groups publish adjacent keys and counts in separate dispatches', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'tiled-shared', 'uint32', 64);
  const keys = createTransientView(graph, 'six-keys', 'uint32', 6);
  const output = {
    keys: graph.createDataView(backing.buffer, {format: 'uint32', length: 2}),
    counts: graph.createDataView(backing.buffer, {format: 'uint32', length: 2, byteOffset: 8}),
    status: {
      count: createTransientView(graph, 'tiled-count', 'uint32', 1),
      overflow: createTransientView(graph, 'tiled-overflow', 'uint32', 1)
    }
  };
  try {
    const create = (scratchMode: 'auto' | 'packed') =>
      new GPUGroupStatistics({keys, columns: [], scratchTileSize: 2, scratchMode, output});
    expect(() => create('auto')).not.toThrow();
    const producer = create('auto');
    const ids = producer.getCommandNodes(graph).map(node => node.id);
    expect(ids.some(id => id.includes('union-publish-counts'))).toBe(true);
    expect(ids.some(id => id.endsWith('keys-publish'))).toBe(true);
    expect(() => create('packed')).toThrow(/aligned binding.*overlap/);
    expect(
      () =>
        new GPUGroupStatistics({
          keys,
          columns: [],
          scratchTileSize: 2,
          output: {
            ...output,
            counts: graph.createDataView(backing.buffer, {
              format: 'uint32',
              length: 2,
              byteOffset: 4
            })
          }
        })
    ).toThrow(/logical.*overlap/);
  } finally {
    device.destroy();
  }
});
