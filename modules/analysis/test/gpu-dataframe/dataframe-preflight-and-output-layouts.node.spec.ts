// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUCubeLayoutTranspose} from '../../src/dataframe-analysis/cube-layout/gpu-cube-layout-transpose';
import {GPUValueCube} from '../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {GPUCompositeScore} from '../../src/dataframe-analysis/composite-indicators/gpu-composite-score';
import {GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} from '../../src/dataframe-analysis/composite-indicators/composite-score-parameters';
import {GPUTemporalReduction} from '../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction';
import {GPUTimeWindowFilter} from '../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {GPUColumnProfile} from '../../src/dataframe-analysis/column-profile/gpu-column-profile';
import {GPU_COLUMN_PROFILE_STATISTIC_COUNT} from '../../src/dataframe-analysis/column-profile/column-profile-parameters';
import {GPUVariogram} from '../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {GPUVariogramModelFit} from '../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {GPUSpatialCorrelogram} from '../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {GPUMarkedPointStatistics} from '../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {GPURipley} from '../../src/dataframe-analysis/pair-statistics/gpu-ripley';
import {GPUCalendarBuckets} from '../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {GPUForestInference} from '../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {GPUPeriodIndex} from '../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUInequality} from '../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {GPUKeyLookup} from '../../src/dataframe-analysis/key-join/gpu-key-join';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2' | 'sint32';
type ViewFactory = <ValueFormat extends Format>(
  format: ValueFormat,
  length?: number
) => GraphDataView<ValueFormat>;
type SourceFactory = <ValueFormat extends Format>(
  format: ValueFormat
) => GraphDataView<ValueFormat>;

const cases: [string, (view: ViewFactory, source: SourceFactory) => GPUCommandNodeProducer][] = [
  [
    'cube transpose',
    (view, source) =>
      new GPUCubeLayoutTranspose({
        values: source('float32'),
        output: view('float32'),
        cellCount: 1,
        periodCount: 1,
        from: 'cell-major'
      })
  ],
  [
    'value cube',
    (view, source) =>
      new GPUValueCube({
        cells: view('uint32'),
        periods: view('uint32'),
        values: source('float32'),
        cellCount: 1,
        periodCount: 1,
        output: {counts: view('uint32')}
      })
  ],
  [
    'composite score',
    (view, source) =>
      new GPUCompositeScore({
        indicators: source('float32'),
        indicatorCount: 1,
        parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
        output: {score: view('float32')}
      })
  ],
  [
    'temporal reduction',
    (view, source) =>
      new GPUTemporalReduction({
        cellIds: view('uint32'),
        timestamps: view('float32'),
        values: source('float32'),
        parameters: view('float32', 2),
        cellCount: 1,
        bucketCount: 1,
        output: {
          counts: view('uint32'),
          minimums: view('float32'),
          maximums: view('float32'),
          first: view('float32'),
          last: view('float32'),
          occupiedSlots: {
            ids: view('uint32'),
            status: {count: view('uint32'), overflow: view('uint32')}
          }
        }
      })
  ],
  [
    'time window',
    (view, source) =>
      new GPUTimeWindowFilter({
        timestamps: source('float32'),
        window: view('float32', 32),
        output: {rows: view('uint32'), status: {count: view('uint32'), overflow: view('uint32')}}
      })
  ],
  [
    'variogram',
    (view, source) =>
      new GPUVariogram({
        positions: source('float32x2'),
        values: view('float32'),
        parameters: view('float32', 32),
        gridSize: [1, 1],
        lagCount: 1,
        output: {semivariances: view('float32')}
      })
  ],
  [
    'spatial correlogram',
    (view, source) =>
      new GPUSpatialCorrelogram({
        positions: source('float32x2'),
        values: view('float32'),
        parameters: view('float32', 8),
        gridSize: [1, 1],
        bandCount: 1,
        output: {moransI: view('float32')}
      })
  ],
  [
    'marked point statistics',
    (view, source) =>
      new GPUMarkedPointStatistics({
        positions: source('float32x2'),
        parameters: view('float32', 6),
        gridSize: [1, 1],
        radiusCount: 1,
        output: {statistics: view('float32', 4), pairCounts: view('uint32'), status: view('uint32')}
      })
  ],
  [
    'Ripley',
    (view, source) =>
      new GPURipley({
        positions: source('float32x2'),
        parameters: view('float32', 32),
        gridSize: [1, 1],
        radiusCount: 1,
        output: {k: view('float32')}
      })
  ]
];

for (const [name, build] of cases) {
  it.each([
    'aligned prefix',
    'storage usage'
  ])(`${name} preflights %s before any scratch is declared`, failure => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view: ViewFactory = (format, length = 1) =>
      createTransientView(graph, `field-${serial++}`, format, length);
    const storage = graph.createTransientBuffer({
      id: 'source',
      byteLength: 512,
      usage: failure === 'storage usage' ? Buffer.COPY_SRC : Buffer.STORAGE
    });
    const source: SourceFactory = format =>
      graph.createDataView(storage, {
        format,
        length: 1,
        byteOffset: failure === 'aligned prefix' ? (format === 'float32x2' ? 248 : 252) : 0
      });
    try {
      const producer = build(view, source);
      if (failure === 'aligned prefix')
        Object.assign(device.limits, {maxStorageBufferBindingSize: 128});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => producer.getCommandNodes(graph)).toThrow(
        failure === 'aligned prefix' ? /256 bytes.*limit 128/ : /STORAGE/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
}

it('allows shared output prefixes across separate passes and aligned packed fields within a pass', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length = 1) =>
    createTransientView(graph, `value-${serial++}`, format, length);
  const buffer = graph.createTransientBuffer({
    id: 'packed',
    byteLength: 4096,
    usage: Buffer.STORAGE
  });
  const output = <ValueFormat extends Format>(
    format: ValueFormat,
    byteOffset: number,
    length = 1
  ) => graph.createDataView(buffer, {format, byteOffset, length});
  try {
    const producers: GPUCommandNodeProducer[] = [
      new GPUInequality({
        id: 'inequality',
        values: view('float32', 2),
        zoneIds: view('uint32', 2),
        zoneCount: 1,
        parameters: view('float32', 4),
        output: {gini: output('float32', 0), mean: output('float32', 4)}
      }),
      new GPUValueCube({
        id: 'cube',
        cells: view('uint32'),
        periods: view('uint32'),
        values: view('float32'),
        cellCount: 1,
        periodCount: 1,
        output: {counts: output('uint32', 0), sums: output('float32', 4)}
      }),
      new GPUKeyLookup({
        id: 'join',
        leftKeys: view('uint32'),
        rightKeys: view('uint32'),
        output: {matched: output('uint32', 0)},
        gather: [{column: view('float32'), output: output('float32', 4)}]
      }),
      new GPUCompositeScore({
        id: 'composite',
        indicators: view('float32'),
        indicatorCount: 1,
        parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
        output: {columnStatistics: output('float32', 0, 4), score: output('float32', 256)}
      }),
      new GPUMarkedPointStatistics({
        id: 'marked',
        positions: view('float32x2'),
        parameters: view('float32', 6),
        gridSize: [1, 1],
        radiusCount: 1,
        output: {
          statistics: output('float32', 0, 4),
          pairCounts: output('uint32', 256),
          status: output('uint32', 512)
        }
      }),
      new GPUSpatialCorrelogram({
        id: 'correlogram',
        positions: view('float32x2'),
        values: view('float32'),
        parameters: view('float32', 8),
        gridSize: [1, 1],
        bandCount: 1,
        output: {moransI: output('float32', 0), peakBands: output('uint32', 4, 2)}
      }),
      new GPUVariogramModelFit({
        id: 'fit',
        distances: view('float32', 2),
        semivariances: view('float32', 2),
        pairCounts: view('uint32', 2),
        models: ['exponential'],
        output: {
          parameters: output('float32x4', 0),
          diagnostics: output('float32x4', 256),
          status: output('uint32', 512)
        }
      }),
      new GPUForestInference({
        id: 'forest',
        rowCount: 1,
        model: {
          featureCount: 1,
          treeCount: 1,
          nodeCount: 1,
          outputCount: 1,
          maximumDepth: 0,
          task: 'regression',
          transform: 'identity'
        },
        nodes: view('uint32', 6),
        trees: view('uint32', 4),
        leafValues: view('float32'),
        features: view('float32'),
        parameters: view('float32', 2),
        output: {
          predictions: output('float32', 0),
          leafIndices: output('uint32', 4),
          validity: output('uint32', 256),
          scores: output('float32', 512)
        }
      }),
      new GPUCalendarBuckets({
        id: 'calendar',
        timestamps: view('uint32x2'),
        parameters: view('sint32', 32),
        output: {
          year: output('sint32', 0),
          month: output('uint32', 256),
          dayOfMonth: output('uint32', 512),
          hour: output('uint32', 768),
          minute: output('uint32', 1024),
          weekday: output('uint32', 1280),
          dayOfYear: output('uint32', 4)
        }
      })
    ];
    producers.push(
      new GPUValueCube({
        id: 'cube-means',
        cells: view('uint32'),
        periods: view('uint32'),
        values: view('float32'),
        cellCount: 1,
        periodCount: 1,
        output: {
          counts: output('uint32', 0),
          sums: output('float32', 256),
          means: output('float32', 512)
        }
      }),
      new GPUColumnProfile({
        id: 'profile',
        columns: [{values: view('float32')}],
        hyperLogLogPrecision: 4,
        histogramBinCount: 1,
        output: {
          statistics: output('float32', 0, GPU_COLUMN_PROFILE_STATISTIC_COUNT),
          hyperLogLogRegisters: output('uint32', 256, 16),
          counts: output('uint32', 512, 2)
        }
      }),
      new GPUInequality({
        id: 'inequality-global',
        values: view('float32'),
        zoneIds: view('uint32'),
        zoneCount: 1,
        parameters: view('float32', 4),
        output: {counts: output('uint32', 0), globalSummary: output('float32', 256, 8)}
      }),
      new GPUKeyLookup({
        id: 'join-inner',
        compactMatchedLeftRows: true,
        leftKeys: view('uint32'),
        rightKeys: view('uint32'),
        gather: [{column: view('float32'), output: output('float32', 1280)}],
        output: {
          rightRows: output('uint32', 0),
          matched: output('uint32', 256),
          rows: {
            ids: output('uint32', 512),
            status: {count: output('uint32', 768), overflow: output('uint32', 1024)}
          }
        }
      }),
      new GPUTemporalReduction({
        id: 'temporal-packed',
        cellIds: view('uint32'),
        timestamps: view('float32'),
        values: view('float32'),
        parameters: view('float32', 2),
        cellCount: 1,
        bucketCount: 1,
        output: {
          counts: output('uint32', 0),
          first: output('float32', 256),
          last: output('float32', 512),
          minimums: output('float32', 768),
          maximums: output('float32', 1024),
          occupiedSlots: {
            ids: output('uint32', 1280),
            status: {count: output('uint32', 1536), overflow: output('uint32', 1792)}
          }
        }
      }),
      new GPUTimeWindowFilter({
        id: 'sorted-packed',
        sorted: true,
        timestamps: view('float32'),
        window: view('float32', 32),
        output: {
          rowRange: output('uint32', 0, 2),
          rows: output('uint32', 256),
          status: {count: output('uint32', 512), overflow: output('uint32', 768)}
        }
      })
    );
    for (const producer of producers) {
      const nodes = producer.getCommandNodes(graph);
      for (const node of nodes) {
        const usages = new Map<unknown, Set<string>>();
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !resource.buffer || !resource.usage.startsWith('storage'))
            continue;
          const handle =
            resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
          const usage = usages.get(handle) ?? new Set<string>();
          usage.add(resource.usage);
          usages.set(handle, usage);
        }
        for (const usage of usages.values()) {
          expect(usage.has('storage-read') && usage.has('storage-read-write'), node.id).toBe(false);
        }
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('rejects co-bound aligned aliases and logical overlap across different passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length = 1) =>
    createTransientView(graph, `value-${serial++}`, format, length);
  const buffer = graph.createTransientBuffer({
    id: 'outputs',
    byteLength: 512,
    usage: Buffer.STORAGE
  });
  const first = graph.createDataView(buffer, {format: 'float32', length: 1});
  const second = graph.createDataView(buffer, {format: 'float32', byteOffset: 4, length: 1});
  try {
    const props = {
      values: view('float32'),
      zoneIds: view('uint32'),
      zoneCount: 1,
      parameters: view('float32', 4)
    };
    expect(() => new GPUInequality({...props, output: {gini: first, theilT: second}})).toThrow(
      /aligned binding.*overlap/
    );
    expect(() => new GPUInequality({...props, output: {gini: first, mean: first}})).toThrow(
      /logical.*overlap/
    );
  } finally {
    device.destroy();
  }
});

it('period and calendar output capacity tails are neither written nor bound', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length = 1) =>
    createTransientView(graph, `capacity-${serial++}`, format, length);
  const storage = graph.createTransientBuffer({
    id: 'capacity-output',
    byteLength: 4096,
    usage: Buffer.STORAGE
  });
  try {
    const periods = new GPUPeriodIndex({
      timestamps: view('uint32x2'),
      temporalContext: {
        encoding: 'int64-words',
        unit: 'millisecond',
        epoch: 'unix',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', origin: 0n, width: 1n, count: 2},
      output: {
        periods: graph.createDataView(storage, {format: 'uint32', length: 1024}),
        status: {
          invalidCount: graph.createDataView(storage, {
            format: 'uint32',
            length: 1,
            byteOffset: 256
          })
        }
      }
    });
    const calendar = new GPUCalendarBuckets({
      timestamps: view('uint32x2'),
      parameters: view('sint32', 4),
      output: {year: graph.createDataView(storage, {format: 'sint32', length: 1024})}
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 512});
    expect(() => periods.getCommandNodes(graph)).not.toThrow();
    expect(() => calendar.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('preflights nested IQR selection and sorted temporal headers before allocating scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length = 1) =>
    createTransientView(graph, `nested-${serial++}`, format, length);
  try {
    const variogram = new GPUVariogram({
      positions: view('float32x2'),
      values: view('float32'),
      parameters: view('float32', 32),
      gridSize: [1, 1],
      lagCount: 1,
      output: {semivariances: view('float32')}
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 256});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => variogram.getCommandNodes(graph)).toThrow(/histogram.*4096/);
    expect(allocate).not.toHaveBeenCalled();
    allocate.mockRestore();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    const base = cases.find(([name]) => name === 'temporal reduction')![1](view, format =>
      view(format)
    ) as GPUTemporalReduction;
    const temporal = new GPUTemporalReduction({...base.props, sorted: true});
    Object.assign(device.limits, {maxStorageBufferBindingSize: 8});
    const temporalAllocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => temporal.getCommandNodes(graph)).toThrow(/temporal scratch.*12/);
    expect(temporalAllocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('time-window filtering rejects published primitive read/write aliases but permits separate publication passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = graph.createTransientBuffer({
    id: 'time-packed',
    byteLength: 2048,
    usage: Buffer.STORAGE
  });
  const field = (byteOffset: number, length = 1) =>
    graph.createDataView(buffer, {format: 'uint32', byteOffset, length});
  const timestamps = createTransientView(graph, 'times', 'float32', 2);
  const window = createTransientView(graph, 'window', 'float32', 32);
  try {
    const props = {
      timestamps,
      window,
      output: {
        outputMask: field(0, 2),
        rows: field(256, 2),
        status: {count: field(512), overflow: field(768)}
      }
    };
    expect(() => new GPUTimeWindowFilter(props)).toThrow(
      /outputs must not share buffers with inputs/
    );
    expect(() =>
      new GPUTimeWindowFilter({
        ...props,
        output: {...props.output, rows: field(256)}
      }).getCommandNodes(graph)
    ).not.toThrow();
    const trackIds = createTransientView(graph, 'tracks', 'uint32', 2);
    expect(
      () =>
        new GPUTimeWindowFilter({
          ...props,
          trackIds,
          output: {...props.output, rows: field(256), trackVisibleCounts: field(1024)}
        })
    ).toThrow(/outputs must not share buffers with inputs/);
    expect(
      () =>
        new GPUTimeWindowFilter({
          ...props,
          timestamps: graph.createDataView(buffer, {
            format: 'float32',
            length: 2,
            byteOffset: 1280
          }),
          output: {...props.output, outputMask: undefined, rows: field(256)}
        })
    ).toThrow(/outputs must not share buffers with inputs/);
  } finally {
    device.destroy();
  }
});
