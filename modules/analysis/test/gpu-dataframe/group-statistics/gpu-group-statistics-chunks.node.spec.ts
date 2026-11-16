// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  GraphVectorView,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it, vi} from 'vitest';
import {
  GPUGroupStatistics,
  type GPUGroupStatistic,
  type GPUGroupStatisticsProps,
  type GPUGroupStatisticsOutput,
  type GPUGroupStatisticsColumnOutput
} from '../../../src/dataframe-analysis/group-statistics/index';
import {
  createNullWebGPUDevice,
  createOutputBuffer,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from './wgsl-reserved-words';
import {
  findConstantNonFiniteBitcasts,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GROUP_STATISTICS_TEST_STORAGE_BINDING_BYTE_LENGTH,
  withGroupStatisticsTestBindingLimit
} from './group-statistics-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

type GroupStatisticsTestProps = GPUGroupStatisticsProps<
  GraphDataView<'uint32' | 'uint32x2'> | GraphVectorView<'uint32' | 'uint32x2'>,
  GraphDataView<'float32'> | GraphVectorView<'float32'>
>;

function createVector<Format extends 'uint32' | 'uint32x2' | 'float32'>(
  graph: GPUCommandGraph,
  id: string,
  format: Format,
  lengths: number[]
): GraphVectorView<Format> {
  const length = lengths.reduce((total, count) => total + count, 0);
  const stride = format === 'uint32x2' ? 2 : 1;
  return new GraphVectorView({
    id,
    name: id,
    format,
    length,
    valueLength: length * stride,
    stride,
    byteStride: stride * 4,
    rowByteLength: stride * 4,
    data: lengths.map((count, index) => createTransientView(graph, `${id}-${index}`, format, count))
  });
}

function createProps(
  graph: GPUCommandGraph,
  statistics: GPUGroupStatistic[] = ['mean']
): GroupStatisticsTestProps {
  return {
    keys: createVector(graph, 'keys', 'uint32x2', [0, 257, 0, 774]),
    mask: createVector(graph, 'mask', 'uint32', [0, 257, 0, 774]),
    columns: [
      {
        values: createVector(graph, 'values', 'float32', [1, 0, 18, 1012, 0]),
        statistics,
        output: {
          ...(statistics.includes('mean')
            ? {means: createTransientView(graph, 'means', 'float32', 8)}
            : {}),
          ...(statistics.includes('variance')
            ? {variances: createTransientView(graph, 'variances', 'float32', 8)}
            : {}),
          ...(statistics.includes('median')
            ? {medians: createTransientView(graph, 'medians', 'float32', 8)}
            : {}),
          ...(statistics.includes('zScore')
            ? {zScores: createTransientView(graph, 'z-scores', 'float32', 1031)}
            : {})
        }
      }
    ],
    output: {
      keys: createTransientView(graph, 'output-keys', 'uint32x2', 8),
      counts: createTransientView(graph, 'output-counts', 'uint32', 8),
      status: {
        count: createTransientView(graph, 'output-count', 'uint32', 1),
        overflow: createTransientView(graph, 'output-overflow', 'uint32', 1)
      }
    }
  };
}

it('dispatches nonempty key and value chunks separately into shared accumulators', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUGroupStatistics(createProps(graph, ['mean', 'variance', 'zScore']));
    const nodes = contributor.getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(new Set(ids).size).toBe(ids.length);
    for (const [variant, counts] of [
      ['prepare', [257, 774]],
      ['sum-scale', [1, 18, 1012]],
      ['accumulate', [1, 18, 1012]],
      ['moment-terms', [1, 18, 1012]],
      ['z-scores', [1, 18, 1012]]
    ] as const) {
      expect(
        nodes
          .filter(node => node.id.includes(`-${variant}-`))
          .map(node => node.workload?.maximumInvocationCount)
      ).toEqual(counts);
    }
    expect(ids.some(id => id.includes('gather-value-bits'))).toBe(false);
    const scales = ids.flatMap((id, index) => (id.includes('-sum-scale-') ? [index] : []));
    const accumulations = ids.flatMap((id, index) => (id.includes('-accumulate-') ? [index] : []));
    expect(Math.max(...scales)).toBeLessThan(Math.min(...accumulations));
    expect(ids.filter(id => id.includes('-moment-merge-'))).toHaveLength(
      Math.ceil(Math.log2(1031))
    );
    // Parse the generated WGSL without a GPU and verify the core eight-binding limit.
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(0);
    for (const source of sources.values()) {
      expect(findConstantNonFiniteBitcasts(stripWGSLComments(source))).toEqual([]);
      expect((source.match(/var<storage/g) ?? []).length).toBeLessThanOrEqual(8);
      const tokens =
        source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(tokens.filter(token => WGSL_RESERVED_WORDS.has(token))).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('accepts end-of-buffer empty key, mask and independent value chunks without binding a backing row', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const scratchTileSize of [undefined, 256]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, ['mean', 'variance', 'median', 'zScore']);
      const createSubvector = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
        name: string,
        format: Format,
        lengths: number[]
      ) => {
        const backing = createTransientView(graph, name, format, 1031);
        let rowOffset = 0;
        const chunks = lengths.map(length => {
          const chunk = graph.createDataView(backing.buffer, {
            format,
            length,
            byteOffset: rowOffset * backing.rowByteLength
          });
          rowOffset += length;
          return chunk;
        });
        expect(rowOffset).toBe(1031);
        const stride = backing.rowByteLength / 4;
        return new GraphVectorView({
          id: name,
          name,
          format,
          length: 1031,
          valueLength: 1031 * stride,
          stride,
          byteStride: stride * 4,
          rowByteLength: stride * 4,
          data: chunks
        });
      };
      const lengths = [0, 1, 256, 0, 774, 0];
      const keys = createSubvector('shared-keys', 'uint32x2', lengths);
      const mask = createSubvector('shared-mask', 'uint32', lengths);
      const values = createSubvector('shared-values', 'float32', [2, 0, 17, 1012, 0]);
      const last = keys.data.at(-1)!;
      const range = getViewBindingRange(last);
      expect(last.byteOffset).toBe(8248);
      expect(range.offset + range.size).toBe(8256);
      expect(last.buffer.byteLength).toBe(8248);
      const contributor = new GPUGroupStatistics({
        ...props,
        keys,
        mask,
        scratchTileSize,
        // The tiled case uses only scalable reductions; the packed case exercises gathers too.
        columns: [
          {
            ...props.columns[0],
            values,
            statistics: scratchTileSize ? ['mean', 'variance'] : props.columns[0].statistics,
            output: scratchTileSize
              ? {means: props.columns[0].output.means, variances: props.columns[0].output.variances}
              : props.columns[0].output
          }
        ]
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(contributor.usesTiledScratch).toBe(scratchTileSize !== undefined);
      for (const node of nodes) {
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
            const binding = getViewBindingRange(resource.buffer);
            expect(binding.offset + binding.size, node.id).toBeLessThanOrEqual(
              resource.buffer.buffer.byteLength
            );
          }
        }
      }
      if (!scratchTileSize) {
        expect(
          nodes
            .filter(node => /-prepare-/.test(node.id))
            .map(node => node.workload?.maximumInvocationCount)
        ).toEqual([1, 256, 774]);
        expect(
          nodes
            .filter(node => /-gather-value-bits-/.test(node.id))
            .map(node => node.workload?.maximumInvocationCount)
        ).toEqual([2, 17, 1012]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('only order statistics gather value chunks into global packed scratch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUGroupStatistics(createProps(graph, ['median']));
    const nodes = contributor.getCommandNodes(graph);
    const gathers = nodes.filter(node => node.id.includes('-gather-value-bits-'));
    expect(gathers.map(node => node.workload?.maximumInvocationCount)).toEqual([1, 18, 1012]);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => node.id.includes('-moment-terms'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('validates every chunk, key-mask topology, graph ownership and input-output aliases', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    expect(
      () =>
        new GPUGroupStatistics({
          ...props,
          mask: createVector(graph, 'wrong-mask', 'uint32', [1, 1030])
        })
    ).toThrow(/group-statistics mask must match the source topology/);
    const values = props.columns[0].values as GraphVectorView<'float32'>;
    const badChunk = graph.createDataView(values.data[3].buffer, {
      format: 'float32',
      length: 506,
      byteStride: 8
    });
    const badValues = new GraphVectorView({
      ...values,
      data: [...values.data.slice(0, 3), badChunk],
      length: 525
    });
    expect(
      () =>
        new GPUGroupStatistics({
          ...props,
          columns: [{...props.columns[0], values: badValues}]
        })
    ).toThrow(/group-statistics columns\[0\].values must be packed/);
    expect(
      () =>
        new GPUGroupStatistics({
          ...props,
          output: {
            ...props.output,
            status: {
              ...props.output.status,
              count: values.data[3] as unknown as GraphDataView<'uint32'>
            }
          }
        })
    ).toThrow(/output.status.count must be packed/);
    const keyChunks = (props.keys as GraphVectorView<'uint32x2'>).data;
    const aliased = graph.createDataView(keyChunks[3].buffer, {format: 'uint32', length: 1});
    expect(
      () =>
        new GPUGroupStatistics({
          ...props,
          output: {...props.output, status: {...props.output.status, count: aliased}}
        })
    ).toThrow(/group-statistics outputs must not share buffers with inputs/);
    const otherGraph = new GPUCommandGraph(device);
    const otherValues = createVector(otherGraph, 'foreign-values', 'float32', [1031]);
    const foreign = new GPUGroupStatistics({
      ...props,
      columns: [{...props.columns[0], values: otherValues}]
    });
    expect(() => foreign.getCommandNodes(graph)).toThrow(
      /group-statistics views must belong to the target graph/
    );
  } finally {
    device.destroy();
  }
});

it('checks global row scratch and individual 64-bit key bindings before allocating scratch', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 8192});
  try {
    // 1031 64-bit keys exceed this binding limit in total, but the two chunks fit, as does
    // each scalar global sort / moment-tree scratch binding. No packed-key gather is needed.
    const graph = new GPUCommandGraph(device);
    expect(() => new GPUGroupStatistics(createProps(graph)).getCommandNodes(graph)).not.toThrow();
    for (const [lengths, message] of [
      [[1031], /input\/output chunk needs 8248 bytes/],
      [[1024, 1025], /keys.length row scratch needs 8196 bytes/]
    ] as const) {
      const rejectedGraph = new GPUCommandGraph(device);
      const props = createProps(rejectedGraph);
      const contributor = new GPUGroupStatistics({
        ...props,
        scratchMode: 'packed',
        keys: createVector(rejectedGraph, 'large-keys', 'uint32x2', [...lengths]),
        mask: undefined,
        columns: []
      });
      const allocate = vi.spyOn(rejectedGraph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(rejectedGraph)).toThrow(message);
      expect(allocate).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('budgets the additional nodes when input chunk counts grow', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const contributor = new GPUGroupStatistics({
      ...props,
      keys: createVector(graph, 'tiny-keys', 'uint32x2', new Array(1031).fill(1)),
      mask: undefined,
      columns: [
        {
          ...props.columns[0],
          values: createVector(graph, 'tiny-values', 'float32', new Array(1031).fill(1))
        }
      ]
    });
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
    const rejectedGraph = new GPUCommandGraph(device);
    const rejectedProps = createProps(rejectedGraph);
    expect(
      () =>
        new GPUGroupStatistics({
          ...rejectedProps,
          keys: createVector(rejectedGraph, 'excessive-keys', 'uint32x2', new Array(32768).fill(1)),
          columns: [],
          mask: undefined
        })
    ).toThrow(/group-statistics.*command.*node/i);
  } finally {
    device.destroy();
  }
});

it('keeps shader and pipeline counts constant as chunk count grows, including unequal chunks and tree levels', () => {
  const device = createNullWebGPUDevice();
  try {
    const rowCount = 4096;
    let expectedPipelineCount: number | undefined;
    let expectedShaderCount: number | undefined;
    for (const chunkCount of [2, 16, 64]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, ['mean', 'variance', 'zScore', 'median']);
      const lengths = new Array(chunkCount).fill(rowCount / chunkCount);
      // Exercise nonzero aligned-binding prefixes and different dispatch sizes as well.
      lengths[0]--;
      lengths[lengths.length - 1]++;
      const backing = createTransientView(graph, 'value-backing', 'float32', rowCount);
      let rowOffset = 0;
      const data = lengths.map(length => {
        const view = graph.createDataView(backing.buffer, {
          format: 'float32',
          length,
          byteOffset: rowOffset * 4
        });
        rowOffset += length;
        return view;
      });
      const values = new GraphVectorView({
        id: 'chunk-values',
        name: 'chunk-values',
        format: 'float32',
        length: rowCount,
        valueLength: rowCount,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data
      });
      const contributor = new GPUGroupStatistics({
        ...props,
        mask: undefined,
        keys: createVector(graph, 'pipeline-keys', 'uint32x2', lengths),
        columns: [
          {
            ...props.columns[0],
            values,
            output: {
              ...props.columns[0].output,
              zScores: createTransientView(graph, 'pipeline-z-scores', 'float32', rowCount)
            }
          }
        ]
      });
      const nodes = contributor.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const shaders = new Set(kernels.map(kernel => kernel.source));
      const pipelines = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
      expectedShaderCount ??= shaders.size;
      expectedPipelineCount ??= pipelines.size;
      expect(shaders.size).toBe(expectedShaderCount);
      expect(pipelines.size).toBe(expectedPipelineCount);
      const stages = kernels.filter(kernel =>
        /-prepare-|-(sum-scale|accumulate|moment-terms|z-scores|gather-value-bits|moment-merge)-/.test(
          kernel.id
        )
      );
      expect(new Set(stages.map(kernel => kernel.source)).size).toBe(7);
      expect(stages.every(kernel => Object.keys(kernel.constants).length === 0)).toBe(true);
      for (const kernel of kernels) {
        const tokens =
          kernel.source
            .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ')
            .match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(tokens.filter(token => WGSL_RESERVED_WORDS.has(token))).toEqual([]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('rejects overlapping output byte ranges across status, columns and table fields, but allows adjacent views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'output-backing', 'uint32', 24);
    const counts = graph.createDataView(backing.buffer, {format: 'uint32', length: 8});
    const means = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 32
    });
    const adjacent = {
      ...props,
      output: {...props.output, counts},
      columns: [{...props.columns[0], output: {means}}]
    };
    expect(() => new GPUGroupStatistics(adjacent)).not.toThrow();
    for (const overlapping of [
      {
        ...adjacent,
        columns: [
          {
            ...props.columns[0],
            output: {
              means: graph.createDataView(backing.buffer, {
                format: 'float32',
                length: 8,
                byteOffset: 28
              })
            }
          }
        ]
      },
      {
        ...props,
        output: {
          ...props.output,
          status: {...props.output.status, overflow: props.output.status.count}
        }
      },
      {...props, columns: [props.columns[0], {...props.columns[0]}]}
    ])
      expect(() => new GPUGroupStatistics(overlapping)).toThrow(
        /group-statistics .* (logical|aligned binding) byte ranges must not overlap/
      );
  } finally {
    device.destroy();
  }
});

it('checks aligned view prefixes and derived group scratch before declaring any resources or nodes', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 8192});
  try {
    for (const failure of ['aligned-input', 'fixed-point-scratch', 'moment-scratch'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, failure === 'moment-scratch' ? ['variance'] : ['mean']);
      let nextProps = props;
      if (failure === 'aligned-input') {
        const backing = createTransientView(graph, 'unaligned-input', 'float32', 2049);
        nextProps = {
          ...props,
          keys: createVector(graph, 'aligned-keys', 'uint32x2', [1024, 1024]),
          mask: undefined,
          columns: [
            {
              ...props.columns[0],
              values: graph.createDataView(backing.buffer, {
                format: 'float32',
                length: 2048,
                byteOffset: 4
              })
            }
          ]
        };
      } else {
        const capacity = failure === 'fixed-point-scratch' ? 700 : 600;
        nextProps = {
          ...props,
          columns: [
            {
              ...props.columns[0],
              output:
                failure === 'moment-scratch'
                  ? {variances: createTransientView(graph, 'large-variances', 'float32', capacity)}
                  : {means: createTransientView(graph, 'large-means', 'float32', capacity)}
            }
          ],
          output: {
            ...props.output,
            keys: createTransientView(graph, 'large-out-keys', 'uint32x2', capacity),
            counts: createTransientView(graph, 'large-out-counts', 'uint32', capacity)
          }
        };
      }
      const contributor = new GPUGroupStatistics(nextProps);
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/group-statistics/);
      expect(allocate).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('checks all generated statistic branches against the full W3C reserved-word list', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const capacity of [256, 257]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      const statistics: GPUGroupStatistic[] = [
        'count',
        'sum',
        'mean',
        'minimum',
        'maximum',
        'variance',
        'standardDeviation',
        'skewness',
        'kurtosis',
        'median',
        'percentiles',
        'mode',
        'uniqueCount',
        'zScore'
      ];
      const floatFields = [
        'sumValues',
        'means',
        'minimums',
        'maximums',
        'variances',
        'standardDeviations',
        'skewness',
        'kurtosis',
        'medians',
        'percentiles',
        'modes',
        'zScores'
      ] as const;
      const floatOutputs = Object.fromEntries(
        floatFields.map(field => [
          field,
          createTransientView(
            graph,
            `all-${field}`,
            'float32',
            field === 'zScores' ? 1031 : field === 'percentiles' ? 2 * capacity : capacity
          )
        ])
      );
      const contributor = new GPUGroupStatistics({
        ...props,
        percentiles: createTransientView(graph, 'all-fractions', 'float32', 2),
        columns: [
          {
            ...props.columns[0],
            statistics,
            output: {
              ...floatOutputs,
              counts: createTransientView(graph, 'all-finite-counts', 'uint32', capacity),
              sums: createTransientView(graph, 'all-sums', 'uint32x2', capacity),
              uniqueCounts: createTransientView(graph, 'all-unique-counts', 'uint32', capacity)
            }
          }
        ],
        output: {
          ...props.output,
          keys: createTransientView(graph, 'all-output-keys', 'uint32x2', capacity),
          counts: createTransientView(graph, 'all-output-counts', 'uint32', capacity)
        }
      });
      const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(kernels.size).toBeGreaterThan(0);
      for (const source of kernels.values()) {
        expect(findConstantNonFiniteBitcasts(stripWGSLComments(source))).toEqual([]);
        const tokens =
          source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ').match(/[A-Za-z_][A-Za-z_0-9]*/g) ??
          [];
        expect(tokens.filter(token => WGSL_RESERVED_WORDS.has(token))).toEqual([]);
        expect((source.match(/var<storage/g) ?? []).length).toBeLessThanOrEqual(8);
      }
    }
  } finally {
    device.destroy();
  }
});

it('rejects adjacent packed outputs whose aligned writable bindings overlap in one dispatch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'finish-sums-backing', 'uint32', 65);
    const counts = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
    const createNextProps = (byteOffset: number): GroupStatisticsTestProps => ({
      ...props,
      columns: [
        {
          ...props.columns[0],
          statistics: ['count', 'mean'],
          output: {
            counts,
            means: graph.createDataView(backing.buffer, {format: 'float32', length: 1, byteOffset})
          }
        }
      ],
      output: {
        ...props.output,
        keys: createTransientView(graph, `one-key-${byteOffset}`, 'uint32x2', 1),
        counts: createTransientView(graph, `one-count-${byteOffset}`, 'uint32', 1)
      }
    });
    expect(() => new GPUGroupStatistics(createNextProps(4))).toThrow(
      /group-statistics columns\[0\].output.counts chunk 0 and columns\[0\].output.means chunk 0 aligned binding byte ranges must not overlap/
    );
    expect(() => new GPUGroupStatistics(createNextProps(256))).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('uses each chunk own workgroup count for strongly unequal inputs without specializing its pipeline', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, ['mean', 'variance', 'zScore', 'median']);
    const contributor = new GPUGroupStatistics({
      ...props,
      keys: createVector(graph, 'unequal-keys', 'uint32x2', [1, 257, 773]),
      mask: undefined,
      columns: [
        {
          ...props.columns[0],
          values: createVector(graph, 'unequal-values', 'float32', [1, 257, 773])
        }
      ]
    });
    const nodes = contributor.getCommandNodes(graph);
    for (const stage of [
      'prepare',
      'sum-scale',
      'accumulate',
      'moment-terms',
      'z-scores',
      'gather-value-bits'
    ]) {
      const dispatches = nodes.filter(node => node.id.includes(`-${stage}-`));
      expect(dispatches.map(node => node.workload?.maximumInvocationCount)).toEqual([1, 257, 773]);
      expect(dispatches.map(node => node.workload?.maximumWorkgroupCount)).toEqual([1, 2, 4]);
      expect(getGPUCommandNodePipelineKeys(dispatches, {includeShaderLayout: true}).size).toBe(1);
    }
    expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
  } finally {
    device.destroy();
  }
});

it('recognizes physical buffer aliases across separate dispatches while permitting adjacent logical ranges', () => {
  const device = createNullWebGPUDevice();
  const buffer = createOutputBuffer(device, 64);
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const counts = importGraphBuffer(graph, 'aliased-table-counts', buffer, 'uint32', 8);
    const meansHandle = importGraphBuffer(graph, 'aliased-column-means', buffer, 'float32', 8);
    const createAliasedProps = (byteOffset: number): GroupStatisticsTestProps => ({
      ...props,
      output: {...props.output, counts},
      columns: [
        {
          ...props.columns[0],
          output: {
            means: graph.createDataView(meansHandle.buffer, {
              format: 'float32',
              length: 8,
              byteOffset
            })
          }
        }
      ]
    });
    expect(() => new GPUGroupStatistics(createAliasedProps(28))).toThrow(
      /logical byte ranges must not overlap/
    );
    expect(() => new GPUGroupStatistics(createAliasedProps(32))).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('keeps public group outputs packed for packed and chunked input callers', () => {
  expectTypeOf<GPUGroupStatisticsOutput['counts']>().toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf<GPUGroupStatisticsOutput['keys']>().toEqualTypeOf<
    GraphDataView<'uint32' | 'uint32x2'>
  >();
  expectTypeOf<GPUGroupStatisticsColumnOutput['means']>().toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf<GPUGroupStatisticsColumnOutput['zScores']>().toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const chunked = createProps(graph);
    const packed = {
      ...chunked,
      mask: undefined,
      keys: createTransientView(graph, 'typed-packed-keys', 'uint32x2', 1031),
      columns: [
        {
          ...chunked.columns[0],
          values: createTransientView(graph, 'typed-packed-values', 'float32', 1031)
        }
      ]
    };
    for (const props of [packed, chunked]) {
      const contributor = new GPUGroupStatistics(props);
      expectTypeOf(contributor.props.output.counts).toEqualTypeOf<GraphDataView<'uint32'>>();
      expect(contributor.props.output.counts.buffer).toBe(props.output.counts.buffer);
    }
  } finally {
    device.destroy();
  }
});

it('keeps real buffer and graph device ownership under the synthetic binding limit and restores limits', async () => {
  const device = createNullWebGPUDevice();
  const originalLimits = device.limits;
  const originalDescriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  const originalMaximum = device.limits.maxStorageBufferBindingSize;
  const buffer = createOutputBuffer(device, 1);
  try {
    // This is the real public import path that failed on WebGPU, not just node generation.
    const length = await withGroupStatisticsTestBindingLimit(device, async limitedDevice => {
      expect(limitedDevice).toBe(device);
      expect(buffer.device).toBe(limitedDevice);
      expect(limitedDevice.limits.maxStorageBufferBindingSize).toBe(
        GROUP_STATISTICS_TEST_STORAGE_BINDING_BYTE_LENGTH
      );
      const graph = new GPUCommandGraph(limitedDevice);
      expect(graph.device).toBe(buffer.device);
      const counts = importGraphBuffer(graph, 'c0-counts', buffer, 'uint32', 1);
      const props = createProps(graph);
      const contributor = new GPUGroupStatistics({
        ...props,
        columns: [
          {
            ...props.columns[0],
            statistics: ['count'],
            output: {
              counts: graph.createDataView(counts.buffer, {format: 'uint32', length: 1})
            }
          }
        ],
        output: {
          ...props.output,
          keys: createTransientView(graph, 'ownership-output-key', 'uint32x2', 1),
          counts: createTransientView(graph, 'ownership-output-count', 'uint32', 1)
        }
      });
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      await Promise.resolve();
      expect(limitedDevice.limits.maxStorageBufferBindingSize).toBe(
        GROUP_STATISTICS_TEST_STORAGE_BINDING_BYTE_LENGTH
      );
      return counts.length;
    });
    expect(length).toBe(1);
    expect(device.limits).toBe(originalLimits);
    expect(device.limits.maxStorageBufferBindingSize).toBe(originalMaximum);
    expect(Object.getOwnPropertyDescriptor(device, 'limits')).toEqual(originalDescriptor);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('restores the exact device limits after synchronous and asynchronous binding-fixture failures', async () => {
  const device = createNullWebGPUDevice();
  const originalLimits = device.limits;
  const originalDescriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  try {
    for (const fail of [
      () => {
        throw new Error('synthetic synchronous fixture failure');
      },
      async () => {
        await Promise.resolve();
        throw new Error('synthetic asynchronous fixture failure');
      }
    ]) {
      await expect(withGroupStatisticsTestBindingLimit(device, fail)).rejects.toThrow(
        /synthetic .* fixture failure/
      );
      expect(device.limits).toBe(originalLimits);
      expect(Object.getOwnPropertyDescriptor(device, 'limits')).toEqual(originalDescriptor);
    }
  } finally {
    device.destroy();
  }
});

it('accepts union-format key views unchanged and preserves precise packed props through inference', () => {
  expectTypeOf<GPUGroupStatisticsProps['keys']>().toEqualTypeOf<
    GraphDataView<'uint32' | 'uint32x2'>
  >();
  expectTypeOf<GPUGroupStatisticsProps['mask']>().toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf<GPUGroupStatisticsProps['columns'][number]['values']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  const device = createNullWebGPUDevice();
  try {
    for (const format of ['uint32', 'uint32x2'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      // This is the same view-of-a-format-union shape returned by the existing parity harness.
      const keys: GraphDataView<'uint32' | 'uint32x2'> = createTransientView(
        graph,
        'union-input',
        format,
        1031
      );
      const outputKeys: GraphDataView<'uint32' | 'uint32x2'> = createTransientView(
        graph,
        'union-output',
        format,
        8
      );
      const contributor = new GPUGroupStatistics({
        ...props,
        keys,
        mask: undefined,
        output: {...props.output, keys: outputKeys}
      });
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      expectTypeOf(contributor.props.keys).toEqualTypeOf<GraphDataView<'uint32' | 'uint32x2'>>();
      expectTypeOf(contributor.props.output.keys).toEqualTypeOf<
        GraphDataView<'uint32' | 'uint32x2'>
      >();
      const packedKeys = createTransientView(graph, 'precise-input', 'uint32', 1031);
      const packedOutputKeys = createTransientView(graph, 'precise-output', 'uint32', 8);
      const packedValues = createTransientView(graph, 'precise-values', 'float32', 1031);
      const packed = new GPUGroupStatistics({
        ...props,
        keys: packedKeys,
        mask: undefined,
        columns: [{...props.columns[0], values: packedValues}],
        output: {...props.output, keys: packedOutputKeys}
      });
      expectTypeOf(packed.props.keys).toEqualTypeOf<GraphDataView<'uint32'>>();
      expectTypeOf(packed.props.output.keys).toEqualTypeOf<GraphDataView<'uint32'>>();
      expectTypeOf<(typeof packed.props.columns)[number]['values']>().toEqualTypeOf<
        GraphDataView<'float32'>
      >();
      expect(packed.props.output.keys.buffer).toBe(packedOutputKeys.buffer);
    }
  } finally {
    device.destroy();
  }
});

it('rejects non-finite and out-of-range numeric shader props before generating any WGSL', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const sumScale of [NaN, Infinity, -Infinity, 1e300, 1e-300, -1, 0, 0.1]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      expect(() => new GPUGroupStatistics({...props, sumScale})).toThrow(
        /group-statistics sumScale must be a positive finite float32 value/
      );
    }
    for (const keyCount of [NaN, Infinity, -Infinity, 2 ** 32, 1.5]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      expect(() => new GPUGroupStatistics({...props, keyCount})).toThrow(
        /group-statistics keyCount must be an integer/
      );
    }
    for (const capacity of [NaN, Infinity, -Infinity, 2 ** 32, 2 ** 30, 1.5, -1, 0]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      Object.defineProperty(props.output.keys, 'length', {value: capacity});
      expect(() => new GPUGroupStatistics(props)).toThrow(
        /group-statistics output.keys.length must be an integer in \[1, 1073741823\]/
      );
    }
    for (const length of [NaN, Infinity, -Infinity, 2 ** 32, 1.5, -1, 0]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      const percentiles = createTransientView(graph, 'bad-fractions', 'float32', 1);
      Object.defineProperty(percentiles, 'length', {value: length});
      expect(() => new GPUGroupStatistics({...props, columns: [], percentiles})).toThrow(
        /group-statistics percentiles.length must be an integer in \[1, 4294967295\]/
      );
    }
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const percentiles = createTransientView(graph, 'excessive-total-fractions', 'float32', 1);
    const percentileOutput = createTransientView(graph, 'excessive-total-output', 'float32', 1);
    // Metadata-only oversized views exercise construction without allocating gigabytes.
    Object.defineProperty(percentiles, 'length', {value: 2 ** 29});
    Object.defineProperty(percentileOutput, 'length', {value: 2 ** 32});
    expect(
      () =>
        new GPUGroupStatistics({
          ...props,
          percentiles,
          columns: [
            {
              values: props.columns[0].values,
              statistics: ['percentiles'],
              output: {percentiles: percentileOutput}
            }
          ]
        })
    ).toThrow(
      /group-statistics output.keys.length \* percentiles.length must be an integer in \[1, 4294967295\]/
    );
  } finally {
    device.destroy();
  }
});

it('emits finite typed numeric constants even at the accepted float32 sum-scale extremes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const sumScale of [2 ** -149, 2 ** -126, 65536, Math.fround(3.4028234663852886e38)]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      const kernels = getGPUCommandNodeKernelSources(
        new GPUGroupStatistics({...props, sumScale}).getCommandNodes(graph)
      );
      let numericConstantCount = 0;
      for (const kernel of kernels) {
        for (const match of kernel.source.matchAll(
          /const\s+\w+\s*:\s*(u32|i32|f32)\s*=\s*([^;]+);/g
        )) {
          const type = match[1];
          const expression = match[2].trim();
          expect(/^(?:nan|[+-]?inf(?:inity)?)$/i.test(expression)).toBe(false);
          if (
            !/^[+-]?(?:0x[0-9a-f]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)[uif]?$/i.test(expression)
          )
            continue;
          const value = Number(expression.replace(/[uif]$/, ''));
          expect(Number.isFinite(value)).toBe(true);
          if (type === 'f32') {
            expect(Number.isFinite(Math.fround(value))).toBe(true);
          } else {
            expect(Number.isInteger(value)).toBe(true);
            expect(value).toBeGreaterThanOrEqual(type === 'u32' ? 0 : -(2 ** 31));
            expect(value).toBeLessThanOrEqual(type === 'u32' ? 2 ** 32 - 1 : 2 ** 31 - 1);
          }
          numericConstantCount++;
        }
      }
      expect(numericConstantCount).toBeGreaterThan(0);
    }
  } finally {
    device.destroy();
  }
});
