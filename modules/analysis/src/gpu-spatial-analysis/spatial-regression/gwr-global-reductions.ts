// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** Scalar result slots start on WebGPU storage binding boundaries, sharing reducer pipelines. */
export const GWR_REDUCTION_SLOT_WORDS = 64;

/** Allocation sizes used in contributor preflight before creating reduction nodes. @internal */
export function getGWRReductionScratchSizes(
  tileCount: number,
  channelCount: number
): readonly (readonly [string, number])[] {
  return [
    [
      'global reduction channels and shrinking published reduction scratch',
      Math.ceil(tileCount / GWR_REDUCTION_SLOT_WORDS) * GWR_REDUCTION_SLOT_WORDS * channelCount * 4
    ],
    ['global reduction scalar slots', channelCount * GWR_REDUCTION_SLOT_WORDS * 4]
  ];
}

/**
 * Fixed-tree published sums, one aligned channel per candidate/statistic. No float atomics or
 * data-sized invocation loops. Repeated candidate passes reuse the same channel/result storage.
 * @internal
 */
export function createGWRCandidateReduction<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tileCount: number;
    candidateWidth: number;
    source: GraphDataView<'float32'>;
  }
): {
  totals: GraphDataView<'float32'>;
  createNodes: (suffix: string) => GPUCommandNode<Parameters>[];
} {
  const {id, tileCount, candidateWidth, source} = props;
  const channelCount = candidateWidth * 5;
  const channelStride = Math.ceil(tileCount / GWR_REDUCTION_SLOT_WORDS) * GWR_REDUCTION_SLOT_WORDS;
  const channels = createTransientView(
    graph,
    `${id}-reduction-channels`,
    'float32',
    channelStride * channelCount
  );
  const totals = createTransientView(
    graph,
    `${id}-reduction-totals`,
    'float32',
    channelCount * GWR_REDUCTION_SLOT_WORDS
  );
  const reductions = Array.from({length: channelCount}, (_, channel) => ({
    input: graph.createDataView(channels.buffer, {
      format: 'float32',
      length: tileCount,
      byteOffset: channel * channelStride * 4
    }),
    output: graph.createDataView(totals.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: channel * GWR_REDUCTION_SLOT_WORDS * 4
    })
  }));
  return {
    totals,
    createNodes: suffix => [
      createWGSLKernelNode(graph, {
        id: `${id}${suffix}-reduction-gather`,
        operation: 'GPUGeographicallyWeightedRegression',
        variant: 'reduction-gather',
        invocationCount: tileCount * channelCount,
        bindings: [
          {name: 'sourceValues', view: source, type: 'f32', access: 'read'},
          {name: 'channels', view: channels, type: 'f32', access: 'read_write'}
        ],
        declarations: `const TILE_COUNT: u32 = ${tileCount}u;\nconst CHANNEL_COUNT: u32 = ${channelCount}u;\nconst CHANNEL_STRIDE: u32 = ${channelStride}u;`,
        body: `let tile = index / CHANNEL_COUNT;
  let channel = index % CHANNEL_COUNT;
  channels[channelsOffset + channel * CHANNEL_STRIDE + tile] = sourceValues[sourceValuesOffset + tile * CHANNEL_COUNT + channel];`
      }),
      ...reductions.flatMap((views, channel) =>
        new GPUReduction({
          id: `${id}${suffix}-sum-${channel}`,
          ...views,
          operation: 'sum'
        }).getCommandNodes(graph)
      )
    ]
  };
}

/** Parallel, centred two-pass total variance from per-tile Chan statistics. @internal */
export function createGWRVarianceReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tileCount: number;
    statistics: GraphDataView<'float32'>;
    summary: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, tileCount, statistics, summary} = props;
  const channels = [0, 1, 2].map(channel =>
    createTransientView(graph, `${id}-variance-channel-${channel}`, 'float32', tileCount)
  );
  const totals = [0, 1, 2].map(channel =>
    createTransientView(graph, `${id}-variance-total-${channel}`, 'float32', 1)
  );
  const anchorRows = createTransientView(graph, `${id}-variance-anchor-rows`, 'float32', tileCount);
  const firstTile = createTransientView(graph, `${id}-variance-first-tile`, 'float32', 1);
  const binding = (name: string, view: GraphDataView, access: 'read' | 'read_write') => ({
    name,
    view,
    access,
    type: 'f32' as const
  });
  const declarations = `const TILE_COUNT: u32 = ${tileCount}u;`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-variance-anchor-rows`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'variance-anchor',
      invocationCount: tileCount,
      bindings: [
        binding('statistics', statistics, 'read'),
        binding('anchorRows', anchorRows, 'read_write')
      ],
      declarations,
      body: 'anchorRows[anchorRowsOffset + index] = select(f32(TILE_COUNT), f32(index), statistics[statisticsOffset + 3u * index] > 0.0);'
    }),
    ...new GPUReduction({
      id: `${id}-variance-first-tile`,
      input: anchorRows,
      output: firstTile,
      operation: 'min'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-variance-centre`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'variance-centre',
      invocationCount: tileCount,
      bindings: [
        binding('statistics', statistics, 'read'),
        binding('firstTile', firstTile, 'read'),
        binding('counts', channels[0], 'read_write'),
        binding('deviations', channels[1], 'read_write')
      ],
      declarations: `${declarations}\n${ROUNDED_DIFFERENCE_WGSL}`,
      body: `let count = statistics[statisticsOffset + 3u * index];
  let anchorIndex = min(u32(firstTile[firstTileOffset]), TILE_COUNT - 1u);
  let anchor = statistics[statisticsOffset + 3u * anchorIndex + 1u];
  let runtimeZero = bitcast<u32>(firstTile[firstTileOffset]) & 0x80000000u;
  counts[countsOffset + index] = count;
  deviations[deviationsOffset + index] = count * getRoundedDifference(statistics[statisticsOffset + 3u * index + 1u], anchor, runtimeZero);`
    }),
    ...[0, 1].flatMap(channel =>
      new GPUReduction({
        id: `${id}-variance-sum-${channel}`,
        input: channels[channel],
        output: totals[channel],
        operation: 'sum'
      }).getCommandNodes(graph)
    ),
    createWGSLKernelNode(graph, {
      id: `${id}-variance-squares`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'variance-squares',
      invocationCount: tileCount,
      bindings: [
        binding('statistics', statistics, 'read'),
        binding('firstTile', firstTile, 'read'),
        binding('totalCount', totals[0], 'read'),
        binding('totalDeviation', totals[1], 'read'),
        binding('squares', channels[2], 'read_write')
      ],
      declarations: `${declarations}\n${ROUNDED_DIFFERENCE_WGSL}`,
      body: `let count = statistics[statisticsOffset + 3u * index];
  let anchorIndex = min(u32(firstTile[firstTileOffset]), TILE_COUNT - 1u);
  let anchor = statistics[statisticsOffset + 3u * anchorIndex + 1u];
  let runtimeZero = bitcast<u32>(firstTile[firstTileOffset]) & 0x80000000u;
  let meanDeviation = totalDeviation[totalDeviationOffset] / max(totalCount[totalCountOffset], 1.0);
  let delta = getRoundedDifference(statistics[statisticsOffset + 3u * index + 1u], anchor, runtimeZero) - meanDeviation;
  squares[squaresOffset + index] = statistics[statisticsOffset + 3u * index + 2u] + count * delta * delta;`
    }),
    ...new GPUReduction({
      id: `${id}-variance-sum-2`,
      input: channels[2],
      output: totals[2],
      operation: 'sum'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-summarize`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'summarize',
      invocationCount: 1,
      bindings: [
        binding('totalSquares', totals[2], 'read'),
        binding('summary', summary, 'read_write')
      ],
      declarations: GET_NAN_WGSL,
      body: `let squares = totalSquares[totalSquaresOffset];
  var value = getNaN();
  if (summary[summaryOffset + 5u] > 0.5 && squares > 0.0) { value = 1.0 - summary[summaryOffset] / squares; }
  summary[summaryOffset + 3u] = value;`
    })
  ];
}

/** Collapse global golden-section extent/count tiles to one record for the unchanged search API. @internal */
export function createGWRSearchBoundsReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tileCount: number;
    tileCandidates: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, tileCount, tileCandidates} = props;
  const channels = [0, 1, 2].map(channel =>
    createTransientView(graph, `${id}-search-bound-channel-${channel}`, 'float32', tileCount)
  );
  const totals = [0, 1, 2].map(channel =>
    createTransientView(graph, `${id}-search-bound-total-${channel}`, 'float32', 1)
  );
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-search-bound-channels`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'search-bound-channels',
      invocationCount: tileCount,
      bindings: [
        {name: 'sourceValues', view: tileCandidates, type: 'f32', access: 'read'},
        ...channels.map((view, channel) => ({
          name: `channel${channel}`,
          view,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      body: channels
        .map(
          (_, channel) =>
            `channel${channel}[channel${channel}Offset + index] = sourceValues[sourceValuesOffset + index * 3u + ${channel}u];`
        )
        .join('\n')
    }),
    ...channels.flatMap((input, channel) =>
      new GPUReduction({
        id: `${id}-search-bound-${channel}`,
        input,
        output: totals[channel],
        operation: channel === 0 ? 'min' : channel === 1 ? 'max' : 'sum'
      }).getCommandNodes(graph)
    ),
    createWGSLKernelNode(graph, {
      id: `${id}-search-bound-record`,
      operation: 'GPUGeographicallyWeightedRegression',
      variant: 'search-bound-record',
      invocationCount: 1,
      bindings: [
        ...totals.map((view, channel) => ({
          name: `total${channel}`,
          view,
          type: 'f32' as const,
          access: 'read' as const
        })),
        {name: 'tileCandidates', view: tileCandidates, type: 'f32', access: 'read_write'}
      ],
      body: totals
        .map(
          (_, channel) =>
            `tileCandidates[tileCandidatesOffset + ${channel}u] = total${channel}[total${channel}Offset];`
        )
        .join('\n')
    })
  ];
}
