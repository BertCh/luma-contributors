// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getFixedPointSource} from './network-coarsening-fixed-point';
import type {GPUNetworkCoarseningOutput} from './gpu-network-coarsening';

/** Final reductions and optional writers, after all superedge accumulation passes. @internal */
export function createCoarseningPublicationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  settings: {
    id: string;
    output: GPUNetworkCoarseningOutput;
    groupCapacity: number;
    groupAccumulatorStride: number;
    summaryLength: number;
    groupAcc: GraphDataView<'uint32'>;
    intraAcc?: GraphDataView<'uint32'>;
    stats: GraphDataView<'uint32'>;
    segmentTotal: GraphDataView<'uint32'>;
    overflowFlag: GraphDataView<'uint32'>;
    usesDataUnit: boolean;
    needsWeightScale: boolean;
    positionUnitExponent: string;
    valueUnitExponent: string;
    weightUnitExponent: string;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    output,
    groupCapacity,
    groupAccumulatorStride,
    summaryLength,
    groupAcc,
    intraAcc,
    stats,
    segmentTotal,
    overflowFlag,
    usesDataUnit,
    needsWeightScale,
    positionUnitExponent,
    valueUnitExponent,
    weightUnitExponent
  } = settings;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const OPERATION = 'GPUNetworkCoarsening';
  // Per-group finish: centroids, bounds, value sums and intra-group outputs.
  const finishBindings: WGSLKernelBinding[] = [
    {
      name: 'groupCounts',
      view: output.groupNodeCount,
      type: 'u32',
      access: 'read'
    },
    {name: 'groupAcc', view: groupAcc, type: 'u32', access: 'read'}
  ];
  const finishStatements: string[] = [];
  if (output.groupCentroid) {
    finishBindings.push({
      name: 'centroidOut',
      view: output.groupCentroid,
      type: 'f32',
      access: 'read_write'
    });
    finishStatements.push(`let positionUnitExponent = ${positionUnitExponent};
  centroidOut[centroidOutOffset + 2u * index] = select(0.0, dequantize(groupAcc[groupAccOffset + base], groupAcc[groupAccOffset + base + 1u], positionUnitExponent) / f32(count), count > 0u);
  centroidOut[centroidOutOffset + 2u * index + 1u] = select(0.0, dequantize(groupAcc[groupAccOffset + base + 2u], groupAcc[groupAccOffset + base + 3u], positionUnitExponent) / f32(count), count > 0u);`);
  }
  if (output.groupBounds) {
    finishBindings.push({
      name: 'boundsOut',
      view: output.groupBounds,
      type: 'f32',
      access: 'read_write'
    });
    finishStatements.push(`boundsOut[boundsOutOffset + 4u * index] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 6u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 1u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 7u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 2u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 8u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 3u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 9u]), count > 0u);`);
  }
  if (output.groupValueSum) {
    finishBindings.push({
      name: 'valueOut',
      view: output.groupValueSum,
      type: 'f32',
      access: 'read_write'
    });
    finishStatements.push(
      `valueOut[valueOutOffset + index] = dequantize(groupAcc[groupAccOffset + base + 4u], groupAcc[groupAccOffset + base + 5u], ${valueUnitExponent});`
    );
  }
  if (usesDataUnit && (output.groupCentroid || output.groupValueSum)) {
    finishBindings.push({name: 'stats', view: stats, type: 'u32', access: 'read'});
  }
  // The intra-group outputs are their own kernel: with the scale word of the weights one
  // kernel for every group output would bind more storage buffers than a device allows.
  if (intraAcc) {
    const intraBindings: WGSLKernelBinding[] = [
      {name: 'intraAcc', view: intraAcc, type: 'u32', access: 'read'}
    ];
    const intraStatements: string[] = [];
    if (output.groupIntraEdgeCount) {
      intraBindings.push({
        name: 'intraCountOut',
        view: output.groupIntraEdgeCount,
        type: 'u32',
        access: 'read_write'
      });
      intraStatements.push(
        'intraCountOut[intraCountOutOffset + index] = intraAcc[intraAccOffset + 3u * index];'
      );
    }
    if (output.groupIntraWeight) {
      intraBindings.push({
        name: 'intraWeightOut',
        view: output.groupIntraWeight,
        type: 'f32',
        access: 'read_write'
      });
      if (needsWeightScale) {
        intraBindings.push({name: 'stats', view: stats, type: 'u32', access: 'read'});
      }
      intraStatements.push(
        `intraWeightOut[intraWeightOutOffset + index] = dequantize(intraAcc[intraAccOffset + 3u * index + 1u], intraAcc[intraAccOffset + 3u * index + 2u], ${weightUnitExponent});`
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-intra-groups`,
        operation: OPERATION,
        variant: 'intra-groups',
        bindings: intraBindings,
        invocationCount: groupCapacity,
        declarations: getFixedPointSource(!usesDataUnit),
        body: intraStatements.join('\n  ')
      })
    );
  }
  if (finishStatements.length > 0) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-groups`,
        operation: OPERATION,
        variant: 'groups',
        bindings: finishBindings,
        invocationCount: groupCapacity,
        declarations: getFixedPointSource(!usesDataUnit),
        body: `let base = index * ${groupAccumulatorStride}u;
  let count = groupCounts[groupCountsOffset + index];
  ${finishStatements.join('\n  ')}`
      })
    );
  }

  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation: OPERATION,
      requiredCount: segmentTotal,
      output: output.edges,
      candidateOverflowSources: output.edges.status.candidateOverflow ? [overflowFlag] : []
    })
  );
  if (output.summary) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'stats', view: stats, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'u32', access: 'read_write'}
        ],
        invocationCount: summaryLength,
        body: 'summary[summaryOffset + index] = stats[statsOffset + index];'
      })
    );
  }
  return nodes;
}
