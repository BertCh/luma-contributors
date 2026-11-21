// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateRequirement,
  validateCapacity,
  validateCommandNodeBudget,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {rasterBinding, validateRasterGraph, validateRasterViews} from './raster-kernel-utils';
import {
  GPURasterRegionGraph,
  getRasterRegionGraphExecutionPlan,
  type GPURasterRegionGraphProps
} from './gpu-raster-region-graph';
import {validateRasterWritableDispatches} from '../raster-time-series/raster-time-series-chunks-b';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

export type GPURasterRegionTile = {
  /** Disjoint core cells, without halo duplication. */
  grid: GPURasterMetadata;
  /** Integer core origin in the destination grid's column/row coordinates. */
  origin: readonly [number, number];
  labels: GraphDataView<'uint32'>;
  /** Optional one-word source-label completeness, combined before assembly. */
  labelsComplete?: GraphDataView<'uint32'>;
  values: GraphDataView<'float32'>;
  /** Optional tile-local label to stable source-region ID; out-of-range labels are background. */
  sourceRegionIds?: GraphDataView<'uint32'>;
};

export type GPURasterRegionGraphTilesProps = Omit<
  GPURasterRegionGraphProps,
  'labels' | 'values'
> & {tiles: readonly GPURasterRegionTile[]};

/** Explicit GPU assembly of a complete, nonoverlapping core partition followed by region merging.
 * The caller supplies equivalence IDs (e.g. from cross-tile components). Shared source IDs merge
 * observations before computing means; all inter-core boundary edges are counted exactly once.
 * No host readback or implicit label reconciliation occurs. */
export function addRasterRegionGraphTiles<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegionGraphTilesProps
): GPURasterRegionGraph & {readonly estimatedRecipeCommandNodeCount: number} {
  const id = props.id ?? 'raster-region-tiles';
  const plan = getRasterRegionGraphExecutionPlan({...props, id}, graph.device);
  const count = props.grid.width * props.grid.height;
  const certificates = [
    ...(props.labelsComplete ? [props.labelsComplete] : []),
    ...props.tiles.flatMap(tile => (tile.labelsComplete ? [tile.labelsComplete] : []))
  ];
  validateRequirement(
    id,
    certificates.length === 0 || Boolean(props.output.status.complete),
    'labelsComplete requires output.status.complete to retain source-label completeness'
  );
  const estimatedRecipeCommandNodeCount =
    plan.estimatedCommandNodeCount +
    props.tiles.length +
    (certificates.length ? 1 + certificates.length : 0);
  validateCommandNodeBudget(id, estimatedRecipeCommandNodeCount);
  const {adjacency, weights, mapping, status} = props.output;
  for (const [name, view, format, length] of [
    ['output.adjacency', adjacency, 'uint32', props.regionCount ** 2],
    ['output.weights', weights, 'float32', props.regionCount ** 2],
    ['output.mapping', mapping, 'uint32', props.regionCount],
    ['output.status.iterationLimitReached', status.iterationLimitReached, 'uint32', 1],
    ['output.status.complete', status.complete, 'uint32', 1]
  ] as const)
    if (view) {
      validateSinglePackedView(
        view,
        [format],
        {id, contributor: 'GPURasterRegionGraphTiles'},
        name
      );
      validateRequirement(
        id,
        view.length >= length && (!name.startsWith('output.status.') || view.length === 1),
        `${name} must hold ${length} entries; received ${view.length}`
      );
    }
  const allInputs = props.tiles.flatMap(tile => [
    tile.labels,
    tile.values,
    ...(tile.sourceRegionIds ? [tile.sourceRegionIds] : [])
  ]);
  allInputs.push(...certificates);
  const allOutputs = [
    adjacency,
    weights,
    mapping,
    status.iterationLimitReached,
    ...(status.complete ? [status.complete] : [])
  ];
  // Preserve route-specific packed-output allowances: Boruvka publishes mapping and diagnostics separately.
  validateRasterWritableDispatches(id, allInputs, allOutputs, [
    {inputs: [], outputs: [adjacency]},
    ...(props.mergeMode === 'adaptive'
      ? [{inputs: [adjacency], outputs: [weights, mapping, status.iterationLimitReached]}]
      : [
          {inputs: [adjacency], outputs: [weights]},
          ...(plan.selectedSearchStructure === 'dense'
            ? [{inputs: [weights], outputs: [mapping, status.iterationLimitReached]}]
            : [
                {inputs: [weights], outputs: [status.iterationLimitReached]},
                {inputs: [], outputs: [mapping]}
              ])
        ]),
    // Certificate folds write private scratch. Final publication reads that private scratch,
    // never the original certificates, so disjoint packed certificate/output words may share
    // an aligned window. The recipe-wide logical input/output rejection still applies.
    ...certificates.map(certificate => ({inputs: [certificate], outputs: []})),
    ...(status.complete ? [{inputs: [], outputs: [status.complete]}] : [])
  ]);
  validateRasterGraph(id, graph, allInputs, allOutputs);
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries([...allInputs, ...allOutputs].map((view, index) => [`view${index}`, view]))
  );
  validateCapacity(
    id,
    'assembled labels and values byteLength',
    count * 4,
    1,
    Math.min(graph.device.limits.maxStorageBufferBindingSize, graph.device.limits.maxBufferSize)
  );
  let area = 0;
  for (let index = 0; index < props.tiles.length; index++) {
    const tile = props.tiles[index];
    for (const [name, view, format] of [
      ['labels', tile.labels, 'uint32'],
      ['values', tile.values, 'float32'],
      ['sourceRegionIds', tile.sourceRegionIds, 'uint32'],
      ['labelsComplete', tile.labelsComplete, 'uint32']
    ] as const)
      if (view)
        validateSinglePackedView(
          view,
          [format],
          {id, contributor: 'GPURasterRegionGraphTiles'},
          `tiles[${index}].${name}`
        );
    const assemblyStorageBindingCount = tile.sourceRegionIds ? 5 : 4;
    validateRequirement(
      id,
      assemblyStorageBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage,
      `tiles[${index}] assembly needs ${assemblyStorageBindingCount} storage buffers per shader stage; received limit ${graph.device.limits.maxStorageBuffersPerShaderStage}`
    );
    const [column, row] = tile.origin;
    validateRequirement(
      id,
      Number.isInteger(column) && Number.isInteger(row) && column >= 0 && row >= 0,
      `tiles[${index}].origin must be nonnegative integers`
    );
    validateRequirement(
      id,
      Number.isInteger(tile.grid.width) &&
        Number.isInteger(tile.grid.height) &&
        tile.grid.width > 0 &&
        tile.grid.height > 0,
      `tiles[${index}].grid width and height must be positive integers`
    );
    validateRequirement(
      id,
      column + tile.grid.width <= props.grid.width && row + tile.grid.height <= props.grid.height,
      `tiles[${index}] must lie inside grid`
    );
    validateRequirement(
      id,
      tile.labels.length === tile.grid.width * tile.grid.height &&
        tile.values.length === tile.labels.length,
      `tiles[${index}] labels and values must hold exactly grid.width times grid.height entries`
    );
    for (const previous of props.tiles.slice(0, index)) {
      validateRequirement(
        id,
        column >= previous.origin[0] + previous.grid.width ||
          previous.origin[0] >= column + tile.grid.width ||
          row >= previous.origin[1] + previous.grid.height ||
          previous.origin[1] >= row + tile.grid.height,
        `tiles[${index}] must not overlap an earlier tile`
      );
    }
    area += tile.grid.width * tile.grid.height;
  }
  // Inside, nonoverlapping rectangles with equal total area form a complete core partition.
  validateRequirement(id, area === count, 'tiles must cover grid completely');
  for (const certificate of certificates) {
    validateSinglePackedView(
      certificate,
      ['uint32'],
      {id, contributor: 'GPURasterRegionGraphTiles'},
      'labelsComplete'
    );
    validateRequirement(
      id,
      certificate.length === 1,
      `labelsComplete must hold exactly one uint32 word; received ${certificate.length}`
    );
  }
  // Every admission check above is read-only; reserve IDs only once the whole recipe fits.
  const labelsComplete = certificates.length
    ? createTransientView(graph, `${id}-labels-complete`, 'uint32', 1)
    : undefined;
  if (labelsComplete) {
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${id}-complete-init`,
        operation: 'GPURasterRegionGraphTiles',
        invocationCount: 1,
        bindings: [rasterBinding('complete', labelsComplete, true)],
        body: 'complete[completeOffset] = 1u;'
      })
    );
    for (const [index, certificate] of certificates.entries())
      graph.add(
        createWGSLKernelNode(graph, {
          id: `${id}-complete-${index}`,
          operation: 'GPURasterRegionGraphTiles',
          invocationCount: 1,
          bindings: [
            rasterBinding('certificate', certificate),
            rasterBinding('complete', labelsComplete, true)
          ],
          body: 'complete[completeOffset] = select(0u, complete[completeOffset], certificate[certificateOffset] != 0u);'
        })
      );
  }
  const labels = createTransientView(graph, `${id}-labels`, 'uint32', count);
  const values = createTransientView(graph, `${id}-values`, 'float32', count);
  for (const [index, tile] of props.tiles.entries()) {
    const inputs = [
      tile.labels,
      tile.values,
      ...(tile.sourceRegionIds ? [tile.sourceRegionIds] : [])
    ];
    validateRasterViews(id, inputs, [labels, values]);
    validateRasterGraph(id, graph, inputs, [labels, values]);
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${id}-assemble-${index}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: tile.labels.length,
        bindings: [
          rasterBinding('sourceLabels', tile.labels),
          rasterBinding('sourceValues', tile.values),
          ...(tile.sourceRegionIds ? [rasterBinding('sourceRegionIds', tile.sourceRegionIds)] : []),
          rasterBinding('labels', labels, true),
          rasterBinding('values', values, true)
        ],
        body: `let destination = (index / ${tile.grid.width}u + ${tile.origin[1]}u) * ${props.grid.width}u + index % ${tile.grid.width}u + ${tile.origin[0]}u;
let label = sourceLabels[sourceLabelsOffset + index];
var sourceLabel = label;
${tile.sourceRegionIds ? `sourceLabel = 0xffffffffu; if (label < ${tile.sourceRegionIds.length}u) { sourceLabel = sourceRegionIds[sourceRegionIdsOffset + label]; }` : ''}
labels[labelsOffset + destination] = sourceLabel; values[valuesOffset + destination] = sourceValues[sourceValuesOffset + index];`
      })
    );
  }
  const contributor = new GPURasterRegionGraph({...props, labels, values, labelsComplete});
  graph.add(contributor);
  return Object.assign(contributor, {estimatedRecipeCommandNodeCount});
}
