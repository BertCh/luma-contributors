// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {rasterBinding, validateRasterGraph, validateRasterViews} from './raster-kernel-utils';
import {GPURasterRegionGraph, type GPURasterRegionGraphProps} from './gpu-raster-region-graph';

export type GPURasterRegionTile = {
  /** Disjoint core cells, without halo duplication. */
  grid: GPURasterMetadata;
  /** Integer core origin in the destination grid's column/row coordinates. */
  origin: readonly [number, number];
  labels: GraphDataView<'uint32'>;
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
): GPURasterRegionGraph {
  const id = props.id ?? 'raster-region-tiles';
  const count = props.grid.width * props.grid.height;
  let area = 0;
  for (let index = 0; index < props.tiles.length; index++) {
    const tile = props.tiles[index];
    const [column, row] = tile.origin;
    assert(Number.isInteger(column) && Number.isInteger(row) && column >= 0 && row >= 0);
    assert(
      Number.isInteger(tile.grid.width) &&
        Number.isInteger(tile.grid.height) &&
        tile.grid.width > 0 &&
        tile.grid.height > 0
    );
    assert(
      column + tile.grid.width <= props.grid.width && row + tile.grid.height <= props.grid.height
    );
    assert(
      tile.labels.length === tile.grid.width * tile.grid.height &&
        tile.values.length === tile.labels.length
    );
    for (const previous of props.tiles.slice(0, index)) {
      assert(
        column >= previous.origin[0] + previous.grid.width ||
          previous.origin[0] >= column + tile.grid.width ||
          row >= previous.origin[1] + previous.grid.height ||
          previous.origin[1] >= row + tile.grid.height
      );
    }
    area += tile.grid.width * tile.grid.height;
  }
  // Inside, nonoverlapping rectangles with equal total area form a complete core partition.
  assert(area === count);
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
  const contributor = new GPURasterRegionGraph({...props, labels, values});
  graph.add(contributor);
  return contributor;
}
