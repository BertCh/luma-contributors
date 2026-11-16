// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURasterDilation, GPURasterErosion} from '@luma.gl/graph/gpu-raster';
import {captureGraphCommandNodes} from '../analysis-core/kernels/gpu-contributor-utils';

/**
 * Use a fixed radius-one public morphology pass so all radii share a shader and pipeline.
 * Repeated Manhattan unit diamonds equal the requested diamond exactly on a rectangle.
 */
const RASTER_MORPHOLOGY_MAXIMUM_PASS_RADIUS = 1;

/**
 * @internal Minimum or maximum of a dense float32 raster over the Manhattan diamond of `radius`
 * cells, clipped to the grid: the erosion and dilation of the SMRF openings.
 *
 * The windows are the published `GPURasterErosion` and `GPURasterDilation` with the `'cross'`
 * structuring element, a `'nodata'` border and the `'ignore'` policy, so cells outside the grid
 * do not take part. One pass covers a radius of one; a larger diamond is a sequence of
 * passes, which is exact: a diamond is the Minkowski sum of smaller diamonds, and clipping to a
 * rectangle preserves shortest Manhattan paths. Every input cell must hold a finite value (SMRF
 * inpaints first); the published classes treat a non-finite sample as missing.
 *
 * @returns the nodes in execution order and the transient view that holds the result (`input`
 * itself for a radius of 0).
 */
export function getPointCloudDiamondMorphologyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'float32'>,
  grid: {width: number; height: number},
  radius: number,
  operation: 'min' | 'max'
): {nodes: GPUCommandNode<Parameters>[]; output: GraphDataView<'float32'>} {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const cells = grid.width * grid.height;
  const Morphology = operation === 'min' ? GPURasterErosion : GPURasterDilation;
  let previous = input;
  let remainingRadius = radius;
  for (let pass = 0; remainingRadius > 0; pass++) {
    const passRadius = Math.min(remainingRadius, RASTER_MORPHOLOGY_MAXIMUM_PASS_RADIUS);
    const output = createTransientView(graph, `${id}-${pass}`, 'float32', cells);
    // Every cell of a dense raster is valid; the published class still requires the view.
    const outputValidity = createTransientView(graph, `${id}-${pass}-validity`, 'uint32', cells);
    const source = previous;
    nodes.push(
      ...captureGraphCommandNodes(graph, () =>
        new Morphology({
          id: `${id}-${pass}`,
          width: grid.width,
          height: grid.height,
          radius: passRadius,
          structuringElement: 'cross',
          borderMode: 'nodata',
          noDataPolicy: 'ignore',
          input: {
            id: `${id}-${pass}-input`,
            format: 'float32',
            storage: {kind: 'buffer', values: source}
          },
          output,
          outputValidity
        }).addToGraph(graph)
      )
    );
    previous = output;
    remainingRadius -= passRadius;
  }
  return {nodes, output: previous};
}
