// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRasterBandViews,
  validateRasterBandBelongsToGraph,
  validateRasterGrid
} from '../raster-band-utils';
import {
  getRasterAlgebraInputNodes,
  getNamedRasterAlgebraInputViews,
  validateRasterAlgebraInput
} from './raster-algebra-input';
import {validateRasterAlgebraView} from './raster-algebra-utils';

/** A legacy band-sequential float view or an ordered collection of published bands. @internal */
export type RasterAlgebraStackInput = GraphDataView<'float32'> | readonly GPURasterBand[];

/** Type guard for readonly band arrays; graph views and chunked vectors are not arrays. @internal */
export function isRasterAlgebraBandStack(
  input: RasterAlgebraStackInput
): input is readonly GPURasterBand[] {
  return Array.isArray(input);
}

/** Original views, including every band's validity, for alias and graph ownership checks. @internal */
export function getNamedRasterAlgebraStackViews(
  input: RasterAlgebraStackInput
): Record<string, GraphDataView | undefined> {
  return isRasterAlgebraBandStack(input)
    ? Object.assign(
        {},
        ...input.map((band, index) => getNamedRasterAlgebraInputViews(band, `stack[${index}]`))
      )
    : {stack: input};
}

/** Original views, including every band's validity, for alias and graph ownership checks. @internal */
export function getRasterAlgebraStackViews(input: RasterAlgebraStackInput): GraphDataView[] {
  return isRasterAlgebraBandStack(input) ? input.flatMap(getRasterBandViews) : [input];
}

/** Construction-time shape and metadata checks; all bands occupy the same complete grid. @internal */
export function validateRasterAlgebraStack(
  id: string,
  input: RasterAlgebraStackInput,
  grid: GPURasterMetadata | undefined,
  layerCount: number,
  cellCount: number
): void {
  validateRequirement(
    id,
    Boolean(input),
    'stack must be a float32 view or an array of raster bands'
  );
  if (!isRasterAlgebraBandStack(input)) {
    validateRasterAlgebraView(id, 'stack', input, 'float32', layerCount * cellCount);
    return;
  }
  validateRequirement(
    id,
    input.length === layerCount,
    `stack must contain exactly ${layerCount} bands`
  );
  validateRequirement(id, Boolean(grid), 'stack requires grid for raster bands');
  validateRequirement(
    id,
    validateRasterGrid(id, grid!) === cellCount,
    `stack grid must contain exactly ${cellCount} cells`
  );
  input.forEach((band, layer) => {
    validateRequirement(
      id,
      Boolean(band) && typeof band === 'object' && 'storage' in band,
      `stack[${layer}] must be a raster band`
    );
    validateRasterAlgebraInput(id, `stack[${layer}]`, band, grid, cellCount);
  });
}

/**
 * Per band: an allowance of two published conversion nodes plus one packing node. Current
 * public buffer/texture conversions use one node; reserve an extra node for their implementation.
 * @internal
 */
const BAND_STACK_NODE_ALLOWANCE = 3;

/** Upper bound for additional nodes of a published-band stack. @internal */
export function getRasterAlgebraStackNodeCount(input: RasterAlgebraStackInput): number {
  return isRasterAlgebraBandStack(input) ? input.length * BAND_STACK_NODE_ALLOWANCE : 0;
}

/**
 * Canonicalizes bands through published raster primitives, then packs their calibrated values in
 * band order. Invalid cells are NaN. Each copy dispatches once per cell with two storage bindings;
 * analytical kernels retain one stack binding even at the maximum layer count. Scratch is owned
 * by the compiled graph; there are no contributor-owned buffers or CPU pixel loops.
 * @internal
 */
export function getRasterAlgebraStackNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: RasterAlgebraStackInput,
  grid: GPURasterMetadata | undefined,
  cellCount: number
): {values: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]} {
  if (!isRasterAlgebraBandStack(input)) return {values: input, nodes: []};
  // Resolve all texture ownership and binding limits before creating any scratch or nodes.
  for (const band of input) validateRasterBandBelongsToGraph(id, graph, band, []);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'stack',
    input.length * cellCount * Float32Array.BYTES_PER_ELEMENT,
    'reduce layerCount or grid dimensions'
  );
  const values = createTransientView(
    graph,
    `${id}-band-stack`,
    'float32',
    input.length * cellCount
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  input.forEach((band, layer) => {
    const source = getRasterAlgebraInputNodes(graph, `${id}-band-${layer}`, band, grid);
    const destination = graph.createDataView(values.buffer, {
      format: 'float32',
      byteOffset: layer * cellCount * Float32Array.BYTES_PER_ELEMENT,
      length: cellCount
    });
    // Published canonicalization scratch has STORAGE usage only. Pack with a storage kernel
    // rather than a buffer-copy node, which would require COPY_SRC on that borrowed scratch.
    nodes.push(
      ...source.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-pack-band-${layer}`,
        operation: 'RasterAlgebraBandStack',
        bindings: [
          {name: 'source', view: source.values, type: 'f32', access: 'read'},
          {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
  });
  return {values, nodes};
}
