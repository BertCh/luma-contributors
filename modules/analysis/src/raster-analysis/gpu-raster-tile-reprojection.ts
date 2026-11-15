// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  GPURasterReprojection,
  getGPURasterReprojectionParameterValues,
  GPU_RASTER_REPROJECTION_PARAMETER_LENGTH,
  type GPURasterReprojectionProps
} from './gpu-raster-reprojection';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterGrid,
  validateRasterParameters
} from './raster-band-utils';
import {validateRequirement} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterExpansionViews, validateRasterExpansionGraph} from './raster-expansion-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';

/** Borrowed tile on the sourceGrid lattice, with half-open bounds relative to its top left. */
export type GPURasterReprojectionSourceTile = {
  /** Includes any caller-supplied halo; storage is packed row-major over these bounds. */
  pixelBounds: readonly [number, number, number, number];
  input: GPURasterBufferBand;
};

export type GPURasterTileReprojectionProps = Omit<GPURasterReprojectionProps, 'input'> & {
  /** Ordered low to high priority. Later tiles overwrite earlier tiles, including nodata. */
  sources: readonly GPURasterReprojectionSourceTile[];
};

/**
 * Explicit GPU mosaic assembly followed by destination-driven reprojection. Source metadata is
 * one shared CRS, affine lattice and resolution; bounds are integer sourceGrid pixel coordinates.
 * This is explicit packing: two float32/uint32 scratch arrays cover the entire bounded sourceGrid.
 * Missing tiles remain invalid. Bilinear taps can cross tile seams and halos; strict requires all
 * positive-weight taps, renormalize uses only available valid taps. No edge clamping or implicit
 * tile fetching occurs. Supply a one-pixel interpolation halo when subdividing source windows.
 *
 * Sampling and projection retain GPURasterReprojection's float32 precision/domain contract.
 * Use bounded local source windows whose buffers fit device storage limits. There is no fixed
 * tile-count limit or per-tile binding array: sequential GPU copy passes use four storage bindings.
 * Reexecuting a compiled graph rereads borrowed input buffers. Destroy after the compiled graph.
 */
export class GPURasterTileReprojection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterTileReprojectionProps;
  private reprojection?: GPURasterReprojection;

  constructor(props: GPURasterTileReprojectionProps) {
    this.id = props.id ?? 'raster-tile-reprojection';
    this.props = props;
    validateRasterGrid(this.id, props.sourceGrid);
    getGPURasterReprojectionParameterValues(props);
    if (props.parameters) {
      validateRasterParameters(this.id, props.parameters, GPU_RASTER_REPROJECTION_PARAMETER_LENGTH);
    }
    validateRasterExpansionViews(
      this.id,
      props.destinationGrid,
      [
        ...props.sources.flatMap(source => getRasterBandViews(source.input)),
        ...(props.parameters ? [props.parameters] : [])
      ],
      Object.values(props.output)
    );
    for (const source of props.sources) {
      const [left, top, right, bottom] = source.pixelBounds;
      // Tile coordinates must describe a nonempty, contained rectangle on the shared lattice.
      validateRequirement(
        this.id,
        source.pixelBounds.every(Number.isSafeInteger),
        'sources pixelBounds must be safe integers'
      );
      validateRequirement(
        this.id,
        left >= 0 && top >= 0 && right > left && bottom > top,
        'sources pixelBounds must be a nonempty [left, top, right, bottom] rectangle at or after the origin'
      );
      validateRequirement(
        this.id,
        right <= props.sourceGrid.width && bottom <= props.sourceGrid.height,
        'sources pixelBounds must lie inside sourceGrid'
      );
    }
  }

  destroy(): void {
    this.reprojection?.destroy();
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRequirement(
      this.id,
      !this.reprojection,
      'getCommandNodes can run once; construct a new contributor for another graph'
    );
    const {props, id} = this;
    validateRasterExpansionGraph(id, graph, [
      ...props.sources.flatMap(source => getRasterBandViews(source.input)),
      props.parameters,
      ...Object.values(props.output)
    ]);
    const count = props.sourceGrid.width * props.sourceGrid.height;
    const values = createTransientView(graph, `${id}-mosaic-values`, 'float32', count);
    const validity = createTransientView(graph, `${id}-mosaic-validity`, 'uint32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-clear`,
        operation: 'GPURasterTileReprojection',
        invocationCount: count,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
        ],
        body: 'values[valuesOffset+index]=0.0;validity[validityOffset+index]=0u;'
      })
    ];
    for (const [sourceIndex, source] of props.sources.entries()) {
      const [left, top, right, bottom] = source.pixelBounds;
      const width = right - left;
      const height = bottom - top;
      const band = getRasterBufferBandNodes(
        graph,
        `${id}-tile-${sourceIndex}`,
        source.input,
        {
          ...props.sourceGrid,
          width,
          height
        },
        true
      );
      nodes.push(
        ...band.nodes,
        createWGSLKernelNode(graph, {
          id: `${id}-route-${sourceIndex}`,
          operation: 'GPURasterTileReprojection',
          invocationCount: width * height,
          bindings: [
            {name: 'source', view: band.band.storage.values, type: 'f32', access: 'read'},
            {name: 'sourceValidity', view: band.band.validity!, type: 'u32', access: 'read'},
            {name: 'values', view: values, type: 'f32', access: 'read_write'},
            {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
          ],
          body: `let destinationIndex=(index/${width}u+${top}u)*${props.sourceGrid.width}u+index%${width}u+${left}u;values[valuesOffset+destinationIndex]=source[sourceOffset+index];validity[validityOffset+destinationIndex]=sourceValidity[sourceValidityOffset+index];`
        })
      );
    }
    this.reprojection = new GPURasterReprojection({
      ...props,
      id: `${id}-sample`,
      input: {
        id: `${id}-mosaic`,
        format: 'float32',
        storage: {kind: 'buffer', values},
        validity
      }
    });
    nodes.push(...this.reprojection.getCommandNodes(graph));
    return nodes;
  }
}
