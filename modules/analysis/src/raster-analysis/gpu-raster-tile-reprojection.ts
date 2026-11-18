// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
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
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  snapshotRasterBand,
  validateRasterGrid,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph,
  validateRasterParameters
} from './raster-band-utils';
import {
  validateCommandNodeBudget,
  validateRequirement,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterExpansionViews, validateRasterExpansionGraph} from './raster-expansion-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';

// Published GPURasterNeighborhood and GPURasterTextureToBuffer each emit one conversion pass.
// Keep the allowance explicit and test the actual graph against the construction estimate.
const BAND_CONVERSION_COMMAND_NODE_ALLOWANCE = 1;

/** Borrowed tile on the sourceGrid lattice, with half-open bounds relative to its top left. */
export type GPURasterReprojectionSourceTile = {
  /** Includes any caller-supplied halo; storage is packed row-major over these bounds. */
  pixelBounds: readonly [number, number, number, number];
  input: GPURasterBand;
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
 * Use bounded local source windows whose buffers fit device storage limits. The construction-time
 * command-node budget is 32,768: 3 fixed nodes plus 2 per source tile (at most 16,382 tiles).
 * Sequential GPU copy passes use four storage bindings, with no per-tile binding array.
 * Reexecuting a compiled graph rereads borrowed input buffers. Destroy after the compiled graph.
 * Source/destination dimensions, affines and CRS JSON values are captured at construction. Caller
 * metadata edits cannot change the validated tile footprint, scratch allocation or sampling grid; use a parameter
 * buffer for supported per-frame affine changes while keeping both shapes fixed.
 */
export class GPURasterTileReprojection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterTileReprojectionProps;
  readonly estimatedCommandNodeCount: number;
  private readonly sourceTiles: readonly GPURasterReprojectionSourceTile[];
  private readonly constructionProps: GPURasterTileReprojectionProps;
  private reprojection?: GPURasterReprojection;

  constructor(props: GPURasterTileReprojectionProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'sourceGrid',
      'destinationGrid',
      'sources',
      'inverseProjection',
      'method',
      'noDataPolicy',
      'parameters',
      'output'
    ]);
    const [sourceGrid, destinationGrid] = [props.sourceGrid, props.destinationGrid].map(grid => {
      const captured = snapshotConstructionFields(grid, [
        'width',
        'height',
        'affine',
        'pixelInterpretation',
        'coordinateReferenceSystem',
        'level',
        'levelZeroOrigin'
      ]);
      return Object.freeze({
        ...captured,
        affine: Object.freeze([...captured.affine]) as typeof grid.affine,
        // The sampler compares serialized CRS values later. Capture that same JSON value now,
        // including nested projection JSON, without retaining caller-owned metadata objects.
        coordinateReferenceSystem: captured.coordinateReferenceSystem
          ? Object.freeze(
              JSON.parse(
                JSON.stringify(
                  snapshotConstructionFields(captured.coordinateReferenceSystem, [
                    'authority',
                    'wellKnownText',
                    'projectionJson'
                  ])
                )
              ) as NonNullable<typeof grid.coordinateReferenceSystem>
            )
          : undefined
      });
    });
    // Capture the lattice used by both tile-bound validation and the later mosaic/sampler.
    // In particular, do not leave affine arrays aliased to caller-owned metadata.
    props = {
      ...props,
      sourceGrid,
      destinationGrid,
      output: snapshotConstructionFields(props.output, ['values', 'validity'])
    };
    this.id = props.id ?? 'raster-tile-reprojection';
    this.estimatedCommandNodeCount =
      2 +
      BAND_CONVERSION_COMMAND_NODE_ALLOWANCE +
      props.sources.length * (1 + BAND_CONVERSION_COMMAND_NODE_ALLOWANCE);
    validateCommandNodeBudget(
      `${this.id} sources.length=${props.sources.length}`,
      this.estimatedCommandNodeCount
    );
    // Reject by length before reading any tile, then keep the validated routing plan private.
    // Clone metadata and bindings, retaining caller-owned GPU view identities and buffer contents.
    this.sourceTiles = Object.freeze(
      props.sources.map(source => {
        const captured = snapshotConstructionFields(source, ['pixelBounds', 'input']);
        return Object.freeze({
          pixelBounds: Object.freeze([...captured.pixelBounds]) as typeof source.pixelBounds,
          input: snapshotRasterBand(captured.input)
        });
      })
    );
    this.constructionProps = Object.freeze({...props, sources: this.sourceTiles});
    this.props = {...this.constructionProps, sources: [...this.sourceTiles]};
    validateRasterGrid(this.id, props.sourceGrid);
    getGPURasterReprojectionParameterValues(props);
    if (props.parameters) {
      validateRasterParameters(this.id, props.parameters, GPU_RASTER_REPROJECTION_PARAMETER_LENGTH);
    }
    validateRasterExpansionViews(
      this.id,
      props.destinationGrid,
      [
        ...this.sourceTiles.flatMap(source => getRasterBandViews(source.input)),
        ...(props.parameters ? [props.parameters] : [])
      ],
      Object.values(props.output)
    );
    for (const source of this.sourceTiles) {
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
      validateRasterTextureBand(this.id, source.input, {
        ...props.sourceGrid,
        width: right - left,
        height: bottom - top
      });
    }
  }

  destroy(): void {
    this.reprojection?.destroy();
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.constructionProps;
    const storageCellCount = storageProps.sourceGrid.width * storageProps.sourceGrid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...this.sourceTiles.flatMap(source => getRasterBandViews(source.input)),
        storageProps.parameters,
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'mosaic and canonical band planes': storageCellCount * 4,
        'affine parameters': GPU_RASTER_REPROJECTION_PARAMETER_LENGTH * 4,
        'projection parameters': storageProps.inverseProjection
          ? storageProps.inverseProjection.packParameters().byteLength
          : 0
      }
    );

    validateRequirement(
      this.id,
      !this.reprojection,
      'getCommandNodes can run once; construct a new contributor for another graph'
    );
    const props = storageProps;
    const {id} = this;
    // Check every tile before the clear pass or any mosaic allocation mutates the graph.
    for (const source of this.sourceTiles) {
      validateRasterBandBelongsToGraph(id, graph, source.input, []);
    }
    validateRasterExpansionGraph(id, graph, [
      ...this.sourceTiles.flatMap(source => getRasterBandViews(source.input)),
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
    for (const [sourceIndex, source] of this.sourceTiles.entries()) {
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
