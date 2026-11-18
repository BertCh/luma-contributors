// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import {RING_SHELL_RECORD_WORDS} from '../../analysis-core/ring-assembly/ring-shell-index';
import {validateRasterAlgebraInputBindings} from '../raster-algebra/raster-algebra-input';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUSegmentRingAssembly,
  type GPUSegmentRingAssemblyOutput
} from '../../analysis-core/ring-assembly/index';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUIsobands} from './gpu-isobands';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {snapshotRasterBand} from '../raster-band-utils';

/**
 * Largest `edgeCapacity` of {@link GPUIsobandRings} on `device`: the boundary edge list holds one
 * `float32x4` (16 bytes) per edge in one storage binding, so 8388608 on a default device (128 MiB
 * bindings).
 */
export function getGPUIsobandRingsMaximumEdgeCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/**
 * Largest ring capacity (`output.ringOffsets.length - 1`) of {@link GPUIsobandRings} on `device`:
 * the ring assembly keeps the bounds of every ring in a transient of four `float32` (16 bytes) per
 * ring in one storage binding, so 8388608 on a default device (128 MiB bindings).
 */
export function getGPUIsobandRingsMaximumRingCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/**
 * Caller-owned outputs of {@link GPUIsobandRings}: every output of `GPUSegmentRingAssembly`
 * (`ringOffsets`, `positions`, `count`, `overflow`, optional `ringAreas`, `ringIsHole`,
 * `ringShells`, `polygons`, ...) plus the band of each ring and the boundary edge status words.
 *
 * `ringGroups` receives the band index of each ring, and holes only attach to shells of their own
 * band. Without `ringGroups` rings are still band-pure.
 */
export type GPUIsobandRingsOutput = GPUSegmentRingAssemblyOutput & {
  /** One-row status words of the boundary edge list. */
  status: GPUSegmentRingAssemblyOutput['status'] & GPUIsobandRingsStatus;
};

/** One-row status words of {@link GPUIsobandRings}. */
export type GPUIsobandRingsStatus = {
  /** Optional one-row scalar receiving the number of band boundary edges before ring assembly. */
  edgeCount?: GraphDataView<'uint32'>;
  /**
   * Required packed uint32 view with at least one row. Its first word receives 1 when more boundary
   * edges existed than `edgeCapacity`; spare words are untouched. Rings are then incomplete.
   */
  edgeOverflow: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUIsobandRings}. The raster inputs match `GPUIsobandsProps`; every
 * per-frame value (breaks, parameters, band window) stays a buffer write.
 */
export type GPUIsobandRingsProps = {
  /** Prefix for generated node IDs. Defaults to `'isoband-rings'`. */
  id?: string;
  /** Raster grid: its width and height (each at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Published band with validity, native nodata and calibration, or legacy packed float32 values. */
  values: GraphDataView<'float32'> | GPURasterBand;
  /** Optional input row filter over the raster: nonzero keeps the cell, zero marks no-data. */
  mask?: GraphDataView<'uint32'>;
  /** Optional finite nodata sentinel compared exactly against sample values. */
  noDataValue?: number;
  /** Break values, see `GPUIsobandsProps.breaks`. */
  breaks: GraphDataView<'float32'>;
  /**
   * Per-frame parameters written with `getGPUIsobandsParameterValues`. Ring vertices are relative
   * to its `localOrigin`. For projected rasters set it to the raster or tile corner: besides the
   * float32 precision, the ring assembly hashes `floor(vertex / vertexTolerance)` as an i32, which
   * saturates beyond `2^31 * vertexTolerance` (214,748 units at the default tolerance).
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time capacity of the intermediate band boundary edge list. A ring of `n` vertices
   * needs `n` edges; budget the total vertex count of every ring of every band. An integer in
   * [1, {@link getGPUIsobandRingsMaximumEdgeCapacity}] (8388608 on a default device).
   */
  edgeCapacity: number;
  /**
   * Vertex matching distance in world units, see `GPUSegmentRingAssemblyProps.vertexTolerance`.
   * Boundary vertices of neighbouring cells are bit-identical, so any value well below the
   * distance between distinct vertices works. Defaults to 1e-4 of the world unit.
   */
  vertexTolerance?: number;
  /** Split rings that touch themselves at a vertex, see `GPUSegmentRingAssemblyProps`. Defaults to true. */
  splitTouchingRings?: boolean;
  /**
   * Caller-owned outputs. The ring capacity `output.ringOffsets.length - 1` is at most
   * {@link getGPUIsobandRingsMaximumRingCapacity} (8388608 on a default device).
   */
  output: GPUIsobandRingsOutput;
};

/**
 * Closed polygon rings of every isoband: shells and holes in GeoArrow layout.
 *
 * Composes `GPUIsobands` (band boundary edges: marching-squares level segments and cell border
 * edges that no neighbouring cell cancels) with `GPUSegmentRingAssembly` (edges chained into closed
 * rings, `ringGroups` = band, holes assigned to their innermost shell of the same band,
 * optional `polygons` regrouped as shell plus holes). Rings follow the segment direction, so shells
 * are counter-clockwise and holes clockwise; the band region lies on the left. Band boundaries are
 * bit-identical to `GPUIsolines` of the same breaks. A band region that touches itself at a
 * single vertex (a saddle whose high corners are joined) is split into separate rings when
 * `splitTouchingRings` is on. Cells with a nodata corner emit no edges, so their border with valid
 * cells becomes part of a ring.
 *
 * Changing breaks, `breakCount`, the extent or the band window never recompiles the graph. The
 * edge and ring capacities are compile-time; the `overflow` and `status.edgeOverflow` words say when they
 * were too small.
 *
 * Row direction: row-zero sample centers are half a cell above `minY` of the outer extent; rows advance
 * toward larger y. Only the width and height of `grid` are read; `grid.affine` is not, and nothing
 * is checked at construction. A north-up grid (row 0 at the top, the default of
 * `makeGPURasterMetadata`) therefore comes out mirrored north to south unless the caller reverses
 * the row order of the raster before upload. Mirroring the output instead (`y' = minY + maxY - y`)
 * also swaps the documented side: high values are then on the right of a segment, and ring
 * windings are reversed. Following the grid affine is deferred (roadmap 12.4).
 */
export class GPUIsobandRings implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUIsobandRingsProps;

  constructor(props: GPUIsobandRingsProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'grid',
      'values',
      'mask',
      'noDataValue',
      'breaks',
      'parameters',
      'edgeCapacity',
      'vertexTolerance',
      'splitTouchingRings',
      'output'
    ]);
    const grid = snapshotConstructionFields(captured.grid, [
      'width',
      'height',
      'affine',
      'pixelInterpretation',
      'coordinateReferenceSystem',
      'level',
      'levelZeroOrigin'
    ]);
    const output = snapshotConstructionFields(captured.output, [
      'ringOffsets',
      'positions',
      'ringAreas',
      'ringIsHole',
      'ringShells',
      'ringGroups',
      'segmentRings',
      'segmentVertices',
      'segmentFlags',
      'polygons',
      'status'
    ]);
    props = Object.freeze({
      ...captured,
      grid: Object.freeze({
        ...grid,
        affine: grid.affine && (Object.freeze([...grid.affine]) as GPURasterMetadata['affine']),
        levelZeroOrigin:
          grid.levelZeroOrigin &&
          (Object.freeze([...grid.levelZeroOrigin]) as readonly [number, number])
      }),
      values: 'storage' in captured.values ? snapshotRasterBand(captured.values) : captured.values,
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, [
          'count',
          'overflow',
          'requiredCount',
          'openSegmentCount',
          'touchingSegmentCount',
          'edgeCount',
          'edgeOverflow'
        ]),
        polygons:
          output.polygons &&
          snapshotConstructionFields(output.polygons, [
            'kind',
            'positions',
            'ringOffsets',
            'polygonOffsets',
            'featureOffsets',
            'sourceIds'
          ])
      })
    });
    this.id = props.id ?? 'isoband-rings';
    this.props = props;
    validateRequirement(
      this.id,
      Boolean(props.output.status.edgeOverflow),
      'output.status.edgeOverflow is required to expose intermediate edge loss'
    );
    validateSinglePackedView(
      props.output.status.edgeOverflow,
      ['uint32'],
      `${this.id} output.status.edgeOverflow`
    );
    validateCapacity(
      this.id,
      'output.status.edgeOverflow.length',
      props.output.status.edgeOverflow.length,
      1
    );
    const {edgeCount, edgeOverflow, ...ringStatus} = props.output.status;
    const finalFields = {
      ...Object.fromEntries(
        Object.entries(props.output).filter(([name]) => name !== 'status' && name !== 'polygons')
      ),
      ...Object.fromEntries(
        Object.entries(ringStatus).map(([name, view]) => [`status.${name}`, view])
      ),
      ...Object.fromEntries(
        Object.entries(props.output.polygons ?? {}).map(([name, view]) => [
          `polygons.${name}`,
          view
        ])
      )
    };
    validateDisjointWritableViews(
      this.id,
      {'output.status.edgeCount': edgeCount, 'output.status.edgeOverflow': edgeOverflow},
      {separateDispatches: true}
    );
    for (const [name, field] of Object.entries(finalFields)) {
      for (const view of getRasterPreflightOutputViews({field})) {
        validateDisjointWritableViews(
          this.id,
          {'output.status.edgeOverflow': edgeOverflow, [`output.${name}`]: view},
          {separateDispatches: true}
        );
        if (edgeCount)
          validateDisjointWritableViews(
            this.id,
            {'output.status.edgeCount': edgeCount, [`output.${name}`]: view},
            {separateDispatches: true}
          );
      }
    }
    if (!Number.isInteger(props.edgeCapacity) || props.edgeCapacity < 1) {
      throw new Error(`${this.id} edgeCapacity must be a positive integer`);
    }
  }

  /** Returns the isoband edge nodes followed by the ring assembly nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateCapacity(
      id,
      'edgeCapacity',
      props.edgeCapacity,
      1,
      getGPUIsobandRingsMaximumEdgeCapacity(graph.device)
    );
    // The ring assembly validates its own output views; this only guards the documented maximum
    // that its transient ring bounds impose.
    if (output?.ringOffsets) {
      validateMaximumCount(
        id,
        'ring capacity (output.ringOffsets.length - 1)',
        output.ringOffsets.length - 1,
        getGPUIsobandRingsMaximumRingCapacity(graph.device)
      );
    }
    const storageProps = this.props;
    validateRasterAlgebraInputBindings(graph, this.id, [storageProps.values], storageProps.grid);
    const storageRingCapacity = storageProps.output.ringOffsets.length - 1;
    const storageNeedsShells = Boolean(
      storageProps.output.ringShells || storageProps.output.polygons
    );
    const storageLeafCapacity = 2 ** Math.ceil(Math.log2(Math.max(1, storageRingCapacity)));
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        storageProps.mask,
        storageProps.breaks,
        storageProps.parameters,
        ...getRasterPreflightOutputViews(storageProps.output),
        ...getRasterPreflightOutputViews(storageProps.output.polygons ?? {})
      ],
      {
        'boundary edges': storageProps.edgeCapacity * 16,
        'cell counts and offsets':
          (storageProps.grid.width - 1) * (storageProps.grid.height - 1) * 4,
        'ring scalar planes': storageRingCapacity * 4,
        'ring counters': 24,
        'shell records': storageNeedsShells ? storageRingCapacity * RING_SHELL_RECORD_WORDS * 4 : 0,
        'shell bounds': storageNeedsShells ? storageRingCapacity * 16 : 0,
        'shell BVH nodes': storageNeedsShells ? (2 * storageLeafCapacity - 1) * 8 : 0
      }
    );

    const edges = createTransientView(graph, `${id}-edges`, 'float32x4', props.edgeCapacity);
    const edgeBands = createTransientView(graph, `${id}-edge-bands`, 'uint32', props.edgeCapacity);
    const edgeCount =
      output.status.edgeCount ?? createTransientView(graph, `${id}-edge-count`, 'uint32', 1);
    const edgeOverflow = output.status.edgeOverflow;
    const isobands = new GPUIsobands({
      id: `${id}-isobands`,
      grid: props.grid,
      values: props.values,
      mask: props.mask,
      noDataValue: props.noDataValue,
      breaks: props.breaks,
      parameters: props.parameters,
      output: {edges, edgeBands, status: {edgeCount, edgeOverflow}}
    });
    const {edgeCount: _edgeCount, edgeOverflow: _edgeOverflow, ...ringStatus} = output.status;
    const assemblyOutput = {...output, status: ringStatus};
    const ringAssembly = new GPUSegmentRingAssembly({
      id: `${id}-assembly`,
      endpoints: edges,
      count: edgeCount,
      groups: edgeBands,
      vertexTolerance: props.vertexTolerance ?? 1e-4,
      interiorSide: 'left',
      cancelOpposingSegments: true,
      splitTouchingRings: props.splitTouchingRings,
      output: assemblyOutput
    });
    return [...isobands.getCommandNodes(graph), ...ringAssembly.getCommandNodes(graph)];
  }
}
