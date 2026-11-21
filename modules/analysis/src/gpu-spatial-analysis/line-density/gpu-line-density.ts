// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterGridCorners} from '../spatial-interpolation/spatial-interpolation-parameters';
import type {
  GPULineGeometryFields,
  GPUOverflowStatusPort
} from '../../analysis-core/contracts/index';
import type {GPUSpatialParameterSchema} from '../../analysis-core/contracts/parameter-schema.js';
import {
  getGPUAuthalicSphereRadius,
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {createSaturatedScanTotalNodes} from '../../analysis-core/kernels/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  defineGPUSpatialParameterSchema,
  packGPUSpatialParameterValues,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  createDensityNode,
  createRecordPublishNode,
  createWalkCountNode,
  createWalkEmitNode,
  type LineDensityGridConstants
} from './line-density-kernels';

import {createLineDensitySeamNodes} from './line-density-seam';

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGeometrySortScratchByteLengths,
  validateGeometryScratchByteLengths
} from '../geometry-edit/geometry-storage-validation';

const OPERATION = 'GPULineDensity';
const LINE_DENSITY_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
});

/** Number of float32 elements in a {@link GPULineDensity} parameter buffer. */
export const GPU_LINE_DENSITY_PARAMETER_LENGTH = 4;

/**
 * Largest `columns * rows` of {@link GPULineDensity}: the per-cell sum runs one 256-thread
 * workgroup per cell and the dispatch holds at most `2^32 - 1` invocations, so
 * `floor((2^32 - 1) / 256)`, `2^24 - 1`.
 */
export const GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT = Math.floor(0xffffffff / 256);

/** Declarative layout of the per-encoding line-density grid. */
export const GPU_LINE_DENSITY_PARAMETER_SCHEMA: GPUSpatialParameterSchema<'float32'> =
  defineGPUSpatialParameterSchema({
    id: 'line-density',
    format: 'float32',
    wordLength: GPU_LINE_DENSITY_PARAMETER_LENGTH,
    fields: [
      {name: 'minimumX', format: 'float32', wordOffset: 0, defaultValue: 0, dynamic: true},
      {name: 'minimumY', format: 'float32', wordOffset: 1, defaultValue: 0, dynamic: true},
      {
        name: 'cellWidth',
        format: 'float32',
        wordOffset: 2,
        defaultValue: 1,
        minimum: Number.MIN_VALUE,
        units: 'position-units',
        dynamic: true
      },
      {
        name: 'cellHeight',
        format: 'float32',
        wordOffset: 3,
        defaultValue: 1,
        minimum: Number.MIN_VALUE,
        units: 'position-units',
        dynamic: true
      }
    ]
  });

/** CPU description of the per-frame grid of {@link GPULineDensity}. */
export type GPULineDensitySettings = {
  /** X of the grid's lower-left corner. */
  minX: number;
  /** Y of the grid's lower-left corner. */
  minY: number;
  /** Cell width in position units (degrees for `'spherical'`). */
  cellWidth: number;
  /** Cell height in position units (degrees for `'spherical'`). */
  cellHeight: number;
};

/**
 * Packs {@link GPULineDensitySettings} into the 4-element float32 layout
 * `[minX, minY, cellWidth, cellHeight]`. Write the result into a `GPUParameterBuffer` between
 * encodings to pan or zoom the grid without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, a cell size is not positive, encoding a minimum moves
 *   it by at least one cell, or `target` is too short. Subtract a common local origin from
 *   positions and grid placement when a minimum cannot preserve sub-cell placement.
 */
export function getGPULineDensityParameterValues(
  settings: GPULineDensitySettings,
  target?: Float32Array
): Float32Array;
/**
 * Packs an axis-aligned raster using the existing minimum/positive-cell-size schema.
 * Also spread getGPULineDensityRasterOutputMapping(grid) into the contributor to map its
 * output rows/columns to raster storage order. Point pixels shift the corner by half a cell.
 */
export function getGPULineDensityParameterValues(
  grid: GPURasterMetadata,
  target?: Float32Array
): Float32Array;
export function getGPULineDensityParameterValues(
  settings: GPULineDensitySettings | GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_LINE_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
    throw new Error(`Line density target must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} elements`);
  }
  if ('affine' in settings) {
    validateRequirement(
      'line-density',
      ['area', 'point'].includes(settings.pixelInterpretation),
      `grid.pixelInterpretation must be area or point; received ${settings.pixelInterpretation}`
    );
    // This layout stores steps directly; it does not derive them from float32 corners.
    const [startX, startY, endX, endY] = getRasterGridCorners('line-density', settings, false);
    return getGPULineDensityParameterValues(
      {
        minX: Math.min(startX, endX),
        minY: Math.min(startY, endY),
        cellWidth: Math.abs(settings.affine[0]),
        cellHeight: Math.abs(settings.affine[4])
      },
      target
    );
  }

  const values = packGPUSpatialParameterValues(GPU_LINE_DENSITY_PARAMETER_SCHEMA, {
    minimumX: settings.minX,
    minimumY: settings.minY,
    cellWidth: settings.cellWidth,
    cellHeight: settings.cellHeight
  });
  for (const [minimumName, minimum, encodedMinimum, cellSizeName, cellSize, encodedCellSize] of [
    ['minX', settings.minX, values[0], 'cellWidth', settings.cellWidth, values[2]],
    ['minY', settings.minY, values[1], 'cellHeight', settings.cellHeight, values[3]]
  ] as const) {
    const displacement = Math.abs(encodedMinimum - minimum);
    validateRequirement(
      'line-density',
      displacement < Math.min(cellSize, encodedCellSize),
      `${minimumName} received ${minimum}, encoded as ${encodedMinimum}, moves by ${displacement}; displacement must be less than one ${cellSizeName} (${cellSize}, encoded ${encodedCellSize}); subtract a common local origin from positions and grid placement`
    );
  }
  target.set(values);
  return target;
}

/** Explicit permutation from physical south/west-first cells to raster storage rows/columns. */
export type GPULineDensityRasterOutputMapping = {
  outputRowOrder: 'south-first' | 'north-first';
  outputColumnOrder: 'west-first' | 'east-first';
};
/** Pair with the metadata parameter overload and matching columns/rows; does not alter geometry. */
export function getGPULineDensityRasterOutputMapping(
  grid: GPURasterMetadata
): GPULineDensityRasterOutputMapping {
  validateRequirement(
    'line-density',
    ['area', 'point'].includes(grid.pixelInterpretation),
    `grid.pixelInterpretation must be area or point; received ${grid.pixelInterpretation}`
  );
  // Use the same origin/positive-step validation as the raster parameter overload.
  getGPULineDensityParameterValues(grid);
  return {
    outputRowOrder: grid.affine[4] < 0 ? 'north-first' : 'south-first',
    outputColumnOrder: grid.affine[0] < 0 ? 'east-first' : 'west-first'
  };
}

/** Caller-owned outputs of {@link GPULineDensity}. */
export type GPULineDensityOutput = {
  /**
   * Total line length inside each cell, `columns * rows` rows in row-major order (cell
   * `row * columns + column`), in the context's length unit (position units for planar,
   * meters by default for longitude/latitude).
   */
  lengths: GraphDataView<'float32'>;
  /**
   * Optional length divided by cell area, in the context's length unit per square length unit,
   * same layout. Spherical cells use the exact spherical area with the authalic radius by default.
   */
  densities?: GraphDataView<'float32'>;
  /**
   * One-row status scalars: `overflow` receives `1` when the piece capacity overflowed (lengths
   * are then low), else `0`, and the optional `requiredCount` receives the unclamped number of
   * segment-cell pieces.
   */
  status: GPUOverflowStatusPort & {
    requiredCount?: GraphDataView<'uint32'>;
    /** Optional one-row count of geographic segments with absolute longitude difference > 180 degrees.
     * These retain the literal longitude/latitude walk; the count resets on every encoding.
     */
    antimeridianUncertainCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPULineDensity}.
 *
 * Per-frame (no recompile): the grid origin and cell size in `parameters`, and every input
 * buffer. Compile-time: view lengths, `columns`, `rows`, `spatialContext`,
 * `recordCapacity`, and which outputs are present.
 */
export type GPULineDensityProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-density'`. */
  id?: string;
  /**
   * Native planar when omitted. A `longitude-latitude` context with metric `'great-circle'`
   * measures lengths in the context's length unit and areas in its square (meters by default).
   */
  spatialContext?: GPUSpatialContext;
  /** Grid columns (compile time). `columns * rows` is at most {@link GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT}. */
  columns: number;
  /** Grid rows (compile time). `columns * rows` is at most {@link GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT}. */
  rows: number;
  /**
   * Compile-time capacity of segment-cell pieces. A segment crossing `k` grid lines produces
   * `k + 1` pieces. Default `max(1024, 4 * positions.length)`. When exceeded, `status.overflow` is set
   * and the cells fed by the dropped pieces (the last segments in row order) are low.
   */
  recordCapacity?: number;
  /** Per-frame packed float32 view written with {@link getGPULineDensityParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Storage order only; cell-boundary ownership remains physical north/east. Default south-first. */
  outputRowOrder?: 'south-first' | 'north-first';
  /** Storage order only; default west-first. */
  outputColumnOrder?: 'west-first' | 'east-first';
  /** Output columns. */
  output: GPULineDensityOutput;
};

/**
 * Line length per grid cell, and that length divided by the cell area (QGIS "Sum line lengths"
 * per grid cell, PostGIS `ST_Length(ST_Intersection(line, cell))` summed). This is not the QGIS or
 * ArcGIS "Line density" tool, which spreads each line over a circular search radius.
 *
 * Every segment is clipped to the grid with Liang-Barsky and walked through the cells it crosses
 * (Amanatides-Woo with integer step counts), emitting one `(cell, length)` piece per crossed
 * cell; in planar coordinates pieces sum to the clipped length exactly up to f32 rounding. Pieces are counted, scanned
 * into slots and emitted (the walk code is shared, so count and emit agree), then reduced per
 * cell by a stable sort and a fixed-order segmented sum, so lengths are bitwise reproducible. No
 * float atomics. A segment lying exactly along a cell boundary counts in the cell above or to the
 * right, and one on the grid's upper or right edge in the last row or column. Segments outside the
 * grid contribute nothing; consecutive lines are not joined, and only segments inside a line are
 * counted (single-vertex lines have length 0).
 *
 * Original settings write south-first rows (row zero has the smallest Y). Raster-metadata
 * parameters keep that physical grid; getGPULineDensityRasterOutputMapping chooses the explicit
 * output permutation to match the raster. Cell-boundary ownership stays north/east. To overlay
 * an original-settings result on a north-up raster, read source row `rows - 1 - rasterRow`.
 *
 * For per-polygon line length (QGIS "Sum line lengths") use {@link GPULineLengthPerPolygon}.
 *
 * Longitude-latitude context. A segment is the straight line between its two vertices in
 * longitude and latitude (it is cut by the grid along that line, as
 * {@link GPULineLengthPerPolygon} cuts it), and each piece is measured as the great-circle
 * distance between the piece's endpoints. The pieces of a long segment therefore do not add up to
 * the great-circle length of the segment, and their total depends on the grid: the segment from
 * (0, 60) to (90, 60) is 4,605 km as one piece and about 5,004 km (the length along the parallel)
 * when cut by 1 degree cells. For segments short against the grid the difference vanishes, so
 * densify long segments along the path they are meant to follow before passing them in.
 * A longitude jump greater than 180 degrees retains this literal route (179 to -179 walks 358
 * degrees). Optional `status.antimeridianUncertainCount` counts these segments per encoding,
 * including segments outside the grid; split or unwrap seam crossings before use if needed.
 *
 * Precision: f32 grid coordinates relative to the grid origin; piece lengths come from f32
 * parameters along the segment (relative error about `1e-6` of the segment length).
 */
export class GPULineDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineDensityProps;
  /** Canonical coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of grid cells. */
  readonly cellCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: 'planar' | 'spherical';
  /** Mean sphere radius for lengths, in the context's length unit; 1 for planar. */
  readonly radius: number;
  /** Authalic sphere radius for cell areas, in the context's length unit; 1 for planar. */
  readonly areaRadius: number;
  /** Resolved piece capacity. */
  readonly recordCapacity: number;

  constructor(props: GPULineDensityProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'positions',
      'lineOffsets',
      'spatialContext',
      'columns',
      'rows',
      'recordCapacity',
      'parameters',
      'outputRowOrder',
      'outputColumnOrder',
      'output'
    ]);
    const output = snapshotConstructionFields(captured.output, ['lengths', 'densities', 'status']);
    props = Object.freeze({
      ...captured,
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, [
          'overflow',
          'requiredCount',
          'antimeridianUncertainCount'
        ])
      })
    });
    validateRequirement(
      props.id ?? 'line-density',
      props.outputRowOrder === undefined ||
        ['south-first', 'north-first'].includes(props.outputRowOrder),
      `outputRowOrder must be south-first or north-first; received ${props.outputRowOrder}`
    );
    validateRequirement(
      props.id ?? 'line-density',
      props.outputColumnOrder === undefined ||
        ['west-first', 'east-first'].includes(props.outputColumnOrder),
      `outputColumnOrder must be west-first or east-first; received ${props.outputColumnOrder}`
    );
    this.id = props.id ?? 'line-density';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      LINE_DENSITY_SUPPORT
    );
    const spherical = this.spatialContext.coordinateSpace === 'longitude-latitude';
    this.coordinateSystem = spherical ? 'spherical' : 'planar';
    this.radius = spherical ? getGPUSphereRadius(this.spatialContext) : 1;
    this.areaRadius = spherical ? getGPUAuthalicSphereRadius(this.spatialContext) : 1;
    this.recordCapacity = props.recordCapacity ?? Math.max(1024, 4 * props.positions.length);
    const {id} = this;
    validateCapacity(id, 'columns', props.columns);
    validateCapacity(id, 'rows', props.rows);
    this.cellCount = props.columns * props.rows;
    validateMaximumCount(id, 'columns x rows', this.cellCount, GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT);
    validateCapacity(id, 'recordCapacity', this.recordCapacity);
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs two positions`);
    }
    validateSinglePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} float32 values`
      );
    }
    const {lengths, densities, status} = props.output;
    const {overflow, requiredCount, antimeridianUncertainCount} = status;
    validateSinglePackedView(lengths, ['float32'], `${id} output.lengths`);
    if (lengths.length !== this.cellCount) {
      throw new Error(`${id} output.lengths must hold ${this.cellCount} rows`);
    }
    if (densities) {
      validateSinglePackedView(densities, ['float32'], `${id} output.densities`);
      if (densities.length !== this.cellCount) {
        throw new Error(`${id} output.densities must hold ${this.cellCount} rows`);
      }
    }
    for (const [name, scalar] of [
      ['overflow', overflow],
      ['requiredCount', requiredCount],
      ['antimeridianUncertainCount', antimeridianUncertainCount]
    ] as const) {
      if (scalar) {
        validateSinglePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one uint32 row`);
        }
      }
    }
    if (antimeridianUncertainCount && antimeridianUncertainCount.length !== 1) {
      throw new Error(
        `${id} output.status.antimeridianUncertainCount must contain exactly one uint32 row`
      );
    }
    validateDisjointWritableViews(
      id,
      {lengths, densities, overflow, requiredCount, antimeridianUncertainCount},
      {separateDispatches: true}
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [lengths, densities, overflow, requiredCount, antimeridianUncertainCount],
      [props.positions, props.lineOffsets, props.parameters]
    );
  }

  /**
   * Returns the count, scan, key fill, emit, publish, cell count, sorted sum and optional density
   * nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, cellCount, recordCapacity} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          props.positions,
          props.lineOffsets,
          props.parameters,
          output.lengths,
          output.densities,
          output.status.overflow,
          output.status.requiredCount,
          output.status.antimeridianUncertainCount
        ].map((view, index) => [String(index), view])
      )
    );
    validateGeometryScratchByteLengths(graph, id, {
      rowCountsAndScan: props.positions.length * 4,
      ...getGeometrySortScratchByteLengths(recordCapacity),
      cellOffsets: (cellCount + 1) * 4
    });
    const rowCount = props.positions.length;
    const constants: LineDensityGridConstants = {
      positions: props.positions,
      lineOffsets: props.lineOffsets,
      columns: props.columns,
      rows: props.rows,
      coordinateSystem: this.coordinateSystem,
      radius: this.radius,
      capacity: recordCapacity,
      outputRowOrder: props.outputRowOrder,
      outputColumnOrder: props.outputColumnOrder
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', recordCapacity);
    const contributions = createTransientView(
      graph,
      `${id}-contributions`,
      'float32',
      recordCapacity
    );
    return [
      ...(output.status.antimeridianUncertainCount
        ? createLineDensitySeamNodes(graph, {
            id,
            constants,
            output: output.status.antimeridianUncertainCount
          })
        : []),
      createWalkCountNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        constants,
        parameters: props.parameters,
        counts
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: id,
        operation: OPERATION,
        counts: counts,
        offsets: starts,
        rowCount: rowCount,
        total: scanTotal
      }),
      // Unwritten slots keep the out-of-range key, which sorts after every cell.
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-keys`,
        operation: OPERATION,
        view: keys,
        type: 'u32',
        value: `${cellCount}u`
      }),
      createWalkEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        constants,
        parameters: props.parameters,
        starts,
        keys,
        contributions
      }),
      createRecordPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity: recordCapacity,
        total: scanTotal,
        overflow: output.status.overflow,
        totalRecords: output.status.requiredCount
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-sum`,
        operation: OPERATION,
        segmentCount: cellCount,
        segmentKeys: keys,
        sumContributions: contributions,
        sums: output.lengths
      }),
      ...(output.densities
        ? [
            createDensityNode<Parameters>(graph, {
              id: `${id}-density`,
              operation: OPERATION,
              columns: props.columns,
              rows: props.rows,
              outputRowOrder: props.outputRowOrder,
              coordinateSystem: this.coordinateSystem,
              areaRadius: this.areaRadius,
              parameters: props.parameters,
              lengths: output.lengths,
              densities: output.densities
            })
          ]
        : [])
    ];
  }
}
