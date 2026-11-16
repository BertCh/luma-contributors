// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {WEB_MERCATOR_EARTH_RADIUS} from '@luma.gl/graph/gpu-project';
import {SCALED_PLANAR_DISTANCE_WGSL} from '../../analysis-core/kernels/scaled-planar-distance-wgsl';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPURasterContext,
  getRasterGroundScale,
  getRasterGroundScaleWGSL,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {validateRasterGridAxisAligned, validateRasterParameters} from '../raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterRowEdgeWGSL} from '../raster-grid/raster-cell-area-scale';
import {
  createPatchKeysNode,
  createPatchUnresolvedNodes,
  getPatchLabelViews,
  getPatchRunSegmentsBody,
  getPatchSegmentCount,
  validatePatchLabels,
  type GPURasterPatchLabels
} from './patch-labels';

const OPERATION = 'GPURasterPatchMetrics';
const NO_ROW = '0xffffffffu';

/** Float32 layout shared with fractional coverage: `[originX, originY, a, b, d, e, determinant, 0]`. */
export const GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH = 8;

/**
 * Packs per-frame affine steps, area and row latitude for GPURasterPatchMetrics.
 * Width, height, pixel interpretation and cellSizeMode remain fixed in the compiled graph.
 *
 * @param grid Raster grid whose affine will be measured.
 * @param target Optional destination of at least eight float32 values.
 */
export function getGPURasterPatchMetricsParameterValues(
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterPatchMetricsParameterValues';
  validateRequirement(
    id,
    target.length >= GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH,
    `target must contain at least ${GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH} values`
  );
  target.set(resolveGPURasterContext(id, {grid}).parameterValues);
  return target;
}

/**
 * Caller-owned per-patch columns of {@link GPURasterPatchMetrics}. Row `r` describes dense label
 * `r + 1`. Every provided column must have the same length (the patch capacity). Provide only the
 * columns you need; at least one is required.
 */
export type GPURasterPatchMetricsOutput = {
  /** Pixel count of each patch (area in pixels). 0 marks an unused row. */
  pixelCounts?: GraphDataView<'uint32'>;
  /**
   * Pixel count times the absolute determinant of the grid affine, in CRS units squared. With a
   * non-uniform `cellSizeMode`, ground square meters: each pixel counts the cell area times the
   * WGS84 ground area per world area of its row.
   */
  areas?: GraphDataView<'float32'>;
  /**
   * Number of exposed pixel faces: faces whose 4-neighbor holds another label, background, an
   * invalid pixel, or (with `countRasterBorder`) lies outside the raster. Exact integer.
   */
  perimeterFaceCounts?: GraphDataView<'uint32'>;
  /**
   * Perimeter length in CRS units: faces between vertically adjacent pixels have the length of one
   * column step of the affine, faces between horizontally adjacent pixels the length of one row
   * step. Holes contribute their inner boundary. With a non-uniform `cellSizeMode`, ground
   * meters: a face counts the step length times the WGS84 ground scale along it at the row of the
   * exposed pixel.
   */
  perimeters?: GraphDataView<'float32'>;
  /** Inclusive minimum pixel column of each patch (0 for an empty row). */
  minColumns?: GraphDataView<'uint32'>;
  /** Inclusive minimum pixel row of each patch (0 for an empty row). */
  minRows?: GraphDataView<'uint32'>;
  /** Inclusive maximum pixel column of each patch (0 for an empty row). */
  maxColumns?: GraphDataView<'uint32'>;
  /** Inclusive maximum pixel row of each patch (0 for an empty row). */
  maxRows?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterPatchMetrics}.
 *
 * Per-frame: label contents and affine through `parameters`. Compile-time: the `grid` shape,
 * pixel interpretation and cellSizeMode,
 * `countRasterBorder`, the output capacity and which columns are present.
 */
export type GPURasterPatchMetricsProps = GPURasterPatchLabels & {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-patch-metrics'`. */
  id?: string;
  /** Whether faces on the raster edge count as perimeter. Default `true`. */
  countRasterBorder?: boolean;
  /**
   * What the affine's units are, for `areas` and `perimeters`. Default `'uniform'`: coordinate
   * units, as before. `'geographic'` (degrees) and `'web-mercator'` (EPSG:3857 meters) give ground
   * square meters and meters on the WGS84 ellipsoid, with the scale taken at each row center (row
   * edges from `grid.affine`). They need an axis-aligned grid. The weighted sums are 64-bit
   * fixed-point integer atomics, so they stay deterministic and independent of the order of the
   * pixels; the sums are exact, but each row's weight is rounded to a multiple of 2^-26 of the
   * largest ground scale the model can reach (the equator for a row's x scale, the pole for its y
   * scale; a fixed constant of the mode). The relative error of a row weight is at most
   * 2^-27 over its weight: about 1e-7 at 60 degrees latitude, about 1e-5 at 89.95 degrees.
   * Compile-time mode; see `parameters` for the row edges.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Per-frame affine packed by getGPURasterPatchMetricsParameterValues (also accepts the
   * identical fractional-coverage layout). All modes read the determinant and both axis lengths;
   * ground modes also read row latitude. A repacked grid must remain axis-aligned in ground
   * modes. Without this view the initial grid affine stays fixed for backwards compatibility.
   */
  parameters?: GraphDataView<'float32'>;
  /** Caller-owned per-patch output columns. */
  output: GPURasterPatchMetricsOutput;
};

// GPURasterRegionMeasurements takes the same dense labels with the same convergence and overflow guards, but yields only pixel counts, intensity statistics, centroids and areas, with no perimeter or bounding-box columns. Here, counts use integer accumulation and ground areas use fixed-point accumulation, independent of summation order; area outputs are float32, with rounding and ground-weight quantization.
/**
 * Per-patch metrics of a dense label raster: area, perimeter and bounding box
 * (the `r.clump` / FRAGSTATS companions to `GPURasterConnectedComponents`).
 *
 * Intensity statistics and centroids stay in `GPURasterRegionMeasurements`; this contributor adds
 * what it lacks, perimeter and extent, and repeats the pixel count and area so one node set is
 * enough for shape ratios. Labels are the output of `GPURasterDenseComponents`; labels above the
 * output capacity are ignored.
 *
 * Unfinished labels are not reported as "no patches": when the `converged` guard reads 0 or the
 * `overflow` guard is nonzero, every provided `uint32` column is filled with
 * {@link GPU_RASTER_PATCH_UNRESOLVED} and every `float32` column with NaN (an extra gated pass, so
 * a finished run pays one tiny dispatch). A finished run never produces either value.
 *
 * Counts, extents and perimeter faces are integer atomics, so results are exact and
 * deterministic; `perimeters` and `areas` are single f32 products of exact integers.
 */
export class GPURasterPatchMetrics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterPatchMetricsProps;
  /** Number of patch rows. */
  readonly patchCapacity: number;

  constructor(props: GPURasterPatchMetricsProps) {
    const id = props.id ?? 'raster-patch-metrics';
    this.id = id;
    this.props = props;
    validatePatchLabels(id, props);
    if (props.parameters) {
      resolveGPURasterContext(id, {grid: props.grid, cellSizeMode: props.cellSizeMode});
      validateRasterParameters(id, props.parameters, GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH);
    } else {
      // Legacy integer metrics never pack the affine; retain their accepted finite grids.
      validateRasterCellSizeMode(id, props.cellSizeMode);
      validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    }
    const columns = Object.entries(props.output).filter(([, view]) => view !== undefined) as [
      string,
      GraphDataView<'uint32'> | GraphDataView<'float32'>
    ][];
    if (columns.length === 0) {
      throw new Error(`${id} output must provide at least one column`);
    }
    this.patchCapacity = columns[0][1].length;
    for (const [name, view] of columns) {
      if (name === 'areas' || name === 'perimeters') {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
      } else {
        validateSinglePackedView(
          view as GraphDataView<'uint32'>,
          ['uint32'],
          `${id} output.${name}`
        );
      }
      if (view.length !== this.patchCapacity) {
        throw new Error(`${id} output columns must have identical lengths`);
      }
    }
    if (this.patchCapacity < 1) {
      throw new Error(`${id} output columns must hold at least one patch row`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      columns.map(([, view]) => view),
      [...getPatchLabelViews(props), ...(props.parameters ? [props.parameters] : [])]
    );
  }

  /** Returns the key, accumulation and finalize nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    const storageCellCount = storageProps.grid.width * storageProps.grid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...getPatchLabelViews(storageProps),
        storageProps.output.areas || storageProps.output.perimeters
          ? storageProps.parameters
          : undefined,
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'patch keys': storageCellCount * 4,
        'patch scalar planes': this.patchCapacity * 4,
        'patch guard state':
          storageProps.status?.converged || storageProps.status?.overflow ? 16 : 0
      }
    );

    const {id, props, patchCapacity} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getPatchLabelViews(props),
      ...(props.parameters ? [props.parameters] : []),
      ...(Object.values(output).filter(view => view !== undefined) as GraphDataView[])
    ]);
    const countBorder = props.countRasterBorder ?? true;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const {node: keysNode, keys} = createPatchKeysNode(graph, id, OPERATION, props, patchCapacity);
    const nodes: GPUCommandNode<Parameters>[] = [keysNode];
    const pixelCounts = output.pixelCounts ?? view('pixel-counts', patchCapacity);
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const needsGeometry = Boolean(output.areas || output.perimeters);
    const weighted =
      cellSizeMode === 'uniform' || !needsGeometry
        ? undefined
        : getGroundWeights(props, cellSizeMode);
    const geometry = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;`;
    const metricGeometry = needsGeometry ? `${geometry}\n${getPatchGeometryWGSL(props)}` : geometry;
    // Only the kernels that bind the georeference (the ones that call getRowWeights) declare it.
    const weightDeclarations = weighted ? `\n${weighted.wgsl}` : '';
    const georeferenceBindings: WGSLKernelBinding[] = props.parameters
      ? [{name: 'georeference', view: props.parameters, type: 'f32', access: 'read'}]
      : [];

    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: pixelCounts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'pixelCounts', view: pixelCounts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: getPatchSegmentCount(width, height),
        declarations: geometry,
        body: getPatchRunSegmentsBody(width, {
          onRun: 'atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);'
        })
      })
    );

    const hasExtent = output.minColumns || output.minRows || output.maxColumns || output.maxRows;
    if (hasExtent) {
      const extents = {
        minColumns: output.minColumns ?? view('min-columns', patchCapacity),
        minRows: output.minRows ?? view('min-rows', patchCapacity),
        maxColumns: output.maxColumns ?? view('max-columns', patchCapacity),
        maxRows: output.maxRows ?? view('max-rows', patchCapacity)
      };
      for (const name of ['minColumns', 'minRows'] as const) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-clear`,
            operation: OPERATION,
            view: extents[name],
            type: 'u32',
            value: NO_ROW
          })
        );
      }
      for (const name of ['maxColumns', 'maxRows'] as const) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-clear`,
            operation: OPERATION,
            view: extents[name],
            type: 'u32',
            value: '0u'
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extents`,
          operation: OPERATION,
          variant: 'extents',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {
              name: 'minColumns',
              view: extents.minColumns,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {name: 'minRows', view: extents.minRows, type: 'atomic<u32>', access: 'read_write'},
            {
              name: 'maxColumns',
              view: extents.maxColumns,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {name: 'maxRows', view: extents.maxRows, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: getPatchSegmentCount(width, height),
          declarations: geometry,
          body: getPatchRunSegmentsBody(width, {
            onRun: `atomicMin(&minColumns[minColumnsOffset + runKey - 1u], runStart);
        atomicMin(&minRows[minRowsOffset + runKey - 1u], row);
        atomicMax(&maxColumns[maxColumnsOffset + runKey - 1u], runEnd - 1u);
        atomicMax(&maxRows[maxRowsOffset + runKey - 1u], row);`
          })
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extents-empty`,
          operation: OPERATION,
          variant: 'extents-empty',
          bindings: [
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'minColumns', view: extents.minColumns, type: 'u32', access: 'read_write'},
            {name: 'minRows', view: extents.minRows, type: 'u32', access: 'read_write'}
          ],
          invocationCount: patchCapacity,
          body: `if (pixelCounts[pixelCountsOffset + index] == 0u) {
    minColumns[minColumnsOffset + index] = 0u;
    minRows[minRowsOffset + index] = 0u;
  }`
        })
      );
    }

    if (output.perimeterFaceCounts || output.perimeters) {
      const acrossRows = view('faces-across-rows', patchCapacity);
      const acrossColumns = view('faces-across-columns', patchCapacity);
      // Non-uniform modes also sum the ground length of the faces as a 64-bit fixed point counter.
      const groundPerimeter =
        weighted && output.perimeters
          ? {
              low: view('perimeter-low', patchCapacity),
              high: view('perimeter-high', patchCapacity)
            }
          : undefined;
      if (groundPerimeter) {
        for (const [name, counter] of [
          ['perimeter-low', groundPerimeter.low],
          ['perimeter-high', groundPerimeter.high]
        ] as const) {
          nodes.push(
            createFillNode<Parameters>(graph, {
              id: `${id}-${name}-clear`,
              operation: OPERATION,
              view: counter,
              type: 'u32',
              value: '0u'
            })
          );
        }
      }
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-faces-rows-clear`,
          operation: OPERATION,
          view: acrossRows,
          type: 'u32',
          value: '0u'
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-faces-columns-clear`,
          operation: OPERATION,
          view: acrossColumns,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-faces`,
          operation: OPERATION,
          variant: 'faces',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'acrossRows', view: acrossRows, type: 'atomic<u32>', access: 'read_write'},
            {name: 'acrossColumns', view: acrossColumns, type: 'atomic<u32>', access: 'read_write'},
            ...(groundPerimeter
              ? [
                  ...georeferenceBindings,
                  {
                    name: 'perimeterLow',
                    view: groundPerimeter.low,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  },
                  {
                    name: 'perimeterHigh',
                    view: groundPerimeter.high,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: getPatchSegmentCount(width, height),
          declarations: `${geometry}${groundPerimeter ? `\n${getPatchGeometryWGSL(props)}${weightDeclarations}` : ''}
const COUNT_BORDER: bool = ${countBorder};

// Whether the neighbor at (column + dx, row + dy) differs from \`key\`.
fn isExposed(column: u32, row: u32, dx: i32, dy: i32, key: u32) -> bool {
  let neighborColumn = i32(column) + dx;
  let neighborRow = i32(row) + dy;
  if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
    return COUNT_BORDER;
  }
  return keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)] != key;
}`,
          // Exposed faces fold per run so the per-pixel loop only accumulates locally and each run
          // issues at most two atomics.
          body: getPatchRunSegmentsBody(width, {
            declareRunState: 'var runRowFaces = 0u;\n  var runColumnFaces = 0u;',
            onRunStart: 'runRowFaces = 0u;\n      runColumnFaces = 0u;',
            onPixel: `if (isExposed(column, row, 0, -1, key)) { runRowFaces++; }
      if (isExposed(column, row, 0, 1, key)) { runRowFaces++; }
      if (isExposed(column, row, -1, 0, key)) { runColumnFaces++; }
      if (isExposed(column, row, 1, 0, key)) { runColumnFaces++; }`,
            onRun: `if (runRowFaces > 0u) { atomicAdd(&acrossRows[acrossRowsOffset + runKey - 1u], runRowFaces); }
        if (runColumnFaces > 0u) { atomicAdd(&acrossColumns[acrossColumnsOffset + runKey - 1u], runColumnFaces); }${
          groundPerimeter
            ? `
        let rowWeights = getRowWeights(row);
        if (runRowFaces > 0u) ${getAddFixedWGSL('perimeterLow', 'perimeterHigh', 'runRowFaces * rowWeights.y')}
        if (runColumnFaces > 0u) ${getAddFixedWGSL('perimeterLow', 'perimeterHigh', 'runColumnFaces * rowWeights.z')}`
            : ''
        }`
          })
        })
      );
      const finalizeBindings: WGSLKernelBinding[] = [
        {name: 'acrossRows', view: acrossRows, type: 'u32', access: 'read'},
        {name: 'acrossColumns', view: acrossColumns, type: 'u32', access: 'read'}
      ];
      if (groundPerimeter) {
        finalizeBindings.push(
          {name: 'perimeterLow', view: groundPerimeter.low, type: 'u32', access: 'read'},
          {name: 'perimeterHigh', view: groundPerimeter.high, type: 'u32', access: 'read'}
        );
      }
      if (output.perimeterFaceCounts) {
        finalizeBindings.push({
          name: 'faceCounts',
          view: output.perimeterFaceCounts,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (output.perimeters) {
        finalizeBindings.push(...georeferenceBindings);
        finalizeBindings.push({
          name: 'perimeters',
          view: output.perimeters,
          type: 'f32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-perimeters`,
          operation: OPERATION,
          variant: 'perimeters',
          bindings: finalizeBindings,
          invocationCount: patchCapacity,
          declarations: output.perimeters
            ? `${metricGeometry}${weighted ? weightDeclarations : ''}`
            : '',
          body: `let rowFaces = acrossRows[acrossRowsOffset + index];
  let columnFaces = acrossColumns[acrossColumnsOffset + index];
  ${output.perimeterFaceCounts ? 'faceCounts[faceCountsOffset + index] = rowFaces + columnFaces;' : ''}
  ${
    output.perimeters
      ? groundPerimeter && weighted
        ? `perimeters[perimetersOffset + index] = ${getFixedValueWGSL('perimeterLow', 'perimeterHigh', 'index')} * getPerimeterReference();`
        : `perimeters[perimetersOffset + index] = f32(rowFaces) * getColumnStep() + f32(columnFaces) * getRowStep();`
      : ''
  }`
        })
      );
    }

    if (output.areas && weighted) {
      const areaLow = view('area-low', patchCapacity);
      const areaHigh = view('area-high', patchCapacity);
      for (const [name, counter] of [
        ['area-low', areaLow],
        ['area-high', areaHigh]
      ] as const) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-clear`,
            operation: OPERATION,
            view: counter,
            type: 'u32',
            value: '0u'
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-ground-areas`,
          operation: OPERATION,
          variant: 'ground-areas',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'areaLow', view: areaLow, type: 'atomic<u32>', access: 'read_write'},
            {name: 'areaHigh', view: areaHigh, type: 'atomic<u32>', access: 'read_write'},
            ...georeferenceBindings
          ],
          invocationCount: getPatchSegmentCount(width, height),
          declarations: `${metricGeometry}${weightDeclarations}`,
          body: getPatchRunSegmentsBody(width, {
            onRun: getAddFixedWGSL(
              'areaLow',
              'areaHigh',
              '(runEnd - runStart) * getRowWeights(row).x'
            )
          })
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-areas`,
          operation: OPERATION,
          variant: 'areas',
          bindings: [
            {name: 'areaLow', view: areaLow, type: 'u32', access: 'read'},
            {name: 'areaHigh', view: areaHigh, type: 'u32', access: 'read'},
            {name: 'areas', view: output.areas, type: 'f32', access: 'read_write'},
            ...georeferenceBindings
          ],
          invocationCount: patchCapacity,
          declarations: metricGeometry,
          body: `areas[areasOffset + index] = ${getFixedValueWGSL('areaLow', 'areaHigh', 'index')} * ${getWGSLFloatLiteral(weighted.areaReference)} * getPixelArea();`
        })
      );
    } else if (output.areas) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-areas`,
          operation: OPERATION,
          variant: 'areas',
          bindings: [
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'areas', view: output.areas, type: 'f32', access: 'read_write'},
            ...georeferenceBindings
          ],
          invocationCount: patchCapacity,
          declarations: metricGeometry,
          body: `areas[areasOffset + index] = f32(pixelCounts[pixelCountsOffset + index]) * getPixelArea();`
        })
      );
    }
    nodes.push(
      ...createPatchUnresolvedNodes(
        graph,
        id,
        OPERATION,
        props,
        (Object.entries(output) as [string, GraphDataView<'uint32'> | GraphDataView<'float32'>][])
          .filter(([, column]) => column !== undefined)
          .map(([name, column]) => ({
            view: column,
            type: name === 'areas' || name === 'perimeters' ? ('f32' as const) : ('u32' as const)
          }))
      )
    );
    return nodes;
  }
}

/** Fixed-point scale of the ground weights: the largest possible row weight is 2^26. */
const FIXED_POINT_SCALE = 67108864;

/** Largest ground scales `[x, y, x * y]` of a mode over the whole latitude range, with 0.1% margin. */
const groundScaleBounds = new Map<string, readonly [number, number, number]>();

function getGroundScaleBounds(
  mode: Exclude<GPURasterCellSizeMode, 'uniform'>
): readonly [number, number, number] {
  let bounds = groundScaleBounds.get(mode);
  if (!bounds) {
    // Latitude sweep in 0.005 degree steps; the edge is latitude degrees, or EPSG:3857 meters at
    // the same latitudes. The x scale peaks at the equator, the y scale at the pole.
    let maximumX = 0;
    let maximumY = 0;
    let maximumArea = 0;
    for (let step = 0; step <= 36000; step++) {
      const latitude = Math.min(90 - step * 0.005, 89.9999999);
      for (const sign of [-1, 1]) {
        const signedLatitude = latitude * sign;
        const edge =
          mode === 'geographic'
            ? signedLatitude
            : WEB_MERCATOR_EARTH_RADIUS * Math.atanh(Math.sin((signedLatitude * Math.PI) / 180));
        const [scaleX, scaleY] = getRasterGroundScale(mode, edge);
        maximumX = Math.max(maximumX, scaleX);
        maximumY = Math.max(maximumY, scaleY);
        maximumArea = Math.max(maximumArea, scaleX * scaleY);
      }
    }
    bounds = [maximumX * 1.001, maximumY * 1.001, maximumArea * 1.001];
    groundScaleBounds.set(mode, bounds);
  }
  return bounds;
}

/**
 * Per-row ground weights for a non-uniform `cellSizeMode`: the WGSL `getRowWeights(row)` that
 * returns `vec3<u32>(area, acrossRows, acrossColumns)` as 2^-26 fixed point fractions of fixed
 * references: areaReference is a constant bound for the mode; the perimeter reference multiplies
 * the mode bounds by the current affine step lengths. The row edges are read
 * from the `georeference` binding when `props.parameters` is given.
 */
function getGroundWeights(
  props: GPURasterPatchMetricsProps,
  mode: Exclude<GPURasterCellSizeMode, 'uniform'>
): {wgsl: string; areaReference: number} {
  const [maximumScaleX, maximumScaleY, maximumArea] = getGroundScaleBounds(mode);
  const areaReference = maximumArea;
  const wgsl = `const AREA_REFERENCE: f32 = ${getWGSLFloatLiteral(areaReference)};
fn getPerimeterReference() -> f32 {
  return max(${getWGSLFloatLiteral(maximumScaleX)} * getColumnStep(), ${getWGSLFloatLiteral(maximumScaleY)} * getRowStep());
}

// Ground weights of one row as 2^-26 fixed point fractions of the largest possible row value.
fn getRowWeights(row: u32) -> vec3<u32> {
  ${getRasterRowEdgeWGSL(props.grid, props.parameters ? 'georeference' : undefined)}
  ${getRasterGroundScaleWGSL(mode)}
  let weights = min(
    vec3<f32>(
      groundScale.x * groundScale.y / AREA_REFERENCE,
      groundScale.x * getColumnStep() / getPerimeterReference(),
      groundScale.y * getRowStep() / getPerimeterReference()
    ),
    vec3<f32>(1.0)
  );
  return vec3<u32>(round(weights * ${getWGSLFloatLiteral(FIXED_POINT_SCALE)}));
}`;
  return {wgsl, areaReference};
}

/** WGSL statement adding \`expression\` to the 64-bit counter (\`low\`, \`high\`) of the current run's patch. */
function getAddFixedWGSL(low: string, high: string, expression: string): string {
  return `{
          let contribution = ${expression};
          let previous = atomicAdd(&${low}[${low}Offset + runKey - 1u], contribution);
          if (previous + contribution < previous) { atomicAdd(&${high}[${high}Offset + runKey - 1u], 1u); }
        }`;
}

/** WGSL expression of a 64-bit fixed-point counter in world-unit fractions of its reference. */
function getFixedValueWGSL(low: string, high: string, index: string): string {
  return `((f32(${high}[${high}Offset + ${index}]) * 4294967296.0 + f32(${low}[${low}Offset + ${index}])) / ${getWGSLFloatLiteral(FIXED_POINT_SCALE)})`;
}

/** Affine metrics read from the per-frame layout, with the legacy fixed-affine fallback. */
function getPatchGeometryWGSL(props: GPURasterPatchMetricsProps): string {
  const [columnX, rowX, , columnY, rowY] = props.grid.affine;
  const read = (slot: number) => `georeference[georeferenceOffset + ${slot}u]`;
  // Keep the square and square root in the shared significand/exponent representation, so
  // representable axis lengths survive even when their squared lengths overflow or underflow.
  const getAxisLength = (firstSlot: number, secondSlot: number) =>
    `convertPlanarDistanceValue(getPlanarDistanceVectorLength(getPlanarDistanceVector(vec2f(${read(firstSlot)}, ${read(secondSlot)}), vec2f(0.0))))`;
  return `${props.parameters ? SCALED_PLANAR_DISTANCE_WGSL : ''}
fn getColumnStep() -> f32 {
  return ${props.parameters ? getAxisLength(2, 4) : getWGSLFloatLiteral(Math.hypot(columnX, columnY))};
}
fn getRowStep() -> f32 {
  return ${props.parameters ? getAxisLength(3, 5) : getWGSLFloatLiteral(Math.hypot(rowX, rowY))};
}
fn getPixelArea() -> f32 {
  return ${props.parameters ? `abs(${read(6)})` : getWGSLFloatLiteral(Math.abs(columnX * rowY - rowX * columnY))};
}`;
}
