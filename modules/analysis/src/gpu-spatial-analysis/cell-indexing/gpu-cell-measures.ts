// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs, fp64} from '@luma.gl/shaders';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../analysis-core/contracts/earth';
import {
  getGPUAreaUnit,
  getGPUAuthalicSphereRadius,
  type GPUMeasureDescription,
  type GPUSpatialUnit
} from '../../analysis-core/contracts/spatial-context';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import type {GPUCellWordOrder} from '../cell-aggregation/index';
import {H3_BOUNDARY_WGSL} from './h3-boundary-wgsl';
import {H3_MEASURES_WGSL} from './h3-measures-wgsl';
import {S2_CELL_MEASURES_WGSL} from './s2-cell-measures-wgsl';
import {QUADBIN_CELL_MEASURES_WGSL} from './quadbin-cell-measures-wgsl';
import type {GPUCellIndexFamily} from './cell-index-families';
import {CELL_MEASURES_FP64_WGSL} from './cell-measures-fp64-wgsl';
import {H3_MEASURES_FP64_WGSL} from './h3-measures-fp64-wgsl';
import {S2_CELL_MEASURES_FP64_WGSL} from './s2-cell-measures-fp64-wgsl';
import {QUADBIN_CELL_MEASURES_FP64_WGSL} from './quadbin-cell-measures-fp64-wgsl';

const OPERATION = 'GPUCellMeasures';

/** Most boundary edges an H3 cell has (hexagons 6, pentagons 5, distortion vertices up to 10). */
export const GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT = 10;

/** Caller-owned outputs of {@link GPUCellMeasures}, all row-aligned with the input cells. */
export type GPUCellMeasuresOutput = {
  /**
   * One `float32` area per cell: square meters by default, square kilometers with `units:
   * 'kilometers'` (H3 `cellAreaM2` / `cellAreaKm2`); 0 for invalid cells.
   */
  areas?: GraphDataView<'float32'>;
  /** One `float32` perimeter per cell in meters or kilometers (sum of all edges before stride truncation); 0 for invalid cells. */
  perimeters?: GraphDataView<'float32'>;
  /**
   * `rows * maximumEdgeCount` `float32` edge lengths in meters (or kilometers). Cell `row` owns
   * entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to
   * vertex `i + 1` (cyclic) of `GPUCellGeometry`. H3 follows `cellToBoundary`; S2 follows
   * face-UV order (00, 01, 11, 10); Quadbin follows north, east, south, west with parallel
   * north/south edges. Entries past the cell's edge count are zero.
   */
  edgeLengths?: GraphDataView<'float32'>;
  /**
   * One `uint32` boundary edge count per cell (6, 5 for pentagons, up to 10 with distortion
   * vertices; 4 for S2/Quadbin; 0 for invalid cells), before any truncation to `maximumEdgeCount`.
   */
  edgeCounts?: GraphDataView<'uint32'>;
  /** One `uint32` per cell: 1 for the 12 pentagons of the resolution, 0 for other cells and invalid cells. */
  pentagons?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellMeasures}.
 *
 * Per-frame (no recompile): the contents of `cells`. Topology (needs a new graph): `family`,
 * `wordOrder`, `maximumEdgeCount`, `units`, `precision`, the row count and which outputs are present.
 */
export type GPUCellMeasuresProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-measures'`. */
  id?: string;
  /** Cell family: H3, S2 or Quadbin. */
  family: Extract<GPUCellIndexFamily, 'h3' | 's2' | 'quadbin'>;
  /** One 64-bit cell per row as two `uint32` words. */
  cells: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /**
   * Fixed `edgeLengths` stride per cell: 6–10 for H3, 4–10 for S2/Quadbin.
   * Defaults to 6 for H3 and 4 otherwise. H3 distortion vertices beyond the stride are truncated.
   */
  maximumEdgeCount?: number;
  /**
   * Unit of `areas`, `perimeters` and `edgeLengths`: `'meters'` (default; square meters for areas)
   * or `'kilometers'` (square kilometers). Before this option the outputs were always kilometers.
   */
  units?: Exclude<GPUSpatialUnit, 'native'>;
  /**
   * Geometry arithmetic: 'float32' (default) or the published 'fp64' double-single module.
   * Use 'fp64' for H3 pentagons and face crossings above resolution 10. S2/Quadbin retain
   * their cancellation-free formulas. Outputs remain float32; this is compile-time topology.
   */
  precision?: 'float32' | 'fp64';
  /** Caller-owned outputs; at least one. */
  output: GPUCellMeasuresOutput;
};

/**
 * Spherical area, perimeter, edge lengths and pentagon flag of H3, S2 and Quadbin cells.
 * H3 follows `cellAreaM2`, `edgeLengthM`, `isPentagon`. Outputs are meters and square meters unless `units: 'kilometers'`;
 * `measures` describes the model (exact spherical polygon areas with great-circle edges on the
 * WGS84 authalic sphere, radius {@link GPU_EARTH_AUTHALIC_RADIUS_METERS}). Quadbin uses
 * parallels and meridians, reporting sphere-rectangle area and sphere-parallel-meridian length
 * models. Its resolution-zero cell is the full Mercator band, excluding the polar caps.
 *
 * One kernel; each thread decodes one cell. H3 areas are the sum of the spherical triangles between
 * the cell center and consecutive boundary vertices, as H3 `cellAreaRads2` defines them, scaled by
 * the squared authalic Earth radius (multiply by `(radius / GPU_EARTH_AUTHALIC_RADIUS_METERS) ** 2`
 * for another sphere).
 *
 * Cell identifiers carry no coordinates, so this class takes no `spatialContext`; its measure
 * model is the discrete-grid authalic sphere, reported in `measures`.
 *
 * Precision, from `H3_MEASURES_WGSL`: hexagons whose boundary stays on the center's icosahedron
 * face (all but the 12 pentagons and the thin set of cells straddling an icosahedron edge) are
 * measured from exact integer lattice offsets and stay within 1e-6 relative error at every
 * resolution 0-15. Pentagons and edge-straddling cells fall back to f32 boundary unit vectors,
 * which resolve about 6e-8 radians (0.4 m). Their measured relative error is 6e-7 at resolution
 * 0, 5e-5 at 3 and 4, at most 5e-4 up to resolution 8 and 2e-2 at 9 and 10 (areas: 2e-5 at 4,
 * 1.5e-4 at 8, 5e-3 at 9 and 10): treat them as unreliable beyond resolution 10. Pentagon
 * flags and edge counts are exact. With `precision: 'fp64'`, boundary projection and
 * face-edge intersections retain double-single coordinates through the published fp64 module;
 * differences are taken before output rounding. The CPU/GPU specs use 2e-5 relative tolerance
 * for all H3 resolutions, including pentagons and face crossings. S2 uses cancellation-free quadratic-map increments through
 * level 30; Quadbin uses hyperbolic difference identities through zoom 26. CPU/GPU parity
 * specs require a 3e-6 relative tolerance for these float32 measures. Invalid and zero cells produce all-zero rows.
 */
export class GPUCellMeasures implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellMeasuresProps;
  /** `edgeLengths` stride per cell. */
  readonly maximumEdgeCount: number;
  /** Resolved output unit. */
  readonly units: 'meters' | 'kilometers';
  /** Machine-readable length and area models and units of the outputs. */
  readonly measures: GPUMeasureDescription;

  constructor(props: GPUCellMeasuresProps) {
    this.id = props.id ?? 'cell-measures';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      ['h3', 's2', 'quadbin'].includes(props.family),
      "family must be 'h3', 's2' or 'quadbin'"
    );
    validateRequirement(
      id,
      props.wordOrder === undefined || ['little-endian', 'high-low'].includes(props.wordOrder),
      "wordOrder must be 'little-endian' or 'high-low'"
    );
    validateRequirement(
      id,
      props.precision === undefined || ['float32', 'fp64'].includes(props.precision),
      "precision must be 'float32' or 'fp64'"
    );
    this.units = props.units ?? 'meters';
    validateRequirement(
      id,
      ['meters', 'kilometers'].includes(this.units),
      "units must be 'meters' or 'kilometers'"
    );
    this.measures = {
      lengthModel: props.family === 'quadbin' ? 'sphere-parallel-meridian' : 'great-circle',
      areaModel: props.family === 'quadbin' ? 'sphere-rectangle' : 'sphere-polygon',
      lengthUnits: this.units,
      areaUnits: getGPUAreaUnit(this.units)
    };
    const minimumEdgeCount = props.family === 'h3' ? 6 : 4;
    this.maximumEdgeCount = props.maximumEdgeCount ?? minimumEdgeCount;
    validateCapacity(
      id,
      'maximumEdgeCount',
      this.maximumEdgeCount,
      minimumEdgeCount,
      GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT
    );
    const {output} = props;
    const outputViews = [
      ['output.areas', output.areas],
      ['output.perimeters', output.perimeters],
      ['output.edgeLengths', output.edgeLengths],
      ['output.edgeCounts', output.edgeCounts],
      ['output.pentagons', output.pentagons]
    ] as const;
    validateSinglePackedView(props.cells, ['uint32x2'], `${id} cells`);
    const rows = props.cells.length;
    validateCapacity(id, 'cells row count', rows);
    validateRequirement(
      id,
      outputViews.some(([, view]) => view),
      'output needs at least one output'
    );
    for (const [name, view] of [
      ['output.areas', output.areas],
      ['output.perimeters', output.perimeters]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
        validateViewLength(id, name, view, rows);
      }
    }
    if (output.edgeLengths) {
      validateSinglePackedView(output.edgeLengths, ['float32'], `${id} output.edgeLengths`);
      validateViewLength(
        id,
        'output.edgeLengths (rows * maximumEdgeCount)',
        output.edgeLengths,
        rows * this.maximumEdgeCount
      );
    }
    for (const [name, view] of [
      ['output.edgeCounts', output.edgeCounts],
      ['output.pentagons', output.pentagons]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        validateViewLength(id, name, view, rows);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      outputViews.map(([, view]) => view),
      [props.cells]
    );
  }

  /** Returns the single measures node `${id}-measures`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumEdgeCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, ...Object.values(output)]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'}
    ];
    const addOutput = (name: string, view: GraphDataView | undefined, type: 'f32' | 'u32') => {
      if (view) {
        bindings.push({name, view, type, access: 'read_write'});
      }
    };
    addOutput('areasOut', output.areas, 'f32');
    addOutput('perimetersOut', output.perimeters, 'f32');
    addOutput('edgeLengthsOut', output.edgeLengths, 'f32');
    addOutput('edgeCountsOut', output.edgeCounts, 'u32');
    addOutput('pentagonsOut', output.pentagons, 'u32');

    const needsMeasures = Boolean(
      output.areas || output.perimeters || output.edgeLengths || output.edgeCounts
    );
    const h3Body = `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  ${output.pentagons ? 'pentagonsOut[pentagonsOutOffset + index] = select(0u, 1u, cellMeasuresIsPentagon(key));' : ''}
  ${
    needsMeasures
      ? `let measures = cellMeasuresGetH3(key);
  ${output.areas ? 'areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;' : ''}
  ${output.perimeters ? 'perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;' : ''}
  ${output.edgeCounts ? 'edgeCountsOut[edgeCountsOutOffset + index] = measures.count;' : ''}
  ${
    output.edgeLengths
      ? `for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] =
      select(0.0, measures.edges[edge] * EARTH_RADIUS, edge < measures.count);
  }`
      : ''
  }`
      : ''
  }`;
    const rectangularBody = `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  let valid = ${props.family === 's2' ? 'dggs_s2_is_valid_cell_id(key)' : 'cellIsValidQuadbin(key)'};
  ${output.pentagons ? 'pentagonsOut[pentagonsOutOffset + index] = 0u;' : ''}
  ${
    needsMeasures
      ? `var measures: ${props.family === 's2' ? 'S2CellMeasures' : 'QuadbinCellMeasures'};
  if (valid) { measures = ${props.family === 's2' ? 'cellMeasuresGetS2' : 'cellMeasuresGetQuadbin'}(key); }
  ${output.areas ? 'areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;' : ''}
  ${output.perimeters ? 'perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;' : ''}
  ${output.edgeCounts ? 'edgeCountsOut[edgeCountsOutOffset + index] = select(0u, 4u, valid);' : ''}
  ${
    output.edgeLengths
      ? `for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    var edgeLength = 0.0;
    if (edge < 4u) { edgeLength = measures.edges[edge] * EARTH_RADIUS; }
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] = edgeLength;
  }`
      : ''
  }`
      : ''
  }`;
    // Every family uses the authalic radius for both areas and edge lengths.
    const radius = getGPUAuthalicSphereRadius({
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: this.units
    });
    const usesFP64 = props.precision === 'fp64';
    const geometrySource = usesFP64
      ? `${CELL_MEASURES_FP64_WGSL}\n${props.family === 'h3' ? H3_MEASURES_FP64_WGSL : props.family === 's2' ? S2_CELL_MEASURES_FP64_WGSL : QUADBIN_CELL_MEASURES_FP64_WGSL}`
      : props.family === 'h3'
        ? `${H3_BOUNDARY_WGSL}\n${H3_MEASURES_WGSL}`
        : props.family === 's2'
          ? S2_CELL_MEASURES_WGSL
          : QUADBIN_CELL_MEASURES_WGSL;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measures`,
        operation: OPERATION,
        variant: `${props.family}${usesFP64 ? '-fp64' : ''}`,
        modules: usesFP64 ? [fp64] : undefined,
        bindings,
        invocationCount: props.cells.length,
        declarations: `const MAXIMUM_EDGE_COUNT: u32 = ${maximumEdgeCount}u;
const EARTH_RADIUS: f32 = ${getWGSLFloatLiteral(radius)};
${dggs.source}
${CELL_KEY_WGSL}
${geometrySource}`,
        body: props.family === 'h3' ? h3Body : rectangularBody
      })
    ];
  }
}
