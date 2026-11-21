// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveCellPrecision, type CellPrecision} from './cell-precision';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {getPlatformInfo, Kernel} from '@luma.gl/graph';
import {dggs, fp64, WGSLShaderAssembler} from '@luma.gl/shaders';

import {
  getWGSLFloatLiteral,
  registerWGSLKernelNodeRecord,
  registerCompiledWGSLKernelNodeRecord,
  type WGSLKernelNodeRecord,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateSinglePackedView,
  validateCapacity,
  validateRequirement
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

import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUCellMeasures';
/** Storage component indices, including fixed-stride edge records, cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Fixed size makes chunk lengths dispatch parameters rather than pipeline topology. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Guard padded 3D workgroups before multiplying their linear index by the workgroup size. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

/** Most boundary edges an H3 cell has (hexagons 6, pentagons 5, distortion vertices up to 10). */
export const GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT = 10;

/** Caller-owned outputs of {@link GPUCellMeasures}, all row-aligned with the input cells. */
type CellMeasuresOutputPorts = {
  /**
   * One `float32` area per cell: square meters by default, square kilometers with `units:
   * 'kilometers'` (H3 `cellAreaM2` / `cellAreaKm2`); 0 for invalid cells.
   */
  areas?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** One `float32` perimeter per cell in meters or kilometers (sum of all edges before stride truncation); 0 for invalid cells. */
  perimeters?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * `rows * maximumEdgeCount` `float32` edge lengths in meters (or kilometers). Cell `row` owns
   * entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to
   * vertex `i + 1` (cyclic) of `GPUCellGeometry`. H3 follows `cellToBoundary`; S2 follows
   * face-UV order (00, 01, 11, 10); Quadbin follows north, east, south, west with parallel
   * north/south edges. Entries past the cell's edge count are zero.
   */
  edgeLengths?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * One `uint32` boundary edge count per cell (6, 5 for pentagons, up to 10 with distortion
   * vertices; 4 for S2/Quadbin; 0 for invalid cells), before any truncation to `maximumEdgeCount`.
   */
  edgeCounts?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** One `uint32` per cell: 1 for the 12 pentagons of the resolution, 0 for other cells and invalid cells. */
  pentagons?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/** Caller-owned measure outputs; select at least one terminal measure. */
export type GPUCellMeasuresOutput = CellMeasuresOutputPorts &
  {
    [Port in keyof CellMeasuresOutputPorts]-?: Required<Pick<CellMeasuresOutputPorts, Port>>;
  }[keyof CellMeasuresOutputPorts];

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
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
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
   * Geometry arithmetic: 'float32' (default) or the published 'double-single' module.
   * Use 'double-single' for H3 pentagons and face crossings above resolution 10. S2/Quadbin retain
   * their cancellation-free formulas. Outputs remain float32; this is compile-time topology.
   */
  precision?: CellPrecision;
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
 * flags and edge counts are exact. With `precision: 'double-single'`, boundary projection and
 * face-edge intersections retain double-single coordinates through the published fp64 module;
 * differences are taken before output rounding. The CPU/GPU specs use 2e-5 relative tolerance
 * for all H3 resolutions, including pentagons and face crossings. S2 uses cancellation-free quadratic-map increments through
 * level 30; Quadbin uses hyperbolic difference identities through zoom 26. CPU/GPU parity
 * specs require a 3e-6 relative tolerance for these float32 measures. Invalid and zero cells produce all-zero rows.
 * Packed and chunked inputs are supported; outputs mirror the input chunks. Every edge-length
 * chunk holds whole fixed-stride records, so no cell crosses chunk boundaries.
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

  /** Upload plus one measures dispatch per nonempty input chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly edgeLengthChunks: readonly GraphDataView[];

  private readonly precision: 'float32' | 'double-single';

  constructor(props: GPUCellMeasuresProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'cells',
      'wordOrder',
      'maximumEdgeCount',
      'units',
      'precision',
      ['output', ['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons']]
    ]);
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
    this.precision = resolveCellPrecision(id, props.precision);
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
    validateRequirement(
      id,
      outputViews.some(([, view]) => view),
      'output needs at least one output'
    );
    this.chunkPlan = planGraphChunkDispatch(id, {
      cells: props.cells,
      areas: output.areas,
      perimeters: output.perimeters,
      edgeCounts: output.edgeCounts,
      pentagons: output.pentagons
    });
    validateCapacity(id, 'cells row count', this.chunkPlan.totalLength);
    for (const chunk of this.chunkPlan.chunks) {
      validateSinglePackedView(
        chunk.views['cells'],
        ['uint32x2'],
        `${id} cells chunk ${chunk.chunkIndex}`
      );
      validateMaximumCount(
        id,
        `cells chunk ${chunk.chunkIndex} components`,
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      for (const [name, format] of [
        ['areas', 'float32'],
        ['perimeters', 'float32'],
        ['edgeCounts', 'uint32'],
        ['pentagons', 'uint32']
      ] as const) {
        const view = chunk.views[name];
        if (view)
          validateSinglePackedView(
            view,
            [format],
            `${id} output.${name} chunk ${chunk.chunkIndex}`
          );
      }
    }
    this.edgeLengthChunks = output.edgeLengths ? getGraphViewChunks(output.edgeLengths) : [];
    if (output.edgeLengths) {
      validateRequirement(
        id,
        output.edgeLengths.length === this.chunkPlan.totalLength * this.maximumEdgeCount,
        'output.edgeLengths length must equal cells rows * maximumEdgeCount'
      );
      validateRequirement(
        id,
        this.edgeLengthChunks.length === this.chunkPlan.chunks.length,
        'output.edgeLengths must match cells chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const edgeLengths = this.edgeLengthChunks[chunk.chunkIndex];
        validateSinglePackedView(
          edgeLengths,
          ['float32'],
          `${id} output.edgeLengths chunk ${chunk.chunkIndex}`
        );
        validateRequirement(
          id,
          edgeLengths.length === chunk.length * this.maximumEdgeCount,
          `output.edgeLengths chunk ${chunk.chunkIndex} length must equal cells chunk rows * maximumEdgeCount (${chunk.length * this.maximumEdgeCount})`
        );
        validateMaximumCount(
          id,
          `output.edgeLengths chunk ${chunk.chunkIndex} components`,
          edgeLengths.length,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    validateDisjointWritableViews(id, output);
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      outputViews.map(([, view]) => view),
      [props.cells]
    );
  }

  /** Uploads chunk parameters once and dispatches every nonempty chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumEdgeCount} = this;
    const {output} = props;
    validateMaximumCount(
      id,
      'storage bindings',
      2 + Object.values(output).filter(Boolean).length,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {edgeLengths: output.edgeLengths});
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
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
    const usesFP64 = this.precision === 'double-single';
    const geometrySource = usesFP64
      ? `${CELL_MEASURES_FP64_WGSL}\n${props.family === 'h3' ? H3_MEASURES_FP64_WGSL : props.family === 's2' ? S2_CELL_MEASURES_FP64_WGSL : QUADBIN_CELL_MEASURES_FP64_WGSL}`
      : props.family === 'h3'
        ? `${H3_BOUNDARY_WGSL}\n${H3_MEASURES_WGSL}`
        : props.family === 's2'
          ? S2_CELL_MEASURES_WGSL
          : QUADBIN_CELL_MEASURES_WGSL;
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    // Every chunk closes over the same source string, including assembled fp64 modules.
    let sharedShaderSource: string | undefined;
    let sharedKernelSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const views = chunk.views;
        const bindings: WGSLKernelBinding[] = [
          {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
          {name: 'cells', view: views['cells'], type: 'u32', access: 'read'}
        ];
        const addOutput = (name: string, view: GraphDataView | undefined, type: 'f32' | 'u32') => {
          if (view) bindings.push({name, view, type, access: 'read_write'});
        };
        addOutput('areasOut', views['areas'], 'f32');
        addOutput('perimetersOut', views['perimeters'], 'f32');
        addOutput('edgeLengthsOut', this.edgeLengthChunks[chunk.chunkIndex], 'f32');
        addOutput('edgeCountsOut', views['edgeCounts'], 'u32');
        addOutput('pentagonsOut', views['pentagons'], 'u32');
        const declarations = bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const source =
          sharedShaderSource ??
          `${declarations}
const MAXIMUM_EDGE_COUNT: u32 = ${maximumEdgeCount}u;
const EARTH_RADIUS: f32 = ${getWGSLFloatLiteral(radius)};
${dggs.source}
${CELL_KEY_WGSL}
${geometrySource}
@compute @workgroup_size(${CHUNK_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  // Bound arrays expose prefix + logical components, including unaligned packed subviews.
  let cellsOffset = arrayLength(&cells) - rowCount * 2u;
  ${output.areas ? 'let areasOutOffset = arrayLength(&areasOut) - rowCount;' : ''}
  ${output.perimeters ? 'let perimetersOutOffset = arrayLength(&perimetersOut) - rowCount;' : ''}
  ${output.edgeLengths ? 'let edgeLengthsOutOffset = arrayLength(&edgeLengthsOut) - rowCount * MAXIMUM_EDGE_COUNT;' : ''}
  ${output.edgeCounts ? 'let edgeCountsOutOffset = arrayLength(&edgeCountsOut) - rowCount;' : ''}
  ${output.pentagons ? 'let pentagonsOutOffset = arrayLength(&pentagonsOut) - rowCount;' : ''}
  ${props.family === 'h3' ? h3Body : rectangularBody}
}`;
        sharedShaderSource = source;
        // The published fp64 integer branch has no live uniform dependency; keep the same
        // arithmetic choice as createWGSLKernelNode, with a chunk-independent source and layout.
        const kernelSource =
          sharedKernelSource ??
          (usesFP64
            ? new WGSLShaderAssembler().assembleWGSLShader({
                platformInfo: getPlatformInfo(graph.device),
                source,
                modules: [fp64],
                defines: {LUMA_FP64_INTEGER_ARITHMETIC: true},
                shaderStage: 'compute',
                computeEntryPoint: 'main',
                scanVertexAttributes: false
              }).source
            : source);
        sharedKernelSource = kernelSource;
        const layout = layouts[chunk.chunkIndex];
        const kernelRecord: WGSLKernelNodeRecord = {
          id: `${id}-measures-${chunk.chunkIndex}`,
          operation: OPERATION,
          variant: `${props.family}${usesFP64 ? '-fp64' : ''}`,
          source: kernelSource,
          body: props.family === 'h3' ? h3Body : rectangularBody,
          entryPoint: 'main',
          workgroupSize: [CHUNK_WORKGROUP_SIZE, 1, 1],
          workgroups: [layout.x, layout.y, layout.z],
          invocationCount: chunk.length,
          constants: {},
          bindings,
          textures: [],
          indirect: false
        };
        registerWGSLKernelNodeRecord(graph, kernelRecord);
        return createGPUComputeCommandNode<Parameters>({
          id: `${id}-measures-${chunk.chunkIndex}`,
          resources: bindings.map(binding => ({
            buffer: binding.view,
            usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
          })),
          workload: {
            operation: OPERATION,
            variant: `${props.family}${usesFP64 ? '-fp64' : ''}`,
            commandCount: 1,
            maximumWorkgroupCount: layout.x * layout.y * layout.z,
            maximumInvocationCount: chunk.length
          },
          compile: function compileRecordedKernel({device}) {
            const kernel = new Kernel(device, {
              id: `${OPERATION}-${props.family}${usesFP64 ? '-fp64' : ''}`,
              source: kernelSource,
              shaderLayout: {
                bindings: bindings.map((binding, location) => ({
                  name: binding.name,
                  group: 0,
                  location,
                  type: binding.access === 'read' ? 'read-only-storage' : 'storage'
                }))
              }
            });
            registerCompiledWGSLKernelNodeRecord(graph, kernelRecord, compileRecordedKernel);
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: Object.fromEntries(
                    bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
                  ),
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      });
    return [...nodes, ...chunkNodes];
  }
}
