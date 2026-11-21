// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {gateDorlingPhase} from './dorling-dispatch-gate';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {createDorlingRowTileNode} from './dorling-row-tile-node';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {GET_SPATIAL_NAN_WGSL, IS_SPATIAL_FINITE_WGSL} from '../shared/spatial-wgsl-point-helpers';

/** Published reductions consume 256 elements per level. */
const REDUCTION_WORKGROUP_SIZE = 256;
/** Static unrolling remains bounded to keep graph compilation and encoding practical. */
const MAXIMUM_DORLING_ITERATIONS = 256;
/** Coordinates, radii and scale stay below overflow in squared float32 distances. */
const MAXIMUM_DORLING_MAGNITUDE = 1e15;
/** Shared keyed grid: at most six radix digits, four scan levels, aggregation and total. */
const DORLING_GRID_COMMAND_NODE_ALLOWANCE = 64;
/** Largest neighbor-grid side; the grid has `side^2` cells. */
const MAXIMUM_NEIGHBOR_GRID_SIDE = 1024;
/** Penetration noise floor: 4 f32 ulps (2^-23 each) of the largest coordinate. */
const COORDINATE_ULP_FLOOR = 4.76837158203125e-7;

/** Four metric words per row plus a spare row must remain addressable by uint32. Device binding limits also apply. */
export const MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS = 0x3ffffffe;
/** Exact all-pairs relaxation is O(n^2) per pass, so the explicit reference retains its cap. */
export const MAXIMUM_GPU_DORLING_CARTOGRAM_ALL_PAIRS_ROWS = 65536;

/** Caller-owned outputs. Rows retain input order, including invalid rows. */
export type GPUDorlingCartogramOutput = {
  positions: GraphDataView<'float32x2'>;
  radii: GraphDataView<'float32'>;
  sourceIds: GraphDataView<'uint32'>;
  /** One float32x2 row: maximum penetration, maximum displacement from initial position. */
  metrics: GraphDataView<'float32x2'>;
  /** Four uint32 words: iterations used, converged, budget exhausted, excluded row count. */
  status: GraphDataView<'uint32'>;
};

export type GPUDorlingCartogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  sourceIds?: GraphDataView<'uint32'>;
  /** Circle area per unit value, in squared position units. Default 1. */
  areaScale?: number;
  /** Static GPU iteration budget, 0..256. Default 100. */
  maximumIterations?: number;
  /** Maximum penetration accepted as converged, in position units. Default 0.001. */
  overlapTolerance?: number;
  /** Attraction toward the original center per pass, 0..1. Default 0.01. May leave residual overlap. */
  attraction?: number;
  /** Rows per relax/diagnose dispatch. Results do not depend on it. Default: one dispatch per pass. */
  rowTileSize?: number;
  /**
   * Compile-time. `true` builds a shared keyed neighbor grid before every pass and the final
   * diagnosis. Cell widths are at least twice the largest radius, enlarged to fit the bounded
   * grid. Every overlapping pair is visited; only correction summation order can differ from
   * all-pairs (absolute tolerance 0.005 position units on the clustered regression fixture).
   * Highly clustered layouts or a dominant circle can still approach O(rows^2).
   * Default `'auto'` uses the grid from DORLING_NEIGHBOR_GRID_MINIMUM_ROWS rows.
   * `false` preserves the all-pairs reference, capped at 65,536 rows.
   */
  neighborGrid?: boolean | 'auto';
  output: GPUDorlingCartogramOutput;
};

/** Measured cut-over: grid overhead exceeds the benefit below this row count. */
export const DORLING_NEIGHBOR_GRID_MINIMUM_ROWS = 32768;

/**
 * Planar Dorling circle relaxation, indexed by default above the named cut-over. The shared
 * grid is rebuilt from the current centers for every pass; the explicit all-pairs reference
 * (`neighborGrid: false`) remains capped at 65,536 rows. Both use synchronous half-penetration
 * corrections averaged over overlapping neighbors. Initial centers are input positions;
 * radii are sqrt(value * areaScale / pi).
 * Origin attraction preserves geography; no GeoDa coordinate parity is claimed. Residual penetration alone
 * determines convergence, measured against `overlapTolerance` or four f32 ulps of the largest
 * coordinate, whichever is larger, because f32 centers cannot resolve smaller penetration. A
 * stationary overlapping layout is never called converged.
 * Zero values are valid, stationary zero-radius points and exert no force. Negative/missing
 * values, nonfinite coordinates, |coordinate| > 1e15, or radius > 1e15 are excluded and emit
 * NaN positions/radii. Source IDs are copied even for excluded rows. No data is read back.
 * Compile-time: all numeric props, input/output lengths, row tiling and neighborGrid.
 * Converged phases use zero indirect workgroups; their published positions, metrics and status
 * remain unchanged. Every encoding resets the gates and iteration counters.
 * Per-frame: input buffer contents. The largest scratch binding is 16 * (rows + 1) bytes;
 * at the default 128 MiB device binding limit this admits 8,388,607 rows. Larger device limits
 * admit larger packed inputs. All views must be packed; caller chunk ownership is unchanged and never destroyed.
 */
export class GPUDorlingCartogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUDorlingCartogramProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumIterations: number;
  readonly areaScale: number;
  readonly overlapTolerance: number;
  readonly neighborGrid: boolean;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUDorlingCartogramProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'positions',
      'values',
      'sourceIds',
      'areaScale',
      'maximumIterations',
      'overlapTolerance',
      'attraction',
      'rowTileSize',
      'neighborGrid',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      ...(snapshot.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
        : {}),
      output: snapshotConstructionFields(snapshot.output, [
        'positions',
        'radii',
        'sourceIds',
        'metrics',
        'status'
      ])
    });
    this.props = props;
    this.id = props.id ?? 'dorling-cartogram';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
    );
    this.maximumIterations = props.maximumIterations ?? 100;
    this.areaScale = props.areaScale ?? 1;
    this.overlapTolerance = props.overlapTolerance ?? 0.001;
    validateRequirement(
      this.id,
      props.neighborGrid === undefined ||
        props.neighborGrid === 'auto' ||
        typeof props.neighborGrid === 'boolean',
      "neighborGrid must be true, false or 'auto'"
    );
    this.neighborGrid =
      props.neighborGrid === undefined || props.neighborGrid === 'auto'
        ? props.positions.length >= DORLING_NEIGHBOR_GRID_MINIMUM_ROWS
        : props.neighborGrid;
    validateCapacity(
      this.id,
      'positions rows',
      props.positions.length,
      0,
      this.neighborGrid
        ? MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS
        : MAXIMUM_GPU_DORLING_CARTOGRAM_ALL_PAIRS_ROWS
    );
    validateCapacity(
      this.id,
      'maximumIterations',
      this.maximumIterations,
      0,
      MAXIMUM_DORLING_ITERATIONS
    );
    for (const [name, value] of [
      ['areaScale', this.areaScale],
      ['overlapTolerance', this.overlapTolerance]
    ] as const) {
      validateFiniteNumber(this.id, name, value, 0, MAXIMUM_DORLING_MAGNITUDE);
      validateRequirement(
        this.id,
        value >= 0 && value <= MAXIMUM_DORLING_MAGNITUDE && (value === 0 || Math.fround(value) > 0),
        `${name} must be zero or a positive float32 value at most ${MAXIMUM_DORLING_MAGNITUDE}`
      );
    }
    validateFiniteNumber(this.id, 'attraction', props.attraction ?? 0.01, 0, 1);
    if (props.rowTileSize !== undefined) {
      validateCapacity(this.id, 'rowTileSize', props.rowTileSize);
    }
    const rows = props.positions.length;
    const entries = [
      ['positions', props.positions, 'float32x2', rows],
      ['values', props.values, 'float32', rows],
      ['sourceIds', props.sourceIds, 'uint32', rows],
      ['output.positions', props.output.positions, 'float32x2', rows],
      ['output.radii', props.output.radii, 'float32', rows],
      ['output.sourceIds', props.output.sourceIds, 'uint32', rows],
      ['output.metrics', props.output.metrics, 'float32x2', 1],
      ['output.status', props.output.status, 'uint32', 4]
    ] as const;
    for (const [name, view, format, length] of entries) {
      if (view) {
        validateSinglePackedView(view, [format], `${this.id} ${name}`);
        validateViewLength(this.id, name, view, length);
      }
    }
    const reductionLevels = Math.max(
      1,
      Math.ceil(Math.log(Math.max(rows, 1)) / Math.log(REDUCTION_WORKGROUP_SIZE))
    );
    const tileCount = Math.ceil(Math.max(rows, 1) / (props.rowTileSize ?? Math.max(rows, 1)));
    const metricNodes = 4 * (reductionLevels + 2) + 1;
    const gridNodes = this.neighborGrid
      ? DORLING_GRID_COMMAND_NODE_ALLOWANCE + 2 * (reductionLevels + 2) + 1
      : 0;
    this.estimatedCommandNodeCount =
      rows === 0
        ? 1
        : 3 +
          (this.neighborGrid ? reductionLevels + 2 : 0) +
          (this.maximumIterations + 1) * (gridNodes + tileCount * 2 + metricNodes + 2) +
          this.maximumIterations;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    const outputs = Object.values(props.output);
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [
      props.positions,
      props.values,
      props.sourceIds
    ]);
    validateSpatialOutputViews(
      this.id,
      props.positions.length > 0
        ? props.output
        : {metrics: props.output.metrics, status: props.output.status},
      [
        ...(props.positions.length > 0
          ? [
              [
                props.output.positions,
                props.output.radii,
                props.output.sourceIds,
                props.output.status
              ]
            ]
          : []),
        [props.output.metrics, props.output.status],
        ...(props.positions.length > 0 && this.maximumIterations > 0
          ? [{read: [props.output.status], write: [props.output.positions]}]
          : [])
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations} = this;
    const {output} = props;
    const rows = props.positions.length;
    // Validate every aligned caller binding and the bounds of every derived allocation before
    // declaring any resources. All scalar slices fit their parent's checked aligned binding.
    validateGraphChunkBindingRanges(graph, id, {
      ...(rows > 0
        ? {
            positions: props.positions,
            values: props.values,
            sourceIds: props.sourceIds,
            positionsOutput: output.positions,
            radii: output.radii,
            sourceIdsOutput: output.sourceIds
          }
        : {}),
      metrics: output.metrics,
      status: output.status
    });
    const gridSide = Math.max(1, Math.min(Math.ceil(Math.sqrt(rows)), MAXIMUM_NEIGHBOR_GRID_SIDE));
    const gridCellCount = gridSide * gridSide;
    const reductionLevels = Math.max(
      1,
      Math.ceil(Math.log(Math.max(rows, 1)) / Math.log(REDUCTION_WORKGROUP_SIZE))
    );
    const tileCount = Math.ceil(Math.max(rows, 1) / (props.rowTileSize ?? Math.max(rows, 1)));
    const phaseNodeBound =
      (this.neighborGrid
        ? DORLING_GRID_COMMAND_NODE_ALLOWANCE + 2 * (reductionLevels + 2) + 1
        : 0) +
      2 * tileCount +
      4 * (reductionLevels + 2) +
      3;
    for (const [name, bytes] of [
      ['positions rows metric scratch', rows === 0 ? 0 : (rows + 1) * 16],
      ['position scratch', rows * 8],
      ['reduction scratch', Math.max(rows * 4, Math.ceil(rows / REDUCTION_WORKGROUP_SIZE) * 8)],
      ['reduced metrics', 16],
      ['tile parameters', 4],
      ['grid cell offsets', this.neighborGrid ? (gridCellCount + 1) * 4 : 0],
      ['grid counts and keys', this.neighborGrid ? Math.max(rows, gridCellCount) * 4 : 0],
      [
        'radix histogram and scan scratch',
        this.neighborGrid ? 16 * Math.ceil(rows / REDUCTION_WORKGROUP_SIZE) * 4 : 0
      ],
      ['phase dispatch', rows === 0 ? 0 : phaseNodeBound * 12]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce positions rows or row tiles, or request a larger maxStorageBufferBindingSize'
      );
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.values,
      props.sourceIds,
      ...Object.values(output)
    ]);
    if (rows === 0) {
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-empty`,
          operation: 'GPUDorlingCartogram',
          variant: 'empty',
          invocationCount: 1,
          bindings: [
            {name: 'metrics', view: output.metrics, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          body: 'metrics[metricsOffset] = 0.0; metrics[metricsOffset+1u] = 0.0; status[statusOffset] = 0u; status[statusOffset+1u] = 1u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u;'
        })
      ];
    }
    const scratch = createTransientView(graph, `${id}-scratch`, 'float32x2', Math.max(rows, 1));
    // The spare row holds largest radius, GPU tile base, diagnosed-state counter and tile length.
    const rowMetrics = createTransientView(graph, `${id}-row-metrics`, 'float32x4', rows + 1);
    const neighborGrid = this.neighborGrid;
    const reductionInput = createTransientView(graph, `${id}-reduction-input`, 'float32', rows);
    const reductionWords = graph.createDataView(reductionInput.buffer, {
      format: 'uint32',
      length: rows
    });
    const reducedMetrics = createTransientView(graph, `${id}-reduced-metrics`, 'float32', 4);
    const scalarView = (view: GraphDataView, word: number, length = 1) =>
      graph.createDataView(view.buffer, {
        format: 'float32',
        length,
        byteOffset: view.byteOffset + word * 4
      });
    const declarations = `const ROWS: u32 = ${rows}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(this.overlapTolerance)};
const MAXIMUM_MAGNITUDE: f32 = ${getWGSLFloatLiteral(MAXIMUM_DORLING_MAGNITUDE)};
const MAXIMUM_ITERATIONS: u32 = ${maximumIterations}u;
${IS_SPATIAL_FINITE_WGSL}
fn valid(value: f32) -> bool { return isSpatialFinite(value); }
${GET_SPATIAL_NAN_WGSL}
fn nan(row: u32) -> f32 { _ = row; return getSpatialNaN(); }`;
    const tileParameters = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-tile-parameters`,
      new Uint32Array([Math.min(rows, props.rowTileSize ?? rows)])
    );
    const nodes: GPUCommandNode<Parameters>[] = [...tileParameters.nodes];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUDorlingCartogram',
        variant: 'initialize',
        invocationCount: Math.max(rows, 1),
        declarations,
        bindings: [
          {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          ...(props.sourceIds
            ? [{name: 'ids', view: props.sourceIds, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'centers', view: output.positions, type: 'f32', access: 'read_write'},
          {name: 'radii', view: output.radii, type: 'f32', access: 'read_write'},
          {name: 'sourceIds', view: output.sourceIds, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `if (index == 0u) { status[statusOffset] = 0u; status[statusOffset+1u] = 0u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u; }
if (index >= ROWS) { return; }
let center = vec2f(initial[initialOffset+index*2u], initial[initialOffset+index*2u+1u]);
let value = values[valuesOffset+index];
let radius = sqrt(value) * ${getWGSLFloatLiteral(Math.sqrt(this.areaScale / Math.PI))};
let included = valid(center.x) && valid(center.y) && all(abs(center) <= vec2f(MAXIMUM_MAGNITUDE)) && valid(value) && value >= 0.0 && valid(radius) && radius <= MAXIMUM_MAGNITUDE;
centers[centersOffset+index*2u] = select(nan(index), center.x, included);
centers[centersOffset+index*2u+1u] = select(nan(index), center.y, included);
radii[radiiOffset+index] = select(nan(index), radius, included);
sourceIds[sourceIdsOffset+index] = ${props.sourceIds ? 'ids[idsOffset+index]' : 'index'};`
      })
    );
    // Initialization for every encoding, including executions after early convergence.
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-reset-state`,
        operation: 'GPUDorlingCartogram',
        variant: 'reset-state',
        invocationCount: 1,
        bindings: [{name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'}],
        declarations,
        body: `rowMetrics[rowMetricsOffset + ROWS * 4u + 1u] = bitcast<f32>(0u);
rowMetrics[rowMetricsOffset + ROWS * 4u + 2u] = bitcast<f32>(0u);`
      })
    );
    const reductionNodes = (reduction: GPUReduction): GPUCommandNode<Parameters>[] => {
      let inputLength = rows;
      const result = reduction.getCommandNodes(graph);
      return result.map((node, level) => {
        const layout =
          level === result.length - 1
            ? {x: 1, y: 1, z: 1}
            : getBoundedDispatchLayout(
                'Dorling reduction',
                inputLength,
                REDUCTION_WORKGROUP_SIZE,
                graph.device.limits.maxComputeWorkgroupsPerDimension
              );
        inputLength = Math.ceil(inputLength / REDUCTION_WORKGROUP_SIZE);
        return {
          ...node,
          workload: {...node.workload, maximumWorkgroupCount: layout.x * layout.y * layout.z}
        };
      });
    };
    // Every tile uses its own dispatch and a shared shader. Base and length live on the GPU,
    // not in shader text, binding offsets or specialization constants.
    const tiledRowNodes = (kernel: WGSLKernelProps<Parameters>) => {
      const count = Math.min(rows, props.rowTileSize ?? rows);
      const stateBindings: WGSLKernelBinding[] = [
        {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'},
        {name: 'tileParameters', view: tileParameters.parameters, type: 'u32', access: 'read'}
      ];
      const result: GPUCommandNode<Parameters>[] = [
        createWGSLKernelNode(graph, {
          id: `${kernel.id}-tile-reset`,
          operation: 'GPUDorlingCartogram',
          variant: 'tile-reset',
          invocationCount: 1,
          declarations,
          bindings: stateBindings,
          body: `rowMetrics[rowMetricsOffset + ROWS * 4u + 1u] = bitcast<f32>(0u);
rowMetrics[rowMetricsOffset + ROWS * 4u + 3u] = bitcast<f32>(tileParameters[tileParametersOffset]);`
        })
      ];
      for (let tileBase = 0, tile = 0; tileBase < rows; tileBase += count, tile++) {
        result.push(
          createDorlingRowTileNode(graph, {
            ...kernel,
            id: `${kernel.id}-tile-${tile}`,
            invocationCount: Math.min(count, rows - tileBase),
            body: `{
let tileLength = bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 3u]);
if (index >= tileLength) { return; }
let index = index + bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 1u]);
${kernel.body}
}`
          }),
          createWGSLKernelNode(graph, {
            id: `${kernel.id}-tile-next-${tile}`,
            operation: 'GPUDorlingCartogram',
            variant: 'tile-next',
            invocationCount: 1,
            declarations,
            bindings: stateBindings,
            body: `let base = rowMetricsOffset + ROWS * 4u + 1u;
let nextBase = bitcast<u32>(rowMetrics[base]) + tileParameters[tileParametersOffset];
rowMetrics[base] = bitcast<f32>(nextBase);
rowMetrics[rowMetricsOffset + ROWS * 4u + 3u] = bitcast<f32>(min(tileParameters[tileParametersOffset], ROWS - min(ROWS, nextBase)));`
          })
        );
      }
      return result;
    };
    // Extract packed scalar columns for the published hierarchical reductions. Scratch is reused
    // between channels and passes; excluded rows use uint32 so counts above 2^24 stay exact.
    const reduceMetrics = (step: number): GPUCommandNode<Parameters>[] => {
      const result: GPUCommandNode<Parameters>[] = [];
      for (let channel = 0; channel < 4; channel++) {
        const excluded = channel === 2;
        result.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-extract-metric-${step}-${channel}`,
            operation: 'GPUDorlingCartogram',
            variant: 'extract-metric',
            invocationCount: rows,
            bindings: [
              {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read'},
              {
                name: 'values',
                view: excluded ? reductionWords : reductionInput,
                type: excluded ? 'u32' : 'f32',
                access: 'read_write'
              }
            ],
            body: `values[valuesOffset + index] = ${excluded ? 'u32' : 'f32'}(rowMetrics[rowMetricsOffset + index * 4u + ${channel}u]);`
          })
        );
        const destination = scalarView(reducedMetrics, channel);
        result.push(
          ...reductionNodes(
            new GPUReduction({
              id: `${id}-reduce-metric-${step}-${channel}`,
              input: excluded ? reductionWords : reductionInput,
              output: excluded
                ? graph.createDataView(destination.buffer, {
                    format: 'uint32',
                    length: 1,
                    byteOffset: destination.byteOffset
                  })
                : destination,
              operation: excluded ? 'sum' : 'max'
            })
          )
        );
      }
      result.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-diagnose-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'diagnose',
          invocationCount: 1,
          declarations,
          bindings: [
            {name: 'reduced', view: reducedMetrics, type: 'u32', access: 'read'},
            {name: 'metrics', view: output.metrics, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'}
          ],
          body: `let overlap = bitcast<f32>(reduced[reducedOffset]);
let displacement = bitcast<f32>(reduced[reducedOffset + 1u]);
let tolerance = max(TOLERANCE, bitcast<f32>(reduced[reducedOffset + 3u]) * ${COORDINATE_ULP_FLOOR});
metrics[metricsOffset] = overlap; metrics[metricsOffset + 1u] = displacement;
let round = bitcast<u32>(rowMetrics[rowMetricsOffset + ROWS * 4u + 2u]);
status[statusOffset] = round;
rowMetrics[rowMetricsOffset + ROWS * 4u + 2u] = bitcast<f32>(round + 1u);
status[statusOffset + 1u] = select(0u, 1u, overlap <= tolerance);
status[statusOffset + 2u] = select(0u, 1u, overlap > tolerance && status[statusOffset] == MAXIMUM_ITERATIONS);
status[statusOffset + 3u] = reduced[reducedOffset + 2u];`
        })
      );
      return result;
    };
    // Extent and cell keys share exactly the same float32 cell mapping as the candidate walk.
    const gridBounds = neighborGrid
      ? createTransientView(graph, `${id}-grid-bounds`, 'float32', 4)
      : undefined;
    const gridKeys = neighborGrid
      ? createTransientView(graph, `${id}-grid-keys`, 'uint32', rows)
      : undefined;
    const gridSourceRows = neighborGrid
      ? createTransientView(graph, `${id}-grid-source-rows`, 'uint32', rows)
      : undefined;
    const gridDeclarations = `${declarations}
const GRID_SIDE: u32 = ${gridSide}u;
fn getGridCell(value: f32, minimum: f32, maximum: f32, largestRadius: f32) -> u32 {
  // A radius-derived cell is enlarged only when the bounded grid cannot cover the extent.
  let width = max(largestRadius * 2.0, (maximum - minimum) / f32(GRID_SIDE));
  if (!(width > 0.0)) { return 0u; }
  return u32(clamp(floor((value - minimum) / width), 0.0, f32(GRID_SIDE - 1u)));
}`;
    // Views returned by the shared builder are distinct per pass. Replace bindings before each walk.
    let gridBindings: WGSLKernelBinding[] = [];
    const visitNeighbors = (
      action: string
    ) => `let largestRadius = rowMetrics[rowMetricsOffset + ROWS * 4u];
  let reach = radius + largestRadius;
  let boundsMinimum = vec2f(gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 2u]);
  let boundsMaximum = vec2f(gridBounds[gridBoundsOffset + 1u], gridBounds[gridBoundsOffset + 3u]);
  let firstColumn = max(getGridCell(center.x - reach, boundsMinimum.x, boundsMaximum.x, largestRadius), 1u) - 1u;
  let lastColumn = min(getGridCell(center.x + reach, boundsMinimum.x, boundsMaximum.x, largestRadius) + 1u, GRID_SIDE - 1u);
  let firstRow = max(getGridCell(center.y - reach, boundsMinimum.y, boundsMaximum.y, largestRadius), 1u) - 1u;
  let lastRow = min(getGridCell(center.y + reach, boundsMinimum.y, boundsMaximum.y, largestRadius) + 1u, GRID_SIDE - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn; cellColumn++) {
      let cell = cellRow * GRID_SIDE + cellColumn;
      let lastSlot = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = cellOffsets[cellOffsetsOffset + cell]; slot < lastSlot; slot++) {
        let other = objectIds[objectIdsOffset + slot];
        ${action}
      }
    }
  }`;
    const addGridBuildNodes = (centers: GraphDataView<'float32x2'>, step: string) => {
      for (let axis = 0; axis < 2; axis++) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-extract-bounds-${step}-${axis}`,
            operation: 'GPUDorlingCartogram',
            variant: 'extract-bounds',
            invocationCount: rows,
            bindings: [
              {name: 'centers', view: centers, type: 'f32', access: 'read'},
              {name: 'values', view: reductionInput, type: 'f32', access: 'read_write'}
            ],
            body: `values[valuesOffset + index] = centers[centersOffset + index * 2u + ${axis}u];`
          }),
          ...reductionNodes(
            new GPUReduction({
              id: `${id}-bounds-${step}-${axis}`,
              input: reductionInput,
              output: scalarView(gridBounds!, axis * 2, 2),
              operation: 'extent'
            })
          )
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-grid-keys-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'grid-keys',
          invocationCount: rows,
          declarations: gridDeclarations,
          bindings: [
            {name: 'centers', view: centers, type: 'f32', access: 'read'},
            {name: 'gridBounds', view: gridBounds!, type: 'f32', access: 'read'},
            {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read'},
            {name: 'keys', view: gridKeys!, type: 'u32', access: 'read_write'},
            {name: 'sourceRows', view: gridSourceRows!, type: 'u32', access: 'read_write'}
          ],
          body: `let center = vec2f(centers[centersOffset + index * 2u], centers[centersOffset + index * 2u + 1u]);
let largestRadius = rowMetrics[rowMetricsOffset + ROWS * 4u];
let column = getGridCell(center.x, gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 1u], largestRadius);
let row = getGridCell(center.y, gridBounds[gridBoundsOffset + 2u], gridBounds[gridBoundsOffset + 3u], largestRadius);
keys[keysOffset + index] = select(${gridCellCount}u, row * GRID_SIDE + column, valid(center.x) && valid(center.y));
sourceRows[sourceRowsOffset + index] = index;`
        })
      );
      const grid = createPointNeighborGridIndexNodes(graph, {
        id: `${id}-grid-${step}`,
        rowCount: rows,
        cellKeys: gridKeys!,
        sourceRows: gridSourceRows!,
        cellCount: gridCellCount
      });
      nodes.push(...grid.nodes);
      gridBindings = [
        {name: 'cellOffsets', view: grid.cellOffsets, type: 'u32', access: 'read'},
        {name: 'objectIds', view: grid.sortedRows, type: 'u32', access: 'read'},
        {name: 'gridBounds', view: gridBounds!, type: 'f32', access: 'read'}
      ];
    };
    // Final state metrics come from a dedicated neighborhood pass; earlier states reuse the
    // penetrations that the following relax pass computes anyway.
    const diagnose = (centers: GraphDataView<'float32x2'>, step: number) => {
      if (neighborGrid) {
        addGridBuildNodes(centers, `diagnose-${step}`);
      }
      const diagnoseRow = `if (index >= ROWS) { return; }
let radius = radii[radiiOffset+index]; var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
if (valid(radius)) {
  let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
  displacement = distance(center,vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]));
  coordinate = max(abs(center.x), abs(center.y));
  if (radius > 0.0) {
    ${
      neighborGrid
        ? visitNeighbors(`let otherRadius = radii[radiiOffset+other];
        if (other == index || !valid(otherRadius) || otherRadius == 0.0) { continue; }
        let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
        overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));`)
        : `for (var other = index+1u; other < ROWS; other++) {
      let otherRadius = radii[radiiOffset+other];
      if (!valid(otherRadius) || otherRadius == 0.0) { continue; }
      let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
      overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));
    }`
    }
  }
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;`;
      nodes.push(
        ...tiledRowNodes({
          id: `${id}-row-diagnose-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'row-diagnose',
          invocationCount: Math.max(rows, 1),
          declarations: neighborGrid ? gridDeclarations : declarations,
          bindings: [
            {name: 'centers', view: centers, type: 'f32', access: 'read'},
            {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
            {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
            {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'},
            ...gridBindings
          ],
          body: diagnoseRow
        })
      );
      nodes.push(...reduceMetrics(step));
    };
    if (neighborGrid) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extract-radius`,
          operation: 'GPUDorlingCartogram',
          variant: 'extract-radius',
          invocationCount: rows,
          declarations,
          bindings: [
            {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
            {name: 'values', view: reductionInput, type: 'f32', access: 'read_write'}
          ],
          body: `let radius = radii[radiiOffset + index];
values[valuesOffset + index] = select(0.0, radius, valid(radius));`
        }),
        ...reductionNodes(
          new GPUReduction({
            id: `${id}-maximum-radius`,
            input: reductionInput,
            output: scalarView(rowMetrics, rows * 4),
            operation: 'max'
          })
        )
      );
    }
    const initialNodes = nodes.splice(0);
    const appendPhase = (phase: GPUCommandNode<Parameters>[], name: string) => {
      nodes.push(...gateDorlingPhase(graph, `${id}-${name}`, phase, output.status, rows));
    };
    if (maximumIterations === 0) {
      diagnose(output.positions, 0);
      appendPhase(nodes.splice(0), 'final');
    }
    for (let step = 1; step <= maximumIterations; step++) {
      const source = output.positions;
      const destinations = scratch;
      const phaseStart = nodes.length;
      // Relax moves state step-1 to a tentative state step and records the metrics of state
      // step-1 (the penetrations are already computed). The reduction then decides convergence
      // of state step-1; commit publishes only a nonconverged tentative state. All rounds use
      // the same canonical source and scratch destination, so skipped rounds cannot change parity.
      if (neighborGrid) {
        addGridBuildNodes(source, `relax-${step}`);
      }
      nodes.push(
        ...tiledRowNodes({
          id: `${id}-relax-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'relax',
          invocationCount: Math.max(rows, 1),
          declarations: neighborGrid ? gridDeclarations : declarations,
          bindings: [
            {name: 'centers', view: source, type: 'f32', access: 'read'},
            {name: 'destinations', view: destinations, type: 'f32', access: 'read_write'},
            {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
            // The grid kernel is at the 8-binding limit, so it drops the converged check; the
            // commit below leaves the canonical positions unchanged when diagnosis finds convergence.
            ...(neighborGrid
              ? []
              : [
                  {
                    name: 'status',
                    view: output.status,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]),
            {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
            {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'},
            ...gridBindings
          ],
          body: `if (index >= ROWS) { return; }
let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
let radius = radii[radiiOffset+index]; var correction = vec2f(0.0); var neighbors = 0u;
var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
let origin = vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]);
if (valid(radius)) { displacement = distance(center, origin); coordinate = max(abs(center.x), abs(center.y)); }
if (valid(radius) && radius > 0.0) {
  ${
    neighborGrid
      ? visitNeighbors(`let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }`)
      : `for (var other = 0u; other < ROWS; other++) {
    let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }
  }`
  }
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;
var moved = center;
if (valid(radius) && radius > 0.0${neighborGrid ? '' : ' && status[statusOffset+1u] == 0u'}) {
  moved = center+correction/f32(max(neighbors,1u))+(origin-center)*${getWGSLFloatLiteral(props.attraction ?? 0.01)};
}
destinations[destinationsOffset+index*2u] = moved.x; destinations[destinationsOffset+index*2u+1u] = moved.y;`
        })
      );
      nodes.push(
        ...reduceMetrics(step - 1),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-commit-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'commit',
          invocationCount: Math.max(rows * 2, 1),
          bindings: [
            {name: 'source', view: destinations, type: 'f32', access: 'read'},
            {name: 'destinations', view: output.positions, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'}
          ],
          body: `if (index < ${rows * 2}u && status[statusOffset+1u] == 0u) { destinations[destinationsOffset+index] = source[sourceOffset+index]; }`
        })
      );
      appendPhase(nodes.splice(phaseStart), `round-${step}`);
      if (step === maximumIterations) {
        const finalStart = nodes.length;
        diagnose(output.positions, step);
        appendPhase(nodes.splice(finalStart), 'final');
      }
    }
    return [...initialNodes, ...nodes];
  }
}
