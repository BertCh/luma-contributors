// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUAnalysisRoute} from '../analysis-core/contracts/analysis-execution-planning';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';
import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphOutputViews,
  captureGraphCommandNodes,
  validateCapacity,
  validateCommandNodeBudget,
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  validateSinglePackedView,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createRasterFractionalTileNode,
  validateRasterFractionalBindingByteLength,
  validateRasterFractionalWritableViews
} from './fractional-coverage/raster-fractional-tile-kernel';
import {getSortedSegmentSumNodes} from '../analysis-core/kernels/sorted-segment-sums';
import {
  createWGSLKernelNode,
  createFillNode,
  getWGSLFloatLiteral
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRasterBandViews,
  snapshotRasterBand,
  snapshotRasterGrid,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from './raster-band-utils';
import {
  getRasterGridGeoreference,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../analysis-core/contracts/raster-grid';
import {
  getRasterRowAreaScaleWGSL,
  getRasterEffectiveAxisAlignmentWGSL
} from './raster-grid/raster-cell-area-scale';
import {
  requireRasterLength,
  validateRasterExpansionPorts,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';
import type {GPURasterFractionalCoverageProps} from './gpu-raster-fractional-coverage';
import {
  GPURasterTextureToBuffer,
  type GPURasterBand,
  type GPURasterBufferBand,
  type GPURasterMetadata
} from '@luma.gl/graph/gpu-raster';

/**
 * Slot of the affine determinant in the georeference view of `GPURasterFractionalCoverage`
 * (`[originX, originY, a, b, d, e, determinant, padding]`). Latitude weighting additionally
 * reads origin Y and row Y; explicit local layout 2 appends discriminator 2 and absolute Y
 * offset in words 8 and 9. Legacy padding word 7 is ignored.
 */
const GEOREFERENCE_DETERMINANT_SLOT = 6;

/** Largest category capacity; dense categories bound scratch space and the per-zone winner scan. */
export const MAXIMUM_CATEGORY_CAPACITY = 65534;

/** uint32 group keys reserve the last word for inactive tuple slots. */
export const MAXIMUM_ZONE_CATEGORY_GROUPS = 0xfffffffe;

/** uint32 tuple row indices reserve the last word for saturated status counts. */
export const MAXIMUM_TUPLES = 0xfffffffe;

/** Full-sort reference limit preserves the original reduction order and scratch bound. */
export const MAXIMUM_REFERENCE_TUPLES = 1048576;

/** The original reference route keeps its historical dense group cap. */
export const MAXIMUM_REFERENCE_ZONE_CATEGORY_GROUPS = 65535;

/** Sort tiles retain the historical 1M-row bound on any single sort allocation. */
export const MAXIMUM_TUPLE_TILE_SIZE = 1048576;

/** One full historical-size sort preserves reduction order for every previously accepted input. */
const DEFAULT_TUPLE_TILE_SIZE = 1048576;

/** Conservative allowance for a 32-bit radix sort, gather, reductions and merge per tile. */
const TUPLE_TILE_COMMAND_NODE_ALLOWANCE = 256;

/** Classify, reset, summary and publication outside the tuple tiles. */
const CATEGORICAL_COMMAND_NODE_ALLOWANCE = 8;

export type GPURasterFractionalCategoricalStatisticsProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Dense, uncalibrated category IDs. Masked/nodata cells contribute only to missing area. */
  input: GPURasterBand<'uint32'>;
  zoneCount: number;
  /** Category IDs are [0, categoryCapacity). Valid IDs outside this range fail the result. */
  categoryCapacity: number;
  /** Compile-time sort tile size, 1..1048576. Normalisation occurs after all tile support is merged. */
  tupleTileSize?: number;
  /** Compile-time default 'tiled'. 'exact' retains a single full sort and the original tuple/group caps. */
  mode?: 'tiled' | 'exact';
  coverage: GPURasterFractionalCoverageProps['output'];
  /**
   * The per-frame georeference view given to `GPURasterFractionalCoverage` (layout of
   * `getGPURasterFractionalCoverageParameterValues`). The cell area is read from it on every
   * encoding as `abs(determinant)`, so the area outputs follow a repacked georeference. When
   * omitted the cell area is `abs(det(grid.affine))` fixed at compile time: pass it whenever the
   * coverage `parameters` are rewritten for another cell size.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * What the affine's units are, for the area outputs. Default `'uniform'`: planar coordinate
   * units squared, as before. `'geographic'` (degrees) and `'web-mercator'` (EPSG:3857 meters)
   * give ground square meters on the WGS84 ellipsoid: each coverage fraction is weighted by the
   * grid-derived WGS84 midpoint ground-area scale (the product of local ground scales
   * at the row midpoint, rather than integrated geodesic cell area), so `coverageSums` then holds weighted coverage and `proportions` are area
   * proportions. They need an axis-aligned grid. Compile-time mode. The row edges come from
   * `parameters` (origin y and row step y, slots 1 and 5, with `grid.height` and
   * `pixelInterpretation` compile-time) on every encoding when it is given, so the ground scale
   * follows a repacked georeference together with the cell area. Without `parameters` they are
   * baked from `grid.affine`, like the planar cell area.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  output: {
    /** Zone-major category rows: zone * categoryCapacity + category. Sum of coverage fractions (weighted by row ground area for a non-uniform `cellSizeMode`). */
    coverageSums: GraphDataView<'float32'>;
    /** Coverage sums multiplied by abs(affine determinant), in squared coordinate units. */
    classAreas: GraphDataView<'float32'>;
    /** Category coverage divided by valid category coverage; NaN without valid support. */
    proportions: GraphDataView<'float32'>;
    /** Per zone; includes missing cells, within the supplied raster footprint only. */
    coveredArea: GraphDataView<'float32'>;
    validArea: GraphDataView<'float32'>;
    missingArea: GraphDataView<'float32'>;
    /** Highest category wins equal-coverage ties. 0xffffffff without a valid result. */
    majority: GraphDataView<'uint32'>;
    /** Lowest present category wins equal-coverage ties. Absent categories never compete. */
    minority: GraphDataView<'uint32'>;
    /** One for complete coverage with positive valid support; zero otherwise. */
    validity: GraphDataView<'uint32'>;
  };
};

/**
 * Fractional categorical composition using stable, sorted zone/category sums. Coverage, area and
 * proportions are separate outputs; no cell counts or observation weights are inferred. Supports
 * 1..65534 categories and uint32-addressable groups and tuple slots, subject to device binding
 * limits. Fixed-size tuple sorts merge sufficient statistics in tile order before normalisation.
 * The default tile size keeps every previously accepted input in one sort with its original
 * reduction order. Above that size, or with an explicit smaller tupleTileSize, float32 sums can
 * differ by reduction-order rounding; fractions are nonnegative and tile support is merged in a
 * fixed order. The 'exact' mode permits at most 1,048,576 total tuple slots and 65,535 groups.
 * Tiled mode permits at most 4,294,967,294 total tuple slots and groups, categoryCapacity at
 * most 65,534, and at most 1,048,576 tuples per sort tile. Caller outputs and derived scratch
 * must each fit device binding/allocation limits. The 32,768-command budget requires
 * 8 + 256 * tiles <= 32,768, hence at most 127 tiles per declaration.
 * The extra group holds missing support. Active tuples must be unique (zone, cell) pairs, as emitted
 * by GPURasterFractionalCoverage; disjoint tile cores may be concatenated with remapped cell IDs.
 *
 * Empty/fully masked complete zones retain zero class support and explicit missing area, with NaN
 * proportions and invalid winners. Overflow, inconsistent tuple counts, malformed active tuples or
 * out-of-range valid categories fail every zone (NaN float outputs). Failed source geometry fails
 * its zone. Float32 area overflow also fails its zone. Missing area covers nodata inside the supplied footprint, not unmapped polygon area.
 * Areas are planar coordinate units squared by default, or WGS84 ground square meters with a
 * `cellSizeMode` of `'geographic'` or `'web-mercator'`. Graph reuse rereads every input each run.
 */
export class GPURasterFractionalCategoricalStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFractionalCategoricalStatisticsProps;
  readonly tupleTileSize: number;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPURasterFractionalCategoricalStatisticsProps) {
    props = snapshotCategoricalProps(props);
    this.id = props.id ?? 'fractional-categorical-statistics';
    this.props = props;
    validateRasterExpansionPorts(
      this.id,
      props,
      {
        'input.storage.values': ['uint32'],
        'input.validity': ['uint32'],
        parameters: ['float32'],
        'coverage.zoneIds': ['uint32'],
        'coverage.cellIds': ['uint32'],
        'coverage.fractions': ['float32'],
        'coverage.zoneValidity': ['uint32'],
        'coverage.status.count': ['uint32'],
        'coverage.status.requiredCount': ['uint32'],
        'coverage.status.overflow': ['uint32'],
        'coverage.status.candidateOverflow': ['uint32'],
        'output.coverageSums': ['float32'],
        'output.classAreas': ['float32'],
        'output.proportions': ['float32'],
        'output.coveredArea': ['float32'],
        'output.validArea': ['float32'],
        'output.missingArea': ['float32'],
        'output.majority': ['uint32'],
        'output.minority': ['uint32'],
        'output.validity': ['uint32']
      },
      [
        'input.validity',
        'parameters',
        'coverage.status.candidateOverflow',
        ...(props.input.storage.kind === 'texture' ? ['input.storage.values'] : [])
      ]
    );
    validateRasterTextureBand(this.id, props.input, props.grid);
    const selection = getCategoricalShapePlan(props);
    for (const constraint of selection.constraints) {
      validateRequirement(
        this.id,
        constraint.satisfied,
        `${constraint.quantity} in ${selection.route} mode received ${constraint.received}; must be in [${constraint.minimum ?? 0}, ${constraint.maximum}]; ${constraint.alternative ?? 'reduce the workload'}`
      );
    }
    this.tupleTileSize = selection.tupleTileSize;
    this.estimatedCommandNodeCount = selection.estimatedCommandNodeCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    const {grid, input, coverage, output, categoryCapacity, zoneCount} = props;
    const cellCount = validateRasterExpansionViews(
      this.id,
      grid,
      [...getRasterBandViews(input), ...getGraphOutputViews(coverage), props.parameters],
      getGraphOutputViews(output)
    );
    validateRasterFractionalWritableViews(this.id, props.output, [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(props.coverage),
      props.parameters
    ]);
    if (props.parameters) {
      validateRasterParameters(this.id, props.parameters, GEOREFERENCE_DETERMINANT_SLOT + 1);
    }
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, grid, props.cellSizeMode);
    if ((props.cellSizeMode ?? 'uniform') !== 'uniform' && !props.parameters) {
      const {northEdge, southEdge} = getRasterGridGeoreference(grid);
      validateRequirement(
        this.id,
        Number.isFinite(Math.fround(northEdge)) && Number.isFinite(Math.fround(southEdge)),
        'grid.affine row edges must fit in finite float32 for cellSizeMode'
      );
    }
    const cellArea = Math.fround(
      Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
    );
    // A category is an identity: implicit calibration must never merge or alter IDs.
    validateRequirement(
      this.id,
      input.format === 'uint32' && (input.scale ?? 1) === 1 && (input.offset ?? 0) === 0,
      "input must be a 'uint32' band without scale or offset"
    );
    validateRequirement(
      this.id,
      Number.isFinite(cellArea) && cellArea > 0,
      'grid.affine must have a finite nonzero determinant in float32'
    );
    validateRequirement(
      this.id,
      input.noDataValue === undefined ||
        (Number.isInteger(input.noDataValue) &&
          input.noDataValue >= 0 &&
          input.noDataValue <= 0xffffffff),
      'input.noDataValue must be an unsigned 32-bit integer'
    );
    for (const view of [
      ...(input.storage.kind === 'buffer' ? [input.storage.values] : []),
      input.validity,
      coverage.zoneIds,
      coverage.cellIds,
      coverage.zoneValidity,
      ...Object.values(coverage.status),
      output.majority,
      output.minority,
      output.validity
    ]) {
      if (view) validateSinglePackedView(view, ['uint32'], this.id);
    }
    for (const view of [
      coverage.fractions,
      output.coverageSums,
      output.classAreas,
      output.proportions,
      output.coveredArea,
      output.validArea,
      output.missingArea
    ]) {
      validateSinglePackedView(view, ['float32'], this.id);
    }
    if (input.storage.kind === 'buffer') {
      requireRasterLength(input.storage.values, cellCount, this.id, 'input values');
    }
    requireRasterLength(input.validity, cellCount, this.id, 'input.validity');
    requireRasterLength(coverage.cellIds, coverage.zoneIds.length, this.id, 'coverage.cellIds');
    requireRasterLength(coverage.fractions, coverage.zoneIds.length, this.id, 'coverage.fractions');
    requireRasterLength(coverage.zoneValidity, zoneCount, this.id, 'coverage.zoneValidity');
    for (const [name, view] of Object.entries(coverage.status)) {
      requireRasterLength(view, 1, this.id, `coverage.status.${name}`);
    }
    for (const name of ['coverageSums', 'classAreas', 'proportions'] as const) {
      requireRasterLength(output[name], zoneCount * categoryCapacity, this.id, `output.${name}`);
    }
    for (const name of [
      'coveredArea',
      'validArea',
      'missingArea',
      'majority',
      'minority',
      'validity'
    ] as const) {
      requireRasterLength(output[name], zoneCount, this.id, `output.${name}`);
    }
  }

  /** Side-effect-free route and device fit check; no resources or command nodes are declared.
   * Invalid option spellings/ tile sizes throw; supported workloads report feasible and constraints.
   */
  static getExecutionPlan(
    props: GPURasterFractionalCategoricalStatisticsProps,
    deviceLimits: Pick<Device['limits'], 'maxStorageBufferBindingSize' | 'maxBufferSize'> &
      Partial<
        Pick<
          Device['limits'],
          | 'maxStorageBuffersPerShaderStage'
          | 'maxComputeWorkgroupSizeX'
          | 'maxComputeInvocationsPerWorkgroup'
          | 'maxComputeWorkgroupStorageSize'
          | 'maxSampledTexturesPerShaderStage'
        >
      >
  ) {
    props = snapshotCategoricalProps(props);
    validateRasterCellSizeMode(props.id ?? 'fractional-categorical-statistics', props.cellSizeMode);
    const plan = getCategoricalShapePlan(props);
    const constraints = [...plan.constraints];
    // Construction is the shared side-effect-free port/option/alias validator. A plan must not
    // certify an input that the constructor rejects; no graph resources are declared here.
    if (plan.feasible) {
      try {
        new GPURasterFractionalCategoricalStatistics(props);
      } catch (error) {
        constraints.push(
          categoricalConstraint(
            error instanceof Error ? error.message : String(error),
            0,
            1,
            1,
            'output',
            plan.route
          )
        );
      }
    }
    for (const [name, count] of getCategoricalBindingCounts(props))
      constraints.push(
        categoricalConstraint(
          `${name} storage binding count`,
          count,
          0,
          deviceLimits.maxStorageBuffersPerShaderStage ?? 8,
          'dispatch',
          plan.route
        )
      );
    if (props.input.storage.kind === 'texture')
      constraints.push(
        categoricalConstraint(
          'input sampled texture binding count',
          1,
          0,
          deviceLimits.maxSampledTexturesPerShaderStage ?? 16,
          'dispatch',
          plan.route
        )
      );
    // The published segment reducer uses 256 lanes and one 256-float workgroup table. The sort
    // also uses up to 256 lanes; conforming WebGPU devices provide at least these resources.
    for (const [name, received, maximum] of [
      ['workgroup size X', 256, deviceLimits.maxComputeWorkgroupSizeX ?? 256],
      ['workgroup invocations', 256, deviceLimits.maxComputeInvocationsPerWorkgroup ?? 256],
      [
        'segment workgroup storage bytes',
        1024,
        deviceLimits.maxComputeWorkgroupStorageSize ?? 16384
      ]
    ] as const)
      constraints.push(categoricalConstraint(name, received, 0, maximum, 'dispatch', plan.route));

    const views = [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(props.coverage),
      props.parameters,
      ...getGraphOutputViews(props.output)
    ];
    const allocations: [string, number, number][] = views
      .filter(view => Boolean(view))
      .map(view => [view!.buffer.id, getViewBindingRange(view!).size, view!.buffer.byteLength]);
    for (const [name, bytes] of getCategoricalScratchByteLengths(props, plan.tupleTileSize))
      allocations.push([name, bytes, bytes]);
    for (const [name, bindingBytes, bufferBytes] of allocations) {
      constraints.push(
        categoricalConstraint(
          `${name} binding bytes`,
          bindingBytes,
          0,
          deviceLimits.maxStorageBufferBindingSize,
          'binding',
          plan.route
        )
      );
      constraints.push(
        categoricalConstraint(
          `${name} buffer bytes`,
          Math.max(bufferBytes, 4),
          0,
          deviceLimits.maxBufferSize,
          'binding',
          plan.route
        )
      );
    }
    return {...plan, feasible: constraints.every(constraint => constraint.satisfied), constraints};
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, coverage, output, zoneCount, categoryCapacity} = props;
    const input = props.input;
    validateRasterBandBelongsToGraph(id, graph, input, []);
    validateRasterStorageBindings(
      graph,
      id,
      [
        ...getRasterBandViews(input),
        ...getGraphOutputViews(coverage),
        props.parameters,
        ...getGraphOutputViews(output)
      ],
      {'canonical band planes': props.grid.width * props.grid.height * 4}
    );
    // Binding ranges include the alignment prefix of nonzero-offset caller views. Preflight
    // every allocation before declaring a resource or node, including the sort's derived scratch.
    for (const view of [
      ...getRasterBandViews(input),
      ...getGraphOutputViews(coverage),
      props.parameters,
      ...getGraphOutputViews(output)
    ]) {
      if (view)
        validateRasterFractionalBindingByteLength(
          graph.device,
          id,
          view.buffer.id,
          getViewBindingRange(view).size,
          'split the raster or coverage into disjoint cores',
          view.buffer.byteLength
        );
    }
    for (const [name, bytes] of getCategoricalScratchByteLengths(props, this.tupleTileSize))
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce zoneCount, categoryCapacity or tupleTileSize'
      );
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(input),
      ...getGraphOutputViews(coverage),
      props.parameters,
      ...getGraphOutputViews(output)
    ]);
    const executionPlan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(
      props,
      graph.device.limits
    );
    for (const constraint of executionPlan.constraints) {
      validateRequirement(
        id,
        constraint.satisfied,
        `${constraint.quantity} received ${constraint.received}; maximum ${constraint.maximum}; reduce zoneCount, categoryCapacity or tupleTileSize`
      );
    }
    const sourceNodes: GPUCommandNode<Parameters>[] = [];
    let nativeInput: GPURasterBufferBand<'uint32'>;
    if (input.storage.kind === 'texture') {
      const textureInput = {...input, storage: input.storage};
      const values = createTransientView(
        graph,
        `${id}-native-values`,
        'uint32',
        grid.width * grid.height
      );
      const validity = createTransientView(
        graph,
        `${id}-native-validity`,
        'uint32',
        grid.width * grid.height
      );
      sourceNodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterTextureToBuffer({
            id: `${id}-native`,
            input: textureInput,
            output: values,
            outputValidity: validity,
            applyCalibration: false
          }).addToGraph(graph)
        )
      );
      nativeInput = {...input, storage: {kind: 'buffer', values}, validity};
    } else nativeInput = {...input, storage: input.storage};
    const cellCount = grid.width * grid.height;
    const tupleCapacity = coverage.zoneIds.length;
    const groupCount = zoneCount * (categoryCapacity + 1);
    // Slot 6 of the coverage georeference is the affine determinant; read per frame when supplied.
    const cellArea = props.parameters
      ? `abs(georeference[georeferenceOffset + ${GEOREFERENCE_DETERMINANT_SLOT}u])`
      : getWGSLFloatLiteral(
          Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
        );
    const georeferenceBindings = props.parameters
      ? [
          {
            name: 'georeference',
            view: props.parameters,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const areaScaleDeclarations =
      cellSizeMode === 'uniform'
        ? undefined
        : getRasterRowAreaScaleWGSL(
            grid,
            cellSizeMode,
            props.parameters ? 'georeference' : undefined
          );
    const areaScaleExpression =
      cellSizeMode === 'uniform' ? 'fraction' : `fraction * getRowAreaScale(cell / ${grid.width}u)`;
    const categories = createTransientView(graph, `${id}-categories`, 'uint32', cellCount);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', this.tupleTileSize);
    const contributions = createTransientView(
      graph,
      `${id}-contributions`,
      'float32',
      this.tupleTileSize
    );
    // State: failure word, current tuple base, active tuple count.
    const failure = createTransientView(graph, `${id}-failure`, 'uint32', 3);
    const tileSums = createTransientView(graph, `${id}-tile-sums`, 'float32', groupCount);
    const sums = createTransientView(graph, `${id}-sums`, 'float32', groupCount);
    const summary = createTransientView(graph, `${id}-summary`, 'float32x4', zoneCount);
    const operation = 'GPURasterFractionalCategoricalStatistics';
    const nodes: GPUCommandNode<Parameters>[] = [
      ...sourceNodes,
      createFillNode(graph, {
        id: `${id}-clear-sums`,
        operation,
        view: sums,
        type: 'f32',
        value: '0.0',
        componentCount: groupCount
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-classify`,
        operation,
        invocationCount: cellCount,
        bindings: [
          {name: 'values', view: nativeInput.storage.values, type: 'u32', access: 'read'},
          ...(nativeInput.validity
            ? [
                {
                  name: 'mask',
                  view: nativeInput.validity,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'categories', view: categories, type: 'u32', access: 'read_write'}
        ],
        body: `let category = values[valuesOffset + index];
let missing = ${nativeInput.validity ? 'mask[maskOffset + index] == 0u' : 'false'}${input.noDataValue === undefined ? '' : ` || category == ${input.noDataValue}u`};
categories[categoriesOffset + index] = select(min(category, ${categoryCapacity + 1}u), ${categoryCapacity}u, missing);
// Unmasked categoryCapacity is invalid, distinct from the missing bucket.
if (!missing && category >= ${categoryCapacity}u) { categories[categoriesOffset + index] = ${categoryCapacity + 1}u; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-reset-status`,
        operation,
        invocationCount: 1,
        bindings: [
          ...(cellSizeMode !== 'uniform' ? georeferenceBindings : []),
          {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
          {name: 'required', view: coverage.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
          ...(coverage.status.candidateOverflow
            ? [
                {
                  name: 'candidateOverflow',
                  view: coverage.status.candidateOverflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'failure', view: failure, type: 'u32', access: 'read_write'}
        ],
        body: `failure[failureOffset] = u32(!(${getRasterEffectiveAxisAlignmentWGSL(grid, cellSizeMode, props.parameters ? 'georeference' : undefined)}) || overflow[overflowOffset] != 0u ${coverage.status.candidateOverflow ? '|| candidateOverflow[candidateOverflowOffset] != 0u' : ''} || count[countOffset] > ${tupleCapacity}u || required[requiredOffset] != count[countOffset]);
failure[failureOffset+1u]=0u;failure[failureOffset+2u]=min(count[countOffset],${tupleCapacity}u);`
      })
    ];
    for (let tile = 0; tile < Math.ceil(tupleCapacity / this.tupleTileSize); tile++) {
      const tileId = `${id}-tile-${tile}`;
      const tileLength = Math.min(this.tupleTileSize, tupleCapacity - tile * this.tupleTileSize);
      const tileKeys = graph.createDataView(keys.buffer, {format: 'uint32', length: tileLength});
      const tileContributions = graph.createDataView(contributions.buffer, {
        format: 'float32',
        length: tileLength
      });
      nodes.push(
        // The published sort consumes fixed scratch, whose inactive suffix sorts after all groups.
        createFillNode(graph, {
          id: `${tileId}-clear-keys`,
          operation,
          view: keys,
          type: 'u32',
          value: `${groupCount}u`,
          componentCount: this.tupleTileSize
        }),
        createFillNode(graph, {
          id: `${tileId}-clear-contributions`,
          operation,
          view: contributions,
          type: 'f32',
          value: '0.0',
          componentCount: this.tupleTileSize
        }),
        createRasterFractionalTileNode(graph, {
          id: `${tileId}-prepare-tuples`,
          operation,
          invocationCount: tileLength,
          rowCount: 'arrayLength(&keys)',
          declarations: areaScaleDeclarations,
          bindings: [
            ...georeferenceBindings,
            {name: 'categories', view: categories, type: 'u32', access: 'read'},
            {name: 'zones', view: coverage.zoneIds, type: 'u32', access: 'read'},
            {name: 'cells', view: coverage.cellIds, type: 'u32', access: 'read'},
            {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
            {name: 'keys', view: tileKeys, type: 'u32', access: 'read_write'},
            {name: 'contributions', view: tileContributions, type: 'f32', access: 'read_write'},
            {name: 'failure', view: failure, type: 'atomic<u32>', access: 'read_write'}
          ],
          body: `keys[keysOffset + index] = ${groupCount}u;
contributions[contributionsOffset + index] = 0.0;
let tuple=atomicLoad(&failure[failureOffset+1u])+index;
if (tuple >= atomicLoad(&failure[failureOffset+2u])) { return; }
let zone = zones[zonesOffset + tuple];
let cell = cells[cellsOffset + tuple];
let fraction = fractions[fractionsOffset + tuple];
if (zone >= ${zoneCount}u || cell >= ${cellCount}u || !(fraction >= 0.0 && fraction <= 1.0)) {
  atomicOr(&failure[failureOffset], 1u); return;
}
if (fraction == 0.0) { return; }
let category = categories[categoriesOffset + cell];
if (category > ${categoryCapacity}u) { atomicOr(&failure[failureOffset], 1u); return; }
keys[keysOffset + index] = zone * ${categoryCapacity + 1}u + category;
contributions[contributionsOffset + index] = ${areaScaleExpression};`
        }),
        ...getSortedSegmentSumNodes(graph, {
          id: `${tileId}-group`,
          operation,
          segmentCount: groupCount,
          segmentKeys: keys,
          sumContributions: contributions,
          sums: tileSums
        }),
        createWGSLKernelNode(graph, {
          id: `${tileId}-merge`,
          operation,
          invocationCount: groupCount,
          bindings: [
            {name: 'partialSums', view: tileSums, type: 'f32', access: 'read'},
            {name: 'sums', view: sums, type: 'f32', access: 'read_write'}
          ],
          body: 'sums[sumsOffset+index]+=partialSums[partialSumsOffset+index];'
        }),
        createWGSLKernelNode(graph, {
          id: `${tileId}-advance`,
          operation,
          invocationCount: 1,
          bindings: [{name: 'state', view: failure, type: 'u32', access: 'read_write'}],
          body: `state[stateOffset+1u]+=min(${this.tupleTileSize}u,0xffffffffu-state[stateOffset+1u]);`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-summarize`,
        operation,
        invocationCount: zoneCount,
        bindings: [
          {name: 'sums', view: sums, type: 'f32', access: 'read'},
          {name: 'failure', view: failure, type: 'u32', access: 'read'},
          {name: 'zoneValidity', view: coverage.zoneValidity, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
          {name: 'majority', view: output.majority, type: 'u32', access: 'read_write'},
          {name: 'minority', view: output.minority, type: 'u32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          ...georeferenceBindings
        ],
        body: `let base = sumsOffset + index * ${categoryCapacity + 1}u;
var observed = 0.0;
var greatest = 0.0;
var least = 3.4028234663852886e38;
var majorityCategory = 0xffffffffu;
var minorityCategory = 0xffffffffu;
for (var category = 0u; category < ${categoryCapacity}u; category++) {
  let support = sums[base + category];
  observed += support;
  if (support > 0.0 && support >= greatest) { greatest = support; majorityCategory = category; }
  if (support > 0.0 && support < least) { least = support; minorityCategory = category; }
}
let missing = sums[base + ${categoryCapacity}u];
let coveredArea = (observed + missing) * ${cellArea};
let finiteArea = coveredArea <= 3.4028234663852886e38;
let complete = failure[failureOffset] == 0u && zoneValidity[zoneValidityOffset + index] != 0u && finiteArea;
let valid = complete && observed > 0.0;
let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
summary[summaryOffset + index * 4u] = select(invalid, observed + missing, complete);
summary[summaryOffset + index * 4u + 1u] = select(invalid, observed, complete);
summary[summaryOffset + index * 4u + 2u] = select(invalid, missing, complete);
summary[summaryOffset + index * 4u + 3u] = f32(complete);
majority[majorityOffset + index] = select(0xffffffffu, majorityCategory, valid);
minority[minorityOffset + index] = select(0xffffffffu, minorityCategory, valid);
validity[validityOffset + index] = u32(valid);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish-classes`,
        operation,
        invocationCount: zoneCount * categoryCapacity,
        bindings: [
          {name: 'sums', view: sums, type: 'f32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'coverage', view: output.coverageSums, type: 'f32', access: 'read_write'},
          {name: 'areas', view: output.classAreas, type: 'f32', access: 'read_write'},
          {name: 'proportions', view: output.proportions, type: 'f32', access: 'read_write'},
          ...georeferenceBindings
        ],
        body: `let zone = index / ${categoryCapacity}u;
let category = index % ${categoryCapacity}u;
let observed = summary[summaryOffset + zone * 4u + 1u];
let complete = summary[summaryOffset + zone * 4u + 3u] != 0.0;
let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
let support = sums[sumsOffset + zone * ${categoryCapacity + 1}u + category];
coverage[coverageOffset + index] = select(invalid, support, complete);
areas[areasOffset + index] = select(invalid, support * ${cellArea}, complete);
proportions[proportionsOffset + index] = select(invalid, support / select(1.0, observed, observed > 0.0), complete && observed > 0.0);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish-areas`,
        operation,
        invocationCount: zoneCount,
        bindings: [
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'covered', view: output.coveredArea, type: 'f32', access: 'read_write'},
          {name: 'observed', view: output.validArea, type: 'f32', access: 'read_write'},
          {name: 'missing', view: output.missingArea, type: 'f32', access: 'read_write'},
          ...georeferenceBindings
        ],
        body: `covered[coveredOffset + index] = summary[summaryOffset + index * 4u] * ${cellArea};
observed[observedOffset + index] = summary[summaryOffset + index * 4u + 1u] * ${cellArea};
missing[missingOffset + index] = summary[summaryOffset + index * 4u + 2u] * ${cellArea};`
      })
    );
    return nodes;
  }
}

/** Shared selection and logical limits, used by declaration and the public planner. */
function getCategoricalShapePlan(props: GPURasterFractionalCategoricalStatisticsProps) {
  const id = props.id ?? 'fractional-categorical-statistics';
  validateRequirement(
    id,
    props.mode === undefined || props.mode === 'tiled' || props.mode === 'exact',
    "mode must be 'tiled' or 'exact'"
  );
  if (props.tupleTileSize !== undefined)
    validateCapacity(id, 'tupleTileSize', props.tupleTileSize, 1, MAXIMUM_TUPLE_TILE_SIZE);
  const route = props.mode ?? 'tiled';
  const tuples = props.coverage.zoneIds.length;
  const tupleTileSize =
    route === 'exact' ? tuples : (props.tupleTileSize ?? Math.min(DEFAULT_TUPLE_TILE_SIZE, tuples));
  const tiles = Math.ceil(tuples / Math.max(tupleTileSize, 1));
  const estimatedCommandNodeCount =
    CATEGORICAL_COMMAND_NODE_ALLOWANCE + tiles * TUPLE_TILE_COMMAND_NODE_ALLOWANCE;
  const constraints = [
    categoricalConstraint(
      'categoryCapacity',
      props.categoryCapacity,
      1,
      MAXIMUM_CATEGORY_CAPACITY,
      'domain',
      route
    ),
    categoricalConstraint(
      'zoneCount',
      props.zoneCount,
      1,
      MAXIMUM_ZONE_CATEGORY_GROUPS,
      'domain',
      route
    ),
    categoricalConstraint(
      'zoneCount times (categoryCapacity + 1)',
      props.zoneCount * (props.categoryCapacity + 1),
      1,
      route === 'exact' ? MAXIMUM_REFERENCE_ZONE_CATEGORY_GROUPS : MAXIMUM_ZONE_CATEGORY_GROUPS,
      'domain',
      route
    ),
    categoricalConstraint(
      'coverage.zoneIds length',
      tuples,
      1,
      route === 'exact' ? MAXIMUM_REFERENCE_TUPLES : MAXIMUM_TUPLES,
      'domain',
      route
    ),
    categoricalConstraint(
      'tupleTileSize',
      tupleTileSize,
      1,
      MAXIMUM_TUPLE_TILE_SIZE,
      'dispatch',
      route
    ),
    categoricalConstraint(
      'command node count (8 + 256 * tiles)',
      estimatedCommandNodeCount,
      0,
      GPU_MAXIMUM_COMMAND_NODE_COUNT,
      'dispatch',
      route
    )
  ];
  if (route === 'exact' && props.tupleTileSize !== undefined)
    constraints.push(
      categoricalConstraint(
        'exact tupleTileSize must cover tuple capacity',
        tuples,
        1,
        props.tupleTileSize,
        'dispatch',
        route
      )
    );
  return {
    route,
    tupleTileSize,
    tiles,
    estimatedCommandNodeCount,
    constraints,
    feasible: constraints.every(constraint => constraint.satisfied),
    accuracyImplications: [
      'Exact planar coverage; changing sort tile size can change float32 sum rounding.'
    ]
  };
}

function categoricalConstraint(
  quantity: string,
  received: number,
  minimum: number,
  maximum: number,
  scope: GPUAnalysisRoute['scope'],
  route: string
): GPUAnalysisRoute & {satisfied: boolean; received: number} {
  return {
    name: quantity,
    kind: scope === 'binding' ? 'allocation' : 'hard-limit',
    quantity,
    unit: scope === 'binding' ? 'bytes' : quantity.startsWith('command') ? 'commands' : 'elements',
    appliesWhen: `mode is ${route}`,
    minimum,
    maximum,
    scope,
    alternative:
      route === 'exact'
        ? 'use tiled mode for larger logical capacities'
        : 'reduce the workload or split disjoint cores',
    accuracyImplications: 'Sort tile changes may change float32 reduction rounding.',
    planner: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterFractionalCategoricalStatistics'
    },
    source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts',
    satisfied: Number.isSafeInteger(received) && received >= minimum && received <= maximum,
    received
  };
}

/** Identical derived binding sizes for the public preflight and command declaration. */
function getCategoricalScratchByteLengths(
  props: GPURasterFractionalCategoricalStatisticsProps,
  tupleTileSize: number
): readonly (readonly [string, number])[] {
  const groups = props.zoneCount * (props.categoryCapacity + 1);
  return [
    ['canonical band planes', props.grid.width * props.grid.height * 4],
    ['categories', props.grid.width * props.grid.height * 4],
    [
      'sort tiles',
      (tupleTileSize <= 256 ? 2 ** Math.ceil(Math.log2(tupleTileSize)) : tupleTileSize) * 4
    ],
    ['segment offsets', (groups + 1) * 4],
    ['group sums', groups * 4],
    ['zone summary', props.zoneCount * 16]
  ];
}

/** Capture declared ports even when inherited/non-enumerable; preserve GPU resource identities. */
function snapshotCategoricalProps(
  props: GPURasterFractionalCategoricalStatisticsProps
): GPURasterFractionalCategoricalStatisticsProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'grid',
    'input',
    'zoneCount',
    'categoryCapacity',
    'tupleTileSize',
    'mode',
    'coverage',
    'parameters',
    'cellSizeMode',
    'output'
  ]);
  const coverage = snapshotConstructionFields(captured.coverage, [
    'zoneIds',
    'cellIds',
    'fractions',
    'zoneValidity',
    'status'
  ]);
  return {
    ...captured,
    grid: snapshotRasterGrid(captured.grid),
    input: snapshotRasterBand(captured.input) as GPURasterBand<'uint32'>,
    coverage: {
      ...coverage,
      status: snapshotConstructionFields(coverage.status, [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow'
      ])
    },
    output: snapshotConstructionFields(captured.output, [
      'coverageSums',
      'classAreas',
      'proportions',
      'coveredArea',
      'validArea',
      'missingArea',
      'majority',
      'minority',
      'validity'
    ])
  };
}

/** Counts match the actual binding lists below and are checked before any declaration. */
function getCategoricalBindingCounts(
  props: GPURasterFractionalCategoricalStatisticsProps
): readonly (readonly [string, number])[] {
  const georeferenceCount = props.parameters ? 1 : 0;
  return [
    [
      'classify',
      2 + Number(Boolean(props.input.validity) || props.input.storage.kind === 'texture')
    ],
    [
      'reset-status',
      4 +
        Number(Boolean(props.coverage.status.candidateOverflow)) +
        ((props.cellSizeMode ?? 'uniform') !== 'uniform' ? georeferenceCount : 0)
    ],
    ['prepare-tuples', 7 + georeferenceCount],
    ['summarize', 7 + georeferenceCount],
    ['publish-classes', 5 + georeferenceCount],
    ['publish-areas', 4 + georeferenceCount]
  ];
}
