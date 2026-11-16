// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  validateCapacity,
  validateCommandNodeBudget,
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
  validateRasterGridAxisAligned,
  validateRasterParameters
} from './raster-band-utils';
import {
  getRasterGridGeoreference,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../analysis-core/raster-grid';
import {getRasterRowAreaScaleWGSL} from './raster-grid/raster-cell-area-scale';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';
import type {GPURasterFractionalCoverageProps} from './gpu-raster-fractional-coverage';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/**
 * Slot of the affine determinant in the georeference view of `GPURasterFractionalCoverage`
 * (`[originX, originY, a, b, d, e, determinant, 0]`), the only word this class reads from it.
 */
const GEOREFERENCE_DETERMINANT_SLOT = 6;

/** Largest category capacity; dense categories bound scratch space and the per-zone winner scan. */
const MAXIMUM_CATEGORY_CAPACITY = 65534;

/** uint32 group keys reserve the last word for inactive tuple slots. */
const MAXIMUM_ZONE_CATEGORY_GROUPS = 0xfffffffe;

/** uint32 tuple row indices reserve the last word for saturated status counts. */
const MAXIMUM_TUPLES = 0xfffffffe;

/** Full-sort reference limit preserves the original reduction order and scratch bound. */
const MAXIMUM_REFERENCE_TUPLES = 1048576;

/** The original reference route keeps its historical dense group cap. */
const MAXIMUM_REFERENCE_ZONE_CATEGORY_GROUPS = 65535;

/** Sort tiles retain the historical 1M-row bound on any single sort allocation. */
const MAXIMUM_TUPLE_TILE_SIZE = 1048576;

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
  input: GPURasterBufferBand<'uint32'>;
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
   * ground area per world area of its row (row edges from `grid.affine`, evaluated at the row
   * center), so `coverageSums` then holds weighted coverage and `proportions` are area
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
 * fixed order. The 'exact' mode retains the original 1M tuple / 65535 group reference limits.
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
    this.id = props.id ?? 'fractional-categorical-statistics';
    this.props = props;
    validateRequirement(
      this.id,
      props.mode === undefined || props.mode === 'tiled' || props.mode === 'exact',
      "mode must be 'tiled' or 'exact'"
    );
    if (props.tupleTileSize !== undefined)
      validateCapacity(this.id, 'tupleTileSize', props.tupleTileSize, 1, MAXIMUM_TUPLE_TILE_SIZE);
    if (props.mode === 'exact') {
      validateCapacity(
        this.id,
        'coverage.zoneIds length in exact mode',
        props.coverage.zoneIds.length,
        1,
        MAXIMUM_REFERENCE_TUPLES
      );
      validateCapacity(
        this.id,
        'zoneCount times (categoryCapacity + 1) in exact mode',
        props.zoneCount * (props.categoryCapacity + 1),
        1,
        MAXIMUM_REFERENCE_ZONE_CATEGORY_GROUPS
      );
      validateRequirement(
        this.id,
        props.tupleTileSize === undefined || props.tupleTileSize >= props.coverage.zoneIds.length,
        'tupleTileSize in exact mode must cover coverage.zoneIds.length'
      );
    }
    this.tupleTileSize =
      props.mode === 'exact'
        ? props.coverage.zoneIds.length
        : (props.tupleTileSize ?? Math.min(DEFAULT_TUPLE_TILE_SIZE, props.coverage.zoneIds.length));
    validateCapacity(this.id, 'tupleTileSize', this.tupleTileSize, 1, MAXIMUM_TUPLE_TILE_SIZE);
    this.estimatedCommandNodeCount =
      CATEGORICAL_COMMAND_NODE_ALLOWANCE +
      Math.ceil(props.coverage.zoneIds.length / this.tupleTileSize) *
        TUPLE_TILE_COMMAND_NODE_ALLOWANCE;
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
    // Dense categories bound scratch space and the per-zone winner scan.
    validateCapacity(this.id, 'categoryCapacity', categoryCapacity, 1, MAXIMUM_CATEGORY_CAPACITY);
    validateRequirement(
      this.id,
      Number.isInteger(zoneCount) &&
        zoneCount >= 1 &&
        zoneCount * (categoryCapacity + 1) <= MAXIMUM_ZONE_CATEGORY_GROUPS,
      `zoneCount must be an integer of at least 1 with zoneCount times (categoryCapacity + 1) at most ${MAXIMUM_ZONE_CATEGORY_GROUPS}`
    );
    validateCapacity(
      this.id,
      'coverage.zoneIds length',
      coverage.zoneIds.length,
      1,
      MAXIMUM_TUPLES
    );
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
      input.storage.values,
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
    requireRasterLength(input.storage.values, cellCount, this.id, 'input values');
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

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, input, coverage, output, zoneCount, categoryCapacity} = props;
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
    for (const [name, bytes] of [
      ['categories', grid.width * grid.height * 4],
      [
        'sort tiles',
        (this.tupleTileSize <= 256
          ? 2 ** Math.ceil(Math.log2(this.tupleTileSize))
          : this.tupleTileSize) * 4
      ],
      ['segment offsets', (zoneCount * (categoryCapacity + 1) + 1) * 4],
      ['group sums', zoneCount * (categoryCapacity + 1) * 4],
      ['zone summary', zoneCount * 16]
    ] as const)
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
          {name: 'values', view: input.storage.values, type: 'u32', access: 'read'},
          ...(input.validity
            ? [{name: 'mask', view: input.validity, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'categories', view: categories, type: 'u32', access: 'read_write'}
        ],
        body: `let category = values[valuesOffset + index];
let missing = ${input.validity ? 'mask[maskOffset + index] == 0u' : 'false'}${input.noDataValue === undefined ? '' : ` || category == ${input.noDataValue}u`};
categories[categoriesOffset + index] = select(min(category, ${categoryCapacity + 1}u), ${categoryCapacity}u, missing);
// Unmasked categoryCapacity is invalid, distinct from the missing bucket.
if (!missing && category >= ${categoryCapacity}u) { categories[categoriesOffset + index] = ${categoryCapacity + 1}u; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-reset-status`,
        operation,
        invocationCount: 1,
        bindings: [
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
        body: `failure[failureOffset] = u32(overflow[overflowOffset] != 0u ${coverage.status.candidateOverflow ? '|| candidateOverflow[candidateOverflowOffset] != 0u' : ''} || count[countOffset] > ${tupleCapacity}u || required[requiredOffset] != count[countOffset]);
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
