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
  validateFiniteNumber,
  validateSinglePackedView,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {createStableLexicographicIndexSortNodes} from '../analysis-core/kernels/stable-lexicographic-index-sort';
import {createSortedSegmentOffsetsNode} from '../analysis-core/kernels/sorted-segment-offsets';
import {validateGraphChunkBindingRanges} from '../analysis-core/kernels/graph-chunk-dispatch';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from './raster-band-utils';
import {validateRasterCellSizeMode, type GPURasterCellSizeMode} from '../analysis-core/raster-grid';
import {getRasterRowAreaScaleWGSL} from './raster-grid/raster-cell-area-scale';
import {validateRasterExpansionGraph} from './raster-expansion-utils';
import {
  GPURaster,
  type GPURasterBufferBand,
  type GPURasterMetadata
} from '@luma.gl/graph/gpu-raster';
import type {GPURasterFractionalCoverageProps} from './gpu-raster-fractional-coverage';
import {
  validateRasterFractionalBindingByteLength,
  validateRasterFractionalWritableViews
} from './fractional-coverage/raster-fractional-tile-kernel';

/**
 * Slot of the affine determinant in the georeference view of `GPURasterFractionalCoverage`
 * (`[originX, originY, a, b, d, e, determinant, 0]`), the only word this class reads from it.
 */
const GEOREFERENCE_DETERMINANT_SLOT = 6;

/** Exact bounded tuple processing; larger domains require explicit partition preparation. */
export const GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_MAXIMUM_TUPLES = 65536;
/** Maximum dynamic quantile probabilities in one invocation. */
export const GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_MAXIMUM_QUANTILES = 32;

/** Per-zone status bits; singleton zones can have valid population results and bit 8 set. */
export const GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_STATUS = {
  INCOMPLETE_COVERAGE: 1,
  INVALID_TUPLE: 2,
  NO_VALID_SUPPORT: 4,
  INSUFFICIENT_SAMPLE_SUPPORT: 8,
  NON_FINITE_RESULT: 16
} as const;

export type GPURasterFractionalZonalDistributionOutput = {
  means: GraphDataView<'float32'>;
  populationVariances: GraphDataView<'float32'>;
  /** Reliability-weighted sample variance, M2 / (W - W2 / W). */
  sampleVariances: GraphDataView<'float32'>;
  /** Kish support W * W / W2; each distinct source cell is one observation. */
  effectiveSampleSizes: GraphDataView<'float32'>;
  weightSums: GraphDataView<'float32'>;
  squaredWeightSums: GraphDataView<'float32'>;
  coveredArea: GraphDataView<'float32'>;
  validArea: GraphDataView<'float32'>;
  /** Zone-major rows: zone * probabilities.length + probability index. */
  quantiles: GraphDataView<'float32'>;
  /** Same shape as quantiles; invalid probabilities produce NaN and 0 here. */
  quantileValidity: GraphDataView<'uint32'>;
  /** Valid mean/population variance/quantiles; sample variance also requires status bit 8 clear. */
  validity: GraphDataView<'uint32'>;
  /** Per-zone bit mask, rewritten on every encoding. */
  status: GraphDataView<'uint32'>;
};

export type GPURasterFractionalZonalDistributionProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBufferBand;
  zoneCount: number;
  /** Independent nonnegative per-cell reliability weights; default one. */
  observationWeights?: GraphDataView<'float32'>;
  /** Positive per-cell areas in caller-declared units; default planar affine determinant. */
  cellAreas?: GraphDataView<'float32'>;
  /**
   * The per-frame georeference view given to `GPURasterFractionalCoverage` (layout of
   * `getGPURasterFractionalCoverageParameterValues`). Without `cellAreas`, the cell area is read
   * from it on every encoding as `abs(determinant)`, so the area outputs follow a repacked
   * georeference. When both are omitted the cell area is `abs(det(grid.affine))` fixed at compile
   * time: pass one of them whenever the coverage `parameters` are rewritten for another cell size.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * What the affine's units are, for the default cell area. Default `'uniform'`: planar coordinate
   * units squared, as before. `'geographic'` (degrees) and `'web-mercator'` (EPSG:3857 meters)
   * multiply the cell area by the WGS84 ground area per world area of the cell's row (row edges
   * from `grid.affine`, evaluated at the row center), so areas are ground square meters. They need
   * an axis-aligned grid and cannot be combined with `cellAreas`, which already declares areas.
   * Compile-time mode. The row edges come from `parameters` (origin y and row step y, slots 1 and
   * 5, with `grid.height` and `pixelInterpretation` compile-time) on every encoding when it is
   * given, so the ground scale follows a repacked georeference together with the cell area. Without
   * `parameters` they are baked from `grid.affine`, like the planar cell area.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Exact distinct-value interpolation (default), or the left-continuous inverse weighted CDF. */
  quantileMethod?: 'exactextract' | 'inverse-cdf';
  /** Unique (zone, cell) tuples, as produced by GPURasterFractionalCoverage. */
  coverage: GPURasterFractionalCoverageProps['output'];
  /** Per-frame probabilities in [0, 1], including endpoints; 1 to 32 rows. */
  probabilities: GraphDataView<'float32'>;
  output: GPURasterFractionalZonalDistributionOutput;
};

/** Largest zone count; bounds the per-zone serial scans and the zone-major quantile result. */
const MAXIMUM_ZONES = 65536;

const OPERATION = 'GPURasterFractionalZonalDistribution';
const RECORD_WORDS = 7;
const SUMMARY_WORDS = 11;
/** The published automatic sort uses local bitonic sorting through 256 rows, radix above it. */
const LOCAL_SORT_MAXIMUM_TUPLES = 256;
const NUMBER_SOURCE = `${IS_FINITE_VALUE_WGSL}
fn missingValue(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }`;

/**
 * Coverage times observation-weight distribution of calibrated finite raster values. Quantiles follow the
 * exactextract 0.3.0 distinct-value histogram interpolation convention. Coverage times observation weights are
 * reliability weights for sample variance, and are never frequency-replicated observations.
 * All tuples are sorted on the GPU; no CPU fallback, readback, or implicit tile merge occurs.
 * Duplicate source cells in a zone are rejected rather than double-counting effective support.
 */
export class GPURasterFractionalZonalDistribution implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFractionalZonalDistributionProps;

  constructor(props: GPURasterFractionalZonalDistributionProps) {
    this.id = props.id ?? 'raster-fractional-zonal-distribution';
    this.props = props;
    validateFiniteNumber(this.id, 'input.scale', props.input.scale ?? 1);
    validateFiniteNumber(this.id, 'input.offset', props.input.offset ?? 0);
    new GPURaster({id: `${this.id} input`, metadata: props.grid, bands: [props.input]});
    for (const view of [...getInputViews(props), ...getGraphOutputViews(props.output)]) {
      validateSinglePackedView(view, [view.format], this.id);
    }
    validateRasterFractionalWritableViews(this.id, props.output, getInputViews(props), [
      [props.output.quantiles, props.output.quantileValidity, props.probabilities],
      [
        props.output.means,
        props.output.populationVariances,
        props.output.sampleVariances,
        props.output.effectiveSampleSizes
      ],
      [
        props.output.weightSums,
        props.output.squaredWeightSums,
        props.output.coveredArea,
        props.output.validArea
      ],
      [props.output.validity, props.output.status]
    ]);
    for (const [name, view] of Object.entries(props.output)) {
      validateSinglePackedView(
        view,
        name === 'status' || name === 'validity' || name === 'quantileValidity'
          ? ['uint32']
          : ['float32'],
        `${this.id} output.${name}`
      );
    }
    for (const [name, view] of Object.entries(props.coverage)) {
      if (name !== 'status') {
        validateSinglePackedView(
          view as GraphDataView,
          name === 'fractions' ? ['float32'] : ['uint32'],
          `${this.id} coverage.${name}`
        );
      }
    }
    for (const [name, view] of Object.entries(props.coverage.status)) {
      validateSinglePackedView(view, ['uint32'], `${this.id} coverage.status.${name}`);
    }
    validateSinglePackedView(props.probabilities, ['float32'], `${this.id} probabilities`);
    for (const [name, view] of Object.entries({
      observationWeights: props.observationWeights,
      cellAreas: props.cellAreas
    })) {
      validateRequirement(
        this.id,
        !view ||
          (view.format === 'float32' && view.length === props.grid.width * props.grid.height),
        `${name} must be a float32 view with one entry per grid cell`
      );
    }
    validateRequirement(
      this.id,
      props.quantileMethod === undefined ||
        ['exactextract', 'inverse-cdf'].includes(props.quantileMethod),
      "quantileMethod must be 'exactextract' or 'inverse-cdf'"
    );
    const capacity = props.coverage.zoneIds.length;
    validateCapacity(this.id, 'zoneCount', props.zoneCount, 0, MAXIMUM_ZONES);
    validateCapacity(
      this.id,
      'coverage.zoneIds length',
      capacity,
      1,
      GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_MAXIMUM_TUPLES
    );
    validateCapacity(
      this.id,
      'probabilities length',
      props.probabilities.length,
      1,
      GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_MAXIMUM_QUANTILES
    );
    validateRequirement(
      this.id,
      props.coverage.cellIds.length === capacity && props.coverage.fractions.length === capacity,
      'coverage.cellIds and coverage.fractions must match the length of coverage.zoneIds'
    );
    validateRequirement(
      this.id,
      props.coverage.zoneValidity.length === props.zoneCount,
      'coverage.zoneValidity must hold zoneCount entries'
    );
    for (const [name, view] of Object.entries(props.coverage.status)) {
      validateRequirement(
        this.id,
        view.length === 1,
        `coverage.status.${name} must hold one entry`
      );
    }
    for (const [name, view] of Object.entries(props.output)) {
      const expectedLength =
        name === 'quantiles' || name === 'quantileValidity'
          ? props.zoneCount * props.probabilities.length
          : props.zoneCount;
      validateRequirement(
        this.id,
        view.length === expectedLength,
        `output.${name} must hold ${expectedLength} entries`
      );
    }
    if (props.parameters) {
      validateRasterParameters(this.id, props.parameters, GEOREFERENCE_DETERMINANT_SLOT + 1);
    }
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateRequirement(
      this.id,
      !props.cellAreas || (props.cellSizeMode ?? 'uniform') === 'uniform',
      'cellSizeMode must be uniform when cellAreas is given (cellAreas already declares the areas)'
    );
    const determinant =
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3];
    // Area support requires an invertible finite planar affine transform.
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(Math.abs(determinant) * capacity)) &&
        Math.fround(Math.abs(determinant)) > 0,
      'grid.affine must have a finite nonzero determinant whose total area fits float32'
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {coverage, output, zoneCount, probabilities} = props;
    const capacity = coverage.zoneIds.length;
    const cellCount = props.grid.width * props.grid.height;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...getInputViews(props), ...getGraphOutputViews(output)].map((view, index) => [
          `view.${index}`,
          view
        ])
      )
    );
    // Slot 6 of the coverage georeference is the affine determinant; read per frame when supplied.
    const cellArea = props.parameters
      ? `abs(georeference[georeferenceOffset + ${GEOREFERENCE_DETERMINANT_SLOT}u])`
      : String(
          Math.abs(
            props.grid.affine[0] * props.grid.affine[4] -
              props.grid.affine[1] * props.grid.affine[3]
          )
        );
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const areaScaleWGSL =
      cellSizeMode === 'uniform'
        ? ''
        : getRasterRowAreaScaleWGSL(
            props.grid,
            cellSizeMode,
            props.parameters ? 'georeference' : undefined
          );
    const recordArea =
      cellSizeMode === 'uniform'
        ? `f32(${cellArea})`
        : `f32(${cellArea}) * select(1.0, getRowAreaScale(cell / ${props.grid.width}u), cell < ${cellCount}u)`;
    validateRasterExpansionGraph(id, graph, [
      ...getInputViews(props),
      ...getGraphOutputViews(output)
    ]);
    // Include the alignment prefix of caller subviews and every derived allocation before
    // canonicalizing the band or declaring sort scratch, so failure leaves the graph unchanged.
    for (const view of [...getInputViews(props), ...getGraphOutputViews(output)]) {
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        view.buffer.id,
        getViewBindingRange(view).size,
        'reduce the packed input or output size',
        view.buffer.byteLength
      );
    }
    for (const [name, bytes] of [
      ['canonical raster band', cellCount * 4],
      ['records', capacity * RECORD_WORDS * 4],
      ['tuple columns', capacity * 4],
      [
        'sort scratch',
        (capacity <= LOCAL_SORT_MAXIMUM_TUPLES ? 2 ** Math.ceil(Math.log2(capacity)) : capacity) * 4
      ],
      ['segment offsets', (zoneCount + 1) * 4],
      ['summary', zoneCount * SUMMARY_WORDS * 4],
      ['control', 3 * 4]
    ] as const) {
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce tuple capacity, zoneCount or grid dimensions'
      );
    }
    const band = getRasterBufferBandNodes(graph, id, props.input, props.grid, true);
    const records = createTransientView(graph, `${id}-records`, 'uint32', capacity * RECORD_WORDS);
    const control = createTransientView(graph, `${id}-control`, 'uint32', 3);
    const centeredValues = createTransientView(graph, `${id}-centered-values`, 'float32', capacity);
    const indices = createTransientView(graph, `${id}-indices`, 'uint32', capacity);
    const zoneKeys = createTransientView(graph, `${id}-zone-keys`, 'uint32', capacity);
    const valueKeys = createTransientView(graph, `${id}-value-keys`, 'uint32', capacity);
    const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', capacity);
    const offsets = createTransientView(graph, `${id}-offsets`, 'uint32', zoneCount + 1);
    const summary = createTransientView(
      graph,
      `${id}-summary`,
      'float32',
      zoneCount * SUMMARY_WORDS
    );
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-sort`,
      operation: OPERATION,
      indices,
      keys: [{view: zoneKeys}, {view: valueKeys}, {view: cellKeys}]
    });
    const nodes: GPUCommandNode<Parameters>[] = [
      ...band.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-reset`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
          {name: 'required', view: coverage.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
          {name: 'control', view: control, type: 'u32', access: 'read_write'}
        ],
        body: `let countValue = count[countOffset];
  control[controlOffset] = min(countValue, ${capacity}u);
  control[controlOffset + 1u] = u32(countValue <= ${capacity}u && required[requiredOffset] == countValue && overflow[overflowOffset] == 0u);
  control[controlOffset + 2u] = 0u;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'values', view: band.band.storage.values, type: 'f32', access: 'read'},
          {name: 'validity', view: band.band.validity!, type: 'u32', access: 'read'},
          {name: 'zoneIds', view: coverage.zoneIds, type: 'u32', access: 'read'},
          {name: 'cellIds', view: coverage.cellIds, type: 'u32', access: 'read'},
          {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
          {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'},
          {name: 'records', view: records, type: 'u32', access: 'read_write'},
          ...(props.parameters
            ? [
                {
                  name: 'georeference',
                  view: props.parameters,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `${NUMBER_SOURCE}${areaScaleWGSL}`,
        body: `let base = recordsOffset + index * ${RECORD_WORDS}u;
  var zone = ${zoneCount}u; var cell = 0xffffffffu; var value = missingValue(index); var fraction = 0.0; var flags = 0u;
  if (index < atomicLoad(&control[controlOffset])) {
    zone = zoneIds[zoneIdsOffset + index]; cell = cellIds[cellIdsOffset + index]; fraction = fractions[fractionsOffset + index];
    if (zone >= ${zoneCount}u) { atomicStore(&control[controlOffset + 2u], 1u); zone = ${zoneCount}u; }
    if (cell < ${cellCount}u && validity[validityOffset + cell] != 0u) { value = values[valuesOffset + cell]; }
    if (cell >= ${cellCount}u || !isFiniteValue(fraction) || fraction < 0.0 || fraction > 1.0) { flags = 2u; }
    else if (isFiniteValue(value) && fraction > 0.0) { flags = 1u; }
  }
  records[base] = zone; records[base + 1u] = cell;
  records[base + 2u] = select(bitcast<u32>(value), 0u, value == 0.0);
  records[base + 3u] = bitcast<u32>(fraction); records[base + 4u] = flags;
  records[base + 5u] = bitcast<u32>(fraction); records[base + 6u] = bitcast<u32>(${recordArea});`
      }),
      ...(props.observationWeights || props.cellAreas
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-observation-support`,
              operation: OPERATION,
              invocationCount: capacity,
              bindings: [
                {name: 'records', view: records, type: 'u32', access: 'read_write'},
                ...(props.observationWeights
                  ? [
                      {
                        name: 'observationWeights',
                        view: props.observationWeights,
                        type: 'f32' as const,
                        access: 'read' as const
                      }
                    ]
                  : []),
                ...(props.cellAreas
                  ? [
                      {
                        name: 'cellAreas',
                        view: props.cellAreas,
                        type: 'f32' as const,
                        access: 'read' as const
                      }
                    ]
                  : [])
              ],
              declarations: NUMBER_SOURCE,
              body: `let base = recordsOffset + index * ${RECORD_WORDS}u; let cell = records[base + 1u];
if (cell >= ${cellCount}u) { return; }
let observationWeight = ${props.observationWeights ? 'observationWeights[observationWeightsOffset + cell]' : '1.0'};
let area: f32 = ${props.cellAreas ? 'cellAreas[cellAreasOffset + cell]' : props.parameters || cellSizeMode !== 'uniform' ? 'bitcast<f32>(records[base + 6u])' : cellArea};
let weight = bitcast<f32>(records[base + 3u]) * observationWeight;
if (!isFiniteValue(observationWeight) || observationWeight < 0.0 || !isFiniteValue(weight) || !isFiniteValue(area) || area <= 0.0) { records[base + 4u] = 2u; }
records[base + 5u] = bitcast<u32>(weight); records[base + 6u] = bitcast<u32>(area);`
            })
          ]
        : []),
      createWGSLKernelNode(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'records', view: records, type: 'u32', access: 'read'},
          {name: 'indices', view: indices, type: 'u32', access: 'read_write'},
          {name: 'zoneKeys', view: zoneKeys, type: 'u32', access: 'read_write'},
          {name: 'valueKeys', view: valueKeys, type: 'u32', access: 'read_write'},
          {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'}
        ],
        body: `let base = recordsOffset + index * ${RECORD_WORDS}u; let bits = records[base + 2u];
  indices[indicesOffset + index] = index; zoneKeys[zoneKeysOffset + index] = records[base];
  valueKeys[valueKeysOffset + index] = select(0xffffffffu, select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u), (bits & 0x7f800000u) != 0x7f800000u);
  cellKeys[cellKeysOffset + index] = records[base + 1u];`
      }),
      ...sorted.nodes,
      createSortedSegmentOffsetsNode(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        segmentCount: zoneCount,
        sortedKeys: sorted.sortedPrimaryKeys,
        segmentOffsets: offsets
      }),
      // Materialize centered values across a dispatch boundary. A backend may reassociate
      // (value - anchor) - centeredMean into value - (anchor + centeredMean) inside one
      // kernel, losing the small mean offset when the common baseline is large.
      createWGSLKernelNode(graph, {
        id: `${id}-center`,
        operation: OPERATION,
        invocationCount: zoneCount,
        bindings: [
          {name: 'records', view: records, type: 'u32', access: 'read'},
          {name: 'indices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'centeredValues', view: centeredValues, type: 'f32', access: 'read_write'}
        ],
        body: `let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
  var anchor = 0.0; var hasAnchor = false;
  for (var cursor = start; cursor < end; cursor++) {
    let originalIndex = indices[indicesOffset + cursor];
    let base = recordsOffset + originalIndex * ${RECORD_WORDS}u;
    var centered = 0.0;
    if (records[base + 4u] == 1u && bitcast<f32>(records[base + 5u]) > 0.0) {
      let value = bitcast<f32>(records[base + 2u]);
      if (!hasAnchor) { anchor = value; hasAnchor = true; }
      centered = value - anchor;
    }
    centeredValues[centeredValuesOffset + originalIndex] = centered;
  }`
      }),
      // Stripe long zones across 256 partials, then reduce a fixed tree. This limits
      // float32 drift without relying on compensation expressions a backend may reassociate.
      createWGSLKernelNode(graph, {
        id: `${id}-moments`,
        operation: OPERATION,
        invocationCount: zoneCount,
        bindings: [
          {name: 'records', view: records, type: 'u32', access: 'read'},
          {name: 'indices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'control', view: control, type: 'u32', access: 'read'},
          {name: 'zoneValidity', view: coverage.zoneValidity, type: 'u32', access: 'read'},
          {name: 'centeredValues', view: centeredValues, type: 'f32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        declarations: NUMBER_SOURCE,
        body: `let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
  var status = select(1u, 0u, control[controlOffset + 1u] != 0u && zoneValidity[zoneValidityOffset + index] != 0u);
  if (control[controlOffset + 2u] != 0u) { status |= 2u; }
  var covered = 0.0; var validArea = 0.0; var weight = 0.0; var squaredWeight = 0.0; var centeredSum = 0.0; var centeredPartials: array<f32, 256>; var supportPartials: array<vec4<f32>, 256>; var anchor = 0.0;
  var validCellCount = 0u; var uniqueCount = 0u; var previousValue = 0.0; var previousCell = 0xffffffffu;
  for (var cursor = start; cursor < end; cursor++) {
    let base = recordsOffset + indices[indicesOffset + cursor] * ${RECORD_WORDS}u;
    let cell = records[base + 1u]; let flags = records[base + 4u];
    if (flags == 2u || (cursor > start && cell == previousCell)) { status |= 2u; }
    previousCell = cell;
    if (flags == 2u) { continue; }
    let fraction = bitcast<f32>(records[base + 3u]); let area = bitcast<f32>(records[base + 6u]); supportPartials[(cursor - start) % 256u].x += fraction * area;
    if (flags != 1u) { continue; }
    supportPartials[(cursor - start) % 256u].y += fraction * area;
    let observationWeight = bitcast<f32>(records[base + 5u]);
    if (observationWeight <= 0.0) { continue; }
    let value = bitcast<f32>(records[base + 2u]);
    if (uniqueCount == 0u) { anchor = value; }
    if (uniqueCount == 0u || value != previousValue) { uniqueCount++; }
    previousValue = value; validCellCount++; supportPartials[(cursor - start) % 256u].z += observationWeight; supportPartials[(cursor - start) % 256u].w += observationWeight * observationWeight;
    centeredPartials[(cursor - start) % 256u] += observationWeight * centeredValues[centeredValuesOffset + indices[indicesOffset + cursor]];
  }
  for (var stride = 128u; stride > 0u; stride /= 2u) { for (var partial = 0u; partial < stride; partial++) { centeredPartials[partial] += centeredPartials[partial + stride]; supportPartials[partial] += supportPartials[partial + stride]; } }
  centeredSum = centeredPartials[0]; covered = supportPartials[0].x; validArea = supportPartials[0].y; weight = supportPartials[0].z; squaredWeight = supportPartials[0].w;
  var centeredMean = 0.0; var mean = missingValue(index); var populationVariance = missingValue(index); var sampleVariance = missingValue(index); var effectiveSupport = 0.0;
  if (weight > 0.0) { centeredMean = centeredSum / weight; mean = anchor + centeredMean; }
  var secondMoment = 0.0; var momentPartials: array<f32, 256>;
  for (var cursor = start; cursor < end; cursor++) {
    let base = recordsOffset + indices[indicesOffset + cursor] * ${RECORD_WORDS}u;
    if (records[base + 4u] != 1u || bitcast<f32>(records[base + 5u]) <= 0.0) { continue; }
    let difference = centeredValues[centeredValuesOffset + indices[indicesOffset + cursor]] - centeredMean;
    momentPartials[(cursor - start) % 256u] += bitcast<f32>(records[base + 5u]) * difference * difference;
  }
  for (var stride = 128u; stride > 0u; stride /= 2u) { for (var partial = 0u; partial < stride; partial++) { momentPartials[partial] += momentPartials[partial + stride]; } }
  secondMoment = momentPartials[0];
  var denominator = 0.0;
  if (weight > 0.0) {
    populationVariance = secondMoment / weight; effectiveSupport = weight * weight / squaredWeight;
    denominator = weight - squaredWeight / weight;
    if (validCellCount >= 2u && denominator > 0.0) { sampleVariance = secondMoment / denominator; }
    if (!isFiniteValue(mean) || !isFiniteValue(populationVariance) || !isFiniteValue(effectiveSupport) || (validCellCount >= 2u && denominator > 0.0 && !isFiniteValue(sampleVariance))) { status |= 16u; }
  } else { status |= 4u; }
  if (validCellCount < 2u || denominator <= 0.0) { status |= 8u; }
  if (!isFiniteValue(covered) || !isFiniteValue(validArea)) { status |= 16u; }
  let complete = (status & 3u) == 0u;
  let valid = (status & 23u) == 0u;
  let base = summaryOffset + index * ${SUMMARY_WORDS}u;
  summary[base] = select(missingValue(index), mean, valid);
  summary[base + 1u] = select(missingValue(index), populationVariance, valid);
  summary[base + 2u] = select(missingValue(index), sampleVariance, valid);
  summary[base + 3u] = select(missingValue(index), effectiveSupport, complete && (status & 16u) == 0u);
  summary[base + 4u] = select(missingValue(index), weight, complete);
  summary[base + 5u] = select(missingValue(index), squaredWeight, complete);
  summary[base + 6u] = select(missingValue(index), covered, complete);
  summary[base + 7u] = select(missingValue(index), validArea, complete);
  summary[base + 8u] = f32(valid); summary[base + 9u] = f32(status); summary[base + 10u] = f32(uniqueCount);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-quantiles`,
        operation: OPERATION,
        invocationCount: zoneCount * probabilities.length,
        bindings: [
          {name: 'records', view: records, type: 'u32', access: 'read'},
          {name: 'indices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'probabilities', view: probabilities, type: 'f32', access: 'read'},
          {name: 'quantiles', view: output.quantiles, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.quantileValidity, type: 'u32', access: 'read_write'}
        ],
        declarations: NUMBER_SOURCE,
        body: `let zone = index / ${probabilities.length}u; let probability = probabilities[probabilitiesOffset + index % ${probabilities.length}u];
  let base = summaryOffset + zone * ${SUMMARY_WORDS}u;
  let valid = summary[base + 8u] != 0.0 && isFiniteValue(probability) && probability >= 0.0 && probability <= 1.0;
  var result = missingValue(index);
  if (valid) {
    let distinctCount = summary[base + 10u]; let totalWeight = summary[base + 4u];
    let quantilePosition = probability * (distinctCount - 1.0) * totalWeight;
    var cursor = offsets[offsetsOffset + zone]; let end = offsets[offsetsOffset + zone + 1u];
    var cumulativeWeight = 0.0; var distinctIndex = 0u; var previousPosition = 0.0; var previousValue = 0.0;
    while (cursor < end) {
      var record = recordsOffset + indices[indicesOffset + cursor] * ${RECORD_WORDS}u;
      if (records[record + 4u] != 1u || bitcast<f32>(records[record + 5u]) <= 0.0) { cursor++; continue; }
      let value = bitcast<f32>(records[record + 2u]); var groupWeight = 0.0;
      while (cursor < end) {
        record = recordsOffset + indices[indicesOffset + cursor] * ${RECORD_WORDS}u;
        if (bitcast<f32>(records[record + 2u]) != value) { break; }
        if (records[record + 4u] == 1u) { groupWeight += bitcast<f32>(records[record + 5u]); }
        cursor++;
      }
      ${props.quantileMethod === 'inverse-cdf' ? 'if (cumulativeWeight + groupWeight >= probability * totalWeight) { result = value; break; }' : ''}
      let position = f32(distinctIndex) * groupWeight + (distinctCount - 1.0) * cumulativeWeight;
      if (${props.quantileMethod === 'inverse-cdf' ? 'false' : 'true'} && distinctIndex > 0u && quantilePosition <= position) {
        result = previousValue + (value - previousValue) * ((quantilePosition - previousPosition) / (position - previousPosition)); break;
      }
      result = value; previousValue = value; previousPosition = position;
      cumulativeWeight += groupWeight; distinctIndex++;
    }
  }
  quantiles[quantilesOffset + index] = result; validity[validityOffset + index] = u32(valid && isFiniteValue(result));`
      })
    ];
    const columns = [
      output.means,
      output.populationVariances,
      output.sampleVariances,
      output.effectiveSampleSizes,
      output.weightSums,
      output.squaredWeightSums,
      output.coveredArea,
      output.validArea
    ];
    for (let firstColumn = 0; firstColumn < columns.length; firstColumn += 4) {
      const selected = columns.slice(firstColumn, firstColumn + 4);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-publish-${firstColumn}`,
          operation: OPERATION,
          invocationCount: zoneCount,
          bindings: [
            {name: 'summary', view: summary, type: 'f32', access: 'read'},
            ...selected.map((view, column) => ({
              name: `column${column}`,
              view,
              type: 'f32' as const,
              access: 'read_write' as const
            }))
          ],
          body: selected
            .map(
              (_, column) =>
                `column${column}[column${column}Offset + index] = summary[summaryOffset + index * ${SUMMARY_WORDS}u + ${firstColumn + column}u];`
            )
            .join('\n')
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish-status`,
        operation: OPERATION,
        invocationCount: zoneCount,
        bindings: [
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `validity[validityOffset + index] = u32(summary[summaryOffset + index * ${SUMMARY_WORDS}u + 8u]);
  status[statusOffset + index] = u32(summary[summaryOffset + index * ${SUMMARY_WORDS}u + 9u]);`
      })
    );
    return nodes;
  }
}

function getInputViews(props: GPURasterFractionalZonalDistributionProps): GraphDataView[] {
  return [
    ...getRasterBandViews(props.input),
    ...getGraphOutputViews(props.coverage),
    props.probabilities,
    props.observationWeights,
    props.cellAreas,
    props.parameters
  ].filter((view): view is GraphDataView => Boolean(view));
}
