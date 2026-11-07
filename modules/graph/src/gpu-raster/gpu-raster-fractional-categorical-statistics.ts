// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {getGraphOutputViews, validatePackedView} from '../utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../utils/sorted-segment-sums';
import {createWGSLKernelNode} from '../utils/wgsl-kernel-nodes';
import {getRasterBandViews} from './raster-band-utils';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';
import type {GPURasterFractionalCoverageProps} from './gpu-raster-fractional-coverage';
import type {GPURasterBufferBand, GPURasterMetadata} from './types';

export type GPURasterFractionalCategoricalStatisticsProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Dense, uncalibrated category IDs. Masked/nodata cells contribute only to missing area. */
  input: GPURasterBufferBand<'uint32'>;
  zoneCount: number;
  /** Category IDs are [0, categoryCapacity). Valid IDs outside this range fail the result. */
  categoryCapacity: number;
  coverage: GPURasterFractionalCoverageProps['output'];
  output: {
    /** Zone-major category rows: zone * categoryCapacity + category. Sum of coverage fractions. */
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
 * 1..65534 categories, at most 65535 zone*(categoryCapacity+1) groups, and 1..1048576 tuple slots.
 * The extra group holds missing support. Active tuples must be unique (zone, cell) pairs, as emitted
 * by GPURasterFractionalCoverage; disjoint tile cores may be concatenated with remapped cell IDs.
 *
 * Empty/fully masked complete zones retain zero class support and explicit missing area, with NaN
 * proportions and invalid winners. Overflow, inconsistent tuple counts, malformed active tuples or
 * out-of-range valid categories fail every zone (NaN float outputs). Failed source geometry fails
 * its zone. Float32 area overflow also fails its zone. Missing area covers nodata inside the supplied footprint, not unmapped polygon area.
 * Areas are planar coordinate units, not geodesic areas. Graph reuse rereads every input each run.
 */
export class GPURasterFractionalCategoricalStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFractionalCategoricalStatisticsProps;

  constructor(props: GPURasterFractionalCategoricalStatisticsProps) {
    this.id = props.id ?? 'fractional-categorical-statistics';
    this.props = props;
    const {grid, input, coverage, output, categoryCapacity, zoneCount} = props;
    const cellCount = validateRasterExpansionViews(
      this.id,
      grid,
      [...getRasterBandViews(input), ...getGraphOutputViews(coverage)],
      getGraphOutputViews(output)
    );
    // Dense categories bound scratch space and the per-zone winner scan.
    assert(
      Number.isInteger(categoryCapacity) && categoryCapacity >= 1 && categoryCapacity <= 65534
    );
    assert(
      Number.isInteger(zoneCount) && zoneCount >= 1 && zoneCount * (categoryCapacity + 1) <= 65535
    );
    assert(coverage.zoneIds.length >= 1 && coverage.zoneIds.length <= 1048576);
    const cellArea = Math.fround(
      Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
    );
    // A category is an identity: implicit calibration must never merge or alter IDs.
    assert(input.format === 'uint32' && (input.scale ?? 1) === 1 && (input.offset ?? 0) === 0);
    assert(Number.isFinite(cellArea) && cellArea > 0);
    assert(
      input.noDataValue === undefined ||
        (Number.isInteger(input.noDataValue) &&
          input.noDataValue >= 0 &&
          input.noDataValue <= 0xffffffff)
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
      if (view) validatePackedView(view, ['uint32'], this.id);
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
      validatePackedView(view, ['float32'], this.id);
    }
    requireRasterLength(input.storage.values, cellCount);
    requireRasterLength(input.validity, cellCount);
    requireRasterLength(coverage.cellIds, coverage.zoneIds.length);
    requireRasterLength(coverage.fractions, coverage.zoneIds.length);
    requireRasterLength(coverage.zoneValidity, zoneCount);
    for (const view of Object.values(coverage.status)) requireRasterLength(view, 1);
    for (const view of [output.coverageSums, output.classAreas, output.proportions])
      requireRasterLength(view, zoneCount * categoryCapacity);
    for (const view of [
      output.coveredArea,
      output.validArea,
      output.missingArea,
      output.majority,
      output.minority,
      output.validity
    ])
      requireRasterLength(view, zoneCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, input, coverage, output, zoneCount, categoryCapacity} = props;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(input),
      ...getGraphOutputViews(coverage),
      ...getGraphOutputViews(output)
    ]);
    const cellCount = grid.width * grid.height;
    const tupleCapacity = coverage.zoneIds.length;
    const groupCount = zoneCount * (categoryCapacity + 1);
    const cellArea = Math.fround(
      Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
    );
    const categories = createTransientView(graph, `${id}-categories`, 'uint32', cellCount);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', tupleCapacity);
    const contributions = createTransientView(
      graph,
      `${id}-contributions`,
      'float32',
      tupleCapacity
    );
    const failure = createTransientView(graph, `${id}-failure`, 'uint32', 1);
    const sums = createTransientView(graph, `${id}-sums`, 'float32', groupCount);
    const summary = createTransientView(graph, `${id}-summary`, 'float32x4', zoneCount);
    const operation = 'GPURasterFractionalCategoricalStatistics';
    return [
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
          {name: 'failure', view: failure, type: 'u32', access: 'read_write'}
        ],
        body: `failure[failureOffset] = u32(overflow[overflowOffset] != 0u || count[countOffset] > ${tupleCapacity}u || required[requiredOffset] != count[countOffset]);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-prepare-tuples`,
        operation,
        invocationCount: tupleCapacity,
        bindings: [
          {name: 'categories', view: categories, type: 'u32', access: 'read'},
          {name: 'zones', view: coverage.zoneIds, type: 'u32', access: 'read'},
          {name: 'cells', view: coverage.cellIds, type: 'u32', access: 'read'},
          {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
          {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'contributions', view: contributions, type: 'f32', access: 'read_write'},
          {name: 'failure', view: failure, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `keys[keysOffset + index] = ${groupCount}u;
contributions[contributionsOffset + index] = 0.0;
if (index >= count[countOffset]) { return; }
let zone = zones[zonesOffset + index];
let cell = cells[cellsOffset + index];
let fraction = fractions[fractionsOffset + index];
if (zone >= ${zoneCount}u || cell >= ${cellCount}u || !(fraction >= 0.0 && fraction <= 1.0)) {
  atomicOr(&failure[failureOffset], 1u); return;
}
if (fraction == 0.0) { return; }
let category = categories[categoriesOffset + cell];
if (category > ${categoryCapacity}u) { atomicOr(&failure[failureOffset], 1u); return; }
keys[keysOffset + index] = zone * ${categoryCapacity + 1}u + category;
contributions[contributionsOffset + index] = fraction;`
      }),
      ...getSortedSegmentSumNodes(graph, {
        id: `${id}-group`,
        operation,
        segmentCount: groupCount,
        segmentKeys: keys,
        sumContributions: contributions,
        sums
      }),
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
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        body: `let base = sumsOffset + index * ${categoryCapacity + 1}u;
var observed = 0.0;
var greatest = 0.0;
var least = 1048577.0;
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
          {name: 'proportions', view: output.proportions, type: 'f32', access: 'read_write'}
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
          {name: 'missing', view: output.missingArea, type: 'f32', access: 'read_write'}
        ],
        body: `covered[coveredOffset + index] = summary[summaryOffset + index * 4u] * ${cellArea};
observed[observedOffset + index] = summary[summaryOffset + index * 4u + 1u] * ${cellArea};
missing[missingOffset + index] = summary[summaryOffset + index * 4u + 2u] * ${cellArea};`
      })
    ];
  }
}
