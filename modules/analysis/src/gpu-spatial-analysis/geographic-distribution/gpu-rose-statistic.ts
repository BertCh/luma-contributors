// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getSpatialAnalysisPackedKernelNodes,
  validateSpatialAnalysisPackedOutputAllocations
} from './spatial-analysis-packed-kernel';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {GUARDED_ATAN2_WGSL, IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGeographicDistributionViews} from './geographic-distribution-validation';
import type {GPUCircularStatisticsContract} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Lanes of the workgroup that owns one group. */
const ROSE_WORKGROUP_SIZE = 256;

/** Bounds bin-owner loops and the caller's group-by-bin storage. */
const ROSE_MAXIMUM_BIN_COUNT = 360;
/** Input row IDs and counts use uint32; the tile loop stops before its increment can wrap. */
const ROSE_MAXIMUM_ROW_COUNT = 0xffffffff;
/** Bound resident workgroups; each cooperatively walks more groups when necessary. */
const ROSE_MAXIMUM_DISPATCH_GROUP_COUNT = 65535;
/** Every output element is addressed by a uint32 index. */
const ROSE_MAXIMUM_OUTPUT_ELEMENT_COUNT = 0xffffffff;

/** Five float32 values per group written by {@link GPURoseStatistic}. */
export const GPU_ROSE_STATISTIC_SUMMARY_STRIDE = 5;

/** Summary offsets: valid count, mean angle, resultant length, circular variance, Rayleigh z. */
export const GPU_ROSE_STATISTIC_SUMMARY = {
  count: 0,
  meanAngle: 1,
  resultantLength: 2,
  circularVariance: 3,
  /** Standard Rayleigh z of the unweighted valid directions (doubled for axial data). */
  rayleighZ: 4,
  stride: GPU_ROSE_STATISTIC_SUMMARY_STRIDE
} as const;

/** Caller-owned outputs of {@link GPURoseStatistic}, rewritten on every encoding. */
export type GPURoseStatisticOutput = {
  /** `groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE` float32 values. */
  summary: GraphDataView<'float32'>;
  /** `groupCount * circular.binCount` weighted observations in angle-bin order. */
  bins: GraphDataView<'float32'>;
};

/** Properties for {@link GPURoseStatistic}. */
export type GPURoseStatisticProps = {
  id?: string;
  /**
   * Coordinate space of `starts` and `ends`. Angles are measured in the plane of the coordinates,
   * so only planar contexts are accepted; omitted means native planar. Longitude/latitude throws
   * at construction (a degree-space angle is not a bearing): project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  starts: GraphDataView<'float32x2'>;
  ends: GraphDataView<'float32x2'>;
  /** Positive finite descriptive weights for bins and weighted moments, not frequency counts. */
  weights?: GraphDataView<'float32'>;
  groupIds?: GraphDataView<'uint32'>;
  groupCount?: number;
  mask?: GraphDataView<'uint32'>;
  /** Angular units and origin affect meanAngle and bins; moments are computed in radians. */
  circular: Omit<GPUCircularStatisticsContract, 'angleUnit' | 'origin'> & {
    angleUnit: 'radians' | 'degrees';
    origin: 'positive-x-counter-clockwise' | 'positive-y-clockwise';
  };
  /** Caller-owned outputs. */
  output: GPURoseStatisticOutput;
};

/**
 * Directional or axial Rose statistic for planar line segments. Directional angles cover
 * `[0, 2 pi)` (or `[0, 360)` degrees); axial angles identify opposite directions and cover
 * `[0, pi)` (or `[0, 180)`). The positive-y clockwise origin reports planar compass bearings.
 * Equal-width bins
 * use a left-closed/right-open convention; computed bin coordinates within four float32 ULPs
 * of an integer boundary snap to that boundary. The mean is weighted by `weights` (one by default),
 * zero-length/non-finite lines are excluded, and bins add 256-row tile partials in source-row
 * order. The angular moment sums use a fixed 256-lane tree order (deterministic, not row order).
 *
 * Rayleigh z uses each valid segment once: `z = n * Rbar^2`, where `Rbar` is the
 * unweighted mean resultant length (angles doubled in axial mode). Descriptive weights do not
 * change z or n, except that non-positive/non-finite weights exclude a segment. In particular,
 * changing length units cannot change z. No p-value is returned; the usual Rayleigh null assumes
 * independent uniform directions. This is not giddy Rose's directional-LISA permutation test.
 *
 * With `groupIds` and more than one group, rows are first stably sorted by group (masked rows and
 * out-of-range group IDs sort last), so each group's workgroup reads only its own rows and the
 * total work is O(rows) instead of O(groups * rows). Tiles then hold 256 rows of one group, still
 * in source-row order.
 */
export class GPURoseStatistic implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURoseStatisticProps;
  readonly groupCount: number;
  /** Resolved spatial context; always planar. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPURoseStatisticProps) {
    this.id = props.id ?? 'rose-statistic';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.groupCount = props.groupCount ?? 1;
    const {id, groupCount} = this;
    validateSinglePackedView(props.starts, ['float32x2'], `${id} starts`);
    validateSinglePackedView(props.ends, ['float32x2'], `${id} ends`);
    if (props.ends.length !== props.starts.length)
      throw new Error(`${id} ends length must equal starts length`);
    for (const [name, view] of [
      ['weights', props.weights],
      ['groupIds', props.groupIds],
      ['mask', props.mask]
    ] as const) {
      if (!view) continue;
      if (name === 'weights') validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      else validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length !== props.starts.length)
        throw new Error(`${id} ${name} length must equal starts length`);
    }
    validateCapacity(id, 'starts.length', props.starts.length, 0, ROSE_MAXIMUM_ROW_COUNT);
    validateCapacity(id, 'groupCount', groupCount);
    validateMaximumCount(
      id,
      'groupCount * summary stride',
      groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE,
      ROSE_MAXIMUM_OUTPUT_ELEMENT_COUNT
    );
    validateMaximumCount(
      id,
      'groupCount * circular.binCount',
      groupCount * props.circular.binCount,
      ROSE_MAXIMUM_OUTPUT_ELEMENT_COUNT
    );
    if (!props.groupIds && groupCount !== 1)
      throw new Error(`${id} groupIds are required for multiple groups`);
    if (props.circular.mode !== 'directional' && props.circular.mode !== 'axial')
      throw new Error(`${id} circular.mode is invalid`);
    if (
      !['radians', 'degrees'].includes(props.circular.angleUnit) ||
      !['positive-x-counter-clockwise', 'positive-y-clockwise'].includes(props.circular.origin)
    ) {
      throw new Error(`${id} circular angle convention is unsupported`);
    }
    validateCapacity(id, 'circular.binCount', props.circular.binCount, 2, ROSE_MAXIMUM_BIN_COUNT);
    const {summary, bins} = props.output;
    validateSinglePackedView(summary, ['float32'], `${id} output.summary`);
    validateSinglePackedView(bins, ['float32'], `${id} output.bins`);
    if (summary.length < groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE)
      throw new Error(`${id} output.summary is too short`);
    if (bins.length < groupCount * props.circular.binCount)
      throw new Error(`${id} output.bins are too short`);
    const inputs = {
      starts: props.starts,
      ends: props.ends,
      weights: props.weights,
      groupIds: props.groupIds,
      mask: props.mask
    };
    const segmented = Boolean(props.groupIds) && groupCount > 1;
    validateGeographicDistributionViews(id, inputs, props.output, [
      {
        reads: {
          ...inputs,
          groupIds: segmented ? undefined : props.groupIds,
          mask: segmented ? undefined : props.mask
        },
        writes: props.output
      }
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, groupCount} = this;
    validateSpatialAnalysisPackedOutputAllocations(graph, id, props.output);
    validateGraphViewsBelongToGraph(id, graph, [
      props.starts,
      props.ends,
      props.weights,
      props.groupIds,
      props.mask,
      props.output.summary,
      props.output.bins
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      starts: props.starts,
      ends: props.ends,
      weights: props.weights,
      groupIds: props.groupIds,
      mask: props.mask,
      ...props.output
    });
    if (props.groupIds && groupCount > 1) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'group sort scratch',
        props.starts.length * 4,
        'reduce starts length'
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'group offsets scratch',
        (groupCount + 1) * 4,
        'reduce groupCount'
      );
    }
    const rowCount = props.starts.length;
    // Sorted segments only pay off when there is more than one group to separate.
    const segmented = Boolean(props.groupIds) && groupCount > 1;
    const nodes: (GPUCommandNode<Parameters> | GPUCommandNode<Parameters>[])[] = [];
    let segmentBindings: WGSLKernelBinding[] = [];
    if (segmented) {
      const sortKeys = createTransientView(graph, `${id}-group-sort-keys`, 'uint32', rowCount);
      const sortIndices = createTransientView(
        graph,
        `${id}-group-sort-indices`,
        'uint32',
        rowCount
      );
      const sortedKeys = createTransientView(graph, `${id}-group-sorted-keys`, 'uint32', rowCount);
      const sortedRows = createTransientView(graph, `${id}-group-sorted-rows`, 'uint32', rowCount);
      const groupOffsets = createTransientView(
        graph,
        `${id}-group-offsets`,
        'uint32',
        groupCount + 1
      );
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-group-sort-prepare`,
          operation: 'GPURoseStatistic',
          variant: 'group-sort-prepare',
          invocationCount: rowCount,
          bindings: [
            {name: 'groupIds', view: props.groupIds!, type: 'u32', access: 'read'},
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
            {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
          ],
          declarations: `const GROUP_COUNT: u32 = ${groupCount}u;`,
          body: `let key = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortKeys[sortKeysOffset + index] = ${props.mask ? 'select(key, GROUP_COUNT, mask[maskOffset + index] == 0u)' : 'key'};
  sortIndices[sortIndicesOffset + index] = index;`
        }),
        ...new GPUSort({
          id: `${id}-group-sort`,
          keys: sortKeys,
          values: sortIndices,
          outputKeys: sortedKeys,
          outputValues: sortedRows,
          keyBits: getSortKeyBits(groupCount)
        }).getCommandNodes(graph),
        ...getSortedSegmentOffsetNodes<Parameters>(graph, {
          id: `${id}-groups`,
          operation: 'GPURoseStatistic',
          groupCount,
          sortedGroupKeys: sortedKeys,
          segmentOffsets: groupOffsets
        })
      );
      segmentBindings = [
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'groupOffsets', view: groupOffsets, type: 'u32', access: 'read'}
      ];
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'starts', view: props.starts, type: 'f32', access: 'read'},
      {name: 'ends', view: props.ends, type: 'f32', access: 'read'},
      ...(props.weights
        ? [{name: 'weights', view: props.weights, type: 'f32' as const, access: 'read' as const}]
        : []),
      ...(segmented
        ? segmentBindings
        : [
            ...(props.groupIds
              ? [
                  {
                    name: 'groupIds',
                    view: props.groupIds,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : [])
          ]),
      {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'},
      {name: 'bins', view: props.output.bins, type: 'f32', access: 'read_write'}
    ];
    const axial = props.circular.mode === 'axial';
    // Unsegmented tiles walk every row and filter by mask and group. Segmented tiles walk the
    // group's sorted segment; its bounds are broadcast with workgroupUniformLoad so the tile loop,
    // which contains barriers, stays in uniform control flow.
    const rowFilter = segmented
      ? 'false'
      : `${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}${props.groupIds ? 'groupIds[groupIdsOffset + row] != group || ' : ''}false`;
    const tileLoopWGSL = segmented
      ? `if (lane == 0u) {
    segmentBegin = 0u;
    segmentEnd = 0u;
    if (isActiveGroup) {
      segmentBegin = groupOffsets[groupOffsetsOffset + group];
      segmentEnd = groupOffsets[groupOffsetsOffset + group + 1u];
    }
  }
  let beginPosition = workgroupUniformLoad(&segmentBegin);
  let endPosition = workgroupUniformLoad(&segmentEnd);
  for (var tile = beginPosition; tile < endPosition;) {
    let position = tile + lane;
    let hasRow = lane < endPosition - tile;
    var row = 0u;
    if (hasRow) { row = sortedRows[sortedRowsOffset + position]; }`
      : `for (var tile = 0u; tile < ROWS;) {
    let row = tile + lane;
    let hasRow = isActiveGroup && lane < ROWS - tile;`;
    nodes.push(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-reduce`,
        operation: 'GPURoseStatistic',
        variant: segmented ? `${props.circular.mode}-segmented` : props.circular.mode,
        // One 256-lane workgroup owns one group. Rows are processed in tiles of 256: every lane
        // does the expensive atan2/sin/cos of its own row, then bin-owner lanes add the tile's
        // weights in row order, so bins keep source-row order. The moment sums use a fixed tree.
        workgroupSize: ROSE_WORKGROUP_SIZE,
        invocationCount:
          Math.min(groupCount, ROSE_MAXIMUM_DISPATCH_GROUP_COUNT) * ROSE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings,
        declarations: `const ROWS: u32 = ${rowCount}u;\nconst GROUP_COUNT: u32 = ${groupCount}u;\nconst LANES: u32 = ${ROSE_WORKGROUP_SIZE}u;\nconst BIN_COUNT: u32 = ${props.circular.binCount}u;\nconst PI: f32 = 3.141592653589793;\nconst PERIOD: f32 = ${axial ? 'PI' : '2.0 * PI'};\n${IS_FINITE_FLOAT_WGSL}\n${GUARDED_ATAN2_WGSL}
${SCALED_QUOTIENT_WGSL}
var<workgroup> tileBin: array<u32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> tileWeight: array<f32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedCount: array<u32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedWeight: array<f32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedCosine: array<f32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedSine: array<f32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedDirectionCosine: array<f32, ${ROSE_WORKGROUP_SIZE}>;
var<workgroup> sharedDirectionSine: array<f32, ${ROSE_WORKGROUP_SIZE}>;${
          segmented ? '\nvar<workgroup> segmentBegin: u32;\nvar<workgroup> segmentEnd: u32;' : ''
        }`,
        body: `let lane = localInvocationIndex;
for(var group=workgroupIndex;group<GROUP_COUNT;group+=INVOCATION_COUNT / LANES){
  let isActiveGroup = group < GROUP_COUNT;
  let binBase = binsOffset + group * BIN_COUNT;
  if (isActiveGroup) {
    for (var bin = lane; bin < BIN_COUNT; bin += LANES) { bins[binBase + bin] = 0.0; }
  }
  var count = 0u;
  var weightTotal = 0.0;
  var cosine = 0.0;
  var sine = 0.0;
  var directionCosine = 0.0;
  var directionSine = 0.0;
  ${tileLoopWGSL}
    var rowBin = 0xffffffffu;
    var rowWeight = 0.0;
    if (hasRow) {
      if (!(${rowFilter})) {
        let start = vec2f(starts[startsOffset + row * 2u], starts[startsOffset + row * 2u + 1u]);
        let end = vec2f(ends[endsOffset + row * 2u], ends[endsOffset + row * 2u + 1u]);
        let delta = end - start;
        let lengthSquared = dot(delta, delta);
        let weight = ${props.weights ? 'weights[weightsOffset + row]' : '1.0'};
        if (lengthSquared > 0.0 && weight > 0.0 && isFiniteFloat(lengthSquared) && isFiniteFloat(weight)) {
          // Transform the vector, rather than a rounded atan2 result. Opposite axial vectors
          // now reach atan2 with identical bits, including signed-zero cardinal components.
          var direction = ${props.circular.origin === 'positive-y-clockwise' ? 'delta.yx' : 'delta'};
          ${axial ? 'if(direction.y<0.0||(direction.y==0.0&&direction.x<0.0)){direction=-direction;}' : ''}
          direction=select(direction,vec2<f32>(0.0),direction==vec2<f32>(0.0));
          var angle = guardedAtan2(direction.y, direction.x);
          if (angle < 0.0) { angle += 2.0 * PI; }
          // Curated formula binding: phi is the doubled axial angle, with a 2*pi period.
          let circularAngle = ${axial ? '2.0 * angle' : 'angle'};
          let unitDirection=direction/sqrt(lengthSquared);
          let cosineAngle = ${axial ? '(unitDirection.x-unitDirection.y)*(unitDirection.x+unitDirection.y)' : 'unitDirection.x'};
          let sineAngle = ${axial ? '2.0*unitDirection.x*unitDirection.y' : 'unitDirection.y'};
          cosine += weight * cosineAngle;
          sine += weight * sineAngle;
          directionCosine += cosineAngle;
          directionSine += sineAngle;
          weightTotal += weight;
          count++;
          var binPosition = (circularAngle / (2.0 * PI)) * f32(BIN_COUNT);
          let nearestBoundary=round(binPosition);
          // Snap only the f32 uncertainty interval around an integer bin boundary (four ULPs).
          if(abs(binPosition-nearestBoundary)<=4.76837158203125e-7*max(1.0,abs(binPosition))){binPosition=nearestBoundary;}
          rowBin = u32(floor(binPosition)) % BIN_COUNT;
          rowWeight = weight;
        }
      }
    }
    tileBin[lane] = rowBin;
    tileWeight[lane] = rowWeight;
    workgroupBarrier();
    if (isActiveGroup) {
      for (var bin = lane; bin < BIN_COUNT; bin += LANES) {
        var added = 0.0;
        for (var other = 0u; other < LANES; other++) {
          if (tileBin[other] == bin) { added += tileWeight[other]; }
        }
        bins[binBase + bin] += added;
      }
    }
    workgroupBarrier();
    if(${segmented ? 'endPosition' : 'ROWS'}-tile<=LANES){break;}
    tile+=LANES;
  }
  sharedCount[lane] = count;
  sharedWeight[lane] = weightTotal;
  sharedCosine[lane] = cosine;
  sharedSine[lane] = sine;
  sharedDirectionCosine[lane] = directionCosine;
  sharedDirectionSine[lane] = directionSine;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedCount[lane] += sharedCount[lane + stride];
      sharedWeight[lane] += sharedWeight[lane + stride];
      sharedCosine[lane] += sharedCosine[lane + stride];
      sharedSine[lane] += sharedSine[lane + stride];
      sharedDirectionCosine[lane] += sharedDirectionCosine[lane + stride];
      sharedDirectionSine[lane] += sharedDirectionSine[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveGroup) {
    let base = summaryOffset + group * ${GPU_ROSE_STATISTIC_SUMMARY_STRIDE}u;
    summary[base] = f32(sharedCount[0]);
    if (sharedWeight[0] > 0.0 && isFiniteFloat(sharedWeight[0])) {
      let totalWeight = sharedWeight[0];
      let meanCosine = getScaledQuotient(sharedCosine[0],totalWeight,1.0,0);
      let meanSine = getScaledQuotient(sharedSine[0],totalWeight,1.0,0);
      var meanAngle = guardedAtan2(meanSine, meanCosine);
      if (meanAngle < 0.0) { meanAngle += 2.0 * PI; }
      ${axial ? 'meanAngle *= 0.5;' : ''}
      let resultant = length(vec2<f32>(meanCosine,meanSine));
      let outputMean = ${props.circular.angleUnit === 'degrees' ? 'meanAngle * (180.0 / PI)' : 'meanAngle'};
      summary[base + 1u] = select(outputMean,0.0,outputMean>=${props.circular.angleUnit === 'degrees' ? (axial ? '180.0' : '360.0') : 'PERIOD'});
      summary[base + 2u] = resultant;
      summary[base + 3u] = 1.0 - resultant;
      let validCount = f32(sharedCount[0]);
      let rayleighZ = (sharedDirectionCosine[0] * sharedDirectionCosine[0] +
        sharedDirectionSine[0] * sharedDirectionSine[0]) / validCount;
      summary[base + 4u] = rayleighZ;
    } else {
      let nan = bitcast<f32>(0x7fc00000u | (group & 0u));
      for (var slot = 1u; slot < ${GPU_ROSE_STATISTIC_SUMMARY_STRIDE}u; slot++) { summary[base + slot] = nan; }
    }
  }
  // Lane zero must finish consuming shared moments before the next group reuses them.
  workgroupBarrier();
}`
      })
    );
    return nodes.flat();
  }
}
