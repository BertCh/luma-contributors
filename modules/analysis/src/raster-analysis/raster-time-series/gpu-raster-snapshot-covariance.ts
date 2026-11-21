// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  FIXED_POINT_WGSL,
  createFixedPointClearNode,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateGraphPorts,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterTemporalCenteringBody,
  RASTER_TEMPORAL_CENTERING_WGSL
} from './raster-temporal-centering';
import {
  RASTER_FINITE_WGSL,
  createRasterCountNodes,
  rasterBinding,
  validateRasterGraph,
  validateRasterViews,
  type RasterScalarView
} from '../remote-sensing/raster-kernel-utils';

/** Slices: the dense temporal matrix grows with the square of the slice count. */
export const MAXIMUM_SNAPSHOT_COVARIANCE_SLICE_COUNT = 256;
/** Cells summed per accumulation invocation, a bounded loop over one block of cells. */
const SNAPSHOT_COVARIANCE_CELL_BLOCK = 256;
/** Accumulation invocations: at most 65,535 workgroups of 256 in one dispatch. */
const MAXIMUM_SNAPSHOT_COVARIANCE_INVOCATION_COUNT = 65535 * 256;
/** Float32 elements of the per-frame parameters: fixed-point unit exponent and value bound. */
export const GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH = 2;

/** Per-frame settings of {@link GPURasterSnapshotCovariance}. */
export type GPURasterSnapshotCovarianceSettings = {
  /** Largest absolute sample; a sample above it excludes its cell. Sets the fixed-point unit. */
  maximumValueMagnitude: number;
};

/** Returns the biased float32 exponent of a term bound, as `getFixedPointExponentBits` does on the GPU. */
function getExponentBits(value: number): number {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0];
  return Math.max((bits >>> 23) & 0xff, 1);
}

/**
 * Packs {@link GPURasterSnapshotCovarianceSettings}: `[unitExponent, maximumValueMagnitude]`.
 * Terms are `weight * anomaly * anomaly`, at most `4 * magnitude^2`; the unit leaves the fixed-point
 * sum `cellCount` terms of headroom, so the 96-bit total cannot overflow. Pass `unitExponent`
 * (element 0) to `getGPURasterSnapshotCovarianceMatrix`.
 */
export function getGPURasterSnapshotCovarianceParameterValues(
  settings: GPURasterSnapshotCovarianceSettings,
  cellCount: number,
  target: Float32Array = new Float32Array(GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterSnapshotCovarianceParameterValues';
  validateRequirement(
    id,
    target.length >= GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH,
    `target must hold at least ${GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH} float32 words; received ${target.length}`
  );
  validateFiniteNumber(id, 'maximumValueMagnitude', settings.maximumValueMagnitude, 0);
  const magnitude = settings.maximumValueMagnitude;
  // Rounded up slightly so float32 rounding of a product cannot exceed the bound's exponent.
  const exponentBits = getExponentBits(4 * magnitude * magnitude * 1.0000005);
  target[0] = exponentBits - 127 + 1 + getFixedPointHeadroomBits(cellCount) - 95;
  target[1] = magnitude;
  return target;
}

/** Properties of {@link GPURasterSnapshotCovariance}. */
export type GPURasterSnapshotCovarianceProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`. A nonfinite or out-of-range sample excludes the whole cell. */
  values: GraphDataView<'float32'>;
  /** Per-observation QA; zero excludes the whole cell. */
  mask?: GraphDataView<'uint32'>;
  /** Optional per-cell area weights in `(0, 1]`, for example `cos(latitude)`. A weight outside that range excludes the cell. Defaults to 1. */
  weights?: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterSnapshotCovarianceParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** Packed upper triangle of `X_w' X_w`, row by row, as 96-bit fixed-point integers (three `uint32` words per entry). Read back and decode with `getGPURasterSnapshotCovarianceMatrix`. */
    covarianceWords: GraphDataView<'uint32'>;
    /** One `uint32`: cells that contributed. Complete cases only; xeofs drops features with missing values the same way. */
    keptCellCount: GraphDataView<'uint32'>;
  };
};

/**
 * The GPU half of the method of snapshots for EOF analysis: the `sliceCount x sliceCount` Gram
 * matrix `X_w' X_w` of the temporal-mean-removed, area-weighted field. Each cell's temporal mean
 * is removed first, and only cells with no missing sample contribute. Terms are `weight * anomaly
 * * anomaly`, so an area weight `w` is applied as `sqrt(w)` on each cell of the data matrix.
 *
 * Temporal centering uses a first-sample anchor only when every subtraction is exact by Sterbenz;
 * otherwise it uses zero, preserving small signed observations. A local 96-bit sum preserves
 * temporal cancellation.
 *
 * The cross-cell sum uses the shared fixed-point accumulator, per block of 256 cells and matrix entry, so it
 * is independent of thread order and accurate past float32. The eigenproblem is deliberately not
 * here: read `covarianceWords` back and call `getGPURasterSnapshotEOFDecomposition` on the CPU
 * (no contributor reads back), then upload the temporal modes to `GPURasterEOFProjection`.
 *
 * The GPU route accepts at most 256 slices per cell. Accumulation requires
 * `ceil(cellCount / 256) * sliceCount * (sliceCount + 1) / 2 <= 16,776,960` invocations.
 *
 * Per-frame: `parameters`. Compile-time: `cellCount`, `sliceCount`, which optional views exist.
 */
export class GPURasterSnapshotCovariance implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly entryCount: number;
  private readonly blockCount: number;

  constructor(readonly props: GPURasterSnapshotCovarianceProps) {
    const id = props.id ?? 'raster-snapshot-covariance';
    this.id = id;
    validateGraphPorts(id, props, {
      values: {required: true, viewType: 'data', formats: ['float32']},
      mask: {required: false, viewType: 'data', formats: ['uint32']},
      weights: {required: false, viewType: 'data', formats: ['float32']},
      parameters: {required: true, viewType: 'data', formats: ['float32']},
      'output.covarianceWords': {required: true, viewType: 'data', formats: ['uint32']},
      'output.keptCellCount': {required: true, viewType: 'data', formats: ['uint32']}
    });
    // Capture constructor options/maps while preserving graph resource identities.
    props = {...props, output: {...props.output}};
    this.props = props;
    const {cellCount, sliceCount} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'sliceCount', sliceCount, 2, MAXIMUM_SNAPSHOT_COVARIANCE_SLICE_COUNT);
    this.entryCount = (sliceCount * (sliceCount + 1)) / 2;
    this.blockCount = Math.ceil(cellCount / SNAPSHOT_COVARIANCE_CELL_BLOCK);
    validateRequirement(
      id,
      this.blockCount * this.entryCount <= MAXIMUM_SNAPSHOT_COVARIANCE_INVOCATION_COUNT,
      `accumulation invocations (cellCount ${cellCount}, sliceCount ${sliceCount}) must be at most ${MAXIMUM_SNAPSHOT_COVARIANCE_INVOCATION_COUNT}`
    );
    const observations = cellCount * sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observations);
    validateRequirement(
      id,
      [props.values, props.mask].every(view => !view || view.length >= observations),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      !props.weights || props.weights.length >= cellCount,
      'weights must hold one entry per cell'
    );
    validateRequirement(
      id,
      props.output.covarianceWords.length >= this.entryCount * 3,
      'output.covarianceWords must hold three words per entry of the upper triangle'
    );
    validateRequirement(
      id,
      props.output.keptCellCount.length >= 1,
      'output.keptCellCount must hold one entry'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_SNAPSHOT_COVARIANCE_PARAMETER_LENGTH} entries`
    );
    validateRasterViews(id, this.getInputs(), this.getOutputs());
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, weights, parameters} = this.props;
    return [values, parameters, ...(mask ? [mask] : []), ...(weights ? [weights] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {covarianceWords, keptCellCount} = this.props.output;
    return [covarianceWords, keptCellCount];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, entryCount, blockCount} = this;
    const {cellCount, sliceCount: slices, values, mask, weights, parameters, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const cellMeans = createTransientView(graph, `${id}-means`, 'float32', cellCount);
    const cellAnchors = createTransientView(graph, `${id}-anchors`, 'float32', cellCount);
    const cellKept = createTransientView(graph, `${id}-kept`, 'uint32', cellCount);
    const weightExpression = weights ? 'weights[weightsOffset + cell]' : '1.0';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cell-means`,
        operation: 'GPURasterSnapshotCovariance',
        invocationCount: cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(mask ? [rasterBinding('mask', mask)] : []),
          ...(weights ? [rasterBinding('weights', weights)] : []),
          rasterBinding('cellMeans', cellMeans, true),
          rasterBinding('cellAnchors', cellAnchors, true),
          rasterBinding('cellKept', cellKept, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}
${FIXED_POINT_WGSL}`,
        body: `let cell = index;
let limit = parameters[parametersOffset + 1u];
var kept = ${weightExpression} > 0.0 && ${weightExpression} <= 1.0;
${getRasterTemporalCenteringBody(slices, 'cell', Boolean(mask))}
cellMeans[cellMeansOffset + cell] = select(0.0, meanOffset, kept);
cellAnchors[cellAnchorsOffset + cell] = anchor;
cellKept[cellKeptOffset + cell] = select(0u, 1u, kept);`
      }),
      createFixedPointClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: 'GPURasterSnapshotCovariance',
        words: output.covarianceWords
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-accumulate`,
        operation: 'GPURasterSnapshotCovariance',
        invocationCount: blockCount * entryCount,
        workgroupSize: SNAPSHOT_COVARIANCE_CELL_BLOCK,
        bindings: [
          rasterBinding('values', values),
          rasterBinding('parameters', parameters),
          ...(weights ? [rasterBinding('weights', weights)] : []),
          rasterBinding('cellMeans', cellMeans),
          rasterBinding('cellAnchors', cellAnchors),
          rasterBinding('cellKept', cellKept),
          {
            name: 'covarianceWords',
            view: output.covarianceWords,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        declarations: `${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('covarianceWords')}
${RASTER_TEMPORAL_CENTERING_WGSL}`,
        body: `let entry = index % ${entryCount}u;
let cellBlock = index / ${entryCount}u;
var row = 0u;
var remaining = entry;
while (remaining >= ${slices}u - row) { remaining -= ${slices}u - row; row++; }
let column = row + remaining;
let unitExponent = i32(parameters[parametersOffset]);
let meanUnitExponent = getFixedPointUnitExponent(getFixedPointExponentBits(parameters[parametersOffset + 1u]), ${getFixedPointHeadroomBits(slices)}u);
var total = vec3<u32>(0u);
let firstCell = cellBlock * ${SNAPSHOT_COVARIANCE_CELL_BLOCK}u;
let endCell = min(firstCell + ${SNAPSHOT_COVARIANCE_CELL_BLOCK}u, ${cellCount}u);
for (var cell = firstCell; cell < endCell; cell++) {
  if (cellKept[cellKeptOffset + cell] == 0u) { continue; }
  let meanOffset = cellMeans[cellMeansOffset + cell];
  let anchor = cellAnchors[cellAnchorsOffset + cell];
  let rowAnomaly = getRasterTemporalCenteredValue(values[valuesOffset + cell * ${slices}u + row], anchor, meanOffset, meanUnitExponent);
  let columnAnomaly = getRasterTemporalCenteredValue(values[valuesOffset + cell * ${slices}u + column], anchor, meanOffset, meanUnitExponent);
  let term = (${weightExpression}) * rowAnomaly * columnAnomaly;
  total = addFixedPoint(total, quantizeFixedPoint(term, unitExponent));
}
addFixedPointToCovarianceWords(entry, total);`
      }),
      ...createRasterCountNodes<Parameters>(graph, {
        id: `${id}-kept-count`,
        operation: 'GPURasterSnapshotCovariance',
        values: cellKept,
        count: cellCount,
        condition: 'value != 0u',
        output: output.keptCellCount
      })
    ];
  }
}
