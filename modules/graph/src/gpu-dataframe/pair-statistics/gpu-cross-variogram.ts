// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Signed empirical cross-semivariance. Times add a second, temporal lag axis. */
export type GPUCrossVariogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  /** Defaults to values, giving the ordinary Matheron semivariogram. */
  secondValues?: GraphDataView<'float32'>;
  times?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** `[maximumDistance, maximumTimeLag]`; positive finite entries. Time ignored without times. */
  parameters: GraphDataView<'float32'>;
  lagCount: number;
  temporalLagCount?: number;
  output: {
    /** Spatial-major bins: spatialLag * temporalLagCount + temporalLag. */
    semivariances: GraphDataView<'float32'>;
    pairCounts: GraphDataView<'uint32'>;
    /** Per bin: 0 valid, 1 empty/invalid parameters, 2 nonfinite accumulated arithmetic. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Deterministic GPU pair accumulation of (x_i-x_j)(y_i-y_j)/2. Negative cross-semivariances are
 * retained. Unordered pairs counted once; bins are left-closed and right-open, except the final
 * edge is included. Complete-case masking includes both variables, coordinates and optional time.
 * Uses bounded dense traversal (at most 16,384 rows and 128 bins), O(rows² * bins) work and
 * O(rows * bins) scratch. No edge correction or implicit projection. Intended for bounded analyses;
 * callers must measure this traversal against the indexed univariate contributor for large data.
 */
export class GPUCrossVariogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUCrossVariogramProps) {
    this.id = props.id ?? 'cross-variogram';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    const temporalCount = props.temporalLagCount ?? 1;
    if (
      !Number.isInteger(props.lagCount) ||
      props.lagCount < 1 ||
      !Number.isInteger(temporalCount) ||
      temporalCount < 1 ||
      props.lagCount * temporalCount > 128 ||
      rows < 1 ||
      rows > 16384 ||
      (!props.times && temporalCount !== 1)
    )
      throw new Error(`${this.id}: invalid bounded pair shape`);
    validatePackedView(props.positions, ['float32x2'], this.id);
    validatePackedView(props.parameters, ['float32'], this.id);
    if (props.parameters.length < (props.times ? 2 : 1))
      throw new Error(`${this.id}: missing lag parameters`);
    for (const view of [props.values, props.secondValues, props.times])
      if (view) {
        validatePackedView(view, ['float32'], this.id);
        if (view.length !== rows) throw new Error(`${this.id}: values must match positions`);
      }
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], this.id);
      if (props.mask.length !== rows) throw new Error(`${this.id}: mask length mismatch`);
    }
    const bins = props.lagCount * temporalCount;
    for (const [name, view] of Object.entries(props.output)) {
      validatePackedView(view, [name === 'semivariances' ? 'float32' : 'uint32'], this.id);
      if (view.length !== bins) throw new Error(`${this.id}: output length must match bins`);
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      Object.values(props.output),
      getBindings(props).map(binding => binding.view)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.positions.length;
    const temporalCount = props.temporalLagCount ?? 1;
    const bins = props.lagCount * temporalCount;
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows * bins * 2);
    const bindings = getBindings(props);
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-pairs`,
        operation: 'GPUCrossVariogram',
        variant: props.times ? 'space-time' : 'cross',
        invocationCount: rows * bins,
        bindings: [
          ...bindings,
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn included(row: u32) -> bool {
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  return isFiniteBits(positions[positionsOffset + 2u * row]) && isFiniteBits(positions[positionsOffset + 2u * row + 1u]) && isFiniteBits(values[valuesOffset + row])
    ${props.secondValues ? '&& isFiniteBits(secondValues[secondValuesOffset + row])' : ''}
    ${props.times ? '&& isFiniteBits(times[timesOffset + row])' : ''};
}`,
        body: `let focus = index / ${bins}u; let bin = index % ${bins}u;
let maximumDistance = parameters[parametersOffset];
let maximumTime = ${props.times ? 'parameters[parametersOffset + 1u]' : '1.0'};
var sum = 0.0; var count = 0.0;
if (included(focus) && maximumDistance > 0.0 && isFiniteBits(maximumDistance) && maximumTime > 0.0 && isFiniteBits(maximumTime)) {
  for (var neighbor = focus + 1u; neighbor < ${rows}u; neighbor++) {
    if (!included(neighbor)) { continue; }
    let delta = vec2<f32>(positions[positionsOffset + 2u * focus] - positions[positionsOffset + 2u * neighbor], positions[positionsOffset + 2u * focus + 1u] - positions[positionsOffset + 2u * neighbor + 1u]);
    let distance = length(delta);
    let time = ${props.times ? 'abs(times[timesOffset + focus] - times[timesOffset + neighbor])' : '0.0'};
    if (!isFiniteBits(distance) || distance > maximumDistance || time > maximumTime) { continue; }
    let spatialBin = min(u32(distance / maximumDistance * ${props.lagCount}.0), ${props.lagCount - 1}u);
    let temporalBin = min(u32(time / maximumTime * ${temporalCount}.0), ${temporalCount - 1}u);
    if (spatialBin * ${temporalCount}u + temporalBin != bin) { continue; }
    let difference = values[valuesOffset + focus] - values[valuesOffset + neighbor];
    let secondDifference = ${props.secondValues ? 'secondValues[secondValuesOffset + focus] - secondValues[secondValuesOffset + neighbor]' : 'difference'};
    sum += 0.5 * difference * secondDifference; count += 1.0;
  }
}
partials[partialsOffset + index * 2u] = sum; partials[partialsOffset + index * 2u + 1u] = count;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPUCrossVariogram',
        variant: 'finish',
        invocationCount: bins,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'result', view: props.output.semivariances, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var sum = 0.0; var count = 0u;
for (var row = 0u; row < ${rows}u; row++) { let base = partialsOffset + (row * ${bins}u + index) * 2u; sum += partials[base]; count += u32(partials[base + 1u]); }
let value = sum / f32(count);
result[resultOffset + index] = select(getNaN(), value, count > 0u && isFiniteBits(value));
counts[countsOffset + index] = count;
status[statusOffset + index] = select(select(2u, 0u, isFiniteBits(value)), 1u, count == 0u);`
      })
    ];
  }
}

/** Space-time semivariogram, with an optional second field for signed cross-semivariance. */
export type GPUSpaceTimeVariogramProps = GPUCrossVariogramProps & {
  times: GraphDataView<'float32'>;
  temporalLagCount: number;
};
export class GPUSpaceTimeVariogram extends GPUCrossVariogram {
  constructor(props: GPUSpaceTimeVariogramProps) {
    super({...props, id: props.id ?? 'space-time-variogram'});
  }
}
function getBindings(props: GPUCrossVariogramProps): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    ...(props.secondValues
      ? [
          {
            name: 'secondValues',
            view: props.secondValues,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : []),
    ...(props.times
      ? [{name: 'times', view: props.times, type: 'f32' as const, access: 'read' as const}]
      : []),
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : [])
  ];
}
