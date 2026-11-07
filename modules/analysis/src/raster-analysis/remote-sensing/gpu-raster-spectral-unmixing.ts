// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterCount,
  validateRasterViews,
  validateRasterGraph,
  getRasterLeastSquaresWGSL
} from './raster-kernel-utils';

export type GPURasterSpectralUnmixingProps = {
  id?: string;
  cellCount: number;
  bandCount: number;
  endmemberCount: number;
  /** Pixel-major spectra and endmember-major spectra, respectively. NaN bands are omitted. */
  values: GraphDataView<'float32'>;
  endmembers: GraphDataView<'float32'>;
  validity?: GraphDataView<'uint32'>;
  constraint?: 'unconstrained' | 'nonnegative' | 'simplex';
  maximumIterations?: number;
  tolerance?: number;
  output: {
    abundances: GraphDataView<'float32'>;
    residual: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
    iterationCounts: GraphDataView<'uint32'>;
  };
};
/** Small QR least squares (up to 256 bands/8 endmembers), or bounded projected-gradient NNLS.
 * Simplex mode enforces nonnegativity and sum one by Euclidean projection, not clipping.
 * Status 0=solved, 1=insufficient valid bands, 2=rank/nonfinite failure, 4=iteration limit.
 * Exhausted constrained fits retain feasible abundances and squared residuals for inspection.
 * tolerance bounds the projected-gradient mapping relative to the initial gradient.
 */
export class GPURasterSpectralUnmixing implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterSpectralUnmixingProps) {
    this.id = props.id ?? 'raster-spectral-unmixing';
    validateRasterCount(props.cellCount);
    validateRasterCount(props.bandCount, 256);
    validateRasterCount(props.endmemberCount, 8);
    assert(props.bandCount >= props.endmemberCount);
    const bands = props.bandCount;
    const members = props.endmemberCount;
    const constrained = (props.constraint ?? 'unconstrained') !== 'unconstrained';
    // Bound explicit per-function working arrays and scalar locals to 8 KiB.
    assert(
      constrained
        ? bands * (members + 2) + 4 * members + 64 <= 2048
        : Math.max(
            bands * (members + 1) + members + 64,
            bands * members + members ** 2 + 2 * members + 32
          ) <= 2048
    );
    validateRasterCount(props.cellCount * props.bandCount);
    validateRasterCount(props.cellCount * props.endmemberCount);
    assert(
      ['unconstrained', 'nonnegative', 'simplex'].includes(props.constraint ?? 'unconstrained')
    );
    validateRasterCount(props.maximumIterations ?? 256, 4096);
    assert(
      Number.isFinite(Math.fround(props.tolerance ?? 1e-5)) &&
        Math.fround(props.tolerance ?? 1e-5) > 0
    );
    assert(
      props.values.length >= props.cellCount * props.bandCount &&
        props.endmembers.length >= props.bandCount * props.endmemberCount
    );
    assert(!props.validity || props.validity.length >= props.cellCount * props.bandCount);
    assert(props.output.abundances.length >= props.cellCount * props.endmemberCount);
    assert(
      [props.output.residual, props.output.status, props.output.iterationCounts].every(
        view => view.length >= props.cellCount
      )
    );
    validateRasterViews(
      this.id,
      [props.values, props.endmembers, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const bands = props.bandCount;
    const members = props.endmemberCount;
    const constrained = (props.constraint ?? 'unconstrained') !== 'unconstrained';
    validateRasterGraph(
      id,
      graph,
      [props.values, props.endmembers, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPURasterSpectralUnmixing',
        invocationCount: props.cellCount,
        workgroupSize: 64,
        bindings: [
          rasterBinding('values', props.values),
          rasterBinding('endmembers', props.endmembers),
          ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
          ...Object.entries(props.output).map(([name, view]) => rasterBinding(name, view, true))
        ],
        declarations:
          RASTER_FINITE_WGSL + (!constrained ? getRasterLeastSquaresWGSL(bands, members) : ''),
        body: `var design: array<f32, ${bands * members}>; var response: array<f32, ${bands}>; var coefficients: array<f32, ${members}>; var count = 0u;
for (var band = 0u; band < ${bands}u; band++) {
  let source = index * ${bands}u + band; let value = values[valuesOffset + source];
  if (!finite(value) ${props.validity ? '|| validity[validityOffset + source] == 0u' : ''}) { continue; }
  var valid = true;
  for (var member = 0u; member < ${members}u; member++) { valid = valid && finite(endmembers[endmembersOffset + member * ${bands}u + band]); }
  if (!valid) { continue; }
  response[band] = value; count++;
  for (var member = 0u; member < ${members}u; member++) { design[member * ${bands}u + band] = endmembers[endmembersOffset + member * ${bands}u + band]; }
}
var code = 1u; var residualValue = missing(); var performed = 0u;
if (count >= ${members}u) {
${
  !constrained
    ? `let fit = leastSquares(design, response); code = 2u; if (fit.valid) { coefficients = fit.coefficients; residualValue = fit.residual; code = 0u; }`
    : `
  var lipschitz = 0.0; var gradientScale = 0.0;
  for (var member = 0u; member < ${members}u; member++) {
    var gradient = 0.0;
    for (var band = 0u; band < ${bands}u; band++) { let value = design[member * ${bands}u + band]; lipschitz += value * value; gradient += value * response[band]; }
    gradientScale = max(gradientScale, abs(gradient));
    ${props.constraint === 'simplex' ? `coefficients[member] = 1.0 / ${members}.0;` : ''}
  }
  code = 2u;
  if (lipschitz > 0.0 && finite(lipschitz)) {
    code = 4u;
    for (var iteration = 0u; iteration < ${props.maximumIterations ?? 256}u; iteration++) {
      var errors: array<f32, ${bands}>;
      for (var band = 0u; band < ${bands}u; band++) { errors[band] = -response[band]; for (var member = 0u; member < ${members}u; member++) { errors[band] += design[member * ${bands}u + band] * coefficients[member]; } }
      var candidate: array<f32, ${members}>;
      for (var member = 0u; member < ${members}u; member++) {
        var gradient = 0.0; for (var band = 0u; band < ${bands}u; band++) { gradient += design[member * ${bands}u + band] * errors[band]; }
        candidate[member] = coefficients[member] - gradient / lipschitz;
      }
      ${
        props.constraint === 'simplex'
          ? `var sorted = candidate;
      for (var member = 1u; member < ${members}u; member++) { let value = sorted[member]; var position = member; loop { if (position == 0u) { break; } if (sorted[position - 1u] >= value) { break; } sorted[position] = sorted[position - 1u]; position--; } sorted[position] = value; }
      var sum = 0.0; var threshold = 0.0;
      for (var member = 0u; member < ${members}u; member++) { sum += sorted[member]; let trial = (sum - 1.0) / f32(member + 1u); if (sorted[member] > trial) { threshold = trial; } }
      for (var member = 0u; member < ${members}u; member++) { candidate[member] = max(candidate[member] - threshold, 0.0); }`
          : `for (var member = 0u; member < ${members}u; member++) { candidate[member] = max(candidate[member], 0.0); }`
      }
      var mapping = 0.0;
      for (var member = 0u; member < ${members}u; member++) { mapping = max(mapping, abs(candidate[member] - coefficients[member]) * lipschitz); }
      coefficients = candidate; performed = iteration + 1u;
      if (mapping <= ${props.tolerance ?? 1e-5} * max(1.0, gradientScale)) { code = 0u; break; }
    }
    residualValue = 0.0;
    for (var band = 0u; band < ${bands}u; band++) { var error = -response[band]; for (var member = 0u; member < ${members}u; member++) { error += design[member * ${bands}u + band] * coefficients[member]; } residualValue += error * error; }
    if (!finite(residualValue)) { code = 2u; }
  }`
}
}
for (var member = 0u; member < ${members}u; member++) { abundances[abundancesOffset + index * ${members}u + member] = select(missing(), coefficients[member], code == 0u || code == 4u); }
residual[residualOffset + index] = residualValue; status[statusOffset + index] = code; iterationCounts[iterationCountsOffset + index] = performed;`
      })
    ];
  }
}
