// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
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
  mask?: GraphDataView<'uint32'>;
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
/** Bands: the per-thread design matrix and response arrays scale with bandCount. */
export const MAXIMUM_UNMIXING_BAND_COUNT = 256;
/** Endmembers: the QR factor and projected-gradient state are dense in endmemberCount squared per thread. */
export const MAXIMUM_UNMIXING_ENDMEMBER_COUNT = 8;
/** Per-thread working memory in floats (8 KiB): beyond this, private arrays spill and occupancy collapses. */
export const UNMIXING_WORKING_MEMORY_FLOAT_COUNT = 2048;
/** Projected-gradient rounds run inside one invocation, so the loop bound is capped. */
export const MAXIMUM_UNMIXING_ITERATIONS = 4096;

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
    validateCapacity(this.id, 'cellCount', props.cellCount);
    validateCapacity(this.id, 'bandCount', props.bandCount, 1, MAXIMUM_UNMIXING_BAND_COUNT);
    validateCapacity(
      this.id,
      'endmemberCount',
      props.endmemberCount,
      1,
      MAXIMUM_UNMIXING_ENDMEMBER_COUNT
    );
    validateRequirement(
      this.id,
      props.bandCount >= props.endmemberCount,
      'bandCount must be at least endmemberCount'
    );
    const bands = props.bandCount;
    const members = props.endmemberCount;
    const constrained = (props.constraint ?? 'unconstrained') !== 'unconstrained';
    validateRequirement(
      this.id,
      constrained
        ? bands * (members + 2) + 4 * members + 64 <= UNMIXING_WORKING_MEMORY_FLOAT_COUNT
        : Math.max(
            bands * (members + 1) + members + 64,
            bands * members + members ** 2 + 2 * members + 32
          ) <= UNMIXING_WORKING_MEMORY_FLOAT_COUNT,
      `per-thread working memory (bandCount ${bands}, endmemberCount ${members}) must fit ${UNMIXING_WORKING_MEMORY_FLOAT_COUNT} floats; reduce bandCount or endmemberCount`
    );
    validateCapacity(this.id, 'cellCount times bandCount', props.cellCount * props.bandCount);
    validateCapacity(
      this.id,
      'cellCount times endmemberCount',
      props.cellCount * props.endmemberCount
    );
    validateRequirement(
      this.id,
      ['unconstrained', 'nonnegative', 'simplex'].includes(props.constraint ?? 'unconstrained'),
      "constraint must be 'unconstrained', 'nonnegative' or 'simplex'"
    );
    validateCapacity(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? 256,
      1,
      MAXIMUM_UNMIXING_ITERATIONS
    );
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(props.tolerance ?? 1e-5)) &&
        Math.fround(props.tolerance ?? 1e-5) > 0,
      'tolerance must be a positive finite float32 value'
    );
    validateRequirement(
      this.id,
      props.values.length >= props.cellCount * props.bandCount &&
        props.endmembers.length >= props.bandCount * props.endmemberCount,
      'values must hold cellCount times bandCount entries and endmembers bandCount times endmemberCount'
    );
    validateRequirement(
      this.id,
      !props.mask || props.mask.length >= props.cellCount * props.bandCount,
      'mask must hold cellCount times bandCount entries'
    );
    validateRequirement(
      this.id,
      props.output.abundances.length >= props.cellCount * props.endmemberCount,
      'output.abundances must hold cellCount times endmemberCount entries'
    );
    validateRequirement(
      this.id,
      [props.output.residual, props.output.status, props.output.iterationCounts].every(
        view => view.length >= props.cellCount
      ),
      'output.residual, output.status and output.iterationCounts must hold cellCount entries'
    );
    validateRasterViews(
      this.id,
      [props.values, props.endmembers, ...(props.mask ? [props.mask] : [])],
      Object.values(props.output),
      [
        {
          inputs: [props.values, props.endmembers, ...(props.mask ? [props.mask] : [])],
          outputs: Object.values(props.output)
        }
      ]
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
      [props.values, props.endmembers, ...(props.mask ? [props.mask] : [])],
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
          ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
          ...Object.entries(props.output).map(([name, view]) => rasterBinding(name, view, true))
        ],
        declarations:
          RASTER_FINITE_WGSL + (!constrained ? getRasterLeastSquaresWGSL(bands, members) : ''),
        body: `var design: array<f32, ${bands * members}>; var response: array<f32, ${bands}>; var coefficients: array<f32, ${members}>; var count = 0u;
for (var band = 0u; band < ${bands}u; band++) {
  let source = index * ${bands}u + band; let value = values[valuesOffset + source];
  if (!finite(value) ${props.mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
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
      if (mapping <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)} * max(1.0, gradientScale)) { code = 0u; break; }
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
