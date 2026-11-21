// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getColumnQuantileNodes} from '../column-classification/column-quantiles-nodes';
import {PAIR_STATISTICS_FLOAT_WGSL} from './pair-statistics-grid';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** @internal Encode/decode pair terms without forming an overflowing reciprocal for tiny units. */
export const PAIR_STATISTICS_UNIT_SCALING_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
fn getPairStatisticsFixedPointScale(unit: f32, unitPower: i32) -> f32 {
  // First f32 above 2^-104: 2^24 / unit is finite at and above this value.
  // Keep the original arithmetic whenever its reciprocal is representable.
  // A nonzero stored exponent does not imply an unrepresentable reciprocal.
  // A subnormal physical unit may flush here: it is below the normal threshold either way,
  // and the fallback retains the original mantissa and exponent.
  let physicalUnit = ldexp(unit, unitPower);
  if (physicalUnit >= ${2 ** -104 * (1 + 2 ** -23)}) { return 16777216.0 / physicalUnit; }
  return 0.0;
}
fn scalePairStatisticsAmount(amount: f32, unit: f32, unitPower: i32, scale: f32) -> f32 {
  if (scale > 0.0) { return amount * scale; }
  if (unit > 0.0) { return getScaledQuotient(amount, unit, 1.0, 24 - unitPower); }
  // Retain the original zero-unit arithmetic and nonfinite-term filtering.
  return amount * scale;
}
// The histogram's nonzero amount is a normal integer, and a selected scale is normal.
// Divide their exact 24-bit significands and round once, ties to even. Float division here
// could lower to a rounded (or flushed) reciprocal and lose the required quotient rounding.
fn getPairStatisticsDecodedQuotient(amount: f32, scale: f32) -> f32 {
  let amountBits = bitcast<u32>(amount);
  let magnitude = amountBits & 0x7fffffffu;
  if (magnitude == 0u || magnitude >= 0x7f800000u) { return amount; }
  let sign = amountBits & 0x80000000u;
  let scaleBits = bitcast<u32>(scale);
  var numerator = (magnitude & 0x7fffffu) | 0x800000u;
  let denominator = (scaleBits & 0x7fffffu) | 0x800000u;
  var exponent = i32(magnitude >> 23u) - i32(scaleBits >> 23u);
  if (numerator < denominator) { numerator <<= 1u; exponent -= 1; }
  if (exponent < -150) { return bitcast<f32>(sign); }
  if (exponent == -150) {
    return bitcast<f32>(sign | select(0u, 1u, numerator > denominator));
  }
  var quotient = 1u;
  var remainder = numerator - denominator;
  // At most 23 fractional bits; a subnormal result needs fewer before rounding.
  let fractionBits = u32(clamp(exponent + 149, 0, 23));
  for (var digit = 0u; digit < fractionBits; digit++) {
    remainder <<= 1u;
    quotient <<= 1u;
    if (remainder >= denominator) { remainder -= denominator; quotient |= 1u; }
  }
  let doubledRemainder = remainder << 1u;
  if (doubledRemainder > denominator || (doubledRemainder == denominator && (quotient & 1u) != 0u)) {
    quotient += 1u;
  }
  if (exponent < -126) { return bitcast<f32>(sign | quotient); }
  if (quotient == 0x1000000u) { quotient >>= 1u; exponent += 1; }
  if (exponent > 127) { return bitcast<f32>(sign | 0x7f800000u); }
  return bitcast<f32>(sign | (u32(exponent + 127) << 23u) | (quotient & 0x7fffffu));
}
fn decodePairStatisticsAmount(amount: f32, unit: f32, unitPower: i32, scale: f32) -> f32 {
  if (scale > 0.0) { return getPairStatisticsDecodedQuotient(amount, scale); }
  if (unit > 0.0) {
    // Multiplying unit by 2^-24 first could underflow. Combine bounded
    // mantissas and integer exponents so compiler reassociation cannot do so.
    let numerator = frexp(amount);
    let factor = frexp(unit);
    return ldexp(numerator.fract * factor.fract, numerator.exp + factor.exp - 24 + unitPower);
  }
  return 0.0;
}`;

/**
 * Smallest fraction of the variance that the fixed-point unit may take. Bounds the adaptive
 * resolution when the interquartile spread underflows; a unit far below the
 * variance costs capacity of the first channel, not accuracy, because terms above 256 units go to
 * the exact large channel.
 */
export const PAIR_STATISTICS_MINIMUM_UNIT_FRACTION = 2 ** -32;

/**
 * Fraction of the variance used as the unit when the interquartile range is zero (more than half
 * of the values tied) or not finite.
 */
export const PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION = 2 ** -12;

/** Standard deviations in the interquartile range of a normal distribution, `2 * 0.6744898`. @internal */
export const PAIR_STATISTICS_INTERQUARTILE_RANGE_PER_SIGMA = 1.3489795;

/** A variance below this (f32) retains the existing constant-column numerical floor. @internal */
export const PAIR_STATISTICS_MINIMUM_VARIANCE = 1e-30;

/**
 * Builds the nodes that write the "typical term variance" `u` of a value column to a
 * two-word transient `[scaledUnit, binaryExponent]`: the robust variance `(IQR / 1.349)^2`
 * from the exact 25% and 75% quantiles of the
 * (masked, finite) values, clamped to `[variance * 2^-32, variance]` with `variance =
 * sumOfSquares / n` from the statistics pass (`variance * 2^-12` when the IQR is 0 or not finite). A kernel reads `u` and scales a pair term whose
 * typical magnitude is one variance (`(z_i - z_j)^2 / 2`, `c_i c_j`) by `2^24 / u`,
 * using exponent-separated arithmetic when that reciprocal is not representable. For variance
 * below 2^-64, calculate/store the unit multiplied by 2^64 and retain exponent -64. Thus even
 * the minimum adaptive fraction above the variance floor stays normal in storage.
 *
 * The plain variance would not do: it is itself dominated by one outlier or sentinel
 * (`variance ~ outlier^2 / n`), so the fixed-point unit would again scale with the maximum. The
 * interquartile range ignores up to a quarter of contaminated rows. `u` is 0 when the variance is
 * not positive or finite (every term then reads as 0, as for a constant column).
 *
 * @internal
 */
export function getPairStatisticsUnitVarianceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** Packed float32 observation column. */
    values: GraphDataView<'float32'>;
    /** Optional row mask. */
    mask?: GraphDataView<'uint32'>;
    /** Cell membership from the same front end as the moments; excluded rows have cellCount. */
    population?: {cellKeys: GraphDataView<'uint32'>; cellCount: number};
    /** The statistics view of `getPairStatisticsInputNodes` (centered values, then moments). */
    statistics: GraphDataView<'float32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; unitVariance: GraphDataView<'float32'>} {
  const {id, operation, values, mask, statistics} = props;
  const rows = values.length;
  const quantileParameters = createTransientView(graph, `${id}-unit-parameters`, 'float32', 6);
  const quantiles = createTransientView(graph, `${id}-unit-quantiles`, 'float32', 2);
  const validCount = createTransientView(graph, `${id}-unit-valid-count`, 'uint32', 1);
  // [unit mantissa, binary exponent]. Tiny units must never be stored as subnormal f32.
  const unitVariance = createTransientView(graph, `${id}-unit-variance`, 'float32', 2);
  const populationMask = props.population
    ? createTransientView(graph, `${id}-unit-population`, 'uint32', rows)
    : mask;
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (props.population) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unit-population`,
        operation,
        variant: 'unit-population',
        invocationCount: rows,
        bindings: [
          {name: 'cellKeys', view: props.population.cellKeys, type: 'u32', access: 'read'},
          {name: 'populationMask', view: populationMask!, type: 'u32', access: 'read_write'}
        ],
        body: `populationMask[populationMaskOffset + index] = select(0u, 1u, cellKeys[cellKeysOffset + index] < ${props.population.cellCount}u);`
      })
    );
  }
  nodes.push(
    // [linear interpolation, filter lower, filter upper, 0, 0.25, 0.75]
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unit-parameters`,
      operation,
      variant: 'unit-parameters',
      bindings: [
        {name: 'quantileParameters', view: quantileParameters, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `quantileParameters[quantileParametersOffset] = 3.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  quantileParameters[quantileParametersOffset + 4u] = 0.25;
  quantileParameters[quantileParametersOffset + 5u] = 0.75;`
    }),
    ...getColumnQuantileNodes<Parameters>(graph, {
      id: `${id}-unit-quantile`,
      operation,
      values,
      mask: populationMask,
      parameters: quantileParameters,
      quantileCount: 2,
      output: {quantiles, validCount}
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unit-variance`,
      operation,
      variant: 'unit-variance',
      bindings: [
        {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
        {name: 'quantiles', view: quantiles, type: 'f32', access: 'read'},
        {name: 'unitVariance', view: unitVariance, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `let variance = statistics[statisticsOffset + MOMENTS + 2u] / statistics[statisticsOffset + MOMENTS];
  let unitPower = select(0.0, -64.0, variance < ${2 ** -64});
  let scaledVariance = ldexp(variance, -i32(unitPower));
  let spread = ldexp((quantiles[quantilesOffset + 1u] - quantiles[quantilesOffset]) / ${PAIR_STATISTICS_INTERQUARTILE_RANGE_PER_SIGMA}, -i32(unitPower) / 2);
  let usable = isFiniteValue(spread) && spread > 0.0;
  let robustVariance = select(scaledVariance * ${PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION}, spread * spread, usable);
  let unit = clamp(robustVariance, scaledVariance * ${PAIR_STATISTICS_MINIMUM_UNIT_FRACTION}, scaledVariance);
  unitVariance[unitVarianceOffset + 1u] = unitPower;
  unitVariance[unitVarianceOffset] =
    select(0.0, unit, isFiniteValue(variance) && variance > ${PAIR_STATISTICS_MINIMUM_VARIANCE});`
    })
  );
  return {nodes, unitVariance};
}
