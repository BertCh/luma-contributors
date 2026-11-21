// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRegressionBindings,
  getRegressionWGSL
} from './gpu-generalized-geographically-weighted-regression';
import type {GPUMultiscaleGeographicallyWeightedRegression} from './gpu-multiscale-geographically-weighted-regression';

/** Full-inference row limit, enforced at MGWR construction. @internal */
export const MGWR_INFERENCE_MAXIMUM_ROW_COUNT = 128;
/** Full-inference n³ × (p+1) × maximumIterations limit, enforced at construction. @internal */
export const MGWR_INFERENCE_MAXIMUM_WORK = 268435456;

/** Propagate every response derivative through the actual selected backfit history. @internal */
export function createMultiscaleInference<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUMultiscaleGeographicallyWeightedRegression,
  parameters: GraphDataView<'float32'>,
  state: GraphDataView<'uint32'>
) {
  const {id, props} = contributor;
  const inference = props.inference!;
  const rows = props.response.length,
    columns = props.predictorCount + 1;
  const length = rows * rows * columns;
  let source = createTransientView(graph, `${id}-response-smoother-first`, 'float32', length),
    destination = createTransientView(graph, `${id}-response-smoother-second`, 'float32', length);
  const bindings = getRegressionBindings({...props, parameters});
  const common = getRegressionWGSL(props, {sphereRadius: contributor.sphereRadius});
  validateGraphViewsBelongToGraph(id, graph, Object.values(inference.output));
  return {
    initialize: () =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-response-smoother`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'initialize-full-inference',
        invocationCount: length,
        bindings: [{name: 'smoother', view: source, type: 'f32', access: 'read_write'}],
        body: 'smoother[smootherOffset + index] = 0.0;'
      }),
    advance: (iteration: number, column: number) => {
      const node = createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-response-smoother-${iteration}-${column}`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-backfit-smoother',
        invocationCount: rows * rows,
        bindings: [
          ...bindings,
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {name: 'destination', view: destination, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read'}
        ],
        declarations: common,
        body: `let focus = index / ${rows}u; let responseRow = index % ${rows}u; let base = index * ${columns}u; for (var component = 0u; component < ${columns}u; component++) { destination[destinationOffset + base + component] = source[sourceOffset + base + component]; } if (state[stateOffset] != 3u || !included(focus) || !included(responseRow)) { return; }
let bandwidth = parameters[parametersOffset + ${column + 1}u]; var denominator = 0.0; var numerator = 0.0;
for (var row = 0u; row < ${rows}u; row++) { if (!included(row)) { continue; } let value = predictor(row, ${column}u); let weight = spatialWeight(focus,row,bandwidth); denominator += weight * value * value; var partial = select(0.0,1.0,row == responseRow); for (var other = 0u; other < ${columns}u; other++) { if (other != ${column}u) { partial -= predictor(row,other) * source[sourceOffset + (row * ${rows}u + responseRow) * ${columns}u + other]; } } numerator += weight * value * partial; }
destination[destinationOffset + base + ${column}u] = select(getNaN(),numerator / denominator,denominator > 0.0);`
      });
      [source, destination] = [destination, source];
      return node;
    },
    finish: () => [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-full-inference-summary`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-inference-summary',
        // contract-allow: single-invocation; full inference requires MGWR_INFERENCE_MAXIMUM_ROW_COUNT = 128 and MGWR_INFERENCE_MAXIMUM_WORK = 268435456, enforced at construction for n³ × (p+1) × maximumIterations.
        invocationCount: 1,
        bindings: [
          {name: 'smoother', view: source, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read'},
          {name: 'codes', view: props.output.status.codes, type: 'u32', access: 'read'},
          {name: 'summary', view: inference.output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: inference.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
// Rebase before subtraction so even opposite finite float32 extremes produce a finite
// residual. The bit path prevents Metal from moving the scale past the subtraction or square.
const INFERENCE_RESIDUAL_SHIFT: u32 = 70u;
fn scaleInferenceResidualOperand(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) { return value; }
  // Operands whose scaled value would be subnormal cannot affect the large residual sum.
  if (magnitude < ((INFERENCE_RESIDUAL_SHIFT + 1u) << 23u)) { return 0.0; }
  return bitcast<f32>(bits - (INFERENCE_RESIDUAL_SHIFT << 23u));
}`,
        body: `status[statusOffset] = 1u; for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); } var trace = 0.0; var traceSquared = 0.0; var residualSquares = 0.0; var count = 0u;
var scaleResiduals = false;
for (var focus = 0u; focus < ${rows}u; focus++) {
  let code = codes[codesOffset + focus];
  if (code == 1u) { continue; }
  if (code != 0u) { return; }
  count++;
  if (!scaleResiduals) {
    let observed = response[responseOffset + focus];
    let predicted = fitted[fittedOffset + focus];
    let observedBits = bitcast<u32>(observed);
    let predictedBits = bitcast<u32>(predicted);
    let oppositeSigns = ((observedBits ^ predictedBits) & 0x80000000u) != 0u;
    let operandMagnitude = max(observedBits & 0x7fffffffu, predictedBits & 0x7fffffffu);
    if (oppositeSigns && operandMagnitude >= 0x7f000000u) {
      // A subtraction of opposite signs can exceed float32 at this magnitude (2^127).
      scaleResiduals = true;
    } else {
      let residual = observed - predicted;
      // At most 128 squares of residuals below 2^60 sum to less than 2^127.
      if ((bitcast<u32>(residual) & 0x7fffffffu) >= 0x5d800000u) {
        scaleResiduals = true;
      } else {
        residualSquares += residual * residual;
      }
    }
  }
for (var responseRow = 0u; responseRow < ${rows}u; responseRow++) { var influence = 0.0; for (var column = 0u; column < ${columns}u; column++) { var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + focus * ${columns - 1}u + column - 1u]; } influence += value * smoother[smootherOffset + (focus * ${rows}u + responseRow) * ${columns}u + column]; } traceSquared += influence * influence; if (focus == responseRow) { trace += influence; } } }
let degrees = f32(count) - ${inference.varianceConvention === 'smoother-df' ? '(2.0 * trace - traceSquared)' : 'trace'};
var variance = getNaN();
let degreesBits = bitcast<u32>(degrees);
if (!scaleResiduals) {
  variance = residualSquares / degrees;
} else if (degreesBits < 0x7f800000u) {
  // With at most 128 rows, scaling both operands by 2^-70 keeps the sum below 2^125.
  var scaledSquares = 0.0;
  for (var focus = 0u; focus < ${rows}u; focus++) {
    if (codes[codesOffset + focus] != 0u) { continue; }
    let scaledResidual = scaleInferenceResidualOperand(response[responseOffset + focus]) -
      scaleInferenceResidualOperand(fitted[fittedOffset + focus]);
    scaledSquares += scaledResidual * scaledResidual;
  }
  if (isFiniteBits(scaledSquares) && scaledSquares > 0.0 && degreesBits < 0x00800000u) {
    // Dividing a residual square of at least 2^120 by these degrees overflows. WGSL rejects
    // a constant infinity, so the bit pattern is a runtime value.
    var infinityBits = 0x7f800000u;
    variance = bitcast<f32>(infinityBits);
  } else if (isFiniteBits(scaledSquares) && scaledSquares > 0.0) {
    // Divide by the significand to avoid a subnormal reciprocal on Metal. In this
    // branch both the scaled quotient and any representable variance are normal.
    let degreesSignificand = bitcast<f32>((degreesBits & 0x007fffffu) | 0x3f800000u);
    let ratioBits = bitcast<u32>(scaledSquares / degreesSignificand);
    let restoredExponent = i32(ratioBits >> 23u) - (i32(degreesBits >> 23u) - 127) +
      2 * i32(INFERENCE_RESIDUAL_SHIFT);
    let exponentBits = u32(min(restoredExponent, 255)) << 23u;
    let fractionBits = ratioBits & 0x007fffffu;
    variance = bitcast<f32>(exponentBits | select(fractionBits, 0u, restoredExponent >= 255));
  }
}
summary[summaryOffset] = trace; summary[summaryOffset + 1u] = traceSquared; summary[summaryOffset + 2u] = variance; summary[summaryOffset + 3u] = degrees; status[statusOffset] = select(2u,0u,count > 0u && degrees > 0.0 && variance >= 0.0 && isFiniteBits(variance) && isFiniteBits(traceSquared));`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-full-inference-publish`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'full-inference-publish',
        invocationCount: rows * columns,
        bindings: [
          {name: 'smoother', view: source, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'codes', view: props.output.status.codes, type: 'u32', access: 'read'},
          {name: 'summary', view: inference.output.summary, type: 'f32', access: 'read'},
          {name: 'status', view: inference.output.status, type: 'u32', access: 'read'},
          {
            name: 'standardErrors',
            view: inference.output.standardErrors,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'effective',
            view: inference.output.effectiveParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: GET_NAN_WGSL,
        body: `let focus = index / ${columns}u; let column = index % ${columns}u; standardErrors[standardErrorsOffset + index] = getNaN(); if (focus == 0u) { effective[effectiveOffset + column] = getNaN(); } if (status[statusOffset] != 0u) { return; }
if (codes[codesOffset + focus] == 0u) {
  var sum = 0.0;
  for (var responseRow = 0u; responseRow < ${rows}u; responseRow++) {
    let derivative = smoother[smootherOffset + (focus * ${rows}u + responseRow) * ${columns}u + column];
    sum += derivative * derivative;
  }
  let variance = summary[summaryOffset + 2u];
  let sumBits = bitcast<u32>(sum) & 0x7fffffffu;
  let varianceBits = bitcast<u32>(variance) & 0x7fffffffu;
  let exponentSum = (sumBits >> 23u) + (varianceBits >> 23u);
  if (sumBits >= 0x7f800000u) {
    standardErrors[standardErrorsOffset + index] = sqrt(sum);
  } else if (exponentSum >= 381u) {
    // A combined unbiased exponent of 127 (biased sum 381) leaves no exponent headroom, and
    // Metal folds sqrt(sum) * sqrt(variance) back into sqrt(sum * variance). Both operands are
    // normal and at least 1 here: take the root of the significand product and restore half the
    // exponent with integer arithmetic, which the compiler cannot reassociate.
    let exponentTotal = i32(exponentSum) - 254;
    let oddExponent = u32(exponentTotal & 1);
    let sumSignificand = bitcast<f32>((sumBits & 0x007fffffu) | 0x3f800000u);
    let varianceSignificand = bitcast<f32>(((varianceBits & 0x007fffffu) | 0x3f800000u) + (oddExponent << 23u));
    let rootBits = bitcast<u32>(sqrt(sumSignificand * varianceSignificand));
    let restoredExponent = i32(rootBits >> 23u) + (exponentTotal - i32(oddExponent)) / 2;
    var infinityBits = 0x7f800000u;
    standardErrors[standardErrorsOffset + index] = bitcast<f32>(
      select((rootBits & 0x007fffffu) | (u32(min(restoredExponent, 254)) << 23u), infinityBits, restoredExponent >= 255)
    );
  } else {
    standardErrors[standardErrorsOffset + index] = sqrt(max(0.0, sum * variance));
  }
}
if (focus == 0u) { var count = 0.0; for (var row = 0u; row < ${rows}u; row++) { if (codes[codesOffset + row] != 0u) { continue; } var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + row * ${columns - 1}u + column - 1u]; } count += value * smoother[smootherOffset + (row * ${rows}u + row) * ${columns}u + column]; } effective[effectiveOffset + column] = count; }`
      })
    ]
  };
}
