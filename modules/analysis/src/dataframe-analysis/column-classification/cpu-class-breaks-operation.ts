// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getClassBreaksProvenance,
  resolveClassBreaksMethods,
  validateClassBreaksNaturalOptions,
  validateClassBreaksInputAssertions,
  validateClassBreaksGraphOptions,
  validateClassBreaksInput
} from './class-breaks-input';
import {
  getClassBreaksOperationPorts,
  snapshotClassBreaksOperationProps
} from './class-breaks-operation-snapshot';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows,
  getCPUOperationInt32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getGPUClassBreaksParameterLength,
  GPU_CLASS_BREAKS_METHOD_CODES,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  type GPUClassBreaksMethod
} from './class-breaks-parameters';
import {
  getGPUColumnQuantilesParameterLength,
  GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH,
  GPU_COLUMN_QUANTILE_INTERPOLATION_CODES
} from './column-quantiles-parameters';
import {computeColumnQuantilesOnCPU} from './cpu-column-quantiles-operation';
import {getComparisonFloat32Key, getOrderedFloat32Key} from './column-classification-shared';
import type {ClassBreaksOperationProps} from './gpu-class-breaks-operation';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';

/** Named props alias for the public CPU operation class. */
export type CPUClassBreaksOperationProps = ClassBreaksOperationProps;

export type CPUClassBreaksOptions = Pick<
  ClassBreaksOperationProps,
  'maximumClassCount' | 'methods' | 'naturalBreaksMode'
>;
export type CPUClassBreaksResult = {
  /** Resolved requested GPU mode and the actual float64 CPU natural-breaks algorithm. */
  provenance?: ReturnType<typeof getClassBreaksProvenance>;
  breaks: Float32Array;
  classCount: Uint32Array;
  classCounts: Uint32Array;
};

function validateClassOptions(id: string, options: CPUClassBreaksOptions): void {
  // The edge array needs maximumClassCount + 1 uint32-addressable rows.
  validateCapacity(id, 'maximumClassCount', options.maximumClassCount, 1, 0xfffffffe);
  validateClassBreaksNaturalOptions(id, options);
  resolveClassBreaksMethods(id, options, false, false);
}

// Exhaustive layered Fisher-Jenks DP over distinct values with their original multiplicities.
// Running central moments avoid subtracting long prefix sums on narrow classes after outliers.
function computeNaturalEdges(sorted: number[], requestedClasses: number): number[] {
  const distinct: number[] = [];
  const frequencies: number[] = [];
  for (const value of sorted) {
    if (distinct.length && distinct[distinct.length - 1] === value)
      frequencies[frequencies.length - 1]++;
    else {
      distinct.push(value);
      frequencies.push(1);
    }
  }
  const count = distinct.length;
  const classes = Math.min(count, requestedClasses);
  let previous = new Float64Array(count + 1).fill(Infinity);
  previous[0] = 0;
  const partitions: Float64Array[] = [];
  for (let layer = 1; layer <= classes; layer++) {
    const current = new Float64Array(count + 1).fill(Infinity);
    const starts = new Float64Array(count + 1);
    for (let start = layer - 1; start < count; start++) {
      if (!Number.isFinite(previous[start])) continue;
      let weight = 0;
      let mean = 0;
      let squares = 0;
      for (let end = start; end < count; end++) {
        const value = distinct[end] - distinct[start];
        const difference = value - mean;
        weight += frequencies[end];
        mean += (difference * frequencies[end]) / weight;
        squares += frequencies[end] * difference * (value - mean);
        const candidate = previous[start] + Math.max(0, squares);
        // Ascending starts and a strict improvement preserve the lowest-predecessor tie rule.
        if (candidate < current[end + 1]) {
          current[end + 1] = candidate;
          starts[end + 1] = start;
        }
      }
    }
    partitions.push(starts);
    previous = current;
  }
  const edges: number[] = [];
  let end = count;
  for (let layer = classes; layer > 1; layer--) {
    const start = partitions[layer - 1][end];
    edges.push(distinct[start]);
    end = start;
  }
  return edges.reverse();
}

/**
 * Computes all class-break methods with float64 moments and exact Fisher-Jenks partitions.
 * Outputs retain the lower-edge classification rule, repeated edges, end-class infinities and
 * NaN tails. Maximum-gap distinctness and class assignment use numeric float32 comparison keys:
 * -0 equals +0 and both enter the upper class at a zero break. Raw key ordering preserves extrema bits.
 * Integer input columns undergo the contributor's initial float32 conversion.
 *
 * What differs from the GPU contributor:
 * - Arithmetic uses float64 with one float32 edge rounding. Quantile ranks and head thresholds
 *   can differ near a float32 boundary. Compare conditioned arithmetic edges within 3e-5 relative;
 *   counts can differ when a computed edge crosses a source value.
 * - Natural breaks always find the global float64 Fisher-Jenks optimum over distinct-value
 *   boundaries, using all source multiplicities, for approximate, exact and monotone GPU modes.
 *   There is no binning, quantization, local refinement or normalized float32 recurrence. Exact
 *   cost ties retain the smallest predecessor, as GPU exact mode does; near ties may differ.
 * - Maximum-break gaps and midpoints use float64; float32 gap overflow or rounded gap ties no
 *   longer affect their order. Equal exact gaps still favor the lower sorted position.
 *   Published edges are float32; a midpoint rounding to the lower observation uses the upper one.
 * - The GPU's 64-class, 2^24-row, bin-count and exact/monotone storage limits are dropped;
 *   uint32 counts and edge-array lengths remain. Natural-break execution/storage settings are
 *   ignored numerically by the CPU operation. Explicit approximate graph declarations still
 *   require maximumClassCount <= naturalBreaksBinCount; implicit CPU defaults retain the exact
 *   natural method without a GPU bin ceiling. Its DP costs O(k u^2) time and O(k u) memory for u distinct values.
 * - Subnormals and finite float64 intermediates beyond float32 range are retained. Non-custom
 *   computed inner edges are clamped to the finite data range before publishing, as on the GPU.
 *
 * @throws If mask length disagrees with values or parameters is shorter than its packed layout.
 */
export function computeClassBreaksOnCPU(
  values: Float32Array | Uint32Array | Int32Array,
  parameters: Float32Array,
  options: CPUClassBreaksOptions,
  mask?: Uint32Array
): CPUClassBreaksResult {
  validateClassOptions('computeClassBreaksOnCPU', options);
  validateCapacity('computeClassBreaksOnCPU', 'values.length', values.length, 0);
  const {maximumClassCount} = options;
  if (mask && mask.length !== values.length)
    throw new Error('mask length must equal values length');
  if (parameters.length < getGPUClassBreaksParameterLength(maximumClassCount))
    throw new Error('parameters must hold the header and maximumClassCount + 1 data values');
  const result: CPUClassBreaksResult = {
    provenance: getClassBreaksProvenance(options, 'cpu'),
    breaks: new Float32Array(maximumClassCount + 1).fill(NaN),
    classCount: Uint32Array.of(0),
    classCounts: new Uint32Array(maximumClassCount)
  };
  const [methodCode, classValue, standardDeviationInterval, rawRatio, hinge] = parameters;
  const method = resolveClassBreaksMethods('computeClassBreaksOnCPU', options, false, false).find(
    candidate => GPU_CLASS_BREAKS_METHOD_CODES[candidate] === methodCode
  );
  if (method === undefined) return result;
  const converted = values instanceof Float32Array ? values : Float32Array.from(values);
  const sorted: number[] = [];
  for (let row = 0; row < converted.length; row++)
    if (Number.isFinite(converted[row]) && (!mask || mask[row] !== 0)) sorted.push(converted[row]);
  sorted.sort((first, second) => getOrderedFloat32Key(first) - getOrderedFloat32Key(second));
  const minimum = sorted[0];
  const maximum = sorted[sorted.length - 1];
  let classes = classValue >= 1 ? Math.min(Math.floor(classValue), maximumClassCount) : 1;
  if (method === 'box-plot') classes = GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT;
  let inner: number[] = [];
  if (method === 'custom') {
    const edges = parameters.subarray(
      GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
      GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH + classes + 1
    );
    if (edges.some(Number.isNaN)) return result;
    result.breaks.set(edges);
  } else {
    if (sorted.length === 0) return result;
    if (minimum === maximum) classes = 1;
    else {
      switch (method) {
        case 'equal-interval':
          for (let edge = 1; edge < classes; edge++)
            inner.push(minimum + ((maximum - minimum) * edge) / classes);
          break;
        case 'quantile':
        case 'box-plot': {
          const quantileCount = method === 'quantile' ? classes - 1 : 3;
          const quantileParameters = new Float32Array(
            getGPUColumnQuantilesParameterLength(quantileCount)
          );
          quantileParameters[0] = GPU_COLUMN_QUANTILE_INTERPOLATION_CODES.linear;
          quantileParameters.set(
            parameters.subarray(
              GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
              GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH + quantileCount
            ),
            GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH
          );
          // One rounding here produces the public quantile output, but box whiskers must retain
          // float64 quartiles internally. Read their ranks directly below for that method.
          if (method === 'quantile' && quantileCount > 0)
            inner = Array.from(
              computeColumnQuantilesOnCPU(
                {values: converted, mask, parameters: quantileParameters},
                quantileCount
              ).quantiles
            );
          else if (method === 'box-plot') {
            const getQuantile = (probability: number) => {
              if (!(probability >= 0 && probability <= 1)) return NaN;
              const height = (sorted.length - 1) * probability;
              const lower = Math.floor(height);
              return (
                sorted[lower] +
                (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * (height - lower)
              );
            };
            const [lower, median, upper] = Array.from(
              quantileParameters.subarray(GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH),
              getQuantile
            );
            const spread = upper - lower;
            inner = [lower - hinge * spread, lower, median, upper, upper + hinge * spread];
          }
          break;
        }
        case 'standard-deviation': {
          let total = 0;
          for (const value of sorted) total += value - minimum;
          const offsetMean = total / sorted.length;
          let squares = 0;
          for (const value of sorted) squares += (value - minimum - offsetMean) ** 2;
          const deviation = Math.sqrt(squares / Math.max(1, sorted.length - 1));
          for (let edge = 1; edge < classes; edge++)
            inner.push(
              minimum +
                offsetMean +
                (edge - 1 - (classes - 2) / 2) * standardDeviationInterval * deviation
            );
          break;
        }
        case 'head-tail': {
          let head = sorted;
          let previousCount = 0;
          const ratio = rawRatio >= 0 ? rawRatio : 0.4;
          while (inner.length + 1 < classes && head.length > 0 && head[0] < head[head.length - 1]) {
            if (inner.length > 0 && (head.length > ratio * previousCount || head.length <= 1))
              break;
            let sum = 0;
            for (const value of head) sum += value - head[0];
            const mean = head[0] + sum / head.length;
            inner.push(mean);
            previousCount = head.length;
            const thresholdKey = getOrderedFloat32Key(mean);
            head = head.filter(value => getOrderedFloat32Key(value) > thresholdKey);
          }
          classes = inner.length + 1;
          break;
        }
        case 'maximum-breaks': {
          const gaps: {gap: number; position: number}[] = [];
          for (let position = 1; position < sorted.length; position++)
            if (
              getComparisonFloat32Key(sorted[position]) !==
              getComparisonFloat32Key(sorted[position - 1])
            )
              gaps.push({gap: sorted[position] - sorted[position - 1], position});
          gaps.sort((first, second) => second.gap - first.gap || first.position - second.position);
          inner = gaps
            .slice(0, classes - 1)
            .map(({position}) => {
              const lower = sorted[position - 1];
              const upper = sorted[position];
              const midpoint = Math.fround((lower + upper) / 2);
              // Lower edges include equality in the upper class; a selected gap must split rows.
              return getComparisonFloat32Key(midpoint) <= getComparisonFloat32Key(lower)
                ? upper
                : midpoint;
            })
            .sort((first, second) => first - second);
          classes = inner.length + 1;
          break;
        }
        case 'natural-breaks':
          inner = computeNaturalEdges(sorted, classes);
          classes = inner.length + 1;
          break;
      }
    }
    if (inner.some(Number.isNaN)) return result;
    result.breaks[0] = minimum;
    result.breaks[classes] = maximum;
    for (let edge = 1; edge < classes; edge++)
      result.breaks[edge] = Math.min(maximum, Math.max(minimum, inner[edge - 1]));
  }
  result.classCount[0] = classes;
  for (let row = 0; row < converted.length; row++) {
    const value = converted[row];
    if (Number.isNaN(value) || (mask && mask[row] === 0)) continue;
    const key = getComparisonFloat32Key(value);
    // Same lower-edge binary search, including raw custom edges used as given.
    let lower = 1;
    let upper = classes;
    while (lower < upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (getComparisonFloat32Key(result.breaks[middle]) <= key) lower = middle + 1;
      else upper = middle;
    }
    result.classCounts[lower - 1]++;
  }
  return result;
}

/**
 * CPU class-break twin: all natural-break modes use the full float64 global optimum.
 * Empty non-custom input is supported: NaN edges, classCount [0], zero classCounts.
 * GPU lowering requires at least one row; guard empty inputs before adding the GPU operation.
 */
export class CPUClassBreaksOperation extends GPUCPUOperation {
  /** Compiled methods for the packed CPU computation, including packed source chunks. */
  readonly methods: readonly GPUClassBreaksMethod[];
  /** Natural-breaks metadata; CPU always uses the exact float64 reference. */
  readonly provenance: ReturnType<typeof getClassBreaksProvenance>;
  constructor(inputProps: ClassBreaksOperationProps) {
    const props = snapshotClassBreaksOperationProps(inputProps);
    const id = props.id ?? 'class-breaks';
    validateClassOptions(id, props);
    if (props.naturalBreaksBinCount !== undefined)
      validateCapacity(id, 'naturalBreaksBinCount', props.naturalBreaksBinCount, 2);
    validateClassBreaksInputAssertions(id, props);
    const methods = validateClassBreaksGraphOptions(id, props, 'cpu');
    const operation = getClassBreaksOperationPorts(props);
    validateOperationGraphNames(
      id,
      operation.inputs,
      Object.fromEntries(
        Object.entries(operation.outputs).map(([port, output]) => [port, output.value])
      )
    );
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const input = inputs['values'];
        if (input === undefined || typeof input === 'number')
          throw new Error(`${id} values must be a vector`);
        validateClassBreaksInput(id, props, input, true);
        const values =
          input.format === 'uint32'
            ? getCPUOperationUint32Rows(id, inputs, 'values')
            : input.format === 'sint32'
              ? getCPUOperationInt32Rows(id, inputs, 'values')
              : getCPUOperationFloat32Rows(id, inputs, 'values');
        const result = computeClassBreaksOnCPU(
          values,
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          {...props, methods},
          props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
        );
        const outputs: Record<string, Float32Array | Uint32Array> = {
          [props.output.breaks]: result.breaks,
          [props.output.classCount]: result.classCount
        };
        if (props.output.classCounts !== undefined)
          outputs[props.output.classCounts] = result.classCounts;
        return outputs;
      }
    });
    this.methods = methods;
    this.provenance = getClassBreaksProvenance({...props, methods: this.methods}, 'cpu');
  }
}
