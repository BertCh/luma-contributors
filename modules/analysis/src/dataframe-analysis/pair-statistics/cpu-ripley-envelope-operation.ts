// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getPhilox4x32,
  getPhiloxUnitFloat,
  PHILOX_PURPOSES
} from '../../analysis-core/kernels/philox-random';
import {
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_RIPLEY_SEED_SLOT,
  GPU_RIPLEY_EDGE_CORRECTION
} from './ripley-parameters';
import {GPU_RIPLEY_WEIGHT_CAP} from './gpu-ripley';
import {
  GPURipleyEnvelopeOperation,
  type RipleyEnvelopeOperationProps
} from './gpu-ripley-envelope-operation';
import {
  getRectangleIsotropicWeight,
  getRectangleSides,
  isRectangleLatticeValid
} from './cpu-inference-rectangle';

/**
 * Input parameters retain the getGPURipleyParameterValues layout, including uint32 seed bits.
 * Only the first radiusCount observedK ordinates are read; trailing capacity is ignored.
 */
export type CPURipleyEnvelopeInput = {
  positions: Float32Array;
  parameters: Float32Array;
  mask?: Uint32Array;
  observedK?: Float32Array;
};
/** GPU gridSize and maximumAttempts affect no rectangle CPU computation. */
export type CPURipleyEnvelopeOptions = Pick<
  RipleyEnvelopeOperationProps,
  'radiusCount' | 'simulationCount' | 'rank' | 'statistic'
>;
/** All rectangle output ports. */
export type CPURipleyEnvelopeResult = {
  lower: Float32Array;
  upper: Float32Array;
  globalLower: Float32Array;
  globalUpper: Float32Array;
  theoretical: Float32Array;
  statistics: Float32Array;
  unplacedCount: Uint32Array;
};

function validateOptions(id: string, options: CPURipleyEnvelopeOptions): void {
  validateCapacity(id, 'radiusCount', options.radiusCount);
  validateCapacity(id, 'rank', options.rank ?? 1);
  validateCapacity(id, 'simulationCount', options.simulationCount, 2 * (options.rank ?? 1));
  validateRequirement(
    id,
    options.statistic === undefined || ['k', 'l', 'l-minus-r'].includes(options.statistic),
    'statistic must be k, l or l-minus-r'
  );
}

/**
 * Binomial CSR rectangle simulations, Ripley K estimation and pointwise/global MAD envelopes.
 * Draws use the contributor's registered Philox purpose and counter order.
 *
 * What differs from the GPU contributor:
 * - This operation form omits the optional prepared polygon window; only rectangles are supported.
 * - Coordinates, distances, K estimates, deviations and inference use float64, rounded only at
 *   final outputs. Near distance thresholds or deviation ties can change counts and p-values.
 * - Isotropic weights retain the documented cap of 100 but omit GPU fixed-point quantization.
 *   Ordinary curves allow 2e-4 relative error, with 2e-5 absolute for L-minus-r cancellation.
 * - GPU-only row, radius, simulation and command-node caps are dropped. Pair searches are exact
 *   all-pairs, and gridSize/maximumAttempts are unused for rectangles.
 * - Invalid lattice bounds/nonpositive maximumDistance exclude all observations. Finite values
 *   whose GPU intermediate width, squared radius or area overflows remain usable in float64.
 * - Nonfinite coordinates are excluded, as on the GPU; NaN observed K ordinates are skipped.
 *
 * @throws If positions is not float32x2 rows, mask length differs, observedK has fewer than
 * radiusCount ordinates, or parameters does not contain the packer's full layout.
 */
export function computeRipleyEnvelopeOnCPU(
  input: CPURipleyEnvelopeInput,
  options: CPURipleyEnvelopeOptions
): CPURipleyEnvelopeResult {
  validateOptions('ripley-envelope', options);
  const {positions, parameters, mask, observedK} = input;
  if (positions.length === 0 || positions.length % 2 !== 0)
    throw new Error('positions must hold nonempty float32x2 rows');
  const rows = positions.length / 2;
  if (mask && mask.length !== rows) throw new Error('mask length must equal positions row count');
  if (observedK && observedK.length < options.radiusCount)
    throw new Error('observedK length must be at least radiusCount');
  if (parameters.length < GPU_RIPLEY_PARAMETER_LENGTH)
    throw new Error('parameters must hold getGPURipleyParameterValues layout');
  const [minimumX, minimumY, maximumX, maximumY, maximumDistance, edgeCorrection] = parameters;
  const bounds = [minimumX, minimumY, maximumX, maximumY];
  const valid = isRectangleLatticeValid(bounds, maximumDistance);
  let included = 0;
  for (let row = 0; row < rows; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    if (
      valid &&
      (mask === undefined || mask[row] !== 0) &&
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= minimumX &&
      x <= maximumX &&
      y >= minimumY &&
      y <= maximumY
    )
      included++;
  }
  const seed = new Uint32Array(parameters.buffer, parameters.byteOffset, parameters.length)[
    GPU_RIPLEY_SEED_SLOT
  ];
  const area = (maximumX - minimumX) * (maximumY - minimumY);
  const radii = Array.from(
    {length: options.radiusCount},
    (_, index) => (maximumDistance * (index + 1)) / options.radiusCount
  );
  const simulatedK: Float64Array[] = [];
  const mode = Math.max(0, Math.min(0xffffffff, Math.trunc(edgeCorrection + 0.5))) || 0;
  for (let simulation = 0; simulation < options.simulationCount; simulation++) {
    const curve = new Float64Array(options.radiusCount).fill(NaN);
    if (included >= 2 && Number.isFinite(area) && area > 0) {
      const points = Array.from({length: included}, (_, row) => {
        const words = getPhilox4x32(
          [row, simulation, 0, 0],
          [seed, PHILOX_PURPOSES.ripleyEnvelopeSimulation.id]
        );
        return [
          minimumX + getPhiloxUnitFloat(words[0]) * (maximumX - minimumX),
          minimumY + getPhiloxUnitFloat(words[1]) * (maximumY - minimumY)
        ];
      });
      const pairSums = new Float64Array(options.radiusCount);
      const borderTotals = new Float64Array(options.radiusCount);
      for (let focus = 0; focus < included; focus++) {
        const sides = getRectangleSides(points[focus][0], points[focus][1], bounds);
        const borderDistance = Math.min(...sides);
        for (let radius = 0; radius < radii.length; radius++)
          if (borderDistance >= radii[radius]) borderTotals[radius]++;
        for (let other = 0; other < included; other++)
          if (other !== focus) {
            const distance = Math.hypot(
              points[focus][0] - points[other][0],
              points[focus][1] - points[other][1]
            );
            if (distance > maximumDistance) continue;
            const weight =
              mode === GPU_RIPLEY_EDGE_CORRECTION.isotropic
                ? getRectangleIsotropicWeight(sides, distance, GPU_RIPLEY_WEIGHT_CAP)
                : 1;
            for (let radius = 0; radius < radii.length; radius++)
              if (
                distance <= radii[radius] &&
                (mode !== GPU_RIPLEY_EDGE_CORRECTION.border || borderDistance >= radii[radius])
              )
                pairSums[radius] += weight;
          }
      }
      for (let radius = 0; radius < radii.length; radius++) {
        if (mode === GPU_RIPLEY_EDGE_CORRECTION.border)
          curve[radius] =
            borderTotals[radius] > 0
              ? (area * pairSums[radius]) / (included * borderTotals[radius])
              : NaN;
        else
          curve[radius] =
            mode === GPU_RIPLEY_EDGE_CORRECTION.none ||
            mode === GPU_RIPLEY_EDGE_CORRECTION.isotropic
              ? (area * pairSums[radius]) / (included * (included - 1))
              : 0;
      }
    }
    simulatedK.push(curve);
  }
  const rank = options.rank ?? 1;
  const statistic = options.statistic ?? 'l-minus-r';
  const getL = (value: number) => Math.sqrt(Math.max(value, 0) / Math.PI);
  const fromK = (value: number, radius: number) =>
    statistic === 'k' ? value : getL(value) - (statistic === 'l-minus-r' ? radius : 0);
  const fromL = (value: number, radius: number) =>
    statistic === 'k'
      ? Math.PI * Math.max(value, 0) ** 2
      : value - (statistic === 'l-minus-r' ? radius : 0);
  const getDeviation = (curve: ArrayLike<number>) => {
    let deviation = NaN;
    for (let radius = 0; radius < radii.length; radius++)
      if (Number.isFinite(curve[radius]))
        deviation = Math.max(
          Number.isNaN(deviation) ? 0 : deviation,
          Math.abs(getL(curve[radius]) - radii[radius])
        );
    return deviation;
  };
  const deviations = simulatedK
    .map(getDeviation)
    .filter(Number.isFinite)
    .sort((first, second) => second - first);
  const critical = deviations.length >= rank ? deviations[rank - 1] : NaN;
  const observed = observedK ? getDeviation(observedK) : NaN;
  const probability = Number.isFinite(observed)
    ? (1 + deviations.filter(value => value >= observed).length) / (deviations.length + 1)
    : NaN;
  const lower = new Float32Array(options.radiusCount).fill(NaN);
  const upper = new Float32Array(options.radiusCount).fill(NaN);
  for (let radius = 0; radius < radii.length; radius++) {
    const column = simulatedK
      .map(curve => curve[radius])
      .filter(Number.isFinite)
      .sort((first, second) => first - second);
    if (column.length >= 2 * rank) {
      lower[radius] = fromK(column[rank - 1], radii[radius]);
      upper[radius] = fromK(column[column.length - rank], radii[radius]);
    }
  }
  return {
    lower,
    upper,
    globalLower: Float32Array.from(radii, radius => fromL(radius - critical, radius)),
    globalUpper: Float32Array.from(radii, radius => fromL(radius + critical, radius)),
    theoretical: Float32Array.from(radii, radius => fromL(radius, radius)),
    statistics: Float32Array.of(included, deviations.length, observed, critical, probability, rank),
    unplacedCount: Uint32Array.of(0)
  };
}

/** Rectangle CPU twin; GPU-only limits and grid search are omitted. */
export class CPURipleyEnvelopeOperation extends GPUCPUOperation {
  constructor(props: RipleyEnvelopeOperationProps) {
    const operation = new GPURipleyEnvelopeOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount);
    validateOptions(id, props);
    resolveGPUSpatialContext(id, props.spatialContext);
    validateRequirement(
      id,
      Object.values(props.output).some(value => value !== undefined),
      'output needs at least one view'
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions must match rowCount`);
        const result = computeRipleyEnvelopeOnCPU(
          {
            positions,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            mask:
              props.mask === undefined ? undefined : getCPUOperationUint32Rows(id, inputs, 'mask'),
            observedK:
              props.observedK === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'observedK')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPURipleyEnvelopeResult]
          ])
        );
      }
    });
  }
}
