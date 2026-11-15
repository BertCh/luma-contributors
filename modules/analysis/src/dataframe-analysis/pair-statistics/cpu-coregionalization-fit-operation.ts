// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSymmetricEigenOnCPU} from '../../analysis-core/kernels/symmetric-eigen';
import {
  getGPUCoregionalizationFitParameterValues,
  COREGIONALIZATION_MODEL_FAMILIES,
  type CoregionalizationStructure,
  validateCoregionalizationFitOptions,
  type CoregionalizationFitOptions
} from './coregionalization-parameters';
import {getVariogramModelShape} from './variogram-model';
import type {CoregionalizationFitOperationProps} from './gpu-coregionalization-fit-operation';

/** Packed input layout shared with GPUCoregionalizationFit. */
export type CPUCoregionalizationFitInput = {
  distances: Float32Array;
  semivariances: Float32Array;
  pairCounts: Uint32Array;
  parameters: Float32Array;
};
/** Identical packed outputs, rounded once after a float64 fit. */
export type CPUCoregionalizationFitResult = {
  matrices: Float32Array;
  loadings: Float32Array;
  diagnostics: Float32Array;
  validity: Uint32Array;
  iterationCount: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  invalidCount: Uint32Array;
};

/**
 * Goulard–Voltz fitting from zero: sequential conditional least squares and PSD projection.
 *
 * What differs from the GPU contributor:
 * - Float64 sufficient statistics, eigendecompositions and coefficients; one rounding at output.
 * - GPU-only bin and sweep caps are dropped; variable/structure bounds keep the kriging layout.
 * - Convergence iteration counts and eigenvector signs/bases can differ at float32 rounding
 *   thresholds or repeated eigenvalues. Compare B and L L^T, not individual factor entries.
 * - Finite float64 intermediate arithmetic can fit bins the GPU rejects on float32 overflow.
 *
 * Empty/invalid bins skip the complete lag matrix. Malformed array lengths throw. Invalid model
 * parameters or a zero-energy structure return validity 0 and NaN coefficients. Budget exits
 * return the current PSD matrices, including the zero model for a zero budget.
 */
export function computeCoregionalizationFitOnCPU(
  input: CPUCoregionalizationFitInput,
  options: CoregionalizationFitOptions
): CPUCoregionalizationFitResult {
  const id = 'coregionalization-fit';
  validateCoregionalizationFitOptions(id, options, false);
  const {variableCount, structureCount} = options;
  const maximumIterations = options.maximumIterations ?? 128;
  const {distances, semivariances, pairCounts, parameters} = input;
  const triangleCount = (variableCount * (variableCount + 1)) / 2;
  const squareCount = variableCount ** 2;
  const coefficientCount = structureCount * squareCount;
  validateRequirement(
    id,
    pairCounts.length === distances.length,
    'pairCounts.length must equal distances.length'
  );
  validateRequirement(
    id,
    semivariances.length === distances.length * triangleCount,
    'semivariances.length must equal distances.length * variableCount * (variableCount + 1) / 2'
  );
  validateRequirement(
    id,
    parameters.length >= 1 + 4 * structureCount,
    'parameters.length must be at least 1 + 4 * structureCount'
  );
  const result: CPUCoregionalizationFitResult = {
    matrices: new Float32Array(coefficientCount).fill(Number.NaN),
    loadings: new Float32Array(coefficientCount).fill(Number.NaN),
    diagnostics: new Float32Array([Number.NaN, Number.NaN, 0, 0]),
    validity: new Uint32Array(1),
    iterationCount: new Uint32Array(1),
    converged: new Uint32Array(1),
    iterationLimitReached: new Uint32Array(1),
    invalidCount: new Uint32Array(1)
  };
  const bins: number[] = [];
  let totalWeight = 0;
  for (let bin = 0; bin < distances.length; bin++) {
    let usable = Number.isFinite(distances[bin]) && distances[bin] > 0 && pairCounts[bin] > 0;
    for (let entry = 0; entry < triangleCount; entry++)
      usable &&= Number.isFinite(semivariances[bin * triangleCount + entry]);
    if (usable) {
      bins.push(bin);
      totalWeight += pairCounts[bin];
    } else {
      result.invalidCount[0]++;
    }
  }
  result.diagnostics[2] = bins.length;
  result.diagnostics[3] = totalWeight;
  if (
    bins.length === 0 ||
    !Number.isFinite(parameters[0]) ||
    parameters[0] <= 0 ||
    parameters[0] > 1
  )
    return result;
  for (let structure = 0; structure < structureCount; structure++) {
    const start = 1 + 4 * structure;
    const [model, nugget, sill, range] = parameters.subarray(start, start + 4);
    if (
      !Number.isInteger(model) ||
      model < 0 ||
      model >= COREGIONALIZATION_MODEL_FAMILIES.length ||
      ![nugget, sill, range].every(Number.isFinite) ||
      nugget < 0 ||
      sill < 0 ||
      nugget + sill <= 0 ||
      range <= 0
    )
      return result;
  }
  const shapes = bins.map(bin =>
    Float64Array.from({length: structureCount}, (_, structure) => {
      const start = 1 + 4 * structure;
      return (
        parameters[start + 1] +
        parameters[start + 2] *
          getVariogramModelShape(
            COREGIONALIZATION_MODEL_FAMILIES[parameters[start]],
            distances[bin] / parameters[start + 3]
          )
      );
    })
  );
  const gram = new Float64Array(structureCount ** 2);
  const targets = new Float64Array(coefficientCount);
  bins.forEach((bin, binIndex) => {
    const weight = pairCounts[bin] / totalWeight;
    for (let structure = 0; structure < structureCount; structure++) {
      for (let other = 0; other < structureCount; other++)
        gram[structure * structureCount + other] +=
          weight * shapes[binIndex][structure] * shapes[binIndex][other];
      let triangle = 0;
      for (let row = 0; row < variableCount; row++) {
        for (let column = row; column < variableCount; column++) {
          const value =
            weight * shapes[binIndex][structure] * semivariances[bin * triangleCount + triangle++];
          targets[structure * squareCount + row * variableCount + column] += value;
          if (row !== column)
            targets[structure * squareCount + column * variableCount + row] += value;
        }
      }
    }
  });
  if (![...gram, ...targets].every(Number.isFinite)) return result;
  for (let structure = 0; structure < structureCount; structure++)
    if (!(gram[structure * structureCount + structure] > 0)) return result;
  const matrices = new Float64Array(coefficientCount);
  const loadings = new Float64Array(coefficientCount);
  let relativeChange = 0;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const previous = matrices.slice();
    for (let structure = 0; structure < structureCount; structure++) {
      const conditional = new Float64Array(squareCount);
      for (let entry = 0; entry < squareCount; entry++) {
        let numerator = targets[structure * squareCount + entry];
        for (let other = 0; other < structureCount; other++)
          if (other !== structure)
            numerator -=
              gram[structure * structureCount + other] * matrices[other * squareCount + entry];
        conditional[entry] = numerator / gram[structure * structureCount + structure];
      }
      if (!conditional.every(Number.isFinite)) return result;
      const eigen = getSymmetricEigenOnCPU(conditional, variableCount);
      for (let row = 0; row < variableCount; row++) {
        for (let column = 0; column < variableCount; column++)
          loadings[structure * squareCount + row * variableCount + column] =
            eigen.vectors[row * variableCount + column] *
            Math.sqrt(Math.max(eigen.values[column], 0));
      }
      for (let row = 0; row < variableCount; row++) {
        for (let column = 0; column < variableCount; column++) {
          let value = 0;
          for (let component = 0; component < variableCount; component++)
            value +=
              loadings[structure * squareCount + row * variableCount + component] *
              loadings[structure * squareCount + column * variableCount + component];
          matrices[structure * squareCount + row * variableCount + column] = value;
        }
      }
    }
    if (!matrices.every(Number.isFinite) || !loadings.every(Number.isFinite)) return result;
    let largest = 0;
    let change = 0;
    for (let entry = 0; entry < coefficientCount; entry++) {
      largest = Math.max(largest, Math.abs(matrices[entry]), Math.abs(previous[entry]));
      change = Math.max(change, Math.abs(matrices[entry] - previous[entry]));
    }
    relativeChange = largest > 0 ? change / largest : 0;
    result.iterationCount[0]++;
    if (change <= parameters[0] * largest) {
      result.converged[0] = 1;
      break;
    }
  }
  let residual = 0;
  bins.forEach((bin, binIndex) => {
    let triangle = 0;
    for (let row = 0; row < variableCount; row++) {
      for (let column = row; column < variableCount; column++) {
        let predicted = 0;
        for (let structure = 0; structure < structureCount; structure++)
          predicted +=
            shapes[binIndex][structure] *
            matrices[structure * squareCount + row * variableCount + column];
        residual +=
          (pairCounts[bin] / totalWeight) *
          (predicted - semivariances[bin * triangleCount + triangle++]) ** 2 *
          (row === column ? 1 : 2);
      }
    }
  });
  if (![...matrices, ...loadings, residual].every(value => Number.isFinite(Math.fround(value)))) {
    result.converged[0] = 0;
    return result;
  }
  result.matrices.set(matrices);
  result.loadings.set(loadings);
  result.diagnostics[0] = residual;
  result.diagnostics[1] = relativeChange;
  result.validity[0] = 1;
  result.iterationLimitReached[0] = 1 - result.converged[0];
  return result;
}

/**
 * Fit a supplied LMC for GPULocalKriging.coregionalizationModels. Read empirical bins back once;
 * inspect validity/convergence before using the model. Invalid fits return an empty model list.
 */
export function fitLinearModelOfCoregionalization(
  input: Omit<CPUCoregionalizationFitInput, 'parameters'>,
  options: {
    variableCount: number;
    structures: readonly CoregionalizationStructure[];
    maximumIterations?: number;
    tolerance?: number;
  }
): CPUCoregionalizationFitResult & {
  coregionalizationModels: {covariance: CoregionalizationStructure; loadings: number[]}[];
} {
  const result = computeCoregionalizationFitOnCPU(
    {
      ...input,
      parameters: getGPUCoregionalizationFitParameterValues(options.structures, options.tolerance)
    },
    {...options, structureCount: options.structures.length}
  );
  const squareCount = options.variableCount ** 2;
  return {
    ...result,
    coregionalizationModels: result.validity[0]
      ? options.structures.map((covariance, index) => ({
          covariance: {...covariance},
          loadings: Array.from(
            result.loadings.subarray(index * squareCount, (index + 1) * squareCount)
          )
        }))
      : []
  };
}

/**
 * Preferred float64 CPU twin for the bounded LMC fit. Same ports as the GPU operation.
 * GPU-only bin and iteration limits are dropped. Compile-time: shape/budget; per-frame: inputs.
 */
export class CPUCoregionalizationFitOperation extends GPUCPUOperation {
  constructor(props: CoregionalizationFitOperationProps) {
    const id = props.id ?? 'coregionalization-fit';
    validateCoregionalizationFitOptions(id, props, false);
    const length = props.structureCount * props.variableCount ** 2;
    const outputNames = {
      ...props.output.status,
      matrices: props.output.matrices,
      loadings: props.output.loadings,
      diagnostics: props.output.diagnostics
    };
    super({
      id,
      inputs: {
        distances: props.distances,
        semivariances: props.semivariances,
        pairCounts: props.pairCounts,
        parameters: props.parameters
      },
      outputs: Object.fromEntries(
        Object.entries(outputNames).map(([name, value]) => [
          value,
          gpuGraph.vector(
            name === 'matrices' || name === 'loadings'
              ? 'float32'
              : name === 'diagnostics'
                ? 'float32x4'
                : 'uint32',
            {length: name === 'matrices' || name === 'loadings' ? length : 1}
          )
        ])
      ),
      execute({inputs}) {
        const result = computeCoregionalizationFitOnCPU(
          {
            distances: getCPUOperationFloat32Rows(id, inputs, 'distances'),
            semivariances: getCPUOperationFloat32Rows(id, inputs, 'semivariances'),
            pairCounts: getCPUOperationUint32Rows(id, inputs, 'pairCounts'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(outputNames).map(([name, value]) => [
            value,
            result[name as keyof CPUCoregionalizationFitResult]
          ])
        );
      }
    });
  }
}
