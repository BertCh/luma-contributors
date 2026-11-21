// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {SpatialLagImpactsOperationProps} from './gpu-spatial-lag-impacts-operation';
import {getSpatialLagImpactsFromTraces} from './spatial-weights-traces-parameters';
import {snapshotSpatialLagImpactsProps} from './spatial-lag-impacts-snapshot';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';

/** Packed input rows, in the contributor's scalar float32 layout. */
export type CPUSpatialLagImpactsInput = {
  coefficients: Float32Array;
  powerTraces: Float32Array;
  powerSums: Float32Array;
  infinityNorm: Float32Array;
  status?: Uint32Array;
};

/** Compile-time settings; the CPU has no 256-power shader limit. */
export type CPUSpatialLagImpactsOptions = Pick<
  SpatialLagImpactsOperationProps,
  | 'coefficientStride'
  | 'rhoIndex'
  | 'firstPredictorIndex'
  | 'predictorCount'
  | 'rowCount'
  | 'maximumPower'
  | 'rhoBound'
>;

/** Three float32 values [direct, indirect, total] per predictor row. */
export type CPUSpatialLagImpactsResult = {impacts: Float32Array};

function validateOptions(id: string, options: CPUSpatialLagImpactsOptions): void {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'rowCount', options.rowCount);
  validateCapacity(id, 'maximumPower', options.maximumPower);
  validateCapacity(id, 'coefficientStride', options.coefficientStride);
  validateCapacity(id, 'rhoIndex', options.rhoIndex, 0);
  const firstPredictorIndex = options.firstPredictorIndex ?? 1;
  validateCapacity(id, 'firstPredictorIndex', firstPredictorIndex, 0);
  validateRequirement(
    id,
    options.rhoIndex < firstPredictorIndex ||
      options.rhoIndex >= firstPredictorIndex + options.predictorCount,
    'rhoIndex must not be a predictor row'
  );
  validateFiniteNumber(id, 'rhoBound', options.rhoBound ?? 1, 0);
  validateRequirement(id, (options.rhoBound ?? 1) > 0, 'rhoBound must be positive');
}

/**
 * Evaluates the documented truncated impacts with the public getSpatialLagImpactsFromTraces
 * helper, in float64, and writes [direct, indirect, total] per predictor in float32.
 *
 * What differs from the GPU contributor:
 * - Float64 intermediate arithmetic, rounded once at output; compare with relative tolerance 3e-6.
 * - The shader-only maximumPower <= 256 limit is dropped; the requested truncation is retained.
 * - rhoBound must be finite. Its float32 representation is used for the same strict bound as
 *   the shader constant, but rho * infinityNorm is evaluated in float64 at that bound.
 * - Non-finite coefficients and traces propagate through the public helper; subnormals are kept.
 *
 * @throws If an input array cannot hold the rows the GPU reads, naming that argument.
 */
export function computeSpatialLagImpactsOnCPU(
  input: CPUSpatialLagImpactsInput,
  options: CPUSpatialLagImpactsOptions
): CPUSpatialLagImpactsResult {
  const id = 'computeSpatialLagImpactsOnCPU';
  validateOptions(id, options);
  const firstPredictorIndex = options.firstPredictorIndex ?? 1;
  const lastRow = Math.max(options.rhoIndex, firstPredictorIndex + options.predictorCount - 1);
  if (input.coefficients.length < lastRow * options.coefficientStride + 1) {
    throw new Error(`${id} coefficients must hold every predictor row and the rho row`);
  }
  for (const name of ['powerTraces', 'powerSums'] as const) {
    if (input[name].length < options.maximumPower) {
      throw new Error(`${id} ${name} must hold maximumPower values`);
    }
  }
  if (input.infinityNorm.length < 1) throw new Error(`${id} infinityNorm must hold one value`);
  if (input.status && input.status.length < 1) throw new Error(`${id} status must hold one value`);
  const impacts = new Float32Array(options.predictorCount * 3).fill(NaN);
  const rho = input.coefficients[options.rhoIndex * options.coefficientStride];
  if (
    !Number.isFinite(rho) ||
    !(Math.abs(rho) * input.infinityNorm[0] < Math.fround(options.rhoBound ?? 1)) ||
    (input.status && input.status[0] !== 0)
  ) {
    return {impacts};
  }
  const powerTraces = input.powerTraces.subarray(0, options.maximumPower);
  const powerSums = input.powerSums.subarray(0, options.maximumPower);
  for (let predictor = 0; predictor < options.predictorCount; predictor++) {
    const beta = input.coefficients[(firstPredictorIndex + predictor) * options.coefficientStride];
    const result = getSpatialLagImpactsFromTraces(
      rho,
      beta,
      powerTraces,
      powerSums,
      options.rowCount,
      input.infinityNorm[0]
    );
    impacts.set([result.direct, result.indirect, result.total], predictor * 3);
  }
  return {impacts};
}

/** CPU twin; the shader power cap is dropped, with the same graph layouts and truncation. */
export class CPUSpatialLagImpactsOperation extends GPUCPUOperation {
  constructor(props: SpatialLagImpactsOperationProps) {
    props = snapshotSpatialLagImpactsProps(props);
    const id = props.id ?? 'spatial-lag-impacts';
    validateOptions(id, props);
    validateOperationGraphNames(
      id,
      {
        coefficients: props.coefficients,
        ...props.traces,
        ...(props.status !== undefined ? {status: props.status} : {})
      },
      props.output
    );
    super({
      id,
      inputs: {
        coefficients: props.coefficients,
        ...props.traces,
        ...(props.status ? {status: props.status} : {})
      },
      outputs: {
        [props.output.impacts]: gpuGraph.vector('float32', {length: props.predictorCount * 3})
      },
      execute({inputs}) {
        const result = computeSpatialLagImpactsOnCPU(
          {
            coefficients: getCPUOperationFloat32Rows(id, inputs, 'coefficients'),
            powerTraces: getCPUOperationFloat32Rows(id, inputs, 'powerTraces'),
            powerSums: getCPUOperationFloat32Rows(id, inputs, 'powerSums'),
            infinityNorm: getCPUOperationFloat32Rows(id, inputs, 'infinityNorm'),
            status: props.status ? getCPUOperationUint32Rows(id, inputs, 'status') : undefined
          },
          props
        );
        return {[props.output.impacts]: result.impacts};
      }
    });
  }
}
