// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {findCPUPolygonPole, readCPUPolygonRows} from '../label-point/cpu-label-point-operation';
import {
  getCPUPolygonRings,
  validateCPUPolygonRows,
  type CPUPolygonRows
} from '../shared/cpu-polygon-rows';
import {
  GPUMaximumInscribedCircleOperation,
  type MaximumInscribedCircleOperationProps
} from './gpu-maximum-inscribed-circle-operation';

/** Circle, witness and status rows, in GPUMaximumInscribedCircle's layout. */
export type CPUMaximumInscribedCircleResult = {
  circles: Float32Array;
  witnesses: Float32Array;
  status: Uint32Array;
};

/**
 * Best-first even-odd polygon circle search with dynamic live-cell storage in binary64.
 *
 * What differs from the GPU contributor:
 * - No cell-capacity or iteration budget; those props are ignored by the CPU operation.
 * - The remaining-error field bounds the completed binary64 search, stopping at tolerance or
 *   binary64 resolution. Status 1 is reserved for inability to subdivide at that resolution.
 * - Collapsed-axis polygons have an exact zero-radius result and zero remaining error.
 * - Float32 arithmetic overflow does not set status 3; only non-finite input does. Finite float32
 *   coordinates are evaluated in binary64 and outputs are rounded once.
 * - Live-slot winner ties, child order, last-vertex fallback and first closest-segment witness
 *   follow the GPU search. Different arithmetic can still resolve near ties differently.
 *
 * @throws If tolerance is not positive and finite, or input arrays are inconsistent.
 */
export function computeMaximumInscribedCircleOnCPU(
  input: CPUPolygonRows,
  tolerance: number
): CPUMaximumInscribedCircleResult {
  validateCPUPolygonRows(input);
  if (!(Number.isFinite(tolerance) && tolerance > 0)) {
    throw new Error('tolerance must be positive and finite');
  }
  const featureCount = input.featureOffsets.length - 1;
  const circles = new Float32Array(featureCount * 4);
  const witnesses = new Float32Array(featureCount * 2);
  const status = new Uint32Array(featureCount);
  for (let feature = 0; feature < featureCount; feature++) {
    const pole = findCPUPolygonPole(input.positions, getCPUPolygonRings(input, feature), tolerance);
    circles.set([pole.centerX, pole.centerY, pole.radius, pole.error], feature * 4);
    witnesses.set([pole.witnessX, pole.witnessY], feature * 2);
    status[feature] = pole.status;
  }
  return {circles, witnesses, status};
}

/** CPU twin; dynamic storage removes the GPU cellCapacity and maximumIterations limits. */
export class CPUMaximumInscribedCircleOperation extends GPUCPUOperation {
  constructor(props: MaximumInscribedCircleOperationProps) {
    const operation = new GPUMaximumInscribedCircleOperation(props);
    const {id} = operation;
    validateCapacity(id, 'featureCount', props.featureCount, 1);
    validateFiniteNumber(id, 'tolerance', props.tolerance);
    validateRequirement(id, props.tolerance > 0, 'tolerance must be greater than 0');
    resolveGPUSpatialContext(id, props.spatialContext);
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
        const input = readCPUPolygonRows(id, inputs);
        validateRequirement(
          id,
          input.featureOffsets.length === props.featureCount + 1,
          'featureOffsets length must equal featureCount + 1'
        );
        const result = computeMaximumInscribedCircleOnCPU(input, props.tolerance);
        return {
          [props.output.circles]: result.circles,
          [props.output.witnesses]: result.witnesses,
          [props.output.status]: result.status
        };
      }
    });
  }
}
