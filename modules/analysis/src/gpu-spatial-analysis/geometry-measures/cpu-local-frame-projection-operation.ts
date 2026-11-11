// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {GPULocalFrame} from '../../analysis-core/contracts/local-frame';
import type {ProjectionBounds} from '@luma.gl/graph/gpu-project';
import {
  GPULocalFrameProjectionOperation,
  type LocalFrameProjectionOperationProps
} from './gpu-local-frame-projection-operation';

/** Absolute float32 frame meters and per-row projection validity. */
export type CPULocalFrameProjectionResult = {output: Float32Array; validity: Uint32Array};

function validateProjectionBounds(id: string, bounds: ProjectionBounds): void {
  validateRequirement(
    id,
    bounds.length === 4 &&
      bounds.every(Number.isFinite) &&
      bounds[0] < bounds[2] &&
      bounds[1] < bounds[3],
    'bounds must contain four finite, increasing longitude/latitude limits'
  );
}

/**
 * Calls the frame's public binary64 projectArray and writes the same absolute frame rows as
 * GPULocalFrameProjection. Uint32 inputs encode little-endian Float64Array word pairs.
 *
 * What differs from the GPU contributor:
 * - Calls projectArray directly, avoiding compiled polynomial patches and their approximation
 *   error. tolerance only controls the GPU patch compiler and is unused by the CPU calculation.
 * - Bounds rejection remains inclusive. Non-finite source or projected rows are rejected with
 *   zero output and validity = 0, including singularities. The CPU accepts domains the GPU
 *   patch compiler may refuse before execution.
 * - Supports only one packed graph chunk, in float32x2 or raw-binary64 uint32x4 format.
 * - Rounds absolute frame meters once at the output, rather than rounding polynomial terms.
 *
 * @throws If positions contains an incomplete row or bounds is inconsistent.
 */
export function computeLocalFrameProjectionOnCPU(
  positions: Float32Array | Uint32Array,
  frame: GPULocalFrame,
  bounds: ProjectionBounds
): CPULocalFrameProjectionResult {
  validateProjectionBounds('computeLocalFrameProjectionOnCPU', bounds);
  const components = positions instanceof Uint32Array ? 4 : 2;
  if (positions.length % components !== 0) {
    throw new Error(`positions must contain complete ${components}-component rows`);
  }
  const rowCount = positions.length / components;
  let coordinates: Float32Array | Float64Array = positions as Float32Array;
  if (positions instanceof Uint32Array) {
    coordinates = new Float64Array(rowCount * 2);
    const words = new DataView(positions.buffer, positions.byteOffset, positions.byteLength);
    for (let coordinate = 0; coordinate < coordinates.length; coordinate++) {
      coordinates[coordinate] = words.getFloat64(coordinate * Float64Array.BYTES_PER_ELEMENT, true);
    }
  }
  const output = frame.projectArray(coordinates);
  const validity = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const longitude = coordinates[row * 2];
    const latitude = coordinates[row * 2 + 1];
    const valid =
      Number.isFinite(longitude) &&
      Number.isFinite(latitude) &&
      longitude >= bounds[0] &&
      longitude <= bounds[2] &&
      latitude >= bounds[1] &&
      latitude <= bounds[3] &&
      Number.isFinite(output[row * 2]) &&
      Number.isFinite(output[row * 2 + 1]);
    validity[row] = valid ? 1 : 0;
    if (!valid) {
      output[row * 2] = 0;
      output[row * 2 + 1] = 0;
    }
  }
  return {output, validity};
}

/** CPU twin; no polynomial compilation or patch-count limit. */
export class CPULocalFrameProjectionOperation extends GPUCPUOperation {
  constructor(props: LocalFrameProjectionOperationProps) {
    const operation = new GPULocalFrameProjectionOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount, 1);
    validateProjectionBounds(id, props.bounds);
    validateFiniteNumber(id, 'tolerance', props.tolerance ?? 0.01);
    validateRequirement(id, (props.tolerance ?? 0.01) > 0, 'tolerance must be greater than 0');
    validateRequirement(
      id,
      props.positionsFormat === undefined ||
        props.positionsFormat === 'float32x2' ||
        props.positionsFormat === 'uint32x4',
      'positionsFormat must be float32x2 or uint32x4'
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
        const positions =
          props.positionsFormat === 'uint32x4'
            ? getCPUOperationUint32Rows(id, inputs, 'positions', 'uint32x4')
            : getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        validateRequirement(
          id,
          positions.length === props.rowCount * (props.positionsFormat === 'uint32x4' ? 4 : 2),
          'positions length must match rowCount'
        );
        const result = computeLocalFrameProjectionOnCPU(positions, props.frame, props.bounds);
        return {
          [props.output]: result.output,
          ...(props.validity === undefined ? {} : {[props.validity]: result.validity})
        };
      }
    });
  }
}
