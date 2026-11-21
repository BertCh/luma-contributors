// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_SOLAR_POSITION_PARAMETER_LENGTH} from './gpu-solar-position';
import {
  GPUSolarPositionOperation,
  snapshotSolarPositionOperationProps,
  type SolarPositionOperationProps
} from './gpu-solar-position-operation';
import {getSolarPosition} from './solar-position';

const MILLISECONDS_PER_DAY = 86400000;
/** J2000.0, the epoch of getSolarTimeParameter and the contributor's packed days. */
const J2000_EPOCH_MILLISECONDS = Date.UTC(2000, 0, 1, 12);

/** Dense solar output rows. */
export type CPUSolarPositionResult = {
  azimuth: Float32Array;
  altitude: Float32Array;
  daylight: Uint32Array;
};

/**
 * Evaluates getSolarPosition for packed longitude/latitude rows using the packed time and
 * daylight threshold from getGPUSolarPositionParameterValues.
 *
 * What differs from the GPU contributor:
 * - Calls the public float64 NOAA helper and rounds outputs once. The GPU's split float32
 *   orbital-angle evaluation differs increasingly far from J2000 and near zenith/nadir.
 * - Daylight can differ for altitudes within the GPU angular error of the strict threshold.
 * - A final float32 azimuth of 360 is canonicalized to 0. The GPU's angular error can place its
 *   answer on either side of this seam, so compare angles circularly and check the range separately.
 * - Longitude arithmetic remains float64, including finite inputs for which the GPU's
 *   multiplication by 4 overflows float32; actual GPU behavior there is unverified.
 * - Nothing else: invalid coordinates yield NaN angles and daylight 0; refraction affects only altitude.
 */
export function computeSolarPositionOnCPU(
  positions: Float32Array,
  parameters: Float32Array,
  options: {refraction?: boolean} = {}
): CPUSolarPositionResult {
  if (positions.length % 2 !== 0)
    throw new Error('positions must contain longitude/latitude pairs');
  if (parameters.length < GPU_SOLAR_POSITION_PARAMETER_LENGTH)
    throw new Error('parameters is too short');
  // The packer documents this tuple; the contributor exports a length, but no offset constants.
  const [dayNumber, dayFraction, daylightAltitudeDegrees] = parameters;
  const timestamp = J2000_EPOCH_MILLISECONDS + (dayNumber + dayFraction) * MILLISECONDS_PER_DAY;
  const length = positions.length / 2;
  const result = {
    azimuth: new Float32Array(length).fill(Number.NaN),
    altitude: new Float32Array(length).fill(Number.NaN),
    daylight: new Uint32Array(length)
  };
  for (let row = 0; row < length; row++) {
    const longitude = positions[row * 2];
    const latitude = positions[row * 2 + 1];
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      Math.abs(latitude) > 90 ||
      !Number.isFinite(timestamp)
    )
      continue;
    const position = getSolarPosition(timestamp, longitude, latitude, options);
    result.azimuth[row] = position.azimuthDegrees;
    if (result.azimuth[row] === 360) result.azimuth[row] = 0;
    result.altitude[row] = position.altitudeDegrees;
    result.daylight[row] = Number(position.geometricAltitudeDegrees > daylightAltitudeDegrees);
  }
  return result;
}

/** CPU solar twin; uses the same row count and optional output ports as the GPU operation. */
export class CPUSolarPositionOperation extends GPUCPUOperation {
  constructor(props: SolarPositionOperationProps) {
    props = snapshotSolarPositionOperationProps(props);
    const operation = new GPUSolarPositionOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount, 1);
    validateRequirement(
      id,
      Object.keys(operation.outputs).length > 0,
      'output must contain at least one port'
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
        const result = computeSolarPositionOnCPU(
          positions,
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          {refraction: props.refraction}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSolarPositionResult]
          ])
        );
      }
    });
  }
}
