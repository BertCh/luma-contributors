// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUHealpixIndexScheme} from './gpu-healpix-point-to-cell';
import {HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';
import {
  GPUHealpixPointToCellOperation,
  type HealpixPointToCellOperationProps
} from './gpu-healpix-point-to-cell-operation';

/**
 * Projects packed longitude/latitude degrees to nested or UNIQ keys, little-endian low/high
 * words, with validity zero for masked or non-finite coordinates. Latitude clamps to [-90,90].
 *
 * What differs from the GPU contributor:
 * - Projection arithmetic uses float64. Keys near cell boundaries can differ from the GPU's
 *   float32 projection, especially at high orders; input float32 quantization remains.
 * - Longitude reduction uses remainder, avoiding cancellation for very large finite longitudes.
 * - The order limit 29 is retained for the HEALPix 64-bit key layout. Empty arrays are accepted
 *   by the pure function; the graph operation, like the GPU contributor, requires at least one row.
 *
 * @throws Error naming positions or mask when packed arrays have inconsistent lengths.
 */
export function computeHealpixPointToCellOnCPU(
  positions: Float32Array,
  options: {order: number; scheme?: GPUHealpixIndexScheme; mask?: Uint32Array}
): {cells: Uint32Array; validity: Uint32Array} {
  const rowCount = positions.length / 2;
  if (!Number.isInteger(rowCount)) throw new Error('positions must contain float32x2 rows');
  if (options.mask && options.mask.length !== rowCount)
    throw new Error('mask length must equal positions row count');
  validateCapacity('healpix-point-to-cell', 'order', options.order, 0, HEALPIX_MAXIMUM_ORDER);
  const sideLength = 2 ** options.order;
  const sideMask = sideLength - 1;
  const cells = new Uint32Array(rowCount * 2);
  const validity = new Uint32Array(rowCount);
  const spreadBits = (value: number): bigint => {
    let result = 0n;
    for (let bit = 0; bit < options.order; bit++) {
      if ((value >>> bit) & 1) result |= 1n << BigInt(2 * bit);
    }
    return result;
  };
  for (let row = 0; row < rowCount; row++) {
    const longitude = positions[row * 2];
    const latitude = positions[row * 2 + 1];
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude) || options.mask?.[row] === 0)
      continue;
    const clampedLatitude = Math.min(90, Math.max(-90, latitude));
    const quarterTurns = (((longitude % 360) + 360) % 360) / 90;
    let face: number;
    let column: number;
    let line: number;
    if (Math.abs(clampedLatitude) <= (Math.asin(2 / 3) * 180) / Math.PI) {
      const vertical = Math.sin((clampedLatitude * Math.PI) / 180);
      const diagonal = sideLength * (0.5 + quarterTurns);
      const displacement = sideLength * vertical * 0.75;
      const ascending = Math.floor(Math.max(diagonal - displacement, 0));
      const descending = Math.floor(Math.max(diagonal + displacement, 0));
      const ascendingFace = Math.floor(ascending / sideLength);
      const descendingFace = Math.floor(descending / sideLength);
      face =
        ascendingFace === descendingFace
          ? ascendingFace | 4
          : ascendingFace < descendingFace
            ? ascendingFace
            : descendingFace + 8;
      column = descending & sideMask;
      line = sideMask - (ascending & sideMask);
    } else {
      const quadrant = Math.min(Math.floor(quarterTurns), 3);
      const fraction = quarterTurns - quadrant;
      const halfColatitude = Math.sin(((90 - Math.abs(clampedLatitude)) * Math.PI) / 360);
      const polarLength = sideLength * Math.sqrt(6) * halfColatitude;
      const ascending = Math.min(Math.floor(fraction * polarLength), sideMask);
      const descending = Math.min(Math.floor((1 - fraction) * polarLength), sideMask);
      face = quadrant + (clampedLatitude >= 0 ? 0 : 8);
      column = clampedLatitude >= 0 ? sideMask - descending : ascending;
      line = clampedLatitude >= 0 ? sideMask - ascending : descending;
    }
    let key =
      (BigInt(face) << BigInt(2 * options.order)) | spreadBits(column) | (spreadBits(line) << 1n);
    if (options.scheme === 'uniq') key += 1n << BigInt(2 * options.order + 2);
    cells[row * 2] = Number(key & 0xffffffffn);
    cells[row * 2 + 1] = Number(key >> 32n);
    validity[row] = 1;
  }
  return {cells, validity};
}

/** CPU twin of the HEALPix graph operation. Every property is compile-time except input contents. */
export class CPUHealpixPointToCellOperation extends GPUCPUOperation {
  constructor(props: HealpixPointToCellOperationProps) {
    const operation = new GPUHealpixPointToCellOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER);
    resolveGPUSpatialContext(
      id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
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
        const result = computeHealpixPointToCellOnCPU(positions, {
          order: props.order,
          scheme: props.scheme,
          mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
        });
        return {
          [props.output.cells]: result.cells,
          ...(props.output.validity ? {[props.output.validity]: result.validity} : {})
        };
      }
    });
  }
}
