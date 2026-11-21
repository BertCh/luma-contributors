// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateOption,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterGrid} from '../raster-band-utils';

/** Outer records are acquisition planes or cell histories; bands are innermost in either case. */
export type RasterStackSourceLayout = 'acquisition-major' | 'cell-major';
export type PrepareRasterStackInputProps = {
  id?: string;
  grid: GPURasterMetadata;
  bandCount: number;
  /** Acquisition-major: one cells-times-bands plane per acquisition. Cell-major: one
   * slices-times-bands history per cell. All records must have identical length. */
  slices: readonly ArrayLike<number>[];
  sourceLayout: RasterStackSourceLayout;
  /** One uint32 per observation, zero rejects it; layout must be explicit when supplied. */
  mask?: readonly Uint32Array[];
  maskLayout?: RasterStackSourceLayout;
};
export type PreparedRasterStackInput = {
  values: Float32Array;
  mask?: Uint32Array;
  cellCount: number;
  sliceCount: number;
  bandCount: number;
};

/** Copies host planes/histories into `(cell * sliceCount + slice) * bandCount + band` order.
 * Counts can be passed directly to either composite constructor. Grid alignment, calibration and
 * acquisition ordering are caller preconditions; no resampling or missing-value conversion occurs. */
export function prepareRasterStackInput(
  props: PrepareRasterStackInputProps
): PreparedRasterStackInput {
  const id = props.id ?? 'prepareRasterStackInput';
  const cellCount = validateRasterGrid(id, props.grid);
  validateCapacity(id, 'bandCount', props.bandCount);
  validateOption(id, 'sourceLayout', props.sourceLayout, ['acquisition-major', 'cell-major']);
  validateCapacity(id, 'slices.length', props.slices.length);
  const sliceCount =
    props.sourceLayout === 'acquisition-major'
      ? props.slices.length
      : props.slices[0].length / props.bandCount;
  validateCapacity(id, 'sliceCount', sliceCount);
  const observationCount = cellCount * sliceCount;
  validateCapacity(id, 'cellCount times sliceCount', observationCount);
  validateCapacity(id, 'valueCount', observationCount * props.bandCount);
  const outerCount = props.sourceLayout === 'acquisition-major' ? sliceCount : cellCount;
  const innerCount = props.sourceLayout === 'acquisition-major' ? cellCount : sliceCount;
  validateRequirement(
    id,
    props.slices.length === outerCount &&
      props.slices.every(record => record.length === innerCount * props.bandCount),
    `slices must contain ${outerCount} records of ${innerCount * props.bandCount} values`
  );
  if (props.mask !== undefined || props.maskLayout !== undefined) {
    validateOption(id, 'maskLayout', props.maskLayout, ['acquisition-major', 'cell-major']);
    const maskOuterCount = props.maskLayout === 'acquisition-major' ? sliceCount : cellCount;
    const maskInnerCount = props.maskLayout === 'acquisition-major' ? cellCount : sliceCount;
    validateRequirement(
      id,
      props.mask !== undefined &&
        props.mask.length === maskOuterCount &&
        props.mask.every(record => record.length === maskInnerCount),
      `mask must contain ${maskOuterCount} records of ${maskInnerCount} observations in maskLayout ${props.maskLayout}`
    );
  }
  const values = new Float32Array(observationCount * props.bandCount);
  const mask = props.mask === undefined ? undefined : new Uint32Array(observationCount);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      const source = props.slices[props.sourceLayout === 'acquisition-major' ? slice : cell];
      const sourceOffset =
        (props.sourceLayout === 'acquisition-major' ? cell : slice) * props.bandCount;
      for (let band = 0; band < props.bandCount; band++) {
        values[(cell * sliceCount + slice) * props.bandCount + band] = source[sourceOffset + band];
      }
      if (mask && props.mask)
        mask[cell * sliceCount + slice] =
          props.mask[props.maskLayout === 'acquisition-major' ? slice : cell][
            props.maskLayout === 'acquisition-major' ? cell : slice
          ];
    }
  }
  return {values, ...(mask ? {mask} : {}), cellCount, sliceCount, bandCount: props.bandCount};
}
