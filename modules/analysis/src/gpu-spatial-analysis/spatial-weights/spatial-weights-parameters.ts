// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a fixed kernel-bandwidth parameter buffer. */
export const GPU_SPATIAL_WEIGHTS_TRANSFORM_PARAMETER_LENGTH = 1;
/** Number of float32 elements in a lattice cell-size parameter buffer. */
export const GPU_LATTICE_WEIGHTS_PARAMETER_LENGTH = 2;
/** Smallest positive float32: sizes and bandwidths must remain positive after packing. */
const MINIMUM_POSITIVE_FLOAT32 = 2 ** -149;

/** Per-frame fixed kernel settings; the neighborhood and kernel profile stay compile-time. */
export type GPUSpatialWeightsTransformSettings = {
  /** Positive fixed bandwidth, in the units of the resident neighbor distances. */
  bandwidth: number;
};

/** Packs `[bandwidth]` for `bandwidth: parameterBuffer.importToGraph(graph)`. */
export function getGPUSpatialWeightsTransformParameterValues(
  settings: GPUSpatialWeightsTransformSettings,
  target: Float32Array = new Float32Array(GPU_SPATIAL_WEIGHTS_TRANSFORM_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUSpatialWeightsTransformParameterValues';
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_SPATIAL_WEIGHTS_TRANSFORM_PARAMETER_LENGTH
  );
  validateFiniteNumber(id, 'bandwidth', settings.bandwidth, MINIMUM_POSITIVE_FLOAT32);
  target[0] = settings.bandwidth;
  return target;
}

/** Per-frame distance scale of a lattice; topology stays compile-time. */
export type GPULatticeWeightsSettings = {
  /** Positive cell width and height. Defaults to `[1, 1]`. */
  cellSize?: readonly [number, number];
};

/** Packs `[cellWidth, cellHeight]` for `cellSize: parameterBuffer.importToGraph(graph)`. */
export function getGPULatticeWeightsParameterValues(
  settings: GPULatticeWeightsSettings = {},
  target: Float32Array = new Float32Array(GPU_LATTICE_WEIGHTS_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPULatticeWeightsParameterValues';
  validateCapacity(id, 'target.length', target.length, GPU_LATTICE_WEIGHTS_PARAMETER_LENGTH);
  const [cellWidth, cellHeight] = settings.cellSize ?? [1, 1];
  validateFiniteNumber(id, 'cellSize[0]', cellWidth, MINIMUM_POSITIVE_FLOAT32);
  validateFiniteNumber(id, 'cellSize[1]', cellHeight, MINIMUM_POSITIVE_FLOAT32);
  target.set([cellWidth, cellHeight]);
  return target;
}

/** Number of float32 elements in a symmetry-tolerance parameter buffer. */
export const GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH = 1;

/** Per-frame reverse-weight comparison settings; the resident CSR layout stays compile-time. */
export type GPUSpatialWeightsSummarySettings = {
  /** Non-negative absolute tolerance for counting unequal reverse weights. Defaults to 0. */
  symmetryTolerance?: number;
};

/** Packs `[symmetryTolerance]` for `symmetryTolerance: parameterBuffer.importToGraph(graph)`. */
export function getGPUSpatialWeightsSummaryParameterValues(
  settings: GPUSpatialWeightsSummarySettings = {},
  target: Float32Array = new Float32Array(GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUSpatialWeightsSummaryParameterValues';
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH
  );
  validateFiniteNumber(id, 'symmetryTolerance', settings.symmetryTolerance ?? 0, 0);
  target[0] = settings.symmetryTolerance ?? 0;
  return target;
}

/** Number of float32 elements in a transpose symmetry-tolerance parameter buffer. */
export const GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH = 1;

/** Per-frame symmetry settings shared by packed and chunked spatial-weights transposes. */
export type GPUSpatialWeightsTransposeSettings = {
  /** Non-negative absolute reverse-weight tolerance. Missing reverse slots always differ. Defaults to 0. */
  symmetryTolerance?: number;
};

/** Packs `[symmetryTolerance]` for `symmetryTolerance: parameterBuffer.importToGraph(graph)`. */
export function getGPUSpatialWeightsTransposeParameterValues(
  settings: GPUSpatialWeightsTransposeSettings = {},
  target: Float32Array = new Float32Array(GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUSpatialWeightsTransposeParameterValues';
  validateCapacity(
    id,
    'target.length',
    target.length,
    GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH
  );
  validateFiniteNumber(id, 'symmetryTolerance', settings.symmetryTolerance ?? 0, 0);
  target[0] = settings.symmetryTolerance ?? 0;
  return target;
}
