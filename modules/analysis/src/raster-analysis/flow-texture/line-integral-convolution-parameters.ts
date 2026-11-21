// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveFlowRasterPlacement,
  resolveFlowLocalOrigin,
  type FlowRasterPlacement
} from './flow-raster-placement';

import {
  validateNonZeroFiniteNumber,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';

/** Number of float32 elements in a {@link GPULineIntegralConvolution} parameter view. */
export const GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPULineIntegralConvolution} uint32 parameter view. */
export const GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH = 4;

/** Philox key word 1 of the LIC white-noise stream, shared by the GPU kernel and the CPU oracle. @internal */
export const LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE =
  PHILOX_PURPOSES.lineIntegralConvolutionNoise.id;

/** Per-frame settings of {@link GPULineIntegralConvolution}. */
export type GPULineIntegralConvolutionSettings = {
  /** Optional common planar origin subtracted in binary64 before packing both placements.
   * Defaults to [0, 0], preserving world-frame packet words. For precise relative placement,
   * pass outputPlacement.origin as localOrigin.
   * LIC outputs are scalars; they require no coordinate reconstruction. */
  localOrigin?: readonly [number, number];
  /**
   * Field placement uses an outer-corner `origin` and signed `cellSize`.
   * Sample `(column, row)` is at origin + `(column + 0.5, row + 0.5)` times
   * cell size. Positive cellHeight places row zero at the smallest y; signed cell sizes set the
   * direction of increasing indices. Zero or non-finite sizes throw. The field is sampled
   * bilinearly between centers, with edge clamping inside its half-open outer bounds.
   */
  fieldPlacement: FlowRasterPlacement;
  /**
   * Output raster placement with an outer-corner `origin` and `cellSize`. LIC starts at each
   * output center, origin + `(column + 0.5, row + 0.5)` times cell size, independently of the
   * field lattice. Signed cell sizes set row/column direction; zero or non-finite sizes throw.
   * grid.affine and pixelInterpretation do not change this placement.
   */
  outputPlacement: FlowRasterPlacement;
  /**
   * Streamline step in output pixels. Defaults to 0.5. The step follows the field in world space;
   * on non-square output cells its world length is the one that spans `stepLength` pixels in that
   * direction.
   */
  stepLength?: number;
  /** A streamline stops where the field is slower than this. Defaults to 0. */
  minimumSpeed?: number;
  /** Animation phase in kernel periods; advance it every frame to make the texture flow. Defaults to 0. */
  phase?: number;
  /** Ripple period in steps. `0` disables the animated ripple (static LIC). Defaults to 0. */
  period?: number;
  /** Seed of the white noise. Defaults to 0. */
  seed?: number;
};

/** Settings accepted by the float parameter packer. */
export type GPULineIntegralConvolutionParameterSettings = GPULineIntegralConvolutionSettings;

/**
 * Packs float parameters of {@link GPULineIntegralConvolution}.
 *
 * Layout: `[fieldOriginX, fieldOriginY, fieldCellWidth, fieldCellHeight, outputOriginX,
 * outputOriginY, outputCellWidth, outputCellHeight, stepLength, minimumSpeed, phase, period]`.
 * By default the packet uses world-frame origin words. With localOrigin, both placement origins are
 * differenced against it in binary64 before narrowing. Choosing the output origin gives precise
 * field/output origin differences and zero output-origin words; cell sizes and units are unchanged.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPULineIntegralConvolutionParameterValues(
  settings: GPULineIntegralConvolutionParameterSettings,
  target: Float32Array = new Float32Array(GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH) {
    throw new Error(
      `Line integral convolution parameter target must hold ${GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH} elements`
    );
  }
  const fieldPlacementWords = resolveFlowRasterPlacement(
    'GPULineIntegralConvolution',
    settings,
    'fieldPlacement',
    'fieldExtent'
  );
  const outputPlacementWords = resolveFlowRasterPlacement(
    'GPULineIntegralConvolution',
    settings,
    'outputPlacement',
    'outputExtent'
  );
  for (const [name, extent] of [
    ['fieldPlacement', fieldPlacementWords],
    ['outputPlacement', outputPlacementWords]
  ] as const) {
    validateNonZeroFiniteNumber('GPULineIntegralConvolution', `${name} cellWidth`, extent[2]);
    validateNonZeroFiniteNumber('GPULineIntegralConvolution', `${name} cellHeight`, extent[3]);
  }
  const [localOriginX, localOriginY] = resolveFlowLocalOrigin(
    'GPULineIntegralConvolution',
    settings.localOrigin
  );
  for (const [name, extent] of [
    ['fieldPlacement', fieldPlacementWords],
    ['outputPlacement', outputPlacementWords]
  ] as const) {
    validateFiniteNumber(
      'GPULineIntegralConvolution',
      name + ' originX - localOrigin',
      extent[0] - localOriginX
    );
    validateFiniteNumber(
      'GPULineIntegralConvolution',
      name + ' originY - localOrigin',
      extent[1] - localOriginY
    );
  }
  target.set([
    fieldPlacementWords[0] - localOriginX,
    fieldPlacementWords[1] - localOriginY,
    fieldPlacementWords[2],
    fieldPlacementWords[3],
    outputPlacementWords[0] - localOriginX,
    outputPlacementWords[1] - localOriginY,
    outputPlacementWords[2],
    outputPlacementWords[3],
    settings.stepLength ?? 0.5,
    settings.minimumSpeed ?? 0,
    settings.phase ?? 0,
    settings.period ?? 0
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPULineIntegralConvolution}. Layout: `[seed, 0, 0, 0]`.
 *
 * @param settings Per-frame settings; only `seed` is read.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPULineIntegralConvolutionUint32ParameterValues(
  settings: Pick<GPULineIntegralConvolutionSettings, 'seed'> = {},
  target: Uint32Array = new Uint32Array(GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH) {
    throw new Error(
      `Line integral convolution uint32 parameter target must hold ${GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH} elements`
    );
  }
  const seed = settings.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Line integral convolution seed must be a uint32 integer');
  }
  target.set([seed, 0, 0, 0]);
  return target;
}
