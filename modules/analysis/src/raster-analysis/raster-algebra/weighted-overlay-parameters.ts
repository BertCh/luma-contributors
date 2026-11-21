// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget, validateRasterOption} from '../raster-band-utils';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest layer count accepted by {@link GPUWeightedOverlay}. */
export const GPU_WEIGHTED_OVERLAY_MAXIMUM_LAYER_COUNT = 16;

/** float32 elements before the per-layer records in a weighted-overlay parameter view. */
const HEADER_LENGTH = 8;
/** float32 elements per layer record. */
const LAYER_LENGTH = 8;
/** Smallest normal float32 reciprocal accepted by the active linear overlay route. */
export const GPU_WEIGHTED_OVERLAY_MINIMUM_NORMAL_SCALE = 2 ** -126;

/** Remap of one {@link GPUWeightedOverlay} input layer. */
export type GPUWeightedOverlayLayerSettings = {
  /** Layer weight. Negative weights are allowed and subtract. */
  weight: number;
  /**
   * `'linear'` (default) maps `[inputMinimum, inputMaximum]` to `[0, 1]`, clamped. `'table'` classifies
   * the value through this layer's rows of `remapBreaks` and returns the matching `remapValues`
   * row; a NaN remap value marks the class as restricted.
   */
  mode?: 'linear' | 'table';
  /** Linear mode: input value mapped to 0. Defaults to 0. */
  inputMinimum?: number;
  /** Linear mode: input value mapped to 1. Defaults to 1. */
  inputMaximum?: number;
  /** Linear mode: use `1 - t` so lower inputs score higher. */
  invert?: boolean;
  /** Table mode: active breaks of this layer, at most the contributor's `maximumBreakCount`. */
  breakCount?: number;
  /** Table mode interval closure, as in `GPURasterReclassify`. Defaults to `'left'`. */
  closed?: 'left' | 'right';
};

/** Per-frame settings of {@link GPUWeightedOverlay}. */
export type GPUWeightedOverlaySettings = {
  /** One record per input layer, in stack order. */
  layers: readonly GPUWeightedOverlayLayerSettings[];
  /** Divide the weighted sum by the sum of absolute weights of the contributing layers. */
  normalizeWeights?: boolean;
  /**
   * `'propagate'` (default): a nodata input makes the cell nodata. `'ignore'`: nodata layers are
   * skipped (and excluded from weight normalization); a cell with no valid layer is nodata.
   * `'strict'` is an alias for propagate.
   */
  noDataPolicy?: 'propagate' | 'strict' | 'ignore';
};

/** Returns the float32 element count of a weighted-overlay parameter view for `layerCount`. */
export function getGPUWeightedOverlayParameterLength(layerCount: number): number {
  return HEADER_LENGTH + LAYER_LENGTH * layerCount;
}

/**
 * Packs per-frame {@link GPUWeightedOverlay} parameters.
 *
 * Layout (float32): header `[normalizeWeights, ignoreNoData, 0, 0, 0, 0, 0, 0]`, then per layer
 * `[weight, mode, inputMinimum, scale, invert, breakCount, closedRight, degenerate]` where
 * `mode` is 0 (linear) or 1 (table), `scale = 1 / (inputMaximum - inputMinimum)` is computed here so the
 * shader multiplies instead of dividing, and `degenerate` is 1 when `inputMaximum == inputMinimum` (the
 * layer then scores 1 for values `>= inputMinimum`, else 0).
 * Active unequal linear ramps must retain endpoint order and a finite normal float32 reciprocal
 * scale (absolute scale at least 2^-126). Table mode ignores retained linear-ramp settings.
 *
 * @param settings Layer records and global options.
 * @param target Optional destination of at least `getGPUWeightedOverlayParameterLength(layers)`.
 * @param options `hasTables: false` declares that the overlay was built without remap tables, so a
 * `'table'` layer throws here instead of scoring NaN (nodata) on every cell. Defaults to true.
 * @throws If the target is too short, a weight or range is not finite, a count is invalid, or a
 * table layer is packed with `hasTables: false`.
 */
export function getGPUWeightedOverlayParameterValues(
  settings: GPUWeightedOverlaySettings,
  target: Float32Array = new Float32Array(
    getGPUWeightedOverlayParameterLength(settings.layers.length)
  ),
  options: {hasTables?: boolean} = {}
): Float32Array {
  const length = getGPUWeightedOverlayParameterLength(settings.layers.length);
  validateParameterTarget('getGPUWeightedOverlayParameterValues', target, length);
  validateRasterOption(
    'getGPUWeightedOverlayParameterValues',
    'noDataPolicy',
    settings.noDataPolicy ?? 'propagate',
    ['propagate', 'strict', 'ignore']
  );
  const candidate = new Float32Array(length);
  candidate[0] = settings.normalizeWeights ? 1 : 0;
  candidate[1] = settings.noDataPolicy === 'ignore' ? 1 : 0;
  for (const [layerIndex, layer] of settings.layers.entries()) {
    validateRasterOption(
      'getGPUWeightedOverlayParameterValues',
      `layer ${layerIndex} mode`,
      layer.mode ?? 'linear',
      ['linear', 'table']
    );
    validateRasterOption(
      'getGPUWeightedOverlayParameterValues',
      `layer ${layerIndex} closed`,
      layer.closed ?? 'left',
      ['left', 'right']
    );
    const linear = layer.mode !== 'table';
    const inputMinimum = layer.inputMinimum ?? 0;
    const inputMaximum = layer.inputMaximum ?? 1;
    const breakCount = layer.breakCount ?? 0;
    for (const [name, value] of [
      ['weight', layer.weight],
      ...(linear
        ? ([
            ['inputMinimum', inputMinimum],
            ['inputMaximum', inputMaximum]
          ] as const)
        : [])
    ] as const) {
      validateRequirement(
        'getGPUWeightedOverlayParameterValues',
        Number.isFinite(value),
        `layer ${layerIndex} ${name} must be finite (got ${value})`
      );
    }
    validateRequirement(
      'getGPUWeightedOverlayParameterValues',
      Number.isSafeInteger(breakCount) && breakCount >= 0,
      `layer ${layerIndex} breakCount must be a non-negative integer (got ${breakCount})`
    );
    validateRequirement(
      'getGPUWeightedOverlayParameterValues',
      layer.mode !== 'table' || options.hasTables !== false,
      `layer ${layerIndex} is a table layer but has no remap tables; provide remap tables or use mode linear`
    );
    const base = HEADER_LENGTH + layerIndex * LAYER_LENGTH;
    const encodedMinimum = Math.fround(inputMinimum);
    const encodedMaximum = Math.fround(inputMaximum);
    validateRequirement(
      'getGPUWeightedOverlayParameterValues',
      !linear ||
        (Number.isFinite(encodedMinimum) &&
          Number.isFinite(encodedMaximum) &&
          (inputMinimum === inputMaximum ||
            Math.sign(encodedMaximum - encodedMinimum) === Math.sign(inputMaximum - inputMinimum))),
      `layer ${layerIndex} inputMinimum/inputMaximum must preserve requested order in float32 (requested [${inputMinimum}, ${inputMaximum}], encoded [${encodedMinimum}, ${encodedMaximum}])`
    );
    const degenerate = linear ? inputMaximum === inputMinimum : encodedMinimum === encodedMaximum;
    const encodedScale = Math.fround(degenerate ? 0 : 1 / (encodedMaximum - encodedMinimum));
    candidate[base] = layer.weight;
    candidate[base + 1] = layer.mode === 'table' ? 1 : 0;
    candidate[base + 2] = !linear && !Number.isFinite(encodedMinimum) ? 0 : encodedMinimum;
    candidate[base + 3] = !linear && !Number.isFinite(encodedScale) ? 0 : encodedScale;
    candidate[base + 4] = layer.invert ? 1 : 0;
    candidate[base + 5] = breakCount;
    candidate[base + 6] = layer.closed === 'right' ? 1 : 0;
    candidate[base + 7] = degenerate ? 1 : 0;
    validateRequirement(
      'getGPUWeightedOverlayParameterValues',
      candidate.subarray(base, base + LAYER_LENGTH).every(Number.isFinite) &&
        (!linear ||
          degenerate ||
          Math.abs(candidate[base + 3]) >= GPU_WEIGHTED_OVERLAY_MINIMUM_NORMAL_SCALE),
      `layer ${layerIndex} weight and breakCount must be finite float32; an active unequal linear ramp requires a normal float32 reciprocal scale (absolute scale >= ${GPU_WEIGHTED_OVERLAY_MINIMUM_NORMAL_SCALE}; got weight ${layer.weight}, breakCount ${breakCount}, scale ${candidate[base + 3]}); use table mode for a wider ramp`
    );
  }
  target.set(candidate);
  return target;
}
