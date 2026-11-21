// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPULineIntegralConvolutionSettings,
  GPUStreamlinesSettings
} from '../../../src/raster-analysis/flow-texture/index';
import type {GPUParticleAdvectionSettings} from '../../../src/raster-analysis/particle-advection/index';

type SavedPlacement = readonly [number, number, number, number];

export type SavedLICSettings = Omit<
  GPULineIntegralConvolutionSettings,
  'fieldPlacement' | 'outputPlacement'
> & {
  fieldExtent: SavedPlacement;
  outputExtent: SavedPlacement;
};

export type SavedStreamlinesSettings = Omit<
  GPUStreamlinesSettings,
  'fieldPlacement' | 'gridPlacement'
> & {
  fieldExtent: SavedPlacement;
  gridExtent: SavedPlacement;
};

export type SavedParticleAdvectionSettings = Omit<
  GPUParticleAdvectionSettings,
  'fieldPlacement'
> & {
  fieldExtent: SavedPlacement;
};

// The golden files store the outer-corner origin and signed steps as one tuple.
function decodePlacement([originX, originY, cellWidth, cellHeight]: SavedPlacement) {
  return {origin: [originX, originY], cellSize: [cellWidth, cellHeight]} as const;
}

export function decodeSavedLICSettings({
  fieldExtent,
  outputExtent,
  ...settings
}: SavedLICSettings): GPULineIntegralConvolutionSettings {
  return {
    ...settings,
    fieldPlacement: decodePlacement(fieldExtent),
    outputPlacement: decodePlacement(outputExtent)
  };
}

export function decodeSavedStreamlinesSettings({
  fieldExtent,
  gridExtent,
  ...settings
}: SavedStreamlinesSettings): GPUStreamlinesSettings {
  return {
    ...settings,
    fieldPlacement: decodePlacement(fieldExtent),
    gridPlacement: decodePlacement(gridExtent)
  };
}

export function decodeSavedParticleAdvectionSettings({
  fieldExtent,
  ...settings
}: SavedParticleAdvectionSettings): GPUParticleAdvectionSettings {
  return {...settings, fieldPlacement: decodePlacement(fieldExtent)};
}
