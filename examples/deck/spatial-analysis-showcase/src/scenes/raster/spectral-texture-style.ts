// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type FocalLayer =
  | 'covariance'
  | 'correlation'
  | 'slope'
  | 'intercept'
  | 'rSquared'
  | 'residualSumSquares'
  | 'pairedCount'
  | 'status';
export type SpectralLayer =
  | 'source'
  | 'smooth'
  | 'box'
  | 'weighted'
  | 'convolution'
  | 'gradient'
  | 'gradientX'
  | 'gradientY'
  | 'sobelX'
  | 'sobelY'
  | 'scharrX'
  | 'scharrY'
  | 'laplacian'
  | 'texture';
export const SPECTRAL_MAP_LAYERS: readonly SpectralLayer[] = [
  'source',
  'smooth',
  'box',
  'weighted',
  'convolution',
  'gradient',
  'gradientX',
  'gradientY',
  'sobelX',
  'sobelY',
  'scharrX',
  'scharrY',
  'laplacian',
  'texture'
];
import type {RampName} from '../../engine/ramps';

export type SpectralStyle = {
  ramp: RampName | 'category';
  extent: readonly [number, number];
  unit: string;
  kind?: 'continuous' | 'categorical';
  palette?: readonly (readonly [number, number, number, number])[];
};

export const SPECTRAL_STATUS_PALETTE = [
  [87, 235, 168, 255],
  [255, 148, 72, 255],
  [107, 158, 255, 255],
  [189, 122, 255, 255],
  [255, 105, 168, 255],
  [255, 92, 92, 255],
  [245, 220, 87, 255]
] as const;

const POSITIVE: SpectralStyle = {ramp: 'ylorrd', extent: [0, 0.6], unit: 'reflectance'};
const SIGNED: SpectralStyle = {
  ramp: 'diverging',
  extent: [-0.08, 0.08],
  unit: 'reflectance / pixel'
};

export const SPECTRAL_LAYER_STYLE: Record<SpectralLayer | FocalLayer, SpectralStyle> = {
  source: POSITIVE,
  smooth: POSITIVE,
  box: POSITIVE,
  weighted: POSITIVE,
  convolution: {ramp: 'diverging', extent: [-0.08, 0.08], unit: 'reflectance / pixel'},
  gradient: {ramp: 'oranges', extent: [0, 0.08], unit: 'reflectance / pixel'},
  gradientX: SIGNED,
  gradientY: SIGNED,
  sobelX: SIGNED,
  sobelY: SIGNED,
  scharrX: SIGNED,
  scharrY: SIGNED,
  laplacian: {ramp: 'diverging', extent: [-0.2, 0.2], unit: 'reflectance / pixel²'},
  texture: {ramp: 'ylorrd', extent: [0, 49], unit: 'GLCM feature'},
  covariance: {ramp: 'diverging', extent: [-0.001, 0.001], unit: 'reflectance²'},
  correlation: {ramp: 'diverging', extent: [-1, 1], unit: 'unitless'},
  slope: {ramp: 'diverging', extent: [-2, 2], unit: 'reflectance / reflectance'},
  intercept: {ramp: 'diverging', extent: [-0.6, 0.6], unit: 'reflectance'},
  rSquared: {ramp: 'ylorrd', extent: [0, 1], unit: 'unitless'},
  residualSumSquares: {ramp: 'ylorrd', extent: [0, 0.1], unit: 'reflectance²'},
  pairedCount: {ramp: 'ylorrd', extent: [0, 49], unit: 'paired samples'},
  status: {
    ramp: 'category',
    extent: [0, 6],
    unit: 'fit status class',
    kind: 'categorical',
    palette: SPECTRAL_STATUS_PALETTE
  }
};

export function getSpectralLayerStyle(
  layer: SpectralLayer | FocalLayer,
  focalRadius = 3
): SpectralStyle {
  if (layer === 'pairedCount') {
    return {...SPECTRAL_LAYER_STYLE.pairedCount, extent: [0, (focalRadius * 2 + 1) ** 2]};
  }
  return SPECTRAL_LAYER_STYLE[layer];
}

export function getSpectralTextureRange(
  metric: string,
  levels: number,
  displayDomain: 'detail' | 'theoretical' = 'detail'
): SpectralStyle {
  switch (metric) {
    case 'correlation':
      return {ramp: 'diverging', extent: [-1, 1], unit: 'unitless GLCM correlation'};
    case 'contrast':
      return {
        ramp: 'ylorrd',
        extent: [0, displayDomain === 'detail' ? 4 : (levels - 1) ** 2],
        unit:
          displayDomain === 'detail'
            ? 'quantized bin² · detail-clipped'
            : 'quantized bin² · theoretical'
      };
    case 'dissimilarity':
      return {
        ramp: 'ylorrd',
        extent: [0, displayDomain === 'detail' ? 2 : levels - 1],
        unit:
          displayDomain === 'detail'
            ? 'quantized bins · detail-clipped'
            : 'quantized bins · theoretical'
      };
    case 'entropy':
      return {ramp: 'ylorrd', extent: [0, 2 * Math.log(levels)], unit: 'nats'};
    case 'homogeneity':
      return {ramp: 'ylorrd', extent: [0, 1], unit: 'unitless GLCM feature'};
    case 'ASM':
      return {ramp: 'ylorrd', extent: [0, 1], unit: 'unitless GLCM feature'};
    default:
      return {ramp: 'ylorrd', extent: [0, 1], unit: 'unitless GLCM feature'};
  }
}
