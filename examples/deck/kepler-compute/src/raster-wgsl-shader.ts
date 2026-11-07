// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  compileBandMath,
  emitBandMathWgsl,
  KEPLER_BAND_COMBINATION_EXPRESSIONS,
  parseBandMath
} from './raster-band-math';

import {clamp} from '@math.gl/core';

/** Texture slots of the raster tile shader: four value bands, four validity masks, colormap, RGBA. */
export const RASTER_WGSL_BAND_SLOT_COUNT = 4;

/**
 * Extra value-only slots (b5..b8) for custom band math over more than four sources. They carry no
 * validity texture (the tile mask is shared and gated through slots 1..4; non-finite results are
 * dropped), which keeps the stage at 14 sampled textures, under WebGPU's default limit of 16.
 */
export const RASTER_WGSL_EXTENDED_SLOT_COUNT = 8;

/**
 * Grayscale draws t,t,t. Colormap samples the colormap. `index` draws Kepler's red channel: Kepler's
 * spectral index modules write `vec4(index, 0, 0, 0)` and nothing else when no colormap is applied.
 */
export type RasterWgslMode = 'grayscale' | 'colormap' | 'rgb' | 'rgba' | 'index';

/** Numeric mode codes written to the `settings.x` uniform. */
export const RASTER_WGSL_MODE_CODES: Readonly<Record<RasterWgslMode, number>> = {
  grayscale: 0,
  colormap: 1,
  rgb: 2,
  rgba: 3,
  index: 4
};

/** What the Kepler `RasterTileLayer` sublayer props decide about one tile's draw. */
export type RasterWgslSubLayerProps = Readonly<{
  /** Position in `images.imageBands` that feeds `b1..b4` (Kepler `renderBandIndexes`). */
  renderBandIndexes?: readonly number[] | null;
  bandCombination?: string;
  /** Custom band math over b1..b8; used when `bandCombination` is not a Kepler preset. */
  bandMathExpression?: string;
  minPixelValue?: number;
  maxPixelValue?: number;
  linearRescalingFactor?: readonly [number, number];
  nonLinearRescaling?: boolean;
  filterEnabled?: boolean;
  filterRange?: readonly [number, number];
  /** Kepler `RasterTileLayer` non-linear rescale parameters (used when `nonLinearRescaling`). */
  gammaContrastFactor?: number;
  sigmoidalContrastFactor?: number;
  sigmoidalBiasFactor?: number;
  saturationValue?: number;
  /** Categorical colormap stepping (Kepler `hasCategoricalColorMap` and the band value range). */
  hasCategoricalColorMap?: boolean;
  dataType?: string;
  minCategoricalBandValue?: number;
  maxCategoricalBandValue?: number;
  /** Per-tile image descriptor: how many bands, whether a colormap or plain RGBA image exists. */
  imageBandCount: number;
  hasColormap: boolean;
  hasRgba: boolean;
  hasMask: boolean;
}>;

/**
 * Kepler's per-channel pipeline before band math: `value * scaler + offset`, then optional gamma
 * (`pow(x, 1 / gamma)`), sigmoidal contrast and saturation. All stages are skipped when disabled.
 */
export type RasterWgslRescale = Readonly<{
  isEnabled: boolean;
  scaler: number;
  offset: number;
  /** 1 disables gamma. */
  gamma: number;
  /** 0 disables sigmoidal contrast. */
  sigmoidalContrast: number;
  sigmoidalBias: number;
  /** 1 disables saturation. */
  saturation: number;
  /** Kepler categorical colormap: texel position is `value * step` of the 256 wide strip. */
  categoricalStep: number;
  isCategorical: boolean;
}>;

export type RasterWgslRenderPlan = Readonly<{
  mode: RasterWgslMode;
  /** WGSL f32 expression over b1..b4. */
  expression: string;
  /** Source expression for diagnostics. */
  sourceExpression: string;
  /** `imageBands` position feeding each of b1..b4 (b1..b8 for extended custom math); unused slots repeat position 0. */
  bandPositions: readonly number[];
  /** Bit i set when b(i+1) is referenced, so its validity gates the pixel. */
  usedBandMask: number;
  domain: readonly [number, number];
  rescale: RasterWgslRescale;
  /** Kepler maps normalized-difference style results from [-1, 1] onto the colormap. */
  isSignedColormap: boolean;
  isFilterEnabled: boolean;
  filterRange: readonly [number, number];
}>;

/** Kepler `dtypeMaxValue`: only integer data types have a finite maximum. */
const INTEGER_DATA_TYPES = new Set(['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32']);

const DISABLED_RESCALE: RasterWgslRescale = {
  isEnabled: false,
  scaler: 1,
  offset: 0,
  gamma: 1,
  sigmoidalContrast: 0,
  sigmoidalBias: 0.5,
  saturation: 1,
  categoricalStep: 0,
  isCategorical: false
};

/**
 * Mirrors Kepler `getModules` (gpu-utils.js): categorical colormaps rescale by `1 / maxPixelValue`;
 * `single` and `rgb` rescale linearly (`1 / (localRange * globalRange)`, offset `-low`) or, with
 * `nonLinearRescaling`, by `1 / maxPixelValue` followed by gamma and sigmoidal contrast; saturation
 * follows for `single` and `rgb`. Index combinations are not rescaled.
 */
export function planRasterRescale(props: RasterWgslSubLayerProps): RasterWgslRescale {
  const combination = props.bandCombination ?? 'rgb';
  const maximum = Number.isFinite(props.maxPixelValue) ? (props.maxPixelValue as number) : 1;
  const minimum = Number.isFinite(props.minPixelValue) ? (props.minPixelValue as number) : 0;
  let rescale: RasterWgslRescale = DISABLED_RESCALE;
  if (props.hasCategoricalColorMap) {
    rescale = {...rescale, isEnabled: true, scaler: 1 / maximum};
  } else if (combination === 'rgb' || combination === 'single') {
    if (!props.nonLinearRescaling) {
      const [low, high] = props.linearRescalingFactor ?? [0, 1];
      rescale = {
        ...rescale,
        isEnabled: true,
        scaler: 1 / ((high - low) * (maximum - minimum)),
        offset: -low
      };
    } else {
      const contrast = props.sigmoidalContrastFactor;
      rescale = {
        ...rescale,
        isEnabled: true,
        scaler: 1 / maximum,
        gamma: props.gammaContrastFactor || 1,
        sigmoidalContrast: Number.isFinite(contrast) ? (contrast as number) : 0,
        sigmoidalBias: Number.isFinite(props.sigmoidalBiasFactor)
          ? (props.sigmoidalBiasFactor as number)
          : 0.5
      };
    }
    const saturation = props.saturationValue;
    if (Number.isFinite(saturation) && saturation !== 1)
      rescale = {...rescale, saturation: saturation as number};
  }
  // Kepler colormap uniforms are i32, so the stepping uses truncated integers.
  const range =
    Math.trunc(props.maxCategoricalBandValue ?? NaN) -
    Math.trunc(props.minCategoricalBandValue ?? NaN);
  const isCategorical =
    INTEGER_DATA_TYPES.has(props.dataType ?? '') &&
    Number.isFinite(props.maxPixelValue) &&
    Number.isFinite(range) &&
    range !== 0;
  return isCategorical
    ? {...rescale, isCategorical, categoricalStep: Math.trunc(maximum) / range}
    : rescale;
}

/** Bit set written to `rescale.w`: 1 rescale, 2 gamma, 4 sigmoidal contrast, 8 saturation, 16 categorical. */
export function getRasterRescaleFlags(rescale: RasterWgslRescale): number {
  return (
    (rescale.isEnabled ? 1 : 0) |
    (rescale.gamma !== 1 ? 2 : 0) |
    (rescale.sigmoidalContrast !== 0 ? 4 : 0) |
    (rescale.saturation !== 1 ? 8 : 0) |
    (rescale.isCategorical ? 16 : 0)
  );
}

/** Highest band an expression may reference (b1..b8). */
function getExpressionSlotCount(bandIndexes: readonly number[]): number {
  return Math.max(0, ...bandIndexes) > RASTER_WGSL_BAND_SLOT_COUNT
    ? RASTER_WGSL_EXTENDED_SLOT_COUNT
    : RASTER_WGSL_BAND_SLOT_COUNT;
}

const SIGNED_COMBINATIONS = new Set([
  'normalizedDifference',
  'enhancedVegetationIndex',
  'soilAdjustedVegetationIndex',
  'modifiedSoilAdjustedVegetationIndex'
]);

/**
 * Decides how one Kepler raster tile is drawn by the WGSL path. Returns null only for sources the
 * path cannot draw: an unparsable custom expression, or one that reads past b8. Custom band math
 * (`bandMathExpression`, or an unknown `bandCombination` that parses as band math) reads up to
 * eight bands; Kepler's own presets read at most three. Saturation applies to `rgb` composites only:
 * a single band has no chroma, so single mode never reads the identity-filled neighbour channels.
 */
export function planRasterRender(props: RasterWgslSubLayerProps): RasterWgslRenderPlan | null {
  const combination = props.bandCombination ?? 'rgb';
  const rawRescale = planRasterRescale(props);
  // Kepler's saturation mixes (b1, identity-filled b2, identity-filled b3); only a real composite means it.
  const rescale: RasterWgslRescale =
    combination === 'rgb' || rawRescale.saturation === 1
      ? rawRescale
      : {...rawRescale, saturation: 1};
  const ordering = props.renderBandIndexes?.length ? props.renderBandIndexes : null;
  const getPositions = (slotCount: number, usedMask: number): number[] =>
    Array.from({length: slotCount}, (_, slot) => {
      const isNeeded = Boolean(usedMask & (1 << slot)) || (rescale.saturation !== 1 && slot < 3);
      const position = ordering?.[slot] ?? (isNeeded ? slot : 0);
      return position >= 0 && position < Math.max(1, props.imageBandCount) ? position : 0;
    });
  const minimum = Number.isFinite(props.minPixelValue) ? (props.minPixelValue as number) : 0;
  const maximum = Number.isFinite(props.maxPixelValue) ? (props.maxPixelValue as number) : 1;
  const range = maximum - minimum;
  const [lowFraction, highFraction] = props.linearRescalingFactor ?? [0, 1];
  const domain: readonly [number, number] = [
    minimum + lowFraction * range,
    minimum + highFraction * range
  ];
  const base = {
    domain,
    rescale,
    isFilterEnabled: Boolean(props.filterEnabled && props.filterRange),
    filterRange: props.filterRange ?? ([0, 0] as const)
  };
  if (props.hasRgba && props.imageBandCount === 0)
    return {
      ...base,
      bandPositions: getPositions(RASTER_WGSL_BAND_SLOT_COUNT, 0),
      mode: 'rgba',
      expression: 'b1',
      sourceExpression: 'rgba',
      usedBandMask: 0,
      isSignedColormap: false
    };
  if (combination === 'rgb') {
    // Composite of three bands through the same rescale used for single bands.
    const usedBandMask = props.imageBandCount >= 3 ? 0b0111 : 0b0001;
    return {
      ...base,
      bandPositions: getPositions(RASTER_WGSL_BAND_SLOT_COUNT, usedBandMask),
      mode: props.imageBandCount >= 3 ? 'rgb' : 'grayscale',
      expression: 'b1',
      sourceExpression: 'rgb',
      usedBandMask,
      isSignedColormap: false
    };
  }
  const isPreset = combination === 'single' || combination in KEPLER_BAND_COMBINATION_EXPRESSIONS;
  const sourceExpression = isPreset
    ? (KEPLER_BAND_COMBINATION_EXPRESSIONS[combination] ?? 'b1')
    : (props.bandMathExpression ?? combination);
  let compiled: ReturnType<typeof compileBandMath>;
  let expression: string;
  try {
    compiled = compileBandMath(sourceExpression);
    expression = emitBandMathWgslFromSource(sourceExpression);
  } catch {
    // An unparsable custom expression keeps Kepler's draw.
    return null;
  }
  if (compiled.bandIndexes.some(bandIndex => bandIndex > RASTER_WGSL_EXTENDED_SLOT_COUNT))
    return null;
  const usedBandMask = compiled.bandIndexes.reduce(
    (mask, bandIndex) => mask | (1 << (bandIndex - 1)),
    0
  );
  const isSigned = SIGNED_COMBINATIONS.has(combination);
  // Kepler's index modules emit vec4(index, 0, 0, 0); without a colormap that is the red channel.
  const isIndex = combination in KEPLER_BAND_COMBINATION_EXPRESSIONS;
  return {
    ...base,
    bandPositions: getPositions(getExpressionSlotCount(compiled.bandIndexes), usedBandMask),
    mode: props.hasColormap ? 'colormap' : isIndex ? 'index' : 'grayscale',
    expression,
    sourceExpression,
    usedBandMask,
    domain: isSigned ? [-1, 1] : domain,
    isSignedColormap: isSigned
  };
}

/** WGSL f32 expression for a band-math source string (b1..b4). */
export function emitBandMathWgslFromSource(expression: string): string {
  return emitBandMathWgsl(parseBandMath(expression));
}

/** Uniform module and binding declarations shared by every variant of the shader. */
export const RASTER_WGSL_MODULE_SOURCE = /* wgsl */ `\
struct RasterWgslUniforms {
  domain: vec4<f32>,
  settings: vec4<f32>,
  bands: vec4<f32>,
  rescale: vec4<f32>,
  tone: vec4<f32>,
};
@group(3) @binding(0) var<uniform> rasterWgsl: RasterWgslUniforms;
@group(3) @binding(1) var valueTexture1: texture_2d<f32>;
@group(3) @binding(2) var valueTexture2: texture_2d<f32>;
@group(3) @binding(3) var valueTexture3: texture_2d<f32>;
@group(3) @binding(4) var valueTexture4: texture_2d<f32>;
@group(3) @binding(5) var validityTexture1: texture_2d<f32>;
@group(3) @binding(6) var validityTexture2: texture_2d<f32>;
@group(3) @binding(7) var validityTexture3: texture_2d<f32>;
@group(3) @binding(8) var validityTexture4: texture_2d<f32>;
@group(3) @binding(9) var colormapTexture: texture_2d<f32>;
@group(3) @binding(10) var rgbaTexture: texture_2d<f32>;
@group(3) @binding(11) var valueTexture5: texture_2d<f32>;
@group(3) @binding(12) var valueTexture6: texture_2d<f32>;
@group(3) @binding(13) var valueTexture7: texture_2d<f32>;
@group(3) @binding(14) var valueTexture8: texture_2d<f32>;
`;

export const RASTER_WGSL_TEXTURE_NAMES = [
  'valueTexture1',
  'valueTexture2',
  'valueTexture3',
  'valueTexture4',
  'validityTexture1',
  'validityTexture2',
  'validityTexture3',
  'validityTexture4',
  'colormapTexture',
  'rgbaTexture',
  'valueTexture5',
  'valueTexture6',
  'valueTexture7',
  'valueTexture8'
] as const;

/** Float value textures cannot be filtered without a device feature, so they declare unfilterable-float. */
export const RASTER_WGSL_UNFILTERABLE_TEXTURE_NAMES: readonly string[] = [
  'valueTexture1',
  'valueTexture2',
  'valueTexture3',
  'valueTexture4',
  'valueTexture5',
  'valueTexture6',
  'valueTexture7',
  'valueTexture8'
];

/**
 * Builds the vertex and fragment WGSL for one band-math expression. `expression` is a WGSL f32
 * expression over `b1..b4`; the shader handles nodata (validity masks, the nodata sentinel and
 * non-finite results), the value filter, grayscale, colormap, three-band RGB and plain RGBA tiles.
 */
export function buildRasterTileWgsl(expression = 'b1'): string {
  // Extended slots are only loaded when the expression reads them.
  const extendedBandLoads = [5, 6, 7, 8]
    .filter(bandIndex => new RegExp(`\\bb${bandIndex}\\b`).test(expression))
    .map(
      bandIndex => `  let b${bandIndex} = rescaleChannel(loadValue(valueTexture${bandIndex}, uv));`
    )
    .join('\n');
  return /* wgsl */ `\
struct RasterAttributes {
  @location(0) positions: vec3<f32>,
  @location(1) texCoords: vec2<f32>,
};
struct RasterVaryings {
  @builtin(position) position: vec4<f32>,
  @location(0) textureCoordinates: vec2<f32>,
};

@vertex
fn vertexMain(attributes: RasterAttributes) -> RasterVaryings {
  var output: RasterVaryings;
  let projected = project_position_to_clipspace_and_commonspace(attributes.positions, vec3<f32>(0.0), vec3<f32>(0.0));
  output.position = projected.clipPosition;
  output.textureCoordinates = attributes.texCoords;
  return output;
}

fn getPixel(dimensions: vec2<u32>, textureCoordinates: vec2<f32>) -> vec2<i32> {
  let size = vec2<f32>(dimensions);
  return vec2<i32>(clamp(floor(textureCoordinates * size), vec2<f32>(0.0), size - vec2<f32>(1.0)));
}

fn loadValue(valueTexture: texture_2d<f32>, textureCoordinates: vec2<f32>) -> f32 {
  return textureLoad(valueTexture, getPixel(textureDimensions(valueTexture), textureCoordinates), 0).r;
}

fn loadValidity(validityTexture: texture_2d<f32>, textureCoordinates: vec2<f32>) -> f32 {
  return textureLoad(validityTexture, getPixel(textureDimensions(validityTexture), textureCoordinates), 0).r;
}

fn isBandInvalid(bit: u32, validityTexture: texture_2d<f32>, textureCoordinates: vec2<f32>) -> bool {
  return (u32(rasterWgsl.bands.x) & bit) != 0u && loadValidity(validityTexture, textureCoordinates) < 0.5;
}

fn isFiniteValue(value: f32) -> bool {
  return abs(value) < 3.0e38;
}

// Kepler flags (rescale.w): 1 rescale, 2 gamma, 4 sigmoidal contrast, 8 saturation, 16 categorical.
fn hasFlag(flag: u32) -> bool {
  return (u32(rasterWgsl.rescale.w) & flag) != 0u;
}

// Kepler linear_rescale, gamma_contrast and sigmoidalContrast modules, applied per channel.
fn rescaleChannel(value: f32) -> f32 {
  if (!hasFlag(1u)) { return value; }
  var x = value * rasterWgsl.rescale.x + rasterWgsl.rescale.y;
  if (hasFlag(2u)) {
    x = pow(max(x, 0.0), 1.0 / max(rasterWgsl.tone.x, 1.0e-8));
  }
  if (hasFlag(4u)) {
    let beta = rasterWgsl.tone.y;
    let alpha = max(rasterWgsl.tone.z, 1.0e-8);
    if (beta > 0.0) {
      let low = 1.0 / (1.0 + exp(beta * alpha));
      let numerator = 1.0 / (1.0 + exp(beta * (alpha - x))) - low;
      let denominator = 1.0 / (1.0 + exp(beta * (alpha - 1.0))) - low;
      x = numerator / denominator;
    } else if (beta < 0.0) {
      let inner = x / (1.0 + exp(beta * alpha - beta)) - x / (1.0 + exp(beta * alpha)) + 1.0 / (1.0 + exp(beta * alpha));
      x = (beta * alpha - log(1.0 / inner - 1.0)) / beta;
    }
  }
  return x;
}

// Kepler saturation module (OpenGL Shading Language chapter 16 weights).
fn saturateColor(rgb: vec3<f32>, adjustment: f32) -> vec3<f32> {
  let intensity = dot(rgb, vec3<f32>(0.2125, 0.7154, 0.0721));
  return mix(vec3<f32>(intensity), rgb, adjustment);
}

fn loadColormapTexel(index: i32, width: i32) -> vec4<f32> {
  return textureLoad(colormapTexture, vec2<i32>(clamp(index, 0, width - 1), 0), 0);
}

// Kepler colormapApply: categorical strips are sampled nearest at value * step, every other
// colormap linearly (clamp to edge) at 0.5 * value + 0.5, i.e. the upper half for 0..1 input.
fn sampleColormap(value: f32) -> vec4<f32> {
  let width = i32(max(rasterWgsl.settings.y, 1.0));
  let widthFloat = f32(width);
  if (hasFlag(16u)) {
    let position = clamp(value * rasterWgsl.tone.w * widthFloat, 0.0, widthFloat - 1.0);
    return loadColormapTexel(i32(floor(position)), width);
  }
  let position = clamp((0.5 * value + 0.5) * widthFloat - 0.5, 0.0, widthFloat - 1.0);
  let lowIndex = i32(floor(position));
  return mix(loadColormapTexel(lowIndex, width), loadColormapTexel(lowIndex + 1, width), position - floor(position));
}

@fragment
fn fragmentMain(varyings: RasterVaryings) -> @location(0) vec4<f32> {
  let mode = i32(rasterWgsl.settings.x);
  let uv = varyings.textureCoordinates;
  if (mode == 3) {
    let rgba = textureLoad(rgbaTexture, getPixel(textureDimensions(rgbaTexture), uv), 0);
    if (rgba.a <= 0.0) { discard; }
    return deckgl_premultiplied_alpha(vec4<f32>(rgba.rgb, rgba.a * layer.opacity));
  }
  if (isBandInvalid(1u, validityTexture1, uv) || isBandInvalid(2u, validityTexture2, uv) ||
      isBandInvalid(4u, validityTexture3, uv) || isBandInvalid(8u, validityTexture4, uv)) {
    discard;
  }
  let rawBand1 = loadValue(valueTexture1, uv);
  let hasNodata = rasterWgsl.settings.z > 0.5;
  let nodata = rasterWgsl.settings.w;
  if (hasNodata && (rawBand1 == nodata)) { discard; }
  var b1 = rescaleChannel(rawBand1);
  var b2 = rescaleChannel(loadValue(valueTexture2, uv));
  var b3 = rescaleChannel(loadValue(valueTexture3, uv));
  let b4 = rescaleChannel(loadValue(valueTexture4, uv));
${extendedBandLoads}
  if (hasFlag(8u)) {
    let saturated = saturateColor(vec3<f32>(b1, b2, b3), rasterWgsl.rescale.z);
    b1 = saturated.x;
    b2 = saturated.y;
    b3 = saturated.z;
  }
  var color = vec4<f32>(1.0);
  let alpha = 1.0;
  if (mode == 2) {
    if (!isFiniteValue(b1) || !isFiniteValue(b2) || !isFiniteValue(b3)) { discard; }
    color = vec4<f32>(clamp(vec3<f32>(b1, b2, b3), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  } else {
    let result = ${expression};
    if (!isFiniteValue(result)) { discard; }
    if (rasterWgsl.bands.y > 0.5 && (result < rasterWgsl.domain.z || result > rasterWgsl.domain.w)) { discard; }
    if (mode == 1) {
      color = sampleColormap(result);
      if (color.a <= 0.0) { discard; }
    } else if (mode == 4) {
      // Kepler spectral index modules emit vec4(index, 0, 0, 0): the red channel, framebuffer clamped.
      color = vec4<f32>(clamp(result, 0.0, 1.0), 0.0, 0.0, 1.0);
    } else {
      let t = clamp(result, 0.0, 1.0);
      color = vec4<f32>(t, t, t, 1.0);
    }
  }
  return deckgl_premultiplied_alpha(vec4<f32>(color.rgb, alpha * layer.opacity));
}
`;
}

export type RasterTileMesh = Readonly<{
  positions: Float32Array;
  texCoords: Float32Array;
  vertexCount: number;
}>;

function getMercatorY(latitude: number): number {
  const clamped = clamp(latitude, -85.0511287798, 85.0511287798);
  return Math.log(Math.tan(Math.PI / 4 + (clamped * Math.PI) / 360));
}

/**
 * Triangle-list quad over `[west, south, east, north]` degrees. Rows are split so the texture
 * coordinate follows Web Mercator (tile pixels are linear in Mercator y, not in latitude).
 */
export function createRasterTileMesh(
  bounds: readonly [number, number, number, number],
  rowCount = 16
): RasterTileMesh {
  const [west, south, east, north] = bounds;
  const northY = getMercatorY(north);
  const southY = getMercatorY(south);
  const rows = Math.max(1, rowCount);
  const vertexCount = rows * 6;
  const positions = new Float32Array(vertexCount * 3);
  const texCoords = new Float32Array(vertexCount * 2);
  let output = 0;
  const emit = (longitude: number, latitude: number, u: number): void => {
    const v = northY === southY ? 0 : (northY - getMercatorY(latitude)) / (northY - southY);
    positions.set([longitude, latitude, 0], output * 3);
    texCoords.set([u, v], output * 2);
    output++;
  };
  for (let row = 0; row < rows; row++) {
    const upper = north - ((north - south) * row) / rows;
    const lower = north - ((north - south) * (row + 1)) / rows;
    emit(west, upper, 0);
    emit(west, lower, 0);
    emit(east, lower, 1);
    emit(west, upper, 0);
    emit(east, lower, 1);
    emit(east, upper, 1);
  }
  return {positions, texCoords, vertexCount};
}
