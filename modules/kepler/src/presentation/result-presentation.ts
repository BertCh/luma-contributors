// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  classifyGPUAnalysisResultValues,
  normalizeGPUAnalysisClassValue,
  getGPUAnalysisColumnPrecision,
  formatGPUAnalysisMeasuredValue,
  formatGPUAnalysisNumber,
  getGPUAnalysisResultClasses,
  validateGPUAnalysisResultEncoding
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  GPUAnalysisClassEncoding,
  GPUAnalysisClassScheme,
  GPUAnalysisResultClass
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogRunResult} from '../analysis/catalog-binding-types';
import type {CatalogOutputTable} from '../analysis/catalog-geojson';
import type {
  KeplerOperationDescriptor,
  KeplerOperationOutputDescriptor
} from '../descriptors/operation-descriptor-types';

// Turns one catalog output table and the descriptor of that output into what a host needs to show
// it: which scale and palette to use, the legend with a count per class, class breaks for numeric
// columns and the notes that say when the plan departs from the descriptor. Pure CPU code. The
// plan never throws on data: a missing column gives no plan, and a column that does not match the
// descriptor gives a categorical plan with a note.

// ---------------------------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------------------------

/** Red, green and blue from 0 to 255. */
export type KeplerResultColor = readonly [number, number, number];

type ResultTone = GPUAnalysisResultClass['tone'];

/**
 * Backgrounds the strongest colours are checked against. White is a light basemap; the dark value
 * is a typical dark basemap land colour.
 */
export const KEPLER_LIGHT_MAP_BACKGROUND: KeplerResultColor = [255, 255, 255];
export const KEPLER_DARK_MAP_BACKGROUND: KeplerResultColor = [20, 20, 24];

/**
 * Smallest contrast ratio of the strongest classes against either background: 3:1 is the WCAG 2.1
 * success criterion 1.4.11 minimum for graphical objects. Only colours with a relative luminance
 * from about 0.13 to 0.30 pass against both backgrounds at once.
 */
export const KEPLER_MINIMUM_MAP_CONTRAST_RATIO = 3;

/**
 * Smallest difference between any two entries that can stand in one legend (all pairs, not only
 * neighbours: noise, other, no value and not tested included), as seen with normal vision and
 * under simulated protanopia, deuteranopia and tritanopia. Two measures are checked because
 * neither is enough alone.
 *
 * CIE76 (Euclidean distance in L*a*b*) is simple and has no hidden parameters; 15 is well above
 * its just-noticeable difference of about 2.3, which allows for small map swatches. It exaggerates
 * differences between saturated colours, so CIEDE2000, which corrects for that and for the
 * lightness of the pair, must also reach 9 (its just-noticeable difference is about 1, and 9 is
 * the lowest value the palette below reaches between greys and hues under any deficiency).
 */
export const KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE = 15;
export const KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000 = 9;

/**
 * Smallest difference between neighbouring stops of an ordered ramp (and between a stop and the
 * no-value colour). Stops are meant to look like steps of one gradient, so the bar is lower than
 * for separate classes: CIE76 of 8 (about three just-noticeable differences) and CIEDE2000 of 6.5.
 */
export const KEPLER_MINIMUM_RAMP_COLOR_DISTANCE = 8;
export const KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000 = 6.5;

/**
 * Smallest contrast against a white map of the lowest class of a ramp and of the weak tones. Below
 * about 1.5 a fill is lost on a white basemap; 1.6 keeps the palest class visible as a tint. This
 * is below the 3:1 of {@link KEPLER_MINIMUM_MAP_CONTRAST_RATIO}, so the fill of such a class alone
 * does not meet it: a plan therefore recommends an outline (`outline`), which does.
 */
export const KEPLER_MINIMUM_LOWEST_CLASS_CONTRAST_RATIO = 1.6;

/**
 * Outline colour that keeps 3:1 against both the white and the dark map (4.5:1 and 4.0:1), for the
 * fills that cannot on their own: draw it about one pixel wide round every class.
 */
export const KEPLER_RECOMMENDED_OUTLINE_COLOR: KeplerResultColor = [118, 118, 118];

/**
 * Colour of each tone. Hot spots run red, cold spots blue, with light tints for the weaker
 * confidence levels and a neutral grey for not significant. The two outlier tones have hues of
 * their own (purple and green) so that they cannot be read as a weaker hot or cold spot. The
 * strongest tones (`strong-high`, `strong-low`, both outliers) lie inside the luminance window that
 * keeps a 3:1 contrast against a white and a dark basemap; the weak tones keep 1.6:1 or more
 * against white. The greys (neutral, excluded, other, no value) differ in lightness by 15 CIE76
 * or more, because a grey stays a grey under every colour-vision deficiency.
 */
const TONE_COLORS: Readonly<Record<ResultTone, KeplerResultColor>> = {
  'strong-high': [200, 35, 45],
  high: [240, 125, 90],
  'weak-high': [250, 175, 140],
  neutral: [190, 190, 190],
  'weak-low': [150, 195, 225],
  low: [90, 150, 205],
  'strong-low': [30, 105, 185],
  'high-outlier': [150, 60, 175],
  'low-outlier': [0, 125, 95],
  category: [86, 180, 233],
  excluded: [70, 70, 70]
};

/** One colour per tone of a class scheme. */
export function getKeplerResultToneColor(tone: ResultTone): KeplerResultColor {
  const color = TONE_COLORS[tone];
  if (!color) {
    throw new Error(
      `getKeplerResultToneColor: argument "tone" must be one of ${Object.keys(TONE_COLORS).join(', ')}; received ${String(tone)}.`
    );
  }
  return color;
}

/**
 * Sequential ramp from low (pale teal, 1.6:1 against white) to high (saturated teal, 3.1:1 against
 * the dark map). Five stops, the most classes a plan uses. The hue differs from the red and blue of
 * the hot and cold tones.
 */
export const KEPLER_SEQUENTIAL_RAMP: readonly KeplerResultColor[] = [
  [166, 214, 198],
  [116, 190, 170],
  [66, 162, 142],
  [26, 134, 122],
  [0, 112, 106]
];

/** Diverging ramp from the low end (blue) through the neutral grey to the high end (red). */
export const KEPLER_DIVERGING_RAMP: readonly KeplerResultColor[] = [
  TONE_COLORS['strong-low'],
  TONE_COLORS.low,
  TONE_COLORS['weak-low'],
  TONE_COLORS.neutral,
  TONE_COLORS['weak-high'],
  TONE_COLORS.high,
  TONE_COLORS['strong-high']
];

/**
 * Categorical palette: the Okabe and Ito (2008) colour-blind-safe set without black, which would
 * vanish on a dark basemap. Every pair differs by at least 15 (CIE76) under both simulations.
 */
export const KEPLER_CATEGORICAL_PALETTE: readonly KeplerResultColor[] = [
  [0, 114, 178],
  [213, 94, 0],
  [0, 158, 115],
  [204, 121, 167],
  [86, 180, 233],
  [230, 159, 0],
  [240, 228, 66]
];

/** Colour of the bucket that collects the categories beyond the limit. */
export const KEPLER_OTHER_CATEGORY_COLOR: KeplerResultColor = [200, 200, 200];

/**
 * Colour of rows that have no value: lighter than every class fill, so that missing data is the
 * least prominent entry of a legend (1.08:1 against white, against 1.6:1 or more for the palest
 * class). The legend row is the label; the map should hatch or outline these rows.
 */
export const KEPLER_MISSING_VALUE_COLOR: KeplerResultColor = [246, 246, 246];

/** Linear interpolation of a ramp at `position` from 0 to 1. */
function sampleRamp(ramp: readonly KeplerResultColor[], position: number): KeplerResultColor {
  const clamped = Math.min(1, Math.max(0, position));
  const scaled = clamped * (ramp.length - 1);
  const lowerIndex = Math.min(ramp.length - 2, Math.floor(scaled));
  const fraction = scaled - lowerIndex;
  const lower = ramp[lowerIndex];
  const upper = ramp[lowerIndex + 1];
  return [
    Math.round(lower[0] + (upper[0] - lower[0]) * fraction),
    Math.round(lower[1] + (upper[1] - lower[1]) * fraction),
    Math.round(lower[2] + (upper[2] - lower[2]) * fraction)
  ];
}

/**
 * `count` colours spread evenly over a ramp, lowest first; one class gets the middle colour.
 * @throws when `count` is not a positive integer.
 */
export function getKeplerResultRampColors(
  palette: 'sequential' | 'diverging',
  count: number
): readonly KeplerResultColor[] {
  if (!Number.isInteger(count) || count < 1 || count > 64) {
    throw new Error(
      `getKeplerResultRampColors: argument "count" must be an integer from 1 to 64; received ${count}.`
    );
  }
  const ramp = palette === 'sequential' ? KEPLER_SEQUENTIAL_RAMP : KEPLER_DIVERGING_RAMP;
  if (count === 1) return [sampleRamp(ramp, 0.5)];
  return Array.from({length: count}, (_, index) => sampleRamp(ramp, index / (count - 1)));
}

// Colour-vision-deficiency simulation: the full-severity matrices of Machado, Oliveira and
// Fernandes, "A Physiologically-based Model for Simulation of Color Vision Deficiency", IEEE
// Transactions on Visualization and Computer Graphics 15(6), 2009, applied to linear sRGB.
const PROTANOPIA_MATRIX: readonly (readonly number[])[] = [
  [0.152286, 1.052583, -0.204868],
  [0.114503, 0.786281, 0.099216],
  [-0.003882, -0.048116, 1.051998]
];
const DEUTERANOPIA_MATRIX: readonly (readonly number[])[] = [
  [0.367322, 0.860646, -0.227968],
  [0.280085, 0.672501, 0.047413],
  [-0.01182, 0.04294, 0.968881]
];

// Tritanopia, same source.
const TRITANOPIA_MATRIX: readonly (readonly number[])[] = [
  [1.255528, -0.076749, -0.178779],
  [-0.078411, 0.930809, 0.147602],
  [0.004733, 0.691367, 0.3039]
];

export type KeplerColorVision = 'normal' | 'protanopia' | 'deuteranopia' | 'tritanopia';

function decodeSrgbChannel(value: number): number {
  const unit = value / 255;
  return unit <= 0.04045 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
}

function encodeSrgbChannel(value: number): number {
  const clamped = Math.min(1, Math.max(0, value));
  const unit = clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * clamped ** (1 / 2.4) - 0.055;
  return Math.round(unit * 255);
}

/** The colour as it looks with the given vision; `normal` returns the colour unchanged. */
export function simulateKeplerColorVision(
  color: KeplerResultColor,
  vision: KeplerColorVision
): KeplerResultColor {
  if (vision === 'normal') return color;
  const matrix =
    vision === 'protanopia'
      ? PROTANOPIA_MATRIX
      : vision === 'deuteranopia'
        ? DEUTERANOPIA_MATRIX
        : TRITANOPIA_MATRIX;
  const linear = color.map(decodeSrgbChannel);
  const mixed = matrix.map(row => row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2]);
  return [encodeSrgbChannel(mixed[0]), encodeSrgbChannel(mixed[1]), encodeSrgbChannel(mixed[2])];
}

function getLinearLuminance(color: KeplerResultColor): number {
  return (
    0.2126729 * decodeSrgbChannel(color[0]) +
    0.7151522 * decodeSrgbChannel(color[1]) +
    0.072175 * decodeSrgbChannel(color[2])
  );
}

/** WCAG contrast ratio of two colours, from 1 to 21. */
export function getKeplerColorContrastRatio(
  first: KeplerResultColor,
  second: KeplerResultColor
): number {
  const firstLuminance = getLinearLuminance(first);
  const secondLuminance = getLinearLuminance(second);
  const lighter = Math.max(firstLuminance, secondLuminance);
  const darker = Math.min(firstLuminance, secondLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

function convertToLab(color: KeplerResultColor): readonly [number, number, number] {
  const red = decodeSrgbChannel(color[0]);
  const green = decodeSrgbChannel(color[1]);
  const blue = decodeSrgbChannel(color[2]);
  // sRGB to CIE XYZ (D65), divided by the D65 white point.
  const x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047;
  const y = 0.2126729 * red + 0.7151522 * green + 0.072175 * blue;
  const z = (0.0193339 * red + 0.119192 * green + 0.9503041 * blue) / 1.08883;
  const transform = (value: number): number =>
    value > 216 / 24389 ? Math.cbrt(value) : ((24389 / 27) * value + 16) / 116;
  const fx = transform(x);
  const fy = transform(y);
  const fz = transform(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** CIE76 colour difference (Euclidean distance in CIE L*a*b*) as seen with the given vision. */
export function getKeplerColorDistance(
  first: KeplerResultColor,
  second: KeplerResultColor,
  vision: KeplerColorVision = 'normal'
): number {
  const firstLab = convertToLab(simulateKeplerColorVision(first, vision));
  const secondLab = convertToLab(simulateKeplerColorVision(second, vision));
  return Math.hypot(
    firstLab[0] - secondLab[0],
    firstLab[1] - secondLab[1],
    firstLab[2] - secondLab[2]
  );
}

/**
 * CIEDE2000 colour difference (Sharma, Wu and Dalal 2005 formulation) as seen with the given
 * vision. It weights lightness, chroma and hue differences by where the pair lies in the colour
 * space, which CIE76 does not.
 */
export function getKeplerColorDifference2000(
  first: KeplerResultColor,
  second: KeplerResultColor,
  vision: KeplerColorVision = 'normal'
): number {
  const [firstL, firstA, firstB] = convertToLab(simulateKeplerColorVision(first, vision));
  const [secondL, secondA, secondB] = convertToLab(simulateKeplerColorVision(second, vision));
  const radians = (degrees: number): number => (degrees * Math.PI) / 180;
  const degrees = (angle: number): number => (angle * 180) / Math.PI;
  const firstChroma = Math.hypot(firstA, firstB);
  const secondChroma = Math.hypot(secondA, secondB);
  const meanChroma = (firstChroma + secondChroma) / 2;
  const chromaPower = meanChroma ** 7;
  const adjustment = 0.5 * (1 - Math.sqrt(chromaPower / (chromaPower + 25 ** 7)));
  const firstAPrime = (1 + adjustment) * firstA;
  const secondAPrime = (1 + adjustment) * secondA;
  const firstChromaPrime = Math.hypot(firstAPrime, firstB);
  const secondChromaPrime = Math.hypot(secondAPrime, secondB);
  const firstHue = (degrees(Math.atan2(firstB, firstAPrime)) + 360) % 360;
  const secondHue = (degrees(Math.atan2(secondB, secondAPrime)) + 360) % 360;
  const deltaL = secondL - firstL;
  const deltaChroma = secondChromaPrime - firstChromaPrime;
  const chromaProduct = firstChromaPrime * secondChromaPrime;
  let deltaHue = 0;
  if (chromaProduct !== 0) {
    deltaHue = secondHue - firstHue;
    if (deltaHue > 180) deltaHue -= 360;
    else if (deltaHue < -180) deltaHue += 360;
  }
  const deltaH = 2 * Math.sqrt(chromaProduct) * Math.sin(radians(deltaHue / 2));
  const meanL = (firstL + secondL) / 2;
  const meanChromaPrime = (firstChromaPrime + secondChromaPrime) / 2;
  let meanHue = firstHue + secondHue;
  if (chromaProduct !== 0) {
    meanHue = (firstHue + secondHue) / 2;
    if (Math.abs(firstHue - secondHue) > 180) meanHue += firstHue + secondHue < 360 ? 180 : -180;
  }
  const hueWeight =
    1 -
    0.17 * Math.cos(radians(meanHue - 30)) +
    0.24 * Math.cos(radians(2 * meanHue)) +
    0.32 * Math.cos(radians(3 * meanHue + 6)) -
    0.2 * Math.cos(radians(4 * meanHue - 63));
  const rotation = 30 * Math.exp(-(((meanHue - 275) / 25) ** 2));
  const chromaPrimePower = meanChromaPrime ** 7;
  const rotationChroma = 2 * Math.sqrt(chromaPrimePower / (chromaPrimePower + 25 ** 7));
  const lightnessScale = 1 + (0.015 * (meanL - 50) ** 2) / Math.sqrt(20 + (meanL - 50) ** 2);
  const chromaScale = 1 + 0.045 * meanChromaPrime;
  const hueScale = 1 + 0.015 * meanChromaPrime * hueWeight;
  const rotationTerm = -Math.sin(radians(2 * rotation)) * rotationChroma;
  const lightnessTerm = deltaL / lightnessScale;
  const chromaTerm = deltaChroma / chromaScale;
  const hueTerm = deltaH / hueScale;
  return Math.sqrt(
    lightnessTerm ** 2 + chromaTerm ** 2 + hueTerm ** 2 + rotationTerm * chromaTerm * hueTerm
  );
}

// ---------------------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------------------

/**
 * Classes of a numeric column. Five: the largest number of classes most readers can still tell
 * apart on a map (the cartographic convention, with seven the upper end); a plan uses fewer when
 * the column has fewer distinct values or many ties.
 */
export const KEPLER_DEFAULT_CLASS_COUNT = 5;

const CLASS_COUNT_REASON =
  'Five classes, the most that readers can reliably tell apart on a map; fewer when the column has fewer distinct values or many ties.';

/**
 * Most categories shown with a colour of their own; the rest go in one `other` bucket. The limit is
 * the size of the colour-blind-safe palette, since more hues than that cannot be told apart.
 */
export const KEPLER_MAXIMUM_CATEGORY_COUNT = KEPLER_CATEGORICAL_PALETTE.length;

/**
 * Share of the non-null values that must be classes of the declared scheme under the declared
 * encoding: all of them. A value outside the scheme is the sign of another producer's code or
 * another concept (a 91 percent column of valid classes with 9 percent of a different scheme's
 * labels passed an earlier 0.9 bar), so any such value makes the plan show categories and name
 * the values. A column that mixes codes of two schemes that share integers cannot be told apart
 * here; the descriptor must name the right scheme (`local-geary-classes` for Geary).
 */
export const KEPLER_MINIMUM_ENCODING_SHARE = 1;

/** Most unrecognised values a note names. */
const MAXIMUM_NAMED_UNRECOGNIZED_VALUES = 5;

/** One legend entry of a class scheme or a categorical column. */
export type KeplerResultLegendEntry = {
  value: number | string | boolean;
  label: string;
  tone: ResultTone;
  /** Rows of the table in this class. */
  count: number;
  color: KeplerResultColor;
};

/** One class of a numeric column; a value belongs to the first class whose `to` it does not exceed. */
export type KeplerResultClassRange = {
  from: number;
  to: number;
  label: string;
  count: number;
  color: KeplerResultColor;
};

/** What a cluster-membership column says about its clusters, from the whole column. */
export type KeplerResultMembership = {
  clusterCount: number;
  noiseCount: number;
  /** Rows in the largest cluster; 0 when there is none. */
  largestClusterCount: number;
};

export type KeplerResultPresentation = {
  column: string;
  /** How values map to colour; class schemes and categorical columns are `categorical`. */
  scale: 'sequential' | 'diverging' | 'categorical';
  /** Colour family: `diverging` for the hot spot scheme, `categorical` for the other schemes. */
  palette: 'sequential' | 'diverging' | 'categorical';
  /** Present when the plan uses the descriptor's class scheme. */
  classScheme?: GPUAnalysisClassScheme;
  classEncoding?: GPUAnalysisClassEncoding;
  /** Share of the non-null values that are classes of the scheme; present with `classScheme`. */
  encodingShare?: number;
  rowCount: number;
  /** Rows with no usable value: null, undefined, NaN, infinite, or text in a numeric column. */
  missingCount: number;
  /** Class scheme classes or categories, in legend order, each with its row count. */
  legend: readonly KeplerResultLegendEntry[];
  /** Rows that are not a class of the scheme (class schemes only). */
  unrecognized?: number;
  /** Categories beyond the limit, folded into the last legend entry (categorical only). */
  other?: {distinctCount: number; count: number};
  membership?: KeplerResultMembership;
  /** Finite minimum and maximum; symmetric about `center` for a diverging scale. */
  domain?: readonly [number, number];
  center?: number;
  /** One precision chosen for the displayed column bounds, also used by the range sentence. */
  significantDigits?: number | 'round-trip';
  /** Rows excluded from inference; hosts must use the excluded fill for these rows. */
  excludedRows?: readonly number[];
  /** Upper bounds of every class but the last: strictly increasing, each a value of the column. */
  breaks: readonly number[];
  classCount?: number;
  classCountReason?: string;
  ranges: readonly KeplerResultClassRange[];
  /**
   * Outline to draw round the fills. `required` is true when a fill alone is below 3:1 against a
   * white or a dark map (the palest classes are); the fill alone then does not meet the contrast.
   */
  outline?: {
    color: KeplerResultColor;
    required: boolean;
    lowestFillContrast: {light: number; dark: number};
    reason: string;
  };
  /** Where the plan departs from the descriptor, or says why a part is empty. */
  notes: readonly string[];
};

function hasColumn(table: CatalogOutputTable, column: string): boolean {
  return Object.prototype.hasOwnProperty.call(table.columns, column);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

type ColumnValues = readonly (string | number | boolean | null)[];

function getPalette(scheme: GPUAnalysisClassScheme): 'diverging' | 'categorical' {
  return scheme === 'hot-spot-confidence-bins' ? 'diverging' : 'categorical';
}

function compareCategoryValues(
  first: string | number | boolean,
  second: string | number | boolean
): number {
  const rank = (value: string | number | boolean): number =>
    typeof value === 'number' ? 0 : typeof value === 'string' ? 1 : 2;
  if (rank(first) !== rank(second)) return rank(first) - rank(second);
  if (first < second) return -1;
  return first > second ? 1 : 0;
}

function makeCategoricalPlan(
  column: string,
  values: ColumnValues,
  notes: string[]
): KeplerResultPresentation {
  const counts = new Map<string | number | boolean, number>();
  let missingCount = 0;
  for (const value of values) {
    if (
      value === null ||
      value === undefined ||
      (typeof value === 'number' && !Number.isFinite(value))
    ) {
      missingCount++;
    } else {
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
  }
  const ordered = [...counts.entries()].sort(
    (first, second) => second[1] - first[1] || compareCategoryValues(first[0], second[0])
  );
  const kept = ordered.slice(0, KEPLER_MAXIMUM_CATEGORY_COUNT);
  const rest = ordered.slice(KEPLER_MAXIMUM_CATEGORY_COUNT);
  const legend: KeplerResultLegendEntry[] = kept.map(([value, count], index) => ({
    value,
    label: String(value),
    tone: 'category',
    count,
    color: KEPLER_CATEGORICAL_PALETTE[index]
  }));
  let other: KeplerResultPresentation['other'];
  if (rest.length) {
    const otherCount = rest.reduce((sum, [, count]) => sum + count, 0);
    other = {distinctCount: rest.length, count: otherCount};
    legend.push({
      value: 'other',
      label: `Other (${formatGPUAnalysisNumber(rest.length)} more values)`,
      tone: 'category',
      count: otherCount,
      color: KEPLER_OTHER_CATEGORY_COLOR
    });
    notes.push(
      `Column "${column}" has ${ordered.length} distinct values; the ${KEPLER_MAXIMUM_CATEGORY_COUNT} most frequent have a colour and the other ${rest.length} share one.`
    );
  }
  return {
    column,
    scale: 'categorical',
    palette: 'categorical',
    rowCount: values.length,
    missingCount,
    legend,
    ...(other ? {other} : {}),
    breaks: [],
    ranges: [],
    notes
  };
}

/** The distinct values of a column that are not classes of the scheme, with their row counts, capped. */
function describeUnrecognizedValues(
  scheme: GPUAnalysisClassScheme,
  encoding: GPUAnalysisClassEncoding,
  values: ColumnValues
): string {
  const counts = new Map<string | number | boolean, number>();
  for (const value of values) {
    if (value === null || value === undefined) continue;
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  const unrecognized = [...counts.entries()]
    .filter(
      ([value]) => classifyGPUAnalysisResultValues(scheme, [value], encoding).unrecognized > 0
    )
    .sort((first, second) => second[1] - first[1]);
  const named = unrecognized
    .slice(0, MAXIMUM_NAMED_UNRECOGNIZED_VALUES)
    .map(
      ([value, count]) =>
        `${typeof value === 'string' ? `"${value}"` : String(value)} (${formatGPUAnalysisNumber(count)} ${count === 1 ? 'row' : 'rows'})`
    );
  const rest = unrecognized.length - named.length;
  return `${named.join(', ')}${rest > 0 ? ` and ${rest} more` : ''}`;
}

function makeClassSchemePlan(
  column: string,
  values: ColumnValues,
  scheme: GPUAnalysisClassScheme,
  encoding: GPUAnalysisClassEncoding,
  notes: string[]
): KeplerResultPresentation | undefined {
  let classes: readonly GPUAnalysisResultClass[];
  try {
    classes = getGPUAnalysisResultClasses(scheme, encoding);
  } catch {
    notes.push(
      `The descriptor names scheme "${scheme}" with encoding "${encoding}", which the analysis package does not have; the column is shown as categories.`
    );
    return undefined;
  }
  const tally = classifyGPUAnalysisResultValues(scheme, values, encoding);
  const encodingShare = validateGPUAnalysisResultEncoding(scheme, encoding, values);
  const nonNullCount = values.filter(value => value !== null && value !== undefined).length;
  if (nonNullCount > 0 && encodingShare < KEPLER_MINIMUM_ENCODING_SHARE) {
    notes.push(
      `Column "${column}" holds values that are not classes of scheme "${scheme}" with encoding "${encoding}" (${formatGPUAnalysisNumber(encodingShare * 100, 1, true)}% are): ${describeUnrecognizedValues(scheme, encoding, values)}; the column is shown as categories.`
    );
    return undefined;
  }

  const legend: KeplerResultLegendEntry[] = [];
  let membership: KeplerResultMembership | undefined;
  if (scheme === 'cluster-membership') {
    const noiseClass = classes[0];
    const noiseCount = tally.counts.get(noiseClass.value) ?? 0;
    const clusters = [...tally.counts.entries()].filter(([value]) => value !== noiseClass.value);
    const bySize = [...clusters].sort(
      (first, second) => second[1] - first[1] || (first[0] as number) - (second[0] as number)
    );
    const shown = bySize
      .slice(0, KEPLER_MAXIMUM_CATEGORY_COUNT)
      .sort((first, second) => (first[0] as number) - (second[0] as number));
    const rest = bySize.slice(KEPLER_MAXIMUM_CATEGORY_COUNT);
    legend.push({
      value: noiseClass.value,
      label: noiseClass.label,
      tone: noiseClass.tone,
      count: noiseCount,
      color: getKeplerResultToneColor(noiseClass.tone)
    });
    shown.forEach(([value, count], index) =>
      legend.push({
        value,
        label: `Cluster ${value}`,
        tone: 'category',
        count,
        color: KEPLER_CATEGORICAL_PALETTE[index]
      })
    );
    if (rest.length) {
      legend.push({
        value: 'other-clusters',
        label: `${formatGPUAnalysisNumber(rest.length)} smaller clusters`,
        tone: 'category',
        count: rest.reduce((sum, [, count]) => sum + count, 0),
        color: KEPLER_OTHER_CATEGORY_COLOR
      });
      notes.push(
        `${clusters.length} clusters found; the ${KEPLER_MAXIMUM_CATEGORY_COUNT} largest have a colour and the other ${rest.length} share one.`
      );
    }
    membership = {
      clusterCount: clusters.length,
      noiseCount,
      largestClusterCount: bySize.length ? bySize[0][1] : 0
    };
  } else {
    for (const resultClass of classes) {
      legend.push({
        value: resultClass.value,
        label: resultClass.label,
        tone: resultClass.tone,
        count: tally.counts.get(resultClass.value) ?? 0,
        color: getKeplerResultToneColor(resultClass.tone)
      });
    }
  }
  if (tally.unrecognized > 0) {
    notes.push(
      `${formatGPUAnalysisNumber(tally.unrecognized)} of ${formatGPUAnalysisNumber(values.length)} rows have no value or a value outside scheme "${scheme}"; they are not in the legend.`
    );
  }
  return {
    column,
    scale: 'categorical',
    palette: getPalette(scheme),
    classScheme: scheme,
    classEncoding: encoding,
    encodingShare,
    rowCount: values.length,
    missingCount: tally.unrecognized,
    legend,
    unrecognized: tally.unrecognized,
    ...(membership ? {membership} : {}),
    breaks: [],
    ranges: [],
    notes
  };
}

/** Every supported vision must distinguish adjacent map classes. */
const COLOR_VISIONS: readonly KeplerColorVision[] = [
  'normal',
  'protanopia',
  'deuteranopia',
  'tritanopia'
];

/**
 * Upper bounds of the classes of sorted finite values. With no more distinct values than classes
 * every value is a class of its own, so values such as 1 and 2 are never merged. Otherwise bound
 * `k` is the value at the `k / classCount` nearest rank; bounds that repeat, or reach the maximum,
 * are dropped, so the bounds are strictly increasing, every class holds at least one value and ties
 * never split. When ties leave fewer classes than wanted, the values above the last bound are
 * divided again, so no run of distinct values is left in the top class only by accident.
 */
function getClassBreaks(
  sorted: Float64Array,
  distinct: readonly number[],
  classCount: number
): number[] {
  if (distinct.length <= classCount) return distinct.slice(0, -1);
  const maximum = sorted[sorted.length - 1];
  const breaks: number[] = [];
  for (let index = 1; index < classCount; index++) {
    const rank = Math.ceil((index * sorted.length) / classCount);
    const bound = sorted[Math.max(0, rank - 1)];
    if (bound < maximum && (breaks.length === 0 || bound > breaks[breaks.length - 1])) {
      breaks.push(bound);
    }
  }
  while (breaks.length < classCount - 1) {
    const last = breaks.length ? breaks[breaks.length - 1] : -Infinity;
    let start = 0;
    while (start < sorted.length && sorted[start] <= last) start++;
    const below = distinct.filter(value => value > last && value < maximum);
    if (!below.length) break;
    const classesLeft = classCount - breaks.length;
    let bound = sorted[start + Math.ceil((sorted.length - start) / classesLeft) - 1];
    if (bound >= maximum) bound = below[below.length - 1];
    breaks.push(bound);
  }
  return breaks;
}

function makeNumericPlan(
  column: string,
  values: ColumnValues,
  scale: 'sequential' | 'diverging',
  center: number,
  notes: string[]
): KeplerResultPresentation {
  const finite = Float64Array.from(values.filter(isFiniteNumber)).sort();
  const missingCount = values.length - finite.length;
  if (
    scale === 'diverging' &&
    finite.length &&
    (finite[0] >= center || finite[finite.length - 1] <= center)
  ) {
    notes.push(
      `Column "${column}" lies on one side of ${formatGPUAnalysisMeasuredValue(center)}, so a sequential scale separates its values.`
    );
    return makeNumericPlan(column, values, 'sequential', center, notes);
  }
  const base = {
    column,
    scale,
    palette: scale,
    rowCount: values.length,
    missingCount,
    legend: [] as KeplerResultLegendEntry[],
    ...(scale === 'diverging' ? {center} : {})
  };
  if (!finite.length) {
    notes.push(
      values.length
        ? `Column "${column}" has no finite value, so there is no scale to draw.`
        : 'The table has no rows, so there is no scale to draw.'
    );
    return {...base, breaks: [], ranges: [], notes};
  }
  const minimum = finite[0];
  const maximum = finite[finite.length - 1];
  const reach = Math.max(Math.abs(maximum - center), Math.abs(center - minimum));
  const domain: readonly [number, number] =
    scale === 'diverging' ? [center - reach, center + reach] : [minimum, maximum];

  const distinct: number[] = [finite[0]];
  for (let index = 1; index < finite.length; index++) {
    if (finite[index] !== finite[index - 1]) distinct.push(finite[index]);
  }
  const ownClasses = distinct.length <= KEPLER_DEFAULT_CLASS_COUNT;
  const breaks = getClassBreaks(finite, distinct, KEPLER_DEFAULT_CLASS_COUNT);
  const colorForRange = (from: number, to: number): KeplerResultColor => {
    const midpoint = from / 2 + to / 2;
    const position = reach === 0 ? 0.5 : 0.5 + (midpoint - center) / reach / 2;
    return sampleRamp(KEPLER_DIVERGING_RAMP, position);
  };
  if (scale === 'diverging') {
    const initialCount = breaks.length + 1;
    // Merge ranges until every pair of surviving fills passes both perceptual limits in every vision.
    while (breaks.length) {
      const bounds = [minimum, ...breaks, maximum];
      const colors = bounds.slice(1).map((bound, index) => colorForRange(bounds[index], bound));
      const mergeIndex = colors.findIndex(
        (color, index) =>
          index > 0 &&
          colors
            .slice(0, index)
            .some(previous =>
              COLOR_VISIONS.some(
                vision =>
                  getKeplerColorDistance(previous, color, vision) <
                    KEPLER_MINIMUM_RAMP_COLOR_DISTANCE ||
                  getKeplerColorDifference2000(previous, color, vision) <
                    KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000
              )
            )
      );
      if (mergeIndex < 0) break;
      breaks.splice(mergeIndex - 1, 1);
    }
    if (breaks.length + 1 < initialCount) {
      notes.push(
        `Visually indistinguishable ranges in column "${column}" were merged into ${breaks.length + 1} classes.`
      );
    }
  }
  const classCount = breaks.length + 1;
  if (distinct.length === 1) {
    notes.push(
      `Every finite value in column "${column}" is ${formatGPUAnalysisMeasuredValue(minimum)}, so there is one class.`
    );
  } else if (ownClasses && classCount === distinct.length) {
    notes.push(
      `Column "${column}" has ${distinct.length} distinct values, so each gets its own class.`
    );
  } else if (scale === 'sequential' && classCount < KEPLER_DEFAULT_CLASS_COUNT) {
    notes.push(
      `Ties in column "${column}" leave ${classCount} classes instead of ${KEPLER_DEFAULT_CLASS_COUNT}, because a class never splits equal values.`
    );
  }

  const countsPerClass = new Array<number>(classCount).fill(0);
  let classIndex = 0;
  for (const value of finite) {
    while (classIndex < breaks.length && value > breaks[classIndex]) classIndex++;
    countsPerClass[classIndex]++;
  }
  const sequentialColors =
    scale === 'sequential' ? getKeplerResultRampColors('sequential', classCount) : [];
  const bounds = [minimum, ...breaks, maximum];
  const significantDigits = getGPUAnalysisColumnPrecision(bounds);
  const boundTexts = bounds.map(value => formatGPUAnalysisMeasuredValue(value, significantDigits));
  const ranges: KeplerResultClassRange[] = countsPerClass.map((count, index) => {
    const from = index === 0 ? minimum : breaks[index - 1];
    const to = index === breaks.length ? maximum : breaks[index];
    let color: KeplerResultColor;
    if (scale === 'sequential') {
      color = sequentialColors[index];
    } else {
      color = colorForRange(from, to);
    }
    const fromText = boundTexts[index];
    const toText = boundTexts[index + 1];
    const label =
      (ownClasses && classCount === distinct.length) || from === to
        ? toText
        : index === 0
          ? `${fromText} to ${toText}`
          : `above ${fromText} to ${toText}`;
    return {from, to, label, count, color};
  });
  return {
    ...base,
    domain,
    significantDigits,
    breaks,
    classCount,
    classCountReason: CLASS_COUNT_REASON,
    ranges,
    notes
  };
}

/**
 * Whether the fills of a plan need an outline to be seen: the lowest classes (palest fills) reach
 * only about 1.6:1 against a white map, below the 3:1 of {@link KEPLER_MINIMUM_MAP_CONTRAST_RATIO}.
 * The returned outline colour keeps 3:1 against both maps. The fill alone does not meet the
 * contrast when `required` is true.
 */
function describeOutline(plan: KeplerResultPresentation): KeplerResultPresentation['outline'] {
  const colors = [
    ...plan.legend.map(entry => entry.color),
    ...plan.ranges.map(range => range.color),
    ...(plan.missingCount > 0 ? [KEPLER_MISSING_VALUE_COLOR] : [])
  ];
  if (!colors.length) return undefined;
  const light = Math.min(
    ...colors.map(color => getKeplerColorContrastRatio(color, KEPLER_LIGHT_MAP_BACKGROUND))
  );
  const dark = Math.min(
    ...colors.map(color => getKeplerColorContrastRatio(color, KEPLER_DARK_MAP_BACKGROUND))
  );
  const required =
    light < KEPLER_MINIMUM_MAP_CONTRAST_RATIO || dark < KEPLER_MINIMUM_MAP_CONTRAST_RATIO;
  return {
    color: KEPLER_RECOMMENDED_OUTLINE_COLOR,
    required,
    lowestFillContrast: {light, dark},
    reason: required
      ? `The palest fill has ${formatGPUAnalysisNumber(light, 2)}:1 contrast against a white map and the darkest ${formatGPUAnalysisNumber(dark, 2)}:1 against a dark map, below the ${KEPLER_MINIMUM_MAP_CONTRAST_RATIO}:1 of WCAG 1.4.11; draw a thin outline in the outline colour (${KEPLER_MINIMUM_MAP_CONTRAST_RATIO}:1 or more on both maps).`
      : 'Every fill keeps 3:1 against both maps.'
  };
}

/**
 * The plan for showing one output table: scale, palette, legend with row counts, domain, class
 * breaks and notes. Returns undefined when the output has no presentation or the table has no such
 * column. A column that does not match the descriptor (a text column given an integer encoding,
 * text in a numeric scale) is not an error: the plan is categorical and a note says why. Nothing
 * here throws on data.
 */
function buildKeplerResultPresentation(
  table: CatalogOutputTable,
  output: KeplerOperationOutputDescriptor
): KeplerResultPresentation | undefined {
  const presentation = output.presentation;
  // An explicit uniform-style exception keeps the host's geometry style; no legend is invented.
  if (output.presentationException || !presentation) return undefined;
  const column = presentation.column;
  if (!hasColumn(table, column)) return undefined;
  const values = table.columns[column];
  if (!Array.isArray(values)) return undefined;
  const notes: string[] = [];

  if (presentation.scale === 'class-scheme') {
    if (presentation.classScheme) {
      const encoding = presentation.classEncoding ?? 'contributor-integer';
      const neighborCounts = table.columns['neighbor_count'];
      // The FDR adapter writes significant=0 for invalid tests as well as tested negatives.
      // Only adjusted_pvalue establishes which rows entered the testing family.
      const adjustedPValues =
        table.id === 'validation-fdr' && presentation.classScheme === 'significance-flag'
          ? table.columns['adjusted_pvalue']
          : undefined;
      const excludedRows = adjustedPValues
        ? values.flatMap((_, row) =>
            typeof adjustedPValues[row] !== 'number' || !Number.isFinite(adjustedPValues[row])
              ? [row]
              : []
          )
        : presentation.classScheme === 'hot-spot-confidence-bins' && neighborCounts
          ? values.flatMap((_, row) => (neighborCounts[row] === 0 ? [row] : []))
          : [];
      const excluded = new Set(excludedRows);
      const eligibleValues = excludedRows.length
        ? values.map((value, row) => (excluded.has(row) ? null : value))
        : values;
      const plan = makeClassSchemePlan(
        column,
        eligibleValues,
        presentation.classScheme,
        encoding,
        notes
      );
      if (plan && excludedRows.length) {
        return {
          ...plan,
          excludedRows,
          missingCount: plan.missingCount - excludedRows.length,
          unrecognized: (plan.unrecognized ?? 0) - excludedRows.length,
          legend: [
            ...plan.legend,
            {
              value: adjustedPValues ? 'excluded-untested' : 'excluded-no-neighbors',
              label: adjustedPValues ? 'Untested' : 'Excluded: no neighbours',
              tone: 'excluded',
              count: excludedRows.length,
              color: getKeplerResultToneColor('excluded')
            }
          ],
          notes: [
            ...notes.filter(note => !note.includes('they are not in the legend')),
            adjustedPValues
              ? 'Rows without finite adjusted p-values were untested; they are excluded from significance counts. Use the excluded fill for these rows.'
              : 'Rows with no neighbours are excluded from hot and cold counts; use the excluded fill for these rows.'
          ]
        };
      }
      if (plan) return plan;
    } else {
      notes.push(
        'The descriptor asks for a class scheme without naming one; the column is shown as categories.'
      );
    }
    return makeCategoricalPlan(column, values, notes);
  }
  if (presentation.scale === 'categorical') return makeCategoricalPlan(column, values, notes);

  const textCount = values.filter(
    value => typeof value === 'string' || typeof value === 'boolean'
  ).length;
  if (textCount > 0 && !values.some(isFiniteNumber)) {
    notes.push(
      `Column "${column}" holds text or true/false values, not numbers, so the ${presentation.scale} scale is replaced by categories.`
    );
    return makeCategoricalPlan(column, values, notes);
  }
  if (textCount > 0) {
    notes.push(
      `${formatGPUAnalysisNumber(textCount)} non-numeric values in column "${column}" are shown as missing.`
    );
  }
  return makeNumericPlan(column, values, presentation.scale, presentation.center ?? 0, notes);
}

export function planKeplerResultPresentation(
  table: CatalogOutputTable,
  output: KeplerOperationOutputDescriptor
): KeplerResultPresentation | undefined {
  const plan = buildKeplerResultPresentation(table, output);
  if (!plan) return undefined;
  const outline = describeOutline(plan);
  return outline ? {...plan, outline} : plan;
}

// ---------------------------------------------------------------------------------------------
// Display style and per-binding presentations
// ---------------------------------------------------------------------------------------------

/**
 * The shape the workbench's display code takes for colouring a layer by a column. Defined here so
 * that this module does not import the workbench.
 */
export type KeplerWorkbenchDisplayStyle = {
  colorColumn?: string;
  colorScale?: 'quantile' | 'quantize' | 'ordinal';
  palette?: 'sequential' | 'diverging' | 'categorical';
  /**
   * Upper bounds of every class but the last, for numeric scales: the legend's own breaks. A host
   * that colours with Kepler's `quantile` scale gets slightly different breaks, because that scale
   * interpolates between values while the plan takes the value at the nearest rank and never splits
   * ties. A host that wants the map to match the legend exactly uses these as the thresholds of a
   * threshold scale.
   */
  thresholds?: readonly number[];
};

/**
 * The workbench display style of a plan. Class schemes and categorical columns map to `ordinal`;
 * sequential and diverging columns map to `quantile` and also carry the plan's `thresholds`. The
 * `quantile` scale's own breaks may differ slightly from the legend's (see `thresholds`). An
 * undefined plan gives an empty style, so a layer keeps its fixed colour.
 */
export function toKeplerWorkbenchDisplayStyle(
  plan: KeplerResultPresentation | undefined
): KeplerWorkbenchDisplayStyle {
  if (!plan) return {};
  return {
    colorColumn: plan.column,
    colorScale: plan.scale === 'categorical' ? 'ordinal' : 'quantile',
    palette: plan.palette,
    ...(plan.scale !== 'categorical' ? {thresholds: plan.breaks} : {})
  };
}

/** One row of a legend as a host draws it. */
export type KeplerResultLegendRow = {
  label: string;
  color: KeplerResultColor;
  count?: number;
};

export type KeplerBindingResultPresentation = {
  tableId: string;
  presentation?: KeplerResultPresentation;
  display: KeplerWorkbenchDisplayStyle;
  legend: readonly KeplerResultLegendRow[];
};

/** The legend of a plan as rows: classes or categories, or the numeric ranges, then rows with no value. */
function getLegendRows(plan: KeplerResultPresentation): KeplerResultLegendRow[] {
  const rows: KeplerResultLegendRow[] = [];
  if (plan.legend.length) {
    for (const entry of plan.legend) {
      rows.push({label: entry.label, color: entry.color, count: entry.count});
    }
  } else {
    for (const range of plan.ranges) {
      rows.push({label: range.label, color: range.color, count: range.count});
    }
  }
  // Rows with no value (and, for a class scheme, rows that are not a class) get a row of their own,
  // so the counts of the legend add up to the rows of the table.
  if (plan.missingCount > 0) {
    rows.push({label: 'No value', color: KEPLER_MISSING_VALUE_COLOR, count: plan.missingCount});
  }
  return rows;
}

/**
 * For every table of a run: its plan, the display style and the legend rows. A table whose output
 * has no presentation, or whose column is absent, has no plan, an empty style and no legend rows.
 */
export function getKeplerBindingResultPresentations(
  result: CatalogRunResult,
  descriptor: KeplerOperationDescriptor
): readonly KeplerBindingResultPresentation[] {
  return result.tables.map(table => {
    const output = descriptor.outputs.find(candidate => candidate.tableId === table.id);
    const presentation = output ? planKeplerResultPresentation(table, output) : undefined;
    return {
      tableId: table.id,
      ...(presentation ? {presentation} : {}),
      display: toKeplerWorkbenchDisplayStyle(presentation),
      legend: presentation ? getLegendRows(presentation) : []
    };
  });
}

/**
 * A publication-only categorical encoding of the exact AX plan. Numeric classes are closed on
 * the upper bound; classifying before Float32 display conversion preserves ties at the breaks.
 * The same legend supplies labels, counts and colours in the map and result card.
 */
export function encodeKeplerResultPresentation(
  table: CatalogOutputTable,
  presentation: KeplerBindingResultPresentation,
  colorColumn?: string
): KeplerResultEncoding | undefined {
  if (colorColumn && colorColumn !== presentation.presentation?.column) {
    const selected = planKeplerResultPresentation(table, {
      tableId: table.id,
      kind: 'attributes',
      presentation: {
        column: colorColumn,
        scale:
          table.columnTypes[colorColumn] === 'string' ||
          table.columnTypes[colorColumn] === 'boolean'
            ? 'categorical'
            : 'sequential'
      }
    });
    presentation = {
      tableId: table.id,
      presentation: selected,
      display: toKeplerWorkbenchDisplayStyle(selected),
      legend: selected ? getLegendRows(selected) : []
    };
  }
  const plan = presentation.presentation;
  if (!plan) return undefined;
  const values = table.columns[plan.column];
  if (!values) return undefined;
  // Labels are the published categories, so Kepler's native legend is meaningful too.
  const usedLabels = new Set<string>();
  const categoryLabels = presentation.legend.map(entry => {
    let label = entry.label;
    let suffix = 2;
    while (usedLabels.has(label)) label = `${entry.label} (${suffix++})`;
    usedLabels.add(label);
    return label;
  });
  const excluded = new Set(plan.excludedRows ?? []);
  const missingIndex = presentation.legend.length - 1;
  const otherIndex =
    plan.other || plan.legend.at(-1)?.value === 'other-clusters'
      ? plan.legend.length - 1
      : undefined;
  // Overflow is a synthetic class: its sentinel can also be a real category value.
  const categories = new Map(
    plan.legend.flatMap((entry, index) =>
      index === otherIndex ? [] : [[entry.value, index] as const]
    )
  );
  const excludedIndex = plan.legend.findIndex(entry => entry.tone === 'excluded');
  let classColumn = `analysis_result_class (${plan.column})`;
  while (hasColumn(table, classColumn)) classColumn += '_';
  const encodedValues = values.map((value, row) => {
    let index: number | undefined;
    if (excluded.has(row) && excludedIndex >= 0) index = excludedIndex;
    else if (plan.scale === 'categorical') {
      // Match the shared classifier's boolean decoding for the two flag schemes.
      const category = plan.classScheme
        ? normalizeGPUAnalysisClassValue(
            plan.classScheme,
            plan.classEncoding ?? 'contributor-integer',
            value
          )
        : value;
      index =
        category === null ||
        category === undefined ||
        (typeof category === 'number' && !Number.isFinite(category))
          ? undefined
          : (categories.get(category) ?? otherIndex);
    } else if (typeof value === 'number' && Number.isFinite(value)) {
      index = plan.breaks.findIndex(bound => value <= bound);
      if (index < 0) index = plan.ranges.length - 1;
    }
    return categoryLabels[index ?? missingIndex];
  });
  return {
    values: encodedValues,
    categories: categoryLabels,
    legend: presentation.legend,
    presentation: plan,
    classColumn,
    table: {
      ...table,
      columns: {...table.columns, [classColumn]: encodedValues},
      columnTypes: {...table.columnTypes, [classColumn]: 'string'}
    }
  };
}

/** One exact publication encoding shared by catalog staging and workflow display. */
export type KeplerResultEncoding = {
  values: string[];
  categories: string[];
  legend: readonly KeplerResultLegendRow[];
  presentation: KeplerResultPresentation;
  classColumn: string;
  table: CatalogOutputTable;
};

/** Native Kepler style for an encoded class column; it never asks Kepler to reclassify values. */
export function getKeplerResultEncodingStyle(encoding: KeplerResultEncoding) {
  const colors = encoding.legend.map(
    entry => `#${entry.color.map(value => value.toString(16).padStart(2, '0')).join('')}`
  );
  return {
    visualChannels: {
      colorField: {name: encoding.classColumn, type: 'string'},
      colorScale: 'customOrdinal'
    },
    visConfig: {
      colorRange: {
        name: encoding.presentation.column,
        type: 'customOrdinal',
        category: 'Custom',
        colors,
        colorMap: colors.map((color, index) => [encoding.categories[index], color] as const)
      },
      ...(encoding.presentation.outline?.required
        ? {
            outline: true,
            stroked: true,
            strokeColor: [...encoding.presentation.outline.color],
            thickness: 1
          }
        : {})
    }
  };
}
