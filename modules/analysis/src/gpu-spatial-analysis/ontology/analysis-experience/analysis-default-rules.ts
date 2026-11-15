// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisExtentSizes} from './analysis-spacing-statistics';
import {
  GPU_ANALYSIS_DEFAULT_RULES,
  type GPUAnalysisDataProfile,
  type GPUAnalysisDefaultResolution,
  type GPUAnalysisDefaultRule,
  type GPUAnalysisSpacingStatistics
} from './analysis-experience-types';

/**
 * Smallest bend of the normalised k-distance curve that counts as a knee. The curve is scaled to
 * [0, 1] on both axes; a straight line bends 0, and the bend is the largest vertical gap between
 * the curve and the chord joining its end points. Chosen from seeded point sets in
 * `scipy.spatial.cKDTree` (k = 3): 12 uniformly random sets of 200 to 5,000 points bend 0.25 to
 * 0.48, because the sorted distances of a random pattern still rise in a tail; four sets of four
 * Gaussian clusters with 5 percent uniform noise bend 0.90 to 0.91, and a lattice with a few
 * corner points bends 0.996. 0.65 sits between the two groups with a margin on both sides.
 * Schubert et al. (2017, section 4.2) note that the plot often has no clear knee; below this value
 * the rule returns undefined so the caller keeps its constant and says so. The bend alone is not
 * enough (see {@link GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN}).
 */
export const GPU_ANALYSIS_KNEE_MINIMUM_BEND = 0.65;

/**
 * Smallest ratio of the k-distance range (largest minus smallest) to its median for a bend to
 * count as a knee. The bend is computed on a curve scaled by its own range, so a lattice with a
 * jitter of 0.001 (range 0.41 of the median) bends 0.995, and a single blob bends 0.7 to 0.8: the
 * scaling makes any curve look sharp. Measured with SciPy `cKDTree` and scikit-learn
 * `DBSCAN(eps = knee, min_samples = k + 1)` (script `knee2.py`, quoted in the spec):
 * - range / median 0.4 to 2.3: lattices with jitter 0.001 and 0.5, uniform random points;
 * - 4.9 to 21.8: one Gaussian blob of 300 to 5,000 points, two or three clean clusters of 1,000
 *   to 3,000 points; DBSCAN at the knee splits them and invents 3 to 7 percent noise;
 * - 41 to 250: four clusters with 5, 15 and 30 percent uniform noise, where DBSCAN at the knee
 *   recovers the noise fraction (5.3, 15.2 and 30.9 percent against 5, 15 and 30).
 * 30 lies above every clean blob up to 5,000 points and below every noisy set. It is a limit of
 * the idea, not a cure: a clean Gaussian blob of 20,000 points or more has a ratio of 50 to 107,
 * its tail is as long as noise would make it, and the rule cannot tell the two apart. That is why
 * the explanation only says what the curve shows and asks to look at the plot.
 */
export const GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN = 30;

/** A curve with fewer quantiles than this has too few steps to show a knee. */
const MINIMUM_KNEE_CURVE_LENGTH = 8;

/** Divisor of Esri's documented default cell size: the shorter side of the extent over 250. */
const CELL_SIZE_DIVISOR = 250;

/** Options of {@link resolveGPUAnalysisDefault}. */
export type GPUAnalysisDefaultOptions = {
  /** Name of the column a `column-*` rule reads; ignored by every other rule. */
  column?: string;
};

/** One default rule as shown to a user and a developer. */
export type GPUAnalysisDefaultRuleRecord = {
  id: GPUAnalysisDefaultRule;
  /** Short name for a settings panel. */
  label: string;
  /** The statistic of the data profile the rule reads. */
  needs: string;
  /** The unit of the value, and whether `factor` applies. */
  unit: string;
  /** Where the rule comes from; says so when it is this library's own choice. */
  source: string;
};

const GPU_ANALYSIS_DEFAULT_RULE_RECORDS: readonly GPUAnalysisDefaultRuleRecord[] = [
  {
    id: 'maximum-nearest-neighbor-distance',
    label: 'Largest nearest-neighbour distance',
    needs: 'spacing statistics: nearestNeighbor.maximum',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      'GeoDa workbook, Distance-band weights (default distance band is the max-min distance); Esri, Spatial Autocorrelation (default distance ensures every feature has a neighbor)'
  },
  {
    id: 'median-nearest-neighbor-distance',
    label: 'Median nearest-neighbour distance',
    needs:
      'spacing statistics: nearestNeighbor.median, or nearestNeighbor.nonZeroMedian when the median is 0',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source: 'This library: the typical spacing of the points. Not a vendor default.'
  },
  {
    id: 'k-distance-knee',
    label: 'Knee of the k-distance curve',
    needs: 'spacing statistics: kDistance',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      'Ester et al. 1996, section 4.2 (sorted k-distance graph for DBSCAN eps); Schubert et al. 2017, "DBSCAN Revisited, Revisited", sections 4.1 to 4.2. The knee is the point furthest from the chord of the normalised curve, a simple form of the idea behind Satopaa et al. 2011, "Finding a Kneedle in a Haystack".'
  },
  {
    id: 'extent-shorter-side-over-250',
    label: 'Shorter side of the extent over 250',
    needs: 'spacing statistics: extentWidth and extentHeight (planar data: extent)',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      'Esri, Kernel Density and Point to Raster: default cell size is the shorter of the width or height of the extent divided by 250'
  },
  {
    id: 'extent-diagonal',
    label: 'Diagonal of the extent',
    needs: 'spacing statistics: extentDiagonal (planar data: extent)',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      'This library: the diagonal of the bounding extent, an upper bound on pairwise distance for planar data. The variogram maximum lag default is one third of it.'
  },
  {
    id: 'extent-west',
    label: 'West edge of the extent',
    needs: 'extent',
    unit: 'Source coordinates: degrees for geographic data, coordinate units for planar data. The factor is ignored.',
    source: 'This library: interpolation bounds follow the dataset extent.'
  },
  {
    id: 'extent-south',
    label: 'South edge of the extent',
    needs: 'extent',
    unit: 'Source coordinates: degrees for geographic data, coordinate units for planar data. The factor is ignored.',
    source: 'This library: interpolation bounds follow the dataset extent.'
  },
  {
    id: 'extent-east',
    label: 'East edge of the extent',
    needs: 'extent',
    unit: 'Source coordinates: degrees for geographic data, coordinate units for planar data. The factor is ignored.',
    source: 'This library: interpolation bounds follow the dataset extent.'
  },
  {
    id: 'extent-north',
    label: 'North edge of the extent',
    needs: 'extent',
    unit: 'Source coordinates: degrees for geographic data, coordinate units for planar data. The factor is ignored.',
    source: 'This library: interpolation bounds follow the dataset extent.'
  },
  {
    id: 'standard-distance',
    label: 'Standard distance',
    needs: 'spacing statistics: standardDistance',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      'Standard distance, the spatial analogue of the standard deviation (the root mean squared distance from the mean centre), as in the Esri Standard Distance tool'
  },
  {
    id: 'normal-reference-bandwidth-scott',
    label: "Scott's rule bandwidth",
    needs: 'spacing statistics: pointCount and standardDistance (unit weights)',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      "scipy.stats.gaussian_kde scotts_factor, Scott 1992, Multivariate Density Estimation; the same quantity as GPUKernelDensity bandwidthRule 'scott'"
  },
  {
    id: 'normal-reference-bandwidth-silverman',
    label: "Silverman's rule bandwidth",
    needs: 'spacing statistics: pointCount and standardDistance (unit weights)',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      "scipy.stats.gaussian_kde silverman_factor, Silverman 1986, Density Estimation for Statistics and Data Analysis; the same quantity as GPUKernelDensity bandwidthRule 'silverman'"
  },
  {
    id: 'extent-shorter-side-over-250-in-source-units',
    label: 'Shorter side of the extent over 250, in coordinate units',
    needs: 'extent (profile.extent, else the spacing statistics extent)',
    unit: 'Source coordinate units: degrees for geographic data, coordinate units for planar data. For parameters measured in coordinate units only; a degree is not a constant ground distance (a degree of longitude shrinks with the cosine of the latitude). The factor applies.',
    source:
      "Esri, Kernel Density and Point to Raster: default cell size is the shorter of the width or height of the extent divided by 250. This rule keeps Esri's quantity in the dataset's own units."
  },
  {
    id: 'outlier-resistant-nearest-neighbor-distance',
    label: 'Upper fence of the nearest-neighbour distances',
    needs: 'spacing statistics: nearestNeighbor.upperFence and aboveUpperFenceCount',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data.',
    source:
      "This library: Tukey's outer fence (upper quartile plus three interquartile ranges, Tukey 1977, Exploratory Data Analysis) of the nearest-neighbour distances, capped at the maximum. A distance band that isolated points do not inflate; the points beyond it have no neighbour."
  },
  {
    id: 'mean-point-spacing',
    label: 'Mean point spacing',
    needs: 'spacing statistics: pointCount, extentWidth and extentHeight',
    unit: 'Ground distance: metres for geographic data, coordinate units for planar data. A cell of factor times this holds about factor squared points on average.',
    source:
      "This layer's rule: the side of the square that holds one point on average, sqrt(extent width times height over point count), computed on every point. Esri's Optimized Hot Spot Analysis instead derives the aggregation cell size from nearest-neighbour distances and the extent; this rule follows point density only and does not reproduce Esri's formula."
  },
  {
    id: 'column-variance',
    label: 'Variance of a column',
    needs: 'column statistics: standardDeviation of the column named by options.column',
    unit: 'The squared unit of the column (reported as column-unit). The factor applies.',
    source: "This library: the square of the column's standard deviation. Not a vendor default."
  },
  {
    id: 'column-standard-deviation',
    label: 'Standard deviation of a column',
    needs: 'column statistics: standardDeviation of the column named by options.column',
    unit: 'The unit of the column (reported as column-unit). The factor applies.',
    source:
      'This library: the spread of one column as a scale for a threshold. Not a vendor default.'
  },
  {
    id: 'column-range',
    label: 'Range of a column',
    needs: 'column statistics: minimum and maximum of the column named by options.column',
    unit: 'The unit of the column (reported as column-unit). The factor applies.',
    source: 'This library: maximum minus minimum of one column. Not a vendor default.'
  },
  {
    id: 'column-sum',
    label: 'Sum of a column',
    needs:
      'column statistics: mean, rowCount, nullCount and optional sampleSize of the column named by options.column',
    unit: 'The unit of the column (reported as column-unit). The factor applies.',
    source:
      'This library: mean times the non-null count; sampled statistics estimate the full sum as mean times rowCount times (sampleSize minus nullCount) over sampleSize. Not a vendor default.'
  }
];

/**
 * Lists every default rule with its label, the statistic it needs and its source, in the order of
 * `GPU_ANALYSIS_DEFAULT_RULES`.
 */
export function listGPUAnalysisDefaultRules(): readonly GPUAnalysisDefaultRuleRecord[] {
  return GPU_ANALYSIS_DEFAULT_RULES.map(id => {
    const record = GPU_ANALYSIS_DEFAULT_RULE_RECORDS.find(candidate => candidate.id === id);
    if (!record) {
      throw new Error(`listGPUAnalysisDefaultRules has no record for rule ${id}`);
    }
    return record;
  });
}

/**
 * Derives one parameter default from what is known about the data.
 *
 * `value` is the rule output times `factor`, or undefined when the profile lacks what the rule
 * needs or the statistic is not usable (no points, no spread, a distance of 0). The caller then
 * keeps its constant and adds the caution `default-from-fallback`. `explanation` is always one
 * sentence; when `value` is undefined it says why. `sampled` is true when the statistics behind
 * the value came from a sample; sample statistics may differ from full-data statistics, and
 * the explanation gives the sample fraction and any estimation applied.
 *
 * Rules
 * - `maximum-nearest-neighbor-distance`: the largest nearest-neighbour distance, so every point
 *   has at least one neighbour. Undefined when it is 0 (all points at one location) or unknown.
 * - `median-nearest-neighbor-distance`: the median nearest-neighbour distance. When it is 0,
 *   uses the median distance from each distinct location to its nearest other distinct location.
 *   This uses the sample's distinct locations when sampled, and is undefined with fewer than two.
 * - `k-distance-knee`: the value of the ascending k-distance curve where it bends furthest from
 *   the straight line between its end points, after scaling both axes to [0, 1]. Undefined when
 *   the curve has fewer than 8 points, is flat, bends less than
 *   {@link GPU_ANALYSIS_KNEE_MINIMUM_BEND}, or spans less than
 *   {@link GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN} times its median.
 * - `extent-shorter-side-over-250`: the shorter side of the extent
 *   divided by 250, a ground distance (metres for geographic data, never degrees). Undefined when
 *   a side is 0 or the coordinate model is unknown.
 * - `extent-diagonal`: the diagonal of the extent in metres or coordinate units.
 * - The extent rules (`extent-shorter-side-over-250`, `extent-shorter-side-over-250-in-source-units`,
 *   `extent-diagonal`, `mean-point-spacing`'s area) read `profile.extent`, the extent of the
 *   geometry (all vertices), and fall back to the spacing statistics' extent only when it is
 *   absent: the statistics see one anchor per line or polygon, whose extent can be a point.
 *   Geographic extents are measured in metres with {@link getGPUAnalysisExtentSizes}.
 * - `extent-shorter-side-over-250-in-source-units`: the same quantity in the units of the source
 *   coordinates (degrees for geographic data), for parameters measured in coordinate units. A
 *   degree is not a constant ground distance.
 * - `extent-west|south|east|north`: the edge of `profile.extent` in source coordinates. `factor`
 *   is not applied to an edge (scaling a coordinate has no meaning), and the resolution reports
 *   `factor` 1. For geographic data a west edge greater than the east edge means the data crosses
 *   the antimeridian (the extent is the smallest arc of longitude); the explanation says so.
 * - `standard-distance`: root mean squared distance from the mean centre.
 * - `normal-reference-bandwidth-scott` and `normal-reference-bandwidth-silverman`: the bandwidth
 *   `GPUKernelDensity` computes for `bandwidthRule: 'scott'` or `'silverman'` with a Gaussian
 *   kernel and unit weights: `h = n^(-1/6) * sqrt((varianceX + varianceY) / 2)`, with the unbiased
 *   (n - 1) variances. In two dimensions both scipy rules give the factor `n^(-1/6)`, so the two
 *   rules return the same value. The sum of the two variances is `standardDistance^2 * n / (n -
 *   1)`, which is why the spacing statistics are enough. Undefined for fewer than 2 points or no
 *   spread. Pass the kernel's canonical-bandwidth scale
 *   (`getGPUKernelDensityCanonicalBandwidthScale`) as `factor` for the quartic and Epanechnikov
 *   kernels. With weights the library uses an effective count and weighted variances that the
 *   profile does not carry, so the value is for unweighted points. A normal-reference rule
 *   oversmooths clustered data. For geographic data the value is a ground distance in metres, the
 *   unit `GPUKernelDensity` expects for projected positions.
 *
 * - `outlier-resistant-nearest-neighbor-distance`: the upper fence of the nearest-neighbour
 *   distances (upper quartile plus three interquartile ranges, capped at the maximum). Isolated
 *   points do not inflate it; the explanation says how many points lie farther than this from
 *   every other point and will have no neighbour.
 * - `mean-point-spacing`: `sqrt(extentWidth * extentHeight / pointCount)` in the spacing unit,
 *   from every valid point (never the sample). Undefined without spacing or when the extent has
 *   no area.
 * - `column-variance`, `column-standard-deviation`, `column-range`: statistics of the one column
 *   of `profile.columns` named by `options.column`, in the column's unit (variance: its square).
 *   Undefined, with the reason, when no column is named, the name is unknown, the column is not
 *   numeric or time, or the statistic is missing or not positive. `sampled` is true when the
 *   column statistics were computed on fewer rows than the column has.
 * - `column-sum`: mean times (rowCount minus nullCount). With sampled statistics, nullCount
 *   describes sampled rows, so the full sum estimate is mean times rowCount times
 *   (sampleSize minus nullCount) over sampleSize. Undefined without a finite mean and valid counts;
 *   finite negative and zero sums are valid. The factor applies after estimating the full sum.
 *
 * Maximum nearest-neighbour distance: when isolated points set the maximum (it lies above the
 * upper fence) the explanation says so, gives the median for scale and names the
 * outlier-resistant rule.
 *
 * @param rule The rule to apply.
 * @param profile What is known about the dataset.
 * @param factor Positive multiplier of the rule output, 1 by default.
 * @param options The column named by a `column-*` rule.
 * @throws RangeError if `factor` is not a positive finite number.
 */
export function resolveGPUAnalysisDefault(
  rule: GPUAnalysisDefaultRule,
  profile: GPUAnalysisDataProfile,
  factor = 1,
  options?: GPUAnalysisDefaultOptions
): GPUAnalysisDefaultResolution {
  if (!Number.isFinite(factor) || factor <= 0) {
    throw new RangeError(
      `resolveGPUAnalysisDefault factor must be a positive finite number; got ${factor}`
    );
  }
  const spacing = profile.spacing;
  const distanceUnit = spacing ? spacing.unit : getSourceUnit(profile);

  switch (rule) {
    case 'maximum-nearest-neighbor-distance':
      return resolveNeighborDistance(
        rule,
        spacing,
        factor,
        spacing?.nearestNeighbor?.maximum,
        'Largest nearest-neighbour distance',
        spacing?.sampled
          ? 'at the unscaled statistic, every sampled point has at least one neighbour within the sample'
          : 'at the unscaled statistic, every point has at least one neighbour'
      );
    case 'median-nearest-neighbor-distance': {
      const nearest = spacing?.nearestNeighbor;
      const useNonZeroMedian = nearest?.median === 0;
      return resolveNeighborDistance(
        rule,
        spacing,
        factor,
        useNonZeroMedian ? nearest?.nonZeroMedian : nearest?.median,
        useNonZeroMedian
          ? 'Median nearest-neighbour distance of distinct locations'
          : 'Median nearest-neighbour distance',
        useNonZeroMedian
          ? `the plain median is 0 because of duplicate locations, so this uses the distance from each distinct location to its nearest other distinct location${spacing?.sampled ? ', over the distinct locations of the sample' : ''}`
          : 'the typical spacing of the points'
      );
    }
    case 'k-distance-knee':
      return resolveKneeDistance(rule, spacing, factor);
    case 'outlier-resistant-nearest-neighbor-distance':
      return resolveOutlierResistantDistance(rule, spacing, factor);
    case 'mean-point-spacing':
      return resolveMeanPointSpacing(rule, profile, factor);
    case 'column-variance':
    case 'column-standard-deviation':
    case 'column-range':
    case 'column-sum':
      return resolveColumnStatistic(rule, profile, factor, options?.column);
    case 'extent-shorter-side-over-250': {
      // A ground distance: metres for geographic data, coordinate units for planar data, from the
      // extent of the geometry (profile.extent) or, without it, of the spacing statistics' anchors.
      const sizes = getExtentSizes(profile);
      const output = sizes ? Math.min(sizes.width, sizes.height) / CELL_SIZE_DIVISOR : NaN;
      const unit =
        sizes?.unit ?? (profile.coordinates?.model === 'geographic' ? 'metre' : 'coordinate-unit');
      return createResolution(
        rule,
        isPositiveFinite(output) ? output * factor : undefined,
        factor,
        unit,
        false,
        isPositiveFinite(output)
          ? `Shorter side of the extent over ${CELL_SIZE_DIVISOR} (${formatQuantity(output, unit)})${describeFactor(factor, output * factor, unit)}, Esri's default cell size for density and point-to-raster`
          : unit === 'metre'
            ? 'No cell size in metres: geographic data needs an extent with a positive width and height'
            : 'No cell size from the extent: it needs an extent with a positive width and height and a known coordinate model'
      );
    }
    case 'extent-shorter-side-over-250-in-source-units': {
      const extent = getFiniteExtent(profile);
      const unit = getSourceUnit(profile);
      let shorterSide = NaN;
      if (extent) {
        const width = extent[2] >= extent[0] ? extent[2] - extent[0] : extent[2] - extent[0] + 360;
        shorterSide = Math.min(
          unit === 'degree' ? width : extent[2] - extent[0],
          extent[3] - extent[1]
        );
      }
      const output = shorterSide / CELL_SIZE_DIVISOR;
      return createResolution(
        rule,
        isPositiveFinite(output) ? output * factor : undefined,
        factor,
        unit,
        false,
        isPositiveFinite(output)
          ? `Shorter side of the extent over ${CELL_SIZE_DIVISOR} in the coordinates of the data (${formatQuantity(output, unit)})${describeFactor(factor, output * factor, unit)}, for parameters measured in coordinate units${unit === 'degree' ? ', where a degree is not a constant ground distance' : ''}`
          : 'No cell size in coordinate units: it needs an extent with a positive width and height'
      );
    }
    case 'extent-diagonal': {
      const sizes = getExtentSizes(profile);
      const diagonal = sizes ? Math.hypot(sizes.width, sizes.height) : NaN;
      const unit = sizes?.unit ?? distanceUnit;
      return createResolution(
        rule,
        isPositiveFinite(diagonal) ? diagonal * factor : undefined,
        factor,
        unit,
        false,
        isPositiveFinite(diagonal)
          ? `Diagonal of the extent (${formatQuantity(diagonal, unit)})${describeFactor(factor, diagonal * factor, unit)}, the diagonal of the bounding extent, an upper bound on pairwise distance for planar data`
          : 'No diagonal: it needs an extent with a positive size and a known coordinate model'
      );
    }
    case 'extent-west':
    case 'extent-south':
    case 'extent-east':
    case 'extent-north': {
      const unit = getSourceUnit(profile);
      const extent = getFiniteExtent(profile);
      const index = {
        'extent-west': 0,
        'extent-south': 1,
        'extent-east': 2,
        'extent-north': 3
      }[rule];
      const side = rule.slice('extent-'.length);
      return createResolution(
        rule,
        extent ? extent[index] : undefined,
        1,
        unit,
        false,
        extent
          ? `The ${side} edge of the data extent (${formatQuantity(extent[index], unit)}), so the result covers the data${
              unit === 'degree' && extent[0] > extent[2] && (index === 0 || index === 2)
                ? ' (the data crosses the antimeridian, so the west edge is greater than the east edge and the extent runs east from the west edge across 180 degrees)'
                : ''
            }`
          : `No ${side} edge: the profile has no finite extent`
      );
    }
    case 'standard-distance': {
      const standardDistance = spacing?.standardDistance;
      const usable = isPositiveFinite(standardDistance) && (spacing?.pointCount ?? 0) > 0;
      return createResolution(
        rule,
        usable && standardDistance !== undefined ? standardDistance * factor : undefined,
        factor,
        distanceUnit,
        false,
        usable && standardDistance !== undefined
          ? `Standard distance (${formatQuantity(standardDistance, distanceUnit)})${describeFactor(factor, standardDistance * factor, distanceUnit)}, the typical distance of a point from the centre of the data`
          : 'No standard distance: it needs spacing statistics with at least two distinct locations'
      );
    }
    case 'normal-reference-bandwidth-scott':
    case 'normal-reference-bandwidth-silverman':
      return resolveNormalReferenceBandwidth(rule, spacing, factor);
    default: {
      const unreachable: never = rule;
      throw new Error(`resolveGPUAnalysisDefault has no rule ${String(unreachable)}`);
    }
  }
}

function resolveNeighborDistance(
  rule: GPUAnalysisDefaultRule,
  spacing: GPUAnalysisSpacingStatistics | undefined,
  factor: number,
  output: number | undefined,
  name: string,
  meaning: string
): GPUAnalysisDefaultResolution {
  const unit = spacing ? spacing.unit : 'coordinate-unit';
  if (!spacing || !isPositiveFinite(output)) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      spacing && spacing.pointCount >= 2
        ? output === undefined || output === 0
          ? `No ${name.toLowerCase()}: the points share locations, so duplicate locations leave no positive nearest-neighbour statistic`
          : `No ${name.toLowerCase()}: the nearest-neighbour statistic must be positive and finite`
        : `No ${name.toLowerCase()}: the profile has no spacing statistics for at least two points`
    );
  }
  const nearest = spacing.nearestNeighbor;
  let reason = meaning;
  if (
    rule === 'maximum-nearest-neighbor-distance' &&
    nearest &&
    nearest.aboveUpperFenceCount > 0 &&
    output > nearest.upperFence
  ) {
    const count = nearest.aboveUpperFenceCount;
    reason = `set by ${formatNumber(count)} isolated ${spacing.sampled ? 'sampled ' : ''}${count === 1 ? 'point' : 'points'} in the unscaled statistic (the median is ${formatQuantity(nearest.median, unit)}), so most points have far closer neighbours; the outlier-resistant rule gives a band that isolated points do not inflate`;
  }
  return createResolution(
    rule,
    output * factor,
    factor,
    unit,
    spacing.sampled,
    `${name} (${formatQuantity(output, unit)})${describeFactor(factor, output * factor, unit)}, ${reason}${describeSample(spacing)}`
  );
}

function resolveOutlierResistantDistance(
  rule: GPUAnalysisDefaultRule,
  spacing: GPUAnalysisSpacingStatistics | undefined,
  factor: number
): GPUAnalysisDefaultResolution {
  const unit = spacing ? spacing.unit : 'coordinate-unit';
  const nearest = spacing?.nearestNeighbor;
  if (!spacing || !nearest || !isPositiveFinite(nearest.upperFence)) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      spacing && spacing.pointCount >= 2
        ? 'No upper fence of the nearest-neighbour distances: it is 0 because most points share locations'
        : 'No upper fence of the nearest-neighbour distances: the profile has no spacing statistics for at least two points'
    );
  }
  const count = nearest.aboveUpperFenceCount;
  const pointLabel = spacing.sampled ? 'sampled points' : 'points';
  const consequence =
    count === 0
      ? `no ${pointLabel} lie beyond the unscaled spacing fence`
      : `${formatNumber(count)} ${spacing.sampled ? 'sampled ' : ''}${count === 1 ? 'point lies' : 'points lie'} beyond the unscaled spacing fence`;
  const qualification =
    'this count describes the unscaled statistic, and actual isolation depends on the final distance band';
  return createResolution(
    rule,
    nearest.upperFence * factor,
    factor,
    unit,
    spacing.sampled,
    `Upper fence of the nearest-neighbour distances (${formatQuantity(nearest.upperFence, unit)})${describeFactor(factor, nearest.upperFence * factor, unit)}, the upper quartile plus three interquartile ranges, a band that isolated points do not inflate; ${consequence}; ${qualification}${describeSample(spacing)}`
  );
}

function resolveMeanPointSpacing(
  rule: GPUAnalysisDefaultRule,
  profile: GPUAnalysisDataProfile,
  factor: number
): GPUAnalysisDefaultResolution {
  const spacing = profile.spacing;
  const unit = spacing ? spacing.unit : 'coordinate-unit';
  const sizes = spacing && spacing.pointCount > 0 ? getExtentSizes(profile) : undefined;
  const output =
    spacing && sizes
      ? Math.sqrt(sizes.width) * (Math.sqrt(sizes.height) / Math.sqrt(spacing.pointCount))
      : NaN;
  if (!spacing || !isPositiveFinite(output)) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      'No mean point spacing: it needs spacing statistics with an extent that has a positive width and height'
    );
  }
  return createResolution(
    rule,
    output * factor,
    factor,
    unit,
    false,
    `Mean point spacing (${formatQuantity(output, unit)})${describeFactor(factor, output * factor, unit)}, the side of the square that holds one point on average over the extent of ${formatNumber(spacing.pointCount)} points, so a cell of this size holds about ${formatNumber(factor * factor)} ${factor * factor === 1 ? 'point' : 'points'} on average`
  );
}

function resolveColumnStatistic(
  rule: 'column-variance' | 'column-standard-deviation' | 'column-range' | 'column-sum',
  profile: GPUAnalysisDataProfile,
  factor: number,
  columnName: string | undefined
): GPUAnalysisDefaultResolution {
  const label = {
    'column-variance': 'variance',
    'column-standard-deviation': 'standard deviation',
    'column-range': 'range',
    'column-sum': 'sum'
  }[rule];
  const fail = (reason: string) =>
    createResolution(
      rule,
      undefined,
      factor,
      'column-unit',
      false,
      `No ${label} of a column: ${reason}`
    );
  if (columnName === undefined || columnName === '') {
    return fail('a column rule needs options.column, the name of the column');
  }
  const column = profile.columns.find(candidate => candidate.name === columnName);
  if (!column) {
    return fail(`the profile has no column named "${columnName}"`);
  }
  if (column.type !== 'number' && column.type !== 'integer' && column.type !== 'time') {
    return fail(`column "${columnName}" is of type ${column.type}, not a number or a time`);
  }
  if (rule === 'column-sum') {
    const sampled = column.sampleSize !== undefined && column.sampleSize < column.rowCount;
    const rowsSeen = column.sampleSize ?? column.rowCount;
    if (
      column.mean === undefined ||
      !Number.isFinite(column.mean) ||
      !Number.isInteger(column.rowCount) ||
      column.rowCount < 0 ||
      !Number.isInteger(rowsSeen) ||
      rowsSeen < 0 ||
      rowsSeen > column.rowCount ||
      !Number.isInteger(column.nullCount) ||
      column.nullCount < 0 ||
      column.nullCount > rowsSeen ||
      (sampled && rowsSeen === 0)
    ) {
      return fail(
        `column "${columnName}" needs a finite mean, non-negative integer rowCount and nullCount, and nullCount no greater than the rows used for the statistics; sampleSize must be a positive integer no greater than rowCount when sampled`
      );
    }
    const nonNullCount = rowsSeen - column.nullCount;
    const estimatedNonNullCount = sampled
      ? column.rowCount * (nonNullCount / rowsSeen)
      : nonNullCount;
    const output = column.mean * estimatedNonNullCount;
    const scaled = output * factor;
    const calculation = sampled
      ? `mean ${formatNumber(column.mean)} times ${formatNumber(column.rowCount)} full rows times the sample non-null share ${formatNumber(nonNullCount)}/${formatNumber(rowsSeen)}, which estimates the full sum from a sample of ${formatNumber(rowsSeen)} rows`
      : `mean ${formatNumber(column.mean)} times ${formatNumber(nonNullCount)} non-null rows`;
    return createResolution(
      rule,
      scaled,
      factor,
      'column-unit',
      sampled,
      `Sum of column "${columnName}" (${formatNumber(output)} in units of the column), ${calculation}${factor === 1 ? '' : ` times ${formatNumber(factor)} = ${formatNumber(scaled)}`}`
    );
  }
  let output: number;
  if (rule === 'column-range') {
    output =
      column.minimum !== undefined && column.maximum !== undefined
        ? column.maximum - column.minimum
        : NaN;
  } else if (column.standardDeviation !== undefined) {
    output =
      rule === 'column-variance'
        ? column.standardDeviation * column.standardDeviation
        : column.standardDeviation;
  } else {
    output = NaN;
  }
  if (!isPositiveFinite(output)) {
    return fail(
      `column "${columnName}" has no ${rule === 'column-range' ? 'minimum and maximum that differ' : 'standard deviation above 0'} in the profile`
    );
  }
  const sampled = column.sampleSize !== undefined && column.sampleSize < column.rowCount;
  const power = rule === 'column-variance' ? 'squared units' : 'units';
  const scaled = output * factor;
  const times = factor === 1 ? '' : ` times ${formatNumber(factor)} = ${formatNumber(scaled)}`;
  return createResolution(
    rule,
    scaled,
    factor,
    'column-unit',
    sampled,
    `${label.charAt(0).toUpperCase()}${label.slice(1)} of column "${columnName}" (${formatNumber(output)} in ${power} of the column)${times}${sampled ? ` (from a sample of ${formatNumber(column.sampleSize ?? 0)} of ${formatNumber(column.rowCount)} rows)` : ''}`
  );
}

function resolveKneeDistance(
  rule: GPUAnalysisDefaultRule,
  spacing: GPUAnalysisSpacingStatistics | undefined,
  factor: number
): GPUAnalysisDefaultResolution {
  const unit = spacing ? spacing.unit : 'coordinate-unit';
  const curve = spacing?.kDistance;
  if (!spacing || !curve || curve.distances.length < MINIMUM_KNEE_CURVE_LENGTH) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      'No knee of the k-distance curve: the profile has no curve of at least 8 points'
    );
  }
  const knee = findKnee(curve.distances);
  const ordinal = formatOrdinal(curve.k);
  if (knee && !isPositiveFinite(knee.value)) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      spacing.sampled,
      'No knee radius: duplicate locations give a zero k-distance at the bend; a radius must be positive and finite'
    );
  }
  if (!knee || knee.bend < GPU_ANALYSIS_KNEE_MINIMUM_BEND) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      `The sorted ${ordinal}-neighbour distance curve has no clear knee (largest bend ${knee ? formatNumber(knee.bend) : 'none'}, below ${GPU_ANALYSIS_KNEE_MINIMUM_BEND}), so no radius is suggested`
    );
  }
  const sorted = [...curve.distances].sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 1
      ? sorted[(sorted.length - 1) / 2]
      : sorted[sorted.length / 2 - 1] +
        (sorted[sorted.length / 2] - sorted[sorted.length / 2 - 1]) / 2;
  const rangeOverMedian =
    median > 0 ? (sorted[sorted.length - 1] - sorted[0]) / median : Number.POSITIVE_INFINITY;
  if (rangeOverMedian < GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      `The sorted ${ordinal}-neighbour distance curve bends ${formatNumber(knee.bend)} but rises by only ${formatNumber(rangeOverMedian)} times its median (at least ${GPU_ANALYSIS_KNEE_MINIMUM_RANGE_OVER_MEDIAN} is needed), so it does not meet this rule's range-to-median threshold and no radius is suggested`
    );
  }
  return createResolution(
    rule,
    knee.value * factor,
    factor,
    unit,
    spacing.sampled,
    `Knee of the sorted ${ordinal}-neighbour distance curve (${formatQuantity(knee.value, unit)})${describeFactor(factor, knee.value * factor, unit)}, where distances start to rise steeply; points whose ${ordinal}-neighbour is farther than this are in sparser surroundings than most, so check the plot${describeSample(spacing)}`
  );
}

/**
 * Finds the knee of an ascending curve: both axes are scaled to [0, 1] and the knee is the first
 * point with the largest vertical gap below the chord joining the end points. Returns undefined
 * for a flat or non-finite curve.
 */
function findKnee(
  distances: readonly number[]
): {index: number; value: number; bend: number} | undefined {
  const last = distances.length - 1;
  const first = distances[0];
  const range = distances[last] - first;
  if (!Number.isFinite(range) || range <= 0) {
    return undefined;
  }
  let bestIndex = 0;
  let bestBend = -Infinity;
  for (let index = 0; index <= last; index++) {
    const bend = index / last - (distances[index] - first) / range;
    if (bend > bestBend) {
      bestBend = bend;
      bestIndex = index;
    }
  }
  return {index: bestIndex, value: distances[bestIndex], bend: bestBend};
}

function resolveNormalReferenceBandwidth(
  rule: 'normal-reference-bandwidth-scott' | 'normal-reference-bandwidth-silverman',
  spacing: GPUAnalysisSpacingStatistics | undefined,
  factor: number
): GPUAnalysisDefaultResolution {
  const unit = spacing ? spacing.unit : 'coordinate-unit';
  const name = rule === 'normal-reference-bandwidth-scott' ? "Scott's" : "Silverman's";
  const pointCount = spacing?.pointCount ?? 0;
  const standardDistance = spacing?.standardDistance;
  if (!spacing || pointCount < 2 || !isPositiveFinite(standardDistance)) {
    return createResolution(
      rule,
      undefined,
      factor,
      unit,
      false,
      `No ${name} rule bandwidth: it needs spacing statistics of at least two distinct locations`
    );
  }
  // Variance per axis, averaged over the two axes, with the unbiased n - 1 divisor, as GPUKernelDensity
  // computes it: standardDistance^2 = (sumSquaresX + sumSquaresY) / n.
  const bandwidth =
    standardDistance * (pointCount ** (-1 / 6) * Math.sqrt(pointCount / (2 * (pointCount - 1))));
  return createResolution(
    rule,
    bandwidth * factor,
    factor,
    unit,
    false,
    `${name} rule bandwidth (${formatQuantity(bandwidth, unit)})${describeFactor(factor, bandwidth * factor, unit)}: the spread of the ${formatNumber(pointCount)} points times n^(-1/6), as the kernel density operation computes it, which smooths clustered data too much`
  );
}

// Helpers

/**
 * Width, height and unit of the extent the extent rules use: `profile.extent` (the extent of the
 * geometry, which for lines and polygons is wider than the anchors the spacing statistics see)
 * when present, otherwise the extent of the spacing statistics. Geographic extents are measured in
 * metres by {@link getGPUAnalysisExtentSizes}; the coordinate model comes from the profile or the
 * spacing statistics, and without either the sizes are unknown.
 */
function getExtentSizes(
  profile: GPUAnalysisDataProfile
): {width: number; height: number; unit: 'metre' | 'coordinate-unit'} | undefined {
  const spacing = profile.spacing && profile.spacing.pointCount > 0 ? profile.spacing : undefined;
  const model = profile.coordinates?.model ?? spacing?.coordinateModel;
  if (!model) {
    return undefined;
  }
  const unit = model === 'geographic' ? 'metre' : 'coordinate-unit';
  if (profile.extent && profile.extent.every(Number.isFinite)) {
    const sizes = getGPUAnalysisExtentSizes(profile.extent, model);
    return {width: sizes.width, height: sizes.height, unit};
  }
  if (spacing && spacing.coordinateModel === model) {
    return {width: spacing.extentWidth, height: spacing.extentHeight, unit};
  }
  return undefined;
}

function createResolution(
  rule: GPUAnalysisDefaultRule,
  value: number | undefined,
  factor: number,
  unit: GPUAnalysisDefaultResolution['unit'],
  sampled: boolean,
  explanation: string
): GPUAnalysisDefaultResolution {
  const isExtentEdge =
    rule === 'extent-west' ||
    rule === 'extent-south' ||
    rule === 'extent-east' ||
    rule === 'extent-north';
  const requiresPositiveValue = !isExtentEdge && rule !== 'column-sum';
  if (value !== undefined && (!Number.isFinite(value) || (requiresPositiveValue && value <= 0))) {
    return {
      rule,
      value: undefined,
      factor,
      unit,
      sampled,
      explanation: `No suggestion for ${rule}: the final scaled value must be ${requiresPositiveValue ? 'positive and finite' : 'finite'}; factor ${formatNumber(factor)} produces a result outside that range.`
    };
  }
  return {rule, value, factor, unit, explanation: `${explanation}.`, sampled};
}

function isPositiveFinite(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value) && value > 0;
}

/** Unit of source coordinates: degrees for geographic, otherwise coordinate units. */
function getSourceUnit(profile: GPUAnalysisDataProfile): 'degree' | 'coordinate-unit' {
  const model = profile.coordinates?.model ?? profile.spacing?.coordinateModel;
  return model === 'geographic' ? 'degree' : 'coordinate-unit';
}

function getFiniteExtent(
  profile: GPUAnalysisDataProfile
): readonly [number, number, number, number] | undefined {
  const extent =
    profile.extent ?? (profile.spacing?.pointCount ? profile.spacing.extent : undefined);
  return extent && extent.every(Number.isFinite) ? extent : undefined;
}

function describeSample(spacing: GPUAnalysisSpacingStatistics): string {
  return spacing.sampled
    ? `. Distances were computed within a sample of ${formatNumber(spacing.sampleSize)} of ${formatNumber(spacing.pointCount)} points (${formatNumber((100 * spacing.sampleSize) / spacing.pointCount)}%). Sample statistics may differ substantially from full-data statistics; no correction is applied`
    : '';
}

function describeFactor(
  factor: number,
  scaled: number,
  unit: GPUAnalysisDefaultResolution['unit']
): string {
  return factor === 1 ? '' : ` times ${formatNumber(factor)} = ${formatQuantity(scaled, unit)}`;
}

/** Fixed locale and three significant digits, so the same value reads the same on every machine. */
function formatNumber(value: number): string {
  return value.toLocaleString('en-US', {maximumSignificantDigits: 3});
}

function formatQuantity(value: number, unit: GPUAnalysisDefaultResolution['unit']): string {
  const text = formatNumber(value);
  if (unit === 'metre') {
    return value >= 10_000 ? `${formatNumber(value / 1000)} km` : `${text} m`;
  }
  if (unit === 'column-unit') {
    return `${text} in column units`;
  }
  if (unit === 'degree') {
    return `${text} degrees`;
  }
  return Math.abs(value) === 1 ? `${text} coordinate unit` : `${text} coordinate units`;
}

function formatOrdinal(rank: number): string {
  const lastTwoDigits = rank % 100;
  if (lastTwoDigits >= 11 && lastTwoDigits <= 13) {
    return `${rank}th`;
  }
  const suffix = {1: 'st', 2: 'nd', 3: 'rd'}[rank % 10] ?? 'th';
  return `${rank}${suffix}`;
}
