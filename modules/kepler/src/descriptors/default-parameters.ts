// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EMERGING_HOT_SPOTS_LIMITS} from '../curated/bindings/curated-emerging-hot-spots-binding';
import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

import {
  guessGPUAnalysisMeasurementLevel,
  resolveGPUAnalysisDefault,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisCoordinateModel,
  type GPUAnalysisDefaultResolution,
  type GPUAnalysisDefaultRule,
  type GPUAnalysisMeasurementLevel
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  getVecchiaNuggetFloor,
  VECCHIA_NUGGET_FLOOR_FRACTION
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogParameterValue
} from '../analysis/catalog-binding-types';
import {
  getCatalogDefaultParameters,
  normalizeCatalogParameters
} from '../analysis/catalog-bindings';
import {
  getCatalogParameterRenameMessage,
  getCatalogValidationIdentity,
  resolveCatalogParameterAliases
} from '../analysis/catalog-parameter-constraints';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {makeLocalFrame} from '../analysis/catalog-local-frame';
import {applyKeplerDatasetProfileDeclarations} from '../profile/dataset-profile';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import type {KeplerOperationDescriptor} from './operation-descriptor-types';

// Resolves the parameters of one binding from what is known about its input datasets. A rule in
// the descriptor derives a distance or a size from the profile; every other parameter keeps the
// binding's default. The result is a complete parameter set that
// `normalizeCatalogParameters` accepts unchanged.

/**
 * Significant digits of a value derived from data. A cell size of 83.4 m or a bandwidth of
 * 1,237 m states a precision the statistic does not have (a nearest-neighbour distance moves with
 * the sample), and a person edits the number by eye, so two digits (83 m, 1,200 m) are kept.
 */
export const DATA_DEFAULT_SIGNIFICANT_DIGITS = 2;

/** How a rule's value is rounded to {@link DATA_DEFAULT_SIGNIFICANT_DIGITS}, and why. */
export type KeplerDefaultRuleRounding = {
  /** `up`: ceiling. `nearest`: ordinary rounding. `none`: the value is kept exactly. */
  direction: 'up' | 'nearest' | 'none';
  reason: string;
};

const REACH_REASON =
  'A reach that must include a neighbour: rounding down can leave the farthest point without one (a knee of 111 m rounded to 110 m made DBSCAN label every point of a regular lattice noise)';

/**
 * Rounding of each default rule, in one place. Rules whose value is a distance that has to reach a
 * neighbour round up; sizes and bandwidths, which only need to be about right, round to nearest;
 * coordinates are not rounded at all (two digits would move a longitude by degrees).
 */
export const KEPLER_DEFAULT_RULE_ROUNDING: Readonly<
  Record<GPUAnalysisDefaultRule, KeplerDefaultRuleRounding>
> = {
  'maximum-nearest-neighbor-distance': {direction: 'up', reason: REACH_REASON},
  'outlier-resistant-nearest-neighbor-distance': {direction: 'up', reason: REACH_REASON},
  'k-distance-knee': {direction: 'up', reason: REACH_REASON},
  'median-nearest-neighbor-distance': {
    direction: 'nearest',
    reason: 'The typical spacing; it is not required to reach anything'
  },
  'extent-shorter-side-over-250': {direction: 'nearest', reason: 'A cell size'},
  'extent-shorter-side-over-250-in-source-units': {direction: 'nearest', reason: 'A cell size'},
  'extent-diagonal': {direction: 'nearest', reason: 'A size'},
  'mean-point-spacing': {
    direction: 'nearest',
    reason: 'A typical spacing; it is not required to reach anything'
  },
  'extent-west': {direction: 'none', reason: 'A coordinate'},
  'extent-south': {direction: 'none', reason: 'A coordinate'},
  'extent-east': {direction: 'none', reason: 'A coordinate'},
  'extent-north': {direction: 'none', reason: 'A coordinate'},
  'normal-reference-bandwidth-scott': {direction: 'nearest', reason: 'A bandwidth'},
  'normal-reference-bandwidth-silverman': {direction: 'nearest', reason: 'A bandwidth'},
  'standard-distance': {direction: 'nearest', reason: 'A spread used as a size'},
  'column-variance': {direction: 'nearest', reason: 'A scale of the values'},
  'column-standard-deviation': {direction: 'nearest', reason: 'A scale of the values'},
  'column-range': {direction: 'nearest', reason: 'A scale of the values'},
  'column-sum': {
    direction: 'up',
    reason: 'A total used to size capacity must not round down and exceed that capacity'
  }
};

/** Source-coordinate edges are positions; every other non-column rule derives a length. */
function getLengthCoordinateModel(
  rule: GPUAnalysisDefaultRule,
  resolution: GPUAnalysisDefaultResolution
): GPUAnalysisCoordinateModel | undefined {
  if (
    rule.startsWith('column-') ||
    ['extent-west', 'extent-south', 'extent-east', 'extent-north'].includes(rule)
  ) {
    return undefined;
  }
  return resolution.unit === 'metre' || resolution.unit === 'degree' ? 'geographic' : 'planar';
}

/** Where one parameter's value came from. */
export type KeplerParameterSource =
  | 'override'
  | 'data'
  | 'constant'
  | 'fallback'
  | 'field'
  | 'preset';

/** How one parameter received its value. */
export type KeplerParameterResolution = {
  parameter: string;
  source: KeplerParameterSource;
  /** The rule's output and explanation, present when a rule ran (also when it fell back). */
  resolution?: GPUAnalysisDefaultResolution;
  /** Coordinate model in which a data-derived length was computed; determines its units. */
  coordinateModel?: GPUAnalysisCoordinateModel;
  /** Present when the value had to be moved: into the parameter's range, or up to fit a grid cap. */
  clamped?: {from: number; to: number; reason?: string};
  /**
   * Why: the layer's reason for a `preset`, what a `fallback` could not get, which measurement
   * level a column-dependent value was chosen for.
   */
  reason?: string;
  /** A placeholder default until the analyst selects the parameter explicitly. */
  requiresChoice?: boolean;
  /**
   * True when the value rests on a guess about the data (a coordinate model guessed from the
   * extent): it is shown as a guess and can be wrong. Absent otherwise.
   */
  guessed?: boolean;
  /**
   * False when the parameter only matters while another parameter holds other values
   * (`activeWhen`). An inactive parameter still has its value, which the catalog needs; its
   * unknowns and fallbacks do not count against a suggestion. Absent means active; this module
   * always sets it.
   */
  active?: boolean;
};

export type KeplerResolvedParameters = {
  parameters: Record<string, CatalogParameterValue>;
  /** One entry per binding parameter, in the binding's order. */
  resolutions: readonly KeplerParameterResolution[];
  /** Parameters shown first: the columns and the descriptor's `basic` parameters. */
  basic: readonly string[];
  /** Parameters folded away. */
  advanced: readonly string[];
  /** Parameters whose resolution is inactive; they are also named in `basic` or `advanced`. */
  inactive: readonly string[];
  /** Data or explicit settings whose execution safety still needs an analyst's decision. */
  confirmations: readonly string[];
};

export type KeplerParameterResolutionOptions = {
  /** Column name or names by the name of the string parameter that receives them. */
  fieldBindings?: Readonly<Record<string, string | readonly string[]>>;
  /**
   * Most cells the binding's grid may have (see {@link KEPLER_GRID_CELL_CAPS}). When given, a cell
   * size derived from the extent is raised until the implied grid fits.
   */
  maximumCellCount?: number;
  /**
   * Where the operation will run. `cpu` applies the CPU work caps of {@link KEPLER_CPU_WORK_CAPS}
   * (a larger cell size keeps a grid operation within the CPU reference engine's work limit).
   * Default `gpu`; a binding with no GPU path runs on the CPU whatever is asked.
   */
  executionEngine?: 'gpu' | 'cpu';
  /** Values a person or a curated analysis fixes; they replace every other source. */
  overrides?: Readonly<Record<string, CatalogParameterValue>>;
};

/** Rounds to a number of significant digits, optionally upwards. */
function roundToSignificantDigits(value: number, digits: number, upwards: boolean): number {
  if (value === 0) {
    return 0;
  }
  const magnitude = 10 ** (Math.floor(Math.log10(Math.abs(value))) - digits + 1);
  const scaled = value / magnitude;
  const rounded = upwards ? Math.ceil(scaled - 1e-9) : Math.round(scaled);
  // The product can carry binary noise (0.1 * 3); toPrecision(15) removes it.
  return Number((rounded * magnitude).toPrecision(15));
}

function clampToRange(
  value: number,
  specification: CatalogParameterSpec
): {value: number; clamped?: {from: number; to: number; reason?: string}} {
  const minimum =
    specification.kind === 'integer' && specification.minimum !== undefined
      ? Math.ceil(specification.minimum)
      : specification.minimum;
  const maximum =
    specification.kind === 'integer' && specification.maximum !== undefined
      ? Math.floor(specification.maximum)
      : specification.maximum;
  let result = value;
  if (minimum !== undefined && result < minimum) result = minimum;
  if (maximum !== undefined && result > maximum) result = maximum;
  return result === value ? {value} : {value: result, clamped: {from: value, to: result}};
}

function mapCoordinateModelToParameter(
  specification: CatalogParameterSpec,
  model: 'geographic' | 'planar'
): string | undefined {
  if (specification.name !== 'coordinateSystem' || specification.kind !== 'enum') {
    return undefined;
  }
  const wanted = model === 'geographic' ? 'EPSG:4326' : 'planar';
  return specification.options?.includes(wanted) ? wanted : undefined;
}

/**
 * Grid cell caps of the bindings whose cell size comes from the extent, read from each binding's
 * code. A grid larger than the cap makes the binding throw, and a descriptor cannot say so because
 * the cap depends on the shape of the extent. Bindings not listed are left alone.
 */
export const KEPLER_GRID_CELL_CAPS: Readonly<
  Record<string, {maximumCellCount: number; source: string}>
> = {
  'curated-emerging-hot-spots': {
    maximumCellCount: EMERGING_HOT_SPOTS_LIMITS.cells,
    source: 'curated-emerging-hot-spots-binding.ts:43'
  },
  // catalog-extra-bindings.ts:745-789 passes MAXIMUM_FEATURES to runGridGenerator.
  'grid-generator': {maximumCellCount: 20000, source: 'catalog-extra-bindings.ts:745-789'},
  // catalog-platform5-recipes-hot-spot.ts:42 (MAXIMUM_HOT_SPOT_CELLS) checked at :81.
  'recipe-hot-spot-lattice': {
    maximumCellCount: 40000,
    source: 'catalog-platform5-recipes-hot-spot.ts:42,81'
  },
  // catalog-platform5-raster.ts:45 (MAXIMUM_RASTER_CELLS) checked at :75, in readPointGrid.
  'raster-density-smoothing': {
    maximumCellCount: 40000,
    source: 'catalog-platform5-raster.ts:45,75'
  },
  'raster-cell-clusters': {maximumCellCount: 40000, source: 'catalog-platform5-raster.ts:45,75'},
  // catalog-platform3-b-bindings.ts:666 and catalog-platform4-statistics-gpu.ts:68: 40000 cells.
  'kriging-grid': {
    maximumCellCount: 40000,
    source: 'catalog-platform3-b-bindings.ts:666; catalog-platform4-statistics-gpu.ts:68'
  },
  // curated-kriging-with-diagnostics-binding.ts:53 (MAXIMUM_CELLS) checked at :290.
  'curated-kriging-with-diagnostics': {
    maximumCellCount: 40000,
    source: 'curated-kriging-with-diagnostics-binding.ts:53,290'
  },
  // catalog-platform3-b-common.ts:36 (MAXIMUM_GRID_CELLS) checked in makeGridSpec at :206; used
  // by preparePointGrid (catalog-platform3-b-grids.ts:132), the line grid (:649),
  // catalog-platform4-raster.ts:281 and catalog-platform5-recipes.ts:640.
  'idw-grid': {maximumCellCount: 250000, source: 'catalog-platform3-b-common.ts:36,206'},
  'kernel-density-grid': {maximumCellCount: 250000, source: 'catalog-platform3-b-common.ts:36,206'},
  'point-density-grid': {maximumCellCount: 250000, source: 'catalog-platform3-b-common.ts:36,206'},
  'line-density-grid': {maximumCellCount: 250000, source: 'catalog-platform3-b-common.ts:36,206'},
  'raster-join-points-in-polygons': {
    maximumCellCount: 250000,
    source: 'catalog-platform3-b-common.ts:36,206; catalog-platform4-raster.ts:281'
  },
  'recipe-straight-line-catchments': {
    maximumCellCount: 250000,
    source: 'catalog-platform3-b-common.ts:36,206; catalog-platform5-recipes.ts:640'
  }
};

/** What a binding limits on the CPU reference engine, and where the code says so. */
export type KeplerCpuWorkCap =
  | {
      /** Grid cells times input rows must not exceed `limit`. */
      kind: 'cells-times-rows';
      limit: number;
      source: string;
    }
  | {
      /** Input rows squared must not exceed `limit` (a pairwise distance matrix or neighbour search). */
      kind: 'rows-squared';
      limit: number;
      source: string;
      /**
       * Parameter and values that avoid the pairwise work (contiguity weights need no distance
       * matrix); the cap then does not apply.
       */
      exceptWhen?: {parameter: string; equals: readonly string[]};
    };

/**
 * CPU work caps by binding id, each read from the code. They apply when the binding runs on the CPU
 * reference engine: always for a binding whose engine is `cpu-reference`, and when the host has no
 * GPU device for the others.
 */
export const KEPLER_CPU_WORK_CAPS: Readonly<Record<string, KeplerCpuWorkCap>> = {
  // catalog-platform3-b-grids.ts:45 (MAXIMUM_CPU_WORK = 6e7) checked by assertCpuWork at :135-138
  // and called from computeIdwField (:154), computeKernelDensityField (:308) and
  // computeKrigingField (:545).
  'idw-grid': {
    kind: 'cells-times-rows',
    limit: 6e7,
    source: 'catalog-platform3-b-grids.ts:45,137,154'
  },
  'kernel-density-grid': {
    kind: 'cells-times-rows',
    limit: 6e7,
    source: 'catalog-platform3-b-grids.ts:45,137,308'
  },
  'kriging-grid': {
    kind: 'cells-times-rows',
    limit: 6e7,
    source: 'catalog-platform3-b-grids.ts:45,137,545'
  },
  // Matrix-based routes use MAXIMUM_PAIR_WORK = 4e7. Bucketed DBSCAN has its own
  // shared point-count predicate, checked by both applicability and execution.
  'optics-ordering': {
    kind: 'rows-squared',
    limit: 4e7,
    source: 'catalog-platform3-b-clusters.ts:33,36-39,272'
  },
  'hierarchical-density-clustering': {
    kind: 'rows-squared',
    limit: 4e7,
    source: 'catalog-platform3-b-clusters.ts:33,36-39,272'
  },
  'local-outlier-factor': {
    kind: 'rows-squared',
    limit: 4e7,
    source: 'catalog-platform3-b-clusters.ts:33,36-39,272'
  },
  // Curated clusters uses the same bucketed DBSCAN, then sorted monotone-chain hulls
  // and linear measurements; its separate 6000-point recipe input cap remains in force.
  // catalog-platform3-b-statistics.ts:81,146: distance and nearest-neighbour weights need
  // count * count <= MAXIMUM_PAIR_WORK (4e7, :81); queen and rook contiguity need no pair work.
  ...Object.fromEntries(
    [
      'spatial-weights',
      'global-spatial-statistics',
      'local-moran-statistics',
      'hotspot-gi-star',
      'local-geary-statistics',
      'local-lisa-variants',
      'global-join-counts-getis-ord',
      'skater-regions'
    ].map(id => [
      id,
      {
        kind: 'rows-squared' as const,
        limit: 4e7,
        source: 'catalog-platform3-b-statistics.ts:81,146',
        exceptWhen: {parameter: 'neighborhood', equals: ['queen', 'rook']}
      }
    ])
  )
};

/** Where a binding runs: a binding with no GPU path runs on the CPU whatever is asked. */
export function getKeplerEffectiveEngine(
  binding: Pick<CatalogOperationBinding, 'engine'>,
  requested: 'gpu' | 'cpu' = 'gpu'
): 'gpu' | 'cpu' {
  return binding.engine === 'cpu-reference' ? 'cpu' : requested;
}

/**
 * The size, in the bindings' own frame, of the grid extent for a profile. The bindings project the
 * data into a local azimuthal equidistant frame (`makeLocalFrame` in catalog-local-frame.ts, shift
 * `minimum`) and grid the projected extent; a longitude and latitude extent measured on the sphere
 * is smaller (at 60 degrees north a 20 by 1 degree box is 1.34 times taller in the frame). So this
 * calls the same helper on points along the profile's extent boundary and its middle lines, with
 * the antimeridian convention (west greater than east means the extent crosses 180). The boundary
 * bounds every point the data can have inside the extent, so the result is at least the data's own
 * frame extent and a grid sized from it never exceeds a cap the data would meet.
 */
export function getKeplerGridFrameExtent(
  profile: KeplerDatasetProfile,
  coordinateSystem: string
): {width: number; height: number} | undefined {
  const extent = profile.extent ?? profile.spacing?.extent;
  if (!extent || extent.some(value => !Number.isFinite(value))) {
    return undefined;
  }
  if (coordinateSystem === 'planar') {
    return {width: extent[2] - extent[0], height: extent[3] - extent[1]};
  }
  const west = extent[0];
  const east = extent[0] > extent[2] ? extent[2] + 360 : extent[2];
  const south = extent[1];
  const north = extent[3];
  const steps = 24;
  const wrap = (longitude: number) => ((((longitude + 180) % 360) + 360) % 360) - 180;
  const positions: [number, number][] = [];
  for (let step = 0; step <= steps; step++) {
    const longitude = wrap(west + ((east - west) * step) / steps);
    const latitude = south + ((north - south) * step) / steps;
    positions.push(
      [longitude, south],
      [longitude, north],
      [longitude, (south + north) / 2],
      [wrap(west), latitude],
      [wrap(east), latitude],
      [wrap((west + east) / 2), latitude]
    );
  }
  const frame = makeLocalFrame('EPSG:4326', positions, {shift: 'minimum'});
  let width = 0;
  let height = 0;
  for (const position of positions) {
    const [x, y] = frame.forward(position);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      width = Math.max(width, x);
      height = Math.max(height, y);
    }
  }
  return {width, height};
}

/**
 * Cells of the grid the bindings build over an extent: `makeGridSpec` in
 * catalog-platform3-b-common.ts:203 gives `floor((extent + 2 * padding) / cellSize) + 1` per side.
 */
function countGridCells(
  width: number,
  height: number,
  padding: number,
  cellSize: number,
  extraCellsPerAxis = 0
): number {
  const columns = Math.max(1, Math.floor((width + 2 * padding) / cellSize) + 1 + extraCellsPerAxis);
  const rows = Math.max(1, Math.floor((height + 2 * padding) / cellSize) + 1 + extraCellsPerAxis);
  return columns * rows;
}

/**
 * Smallest cell size (to {@link DATA_DEFAULT_SIGNIFICANT_DIGITS} digits, rounded up) at or above
 * `cellSize` whose grid over the extent has at most `maximumCellCount` cells.
 */
function raiseCellSizeToFit(
  cellSize: number,
  width: number,
  height: number,
  padding: number,
  maximumCellCount: number,
  extraCellsPerAxis = 0
): number {
  // Start from the continuous solution of (w / c) * (h / c) = cap, then step up until it fits.
  let candidate = Math.max(
    cellSize,
    Math.sqrt(((width + 2 * padding) * (height + 2 * padding)) / maximumCellCount)
  );
  candidate = roundToSignificantDigits(candidate, DATA_DEFAULT_SIGNIFICANT_DIGITS, true);
  // Step one unit of the last kept digit at a time, so the result is the smallest value at that
  // precision. The start is close to the answer, so only a few steps are taken.
  for (let step = 0; step < 10000; step++) {
    if (countGridCells(width, height, padding, candidate, extraCellsPerAxis) <= maximumCellCount) {
      return candidate;
    }
    const unit = 10 ** (Math.floor(Math.log10(candidate)) - DATA_DEFAULT_SIGNIFICANT_DIGITS + 1);
    candidate = Number((candidate + unit).toPrecision(15));
  }
  return candidate;
}

/** Leave one percent below the nearest pair distance for float32 seed quantization. */
const CATCHMENT_CELL_DIAGONAL_MARGIN = 0.99;
/** A square's diagonal must remain smaller than the nearest pair distance to separate every seed. */
const CATCHMENT_SPACING_CELL_FACTOR = Math.SQRT1_2 * CATCHMENT_CELL_DIAGONAL_MARGIN;
/** The generator subtracts this tolerance before ceil to avoid an extra cell at exact divisibility. */
const GENERATOR_CELL_BOUNDARY_TOLERANCE = 1e-9;

/** The catchment binding grids all inputs in one frame, so its cap must use their union. */
function getCatchmentGridProfile(
  profilesByInput: Readonly<Record<string, KeplerDatasetProfile | undefined>>
): KeplerDatasetProfile | undefined {
  const profiles = Object.values(profilesByInput).filter(
    (profile): profile is KeplerDatasetProfile => profile !== undefined
  );
  const first = profiles[0];
  if (!first) {
    return undefined;
  }
  const extents = profiles.map(profile => profile.extent ?? profile.spacing?.extent);
  if (extents.some(extent => extent === undefined)) {
    return undefined;
  }
  const bounds = extents as (readonly [number, number, number, number])[];
  const geographic = first.coordinates?.model === 'geographic';
  const west = bounds[0][0];
  const intervals = bounds.map(extent => {
    const start = geographic
      ? west + ((((extent[0] - west + 180) % 360) + 360) % 360) - 180
      : extent[0];
    return [
      start,
      start +
        (geographic && extent[2] < extent[0] ? extent[2] + 360 - extent[0] : extent[2] - extent[0])
    ];
  });
  const minimumX = Math.min(...intervals.map(interval => interval[0]));
  const maximumX = Math.max(...intervals.map(interval => interval[1]));
  return {
    ...first,
    extent: [
      minimumX,
      Math.min(...bounds.map(extent => extent[1])),
      maximumX,
      Math.max(...bounds.map(extent => extent[3]))
    ]
  };
}

/**
 * Resolves every parameter of a binding.
 *
 * Order of precedence: an override; a column bound to a string parameter (`field`); a data rule of
 * the descriptor (`data`), rounded to {@link DATA_DEFAULT_SIGNIFICANT_DIGITS} significant digits,
 * rounded to a whole number for an `integer` parameter and moved into the parameter's
 * `minimum`/`maximum` (the move is recorded in `clamped`); the binding's default (`constant`, or
 * `fallback` when a rule had been asked for and the profile could not give a value).
 *
 * The `coordinateSystem` parameter, when the binding has one with the options `EPSG:4326` and
 * `planar`, follows the coordinate model of the first input profile that states one, so that a
 * distance in metres is not read as planar units. That is recorded as `data`.
 *
 * When `options.maximumCellCount` is given, a `cellSize` derived from the data (the extent rule
 * or the point spacing) is raised, after the steps above, to the smallest value whose grid over the profile's extent
 * (from its spacing statistics) has at most that many cells; the move is recorded in `clamped`
 * with a reason. Neither rule knows the cap: 250 cells on the shorter side of an elongated extent,
 * or a tight cluster that sets the spacing for a wide extent, exceeds a 40,000-cell cap.
 *
 * @throws Error when `fieldBindings` names a parameter that is not a string parameter of the
 * binding, or a joined column list longer than the parameter's `maximumLength`.
 */
export function resolveKeplerOperationParameters(
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  profilesByInput: Readonly<Record<string, KeplerDatasetProfile | undefined>>,
  options: KeplerParameterResolutionOptions = {}
): KeplerResolvedParameters {
  const functionName = 'resolveKeplerOperationParameters';
  const confirmations: string[] = [];
  let maximumCellCount =
    options.maximumCellCount ??
    (['grid-generator', 'curated-emerging-hot-spots'].includes(binding.id)
      ? KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount
      : undefined);
  const fieldBindings = options.fieldBindings ?? {};
  for (const name of Object.keys(options.overrides ?? {})) {
    const message = getCatalogParameterRenameMessage(binding.parameters, name);
    if (message) validateRequirement(getCatalogValidationIdentity(binding.id), false, message);
  }
  const staticDefaults = getCatalogDefaultParameters(binding);
  const overrides = resolveCatalogParameterAliases(
    binding.parameters,
    options.overrides ?? {},
    (name, problem) => {
      validateRequirement(getCatalogValidationIdentity(binding.id), false, `${name} ${problem}`);
      throw new Error('unreachable');
    }
  );
  const specificationsByName = new Map(
    binding.parameters.map(parameter => [parameter.name, parameter])
  );
  for (const name of Object.keys(fieldBindings)) {
    const specification = specificationsByName.get(name);
    if (!specification || specification.kind !== 'string') {
      throw new Error(
        `${functionName} fieldBindings.${name} must name a string parameter of ${binding.id}; its string parameters are ${binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => parameter.name)
          .join(', ')}`
      );
    }
  }

  const policiesByName = new Map(
    descriptor.parameters.map(parameter => [parameter.name, parameter])
  );
  const fieldParameterNames = new Set(
    descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter))
  );
  const coordinateModel = binding.inputs
    .map(input => {
      const profile = profilesByInput[input.name];
      return profile?.coordinates?.model ?? profile?.spacing?.coordinateModel;
    })
    .find(model => model !== undefined);

  const parameters: Record<string, CatalogParameterValue> = {};
  const resolutionsByName = new Map<string, KeplerParameterResolution>();
  const basic: string[] = [];
  const advanced: string[] = [];

  /** The column bound to a role of an input, from a person's binding or an override. */
  const findBoundColumn = (
    inputName: string,
    role: string | undefined
  ): GPUAnalysisColumnProfile | undefined => {
    const field = descriptor.inputs
      .find(input => input.name === inputName)
      ?.fields.find(candidate => candidate.role === role);
    if (!field) {
      return undefined;
    }
    const bound = fieldBindings[field.parameter] ?? overrides[field.parameter];
    const first =
      typeof bound === 'string'
        ? bound.split(',')[0].trim()
        : (bound as readonly unknown[] | undefined)?.[0];
    return typeof first === 'string'
      ? profilesByInput[inputName]?.columns.find(column => column.name === first)
      : undefined;
  };

  const getMeasurementLevel = (
    column: GPUAnalysisColumnProfile
  ): GPUAnalysisMeasurementLevel | undefined =>
    column.measurement ?? guessGPUAnalysisMeasurementLevel(column)?.level;

  // Parameters whose factor depends on another parameter's value are resolved after it.
  const dependencyOf = new Map<string, string>();
  for (const entry of descriptor.parameters) {
    if (entry.defaultPolicy.from === 'data' && entry.defaultPolicy.factorByParameter) {
      dependencyOf.set(entry.name, entry.defaultPolicy.factorByParameter.parameter);
    }
  }
  const order: string[] = [];
  const visiting: string[] = [];
  const visit = (name: string): void => {
    if (order.includes(name)) {
      return;
    }
    if (visiting.includes(name)) {
      throw new Error(
        `${functionName} the descriptor of ${binding.id} has a cycle in factorByParameter: ${[...visiting.slice(visiting.indexOf(name)), name].join(' -> ')}`
      );
    }
    visiting.push(name);
    const dependency = dependencyOf.get(name);
    if (dependency !== undefined && specificationsByName.has(dependency)) {
      visit(dependency);
    }
    visiting.pop();
    order.push(name);
  };
  // Length rules must see the final override, preset or inferred coordinate setting.
  if (specificationsByName.has('coordinateSystem')) {
    visit('coordinateSystem');
  }
  for (const specification of binding.parameters) {
    visit(specification.name);
  }

  const resolveOne = (declaredSpecification: CatalogParameterSpec): void => {
    const specification = {
      ...declaredSpecification,
      defaultValue: staticDefaults[declaredSpecification.name]
    };
    const name = specification.name;
    const policy = policiesByName.get(name);
    const set = (
      value: CatalogParameterValue | undefined,
      resolution: Omit<KeplerParameterResolution, 'parameter' | 'active'>
    ) => {
      if (value !== undefined) parameters[name] = Array.isArray(value) ? [...value] : value;
      resolutionsByName.set(name, {parameter: name, active: true, ...resolution});
    };
    const fallBack = (reason: string, ruleResolution?: GPUAnalysisDefaultResolution) =>
      set(specification.defaultValue, {
        source: 'fallback',
        reason,
        ...(ruleResolution ? {resolution: ruleResolution} : {})
      });

    if (name in overrides) {
      set(overrides[name], {source: 'override'});
      return;
    }
    if (name in fieldBindings) {
      const bound = fieldBindings[name];
      const text = typeof bound === 'string' ? bound : bound.join(',');
      if (text.length > (specification.maximumLength ?? 64)) {
        throw new Error(
          `${functionName} fieldBindings.${name} is ${text.length} characters; ${binding.id} accepts at most ${specification.maximumLength ?? 64}`
        );
      }
      set(text, {source: 'field'});
      return;
    }
    const explicitField = descriptor.inputs
      .flatMap(input => input.fields)
      .find(field => field.parameter === name && field.autoAssign === false);
    if (explicitField) {
      set('', {source: 'constant', reason: 'This column must be selected explicitly'});
      return;
    }
    for (const input of descriptor.inputs) {
      const field = input.fields.find(candidate => candidate.parameter === name);
      const coordinates = profilesByInput[input.name]?.pointColumns;
      const coordinate =
        field?.autoAssign === 'point-longitude'
          ? coordinates?.longitude
          : field?.autoAssign === 'point-latitude'
            ? coordinates?.latitude
            : undefined;
      if (coordinate !== undefined) {
        set(coordinate, {source: 'data', reason: `Declared pointColumns of ${input.name}`});
        return;
      }
    }
    const defaultPolicy = policy?.defaultPolicy;
    if (defaultPolicy?.from === 'time-encoding') {
      const column = findBoundColumn(defaultPolicy.input, defaultPolicy.role);
      if (
        column?.type === 'time' &&
        'timestampEncoding' in column &&
        column.timestampEncoding === 'iso8601'
      ) {
        set('iso8601', {source: 'data', reason: `${column.name} contains ISO timestamp text`});
      } else {
        const reason = column
          ? `Choose ${name} explicitly for ${column.name}: the profile cannot certify its timestamp encoding`
          : undefined;
        set(specification.defaultValue, {
          source: 'constant',
          ...(reason ? {reason, requiresChoice: true} : {})
        });
        if (reason) confirmations.push(reason);
      }
      return;
    }
    if (defaultPolicy?.from === 'preset') {
      const inputName = defaultPolicy.input ?? binding.inputs[0]?.name;
      const profile = inputName === undefined ? undefined : profilesByInput[inputName];
      const geometry = profile?.catalogGeometry ?? (profile?.pointColumns ? 'point' : undefined);
      if (defaultPolicy.whenGeometry !== undefined && geometry !== defaultPolicy.whenGeometry) {
        set(specification.defaultValue, {source: 'constant'});
        return;
      }
      try {
        const normalized = normalizeCatalogParameters(binding, {[name]: defaultPolicy.value});
        set(normalized[name], {source: 'preset', reason: defaultPolicy.reason});
      } catch (error) {
        fallBack(
          `The preset ${JSON.stringify(defaultPolicy.value)} (${defaultPolicy.reason}) is not accepted by ${binding.id}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
      return;
    }
    const modelValue =
      coordinateModel === undefined
        ? undefined
        : mapCoordinateModelToParameter(specification, coordinateModel);
    if (modelValue !== undefined) {
      const stated = binding.inputs
        .map(input => profilesByInput[input.name]?.coordinates)
        .find(coordinates => coordinates !== undefined);
      const guessed = stated?.source === 'guess';
      set(modelValue, {
        source: 'data',
        ...(guessed
          ? {
              guessed: true,
              reason: `The coordinate model is a guess (${stated?.model}${stated?.confidence === 'low' ? ', low confidence' : ''}) from the extent, not a declaration`
            }
          : {})
      });
      return;
    }
    if (defaultPolicy?.from === 'field-measurement') {
      const column = findBoundColumn(defaultPolicy.input, defaultPolicy.role);
      if (!column) {
        fallBack(
          `No column is bound to the role "${defaultPolicy.role}", so its measurement level is unknown`
        );
        return;
      }
      const level = getMeasurementLevel(column);
      const chosen = level === undefined ? undefined : defaultPolicy.values[level];
      if (level === undefined || chosen === undefined) {
        fallBack(
          level === undefined
            ? `The measurement level of ${column.name} is unknown`
            : `No value is listed for ${column.name}, which measures ${level}`
        );
        return;
      }
      set(chosen, {source: 'data', reason: `${column.name} measures ${level}`});
      return;
    }
    if (
      defaultPolicy?.from === 'data' &&
      (specification.kind === 'number' || specification.kind === 'integer')
    ) {
      const profile = profilesByInput[defaultPolicy.input];
      const isColumnRule = defaultPolicy.rule.startsWith('column-');
      const column = isColumnRule
        ? findBoundColumn(defaultPolicy.input, defaultPolicy.role)
        : undefined;
      if (isColumnRule && !column) {
        fallBack(
          `No column is bound to the role "${defaultPolicy.role ?? ''}" of ${defaultPolicy.input}, which ${defaultPolicy.rule} reads`
        );
        return;
      }
      if (defaultPolicy.rule === 'column-sum' && column) {
        if (column.sampleSize !== undefined && column.sampleSize < column.rowCount) {
          fallBack(
            `Cannot derive ${name}: column-sum for ${column.name} uses sampled column statistics; a sampled sum cannot bound the allocation. Profile every value or choose ${name} explicitly`
          );
          return;
        }
        if (column.minimum !== undefined && column.minimum < 0) {
          fallBack(
            `Cannot derive ${name}: ${column.name} contains negative values (minimum ${column.minimum}); its signed column-sum does not bound the allocation rendered from non-negative values`
          );
          return;
        }
      }
      let factor = defaultPolicy.factor ?? 1;
      if (defaultPolicy.factorByParameter) {
        const controlling = parameters[defaultPolicy.factorByParameter.parameter];
        factor *= defaultPolicy.factorByParameter.factors[String(controlling)] ?? 1;
      }
      const resolveRule = (rule: GPUAnalysisDefaultRule, ruleFactor: number) => {
        if (!profile) {
          return undefined;
        }
        const finalModel =
          parameters['coordinateSystem'] === 'planar'
            ? 'planar'
            : parameters['coordinateSystem'] === 'EPSG:4326'
              ? 'geographic'
              : undefined;
        const changesModel =
          finalModel &&
          !rule.startsWith('column-') &&
          (profile.coordinates?.model !== finalModel ||
            (profile.spacing && profile.spacing.coordinateModel !== finalModel));
        const effectiveProfile = changesModel
          ? applyKeplerDatasetProfileDeclarations(profile, {coordinateModel: finalModel})
          : profile;
        const derived = resolveGPUAnalysisDefault(
          rule,
          effectiveProfile,
          ruleFactor,
          column ? {column: column.name} : undefined
        );
        const computedModel = getLengthCoordinateModel(rule, derived);
        if (
          changesModel &&
          !['extent-west', 'extent-south', 'extent-east', 'extent-north'].includes(rule) &&
          (effectiveProfile.spacing?.coordinateModel !== finalModel || computedModel !== finalModel)
        ) {
          return {
            ...derived,
            value: undefined,
            explanation: `Cannot derive ${name}: re-profiling for the chosen coordinate system ${finalModel} provided no spacing in that model; profile the dataset again with its coordinate model declared`
          };
        }
        return derived;
      };
      const resolution = resolveRule(defaultPolicy.rule, factor);
      const accept = (
        rule: GPUAnalysisDefaultRule,
        derived: GPUAnalysisDefaultResolution | undefined,
        reason?: string
      ): boolean => {
        const rawValue = derived?.value;
        if (rawValue === undefined || !Number.isFinite(rawValue) || rawValue <= 0) {
          return false;
        }
        const rounding = KEPLER_DEFAULT_RULE_ROUNDING[rule].direction;
        let value =
          rounding === 'none'
            ? rawValue
            : roundToSignificantDigits(
                rawValue,
                DATA_DEFAULT_SIGNIFICANT_DIGITS,
                rounding === 'up'
              );
        if (specification.kind === 'integer') {
          value = Math.max(1, Math.round(value));
        }
        const {value: clampedValue, clamped} = clampToRange(value, specification);
        set(clampedValue, {
          source: 'data',
          resolution: derived,
          ...(derived && getLengthCoordinateModel(rule, derived)
            ? {coordinateModel: getLengthCoordinateModel(rule, derived)}
            : {}),
          ...(reason ? {reason} : {}),
          ...(clamped ? {clamped} : {})
        });
        return true;
      };
      if (accept(defaultPolicy.rule, resolution)) {
        return;
      }
      // The rule had nothing to give for this data (no clear knee): try the descriptor's second
      // rule, which still follows the data's scale, before the binding's constant.
      if (defaultPolicy.fallback && profile) {
        const second = resolveRule(
          defaultPolicy.fallback.rule,
          defaultPolicy.fallback.factor ?? 1
        )!;
        const both = `${resolution?.explanation ?? `${defaultPolicy.rule} could not be read`}; then ${second.explanation}`;
        if (accept(defaultPolicy.fallback.rule, second, both)) {
          return;
        }
        fallBack(`${both}; neither gave a value`, second);
        return;
      }
      fallBack(
        resolution
          ? resolution.explanation
          : `No profile of ${defaultPolicy.input} is available for ${defaultPolicy.rule}`,
        resolution
      );
      return;
    }
    set(specification.defaultValue, {source: 'constant'});
  };

  for (const specification of binding.parameters) {
    const policy = policiesByName.get(specification.name);
    const level = fieldParameterNames.has(specification.name)
      ? 'basic'
      : (policy?.level ?? 'advanced');
    (level === 'basic' ? basic : advanced).push(specification.name);
  }
  for (const name of order) {
    resolveOne(specificationsByName.get(name) as CatalogParameterSpec);
  }
  // The public Vecchia packers enforce a model-specific numerical floor. Adjust only the
  // generated nugget, after the selected model and sill resolve; explicit settings stay intact
  // and applicability reports any incompatibility before execution.
  if (binding.id.startsWith('validation-vecchia-') && !('nugget' in overrides)) {
    const spatialFloor = getVecchiaNuggetFloor({
      model: parameters['model'] as 'exponential' | 'matern-3/2',
      sill: parameters['sill'] as number
    });
    // Spatial floor is f*sill. Space-time uses total variance: n >= f*(sill+n).
    const floor =
      binding.id === 'validation-vecchia-space-time'
        ? spatialFloor / (1 - VECCHIA_NUGGET_FLOOR_FRACTION)
        : spatialFloor;
    const previous = parameters['nugget'] as number;
    if (floor > previous) {
      parameters['nugget'] = floor;
      const reason = `Model-specific nugget floor for ${parameters['model']} at the resolved sill ${parameters['sill']}${binding.id === 'validation-vecchia-space-time' ? ', using total variance' : ''}`;
      resolutionsByName.set('nugget', {
        parameter: 'nugget',
        active: true,
        source: 'preset',
        reason,
        clamped: {from: previous, to: floor, reason}
      });
    }
  }
  // Each MGTWR component corresponds to one coefficient, including the intercept. Generate
  // unedited lists from the resolved scalar scales and mapped predictors, not the binding's
  // two-entry fixture arrays. Explicit lists are preserved for applicability to validate.
  if (binding.id === 'regression-mgtwr') {
    const predictorNames = String(parameters['predictorProperties'])
      .split(',')
      .map(name => name.trim());
    if (
      predictorNames.length >= 1 &&
      predictorNames.length <= 11 &&
      predictorNames.every(Boolean)
    ) {
      for (const [name, scalar] of [
        ['componentBandwidths', 'bandwidth'],
        ['componentTimeScales', 'timeScale']
      ] as const) {
        if (name in overrides) continue;
        const value = parameters[scalar] as number;
        parameters[name] = Array(predictorNames.length + 1).fill(value);
        resolutionsByName.set(name, {
          parameter: name,
          active: true,
          source: 'preset',
          reason: `One ${scalar} value (${value}) per coefficient: intercept plus ${predictorNames.length} selected predictors`
        });
      }
    }
  }
  // A caller may lower the dot allocation cap; keep a complete non-negative sum within it.
  // For rejected signed totals, an exact maximum bounds the positive allocation conservatively;
  // this remains a fallback requiring mapping, rather than a usable default from the signed sum.
  const dotResolution = resolutionsByName.get('dotValue');
  if (
    binding.id === 'geometry-dot-density' &&
    dotResolution &&
    (dotResolution.source === 'data' || dotResolution.source === 'fallback')
  ) {
    const dotPolicy = policiesByName.get('dotValue')?.defaultPolicy;
    const maximumDots = Number(parameters['maximumDots']);
    const column =
      dotPolicy?.from === 'data' ? findBoundColumn(dotPolicy.input, dotPolicy.role) : undefined;
    const signedFallback =
      dotResolution.source === 'fallback' &&
      column &&
      (column.minimum ?? 0) < 0 &&
      (column.sampleSize === undefined || column.sampleSize >= column.rowCount);
    const allocationMean =
      dotResolution.source === 'data'
        ? column?.mean
        : signedFallback && column.maximum !== undefined
          ? Math.max(0, column.maximum)
          : undefined;
    const total =
      column && allocationMean !== undefined
        ? allocationMean * (column.rowCount - column.nullCount)
        : undefined;
    if (total !== undefined && Number.isFinite(total) && maximumDots > 0) {
      const minimumDotValue = total / maximumDots;
      const previousDotValue = Number(parameters['dotValue']);
      if (minimumDotValue > previousDotValue) {
        const value = roundToSignificantDigits(
          minimumDotValue,
          DATA_DEFAULT_SIGNIFICANT_DIGITS,
          true
        );
        const specification = specificationsByName.get('dotValue')!;
        const clamped = clampToRange(value, specification);
        parameters['dotValue'] = clamped.value;
        if (signedFallback) {
          dotResolution.reason +=
            '; the conservative fallback bounds each non-missing row by the complete column maximum; correct the negative values or choose the value per dot explicitly';
        }
        dotResolution.clamped = {
          from: previousDotValue,
          to: clamped.value,
          reason: `The complete column ${signedFallback ? 'maximum bound' : 'total'} must fit the chosen maximumDots allocation of ${maximumDots}`
        };
        if (clamped.value < minimumDotValue) {
          confirmations.push(
            `dotValue needs at least ${minimumDotValue} to fit maximumDots ${maximumDots}, beyond its allowed range`
          );
        }
      }
    }
  }
  const resolutions = binding.parameters.map(
    specification => resolutionsByName.get(specification.name) as KeplerParameterResolution
  );
  const inactive: string[] = [];
  for (const resolution of resolutions) {
    const activeWhen = policiesByName.get(resolution.parameter)?.activeWhen;
    if (activeWhen) {
      resolution.active = evaluateGPUAnalysisParameterCondition(activeWhen, parameters);
      if (!resolution.active) {
        inactive.push(resolution.parameter);
      }
    }
  }

  const emergingHotSpots = binding.id === 'curated-emerging-hot-spots';
  // With automatic temporal coverage, reserve the largest supported slice count. This is a
  // conservative bound until the recipe maps the exact calendar range during preparation.
  if (emergingHotSpots) {
    const sliceCount = Number(parameters['sliceCount']) || EMERGING_HOT_SPOTS_LIMITS.slices;
    maximumCellCount = Math.min(
      maximumCellCount ?? Infinity,
      EMERGING_HOT_SPOTS_LIMITS.cells,
      Math.floor(EMERGING_HOT_SPOTS_LIMITS.bins / sliceCount)
    );
  }
  // Alignment of the emerging grid to cell-size multiples can add a cell on each axis, unlike
  // the other bindings' grids that start at the data minimum. Bound that extra cell explicitly.
  const extraCellsPerAxis = emergingHotSpots ? 1 : 0;
  // Grid size limits: the binding's cell cap and, on the CPU engine, its work cap.
  const cpuCap =
    getKeplerEffectiveEngine(binding, options.executionEngine) === 'cpu'
      ? KEPLER_CPU_WORK_CAPS[binding.id]
      : undefined;
  if (maximumCellCount !== undefined || cpuCap?.kind === 'cells-times-rows') {
    for (const resolution of resolutions) {
      const policy = policiesByName.get(resolution.parameter)?.defaultPolicy;
      if (
        resolution.source !== 'data' ||
        policy?.from !== 'data' ||
        resolution.parameter !== 'cellSize'
      ) {
        continue;
      }
      const profile =
        binding.id === 'recipe-straight-line-catchments'
          ? getCatchmentGridProfile(profilesByInput)
          : profilesByInput[policy.input];
      const current = parameters[resolution.parameter];
      const coordinateSystem =
        typeof parameters['coordinateSystem'] === 'string'
          ? parameters['coordinateSystem']
          : profile?.coordinates?.model === 'planar'
            ? 'planar'
            : 'EPSG:4326';
      const size = profile ? getKeplerGridFrameExtent(profile, coordinateSystem) : undefined;
      if (!profile || !size || typeof current !== 'number') {
        continue;
      }
      // The tighter of the cell cap and the CPU work cap (cells times rows).
      const limits: {count: number; label: string}[] = [];
      if (maximumCellCount !== undefined) {
        limits.push({
          count: maximumCellCount,
          label: `${binding.id} allows at most ${maximumCellCount.toLocaleString('en-US')} cells${emergingHotSpots ? ' within its cell and cube-bin limits' : ''}`
        });
      }
      if (cpuCap?.kind === 'cells-times-rows' && profile.rowCount > 0) {
        const count = Math.max(1, Math.floor(cpuCap.limit / profile.rowCount));
        limits.push({
          count,
          label: `on the CPU engine ${binding.id} allows at most ${count.toLocaleString('en-US')} cells for ${profile.rowCount.toLocaleString('en-US')} rows (cells times rows within ${cpuCap.limit.toExponential(0)})`
        });
      }
      const tightest = limits.reduce((best, limit) => (limit.count < best.count ? limit : best));
      const padding =
        typeof parameters['padding'] === 'number' && parameters['padding'] > 0
          ? parameters['padding']
          : 0;
      const requiredSize = raiseCellSizeToFit(
        current,
        size.width,
        size.height,
        padding,
        tightest.count,
        extraCellsPerAxis
      );
      const raised = Math.min(
        requiredSize,
        specificationsByName.get(resolution.parameter)?.maximum ?? Infinity
      );
      if (raised < requiredSize) {
        confirmations.push(
          `${binding.id} cellSize needs at least ${requiredSize} to fit ${tightest.count} cells, beyond its allowed maximum; reduce the input extent`
        );
      }
      if (raised !== current) {
        const impliedCount = countGridCells(
          size.width,
          size.height,
          padding,
          current,
          extraCellsPerAxis
        );
        parameters[resolution.parameter] = raised;
        resolution.clamped = {
          from: resolution.clamped?.from ?? current,
          to: raised,
          reason: `A cell size of ${current} would make a grid of ${impliedCount.toLocaleString('en-US')} cells over this extent in the operation's own local frame; ${tightest.label}`
        };
      }
    }
  }
  if (emergingHotSpots && maximumCellCount !== undefined) {
    const profile = profilesByInput['points'];
    const extent = profile
      ? getKeplerGridFrameExtent(profile, String(parameters['coordinateSystem']))
      : undefined;
    if (!extent) {
      confirmations.push(
        `${binding.id} needs a complete extent to check cellSize against its cell and cube-bin limits`
      );
    } else if (
      countGridCells(
        extent.width,
        extent.height,
        0,
        Number(parameters['cellSize']),
        extraCellsPerAxis
      ) > maximumCellCount
    ) {
      confirmations.push(
        `${binding.id} cellSize may exceed the ${maximumCellCount} cells allowed by its cell and cube-bin limits; increase cellSize or reduce sliceCount`
      );
    }
  }
  if (binding.id === 'grid-generator' && maximumCellCount !== undefined) {
    const profile = profilesByInput['extent'];
    const extent = profile?.extent ?? profile?.spacing?.extent;
    if (extent) {
      // The generator reads raw coordinates, rather than unwrapping or projecting them. For a
      // wrapped profile the full longitude range safely bounds both -180..180 and 0..360 inputs.
      const width = extent[0] > extent[2] ? 360 : extent[2] - extent[0];
      const height = extent[3] - extent[1];
      const countCells = () =>
        Math.max(
          1,
          Math.ceil(width / Number(parameters['cellWidth']) - GENERATOR_CELL_BOUNDARY_TOLERANCE)
        ) *
        Math.max(
          1,
          Math.ceil(height / Number(parameters['cellHeight']) - GENERATOR_CELL_BOUNDARY_TOLERANCE)
        );
      const adjustable = resolutions.filter(
        resolution =>
          ['cellWidth', 'cellHeight'].includes(resolution.parameter) && resolution.source === 'data'
      );
      while (countCells() > maximumCellCount && adjustable.length > 0) {
        let moved = false;
        for (const resolution of adjustable) {
          const current = Number(parameters[resolution.parameter]);
          const increment =
            10 ** (Math.floor(Math.log10(current)) - DATA_DEFAULT_SIGNIFICANT_DIGITS + 1);
          const maximum = specificationsByName.get(resolution.parameter)?.maximum ?? Infinity;
          const next = Math.min(
            maximum,
            Number((current + increment).toPrecision(DATA_DEFAULT_SIGNIFICANT_DIGITS))
          );
          if (next > current) {
            parameters[resolution.parameter] = next;
            resolution.clamped = {
              from: resolution.clamped?.from ?? current,
              to: next,
              reason: `The raw-coordinate grid must contain at most ${maximumCellCount.toLocaleString('en-US')} cells after rounding`
            };
            moved = true;
          }
        }
        if (!moved) {
          break;
        }
      }
      if (countCells() > maximumCellCount) {
        confirmations.push(
          `grid-generator cellWidth and cellHeight produce ${countCells().toLocaleString('en-US')} cells; increase their sizes to stay within ${maximumCellCount.toLocaleString('en-US')}`
        );
      }
    }
  }
  if (binding.id === 'recipe-straight-line-catchments') {
    const facilities = profilesByInput['facilities'];
    const union = getCatchmentGridProfile(profilesByInput);
    const cellSize = parameters['cellSize'];
    const spacing = facilities?.spacing;
    const minimumDistance = spacing?.nearestNeighbor?.minimum;
    const resolution = resolutions.find(entry => entry.parameter === 'cellSize');
    const seeds = facilities?.catchmentSeeds;
    const observations = profilesByInput['observations'];
    const exactZeroCollisions =
      seeds !== undefined &&
      seeds.facilityRevision === facilities?.revision &&
      seeds.observationsDatasetId === observations?.datasetId &&
      seeds.observationsRevision === observations?.revision &&
      seeds.coordinateSystem === parameters['coordinateSystem'] &&
      seeds.cellSize === cellSize &&
      seeds.collisionCount === 0;
    if (
      facilities &&
      union &&
      typeof cellSize === 'number' &&
      facilities.rowCount > 1 &&
      !exactZeroCollisions
    ) {
      const duplicateCount = spacing?.duplicateLocationCount ?? 0;
      const safeSize =
        minimumDistance && minimumDistance > 0
          ? minimumDistance * CATCHMENT_SPACING_CELL_FACTOR
          : undefined;
      const size = getKeplerGridFrameExtent(union, String(parameters['coordinateSystem']));
      const canGuarantee =
        parameters['coordinateSystem'] === 'planar' &&
        spacing &&
        !spacing.sampled &&
        duplicateCount === 0;
      const minimumSize = specificationsByName.get('cellSize')?.minimum ?? 0;
      if (
        canGuarantee &&
        safeSize !== undefined &&
        safeSize >= minimumSize &&
        cellSize > safeSize &&
        size &&
        maximumCellCount !== undefined &&
        countGridCells(size.width, size.height, 0, safeSize) <= maximumCellCount &&
        resolution?.source === 'data'
      ) {
        parameters['cellSize'] = safeSize;
        resolution.clamped = {
          from: resolution.clamped?.from ?? cellSize,
          to: safeSize,
          reason:
            'Keep distinct facilities in separate seed cells: cell diagonal is smaller than the measured minimum facility spacing'
        };
      } else if (
        duplicateCount > 0 ||
        !canGuarantee ||
        safeSize === undefined ||
        cellSize > safeSize
      ) {
        const facilityExtent = facilities.extent ?? spacing?.extent;
        const unionExtent = union.extent;
        let collisionLowerBound = duplicateCount;
        if (parameters['coordinateSystem'] === 'planar' && facilityExtent && unionExtent) {
          const inverse = Math.fround(1 / Math.fround(cellSize));
          const cellIndex = (value: number, origin: number) =>
            Math.floor(Math.fround(Math.fround(value - origin) * inverse));
          const columns =
            cellIndex(facilityExtent[2], unionExtent[0]) -
            cellIndex(facilityExtent[0], unionExtent[0]) +
            1;
          const rows =
            cellIndex(facilityExtent[3], unionExtent[1]) -
            cellIndex(facilityExtent[1], unionExtent[1]) +
            1;
          collisionLowerBound = Math.max(collisionLowerBound, facilities.rowCount - columns * rows);
        }
        confirmations.push(
          `Facility seed collisions at cellSize ${cellSize}: ${collisionLowerBound > 0 ? `at least ${collisionLowerBound}` : 'an unknown number of'} facilities share or may share a seed cell. The profile has no facility positions for an exact count; choose a finer resolution within the cell cap or accept empty catchments`
        );
      }
    }
  }
  return {parameters, resolutions, basic, advanced, inactive, confirmations};
}
