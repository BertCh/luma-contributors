// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';

import {
  defineGPUSpatialParameterSchema,
  type GPUSpatialParameterSchema
} from '../../analysis-core/contracts/index';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveKernelShapeName} from '../shared/kernel-shape-name';

/** Number of float32 elements in a {@link GPUNeighborSearch} parameter view. */
export const GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH = 12;

/**
 * Words of a {@link GPUNeighborSearch} parameter view with `times`: the base layout, then
 * `timeRadius`, `minimumTime` and `maximumTime`.
 */
export const GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH = 15;

/**
 * Default `distanceFloor` of every inverse-distance weight in the library
 * (`GPUNeighborSearch` and `GPUNearestFeatureWeights`), in the length unit of the distances:
 * `w = max(d, distanceFloor)^-power`, so two coincident points (`d = 0`) weigh
 * `distanceFloor^-power`, `1e6` at the default power 1. libpysal divides by zero there.
 *
 * @internal
 */
export const GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR = 1e-6;

/** Declarative layout of the packed neighbor-search view. */
export const GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA: GPUSpatialParameterSchema<'float32'> =
  defineGPUSpatialParameterSchema({
    id: 'neighbor-search',
    format: 'float32',
    wordLength: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    renamedFields: {radius: 'maximumDistance', weightKind: 'weightType'},
    fields: [
      {name: 'minimumX', format: 'float32', wordOffset: 0, defaultValue: 0, dynamic: true},
      {name: 'minimumY', format: 'float32', wordOffset: 1, defaultValue: 0, dynamic: true},
      {name: 'maximumX', format: 'float32', wordOffset: 2, defaultValue: 0, dynamic: true},
      {name: 'maximumY', format: 'float32', wordOffset: 3, defaultValue: 0, dynamic: true},
      {
        name: 'maximumDistance',
        format: 'float32',
        wordOffset: 4,
        defaultValue: Infinity,
        allowNonFinite: true,
        units: 'position-units',
        dynamic: true
      },
      {name: 'weightType', format: 'float32', wordOffset: 5, defaultValue: 0, dynamic: true},
      {name: 'power', format: 'float32', wordOffset: 6, defaultValue: 1, dynamic: true},
      {
        name: 'distanceFloor',
        format: 'float32',
        wordOffset: 7,
        defaultValue: GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR,
        minimum: 0,
        units: 'position-units',
        dynamic: true
      },
      {name: 'kernel', format: 'float32', wordOffset: 8, defaultValue: 1, dynamic: true},
      {
        name: 'rowStandardize',
        format: 'float32',
        wordOffset: 9,
        defaultValue: 0,
        minimum: 0,
        maximum: 1,
        dynamic: true
      }
    ]
  });

/** Encoded `weightType` values stored in parameter slot 5. */
export const GPU_NEIGHBOR_SEARCH_WEIGHT_TYPE = Object.freeze({
  /** `w = 1`. */
  binary: 0,
  /** `w = max(d, distanceFloor)^-power`. */
  'inverse-distance': 1,
  /** `w = K(d / bandwidth)` for the selected kernel. */
  kernel: 2
} as const);

/** Encoded `kernel` values stored in parameter slot 8. */
export const GPU_NEIGHBOR_SEARCH_KERNEL = {
  /** `exp(-z^2 / 2) / sqrt(2 pi)`. */
  gaussian: 0,
  /** `1 - z`. */
  triangular: 1,
  /** `3/4 (1 - z^2)`, PySAL's `quadratic`. */
  epanechnikov: 2,
  /** `15/16 (1 - z^2)^2`, PySAL's `quartic`. */
  bisquare: 3,
  /** Alias of bisquare: `15/16 (1 - z^2)^2`. */
  quartic: 3,
  /** `1/2`. */
  uniform: 4
} as const;

/** Shared weight-rule spelling; nearest-feature weights support its first two values. */
export type GPUNeighborSearchWeightType = keyof typeof GPU_NEIGHBOR_SEARCH_WEIGHT_TYPE;

/** Kernel profile used when `weightType` is `'kernel'`. */
export type GPUNeighborSearchKernel = keyof typeof GPU_NEIGHBOR_SEARCH_KERNEL;

/** Per-frame settings of {@link GPUNeighborSearch}, packed by {@link getGPUNeighborSearchParameterValues}. */
export type GPUNeighborSearchSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and self-join rows outside
   * it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor,
   * so it must cover every target of interest (count the dropped rows with `invalidCount`). The
   * queries of a cross join are not clipped by it: a query outside the bounds still finds
   * its exact neighbors (great-circle latitude must be within +-90). For a great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude,
   * maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world).
   */
  bounds: readonly [number, number, number, number];
  /** Distance band in spatial-context length units; finite positive float32 in radius mode. */
  searchRadius?: number;
  /** Optional kNN distance cutoff in spatial-context length units; omitted means unbounded. */
  maximumDistance?: number;
  /** Weight rule, defaults to `'binary'`. */
  weightType?: GPUNeighborSearchWeightType;
  /** Inverse-distance exponent. Defaults to 1. */
  power?: number;
  /**
   * Inverse distance uses `max(d, distanceFloor)`, in the length unit of the distances. Defaults
   * to `1e-6`, the default of `GPUNearestFeatureWeights` too, so coincident points (`d = 0`) get
   * the finite weight `distanceFloor^-power` (`1e6` at power 1). An explicit 0 removes the floor:
   * coincident points then get weight 0, because their weight is not finite, and a zero-weight
   * slot is not a neighbor. Under the default, a coincident neighbor outweighs every ordinary
   * neighbor by many orders of magnitude and takes nearly all of a row-standardized row; pass a
   * floor on the scale of the data when positions repeat (see the class documentation).
   */
  distanceFloor?: number;
  /** Kernel profile for `weightType: 'kernel'`. Defaults to `'triangular'`. */
  kernel?: GPUNeighborSearchKernel;
  /** Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. */
  rowStandardize?: boolean;
  /**
   * Time window of a space-time search, in the `temporalContext` unit: a neighbor satisfies
   * `|t_target - t_query| <= timeRadius`. Non-positive or non-finite makes every row empty.
   * Required with `times`, and selects the
   * {@link GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH}-word layout.
   */
  timeRadius?: number;
  /**
   * `[minimumTime, maximumTime]` covered by the index. Rows outside it are invalid, like rows
   * outside `bounds`. Required with `timeRadius`.
   */
  timeBounds?: readonly [number, number];
};

/** Mode-aware settings; the original settings-only overload remains available. */
export type GPUNeighborSearchModeSettings = GPUNeighborSearchSettings &
  (
    | {mode: 'radius'; searchRadius: number; maximumDistance?: never}
    | {mode: 'knn'; searchRadius?: never; maximumDistance?: number}
  );

/** Validates and defaults the weight rule. @internal */
export function resolveNeighborSearchWeightType(
  settings: Pick<GPUNeighborSearchSettings, 'weightType'>
): GPUNeighborSearchWeightType {
  const {weightType} = settings;
  validateRequirement(
    'GPUNeighborSearch',
    !('weightKind' in settings),
    'weightKind was renamed; use weightType'
  );
  validateRequirement(
    'GPUNeighborSearch',
    weightType === undefined ||
      (typeof weightType === 'string' &&
        Object.hasOwn(GPU_NEIGHBOR_SEARCH_WEIGHT_TYPE, weightType)),
    `weightType received ${String(weightType)}; expected binary, inverse-distance or kernel`
  );
  return weightType ?? 'binary';
}

/**
 * The weight transform {@link GPUNeighborSearchSettings} select, for
 * `GPUNeighborSearchProps.weightTransform`: `'row'` with `rowStandardize`, else `'binary'`,
 * `'raw'` (inverse distance) or `'kernel'` from the resolved `weightType`.
 */
export function getGPUNeighborSearchWeightTransform(
  settings: Pick<GPUNeighborSearchSettings, 'weightType' | 'rowStandardize'>
): 'binary' | 'raw' | 'kernel' | 'row' {
  const weightType = resolveNeighborSearchWeightType(settings);
  if (settings.rowStandardize) {
    return 'row';
  }
  return weightType === 'inverse-distance' ? 'raw' : weightType === 'kernel' ? 'kernel' : 'binary';
}

/**
 * Packs {@link GPUNeighborSearchSettings} into the float32 layout read by the kernels.
 *
 * Layout: `[minX, minY, maxX, maxY, maximumDistance, weightType, power, distanceFloor, kernel,
 * rowStandardize, 0, 0]`, followed by `[timeRadius, minimumTime, maximumTime]` when `timeRadius`
 * is given.
 */
export function getGPUNeighborSearchParameterValues(
  settings: GPUNeighborSearchModeSettings
): Float32Array;
/** Legacy settings-only overload; use the mode-aware overload to validate the distance band. */
export function getGPUNeighborSearchParameterValues(
  settings: GPUNeighborSearchSettings & {mode?: never}
): Float32Array;
export function getGPUNeighborSearchParameterValues(
  settings: GPUNeighborSearchSettings & {mode?: 'radius' | 'knn'}
): Float32Array {
  const {mode, searchRadius, maximumDistance} = settings;
  validateRequirement(
    'GPUNeighborSearch',
    !('radius' in settings),
    mode === 'radius'
      ? 'radius was renamed; use searchRadius'
      : mode === 'knn'
        ? 'radius was renamed; use maximumDistance'
        : 'radius was renamed; use searchRadius in radius mode or maximumDistance in kNN mode'
  );
  validateRequirement(
    'GPUNeighborSearch',
    mode === undefined || mode === 'radius' || mode === 'knn',
    `mode received ${String(mode)}; expected radius or knn`
  );
  validateRequirement(
    'GPUNeighborSearch',
    !(searchRadius !== undefined && maximumDistance !== undefined) &&
      !(mode === 'radius' && maximumDistance !== undefined) &&
      !(mode === 'knn' && searchRadius !== undefined),
    `searchRadius ${String(searchRadius)} and maximumDistance ${String(maximumDistance)} must match mode ${String(mode)}`
  );
  const distance = searchRadius ?? maximumDistance ?? Infinity;
  if (mode === 'radius' || searchRadius !== undefined) {
    validateRequirement(
      'GPUNeighborSearch',
      Math.fround(distance) > 0 && Number.isFinite(Math.fround(distance)),
      `searchRadius must pack as a finite positive float32; received ${String(distance)}; supply searchRadius in radius mode`
    );
  }
  if (maximumDistance !== undefined) {
    validateRequirement(
      'GPUNeighborSearch',
      maximumDistance === Infinity ||
        (Math.fround(maximumDistance) > 0 && Number.isFinite(Math.fround(maximumDistance))),
      `maximumDistance must be positive finite float32 or Infinity; received ${String(maximumDistance)}`
    );
  }
  if (
    settings.spatialContext?.coordinateSpace !== 'longitude-latitude' &&
    Number.isFinite(distance) &&
    distance > 0
  ) {
    const packedDistance = Math.fround(distance);
    const prop = searchRadius !== undefined ? 'searchRadius' : 'maximumDistance';
    validateRequirement(
      'GPUNeighborSearch',
      Number.isFinite(Math.fround(packedDistance * packedDistance)),
      `${prop} square must remain finite in float32 for planar search; received ${distance}; use a smaller distance or rescale planar positions`
    );
  }
  if ((settings.timeRadius === undefined) !== (settings.timeBounds === undefined)) {
    throw new Error('GPUNeighborSearch timeRadius and timeBounds must be given together');
  }
  const values = new Float32Array(
    settings.timeRadius === undefined
      ? GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
      : GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
  );
  if (settings.timeRadius !== undefined && settings.timeBounds) {
    values[12] = settings.timeRadius;
    values[13] = settings.timeBounds[0];
    values[14] = settings.timeBounds[1];
  }
  const [minimumX, minimumY, maximumX, maximumY] = settings.bounds;
  const weightType = resolveNeighborSearchWeightType(settings);
  const kernel = resolveKernelShapeName('GPUNeighborSearch', settings.kernel ?? 'triangular', [
    'gaussian',
    'triangular',
    'epanechnikov',
    'quartic',
    'uniform'
  ] as const);
  warnGPUShortLongitudeLatitudeDistance('GPUNeighborSearch', settings.spatialContext, distance);
  values[0] = minimumX;
  values[1] = minimumY;
  values[2] = maximumX;
  values[3] = maximumY;
  values[4] = distance;
  values[5] = GPU_NEIGHBOR_SEARCH_WEIGHT_TYPE[weightType];
  values[6] = settings.power ?? 1;
  values[7] = settings.distanceFloor ?? GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR;
  values[8] = GPU_NEIGHBOR_SEARCH_KERNEL[kernel];
  values[9] = settings.rowStandardize ? 1 : 0;
  return values;
}
