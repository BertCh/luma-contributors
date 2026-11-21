// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR,
  type VariogramModelType
} from '../../dataframe-analysis/pair-statistics/variogram-model';
import {
  getGneitingSpatialScale,
  getGneitingTemporalScale,
  type SpaceTimeVariogramModel,
  type SpaceTimeVariogramStructure
} from '../../dataframe-analysis/pair-statistics/space-time-variogram-model';

export {MATERN_3_2_RANGE_FACTOR, MATERN_5_2_RANGE_FACTOR};

/** Covariance families of {@link GPUPredictionCovariance}. */
export type GPUPredictionCovarianceModel = VariogramModelType | 'matern';

/** Supported closed-form Matern smoothness values. */
export type GPUPredictionCovarianceSmoothness = 0.5 | 1.5 | 2.5;

/**
 * Shared isotropic covariance `C(h) = sill * correlation(h / range) + nugget` at coincident
 * positions. `range` is the effective range of `fitVariogramModel`: the exponential, Gaussian and
 * Matern correlations fall to `exp(-3)` (about 5%) at `range`, and the spherical reaches 0 there.
 */
export type GPUPredictionCovariance = {
  /** Family. `'matern'` needs `smoothness`; `'matern-3/2'` and `'matern-5/2'` do not. */
  model: GPUPredictionCovarianceModel;
  /** Effective range, positive. */
  range: number;
  /** Partial sill, at least 0. */
  sill: number;
  /** Process microscale variance; observation error is separately supplied per observation. */
  nugget?: number;
  /** Matern smoothness `nu`: `0.5` (the exponential model), `1.5` or `2.5`. */
  smoothness?: GPUPredictionCovarianceSmoothness;
};

/**
 * Below this argument the Matern 3/2 and 5/2 correlations are evaluated by a Taylor series instead
 * of the closed form. At `a < 0.1` the series terms beyond `a^6` are below 2e-10, while the f32 form
 * `(1 + a) exp(-a)` multiplies two rounded factors of nearly equal magnitude: it can round above
 * 1 and loses the a^2 structure of `1 - correlation`, which is what separates near-duplicate rows.
 *
 * @internal
 */
export const MATERN_SERIES_ARGUMENT_LIMIT = 0.1;

/**
 * Model codes of the per-frame covariance layout (`GPU_KRIGING_PARAMETER_LENGTH` uses the first
 * three): 0 spherical, 1 exponential (and Matern 1/2), 2 Gaussian, 3 Matern 3/2, 4 Matern 5/2.
 *
 * @internal
 */
export const MAXIMUM_PREDICTION_COVARIANCE_MODEL_CODE = 4;

/**
 * Returns the model code of the per-frame layout.
 *
 * @internal
 */
export function getPredictionCovarianceModelCode(model: GPUPredictionCovariance): number {
  switch (model.model) {
    case 'spherical':
      return 0;
    case 'exponential':
      return 1;
    case 'gaussian':
      return 2;
    case 'matern-3/2':
      return 3;
    case 'matern-5/2':
      return 4;
    case 'matern':
      switch (model.smoothness) {
        case 0.5:
          return 1;
        case 1.5:
          return 3;
        case 2.5:
          return 4;
        default:
          throw new Error(
            'getPredictionCovarianceModelCode covariance.smoothness must be 0.5, 1.5 or 2.5 for matern'
          );
      }
    default:
      throw new Error(
        'getPredictionCovarianceModelCode covariance.model must be a supported covariance family'
      );
  }
}

/** @internal */
export function validatePredictionCovariance(
  model: GPUPredictionCovariance,
  id: string = 'PredictionCovariance',
  field: string = 'covariance'
): void {
  validateRequirement(
    id,
    ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2', 'matern'].includes(
      model.model
    ),
    `${field}.model must be spherical, exponential, gaussian, matern-3/2, matern-5/2 or matern`
  );
  validateRequirement(
    id,
    model.model !== 'matern' || [0.5, 1.5, 2.5].includes(model.smoothness ?? -1),
    `${field}.smoothness must be 0.5, 1.5 or 2.5 for matern`
  );
  validateRequirement(
    id,
    Number.isFinite(model.range) &&
      Number.isFinite(Math.fround(model.range)) &&
      Math.fround(model.range) > 0,
    `${field}.range must be finite and positive after float32 encoding; received ${model.range}`
  );
  validateRequirement(
    id,
    Number.isFinite(model.sill) && Number.isFinite(Math.fround(model.sill)) && model.sill >= 0,
    `${field}.sill must be finite and nonnegative after float32 encoding; received ${model.sill}`
  );
  validateRequirement(
    id,
    Number.isFinite(model.nugget ?? 0) &&
      Number.isFinite(Math.fround(model.nugget ?? 0)) &&
      (model.nugget ?? 0) >= 0,
    `${field}.nugget must be finite and nonnegative after float32 encoding; received ${model.nugget ?? 0}`
  );
  validateRequirement(
    id,
    model.sill + (model.nugget ?? 0) > 0,
    `${field}.sill plus ${field}.nugget must be positive`
  );
}

/** WGSL statements declaring `maternArgument`, and an expression in it for a Matern correlation. */
function getMaternCorrelationWGSL(
  rangeFactor: number,
  series: string,
  closedForm: string
): {statements: string; expression: string} {
  return {
    statements: `let maternArgument = ${getWGSLFloatLiteral(rangeFactor)} * scaled;`,
    expression: `select(${closedForm}, ${series}, maternArgument < ${getWGSLFloatLiteral(MATERN_SERIES_ARGUMENT_LIMIT)})`
  };
}

/**
 * WGSL for one model code: optional statements, then a correlation expression over `scaled`.
 *
 * This is the covariance form of the model family. The semivariogram form (`1 - correlation`,
 * used by `GPUKriging` and the variogram fit) is `VARIOGRAM_MODEL_SHAPE_WGSL` in
 * `dataframe-analysis/pair-statistics/variogram-model.ts`, with the same model codes, range
 * factors and Matern series coefficients. The two are separate on purpose: each form keeps its
 * digits at a different end (a shape at short lags, a correlation at long lags), so neither can be
 * computed as one minus the other in float32. The exponential and Gaussian correlations need no
 * series here: `exp(-x)` near 1 is already the nearest float32 to the correlation.
 */
function getCorrelationWGSL(modelCode: number): {statements: string; expression: string} {
  switch (modelCode) {
    case 0:
      return {
        statements: '',
        expression:
          'select(1.0 - 1.5 * scaled + 0.5 * scaled * scaled * scaled, 0.0, scaled >= 1.0)'
      };
    case 1:
      return {statements: '', expression: 'exp(-3.0 * scaled)'};
    case 2:
      return {statements: '', expression: 'exp(-3.0 * scaled * scaled)'};
    case 3:
      // 1 - a^2 (1/2 - a (1/3 - a (1/8 - a (1/30 - a / 144)))).
      return getMaternCorrelationWGSL(
        MATERN_3_2_RANGE_FACTOR,
        `1.0 - maternArgument * maternArgument * (${getWGSLFloatLiteral(1 / 2)} - maternArgument * (${getWGSLFloatLiteral(1 / 3)} - maternArgument * (${getWGSLFloatLiteral(1 / 8)} - maternArgument * (${getWGSLFloatLiteral(1 / 30)} - maternArgument * ${getWGSLFloatLiteral(1 / 144)}))))`,
        '(1.0 + maternArgument) * exp(-maternArgument)'
      );
    case 4:
      // 1 - a^2 (1/6 - a^2 (1/24 - a (1/45 - a / 144))).
      return getMaternCorrelationWGSL(
        MATERN_5_2_RANGE_FACTOR,
        `1.0 - maternArgument * maternArgument * (${getWGSLFloatLiteral(1 / 6)} - maternArgument * maternArgument * (${getWGSLFloatLiteral(1 / 24)} - maternArgument * (${getWGSLFloatLiteral(1 / 45)} - maternArgument * ${getWGSLFloatLiteral(1 / 144)})))`,
        `(1.0 + maternArgument + maternArgument * maternArgument * ${getWGSLFloatLiteral(1 / 3)}) * exp(-maternArgument)`
      );
    default:
      throw new Error('getCorrelationWGSL modelCode must be an integer in [0, 4]');
  }
}

/**
 * `fn covariance(first, second) -> f32` with the model compiled in as literals.
 *
 * @internal
 */
export function getPredictionCovarianceWGSL(model: GPUPredictionCovariance): string {
  const {statements, expression} = getCorrelationWGSL(getPredictionCovarianceModelCode(model));
  return `fn covariance(first: vec2<f32>, second: vec2<f32>) -> f32 {
    let scaled = distance(first, second) / ${getWGSLFloatLiteral(model.range)};
    ${statements}
    return ${getWGSLFloatLiteral(model.sill)} * (${expression}) + select(0.0, ${getWGSLFloatLiteral(model.nugget ?? 0)}, all(first == second));
  }`;
}

/** Number of float32 elements in a {@link GPULocalKriging} parameter view for one structure. */
export const GPU_LOCAL_KRIGING_PARAMETER_LENGTH = 8;

/**
 * Number of float32 elements in a {@link GPULocalKriging} parameter view.
 *
 * @param structureCount Covariance structures, 1..4.
 * @param variableCount Co-kriging variables, 2..4, or 1 for the univariate modes.
 */
export function getGPULocalKrigingParameterLength(
  structureCount: number = 1,
  variableCount: number = 1
): number {
  return 4 + 4 * structureCount + (variableCount > 1 ? structureCount * variableCount ** 2 : 0);
}

/** Per-frame settings of {@link GPULocalKriging}. */
type LocalKrigingSettingsOptions = {
  /** Covariance of the univariate modes and of the two-by-two `coregionalization` shortcut. */
  covariance?: GPUPredictionCovariance;
  /** Two-by-two row-major loading matrix `L`, with `B = L L^T`, as in the contributor props. */
  coregionalization?: readonly [number, number, number, number];
  /** One to four covariance structures with row-major square loading factors. */
  coregionalizationModels?: readonly {
    covariance: GPUPredictionCovariance;
    loadings: readonly number[];
  }[];
  /** Co-kriging variables, 2..4. Defaults to 2 when loadings are supplied. */
  variableCount?: number;
};

/** At least one covariance source is required; complete legacy option bags remain valid. */
export type GPULocalKrigingSettings = LocalKrigingSettingsOptions &
  (
    | {covariance: GPUPredictionCovariance}
    | {coregionalizationModels: NonNullable<LocalKrigingSettingsOptions['coregionalizationModels']>}
  );

/**
 * Packs per-frame {@link GPULocalKriging} covariance parameters, so a parameter sweep writes one
 * buffer instead of recompiling the kernel.
 *
 * Layout (float32): `[covarianceScale, 0, 0, 0]`, then four values for each of `structureCount`
 * structures `[modelCode, nugget, sill, range]` (model code 0 spherical, 1 exponential or Matern
 * 1/2, 2 Gaussian, 3 Matern 3/2, 4 Matern 5/2), then for co-kriging `variableCount^2` coefficients
 * `B = L L^T` row-major for each structure. `covarianceScale` is the largest structure variance
 * times its largest cross loading; the system is divided by it so entries stay O(1) next to unit
 * drift rows. An initialized model whose float32 scale is zero uses scale 1, preserving valid
 * zero-loading process covariances. A zero `covarianceScale` in an unwritten buffer still gives
 * status 1 for every query.
 *
 * @param settings The covariance model, one per structure when co-kriging.
 * @param target Optional destination of at least the parameter length.
 * @throws If the settings or loadings are invalid or the target is too short.
 */
export function getGPULocalKrigingParameterValues(
  settings: GPULocalKrigingSettings,
  target?: Float32Array
): Float32Array {
  const structures =
    settings.coregionalizationModels ??
    (settings.coregionalization && settings.covariance
      ? [{covariance: settings.covariance, loadings: settings.coregionalization}]
      : settings.covariance
        ? [{covariance: settings.covariance, loadings: [1]}]
        : []);
  validateRequirement(
    'getGPULocalKrigingParameterValues',
    structures.length > 0,
    'supply covariance or coregionalizationModels'
  );
  const hasLoadings = Boolean(settings.coregionalizationModels || settings.coregionalization);
  const variableCount = settings.variableCount ?? (hasLoadings ? 2 : 1);
  const length = getGPULocalKrigingParameterLength(structures.length, variableCount);
  validateRequirement(
    'getGPULocalKrigingParameterValues',
    !(
      structures.length < 1 ||
      structures.length > 4 ||
      !Number.isInteger(variableCount) ||
      variableCount < (hasLoadings ? 2 : 1) ||
      variableCount > (hasLoadings ? 4 : 1)
    ),
    'coregionalizationModels must contain 1..4 structures; variableCount must be an integer in [2,4] with loadings, or 1 without'
  );
  const destination = new Float32Array(length);
  const output = target ?? destination;
  validateRequirement(
    'getGPULocalKrigingParameterValues',
    !(output.length < length),
    `Local kriging parameter target must hold ${length} elements`
  );

  let covarianceScale = 0;
  structures.forEach(({covariance, loadings}, structureIndex) => {
    validatePredictionCovariance(
      covariance,
      'getGPULocalKrigingParameterValues',
      settings.coregionalizationModels
        ? `coregionalizationModels[${structureIndex}].covariance`
        : 'covariance'
    );
    validateRequirement(
      'getGPULocalKrigingParameterValues',
      !(loadings.length !== variableCount ** 2 || !loadings.every(Number.isFinite)),
      'coregionalizationModels.loadings must contain variableCount squared finite entries'
    );
    const structureOffset = 4 + structureIndex * 4;
    destination[structureOffset] = getPredictionCovarianceModelCode(covariance);
    destination[structureOffset + 1] = covariance.nugget ?? 0;
    destination[structureOffset + 2] = covariance.sill;
    destination[structureOffset + 3] = covariance.range;
    let loadingScale = variableCount > 1 ? 0 : 1;
    if (variableCount > 1) {
      const coefficientOffset = 4 + 4 * structures.length + structureIndex * variableCount ** 2;
      for (let pair = 0; pair < variableCount ** 2; pair++) {
        const coefficient = Math.fround(
          getCoregionalizationCoefficient(loadings, variableCount, pair)
        );
        validateRequirement(
          'getGPULocalKrigingParameterValues',
          Number.isFinite(coefficient),
          `coregionalizationModels[${structureIndex}] coefficient[${pair}] must be finite after float32 encoding; received ${coefficient}`
        );
        destination[coefficientOffset + pair] = coefficient;
        loadingScale = Math.max(loadingScale, Math.abs(coefficient));
      }
    }
    covarianceScale = Math.max(
      covarianceScale,
      (covariance.sill + (covariance.nugget ?? 0)) * loadingScale
    );
  });
  // Zero loadings define a valid PSD process covariance. Reserve zero for an unwritten buffer,
  // including when an initialized scale rounds to zero in the float32 parameter layout.
  const normalizationScale = Math.fround(covarianceScale);
  validateRequirement(
    'getGPULocalKrigingParameterValues',
    Number.isFinite(normalizationScale),
    `normalizationScale must be finite after float32 encoding; received ${covarianceScale}`
  );
  destination[0] = normalizationScale === 0 ? 1 : normalizationScale;
  output.set(destination);
  return output;
}

/**
 * Coefficient `(L L^T)[pair]` of a row-major square loading matrix.
 *
 * @internal
 */
export function getCoregionalizationCoefficient(
  loadings: readonly number[],
  variableCount: number,
  pair: number
): number {
  const first = Math.floor(pair / variableCount);
  const second = pair % variableCount;
  let value = 0;
  for (let component = 0; component < variableCount; component++) {
    value +=
      loadings[first * variableCount + component] * loadings[second * variableCount + component];
  }
  return value;
}

/** `covarianceCorrelation(modelCode, scaled)`, branching on the per-frame model code. */
function getCovarianceCorrelationWGSL(): string {
  const correlationCases = [0, 1, 2, 3, 4]
    .map(modelCode => {
      const {statements, expression} = getCorrelationWGSL(modelCode);
      return `    ${modelCode === MAXIMUM_PREDICTION_COVARIANCE_MODEL_CODE ? 'default' : `case ${modelCode}u`} { ${statements} return ${expression}; }`;
    })
    .join('\n');
  return `fn covarianceCorrelation(modelCode: u32, scaled: f32) -> f32 {
  switch modelCode {
${correlationCases}
  }
}`;
}

/**
 * Per-frame covariance functions over the `params` storage binding: `loadCovarianceParameters()`,
 * returning false for an invalid buffer, and `crossCovariance(first, second, firstVariable,
 * secondVariable)`.
 *
 * @internal
 */
export function getPerFramePredictionCovarianceWGSL(
  structureCount: number,
  variableCount: number
): string {
  const coefficientCount = structureCount * variableCount * variableCount;
  const usesCoefficients = variableCount > 1;
  return `var<private> covarianceStructures: array<vec4<f32>, ${structureCount}>;
${usesCoefficients ? `var<private> crossCoefficients: array<f32, ${coefficientCount}>;` : ''}
${getCovarianceCorrelationWGSL()}
fn structureCovariance(structure: u32, first: vec2<f32>, second: vec2<f32>) -> f32 {
  let values = covarianceStructures[structure];
  let scaled = distance(first, second) / values.w;
  return values.z * covarianceCorrelation(u32(values.x), scaled) + select(0.0, values.y, all(first == second));
}
fn loadCovarianceParameters() -> bool {
  if (!(params[paramsOffset] > 0.0)) { return false; }
  for (var structure = 0u; structure < ${structureCount}u; structure++) {
    let base = paramsOffset + 4u + structure * 4u;
    let values = vec4<f32>(params[base], params[base + 1u], params[base + 2u], params[base + 3u]);
    if (!(values.x >= 0.0 && values.x <= ${MAXIMUM_PREDICTION_COVARIANCE_MODEL_CODE}.0) || !(values.y >= 0.0) || !(values.z >= 0.0) || !(values.w > 0.0)) { return false; }
    covarianceStructures[structure] = values;
  }
${usesCoefficients ? `  for (var coefficient = 0u; coefficient < ${coefficientCount}u; coefficient++) { crossCoefficients[coefficient] = params[paramsOffset + ${4 + 4 * structureCount}u + coefficient]; }` : ''}
  return true;
}
fn crossCovariance(first: vec2<f32>, second: vec2<f32>, firstVariable: u32, secondVariable: u32) -> f32 {
  ${usesCoefficients ? `var total = 0.0;\n  for (var structure = 0u; structure < ${structureCount}u; structure++) { total += structureCovariance(structure, first, second) * crossCoefficients[structure * ${variableCount * variableCount}u + firstVariable * ${variableCount}u + secondVariable]; }\n  return total;` : 'return structureCovariance(0u, first, second);'}
}`;
}

/** Number of float32 elements in a {@link GPUSpaceTimeKriging} parameter view. */
export const GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH = 28;

/** Per-frame settings of {@link GPUSpaceTimeKriging}. */
export type GPUSpaceTimeKrigingSettings = {
  /** Space-time covariance, for example the result of `fitSpaceTimeVariogramModel`. */
  covariance: SpaceTimeVariogramModel;
  /**
   * Spatial units per unit of time of the neighbor metric `sqrt(h^2 + (anisotropy u)^2)`.
   * Defaults to the `anisotropy` of a metric or sum-metric covariance; required otherwise.
   */
  searchAnisotropy?: number;
  /** Largest spatial lag of a neighbor. Defaults to infinity. */
  searchRadius?: number;
  /** Largest absolute time lag of a neighbor. Defaults to infinity. */
  maximumTimeLag?: number;
  /** Known mean of `simple` mode. Defaults to 0. */
  mean?: number;
};

const SPACE_TIME_TYPE_CODES: Record<SpaceTimeVariogramModel['type'], number> = {
  separable: 0,
  'product-sum': 1,
  metric: 2,
  'sum-metric': 3,
  gneiting: 4
};

function validateStructure(
  structure: SpaceTimeVariogramStructure,
  id: string,
  field: string
): void {
  validatePredictionCovariance(
    {
      model: structure.model,
      nugget: structure.nugget,
      sill: structure.sill,
      range: structure.range
    },
    id,
    field
  );
}

function getStructureTotal(structure: SpaceTimeVariogramStructure): number {
  return structure.nugget + structure.sill;
}

function validateSpaceTimeFinitePositive(id: string, name: string, value: number): void {
  validateRequirement(
    id,
    Number.isFinite(value) && value > 0,
    `${name} must be positive and finite`
  );
}

/** Returns `C(0, 0)` of a validated model. */
function getSpaceTimeCovarianceScale(model: SpaceTimeVariogramModel): number {
  switch (model.type) {
    case 'separable':
      return model.sill;
    case 'product-sum': {
      const spatial = getStructureTotal(model.spatial);
      const temporal = getStructureTotal(model.temporal);
      return model.k * spatial * temporal + spatial + temporal;
    }
    case 'metric':
      return getStructureTotal(model.joint);
    case 'sum-metric':
      return (
        getStructureTotal(model.spatial) +
        getStructureTotal(model.temporal) +
        getStructureTotal(model.joint)
      );
    case 'gneiting':
      return model.nugget + model.sill;
    default:
      throw new Error(
        'getSpaceTimeCovarianceScale covariance.type must be separable, product-sum, metric, sum-metric or gneiting'
      );
  }
}

function writeStructure(
  target: Float32Array,
  offset: number,
  structure: SpaceTimeVariogramStructure
): void {
  target[offset] = getPredictionCovarianceModelCode({
    model: structure.model,
    range: structure.range,
    sill: structure.sill
  });
  target[offset + 1] = structure.nugget;
  target[offset + 2] = structure.sill;
  target[offset + 3] = structure.range;
}

/**
 * Packs per-frame {@link GPUSpaceTimeKriging} parameters.
 *
 * Layout (float32): `[covarianceScale, typeCode (0 separable, 1 product-sum, 2 metric, 3 sum-metric,
 * 4 gneiting), anisotropy, separable sill or product-sum k]`, then three structures
 * `[modelCode, nugget, sill, range]` (spatial, temporal, joint), then the Gneiting
 * `[sill, nugget, a, c, alpha, gamma, beta, tau]`, then
 * `[searchRadius, maximumTimeLag, searchAnisotropy, mean]`. A zero `covarianceScale` (an unwritten
 * buffer) gives status 1 for every query.
 *
 * @param settings Covariance and neighbor search.
 * @param target Optional destination of at least 28 elements.
 * @throws If a parameter is out of range, the target is too short, or `searchAnisotropy` is
 *   missing for a model without an anisotropy.
 */
export function getGPUSpaceTimeKrigingParameterValues(
  settings: GPUSpaceTimeKrigingSettings,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUSpaceTimeKrigingParameterValues';
  validateRequirement(
    id,
    !(target.length < GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH),
    `Space-time kriging parameter target must hold ${GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH} elements`
  );
  const {covariance} = settings;
  const destination = target;
  target = new Float32Array(GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH);
  target[1] = SPACE_TIME_TYPE_CODES[covariance.type];
  let modelAnisotropy: number | undefined;
  switch (covariance.type) {
    case 'separable': {
      validateStructure(covariance.spatial, id, 'covariance.spatial');
      validateStructure(covariance.temporal, id, 'covariance.temporal');
      validateRequirement(
        id,
        !(
          Math.abs(getStructureTotal(covariance.spatial) - 1) > 1e-6 ||
          Math.abs(getStructureTotal(covariance.temporal) - 1) > 1e-6
        ),
        'Separable space-time structures must be standardized to a total of 1'
      );
      validateSpaceTimeFinitePositive(id, 'covariance.sill', covariance.sill);
      target[3] = covariance.sill;
      writeStructure(target, 4, covariance.spatial);
      writeStructure(target, 8, covariance.temporal);
      break;
    }
    case 'product-sum': {
      validateStructure(covariance.spatial, id, 'covariance.spatial');
      validateStructure(covariance.temporal, id, 'covariance.temporal');
      validateRequirement(
        id,
        !(!Number.isFinite(covariance.k) || !(covariance.k >= 0)),
        'Space-time covariance k must be non-negative and finite'
      );
      target[3] = covariance.k;
      writeStructure(target, 4, covariance.spatial);
      writeStructure(target, 8, covariance.temporal);
      break;
    }
    case 'metric': {
      validateStructure(covariance.joint, id, 'covariance.joint');
      validateSpaceTimeFinitePositive(id, 'covariance.anisotropy', covariance.anisotropy);
      modelAnisotropy = covariance.anisotropy;
      writeStructure(target, 12, covariance.joint);
      break;
    }
    case 'sum-metric': {
      validateStructure(covariance.spatial, id, 'covariance.spatial');
      validateStructure(covariance.temporal, id, 'covariance.temporal');
      validateStructure(covariance.joint, id, 'covariance.joint');
      validateSpaceTimeFinitePositive(id, 'covariance.anisotropy', covariance.anisotropy);
      modelAnisotropy = covariance.anisotropy;
      writeStructure(target, 4, covariance.spatial);
      writeStructure(target, 8, covariance.temporal);
      writeStructure(target, 12, covariance.joint);
      break;
    }
    case 'gneiting': {
      validateSpaceTimeFinitePositive(id, 'covariance.sill', covariance.sill);
      validateSpaceTimeFinitePositive(id, 'covariance.spatialRange', covariance.spatialRange);
      validateSpaceTimeFinitePositive(id, 'covariance.temporalRange', covariance.temporalRange);
      validateRequirement(
        id,
        !(
          !(covariance.nugget >= 0) ||
          !(covariance.alpha > 0 && covariance.alpha <= 1) ||
          !(covariance.gamma > 0 && covariance.gamma <= 1) ||
          !(covariance.beta >= 0 && covariance.beta <= 1) ||
          !(covariance.tau >= covariance.beta) ||
          !Number.isFinite(covariance.tau)
        ),
        'Gneiting covariance parameters require finite nugget >= 0, sill > 0, alpha/gamma in (0,1], beta in [0,1], finite tau >= beta'
      );
      target[16] = covariance.sill;
      target[17] = covariance.nugget;
      target[18] = getGneitingTemporalScale(
        covariance.temporalRange,
        covariance.alpha,
        covariance.tau
      );
      target[19] = getGneitingSpatialScale(covariance.spatialRange, covariance.gamma);
      target[20] = covariance.alpha;
      target[21] = covariance.gamma;
      target[22] = covariance.beta;
      target[23] = covariance.tau;
      for (const [field, offset] of [
        ['temporal coefficient a', 18],
        ['spatial coefficient c', 19],
        ['alpha', 20],
        ['gamma', 21],
        ['tau', 23]
      ] as const) {
        validateRequirement(
          id,
          Number.isFinite(target[offset]) && target[offset] > 0,
          `Gneiting ${field} must be finite and positive after float32 encoding; received ${target[offset]}`
        );
      }
      validateRequirement(
        id,
        Number.isFinite(target[16]) &&
          Number.isFinite(target[17]) &&
          Number.isFinite(target[22]) &&
          target[22] >= 0 &&
          target[22] <= 1 &&
          target[23] >= target[22],
        'Gneiting sill, nugget, beta and tau must remain finite and in their domains after float32 encoding'
      );
      break;
    }
    default:
      throw new Error(
        'getGPUSpaceTimeKrigingParameterValues covariance.type must be separable, product-sum, metric, sum-metric or gneiting'
      );
  }
  const searchAnisotropy = settings.searchAnisotropy ?? modelAnisotropy;
  validateRequirement(
    id,
    !(searchAnisotropy === undefined),
    'searchAnisotropy is required for a model without an anisotropy'
  );
  validateSpaceTimeFinitePositive(id, 'searchAnisotropy', searchAnisotropy);
  const scale = getSpaceTimeCovarianceScale(covariance);
  validateSpaceTimeFinitePositive(id, 'covariance total variance', scale);
  target[0] = scale;
  target[2] = modelAnisotropy ?? 0;
  if (modelAnisotropy !== undefined)
    validateRequirement(
      id,
      Number.isFinite(target[2]) && target[2] > 0,
      `covariance.anisotropy must remain finite and positive after float32 encoding; received ${modelAnisotropy}`
    );
  target[24] = settings.searchRadius ?? Infinity;
  target[25] = settings.maximumTimeLag ?? Infinity;
  target[26] = searchAnisotropy;
  target[27] = settings.mean ?? 0;
  validateRequirement(
    id,
    Number.isFinite(target[0]) && target[0] > 0 && Number.isFinite(target[26]) && target[26] > 0,
    `covariance normalization scale and searchAnisotropy must remain finite and positive after float32 encoding; received ${scale}, ${searchAnisotropy}`
  );
  destination.set(target);
  return destination;
}

/**
 * Per-frame space-time covariance over the `params` binding: `loadSpaceTimeCovariance()`, returning
 * false for an unwritten buffer, and `spaceTimeCovariance(spatialLag, timeLag, samePlace, sameTime)`.
 *
 * @internal
 */
export function getPerFrameSpaceTimeCovarianceWGSL(): string {
  return `var<private> stHeader: vec4<f32>;
var<private> stStructures: array<vec4<f32>, 3>;
var<private> stGneiting: array<f32, 8>;
${getCovarianceCorrelationWGSL()}
fn stRead(offset: u32) -> f32 { return params[paramsOffset + offset]; }
fn stStructureCovariance(slot: u32, lag: f32, coincident: bool) -> f32 {
  let values = stStructures[slot];
  return values.z * covarianceCorrelation(u32(values.x), lag / values.w) + select(0.0, values.y, coincident);
}
fn loadSpaceTimeCovariance() -> bool {
  stHeader = vec4<f32>(stRead(0u), stRead(1u), stRead(2u), stRead(3u));
  if (!(stHeader.x > 0.0) || !(stHeader.y >= 0.0 && stHeader.y <= 4.0)) { return false; }
  for (var slot = 0u; slot < 3u; slot++) {
    let base = 4u + slot * 4u;
    stStructures[slot] = vec4<f32>(stRead(base), stRead(base + 1u), stRead(base + 2u), stRead(base + 3u));
  }
  for (var entry = 0u; entry < 8u; entry++) { stGneiting[entry] = stRead(16u + entry); }
  return true;
}
fn spaceTimeCovariance(spatialLag: f32, timeLag: f32, samePlace: bool, sameTime: bool) -> f32 {
  let typeCode = u32(stHeader.y);
  if (typeCode == 2u || typeCode == 3u) {
    let metricLag = sqrt(spatialLag * spatialLag + stHeader.z * stHeader.z * timeLag * timeLag);
    let joint = stStructureCovariance(2u, metricLag, samePlace && sameTime);
    if (typeCode == 2u) { return joint; }
    return stStructureCovariance(0u, spatialLag, samePlace) + stStructureCovariance(1u, timeLag, sameTime) + joint;
  }
  if (typeCode == 4u) {
    let sill = stGneiting[0];
    let nugget = stGneiting[1];
    let a = stGneiting[2];
    let c = stGneiting[3];
    let psi = a * select(pow(timeLag, 2.0 * stGneiting[4]), 0.0, timeLag <= 0.0) + 1.0;
    let spatialPower = select(pow(spatialLag, 2.0 * stGneiting[5]), 0.0, spatialLag <= 0.0);
    return sill * pow(psi, -stGneiting[7]) * exp(-c * spatialPower * pow(psi, -stGneiting[6] * stGneiting[5])) + select(0.0, nugget, samePlace && sameTime);
  }
  let spatial = stStructureCovariance(0u, spatialLag, samePlace);
  let temporal = stStructureCovariance(1u, timeLag, sameTime);
  if (typeCode == 0u) { return stHeader.w * spatial * temporal; }
  return stHeader.w * spatial * temporal + spatial + temporal;
}`;
}
