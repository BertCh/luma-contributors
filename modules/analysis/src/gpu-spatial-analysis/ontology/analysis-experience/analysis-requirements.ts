// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisColumnProfile,
  GPUAnalysisDataProfile,
  GPUAnalysisRequirementResult,
  GPUAnalysisRequirementSpec
} from './analysis-experience-types';

// Evaluates the preconditions an operation declares against what a data profile knows. A profile
// settles some of them (a row count, a column minimum) and cannot settle others (connected
// weights). Where it cannot, the result is `unknown` with evidence `not-checkable`: that is an
// honest answer, not an error, and a host asks the user or runs a GPU validator instead.

/** Bound columns by role id: one name, or several for a role that takes many. */
export type GPUAnalysisBoundColumns = Readonly<Record<string, string | readonly string[]>>;

/** Summary of a set of requirement results for one input. */
export type GPUAnalysisRequirementSummary = {
  /** `blocked`: a blocking requirement failed. `needs-confirmation`: one is unknown. */
  status: 'ready' | 'needs-confirmation' | 'blocked';
  /** Blocking requirements that failed. */
  blocking: readonly GPUAnalysisRequirementResult[];
  /** Blocking requirements the profile could not settle. */
  unknown: readonly GPUAnalysisRequirementResult[];
  /** Warning-level requirements that failed or could not be settled. */
  warnings: readonly GPUAnalysisRequirementResult[];
};

/** Kinds whose rows are areas, for which neighbours come from contiguity. */
const AREA_KINDS: readonly string[] = ['polygons', 'cells', 'regions'];

/** Distinct values a column needs to vary at all. */
const VARYING_DISTINCT_MINIMUM = 2;

function formatNumber(value: number): string {
  return Number.isInteger(value)
    ? value.toLocaleString('en-US', {maximumFractionDigits: 0})
    : value.toLocaleString('en-US', {maximumSignificantDigits: 4});
}

function formatCount(value: number, singular: string, plural = `${singular}s`): string {
  return `${formatNumber(value)} ${value === 1 ? singular : plural}`;
}

function unknownResult(
  requirement: GPUAnalysisRequirementSpec,
  message: string,
  evidence: GPUAnalysisRequirementResult['evidence'] = 'not-checkable'
): GPUAnalysisRequirementResult {
  return {
    id: requirement.id,
    ...(requirement.role === undefined ? {} : {role: requirement.role}),
    severity: requirement.severity,
    state: 'unknown',
    evidence,
    message,
    ...(requirement.remedy === undefined ? {} : {remedy: requirement.remedy})
  };
}

function decidedResult(
  requirement: GPUAnalysisRequirementSpec,
  passed: boolean,
  evidence: GPUAnalysisRequirementResult['evidence'],
  message: string
): GPUAnalysisRequirementResult {
  return {
    id: requirement.id,
    ...(requirement.role === undefined ? {} : {role: requirement.role}),
    severity: requirement.severity,
    state: passed ? 'passed' : 'failed',
    evidence,
    message,
    ...(requirement.remedy === undefined || passed ? {} : {remedy: requirement.remedy})
  };
}

function requireLimit(requirement: GPUAnalysisRequirementSpec): number {
  const {limit} = requirement;
  if (limit === undefined || !Number.isFinite(limit) || limit < 0) {
    throw new Error(
      `evaluateGPUAnalysisRequirement: requirement.limit of "${requirement.id}" must be a finite number of at least 0, received ${String(limit)}`
    );
  }
  return limit;
}

function evaluateCountLimit(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  const limit = requireLimit(requirement);
  if (requirement.id === 'minimum-rows') {
    return decidedResult(
      requirement,
      profile.rowCount >= limit,
      'profile',
      `${formatCount(profile.rowCount, 'row')}; at least ${formatNumber(limit)} ${limit === 1 ? 'is' : 'are'} needed`
    );
  }
  if (requirement.id === 'maximum-rows') {
    return decidedResult(
      requirement,
      profile.rowCount <= limit,
      'profile',
      `${formatCount(profile.rowCount, 'row')}; at most ${formatNumber(limit)} ${limit === 1 ? 'is' : 'are'} supported`
    );
  }
  const networkQuantity = {
    'maximum-network-edges': ['edgeCount', 'physical edge'],
    'maximum-network-nodes': ['nodeCount', 'network node'],
    'maximum-network-arcs': ['arcCount', 'traversable arc']
  } as const;
  if (requirement.id in networkQuantity) {
    const [quantity, label] = networkQuantity[requirement.id as keyof typeof networkQuantity];
    const count = profile.network?.[quantity];
    if (count === undefined)
      return unknownResult(
        requirement,
        `The ${label} count for the selected topology is not known; at most ${formatNumber(limit)} are supported`
      );
    return decidedResult(
      requirement,
      count <= limit,
      'profile',
      `${formatCount(count, label)}; at most ${formatNumber(limit)} are supported`
    );
  }
  // A point has one vertex, so a point table's vertex count is its row count.
  const vertexCount =
    profile.vertexCount ?? (profile.kind === 'points' ? profile.rowCount : undefined);
  if (vertexCount === undefined) {
    return unknownResult(
      requirement,
      `The vertex count is not known; at most ${formatNumber(limit)} ${limit === 1 ? 'is' : 'are'} supported`
    );
  }
  return decidedResult(
    requirement,
    vertexCount <= limit,
    'profile',
    `${formatCount(vertexCount, 'vertex', 'vertices')}; at most ${formatNumber(limit)} ${limit === 1 ? 'is' : 'are'} supported`
  );
}

/** Longitudes up to this value are allowed for degrees (0 to 360 longitudes exist). */
const LONGITUDE_MAXIMUM_DEGREES = 360;
const LONGITUDE_MINIMUM_DEGREES = -180;
const LATITUDE_LIMIT_DEGREES = 90;

/**
 * Sanity check of a DECLARED coordinate model against the extent [minimumX, minimumY, maximumX,
 * maximumY]. A declared geographic model whose extent lies outside longitude -180 to 360 or
 * latitude -90 to 90 is implausible (the values cannot be degrees). A declared planar model is
 * never contradicted by an extent, because planar units can take any values. A non-finite extent
 * is not checked.
 */
export function checkGPUAnalysisCoordinateDeclaration(
  extent: readonly [number, number, number, number],
  model: 'geographic' | 'planar'
): {plausible: boolean; message: string} {
  if (model === 'planar' || !extent.every(value => Number.isFinite(value))) {
    return {
      plausible: true,
      message: 'The declared coordinate system is not contradicted by the extent'
    };
  }
  const [minimumX, minimumY, maximumX, maximumY] = extent;
  const plausible =
    minimumX >= LONGITUDE_MINIMUM_DEGREES &&
    maximumX <= LONGITUDE_MAXIMUM_DEGREES &&
    minimumY >= -LATITUDE_LIMIT_DEGREES &&
    maximumY <= LATITUDE_LIMIT_DEGREES;
  return plausible
    ? {plausible, message: 'The declared geographic coordinates fit within degrees'}
    : {
        plausible,
        message: `The coordinates are declared geographic but the extent runs from ${formatNumber(minimumX)} to ${formatNumber(maximumX)} in x and ${formatNumber(minimumY)} to ${formatNumber(maximumY)} in y, so the values cannot be degrees`
      };
}

/** Remedy for a guessed coordinate model that contradicts a requirement. */
const CONFIRM_COORDINATES_REMEDY = 'Confirm the coordinate system';

function evaluateCoordinateModel(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  const {coordinates} = profile;
  const wanted = requirement.id === 'planar-coordinates' ? 'planar' : 'geographic';
  if (coordinates === undefined) {
    return unknownResult(
      requirement,
      `The coordinate system is not known; ${wanted} coordinates are needed`
    );
  }
  if (coordinates.source === 'declared') {
    const extent = profile.extent ?? profile.spacing?.extent;
    if (extent !== undefined) {
      const check = checkGPUAnalysisCoordinateDeclaration(extent, coordinates.model);
      if (!check.plausible) {
        return decidedResult(requirement, false, 'declaration', check.message);
      }
    }
    return decidedResult(
      requirement,
      coordinates.model === wanted,
      'declaration',
      `The coordinates are ${coordinates.model} (declared); ${wanted} coordinates are needed`
    );
  }
  const message = `The coordinates look ${coordinates.model} (guessed from the extent); ${wanted} coordinates are needed`;
  if (coordinates.confidence !== 'low' && coordinates.model === wanted) {
    return decidedResult(requirement, true, 'guess', message);
  }
  // A low-confidence or contradicting guess is a question for the user, never a failure.
  return {
    ...unknownResult(requirement, message, 'guess'),
    remedy:
      requirement.remedy === undefined
        ? CONFIRM_COORDINATES_REMEDY
        : `${CONFIRM_COORDINATES_REMEDY}; ${requirement.remedy}`
  };
}

function evaluateDeclaredCoordinates(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  const {coordinates} = profile;
  if (coordinates === undefined) {
    return unknownResult(
      requirement,
      'The coordinate system is not known; it must be declared before the analysis runs'
    );
  }
  if (coordinates.source === 'declared') {
    return decidedResult(
      requirement,
      true,
      'declaration',
      `The coordinate system is declared as ${coordinates.model}; a declaration is needed`
    );
  }
  return unknownResult(
    requirement,
    `The coordinate system is only a guess (${coordinates.model}, from the extent); it must be declared before the analysis runs`,
    'guess'
  );
}

function evaluateRowSemantics(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  const {expects} = requirement;
  if (expects === undefined || expects.length === 0) {
    throw new Error(
      'evaluateGPUAnalysisRequirement: requirement.expects of "row-semantics" must list at least one row semantics'
    );
  }
  const wanted = expects.join(' or ');
  const {rowSemantics} = profile;
  if (rowSemantics === undefined) {
    return unknownResult(
      requirement,
      `What the rows stand for is not known; ${wanted} are expected`
    );
  }
  const message = `The rows are ${rowSemantics.value} (${rowSemantics.source === 'declared' ? 'declared' : 'guessed'}); ${wanted} are expected`;
  if (rowSemantics.source === 'guess' && rowSemantics.confidence === 'low') {
    return unknownResult(requirement, `${message}; confirm what the rows represent`, 'guess');
  }
  return decidedResult(
    requirement,
    expects.includes(rowSemantics.value),
    rowSemantics.source === 'declared' ? 'declaration' : 'guess',
    message
  );
}

function evaluateNeighbors(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  if (profile.kind === 'points') {
    // Every point has a nearest other point as soon as there are two points.
    return decidedResult(
      requirement,
      profile.rowCount >= 2,
      'profile',
      `${formatCount(profile.rowCount, 'point')}; every point has a nearest neighbor when there are at least 2`
    );
  }
  if (AREA_KINDS.includes(profile.kind)) {
    if (profile.contiguity === undefined) {
      return unknownResult(
        requirement,
        'Neighbor counts under contiguity have not been computed; every area needs a neighbor'
      );
    }
    const {islandCount} = profile.contiguity;
    return decidedResult(
      requirement,
      islandCount === 0,
      'profile',
      `${formatCount(islandCount, 'area has', 'areas have')} no neighbor under contiguity; none may be without one`
    );
  }
  return unknownResult(
    requirement,
    `Neighbors cannot be checked from a profile of ${profile.kind}; every row needs a neighbor`
  );
}

function evaluateDuplicates(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile
): GPUAnalysisRequirementResult {
  const {spacing} = profile;
  if (spacing === undefined) {
    return unknownResult(
      requirement,
      'Spacing statistics have not been computed; no two rows may share a location'
    );
  }
  const duplicates = spacing.duplicateLocationCount;
  const fact = `${formatCount(duplicates, 'point shares', 'points share')} a location with an earlier point`;
  if (!spacing.sampled) {
    return decidedResult(requirement, duplicates === 0, 'profile', `${fact}; none may`);
  }
  const sample = `in a sample of ${formatNumber(spacing.sampleSize)} of ${formatNumber(spacing.pointCount)} points`;
  // "No duplicates" is universal: a sample can refute it but never establish it.
  return duplicates > 0
    ? decidedResult(requirement, false, 'sampled-profile', `${fact} (${sample}); none may`)
    : unknownResult(
        requirement,
        `${fact} (${sample}, which cannot show that no other points do); none may`,
        'sampled-profile'
      );
}

type ColumnCheck = {
  passed: boolean;
  message: string;
};

type ColumnOutcome =
  | {state: 'unknown'; message: string; sampled?: boolean}
  | {state: 'decided'; passed: boolean; message: string; sampled: boolean};

function checkDistinctCount(
  column: GPUAnalysisColumnProfile,
  limit: number,
  message: string,
  maximum: boolean = false
): ColumnOutcome {
  const sampleSize = column.sampleSize ?? column.rowCount;
  const sampled = sampleSize < column.rowCount;
  const sample = sampled
    ? ` (in a sample of ${formatNumber(sampleSize)} of ${formatNumber(column.rowCount)} rows)`
    : '';
  let lowerBound = column.distinctCountLowerBound;
  let upperBound = column.distinctCountUpperBound;
  const exactCount = column.distinctCountIsEstimate ? undefined : column.distinctCount;
  if (lowerBound === undefined && upperBound === undefined) {
    if (column.distinctCountIsEstimate || column.distinctCount === undefined) {
      return {
        state: 'unknown',
        sampled,
        message: `${message}${sample}; the estimated count has no guaranteed error bound, so confirm with an exact count`
      };
    }
    lowerBound = upperBound = column.distinctCount;
  }
  if (
    lowerBound === undefined ||
    upperBound === undefined ||
    !Number.isSafeInteger(lowerBound) ||
    !Number.isSafeInteger(upperBound) ||
    lowerBound < 0 ||
    (exactCount !== undefined && !Number.isSafeInteger(exactCount))
  ) {
    return {
      state: 'unknown',
      sampled,
      message: `${message}${sample}; the guaranteed distinct bounds are incomplete or inconsistent, so refine with an exact count`
    };
  }
  // Bounds are inclusive guarantees, not necessarily tight. Intersect independent evidence
  // in the inspected population before accounting for any uninspected rows. A loose upper
  // bound may exceed the row ceiling, and an exact count may lie anywhere inside the bounds.
  lowerBound = Math.max(lowerBound, exactCount ?? 0);
  upperBound = Math.min(upperBound, sampleSize - column.nullCount, exactCount ?? Infinity);
  if (lowerBound > upperBound) {
    return {
      state: 'unknown',
      sampled,
      message: `${message}${sample}; the guaranteed distinct bounds, non-null row ceiling and exact count have an empty intersection, so refine with an exact count`
    };
  }
  // Each uninspected row may introduce one new value, including when every sampled row is null.
  const populationUpperBound = upperBound + (sampled ? column.rowCount - sampleSize : 0);
  const passed = maximum ? populationUpperBound <= limit : lowerBound >= limit;
  const failed = maximum ? lowerBound > limit : populationUpperBound < limit;
  const bounds =
    column.distinctCountLowerBound !== undefined || sampled
      ? `; guaranteed population distinct bounds [${formatNumber(lowerBound)}, ${formatNumber(populationUpperBound)}]`
      : '';
  if (!passed && !failed) {
    return {
      state: 'unknown',
      sampled,
      message: `${message}${sample}${bounds}; ${sampled && !maximum ? 'the sample cannot rule it out for the whole column; ' : ''}refine with an exact count near this threshold`
    };
  }
  return {state: 'decided', passed, sampled, message: `${message}${sample}${bounds}`};
}

/** Checks descriptor bounds, never the rows of a dataset. */
function describeNumericRange(requirement: GPUAnalysisRequirementSpec): string {
  const {minimum, maximum, integer} = requirement;
  if (
    (minimum === undefined && maximum === undefined && integer !== true) ||
    (minimum !== undefined && !Number.isFinite(minimum)) ||
    (maximum !== undefined && !Number.isFinite(maximum)) ||
    (minimum !== undefined && maximum !== undefined && minimum > maximum)
  ) {
    throw new Error(
      'evaluateGPUAnalysisRequirement: numeric-range requires finite ordered minimum/maximum bounds or integer: true'
    );
  }
  return [
    integer ? 'integer values' : 'numeric values',
    ...(minimum === undefined ? [] : [`at least ${formatNumber(minimum)}`]),
    ...(maximum === undefined ? [] : [`at most ${formatNumber(maximum)}`])
  ].join(' ');
}

function checkColumn(
  requirement: GPUAnalysisRequirementSpec,
  column: GPUAnalysisColumnProfile
): ColumnOutcome {
  const name = column.name;
  const sampled = column.sampleSize !== undefined && column.sampleSize < column.rowCount;
  const sample = `in a sample of ${formatNumber(column.sampleSize ?? 0)} of ${formatNumber(column.rowCount)} rows`;
  // A universal property (all positive, none missing) can be refuted by a sample but not
  // established; an existential one (varies, has N distinct values) the other way round. The
  // direction a sample cannot settle is unknown.
  const decide = (check: ColumnCheck, kind: 'universal' | 'existential'): ColumnOutcome => {
    if (!sampled) {
      return {state: 'decided', passed: check.passed, message: check.message, sampled};
    }
    if (kind === 'universal' ? check.passed : !check.passed) {
      return {
        state: 'unknown',
        sampled: true,
        message: `${check.message} (${sample}, which cannot ${kind === 'universal' ? 'show it holds for every row' : 'rule it out for the whole column'})`
      };
    }
    return {
      state: 'decided',
      passed: check.passed,
      message: `${check.message} (${sample})`,
      sampled
    };
  };
  const minimum = Number.isFinite(column.minimum) ? column.minimum : undefined;
  const maximum = Number.isFinite(column.maximum) ? column.maximum : undefined;

  switch (requirement.id) {
    case 'numeric-range': {
      const wanted = describeNumericRange(requirement);
      const integral = column.integral ?? (column.type === 'integer' ? true : undefined);
      const outside =
        (requirement.minimum !== undefined &&
          minimum !== undefined &&
          minimum < requirement.minimum) ||
        (requirement.maximum !== undefined &&
          maximum !== undefined &&
          maximum > requirement.maximum) ||
        (requirement.integer === true && integral === false);
      if (outside)
        return decide(
          {passed: false, message: `${name} violates the required range: ${wanted}`},
          'universal'
        );
      if (
        (requirement.minimum !== undefined && minimum === undefined) ||
        (requirement.maximum !== undefined && maximum === undefined) ||
        (requirement.integer === true && integral === undefined)
      )
        return {
          state: 'unknown',
          sampled,
          message: `The profile cannot establish ${wanted} for ${name}`
        };
      return decide(
        {passed: true, message: `${name} satisfies the required range: ${wanted}`},
        'universal'
      );
    }
    case 'positive-values':
      if (minimum === undefined) {
        return {
          state: 'unknown',
          message: `No minimum was computed for ${name}; every value must be above 0`
        };
      }
      return decide(
        {
          passed: minimum > 0,
          message: `${name} has a minimum of ${formatNumber(minimum)}; every value must be above 0`
        },
        'universal'
      );
    case 'non-negative-values':
      if (minimum === undefined) {
        return {
          state: 'unknown',
          message: `No minimum was computed for ${name}; no value may be below 0`
        };
      }
      return decide(
        {
          passed: minimum >= 0,
          message: `${name} has a minimum of ${formatNumber(minimum)}; no value may be below 0`
        },
        'universal'
      );
    case 'no-missing-values':
      return decide(
        {
          passed: column.nullCount === 0,
          message: `${name} has ${formatCount(column.nullCount, 'missing value')}; none are allowed`
        },
        'universal'
      );
    case 'varying-values': {
      if (minimum !== undefined && maximum !== undefined) {
        return decide(
          {
            passed: minimum < maximum,
            message: `${name} ranges from ${formatNumber(minimum)} to ${formatNumber(maximum)}; the values must differ`
          },
          'existential'
        );
      }
      if (column.distinctCount !== undefined || column.distinctCountLowerBound !== undefined) {
        return checkDistinctCount(
          column,
          VARYING_DISTINCT_MINIMUM,
          `${name} has ${column.distinctCountIsEstimate ? 'about ' : ''}${formatCount(column.distinctCount ?? (column.distinctCountLowerBound as number), 'distinct value')}; at least ${VARYING_DISTINCT_MINIMUM} are needed for the values to differ`
        );
      }
      return {
        state: 'unknown',
        message: `No range or distinct count was computed for ${name}; the values must differ`
      };
    }
    case 'minimum-distinct-values':
    case 'maximum-distinct-values': {
      const limit = requireLimit(requirement);
      const maximum = requirement.id === 'maximum-distinct-values';
      const direction = maximum ? 'at most' : 'at least';
      if (column.distinctCount === undefined && column.distinctCountLowerBound === undefined) {
        return {
          state: 'unknown',
          message: `No distinct count was computed for ${name}; ${direction} ${formatNumber(limit)} are needed`
        };
      }
      const about = column.distinctCountIsEstimate ? 'about ' : '';
      return checkDistinctCount(
        column,
        limit,
        `${name} has ${about}${formatCount(column.distinctCount ?? (column.distinctCountLowerBound as number), 'distinct value')}; ${direction} ${formatNumber(limit)} ${limit === 1 ? 'is' : 'are'} needed`,
        maximum
      );
    }
    default:
      return {state: 'unknown', message: `${requirement.id} is not a column requirement`};
  }
}

function resolveBoundNames(bound: string | readonly string[] | undefined): readonly string[] {
  if (bound === undefined) {
    return [];
  }
  return typeof bound === 'string' ? [bound] : bound;
}

function evaluateColumnRequirement(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile,
  boundColumns: GPUAnalysisBoundColumns
): GPUAnalysisRequirementResult {
  if (requirement.role === undefined) {
    throw new Error(
      `evaluateGPUAnalysisRequirement: requirement.role of "${requirement.id}" must name a field role`
    );
  }
  if (
    requirement.id === 'minimum-distinct-values' ||
    requirement.id === 'maximum-distinct-values'
  ) {
    requireLimit(requirement);
  }
  const names = resolveBoundNames(boundColumns[requirement.role]);
  if (names.length === 0) {
    return unknownResult(
      requirement,
      `No column is chosen for the role "${requirement.role}" yet, so ${requirement.id} cannot be checked`
    );
  }
  const outcomes: ColumnOutcome[] = names.map(name => {
    const column = profile.columns.find(candidate => candidate.name === name);
    return column === undefined
      ? {
          state: 'unknown',
          message: `The profile has no column named ${name}; ${requirement.id} cannot be checked`
        }
      : checkColumn(requirement, column);
  });

  const failed = outcomes.filter(outcome => outcome.state === 'decided' && !outcome.passed);
  const unknown = outcomes.filter(outcome => outcome.state === 'unknown');
  const anySampled = outcomes.some(outcome => outcome.state === 'decided' && outcome.sampled);
  const evidence = anySampled ? 'sampled-profile' : 'profile';

  if (failed.length > 0) {
    const others = failed.length - 1;
    const message =
      others === 0
        ? failed[0].message
        : `${failed[0].message}; ${others === 1 ? '1 other bound column also fails' : `${formatNumber(others)} other bound columns also fail`}`;
    return decidedResult(requirement, false, evidence, message);
  }
  if (unknown.length > 0) {
    return unknownResult(
      requirement,
      unknown[0].message,
      unknown[0].state === 'unknown' && unknown[0].sampled ? 'sampled-profile' : 'not-checkable'
    );
  }
  const others = outcomes.length - 1;
  const message =
    others === 0
      ? outcomes[0].message
      : `${outcomes[0].message}; ${others === 1 ? '1 other bound column passes' : `${formatNumber(others)} other bound columns pass`} too`;
  return decidedResult(requirement, true, evidence, message);
}

/**
 * Evaluates one declared requirement against a profile and the columns bound to the operation's
 * field roles. Facts that settle each requirement:
 * - `minimum-rows`, `maximum-rows`: `profile.rowCount` against `limit`.
 * - `maximum-vertices`: `profile.vertexCount` against `limit`; for point data the row count; else
 *   unknown.
 * - `maximum-network-edges`, `maximum-network-nodes`, `maximum-network-arcs`: exact physical
 *   edges, builder nodes or traversable directed arcs for the selected topology. Unknown without
 *   network facts; the host must select facts for its revision and topology settings first.
 * - `planar-coordinates`, `geographic-coordinates`: `profile.coordinates.model`. Evidence is
 *   `declaration` or `guess` by its source; unknown when absent. A declared model passes or
 *   fails; a guessed one passes when it agrees with confidence above low and is `unknown` (evidence `guess`, remedy
 *   `Confirm the coordinate system`) when it contradicts, never `failed`.
 * - `declared-coordinates`: passed only when `profile.coordinates.source` is `declared`; a guess
 *   is unknown with evidence `guess`, because the user can still declare it.
 * - `row-semantics`: `profile.rowSemantics.value` against `expects`; unknown until present,
 *   and unknown for low-confidence guesses whether matching or contradicting.
 * - `positive-values`, `non-negative-values`: the minimum of each column bound to `role`.
 * - `numeric-range`: inclusive minimum/maximum and integrality of each bound column, or the
 *   named numeric parameter in `parameters`. Missing statistics stay unknown. Sampled columns
 *   can refute a range but cannot establish it for every row.
 * - `no-missing-values`: the null count of each bound column.
 * - `varying-values`: minimum below maximum, else distinct count of at least 2.
 * - `minimum-distinct-values`, `maximum-distinct-values`: inclusive guaranteed distinct bounds
 *   against `limit`. An estimate alone remains unknown. Bounds that straddle the threshold
 *   require exact refinement. Maximum distinct values expresses category capacity.
 *   Value requirements are `unknown` when no column is bound or the statistic is absent. When a
 *   bound column's statistics are sampled (`sampleSize` below `rowCount`) the evidence is
 *   `sampled-profile` and each id follows one rule. A universal property can be refuted by a
 *   sample but not established: `positive-values`, `non-negative-values`, `numeric-range`, `no-missing-values`
 *   and `no-duplicate-locations` fail when the sample shows a counterexample and are `unknown`
 *   when the sample satisfies them. An existential property can be established but not
 *   refuted: `varying-values` and `minimum-distinct-values` pass when the sample shows enough
 *   variety. Distinct upper bounds add one potential value per uninspected row, so a sampled
 *   capacity passes only when that full-population upper bound fits. The message names the sample.
 * - `every-row-has-a-neighbor`: for areas, `profile.contiguity.islandCount` being 0; for points,
 *   always true from two rows on (each point has a nearest neighbor); unknown for other kinds.
 * - `no-duplicate-locations`: `profile.spacing.duplicateLocationCount` being 0 (sampling above).
 * - `planar-coordinates`, `geographic-coordinates` also check a declared geographic model against
 *   the extent (`profile.extent`, else `profile.spacing.extent`) with
 *   `checkGPUAnalysisCoordinateDeclaration`: coordinates that cannot be degrees fail it.
 * - `connected-weights`, `disjoint-coverage`: never checkable from a profile; always unknown.
 * Throws when the requirement itself is malformed (a missing `limit`, `role` or `expects`).
 */
export function evaluateGPUAnalysisRequirement(
  requirement: GPUAnalysisRequirementSpec,
  profile: GPUAnalysisDataProfile,
  boundColumns: GPUAnalysisBoundColumns,
  parameters: Readonly<Record<string, unknown>> = {}
): GPUAnalysisRequirementResult {
  if (requirement.id === 'numeric-range') {
    const wanted = describeNumericRange(requirement);
    if (requirement.parameter !== undefined) {
      if (requirement.role !== undefined)
        throw new Error(
          'evaluateGPUAnalysisRequirement: numeric-range must name either a parameter or a role, not both'
        );
      const value = parameters[requirement.parameter];
      if (value === undefined)
        return unknownResult(
          requirement,
          `Parameter ${requirement.parameter} is unknown; requires ${wanted}`
        );
      const passed =
        typeof value === 'number' &&
        Number.isFinite(value) &&
        (requirement.minimum === undefined || value >= requirement.minimum) &&
        (requirement.maximum === undefined || value <= requirement.maximum) &&
        (!requirement.integer || Number.isInteger(value));
      return decidedResult(
        requirement,
        passed,
        'declaration',
        `${requirement.parameter} is ${String(value)}; requires ${wanted}`
      );
    }
    return evaluateColumnRequirement(requirement, profile, boundColumns);
  }
  switch (requirement.id) {
    case 'minimum-rows':
    case 'maximum-rows':
    case 'maximum-vertices':
    case 'maximum-network-edges':
    case 'maximum-network-nodes':
    case 'maximum-network-arcs':
      return evaluateCountLimit(requirement, profile);
    case 'planar-coordinates':
    case 'geographic-coordinates':
      return evaluateCoordinateModel(requirement, profile);
    case 'declared-coordinates':
      return evaluateDeclaredCoordinates(requirement, profile);
    case 'row-semantics':
      return evaluateRowSemantics(requirement, profile);
    case 'positive-values':
    case 'non-negative-values':
    case 'no-missing-values':
    case 'varying-values':
    case 'minimum-distinct-values':
    case 'maximum-distinct-values':
      return evaluateColumnRequirement(requirement, profile, boundColumns);
    case 'every-row-has-a-neighbor':
      return evaluateNeighbors(requirement, profile);
    case 'no-duplicate-locations':
      return evaluateDuplicates(requirement, profile);
    case 'connected-weights':
      return unknownResult(
        requirement,
        'Whether the weights connect every area cannot be checked from a profile; a validator or the user must settle it'
      );
    case 'disjoint-coverage':
      return unknownResult(
        requirement,
        'Whether the areas overlap cannot be checked from a profile; a validator or the user must settle it'
      );
  }
}

/** Evaluates every requirement of one input, in order. */
export function evaluateGPUAnalysisRequirements(
  requirements: readonly GPUAnalysisRequirementSpec[],
  profile: GPUAnalysisDataProfile,
  boundColumns: GPUAnalysisBoundColumns,
  parameters: Readonly<Record<string, unknown>> = {}
): readonly GPUAnalysisRequirementResult[] {
  return requirements.map(requirement =>
    evaluateGPUAnalysisRequirement(requirement, profile, boundColumns, parameters)
  );
}

/**
 * Reduces results to one status. `blocked` when a blocking requirement failed (even if others are
 * unknown); `needs-confirmation` when none failed but a blocking one is unknown; otherwise `ready`.
 * Warning-level results never change the status; they are listed in `warnings` when they failed
 * or are unknown.
 */
export function summarizeGPUAnalysisRequirementResults(
  results: readonly GPUAnalysisRequirementResult[]
): GPUAnalysisRequirementSummary {
  const blocking = results.filter(
    result => result.severity === 'block' && result.state === 'failed'
  );
  const unknown = results.filter(
    result => result.severity === 'block' && result.state === 'unknown'
  );
  const warnings = results.filter(
    result => result.severity === 'warn' && result.state !== 'passed'
  );
  const status =
    blocking.length > 0 ? 'blocked' : unknown.length > 0 ? 'needs-confirmation' : 'ready';
  return {status, blocking, unknown, warnings};
}
