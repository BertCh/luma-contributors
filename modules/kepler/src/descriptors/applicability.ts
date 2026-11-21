// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  evaluateGPUAnalysisParameterCondition,
  getGPUAnalysisParameterComparisons
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

import {
  assignGPUAnalysisFieldRoles,
  evaluateGPUAnalysisRequirements,
  getGPUAnalysisCatalogGeometryForDataKind,
  getGPUAnalysisQuestion,
  GPU_ANALYSIS_QUESTIONS,
  guessGPUAnalysisMeasurementLevel,
  matchGPUAnalysisFieldRole,
  suggestGPUAnalysisRatePairs,
  tokenizeGPUAnalysisColumnName,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisColumnType,
  type GPUAnalysisFieldRoleSpec,
  type GPUAnalysisQuestion,
  type GPUAnalysisRequirementResult,
  type GPUAnalysisRequirementSpec,
  type GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  getGPUVecchiaLogLikelihoodParameterValues,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  isValidationVecchiaCovarianceUsable,
  makeValidationSpaceTimeCovariance
} from '../analysis/catalog-validation-vecchia';
import {
  isDbscanCpuPointCountAdmitted,
  MAXIMUM_DBSCAN_CPU_POINTS
} from '../analysis/catalog-platform3-b-clusters';
import type {
  CatalogInputSpec,
  CatalogOperationBinding,
  CatalogParameterValue
} from '../analysis/catalog-binding-types';
import {KEPLER_CURATED_ANALYSES} from '../curated/curated-analyses';
import type {KeplerCuratedAnalysis} from '../curated/curated-analysis-types';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {getKeplerDatasetNetworkProfile} from '../profile/dataset-network-profile';
import {
  getKeplerEffectiveEngine,
  KEPLER_CPU_WORK_CAPS,
  KEPLER_GRID_CELL_CAPS,
  resolveKeplerOperationParameters,
  type KeplerParameterResolution
} from './default-parameters';
import {
  KEPLER_OPERATION_DESCRIPTORS,
  resolveKeplerOperationDescriptor
} from './operation-descriptors';
import type {
  KeplerOperationDescriptor,
  KeplerOperationInputDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// The suggestion engine: given the datasets a person has loaded, which analyses fit, why, with
// their columns and parameters filled in, and what is missing for the ones that do not. A pure
// function of its arguments with no model in it, so every ranking can be explained and tested.
//
// Hard rules (Show Me's availability conditions, Mackinlay et al. 2007) decide whether an analysis
// can run: input kinds present, required columns satisfiable, capacity. Soft rules (Draco's
// weighted costs, Moritz et al. 2018) order what can. An analysis that cannot run stays visible
// with the one thing it lacks.

// ---------------------------------------------------------------------------------------------
// Input compatibility
// ---------------------------------------------------------------------------------------------

/**
 * Which kinds of binding input a loaded dataset of each kind can fill. This is the first of two
 * tests; the second is that the geometry the catalog reads from the dataset (`point`, `line`,
 * `polygon`) is one the binding input accepts (`CatalogInputSpec.accepts`).
 *
 * - A point, line or polygon dataset can fill an input of its own kind and the loose kinds
 *   (`regions`, `cells`, `attributes`), and also the other geometry kinds when the binding
 *   accepts its geometry (`local-moran-statistics` reads `polygons` but accepts points).
 * - A `network` input accepts a line dataset (and a `network` dataset): the catalog builds the
 *   graph from lines.
 * - `trajectories` inputs accept point and line datasets, as the binding's geometry says.
 * This table governs feature ports only. Typed table and raster ports match geometry-free
 * `attributes` and `surface` profiles directly in isKeplerDatasetCompatibleWithInput.
 */
export const KEPLER_DATASET_KIND_INPUT_KINDS: Readonly<
  Partial<Record<GPUSpatialAnalysisDataKind, readonly GPUSpatialAnalysisDataKind[]>>
> = {
  points: ['points', 'lines', 'polygons', 'regions', 'cells', 'attributes', 'trajectories'],
  lines: [
    'points',
    'lines',
    'polygons',
    'regions',
    'cells',
    'attributes',
    'trajectories',
    'network'
  ],
  polygons: ['points', 'lines', 'polygons', 'regions', 'cells', 'attributes'],
  network: ['network', 'lines', 'regions', 'attributes', 'trajectories'],
  trajectories: ['trajectories', 'points', 'lines', 'attributes'],
  cells: ['cells', 'regions', 'polygons', 'points', 'attributes'],
  attributes: []
};

type CatalogGeometry = 'point' | 'line' | 'polygon';

/** The geometry the catalog can read from a dataset, from its profile. */
export function getKeplerDatasetGeometry(
  dataset: KeplerDatasetProfile
): CatalogGeometry | undefined {
  return dataset.catalogGeometry ?? (dataset.pointColumns ? 'point' : undefined);
}

const GEOMETRY_FAMILIES: readonly CatalogGeometry[] = ['point', 'line', 'polygon'];

/** Every geometry family a dataset holds: more than one for a layer that mixes them. */
export function getKeplerDatasetGeometries(dataset: KeplerDatasetProfile): CatalogGeometry[] {
  if (dataset.mixedGeometry) {
    const mixed = dataset.mixedGeometry;
    const present = GEOMETRY_FAMILIES.filter(family => mixed[family] > 0);
    if (present.length > 0) {
      return present;
    }
  }
  const geometry = getKeplerDatasetGeometry(dataset);
  return geometry === undefined ? [] : [geometry];
}

/**
 * Whether a loaded dataset can fill a binding input: its kind is among the kinds
 * {@link KEPLER_DATASET_KIND_INPUT_KINDS} lists for the input's kind, and a geometry it holds is
 * one the input accepts (a layer that mixes families is judged later: the catalog rejects an input
 * whose features are not all of an accepted kind). `restrictedGeometry` narrows the accepted
 * geometry further (a curated analysis whose slot is `polygons` takes areas only). Typed table
 * ports instead read geometry-free attributes profiles; raster ports read surface profiles whose
 * rowCount is the decoded cell count, not the workflow metadata table's zero rows.
 */
export function isKeplerDatasetCompatibleWithInput(
  dataset: KeplerDatasetProfile,
  input: CatalogInputSpec,
  inputKind: GPUSpatialAnalysisDataKind,
  restrictedGeometry?: CatalogGeometry
): boolean {
  if (
    input.acceptsTable &&
    dataset.kind === 'attributes' &&
    dataset.geometryColumns.length === 0 &&
    getKeplerDatasetGeometries(dataset).length === 0
  )
    return restrictedGeometry === undefined;
  if (input.dataKind === 'table' || input.dataKind === 'raster') {
    const expectedKind = input.dataKind === 'table' ? 'attributes' : 'surface';
    return (
      restrictedGeometry === undefined &&
      inputKind === expectedKind &&
      dataset.kind === expectedKind &&
      dataset.geometryColumns.length === 0 &&
      getKeplerDatasetGeometries(dataset).length === 0
    );
  }
  const geometries = getKeplerDatasetGeometries(dataset).filter(
    geometry =>
      input.accepts.includes(geometry) &&
      (restrictedGeometry === undefined || geometry === restrictedGeometry)
  );
  if (geometries.length === 0) {
    return false;
  }
  return (KEPLER_DATASET_KIND_INPUT_KINDS[dataset.kind] ?? []).includes(inputKind);
}

// ---------------------------------------------------------------------------------------------
// Applicability of one binding
// ---------------------------------------------------------------------------------------------

export type KeplerApplicabilityStatus =
  | 'ready'
  | 'needs-confirmation'
  | 'needs-mapping'
  | 'blocked';

/** A requirement result and the binding input it was evaluated for. */
export type KeplerApplicabilityRequirement = GPUAnalysisRequirementResult & {input: string};

/** One thing a person has to supply or settle before the analysis can run. */
export type KeplerApplicabilityMissing = {
  what: string;
  /** What to do about it. Always present on entries this module writes. */
  remedy?: string;
};

/**
 * A guessed or unknown fact that the result depends on. A host can ask the person once and the
 * question settles every analysis that carries the fact.
 */
export type KeplerGuessedFact = {
  kind: 'row-semantics' | 'coordinate-model' | 'measurement';
  datasetId: string;
  /** The column, for a measurement fact. */
  column?: string;
  /** How the fact stands for this analysis: it fits, contradicts the analysis, or is not known. */
  effect?: 'fits' | 'contradicts' | 'unknown';
};

export type KeplerOperationApplicability = {
  bindingId: string;
  status: KeplerApplicabilityStatus;
  /** Binding input name to dataset id; undefined when no loaded dataset fits. */
  inputAssignments: Readonly<Record<string, string | undefined>>;
  /** Equally supported complete assignments; the host asks the analyst to choose one. */
  inputAssignmentAlternatives: readonly Readonly<Record<string, string | undefined>>[];
  /** Actual output kinds, including geometry inherited through `kindFrom`, by table id. */
  outputKinds: Readonly<Record<string, GPUSpatialAnalysisDataKind>>;
  /** Binding string parameter to the column or columns written into it. */
  fieldBindings: Readonly<Record<string, string | readonly string[]>>;
  /** Default-resolution details shared with executable workbench controls. */
  inactive: readonly string[];
  confirmations: readonly string[];
  /** Descriptor requirements and the binding's own capacities, for the inputs that have a dataset. */
  requirements: readonly KeplerApplicabilityRequirement[];
  /** Everything a person has to supply or settle; empty exactly when the status is `ready`. */
  missing: readonly KeplerApplicabilityMissing[];
  /** Plain phrases for why this fits: `points with a time column`, `48,210 rows`. */
  reasons: readonly string[];
  /** Things to read with care; none of them stops the analysis. */
  warnings: readonly string[];
  /** The part of `warnings` that says a column is not what the method is meant for. */
  measurementWarnings: readonly string[];
  /**
   * Warnings about a column the person chose (or a parameter they set) that the engine would not
   * have chosen. They stay visible and do not change the status: the choice is theirs.
   */
  acknowledgedWarnings: readonly string[];
  /** The descriptor's notes, passed through unchanged. */
  notes: readonly string[];
  /** Roles that received a column, for ranking by how much of the data is used. */
  filledRoles: readonly {
    input: string;
    role: string;
    required: boolean;
    /** Type of the first column bound to the role. */
    columnType?: GPUAnalysisColumnType;
    /** True when the column is a declared or high-confidence fit; a low-confidence guess is false. */
    confident?: boolean;
  }[];
  /**
   * Candidate columns for roles left empty on purpose (a weight or value role that changes what
   * the analysis computes is never filled unasked), best first.
   */
  roleCandidates: Readonly<Record<string, readonly string[]>>;
  /** Facts the result rests on that are only guessed or unknown. */
  guessedFacts: readonly KeplerGuessedFact[];
  /** True when the person declared what the rows are and the analysis is made for another meaning. */
  declaredContradiction: boolean;
  /** True when a required column has no proposal at all (the person must supply or add one). */
  hasEmptyRole: boolean;
  /**
   * Facts a subject-only call cannot know (measurement levels, coordinates, row counts). They do
   * not change the status in that mode and are listed here.
   */
  caveats: readonly string[];
  parameters: Readonly<Record<string, CatalogParameterValue>>;
  resolutions: readonly KeplerParameterResolution[];
  basic: readonly string[];
  advanced: readonly string[];
};

export type KeplerApplicabilityOptions = {
  /** Columns a person chose, by the name of the binding string parameter that receives them. */
  fieldBindings?: Readonly<Record<string, string | readonly string[]>>;
  /** Require the selected assignment to use this dataset in at least one input. */
  requiredDatasetId?: string;
  /** Dataset choices made by the analyst, by binding input name. */
  inputAssignments?: Readonly<Record<string, string>>;
  /** Curated slot names whose whole words may identify a dataset for a binding input. */
  inputNames?: Readonly<Record<string, readonly string[]>>;
  /** Parameter values a person or a curated analysis fixes. */
  overrides?: Readonly<Record<string, CatalogParameterValue>>;
  /** Geometry an input is restricted to, by input name. */
  inputGeometries?: Readonly<Record<string, CatalogGeometry>>;
  /**
   * True when the row counts are not known (a host that has only the columns). The row and vertex
   * requirements then evaluate to `unknown` instead of judging a count of zero.
   */
  unknownRowCount?: boolean;
  /**
   * Where the operation will run; default `gpu`. On `cpu` a binding with no CPU engine is blocked
   * and the CPU work caps apply (a larger cell size, or a block naming the row limit).
   */
  executionEngine?: 'gpu' | 'cpu';
  /**
   * True for a judgement made from the shape of a dataset alone (no statistics): what such a call
   * cannot know does not lower the status and is listed in `caveats` instead.
   */
  subjectOnly?: boolean;
};

/** Role ids that make a phrase of their own in a reason. */
const ROLE_PHRASES: Readonly<Record<string, string>> = {
  time: 'a time column',
  category: 'a category column',
  group: 'a category column'
};

const KIND_PHRASES: Readonly<Partial<Record<GPUSpatialAnalysisDataKind, string>>> = {
  points: 'points',
  lines: 'lines',
  polygons: 'areas',
  network: 'street or network lines',
  regions: 'features',
  cells: 'cells',
  attributes: 'features',
  trajectories: 'tracks'
};

const COLUMN_TYPE_PHRASES: Readonly<Record<string, string>> = {
  number: 'numeric',
  integer: 'numeric',
  category: 'category',
  time: 'time',
  boolean: 'yes/no',
  text: 'text'
};

const COORDINATE_REQUIREMENTS: readonly string[] = [
  'planar-coordinates',
  'geographic-coordinates',
  'declared-coordinates'
];

function formatCountText(value: number): string {
  return value.toLocaleString('en-US');
}

function describeKindPhrase(kind: GPUSpatialAnalysisDataKind): string {
  return KIND_PHRASES[kind] ?? 'features';
}

function describeRoleColumnTypes(role: GPUAnalysisFieldRoleSpec): string {
  const phrases = [...new Set(role.accepts.map(type => COLUMN_TYPE_PHRASES[type] ?? type))];
  return phrases.join(' or ');
}

function isRoleRequired(
  role: GPUAnalysisFieldRoleSpec,
  parameters: Readonly<Record<string, CatalogParameterValue>>
): boolean {
  if (typeof role.required === 'boolean') {
    return role.required;
  }
  return evaluateGPUAnalysisParameterCondition(role.required, parameters);
}

function hasNameAffinity(
  role: GPUAnalysisFieldRoleSpec,
  column: GPUAnalysisColumnProfile
): boolean {
  const roleWords = tokenizeGPUAnalysisColumnName(role.role);
  const columnWords = tokenizeGPUAnalysisColumnName(column.name);
  return roleWords.length > 0 && roleWords.every(word => columnWords.includes(word));
}

// ---------------------------------------------------------------------------------------------
// Column facts and role assignment
// ---------------------------------------------------------------------------------------------

/** Words that name a coordinate column. */
const COORDINATE_COLUMN_WORDS: readonly string[] = [
  'lon',
  'lng',
  'long',
  'longitude',
  'lat',
  'latitude',
  'easting',
  'northing'
];

/**
 * Role ids that mean "the quantity being explained" and "the quantities that explain it". Every
 * descriptor that has such roles uses these two ids (the regression bindings: `response` and
 * `predictors`), which a specification checks, so the engine can recognise a circular fill.
 */
export const KEPLER_RESPONSE_ROLE = 'response';
export const KEPLER_PREDICTOR_ROLE = 'predictors';

/** Role ids of the two parts of a rate; they are filled together from one plausible pair. */
const CASES_ROLE = 'cases';
const POPULATION_ROLE = 'population';

/** Words that name the denominator of a rate (people exposed). */
const POPULATION_WORDS: readonly string[] = ['population', 'pop', 'residents', 'households'];

/**
 * Words that give name evidence for a time column. A role whose meaning is specific (time) is
 * filled only by a column with such evidence or of the profile's `time` type, never by a number
 * whose only qualification is that it is a number.
 */
const TIME_WORDS: readonly string[] = [
  'time',
  'date',
  'timestamp',
  'datetime',
  'epoch',
  'when',
  'year',
  'month',
  'day',
  'hour',
  'created',
  'observed',
  'occurred',
  'recorded'
];

/** Words that give name evidence for a travel cost column. */
const COST_WORDS: readonly string[] = [
  'cost',
  'time',
  'length',
  'distance',
  'duration',
  'minutes',
  'seconds',
  'impedance',
  'travel',
  'weight'
];

type SpecificRoleKind = 'time' | 'location' | 'one-way' | 'cost' | 'filter';

/** The meaning of a role that only a column with evidence may fill; undefined for a general role. */
function getSpecificRoleKind(field: GPUAnalysisFieldRoleSpec): SpecificRoleKind | undefined {
  const roleTokens = tokenizeGPUAnalysisColumnName(field.role);
  const labelTokens = tokenizeGPUAnalysisColumnName(field.label);
  if (field.role === 'time' || labelTokens.includes('time') || labelTokens.includes('timestamp')) {
    return 'time';
  }
  if (roleTokens.includes('oneway') || (roleTokens.includes('one') && roleTokens.includes('way'))) {
    return 'one-way';
  }
  if (roleTokens.includes('location')) return 'location';
  if (roleTokens.includes('cost')) {
    return 'cost';
  }
  if (roleTokens.includes('filter')) {
    return 'filter';
  }
  return undefined;
}

function hasEvidenceFor(kind: SpecificRoleKind, column: GPUAnalysisColumnProfile): boolean {
  const tokens = tokenizeGPUAnalysisColumnName(column.name);
  switch (kind) {
    case 'time':
      return column.type === 'time' || tokens.some(token => TIME_WORDS.includes(token));
    case 'location':
      return (
        column.measurement === 'identifier' ||
        tokens.some(token => ['location', 'site', 'station', 'zone'].includes(token))
      );
    case 'one-way':
      return (
        tokens.includes('oneway') ||
        tokens.includes('direction') ||
        (tokens.includes('one') && tokens.includes('way'))
      );
    case 'cost':
      return tokens.some(token => COST_WORDS.includes(token));
    case 'filter':
      return false;
  }
}

/** How sure the profile is of a column's measurement level. */
function getColumnConfidence(column: GPUAnalysisColumnProfile): 'declared' | 'high' | 'low' {
  if (column.measurementSource === 'declared') {
    return 'declared';
  }
  const confidence =
    column.measurementConfidence ?? guessGPUAnalysisMeasurementLevel(column)?.confidence;
  return confidence === 'high' ? 'high' : 'low';
}

/** Optional quantities that change the calculation must be selected explicitly. */
function isComputingOptionalRole(field: GPUAnalysisFieldRoleSpec, required: boolean): boolean {
  return !required && /^(weight|value|second-?value|scale|error-?variance)$/i.test(field.role);
}

/** Columns that hold a position: the profile's point columns and columns named like one. */
function getCoordinateColumnNames(
  dataset: KeplerDatasetProfile,
  columns: readonly GPUAnalysisColumnProfile[]
): Set<string> {
  const names = new Set<string>();
  if (dataset.pointColumns) {
    names.add(dataset.pointColumns.longitude);
    names.add(dataset.pointColumns.latitude);
  }
  for (const column of columns) {
    const words = tokenizeGPUAnalysisColumnName(column.name);
    if (words.length === 1 && COORDINATE_COLUMN_WORDS.includes(words[0])) {
      names.add(column.name);
    }
  }
  return names;
}

function toNames(value: string | readonly string[] | undefined): readonly string[] {
  return value === undefined || value === '' ? [] : typeof value === 'string' ? [value] : value;
}

/**
 * Replaces each field's `measurement` by `measurementByGeometry` for the geometry of the dataset
 * that is assigned, so role matching, warnings and the preferred-measurement score all read the
 * levels that suit that geometry.
 */
function applyMeasurementByGeometry(
  inputDescriptor: KeplerOperationInputDescriptor,
  geometry: CatalogGeometry | undefined
): KeplerOperationInputDescriptor {
  if (geometry === undefined) {
    return inputDescriptor;
  }
  return {
    ...inputDescriptor,
    fields: inputDescriptor.fields.map(field => {
      const levels = field.measurementByGeometry?.[geometry];
      return levels === undefined ? field : {...field, measurement: levels};
    })
  };
}

/** Columns whose names can be written into the input's string parameters. */
function getUsableColumns(
  profile: KeplerDatasetProfile,
  binding: CatalogOperationBinding,
  inputDescriptor: KeplerOperationInputDescriptor
): readonly GPUAnalysisColumnProfile[] {
  const parametersByName = new Map(
    binding.parameters.map(parameter => [parameter.name, parameter])
  );
  const maximumLength = Math.min(
    ...inputDescriptor.fields.map(
      field => parametersByName.get(field.parameter)?.maximumLength ?? 64
    ),
    64
  );
  const allowsMany = inputDescriptor.fields.some(field => field.multiple !== undefined);
  return profile.columns.filter(
    column => column.name.length <= maximumLength && !(allowsMany && column.name.includes(','))
  );
}

type RoleProblem = {
  what: string;
  remedy: string;
  /** True when no column is proposed for the role at all. */
  emptyRole?: boolean;
};

type RoleAssignment = {
  /** Column or columns by role id. */
  roleColumns: Record<string, string | readonly string[]>;
  unfilledRoles: readonly string[];
  warnings: readonly string[];
  /** Required roles filled by a poor column, or left empty, so that a person should choose. */
  mappingReasons: readonly RoleProblem[];
  /** Reasons the analysis cannot read the data at all (a time column of a type it cannot read). */
  hardReasons: readonly RoleProblem[];
  /** Candidates for roles left empty on purpose, best first. */
  roleCandidates: Readonly<Record<string, readonly string[]>>;
};

function assignRoles(
  inputDescriptor: KeplerOperationInputDescriptor,
  dataset: KeplerDatasetProfile,
  columns: readonly GPUAnalysisColumnProfile[],
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  userColumns: Readonly<Record<string, string | readonly string[]>>
): RoleAssignment {
  const roleColumns: Record<string, string | readonly string[]> = {};
  const takenNames = new Set<string>();
  const userRoles = new Set<string>();
  for (const field of inputDescriptor.fields) {
    const pointCoordinate =
      field.autoAssign === 'point-longitude'
        ? dataset.pointColumns?.longitude
        : field.autoAssign === 'point-latitude'
          ? dataset.pointColumns?.latitude
          : undefined;
    const chosen = userColumns[field.parameter] ?? pointCoordinate;
    if (chosen !== undefined) {
      roleColumns[field.role] = chosen;
      userRoles.add(field.role);
      if (typeof field.required === 'boolean' || isRoleRequired(field, parameters)) {
        for (const name of toNames(chosen)) {
          takenNames.add(name);
        }
      }
    }
  }
  const unfilled: string[] = [];
  const warnings: string[] = [];
  const mappingReasons: RoleProblem[] = [];
  const hardReasons: RoleProblem[] = [];
  const roleCandidates: Record<string, string[]> = {};
  for (const field of inputDescriptor.fields) {
    const chosen = roleColumns[field.role];
    if (
      chosen === undefined ||
      (typeof field.required !== 'boolean' && !isRoleRequired(field, parameters))
    ) {
      continue;
    }
    const names = toNames(chosen);
    const minimum =
      isRoleRequired(field, parameters) || names.length > 0 ? (field.multiple?.minimum ?? 1) : 0;
    const maximum = field.multiple?.maximum ?? 1;
    if (names.length < minimum || names.length > maximum || new Set(names).size !== names.length) {
      mappingReasons.push({
        what: `evaluateKeplerOperationApplicability fieldBindings.${field.parameter} for "${field.label}" must contain ${minimum} to ${maximum} distinct columns; got ${names.length}`,
        remedy: `Choose ${minimum} to ${maximum} distinct columns`
      });
      delete roleColumns[field.role];
    } else if (
      names.some(
        name =>
          !columns.some(
            column => column.name === name && matchGPUAnalysisFieldRole(field, [column]).length > 0
          )
      )
    ) {
      hardReasons.push({
        what: `evaluateKeplerOperationApplicability fieldBindings.${field.parameter} for "${field.label}" must name ${describeRoleColumnTypes(field)} columns in "${dataset.label}": ${names
          .map(name => {
            const column = columns.find(candidate => candidate.name === name);
            return column
              ? `${name} holds ${column.type} values, but this role reads ${describeRoleColumnTypes(field)} columns`
              : `"${name}" is not in this dataset`;
          })
          .join('; ')}`,
        remedy: 'Choose compatible columns from this dataset'
      });
      delete roleColumns[field.role];
    } else if (!field.multiple && names.length === 1) {
      roleColumns[field.role] = names[0];
    }
  }
  const coordinateNames = getCoordinateColumnNames(dataset, columns);
  const available = columns.filter(column => !takenNames.has(column.name));
  // A coordinate column never fills a role while another candidate exists.
  let pool = available.filter(column => !coordinateNames.has(column.name));
  const general: GPUAnalysisFieldRoleSpec[] = [];

  // Roles whose meaning is specific, roles that change what is computed, and the parts of a rate
  // are not filled from a column by its storage type alone.
  for (const field of inputDescriptor.fields) {
    if (userRoles.has(field.role)) {
      continue;
    }
    const required = isRoleRequired(field, parameters);
    if (field.autoAssign === false) {
      roleCandidates[field.role] = matchGPUAnalysisFieldRole(field, pool)
        .map(candidate => candidate.name)
        .slice(0, 5);
      if (required) unfilled.push(field.role);
      continue;
    }
    if (field.autoAssign !== undefined) {
      roleCandidates[field.role] = matchGPUAnalysisFieldRole(field, columns)
        .map(candidate => candidate.name)
        .slice(0, 5);
      if (
        required ||
        (field.autoAssign !== 'explicit' &&
          (dataset.pointColumns ||
            (dataset.geometryFeatureCounts === undefined &&
              dataset.geometryColumns.length === 0 &&
              dataset.catalogGeometry !== 'line')))
      ) {
        mappingReasons.push({
          what: `Choose the ${field.autoAssign === 'explicit' ? 'explicit' : 'native point'} column for "${field.label}"`,
          remedy: 'Declare both pointColumns or explicitly map the required fields',
          emptyRole: true
        });
      }
      continue;
    }
    const specific = getSpecificRoleKind(field);
    if (specific !== undefined) {
      const evidence = pool.filter(column => hasEvidenceFor(specific, column));
      const matches = matchGPUAnalysisFieldRole(field, evidence);
      // A named location identifier outranks an unrelated numeric category.
      const match =
        (specific === 'location'
          ? matches.find(candidate =>
              hasNameAffinity(field, evidence.find(column => column.name === candidate.name)!)
            )
          : undefined) ?? matches[0];
      if (match) {
        roleColumns[field.role] = match.name;
        pool = pool.filter(column => column.name !== match.name);
        if (match.warning !== undefined) {
          warnings.push(`${field.label}: ${match.warning}`);
        }
        continue;
      }
      const anyType = matchGPUAnalysisFieldRole(field, pool).map(candidate => candidate.name);
      roleCandidates[field.role] = anyType.slice(0, 5);
      if (specific === 'time' && evidence.length > 0) {
        hardReasons.push({
          what: `"${field.label}": ${evidence.map(column => column.name).join(', ')} holds the time as ${evidence[0].type}, which ${describeRoleColumnTypes(field)} columns only are read`,
          remedy: 'Convert the time column to numbers (for example epoch seconds)'
        });
      } else if (required) {
        if (anyType.length === 0) {
          unfilled.push(field.role);
        } else {
          mappingReasons.push({
            what: `"${field.label}": no column is named or typed like ${specific === 'time' ? 'a time' : specific === 'location' ? 'a location identifier' : specific === 'one-way' ? 'a one-way flag' : 'a cost'}, so none was proposed`,
            remedy: `Choose the column that holds this (${anyType.slice(0, 3).join(', ')} fit by type only)`,
            emptyRole: true
          });
        }
      }
      continue;
    }
    if (isComputingOptionalRole(field, required)) {
      roleCandidates[field.role] = matchGPUAnalysisFieldRole(field, pool)
        .filter(candidate => candidate.fit === 'preferred')
        .map(candidate => candidate.name)
        .slice(0, 5);
      continue;
    }
    general.push(field);
  }

  // Cases and population are filled together from the best plausible rate pair, or not at all.
  const casesField = general.find(field => field.role === CASES_ROLE);
  const populationField = general.find(field => field.role === POPULATION_ROLE);
  let remaining = general;
  if (casesField && populationField) {
    remaining = general.filter(field => field !== casesField && field !== populationField);
    const pair = suggestGPUAnalysisRatePairs(pool)[0];
    if (pair) {
      roleColumns[CASES_ROLE] = pair.numerator;
      roleColumns[POPULATION_ROLE] = pair.denominator;
      pool = pool.filter(
        column => column.name !== pair.numerator && column.name !== pair.denominator
      );
    } else {
      mappingReasons.push({
        what: `"${casesField.label}" and "${populationField.label}": no pair of a count of cases and a larger population count was found, so nothing was proposed`,
        remedy: 'Add a count of cases and a population column to the layer, or choose them',
        emptyRole: true
      });
    }
  }

  // The remaining roles by the roles module's assignment, without the parts of a rate on both
  // sides of a model.
  const parameterValues = Object.fromEntries(
    Object.entries(parameters).filter(([, value]) => !Array.isArray(value))
  ) as Record<string, string | number | boolean>;
  const run = (candidates: readonly GPUAnalysisColumnProfile[]) =>
    assignGPUAnalysisFieldRoles(remaining, candidates, parameterValues);
  const baselinePool = [...pool, ...available.filter(column => coordinateNames.has(column.name))];
  const baseline = run(baselinePool);
  const excluded = new Set<string>();
  const ratePairs = suggestGPUAnalysisRatePairs(pool);
  const circularPartners = new Map<string, string>();
  let result = run(pool);
  for (let step = 0; step < pool.length; step++) {
    const responses = toNames(
      result.assignment[KEPLER_RESPONSE_ROLE] ?? roleColumns[KEPLER_RESPONSE_ROLE]
    );
    const predictors = toNames(result.assignment[KEPLER_PREDICTOR_ROLE]);
    let predictorToDrop: string | undefined;
    // A rate and a population on opposite sides: the rate was almost surely divided by it.
    for (const response of responses) {
      const responseColumn = pool.find(column => column.name === response);
      if (responseColumn?.measurement !== 'rate') {
        continue;
      }
      const denominator = predictors.find(name =>
        tokenizeGPUAnalysisColumnName(name).some(token => POPULATION_WORDS.includes(token))
      );
      if (denominator !== undefined) {
        predictorToDrop = denominator;
        circularPartners.set(denominator, `probably the denominator of ${response}`);
      }
    }
    for (const pair of predictorToDrop === undefined ? ratePairs : []) {
      if (responses.includes(pair.numerator) && predictors.includes(pair.denominator)) {
        predictorToDrop = pair.denominator;
        circularPartners.set(pair.denominator, `the denominator of ${pair.numerator}`);
      } else if (responses.includes(pair.denominator) && predictors.includes(pair.numerator)) {
        predictorToDrop = pair.numerator;
        circularPartners.set(pair.numerator, `the numerator over ${pair.denominator}`);
      }
      if (predictorToDrop !== undefined) {
        break;
      }
    }
    if (predictorToDrop === undefined) {
      break;
    }
    excluded.add(predictorToDrop);
    result = run(pool.filter(column => !excluded.has(column.name)));
  }
  Object.assign(roleColumns, result.assignment);
  warnings.push(...result.warnings);

  // A required role that only an excluded column could fill keeps that column, as a question.
  for (const roleId of result.unfilledRoles) {
    const field = remaining.find(candidate => candidate.role === roleId);
    const names = toNames(baseline.assignment[roleId]);
    if (!field || names.length === 0) {
      unfilled.push(roleId);
      continue;
    }
    roleColumns[roleId] = baseline.assignment[roleId] as string | readonly string[];
    const why = names
      .map(name =>
        coordinateNames.has(name)
          ? `${name} is a coordinate column`
          : `${name} is ${circularPartners.get(name) ?? 'already used'} of the response, which makes the fit circular`
      )
      .join('; ');
    // Nothing good is proposed for the role: the poor column is kept only so that the person can
    // see what it was, and the analysis is treated as lacking the column.
    mappingReasons.push({
      what: `"${field.label}": the only candidate is poor: ${why}`,
      remedy: 'Choose the column that holds this quantity, or add one',
      emptyRole: true
    });
  }

  // A low-confidence guess does not fill an optional role either: it would change the result.
  for (const field of remaining) {
    const required = isRoleRequired(field, parameters);
    const assigned = roleColumns[field.role];
    if (required || assigned === undefined || userRoles.has(field.role) || !field.measurement) {
      continue;
    }
    const column = columns.find(candidate => candidate.name === toNames(assigned)[0]);
    if (column && getColumnConfidence(column) === 'low') {
      delete roleColumns[field.role];
      roleCandidates[field.role] = toNames(assigned).slice(0, 5) as string[];
    }
  }
  return {
    roleColumns,
    unfilledRoles: unfilled,
    warnings,
    mappingReasons,
    hardReasons,
    roleCandidates
  };
}

// ---------------------------------------------------------------------------------------------
// Assigning datasets to inputs
// ---------------------------------------------------------------------------------------------

type AssignedInput = {
  inputName: string;
  dataset: KeplerDatasetProfile;
  roles: RoleAssignment;
};

type InputPairing = {
  dataset: KeplerDatasetProfile;
  roles: RoleAssignment;
  fit: number;
};

/** Fit of a dataset that has the input's own kind, against one that fits only by geometry. */
const EXACT_KIND_FIT = 4;
const GEOMETRY_ONLY_FIT = 1;
/** A matching row meaning distinguishes assignments more strongly than a matching slot name. */
const SEMANTICS_ASSIGNMENT_FIT = 5;
/** A dataset whose id or label names the input (`before` for the input `before`) is preferred for it. */
const NAME_AFFINITY_FIT = 3;
const GENERIC_INPUT_WORDS: readonly string[] = ['points', 'lines', 'polygons', 'features', 'layer'];

/**
 * Datasets tried for one input. The assignment search is exhaustive over these, so five inputs
 * with this many datasets each cost at most 6^5 = 7,776 leaves (a few milliseconds); the datasets
 * beyond the best six by fit, then by id, are not tried, and the result says how many.
 */
export const KEPLER_MAXIMUM_DATASETS_PER_INPUT = 6;

/** Bound the alternatives shown in a form; the complete search still counts every tie. */
const MAXIMUM_ASSIGNMENT_ALTERNATIVES = 8;

function buildInputAssignments(
  pairings: readonly (readonly InputPairing[])[],
  countProblems: (choice: readonly (InputPairing | undefined)[]) => number,
  requiredDatasetId?: string
): {
  choice: (InputPairing | undefined)[];
  alternatives: (InputPairing | undefined)[][];
  count: number;
} {
  let best:
    | {filled: number; problems: number; fit: number; choice: (InputPairing | undefined)[]}
    | undefined;
  let alternatives: (InputPairing | undefined)[][] = [];
  let count = 0;
  const choice: (InputPairing | undefined)[] = [];
  const used = new Set<string>();
  const visit = (index: number, filled: number, fit: number): void => {
    if (index === pairings.length) {
      if (requiredDatasetId !== undefined && !used.has(requiredDatasetId)) return;
      const problems = countProblems(choice);
      if (
        !best ||
        filled > best.filled ||
        (filled === best.filled &&
          (problems < best.problems || (problems === best.problems && fit > best.fit)))
      ) {
        best = {filled, problems, fit, choice: [...choice]};
        alternatives = [[...choice]];
        count = 1;
      } else if (filled === best.filled && problems === best.problems && fit === best.fit) {
        count++;
        if (alternatives.length < MAXIMUM_ASSIGNMENT_ALTERNATIVES) {
          alternatives.push([...choice]);
        }
      }
      return;
    }
    for (const pairing of pairings[index]) {
      if (used.has(pairing.dataset.datasetId)) {
        continue;
      }
      used.add(pairing.dataset.datasetId);
      choice[index] = pairing;
      visit(index + 1, filled + 1, fit + pairing.fit);
      used.delete(pairing.dataset.datasetId);
    }
    choice[index] = undefined;
    visit(index + 1, filled, fit);
  };
  visit(0, 0, 0);
  return {choice: best?.choice ?? [], alternatives, count};
}

/** Columns shared by roles that feed one parameter must be readable from every assigned layer. */
function getSharedFieldCandidates(
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  choice: readonly (InputPairing | undefined)[]
): Map<string, string[]> {
  const groups = new Map<
    string,
    {
      field: KeplerOperationInputDescriptor['fields'][number];
      dataset: KeplerDatasetProfile;
      columns: readonly GPUAnalysisColumnProfile[];
    }[]
  >();
  binding.inputs.forEach((input, index) => {
    const pairing = choice[index];
    const inputDescriptor = descriptor.inputs.find(entry => entry.name === input.name);
    if (!pairing || !inputDescriptor) {
      return;
    }
    for (const field of applyMeasurementByGeometry(
      inputDescriptor,
      getKeplerDatasetGeometry(pairing.dataset)
    ).fields) {
      const group = groups.get(field.parameter) ?? [];
      const usable = getUsableColumns(pairing.dataset, binding, inputDescriptor);
      const coordinateNames = getCoordinateColumnNames(pairing.dataset, usable);
      group.push({
        field,
        dataset: pairing.dataset,
        columns: usable.filter(column => !coordinateNames.has(column.name))
      });
      groups.set(field.parameter, group);
    }
  });
  const shared = new Map<string, string[]>();
  for (const [parameter, members] of groups) {
    if (members.length < 2) {
      continue;
    }
    shared.set(
      parameter,
      members[0].columns
        .filter(column =>
          members.every(member =>
            member.columns.some(
              other =>
                other.name === column.name &&
                (other.type === column.type ||
                  (['integer', 'number'].includes(other.type) &&
                    ['integer', 'number'].includes(column.type))) &&
                matchGPUAnalysisFieldRole(member.field, [other]).length > 0
            )
          )
        )
        .map(column => column.name)
    );
  }
  return shared;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function getRowNoun(count: number): string {
  return count === 1 ? 'row' : 'rows';
}

function overrideRowCountResult(
  result: GPUAnalysisRequirementResult
): GPUAnalysisRequirementResult {
  if (
    result.id !== 'minimum-rows' &&
    result.id !== 'maximum-rows' &&
    result.id !== 'maximum-vertices'
  ) {
    return result;
  }
  return {
    ...result,
    state: 'unknown',
    evidence: 'not-checkable',
    message: 'The number of rows is not known yet',
    ...(result.remedy === undefined ? {} : {remedy: result.remedy})
  };
}

/** A failure that rests on a guess is a question for the person, never a block. */
function softenGuessedFailure(result: GPUAnalysisRequirementResult): GPUAnalysisRequirementResult {
  if (result.severity !== 'block' || result.state !== 'failed' || result.evidence !== 'guess') {
    return result;
  }
  return {
    ...result,
    state: 'unknown',
    // Coordinate guesses are already `unknown` upstream; the row semantics still fail.
    remedy:
      result.id === 'row-semantics'
        ? 'Confirm what the rows stand for'
        : 'Confirm the guessed value'
  };
}

// ---------------------------------------------------------------------------------------------
// Checks across datasets
// ---------------------------------------------------------------------------------------------

/** Bounding boxes of two profiles overlap, with the antimeridian convention (west above east). */
function extentsOverlap(
  left: KeplerDatasetProfile,
  right: KeplerDatasetProfile
): boolean | undefined {
  const a = left.extent ?? left.spacing?.extent;
  const b = right.extent ?? right.spacing?.extent;
  if (!a || !b || left.coordinates?.model !== right.coordinates?.model) {
    return undefined;
  }
  const geographic = left.coordinates?.model === 'geographic';
  const margin =
    0.05 *
    Math.max(
      Math.abs(a[2] - a[0]) || 0,
      Math.abs(a[3] - a[1]) || 0,
      Math.abs(b[2] - b[0]) || 0,
      Math.abs(b[3] - b[1]) || 0
    );
  const overlapsLatitude = a[1] - margin <= b[3] && b[1] - margin <= a[3];
  if (!overlapsLatitude) {
    return false;
  }
  const spans = (box: readonly number[]): [number, number] => {
    const east = geographic && box[0] > box[2] ? box[2] + 360 : box[2];
    return [box[0], east];
  };
  const [aWest, aEast] = spans(a);
  const [bWest, bEast] = spans(b);
  const shifts = geographic ? [-360, 0, 360] : [0];
  return shifts.some(shift => aWest - margin <= bEast + shift && bWest + shift - margin <= aEast);
}

/** Only a dataset id establishes identity; equal metadata can describe independent observations. */
function isSameLayer(left: KeplerDatasetProfile, right: KeplerDatasetProfile): boolean {
  return (left.sourceDatasetId ?? left.datasetId) === (right.sourceDatasetId ?? right.datasetId);
}

// ---------------------------------------------------------------------------------------------
// Applicability of one binding
// ---------------------------------------------------------------------------------------------

type MissingKind = 'hard' | 'mapping' | 'confirmation';
type MissingEntry = KeplerApplicabilityMissing & {
  kind: MissingKind;
  emptyRole?: boolean;
  parameterConfirmation?: boolean;
};

/** Parameter units that are lengths in the coordinate frame of the data. */
const LENGTH_UNITS: readonly string[] = ['crs', 'm', 'metre', 'meter', 'coordinate units'];

/** Row semantics a person can declare, as options of a question. */
const ROW_SEMANTICS_OPTIONS = ['events', 'objects', 'surface-samples'] as const;

/** JSON with the keys of every object in sorted order, so equal options give equal keys. */
function stableStringify(value: unknown): string {
  return JSON.stringify(value, (_key, entry) =>
    entry && typeof entry === 'object' && !Array.isArray(entry)
      ? Object.fromEntries(
          Object.entries(entry).sort(([left], [right]) => compareText(left, right))
        )
      : entry
  );
}

type ApplicabilityCacheNode = {
  next: WeakMap<object, ApplicabilityCacheNode>;
  results: Map<string, KeplerOperationApplicability>;
  profileFacts?: string;
};

const APPLICABILITY_CACHE = new WeakMap<object, WeakMap<object, ApplicabilityCacheNode>>();

function getCacheNode(
  binding: object,
  descriptor: object,
  datasets: readonly object[]
): ApplicabilityCacheNode {
  let byDescriptor = APPLICABILITY_CACHE.get(binding);
  if (!byDescriptor) {
    byDescriptor = new WeakMap();
    APPLICABILITY_CACHE.set(binding, byDescriptor);
  }
  let node: ApplicabilityCacheNode | undefined = byDescriptor.get(descriptor);
  if (!node) {
    node = {next: new WeakMap(), results: new Map()};
    byDescriptor.set(descriptor, node);
  }
  for (const dataset of datasets) {
    let following: ApplicabilityCacheNode | undefined = node.next.get(dataset);
    if (!following) {
      following = {next: new WeakMap(), results: new Map()};
      node.next.set(dataset, following);
    }
    node = following;
  }
  return node;
}

/**
 * Decides whether one analysis can run on the loaded datasets, which dataset fills each input,
 * which column fills each role, and what the parameters are. The result is cached per binding,
 * descriptor, profile facts and options, so
 * a host that asks again for the same state reuses the result. Profiles should be immutable values,
 * replaced on change; a facts fingerprint also invalidates results for mutable hosts. Do not modify
 * a returned result.
 *
 * Rules
 * - An input is filled by a dataset when {@link isKeplerDatasetCompatibleWithInput} holds (the
 *   compatibility table is {@link KEPLER_DATASET_KIND_INPUT_KINDS}). Datasets are considered in
 *   id order, so the result does not depend on the order they were loaded in.
 * - Each dataset fills at most one input. The assignment that fills most inputs wins; then
 *   complete assignments whose columns fit, then kind, row meaning and whole-word slot names.
 *   Equally supported complete assignments require an analyst's choice.
 *   At most {@link KEPLER_MAXIMUM_DATASETS_PER_INPUT} datasets are tried per input.
 * - Columns: a role named after a meaning (time, one-way, cost) is filled only by a column with
 *   name or type evidence; a weight or value role that is optional is never filled unasked and its
 *   candidates are listed; cases and population are filled together from the best plausible rate
 *   pair or not at all; coordinate columns and identifiers do not fill numeric roles while another
 *   candidate exists; the two parts of a rate are never on both sides of a regression.
 * - A column a person chose (`fieldBindings`, or an override of a column parameter) is checked
 *   against the role's accepted types (blocked) and its intended measurement (a warning that
 *   stays, `acknowledgedWarnings`, status unchanged).
 * - `blocked`: an input cannot be filled, a required column has no candidate or is unreadable, a
 *   chosen column is missing or of the wrong type, the layer mixes geometry families the input does
 *   not read, declared coordinate models disagree, an explicitly required input pair does not
 *   overlap, a `block` requirement
 *   failed on declared or measured facts, a coordinate declaration contradicts the extent, or the
 *   operation needs a GPU and the host has none.
 * - `needs-mapping`: it could run, but several columns fit equally well, there is no plausible rate pair, a
 *   length default could not be derived from the data, or a required column has no proposal.
 * - `needs-confirmation`: an operational `block` requirement is unknown (for example capacity),
 *   or declared row semantics contradict the analysis. Unknown row meaning and guessed metadata
 *   stay visible as warnings and questions, without preventing a run or a field correction,
 *   unless an explicit blocking `declared-coordinates` requirement asks for a declaration.
 * - `ready`: none of the above. Warning-level requirements never change the status.
 * - Capacity comes from the binding itself even when the descriptor does not state it.
 */
export function evaluateKeplerOperationApplicability(
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  datasets: readonly KeplerDatasetProfile[],
  options: KeplerApplicabilityOptions = {}
): KeplerOperationApplicability {
  const ordered = [...datasets].sort((left, right) => compareText(left.datasetId, right.datasetId));
  const node = getCacheNode(binding, descriptor, ordered);
  const profileFacts = stableStringify(ordered);
  if (node.profileFacts !== profileFacts) {
    node.results.clear();
    node.profileFacts = profileFacts;
  }
  const key = stableStringify(options);
  const cached = node.results.get(key);
  if (cached) {
    return cached;
  }
  const result = evaluateApplicability(binding, descriptor, ordered, options);
  node.results.set(key, result);
  return result;
}

function evaluateApplicability(
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  datasets: readonly KeplerDatasetProfile[],
  options: KeplerApplicabilityOptions
): KeplerOperationApplicability {
  // Reuse validated publication mappings, including embedded time units. Explicit user choices
  // take precedence, and coordinate declarations may have changed since initial profiling.
  const nativeProfiles = descriptor.inputs
    .filter(
      input =>
        input.kind === 'trajectories' &&
        binding.inputs.some(specification => specification.name === input.name)
    )
    .flatMap(input =>
      datasets.filter(
        dataset =>
          dataset.trajectory &&
          (!options.inputAssignments?.[input.name] ||
            options.inputAssignments[input.name] === dataset.datasetId)
      )
    );
  const nativeProfile = nativeProfiles.length === 1 ? nativeProfiles[0] : undefined;
  const overrides = {
    ...nativeProfile?.trajectory?.parameters,
    ...(nativeProfile?.coordinates
      ? {coordinateSystem: nativeProfile.coordinates.model === 'planar' ? 'planar' : 'EPSG:4326'}
      : {}),
    ...options.overrides
  };
  const inputDescriptors = new Map(descriptor.inputs.map(input => [input.name, input]));
  // A column parameter set by an override is a column the person chose.
  const fieldParameters = new Set(
    descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter))
  );
  const userColumns: Record<string, string | readonly string[]> = {};
  for (const [name, value] of Object.entries(overrides)) {
    if (fieldParameters.has(name) && typeof value === 'string') {
      userColumns[name] = value.includes(',') ? value.split(',').map(part => part.trim()) : value;
    }
  }
  Object.assign(userColumns, options.fieldBindings ?? {});
  // Field overrides are handled as mappings, so inactive choices never reach parameter validation.
  const parameterOverrides = Object.fromEntries(
    Object.entries(overrides).filter(([name]) => !fieldParameters.has(name))
  );

  // Parameter values that the conditional roles are judged against, before any column is chosen.
  const baseParameters = resolveKeplerOperationParameters(
    binding,
    descriptor,
    {},
    {overrides: parameterOverrides}
  ).parameters;

  const entries: MissingEntry[] = [];
  const add = (
    kind: MissingKind,
    what: string,
    remedy: string,
    extra: {emptyRole?: boolean; parameterConfirmation?: boolean} = {}
  ): void => {
    if (!entries.some(entry => entry.what === what)) {
      entries.push({kind, what, remedy, ...extra});
    }
  };

  const executionEngine = options.executionEngine ?? 'gpu';
  if (binding.inputs.length === 0) {
    add(
      'hard',
      'This operation reads no layer, so there is nothing to suggest it for',
      'Open it from the catalog'
    );
  }
  if (binding.engine === 'gpu' && executionEngine === 'cpu') {
    add('hard', `${binding.title} needs a GPU device`, 'Run it where a GPU device is available');
  }

  // Candidate datasets per input: those that fit, best six by fit.
  const droppedNotes: string[] = [];
  const pairings = binding.inputs.map(input => {
    const inputDescriptor = inputDescriptors.get(input.name) ?? {
      name: input.name,
      kind: 'points' as GPUSpatialAnalysisDataKind,
      fields: []
    };
    const all = datasets
      .filter(
        dataset =>
          options.inputAssignments?.[input.name] === undefined ||
          options.inputAssignments[input.name] === dataset.datasetId
      )
      .filter(dataset =>
        isKeplerDatasetCompatibleWithInput(
          dataset,
          input,
          inputDescriptor.kind,
          options.inputGeometries?.[input.name]
        )
      )
      .map((dataset): InputPairing => {
        const effective = applyMeasurementByGeometry(
          inputDescriptor,
          getKeplerDatasetGeometry(dataset)
        );
        const roles = assignRoles(
          effective,
          dataset,
          getUsableColumns(dataset, binding, effective),
          baseParameters,
          userColumns
        );
        const datasetWords = tokenizeGPUAnalysisColumnName(`${dataset.datasetId} ${dataset.label}`);
        const curatedNames =
          options.inputNames?.[input.name] ??
          KEPLER_CURATED_ANALYSES.flatMap(analysis =>
            analysis.execution.status === 'available' && analysis.execution.bindingId === binding.id
              ? analysis.slots
                  .filter(
                    slot =>
                      analysis.execution.status === 'available' &&
                      analysis.execution.inputs[slot.name] === input.name
                  )
                  .map(slot => slot.name)
              : []
          );
        const nameMatches = [input.name, ...curatedNames].some(name => {
          const words = tokenizeGPUAnalysisColumnName(name).filter(
            word => !GENERIC_INPUT_WORDS.includes(word)
          );
          return words.length > 0 && words.every(word => datasetWords.includes(word));
        });
        const semantics = dataset.rowSemantics?.value;
        const semanticsFit =
          inputDescriptor.rowSemantics?.length && semantics
            ? inputDescriptor.rowSemantics.includes(semantics)
              ? SEMANTICS_ASSIGNMENT_FIT
              : -SEMANTICS_ASSIGNMENT_FIT
            : 0;
        const fit =
          (nameMatches ? NAME_AFFINITY_FIT : 0) +
          semanticsFit +
          (dataset.kind === inputDescriptor.kind ? EXACT_KIND_FIT : GEOMETRY_ONLY_FIT);
        return {dataset, roles, fit};
      });
    if (all.length <= KEPLER_MAXIMUM_DATASETS_PER_INPUT) {
      return all;
    }
    const kept = [...all]
      .sort(
        (left, right) =>
          Number(right.dataset.datasetId === options.requiredDatasetId) -
            Number(left.dataset.datasetId === options.requiredDatasetId) ||
          right.fit - left.fit ||
          compareText(left.dataset.datasetId, right.dataset.datasetId)
      )
      .slice(0, KEPLER_MAXIMUM_DATASETS_PER_INPUT)
      .sort((left, right) => compareText(left.dataset.datasetId, right.dataset.datasetId));
    droppedNotes.push(
      `${all.length - kept.length} more loaded layers fit "${input.label}" and were not tried (only the best ${KEPLER_MAXIMUM_DATASETS_PER_INPUT} by fit${options.requiredDatasetId === undefined ? '' : ', reserving a place for the required dataset,'} are)`
    );
    return kept;
  });
  const assignment = buildInputAssignments(
    pairings,
    choice => {
      const roleProblems = choice.reduce(
        (total, pairing) =>
          total +
          (pairing
            ? pairing.roles.unfilledRoles.length +
              pairing.roles.hardReasons.length +
              pairing.roles.mappingReasons.length
            : 0),
        0
      );
      const shared = getSharedFieldCandidates(binding, descriptor, choice);
      return (
        roleProblems +
        [...shared].filter(([parameter, candidates]) => {
          const needed =
            (toNames(userColumns[parameter]).length > 0 &&
              descriptor.inputs.some(input =>
                input.fields.some(
                  field =>
                    field.parameter === parameter &&
                    (typeof field.required === 'boolean' || isRoleRequired(field, baseParameters))
                )
              )) ||
            descriptor.inputs.some(input =>
              input.fields.some(
                field => field.parameter === parameter && isRoleRequired(field, baseParameters)
              )
            );
          const selected = toNames(userColumns[parameter]);
          return (
            needed && (candidates.length === 0 || selected.some(name => !candidates.includes(name)))
          );
        }).length
      );
    },
    options.requiredDatasetId
  );
  const chosen = assignment.choice;
  const inputAssignmentAlternatives = assignment.alternatives.map(choice =>
    Object.fromEntries(
      binding.inputs.map((input, index) => [input.name, choice[index]?.dataset.datasetId])
    )
  );
  if (assignment.count > 1 && chosen.every(Boolean)) {
    add(
      'mapping',
      `Choose which dataset fills each input: ${assignment.count} equally supported assignments (${inputAssignmentAlternatives
        .map(alternative =>
          Object.entries(alternative)
            .map(([input, dataset]) => `${input}=${dataset}`)
            .join(', ')
        )
        .join(' or ')})`,
      'Choose the dataset for each input'
    );
  }
  const sharedCandidates = getSharedFieldCandidates(binding, descriptor, chosen);
  for (const [parameter, candidates] of sharedCandidates) {
    const members = descriptor.inputs.flatMap(input =>
      input.fields.filter(field => field.parameter === parameter)
    );
    const needed =
      members.some(field => isRoleRequired(field, baseParameters)) ||
      (toNames(userColumns[parameter]).length > 0 &&
        members.some(
          field => typeof field.required === 'boolean' || isRoleRequired(field, baseParameters)
        ));
    if (!needed) {
      continue;
    }
    if (candidates.length === 0) {
      add(
        'mapping',
        `Choose a common column for ${parameter}: every input feeding this parameter must contain the same compatible column`,
        'Rename or add a compatible column present in all of these datasets'
      );
      chosen.forEach((pairing, index) => {
        if (!pairing) {
          return;
        }
        for (const field of inputDescriptors.get(binding.inputs[index].name)?.fields ?? []) {
          if (field.parameter === parameter) {
            delete pairing.roles.roleColumns[field.role];
            pairing.roles.unfilledRoles = pairing.roles.unfilledRoles.filter(
              role => role !== field.role
            );
          }
        }
      });
    } else if (userColumns[parameter] === undefined) {
      const field = members[0];
      const firstProfile = chosen.find(pairing =>
        pairing?.dataset.columns.some(column => candidates.includes(column.name))
      )?.dataset;
      const commonColumns =
        firstProfile?.columns.filter(column => candidates.includes(column.name)) ?? [];
      const common = matchGPUAnalysisFieldRole(field, commonColumns)[0]?.name;
      if (common !== undefined) {
        chosen.forEach((pairing, index) => {
          if (!pairing) {
            return;
          }
          for (const member of inputDescriptors.get(binding.inputs[index].name)?.fields ?? []) {
            if (member.parameter === parameter) {
              pairing.roles.roleColumns[member.role] = common;
              pairing.roles.unfilledRoles = pairing.roles.unfilledRoles.filter(
                role => role !== member.role
              );
            }
          }
        });
      }
    }
  }

  const assigned: AssignedInput[] = [];
  const inputAssignments: Record<string, string | undefined> = {};
  binding.inputs.forEach((input, index) => {
    const pairing = chosen[index];
    inputAssignments[input.name] = pairing?.dataset.datasetId;
    if (pairing) {
      assigned.push({inputName: input.name, dataset: pairing.dataset, roles: pairing.roles});
    }
  });

  // Inputs no dataset fills.
  const filledKinds = new Set<string>();
  for (const input of binding.inputs) {
    const kind = inputDescriptors.get(input.name)?.kind ?? 'points';
    if (inputAssignments[input.name] !== undefined) {
      filledKinds.add(kind);
      continue;
    }
    const phrase =
      input.dataKind === 'table'
        ? 'geometry-free table rows'
        : input.dataKind === 'raster'
          ? 'numeric raster cells'
          : describeKindPhrase(kind);
    const second = filledKinds.has(kind) ? 'second ' : '';
    const hasTable = datasets.some(dataset => getKeplerDatasetGeometry(dataset) === undefined);
    add(
      'hard',
      `A ${second}dataset of ${phrase} for "${input.label}"`,
      input.dataKind === 'table'
        ? 'Load a geometry-free table with the required columns'
        : input.dataKind === 'raster'
          ? 'Load one bounded numeric raster band/grid'
          : datasets.length === 0
            ? `Load a dataset of ${phrase}`
            : hasTable && !second
              ? `Load a layer of ${phrase}, or give a table coordinate columns or a geometry column`
              : `Load a layer of ${phrase}`
    );
    filledKinds.add(kind);
  }

  // Columns, requirements and parameters of the inputs that have a dataset.
  const fieldBindings: Record<string, string | readonly string[]> = {};
  const filledRoles: {
    input: string;
    role: string;
    required: boolean;
    columnType?: GPUAnalysisColumnType;
    confident?: boolean;
  }[] = [];
  const measurementWarnings: string[] = [];
  const acknowledgedWarnings: string[] = [];
  const warnings: string[] = [...droppedNotes];
  const requirements: KeplerApplicabilityRequirement[] = [];
  const guessedFacts: KeplerGuessedFact[] = [];
  const roleCandidates: Record<string, readonly string[]> = {};
  if (nativeProfiles.length > 1) {
    add(
      'mapping',
      'Choose which validated trajectory dataset to analyze',
      'Choose the tracks input dataset so its identity mappings and timestamp units can be applied'
    );
  }
  const unusedColumns: string[] = [];
  let declaredContradiction = false;

  const requirementParameters = resolveKeplerOperationParameters(
    binding,
    descriptor,
    Object.fromEntries(assigned.map(entry => [entry.inputName, entry.dataset])),
    {
      overrides: parameterOverrides,
      executionEngine,
      maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount,
      fieldBindings: Object.fromEntries(
        assigned.flatMap(entry =>
          (inputDescriptors.get(entry.inputName)?.fields ?? []).flatMap(field => {
            const value = entry.roles.roleColumns[field.role];
            const active =
              typeof field.required === 'boolean' || isRoleRequired(field, baseParameters);
            // A field can determine its own controller through a measurement default. Retain
            // that proposal for resolving the controller, then judge activity on its final value.
            const controllers =
              typeof field.required === 'boolean'
                ? []
                : getGPUAnalysisParameterComparisons(field.required).map(comparison =>
                    'parameter' in comparison ? comparison.parameter : comparison.whenParameter
                  );
            const determinesController = controllers.some(controller => {
              const controllerPolicy = descriptor.parameters.find(
                parameter => parameter.name === controller
              )?.defaultPolicy;
              return (
                !(controller in parameterOverrides) &&
                controllerPolicy?.from === 'field-measurement' &&
                controllerPolicy.input === entry.inputName &&
                controllerPolicy.role === field.role
              );
            });
            return value === undefined || (!active && !determinesController)
              ? []
              : [[field.parameter, value]];
          })
        )
      )
    }
  ).parameters;

  for (const assignment of assigned) {
    const {inputName, dataset} = assignment;
    const declaredInput = inputDescriptors.get(inputName);
    const input = binding.inputs.find(candidate => candidate.name === inputName);
    if (!declaredInput || !input) {
      continue;
    }
    const measuredInputDescriptor = applyMeasurementByGeometry(
      declaredInput,
      getKeplerDatasetGeometry(dataset)
    );
    const inputDescriptor = {
      ...measuredInputDescriptor,
      fields: measuredInputDescriptor.fields.filter(field => {
        const active =
          typeof field.required === 'boolean' || isRoleRequired(field, requirementParameters);
        if (!active && userColumns[field.parameter] !== undefined) {
          unusedColumns.push(
            `The column ${toNames(userColumns[field.parameter]).join(', ')} chosen for "${field.label}" is not used with the current settings`
          );
        }
        return active;
      })
    };
    const columns = getUsableColumns(dataset, binding, inputDescriptor);
    const roles =
      inputDescriptor.fields.length !== measuredInputDescriptor.fields.length ||
      inputDescriptor.fields.some(
        field =>
          isRoleRequired(field, baseParameters) !== isRoleRequired(field, requirementParameters)
      )
        ? assignRoles(inputDescriptor, dataset, columns, requirementParameters, userColumns)
        : assignment.roles;
    assignment.roles = roles;
    const unfilledRoleIds = new Set(roles.unfilledRoles);
    const userRoleIds = new Set(
      inputDescriptor.fields
        .filter(field => userColumns[field.parameter] !== undefined)
        .map(field => field.role)
    );
    Object.assign(roleCandidates, roles.roleCandidates);

    // A layer that mixes geometry families is rejected by the catalog.
    const families = getKeplerDatasetGeometries(dataset);
    const rejected = families.filter(family => !input.accepts.includes(family));
    if (dataset.mixedGeometry && rejected.length > 0) {
      const counts = rejected
        .map(family => `${dataset.mixedGeometry?.[family].toLocaleString('en-US')} ${family}s`)
        .join(' and ');
      add(
        'hard',
        `"${dataset.label}" mixes geometry kinds: ${counts} that "${input.label}" does not read (it reads ${input.accepts.join(', ')})`,
        `Split or filter the layer so that it holds only ${input.accepts.join(' or ')} features`
      );
    }

    for (const field of inputDescriptor.fields) {
      const boundColumns = roles.roleColumns[field.role];
      const required = isRoleRequired(field, requirementParameters);
      if (boundColumns === undefined) {
        if (unfilledRoleIds.has(field.role)) {
          add(
            'hard',
            `A ${describeRoleColumnTypes(field)} column for "${field.label}" in "${dataset.label}"`,
            `Add a ${describeRoleColumnTypes(field)} column to "${dataset.label}", or choose another dataset`
          );
        }
        continue;
      }
      const names = toNames(boundColumns);
      fieldBindings[field.parameter] = boundColumns;
      // An explicit empty optional mapping selects fallback behavior, not a filled role.
      if (names.length === 0) continue;
      const firstColumn = columns.find(column => column.name === names[0]);
      const confidence = firstColumn ? getColumnConfidence(firstColumn) : 'low';
      filledRoles.push({
        input: inputName,
        role: field.role,
        required,
        confident: confidence !== 'low' || !field.measurement,
        ...(firstColumn ? {columnType: firstColumn.type} : {})
      });
      for (const name of names) {
        if (!dataset.columns.some(column => column.name === name)) {
          add(
            'hard',
            `The column "${name}" chosen for "${field.label}" is not in "${dataset.label}"`,
            'Choose a column of the dataset'
          );
        }
      }
      if (userRoleIds.has(field.role)) {
        // A person's choice is final, but it is checked: its type blocks, its meaning is a
        // warning that stays visible and is marked as accepted by the choice.
        for (const name of names) {
          const column = dataset.columns.find(candidate => candidate.name === name);
          if (!column) {
            continue;
          }
          const match = matchGPUAnalysisFieldRole(field, [column])[0];
          if (!match) {
            add(
              'hard',
              `"${field.label}": ${name} holds ${column.type} values, but this role reads ${describeRoleColumnTypes(field)} columns`,
              `Choose a ${describeRoleColumnTypes(field)} column`
            );
          } else if (match.warning !== undefined) {
            const text = `${field.label}: ${match.warning} (kept because you chose it)`;
            acknowledgedWarnings.push(text);
            warnings.push(text);
          }
        }
        continue;
      }
      if (!required) {
        continue;
      }
      // What to explain and what explains it is the analyst's question, not a fact in the data: a
      // model's columns are proposed and never presented as settled.
      if (field.role === KEPLER_RESPONSE_ROLE || field.role === KEPLER_PREDICTOR_ROLE) {
        add(
          'mapping',
          `Choose the columns of the model: ${names.join(', ')} was proposed for "${field.label}", but what to explain and what explains it is your question`,
          'Choose the column to explain and the columns that may explain it'
        );
      }
      // A required role filled by a guess is checked for fit, confidence and ambiguity.
      const takenElsewhere = Object.entries(roles.roleColumns)
        .filter(([role]) => role !== field.role)
        .flatMap(([, value]) => [...toNames(value)]);
      const matches = matchGPUAnalysisFieldRole(
        field,
        columns.filter(
          column =>
            !sharedCandidates.has(field.parameter) ||
            sharedCandidates.get(field.parameter)!.includes(column.name)
        ),
        {
          excludedColumnNames: takenElsewhere
        }
      );
      const firstName = names[0];
      const chosenMatch = matches.find(match => match.name === firstName);
      const typeOfRole = (field.accepts as readonly string[]).includes(field.role)
        ? field.role
        : undefined;
      // A level the profile only guessed (low confidence) is a fact to ask about, whichever way
      // the role matcher words it.
      if (confidence === 'low' && field.measurement && firstColumn) {
        guessedFacts.push({
          kind: 'measurement',
          datasetId: dataset.datasetId,
          column: firstName,
          effect: 'unknown'
        });
      }
      if (
        typeOfRole !== undefined &&
        firstColumn !== undefined &&
        firstColumn.type !== typeOfRole
      ) {
        add(
          'mapping',
          `Choose the column for "${field.label}": ${firstName} is of type ${firstColumn.type}, not ${typeOfRole}`,
          `Choose a column that holds a ${typeOfRole}, or confirm that ${firstName} does`
        );
      } else if (chosenMatch?.fit === 'accepted') {
        const ratePair = field.measurement?.includes('rate')
          ? suggestGPUAnalysisRatePairs(columns)[0]
          : undefined;
        warnings.push(
          `Check the guessed column for "${field.label}": ${chosenMatch.warning ?? `${firstName} is not clearly meant for it`}. ` +
            (ratePair
              ? `Make a rate first: divide ${ratePair.numerator} by ${ratePair.denominator}, then choose it`
              : 'Choose the column that holds this quantity, or declare what a column measures')
        );
      } else if (confidence === 'low' && field.measurement && firstColumn) {
        warnings.push(
          `Check the guessed column for "${field.label}": ${firstName} is only a guess (${firstColumn.measurementReason ?? 'from its name and values'}). Declare what the column measures, or choose another column`
        );
      } else if (!field.multiple) {
        // `role.accepts` lists the preferred column types first, so only columns of the same
        // preference as the chosen one are rivals for it.
        const getTypeRank = (name: string): number => {
          const type = columns.find(column => column.name === name)?.type;
          const direct = field.accepts.findIndex(accepted => accepted === type);
          return direct >= 0 ? direct : field.accepts.findIndex(accepted => accepted === 'number');
        };
        const preferred = matches.filter(
          match => match.fit === 'preferred' && getTypeRank(match.name) === getTypeRank(firstName)
        );
        if (
          preferred.length >= 2 &&
          firstColumn !== undefined &&
          !hasNameAffinity(field, firstColumn) &&
          firstColumn.measurementSource !== 'declared'
        ) {
          add(
            'mapping',
            `Several columns could be "${field.label}": ${preferred
              .slice(0, 3)
              .map(match => match.name)
              .join(', ')}`,
            'Choose the column that holds this quantity'
          );
        }
      }
    }
    for (const warning of roles.warnings) {
      measurementWarnings.push(warning);
      warnings.push(warning);
    }
    for (const problem of roles.mappingReasons) {
      add('mapping', problem.what, problem.remedy, problem.emptyRole ? {emptyRole: true} : {});
      // A poor fill is a measurement problem like any other: it costs the same in the ranking.
      measurementWarnings.push(problem.what);
      warnings.push(problem.what);
    }
    for (const problem of roles.hardReasons) {
      add('hard', problem.what, problem.remedy);
    }
    // A circular pair the person chose is theirs to keep, with the warning.
    const responseNames = toNames(roles.roleColumns[KEPLER_RESPONSE_ROLE]);
    const predictorNames = toNames(roles.roleColumns[KEPLER_PREDICTOR_ROLE]);
    if (
      userRoleIds.has(KEPLER_RESPONSE_ROLE) &&
      userRoleIds.has(KEPLER_PREDICTOR_ROLE) &&
      responseNames.length > 0
    ) {
      for (const pair of suggestGPUAnalysisRatePairs(columns)) {
        if (
          (responseNames.includes(pair.numerator) && predictorNames.includes(pair.denominator)) ||
          (responseNames.includes(pair.denominator) && predictorNames.includes(pair.numerator))
        ) {
          const text = `${pair.numerator} and ${pair.denominator} are the two parts of a rate, so explaining one by the other is circular (kept because you chose them)`;
          acknowledgedWarnings.push(text);
          warnings.push(text);
        }
      }
    }

    // Requirements: the descriptor's, plus the binding's own capacities where the descriptor is
    // silent or states a looser limit. A requirement about a column role that has no column cannot
    // be evaluated: the role is optional and unused, or it is missing and already listed.
    const selectedTableRoute =
      input.dataKind === 'table' ||
      (input.acceptsTable &&
        dataset.kind === 'attributes' &&
        getKeplerDatasetGeometries(dataset).length === 0);
    const own: (GPUAnalysisRequirementSpec &
      Pick<KeplerOperationRequirement, 'statisticsScope'>)[] = descriptor.requirements
      .filter(
        requirement =>
          requirement.input === inputName &&
          !(selectedTableRoute && requirement.id === 'maximum-vertices') &&
          evaluateGPUAnalysisParameterCondition(requirement.activeWhen, requirementParameters) &&
          (requirement.role === undefined ||
            toNames(roles.roleColumns[requirement.role]).length > 0)
      )
      .map(requirement => {
        const policy = requirement.limitFromParameter;
        if (!policy) return requirement;
        const value = requirementParameters[policy.parameter];
        if (typeof value !== 'number' || !Number.isFinite(value))
          throw new Error(
            `evaluateKeplerOperationApplicability: limitFromParameter.${policy.parameter} must be a finite numeric parameter`
          );
        return {
          ...requirement,
          limit: (policy.absolute ? Math.abs(value) : value) + (policy.offset ?? 0)
        };
      });
    // Surface profile rows count decoded cells, independently of a workflow's empty metadata
    // table. Non-feature ports have no geometry vertex capacity.
    const capacities: {id: 'maximum-rows' | 'maximum-vertices'; limit: number; remedy: string}[] =
      input.dataKind === 'raster'
        ? [
            {
              id: 'maximum-rows',
              limit: input.maximumCells ?? 0,
              remedy:
                'Select a raster tile within maximumCells or use a binding with a larger cell capacity'
            }
          ]
        : [
            {
              id: 'maximum-rows',
              limit: input.maximumFeatures,
              remedy: 'Filter the layer to the area of interest or aggregate it first'
            },
            ...(selectedTableRoute
              ? []
              : [
                  {
                    id: 'maximum-vertices' as const,
                    limit: input.maximumVertices,
                    remedy: 'Simplify the geometry or split the layer into tiles'
                  }
                ])
          ];
    for (const capacity of capacities) {
      const stated = own.filter(requirement => requirement.id === capacity.id);
      if (!stated.some(requirement => (requirement.limit ?? Infinity) <= capacity.limit)) {
        own.push({
          id: capacity.id,
          severity: 'block',
          limit: capacity.limit,
          remedy: capacity.remedy
        });
      }
    }
    const results = evaluateGPUAnalysisRequirements(
      own,
      {
        ...dataset,
        network: getKeplerDatasetNetworkProfile(
          dataset,
          requirementParameters,
          getKeplerEffectiveEngine(binding, executionEngine),
          binding.id
        )
      },
      roles.roleColumns,
      requirementParameters
    );
    for (const requirementResult of results) {
      const retainedGeometryUnknown =
        requirementResult.state === 'failed' &&
        !dataset.pointColumns &&
        dataset.geometryFeatureCounts?.withoutGeometry !== 0 &&
        own.some(
          requirement =>
            requirement.id === requirementResult.id &&
            requirement.role === requirementResult.role &&
            requirement.statisticsScope === 'retained-geometry'
        );
      const result = retainedGeometryUnknown
        ? {
            ...requirementResult,
            state: 'unknown' as const,
            evidence: 'not-checkable' as const,
            message: `${requirementResult.message}; aggregate statistics cannot identify missing values on retained geometry rows`
          }
        : input.dataKind === 'raster' && requirementResult.id === 'maximum-rows'
          ? {
              ...requirementResult,
              message: requirementResult.message.replace(/\brow(s?)\b/g, 'cell$1')
            }
          : requirementResult;
      const adjusted = softenGuessedFailure(
        options.unknownRowCount ? overrideRowCountResult(result) : result
      );
      requirements.push({...adjusted, input: inputName});
      const roleIsUnfilled = adjusted.role !== undefined && unfilledRoleIds.has(adjusted.role);
      // For points every row is one vertex, so a vertex limit repeats the row limit.
      const repeatsRowLimit =
        adjusted.id === 'maximum-vertices' &&
        dataset.kind === 'points' &&
        (dataset.vertexCount ?? dataset.rowCount) === dataset.rowCount &&
        results.some(other => other.id === 'maximum-rows' && other.state === 'failed');
      if (adjusted.id === 'row-semantics') {
        const stated = dataset.rowSemantics;
        const expected = own.find(requirement => requirement.id === 'row-semantics')?.expects;
        const contradictsGuess =
          stated?.source === 'guess' && expected && !expected.includes(stated.value);
        if (adjusted.state === 'failed' || contradictsGuess) {
          if (stated?.source === 'declared') {
            declaredContradiction = true;
            add(
              'confirmation',
              `${adjusted.message}. You declared what these rows are, and this analysis is made for something else`,
              'Choose an analysis made for what the rows are, or change what the rows are declared to stand for'
            );
          } else {
            warnings.push(`${adjusted.message}. Confirm what the rows stand for`);
          }
        }
        if (stated?.source !== 'declared') {
          guessedFacts.push({
            kind: 'row-semantics',
            datasetId: dataset.datasetId,
            effect:
              adjusted.state === 'passed'
                ? 'fits'
                : adjusted.state === 'failed'
                  ? 'contradicts'
                  : 'unknown'
          });
        }
      }
      // Meaning guessed from metadata is guidance, not evidence of a failed precondition.
      // Keep the unknown result and its evidence visible, and offer a declaration in the host.
      // An explicit coordinate-declaration requirement still asks for confirmation before running.
      if (
        adjusted.state === 'unknown' &&
        (adjusted.id === 'row-semantics' || adjusted.evidence === 'guess')
      ) {
        warnings.push(`${adjusted.message}${adjusted.remedy ? `. ${adjusted.remedy}` : ''}`);
        if (adjusted.id !== 'declared-coordinates') continue;
      }
      if (adjusted.severity === 'block' && !roleIsUnfilled) {
        if (repeatsRowLimit) {
          continue;
        }
        if (adjusted.state === 'failed') {
          add(
            'hard',
            adjusted.message,
            adjusted.remedy ?? 'Change the data or the settings so that this holds'
          );
        } else if (adjusted.state === 'unknown') {
          add('confirmation', adjusted.message, adjusted.remedy ?? 'Confirm this before running');
        }
      } else if (adjusted.severity === 'warn' && adjusted.state !== 'passed' && !roleIsUnfilled) {
        warnings.push(
          adjusted.remedy === undefined
            ? adjusted.message
            : `${adjusted.message}. ${adjusted.remedy}`
        );
      }
    }
  }

  // Spatial inputs share a frame even when the operation exposes no coordinateSystem setting.
  const sharesSpatialFrame =
    assigned.length > 1 && !descriptor.inputs.every(input => input.kind === 'attributes');
  // Layers that fill several inputs of one analysis.
  for (let first = 0; first < assigned.length; first++) {
    for (let second = first + 1; second < assigned.length; second++) {
      const left = assigned[first].dataset;
      const right = assigned[second].dataset;
      const leftLabel = binding.inputs.find(
        input => input.name === assigned[first].inputName
      )?.label;
      const rightLabel = binding.inputs.find(
        input => input.name === assigned[second].inputName
      )?.label;
      if (isSameLayer(left, right)) {
        if (descriptor.questions.includes('what-changed')) {
          add(
            'hard',
            `"${left.label}" and "${right.label}" hold the same rows, so they are not two states to compare`,
            'Load the other period as its own layer'
          );
        } else {
          warnings.push(
            `"${leftLabel}" and "${rightLabel}" are filled by layers with the same rows`
          );
        }
      }
      if (
        sharesSpatialFrame &&
        left.coordinates?.source === 'declared' &&
        right.coordinates?.source === 'declared' &&
        left.coordinates.model !== right.coordinates.model
      ) {
        add(
          'hard',
          `"${left.label}" and "${right.label}" declare different coordinate models for inputs that share positions`,
          'Transform the datasets into one coordinate system before running'
        );
      }
      const needsOverlap = descriptor.overlappingInputs?.some(
        ([firstInput, secondInput]) =>
          (firstInput === assigned[first].inputName &&
            secondInput === assigned[second].inputName) ||
          (secondInput === assigned[first].inputName && firstInput === assigned[second].inputName)
      );
      if (needsOverlap && extentsOverlap(left, right) === false) {
        add(
          'hard',
          `"${left.label}" and "${right.label}" do not overlap in space, so nothing can relate them`,
          'Use layers of the same area, or check the coordinates of one of them'
        );
      }
    }
  }

  // Parameters.
  const profilesByInput = Object.fromEntries(
    assigned.map(entry => [entry.inputName, entry.dataset])
  );
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
    overrides: parameterOverrides,
    maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount,
    executionEngine,
    fieldBindings: Object.fromEntries(
      Object.entries(fieldBindings).filter(
        ([parameter]) => binding.parameters.find(spec => spec.name === parameter)?.kind === 'string'
      )
    )
  });
  // Validate scalar and list dependencies against the same public packers and dimensions as
  // the adapters. These checks inspect selected parameters, never individual observation rows.
  if (binding.id.startsWith('validation-vecchia-')) {
    const covariance = {
      model: resolved.parameters['model'] as 'exponential' | 'matern-3/2',
      sill: resolved.parameters['sill'] as number,
      nugget: resolved.parameters['nugget'] as number,
      range: resolved.parameters['range'] as number
    };
    try {
      if (!isValidationVecchiaCovarianceUsable(covariance)) {
        throw new Error(
          'covariance requires a finite positive range and total sill at Float32 precision'
        );
      }
      if (binding.id === 'validation-vecchia-space-time') {
        getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
          makeValidationSpaceTimeCovariance(covariance, resolved.parameters['anisotropy'] as number)
        );
      } else {
        getGPUVecchiaLogLikelihoodParameterValues(covariance);
      }
    } catch (error) {
      add(
        'hard',
        `${binding.id}: ${error instanceof Error ? error.message : String(error)}`,
        'Choose an admissible covariance; raise nugget to the model-specific floor for the selected sill, or use the exponential model'
      );
    }
  }
  if (binding.id === 'regression-mgtwr') {
    const predictorCount = String(resolved.parameters['predictorProperties']).split(',').length;
    for (const name of ['componentBandwidths', 'componentTimeScales']) {
      const components = resolved.parameters[name] as readonly number[];
      if (components.length !== predictorCount + 1) {
        add(
          'hard',
          `${binding.id}: ${name} has ${components.length} entries; ${predictorCount + 1} coefficients require one entry each, intercept first`,
          `Supply ${predictorCount + 1} values for ${name}, or clear its override to generate defaults from the predictor list`
        );
      }
    }
  }
  if (binding.id === 'validation-ensemble-scores' && resolved.parameters['fair'] === true) {
    const memberCount = String(resolved.parameters['memberProperties'])
      .split(',')
      .filter(name => name.trim()).length;
    if (memberCount < 2) {
      add(
        'hard',
        `${binding.id}: fair scoring requires at least 2 members; ${memberCount} selected`,
        'Select at least 2 ensemble member columns or turn fair scoring off'
      );
    }
  }
  for (const confirmation of resolved.confirmations) {
    const requiresChoice = resolved.resolutions.some(
      resolution => resolution.requiresChoice && resolution.reason === confirmation
    );
    add(
      'confirmation',
      confirmation,
      requiresChoice
        ? 'Choose the timestamp encoding explicitly'
        : 'Adjust the resolution or confirm the stated approximation',
      {parameterConfirmation: requiresChoice}
    );
  }
  for (const parameter of descriptor.parameters) {
    if (
      parameter.confirmation &&
      evaluateGPUAnalysisParameterCondition(parameter.activeWhen, resolved.parameters) &&
      resolved.parameters[parameter.name] !== true
    ) {
      add('confirmation', parameter.confirmation.message, parameter.confirmation.remedy, {
        parameterConfirmation: true
      });
    }
  }
  let lengthDefaultFromData = false;
  for (const resolution of resolved.resolutions) {
    const specification = binding.parameters.find(spec => spec.name === resolution.parameter);
    const label = specification?.label ?? resolution.parameter;
    const policy = descriptor.parameters.find(
      entry => entry.name === resolution.parameter
    )?.defaultPolicy;
    if (resolution.source === 'fallback' && resolution.active !== false) {
      if (policy?.from === 'data' && policy.rule === 'column-sum') {
        add(
          'mapping',
          `${label} could not be derived safely (${resolution.reason ?? 'the column total is unavailable'}); choose allocation settings before running`,
          `Profile all non-negative values or choose ${label} explicitly to fit the allocation limit`
        );
      } else if (policy?.from === 'data' && LENGTH_UNITS.includes(specification?.unit ?? '')) {
        // A length that could not be read from the data is not a safe constant: its unit is the
        // data's own, and the binding's number may be larger than the whole layer.
        add(
          'mapping',
          `${label} could not be derived from this data (${resolution.reason ?? 'the profile has no statistic for it'}); the default of ${String(resolved.parameters[resolution.parameter])} is not tied to this layer's scale`,
          `Set ${label} yourself, in the layer's own units`
        );
      } else {
        warnings.push(
          `${label} could not be derived from the data; the default of ${String(
            resolved.parameters[resolution.parameter]
          )} stays`
        );
      }
    }
    if (resolution.resolution?.sampled && resolution.source === 'data') {
      warnings.push(`${label} comes from statistics of a sample of the points`);
    }
    if (
      resolution.source === 'data' &&
      resolution.active !== false &&
      policy?.from === 'data' &&
      (resolution.resolution?.unit === 'metre' || resolution.resolution?.unit === 'coordinate-unit')
    ) {
      lengthDefaultFromData = true;
    }
  }
  // Profiles contain spacing summaries, not the neighbour graph: warn about chaining without
  // claiming to know the resulting partition before execution.
  const clusteringRadius = (
    {
      'density-clustering': 'epsilon',
      'curated-clusters-and-footprints': 'epsilon',
      'optics-ordering': 'extractionDistance',
      'hierarchical-density-clustering': 'cutDistance'
    } as Readonly<Record<string, string>>
  )[binding.id];
  if (clusteringRadius) {
    const radius = resolved.parameters[clusteringRadius];
    const radiusResolution = resolved.resolutions.find(
      entry => entry.parameter === clusteringRadius
    );
    const spacing = profilesByInput['points']?.spacing;
    if (
      radiusResolution?.source === 'data' &&
      radiusResolution.active !== false &&
      radiusResolution.coordinateModel === spacing?.coordinateModel &&
      typeof radius === 'number' &&
      spacing?.nearestNeighbor &&
      radius >= spacing.nearestNeighbor.maximum
    ) {
      warnings.push(
        `At ${clusteringRadius}=${radius}, chains of neighbours can join every point into one cluster; reduce ${clusteringRadius} to look for smaller groups and review minimumSamples and the resulting noise`
      );
    }
  }
  if (
    binding.id === 'cell-compare' &&
    resolved.parameters['statistic'] === 'count' &&
    ['before', 'after'].some(inputName => {
      const semantics = profilesByInput[inputName]?.rowSemantics?.value;
      return semantics === 'objects' || semantics === 'surface-samples';
    })
  ) {
    warnings.push(
      'Count compares numbers of point locations and ignores their measurements: identical station locations in both periods produce zero change in every cell, even when their measured values differ; choose statistic=sum (or mean) and map valueProperty to the measurement column in both periods'
    );
  }
  // Reach rules count isolated points; say so when there are some.
  for (const entry of descriptor.parameters) {
    const policy = entry.defaultPolicy;
    const resolution = resolved.resolutions.find(candidate => candidate.parameter === entry.name);
    if (
      policy.from !== 'data' ||
      resolution?.active === false ||
      resolution?.source !== 'data' ||
      (policy.rule !== 'maximum-nearest-neighbor-distance' &&
        policy.rule !== 'outlier-resistant-nearest-neighbor-distance')
    ) {
      continue;
    }
    const spacing = profilesByInput[policy.input]?.spacing;
    const isolated = spacing?.nearestNeighbor?.aboveUpperFenceCount ?? 0;
    if (spacing && isolated > 0) {
      const label = binding.parameters.find(spec => spec.name === entry.name)?.label ?? entry.name;
      warnings.push(
        `${isolated.toLocaleString('en-US')} of ${spacing.pointCount.toLocaleString('en-US')} points are isolated: each is farther from its nearest neighbour than the upper fence of the spacing (${spacing.nearestNeighbor?.upperFence.toLocaleString('en-US', {maximumSignificantDigits: 3})} ${spacing.unit === 'metre' ? 'm' : 'units'}); ${label} is read from the data with that in mind`
      );
    }
  }

  // Coordinates: a low-confidence guess is a question before any metric work, and a declaration
  // that the extent contradicts is wrong.
  const hasCoordinateRequirement = descriptor.requirements.some(requirement =>
    COORDINATE_REQUIREMENTS.includes(requirement.id)
  );
  const coordinateRequirementWarned = requirements.some(
    result =>
      COORDINATE_REQUIREMENTS.includes(result.id) &&
      result.state === 'unknown' &&
      result.evidence === 'guess'
  );
  for (const {dataset} of assigned) {
    const coordinates = dataset.coordinates;
    if (coordinates?.source === 'guess' && !coordinateRequirementWarned) {
      warnings.push(
        `The coordinates of "${dataset.label}" are a guess (${coordinates.model}); confirm them`
      );
    }
    if (coordinates?.source === 'guess') {
      const metric = hasCoordinateRequirement || lengthDefaultFromData || sharesSpatialFrame;
      if (metric) {
        guessedFacts.push({
          kind: 'coordinate-model',
          datasetId: dataset.datasetId,
          effect: 'unknown'
        });
      }
      if (sharesSpatialFrame || (coordinates.confidence === 'low' && metric)) {
        warnings.push(
          `The coordinate system of "${dataset.label}" is a ${coordinates.confidence === 'low' ? 'low-confidence ' : ''}guess (${coordinates.model}${coordinates.reason ? `: ${coordinates.reason}` : ''}), and this analysis relates positions in it`
        );
      }
    }
    const extent = dataset.extent;
    if (
      coordinates?.source === 'declared' &&
      coordinates.model === 'geographic' &&
      extent &&
      (Math.abs(extent[1]) > 90 || Math.abs(extent[3]) > 90 || extent[0] < -180 || extent[2] > 360)
    ) {
      add(
        'hard',
        `"${dataset.label}" is declared as longitude and latitude, but its extent (${extent.map(value => Number(value.toPrecision(5))).join(', ')}) cannot be degrees`,
        'Declare the coordinate system the values are really in (planar, for projected coordinates)'
      );
    }
    // A coordinate system chosen for this run that contradicts what the layer was declared as.
    // The choice stands (the defaults are derived in the chosen model), and the reader is told
    // what it means for the distances.
    const chosenCoordinateSystem = parameterOverrides['coordinateSystem'];
    const chosenModel =
      chosenCoordinateSystem === 'planar'
        ? 'planar'
        : chosenCoordinateSystem === 'EPSG:4326'
          ? 'geographic'
          : undefined;
    if (
      chosenModel !== undefined &&
      coordinates?.source === 'declared' &&
      coordinates.model !== chosenModel
    ) {
      warnings.push(
        chosenModel === 'planar'
          ? `"${dataset.label}" is declared as longitude and latitude, but this run is set to planar coordinates: distances and sizes are measured in degrees, which are not equal lengths east-west and north-south. Set the run to match the layer, or change what the layer is declared as`
          : `"${dataset.label}" is declared as planar coordinates, but this run is set to longitude and latitude: its values are read as degrees. Set the run to match the layer, or change what the layer is declared as`
      );
    }
  }

  // The CPU engine's work caps for pairwise work (grid work raised the cell size above).
  if (
    binding.id === 'density-clustering' &&
    getKeplerEffectiveEngine(binding, executionEngine) === 'cpu' &&
    assigned[0] &&
    !isDbscanCpuPointCountAdmitted(assigned[0].dataset.rowCount)
  ) {
    add(
      'hard',
      `CPU DBSCAN accepts at most ${MAXIMUM_DBSCAN_CPU_POINTS} total points; received ${assigned[0].dataset.rowCount}`,
      'Filter or partition the layer'
    );
  }
  const cpuCap = KEPLER_CPU_WORK_CAPS[binding.id];
  if (
    cpuCap?.kind === 'rows-squared' &&
    getKeplerEffectiveEngine(binding, executionEngine) === 'cpu'
  ) {
    const except = cpuCap.exceptWhen;
    const excepted =
      except !== undefined && evaluateGPUAnalysisParameterCondition(except, resolved.parameters);
    const firstInput = assigned[0];
    if (!excepted && firstInput && firstInput.dataset.rowCount ** 2 > cpuCap.limit) {
      const limit = Math.floor(Math.sqrt(cpuCap.limit));
      add(
        'hard',
        `${firstInput.dataset.rowCount.toLocaleString('en-US')} rows; at most ${limit.toLocaleString('en-US')} on the CPU engine (the pairwise work is rows squared, within ${cpuCap.limit.toExponential(0)})`,
        'Filter or sample the layer, or run the analysis where a GPU device is available'
      );
    }
  }

  // Reasons.
  const reasons: string[] = [...unusedColumns];
  const prefixed = assigned.length > 1;
  for (const {inputName, dataset} of assigned) {
    const phrases = [
      ...new Set(
        filledRoles
          .filter(entry => entry.input === inputName)
          .map(entry => ROLE_PHRASES[entry.role] ?? `a ${entry.role.replace(/-/g, ' ')} column`)
      )
    ];
    const withClause =
      phrases.length === 0
        ? ''
        : ` with ${phrases.length === 1 ? phrases[0] : `${phrases.slice(0, -1).join(', ')} and ${phrases[phrases.length - 1]}`}`;
    const prefix = prefixed ? `${dataset.label}: ` : '';
    reasons.push(`${prefix}${describeKindPhrase(dataset.kind)}${withClause}`);
    reasons.push(`${prefix}${formatCountText(dataset.rowCount)} ${getRowNoun(dataset.rowCount)}`);
  }
  if (assigned.length > 1) {
    reasons.push(`uses ${assigned.length} of the loaded datasets`);
  }

  // A judgement from the shape of a dataset alone does not know measurement levels, coordinates
  // or row counts: what rests on them is a caveat, not a lower status.
  const caveats: string[] = [];
  const effectiveEntries = options.subjectOnly
    ? entries.filter(entry => {
        if (entry.kind === 'hard' || entry.parameterConfirmation) {
          return true;
        }
        caveats.push(entry.what);
        return false;
      })
    : entries;
  const status: KeplerApplicabilityStatus = effectiveEntries.some(entry => entry.kind === 'hard')
    ? 'blocked'
    : effectiveEntries.some(entry => entry.kind === 'mapping')
      ? 'needs-mapping'
      : effectiveEntries.some(entry => entry.kind === 'confirmation')
        ? 'needs-confirmation'
        : 'ready';

  return {
    bindingId: binding.id,
    status,
    inputAssignments,
    inputAssignmentAlternatives: assignment.count > 1 ? inputAssignmentAlternatives : [],
    outputKinds: Object.fromEntries(
      descriptor.outputs.map(output => [
        output.tableId,
        (output.kindFrom
          ? assigned.find(input => input.inputName === output.kindFrom)?.dataset.kind
          : undefined) ?? output.kind
      ])
    ),
    fieldBindings,
    requirements,
    missing: effectiveEntries.map(({what, remedy}) => ({what, remedy})),
    reasons,
    warnings: [...new Set(warnings)],
    measurementWarnings: [...new Set(measurementWarnings)],
    acknowledgedWarnings: [...new Set(acknowledgedWarnings)],
    notes: descriptor.notes ?? [],
    filledRoles,
    roleCandidates,
    guessedFacts,
    declaredContradiction,
    hasEmptyRole: effectiveEntries.some(entry => entry.emptyRole === true),
    caveats,
    inactive: resolved.inactive,
    confirmations: resolved.confirmations,
    parameters: resolved.parameters,
    resolutions: resolved.resolutions,
    basic: resolved.basic,
    advanced: resolved.advanced
  };
}

// ---------------------------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------------------------

/**
 * A curated analysis is a question-named, pre-configured way to run one operation. It ranks above
 * a bare operation with the same fit because people ask questions, not operation names
 * (design README 4.5: "templates above single operations"; the Voyager studies, Wongsuphasawat et
 * al. 2015 and 2017: users bookmarked views that the recommender added, 69 percent of the
 * bookmarks, when they were shown as ready answers).
 */
export const KEPLER_CURATED_BONUS = 30;

/**
 * Points per step of descriptor prominence: prominence 1 adds twice this, 2 adds it once, 3 adds
 * nothing. Prominence is the authors' judgement of which operations lead for their data. Show Me
 * ranks by hand-written specialisation (Mackinlay et al. 2007: "higher ranked commands typically
 * have more specialized conditions"); this is the same hand-written order.
 */
export const KEPLER_PROMINENCE_WEIGHT = 10;

/**
 * Bonus for each filled optional column, at most {@link KEPLER_OPTIONAL_ROLE_LIMIT} of them. An
 * analysis that uses a category or a weight the data has says more than one that ignores it
 * (README 4.5 "uses more of what the dataset has"; Voyager: 69 percent of bookmarks included an
 * automatically added variable).
 */
export const KEPLER_OPTIONAL_ROLE_BONUS = 4;
export const KEPLER_OPTIONAL_ROLE_LIMIT = 2;

/**
 * Bonus for a filled column in the role `time`. Time is the one column that turns a static
 * pattern question into a change question; Tableau's Show Me ranks date plus measure above plain
 * charts for the same reason.
 */
export const KEPLER_TIME_COLUMN_BONUS = 12;

/**
 * Bonus for each loaded dataset beyond the first that an analysis uses. Pair rules (points and
 * areas, facilities and a network) fire when a second dataset is loaded (README 4.5, "Two
 * datasets"); an analysis that ignores half the data should not tie with one that joins it. It
 * was 10 while an optional value column was filled unasked (worth another 4 to the analyses that
 * used one); that fill is gone, so the margin of a pair rule is kept at 15.
 */
export const KEPLER_EXTRA_DATASET_BONUS = 15;

/**
 * Bonus when every column role is filled with a column of the kind the method is meant for and
 * there is no measurement warning. Draco treats a mismatch of measurement level as a violated
 * soft constraint (Moritz et al. 2018; Draco 2, Yang et al. 2023: aggregating an ordinal field
 * with a mean is penalised).
 */
export const KEPLER_PREFERRED_MEASUREMENT_BONUS = 6;

/**
 * Penalty for each column that is not what the method is meant for (counts given to a rate
 * method, an identifier given to a numeric role). Large, so that a hot spot of raw counts never
 * leads when a rate is expected: counts follow population, and a hot spot map of counts maps
 * where people live (Esri and GeoDa guidance in the templates note).
 */
export const KEPLER_MEASUREMENT_WARNING_PENALTY = -20;

/**
 * Penalty for each warning-level requirement that failed (fewer than 30 rows for a significance
 * analysis, rows of the wrong kind). The analysis still runs; it should not lead. Esri documents
 * local statistics as unreliable below 30 features.
 */
export const KEPLER_FAILED_WARNING_PENALTY = -10;

/**
 * Penalty for an analysis whose columns a person has to choose. Predictable suggestions matter:
 * skilled Show Me users found it "hard to predict what view will be built" (Mackinlay et al.
 * 2007), and a suggestion that needs a decision is less predictable than one that runs.
 */
export const KEPLER_NEEDS_MAPPING_PENALTY = -10;

/**
 * Penalty for each fact the profile could not settle (a requirement that is unknown, a default
 * that fell back to a constant). Fewer unknowns means fewer surprises, and Show Me leaves an
 * option unranked when deciding it would need an expensive scan.
 */
export const KEPLER_UNKNOWN_PENALTY = -3;

/**
 * Penalty for operations whose only question is `prepare-data`. People come to ask where and
 * how much, not to repair geometry; the Voyager studies found suggestions work when they extend
 * the question being explored. Preparation steps are listed in "Browse all" and offered as next
 * steps of a result.
 */
export const KEPLER_PLUMBING_PENALTY = -15;

/**
 * Penalty when the dataset's declared row semantics contradict what the method is meant for
 * (readings of a surface given to an event-pattern method, or the reverse). Large enough to cancel
 * a curated analysis of prominence 1 (30 + 20), because the result is not a weaker answer but an
 * answer to another question: a density or hot spot map of rain gauge readings maps where the gauges stand
 * (Esri's hot spot guidance: the incident points must be events, not samples of a field). A
 * guessed semantics carries the ordinary penalty.
 */
export const KEPLER_ROW_SEMANTICS_MISMATCH_PENALTY = -50;

/**
 * Bonus when the dataset's declared row semantics are the ones the method is meant for (a surface
 * method on readings, an event-pattern method on incidents). The method that was designed for what
 * the rows are should lead the one that merely accepts them (Esri's hot spot guidance separates
 * incident data from weighted features for this reason).
 */
export const KEPLER_SEMANTICS_MATCH_BONUS = 8;

/**
 * Bonus for each geometry fewer than three that the most specific filled input accepts. An
 * operation that reads only points, or only streets, says more about its data than one that reads
 * any feature; Show Me ranks the more specialised commands first (Mackinlay et al. 2007: "higher
 * ranked commands typically have more specialized conditions").
 */
export const KEPLER_SPECIFIC_INPUT_BONUS = 4;

/**
 * Penalty for a comparison of two states ('what changed') whose two datasets share fewer than
 * half of their column names. Before and after are two observations of the same kind of record;
 * a facilities layer and a demand layer fill the same two slots by type but are not two periods.
 */
export const KEPLER_UNLIKE_STATES_PENALTY = -20;

/** Share of column names two datasets need in common to read as two states of one thing. */
export const KEPLER_LIKE_STATES_OVERLAP = 0.5;

/**
 * Bonus for a comparison of two states whose layers do share at least half of their column names
 * and are not the same layer: the strongest evidence a change analysis can have that the two
 * layers are two observations of one thing. The mirror of the penalty above.
 * Together with the extra-dataset bonus, this keeps a matching two-state comparison ahead of
 * separate time-aware analyses of either state, which cannot answer the between-state question.
 */
export const KEPLER_LIKE_STATES_BONUS = 15;

/**
 * Extra penalty when a layer has fewer than half the rows an analysis needs for its results to
 * mean anything (a significance or clustering analysis on 12 points when it wants 30). The
 * ordinary failed-warning penalty (-10) says "read with care"; this says "do not lead with it".
 */
export const KEPLER_FAR_BELOW_MINIMUM_PENALTY = -20;

/** Default number of suggestions; the research on suggestion lists finds 5 to 8 usable. */
export const KEPLER_DEFAULT_SUGGESTION_COUNT = 6;

/** Number of greyed-out "one step away" analyses shown beside the suggestions. */
export const KEPLER_BLOCKED_SUGGESTION_COUNT = 4;

/** Share of the requested budget one primary question may take, rounded down. */
export const KEPLER_QUESTION_SLOT_SHARE = 1 / 3;

/** Specialist candidates live under "Browse all" unless validated native evidence promotes them. */
export const KEPLER_HIDDEN_PROMINENCE = 3;

/**
 * Weight of row semantics that are only guessed (low confidence), when they fit what an analysis
 * is made for. A fact the profile guessed should lean the order, never settle it, so it counts for
 * half of a declaration (+8).
 */
export const KEPLER_GUESSED_SEMANTICS_MATCH_BONUS = 4;

/**
 * Penalty when guessed row semantics contradict what an analysis is made for. Weaker than a
 * declared contradiction (-50), but stronger than mapping an uncertain measurement (-30).
 * Counting sensor locations must not lead interpolation solely because a reading needs confirmation.
 * The guess may be wrong, and the analysis is offered with a
 * question (status `needs-confirmation`) instead of being removed.
 */
export const KEPLER_GUESSED_SEMANTICS_MISMATCH_PENALTY = -35;

/**
 * Most candidates one single-input analysis is offered for: one per compatible loaded dataset, the
 * first this many by dataset id. The rest are counted in `dropped`.
 */
export const KEPLER_MAXIMUM_CANDIDATES_PER_BINDING = 6;

/**
 * Former candidate-window size, retained for callers importing this constant. Question discovery
 * now tests alternative answers against the top three, including facts used outside this window.
 */
export const KEPLER_QUESTION_WINDOW = 10;

/**
 * Order of the curated analyses as a first step, used after the score and the descriptor's
 * prominence tie: the README 4.6 table order (where things are, whether that is more than chance,
 * how groups form, then change, surfaces, areas, reach). It is a documented judgement of how
 * common each analysis is as a first question, so a tie never falls to an id.
 */
export const KEPLER_CURATED_ANALYSIS_ORDER: Readonly<Record<string, number>> = Object.fromEntries(
  KEPLER_CURATED_ANALYSES.map((analysis, index) => [analysis.id, index])
);

/** Order of bare operations after every curated analysis: their position in the registry. */
const BARE_OPERATION_ORDER_OFFSET = 1000;
const BARE_OPERATION_ORDER: ReadonlyMap<string, number> = new Map(
  KEPLER_OPERATION_DESCRIPTORS.map((descriptor, index) => [descriptor.bindingId, index])
);

/** A question a host can ask once to settle the order of several analyses. */
export type KeplerSuggestionQuestion = {
  id: 'row-semantics' | 'coordinate-model' | 'measurement';
  datasetId: string;
  /** The column, for a measurement question. */
  column?: string;
  prompt: string;
  options: readonly string[];
  /** Suggestion ids whose rank or status depends on the answer. */
  affects: readonly string[];
};

export type KeplerAnalysisSuggestion = {
  /** `curated:<id>` or `binding:<id>`, followed by `@<dataset id>` for one of several layers. */
  id: string;
  bindingId: string;
  curatedAnalysisId?: string;
  /** Loaded dataset ids the analysis is filled with, by binding input order. */
  datasetIds: readonly string[];
  /** The curated analysis's question as a person asks it, or the descriptor's action phrase. */
  title: string;
  question: GPUAnalysisQuestion;
  reasons: readonly string[];
  /** The descriptor's notes, passed through. */
  notes: readonly string[];
  status: KeplerApplicabilityStatus;
  applicability: KeplerOperationApplicability;
  /** Sum of `scoreTerms`. */
  score: number;
  /** What each soft rule contributed, by rule name. */
  scoreTerms: Readonly<Record<string, number>>;
};

export type KeplerSuggestionOptions = {
  /** Anchor candidates to this loaded dataset before ranking, diversity and count limits. */
  requiredDatasetId?: string;
  /** Number of suggestions, at least 1. Default {@link KEPLER_DEFAULT_SUGGESTION_COUNT}. */
  maximumCount?: number;
  /** Curated analyses to offer. Default: all of them. */
  curated?: readonly KeplerCuratedAnalysis[];
  /** Where the analyses will run; see {@link KeplerApplicabilityOptions.executionEngine}. */
  executionEngine?: 'gpu' | 'cpu';
};

export type KeplerSuggestionResult = {
  suggestions: readonly KeplerAnalysisSuggestion[];
  /**
   * Up to four curated or prominence-1 analyses that lack exactly one thing, for the greyed-out
   * rows: a dataset, a column (the remedy names the kind of column), a capacity.
   */
  blocked: readonly KeplerAnalysisSuggestion[];
  /**
   * Suggestions of a question that did not fit its share of the slots, by question, best first
   * (see {@link suggestKeplerAnalyses}).
   */
  more: Readonly<Partial<Record<GPUAnalysisQuestion, readonly KeplerAnalysisSuggestion[]>>>;
  /** Questions that would settle the order of the best candidates; ask at most one at a time. */
  questions: readonly KeplerSuggestionQuestion[];
  /** What was not tried because of the search limits, in words. */
  dropped: readonly string[];
  /** Counts over every candidate, including those not shown. */
  total: {ready: number; needsInput: number; blocked: number};
};

type Candidate = KeplerAnalysisSuggestion & {
  prominence: 1 | 2 | 3;
  isCurated: boolean;
  /** Loaded datasets the analysis already uses. */
  assignedCount: number;
  /** Position of the analysis in the documented order. */
  order: number;
};

/** Distinct column names the two datasets share, over the distinct names of both. */
function getColumnNameOverlap(left: KeplerDatasetProfile, right: KeplerDatasetProfile): number {
  const leftNames = new Set(left.columns.map(column => column.name));
  const rightNames = new Set(right.columns.map(column => column.name));
  const union = new Set([...leftNames, ...rightNames]);
  if (union.size === 0) {
    return 1;
  }
  let shared = 0;
  for (const name of leftNames) {
    if (rightNames.has(name)) shared++;
  }
  return shared / union.size;
}

/** The question an operation leads with: its first that is not data preparation. */
export function getKeplerPrimaryQuestion(
  descriptor: KeplerOperationDescriptor
): GPUAnalysisQuestion {
  return descriptor.questions.find(question => question !== 'prepare-data') ?? 'prepare-data';
}

function computeScoreTerms(
  binding: CatalogOperationBinding,
  applicability: KeplerOperationApplicability,
  descriptor: KeplerOperationDescriptor,
  isCurated: boolean,
  datasets: readonly KeplerDatasetProfile[]
): Record<string, number> {
  // Only a column that is a declared or high-confidence fit counts as used: a low-confidence
  // guess is not evidence that the data has what the analysis can use.
  const confidentOptionalCount = applicability.filledRoles.filter(
    entry => !entry.required && entry.role !== 'time' && entry.confident === true
  ).length;
  const assignedIds = Object.values(applicability.inputAssignments).filter(
    (id): id is string => id !== undefined
  );
  // The same fact (coordinates only guessed) is unknown for every input; it counts once.
  const unknownFacts = new Set(
    applicability.requirements
      .filter(result => result.state === 'unknown')
      .map(result => `${result.id}/${result.role ?? ''}`)
  );
  const unknownCount =
    unknownFacts.size +
    applicability.resolutions.filter(
      resolution => resolution.source === 'fallback' && resolution.active !== false
    ).length;
  const semanticsResults = applicability.requirements.filter(
    result => result.id === 'row-semantics'
  );
  const scoredSemantics = semanticsResults.map(result => {
    const profile = datasets.find(
      dataset => dataset.datasetId === applicability.inputAssignments[result.input]
    );
    const expects = descriptor.requirements.find(
      requirement => requirement.input === result.input && requirement.id === 'row-semantics'
    )?.expects;
    if (result.state === 'unknown' && profile?.rowSemantics && expects) {
      return {...result, state: expects.includes(profile.rowSemantics.value) ? 'passed' : 'failed'};
    }
    return result;
  });
  const declaredMismatches = scoredSemantics.filter(
    result => result.state === 'failed' && result.evidence === 'declaration'
  ).length;
  const guessedMismatches = scoredSemantics.filter(
    result => result.state === 'failed' && result.evidence !== 'declaration'
  ).length;
  // Conditional count-only guards are negative evidence for surface samples, not extra
  // information used by a row counter when they pass.
  const matchingSemantics = scoredSemantics.filter(
    result =>
      !descriptor.requirements.some(
        requirement =>
          requirement.input === result.input &&
          requirement.id === 'row-semantics' &&
          requirement.activeWhen &&
          'parameter' in requirement.activeWhen &&
          requirement.activeWhen.parameter === 'valueProperty' &&
          requirement.activeWhen.equals.includes('')
      )
  );
  const declaredMatches = matchingSemantics.filter(
    result => result.state === 'passed' && result.evidence === 'declaration'
  ).length;
  const guessedMatches = matchingSemantics.filter(
    result => result.state === 'passed' && result.evidence !== 'declaration'
  ).length;
  const failedWarnings = applicability.requirements.filter(
    result =>
      result.severity === 'warn' && result.state === 'failed' && result.id !== 'row-semantics'
  ).length;
  const mostSpecificInput = Math.max(
    0,
    ...binding.inputs
      .filter(input => applicability.inputAssignments[input.name] !== undefined)
      .map(input => 3 - input.accepts.length)
  );
  // A minimum-rows warning that failed by more than half.
  let farBelowMinimum = 0;
  for (const result of applicability.requirements) {
    if (result.id !== 'minimum-rows' || result.severity !== 'warn' || result.state !== 'failed') {
      continue;
    }
    const limit = descriptor.requirements.find(
      requirement =>
        requirement.input === result.input &&
        requirement.id === 'minimum-rows' &&
        requirement.severity === 'warn'
    )?.limit;
    const rows = datasets.find(
      dataset => dataset.datasetId === applicability.inputAssignments[result.input]
    )?.rowCount;
    if (limit !== undefined && rows !== undefined && rows * 2 < limit) {
      farBelowMinimum = KEPLER_FAR_BELOW_MINIMUM_PENALTY;
    }
  }
  let unlikeStates = 0;
  let likeStates = 0;
  if (descriptor.questions.includes('what-changed') && assignedIds.length === 2) {
    const [left, right] = assignedIds.map(id => datasets.find(dataset => dataset.datasetId === id));
    if (left && right) {
      if (getColumnNameOverlap(left, right) < KEPLER_LIKE_STATES_OVERLAP) {
        unlikeStates = KEPLER_UNLIKE_STATES_PENALTY;
      } else {
        likeStates = KEPLER_LIKE_STATES_BONUS;
      }
    }
  }
  return {
    curated: isCurated ? KEPLER_CURATED_BONUS : 0,
    // Exact native track/time evidence is more specific than point/line geometry alone.
    nativeKind: descriptor.inputs.some(
      input =>
        input.kind === 'trajectories' &&
        datasets.some(
          dataset =>
            dataset.datasetId === applicability.inputAssignments[input.name] && dataset.trajectory
        )
    )
      ? KEPLER_CURATED_BONUS + KEPLER_PROMINENCE_WEIGHT
      : 0,
    prominence: (3 - descriptor.prominence) * KEPLER_PROMINENCE_WEIGHT,
    // A timestamp is useful only when the method fits what the rows represent. Counting
    // sensor locations through time does not become a surface-value analysis.
    timeColumn:
      declaredMismatches + guessedMismatches === 0 &&
      applicability.filledRoles.some(entry => entry.role === 'time')
        ? KEPLER_TIME_COLUMN_BONUS
        : 0,
    optionalColumns:
      Math.min(confidentOptionalCount, KEPLER_OPTIONAL_ROLE_LIMIT) * KEPLER_OPTIONAL_ROLE_BONUS,
    extraDatasets: Math.max(0, assignedIds.length - 1) * KEPLER_EXTRA_DATASET_BONUS,
    specificInput: mostSpecificInput * KEPLER_SPECIFIC_INPUT_BONUS,
    semanticsMatch:
      declaredMatches * KEPLER_SEMANTICS_MATCH_BONUS +
      guessedMatches * KEPLER_GUESSED_SEMANTICS_MATCH_BONUS,
    // No column the method is not meant for. An analysis with no column roles qualifies too: it
    // asks nothing of the data's measurements, which is what the bonus rewards.
    preferredMeasurement:
      applicability.measurementWarnings.length === 0 ? KEPLER_PREFERRED_MEASUREMENT_BONUS : 0,
    measurementWarnings:
      applicability.measurementWarnings.length * KEPLER_MEASUREMENT_WARNING_PENALTY,
    rowSemanticsMismatch:
      declaredMismatches * KEPLER_ROW_SEMANTICS_MISMATCH_PENALTY +
      guessedMismatches * KEPLER_GUESSED_SEMANTICS_MISMATCH_PENALTY,
    failedWarnings: failedWarnings * KEPLER_FAILED_WARNING_PENALTY,
    farBelowMinimum,
    unlikeStates,
    likeStates,
    needsMapping: applicability.status === 'needs-mapping' ? KEPLER_NEEDS_MAPPING_PENALTY : 0,
    unknowns: unknownCount * KEPLER_UNKNOWN_PENALTY,
    plumbing: descriptor.questions.every(question => question === 'prepare-data')
      ? KEPLER_PLUMBING_PENALTY
      : 0
  };
}

function sumTerms(terms: Readonly<Record<string, number>>): number {
  return Object.values(terms).reduce((total, value) => total + value, 0);
}

/**
 * Candidate order: score, then descriptor prominence, then the documented order of analyses
 * ({@link KEPLER_CURATED_ANALYSIS_ORDER}; bare operations after, in registry order), then the
 * dataset ids the analysis is filled with, and only then the suggestion id. Two different
 * analyses never tie on all of these, so an alphabetical id never decides between them.
 */
function compareCandidates(left: Candidate, right: Candidate): number {
  return (
    right.score - left.score ||
    left.prominence - right.prominence ||
    left.order - right.order ||
    compareText(left.datasetIds.join(','), right.datasetIds.join(',')) ||
    compareText(left.id, right.id)
  );
}

function evaluateCandidate(
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  datasets: readonly KeplerDatasetProfile[],
  curated: KeplerCuratedAnalysis | undefined,
  executionEngine: 'gpu' | 'cpu',
  idSuffix: string,
  requiredDatasetId?: string
): Candidate {
  // Preserve D5's full specialist metadata. Exact native track/time validation unlocks only
  // the existing bounded thinning operation; geometry alone never promotes the family.
  const prominence =
    binding.id === 'trajectory-thinning' && datasets.length === 1 && datasets[0].trajectory
      ? 1
      : descriptor.prominence;
  let options: KeplerApplicabilityOptions = {executionEngine, requiredDatasetId};
  if (curated && curated.execution.status === 'available') {
    const execution = curated.execution;
    const inputGeometries: Record<string, CatalogGeometry> = {};
    for (const slot of curated.slots) {
      const inputName = execution.inputs[slot.name];
      const geometry = getGPUAnalysisCatalogGeometryForDataKind(slot.kind, 'input');
      if (
        inputName !== undefined &&
        (geometry === 'point' || geometry === 'line' || geometry === 'polygon')
      ) {
        inputGeometries[inputName] = geometry;
      }
    }
    const inputNames = Object.fromEntries(
      binding.inputs.map(input => [
        input.name,
        curated.slots
          .filter(slot => execution.inputs[slot.name] === input.name)
          .map(slot => slot.name)
      ])
    );
    options = {
      executionEngine,
      requiredDatasetId,
      overrides: execution.parameters,
      inputGeometries,
      inputNames
    };
  }
  const applicability = evaluateKeplerOperationApplicability(
    binding,
    descriptor,
    datasets,
    options
  );
  const scoreTerms = computeScoreTerms(
    binding,
    applicability,
    {...descriptor, prominence},
    curated !== undefined,
    datasets
  );
  const datasetIds = binding.inputs
    .map(input => applicability.inputAssignments[input.name])
    .filter((id): id is string => id !== undefined);
  return {
    id: `${curated ? `curated:${curated.id}` : `binding:${binding.id}`}${idSuffix}`,
    bindingId: binding.id,
    ...(curated ? {curatedAnalysisId: curated.id} : {}),
    datasetIds,
    title: curated ? curated.title : descriptor.action,
    question: curated ? curated.question : getKeplerPrimaryQuestion(descriptor),
    reasons: applicability.reasons,
    notes: applicability.notes,
    status: applicability.status,
    applicability,
    score: sumTerms(scoreTerms),
    scoreTerms,
    prominence,
    isCurated: curated !== undefined,
    assignedCount: datasetIds.length,
    order: curated
      ? (KEPLER_CURATED_ANALYSIS_ORDER[curated.id] ?? BARE_OPERATION_ORDER_OFFSET - 1)
      : BARE_OPERATION_ORDER_OFFSET + (BARE_OPERATION_ORDER.get(binding.id) ?? 0)
  };
}

function toSuggestion(candidate: Candidate): KeplerAnalysisSuggestion {
  const {
    prominence: _prominence,
    isCurated: _isCurated,
    assignedCount: _assignedCount,
    order: _order,
    ...suggestion
  } = candidate;
  return suggestion;
}

/** Strictly cap each question at a third of the requested budget; retain all others in Browse all. */
function selectDiverse(
  sorted: readonly Candidate[],
  maximumCount: number
): {
  selected: KeplerAnalysisSuggestion[];
  more: Partial<Record<GPUAnalysisQuestion, KeplerAnalysisSuggestion[]>>;
} {
  const perQuestionLimit = Math.max(1, Math.floor(maximumCount * KEPLER_QUESTION_SLOT_SHARE));
  const counts = new Map<GPUAnalysisQuestion, number>();
  const selectedIds = new Set<string>();
  const targets = new Set<string>();
  // Offer different analyses before repeating one for another layer. Otherwise two highly
  // ranked single-layer answers can exhaust a question's share and hide a paired analysis.
  for (const allowRepeatedTarget of [false, true]) {
    for (const candidate of sorted) {
      const target = candidate.curatedAnalysisId ?? candidate.bindingId;
      const used = counts.get(candidate.question) ?? 0;
      if (
        selectedIds.has(candidate.id) ||
        selectedIds.size >= maximumCount ||
        used >= perQuestionLimit ||
        (!allowRepeatedTarget && targets.has(target))
      ) {
        continue;
      }
      selectedIds.add(candidate.id);
      targets.add(target);
      counts.set(candidate.question, used + 1);
    }
  }
  const selected: KeplerAnalysisSuggestion[] = [];
  const more: Partial<Record<GPUAnalysisQuestion, KeplerAnalysisSuggestion[]>> = {};
  // Keep both lists in score order, including repeats deferred by the first pass.
  for (const candidate of sorted) {
    if (selectedIds.has(candidate.id)) {
      selected.push(toSuggestion(candidate));
    } else {
      const hidden = more[candidate.question] ?? [];
      hidden.push(toSuggestion(candidate));
      more[candidate.question] = hidden;
    }
  }
  return {selected, more};
}

function evaluateCandidates(
  bindings: readonly CatalogOperationBinding[],
  datasets: readonly KeplerDatasetProfile[],
  curatedAnalyses: readonly KeplerCuratedAnalysis[],
  executionEngine: 'gpu' | 'cpu',
  dropped: string[],
  requiredDatasetId?: string
): Candidate[] {
  const candidates: Candidate[] = [];
  for (const binding of bindings) {
    const {descriptor} = resolveKeplerOperationDescriptor(binding);
    const curatedForBinding = curatedAnalyses.filter(
      analysis =>
        analysis.execution.status === 'available' && analysis.execution.bindingId === binding.id
    );
    // A curated analysis stands in for its own binding: both never appear.
    const variants: (KeplerCuratedAnalysis | undefined)[] =
      curatedForBinding.length === 0 ? [undefined] : curatedForBinding;
    for (const curated of variants) {
      if (requiredDatasetId !== undefined && binding.inputs.length === 1) {
        // A focused card gets its own layer even when it is a twin or beyond the global cap.
        candidates.push(
          evaluateCandidate(
            binding,
            descriptor,
            datasets.filter(dataset => dataset.datasetId === requiredDatasetId),
            curated,
            executionEngine,
            `@${requiredDatasetId}`,
            requiredDatasetId
          )
        );
        continue;
      }
      // An analysis of one layer is offered for every compatible loaded layer, not only the
      // first: with two point layers each gets its own suggestion.
      if (binding.inputs.length === 1 && datasets.length > 1) {
        const input = binding.inputs[0];
        const inputKind =
          descriptor.inputs.find(entry => entry.name === input.name)?.kind ?? 'points';
        let restricted: CatalogGeometry | undefined;
        if (curated?.execution.status === 'available') {
          const slot = curated.slots.find(
            entry =>
              curated.execution.status === 'available' &&
              curated.execution.inputs[entry.name] === input.name
          );
          const geometry = slot
            ? getGPUAnalysisCatalogGeometryForDataKind(slot.kind, 'input')
            : undefined;
          restricted =
            geometry === 'point' || geometry === 'line' || geometry === 'polygon'
              ? geometry
              : undefined;
        }
        const compatible: KeplerDatasetProfile[] = [];
        for (const dataset of [...datasets]
          .filter(entry => isKeplerDatasetCompatibleWithInput(entry, input, inputKind, restricted))
          .sort((left, right) => compareText(left.datasetId, right.datasetId))) {
          // A layer that holds the same rows as one already offered gets no suggestion of its own.
          const twin = compatible.find(kept => isSameLayer(kept, dataset));
          if (twin) {
            dropped.push(
              `"${dataset.label}" holds the same rows as "${twin.label}" and was not offered its own suggestion`
            );
            continue;
          }
          compatible.push(dataset);
        }
        if (compatible.length > 0) {
          const kept = compatible.slice(0, KEPLER_MAXIMUM_CANDIDATES_PER_BINDING);
          if (compatible.length > kept.length) {
            dropped.push(
              `${compatible.length - kept.length} more layers fit ${curated ? curated.name : descriptor.action} and were not offered a suggestion of their own (the first ${KEPLER_MAXIMUM_CANDIDATES_PER_BINDING} by id are)`
            );
          }
          for (const dataset of kept) {
            candidates.push(
              evaluateCandidate(
                binding,
                descriptor,
                [dataset],
                curated,
                executionEngine,
                `@${dataset.datasetId}`
              )
            );
          }
          continue;
        }
      }
      candidates.push(
        evaluateCandidate(
          binding,
          descriptor,
          datasets,
          curated,
          executionEngine,
          '',
          requiredDatasetId
        )
      );
    }
  }
  return requiredDatasetId === undefined
    ? candidates
    : candidates.filter(candidate => candidate.datasetIds.includes(requiredDatasetId));
}

/** Counterfactual copies share cached applicability across repeated ranking calls. */
const COUNTERFACTUAL_PROFILES = new WeakMap<
  KeplerDatasetProfile,
  {facts: string; profiles: Map<string, KeplerDatasetProfile>}
>();

function getCounterfactualProfile(
  dataset: KeplerDatasetProfile,
  fact: KeplerGuessedFact,
  answer: string
): KeplerDatasetProfile {
  const facts = stableStringify(dataset);
  let cached = COUNTERFACTUAL_PROFILES.get(dataset);
  if (!cached || cached.facts !== facts) {
    cached = {facts, profiles: new Map()};
    COUNTERFACTUAL_PROFILES.set(dataset, cached);
  }
  const key = `${fact.kind}|${fact.column ?? ''}|${answer}`;
  const previous = cached.profiles.get(key);
  if (previous) {
    return previous;
  }
  let profile: KeplerDatasetProfile;
  if (fact.kind === 'row-semantics') {
    profile = {
      ...dataset,
      rowSemantics: {
        value: answer as NonNullable<KeplerDatasetProfile['rowSemantics']>['value'],
        source: 'declared'
      }
    };
  } else if (fact.kind === 'coordinate-model') {
    profile = {
      ...dataset,
      coordinates: {model: answer as 'planar' | 'geographic', source: 'declared'}
    };
  } else {
    profile = {
      ...dataset,
      columns: dataset.columns.map(column =>
        column.name === fact.column
          ? {
              ...column,
              measurement: answer as 'count' | 'amount' | 'rate',
              measurementSource: 'declared' as const
            }
          : column
      )
    };
  }
  cached.profiles.set(key, profile);
  return profile;
}

/** Questions a host can ask to settle the order of the best candidates. */
function buildQuestions(
  window: readonly Candidate[],
  datasets: readonly KeplerDatasetProfile[],
  changesLeaders: (fact: KeplerGuessedFact, alternatives: readonly string[]) => boolean
): KeplerSuggestionQuestion[] {
  const groups = new Map<string, {fact: KeplerGuessedFact; affects: string[]}>();
  for (const candidate of window) {
    for (const fact of candidate.applicability.guessedFacts) {
      const key = `${fact.kind}|${fact.datasetId}|${fact.column ?? ''}`;
      const group = groups.get(key) ?? {fact, affects: []};
      group.affects.push(candidate.id);
      groups.set(key, group);
    }
  }
  const questions: KeplerSuggestionQuestion[] = [];
  for (const {fact, affects} of groups.values()) {
    const label =
      datasets.find(dataset => dataset.datasetId === fact.datasetId)?.label ?? fact.datasetId;
    if (fact.kind === 'row-semantics') {
      if (changesLeaders(fact, ROW_SEMANTICS_OPTIONS)) {
        questions.push({
          id: 'row-semantics',
          datasetId: fact.datasetId,
          prompt: `What does each row of "${label}" stand for?`,
          options: ROW_SEMANTICS_OPTIONS,
          affects
        });
      }
    } else if (fact.kind === 'coordinate-model' && changesLeaders(fact, ['geographic', 'planar'])) {
      questions.push({
        id: 'coordinate-model',
        datasetId: fact.datasetId,
        prompt: `Are the coordinates of "${label}" longitude and latitude in degrees, or planar (a metric or local system)?`,
        options: ['geographic', 'planar'],
        affects
      });
    } else if (fact.kind === 'measurement' && changesLeaders(fact, ['count', 'amount', 'rate'])) {
      questions.push({
        id: 'measurement',
        datasetId: fact.datasetId,
        ...(fact.column === undefined ? {} : {column: fact.column}),
        prompt: `What does "${fact.column ?? 'this column'}" in "${label}" measure?`,
        options: ['count', 'amount', 'rate'],
        affects
      });
    }
  }
  // The question that settles the most candidates first.
  return questions.sort(
    (left, right) =>
      right.affects.length - left.affects.length ||
      compareText(`${left.id}${left.datasetId}`, `${right.id}${right.datasetId}`)
  );
}

/** Keep the most recent suggestion state per immutable binding registry; profile facts invalidate it. */
const SUGGESTION_CACHE = new WeakMap<
  readonly CatalogOperationBinding[],
  {key: string; result: KeplerSuggestionResult}
>();

/**
 * Chooses the analyses to offer for the loaded datasets.
 *
 * Every binding is evaluated ({@link evaluateKeplerOperationApplicability}); a binding that a
 * curated analysis runs is offered only as that analysis. An analysis of one layer is offered for
 * every compatible loaded layer (up to {@link KEPLER_MAXIMUM_CANDIDATES_PER_BINDING}), so the
 * result does not depend on the order the layers were loaded in; an analysis of several layers
 * gets the best assignment ({@link KEPLER_MAXIMUM_DATASETS_PER_INPUT} tried per input). What the
 * limits left out is named in `dropped`. With `requiredDatasetId`, every assignment must contain
 * that dataset before ranking or diversity limits apply; other loaded layers remain partners.
 *
 * Candidates that can run are scored by the soft rules above (`scoreTerms` shows each one's
 * contribution) and taken best first; ties are broken by prominence, then the documented order of
 * analyses ({@link KEPLER_CURATED_ANALYSIS_ORDER}), then dataset ids, never by an analysis id.
 * Left out of `suggestions` (they stay under "Browse all"): blocked candidates, operations of
 * prominence 3, analyses whose declared row meaning they contradict, and analyses that lack a
 * required column altogether. Each question takes at most a third of the requested slots, rounded
 * down (at least one for budgets below three). Too few families produce a shorter list. Every
 * omitted runnable candidate stays in `more`.
 *
 * `blocked` lists up to four curated or prominence-1 analyses that lack exactly one thing (a
 * dataset, a column, a capacity) with how to supply it, ordered by how many loaded datasets they
 * already use and then by score. `questions` lists what a host can ask, one at a time, to settle
 * the order of the best candidates when it rests on a guessed or unknown fact.
 *
 * @throws RangeError when `maximumCount` is not an integer of at least 1.
 */
export function suggestKeplerAnalyses(
  bindings: readonly CatalogOperationBinding[],
  datasets: readonly KeplerDatasetProfile[],
  options: KeplerSuggestionOptions = {}
): KeplerSuggestionResult {
  const maximumCount = options.maximumCount ?? KEPLER_DEFAULT_SUGGESTION_COUNT;
  if (!Number.isInteger(maximumCount) || maximumCount < 1) {
    throw new RangeError(
      `suggestKeplerAnalyses maximumCount must be an integer of at least 1; got ${maximumCount}`
    );
  }
  const cacheKey = stableStringify({
    datasets: [...datasets].sort((left, right) => compareText(left.datasetId, right.datasetId)),
    bindingIds: bindings.map(binding => binding.id),
    options
  });
  const cached = SUGGESTION_CACHE.get(bindings);
  if (cached?.key === cacheKey) {
    return cached.result;
  }
  const bindingIds = new Set(bindings.map(binding => binding.id));
  const curatedAnalyses = (options.curated ?? KEPLER_CURATED_ANALYSES).filter(
    analysis =>
      analysis.execution.status === 'available' && bindingIds.has(analysis.execution.bindingId)
  );
  const dropped: string[] = [];
  const candidates = evaluateCandidates(
    bindings,
    datasets,
    curatedAnalyses,
    options.executionEngine ?? 'gpu',
    dropped,
    options.requiredDatasetId
  )
    // An operation that reads no layer has nothing to be suggested for.
    .filter(
      candidate =>
        !bindings.find(binding => binding.id === candidate.bindingId && binding.inputs.length === 0)
    )
    .sort(compareCandidates);

  const runnable = candidates.filter(
    candidate =>
      candidate.status !== 'blocked' &&
      candidate.prominence < KEPLER_HIDDEN_PROMINENCE &&
      !candidate.applicability.declaredContradiction &&
      !candidate.applicability.hasEmptyRole
  );
  // The greyed-out rows should be about the data a person has: an analysis that already uses a
  // loaded dataset and lacks one thing comes before one that uses none. A column that no
  // plausible candidate fills counts as one thing too.
  const oneStepAway = candidates
    .filter(
      candidate =>
        (candidate.status === 'blocked' ||
          (candidate.status === 'needs-mapping' && candidate.applicability.hasEmptyRole)) &&
        candidate.applicability.missing.length === 1 &&
        (candidate.isCurated || candidate.prominence === 1)
    )
    .sort(
      (left, right) => right.assignedCount - left.assignedCount || compareCandidates(left, right)
    );
  const chosen = selectDiverse(runnable, maximumCount);
  // The separate four-row remedy list prioritizes loaded inputs, rather than filling the
  // recommendation row's diversity budget with unrelated missing layers.
  const greyed = oneStepAway.slice(0, KEPLER_BLOCKED_SUGGESTION_COUNT).map(toSuggestion);
  const result: KeplerSuggestionResult = {
    suggestions: chosen.selected,
    blocked: greyed,
    more: chosen.more,
    questions: buildQuestions(candidates, datasets, (fact, alternatives) => {
      const current = chosen.selected
        .slice(0, 3)
        .map(candidate => `${candidate.id}:${candidate.status}`)
        .join('|');
      return alternatives.some(answer => {
        const changed = datasets.map(dataset => {
          if (dataset.datasetId !== fact.datasetId) {
            return dataset;
          }
          return getCounterfactualProfile(dataset, fact, answer);
        });
        const alternatives = evaluateCandidates(
          bindings,
          changed,
          curatedAnalyses,
          options.executionEngine ?? 'gpu',
          [],
          options.requiredDatasetId
        )
          .filter(
            candidate =>
              candidate.status !== 'blocked' &&
              candidate.prominence < KEPLER_HIDDEN_PROMINENCE &&
              !candidate.applicability.declaredContradiction &&
              !candidate.applicability.hasEmptyRole
          )
          .sort(compareCandidates);
        const alternative = selectDiverse(alternatives, maximumCount)
          .selected.slice(0, 3)
          .map(candidate => `${candidate.id}:${candidate.status}`)
          .join('|');
        return alternative !== current;
      });
    }),
    dropped: [...new Set(dropped)],
    total: {
      ready: candidates.filter(candidate => candidate.status === 'ready').length,
      needsInput: candidates.filter(
        candidate =>
          candidate.status === 'needs-confirmation' || candidate.status === 'needs-mapping'
      ).length,
      blocked: candidates.filter(candidate => candidate.status === 'blocked').length
    }
  };
  SUGGESTION_CACHE.set(bindings, {key: cacheKey, result});
  return result;
}

// ---------------------------------------------------------------------------------------------
// Browse all
// ---------------------------------------------------------------------------------------------

export type KeplerApplicableBinding = {
  bindingId: string;
  title: string;
  prominence: 1 | 2 | 3;
  /** True when no hand-written descriptor exists and a conservative one was derived. */
  derived: boolean;
  applicability: KeplerOperationApplicability;
};

export type KeplerApplicableBindingGroup = {
  question: GPUAnalysisQuestion;
  /** The question as a person asks it. */
  label: string;
  bindings: readonly KeplerApplicableBinding[];
};

/**
 * Every binding that is not blocked for the loaded datasets, grouped by primary question in the
 * ontology's question order and, inside a group, by prominence and then id. This is the list
 * behind "Browse all that fit": nothing is hidden by rank or by the diversity limits.
 * `requiredDatasetId` anchors assignment search to a dataset card without excluding its partners.
 */
export function listKeplerApplicableBindings(
  bindings: readonly CatalogOperationBinding[],
  datasets: readonly KeplerDatasetProfile[],
  options: Pick<KeplerApplicabilityOptions, 'requiredDatasetId'> = {}
): readonly KeplerApplicableBindingGroup[] {
  const groups = new Map<GPUAnalysisQuestion, KeplerApplicableBinding[]>();
  for (const binding of bindings) {
    const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
    const applicability = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      datasets,
      options
    );
    if (
      applicability.status === 'blocked' ||
      (options.requiredDatasetId !== undefined &&
        !Object.values(applicability.inputAssignments).includes(options.requiredDatasetId))
    ) {
      continue;
    }
    const question = getKeplerPrimaryQuestion(descriptor);
    const group = groups.get(question) ?? [];
    group.push({
      bindingId: binding.id,
      title: descriptor.action,
      prominence: descriptor.prominence,
      derived: source === 'derived',
      applicability
    });
    groups.set(question, group);
  }
  return GPU_ANALYSIS_QUESTIONS.filter(question => groups.has(question)).map(question => ({
    question,
    label: getGPUAnalysisQuestion(question).label,
    bindings: [...(groups.get(question) ?? [])].sort(
      (left, right) =>
        left.prominence - right.prominence ||
        (left.bindingId < right.bindingId ? -1 : left.bindingId > right.bindingId ? 1 : 0)
    )
  }));
}
