// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

import {
  assignGPUAnalysisFieldRoles,
  matchGPUAnalysisFieldRole,
  tokenizeGPUAnalysisColumnName,
  getGPUAnalysisCatalogGeometryForDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  GPUAnalysisColumnProfile,
  GPUAnalysisFieldRoleSpec,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {isKeplerDatasetCompatibleWithInput} from '../descriptors/applicability';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {getCatalogBinding} from '../analysis/catalog-bindings';
import type {KeplerOperationGeometry} from '../descriptors/operation-descriptor-types';
import {getKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS} from './curated-analysis-samples';
import type {KeplerCuratedAnalysis, KeplerCuratedAnalysisSlot} from './curated-analysis-types';

// The step between "I picked a curated analysis" and "it runs on my data": bind each typed slot to
// one loaded dataset and each field role to columns of it, pre-filled from the profiles, with every
// loaded dataset and column listed so that a host can grey out those that do not fit and say why.
//
// A mapping is derived, never edited in place. It is a pure function of four things: the analysis,
// the datasets, the parameter values that decide which roles are required, and the user's own
// choices (`choices`). Updating a mapping adds one choice and derives again, so dependent choices
// (the roles of a slot whose dataset changed) are always consistent. Only the user's choices are
// saved; automatic ones are derived again on replay, which is what lets a saved mapping meet
// datasets with different columns.
//
// Rules for the automatic choice (the report of this stream repeats them):
// - A dataset fits a slot through shared binding kind/geometry compatibility, narrowed to the
//   slot's geometry (polygon-backed cells also fit polygon slots), and every role required now
//   can be given its own column of an accepted type. Row semantics outside the slot's list are a
//   warning on the candidate, not a block, as for measurement levels.
// - Slots are filled jointly: no dataset fills two slots unless it is the only one that fits the
//   second slot, and then the second slot is flagged as a conflict.
// - Among fitting datasets a score decides: slot name inside the dataset label or id, row
//   semantics, required roles that have a `preferred` column, columns named for their role.
//   Equal scores are ambiguous: the choice is made by dataset id but flagged `choose`.
// - A role takes `preferred` columns first, then those named for the role or for the sample column
//   (whole-word match through `tokenizeGPUAnalysisColumnName`, never substring), then column order.
//   A required role is `settled` when its column is the only preferred one left or is named for the
//   role and no other preferred column is; `choose` when several fit equally or only a column with
//   a warning (`accepted`) is left.
// - An optional role (not required with the current parameters) is never filled automatically: a
//   column there changes what the analysis counts or weights. Its candidates are offered and its
//   `reason` says so; clearing it removes the user's choice.
// - Row semantics the user declared that contradict the slot's make the slot `choose`; a guess or
//   absent semantics is only a warning on the candidate.
// - Roles of two slots that feed one binding parameter share one column: the candidates are the
//   columns every dataset has under the same name with a compatible type.
// - A column name the binding parameter cannot hold (too long, or a comma in a list role) is
//   listed as `unavailable` and never chosen.
// - `ready` means complete and nothing flagged `choose`; the user's explicit choice settles a flag
//   and leaves an `acknowledged` problem behind. Guessed coordinates are a `check` problem that does
//   not change `ready`.

/** Slots and roles whose required-ness depends on parameters need their values; presets give them. */
export type KeplerCuratedAnalysisMappingOptions = {
  /** Parameter values added to the analysis presets when deciding which roles are required now. */
  parameters?: Readonly<Record<string, CatalogParameterValue>>;
};

/** `fits`: can fill the slot. `wrong-kind`: another kind of data. `missing-columns`: right kind, lacks a needed column. */
export type KeplerCuratedMappingDatasetFit = 'fits' | 'wrong-kind' | 'missing-columns';

/** One loaded dataset as a candidate for one slot. Every loaded dataset appears, whatever its fit. */
export type KeplerCuratedMappingDatasetCandidate = {
  datasetId: string;
  label: string;
  fit: KeplerCuratedMappingDatasetFit;
  /** Why it fits or does not, in a sentence a host can show beside a greyed entry. */
  reason: string;
  /** Present on a fitting dataset that is used with a caveat, for example row semantics. */
  warning?: string;
};

/** One column as a candidate for one role. Only columns of an accepted type are listed. */
export type KeplerCuratedMappingColumnCandidate = {
  name: string;
  fit: 'preferred' | 'accepted';
  /** The analysis package's warning, plus `Also used for <role>` when another role holds the column. */
  warning?: string;
  /** Present when the column is listed but cannot be chosen (a name too long or with a comma); says why. */
  unavailable?: string;
};

/**
 * `settled`: nothing to check. `choose`: the value shown is a first pick the user should confirm.
 * `empty`: nothing chosen.
 */
export type KeplerCuratedMappingChoiceState = 'settled' | 'choose' | 'empty';

export type KeplerCuratedMappingRole = {
  role: string;
  label: string;
  /** Required with the parameter values the mapping was derived for. */
  required: boolean;
  multiple?: {minimum: number; maximum: number};
  /**
   * `<slot>.<role>` of the roles in other slots that feed the same binding parameter. The binding
   * reads one column name for all of them, so candidates are the columns every dataset has (same
   * name, compatible type) and a choice is made once for all.
   */
  sharedWith?: readonly string[];
  candidates: readonly KeplerCuratedMappingColumnCandidate[];
  /** Chosen columns; empty when none. A role without `multiple` has at most one. */
  columns: readonly string[];
  source: 'automatic' | 'user' | 'none';
  state: KeplerCuratedMappingChoiceState;
  /** Why the state is `choose`, or why an optional role was left empty. */
  reason?: string;
  /** Warnings of the chosen columns. */
  warnings: readonly string[];
};

export type KeplerCuratedMappingSlot = {
  slot: string;
  label: string;
  kind: GPUSpatialAnalysisDataKind;
  /** Every loaded dataset: fitting ones first (best first), then missing-columns, then wrong-kind. */
  candidates: readonly KeplerCuratedMappingDatasetCandidate[];
  datasetId?: string;
  source: 'automatic' | 'user' | 'none';
  state: KeplerCuratedMappingChoiceState;
  /** True when `datasetId` is also the dataset of an earlier slot because no other dataset fits. */
  conflict: boolean;
  reason?: string;
  /** Caveat on the chosen dataset (row semantics other than the analysis expects). */
  warning?: string;
  /** The coordinate model of the chosen dataset and whether it is a guess; absent when the profile has none. */
  coordinates?: {model: 'geographic' | 'planar'; source: 'guess' | 'declared'; reason?: string};
  roles: readonly KeplerCuratedMappingRole[];
};

export type KeplerCuratedMappingProblem = {
  kind:
    | 'slot-empty'
    | 'slot-conflict'
    | 'role-empty'
    | 'needs-choice'
    | 'column-warning'
    | 'dataset-warning'
    | 'saved-choice-lost'
    | 'dataset-duplicate'
    | 'shared-column-missing'
    | 'field-too-long'
    | 'coordinate-guess';
  /** `blocks`: the analysis cannot run. `check`: it runs, but the user should look. */
  severity: 'blocks' | 'check';
  /**
   * True when the user made the choice the problem is about (a column or dataset picked in spite
   * of its warning). `ready` ignores acknowledged problems; a host can show them quietly.
   */
  acknowledged: boolean;
  /** Absent for a problem about the dataset list as a whole. */
  slot?: string;
  role?: string;
  message: string;
};

/** The user's own choices: the only part of a mapping that is saved. */
export type KeplerCuratedAnalysisMappingChoices = {
  analysisId: string;
  /** Slot name to dataset id. */
  datasets: Readonly<Record<string, string>>;
  /** `<slot>.<role>` to columns. */
  columns: Readonly<Record<string, readonly string[]>>;
};

export type KeplerCuratedAnalysisMapping = {
  analysisId: string;
  /** Preset and caller parameter values that decided which roles are required. */
  parameterValues: Readonly<Record<string, CatalogParameterValue>>;
  /** Settings supplied by the caller, including values equal to an inherited preset. */
  explicitParameterNames?: readonly string[];
  slots: readonly KeplerCuratedMappingSlot[];
  /** The "3 of 3 mapped" counter: every slot plus every role required now. */
  progress: {mapped: number; required: number};
  /** Nothing blocks a run. */
  complete: boolean;
  /** Complete and nothing needs the user's confirmation. */
  ready: boolean;
  problems: readonly KeplerCuratedMappingProblem[];
  choices: KeplerCuratedAnalysisMappingChoices;
};

export type KeplerCuratedAnalysisMappingChange =
  | {slot: string; datasetId: string}
  | {slot: string; role: string; columns: readonly string[]};

export type KeplerCuratedAnalysisRequest = {
  bindingId: string;
  /** Catalog binding input name to dataset id. */
  inputDatasetIds: Readonly<Record<string, string>>;
  /**
   * Presets, then field parameters, then overrides. Parameters none of these set are absent: the
   * module that derives defaults from the profile fills them, and the catalog fills the rest.
   */
  parameters: Record<string, CatalogParameterValue>;
};

export type KeplerCuratedAnalysisAvailability = {
  analysisId: string;
  status: 'ready' | 'needs-mapping' | 'needs-data' | 'needs-binding';
  mapping?: KeplerCuratedAnalysisMapping;
  reason: string;
};

// ---------------------------------------------------------------------------------------------
// Scores and wording
// ---------------------------------------------------------------------------------------------

/**
 * Dataset score weights. Each tier outweighs the whole tier below it: a slot has at most a few
 * roles, so the role terms stay under 10 and cannot overflow into the row-semantics term.
 */
const SLOT_NAME_SCORE = 1000;
const ROW_SEMANTICS_MATCH_SCORE = 100;
const ROW_SEMANTICS_UNKNOWN_SCORE = 50;
const PREFERRED_ROLE_SCORE = 10;
const ROLE_NAME_SCORE = 1;

/** How a data kind reads in a sentence. */
const KIND_PHRASES: Readonly<Partial<Record<GPUSpatialAnalysisDataKind, string>>> = {
  points: 'points',
  lines: 'lines',
  polygons: 'polygons',
  network: 'lines (a street network)',
  surface: 'a raster',
  trajectories: 'trajectories',
  cells: 'cells',
  attributes: 'a table'
};

function describeKind(kind: GPUSpatialAnalysisDataKind): string {
  return KIND_PHRASES[kind] ?? kind;
}

function getRoleKey(slotName: string, roleName: string): string {
  return `${slotName}.${roleName}`;
}

function joinWithAnd(items: readonly string[]): string {
  return items.length <= 1
    ? items.join('')
    : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

function describeColumnTypes(role: GPUAnalysisFieldRoleSpec): string {
  return role.accepts.join(' or ');
}

// ---------------------------------------------------------------------------------------------
// Parameters and roles
// ---------------------------------------------------------------------------------------------

function getPresetParameters(
  analysis: KeplerCuratedAnalysis
): Readonly<Record<string, CatalogParameterValue>> {
  return analysis.execution.status === 'available' ? analysis.execution.parameters : {};
}

function isRoleRequired(
  role: GPUAnalysisFieldRoleSpec,
  parameterValues: Readonly<Record<string, CatalogParameterValue>>
): boolean {
  if (typeof role.required === 'boolean') {
    return role.required;
  }
  return evaluateGPUAnalysisParameterCondition(role.required, parameterValues);
}

/** The analysis package compares scalar parameter values only; lists cannot decide a role. */
function getScalarParameters(
  parameterValues: Readonly<Record<string, CatalogParameterValue>>
): Record<string, string | number | boolean> {
  const scalars: Record<string, string | number | boolean> = {};
  for (const [name, value] of Object.entries(parameterValues)) {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      scalars[name] = value;
    }
  }
  return scalars;
}

function getSampleColumn(
  analysis: KeplerCuratedAnalysis,
  slotName: string,
  roleName: string
): string | undefined {
  const sampleId = analysis.sampleId;
  if (sampleId === undefined || !(sampleId in KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS)) {
    return undefined;
  }
  const columns =
    KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS[
      sampleId as keyof typeof KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS
    ];
  return columns[getRoleKey(slotName, roleName)];
}

function isSameTokenList(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((token, index) => token === right[index]);
}

/**
 * 0: the whole name equals the role id or the sample's column (as word lists). 1: the role id's
 * words are all words of the name. 2: no name evidence. Whole words only, never substrings.
 */
function getNameRank(
  roleName: string,
  sampleColumn: string | undefined,
  columnName: string
): number {
  const columnTokens = tokenizeGPUAnalysisColumnName(columnName);
  const roleTokens = tokenizeGPUAnalysisColumnName(roleName);
  if (isSameTokenList(columnTokens, roleTokens)) {
    return 0;
  }
  if (
    sampleColumn !== undefined &&
    isSameTokenList(columnTokens, tokenizeGPUAnalysisColumnName(sampleColumn))
  ) {
    return 0;
  }
  if (roleTokens.length > 0 && roleTokens.every(token => columnTokens.includes(token))) {
    return 1;
  }
  return 2;
}

type RankedColumn = KeplerCuratedMappingColumnCandidate & {nameRank: number};

/** Matcher order with `preferred` first, then by name evidence; the matcher's order breaks ties. */
function rankColumns(
  role: GPUAnalysisFieldRoleSpec,
  sampleColumn: string | undefined,
  columns: readonly GPUAnalysisColumnProfile[],
  excludedColumnNames: readonly string[]
): RankedColumn[] {
  const matches = matchGPUAnalysisFieldRole(role, columns, {excludedColumnNames});
  return matches
    .map((match, index) => ({
      ...match,
      nameRank: getNameRank(role.role, sampleColumn, match.name),
      index
    }))
    .sort((left, right) => {
      const leftFit = left.fit === 'preferred' ? 0 : 1;
      const rightFit = right.fit === 'preferred' ? 0 : 1;
      return leftFit - rightFit || left.nameRank - right.nameRank || left.index - right.index;
    })
    .map(({index: _index, ...ranked}) => ranked);
}

function getWantedColumnCount(role: GPUAnalysisFieldRoleSpec): number {
  const minimum = Math.max(role.multiple?.minimum ?? 1, 1);
  return Math.min(minimum, role.multiple?.maximum ?? 1);
}

// ---------------------------------------------------------------------------------------------
// Dataset candidates
// ---------------------------------------------------------------------------------------------

type DatasetEvaluation = {
  candidate: KeplerCuratedMappingDatasetCandidate;
  profile: KeplerDatasetProfile;
  score: number;
  /** The rows' semantics were declared by the user and are not the ones the slot lists. */
  declaredMismatch?: boolean;
};

function isKindCompatible(
  analysis: KeplerCuratedAnalysis,
  slot: KeplerCuratedAnalysisSlot,
  profile: KeplerDatasetProfile
): boolean {
  const execution = analysis.execution;
  const input =
    execution.status === 'available'
      ? getCatalogBinding(execution.bindingId)?.inputs.find(
          candidate => candidate.name === execution.inputs[slot.name]
        )
      : undefined;
  const geometry = getGPUAnalysisCatalogGeometryForDataKind(slot.kind, 'input');
  const restrictedGeometry =
    geometry === 'point' || geometry === 'line' || geometry === 'polygon' ? geometry : undefined;
  return isKeplerDatasetCompatibleWithInput(
    profile,
    input ?? {
      name: slot.name,
      label: slot.label,
      ...(slot.kind === 'surface' ? {dataKind: 'raster' as const} : {}),
      accepts: restrictedGeometry ? [restrictedGeometry] : ['point', 'line', 'polygon'],
      maximumFeatures: 0,
      maximumVertices: 0
    },
    slot.kind,
    restrictedGeometry
  );
}

function hasSlotNameInDataset(slotName: string, profile: KeplerDatasetProfile): boolean {
  const slotTokens = tokenizeGPUAnalysisColumnName(slotName);
  const datasetTokens = [
    ...tokenizeGPUAnalysisColumnName(profile.label),
    ...tokenizeGPUAnalysisColumnName(profile.datasetId)
  ];
  return slotTokens.length > 0 && slotTokens.every(token => datasetTokens.includes(token));
}

function evaluateDataset(
  analysis: KeplerCuratedAnalysis,
  slot: KeplerCuratedAnalysisSlot,
  profile: KeplerDatasetProfile,
  parameterValues: Readonly<Record<string, CatalogParameterValue>>
): DatasetEvaluation {
  const base = {datasetId: profile.datasetId, label: profile.label};
  if (!isKindCompatible(analysis, slot, profile)) {
    return {
      profile,
      score: 0,
      candidate: {
        ...base,
        fit: 'wrong-kind',
        reason: `${slot.label} needs ${describeKind(slot.kind)}; ${profile.label} holds ${describeKind(profile.kind)}.`
      }
    };
  }
  const assignment = assignGPUAnalysisFieldRoles(
    slot.fields,
    profile.columns,
    getScalarParameters(parameterValues)
  );
  if (assignment.unfilledRoles.length > 0) {
    const labels = assignment.unfilledRoles.map(
      roleName => slot.fields.find(field => field.role === roleName)?.label ?? roleName
    );
    return {
      profile,
      score: 0,
      candidate: {
        ...base,
        fit: 'missing-columns',
        reason: `${profile.label} has no column for ${joinWithAnd(labels)}.`
      }
    };
  }

  let score = hasSlotNameInDataset(slot.name, profile) ? SLOT_NAME_SCORE : 0;
  let warning: string | undefined;
  let declaredMismatch = false;
  const semantics = profile.rowSemantics?.value;
  if (slot.rowSemantics === undefined || slot.rowSemantics.length === 0) {
    score += ROW_SEMANTICS_MATCH_SCORE;
  } else if (semantics === undefined) {
    score += ROW_SEMANTICS_UNKNOWN_SCORE;
  } else if (slot.rowSemantics.includes(semantics)) {
    score += ROW_SEMANTICS_MATCH_SCORE;
  } else {
    declaredMismatch = profile.rowSemantics?.source === 'declared';
    const source = declaredMismatch ? 'declared' : 'a guess';
    warning = `Rows are ${semantics} (${source}); this analysis is meant for ${slot.rowSemantics.join(' or ')} rows.`;
  }
  for (const role of slot.fields) {
    if (!isRoleRequired(role, parameterValues)) {
      continue;
    }
    const sampleColumn = getSampleColumn(analysis, slot.name, role.role);
    const ranked = rankColumns(role, sampleColumn, profile.columns, []);
    if (ranked.some(column => column.fit === 'preferred')) {
      score += PREFERRED_ROLE_SCORE;
    }
    if (ranked.some(column => column.nameRank < 2)) {
      score += ROLE_NAME_SCORE;
    }
  }
  const rowText = `${profile.rowCount.toLocaleString('en-US')} ${profile.rowCount === 1 ? 'row' : 'rows'}`;
  return {
    profile,
    score,
    declaredMismatch,
    candidate: {
      ...base,
      fit: 'fits',
      reason:
        slot.fields.length === 0
          ? `${describeKind(profile.kind)}, ${rowText}.`
          : `${describeKind(profile.kind)}, ${rowText}; every required field has a column.`,
      ...(warning === undefined ? {} : {warning})
    }
  };
}

const FIT_ORDER: Readonly<Record<KeplerCuratedMappingDatasetFit, number>> = {
  fits: 0,
  'missing-columns': 1,
  'wrong-kind': 2
};

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function evaluateSlotDatasets(
  analysis: KeplerCuratedAnalysis,
  slot: KeplerCuratedAnalysisSlot,
  datasets: readonly KeplerDatasetProfile[],
  parameterValues: Readonly<Record<string, CatalogParameterValue>>
): DatasetEvaluation[] {
  return datasets
    .map(profile => evaluateDataset(analysis, slot, profile, parameterValues))
    .sort(
      (left, right) =>
        FIT_ORDER[left.candidate.fit] - FIT_ORDER[right.candidate.fit] ||
        right.score - left.score ||
        compareText(left.candidate.datasetId, right.candidate.datasetId)
    );
}

// ---------------------------------------------------------------------------------------------
// Joint dataset assignment
// ---------------------------------------------------------------------------------------------

type SlotAssignment = {
  datasetId?: string;
  source: 'automatic' | 'user' | 'none';
  state: KeplerCuratedMappingChoiceState;
  conflict: boolean;
};

/**
 * Chooses a dataset for every slot that the user has not fixed. Distinct datasets are tried first
 * (most slots filled, then highest total score, then smallest dataset ids in slot order); every
 * assignment that ties the best is kept so that a slot whose dataset could swap with another's is
 * reported as a choice. A slot still empty gets its best fitting dataset again, flagged as a
 * conflict. The search is exhaustive; slots number at most three and fitting datasets are few.
 */
function assignDatasets(
  slots: readonly KeplerCuratedAnalysisSlot[],
  evaluations: readonly (readonly DatasetEvaluation[])[],
  pinnedDatasetIds: readonly (string | undefined)[]
): SlotAssignment[] {
  const result: SlotAssignment[] = slots.map(() => ({
    source: 'none',
    state: 'empty',
    conflict: false
  }));
  const used = new Set<string>();
  const seenPinned = new Set<string>();
  const freeSlotIndexes: number[] = [];

  slots.forEach((_slot, slotIndex) => {
    const pinned = pinnedDatasetIds[slotIndex];
    if (pinned === undefined) {
      freeSlotIndexes.push(slotIndex);
      return;
    }
    result[slotIndex] = {
      datasetId: pinned,
      source: 'user',
      state: 'settled',
      conflict: seenPinned.has(pinned)
    };
    seenPinned.add(pinned);
    used.add(pinned);
  });

  const fitting = freeSlotIndexes.map(slotIndex =>
    evaluations[slotIndex].filter(evaluation => evaluation.candidate.fit === 'fits')
  );
  let bestFilled = -1;
  let bestScore = -1;
  let bestChoice: (string | undefined)[] = [];
  let optimalSets: Set<string | undefined>[] = freeSlotIndexes.map(() => new Set());
  const current: (string | undefined)[] = [];

  const search = (position: number, filled: number, score: number): void => {
    if (position === freeSlotIndexes.length) {
      if (filled > bestFilled || (filled === bestFilled && score > bestScore)) {
        bestFilled = filled;
        bestScore = score;
        bestChoice = [...current];
        optimalSets = current.map(datasetId => new Set([datasetId]));
      } else if (filled === bestFilled && score === bestScore) {
        current.forEach((datasetId, index) => optimalSets[index].add(datasetId));
      }
      return;
    }
    for (const evaluation of fitting[position]) {
      const datasetId = evaluation.candidate.datasetId;
      if (used.has(datasetId)) {
        continue;
      }
      used.add(datasetId);
      current[position] = datasetId;
      search(position + 1, filled + 1, score + evaluation.score);
      used.delete(datasetId);
    }
    current[position] = undefined;
    search(position + 1, filled, score);
  };
  search(0, 0, 0);

  freeSlotIndexes.forEach((slotIndex, position) => {
    const datasetId = bestChoice[position];
    if (datasetId !== undefined) {
      result[slotIndex] = {
        datasetId,
        source: 'automatic',
        // With a slot left over, which slot got the dataset is not the question; the missing one is.
        state:
          optimalSets[position].size > 1 && bestFilled === freeSlotIndexes.length
            ? 'choose'
            : 'settled',
        conflict: false
      };
      return;
    }
    const fallback = fitting[position][0];
    if (fallback !== undefined) {
      result[slotIndex] = {
        datasetId: fallback.candidate.datasetId,
        source: 'automatic',
        state: 'settled',
        conflict: true
      };
    }
  });
  return result;
}

// ---------------------------------------------------------------------------------------------
// Role derivation
// ---------------------------------------------------------------------------------------------

type RoleChosen = {
  columns: readonly string[];
  source: 'automatic' | 'user' | 'none';
  state: KeplerCuratedMappingChoiceState;
  reason?: string;
};

/** Settled when the chosen columns are the only preferred ones left, or the only ones with name evidence. */
function isSettledChoice(
  chosen: readonly RankedColumn[],
  preferred: readonly RankedColumn[],
  wanted: number
): boolean {
  if (preferred.length <= wanted) {
    return true;
  }
  const chosenNames = new Set(chosen.map(column => column.name));
  const worstChosenRank = Math.max(...chosen.map(column => column.nameRank));
  if (worstChosenRank >= 2) {
    return false;
  }
  return preferred
    .filter(column => !chosenNames.has(column.name))
    .every(column => column.nameRank > worstChosenRank);
}

function describeAlternatives(
  preferred: readonly RankedColumn[],
  chosen: readonly string[]
): string {
  const others = preferred.map(column => column.name).filter(name => !chosen.includes(name));
  const shown = others.slice(0, 3);
  return `${shown.join(', ')}${others.length > shown.length ? ' and others' : ''}`;
}

type SlotRoleInput = {
  analysis: KeplerCuratedAnalysis;
  slot: KeplerCuratedAnalysisSlot;
  profile: KeplerDatasetProfile;
  parameterValues: Readonly<Record<string, CatalogParameterValue>>;
  /** Columns the user fixed, by role id; already checked against the candidates. */
  pinned: ReadonlyMap<string, readonly string[]>;
  /** Roles whose saved choice was lost, with the sentence that says so. */
  lost: ReadonlyMap<string, string>;
  /** Per role, columns that cannot be used (name too long or with a comma) and why. */
  blocked: ReadonlyMap<string, ReadonlyMap<string, string>>;
  /** Per role, columns left out because another slot's dataset lacks them (shared parameter). */
  notShared: ReadonlyMap<string, ReadonlySet<string>>;
  /** Per role, the choice copied from the slot that leads a shared parameter. */
  forced: ReadonlyMap<string, RoleChosen>;
  /** Per role, the other slots' `<slot>.<role>` keys that feed the same binding parameter. */
  sharedWith: ReadonlyMap<string, readonly string[]>;
};

/** Columns kept out of a role's candidates, listed or not. */
function getHiddenColumns(input: SlotRoleInput, roleName: string): string[] {
  return [...(input.notShared.get(roleName) ?? [])];
}

function describeOptionalRole(
  analysis: KeplerCuratedAnalysis,
  role: GPUAnalysisFieldRoleSpec,
  usable: readonly RankedColumn[]
): string {
  if (usable.length === 0) {
    return `No ${describeColumnTypes(role)} column is available for this optional field.`;
  }
  const shown = usable
    .slice(0, 3)
    .map(column => column.name)
    .join(', ');
  return `${role.label} is optional and is never filled automatically: a column here changes what ${analysis.name} computes compared with leaving it empty. ${usable.length} ${usable.length === 1 ? 'column fits' : 'columns fit'} (${shown}${usable.length > 3 ? ', ...' : ''}); choose one to use it.`;
}

function deriveRoleChoices(input: SlotRoleInput): Map<string, RoleChosen> {
  const {analysis, slot, profile, parameterValues, pinned, lost, blocked, forced} = input;
  const choices = new Map<string, RoleChosen>();
  const used = new Set<string>();
  const excludedFor = (roleName: string, taken: readonly string[]): string[] => [
    ...taken,
    ...(blocked.get(roleName)?.keys() ?? []),
    ...getHiddenColumns(input, roleName)
  ];

  for (const [roleName, columns] of pinned) {
    choices.set(roleName, {columns, source: 'user', state: 'settled'});
    columns.forEach(column => used.add(column));
  }
  for (const [roleName, copy] of forced) {
    if (!pinned.has(roleName)) {
      choices.set(roleName, copy);
      copy.columns.forEach(column => used.add(column));
    }
  }

  const pending = slot.fields
    .map((role, index) => ({role, index, required: isRoleRequired(role, parameterValues)}))
    .filter(entry => !choices.has(entry.role.role))
    .map(entry => {
      const ranked = rankColumns(
        entry.role,
        getSampleColumn(analysis, slot.name, entry.role.role),
        profile.columns,
        excludedFor(entry.role.role, [...used])
      );
      return {
        ...entry,
        preferredCount: ranked.filter(column => column.fit === 'preferred').length,
        candidateCount: ranked.length
      };
    })
    .sort(
      (left, right) =>
        Number(right.required) - Number(left.required) ||
        (left.required && right.required
          ? left.preferredCount - right.preferredCount || left.candidateCount - right.candidateCount
          : 0) ||
        left.index - right.index
    );

  for (const {role, required} of pending) {
    const ranked = rankColumns(
      role,
      getSampleColumn(analysis, slot.name, role.role),
      profile.columns,
      excludedFor(role.role, [...used])
    );
    const lostReason = lost.get(role.role);
    if (!required) {
      // An optional column changes what the analysis counts or weights, so the user decides.
      choices.set(role.role, {
        columns: [],
        source: 'none',
        state: 'empty',
        reason: lostReason ?? describeOptionalRole(analysis, role, ranked)
      });
      continue;
    }
    const preferred = ranked.filter(column => column.fit === 'preferred');
    const wanted = getWantedColumnCount(role);
    if (ranked.length < wanted) {
      choices.set(role.role, {columns: [], source: 'none', state: 'empty', reason: lostReason});
      continue;
    }
    const chosen = ranked.slice(0, wanted);
    const accepted = chosen.find(column => column.fit === 'accepted');
    let state: KeplerCuratedMappingChoiceState = 'settled';
    let reason: string | undefined;
    if (accepted !== undefined) {
      state = 'choose';
      reason = accepted.warning ?? `${accepted.name} fits by type only.`;
    } else if (!isSettledChoice(chosen, preferred, wanted)) {
      state = 'choose';
      reason = `${preferred.length} columns could fill ${role.label}; ${chosen.map(column => column.name).join(', ')} came first, then ${describeAlternatives(
        preferred,
        chosen.map(column => column.name)
      )}.`;
    }
    if (lostReason !== undefined) {
      state = 'choose';
      reason = lostReason;
    }
    chosen.forEach(column => used.add(column.name));
    choices.set(role.role, {
      columns: chosen.map(column => column.name),
      source: 'automatic',
      state,
      ...(reason === undefined ? {} : {reason})
    });
  }
  return choices;
}

function buildRoles(
  input: SlotRoleInput,
  chosenByRole: Map<string, RoleChosen>
): KeplerCuratedMappingRole[] {
  const {analysis, slot, profile, parameterValues, blocked, sharedWith} = input;
  return slot.fields.map(role => {
    const chosen = chosenByRole.get(role.role) ?? {
      columns: [],
      source: 'none' as const,
      state: 'empty' as const
    };
    const usedByOther = new Map<string, string>();
    for (const other of slot.fields) {
      if (other.role === role.role) {
        continue;
      }
      for (const column of chosenByRole.get(other.role)?.columns ?? []) {
        usedByOther.set(column, other.label);
      }
    }
    const candidates = rankColumns(
      role,
      getSampleColumn(analysis, slot.name, role.role),
      profile.columns,
      getHiddenColumns(input, role.role)
    ).map(({nameRank: _nameRank, ...candidate}) => {
      const otherLabel = usedByOther.get(candidate.name);
      const unavailable = blocked.get(role.role)?.get(candidate.name);
      const note = otherLabel === undefined ? undefined : `Also used for ${otherLabel}.`;
      return {
        ...candidate,
        ...(note === undefined
          ? {}
          : {warning: candidate.warning ? `${candidate.warning}; ${note}` : note}),
        ...(unavailable === undefined ? {} : {unavailable})
      };
    });
    const warnings = chosen.columns.flatMap(column => {
      const warning = candidates.find(candidate => candidate.name === column)?.warning;
      return warning === undefined ? [] : [warning];
    });
    const sharing = sharedWith.get(role.role);
    return {
      role: role.role,
      label: role.label,
      required: isRoleRequired(role, parameterValues),
      ...(role.multiple === undefined ? {} : {multiple: role.multiple}),
      ...(sharing === undefined ? {} : {sharedWith: sharing}),
      candidates,
      columns: chosen.columns,
      source: chosen.source,
      state: chosen.state,
      ...(chosen.reason === undefined ? {} : {reason: chosen.reason}),
      warnings
    };
  });
}

// ---------------------------------------------------------------------------------------------
// Inputs, limits and shared parameters
// ---------------------------------------------------------------------------------------------

/** Longest string the catalog accepts for a string parameter that states no `maximumLength` itself. */
const DEFAULT_STRING_PARAMETER_LENGTH = 64;

/** Separator of a column list written into one binding parameter (descriptor contract). */
const COLUMN_LIST_SEPARATOR = ',';

type KeplerProfileColumnCheck = {name: unknown; type: unknown};

function describeValueType(value: unknown): string {
  return value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
}

/**
 * Checks the members of each profile that this module reads, so that a malformed profile gives an
 * error that names the function and the member instead of a TypeError further in.
 */
function assertDatasetProfiles(caller: string, datasets: readonly KeplerDatasetProfile[]): void {
  if (!Array.isArray(datasets)) {
    throw new Error(`${caller}: datasets must be an array of dataset profiles`);
  }
  datasets.forEach((dataset, index) => {
    const at = `datasets[${index}]`;
    if (typeof dataset !== 'object' || dataset === null) {
      throw new Error(
        `${caller}: ${at} must be a dataset profile object; got ${describeValueType(dataset)}`
      );
    }
    for (const member of ['datasetId', 'label', 'kind'] as const) {
      const value: unknown = dataset[member];
      if (typeof value !== 'string' || (member !== 'label' && value.length === 0)) {
        throw new Error(
          `${caller}: ${at}.${member} must be a${member === 'label' ? '' : ' non-empty'} string; got ${describeValueType(value)}`
        );
      }
    }
    if (
      typeof dataset.rowCount !== 'number' ||
      !Number.isFinite(dataset.rowCount) ||
      dataset.rowCount < 0
    ) {
      throw new Error(
        `${caller}: ${at}.rowCount must be a finite number of at least 0; got ${String(dataset.rowCount)}`
      );
    }
    if (!Array.isArray(dataset.columns)) {
      throw new Error(
        `${caller}: ${at}.columns must be an array; got ${describeValueType(dataset.columns)}`
      );
    }
    dataset.columns.forEach((column: KeplerProfileColumnCheck, columnIndex: number) => {
      if (typeof column?.name !== 'string' || typeof column?.type !== 'string') {
        throw new Error(
          `${caller}: ${at}.columns[${columnIndex}] must have a string name and a string type`
        );
      }
    });
  });
}

/** First dataset of each id is used; later ones are dropped and reported. */
function removeDuplicateDatasets(datasets: readonly KeplerDatasetProfile[]): {
  datasets: KeplerDatasetProfile[];
  duplicateIds: string[];
} {
  const seen = new Set<string>();
  const duplicateIds: string[] = [];
  const kept: KeplerDatasetProfile[] = [];
  for (const dataset of datasets) {
    if (seen.has(dataset.datasetId)) {
      if (!duplicateIds.includes(dataset.datasetId)) {
        duplicateIds.push(dataset.datasetId);
      }
      continue;
    }
    seen.add(dataset.datasetId);
    kept.push(dataset);
  }
  return {datasets: kept, duplicateIds};
}

function getDescriptorGeometry(
  kind: GPUSpatialAnalysisDataKind
): KeplerOperationGeometry | undefined {
  if (kind === 'points') return 'point';
  if (kind === 'lines' || kind === 'network') return 'line';
  if (kind === 'polygons') return 'polygon';
  return undefined;
}

/**
 * The analysis as matching sees it: where the binding's descriptor lists measurement levels by
 * geometry (a total per area mostly measures area size, so polygons list rates only), those levels
 * replace the role's own list for that slot. The analysis itself is not changed.
 */
function applyDescriptorMeasurements(analysis: KeplerCuratedAnalysis): KeplerCuratedAnalysis {
  const execution = analysis.execution;
  if (execution.status !== 'available') {
    return analysis;
  }
  const descriptor = getKeplerOperationDescriptor(execution.bindingId);
  if (descriptor === undefined) {
    return analysis;
  }
  return {
    ...analysis,
    slots: analysis.slots.map(slot => {
      const geometry = getDescriptorGeometry(slot.kind);
      const input = descriptor.inputs.find(entry => entry.name === execution.inputs[slot.name]);
      if (geometry === undefined || input === undefined) {
        return slot;
      }
      return {
        ...slot,
        fields: slot.fields.map(role => {
          const parameter = execution.fieldParameters[getRoleKey(slot.name, role.role)];
          const levels = input.fields.find(field => field.parameter === parameter)
            ?.measurementByGeometry?.[geometry];
          return levels === undefined ? role : {...role, measurement: levels};
        })
      };
    })
  };
}

/** The binding parameter that receives a role's columns, and how long its text may be. */
function getFieldLimit(
  analysis: KeplerCuratedAnalysis,
  slotName: string,
  roleName: string
): {parameter: string; maximumLength: number} | undefined {
  const execution = analysis.execution;
  if (execution.status !== 'available') {
    return undefined;
  }
  const parameter = execution.fieldParameters[getRoleKey(slotName, roleName)];
  if (parameter === undefined) {
    return undefined;
  }
  const specification = getCatalogBinding(execution.bindingId)?.parameters.find(
    entry => entry.name === parameter
  );
  if (specification === undefined || specification.kind !== 'string') {
    return undefined;
  }
  return {parameter, maximumLength: specification.maximumLength ?? DEFAULT_STRING_PARAMETER_LENGTH};
}

/** Columns a role cannot use at all, with the reason: a name the parameter cannot hold. */
function getBlockedColumns(
  analysis: KeplerCuratedAnalysis,
  slotName: string,
  role: GPUAnalysisFieldRoleSpec,
  profile: KeplerDatasetProfile
): Map<string, string> {
  const blocked = new Map<string, string>();
  const limit = getFieldLimit(analysis, slotName, role.role);
  for (const column of profile.columns) {
    if (role.multiple && column.name.includes(COLUMN_LIST_SEPARATOR)) {
      blocked.set(
        column.name,
        `"${column.name}" contains a comma; ${role.label} is written as a comma-separated list, so the name cannot be used.`
      );
    } else if (limit !== undefined && column.name.length > limit.maximumLength) {
      blocked.set(
        column.name,
        `"${column.name}" has ${column.name.length} characters; ${limit.parameter} holds at most ${limit.maximumLength}.`
      );
    }
  }
  return blocked;
}

/** Why a list of columns cannot be written into the role's parameter, if it cannot. */
function checkColumnListLength(
  analysis: KeplerCuratedAnalysis,
  slotName: string,
  role: GPUAnalysisFieldRoleSpec,
  columns: readonly string[]
): string | undefined {
  const limit = getFieldLimit(analysis, slotName, role.role);
  if (limit === undefined || columns.length === 0) {
    return undefined;
  }
  const length = columns.join(COLUMN_LIST_SEPARATOR).length;
  return length > limit.maximumLength
    ? `${columns.length} ${columns.length === 1 ? 'column name is' : 'column names are'} ${length} characters written out, and ${limit.parameter} holds at most ${limit.maximumLength}; choose fewer or shorter ones.`
    : undefined;
}

type SharedGroupMember = {slot: string; role: string; key: string};

/** Roles of different slots that feed one binding parameter; the binding reads one column name for all. */
type SharedGroup = {parameter: string; members: readonly SharedGroupMember[]};

function getSharedGroups(analysis: KeplerCuratedAnalysis): SharedGroup[] {
  const execution = analysis.execution;
  if (execution.status !== 'available') {
    return [];
  }
  const byParameter = new Map<string, SharedGroupMember[]>();
  for (const slot of analysis.slots) {
    for (const role of slot.fields) {
      const key = getRoleKey(slot.name, role.role);
      const parameter = execution.fieldParameters[key];
      if (parameter !== undefined) {
        byParameter.set(parameter, [
          ...(byParameter.get(parameter) ?? []),
          {slot: slot.name, role: role.role, key}
        ]);
      }
    }
  }
  return [...byParameter.entries()]
    .filter(([, members]) => new Set(members.map(member => member.slot)).size > 1)
    .map(([parameter, members]) => ({parameter, members}));
}

function areColumnTypesCompatible(left: string, right: string): boolean {
  const numeric = (type: string): boolean => type === 'number' || type === 'integer';
  return left === right || (numeric(left) && numeric(right));
}

function isSameList(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

// ---------------------------------------------------------------------------------------------
// Deriving a mapping
// ---------------------------------------------------------------------------------------------

function mergeParameters(
  analysis: KeplerCuratedAnalysis,
  extra: Readonly<Record<string, CatalogParameterValue>> | undefined
): Record<string, CatalogParameterValue> {
  return {...getPresetParameters(analysis), ...extra};
}

type LostMessage = {slot: string; role?: string; message: string};

function deriveMapping(
  caller: string,
  rawAnalysis: KeplerCuratedAnalysis,
  rawDatasets: readonly KeplerDatasetProfile[],
  choices: KeplerCuratedAnalysisMappingChoices,
  parameterValues: Readonly<Record<string, CatalogParameterValue>>,
  explicitParameterNames: readonly string[] = []
): KeplerCuratedAnalysisMapping {
  assertDatasetProfiles(caller, rawDatasets);
  const {datasets, duplicateIds} = removeDuplicateDatasets(rawDatasets);
  const analysis = applyDescriptorMeasurements(rawAnalysis);
  const lostMessages: LostMessage[] = [];
  const slotByName = new Map(analysis.slots.map(slot => [slot.name, slot]));

  // Saved choices that name a slot or role the analysis no longer has are dropped and reported.
  for (const slotName of Object.keys(choices.datasets)) {
    if (!slotByName.has(slotName)) {
      lostMessages.push({
        slot: slotName,
        message: `Saved dataset for "${slotName}" was dropped: ${analysis.name} has no such slot.`
      });
    }
  }
  const roleKeys = new Set(
    analysis.slots.flatMap(slot => slot.fields.map(role => getRoleKey(slot.name, role.role)))
  );
  for (const key of Object.keys(choices.columns)) {
    if (!roleKeys.has(key)) {
      lostMessages.push({
        slot: key.split('.')[0],
        message: `Saved columns for "${key}" were dropped: ${analysis.name} has no such field.`
      });
    }
  }

  const evaluations = analysis.slots.map(slot =>
    evaluateSlotDatasets(analysis, slot, datasets, parameterValues)
  );
  const lostSlots = new Map<string, string>();
  const effectiveDatasets: Record<string, string> = {};
  const pinnedDatasetIds = analysis.slots.map((slot, slotIndex) => {
    const saved = choices.datasets[slot.name];
    if (saved === undefined) {
      return undefined;
    }
    const evaluation = evaluations[slotIndex].find(entry => entry.candidate.datasetId === saved);
    if (evaluation === undefined || evaluation.candidate.fit !== 'fits') {
      const why =
        evaluation === undefined
          ? 'is not loaded'
          : `no longer fits (${evaluation.candidate.reason})`;
      const message = `Saved dataset "${saved}" for ${slot.label} ${why}; a dataset was chosen again.`;
      lostMessages.push({slot: slot.name, message});
      lostSlots.set(slot.name, message);
      return undefined;
    }
    effectiveDatasets[slot.name] = saved;
    return saved;
  });

  const assignments = assignDatasets(analysis.slots, evaluations, pinnedDatasetIds);
  const profileBySlot = new Map<string, KeplerDatasetProfile>();
  analysis.slots.forEach((slot, slotIndex) => {
    const profile = datasets.find(
      dataset => dataset.datasetId === assignments[slotIndex].datasetId
    );
    if (profile !== undefined) {
      profileBySlot.set(slot.name, profile);
    }
  });

  // Per role: unusable names, offered columns and the columns that two slots share.
  const blockedByKey = new Map<string, Map<string, string>>();
  const offeredByKey = new Map<string, Set<string>>();
  for (const slot of analysis.slots) {
    const profile = profileBySlot.get(slot.name);
    if (profile === undefined) {
      continue;
    }
    for (const role of slot.fields) {
      const key = getRoleKey(slot.name, role.role);
      const blocked = getBlockedColumns(analysis, slot.name, role, profile);
      blockedByKey.set(key, blocked);
      offeredByKey.set(
        key,
        new Set(
          rankColumns(role, undefined, profile.columns, [...blocked.keys()]).map(
            column => column.name
          )
        )
      );
    }
  }

  const groups = getSharedGroups(analysis);
  const groupByKey = new Map<string, SharedGroup>();
  const sharedMissing = new Map<string, string>();
  const sharedAllowedByKey = new Map<string, Set<string>>();
  for (const group of groups) {
    group.members.forEach(member => groupByKey.set(member.key, group));
    const memberProfiles = group.members.map(member => profileBySlot.get(member.slot));
    if (memberProfiles.some(profile => profile === undefined)) {
      continue;
    }
    const allowed = new Set<string>();
    const [first] = group.members;
    for (const name of offeredByKey.get(first.key) ?? []) {
      const types = group.members.map(
        member => profileBySlot.get(member.slot)!.columns.find(column => column.name === name)?.type
      );
      const everywhere = group.members.every(
        (member, index) => offeredByKey.get(member.key)?.has(name) && types[index] !== undefined
      );
      if (everywhere && types.every(type => areColumnTypesCompatible(type!, types[0]!))) {
        allowed.add(name);
      }
    }
    for (const member of group.members) {
      sharedAllowedByKey.set(member.key, allowed);
      offeredByKey.set(member.key, allowed);
    }
    if (allowed.size === 0) {
      const described = group.members.map((member, index) => {
        const profile = memberProfiles[index]!;
        return `${profile.label} (${analysis.slots.find(slot => slot.name === member.slot)?.label})`;
      });
      sharedMissing.set(group.parameter, joinWithAnd(described));
    }
  }

  const validateColumns = (
    slotName: string,
    role: GPUAnalysisFieldRoleSpec,
    columns: readonly string[]
  ): string | undefined => {
    const key = getRoleKey(slotName, role.role);
    const offered = offeredByKey.get(key) ?? new Set<string>();
    const missing = columns.filter(column => !offered.has(column));
    const maximum = role.multiple?.maximum ?? 1;
    const minimum = role.multiple ? role.multiple.minimum : 1;
    const countOk =
      columns.length <= maximum &&
      (columns.length >= minimum ||
        (columns.length === 0 && !isRoleRequired(role, parameterValues))) &&
      new Set(columns).size === columns.length;
    const profile = profileBySlot.get(slotName);
    if (missing.length > 0) {
      return `Saved ${missing.length === 1 ? 'column' : 'columns'} ${missing.map(name => `"${name}"`).join(', ')} for ${role.label} ${missing.length === 1 ? 'is' : 'are'} not ${missing.length === 1 ? 'an available column' : 'available columns'} of ${profile?.label}; a column was chosen again.`;
    }
    if (!countOk || columns.length === 0) {
      return `Saved columns for ${role.label} no longer fit the field's limits (a list of distinct columns, ${minimum === maximum ? minimum : `${minimum} to ${maximum}`} of them); a column was chosen again.`;
    }
    const tooLong = checkColumnListLength(analysis, slotName, role, columns);
    return tooLong === undefined ? undefined : `Saved columns for ${role.label}: ${tooLong}`;
  };

  // Saved column choices that still hold, by role key.
  const pinnedByKey = new Map<string, readonly string[]>();
  const lostByKey = new Map<string, string>();
  const noteLost = (slotName: string, roleName: string, message: string): void => {
    lostMessages.push({slot: slotName, role: roleName, message});
    lostByKey.set(getRoleKey(slotName, roleName), message);
  };
  for (const group of groups) {
    let winner: readonly string[] | undefined;
    for (const member of group.members) {
      const saved = choices.columns[member.key];
      const slotSpec = slotByName.get(member.slot)!;
      const role = slotSpec.fields.find(field => field.role === member.role)!;
      if (saved === undefined) {
        continue;
      }
      if (profileBySlot.get(member.slot) === undefined) {
        noteLost(
          member.slot,
          member.role,
          `Saved columns for ${role.label} were dropped: ${slotSpec.label} has no dataset.`
        );
        continue;
      }
      const problem = validateColumns(member.slot, role, saved);
      if (problem !== undefined) {
        noteLost(member.slot, member.role, problem);
      } else if (winner === undefined) {
        winner = saved;
      } else if (!isSameList(winner, saved)) {
        noteLost(
          member.slot,
          member.role,
          `Saved columns for ${role.label} differ from the other slot that feeds ${group.parameter}; both must name the same columns, so the first was kept.`
        );
      }
    }
    if (winner !== undefined) {
      group.members.forEach(member => pinnedByKey.set(member.key, winner!));
    }
  }
  for (const slot of analysis.slots) {
    for (const role of slot.fields) {
      const key = getRoleKey(slot.name, role.role);
      const saved = choices.columns[key];
      if (saved === undefined || groupByKey.has(key)) {
        continue;
      }
      if (profileBySlot.get(slot.name) === undefined) {
        noteLost(
          slot.name,
          role.role,
          `Saved columns for ${role.label} were dropped: ${slot.label} has no dataset.`
        );
        continue;
      }
      const problem = validateColumns(slot.name, role, saved);
      if (problem === undefined) {
        pinnedByKey.set(key, saved);
      } else {
        noteLost(slot.name, role.role, problem);
      }
    }
  }

  const leaderChoice = new Map<string, RoleChosen>();
  const slots: KeplerCuratedMappingSlot[] = analysis.slots.map((slot, slotIndex) => {
    const assignment = assignments[slotIndex];
    const lostSlot = lostSlots.get(slot.name);
    const profile = profileBySlot.get(slot.name);
    const evaluation = evaluations[slotIndex].find(
      entry => entry.candidate.datasetId === assignment.datasetId
    );
    let state = assignment.state;
    let reason: string | undefined;
    if (assignment.source === 'automatic' && state === 'choose') {
      reason = `Several datasets fit ${slot.label} equally well; ${assignment.datasetId} was taken first.`;
    }
    // A declared mismatch is the user's own statement about the data; it needs a decision.
    if (
      state === 'settled' &&
      assignment.source === 'automatic' &&
      evaluation?.declaredMismatch === true
    ) {
      state = 'choose';
      reason = evaluation.candidate.warning;
    }
    if (lostSlot !== undefined && assignment.datasetId !== undefined) {
      state = 'choose';
      reason = lostSlot;
    }

    let roles: KeplerCuratedMappingRole[];
    if (profile === undefined) {
      roles = slot.fields.map(role => ({
        role: role.role,
        label: role.label,
        required: isRoleRequired(role, parameterValues),
        ...(role.multiple === undefined ? {} : {multiple: role.multiple}),
        candidates: [],
        columns: [],
        source: 'none',
        state: 'empty',
        warnings: []
      }));
    } else {
      const pinned = new Map<string, readonly string[]>();
      const lost = new Map<string, string>();
      const blocked = new Map<string, ReadonlyMap<string, string>>();
      const notShared = new Map<string, ReadonlySet<string>>();
      const forced = new Map<string, RoleChosen>();
      const sharedWith = new Map<string, readonly string[]>();
      for (const role of slot.fields) {
        const key = getRoleKey(slot.name, role.role);
        const pin = pinnedByKey.get(key);
        if (pin !== undefined) {
          pinned.set(role.role, pin);
        }
        const lostText = lostByKey.get(key);
        if (lostText !== undefined) {
          lost.set(role.role, lostText);
        }
        blocked.set(role.role, blockedByKey.get(key) ?? new Map());
        const allowed = sharedAllowedByKey.get(key);
        if (allowed !== undefined) {
          notShared.set(
            role.role,
            new Set(profile.columns.map(column => column.name).filter(name => !allowed.has(name)))
          );
        }
        const group = groupByKey.get(key);
        if (group !== undefined) {
          sharedWith.set(
            role.role,
            group.members.filter(member => member.key !== key).map(member => member.key)
          );
          const leader = leaderChoice.get(group.parameter);
          if (leader !== undefined && group.members[0].key !== key) {
            forced.set(role.role, leader);
          }
        }
      }
      const input: SlotRoleInput = {
        analysis,
        slot,
        profile,
        parameterValues,
        pinned,
        lost,
        blocked,
        notShared,
        forced,
        sharedWith
      };
      const chosenByRole = deriveRoleChoices(input);
      for (const group of groups) {
        if (group.members[0].slot === slot.name) {
          const chosen = chosenByRole.get(group.members[0].role);
          if (chosen !== undefined && chosen.columns.length > 0) {
            leaderChoice.set(group.parameter, chosen);
          }
        }
      }
      roles = buildRoles(input, chosenByRole);
    }
    return {
      slot: slot.name,
      label: slot.label,
      kind: slot.kind,
      candidates: evaluations[slotIndex].map(entry => entry.candidate),
      ...(assignment.datasetId === undefined ? {} : {datasetId: assignment.datasetId}),
      source: assignment.source,
      state,
      conflict: assignment.conflict,
      ...(reason === undefined ? {} : {reason}),
      ...(evaluation?.candidate.warning === undefined
        ? {}
        : {warning: evaluation.candidate.warning}),
      ...(profile?.coordinates === undefined
        ? {}
        : {
            coordinates: {
              model: profile.coordinates.model,
              source: profile.coordinates.source,
              ...(profile.coordinates.reason === undefined
                ? {}
                : {reason: profile.coordinates.reason})
            }
          }),
      roles
    };
  });

  const problems = collectProblems(
    analysis,
    slots,
    lostMessages,
    duplicateIds,
    sharedMissing,
    groups
  );
  const requiredRoles = slots.flatMap(slot => slot.roles.filter(role => role.required));
  const progress = {
    mapped:
      slots.filter(slot => slot.datasetId !== undefined && !slot.conflict).length +
      requiredRoles.filter(role => role.columns.length > 0).length,
    required: slots.length + requiredRoles.length
  };
  const complete = !problems.some(problem => problem.severity === 'blocks');
  const ready =
    complete &&
    slots.every(slot => slot.state !== 'choose') &&
    requiredRoles.every(role => role.state !== 'choose');

  // Only choices that still hold are kept, so a saved mapping never carries dead keys.
  const effectiveColumns: Record<string, readonly string[]> = {};
  for (const [key, columns] of pinnedByKey) {
    effectiveColumns[key] = columns;
  }
  return {
    analysisId: rawAnalysis.id,
    parameterValues,
    explicitParameterNames,
    slots,
    progress,
    complete,
    ready,
    problems,
    choices: {analysisId: rawAnalysis.id, datasets: effectiveDatasets, columns: effectiveColumns}
  };
}

function collectProblems(
  analysis: KeplerCuratedAnalysis,
  slots: readonly KeplerCuratedMappingSlot[],
  lostMessages: readonly LostMessage[],
  duplicateIds: readonly string[],
  sharedMissing: ReadonlyMap<string, string>,
  groups: readonly SharedGroup[]
): KeplerCuratedMappingProblem[] {
  const problems: KeplerCuratedMappingProblem[] = [];
  for (const datasetId of duplicateIds) {
    problems.push({
      kind: 'dataset-duplicate',
      severity: 'check',
      acknowledged: false,
      message: `Dataset id "${datasetId}" is loaded more than once; the first one was used and the rest ignored.`
    });
  }
  for (const lost of lostMessages) {
    problems.push({
      kind: 'saved-choice-lost',
      severity: 'check',
      acknowledged: false,
      slot: lost.slot,
      ...(lost.role === undefined ? {} : {role: lost.role}),
      message: lost.message
    });
  }
  const sharedMissingKeys = new Set<string>();
  for (const group of groups) {
    const described = sharedMissing.get(group.parameter);
    if (described === undefined) {
      continue;
    }
    const required = group.members.some(
      member =>
        slots.find(slot => slot.slot === member.slot)?.roles.find(role => role.role === member.role)
          ?.required
    );
    if (!required) {
      continue;
    }
    group.members.forEach(member => sharedMissingKeys.add(member.key));
    problems.push({
      kind: 'shared-column-missing',
      severity: 'blocks',
      acknowledged: false,
      slot: group.members[0].slot,
      role: group.members[0].role,
      message: `${described} are read with one column name (${group.parameter}), but they have no column of that name and a compatible type in common. Rename a column or choose other datasets.`
    });
  }
  for (const slot of slots) {
    if (slot.datasetId === undefined) {
      const fitting = slot.candidates.filter(candidate => candidate.fit === 'fits').length;
      problems.push({
        kind: 'slot-empty',
        severity: 'blocks',
        acknowledged: false,
        slot: slot.slot,
        message:
          fitting === 0
            ? `${slot.label}: no loaded dataset fits (it needs ${describeKind(slot.kind)}). ${analysis.name} cannot run without it.`
            : `${slot.label}: no dataset chosen. ${analysis.name} cannot run without it.`
      });
    } else if (slot.conflict) {
      problems.push({
        kind: 'slot-conflict',
        severity: 'blocks',
        acknowledged: false,
        slot: slot.slot,
        message: `${slot.label}: ${slot.datasetId} is the only dataset that fits and it already fills an earlier slot of the same kind. Load another dataset, or the two inputs would be the same data.`
      });
    } else if (slot.state === 'choose' && !hasLostMessage(lostMessages, slot.slot)) {
      problems.push({
        kind: 'needs-choice',
        severity: 'check',
        acknowledged: false,
        slot: slot.slot,
        message: `${slot.label}: ${slot.reason ?? 'check the dataset.'}`
      });
    } else if (slot.warning !== undefined) {
      problems.push({
        kind: 'dataset-warning',
        severity: 'check',
        acknowledged: slot.source === 'user',
        slot: slot.slot,
        message: `${slot.label}: ${slot.warning}`
      });
    }
    if (slot.coordinates?.source === 'guess') {
      problems.push({
        kind: 'coordinate-guess',
        severity: 'check',
        acknowledged: false,
        slot: slot.slot,
        message: `${slot.label}: the coordinates are guessed to be ${slot.coordinates.model}${slot.coordinates.reason === undefined ? '' : ` (${slot.coordinates.reason})`}. Declare the coordinate model if that is wrong; distances depend on it.`
      });
    }
    for (const role of slot.roles) {
      const spec = getRoleSpec(analysis, slot.slot, role.role);
      const key = getRoleKey(slot.slot, role.role);
      if (role.required && role.columns.length === 0) {
        if (sharedMissingKeys.has(key)) {
          continue;
        }
        problems.push({
          kind: 'role-empty',
          severity: 'blocks',
          acknowledged: false,
          slot: slot.slot,
          role: role.role,
          message:
            slot.datasetId === undefined
              ? `${role.label}: needs a column once ${slot.label} has a dataset. ${analysis.name} cannot run without it.`
              : `${role.label}: no ${describeColumnTypes(spec)} column of ${slot.datasetId} is left for it. ${analysis.name} cannot run without it.`
        });
        continue;
      }
      const tooLong = checkColumnListLength(analysis, slot.slot, spec, role.columns);
      if (tooLong !== undefined) {
        problems.push({
          kind: 'field-too-long',
          severity: 'blocks',
          acknowledged: false,
          slot: slot.slot,
          role: role.role,
          message: `${role.label}: ${tooLong}`
        });
      }
      if (
        role.required &&
        role.state === 'choose' &&
        !hasLostMessage(lostMessages, slot.slot, role.role)
      ) {
        problems.push({
          kind: 'needs-choice',
          severity: 'check',
          acknowledged: false,
          slot: slot.slot,
          role: role.role,
          message: `${role.label}: ${role.reason ?? 'check the column.'}`
        });
      } else if (role.source === 'user' && role.warnings.length > 0) {
        // The user chose this column knowing the warning; `ready` ignores it.
        problems.push({
          kind: 'column-warning',
          severity: 'check',
          acknowledged: true,
          slot: slot.slot,
          role: role.role,
          message: `${role.label}: ${role.warnings.join(' ')}`
        });
      }
    }
  }
  return problems;
}

/** A lost saved choice already explains why the slot or role is flagged; do not say it twice. */
function hasLostMessage(
  lostMessages: readonly {slot: string; role?: string}[],
  slotName: string,
  roleName?: string
): boolean {
  return lostMessages.some(lost => lost.slot === slotName && lost.role === roleName);
}

function getRoleSpec(
  analysis: KeplerCuratedAnalysis,
  slotName: string,
  roleName: string
): GPUAnalysisFieldRoleSpec {
  return analysis.slots
    .find(slot => slot.name === slotName)!
    .fields.find(role => role.role === roleName)!;
}

function createEmptyChoices(analysis: KeplerCuratedAnalysis): KeplerCuratedAnalysisMappingChoices {
  return {analysisId: analysis.id, datasets: {}, columns: {}};
}

// ---------------------------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------------------------

/**
 * Pre-fills the mapping of one analysis on the loaded datasets. Never throws on the data's shape:
 * a dataset that does not fit is listed with its reason, a repeated dataset id is ignored after
 * the first and reported, and what is missing is reported in `problems`. It throws only for a
 * malformed dataset profile (a member this module reads is missing or has the wrong type).
 */
export function createKeplerCuratedAnalysisMapping(
  analysis: KeplerCuratedAnalysis,
  datasets: readonly KeplerDatasetProfile[],
  options?: KeplerCuratedAnalysisMappingOptions
): KeplerCuratedAnalysisMapping {
  return deriveMapping(
    'createKeplerCuratedAnalysisMapping',
    analysis,
    datasets,
    createEmptyChoices(analysis),
    mergeParameters(analysis, options?.parameters),
    Object.keys(options?.parameters ?? {})
  );
}

function describeColumnLimits(minimum: number, maximum: number): string {
  return minimum === maximum ? String(minimum) : `${minimum} to ${maximum}`;
}

/**
 * Applies one user choice and derives the rest again. Choosing a dataset drops the column choices
 * of that slot (and of the roles that share a binding parameter with them) unless the dataset is
 * the one already chosen. An empty column list on an optional role removes the user's choice.
 * Throws when the choice cannot be made: unknown slot, role or dataset, a dataset that does not
 * fit, `columns` that is not an array, a column that is not offered for the role (including a
 * name with a comma in a list role or one too long for the binding parameter), a duplicate
 * column, a column count outside the role's limits, or a list too long to write into the
 * parameter. A role that shares its binding parameter with a role of another slot is set in both.
 */
export function updateKeplerCuratedAnalysisMapping(
  mapping: KeplerCuratedAnalysisMapping,
  change: KeplerCuratedAnalysisMappingChange,
  analysis: KeplerCuratedAnalysis,
  datasets: readonly KeplerDatasetProfile[],
  options?: KeplerCuratedAnalysisMappingOptions
): KeplerCuratedAnalysisMapping {
  const caller = 'updateKeplerCuratedAnalysisMapping';
  if (mapping.analysisId !== analysis.id) {
    throw new Error(
      `${caller}: mapping is for analysis "${mapping.analysisId}", not "${analysis.id}"`
    );
  }
  const slotSpec = analysis.slots.find(slot => slot.name === change.slot);
  if (slotSpec === undefined) {
    throw new Error(
      `${caller}: slot "${change.slot}" is not a slot of "${analysis.id}"; accepted: ${analysis.slots.map(slot => slot.name).join(', ')}`
    );
  }
  const parameterValues =
    options?.parameters === undefined
      ? mapping.parameterValues
      : mergeParameters(analysis, options.parameters);
  const explicitParameterNames =
    options?.parameters === undefined
      ? (mapping.explicitParameterNames ?? [])
      : Object.keys(options.parameters);
  const current = deriveMapping(
    caller,
    analysis,
    datasets,
    mapping.choices,
    parameterValues,
    explicitParameterNames
  );
  const currentSlot = current.slots.find(
    slot => slot.slot === change.slot
  ) as KeplerCuratedMappingSlot;
  const groups = getSharedGroups(analysis);

  if (!('role' in change)) {
    const candidate = currentSlot.candidates.find(entry => entry.datasetId === change.datasetId);
    if (candidate === undefined) {
      throw new Error(
        `${caller}: slot "${change.slot}": dataset "${change.datasetId}" is not loaded; loaded: ${datasets.map(dataset => dataset.datasetId).join(', ') || 'none'}`
      );
    }
    if (candidate.fit !== 'fits') {
      throw new Error(
        `${caller}: slot "${change.slot}": dataset "${change.datasetId}" cannot be used (${candidate.fit}): ${candidate.reason}`
      );
    }
    const keepColumns = currentSlot.datasetId === change.datasetId;
    const droppedKeys = new Set<string>();
    if (!keepColumns) {
      for (const role of slotSpec.fields) {
        const key = getRoleKey(change.slot, role.role);
        droppedKeys.add(key);
        groups
          .filter(group => group.members.some(member => member.key === key))
          .forEach(group => group.members.forEach(member => droppedKeys.add(member.key)));
      }
    }
    return deriveMapping(
      caller,
      analysis,
      datasets,
      {
        analysisId: analysis.id,
        datasets: {...mapping.choices.datasets, [change.slot]: change.datasetId},
        columns: Object.fromEntries(
          Object.entries(mapping.choices.columns).filter(([key]) => !droppedKeys.has(key))
        )
      },
      parameterValues,
      explicitParameterNames
    );
  }

  const roleKey = getRoleKey(change.slot, change.role);
  const roleSpec = slotSpec.fields.find(role => role.role === change.role);
  if (roleSpec === undefined) {
    throw new Error(
      `${caller}: role "${change.role}" is not a role of slot "${change.slot}"; accepted: ${slotSpec.fields.map(role => role.role).join(', ') || 'none'}`
    );
  }
  const currentRole = currentSlot.roles.find(
    role => role.role === change.role
  ) as KeplerCuratedMappingRole;
  if (currentSlot.datasetId === undefined) {
    throw new Error(
      `${caller}: slot "${change.slot}" has no dataset; choose a dataset before choosing columns for "${change.role}"`
    );
  }
  if (!Array.isArray(change.columns) || change.columns.some(column => typeof column !== 'string')) {
    throw new Error(
      `${caller}: role "${roleKey}": columns must be an array of column names; got ${describeValueType(change.columns)}`
    );
  }
  const columns = [...change.columns];
  const minimum = roleSpec.multiple ? roleSpec.multiple.minimum : 1;
  const maximum = roleSpec.multiple?.maximum ?? 1;
  const required = isRoleRequired(roleSpec, parameterValues);
  if (columns.length === 0 && required) {
    throw new Error(
      `${caller}: role "${roleKey}" is required with the current parameters; choose ${describeColumnLimits(minimum, maximum)} column(s)`
    );
  }
  if (columns.length > maximum || (columns.length > 0 && columns.length < minimum)) {
    throw new Error(
      `${caller}: role "${roleKey}" takes ${describeColumnLimits(minimum, maximum)} column(s); got ${columns.length}`
    );
  }
  if (new Set(columns).size !== columns.length) {
    throw new Error(`${caller}: role "${roleKey}" lists the same column twice`);
  }
  const profile = datasets.find(dataset => dataset.datasetId === currentSlot.datasetId);
  for (const column of columns) {
    const candidate = currentRole.candidates.find(entry => entry.name === column);
    if (candidate?.unavailable !== undefined) {
      throw new Error(`${caller}: role "${roleKey}": ${candidate.unavailable}`);
    }
    if (candidate !== undefined) {
      continue;
    }
    const profileColumn = profile?.columns.find(entry => entry.name === column);
    const shared = currentRole.sharedWith !== undefined && profileColumn !== undefined;
    throw new Error(
      profileColumn === undefined
        ? `${caller}: role "${roleKey}": "${column}" is not a column of ${currentSlot.datasetId}`
        : shared
          ? `${caller}: role "${roleKey}": "${column}" is not offered because it is not shared with ${currentRole.sharedWith?.join(', ')} (same name, compatible type)`
          : `${caller}: role "${roleKey}": "${column}" is ${profileColumn.type}; ${roleSpec.label} takes ${describeColumnTypes(roleSpec)}`
    );
  }
  const tooLong = checkColumnListLength(analysis, change.slot, roleSpec, columns);
  if (tooLong !== undefined) {
    throw new Error(`${caller}: role "${roleKey}": ${tooLong}`);
  }

  const keys = new Set([roleKey]);
  groups
    .filter(group => group.members.some(member => member.key === roleKey))
    .forEach(group => group.members.forEach(member => keys.add(member.key)));
  const nextColumns: Record<string, readonly string[]> = Object.fromEntries(
    Object.entries(mapping.choices.columns).filter(([key]) => !keys.has(key))
  );
  if (columns.length > 0) {
    keys.forEach(key => {
      nextColumns[key] = columns;
    });
  }
  return deriveMapping(
    caller,
    analysis,
    datasets,
    {analysisId: analysis.id, datasets: mapping.choices.datasets, columns: nextColumns},
    parameterValues,
    explicitParameterNames
  );
}

/**
 * Derives a mapping again from saved choices. A saved dataset or column that is gone, or no longer
 * fits, a key for a slot or role the analysis no longer has, and a duplicated column are reported
 * in `problems` (`saved-choice-lost`) and dropped; what they stood for is chosen again
 * automatically and flagged `choose`. Nothing throws on data. Throws only when the choices belong
 * to another analysis.
 */
export function replayKeplerCuratedAnalysisMapping(
  analysis: KeplerCuratedAnalysis,
  choices: KeplerCuratedAnalysisMappingChoices,
  datasets: readonly KeplerDatasetProfile[],
  options?: KeplerCuratedAnalysisMappingOptions
): KeplerCuratedAnalysisMapping {
  const caller = 'replayKeplerCuratedAnalysisMapping';
  if (choices.analysisId !== analysis.id) {
    throw new Error(
      `${caller}: choices are for analysis "${choices.analysisId}", not "${analysis.id}"`
    );
  }
  return deriveMapping(
    caller,
    analysis,
    datasets,
    choices,
    mergeParameters(analysis, options?.parameters),
    Object.keys(options?.parameters ?? {})
  );
}

/**
 * Builds the catalog run request of a complete mapping of an available analysis. Order of
 * parameters: the effective settings retained by the mapping, then mapped fields, then explicit overrides.
 *
 * - A role with several columns is written as a comma-separated list, as the descriptor contract
 *   and the defaults module do. A name with a comma, or a list longer than the parameter's
 *   `maximumLength`, throws here (the mapping already reports both before it is complete).
 * - Roles of two slots that feed one parameter must hold the same columns.
 * - When a slot's dataset has planar coordinates and the binding has a `coordinateSystem`
 *   parameter, the preset for it is left out so that the defaults module derives it from the
 *   profile; explicit mapping settings and request overrides still win.
 * - Parameters none of these set stay absent: the defaults module fills those that follow the
 *   data, and the catalog's own defaults fill the rest.
 * - Equations of the seemingly-unrelated-regressions binding use a different separator; no
 *   curated analysis uses it, so this function does not write that form.
 */
export function buildKeplerCuratedAnalysisRequest(
  analysis: KeplerCuratedAnalysis,
  mapping: KeplerCuratedAnalysisMapping,
  options?: {parameterOverrides?: Readonly<Record<string, CatalogParameterValue>>}
): KeplerCuratedAnalysisRequest {
  const caller = 'buildKeplerCuratedAnalysisRequest';
  if (mapping.analysisId !== analysis.id) {
    throw new Error(
      `${caller}: mapping is for analysis "${mapping.analysisId}", not "${analysis.id}"`
    );
  }
  const execution = analysis.execution;
  if (execution.status !== 'available') {
    throw new Error(
      `${caller}: analysis "${analysis.id}" cannot run yet (needs-binding): ${execution.reason}`
    );
  }
  const overrides = options?.parameterOverrides ?? {};
  const parameterValues = {...mapping.parameterValues, ...overrides};

  // An override can make a conditional role required; the mapping was derived without it.
  const missing: string[] = [];
  for (const slot of analysis.slots) {
    const mappedSlot = mapping.slots.find(entry => entry.slot === slot.name);
    for (const role of slot.fields) {
      const mappedRole = mappedSlot?.roles.find(entry => entry.role === role.role);
      if (mappedRole && mappedRole.required !== isRoleRequired(role, mapping.parameterValues)) {
        throw new Error(
          `${caller}: mapping.parameterValues is inconsistent with role "${slot.name}.${role.role}"; use the parameter values with which the mapping was derived or update the mapping`
        );
      }
      if (isRoleRequired(role, parameterValues) && (mappedRole?.columns.length ?? 0) === 0) {
        missing.push(`${slot.label}: ${role.label} has no column`);
      }
    }
    if (mappedSlot?.datasetId === undefined || mappedSlot.conflict) {
      missing.push(`${slot.label} has no dataset of its own`);
    }
  }
  if (!mapping.complete || missing.length > 0) {
    const blocking = mapping.problems
      .filter(problem => problem.severity === 'blocks')
      .map(problem => problem.message);
    throw new Error(
      `${caller}: the mapping of "${analysis.id}" is not complete: ${[...new Set([...blocking, ...missing])].join(' ')}`
    );
  }

  const fieldParameterNames = new Set(Object.values(execution.fieldParameters));
  for (const name of Object.keys(overrides)) {
    if (fieldParameterNames.has(name)) {
      throw new Error(
        `${caller}: parameterOverrides sets "${name}", which the mapping fills with a column; change the mapping instead`
      );
    }
  }

  const inputDatasetIds: Record<string, string> = {};
  for (const slot of analysis.slots) {
    const inputName = execution.inputs[slot.name];
    const datasetId = mapping.slots.find(entry => entry.slot === slot.name)?.datasetId as string;
    if (inputName === undefined) {
      throw new Error(
        `${caller}: analysis "${analysis.id}" lists slot "${slot.name}" but no binding input for it in execution.inputs`
      );
    }
    inputDatasetIds[inputName] = datasetId;
  }

  const parameters: Record<string, CatalogParameterValue> = {...mapping.parameterValues};
  const columnsByParameter = new Map<string, {key: string; columns: readonly string[]}>();
  for (const [roleKey, parameterName] of Object.entries(execution.fieldParameters)) {
    const separator = roleKey.indexOf('.');
    const slotName = roleKey.slice(0, separator);
    const roleName = roleKey.slice(separator + 1);
    const columns =
      mapping.slots
        .find(entry => entry.slot === slotName)
        ?.roles.find(entry => entry.role === roleName)?.columns ?? [];
    // Field settings belong to the mapped roles, including an explicitly empty optional role.
    if (!columnsByParameter.has(parameterName)) delete parameters[parameterName];
    if (columns.length === 0) {
      continue;
    }
    const earlier = columnsByParameter.get(parameterName);
    if (earlier !== undefined) {
      if (!isSameList(earlier.columns, columns)) {
        throw new Error(
          `${caller}: roles "${earlier.key}" and "${roleKey}" both feed parameter "${parameterName}" but hold different columns (${earlier.columns.join(', ')} and ${columns.join(', ')})`
        );
      }
      continue;
    }
    columnsByParameter.set(parameterName, {key: roleKey, columns});
    const role = analysis.slots
      .find(slot => slot.name === slotName)
      ?.fields.find(field => field.role === roleName);
    if (role?.multiple && columns.some(column => column.includes(COLUMN_LIST_SEPARATOR))) {
      throw new Error(
        `${caller}: role "${roleKey}": a column name contains a comma, which separates the list written into "${parameterName}"`
      );
    }
    const tooLong =
      role === undefined ? undefined : checkColumnListLength(analysis, slotName, role, columns);
    if (tooLong !== undefined) {
      throw new Error(`${caller}: role "${roleKey}": ${tooLong}`);
    }
    parameters[parameterName] = columns.join(COLUMN_LIST_SEPARATOR);
  }

  const hasPlanarSlot = mapping.slots.some(slot => slot.coordinates?.model === 'planar');
  if (
    hasPlanarSlot &&
    !('coordinateSystem' in overrides) &&
    !mapping.explicitParameterNames?.includes('coordinateSystem') &&
    getCatalogBinding(execution.bindingId)?.parameters.some(
      parameter => parameter.name === 'coordinateSystem'
    )
  ) {
    delete parameters['coordinateSystem'];
  }
  Object.assign(parameters, overrides);
  return {bindingId: execution.bindingId, inputDatasetIds, parameters};
}

const STATUS_ORDER: Readonly<Record<KeplerCuratedAnalysisAvailability['status'], number>> = {
  ready: 0,
  'needs-mapping': 1,
  'needs-data': 2,
  'needs-binding': 3
};

/**
 * Says where every analysis stands for the loaded datasets. `ready`: complete and nothing to
 * confirm (see `KeplerCuratedAnalysisMapping.ready`). `needs-mapping`: every slot has a fitting
 * dataset of its own but something is unmapped, ambiguous, or contradicts what the user declared
 * about the data. `needs-data`: some slot has no fitting dataset, or two slots of one kind share
 * the only one. `needs-binding`: the analysis cannot run yet. Order: status (in that order), then
 * wave, then position in `analyses`.
 */
export function listKeplerCuratedAnalysesForDatasets(
  analyses: readonly KeplerCuratedAnalysis[],
  datasets: readonly KeplerDatasetProfile[]
): readonly KeplerCuratedAnalysisAvailability[] {
  const entries = analyses.map((analysis, index) => {
    if (analysis.execution.status === 'needs-binding') {
      return {
        index,
        wave: analysis.wave,
        availability: {
          analysisId: analysis.id,
          status: 'needs-binding' as const,
          reason: analysis.execution.reason
        } satisfies KeplerCuratedAnalysisAvailability
      };
    }
    const mapping = createKeplerCuratedAnalysisMapping(analysis, datasets);
    return {
      index,
      wave: analysis.wave,
      availability: describeAvailability(analysis, mapping, datasets)
    };
  });
  return entries
    .sort(
      (left, right) =>
        STATUS_ORDER[left.availability.status] - STATUS_ORDER[right.availability.status] ||
        left.wave - right.wave ||
        left.index - right.index
    )
    .map(entry => entry.availability);
}

function describeAvailability(
  analysis: KeplerCuratedAnalysis,
  mapping: KeplerCuratedAnalysisMapping,
  datasets: readonly KeplerDatasetProfile[]
): KeplerCuratedAnalysisAvailability {
  const base = {analysisId: analysis.id, mapping};
  const slotWithoutFit = mapping.slots.find(
    slot => !slot.candidates.some(candidate => candidate.fit === 'fits')
  );
  if (slotWithoutFit !== undefined) {
    const sameKindMissing = slotWithoutFit.candidates.filter(
      candidate => candidate.fit === 'missing-columns'
    );
    const reason =
      datasets.length === 0
        ? `No dataset is loaded; ${slotWithoutFit.label} needs ${describeKind(slotWithoutFit.kind)}.`
        : sameKindMissing.length > 0
          ? `${slotWithoutFit.label} needs ${describeKind(slotWithoutFit.kind)} with more columns: ${sameKindMissing[0].reason}`
          : `${slotWithoutFit.label} needs ${describeKind(slotWithoutFit.kind)}; none of the ${datasets.length} loaded ${datasets.length === 1 ? 'dataset holds' : 'datasets hold'} it.`;
    return {...base, status: 'needs-data', reason};
  }
  const conflict = mapping.slots.find(slot => slot.conflict);
  if (conflict !== undefined) {
    const sameKind = mapping.slots
      .filter(slot => slot.kind === conflict.kind)
      .map(slot => slot.label);
    return {
      ...base,
      status: 'needs-data',
      reason: `${joinWithAnd(sameKind)} each need ${describeKind(conflict.kind)}, but only one loaded dataset fits.`
    };
  }
  const sharedMissing = mapping.problems.find(problem => problem.kind === 'shared-column-missing');
  if (sharedMissing !== undefined) {
    return {...base, status: 'needs-data', reason: sharedMissing.message};
  }
  if (mapping.ready) {
    const summary = mapping.slots.map(slot => `${slot.label}: ${slot.datasetId}`).join('; ');
    return {...base, status: 'ready', reason: `Every field is mapped (${summary}).`};
  }
  const first =
    mapping.problems.find(
      problem =>
        !problem.acknowledged &&
        problem.kind !== 'saved-choice-lost' &&
        problem.kind !== 'coordinate-guess' &&
        problem.kind !== 'dataset-duplicate'
    ) ?? mapping.problems[0];
  const count = mapping.progress;
  return {
    ...base,
    status: 'needs-mapping',
    reason: `${count.mapped} of ${count.required} mapped${first === undefined ? '' : `; ${first.message}`}`
  };
}

// ---------------------------------------------------------------------------------------------
// Agreement with the descriptor
// ---------------------------------------------------------------------------------------------

function describeLevels(levels: readonly string[] | undefined): string {
  return levels === undefined ? 'any level' : levels.join(', ');
}

function isSameSet(
  left: readonly string[] | undefined,
  right: readonly string[] | undefined
): boolean {
  if (left === undefined || right === undefined) {
    return left === right;
  }
  return left.length === right.length && left.every(value => right.includes(value));
}

/**
 * Compares an available analysis with the descriptor of its binding and returns every
 * disagreement as a sentence; an empty list means they agree. Compared: the analysis's question
 * against the descriptor's questions; each slot's input, kind and row semantics (same set; none
 * means any); each role's binding parameter and role id, accepted types, measurement levels (the
 * slot's must be a subset of the descriptor's for the slot's geometry, honouring
 * `measurementByGeometry`; a slot that lists none while the descriptor does is reported),
 * required flag and `multiple`; descriptor fields and inputs the analysis does not offer; and
 * exposed parameters (each must be `basic` and active under the presets; `basic` active ones left unexposed are listed, except `coordinateSystem`, which follows the data's profile). An analysis that needs a binding returns an empty list: there is nothing to compare.
 */
export function validateKeplerCuratedAnalysis(analysis: KeplerCuratedAnalysis): string[] {
  const execution = analysis.execution;
  if (execution.status !== 'available') {
    return [];
  }
  const id = analysis.id;
  const descriptor = getKeplerOperationDescriptor(execution.bindingId);
  if (descriptor === undefined) {
    return [`${id}: binding "${execution.bindingId}" has no descriptor to compare with.`];
  }
  const sentences: string[] = [];
  if (!descriptor.questions.includes(analysis.question)) {
    sentences.push(
      `${id}: question "${analysis.question}" is not one the descriptor lists (${descriptor.questions.join(', ')}).`
    );
  }
  const coveredInputs = new Set<string>();
  const coveredParameters = new Set<string>();
  for (const slot of analysis.slots) {
    const inputName = execution.inputs[slot.name];
    const input = descriptor.inputs.find(entry => entry.name === inputName);
    if (input === undefined) {
      sentences.push(
        `${id}: slot "${slot.name}" maps to binding input "${inputName}", which the descriptor does not have.`
      );
      continue;
    }
    coveredInputs.add(input.name);
    if (input.kind !== slot.kind) {
      sentences.push(
        `${id}: slot "${slot.name}" is ${slot.kind} but the descriptor input "${input.name}" is ${input.kind}.`
      );
    }
    if (!isSameSet(slot.rowSemantics, input.rowSemantics)) {
      sentences.push(
        `${id}: slot "${slot.name}" expects ${describeLevels(slot.rowSemantics).replace('any level', 'any row semantics')} rows but the descriptor expects ${describeLevels(input.rowSemantics).replace('any level', 'any row semantics')}.`
      );
    }
    const geometry = getDescriptorGeometry(slot.kind);
    for (const role of slot.fields) {
      const roleKey = getRoleKey(slot.name, role.role);
      const parameter = execution.fieldParameters[roleKey];
      if (parameter === undefined) {
        sentences.push(
          `${id}: role "${roleKey}" has no entry in execution.fieldParameters, so no column reaches the binding.`
        );
        continue;
      }
      coveredParameters.add(`${input.name}.${parameter}`);
      const field = input.fields.find(entry => entry.parameter === parameter);
      if (field === undefined) {
        sentences.push(
          `${id}: role "${roleKey}" feeds "${parameter}", which is not a field of descriptor input "${input.name}".`
        );
        continue;
      }
      if (field.role !== role.role) {
        sentences.push(
          `${id}: role "${roleKey}" is called "${field.role}" in the descriptor (same parameter "${parameter}").`
        );
      }
      if (!isSameSet(role.accepts, field.accepts)) {
        sentences.push(
          `${id}: role "${roleKey}" accepts ${role.accepts.join(', ')} but the descriptor accepts ${field.accepts.join(', ')}.`
        );
      }
      const descriptorLevels =
        (geometry === undefined ? undefined : field.measurementByGeometry?.[geometry]) ??
        field.measurement;
      if (descriptorLevels !== undefined) {
        const extra = (role.measurement ?? ['(any level)']).filter(
          level => !descriptorLevels.includes(level as never)
        );
        if (extra.length > 0) {
          sentences.push(
            `${id}: role "${roleKey}" accepts measurement levels ${describeLevels(role.measurement)} but the descriptor allows only ${describeLevels(descriptorLevels)} for ${slot.kind}; not allowed: ${extra.join(', ')}.`
          );
        }
      }
      if (JSON.stringify(role.required) !== JSON.stringify(field.required)) {
        sentences.push(
          `${id}: role "${roleKey}" is required ${JSON.stringify(role.required)} but the descriptor says ${JSON.stringify(field.required)}.`
        );
      }
      if (JSON.stringify(role.multiple) !== JSON.stringify(field.multiple)) {
        sentences.push(
          `${id}: role "${roleKey}" takes ${JSON.stringify(role.multiple) ?? 'one column'} but the descriptor says ${JSON.stringify(field.multiple) ?? 'one column'}.`
        );
      }
    }
    for (const field of input.fields) {
      if (!coveredParameters.has(`${input.name}.${field.parameter}`)) {
        sentences.push(
          `${id}: descriptor role "${field.role}" (parameter "${field.parameter}") of input "${input.name}" is not offered by slot "${slot.name}".`
        );
      }
    }
  }
  for (const input of descriptor.inputs) {
    if (!coveredInputs.has(input.name)) {
      sentences.push(`${id}: descriptor input "${input.name}" has no slot.`);
    }
  }
  // A parameter is active when its `activeWhen` holds under the presets and the binding's defaults.
  const bindingDefaults = Object.fromEntries(
    (getCatalogBinding(execution.bindingId)?.parameters ?? []).flatMap(parameter =>
      parameter.defaultValue === undefined ? [] : [[parameter.name, parameter.defaultValue]]
    )
  );
  const effectiveValues: Record<string, CatalogParameterValue> = {
    ...bindingDefaults,
    ...execution.parameters
  };
  const isActive = (parameter: (typeof descriptor.parameters)[number]): boolean =>
    evaluateGPUAnalysisParameterCondition(parameter.activeWhen, effectiveValues);

  for (const name of analysis.exposedParameters) {
    const parameter = descriptor.parameters.find(entry => entry.name === name);
    if (parameter === undefined) {
      sentences.push(`${id}: exposed parameter "${name}" has no entry in the descriptor.`);
    } else if (parameter.level !== 'basic') {
      sentences.push(`${id}: exposed parameter "${name}" is ${parameter.level} in the descriptor.`);
    } else if (!isActive(parameter)) {
      sentences.push(
        `${id}: exposed parameter "${name}" is inactive under the presets (it only applies when ${JSON.stringify(parameter.activeWhen)}).`
      );
    }
  }
  // `coordinateSystem` comes from the profile of the data, so it is never exposed by an analysis.
  const unexposed = descriptor.parameters
    .filter(
      parameter =>
        parameter.level === 'basic' &&
        parameter.name !== 'coordinateSystem' &&
        isActive(parameter) &&
        !analysis.exposedParameters.includes(parameter.name)
    )
    .map(parameter => parameter.name);
  if (unexposed.length > 0) {
    sentences.push(`${id}: descriptor basic parameters not exposed: ${unexposed.join(', ')}.`);
  }
  return sentences;
}

// ---------------------------------------------------------------------------------------------
// Saving and replaying choices
// ---------------------------------------------------------------------------------------------

/** Version of the saved form; `parse` rejects any other. */
const MAPPING_FORMAT_VERSION = 1;

/** Saves the user's own choices as JSON text with sorted keys, so equal choices give equal text. */
export function serializeKeplerCuratedAnalysisMapping(
  mapping: KeplerCuratedAnalysisMapping
): string {
  const sortRecord = <Value>(record: Readonly<Record<string, Value>>): Record<string, Value> =>
    Object.fromEntries(Object.entries(record).sort(([left], [right]) => compareText(left, right)));
  return JSON.stringify({
    version: MAPPING_FORMAT_VERSION,
    analysisId: mapping.choices.analysisId,
    datasets: sortRecord(mapping.choices.datasets),
    columns: sortRecord(mapping.choices.columns)
  });
}

/** Keys that would reach the prototype of a plain object. */
const UNSAFE_KEYS: readonly string[] = ['__proto__', 'constructor', 'prototype'];

function assertSafeKey(caller: string, section: string, key: string): void {
  if (UNSAFE_KEYS.some(unsafe => key === unsafe || key.split('.').includes(unsafe))) {
    throw new Error(`${caller}: ${section} key "${key}" is not allowed`);
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Reads what `serializeKeplerCuratedAnalysisMapping` wrote (JSON text or the parsed object).
 * Strict: wrong version, unknown keys, and values of the wrong type throw. Whether the datasets
 * and columns still exist is not checked here; `replayKeplerCuratedAnalysisMapping` reports that.
 */
export function parseKeplerCuratedAnalysisMapping(
  input: unknown
): KeplerCuratedAnalysisMappingChoices {
  const caller = 'parseKeplerCuratedAnalysisMapping';
  let value = input;
  if (typeof input === 'string') {
    try {
      value = JSON.parse(input);
    } catch {
      throw new Error(`${caller}: input is not valid JSON`);
    }
  }
  if (!isPlainObject(value)) {
    throw new Error(
      `${caller}: input must be an object with version, analysisId, datasets and columns`
    );
  }
  for (const key of Object.keys(value)) {
    if (!['version', 'analysisId', 'datasets', 'columns'].includes(key)) {
      throw new Error(
        `${caller}: unknown key "${key}"; accepted: version, analysisId, datasets, columns`
      );
    }
  }
  if (value['version'] !== MAPPING_FORMAT_VERSION) {
    throw new Error(
      `${caller}: version must be ${MAPPING_FORMAT_VERSION}; got ${String(value['version'])}`
    );
  }
  if (typeof value['analysisId'] !== 'string' || value['analysisId'].length === 0) {
    throw new Error(`${caller}: analysisId must be a non-empty string`);
  }
  if (!isPlainObject(value['datasets']) || !isPlainObject(value['columns'])) {
    throw new Error(`${caller}: datasets and columns must be objects`);
  }
  const datasets: Record<string, string> = {};
  for (const [slot, datasetId] of Object.entries(value['datasets'])) {
    assertSafeKey(caller, 'datasets', slot);
    if (typeof datasetId !== 'string' || datasetId.length === 0) {
      throw new Error(`${caller}: datasets["${slot}"] must be a non-empty dataset id string`);
    }
    datasets[slot] = datasetId;
  }
  const columns: Record<string, readonly string[]> = {};
  for (const [roleKey, names] of Object.entries(value['columns'])) {
    assertSafeKey(caller, 'columns', roleKey);
    if (
      !roleKey.includes('.') ||
      !Array.isArray(names) ||
      names.some(name => typeof name !== 'string' || name.length === 0)
    ) {
      throw new Error(
        `${caller}: columns["${roleKey}"] must be an array of non-empty column names under a "<slot>.<role>" key`
      );
    }
    if (names.length === 0) {
      throw new Error(
        `${caller}: columns["${roleKey}"] is empty; an empty choice is not saved (a role without a saved choice is left to the automatic one)`
      );
    }
    columns[roleKey] = names as string[];
  }
  return {analysisId: value['analysisId'], datasets, columns};
}
