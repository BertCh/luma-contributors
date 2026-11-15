// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisColumnProfile,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {getCatalogBinding} from '../analysis/catalog-bindings';
import {KEPLER_CURATED_ANALYSES} from '../curated/curated-analyses';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleParameters,
  KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS,
  type KeplerCuratedAnalysisSampleId
} from '../curated/curated-analysis-samples';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {evaluateKeplerOperationApplicability} from './applicability';
import {KEPLER_GRID_CELL_CAPS, resolveKeplerOperationParameters} from './default-parameters';
import {resolveKeplerOperationDescriptor} from './operation-descriptors';

// Thin adapters for the analysis workbench. They use local structural types only, so the
// workbench and this module share no import: the join keys are the catalog binding id and the
// binding's parameter names.

/** A parameter value the workbench can hold in a form field. A list of numbers is omitted. */
export type KeplerWorkbenchParameterValue = string | number | boolean;

/** What the workbench knows about a dataset it has: geometry kind, size and typed columns. */
export type KeplerWorkbenchSubject = {
  /**
   * `features` is a collection whose geometry is mixed or not yet known; it is judged as the first
   * of point, polygon and line that the binding accepts.
   */
  kind: 'point' | 'line' | 'polygon' | 'features' | 'table';
  /** Absent when not known; row and vertex limits are then `unknown`, not failed. */
  rowCount?: number;
  /** Total vertices, when known; polygon and line operations limit them. */
  vertexCount?: number;
  /** Longitude and latitude columns of a table: such a table is a layer of points. */
  pointColumns?: {longitude: string; latitude: string};
  columns: readonly {name: string; type: 'string' | 'float64' | 'int32' | 'boolean'}[];
};

/** What the adapters accept besides the subject. */
export type KeplerWorkbenchOptions = {
  /** Where the operation will run; default `gpu`. See `KeplerApplicabilityOptions`. */
  executionEngine?: 'gpu' | 'cpu';
};

export type KeplerWorkbenchApplicability = {
  status: 'passed' | 'failed' | 'unknown';
  reason: string;
  remedy?: string;
  /** Resolved parameters; parameters that are lists of numbers are omitted. */
  parameters: Record<string, KeplerWorkbenchParameterValue>;
  /** Resolved output kinds by table id, including geometry inherited from the input. */
  outputKinds: Readonly<Record<string, GPUSpatialAnalysisDataKind>>;
};

export type KeplerWorkbenchParameterDefaults = {
  values: Record<string, KeplerWorkbenchParameterValue>;
  basic: string[];
  advanced: string[];
  /** Parameters that do not matter for the current values of the others; also given in `values`. */
  inactive: string[];
};

export type KeplerWorkbenchCuratedPreset = {
  bindingId: string;
  /** The method's name, for a button: `Hot spot map`. */
  title: string;
  /** The curated presets, the parameters that suit the sample and the sample's column choices. */
  parameters: Record<string, KeplerWorkbenchParameterValue>;
  /** GeoJSON text of the sample, by binding input name. Absent when the analysis has no sample. */
  sampleGeoJSONByInput?: Record<string, string>;
};

const SUBJECT_COLUMN_TYPES: Readonly<Record<string, GPUAnalysisColumnProfile['type']>> = {
  string: 'category',
  float64: 'number',
  int32: 'integer',
  boolean: 'boolean'
};

function omitNumberLists(
  values: Readonly<Record<string, CatalogParameterValue>>
): Record<string, KeplerWorkbenchParameterValue> {
  const result: Record<string, KeplerWorkbenchParameterValue> = {};
  for (const [name, value] of Object.entries(values)) {
    if (!Array.isArray(value)) {
      result[name] = value as KeplerWorkbenchParameterValue;
    }
  }
  return result;
}

function requireBinding(functionName: string, bindingId: string) {
  const binding = getCatalogBinding(bindingId);
  if (!binding) {
    throw new Error(`${functionName} bindingId "${bindingId}" is not a registered catalog binding`);
  }
  return binding;
}

function buildSubjectProfile(
  subject: KeplerWorkbenchSubject,
  geometry: 'point' | 'line' | 'polygon' | undefined
): KeplerDatasetProfile {
  const rowCount = subject.rowCount ?? 0;
  // A table with longitude and latitude columns is a layer of points.
  const effective = geometry ?? (subject.pointColumns ? 'point' : undefined);
  return {
    datasetId: 'workbench-subject',
    label: 'This dataset',
    kind:
      effective === 'point'
        ? 'points'
        : effective === 'line'
          ? 'lines'
          : effective === 'polygon'
            ? 'polygons'
            : 'attributes',
    rowCount,
    ...(subject.vertexCount === undefined ? {} : {vertexCount: subject.vertexCount}),
    ...(effective === undefined ? {} : {catalogGeometry: effective}),
    ...(subject.pointColumns ? {pointColumns: subject.pointColumns} : {}),
    geometryColumns: [],
    columns: subject.columns.map(column => ({
      name: column.name,
      type: SUBJECT_COLUMN_TYPES[column.type],
      rowCount,
      nullCount: 0
    }))
  };
}

/**
 * Judges one operation for the workbench's current subject.
 *
 * Status mapping. The engine has four statuses and the workbench three:
 * - `ready` is `passed`; `blocked` is `failed` (the reason names what is missing).
 * - `needs-confirmation` and `needs-mapping` are `unknown` when a profile is given: a person has
 *   something to confirm or choose.
 * - Without a profile the call knows only the columns' storage types, the geometry kind and
 *   (optionally) the row and vertex counts. It cannot know what a column measures, whether the
 *   coordinates are degrees, or the spacing of the points, and a person can do nothing about those
 *   from here. So in that mode they never lower the status: the result is `passed` and `reason`
 *   ends with "not known from the subject alone: ..." naming them. `unknown` is then not returned.
 *
 * Without a profile one is built from the subject: typed columns, no coordinate model and no
 * spacing statistics. A table with `pointColumns` is a layer of points. `parameters` are the
 * resolved parameters without the lists of numbers, which a form field cannot hold.
 */
export function evaluateKeplerWorkbenchApplicability(
  bindingId: string,
  subject: KeplerWorkbenchSubject,
  profile?: KeplerDatasetProfile,
  adapterOptions: KeplerWorkbenchOptions = {}
): KeplerWorkbenchApplicability {
  const binding = getCatalogBinding(bindingId);
  if (!binding) {
    return {
      status: 'failed',
      reason: `"${bindingId}" is not a registered operation`,
      remedy: 'Choose an operation from the catalog',
      parameters: {},
      outputKinds: {}
    };
  }
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const geometries: ('point' | 'line' | 'polygon' | undefined)[] =
    subject.kind === 'table'
      ? [undefined]
      : subject.kind === 'features'
        ? ['point', 'polygon', 'line']
        : [subject.kind];
  const options = {
    unknownRowCount: subject.rowCount === undefined,
    subjectOnly: profile === undefined,
    ...(adapterOptions.executionEngine ? {executionEngine: adapterOptions.executionEngine} : {})
  };
  let applicability = evaluateKeplerOperationApplicability(
    binding,
    descriptor,
    [profile ?? buildSubjectProfile(subject, geometries[0])],
    options
  );
  if (!profile && subject.kind === 'features') {
    for (const geometry of geometries) {
      const candidate = evaluateKeplerOperationApplicability(
        binding,
        descriptor,
        [buildSubjectProfile(subject, geometry)],
        options
      );
      if (candidate.status !== 'blocked') {
        applicability = candidate;
        break;
      }
    }
  }
  const first = applicability.missing[0];
  const extra = applicability.missing.length - 1;
  return {
    status:
      applicability.status === 'ready'
        ? 'passed'
        : applicability.status === 'blocked'
          ? 'failed'
          : 'unknown',
    reason:
      first === undefined
        ? [
            applicability.reasons.join('; '),
            ...(applicability.caveats.length > 0
              ? [`not known from the subject alone: ${applicability.caveats.join('; ')}`]
              : [])
          ]
            .filter(part => part.length > 0)
            .join('; ')
        : extra > 0
          ? `${first.what} (and ${extra} more)`
          : first.what,
    ...(first?.remedy === undefined ? {} : {remedy: first.remedy}),
    parameters: omitNumberLists(applicability.parameters),
    outputKinds: applicability.outputKinds
  };
}

/**
 * Default parameter values for an operation: derived from the profile where the descriptor says
 * so (every input is read from the one profile), the binding's defaults otherwise. Without a
 * profile every value is the binding's default. `basic` and `advanced` split the parameter names.
 *
 * @throws Error when `bindingId` is not registered.
 */
export function getKeplerWorkbenchParameterDefaults(
  bindingId: string,
  profile?: KeplerDatasetProfile,
  adapterOptions: KeplerWorkbenchOptions = {}
): KeplerWorkbenchParameterDefaults {
  const binding = requireBinding('getKeplerWorkbenchParameterDefaults', bindingId);
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const profilesByInput = Object.fromEntries(binding.inputs.map(input => [input.name, profile]));
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
    maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount,
    ...(adapterOptions.executionEngine ? {executionEngine: adapterOptions.executionEngine} : {})
  });
  const values = omitNumberLists(resolved.parameters);
  return {
    values,
    basic: resolved.basic.filter(name => name in values && !resolved.inactive.includes(name)),
    advanced: resolved.advanced.filter(name => name in values && !resolved.inactive.includes(name)),
    inactive: resolved.inactive.filter(name => name in values)
  };
}

/**
 * Presets for the curated analyses that run today, one per analysis. Each carries the parameters
 * that make it run on its own sample (the curated presets, then the sample's scale parameters,
 * then the sample's columns written into the column parameters) and the sample as GeoJSON text by
 * binding input name. Lists of numbers are omitted from `parameters`.
 */
export function getKeplerWorkbenchCuratedPresets(): KeplerWorkbenchCuratedPreset[] {
  const presets: KeplerWorkbenchCuratedPreset[] = [];
  for (const analysis of KEPLER_CURATED_ANALYSES) {
    if (analysis.execution.status !== 'available') {
      continue;
    }
    const execution = analysis.execution;
    const parameters: Record<string, CatalogParameterValue> = {
      ...execution.parameters,
      ...getKeplerCuratedAnalysisSampleParameters(analysis.id)
    };
    let sampleGeoJSONByInput: Record<string, string> | undefined;
    if (analysis.sampleId !== undefined) {
      const sampleColumns =
        KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS[
          analysis.sampleId as KeplerCuratedAnalysisSampleId
        ] ?? {};
      for (const [slotRole, parameter] of Object.entries(execution.fieldParameters)) {
        if (slotRole in sampleColumns) {
          parameters[parameter] = sampleColumns[slotRole];
        }
      }
      const sample = createKeplerCuratedAnalysisSample(analysis.sampleId);
      sampleGeoJSONByInput = {};
      for (const [slotName, collection] of Object.entries(sample)) {
        const inputName = execution.inputs[slotName];
        if (inputName !== undefined) {
          sampleGeoJSONByInput[inputName] = JSON.stringify(collection);
        }
      }
    }
    presets.push({
      bindingId: execution.bindingId,
      title: analysis.name,
      parameters: omitNumberLists(parameters),
      ...(sampleGeoJSONByInput ? {sampleGeoJSONByInput} : {})
    });
  }
  return presets;
}
