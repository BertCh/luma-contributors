// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisColumnProfile,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {WorkflowData} from '../workflows/workflow-types';
import {
  applyKeplerDatasetProfileDeclarations,
  profileKeplerColumnTable
} from '../profile/dataset-profile';
import type {KeplerApplicabilityOptions, KeplerOperationApplicability} from './applicability';
import type {
  CatalogOperationBinding,
  CatalogParameterValue
} from '../analysis/catalog-binding-types';
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
import {type KeplerParameterResolution} from './default-parameters';
import {resolveKeplerOperationDescriptor} from './operation-descriptors';

// AX adapters over the actual workflow parameter and immutable data contracts. Catalog binding
// ids, input port names and field-role parameters remain the join keys.

/** The workbench and catalog share the complete parameter value contract. */
export type KeplerWorkbenchParameterValue = CatalogParameterValue;

export type KeplerWorkbenchProfiles = Readonly<Record<string, KeplerDatasetProfile | undefined>>;

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
  profilesByInput?: KeplerWorkbenchProfiles;
  overrides?: Readonly<Record<string, CatalogParameterValue>>;
};

export type KeplerWorkbenchApplicability = {
  status: 'passed' | 'failed' | 'unknown';
  reason: string;
  remedy?: string;
  /** Complete resolved parameters, including number lists. */
  parameters: Record<string, KeplerWorkbenchParameterValue>;
  /** Resolved output kinds by table id, including geometry inherited from the input. */
  outputKinds: Readonly<Record<string, GPUSpatialAnalysisDataKind>>;
};

export type KeplerWorkbenchParameterDefaults = {
  values: Record<string, KeplerWorkbenchParameterValue>;
  resolutions: readonly KeplerParameterResolution[];
  confirmations: readonly string[];
  basic: string[];
  advanced: string[];
  /** Parameters that do not matter for the current values of the others; also given in `values`. */
  inactive: string[];
};

export type KeplerWorkbenchCuratedPreset = {
  curatedAnalysisId: string;
  bindingId: string;
  /** The method's name, for a button: `Hot spot map`. */
  title: string;
  /** Method settings only, independent of sample field names and sample scale. */
  executionParameters: Record<string, KeplerWorkbenchParameterValue>;
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

function copyParameters(
  values: Readonly<Record<string, CatalogParameterValue>>
): Record<string, CatalogParameterValue> {
  return Object.fromEntries(
    Object.entries(values).map(([name, value]) => [name, Array.isArray(value) ? [...value] : value])
  );
}

/** Evaluate the connected input assignment exactly; never rematch profiles between ports. */
export function evaluateKeplerWorkbenchInputs(
  bindingId: string | CatalogOperationBinding,
  profilesByInput: KeplerWorkbenchProfiles,
  options: Omit<KeplerApplicabilityOptions, 'inputAssignments' | 'subjectOnly'> = {}
): KeplerOperationApplicability {
  const binding =
    typeof bindingId === 'string'
      ? requireBinding('evaluateKeplerWorkbenchInputs', bindingId)
      : bindingId;
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  // Distinct aliases preserve assignments even when the same dataset feeds two ports.
  const datasets = binding.inputs.flatMap(input => {
    const profile = Object.hasOwn(profilesByInput, input.name)
      ? profilesByInput[input.name]
      : undefined;
    if (!profile) return [];
    const coordinateSystem = options.overrides?.['coordinateSystem'];
    const effective =
      profile.coordinates?.source !== 'declared' &&
      (coordinateSystem === 'planar' || coordinateSystem === 'EPSG:4326')
        ? applyKeplerDatasetProfileDeclarations(profile, {
            coordinateModel: coordinateSystem === 'planar' ? 'planar' : 'geographic'
          })
        : profile;
    const datasetId = `workbench-input:${input.name}`;
    return [
      {
        ...effective,
        datasetId,
        sourceDatasetId: effective.sourceDatasetId ?? effective.datasetId,
        ...(effective.network && effective.network.datasetId === effective.datasetId
          ? {network: {...effective.network, datasetId}}
          : {}),
        ...(effective.catchmentSeeds
          ? {
              catchmentSeeds: {
                ...effective.catchmentSeeds,
                observationsDatasetId:
                  effective.catchmentSeeds.observationsDatasetId ===
                  profilesByInput['observations']?.datasetId
                    ? 'workbench-input:observations'
                    : effective.catchmentSeeds.observationsDatasetId
              }
            }
          : {})
      }
    ];
  });
  return evaluateKeplerOperationApplicability(binding, descriptor, datasets, {
    ...options,
    inputAssignments: Object.fromEntries(
      binding.inputs.map(input => [input.name, `workbench-input:${input.name}`])
    )
  });
}

const DATA_PROFILE_CACHE = new WeakMap<WorkflowData, KeplerDatasetProfile>();

/** One bounded profile per immutable intermediate; host source profiles retain declarations. */
export function profileKeplerWorkbenchData(data: WorkflowData): KeplerDatasetProfile {
  const cached = DATA_PROFILE_CACHE.get(data);
  if (cached) return cached;
  if (data.kind === 'raster' && data.raster) {
    const rasterProfile: KeplerDatasetProfile = {
      datasetId: data.table.id,
      label: data.table.label,
      kind: 'surface',
      rowCount: data.raster.grid.width * data.raster.grid.height,
      geometryColumns: [],
      columns: [],
      coordinates: {
        model: data.raster.crs === 'EPSG:4326' ? 'geographic' : 'planar',
        source: 'declared'
      }
    };
    DATA_PROFILE_CACHE.set(data, rasterProfile);
    return rasterProfile;
  }
  const profile = profileKeplerColumnTable(
    {
      rowCount: data.table.rowCount,
      geometryColumn: data.table.geometryColumn,
      columns: Object.entries(data.table.columns).map(([name, values]) => ({
        name,
        values,
        type: data.table.columnTypes[name]
      }))
    },
    {
      datasetId: data.table.id,
      label: data.table.label,
      maximumSampleSize: 1024,
      inferPointColumns: data.kind !== 'table',
      coordinateSystem: data.table.coordinateSystem
    }
  );
  DATA_PROFILE_CACHE.set(data, profile);
  return profile;
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

/** A subject without statistics leaves mandatory unknown evidence unresolved. */
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
    unknownRowCount: !profile && !adapterOptions.profilesByInput && subject.rowCount === undefined,
    subjectOnly: false,
    ...(adapterOptions.overrides ? {overrides: adapterOptions.overrides} : {}),
    ...(adapterOptions.executionEngine ? {executionEngine: adapterOptions.executionEngine} : {})
  };
  let applicability = adapterOptions.profilesByInput
    ? evaluateKeplerWorkbenchInputs(bindingId, adapterOptions.profilesByInput, options)
    : evaluateKeplerOperationApplicability(
        binding,
        descriptor,
        [profile ?? buildSubjectProfile(subject, geometries[0])],
        options
      );
  if (!profile && !adapterOptions.profilesByInput && subject.kind === 'features') {
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
    parameters: copyParameters(applicability.parameters),
    outputKinds: applicability.outputKinds
  };
}

/** Resolve defaults and their provenance using each connected input's own profile. */
export function getKeplerWorkbenchParameterDefaults(
  bindingId: string | CatalogOperationBinding,
  profile?: KeplerDatasetProfile,
  adapterOptions: KeplerWorkbenchOptions = {}
): KeplerWorkbenchParameterDefaults {
  const binding =
    typeof bindingId === 'string'
      ? requireBinding('getKeplerWorkbenchParameterDefaults', bindingId)
      : bindingId;
  // Legacy single-profile callers bind only the first port, never every port.
  const profilesByInput =
    adapterOptions.profilesByInput ??
    (profile && binding.inputs[0] ? {[binding.inputs[0].name]: profile} : {});
  // Applicability chooses role columns before resolving data-dependent defaults. The form
  // consumes that same parameter set and provenance rather than resolving without mappings.
  const resolved = evaluateKeplerWorkbenchInputs(binding, profilesByInput, {
    overrides: adapterOptions.overrides,
    ...(adapterOptions.executionEngine ? {executionEngine: adapterOptions.executionEngine} : {})
  });
  const values = copyParameters(resolved.parameters);
  return {
    values,
    resolutions: resolved.resolutions,
    confirmations: resolved.confirmations,
    basic: resolved.basic.filter(name => name in values && !resolved.inactive.includes(name)),
    advanced: resolved.advanced.filter(name => name in values && !resolved.inactive.includes(name)),
    inactive: resolved.inactive.filter(name => name in values)
  };
}

/**
 * Presets for the curated analyses that run today, one per analysis. Each carries the parameters
 * that make it run on its own sample (the curated presets, then the sample's scale parameters,
 * then the sample's columns written into the column parameters) and the sample as GeoJSON text by
 * binding input name. Number lists are preserved.
 */
let curatedPresets: KeplerWorkbenchCuratedPreset[] | undefined;

export function getKeplerWorkbenchCuratedPresets(): KeplerWorkbenchCuratedPreset[] {
  if (curatedPresets) return curatedPresets;
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
      // The GeoJSON bridge must not serialize a decoded raster as feature data. Typed inputs
      // remain available through the shared sample factory, with number-list settings intact.
      if (Object.values(sample).every(input => input.type === 'FeatureCollection')) {
        sampleGeoJSONByInput = {};
        for (const [slotName, collection] of Object.entries(sample)) {
          const inputName = execution.inputs[slotName];
          if (inputName !== undefined) {
            sampleGeoJSONByInput[inputName] = JSON.stringify(collection);
          }
        }
      }
    }
    presets.push({
      curatedAnalysisId: analysis.id,
      executionParameters: copyParameters(execution.parameters),
      bindingId: execution.bindingId,
      title: analysis.name,
      parameters: copyParameters(parameters),
      ...(sampleGeoJSONByInput ? {sampleGeoJSONByInput} : {})
    });
  }
  curatedPresets = presets;
  return presets;
}
