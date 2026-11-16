// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  applyKeplerDatasetProfileDeclarations,
  countCollectionVertices,
  evaluateKeplerOperationApplicability,
  isKeplerDatasetCompatibleWithInput,
  normalizeCatalogParameters,
  parseKeplerDatasetProfileDeclarations,
  parseCatalogFeatureCollection,
  profileKeplerFeatureCollection,
  resolveKeplerOperationDescriptor,
  getKeplerEffectiveEngine,
  profileKeplerDatasetNetwork,
  withKeplerDatasetNetworkProfile,
  type CatalogOperationBinding,
  type CatalogFeatureCollection,
  type CatalogParameterValue,
  type KeplerDatasetProfile,
  type KeplerParameterResolution
} from '@luma-contributors/kepler';
import {
  getCatalogSampleInputs,
  getCatalogSampleInputDeclarations,
  readCatalogPointColumns
} from './analysis-catalog-input';
import {
  ANALYSIS_PROFILE_LIMITS,
  type AnalysisDatasetContextEntry
} from './analysis-dataset-context';

export type AnalysisCatalogInputDraft = {
  mode: 'sample' | 'text' | 'native';
  text: string;
  datasetId: string;
  field: string;
  /** Versioned declarations carried when another result becomes this text input. */
  profileDeclarations?: string;
};

export const EMPTY_CATALOG_INPUT_DRAFT: AnalysisCatalogInputDraft = {
  mode: 'sample',
  text: '',
  datasetId: '',
  field: ''
};

export type AnalysisCatalogInputProfiles = {
  profiles: Record<string, KeplerDatasetProfile | undefined>;
  problems: string[];
};

/** Reads only already-held GeoJSON and profile snapshots, never native/resident rows. */
export function resolveAnalysisCatalogInputProfiles(
  binding: CatalogOperationBinding,
  drafts: Readonly<Record<string, AnalysisCatalogInputDraft>>,
  getEntry: (datasetId: string) => AnalysisDatasetContextEntry | undefined,
  checkedProfiles?: Readonly<Record<string, KeplerDatasetProfile>>,
  coordinateSystem?: 'planar' | 'EPSG:4326'
): AnalysisCatalogInputProfiles {
  const profiles: AnalysisCatalogInputProfiles['profiles'] = {};
  const problems: string[] = [];
  let samples: Record<string, string> | undefined;
  for (const input of binding.inputs) {
    const draft = drafts[`${binding.id}:${input.name}`] ?? EMPTY_CATALOG_INPUT_DRAFT;
    try {
      if (input.dataKind && input.dataKind !== 'features') {
        throw new Error(
          'This catalog form reads GeoJSON features only. Open the typed workflow input instead.'
        );
      }
      if (draft.mode === 'native') {
        const entry = getEntry(draft.datasetId);
        const columns = readCatalogPointColumns(draft.field);
        const matches =
          entry &&
          (columns
            ? columns.every(Boolean) &&
              columns[0] === entry.profile.pointColumns?.longitude &&
              columns[1] === entry.profile.pointColumns?.latitude
            : Boolean(draft.field) && draft.field === entry.dataset.geometryColumn);
        if (checkedProfiles?.[input.name]) {
          // The panel admits checked profiles only for the current source, revision and mapping.
          // Keep their exact counts while carrying forward declarations from shared metadata.
          const shared = matches ? entry?.profile : undefined;
          const coordinateModel =
            shared?.coordinates?.source === 'declared'
              ? shared.coordinates.model
              : coordinateSystem === 'planar'
                ? 'planar'
                : coordinateSystem === 'EPSG:4326'
                  ? 'geographic'
                  : undefined;
          const checked = applyKeplerDatasetProfileDeclarations(checkedProfiles[input.name], {
            ...(coordinateModel ? {coordinateModel} : {}),
            ...(shared?.rowSemantics?.source === 'declared'
              ? {rowSemantics: shared.rowSemantics.value}
              : {}),
            measurement: Object.fromEntries(
              (shared?.columns ?? []).flatMap(column =>
                column.measurementSource === 'declared' && column.measurement
                  ? [[column.name, column.measurement]]
                  : []
              )
            )
          });
          profiles[input.name] = shared
            ? {
                ...checked,
                datasetId: shared.datasetId,
                label: shared.label,
                revision: shared.revision,
                geometryColumns: shared.geometryColumns,
                pointColumns: shared.pointColumns
              }
            : checked;
          if (draft.profileDeclarations)
            profiles[input.name] = applyKeplerDatasetProfileDeclarations(
              profiles[input.name]!,
              parseKeplerDatasetProfileDeclarations(draft.profileDeclarations)
            );
          continue;
        }
        if (!entry)
          throw new Error(
            'Choose a current dataset, then Check selected inputs to compute its profile.'
          );
        if (!matches)
          throw new Error(
            'This mapping has not been profiled. Choose the shared geometry or coordinate columns, or Check selected inputs.'
          );
        profiles[input.name] = entry.profile;
        if (draft.profileDeclarations)
          profiles[input.name] = applyKeplerDatasetProfileDeclarations(
            entry.profile,
            parseKeplerDatasetProfileDeclarations(draft.profileDeclarations)
          );
        if (!entry.dataset.provider.readColumnValues)
          problems.push(
            `${input.label}: original source readback unavailable. Prepare a source with original column readback.`
          );
        if (entry.dataset.rowCount > ANALYSIS_PROFILE_LIMITS.maximumRows)
          problems.push(
            `${input.label}: ${entry.dataset.rowCount} rows exceed the host readback limit ${ANALYSIS_PROFILE_LIMITS.maximumRows}. Filter or aggregate the dataset first.`
          );
        if (entry.limitation)
          problems.push(
            `${input.label}: ${entry.limitation}. Resolve the profile limitation before Run.`
          );
        continue;
      }
      if (draft.mode === 'text' && draft.text.length > ANALYSIS_PROFILE_LIMITS.maximumReadbackBytes)
        throw new Error(
          `GeoJSON text exceeds the host size limit ${ANALYSIS_PROFILE_LIMITS.maximumReadbackBytes}. Use a smaller source.`
        );
      if (draft.mode === 'sample') samples ??= getCatalogSampleInputs(binding.id);
      if (draft.mode === 'sample' && !samples![input.name])
        throw new Error(
          'No built-in sample is available for this input. Choose a dataset or GeoJSON text.'
        );
      const collection = parseCatalogFeatureCollection(
        draft.mode === 'sample' ? samples![input.name] : draft.text
      );
      profiles[input.name] =
        checkedProfiles?.[input.name] ??
        profileAnalysisCatalogInput(binding, input.name, collection);
      const sampleDeclarations =
        draft.mode === 'sample'
          ? getCatalogSampleInputDeclarations(binding.id, input.name)
          : undefined;
      if (sampleDeclarations)
        profiles[input.name] = applyKeplerDatasetProfileDeclarations(
          profiles[input.name]!,
          sampleDeclarations
        );
      if (draft.profileDeclarations)
        profiles[input.name] = applyKeplerDatasetProfileDeclarations(
          profiles[input.name]!,
          parseKeplerDatasetProfileDeclarations(draft.profileDeclarations)
        );
      if (coordinateSystem)
        profiles[input.name] = applyKeplerDatasetProfileDeclarations(profiles[input.name]!, {
          coordinateModel: coordinateSystem === 'planar' ? 'planar' : 'geographic'
        });
    } catch (failure) {
      profiles[input.name] = undefined;
      problems.push(
        `${input.label}: ${failure instanceof Error ? failure.message : String(failure)} Choose a compatible source or supply valid GeoJSON text.`
      );
    }
  }
  return {profiles, problems};
}

/** Explicit input checking and held-GeoJSON defaults use the same bounded AX profiler. */
export function profileAnalysisCatalogInput(
  binding: CatalogOperationBinding,
  inputName: string,
  collection: CatalogFeatureCollection,
  options?: {
    parameters: Readonly<Record<string, CatalogParameterValue>>;
    executionEngine: 'cpu' | 'gpu';
  }
): KeplerDatasetProfile {
  const input = binding.inputs.find(candidate => candidate.name === inputName)!;
  validateRequirement(
    binding.id,
    collection.features.length <= input.maximumFeatures,
    `${input.name}: ${collection.features.length} features exceed the limit ${input.maximumFeatures}. Filter or aggregate the source first.`
  );
  validateRequirement(
    binding.id,
    countCollectionVertices(collection) <= input.maximumVertices,
    `${input.name}: geometry exceeds the limit ${input.maximumVertices} vertices. Simplify the source first.`
  );
  const profile = profileKeplerFeatureCollection(collection, {
    datasetId: `${binding.id}:${input.name}`,
    label: input.label,
    maximumSampleSize: ANALYSIS_PROFILE_LIMITS.maximumSampleSize
  });
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const needsNetwork = descriptor.requirements.some(
    requirement =>
      requirement.input === inputName &&
      ['maximum-network-edges', 'maximum-network-nodes', 'maximum-network-arcs'].includes(
        requirement.id
      )
  );
  return options && needsNetwork
    ? withKeplerDatasetNetworkProfile(
        profile,
        profileKeplerDatasetNetwork(
          profile,
          collection,
          options.parameters,
          getKeplerEffectiveEngine(binding, options.executionEngine),
          binding.id
        )
      )
    : profile;
}

/** Explicit edits are separate from defaults: refinement can replace only unedited values. */
export function createAnalysisCatalogFormModel(
  binding: CatalogOperationBinding,
  profiles: Readonly<Record<string, KeplerDatasetProfile | undefined>>,
  options: {
    overrides?: Readonly<Record<string, unknown>>;
    executionEngine?: 'cpu' | 'gpu';
    running?: boolean;
    inputProblems?: readonly string[];
  } = {}
) {
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const problems = [...(options.inputProblems ?? [])];
  const overrides: Record<string, CatalogParameterValue> = {};
  // Validate the edited parameter alone, so normalizing does not turn defaults into overrides.
  for (const [name, value] of Object.entries(options.overrides ?? {})) {
    const specification = binding.parameters.find(parameter => parameter.name === name);
    try {
      overrides[name] = normalizeCatalogParameters(
        {id: binding.id, parameters: specification ? [specification] : []},
        {[name]: value}
      )[name];
    } catch {
      problems.push(
        `${binding.id}: ${specification?.label ?? name} needs a valid ${specification?.kind ?? 'declared parameter'}${specification?.minimum !== undefined ? ` in ${specification.minimum} to ${specification.maximum ?? 'unbounded'}` : ''}. Correct this setting before Run.`
      );
    }
  }
  const executionEngine = options.executionEngine ?? 'gpu';
  // Empty strings are deliberate unbindings. The checker ignores them in overrides, so pass
  // field edits explicitly to prevent it from silently proposing a replacement column.
  const fieldBindings = Object.fromEntries(
    descriptor.inputs.flatMap(input =>
      input.fields.flatMap(field => {
        const value = overrides[field.parameter];
        return typeof value === 'string'
          ? [
              [
                field.parameter,
                value.includes(',') ? value.split(',').map(name => name.trim()) : value
              ]
            ]
          : [];
      })
    )
  );
  const applicability = evaluateKeplerOperationApplicability(
    binding,
    descriptor,
    [
      ...new Map(
        binding.inputs.flatMap(input =>
          profiles[input.name]
            ? [[profiles[input.name]!.datasetId, profiles[input.name]!] as const]
            : []
        )
      ).values()
    ],
    {
      inputAssignments: Object.fromEntries(
        binding.inputs.map(input => [
          input.name,
          profiles[input.name]?.datasetId ?? `${binding.id}:unassigned:${input.name}`
        ])
      ),
      overrides,
      fieldBindings,
      executionEngine
    }
  );
  // Execution consumes exactly what the checker approved, including its grid/CPU work caps and
  // canonical field mappings. A second resolution can disagree even when both calls look alike.
  const resolved = {
    parameters: {...applicability.parameters},
    resolutions: applicability.resolutions.map(resolution =>
      Object.hasOwn(overrides, resolution.parameter)
        ? {...resolution, source: 'override' as const}
        : resolution
    ),
    basic: applicability.basic,
    advanced: applicability.advanced,
    inactive: applicability.resolutions
      .filter(resolution => resolution.active === false)
      .map(resolution => resolution.parameter)
  };
  // The checker is also a suggestion engine: no-input demonstrations cannot be suggested for a
  // dataset, but remain accessible here. Remove only that catalog-entry navigation instruction.
  const missing = applicability.missing.filter(
    item =>
      !(
        binding.inputs.length === 0 &&
        item.what === 'This operation reads no layer, so there is nothing to suggest it for'
      )
  );
  const status =
    problems.length > 0
      ? 'blocked'
      : binding.inputs.length === 0 && missing.length === 0
        ? 'ready'
        : applicability.status;
  const capacities = binding.inputs.map(input => ({
    input: input.name,
    label: input.label,
    rows: profiles[input.name]?.rowCount,
    vertices:
      profiles[input.name]?.vertexCount ??
      (profiles[input.name]?.kind === 'points' ? profiles[input.name]?.rowCount : undefined),
    maximumRows: input.dataKind === 'raster' ? input.maximumCells : input.maximumFeatures,
    maximumVertices: input.maximumVertices
  }));
  const selectionChecks = binding.inputs.flatMap(input => {
    const profile = profiles[input.name];
    if (!profile)
      return [
        {
          input: input.name,
          state: 'unknown',
          evidence: 'not-checkable',
          message: `${input.label}: no current profile for this input assignment`,
          remedy: 'Choose the dataset and geometry mapping, then check the selected inputs'
        }
      ];
    const inputDescriptor = descriptor.inputs.find(candidate => candidate.name === input.name)!;
    return isKeplerDatasetCompatibleWithInput(profile, input, inputDescriptor.kind)
      ? []
      : [
          {
            input: input.name,
            state: 'failed',
            evidence: 'profile',
            message: `${input.label}: selected ${profile.kind} profile (${profile.catalogGeometry ?? 'no geometry'}) does not fit ${inputDescriptor.kind} / ${input.accepts.join('/')}`,
            remedy:
              'Choose a source with the required kind and geometry, or prepare a compatible source'
          }
        ];
  });
  return {
    descriptor,
    resolved,
    applicability,
    status,
    missing,
    capacities,
    selectionChecks,
    problems,
    parameters: {...resolved.parameters, ...options.overrides},
    runDisabled: Boolean(options.running) || problems.length > 0 || status !== 'ready'
  };
}

/** Keep the resolver's source, rule, guess and clamping visible beside the actual value. */
export function describeAnalysisCatalogParameter(resolution: KeplerParameterResolution): string {
  return [
    resolution.source,
    resolution.reason,
    resolution.resolution?.explanation,
    resolution.guessed ? 'based on a guess; confirm dataset meaning' : undefined,
    resolution.clamped
      ? `adjusted from ${resolution.clamped.from} to ${resolution.clamped.to}: ${resolution.clamped.reason ?? 'parameter limit'}`
      : undefined,
    resolution.active === false ? 'inactive for these settings' : undefined
  ]
    .filter(Boolean)
    .join(' · ');
}
