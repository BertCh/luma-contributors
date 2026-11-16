// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState, useSyncExternalStore} from 'react';
import type * as arrow from 'apache-arrow';
import type KeplerTable from '@kepler.gl/table';
import type {Device} from '@luma.gl/core';
import {
  KEPLER_CATALOG_BINDINGS,
  completeNetworkCategoricalCatalogEntries,
  describeKeplerCatalogResult,
  getCatalogBinding,
  listCatalogEntries,
  matchCatalogInvocationInputs,
  parseCatalogInvocation,
  parseCatalogFeatureCollection,
  listCatalogLensScenarios,
  runCatalogLensScenario,
  runCatalogOperationAsync,
  resolveKeplerOperationDescriptor,
  parseKeplerDatasetProfileDeclarations,
  serializeKeplerDatasetProfileDeclarations,
  serializeCatalogInvocation,
  type CatalogLensScenario,
  type CatalogFeatureCollection,
  type CatalogOperationBinding,
  type CatalogOutputTable,
  type CatalogRunResult,
  type CatalogParameterValue,
  type KeplerDatasetProfile,
  type KeplerDatasetRegistry,
  AnalysisPreviewSession,
  getAnalysisSensitivityUnavailableReason,
  type AnalysisSensitivityProvenance
} from '@luma-contributors/kepler';
import {AnalysisSensitivityCleanupStatus} from './analysis-sensitivity-strip';
import {AnalysisResultCard} from './analysis-result-card';
import {
  createAnalysisResultCardModel,
  createAnalysisResultNextDraft,
  makeAnalysisResultWorkflow,
  hasAnalysisResultStackEntry,
  keepAnalysisResultWorkflow,
  type AnalysisResultStackEntry,
  type AnalysisResultCardModel,
  type AnalysisResultNextAction
} from './analysis-result-card-model';
import type {WorkbenchState, WorkbenchStore} from './workflow/workbench-store';
import {includeSpacetimeCatalogEntries} from './analysis-catalog-spacetime';
import {
  prepareAnalysisResultPublication,
  showAnalysisResultReport,
  type AnalysisResultPublication
} from './analysis-result-publication';
import {
  TRAJECTORY_POINT_FIELD,
  isTrajectoryCatalogOperation,
  makeCatalogTrajectoryNativeSource,
  getCatalogTrajectoryTripOptions,
  type CatalogTrajectorySource
} from './analysis-catalog-trajectory';
import {readCatalogNativeInput} from './analysis-catalog-native-input';

import type {SandboxTripOptions} from './trip-source-options';
import {
  AnalysisCatalogLensMapping,
  EMPTY_LENS_MAPPING,
  readLensUserData,
  type LensMappingDraft
} from './analysis-catalog-lens-mapping';
import {
  listCatalogInvocations,
  recordCatalogInvocation,
  subscribeCatalogInvocations,
  type CatalogInvocationRecord
} from './catalog-invocation-store';
import {
  findNativeGeometryFields,
  makeCatalogPointColumns,
  readCatalogPointColumns,
  getCatalogSampleInputs
} from './analysis-catalog-input';

import {
  getCatalogResidentDatasetLabel,
  getCatalogResidentFields,
  getCatalogResidentGeometryFields,
  type CatalogResidentDataset
} from './analysis-catalog-resident-input';

import {analysisDatasetContext} from './analysis-dataset-context';
import {AnalysisRowMeaning} from './analysis-profile-declarations';
import {
  isAnalysisSuggestionLaunchCurrent,
  type AnalysisSuggestionLaunch
} from './analysis-suggestion-model';
import {
  createAnalysisCatalogFormModel,
  EMPTY_CATALOG_INPUT_DRAFT,
  profileAnalysisCatalogInput,
  resolveAnalysisCatalogInputProfiles,
  type AnalysisCatalogInputDraft as InputDraft
} from './analysis-catalog-form-model';

import {AnalysisSheet, AnalysisSheetParameter, AnalysisSheetPreviewScope} from './analysis-sheet';
import {createAnalysisSheetModel, getAnalysisSheetPreviewDelay} from './analysis-sheet-model';
import {createAnalysisPreviewController} from './analysis-preview-controller';
import {createAnalysisRunIdentity} from './analysis-run-identity';

const subscribeNoAnalysisStack = (_listener: () => void) => () => {};
const getNoAnalysisStack = () => null;

/** Catalog discovery plus the typed run form for operations that have executable bindings. */
export function AnalysisCatalogPanel({
  datasets,
  residentDatasets = [],
  trajectorySources = [],
  device,
  registry,
  getCurrentRegistry,
  launchRequest,
  onLaunchConsumed,
  onPublish
}: {
  launchRequest?: AnalysisSuggestionLaunch;
  onLaunchConsumed?(launch: AnalysisSuggestionLaunch): void;
  datasets: readonly KeplerTable[];
  residentDatasets?: readonly CatalogResidentDataset[];
  trajectorySources?: readonly CatalogTrajectorySource[];
  /** GPU device of the compute runtime. Without it only operations with a CPU engine run. */
  device?: Device | null;
  /** Shared runtime registry admits overlapping retained previews before allocation. */
  registry?: KeplerDatasetRegistry;
  /** Live host identity, including replacement before the next React render. */
  getCurrentRegistry?(): KeplerDatasetRegistry | undefined;
  /** The host stages the table in the Sources workflow; nothing is uploaded here. */
  onPublish(
    table: arrow.Table,
    label: string,
    geometryColumn?: string,
    tripOptions?: SandboxTripOptions,
    analysisResult?: AnalysisResultPublication
  ): void;
}) {
  const [lensScenarios, setLensScenarios] = useState<CatalogLensScenario[]>([]);
  const [lensScenarioId, setLensScenarioId] = useState('');
  const [lensMapping, setLensMapping] = useState<LensMappingDraft>(EMPTY_LENS_MAPPING);
  const [engineChoice, setEngineChoice] = useState<'auto' | 'cpu'>('auto');
  const [running, setRunning] = useState(false);
  const previewSession = useRef<ReturnType<typeof createAnalysisPreviewController> | undefined>(
    undefined
  );
  const runController = useRef<AbortController | undefined>(undefined);
  const runSequence = useRef(0);
  const configurationGeneration = useRef(0);
  const sheetMounted = useRef(true);
  useEffect(() => {
    sheetMounted.current = true;
    return () => {
      sheetMounted.current = false;
    };
  }, []);
  const [sensitivitySelection, setSensitivitySelection] = useState(0);
  const restoredSensitivity = useRef<AnalysisSensitivityProvenance | undefined>(undefined);
  const sensitivitySearch = useRef<AnalysisSensitivityProvenance | undefined>(undefined);
  const [previewEnabled, setPreviewEnabled] = useState(false);
  const [previewGeneration, setPreviewGeneration] = useState(0);
  const lastApply = useRef<'live' | 'rebuild'>('rebuild');
  const cancelPendingRun = () => {
    setRunning(false);
    runSequence.current++;
    runController.current?.abort();
    previewSession.current?.cancel();
  };
  const invalidateConfiguration = () => {
    sensitivitySearch.current = undefined;
    restoredSensitivity.current = undefined;
    configurationGeneration.current++;
    cancelPendingRun();
  };
  const entries = useMemo(
    () =>
      completeNetworkCategoricalCatalogEntries(
        includeSpacetimeCatalogEntries(listCatalogEntries({lensScenarios}))
      ),
    [lensScenarios]
  );
  const [operationId, setOperationId] = useState(KEPLER_CATALOG_BINDINGS[0].id);
  const binding = getCatalogBinding(operationId) as CatalogOperationBinding;
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [drafts, setDrafts] = useState<Record<string, InputDraft>>({});
  const [result, setResult] = useState<CatalogRunResult | null>(null);
  const [completedRun, setCompletedRun] = useState<{
    model: AnalysisResultCardModel;
    inputs: Record<string, CatalogFeatureCollection>;
    id: number;
  }>();
  const [stackedRun, setStackedRun] = useState<{
    runId: number;
    entry: AnalysisResultStackEntry;
    store: WorkbenchStore;
  }>();
  const stackState = useSyncExternalStore<WorkbenchState | null>(
    stackedRun?.store.subscribe ?? subscribeNoAnalysisStack,
    stackedRun?.store.getState ?? getNoAnalysisStack,
    getNoAnalysisStack
  );
  const clearResult = () => {
    setResult(null);
    setCompletedRun(undefined);
    setResultContext(undefined);
  };
  const [error, setError] = useState('');
  const [invocationText, setInvocationText] = useState('');
  const [showBlocked, setShowBlocked] = useState(false);
  const [dismissedWarnings, setDismissedWarnings] = useState<string[]>([]);
  const controlProblemsRef = useRef<Record<string, string>>({});
  const [controlProblems, setControlProblems] = useState(controlProblemsRef.current);
  const hasInvalidControls = () =>
    Object.keys(controlProblemsRef.current).some(key => key.startsWith(`${operationId}:`));
  const reportControlProblem = (parameter: string, problem: string | null) => {
    const key = `${operationId}:${parameter}`;
    const previous = controlProblemsRef.current;
    if ((previous[key] ?? null) === problem) return;
    const next = {...previous};
    if (problem === null) delete next[key];
    else next[key] = problem;
    // Action guards read the ref before React has rendered the new preflight state.
    controlProblemsRef.current = next;
    setControlProblems(next);
    if (problem !== null) cancelPendingRun();
  };

  const consumedLaunch = useRef<AnalysisSuggestionLaunch | undefined>(undefined);
  useEffect(() => {
    if (!launchRequest || consumedLaunch.current === launchRequest) return;
    consumedLaunch.current = launchRequest;
    // Acknowledge both accepted and rejected requests in the persistent host.
    onLaunchConsumed?.(launchRequest);
    const current = isAnalysisSuggestionLaunchCurrent(launchRequest);
    if (!current) {
      setError('Suggested input changed. Reopen the suggestion for the current dataset revision.');
      return;
    }
    invalidateConfiguration();
    setPreviewEnabled(false);
    setOperationId(launchRequest.bindingId);
    setDrafts({...launchRequest.drafts});
    setOverrides({...launchRequest.overrides});
    setEngineChoice('auto');
    clearResult();
    setError('');
    setCheckedInputs(undefined);
    // The catalog also mounts when a source modal closes; focus after its launch state commits.
    requestAnimationFrame(() => {
      const operation = document.getElementById('catalog-operation');
      operation?.focus();
      operation?.scrollIntoView({block: 'nearest'});
    });
  }, [launchRequest, onLaunchConsumed]);
  const lastInputs = useRef<Record<string, CatalogFeatureCollection>>({});
  const runnableCount = entries.filter(entry => entry.runnable !== 'no').length;
  const blocked = entries.filter(entry => entry.runnable === 'no');
  const isTrajectory = isTrajectoryCatalogOperation(operationId);
  const inputDatasets = useMemo(
    () =>
      isTrajectory
        ? [...datasets, ...trajectorySources.map(makeCatalogTrajectoryNativeSource)]
        : datasets,
    [isTrajectory, datasets, trajectorySources]
  );
  const geometryDatasets = [
    ...inputDatasets.map(dataset => ({
      dataset,
      fields: [
        ...findNativeGeometryFields(dataset as never),
        // Manual point mapping is a UI choice, not a discovered geometry field.
        ...(dataset.fields.filter(field => field.name.trim()).length >= 2
          ? [makeCatalogPointColumns('', '')]
          : []),
        ...(isTrajectory ? [TRAJECTORY_POINT_FIELD] : [])
      ]
    })),
    ...residentDatasets.map(resident => ({
      dataset: {
        id: resident.id,
        label: getCatalogResidentDatasetLabel(resident),
        fields: getCatalogResidentFields(resident)
      },
      fields: getCatalogResidentGeometryFields(resident)
    }))
  ];

  const profileVersion = useSyncExternalStore(
    analysisDatasetContext.subscribe,
    analysisDatasetContext.getVersion,
    analysisDatasetContext.getVersion
  );
  const [checking, setChecking] = useState(false);
  const sourceIdentities = [
    ...datasets.map(dataset => dataset.dataContainer),
    ...trajectorySources.map(source => source.table),
    ...residentDatasets.flatMap(dataset => [dataset.owner ?? dataset.provider, dataset.provider])
  ];
  // Metadata revisions need not change when a native container/provider is replaced.
  // Keep cache identity monotonic so replacing A with B (or later A again) rebuilds inputs.
  const sourceIdentityState = useRef({identities: sourceIdentities, generation: 0});
  if (
    sourceIdentityState.current.identities.length !== sourceIdentities.length ||
    sourceIdentities.some(
      (source, index) => source !== sourceIdentityState.current.identities[index]
    )
  ) {
    sourceIdentityState.current = {
      identities: sourceIdentities,
      generation: sourceIdentityState.current.generation + 1
    };
  }
  const readSourceRevisions = () =>
    JSON.stringify([
      datasets.map(dataset => [
        dataset.id,
        dataset.dataContainer.numRows(),
        'dataRevision' in dataset ? dataset.dataRevision : undefined
      ]),
      residentDatasets.map(dataset => [
        dataset.id,
        dataset.revision,
        dataset.provider.getRevision(),
        dataset.isCurrent()
      ])
    ]);
  const sourceRevisions = readSourceRevisions();
  const checksDependOnSettings =
    isTrajectory ||
    resolveKeplerOperationDescriptor(binding).descriptor.requirements.some(requirement =>
      ['maximum-network-edges', 'maximum-network-nodes', 'maximum-network-arcs'].includes(
        requirement.id
      )
    );
  const [checkedInputs, setCheckedInputs] = useState<{
    drafts: typeof drafts;
    overrides: typeof overrides;
    operationId: string;
    sourceIdentities: readonly object[];
    sourceRevisions: string;
    profileVersion: number;
    profiles: Record<string, KeplerDatasetProfile>;
    engineChoice: typeof engineChoice;
    device: typeof device;
  }>();
  const checkedProfiles =
    checkedInputs &&
    checkedInputs.operationId === operationId &&
    checkedInputs.drafts === drafts &&
    (!checksDependOnSettings ||
      (checkedInputs.overrides === overrides &&
        checkedInputs.engineChoice === engineChoice &&
        checkedInputs.device === device)) &&
    checkedInputs.sourceRevisions === sourceRevisions &&
    checkedInputs.profileVersion === profileVersion &&
    checkedInputs.sourceIdentities.length === sourceIdentities.length &&
    checkedInputs.sourceIdentities.every((source, index) => source === sourceIdentities[index])
      ? checkedInputs.profiles
      : undefined;
  const latestInputSelection = useRef({
    drafts,
    operationId,
    sourceRevisions,
    sourceIdentities,
    profileVersion,
    overrides,
    engineChoice,
    device
  });
  latestInputSelection.current = {
    drafts,
    operationId,
    sourceRevisions,
    sourceIdentities,
    profileVersion,
    overrides,
    engineChoice,
    device
  };
  const selectedCoordinateSystem =
    overrides['coordinateSystem'] === 'planar' || overrides['coordinateSystem'] === 'EPSG:4326'
      ? overrides['coordinateSystem']
      : undefined;
  const inputProfiles = useMemo(
    () =>
      resolveAnalysisCatalogInputProfiles(
        binding,
        drafts,
        analysisDatasetContext.get,
        checkedProfiles,
        selectedCoordinateSystem
      ),
    [binding, drafts, profileVersion, checkedProfiles, selectedCoordinateSystem]
  );
  const form = createAnalysisCatalogFormModel(binding, inputProfiles.profiles, {
    overrides,
    executionEngine: engineChoice === 'cpu' || !device ? 'cpu' : 'gpu',
    running: running || checking,
    inputProblems: [
      ...inputProfiles.problems,
      ...Object.entries(controlProblems).flatMap(([key, problem]) =>
        key.startsWith(`${operationId}:`)
          ? [`${binding.id}: ${key.slice(operationId.length + 1)} ${problem}`]
          : []
      )
    ]
  });
  const parameters = form.resolved.parameters;
  const sheet = createAnalysisSheetModel(binding, form, inputProfiles.profiles);
  const previewAvailable = Boolean(
    binding.planPreview && device && registry && engineChoice === 'auto'
  );
  const parameterKey = JSON.stringify(parameters);
  const revisionKey = JSON.stringify([
    operationId,
    drafts,
    sourceRevisions,
    sourceIdentityState.current.generation,
    profileVersion,
    engineChoice
  ]);
  const latestExecution = useRef({revisionKey, parameterKey, device, registry, sourceIdentities});
  latestExecution.current = {revisionKey, parameterKey, device, registry, sourceIdentities};
  const [resultContext, setResultContext] = useState<{
    revisionKey: string;
    parameterKey: string;
    isCurrent: () => boolean;
    profiles: typeof inputProfiles.profiles;
    resolutions: typeof form.resolved.resolutions;
  }>();
  const resultCurrent = Boolean(
    !hasInvalidControls() &&
      resultContext &&
      resultContext.revisionKey === revisionKey &&
      resultContext.parameterKey === parameterKey &&
      resultContext.isCurrent()
  );
  useEffect(() => {
    const session = previewAvailable
      ? new AnalysisPreviewSession({binding, device: device!, registry: registry!})
      : undefined;
    const preview = session
      ? createAnalysisPreviewController({
          session,
          getRuntime: () =>
            getCurrentRegistry ? getCurrentRegistry() : latestExecution.current.registry,
          getRevisionKey: () => latestExecution.current.revisionKey
        })
      : undefined;
    previewSession.current = preview;
    return () => {
      cancelPendingRun();
      if (previewSession.current === preview) previewSession.current = undefined;
      void preview?.destroy().catch(failure => setError(String(failure)));
    };
  }, [binding, device, registry, previewAvailable, previewGeneration]);
  const createResultIdentity = () => {
    const snapshot = latestExecution.current;
    const generation = configurationGeneration.current;
    const deviceEpoch = registry?.deviceEpoch;
    return createAnalysisRunIdentity({
      // Completed arrays remain valid after request cancellation. Their identity tracks only
      // source/settings/runtime changes; pending requests retain a separate abortable identity.
      signal: new AbortController().signal,
      getRuntime: () =>
        getCurrentRegistry
          ? getCurrentRegistry()
          : (latestExecution.current.registry ?? latestExecution.current.device),
      getRevisionKey: () =>
        JSON.stringify([latestExecution.current.revisionKey, latestExecution.current.parameterKey]),
      isCurrent: () => {
        const latest = latestExecution.current;
        return (
          sheetMounted.current &&
          configurationGeneration.current === generation &&
          analysisDatasetContext.getVersion() === profileVersion &&
          latest.revisionKey === snapshot.revisionKey &&
          latest.parameterKey === snapshot.parameterKey &&
          latest.device === snapshot.device &&
          latest.registry === snapshot.registry &&
          latest.sourceIdentities.length === snapshot.sourceIdentities.length &&
          latest.sourceIdentities.every(
            (source, index) => source === snapshot.sourceIdentities[index]
          ) &&
          registry?.deviceEpoch === deviceEpoch &&
          readSourceRevisions() === sourceRevisions
        );
      }
    });
  };
  const changeParameter = (name: string, value: unknown, apply: 'live' | 'rebuild') => {
    const previous = parameters[name];
    const unchanged =
      Object.is(previous, value) ||
      (Array.isArray(previous) &&
        Array.isArray(value) &&
        previous.length === value.length &&
        previous.every((item, index) => Object.is(item, value[index])));
    // A ready form needs no new override for the same canonical value. Before readiness,
    // an explicit value can still confirm a fallback or the declared coordinate system.
    if (
      unchanged &&
      (form.status === 'ready' ||
        (resultContext?.revisionKey === revisionKey &&
          resultContext.parameterKey === parameterKey &&
          resultContext.isCurrent()))
    )
      return;
    invalidateConfiguration();
    lastApply.current = apply;
    setOverrides(current => ({...current, [name]: value}));
  };
  const createResultIdentityForSensitivity = createResultIdentity().isCurrent;
  const renderParameter = (row: (typeof sheet.rows)[number]) => (
    <AnalysisSheetParameter
      key={`${binding.id}:${row.specification.name}`}
      spec={binding.parameters.find(candidate => candidate.name === row.specification.name)!}
      row={row}
      value={form.parameters[row.specification.name] as CatalogParameterValue}
      source={row.source}
      sensitivity={
        previewAvailable &&
        form.status === 'ready' &&
        !hasInvalidControls() &&
        !getAnalysisSensitivityUnavailableReason(binding, row.specification.name)
          ? {
              binding,
              device: device!,
              registry: registry!,
              parameters,
              revisionKey,
              readInputs,
              isCurrent: () => !hasInvalidControls() && createResultIdentityForSensitivity(),
              onChoose: provenance => {
                changeParameter(row.specification.name, provenance.chosenValue!, 'live');
                sensitivitySearch.current = provenance;
                setResultContext(undefined);
                setSensitivitySelection(current => current + 1);
                setPreviewEnabled(true);
              }
            }
          : undefined
      }
      onChange={value => changeParameter(row.specification.name, value, row.apply)}
      onValidityChange={problem => reportControlProblem(row.specification.name, problem)}
    />
  );

  const [savedInvocations, setSavedInvocations] = useState(listCatalogInvocations());
  useEffect(
    () => subscribeCatalogInvocations(() => setSavedInvocations(listCatalogInvocations())),
    []
  );
  const loadSavedInvocation = (record: CatalogInvocationRecord) =>
    invoke(() => {
      invalidateConfiguration();
      setPreviewEnabled(false);
      const restored = parseCatalogInvocation(record.invocation);
      restoredSensitivity.current = record.sensitivity;
      setOperationId(restored.binding.id);
      setOverrides({...restored.invocation.parameters});
      setEngineChoice(record.engine);
      setDrafts(current => ({
        ...current,
        ...Object.fromEntries(
          Object.entries(record.inputs).map(([name, input]) => [
            `${record.operationId}:${name}`,
            {
              mode: input.mode,
              text: input.mode === 'text' ? input.text : '',
              datasetId: input.mode === 'native' ? input.datasetId : '',
              field: input.mode === 'native' ? input.field : ''
            }
          ])
        )
      }));
      clearResult();
    });

  const getDraft = (name: string): InputDraft =>
    drafts[`${operationId}:${name}`] ?? EMPTY_CATALOG_INPUT_DRAFT;
  const setDraft = (name: string, patch: Partial<InputDraft>) => {
    invalidateConfiguration();
    setPreviewEnabled(false);
    setDrafts(current => {
      const previous = current[`${operationId}:${name}`] ?? EMPTY_CATALOG_INPUT_DRAFT;
      const next = {...previous, ...patch};
      if (
        !Object.hasOwn(patch, 'profileDeclarations') &&
        (patch.mode !== undefined || patch.datasetId !== undefined || patch.text !== undefined)
      )
        delete next.profileDeclarations;
      // Shared publication roles seed a newly chosen source; subsequent explicit mapping edits stay.
      if (patch.datasetId !== undefined || patch.field === makeCatalogPointColumns('', '')) {
        const entry = analysisDatasetContext.get(next.datasetId);
        const columns = entry?.profile.pointColumns;
        if (columns) next.field = makeCatalogPointColumns(columns.longitude, columns.latitude);
        else if (patch.field !== makeCatalogPointColumns('', ''))
          next.field = entry?.dataset.geometryColumn ?? '';
      }
      return {...current, [`${operationId}:${name}`]: next};
    });
  };
  const selectOperation = (id: string) => {
    invalidateConfiguration();
    setPreviewEnabled(false);
    setOperationId(id);
    setOverrides({});
    clearResult();
    setError('');
  };
  const readInputs = async (): Promise<Record<string, CatalogFeatureCollection>> => {
    const samples = getCatalogSampleInputs(binding.id);
    const inputs: Record<string, CatalogFeatureCollection> = {};
    for (const spec of binding.inputs) {
      const draft = getDraft(spec.name);
      if (draft.mode === 'sample')
        inputs[spec.name] = parseCatalogFeatureCollection(samples[spec.name]);
      else if (draft.mode === 'text') inputs[spec.name] = parseCatalogFeatureCollection(draft.text);
      else {
        const resident = residentDatasets.find(dataset => dataset.id === draft.datasetId);
        const source = inputDatasets.find(dataset => dataset.id === draft.datasetId);
        inputs[spec.name] = await readCatalogNativeInput({
          resident,
          source,
          field: draft.field,
          isTrajectory: isTrajectoryCatalogOperation(operationId, spec.name),
          parameters,
          maximumFeatures: spec.maximumFeatures
        });
      }
    }
    return inputs;
  };
  const checkSelectedInputs = async () => {
    const selection = latestInputSelection.current;
    setCheckedInputs(undefined);
    setChecking(true);
    try {
      const inputs = await readInputs();
      const profiles = Object.fromEntries(
        binding.inputs.map(input => [
          input.name,
          profileAnalysisCatalogInput(binding, input.name, inputs[input.name], {
            parameters,
            executionEngine: engineChoice === 'cpu' || !device ? 'cpu' : 'gpu'
          })
        ])
      );
      const latest = latestInputSelection.current;
      if (
        latest.drafts !== selection.drafts ||
        latest.operationId !== selection.operationId ||
        latest.overrides !== selection.overrides ||
        latest.engineChoice !== selection.engineChoice ||
        latest.device !== selection.device ||
        latest.sourceRevisions !== selection.sourceRevisions ||
        latest.profileVersion !== selection.profileVersion ||
        latest.sourceIdentities.length !== selection.sourceIdentities.length ||
        latest.sourceIdentities.some(
          (source, index) => source !== selection.sourceIdentities[index]
        )
      )
        throw new Error('Inputs changed during checking. Check the current selection again.');
      setCheckedInputs({...selection, profiles});
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setChecking(false);
    }
  };
  const invoke = async (action: () => void | Promise<void>) => {
    try {
      await action();
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const run = async (automatic = false) => {
    if (
      hasInvalidControls() ||
      form.status !== 'ready' ||
      form.problems.length > 0 ||
      checking ||
      (automatic && !previewAvailable)
    )
      return;
    cancelPendingRun();
    const sequence = runSequence.current;
    const controller = new AbortController();
    runController.current = controller;
    const resultIdentity = createResultIdentity();
    const identity = createAnalysisRunIdentity({
      signal: controller.signal,
      getRuntime: () =>
        getCurrentRegistry
          ? getCurrentRegistry()
          : (latestExecution.current.registry ?? latestExecution.current.device),
      getRevisionKey: () =>
        JSON.stringify([latestExecution.current.revisionKey, latestExecution.current.parameterKey]),
      isCurrent: () =>
        sequence === runSequence.current && !hasInvalidControls() && resultIdentity.isCurrent()
    });
    const isCurrent = identity.isCurrent;
    const search = sensitivitySearch.current ?? restoredSensitivity.current;
    const sensitivity =
      search &&
      (search.revisionKey === revisionKey || search === restoredSensitivity.current) &&
      search.chosenValue === parameters[search.parameter]
        ? search
        : undefined;
    const reportContext = structuredClone({
      ...(sensitivity ? {parameterSearch: true} : {}),
      resolutions: form.resolved.resolutions,
      profiles: inputProfiles.profiles,
      fieldBindings: form.applicability.fieldBindings,
      warnings: form.applicability.warnings
    });
    const commit = (rawOutput: CatalogRunResult) => {
      if (!isCurrent()) return;
      const output = sensitivity ? {...rawOutput, sensitivity} : rawOutput;
      lastInputs.current = inputs;
      const baseReport = describeKeplerCatalogResult(
        output,
        binding,
        form.descriptor,
        reportContext
      );
      const report = {
        ...baseReport,
        chosen: [
          ...baseReport.chosen,
          ...(sensitivity
            ? [
                `Exploratory Radius search: ${sensitivity.values.join(', ')}; chosen ${sensitivity.chosenValue}. Population: ${sensitivity.population.scope}; selection ${sensitivity.population.selectionKey}.`
              ]
            : []),
          `Population: full selected inputs; ${Object.entries(reportContext.profiles)
            .map(
              ([name, profile]) =>
                `${name}: ${profile?.rowCount ?? 'unknown'} rows, dataset ${profile?.datasetId ?? 'unknown'}, revision ${profile?.revision ?? 'held source'}`
            )
            .join('; ')}.`
        ]
      };
      const model = createAnalysisResultCardModel(output, reportContext, {
        descriptor: form.descriptor,
        report,
        featureInputsOnly: true,
        executionEngine: device && engineChoice !== 'cpu' ? 'gpu' : 'cpu'
      });
      setCompletedRun({model, inputs, id: sequence});
      setResult(output);
      setResultContext({
        revisionKey,
        parameterKey,
        isCurrent: resultIdentity.isCurrent,
        profiles: inputProfiles.profiles,
        resolutions: form.resolved.resolutions
      });
      setError('');
    };
    let inputs: Record<string, CatalogFeatureCollection> = {};
    setRunning(true);
    try {
      inputs = await readInputs();
      if (!isCurrent()) return;
      if (
        sensitivity &&
        !matchCatalogInvocationInputs(
          parseCatalogInvocation(sensitivity.invocation).invocation,
          inputs
        )
      )
        throw new Error(
          'The recorded sensitivity population changed. Explore Radius again before keeping this search.'
        );
      // Saved with the workspace: normalized parameters, input hashes and where each input came from.
      if (!automatic || sensitivity)
        recordCatalogInvocation({
          operationId: binding.id,
          invocation: serializeCatalogInvocation(binding, parameters, inputs),
          inputs: Object.fromEntries(
            binding.inputs.map(spec => {
              const draft = getDraft(spec.name);
              return [
                spec.name,
                draft.mode === 'native'
                  ? {mode: 'native' as const, datasetId: draft.datasetId, field: draft.field}
                  : draft.mode === 'text'
                    ? {mode: 'text' as const, text: draft.text}
                    : {mode: 'sample' as const}
              ];
            })
          ),
          engine: engineChoice,
          ...(sensitivity ? {sensitivity} : {})
        });
      if (previewAvailable && previewSession.current) {
        const accepted = await previewSession.current.request({
          parameters,
          inputs,
          signal: controller.signal,
          isCurrent,
          commit
        });
        if (accepted && isCurrent()) setPreviewEnabled(true);
      } else {
        const output = await runCatalogOperationAsync(binding, parameters, inputs, {
          device: device ?? undefined,
          engine: engineChoice,
          signal: controller.signal
        });
        commit(output);
      }
    } catch (failure) {
      if (isCurrent()) setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (sequence === runSequence.current) setRunning(false);
    }
  };
  const controlsInvalid = hasInvalidControls();
  const latestRun = useRef(run);
  latestRun.current = run;
  useEffect(() => {
    if (!previewEnabled || !previewAvailable || resultCurrent || hasInvalidControls()) return;
    const timer = setTimeout(
      () => void latestRun.current(true),
      getAnalysisSheetPreviewDelay(lastApply.current)
    );
    return () => clearTimeout(timer);
  }, [
    parameterKey,
    revisionKey,
    previewEnabled,
    previewAvailable,
    controlsInvalid,
    sensitivitySelection
  ]);
  // Cancelling a request preserves the accepted result and stops automatic scheduling.
  const cancel = () => {
    cancelPendingRun();
    setPreviewEnabled(false);
    setRunning(false);
  };
  const loadLensScenarios = async () => {
    try {
      // Scenario modules register themselves on import; this is only paid when asked for.
      await import('@luma-contributors/lenses/computation-lenses/scenarios/index');
      const scenarios = listCatalogLensScenarios();
      setLensScenarios(scenarios);
      setLensScenarioId(current => current || scenarios[0]?.id || '');
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const runLensScenario = async () => {
    cancel();
    const sequence = runSequence.current;
    const resultIdentity = createResultIdentity();
    setRunning(true);
    try {
      // Lens scenarios dispatch GPU contributors on embedded datasets and need a device.
      if (!device) throw new Error('GPU device required');
      const userDataset = lensMapping.datasetId
        ? datasets.find(dataset => dataset.id === lensMapping.datasetId)
        : undefined;
      // With a table chosen, the mapped numeric fields stand in for the embedded dataset.
      const output = await runCatalogLensScenario(device, lensScenarioId, {
        ...(userDataset
          ? {userData: readLensUserData(userDataset as never, lensMapping.mapping)}
          : {})
      });
      if (sequence !== runSequence.current || !resultIdentity.isCurrent()) return;
      setResultContext({
        revisionKey,
        parameterKey,
        profiles: {},
        resolutions: [],
        isCurrent: resultIdentity.isCurrent
      });
      setCompletedRun(undefined);
      setResult({
        operationId: lensScenarioId,
        engine: 'gpu',
        dispatchedContributors: output.dispatchedContributors ?? [],
        parameters: {},
        tables: output.tables,
        diagnostics: output.diagnostics,
        summary: output.summary
      });
      setError('');
    } catch (failure) {
      if (sequence === runSequence.current)
        setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (sequence === runSequence.current) setRunning(false);
    }
  };
  const serialize = () =>
    invoke(async () => {
      if (hasInvalidControls())
        throw new Error('Correct invalid settings before serializing an invocation.');
      const inputs = await readInputs();
      lastInputs.current = inputs;
      setInvocationText(serializeCatalogInvocation(binding, parameters, inputs));
    });
  const restore = () =>
    invoke(() => {
      cancel();
      invalidateConfiguration();
      const restored = parseCatalogInvocation(invocationText);
      setOperationId(restored.binding.id);
      setOverrides({...restored.invocation.parameters});
      clearResult();
      if (!matchCatalogInvocationInputs(restored.invocation, lastInputs.current))
        setError('Parameters restored. Current inputs differ from the serialized invocation.');
    });
  const publish = (table: CatalogOutputTable) =>
    invoke(() => {
      if (hasInvalidControls())
        throw new Error('Correct invalid settings before keeping this result.');
      if (!resultCurrent || !resultContext?.isCurrent())
        throw new Error(
          'Settings or inputs changed. Run the full population before keeping this result.'
        );

      const currentRun = completedRun?.model.result === result ? completedRun.model : undefined;
      const presentation = currentRun
        ? currentRun.presentations.find(entry => entry.tableId === table.id)
        : undefined;
      const report = currentRun?.report;
      const staged = prepareAnalysisResultPublication(table, presentation, report);
      if (staged.analysisResult.destination === 'report') {
        showAnalysisResultReport({
          id: table.id,
          label: table.label,
          analysisResult: staged.analysisResult
        });
        return;
      }
      onPublish(
        staged.table,
        table.label,
        table.geometryColumn,
        getCatalogTrajectoryTripOptions(table),
        staged.analysisResult
      );
    });
  const openNext = (action: AnalysisResultNextAction) => {
    if (
      hasInvalidControls() ||
      !resultContext?.isCurrent() ||
      !completedRun ||
      action.applicability.status === 'blocked'
    )
      return;
    invalidateConfiguration();
    setPreviewEnabled(false);
    const next = createAnalysisResultNextDraft(completedRun.model, action);
    setDrafts(current => ({...current, ...next.drafts}));
    setOperationId(next.operationId);
    setOverrides(next.overrides);
    clearResult();
    setError('');
  };
  const keepRun = async (save: boolean) => {
    if (
      hasInvalidControls() ||
      !resultContext?.isCurrent() ||
      !completedRun ||
      completedRun.model.result !== result
    )
      return;
    const {getWorkbenchStore} = await import('./workflow/workbench-app');
    if (hasInvalidControls() || !resultContext?.isCurrent()) return;
    const store = getWorkbenchStore();
    const document = makeAnalysisResultWorkflow(
      completedRun.model,
      completedRun.inputs,
      store.registry
    );
    const entry = keepAnalysisResultWorkflow(document, store, save);
    setStackedRun({runId: completedRun.id, entry, store});
    store.setDock({open: true, inspectorTab: 'parameters'});
  };

  return (
    <section
      aria-label="Analysis catalog"
      data-testid="analysis-catalog"
      style={{borderTop: '1px solid #405168', marginTop: 18, paddingTop: 10}}
    >
      <h2 style={{fontSize: 16, marginBottom: 8}}>Contributor catalog</h2>
      <p className="hint" data-testid="catalog-summary">
        {runnableCount} of {entries.length} contributors are runnable here;{' '}
        {KEPLER_CATALOG_BINDINGS.length} bindings are registered. With a GPU device bindings
        dispatch the real graph contributors; the CPU reference engine is the oracle and the
        fallback when no device is available.{' '}
        <span data-testid="catalog-device">
          {device ? `GPU device: ${device.type}` : 'No GPU device'}
        </span>
      </p>
      <label htmlFor="catalog-engine">Engine</label>
      <select
        id="catalog-engine"
        value={engineChoice}
        onChange={event => {
          const next = event.target.value as 'auto' | 'cpu';
          if (next === engineChoice) return;
          cancel();
          invalidateConfiguration();
          setEngineChoice(next);
        }}
        style={{width: '100%'}}
      >
        <option value="auto">GPU when available</option>
        <option value="cpu">CPU reference</option>
      </select>
      <button
        type="button"
        aria-expanded={showBlocked}
        aria-controls="catalog-blocked-list"
        onClick={() => setShowBlocked(value => !value)}
      >
        {showBlocked ? 'Hide' : 'Show'} {blocked.length} not runnable
      </button>
      {showBlocked && (
        <ul
          id="catalog-blocked-list"
          data-testid="catalog-blocked"
          style={{maxHeight: 160, overflow: 'auto', fontSize: 12}}
        >
          {blocked.slice(0, 300).map(entry => (
            <li key={entry.contributorName} title={entry.reason}>
              {entry.contributorName} · {entry.capabilityId}
            </li>
          ))}
          <li>{blocked[0]?.reason}</li>
        </ul>
      )}
      <details data-testid="catalog-lens">
        <summary>Lens scenarios ({lensScenarios.length} loaded)</summary>
        <p className="hint">
          Registered lens scenarios run a contributor on an embedded dataset and show its recorded
          graph. Scenarios backed by an embedded dataframe dataset can instead run on one of your
          tables through a typed field mapping. Loading registers the scenario modules.
        </p>
        <button data-testid="catalog-lens-load" onClick={loadLensScenarios}>
          Load lens scenarios
        </button>
        {lensScenarios.length > 0 && (
          <>
            <select
              aria-label="Lens scenario"
              value={lensScenarioId}
              onChange={event => {
                setLensScenarioId(event.target.value);
                setLensMapping(EMPTY_LENS_MAPPING);
              }}
              style={{width: '100%'}}
            >
              {lensScenarios.map(scenario => (
                <option key={scenario.id} value={scenario.id}>
                  {scenario.element} · {scenario.title}
                </option>
              ))}
            </select>
            <AnalysisCatalogLensMapping
              scenarioId={lensScenarioId}
              datasets={datasets}
              draft={lensMapping}
              onChange={setLensMapping}
            />
            <button
              data-testid="catalog-lens-run"
              onClick={runLensScenario}
              disabled={running || !device}
            >
              Run scenario
            </button>
          </>
        )}
      </details>
      <label htmlFor="catalog-operation">Operation</label>
      <select
        id="catalog-operation"
        value={operationId}
        style={{width: '100%'}}
        onChange={event => selectOperation(event.target.value)}
      >
        {KEPLER_CATALOG_BINDINGS.map(candidate => (
          <option key={candidate.id} value={candidate.id}>
            {candidate.domain} · {candidate.title} [
            {candidate.engine === 'gpu'
              ? 'GPU'
              : candidate.engine === 'gpu-or-cpu'
                ? 'GPU/CPU'
                : 'CPU'}
            ]
          </option>
        ))}
      </select>
      <p className="hint">{binding.summary}</p>
      <p className="hint" data-testid="catalog-limits">
        Limits: {binding.limitation}
      </p>
      <details>
        <summary>Contract</summary>
        <ul style={{fontSize: 12}}>
          {binding.declaredBehavior.map(line => (
            <li key={line}>{line}</li>
          ))}
          <li>Limitation: {binding.limitation}</li>
          <li>Cleanup: {binding.cleanup}</li>
          <li>
            Outputs:{' '}
            {binding.outputs
              .map(
                output =>
                  `${output.label} (≤ ${output.maximumRows} rows; ${output.identityColumns.join(', ')})`
              )
              .join('; ')}
          </li>
          <li>Contributors mirrored: {binding.contributorNames.join(', ') || 'none'}</li>
          <li>GPU contributors dispatched: {binding.gpuContributorNames.join(', ') || 'none'}</li>
        </ul>
      </details>
      <AnalysisSheet>
        <AnalysisSensitivityCleanupStatus />
        <h2>{binding.title}</h2>
        {sheet.methodDifference && (
          <p data-testid="ax-method-difference">{sheet.methodDifference}</p>
        )}
        <section aria-label="Data">
          <h3>Data</h3>
          {binding.inputs.map(spec => {
            const draft = getDraft(spec.name);
            const selectedDataset = geometryDatasets.find(
              item => item.dataset.id === draft.datasetId
            );
            return (
              <fieldset key={spec.name} data-catalog-input={spec.name} style={{marginTop: 8}}>
                <legend>
                  {spec.label} ({spec.accepts.join('/')}, ≤ {spec.maximumFeatures} features, ≤{' '}
                  {spec.maximumVertices} vertices)
                </legend>
                {inputProfiles.profiles[spec.name] && (
                  <AnalysisRowMeaning
                    profile={inputProfiles.profiles[spec.name]!}
                    onDeclare={rowSemantics => {
                      if (draft.mode === 'native' && analysisDatasetContext.get(draft.datasetId)) {
                        analysisDatasetContext.setDeclarations(draft.datasetId, {
                          ...analysisDatasetContext.getDeclarations(draft.datasetId),
                          rowSemantics
                        });
                      } else {
                        setDraft(spec.name, {
                          profileDeclarations: serializeKeplerDatasetProfileDeclarations({
                            ...(draft.profileDeclarations
                              ? parseKeplerDatasetProfileDeclarations(draft.profileDeclarations)
                              : {}),
                            rowSemantics
                          })
                        });
                      }
                    }}
                  />
                )}
                <select
                  aria-label={`${spec.label} source`}
                  value={draft.mode}
                  onChange={event =>
                    setDraft(spec.name, {mode: event.target.value as InputDraft['mode']})
                  }
                  style={{width: '100%'}}
                >
                  <option value="sample">Built-in sample</option>
                  <option value="text">GeoJSON text</option>
                  <option value="native">Native dataset</option>
                </select>
                {draft.mode === 'text' && (
                  <textarea
                    aria-label={`${spec.label} GeoJSON`}
                    rows={4}
                    value={draft.text}
                    onChange={event => setDraft(spec.name, {text: event.target.value})}
                    style={{width: '100%', boxSizing: 'border-box'}}
                  />
                )}
                {draft.mode === 'native' && (
                  <>
                    <select
                      aria-label={`${spec.label} dataset`}
                      value={draft.datasetId}
                      onChange={event =>
                        setDraft(spec.name, {datasetId: event.target.value, field: ''})
                      }
                      style={{width: '100%'}}
                    >
                      <option value="">Choose dataset</option>
                      {geometryDatasets
                        .filter(item => item.fields.length)
                        .map(item => (
                          <option key={item.dataset.id} value={item.dataset.id}>
                            {item.dataset.label}
                          </option>
                        ))}
                    </select>
                    <select
                      aria-label={`${spec.label} geometry field`}
                      value={
                        readCatalogPointColumns(draft.field)
                          ? makeCatalogPointColumns('', '')
                          : draft.field
                      }
                      onChange={event => setDraft(spec.name, {field: event.target.value})}
                      style={{width: '100%'}}
                    >
                      <option value="">Choose geometry field</option>
                      {(selectedDataset?.fields ?? [])
                        .filter(
                          field =>
                            field !== TRAJECTORY_POINT_FIELD ||
                            isTrajectoryCatalogOperation(operationId, spec.name)
                        )
                        .map(field => (
                          <option key={field} value={field}>
                            {readCatalogPointColumns(field) ? 'Point from numeric columns' : field}
                          </option>
                        ))}
                    </select>
                    {readCatalogPointColumns(draft.field) &&
                      ([0, 1] as const).map(axis => (
                        <label key={axis}>
                          {axis === 0 ? 'X / longitude column' : 'Y / latitude column'}
                          <select
                            aria-label={`${spec.label} ${axis === 0 ? 'X / longitude' : 'Y / latitude'} column`}
                            value={readCatalogPointColumns(draft.field)![axis]}
                            onChange={event => {
                              const columns = readCatalogPointColumns(draft.field)!;
                              columns[axis] = event.target.value;
                              setDraft(spec.name, {field: makeCatalogPointColumns(...columns)});
                            }}
                            style={{width: '100%'}}
                          >
                            <option value="">Choose numeric column</option>
                            {selectedDataset?.dataset.fields.map(field => (
                              <option key={field.name} value={field.name}>
                                {field.name}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                  </>
                )}
                {sheet.rows.filter(row => row.input === spec.name).map(renderParameter)}
              </fieldset>
            );
          })}
        </section>
        <section aria-label="Key settings">
          <h3>Key settings</h3>
          {sheet.basic.map(renderParameter)}
        </section>
        {savedInvocations.length > 0 && (
          <details data-testid="catalog-saved-invocations" open>
            <summary>
              Saved runs ({savedInvocations.length}, travel with the workspace bundle)
            </summary>
            <ul style={{fontSize: 12}}>
              {savedInvocations.map(record => (
                <li key={record.id}>
                  {record.id}{' '}
                  <button onClick={() => loadSavedInvocation(record)}>Load into form</button>
                </li>
              ))}
            </ul>
          </details>
        )}
        <section aria-label="Checks" data-testid="catalog-preflight">
          <h3>Checks</h3>
          <p role="status" data-testid="catalog-preflight-status">
            {form.status}
          </p>
          {form.capacities.map(capacity => (
            <p key={capacity.input} data-testid="catalog-input-capacity" className="hint">
              {capacity.label}: {capacity.rows ?? 'unknown'} / {capacity.maximumRows} rows;{' '}
              {capacity.vertices ?? 'unknown'} / {capacity.maximumVertices} vertices.
            </p>
          ))}
          <ul>
            {form.problems.map(problem => (
              <li key={problem}>Invalid input or setting: {problem}</li>
            ))}
            {form.selectionChecks.map(check => (
              <li key={check.input} data-state={check.state}>
                {check.state} ({check.evidence}): {check.message}. Remedy: {check.remedy}
              </li>
            ))}
            {form.applicability.requirements.map((requirement, index) => (
              <li
                key={`${requirement.input}:${requirement.id}:${index}`}
                data-state={requirement.state}
              >
                {requirement.input}: {requirement.state} ({requirement.evidence}) —{' '}
                {requirement.message}
                {requirement.remedy ? ` Remedy: ${requirement.remedy}` : ''}
              </li>
            ))}
            {form.missing.map(item => (
              <li key={item.what}>
                {item.what}. Remedy: {item.remedy}
              </li>
            ))}
            {form.applicability.warnings
              .filter(
                message =>
                  !dismissedWarnings.includes(JSON.stringify([revisionKey, parameterKey, message]))
              )
              .map(message => (
                <li key={message}>
                  Warning: {message}{' '}
                  <button
                    type="button"
                    aria-label={`Dismiss warning: ${message}`}
                    onClick={() =>
                      setDismissedWarnings(current => [
                        ...current,
                        JSON.stringify([revisionKey, parameterKey, message])
                      ])
                    }
                  >
                    Dismiss
                  </button>
                </li>
              ))}
            {form.applicability.notes.map(message => (
              <li key={message}>{message}</li>
            ))}
          </ul>
          {form.applicability.guessedFacts.length > 0 && (
            <p>
              Confirm guesses in Dataset meaning and analysis limits above, then review these
              checks. For GeoJSON text or a checked native mapping, choose the Coordinate system
              setting in this form to confirm its source units.
            </p>
          )}
          {binding.inputs.length > 0 && (
            <>
              <p className="hint">
                Check selected inputs to refine their profiles and network topology for these
                settings. Original source values are read within the displayed limits; changed
                mappings, settings or engines require a new check.
              </p>
              <button
                type="button"
                data-testid="catalog-check-inputs"
                onClick={checkSelectedInputs}
                disabled={running || checking}
              >
                {checking ? 'Checking inputs…' : 'Check selected inputs'}
              </button>
            </>
          )}
          {Object.keys(overrides).length > 0 && (
            <button
              type="button"
              onClick={() => {
                invalidateConfiguration();
                setOverrides({});
              }}
            >
              Reset edits to resolved defaults
            </button>
          )}
        </section>
        <details data-testid="ax-all-settings">
          <summary>All settings</summary>
          {sheet.advanced.map(renderParameter)}
        </details>
        <AnalysisSheetPreviewScope
          available={previewAvailable}
          reason={
            previewAvailable
              ? 'Run once to enable retained preview updates. Rebuilds keep the previous result until replacement succeeds.'
              : !binding.planPreview
                ? 'Retained live preview is unavailable for this binding. Change settings, then Run.'
                : 'Retained preview requires the GPU engine and an active runtime. Change settings, then Run.'
          }
        />
        <p style={{marginTop: 8}}>
          <button data-testid="catalog-run" onClick={() => void run()} disabled={form.runDisabled}>
            {running ? 'Running' : 'Run'}
          </button>{' '}
          <button
            data-testid="ax-preview-cancel"
            onClick={cancel}
            disabled={!running && !previewEnabled}
          >
            Cancel preview
          </button>{' '}
          <button onClick={serialize} disabled={form.problems.length > 0 || checking}>
            Serialize invocation
          </button>{' '}
          <button
            data-testid="ax-clear-result"
            onClick={() => {
              cancel();
              invalidateConfiguration();
              clearResult();
              setPreviewGeneration(current => current + 1);
            }}
            disabled={!result}
          >
            Clear result
          </button>
        </p>
        {invocationText && (
          <>
            <textarea
              aria-label="Serialized invocation"
              rows={3}
              value={invocationText}
              onChange={event => setInvocationText(event.target.value)}
              style={{width: '100%', boxSizing: 'border-box'}}
            />
            <button onClick={restore}>Restore parameters</button>
          </>
        )}
        {error && (
          <p role="alert" className="error" data-testid="catalog-error">
            {error}
          </p>
        )}
        {result && (
          <div
            data-testid="catalog-result"
            data-current={resultCurrent}
            data-parameters={JSON.stringify(result.parameters)}
          >
            {!resultCurrent && (
              <p role="status">
                {controlsInvalid
                  ? 'Correct invalid settings before running or keeping this result.'
                  : 'Previous result. Run the current full population before keeping it.'}
              </p>
            )}
            {completedRun?.model.result === result && (
              <AnalysisResultCard
                key={completedRun.id}
                model={completedRun.model}
                actionsDisabledReason={
                  resultCurrent
                    ? undefined
                    : 'Correct settings and run the current full population before keeping this result.'
                }
                onNext={openNext}
                inStack={
                  stackedRun?.runId === completedRun.id &&
                  hasAnalysisResultStackEntry(stackedRun.entry, stackState?.workflows ?? [])
                }
                onAddToStack={() => keepRun(false)}
                onSaveWorkflow={() => keepRun(true)}
              />
            )}
            <p className="hint" data-testid="catalog-engine">
              Engine: {result.engine === 'gpu' ? 'GPU' : 'CPU reference'}
              {result.dispatchedContributors.length
                ? ` (${result.dispatchedContributors.join(', ')})`
                : ''}
              {result.fallbackReason ? ` · GPU not used: ${result.fallbackReason}` : ''}
            </p>
            <p className="hint" data-testid="catalog-result-summary">
              {Object.entries(result.summary)
                .map(([key, value]) => `${key}: ${value}`)
                .join(' · ')}
            </p>
            <ul style={{fontSize: 12}}>
              {result.diagnostics.map(item => (
                <li key={item.code}>
                  [{item.severity}] {item.code} ×{item.count}: {item.message}
                </li>
              ))}
            </ul>
            {result.tables.map(table => (
              <div key={table.id}>
                <strong>
                  {table.label}: {table.rowCount} rows
                </strong>{' '}
                <button
                  data-testid="catalog-publish"
                  disabled={table.rowCount === 0 || !resultCurrent}
                  onClick={() => publish(table)}
                >
                  Stage in Sources
                </button>
                <TablePreview table={table} />
              </div>
            ))}
            <p className="hint">
              Staged results appear in the Sources tab; publishing uploads them as a native dataset.
            </p>
          </div>
        )}
      </AnalysisSheet>
    </section>
  );
}

function TablePreview({table}: {table: CatalogOutputTable}) {
  const names = Object.keys(table.columnTypes).filter(name => name !== table.geometryColumn);
  return (
    <div style={{overflow: 'auto', maxHeight: 160}}>
      <table aria-label="Result preview" style={{fontSize: 11}}>
        <thead>
          <tr>
            {names.map(name => (
              <th key={name} scope="col">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from(
            {length: Math.min(table.id.endsWith('-diagnostics') ? 64 : 8, table.rowCount)},
            (_, row) => (
              <tr key={row}>
                {names.map(name => {
                  const value = table.columns[name][row];
                  return (
                    <td key={name}>
                      {typeof value === 'number' && !Number.isInteger(value)
                        ? value.toFixed(3)
                        : String(value ?? '')}
                    </td>
                  );
                })}
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}
