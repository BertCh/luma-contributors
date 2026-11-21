// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EMBEDDED_DATASET_LOADERS} from '../../embedded-dataset-loaders.js';
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createGPUComputationRecord,
  getGPUComputationAdapter,
  createGPUComputationCaptures,
  type CreateGPUComputationRecordProps,
  type GPUComputationCapture,
  type GPUComputationCaptureElementType,
  type GPUComputationRecordWithSidecars,
  type GPUComputationKeySpace,
  type GPUComputationRelation
} from '@luma-contributors/analysis/computation-record';
import {
  describeGPUCommandGraph,
  type GPUAnalysisChoice,
  type GPUCommandGraphDescription,
  type GPUComputationCaptureGeometry,
  type GPUComputationFormulaRef
} from '@luma-contributors/analysis/computation-record';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addLensViewSnapshot,
  LENS_RECORD_SOURCE_TEXT,
  LensCommandGraph,
  readLensViewSnapshot,
  submitLensGraph
} from '../../lens-view-snapshot';
import {getDataFrameOperatorEntry} from '../../../gpu-dataframe/computation-pack/pack';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import {type GPUComputationAxis} from '@luma-contributors/analysis/computation-record';

import type {LensDatasetResolver} from '../../types';
import {applyLensUserInput} from '../../user-input';
import {assertValidLensRecord, assertValidLensSidecars} from '../../validate-lens-record';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

/** Loads all Chicago tract columns, including nullable income. */
export function loadDataFrameLensData(
  name: 'f-chicago-tracts',
  resolver?: LensDatasetResolver
): Promise<typeof import('../../data/f-chicago-tracts.json')>;
/** Loads the deterministic sample of Chicago observations. */
export function loadDataFrameLensData(
  name: 'f-chicago-nature',
  resolver?: LensDatasetResolver
): Promise<typeof import('../../data/f-chicago-nature.json')>;
/** Loads the deterministic sample of NOAA weather stations. */
export function loadDataFrameLensData(
  name: 'f-ghcn-stations',
  resolver?: LensDatasetResolver
): Promise<typeof import('../../data/f-ghcn-stations.json')>;
/** Accepts a runtime choice of dataset as well as the precisely typed literal overloads. */
export function loadDataFrameLensData(
  name: 'f-chicago-tracts' | 'f-chicago-nature' | 'f-ghcn-stations',
  resolver?: LensDatasetResolver
): Promise<
  | typeof import('../../data/f-chicago-tracts.json')
  | typeof import('../../data/f-chicago-nature.json')
  | typeof import('../../data/f-ghcn-stations.json')
>;
/** Loads a deterministic local dataset only when a dataframe lens is activated. */
export async function loadDataFrameLensData(
  name: 'f-chicago-tracts' | 'f-chicago-nature' | 'f-ghcn-stations',
  resolver?: LensDatasetResolver
) {
  // While a host runs the scenario on user data (user-input.ts), mapped columns replace these.
  switch (name) {
    case 'f-chicago-tracts':
      return applyLensUserInput(
        (await EMBEDDED_DATASET_LOADERS['f-chicago-tracts.json']()).default,
        resolver
      );
    case 'f-chicago-nature':
      return applyLensUserInput(
        (await EMBEDDED_DATASET_LOADERS['f-chicago-nature.json']()).default,
        resolver
      );
    case 'f-ghcn-stations':
      return applyLensUserInput(
        (await EMBEDDED_DATASET_LOADERS['f-ghcn-stations.json']()).default,
        resolver
      );
  }
}

/** Creates the command graph a dataframe lens authors its example into. */
export function createDataFrameLensGraph<Parameters = void>(
  device: Device,
  id: string
): GPUCommandGraph<Parameters> {
  return new LensCommandGraph<Parameters>(device, {id});
}

/** Assembles a real dataframe run and its readbacks under the same deterministic run identity. */
export function createDataFrameLensRecord(props: {
  scenarioId: string;
  device: Device;
  graphs: CreateGPUComputationRecordProps['graphs'];
  formulas?: CreateGPUComputationRecordProps['formulas'];
  keySpaces: readonly GPUComputationKeySpace[];
  relations?: readonly GPUComputationRelation[];
  choices?: CreateGPUComputationRecordProps['choices'];
  captures: readonly GPUComputationCapture[];
}): GPUComputationRecordWithSidecars {
  const record = createGPUComputationRecord({
    id: props.scenarioId,
    adapter: getGPUComputationAdapter(props.device),
    domains: ['gpu-dataframe'],
    graphs: props.graphs,
    formulas: mergeGPUCoreFormulaRefsForGraphs(props.graphs, props.formulas),
    choices: props.choices,
    axes: {keySpaces: props.keySpaces, relations: props.relations ?? []},
    datasetVersion: `${props.scenarioId}:1`
  });
  return {
    ...record,
    captures: createGPUComputationCaptures({
      runId: record.runId,
      executionId: record.executionId,
      snapshots: props.captures
    })
  };
}

/** Tags actual GPU readback as a table column, without inventing geometry. */
export function createTableCapture(
  props: Pick<
    GPUComputationCapture,
    'id' | 'name' | 'keySpace' | 'role' | 'values' | 'elementType' | 'meaning'
  >
): GPUComputationCapture {
  return {...props, provenance: 'gpu-readback'};
}

/** One graph view a pair-statistics scenario snapshots, with the tags the result view reads. */
export type PairStatisticsCaptureSpec = {
  /** Packed input or output view; its capture name is the key it is registered under. */
  view: GraphDataView;
  /** Input, output or (for slices of an output that a formula symbol names) intermediate. */
  role: 'input' | 'output' | 'intermediate';
  /** Key space the rows index, declared in the record's `keySpaces`. */
  keySpace: string;
  /** Defaults to the view's 32-bit scalar type. */
  elementType?: GPUComputationCaptureElementType;
  /** Points geometry of projected positions. */
  geometry?: GPUComputationCaptureGeometry;
  /** Free-form note shown with the column. */
  meaning?: string;
};

/**
 * Adds an end-of-run snapshot tap to the authored graph, compiles and runs it once, and returns the
 * tagged readbacks with the compiled graph description. The tap adds one copy node (observer
 * effect); values are `gpu-readback`. The compiled graph is released before returning.
 */
export async function capturePairStatisticsViews(
  device: Device,
  graph: GPUCommandGraph<undefined>,
  graphId: string,
  specs: Readonly<Record<string, PairStatisticsCaptureSpec>>
): Promise<{captures: GPUComputationCapture[]; description: GPUCommandGraphDescription}> {
  const prepared = preparePairStatisticsViews(device, graph, graphId, specs);
  try {
    return {description: prepared.description, captures: await prepared.read()};
  } finally {
    prepared.destroy();
  }
}

/** Declares the complete observation graph; read() is the only submission/readback boundary. */
export function preparePairStatisticsViews(
  device: Device,
  graph: GPUCommandGraph<undefined>,
  graphId: string,
  specs: Readonly<Record<string, PairStatisticsCaptureSpec>>
) {
  const names = Object.keys(specs);
  const snapshot = addLensViewSnapshot(graph, device, {
    id: graph.id,
    views: Object.fromEntries(names.map(name => [name, specs[name].view])),
    elementTypes: Object.fromEntries(
      names.flatMap(name => {
        const elementType = specs[name].elementType;
        return elementType ? [[name, elementType]] : [];
      })
    )
  });
  let compiled: ReturnType<typeof graph.compile>;
  try {
    compiled = graph.compile();
  } catch (error) {
    snapshot.arena.destroy();
    throw error;
  }
  try {
    const description = describeGPUCommandGraph(compiled, {
      graphId,
      domain: 'gpu-dataframe',
      graph,
      sourceText: LENS_RECORD_SOURCE_TEXT
    });
    return {
      description,
      async read(): Promise<GPUComputationCapture[]> {
        submitLensGraph(device, compiled, {parameters: undefined});
        const readback = await readLensViewSnapshot(snapshot, {graphId});
        return readback.map(capture => {
          const spec = specs[capture.name];
          return {
            ...capture,
            role: spec.role,
            keySpace: spec.keySpace,
            ...(spec.geometry && {geometry: spec.geometry}),
            ...(spec.meaning && {meaning: spec.meaning})
          };
        });
      },
      destroy() {
        compiled.destroy();
        snapshot.arena.destroy();
      }
    };
  } catch (error) {
    compiled.destroy();
    snapshot.arena.destroy();
    throw error;
  }
}

/** Assembles choices and structure before reading a prepared pair-statistics graph. */
export async function runDataFrameLensRecord(
  props: Omit<Parameters<typeof createDataFrameLensRecord>[0], 'graphs' | 'captures'> & {
    prepared: ReturnType<typeof preparePairStatisticsViews>;
  }
): Promise<GPUComputationRecordWithSidecars> {
  try {
    const result = createDataFrameLensRecord({
      ...props,
      graphs: [props.prepared.description],
      captures: []
    });
    return {
      ...result,
      captures: createGPUComputationCaptures({
        runId: result.runId,
        executionId: result.executionId,
        snapshots: await props.prepared.read()
      })
    };
  } finally {
    props.prepared.destroy();
  }
}

/** Returns the curated pack formula of an element, or none when the pack declares it procedural. */
export function getDataFrameElementFormulas(element: string): GPUComputationFormulaRef[] {
  const formula = getDataFrameOperatorEntry(element)?.formula;
  return formula && 'tex' in formula ? [formula] : [];
}

/** Builds one recorded analyst choice; sweeping is off unless the caller says a sweep is meaningful. */
export function createDataFrameChoice(
  stage: string,
  name: string,
  value: GPUAnalysisChoice['value'],
  options: Pick<GPUAnalysisChoice, 'alternatives' | 'unit'> & {sweepable?: boolean} = {}
): GPUAnalysisChoice {
  const {sweepable = false, ...rest} = options;
  return {stage, name, value, sweepable, ...rest};
}

/** Inputs of {@link runDataFrameContributorLens}. */
export type DataFrameContributorLensProps = {
  /** Scenario id, also the record id and the prefix of the compiled graph id. */
  scenarioId: string;
  /** Exported element name; its pack entry supplies the formula. */
  element: string;
  /** Recording graph that already holds the contributor and its inputs. */
  graph: GPUCommandGraph<undefined>;
  /** Views to snapshot at the end of the run, tagged for the result view. */
  specs: Readonly<Record<string, PairStatisticsCaptureSpec>>;
  /** Every key space the captures use. */
  keySpaces: readonly GPUComputationKeySpace[];
  /** Key space a datum pin indexes; it becomes the record's data axis. */
  datumKeySpace: string;
  /** Relations between key spaces, for example the identity from invocation to row. */
  relations?: readonly GPUComputationRelation[];
  /** Analyst choices the example made, with alternatives from the exported constants. */
  choices: readonly GPUAnalysisChoice[];
  /** Dataset identity folded into the run id. */
  datasetVersion: string;
  /** Adds captures that are exact reshapes of the readback, for example one column of a table. */
  deriveCaptures?: (captures: readonly GPUComputationCapture[]) => readonly GPUComputationCapture[];
  /** Names after deriveCaptures reshapes readbacks; checked against the real derived sidecar. */
  derivedCaptureNames?: readonly string[];
};

/**
 * Compiles the recording graph, runs it once with an end-of-run snapshot of the tagged views, and
 * returns the validated record with its captures. The snapshot adds one submission after the run
 * (observer effect). Imported buffers stay caller-owned; compiled transients are released here.
 */
export async function runDataFrameContributorLens(
  device: Device,
  props: DataFrameContributorLensProps
): Promise<GPUComputationRecordWithSidecars> {
  const graphId = `${props.scenarioId}-compiled`;
  const prepared = preparePairStatisticsViews(device, props.graph, graphId, props.specs);
  try {
    const {description} = prepared;
    validateRequirement(
      props.scenarioId,
      !props.deriveCaptures || props.derivedCaptureNames !== undefined,
      'derivedCaptureNames must declare the names returned by deriveCaptures'
    );
    const datumSpace = props.keySpaces.find(keySpace => keySpace.id === props.datumKeySpace);
    const axes: GPUComputationAxis[] = [
      {
        id: 'entity',
        label: datumSpace?.label ?? props.datumKeySpace,
        kind: 'data',
        keySpace: props.datumKeySpace,
        ...(datumSpace?.size !== undefined && {size: datumSpace.size})
      }
    ];
    const record = createGPUComputationRecord({
      id: props.scenarioId,
      adapter: getGPUComputationAdapter(device),
      domains: ['gpu-dataframe'],
      graphs: [description],
      axes: {axes, keySpaces: props.keySpaces, relations: props.relations ?? []},
      formulas: mergeGPUCoreFormulaRefsForGraphs(
        [description],
        getDataFrameElementFormulas(props.element)
      ),
      choices: props.choices,
      datasetVersion: props.datasetVersion
    });
    // A malformed record must fail here instead of rendering a plausible but wrong card.
    assertValidLensRecord(record);
    const readback = await prepared.read();
    const snapshots = props.deriveCaptures ? props.deriveCaptures(readback) : readback;
    if (props.derivedCaptureNames)
      validateRequirement(
        props.scenarioId,
        JSON.stringify(snapshots.map(capture => capture.name).sort()) ===
          JSON.stringify([...props.derivedCaptureNames].sort()),
        'derivedCaptureNames must equal the names returned by deriveCaptures'
      );
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      executionId: record.executionId,
      snapshots: [...snapshots]
    });
    assertValidLensSidecars(record, {captures});
    return {...record, captures};
  } finally {
    prepared.destroy();
  }
}
