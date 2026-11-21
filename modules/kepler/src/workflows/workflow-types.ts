// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AnalysisSensitivityProvenance} from '../analysis/analysis-sensitivity';
import type {WorkflowAnalysisTemplate} from './workflow-analysis-template';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor
} from '../descriptors/operation-descriptor-types';
import type {KeplerWorkbenchCuratedPreset} from '../descriptors/workbench-adapters';
import type {KeplerOperationApplicability} from '../descriptors/applicability';
import type {KeplerBindingResultPresentation} from '../presentation/result-presentation';
import type {KeplerResultReport, KeplerResultReportContext} from '../presentation/result-report';
import type {CatalogRasterInput} from '../analysis/catalog-raster-input';
import type {Device} from '@luma.gl/core';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunResult
} from '../analysis/catalog-binding-types';
import type {CatalogDiagnostic, CatalogOutputTable} from '../analysis/catalog-geojson';

/**
 * Analysis workflows: a serialisable graph of steps over the catalog operations.
 *
 * One document is the source of truth. The canvas, the step list, the assistant draft and the
 * JSON and TypeScript exports are views over it. Every node is a step with an operation; data
 * enters through source operations and moves between steps as tables (a table may carry one
 * GeoJSON-text geometry column). Staleness is derived from content keys, never stored.
 */

export const WORKFLOW_SCHEMA_VERSION = 1;

/** Named limits enforced by the parser and the edit operations. Messages name the limit. */
export const WORKFLOW_LIMITS = {
  maximumWorkflows: 16,
  /** Workflows a user saved as reusable templates in one workspace. */
  maximumUserTemplates: 32,
  maximumSteps: 64,
  maximumInputs: 16,
  maximumIdCharacters: 48,
  maximumTitleCharacters: 80,
  maximumNoteCharacters: 500,
  /** Characters of one `text` parameter value (embedded GeoJSON of a source step). */
  maximumTextCharacters: 262144,
  /** Bounded raster ownership lists; parameter specs may impose a smaller limit. */
  maximumParameterNumbers: 65536,
  /** Characters of one serialised document. */
  maximumDocumentCharacters: 1048576,
  /** Rows a source operation may hand to the graph. */
  maximumSourceRows: 100000,
  maximumCanvasCoordinate: 100000
} as const;

// ---------------------------------------------------------------------------------------------
// Ports and data
// ---------------------------------------------------------------------------------------------

export type WorkflowGeometryKind = 'point' | 'line' | 'polygon';

/**
 * Kind of data on a port. `features` is geometry of a kind not known until run time (a source,
 * or an operation whose output geometry follows its input). `table` has no geometry column.
 * `raster` carries a bounded decoded band/grid independently of table rows.
 */
export type WorkflowPortKind = WorkflowGeometryKind | 'features' | 'table' | 'raster';

/** Static answer to "may this output feed this input"; `maybe` is settled when the step runs. */
export type WorkflowPortCompatibility = 'yes' | 'maybe' | 'no';

/**
 * Data handed between steps. Scalar/feature data uses plain column arrays, declared column types
 * and at most one GeoJSON-text geometry column. Raster data uses the separate band/grid payload
 * with a table for its id, label and optional display/provenance rows. `kind` determines which
 * representation to read; raster capacity is measured in cells independently of display rows.
 */
export type WorkflowData = {
  table: CatalogOutputTable;
  kind: WorkflowPortKind;
  /** Bounded decoded band/grid when kind=raster; table is an empty metadata shell. */
  raster?: CatalogRasterInput;
  /** AX plans from this exact output, reused by the result view and map publication. */
  analysisPresentation?: KeplerBindingResultPresentation;
  analysisReport?: KeplerResultReport;
  /** Exact catalog run and its reporting evidence, retained independently of later step edits. */
  analysisRun?: {result: CatalogRunResult; context: KeplerResultReportContext};
  /** Source profile snapshot, valid only for these immutable rows. */
  analysisProfile?: KeplerDatasetProfile;
  /**
   * Column whose value becomes the feature id when the table is read as features; row index when
   * absent. Downstream identity columns (`source_id`) then refer to this column.
   */
  identityColumn?: string;
};

// ---------------------------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------------------------

export type WorkflowParameterKind =
  | 'number'
  | 'integer'
  | 'boolean'
  | 'enum'
  | 'string'
  | 'number-list'
  /** Multi-line text up to `maximumLength` characters (embedded GeoJSON). */
  | 'text'
  /** Id of a host dataset; the form offers `WorkflowHost.listDatasets()`. */
  | 'dataset'
  /** One column name; the form suggests the columns of `columnsFrom`. Free text stays allowed. */
  | 'column'
  /** Comma-separated column names. */
  | 'column-list'
  /** Arithmetic or boolean expression over columns (see workflow-expression.ts). */
  | 'expression';

export type WorkflowParameterSpec = {
  name: string;
  label: string;
  kind: WorkflowParameterKind;
  description?: string;
  /** Physical unit; `crs` when it follows the declared coordinate system. */
  unit?: string;
  /** Canonical name of a retained saved spelling. */
  aliasFor?: string;
  /** Retired name of this parameter, used in validation messages. */
  renamedFrom?: string;
  nonzero?: boolean;
  /** The unscaled value and its product with this scale must each remain finite and nonzero as Float32. */
  nonzeroFloat32Scale?: number;
  coordinateModeUnits?: {geographic: string; planar: string};
  minimum?: number;
  maximum?: number;
  options?: readonly string[];
  maximumLength?: number;
  /** Absent when the catalog derives the value from its input at execution. */
  defaultValue?: CatalogParameterValue;
  /**
   * For `column`, `column-list` and `expression`: the input port whose columns are suggested, or
   * the name of a `dataset` parameter of the same operation. Absent: columns of every input.
   */
  columnsFrom?: string;
  /** The form hides the parameter unless another parameter currently has one of these values. */
  visibleWhen?: {parameter: string; equals: readonly CatalogParameterValue[]};
  fieldRole?: KeplerOperationFieldDescriptor;
  level?: 'basic' | 'advanced';
  apply?: 'live' | 'rebuild';
};

export type WorkflowInputPortSpec = {
  name: string;
  label: string;
  accepts: readonly WorkflowPortKind[];
  /** Scalar catalog ports reject known geometry; generic table operations can retain it. */
  geometryFree?: boolean;
  /** Table is an alternative to these geometry kinds, rather than a generic geometry carrier. */
  acceptsTable?: boolean;
  maximumRows: number;
  /** Cell capacity for raster inputs, independent of tabular row capacity. */
  maximumCells?: number;
  /** Vertex capacity of geometry inputs; absent for table inputs. */
  maximumVertices?: number;
  /** An optional input may stay unconnected. */
  optional?: boolean;
};

export type WorkflowOutputPortSpec = {
  /** Table id of the output; unique within the operation. */
  name: string;
  label: string;
  kind: WorkflowPortKind;
  /** Input port whose geometry kind this output preserves, including a geometry-free table. */
  kindFrom?: string;
  /** Columns that tie each row to its source feature and to its new identity. */
  identityColumns: readonly string[];
  maximumRows: number;
};

export type WorkflowOperationCategory =
  | 'source'
  | 'table'
  | 'geometry'
  | 'network'
  | 'aggregation'
  | 'interpolation'
  | 'density'
  | 'clustering'
  | 'raster'
  | 'analysis';

/** Engines an operation can use. `host` runs in the host application (sources). */
export type WorkflowEngineSupport = 'cpu-reference' | 'gpu-or-cpu' | 'gpu' | 'host';

/** Engine that produced a result. */
export type WorkflowEngine = 'cpu-reference' | 'gpu' | 'host';

export type WorkflowEngineChoice = 'auto' | 'cpu';

export type WorkflowRunContext = {
  /** GPU device of the host, null when none is available. */
  device: Device | null;
  engine: WorkflowEngineChoice;
  host: WorkflowHost;
  signal: AbortSignal;
  workflowId: string;
  stepId: string;
  /** Explicit document choices only; keeps derived defaults distinguishable from overrides. */
  parameterOverrides?: Readonly<Record<string, CatalogParameterValue>>;
  curatedAnalysisId?: string;
};

export type WorkflowOperationOutput = {
  curatedAnalysisId?: string;
  resolvedParameters?: Record<string, CatalogParameterValue>;
  /** One entry per declared output port, keyed by port name. */
  outputs: Record<string, WorkflowData>;
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
  engine: WorkflowEngine;
  dispatchedContributors: readonly string[];
  /** Why the GPU engine did not run when it could have. */
  fallbackReason?: string;
};

export type WorkflowOperation = {
  /** Actual adapted definition, including bindings outside the global catalog. */
  analysisBinding?: CatalogOperationBinding;
  analysisDescriptor?: KeplerOperationDescriptor;
  analysisPresets?: readonly KeplerWorkbenchCuratedPreset[];
  id: string;
  title: string;
  category: WorkflowOperationCategory;
  summary: string;
  /** `catalog`: adapted from a `CatalogOperationBinding`. `builtin`: defined by the workflow module. */
  origin: 'catalog' | 'builtin';
  /** Part of every cache key; bump when results change for the same inputs and parameters. */
  version: string;
  engine: WorkflowEngineSupport;
  inputs: readonly WorkflowInputPortSpec[];
  parameters: readonly WorkflowParameterSpec[];
  outputs: readonly WorkflowOutputPortSpec[];
  capabilityIds: readonly string[];
  /** GPU contributor exports whose semantics the operation mirrors or dispatches. */
  contributorNames: readonly string[];
  declaredBehavior: readonly string[];
  limitation: string;
  /** Coerces and range-checks raw values; throws a `WorkflowError` naming the parameter and limit. */
  normalizeParameters(
    raw: Readonly<Record<string, unknown>>
  ): Record<string, CatalogParameterValue>;
  run(
    context: WorkflowRunContext,
    inputs: Readonly<Record<string, WorkflowData>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): Promise<WorkflowOperationOutput>;
  /**
   * Sources only: a key that changes when the external data the step reads changes (dataset
   * revision, selection revision). Called synchronously while keys are derived.
   */
  getExternalRevisionKey?(
    host: WorkflowHost,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): string;
};

export type WorkflowOperationRegistry = {
  list(): readonly WorkflowOperation[];
  get(operationId: string): WorkflowOperation | undefined;
};

// ---------------------------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------------------------

export type WorkflowDocument = {
  schemaVersion: typeof WORKFLOW_SCHEMA_VERSION;
  /** Unique among the workflows of a workspace. */
  id: string;
  title: string;
  description?: string;
  /** Parameters promoted to the workflow level; steps reference them by id. */
  inputs: WorkflowInput[];
  steps: WorkflowStep[];
  settings: WorkflowSettings;
};

export type WorkflowSettings = {
  /** Default engine of every step; a step may override it. */
  engine: WorkflowEngineChoice;
  /** Re-run stale steps after an edit or an external change (debounced). Off by default. */
  autoRun: boolean;
};

/** A workflow-level parameter. Its kind, range and options constrain every step that binds it. */
export type WorkflowInput = {
  id: string;
  label: string;
  kind: 'number' | 'integer' | 'boolean' | 'enum' | 'string' | 'number-list';
  value: CatalogParameterValue;
  unit?: string;
  minimum?: number;
  maximum?: number;
  options?: readonly string[];
};

export type WorkflowStep = {
  /** Historical search evidence retained when a result is kept; never changes execution. */
  sensitivity?: AnalysisSensitivityProvenance;
  /** Curated registry identity of the preset used to configure this step. */
  curatedAnalysisId?: string;
  /** Unique within the document; `[a-z0-9][a-z0-9-]*`. */
  id: string;
  /** Unknown ids load as placeholders that cannot run, so saved maps survive catalog changes. */
  operationId: string;
  /** Display name; the operation title when absent. */
  label?: string;
  /** Input port name to the upstream output it reads. One connection per input port. */
  inputs: Record<string, WorkflowConnection>;
  /** Only parameters that differ from the operation default need an entry. */
  parameters: Record<string, WorkflowParameterBinding>;
  /** A disabled step does not run and blocks the steps that read it. */
  disabled?: boolean;
  /** Output port names shown on the map as host datasets. */
  display?: string[];
  /** Overrides `settings.engine`. */
  engine?: WorkflowEngineChoice;
  /** `stop` turns a GPU fallback into a step error. Default `warn`. */
  onFallback?: 'warn' | 'stop';
  /** How a displayed output is drawn, by output port name. Never affects a result key. */
  displayStyle?: Record<string, WorkflowDisplayStyle>;
  /** Canvas position; never read by execution. */
  position: {x: number; y: number};
  note?: string;
};

/** Colouring of a displayed output. Absent fields leave the host's default. */
export type WorkflowDisplayStyle = {
  /** Column that drives the fill colour. */
  colorColumn?: string;
  /**
   * `quantile` and `quantize` for numeric columns, `ordinal` for categories. Absent: the host
   * chooses from the column type.
   */
  colorScale?: 'quantile' | 'quantize' | 'ordinal';
  /** `diverging` suits signed statistics (z-scores); absent: sequential, or categorical for ordinal. */
  palette?: 'sequential' | 'diverging' | 'categorical';
};

export type WorkflowConnection = {stepId: string; output: string};

export type WorkflowParameterBinding =
  | {from: 'static'; value: CatalogParameterValue}
  | {from: 'input'; inputId: string};

/** The saved extension member: `settings.extensions.workflows` of the sandbox state. */
export type WorkflowCollection = {
  schemaVersion: typeof WORKFLOW_SCHEMA_VERSION;
  workflows: WorkflowDocument[];
  activeWorkflowId?: string;
  /**
   * Workflows the user saved as templates: reusable, parameterised by their workflow inputs, and
   * never run in place. Ids are distinct from each other (they may equal a workflow id).
   */
  templates?: WorkflowDocument[];
  analysisTemplates?: WorkflowAnalysisTemplate[];
};

// ---------------------------------------------------------------------------------------------
// Errors and issues
// ---------------------------------------------------------------------------------------------

export type WorkflowIssueCode =
  // document structure
  | 'malformed'
  | 'limit-exceeded'
  | 'duplicate-id'
  | 'cycle'
  // step configuration
  | 'unknown-operation'
  | 'unknown-input-port'
  | 'unknown-output-port'
  | 'unknown-step'
  | 'unknown-parameter'
  | 'unknown-workflow-input'
  | 'input-required'
  | 'incompatible-port'
  | 'unverified-port'
  | 'analysis-requirement'
  | 'parameter-out-of-range'
  | 'capacity-may-exceed'
  | 'upstream-disabled'
  | 'upstream-not-configured'
  | 'display-unknown-output'
  // run time
  | 'capacity-exceeded'
  | 'geometry-required'
  | 'geometry-kind-not-accepted'
  | 'gpu-fallback'
  | 'gpu-required'
  | 'dataset-not-found'
  | 'dataset-read-failed'
  | 'host-capability-missing'
  | 'operation-failed'
  | 'publish-failed'
  | 'cancelled';

export type WorkflowIssue = {
  severity: 'error' | 'warning' | 'info';
  code: WorkflowIssueCode;
  /** Starts with the step label when the issue belongs to a step, and names the prop and limit. */
  message: string;
  stepId?: string;
  /** Input or output port name. */
  port?: string;
  parameter?: string;
};

/** Thrown by the parser, the edit operations and operations; carries the same fields as an issue. */
export class WorkflowError extends Error {
  override readonly name = 'WorkflowError';
  constructor(
    readonly code: WorkflowIssueCode,
    message: string,
    readonly details: {stepId?: string; port?: string; parameter?: string; cause?: unknown} = {}
  ) {
    super(message, details.cause === undefined ? undefined : {cause: details.cause});
  }
}

// ---------------------------------------------------------------------------------------------
// Planning and run state
// ---------------------------------------------------------------------------------------------

/**
 * - `not-configured`: an error issue stops the step from running (missing input, bad parameter,
 *   unknown operation).
 * - `blocked`: configured, but an upstream step is disabled, not configured or failed.
 * - `ready`: can run, has no result.
 * - `queued` / `running`: part of the run in progress.
 * - `done`: holds a result for its current key.
 * - `stale`: holds a result for an older key.
 * - `error`: the last run of the current key failed.
 * - `disabled`: switched off by the user.
 */
export type WorkflowStepStatus =
  | 'not-configured'
  | 'blocked'
  | 'ready'
  | 'queued'
  | 'running'
  | 'done'
  | 'stale'
  | 'error'
  | 'disabled';

export type WorkflowStepResult = {
  sensitivity?: AnalysisSensitivityProvenance;
  curatedAnalysisId?: string;
  /** Content key the result was computed for. */
  key: string;
  outputs: Record<string, WorkflowData>;
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
  engine: WorkflowEngine;
  dispatchedContributors: readonly string[];
  fallbackReason?: string;
  /** Normalised parameters the run used (workflow inputs resolved). */
  parameters: Record<string, CatalogParameterValue>;
  durationMilliseconds: number;
  /** True when the result came from the node cache without running the operation. */
  cacheHit: boolean;
  /** Epoch milliseconds. */
  completedAt: number;
};

export type WorkflowPublishedOutput = {datasetId: string; label: string; key: string};

export type WorkflowStepState = {
  status: WorkflowStepStatus;
  /** Current content key; null while the step or anything upstream cannot be keyed. */
  key: string | null;
  issues: WorkflowIssue[];
  /** Last successful result; compare `result.key` with `key` to tell done from stale. */
  result?: WorkflowStepResult;
  /** Failure of the last run for the current key. */
  error?: WorkflowIssue;
  /** Host datasets published for displayed outputs, by output port name. */
  published?: Record<string, WorkflowPublishedOutput>;
};

/** Static plan of a document: order, per-step parameters, keys and issues. Pure and synchronous. */
export type WorkflowPlan = {
  /** Step ids in a deterministic topological order (ties by position.x, position.y, id). */
  order: string[];
  /** Document-level issues (cycle, limits). Step issues are under `steps`. */
  issues: WorkflowIssue[];
  steps: Record<string, WorkflowPlannedStep>;
};

export type WorkflowPlannedStep = {
  analysis?: KeplerOperationApplicability;
  stepId: string;
  operation?: WorkflowOperation;
  /** Normalised parameters with workflow inputs resolved; absent when normalisation failed. */
  parameters?: Record<string, CatalogParameterValue>;
  engine: WorkflowEngineChoice;
  /** Null when the step or anything upstream cannot run. */
  key: string | null;
  /** `not-configured`, `blocked`, `disabled` or `ready`; the runner refines `ready`. */
  status: Extract<WorkflowStepStatus, 'not-configured' | 'blocked' | 'disabled' | 'ready'>;
  issues: WorkflowIssue[];
  upstream: string[];
  downstream: string[];
};

export type WorkflowRunRecordStep = {
  sensitivity?: AnalysisSensitivityProvenance;
  stepId: string;
  operationId: string;
  key: string;
  status: 'done' | 'error' | 'cancelled' | 'skipped';
  cacheHit: boolean;
  engine?: WorkflowEngine;
  fallbackReason?: string;
  dispatchedContributors?: readonly string[];
  durationMilliseconds: number;
  /** Rows of each output. */
  outputRows?: Record<string, number>;
  /** Upstream edges read by this step: input port, producing step, output and its key. */
  inputs: Array<{port: string; stepId: string; output: string; key: string; rowCount: number}>;
  error?: WorkflowIssue;
};

/** Provenance of one run, in execution order. Serialisable. */
export type WorkflowRunRecord = {
  schemaVersion: 1;
  runId: string;
  /** Persisted unique execution identity; absent only in saved legacy runs. */
  executionId?: string;
  /** Computational dependency recorded at execution time; absent on historical runs. */
  dependencyRevision?: string;
  workflowId: string;
  /** Step the run was asked to reach; absent for a whole-workflow run. */
  target?: string;
  startedAt: number;
  durationMilliseconds: number;
  status: 'done' | 'error' | 'cancelled';
  steps: WorkflowRunRecordStep[];
};

export type WorkflowRunnerState = {
  /** Increases on every change a subscriber could observe. */
  revision: number;
  running: boolean;
  /** Step currently executing. */
  runningStepId?: string;
  order: string[];
  issues: WorkflowIssue[];
  steps: Record<string, WorkflowStepState>;
  lastRun?: WorkflowRunRecord;
};

export type WorkflowRunOptions = {
  /** Run only this step and what it needs ("run to here"). */
  target?: string;
  /** Ignore cached results for the steps of this run. */
  force?: boolean;
};

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------

export type WorkflowDatasetField = {
  name: string;
  /** Host field type name (`real`, `integer`, `string`, `geojson`, `timestamp`, ...). */
  type: string;
};

export type WorkflowDatasetProfile = {
  id: string;
  label: string;
  rowCount: number;
  /** GPU-resident datasets are read back through a bounded extraction. */
  resident: boolean;
  fields: readonly WorkflowDatasetField[];
  /** Fields whose values read as GeoJSON geometry. */
  geometryFields: readonly string[];
  /** Best guess of the point coordinate columns, when the dataset has them. */
  pointColumns?: {longitude: string; latitude: string};
};

export type WorkflowDatasetRequest = {
  datasetId: string;
  /** Label recorded with the step; used to find the dataset again after a workspace replay. */
  datasetLabel?: string;
  geometry:
    | {mode: 'geojson-field'; field: string}
    | {mode: 'point-columns'; longitude: string; latitude: string}
    | {mode: 'none'};
  /** `selected` reads only the rows that pass the host's current filters and selection. */
  population: 'all' | 'selected';
  maximumRows: number;
};

export type WorkflowPublishRequest = {
  workflowId: string;
  workflowTitle: string;
  stepId: string;
  output: string;
  label: string;
  data: WorkflowData;
  /** Earlier output; the host may return a new id and retires this one after the successor is ready. */
  previousDatasetId?: string;
  /** Requested colouring; a host that cannot apply it publishes with its default. */
  style?: WorkflowDisplayStyle;
  /** Synchronous execution identity check; hosts must check before committing an async publication. */
  isCurrent?: () => boolean;
};

/** A table offered to a SQL query under a fixed name (`input1`, `input2`, `input3`). */
export type WorkflowSqlRequest = {
  /** One read-only `SELECT` or `WITH` statement. */
  query: string;
  tables: Readonly<Record<string, WorkflowData>>;
  maximumRows: number;
};

/** A shape the user drew on the host's map. */
export type WorkflowDrawing = {
  id: string;
  label: string;
  kind: WorkflowGeometryKind;
};

/** What a host application supplies. Everything else in the module is host-independent. */
export type WorkflowHost = {
  getDevice(): Device | null;
  /** Required for replaceable runtimes: changes on replacement/loss even if device/data are unchanged. */
  getRuntimeRevisionKey?(): string;
  listDatasets(): readonly WorkflowDatasetProfile[];
  /** Shared, revision-checked AX service; missing evidence is never silently certified. */
  getAnalysisProfile?(datasetId: string): KeplerDatasetProfile | undefined;
  /** Synchronous; changes whenever a read of the same request would return different rows. */
  getDatasetRevisionKey(request: WorkflowDatasetRequest): string;
  readDataset(request: WorkflowDatasetRequest, signal: AbortSignal): Promise<WorkflowData>;
  /** Shows a step output on the map as a host dataset. Absent: display flags are ignored. */
  publishOutput?(
    request: WorkflowPublishRequest,
    signal: AbortSignal
  ): Promise<{
    datasetId: string;
    label: string;
    /**
     * Optional two-phase handoff. The caller checks currentness, then calls commit without
     * awaiting between that check and acceptance. A stale request throws synchronously;
     * the caller retracts the unaccepted successor. The returned promise only retires the
     * predecessor: cancellation or cleanup failure after acceptance must keep the successor.
     */
    commit?: () => Promise<void>;
  }>;
  /** Removes a dataset published by `publishOutput`. */
  retractOutput?(datasetId: string): Promise<void>;
  /**
   * Runs one read-only SQL statement over the given tables and returns the result as a table (a
   * geometry column is kept when the query selects one). Absent: the SQL operation cannot run.
   */
  runSql?(request: WorkflowSqlRequest, signal: AbortSignal): Promise<WorkflowData>;
  /** Shapes drawn on the map. Absent: the drawing source cannot run. */
  listDrawings?(): readonly WorkflowDrawing[];
  /** The drawn shapes as features (`drawing_id` identity column, `label` column). */
  readDrawings?(drawingIds: readonly string[] | 'all', signal: AbortSignal): Promise<WorkflowData>;
  /** Synchronous; changes whenever `readDrawings` would return different features. */
  getDrawingRevisionKey?(): string;
  /** Moves the map to a dataset published by `publishOutput`. */
  focusOutput?(datasetId: string): void;
};
