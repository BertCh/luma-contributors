# Kepler analysis workflows: research and design (2026-11-14)

Question from the maintainer: Kepler has no UI for analysis workflows; what is the best way to support user-defined workflows and composing functions into them?

Scope: research and design only. Nothing here is implemented. "Verified" means a source file, built output or primary document was read. "Inferred" means a conclusion drawn from those reads that nothing has exercised. External results came through search and fetch summaries, so exact UI details need a recheck against the primary documentation before they drive a design.

## 1. Problem and what exists today

Analysis in the Kepler 3.3 host (`examples/deck/kepler-compute`) is reached one operation at a time:

- **Layers > Analysis** is a details section added by the replaced `LayerManager` factory (`src/native-workflows.tsx:17-153`, `src/map-container.tsx:342-484`). Verified.
- The catalog panel (`src/analysis-catalog-panel.tsx:118-186`) picks one binding, one input source per input (sample, pasted text, or a native dataset plus geometry field), runs it, and stages the result as a source candidate in the Sources tab (`src/app.tsx:496-512`). The user must confirm the publication there. Verified.
- A published catalog result becomes a GPU-resident dataset, and the catalog panel's input list excludes resident datasets (`app.tsx:500-502`). Inferred from those two facts: a catalog result cannot feed a second catalog operation without the user exporting and re-importing it.
- Catalog runs are saved as a flat list (at most 32, replaced wholesale on restore) under `settings.extensions.catalogInvocations` (`src/catalog-invocation-store.ts:10-130`). The records have no edges. Verified.
- The assistant router (`src/assistant-operation-router.ts`) reaches only the fixed recipe names of `runAnalysis`; no catalog or workflow tool exists. Verified by grep of `assistant-*.ts`.
- The only "workflow" content is documentation: the website workflows page models chains as figures built from independent runs, and the roadmap notes this (`dev-docs/roadmap.md:244`, WEB-HIW-03). `ComputationDrawer` was never built (`roadmap.md:196`).

What Kepler 3.3 itself offers as injection points (verified in `node_modules/@kepler.gl/components`):

- `SidePanelFactory` depends on `CustomPanelsFactory`, whose default is an empty `panels` array of `{id, label, iconComponent, component}`. The host does not provide it (`side-panel/custom-panel.d.ts`, `side-panel.js:71-101`).
- `ModalContainer` accepts string ids and `{id, template, modalProps}`; the host already adds a `residentMemory` modal (`src/resident-workflow-guards.tsx:15-75`).
- `BottomWidgetFactory` is time-widget oriented, not a general drawer.
- `KeplerGlSchema.getConfigToSave` builds the saved config by iterating the registered reducer schema keys and calling each schema's `save`, so a top-level key such as `workflows` is not saved (verified by reading `node_modules/@kepler.gl/schemas/dist/esm/schema-manager.js:155-168`; a save round trip was not run). Kepler's own Save Map and cloud-provider path therefore cannot carry a workflow.
- The host's own route for analysis state is `sandbox state` with an opaque `settings.extensions` member, validated whole before replay, with unknown members ignored (`src/sandbox-host-extensions.ts:15-99`). This route is host-only: it is produced by `saveSandbox` (`src/sandbox-state.ts:353-398`) and replayed by `replaySandbox`, not by Kepler's Save Map. The members are hard-coded, not a registry: `collectSandboxExtensions`, `validateSandboxExtensions` and `applySandboxExtensions` each name `catalogInvocations` and `streamedPointWeights` explicitly, so a `workflows` member needs edits to all three plus the bundle validation at `sandbox-state.ts:668,692` (verified).

## 2. Prior art lessons

| System | What it does well | Where it fails or costs | Lesson for Kepler | Sources |
| --- | --- | --- | --- | --- |
| ArcGIS geoprocessing history and ModelBuilder | Every run is a structured record (inputs, outputs, parameters, duration, messages); entries reopen with prior settings, copy as Python, and several selected entries become a connected chain in a model. Run entire, to a step, or step by step. Export to Python makes parameters function arguments | Iterators, logical tools and model-only tools do not export to Python | History first: promote logged runs to a workflow. Avoid control flow in v1, or design the exporter with it | https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/geoprocessing-history.html ; https://pro.arcgis.com/en/pro-app/latest/help/analysis/geoprocessing/modelbuilder/exporting-a-model-to-python.htm |
| QGIS Model Designer | Each parameter is a static value, an expression, a model input, or an upstream output. Outputs are exposed explicitly; steps can be deactivated; Python export; validation | Algorithms with an unknown number of outputs are unsupported; circular dependencies blocked; model must be saved before it appears | Four-way parameter binding maps onto `CatalogParameterSpec`. Fixed outputs per binding already satisfy the output restriction | https://docs.qgis.org/3.40/en/docs/user_manual/processing/modeler.html |
| CARTO Workflows | Canvas of components; variables; named and published versions; run via API, schedule, MCP tool; cache option for unchanged parameters | Compiles to SQL in the warehouse (inferred), so the sharing model does not transfer. Error display and run-to-here were not found in the docs read | Draft versus published split; a saved workflow registered as an assistant tool is a later option | https://docs.carto.com/carto-user-manual/workflows/version-history ; https://docs.carto.com/carto-user-manual/workflows/components |
| KNIME | Per-node status (not configured, configured, executed, error), reset downstream on change, execute selected, mandatory ports must be connected | Canvas costs screen space | Adopt the status vocabulary and downstream reset | https://docs.knime.com/latest/analytics_platform_user_guide/index.html |
| Hex | Dependencies inferred from code, edits re-run downstream only, a subset of inputs exposed in a simpler app view | Needs code literacy | Author in the workflow panel, surface a few parameters elsewhere; catalog dependencies are explicit so no inference is needed (inferred) | https://docs.hex.tech/product-docs/develop-logic/compute-model/reactive-execution |
| Google Earth Engine | Client builds a serializable expression DAG evaluated lazily for the visible region | Deferred execution can surprise on cost | A small serializable DAG is a proven model | https://developers.google.com/earth-engine/deferred_execution |
| Node-RED | Simple flow editor | Message passing of large payloads blocks or crashes; large editors are slow (forum reports) | Steps must pass references to resident tables, not data | https://discourse.nodered.org/t/how-much-is-big-is-node-red/68950 |
| Foursquare Studio, Felt | Form-per-operation plus SQL or chat; Felt exposes an MCP server | No user-composed graph found in either (Studio: none found; Felt: SDK analysis unconfirmed) | A saved, re-runnable chain inside the map is a gap (inferred) | https://docs.foursquare.com/analytics-products/docs/fsq-spatial-workbench-getting-started ; https://felt.com/blog/introducing-felt-mcp-server |
| Kepler AI assistant | Chat, function calling with type hints, ask for missing parameters | One tool call at a time; custom tools documented elsewhere (OpenAssistant) | Catalog parameter specs map onto tool schemas; plans should be reviewable | https://unpkg.com/kepler.gl@3.1.8/src/ai-assistant/README.md ; https://geodaai.github.io/openassistant/guide/tools |
| React Flow, Rete.js, LiteGraph | React Flow `@xyflow/react` is MIT, peer `react >=17`; Rete.js MIT; LiteGraph fork archived and licence not confirmed | React 19 and host-style compatibility of React Flow not tested | If a canvas is wanted, use React Flow as a view over the same spec | https://www.npmjs.com/package/@xyflow/react ; https://github.com/retejs/rete |

Cross-cutting findings that repeat: one serialisable step graph as source of truth with several views; code export fails on control flow; discoverability at hundreds of functions needs search and facets; errors need to attach to the failing step.

Not researched or unverified: Alteryx, Dataiku, Node-RED licence, Observable runtime internals, GEE sharing and Apps, QGIS history-to-code, KNIME Workflow Coach ranking, and any user studies of workflow-builder failures.

## 3. Internal assets and gaps

### Assets (all verified unless marked)

| Asset | What it gives a workflow | Source |
| --- | --- | --- |
| `CatalogOperationBinding` | Operation id, domain, `capabilityIds`, typed inputs (accepted geometry kind, maximum features and vertices), typed parameters (kind, unit, min, max, options, default), outputs (table id, geometry kind, identity columns, maximum rows), engine support, declared behaviour | `modules/kepler/src/analysis/catalog-binding-types.ts:56-225` |
| `normalizeCatalogParameters`, `validateCatalogInputs` | Range checks and capacity checks before any work | `catalog-bindings.ts:529-620` |
| `runCatalogOperationAsync` and `CatalogRunResult` | GPU run with CPU fallback, `engine`, `fallbackReason`, `diagnostics`, `summary` per run | `catalog-bindings.ts:650-733` |
| `serializeCatalogInvocation` / `parseCatalogInvocation` / `matchCatalogInvocationInputs` | Versioned record of operation, normalised parameters and input content hashes | `catalog-bindings.ts:735-800` |
| `listCatalogEntries` | 283 contributor entries joined to bindings by contributor name; 129 runnable through a binding, 153 not, with reasons (counts from built `modules/kepler/dist`, stale relative to the parallel binding effort) | `catalog-bindings.ts:831-927` |
| Ontology | 15 data kinds (points, lines, polygons, spatial-weights, surface, network, attributes ...) as operator inputs and outputs, and a query by kind | `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:314-374` |
| `KeplerAnalysisRecipe` | Revision-aware, strictly parsed, saved by field name; restore fails on stale dataset, selection or weights revision | `modules/kepler/src/recipes/analysis-recipe.ts:183-346,965-990` |
| Dataset registry request protocol | `beginRequest` leases inputs and records revisions; `publishResult` returns null when superseded, aborted, device epoch changed or an input revision changed; revision bumps cancel in-flight requests | `modules/kepler/src/runtime/dataset-registry.ts:131-139,498-560,609-700` |
| Point-weights cache | Content-keyed, refcounted, memory-reserved cache that the hot-spot graph imports; already a two-node DAG with a materialised edge | `examples/deck/kepler-compute/src/analysis-point-weights-cache.ts:25-325` |
| `ResidentResultRecipes` | Lineage with dependency order and cycle detection for join, spatial-join, suitability results | `examples/deck/kepler-compute/src/resident-result-recipes.ts:12-76` |
| Execution router | Per-request route (luma, duckdb, analysis-backend, unavailable) with reasons (precision, capacity, memory-budget) | `modules/kepler/src/execution/execution-router.ts:21-160` |
| `GPUComputationRecord` with `workflow?: GPUWorkflowRecord` | Ordered per-run trace of graph invocations; schema is fork-local | `modules/analysis/src/computation-record/computation-record-types.ts:487-522` |
| Composite `addXRecipe` functions | 23 pre-built chains (cluster-and-outline, hot-spot-analysis, change-of-support ...) with overridable inputs and outputs | `modules/analysis/src/gpu-spatial-analysis/recipes/recipe-utils.ts:1-70` |
| `settings.extensions` | A forward-compatible slot, validated whole before replay, carried by saved maps and bundles | `sandbox-host-extensions.ts:15-99`, `modules/kepler/src/recipes/sandbox-state.ts:12-60` |

### Gaps

1. No workflow document: no step edges, no step-output reference. Catalog parameters are scalars and inputs are GeoJSON from a manual source.
2. Two data models. Catalog bindings take `CatalogFeatureCollection` and return plain-array tables, with per-binding `maximumFeatures` values that range from 1 to 20,000 (a grep of `modules/kepler/src/analysis/catalog-*.ts` shows 20,000 six times, 5,000 four times, plus 1, 32 and 500) and 20,000 rows for native input (`analysis-catalog-input.ts:10`); GPU upload and readback happen per invocation (`catalog-gpu-session.ts:53-143`). Recipes and the resident runtime use GPU-resident columns with revisions and caps up to 2^24 rows. Chaining two catalog bindings today means readback, GeoJSON, upload at every edge. No table-to-FeatureCollection adapter was found.
3. Three type vocabularies: binding geometry kinds, ontology data kinds, website catalog data types. Only geometry kinds are on the bindings, so weights, rasters, networks and attribute columns cannot be ports.
4. Invocation hashes identify inputs but not upstream steps. The hash is a 32-bit FNV-1a hash of `JSON.stringify` of the collection (`catalog-bindings.ts:743-745`, `catalog-geojson.ts:157-164`, verified); that is too weak for cache identity over user data (inferred risk: a collision serves a wrong result silently).
5. No dependency-aware invalidation or re-run, and no structured error codes (messages are thrown strings).
6. The assistant cannot enumerate bindings or compose steps.
7. Share links support one fixture and at most one analysis (`sandbox-share.ts:14-68`).
8. Capabilities that are not catalog bindings (dataset-state panels such as Moran, kriging, OLS, cartogram) have no step form yet.
9. The parallel effort adds about 100 bindings; their shape (GeoJSON-in or also graph-adding) is not known.

## 4. Design options

Each option fixes five choices. The options differ mainly in how far they go beyond the existing invocation record.

### Option A: Recipe list over catalog invocations (linear, history-first)

- UI: a "Workflows" side-panel tab (`CustomPanelsFactory`) with an ordered step list; each step has a generated parameter form, a status badge, run, and "run to here". "Promote history to workflow" turns selected catalog runs into a chain.
- Spec: an ordered list of steps. A step input references a dataset or the output of an earlier step in the list. No fan-in beyond references.
- Execution: each step runs through `runCatalogOperationAsync`; intermediates are kept in memory as CPU tables and converted to feature collections for the next step.
- Persistence: new `settings.extensions.workflows` member.
- Assistant: proposes a list as a draft the user approves.
- Costs: smallest; no dependency; fits the narrow sidebar. Risks: inherits the 5,000 to 20,000 feature caps and a CPU hand-off at every edge, so it is useful for small and medium data only; a list cannot show branches.

### Option B: DAG document with several views and a hybrid execution model (recommended)

- UI: the same side-panel step list as the primary editor, a modal with a read-only or lightly editable graph (React Flow, lazy loaded) later, and a run trace view reusing existing DOT and trace exports.
- Spec: a DAG of steps with typed ports and four-way parameter binding (section 5.1). The list view is a topological rendering of the DAG.
- Execution: per-step node as the unit. A step fuses internally (as recipes do). Edges are materialised through the dataset registry request protocol, one request slot per step, with a planner that may fuse linear single-consumer chains later. Two data forms: a materialising adapter for existing GeoJSON bindings and a resident-native form for new bindings.
- Persistence: `settings.extensions.workflows`, strict whole-member validation, portable bundle for free.
- Assistant: `defineWorkflow` draft operation validated by the same parser, approved as a whole.
- Costs: needs a binding contract extension (port types, resident form), a hand-off decision, cache and invalidation layer. Risks: scope; depends on the parallel binding effort agreeing on the contract.

### Option C: Full canvas product (node editor first, whole-graph fusion)

- UI: a canvas as the primary editor, in a wide modal or docked panel.
- Spec: a graph with iterators, branches and nested sub-workflows.
- Execution: compile the whole workflow into one `GPUCommandGraph`.
- Costs and risks: the narrow Kepler sidebar is a poor canvas host; the prior art shows iterators and branches are exactly what stops code export; whole-graph fusion loses per-step cancellation, memory admission, engine mixing and incremental recompute, and each contributor is capped at 32,768 command nodes (`modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts:483-497`, `GPU_MAXIMUM_COMMAND_NODE_COUNT`; the original path was wrong) with no enforced total across a graph. Heavy up-front cost and no evidence users need it before A or B is in use. Not recommended.

Execution-model comparison (inferred, not benchmarked):

| Model | Strengths | Weaknesses |
| --- | --- | --- |
| Fuse the whole workflow | Best transient aliasing, no intermediate publication | Loses per-step stale rules, admission, engine mixing; one oversized step blocks everything; node cap |
| Materialise every step | Simplest invalidation, caching, provenance, CPU and GPU mixing; matches registry and weights cache | One reservation and compile per step even for trivial steps |
| Hybrid (recommended) | Consistent with how the repo already works (fusion inside recipes, materialised cross-step cache) | Needs a planner and a node-count and scratch budget |

## 5. Recommendation

Adopt Option B, delivered in the order of Option A: ship the list editor over a DAG-capable document first, keep the graph view optional. Keep one execution path and one serialised document; list, graph, assistant plan and JSON or TypeScript export are views over it.

### 5.1 Draft workflow spec (schema version 1)

```ts
type WorkflowDocument = {
  schemaVersion: 1;
  id: string;
  title: string;
  // Parameters surfaced to the user at run time, referenced by steps.
  inputs: WorkflowInput[];
  steps: WorkflowStep[];
  // Optional layout for a graph view; never read by execution.
  layout?: Record<string, {x: number; y: number}>;
};

type WorkflowInput = {
  id: string;
  kind: 'dataset' | 'number' | 'integer' | 'boolean' | 'enum' | 'string';
  label: string;
  default?: KeplerJSONValue;
  // For dataset inputs: accepted port types.
  accepts?: WorkflowPortType[];
};

type WorkflowStep = {
  id: string;                         // unique within the document
  kind: 'catalog' | 'recipe' | 'sql'; // catalog is the v1 target
  operationId: string;                // getCatalogBinding(id); unknown ids load disabled
  label?: string;
  disabled?: boolean;                 // QGIS-style deactivation
  inputs: Record<string, WorkflowInputBinding>;      // keyed by CatalogInputSpec.name
  parameters: Record<string, WorkflowParameterBinding>; // keyed by CatalogParameterSpec.name
  execution?: {
    engine?: 'auto' | 'gpu' | 'cpu';  // bounded by the binding's engine support
    onFallback?: 'warn' | 'stop';
    publish?: 'transient' | 'resident'; // resident = visible Kepler dataset
    run?: 'manual' | 'live';          // live is opt-in per step
  };
  // Staleness checks recorded at last successful run, never the reference itself.
  provenance?: {
    algorithmVersion: string;
    inputRevisions: Array<{ref: string; datasetId?: string; revision?: number; contentHash?: string}>;
    engine: 'gpu' | 'cpu';
    fallbackReason?: string;
    diagnostics?: string[];
    runId?: string;                   // joins to a GPUComputationRecord
  };
};

type WorkflowInputBinding =
  | {from: 'dataset'; datasetId: string; field?: string; population?: 'all' | 'selected'}
  | {from: 'workflow-input'; inputId: string}
  | {from: 'step'; stepId: string; output: string}   // CatalogOutputSpec.tableId
  | {from: 'text'; text: string}                     // mirrors CatalogInputReference
  | {from: 'sample'; sampleId: string};

type WorkflowParameterBinding =
  | {from: 'static'; value: KeplerJSONValue}
  | {from: 'workflow-input'; inputId: string}
  | {from: 'step'; stepId: string; output: string; column?: string};
```

Rules the parser and validator enforce at edit time and at construction, not per row:

- Strict parse; unknown fields rejected for the document; whole `workflows` member rejected on any error, matching `validateSandboxExtensions`.
- The graph is acyclic; step ids unique; every `from: 'step'` target exists and is upstream.
- Parameter values are checked by `normalizeCatalogParameters` for the bound operation; messages name the step, the prop and the limit, following the contributor contract.
- Port types must match: `CatalogOutputSpec.geometry` against `CatalogInputSpec.accepts`, and declared `maximumRows` against the downstream `maximumFeatures`. Port types should grow from geometry kinds to an `ontologyKind` drawn from `GPU_SPATIAL_ANALYSIS_DATA_KINDS` (new optional field on the input and output specs).
- Unknown `operationId` values load as disabled placeholders so saved maps survive catalog changes.
- A `from: 'step'` parameter binding is limited to scalar outputs declared by the producer; v1 may omit it (open decision 4).
- Expressions (QGIS-style) are not in v1.

### 5.2 UI placement in Kepler

- Primary: a "Workflows" tab in the Kepler side panel by providing `CustomPanelsFactory` in the host injector (`src/map-container.tsx`, next to the other `.provide` calls) with `panels: [{id: 'workflows', label: 'Workflows', iconComponent, component}]`, and a new `workflows` field on the `NativeWorkflows` context. No Kepler fork. The side panel is narrow, which suits a step list.
- Secondary: a wide modal following the `residentMemory` pattern for the graph view and run trace. A bottom drawer would mean replacing `BottomWidgetFactory`, so it is deferred.
- Palette: a searchable list built from `KEPLER_CATALOG_BINDINGS` and `listCatalogEntries` (title, summary, domain, `capabilityIds`, runnable status and reason), with a "compatible next step" filter on output kind against accepted input kinds.
- Step form: generated from `CatalogParameterSpec` (kind, unit, min, max, options, default); the existing hand-written `analysis-*-panel.tsx` forms are not reused.
- Status vocabulary: not configured, ready, running, done, stale, error; changing a step marks downstream steps stale; "run to here" and "run all"; each step shows engine, `fallbackReason` and diagnostics.
- Do not add a workflows key to Kepler's map config; persistence goes through the sandbox extension (section 5.4).

### 5.3 Execution model

Verified building blocks, inferred composition:

- Unit of execution is a step. Inside a step the binding or recipe fuses its own graph as today.
- Each step uses one registry request in a slot named for the step (for example `wf:<workflowId>:<stepId>`). Upstream revision bumps then cancel in-flight dependents, and stale publications are dropped by `publishResult`. The registry cancels requests but does not recompute dependents, so a small CPU dirty-flag layer marks dependents stale and recomputes in topological order.
  - Limit found in review (verified): `beginRequest` takes dataset ids only and leases registry datasets (`dataset-registry.ts:609-645`), and `publishResult` registers a new resident dataset (`:648-681`). The catalog panel today never calls the registry; it calls `runCatalogOperationAsync` directly (`analysis-catalog-panel.tsx:182`, grep shows `beginRequest` only in the resident-* runtimes). A step with `publish: 'transient'` produces no registry dataset, so cancellation and stale-drop only protect resident steps; transient intermediates need the CPU dirty-flag layer alone, and a downstream step cannot lease them. The design must say which.
- Node cache key: `(algorithmVersion, normalised parameters, engine, input dataset revisions, selection revision when population is selected, device epoch)`. Reuse the point-weights cache pattern (refcount, reservations, evict unreferenced non-latest entries), lifted from the example host into `modules/kepler`. For imported or serialised inputs use a stronger hash than the 32-bit text hash; for resident inputs key on `(datasetId, revision)`.
- Data hand-off. v1: a materialising adapter for current GeoJSON bindings (table to feature collection, within the binding's caps), priced and shown as a CPU boundary in the plan. Target: a resident-native binding form (typed column or dataset refs plus an `addX(graph, props)` function) so adjacent steps avoid readback. Which form the new bindings use is an open decision.
- Planner (later phase): fuse a linear chain only when every intermediate has one consumer, is not pinned visible, all steps use the GPU engine on the same device and population, and summed `estimatedCommandNodeCount` and compiled scratch stay under a workflow cap well below 32,768. Otherwise materialise. The decisions are shown as data.
- Pre-flight: before running, sum declared maximum rows and capacities and router byte estimates against the available budget; refuse at plan time with a message naming the step and the limit.
- Publication gate: one small status read per step (overflow, required count, invalid count) decides whether the result may feed downstream; an overflowed result is a step error, never silently passed on.
- Selection and live runs: selections are not saved as masks. Live re-run on filter or selection change is opt-in per step with debounce; the default is manual run for heavy steps. This is inferred; no prior art studied handles GPU-resident filter changes.
- Provenance: one `GPUWorkflowRecord` per run, one `GPUComputationRecord` graph per step, with edges (`step id` to upstream `step id`, dataset revision) as an extension of the ordered `invocations` list, plus engine, fallback and cache hit or miss per step. The record schema is fork-local, so this is allowed to change.

### 5.4 Persistence and sharing

- New member `settings.extensions.workflows` = `{schemaVersion: 1, workflows: WorkflowDocument[]}` with its own strict parser, versioned independently of the sandbox schema, and a store modelled on `catalog-invocation-store.ts`. It rides along with host sandbox saves and portable bundles only (corrected: not "saved maps"; Kepler's own Save Map drops it, see section 1). Dataset ids inside `from: 'dataset'` bindings must survive replay's dataset re-creation, which the document does not yet address.
- Last results are not stored in the document; provenance holds only staleness checks.
- The existing `catalogInvocations` list is migrated by "promote history to workflow", not deleted.
- Share links need a separate size-bounded workflow link or a lifted one-analysis restriction. Whether the Workflows tab appears in the standalone viewer is unexamined.

### 5.5 How the new catalog bindings plug in

- Steps reference `operationId` only through `getCatalogBinding`, so every binding registered in `KEPLER_CATALOG_BINDINGS` appears in the palette and gets a generated form and static validation with no UI change.
- The binding shape is the only stable contract (70 or more `catalog-*.ts` files and about 100 more bindings in flight). Two additive fields are requested of the parallel effort: an optional port type (`ontologyKind`) on input and output specs, and an optional resident-native form. Existing bindings keep working through the adapter.
- An explicit operation-id join between the binding, the ontology operator and the website catalog entry replaces name matching.
- Assistant: the same specs generate tool definitions (list operations, describe one, propose a workflow draft). The draft is a `WorkflowDocument` shown as a step list; approval is per workflow, following the router's per-operation `approved` guard. The host router already accepts published tool shapes; `@kepler.gl/ai-assistant` is not installed in this repository (verified), so this is a host-side operation, not a Kepler feature.
- Composite `addXRecipe` chains (hot spots, cluster-and-outline and others) are shipped as templates; a few should be decomposed into step definitions as a test of the model.

## 6. Phased plan

| ID | Work | Gate |
| --- | --- | --- |
| KE-WF-01 | Spike: add the missing table-to-feature-collection adapter and chain two existing catalog bindings in a node test; record the row caps and the readback cost per edge | A node spec runs buffer then dissolve (or an equivalent pair) end to end and states measured caps |
| KE-WF-02 | Port types: optional `ontologyKind` on `CatalogInputSpec` and `CatalogOutputSpec`; explicit operation-id join to ontology operators and website entries; agree the field with the parallel binding effort | `yarn typecheck`; a spec lists compatible next steps for three bindings; `yarn check-contributor-contract` unchanged |
| KE-WF-03 | `WorkflowDocument` types, strict parser and validator (acyclic, port types, parameter normalisation, unknown ids as disabled) in `modules/kepler` | Node specs for valid, cyclic, stale and unknown-id documents; `yarn test-node` |
| KE-WF-04 | Executor: per-step request slots, dirty-flag invalidation, node cache lifted from the point-weights cache, publication status gate, cancellation | Spec showing an upstream revision bump cancels and marks dependents stale; weights then hot spots expressed as a two-step workflow |
| KE-WF-05 | Persistence: `settings.extensions.workflows`, store, save and restore round trip, portable bundle inclusion | Headless round trip preserves a workflow; malformed member rejected whole without touching other extensions |
| KE-WF-06 | Workflows side-panel tab (`CustomPanelsFactory`): palette search, generated forms, status badges, run to here, "promote history to workflow" | Headless smoke: build and run a two-step workflow from the catalog; screenshot reviewed; independent review (passing specs alone were not enough on earlier batches) |
| KE-WF-07 | Pre-flight and diagnostics: peak byte and node-count estimate, CPU-boundary display, step-level errors naming prop and limit, `GPUWorkflowRecord` per run with edges | Estimate within a stated tolerance of measured scratch on three chains; trace opens in the existing computation views |
| KE-WF-08 | Resident-native binding form and the hybrid planner; measure fused versus materialised on three chains (weights to hot spots, standardise to weighted sum, join to group-by) | Measured GPU and compile timings recorded in this folder; planner thresholds set from them |
| KE-WF-09 | Assistant: generated tool definitions, `defineWorkflow` draft operation, whole-workflow approval | Router spec: invalid drafts rejected by the shared parser; draft shown as a step list before any run |
| KE-WF-10 | Templates from composite recipes and lens scenario packs; optional lazy React Flow graph view over the same document | React 19 and host style smoke for React Flow before adding the dependency; licence check against SPDX rules |
| KE-WF-11 | JSON export, then TypeScript export with parameters as function arguments | Exported TypeScript reproduces a saved workflow's result on a fixture |

Deferred until a use is shown: iterators, branching, nested workflows, batch over many layers, expressions, live re-execution on every filter change. If any is added, the exporter is designed in the same change.

## 7. Open decisions for the maintainer

1. Data hand-off between steps: CPU tables with the catalog caps in v1 (KE-WF-01 to KE-WF-07), or require a resident-native form before shipping the editor? This decides which operations can be chained at scale.
2. Binding contract: will the parallel binding effort add `ontologyKind` ports and an optional resident-native form now, or ship GeoJSON-in, table-out only?
3. Recipes (revision-aware, column based) and catalog bindings (GeoJSON based): two step kinds, or expose recipes through the binding registry?
4. Step-output parameter binding (`from: 'step'` on a parameter) and expressions: in v1, later, or never?
5. Visibility: is a workflow a host-only extension (recommended, `settings.extensions`), or should it appear in Kepler's `visState`; and does the standalone viewer and share link carry it?
6. Stale input policy on re-run: error (as `restoreKeplerAnalysisRecipe` does), warn (as catalog hash matching does), or re-run automatically?
7. Selection semantics for a step with population `selected`: recompute on every selection change, or hold the previous result marked stale?
8. Engine policy: per-step engine choice with a recorded fallback (recommended), or require a single engine per workflow; and whether a CPU step's result may be reused where a GPU step ran (the cache key currently includes the engine).
9. Graph view: is a canvas wanted at all in this phase? If so, approve React Flow (MIT) as a dependency after a React 19 smoke test.
10. Assistant scope: draft-only with whole-workflow approval (recommended), and whether saved workflows become assistant tools later.
11. Licensing: lenses and sample datasets carry non-commercial or share-alike terms (`THIRD-PARTY-DATA.md`); decide whether shipped workflow templates may reference sample data.
12. Hash for cache identity: replace the 32-bit text hash with a stronger digest for imported or serialised inputs.

## 8. Review notes (2026-11-14)

Changed in this review (all verified against the repo):

- Corrected the 32,768 command-node citation to `analysis-core/kernels/gpu-contributor-utils.ts:483-497`.
- Upgraded the Kepler config-drop claim from inferred to verified by reading `schema-manager.js:155-168`, and stated that Kepler's own Save Map and cloud path cannot carry workflows; only host sandbox saves and bundles can. Noted the extension members are hard-coded in three functions plus bundle validation.
- Replaced "about 5,000 features per binding" with the measured range (1 to 20,000).
- Named the hash as FNV-1a 32-bit and cited `catalog-geojson.ts:157-164`.
- Added the registry limit: the catalog path never uses `beginRequest`, and transient step outputs are not registry datasets, so the section 5.3 cancellation story applies only to resident steps.
- Corrected "saved maps" to "host sandbox saves" in 5.4.

Remaining concerns, not fixed:

- Gates without a number: KE-WF-06 ("screenshot reviewed"), KE-WF-07 ("stated tolerance"), KE-WF-08 ("planner thresholds set from them") and KE-WF-10 ("smoke") need a threshold, for example estimate within 25 percent of measured scratch, and a named fixture. KE-WF-05 should include a negative case showing Kepler Save Map drops the member, so the limit is documented rather than discovered.
- Settled-looking choices that are the maintainer's: Option B and the hybrid model are labelled "recommended" before any measurement (KE-WF-01 and KE-WF-08 are the evidence), and the `kind: 'sql'` step type appears in the spec without any section or decision covering it. Treat the hybrid planner as a hypothesis.
- Prior art missing (from general knowledge, not fetched, so recheck before citing): Power Query and Tableau Prep "applied steps" lists, which are the strongest precedent for a linear, history-first list with per-step preview; Grasshopper and Houdini or Blender node graphs, whose documented lesson is wiring sprawl and the need for groups or sub-graphs; Airflow and Dagster for dependency-driven re-run and asset lineage; Jupyter and Observable on hidden state and ordering. UX lessons not covered: per-step preview on the map before commit, undo and redo of workflow edits, progress and cancel display, and how a workflow behaves when its input dataset is replaced or filtered.
- The 129 runnable and 283 entry counts come from a stale `modules/kepler/dist`; the working tree has uncommitted catalog edits, so recount after the parallel effort.
- The repo's `AGENTS.md` pins luma.gl alpha.13 while `node_modules/@luma.gl/core` is alpha.14; the document already lists this as open.
- Lens scenario packs and the website workflows page are mentioned only in KE-WF-10; they are the nearest existing example of a multi-step chain and could inform the step model sooner.

Facts still to check before relying on them: that a Kepler save and load round trip drops unknown map-config keys (now read in code, still not run); React Flow under React 19 and the host styles; which adapters use `createExecution`; the binding and entry counts after the parallel effort lands; the luma.gl version actually installed (alpha.14 in `node_modules`, alpha.13 in `AGENTS.md`).

## 9. Experiments and build decisions (2026-11-14, later the same day)

Sections 1 to 8 are the design study. This section records what was run before the build started and which of the section 7 decisions the build took. The maintainer asked for an ambitious build borrowing from noodles.gl; the choices below follow the study's recommendations unless stated, and each can be reversed without changing the document format.

### 9.1 What was run

| Question | What was run | Result |
| --- | --- | --- |
| Does chaining catalog bindings work through a table-to-features adapter (KE-WF-01)? | A node probe against the built package: `random-points-in-polygon` then `voronoi-diagram`; `kernel-density-grid` then `hotspot-gi-star` | The first chain ran (10 points, 10 cells). The second was refused by the catalog (`Input feature capacity exceeded`): a 64 by 107 grid has 6,400 cells and Gi* accepts 3,000. Capacities differ between adjacent operations in ordinary chains, so the runner checks every hand-off against the consumer's declared capacity and names the limit |
| Does React Flow work under React 19 in the host (section 8, unverified)? | `@xyflow/react` 12.12.1 (MIT, peer `react >=17`) added to the example in a staging copy; a three-node canvas in a dock below the map; headless WebGPU Chromium | Renders in dark mode, nodes drag, the minimap and controls work, no page errors. The only console messages are Kepler's existing React 19 `element.ref` warnings |
| Does `CustomPanelsFactory` add a side-panel tab without a fork? | The same spike provided the factory with one panel | A fifth tab appears and renders its component. The component must come from `panels[].component`; rendering it from the factory element as well shows it twice |
| Can a dock sit below the map without replacing a Kepler factory? | The app passed `height - dockHeight` to `KeplerGl` and rendered the dock as a sibling | The map canvas resized to 720 of 1000 pixels and stayed interactive |
| Which bindings can chain? | Inventory of the 130 bindings in the built package | 92 take one input, 33 two, 5 three; 116 have one output, 14 two; outputs are 51 polygon, 31 line, 37 point and 25 without geometry. 105 string parameters name a property of the input (`valueProperty`, `groupProperty`, ...), so column suggestions from the upstream result matter. 11 bindings are GPU-only |

Not run: a save through Kepler's own Save Map (the member is dropped by construction, section 1); React Flow with more than a few dozen nodes; any measurement of a GPU-resident hand-off.

### 9.2 What noodles.gl contributed

Read from its repository documents (`AGENTS.md`, `dev-docs/architecture.md`, `dev-docs/tech-stack.md`; Apache-2.0; no code was copied):

- One document of nodes and edges with node positions inside it, and several editors over it.
- Houdini-style keys on the network editor: Tab adds a node, B bypasses, D displays, F frames.
- A display flag on a node decides what the map shows.
- Undo by document snapshot, with edits under the same tag inside 500 ms forming one step.
- Parameters are literals; noodles.gl also has expressions and references on parameters and a timeline, which this build leaves out.
- An assistant edits the document through tools; here it only proposes a draft.

What does not transfer: noodles.gl lowers its graph to SQL, CPU or GPU plans and routes value edits without recompiling. The catalog operations here are whole analyses with declared capacities, so the unit of execution is a step and the hand-off is a table.

### 9.3 Decisions taken for the build

| Section 7 decision | Taken |
| --- | --- |
| 1 Data hand-off | CPU tables within the catalog capacities. `WorkflowData` is the seam for a resident form |
| 2 Binding contract | Unchanged. Port kinds are derived by an adapter; `ontologyKind` is not added |
| 3 Recipes | Not a step kind. Resident analyses are reached by displaying a step output (it becomes a resident dataset) and by reading a resident dataset, or its current selection, as a source |
| 4 Step-output parameter binding, expressions | No. Parameters can be promoted to workflow inputs; expressions exist only inside the filter and calculate operations |
| 5 Visibility | Host-only extension `settings.extensions.workflows`. The standalone viewer and share links do not carry workflows |
| 6 Stale input policy | Derived from content keys. A stale result stays visible and is marked; re-run is manual unless auto-run is on |
| 7 Selection semantics | A source reading the selection is re-keyed when the selection changes and is stale until run |
| 8 Engine policy | Workflow default with a per-step override; the engine choice is part of the key of GPU-capable steps only |
| 9 Graph view | Yes: React Flow canvas in a bottom dock, plus the step list in the side panel |
| 10 Assistant scope | Draft only, approved as a whole |
| 11 Template data | Generated synthetic geometry only |
| 12 Hash | A 128-bit content hash replaces the 32-bit text hash for workflow keys; the catalog invocation hash is unchanged |

Differences from the draft spec in section 5.1: inputs are always connections from another step (data sources are steps), so `WorkflowInputBinding` has one form; step positions live on the step; `provenance` is not stored in the document (the runner keeps results and the run record); `kind: 'recipe' | 'sql'` is not in the document.
