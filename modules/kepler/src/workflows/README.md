# Analysis workflows

A workflow is a serialisable graph of steps over the catalog operations. This directory is the host-independent part: the document, the operations, the planner, the runner and the exports. It has no React, DOM or Kepler imports. The workbench UI that edits and runs workflows lives in `examples/deck/kepler-compute/src/workflow/`.

The design study, the experiments and the decisions behind this shape are in `dev-docs/research/kepler-workflows-2026-11-14/README.md`.

## Model

- Every node is a `WorkflowStep` with an `operationId`. Data enters through source steps (`source-dataset` reads a host dataset, `source-geojson` holds embedded GeoJSON, `source-drawing` reads shapes drawn on the host's map) and moves from an output port of one step to an input port of another.
- Data between steps is `WorkflowData`: a `CatalogOutputTable` (plain column arrays, declared types, at most one GeoJSON-text geometry column) plus the kind the rows actually hold (`point`, `line`, `polygon`, `features` for mixed, `table` for none).
- Operations come from a registry (`getWorkflowOperationRegistry()`): three sources, nine table operations, a SQL step (`table-sql`, run by the host's SQL engine) and one operation per binding in `KEPLER_CATALOG_BINDINGS`, adapted by `makeWorkflowOperationFromCatalogBinding`. A binding added to the catalog appears as an operation with no further work.
- Parameters are literals or bindings to workflow inputs (parameters promoted to the workflow level). There are no expressions on parameters; expressions exist only inside `table-filter-rows` and `table-calculate-column`.

```ts
import {
  WorkflowRunner,
  addWorkflowStep,
  createWorkflowDocument,
  getWorkflowOperationRegistry
} from '@luma-contributors/kepler';

const registry = getWorkflowOperationRegistry();
let document = createWorkflowDocument({title: 'Catchments'});
const source = addWorkflowStep(document, registry, {
  operationId: 'source-geojson',
  parameters: {text: facilitiesGeoJSON}
});
const cells = addWorkflowStep(source.document, registry, {
  operationId: 'voronoi-diagram',
  connectFrom: {stepId: source.stepId, output: 'data'}
});
document = cells.document;

const runner = new WorkflowRunner({host, registry});
runner.setDocument(document);
const record = await runner.run();
const table = runner.getState().steps[cells.stepId].result?.outputs['voronoi-cells'].table;
```

## Status is derived from content keys

`planWorkflow` gives every runnable step a key: a hash of the operation and its version, the normalised parameters, the engine choice (for GPU-capable operations only), the external revision of a source, and the keys of the steps it reads. `WorkflowRunner` keeps results by key. A step is `done` when it holds a result for its current key and `stale` when it holds one for an older key. Consequences:

- Editing a parameter invalidates that step and its dependents, and nothing else.
- A change in a host dataset, filter or selection re-keys the sources that read it (`refreshExternal()`).
- Undoing an edit returns the affected steps to `done` without running anything, because the earlier results are still cached.
- Label, note, position and display flags never invalidate a result.

## Checks at every hand-off

Adjacent operations have different capacities (a kernel density grid can emit more cells than Getis-Ord Gi\* accepts). Before a step runs, the runner checks each input against the consumer's declared kinds, row capacity and vertex capacity, and refuses with a message that names the port, the limit and the actual count. The planner reports the same pairs earlier as notes when the producer's declared maximum exceeds the consumer's.

## Files

| File | Holds |
| --- | --- |
| `workflow-types.ts` | Every type, `WORKFLOW_LIMITS`, `WorkflowError`, the `WorkflowHost` interface |
| `workflow-document.ts` | Strict parser and canonical serialiser of documents and of the saved collection |
| `workflow-data.ts` | Table and feature conversion, kind detection, 128-bit content hashes, column summaries, CSV and GeoJSON text |
| `workflow-parameters.ts` | Coercion, range checks and messages for parameter values |
| `workflow-expression.ts` | The expression language of the filter and calculate operations (hand-written parser, no `eval`) |
| `workflow-operations.ts` | Catalog adapter, registry, port compatibility, palette search |
| `workflow-source-operations.ts` | `source-dataset`, `source-geojson`, `source-drawing` |
| `workflow-sql-operation.ts` | `table-sql` and the validator that admits one read-only `SELECT` statement without file or network readers |
| `workflow-table-operations.ts` | Join attributes, filter rows, calculate column, select columns, sort rows, summarise, concatenate, points from columns, drop geometry |
| `workflow-plan.ts` | Topological order, issues, status and keys |
| `workflow-edit.ts`, `workflow-history.ts` | Immutable edit operations (steps, connections, parameters, workflow inputs, display style, copy and paste, inlining another workflow, swapping a step's operation), layered layout, snapshot undo with coalescing |
| `workflow-runner.ts` | Execution, node cache, hand-off checks, cancel, display publication, run record |
| `workflow-templates.ts` | Six ready-made workflows over generated geometry |
| `workflow-export.ts` | JSON and TypeScript export, plain-text outline, promotion of catalog history |
| `workflow-assistant.ts` | Tool definitions for an assistant: list and describe operations, propose a draft, list and describe saved workflows, change workflow inputs, run a workflow |
| `workflow-suggestions.ts` | One-click repairs for issues (insert a sort with a limit before an over-capacity input, join a keyed table back to its features, connect a compatible output, reset a parameter) |
| `workflow-recommendations.ts` | A short rule-based list of next steps for some data, from its kind, row count and columns |
| `workflow-report.ts` | Run report in Markdown and the graph in Graphviz DOT |

## Host interface

A host supplies `WorkflowHost`: the GPU device and the datasets it can offer (`listDatasets`, `readDataset`, `getDatasetRevisionKey`). Everything else is optional and the operations that need it refuse with `host-capability-missing` when it is absent: `publishOutput` / `retractOutput` / `focusOutput` so that a step output flagged for display becomes a dataset in the host (with an optional colour-by style), `runSql` for the SQL step, and `listDrawings` / `readDrawings` / `getDrawingRevisionKey` for drawn shapes. The module never reaches into a host.

## Limits

- Hand-off is by table on the CPU, within the catalog capacities (1 to 100,000 features per input, depending on the operation). There is no GPU-resident hand-off between steps; `WorkflowData` is where one would be added.
- Operations are whole catalog analyses; a workflow is not fused into one GPU graph.
- No iterators, branches or step-output parameter bindings. Composition is by copy: a workflow or a saved template is inlined into another (`insertWorkflowSteps`), not referenced, so later edits to the original do not propagate.
- A table whose column names are integer-like strings (`"10"`, `"2"`) lists those columns first in numeric order, because tables are keyed objects. `workflow-table-operations.node.spec.ts` records this as an expected failure.
- A document holds at most 64 steps and 16 workflow inputs; a workspace at most 16 workflows (`WORKFLOW_LIMITS`).
