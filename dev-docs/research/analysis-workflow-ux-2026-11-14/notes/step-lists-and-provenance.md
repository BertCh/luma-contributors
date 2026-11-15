# Step lists, history-as-workflow, notebooks and provenance UIs (as of 2026-11-14)

Scope: linear "applied steps" pipelines and history-based workflow UIs as alternatives to a node canvas, for
a narrow side panel next to a Kepler.gl map. Every claim carries a URL. Items marked **[unverified]** come
from prior knowledge or secondary sources and were not confirmed against a primary page in this session.

## Power Query (Excel, Power BI): Applied Steps, preview, errors, profiling, folding, references

### Takeaway
Power Query is the closest precedent: a linear, named, editable step list where selecting any step previews
the data at that point, steps carry a settings gear, and an upstream change that removes a column turns a
later step into a blocking step-level error rather than silently dropping it. Profiling runs on the first
1,000 rows by default, with a one-click switch to the entire dataset.

### Cited Findings
- The Applied steps list lives in the Query settings pane; each transformation appears as a named step (for
  example "Renamed columns"). "Selecting any step displays the results of that particular step, so you can
  see exactly how your data changes as you add steps." — [Microsoft Learn: Applied Steps](https://learn.microsoft.com/en-us/power-query/applied-steps)
- Step parts: delete icon (shown when the step is selected), step icon (by type, not user-changeable, Online
  only), step name (auto-assigned, renamable), step description (empty by default), step settings (gear;
  not all steps have settings), query folding indicator (Online only). Hover in Online shows name, label,
  description, settings and folding info. — [Applied Steps](https://learn.microsoft.com/en-us/power-query/applied-steps)
- Right-click step menu: Edit settings (greyed out when the step has none; opens the step's original dialog,
  for example the OData source page or Cluster values page), Rename, Delete (or X), **Delete until end**
  (deletes the step and all later steps, with a confirmation dialog), **Insert step after** (inserting an
  intermediate step shows a confirmation dialog; the next transformation the user makes is linked to the
  new selected step), Move before / Move after, drag-and-drop to move more than one position,
  **Extract Previous** (moves all steps before the selected one into a new named query that the current
  query then references, for reuse on other sources), Properties (name and description; a description adds
  a small icon and tooltip). — [Applied Steps](https://learn.microsoft.com/en-us/power-query/applied-steps)
- Errors: a **step-level error** "prevents the query from loading and displays the error components in a
  yellow pane" with reason, message and detail (for example `Expression.Error: The column 'Column' of the
  table wasn't found`), with Show details. The documented cause is exactly schema drift: the source file's
  column was renamed so a later Rename step can no longer find it; suggested fix is to delete or edit the
  step. **Cell-level errors** do not block loading; cells show "Error", selecting the cell whitespace shows
  the error pane, and Remove errors / Replace errors / Keep errors are offered. — [Microsoft Learn: Dealing with errors](https://learn.microsoft.com/en-us/power-query/dealing-with-errors)
- Data profiling: Column quality (Valid green, Error red, Empty dark grey, Unknown dashed green, Unexpected
  error dashed red, with percentages under each column header), Column distribution (frequency bars with
  distinct and unique counts), Column profile (statistics plus value distribution below the preview;
  right-click a bar for transformations on that value; Group by, Copy). "By default, Power Query performs
  this data profiling over the first 1,000 rows"; a status-bar message "Column profiling based on top 1000
  rows" toggles to "entire data set". Enabled from the View tab. — [Microsoft Learn: Data profiling tools](https://learn.microsoft.com/en-us/power-query/data-profiling-tools)
- Query folding indicators (Online only) show per step whether the query **up to that step** folds:
  Folding, Not folding, Might fold (determined at runtime), Opaque (inconclusive), Unknown (no query plan,
  error or non-table). The state is not sequential: a later step can fold again after a non-folding one if
  the non-folding column is removed. — [Microsoft Learn: Query folding indicators](https://learn.microsoft.com/en-us/power-query/step-folding-indicators)
- Reference versus Duplicate: Duplicate copies the query's steps into an independent query; Reference
  creates a new query whose source is the referenced query's final output; the Query Dependencies view (View
  tab) shows references as dependency edges but not duplicates. A reference always builds on the last step
  of the referenced query. Whether the source is fetched once or many times is disputed between sources. —
  [RADACAD](https://radacad.com/reference-vs-duplicate-in-power-bi-power-query-back-to-basics);
  [SumProduct](https://www.sumproduct.com/blog/article/power-query-blogs/power-query-reliable-references);
  Microsoft guidance page exists at [Referencing Power Query queries](https://learn.microsoft.com/power-bi/guidance/power-query-referenced-queries) (content not fetched).
- **[unverified]** The Advanced Editor shows the whole query as M `let ... in` code where each applied step is
  one named binding; the formula bar shows the selected step's M expression; parameters (Manage Parameters)
  can be referenced in step settings. Folding sample code in the folding page confirms the step-as-binding
  shape — [Query folding indicators](https://learn.microsoft.com/en-us/power-query/step-folding-indicators).

### Inferences
- A step list where each step is a named binding over the previous step's output maps directly onto typed
  analysis operations: the "M code" equivalent is an exported TypeScript or JSON recipe.
- Copy Power Query's split between blocking step errors (operation cannot run: missing column, wrong
  geometry type) and per-row errors (counted and surfaced in a quality bar, not blocking).
- "Extract Previous" is a clean answer to "a result cannot feed another operation": cut the list at a step,
  turn the prefix into a named reusable input. "Reference" gives branching without a canvas.
- The profile-on-1,000-rows default with an explicit status-bar toggle is a proven pattern for
  "preview on sample, run in full".

### Gaps
- Primary Microsoft pages for the Advanced Editor, Query Dependencies view and parameters were not fetched.
- No Microsoft documentation found on undo/redo semantics in the Power Query editor.

## Tableau Prep, Trifacta/Alteryx, OpenRefine, recorded macros, dbt lineage

### Takeaway
Tableau Prep combines a small flow graph with a per-step Changes list that can be reordered by drag, plus a
profile pane; OpenRefine is the strongest precedent for "history becomes a reusable workflow" via
Extract/Apply of operation JSON, and its recent column-mapping work shows the dataset-replacement problem is
real and still not fully solved.

### Cited Findings
- Tableau Prep: each change adds an annotation on the step in the Flow pane and an entry in the step's
  Changes pane; you can "drag changes up or down to change the order in which they're applied and add a
  description"; in Aggregate, Pivot, Join and Union steps "the order of these changes is applied by the
  system and cannot be changed"; remove a change with its X; hovering a step's cleaning icon shows its
  annotations, clicking one highlights the change and field. Undo is in the top menu and on the Flow pane.
  The profile pane (per-field profile cards) is the default view; a data grid view shows rows; recommendations
  appear as a light bulb on profile cards. "Pause data updates" switches to list view and disables
  value-dependent operations so users can batch edits. — [Tableau Help: Clean and shape data](https://help.tableau.com/current/prep/en-us/prep_clean.htm)
- **[unverified]** Tableau Prep input steps have Data Sample settings (default automatic sample, use all
  data, fixed number of rows, quick or random selection) and flows run on all data when output; page
  prep_configure.htm returned 404 in this session.
- OpenRefine History (Undo/Redo) tab: lists every change; step 0 is project creation; current state
  highlighted, later steps greyed; clicking a step returns to that state; clicking a later step redoes up to
  it; a new operation after stepping back erases the greyed steps (linear undo, no branches); history saved
  with the project and in archive exports. **Extract…** lists operations up to the current state with
  checkboxes and shows JSON to copy; **Apply…** in another project pastes it. "Not all operations can be
  extracted"; single-cell edits cannot be replicated. — [OpenRefine manual: History](https://openrefine.org/docs/manual/running#history-undoredo)
- Applying history JSON to a dataset with different column names: users report 3.8 applied operations
  loosely even with missing columns, 3.9+ validates strictly and broke such workflows; a maintainer pointed to
  a column-mapping dialog arriving in 3.10 snapshots; an October 2025 report says almost no errors are
  reported to the user, linked to issue #7500 "Allow users to partially apply histories". Status of 3.10 not
  confirmed from release notes. — [OpenRefine forum thread](https://forum.openrefine.org/t/apply-operation-history-from-json-file-not-working/1582);
  [page 2](https://forum.openrefine.org/t/apply-operation-history-from-json-file-not-working/1582?page=2);
  [invalid JSON thread](https://forum.openrefine.org/t/applying-invalid-json-causes-unrecoverable-error/2563/2)
- **[unverified]** Trifacta / Alteryx Designer Cloud "recipes" are ordered step lists (Wrangle language)
  edited beside a grid on a sample, with column histograms and data-quality bars, and run as jobs on the full
  data; Excel/Sheets macro recorders capture actions as VBA/Apps Script with absolute or relative
  references; dbt lineage is a DAG of models with `ref()` and state-based selection (`state:modified+`). No
  primary pages fetched for these.

### Inferences
- Tableau Prep shows the hybrid that fits a narrow panel: steps are coarse, each step holds an ordered,
  reorderable change list; system-ordered steps lock reordering. A side-panel step list with a few
  lock-ordered "join/aggregate" steps is a known pattern.
- OpenRefine's experience says: when re-applying a pipeline to a new dataset, ask for column mapping
  **before** applying, and report per-step success, skip or failure; do not silently partially apply.

### Gaps
- Tableau Prep sampling and input-schema-change behaviour could not be verified (404).
- Trifacta/Alteryx, macro recorders and dbt not verified from primary docs in this session.

## Creative tools: History versus Actions, non-destructive stacks

### Takeaway
Creative tools separate two things the analysis app currently conflates: an ephemeral undo **History** and
a reusable, recordable **Action**; and non-destructive **stacks** (modifiers, smart filters, effects) that
re-evaluate live, can be toggled per item, reordered, and "applied" to bake. Kepler's layer and filter list
is already such a stack.

### Cited Findings
- Adobe Actions documentation (helpx.adobe.com/photoshop/using/creating-actions.html) returned HTTP 403.
  **[unverified]** Photoshop Actions record a sequence of commands; each recorded step has a "toggle dialog
  on/off" (modal control) that pauses playback to show the dialog for user input; "Insert Stop" adds a
  message pause; individual commands can be unchecked to skip; Batch and Create Droplet run an action over a
  folder. History panel is a per-document, session-limited list of states with snapshots; it is not
  reusable.
- Blender manual page returned 403. **[unverified]** Blender modifier stack evaluates top to bottom; each
  modifier panel has toggles for edit mode, viewport and render, a menu with Apply, Duplicate, Move to first
  / last, collapse, and shows warnings inline in the panel.
- **[unverified]** Photoshop Smart Filters attach a filter list to a smart object, each with visibility eye,
  blending options and re-editable parameters; Lightroom and Capture One keep a non-destructive edit history
  with snapshots; After Effects effect stacks run top to bottom with per-effect fx toggles.
- VisTrails describes its change-based provenance as working "with any system that has undo/redo" —
  [Freire and Silva eScience 2008 slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/julianafreire-escience2008.pdf)

### Inferences
- For the step-list spec: give every step a "bypass" toggle (eye) that keeps parameters but passes input
  through, distinct from delete; downstream steps go stale when toggled.
- Modal controls map to "ask each time" parameters when a saved workflow is replayed: mark a parameter as
  prompted on replay (also Apple Shortcuts' Ask Each Time, below).
- An analysis stack beside Kepler's layer list: each analysis step's output is a dataset; its "eye" could
  control the result layer, while the step's bypass controls computation. Keep the two separate.

### Gaps
- All Adobe and Blender primary pages blocked (403); these claims need verifying before citing.

## Automation step lists: Apple Shortcuts, Zapier, Make, GitHub Actions, Retool

### Takeaway
Automation editors solve "a result feeds a later step" without a canvas by inserting typed tokens
referencing earlier steps' outputs into later steps' parameters, and by requiring per-step test runs to
produce sample outputs for mapping.

### Cited Findings
- Apple Shortcuts: tap a field, Select Variable; "each action's output" appears as a blue token beneath that
  action; tapping inserts it where "the stored data will be placed when the shortcut is run". Set Variable
  stores the previous output under a name (advised only for values that change). A parameter can be replaced
  by **Ask Each Time** so the user is prompted at run time; tokens are removed with delete or Clear. —
  [Apple Support: Use variables in Shortcuts](https://support.apple.com/guide/shortcuts/use-variables-apdd02c2780c/ios)
- **[unverified]** Shortcuts magic variables can be tapped to choose a property or type coercion and renamed.
- Zapier: testing a step creates a test record approximating live data; trigger, Filter and Paths steps must
  be tested before Publish is enabled; mapping uses a plus icon to pick a field from an earlier step's
  sample output; mapping only goes from previous to later steps; changing settings can leave outdated
  mappings, which show a yellow warning; when a step errors it produces no output. Zapier also has a
  "bird's-eye view of dependencies between Zap steps". — [Zapier: Map fields](https://help.zapier.com/hc/en-us/articles/8496343026701);
  [Zapier: Enter data in Zap fields](https://help.zapier.com/hc/en-us/articles/8495924713997);
  [Zapier platform: sample data](https://docs.zapier.com/platform/build/sample-data);
  [Zapier: dependencies between steps](https://help.zapier.com/hc/en-us/articles/34444909619981-Get-a-bird-s-eye-view-of-dependencies-between-Zap-steps)
- **[unverified]** Make uses a canvas of modules with a mapping panel; GitHub Actions uses `needs:` for job
  DAGs and `steps.<id>.outputs` for step references; Retool Workflows uses a canvas with per-block run and
  output inspection. Not fetched.

### Inferences
- An analysis step parameter of type "dataset" can be filled by a token "Result of step 3: Hotspots" instead
  of a dataset picker; this gives DAG power inside a list. Zapier's yellow "outdated mapping" warning is the
  pattern for a token whose referenced column disappeared.
- Zapier's "test step to produce sample output" suggests a per-step "preview on sample" run that also
  supplies the schema for downstream mapping.

### Gaps
- Make, GitHub Actions and Retool not verified from primary sources.

## Notebooks and reactive documents

### Takeaway
Studies show manual, order-dependent re-execution causes hidden state and poor reproducibility; reactive
systems (marimo, Hex, Observable) derive execution order from dependencies and either auto-run descendants
or, for expensive work, mark them stale. That "lazy: mark stale, run on request" mode is the model for GPU
analysis steps.

### Cited Findings
- Pimentel et al., MSR 2019, 1.4 million GitHub notebooks: of 863,878 attempted executions only about 24%
  ran without errors and about 4% reproduced the same results; among notebooks with unambiguous execution
  order 36.36% had out-of-order cells and 76.90% had skipped execution counts; causes include missing
  dependencies, hidden states and out-of-order execution. A later journal version suggests the original may
  have underestimated reproducibility. — [Pimentel et al. 2019 PDF](https://leomurta.github.io/papers/pimentel2019a.pdf);
  [EMSE 2021 extension](https://link.springer.com/10.1007/s10664-021-09961-9)
- Chattopadhyay et al., CHI 2020 (honourable mention), interviews (n=20) and survey (n=156): nine pain
  points across the workflow, including reliability at scale, archiving history within and between
  notebooks, sharing with non-data-scientists, reproduction and reuse, and notebooks as products; data
  transformation found "challenging and cumbersome". — [Paper PDF](https://microsoft.com/en-us/research/uploads/prod/2020/03/chi20c-sub8173-cam-i16.pdf);
  [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/old-tools-new-tricks-improving-the-computational-notebook-experience-for-data-scientists/)
- Kery, Horvath and Myers, Variolite, CHI 2017: data scientists avoid standard version control for
  exploratory work (3 of 10 interviewees used it for exploratory analysis versus 9 of 10 for other work);
  prototype lets users box code chunks into local alternative versions. — [CMU HCII preview](https://hcii.cmu.edu/news/chi-preview-variolite-supporting-exploratory-programming-data-scientists);
  [DOI via Unpaywall](https://unpaywall.org/10.1145%2F3025453.3025597)
- marimo: static analysis of each cell's references and definitions forms a DAG; running a cell runs its
  descendants; execution order follows variable relationships, not position; mutations are not tracked;
  deleting a cell deletes its globals and dependents re-run or are marked stale; **lazy** runtime marks
  cells stale instead of running (for expensive notebooks); disabled cells block themselves and dependents;
  interrupting a cell "cancels execution of any queued dependent cells". — [marimo docs: Reactivity](https://docs.marimo.io/guides/reactivity/)
- Hex: Auto run mode is the default for logic and the only mode for apps; "Cell only" is traditional
  notebook behaviour; apps show a chosen subset of cells but interactions may run more cells than shown;
  app default shows results from the last publish or scheduled run, alternatively runs top to bottom on
  open; background refreshes use default input values. — [Hex: Run modes and cell staleness](https://learn.hex.tech/docs/explore-data/projects/project-execution/run-modes-and-cell-staleness);
  [Hex: App run settings](https://learn.hex.tech/docs/share-insights/apps/app-run-settings)
- **[unverified]** Observable notebooks re-run cells reactively on dependency change; Streamlit reruns the
  whole script top to bottom on each widget change with `st.cache_data` for memoisation; Quarto and Evidence
  render parameterised reports. Not fetched.

### Inferences
- Do not allow out-of-order manual execution with hidden state: a step's result must always be a function
  of its inputs and parameters, with stale marked explicitly.
- Default to lazy for expensive GPU steps (mark stale, "Run stale" button), auto for cheap ones; mirror
  marimo's "cancel also cancels queued dependents".
- Hex's app view argues for "promote workflow to a tool": expose a chosen subset of parameters as inputs.

### Gaps
- Observable, Streamlit, Quarto and Evidence primary docs not fetched; marimo's visual stale styling not
  documented on the fetched page.

## Geoprocessing history: ArcGIS Pro, QGIS, GRASS, Earth Engine

### Takeaway
ArcGIS Pro and QGIS both treat every tool run as a history entry that can be re-opened with the previous
parameters and exported as code; ArcGIS Pro additionally offers "Add To Model", the main precedent for
promoting history into a reusable parameterised workflow.

### Cited Findings
- ArcGIS Pro geoprocessing history: entry context menu Open (reopens the tool with previous settings; also
  double-click), **Add To Model** (adds tool and required data to an open or new model), Copy Python Command,
  Send To Python window, Send To Notebook, Save As Python Script (multi-select), Edit Description, Add To /
  Remove From History Favorites, Remove. Entries saved with the project; sort by Status, Name, Start Time;
  filter by Type, Status, Input Layer; each entry shows In Progress, Completed or Failed and stores info,
  warning and error messages and output reports. Options: Write to Geoprocessing History (on), XML log
  (off), Write to dataset metadata (on). — [ArcGIS Pro: Geoprocessing history](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/geoprocessing-history.html)
- **[unverified]** ArcGIS Pro "Share As Geoprocessing Package" from a history entry; not on the fetched page.
- QGIS Processing history (3.44 docs): every algorithm execution stored with date, time and parameters as
  command-line expressions; double-click opens the dialog prefilled for re-run; right-click Copy as Python
  Command, Copy as qgis_process Command (includes environment such as units and ellipsoid), Copy as JSON
  (paste into an algorithm dialog to restore values), Create Test. — [QGIS docs: History manager](https://docs.qgis.org/latest/en/docs/user_manual/processing/history.html)
- **[unverified]** QGIS Model Designer is separate (canvas); no "create model from history" documented on
  the fetched page. GRASS keeps shell command history; Earth Engine Code Editor keeps script revisions in
  git-backed repositories. Not fetched.

### Inferences
- History entries should store full parameter snapshots so "Open with previous settings" and "Copy as code"
  are trivial; "Add to workflow" (append this run as a step) is the ArcGIS-style promotion.
- QGIS's "Copy as JSON then paste into a dialog" is a cheap parameter-transfer primitive worth copying.

### Gaps
- ArcGIS "save as package", GRASS and Earth Engine not verified.

## Provenance research and libraries (VisTrails, Trrack, undo models, PROV)

### Takeaway
Trrack (BSD-3-Clause, `@trrack/core` plus `@trrack/vis-react`) is the directly usable React-compatible
library: action-based provenance graph, branching undo (new action after undo creates a branch), node
annotations, JSON export. VisTrails supplies the conceptual model (version tree of workflow changes,
analogies, parameter exploration).

### Cited Findings
- Trrack: history and provenance library for web visualisations, tracks state in a directed provenance graph;
  `@trrack/core` is framework-agnostic; `@trrack/vis-react` renders the graph in React (peers react,
  react-dom, @trrack/core); licence BSD-3-Clause; no GitHub releases listed; about 25 stars. —
  [GitHub Trrack/trrack](https://github.com/Trrack/trrack)
- Trrack concepts: register named actions; each applied action records a new node linked to its parent;
  stores full checkpoints when many keys change or patches when few; step back and forward; save a node ID
  and return to it; "taking a new action after an undo creates a new branch rather than destroying the old
  one"; attach arbitrary data to any node; export the whole graph to JSON. — [Trrack getting started](https://apps.vdl.sci.utah.edu/trrackjs/docs/tutorial/getting-started)
- VisTrails: stores the trail of workflow changes as a version tree where each node is a workflow version
  and each edge a change; selecting a version lets users examine, modify, execute, tag and annotate it;
  analogies transfer a change from one version pair to another (drag source onto target; also in the
  Spreadsheet, where analogies cannot be named); parameter explorations show results side by side in the
  VisTrails Spreadsheet. — [VisTrails: What is VisTrails](https://www.vistrails.org/usersguide/dev/html/intro.html);
  [VisTrails: Analogies](https://www.vistrails.org/usersguide/html/analogies.html);
  [Enabling Reproducible Science with VisTrails](https://arxiv.org/pdf/1309.1784)
- **[unverified]** Cutler, Gadhave and Lex, "Trrack: A Library for Provenance-Tracking in Web-Based
  Visualizations", IEEE VIS 2020 short paper; "Loops" (Eckelt et al., 2024) is a JupyterLab extension
  visualising notebook provenance and cell differences; Lyra 2 records interaction history for Vega-Lite;
  Chimera (Kurlander and Feiner) introduced editable graphical histories; undo models are linear, branching
  (history tree) and selective (undo an earlier operation while keeping later ones). W3C PROV-DM models
  Entity, Activity, Agent with used / wasGeneratedBy relations — [W3C PROV-DM](https://www.w3.org/TR/prov-dm/) (not fetched).

### Inferences
- Use a linear step list as the editable artefact, and record edits and runs into a Trrack-like branching
  graph in the background; expose branches only as "versions" of a saved workflow, not as a canvas.
- Selective undo is effectively "bypass or delete step N and re-run downstream" in a step list, which a
  dependency-tracked list makes safe.
- A PROV-shaped export (run = activity, dataset = entity, operation version = plan) is cheap to add to the
  existing flat run record.

### Gaps
- Trrack API details (registry, ephemeral actions, events) and the Loops paper not fetched; Trrack
  maintenance status (no releases) should be checked on npm.

## Orchestration vocabulary: Dagster, Airflow, Make, Bazel

### Takeaway
Dagster's vocabulary fits an interactive analysis panel: an output is "unsynced" when its code version or
an upstream data version changed, staleness is not transitive through code changes alone, and a
"Materialize unsynced" action re-runs only out-of-date and never-run items.

### Cited Findings
- Dagster: an asset is out of sync when its code version or input data versions changed since last
  materialisation; UI label "Unsynced" on graph nodes and sidebar with an (i) hover showing the mismatch;
  "Never materialized" for unrun assets; reasons include code version change, dependencies added or
  removed, upstream data version change; staleness is **not transitive** through code changes, only through
  data version changes, and an explicit unchanged `DataVersion` keeps downstream in sync; **Materialize
  unsynced** (toggle beside Materialize) runs only out-of-sync and never-materialised assets; observable
  source assets compute their own data version to detect upstream changes. — [Dagster: Asset versioning and caching](https://dagster.io/docs/guides/build/assets/asset-versioning-and-caching)
- **[unverified]** Airflow task instances show states (queued, running, success, failed, upstream_failed,
  skipped) and "Clear" with downstream to re-run; Make re-runs targets whose prerequisites have newer
  timestamps; Bazel uses content hashes of inputs for cache hits. Not fetched.

### Inferences
- Map the proposed statuses: not configured, ready (never materialised), running, done (synced), stale
  (unsynced, with reason: parameter changed, upstream result changed, source dataset changed), error, plus
  Airflow's "upstream failed / blocked" as a distinct status so a downstream step does not show its own
  error.
- Content-hash results (dataset version + operation id/version + parameters) so an unchanged re-run does
  not invalidate downstream steps (Dagster's explicit data version).
- "Run to here" = materialise this step and its unsynced ancestors; "Run stale" = Materialize unsynced.

### Gaps
- Airflow, Make and Bazel primary docs not fetched.

## Edge cases and how products handle them

### Takeaway
Products handle schema drift by failing the dependent step visibly (Power Query yellow step error, Zapier
yellow outdated-mapping warning, OpenRefine column mapping), handle preview-versus-full via explicit sample
indicators, and handle long runs with cancel that also cancels queued dependents; undo of runs (as opposed
to edits) is rarely offered.

### Cited Findings
- Source replaced or column renamed: Power Query raises a blocking step-level error at the first step that
  references the missing column — [Dealing with errors](https://learn.microsoft.com/en-us/power-query/dealing-with-errors).
  OpenRefine moved from loose to strict validation and toward a pre-apply column-mapping dialog —
  [OpenRefine forum](https://forum.openrefine.org/t/apply-operation-history-from-json-file-not-working/1582).
  Zapier flags outdated mappings in yellow — [Zapier: Enter data in Zap fields](https://help.zapier.com/hc/en-us/articles/8495924713997).
- Upstream source change detection: Dagster observable source assets compute a data version that marks
  downstream unsynced — [Dagster versioning](https://dagster.io/docs/guides/build/assets/asset-versioning-and-caching).
- Sample then full: Power Query profiles first 1,000 rows with a status-bar toggle —
  [Data profiling tools](https://learn.microsoft.com/en-us/power-query/data-profiling-tools); Zapier tests
  steps on a sample record — [Zapier map fields](https://help.zapier.com/hc/en-us/articles/8496343026701).
- Long-running and cancel: marimo interrupt cancels queued dependents; lazy mode avoids auto-running
  expensive cells — [marimo reactivity](https://docs.marimo.io/guides/reactivity/). ArcGIS Pro history
  shows In Progress, Completed, Failed per entry — [ArcGIS Pro history](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/geoprocessing-history.html).
- Batch edits without recompute: Tableau Prep "Pause data updates" — [Tableau Prep clean](https://help.tableau.com/current/prep/en-us/prep_clean.htm).
- Undo: OpenRefine linear undo discards redo branch on a new edit — [OpenRefine history](https://openrefine.org/docs/manual/running#history-undoredo);
  Trrack branches instead — [Trrack getting started](https://apps.vdl.sci.utah.edu/trrackjs/docs/tutorial/getting-started).
- Export to code: ArcGIS Pro Copy Python / Save As Python Script; QGIS Copy as Python / qgis_process / JSON;
  OpenRefine operation JSON — sources above.

### Inferences
- On dataset replace: diff the new schema against each step's referenced columns, offer a mapping dialog
  before re-running, mark affected steps "needs attention" (not error) until mapped.
- Upstream Kepler filters: treat the filtered view as an explicit input version (filter state in the hash)
  and show "input changed by filter" stale reason; let the user pin a step to the unfiltered dataset.
- Undo: edits go on an undo stack; runs are not undone but results are versioned and kept (cache), so undoing
  a parameter edit restores the cached earlier result without recompute.
- Naming: auto-names from operation ("Hotspots (Gi*)"), editable, with descriptions, following Power Query.

### Gaps
- No primary source found for a product that offers undo of a run itself; for partial failure semantics
  beyond Airflow-style "upstream failed" (unverified); and for Kepler.gl-specific filter interactions.
