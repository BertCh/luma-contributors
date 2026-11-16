# Analysis experience: research and improvement plan (2026-11-14)

Question from the maintainer: research the analysis workflow and the UI for doing analysis, here and in similar projects. It does not have to be a visual builder. Lean into inferring analyses and workflows from the type of the input, and into a curated set of pre-configured workflows. Produce a plan of improvement.

Scope: research and design. The library layer of the plan was implemented afterwards, on the same day; section 9 says what exists, how it was checked and what it cannot do. No host surface uses it yet. This report widens [kepler-workflows-2026-11-14](../kepler-workflows-2026-11-14/README.md), which designed a workflow document and executor (KE-WF) and listed what it had not researched. That report stands; section 4.9 says how the two fit.

How it was produced, on `fbb78ec5` with a dirty working tree (the Kepler port and a backlog pass were editing the checkout):

- Five read-only reviews of the repository by Codex `gpt-6.1-sol` agents, which read source and ran counting commands. No GPU run, browser run, build or test was made. The lead spot-checked four claims against source (`app.tsx:498-502`, `catalog-binding-types.ts:34-62`, `catalog-lens.ts:35-63`, the constant `distanceBand` default).
- Eight web research passes by Claude agents. Each note marks what was read on a primary page and what is unverified. Esri moved its documentation during 2026 and several pages did not load; those claims are marked in the notes.
- Internal statements below are "verified" when a reviewer read the code and "inferred" otherwise. Action counts are counted from code, not measured in a browser.

The thirteen notes are in [notes/](./notes); the index is at the end. Every external claim below has its URL in the note named beside it.

## 1. Summary

The library has the operations (529 catalogued operators, 41 composite recipes, 130 Kepler bindings) and a large amount of curated material (417 lens scenarios, 173 showcase stories, 532 website entries with explanations). What it lacks is a path from "I loaded a dataset" to "I am looking at a suitable analysis of it". Today that path is a 24-item operation menu, eleven hand-written configuration panels (the point forms among them start with blank coordinate roles), and a generated catalog that cannot see prepared datasets.

Among the web map products reviewed, none documents suggesting an analysis from the dataset's columns (the sources fetched show none; that is not proof that none exists). kepler.gl infers layers only. The closest precedent, ArcGIS Insights, was retired in 2026. This looks like an open position. Whether a client-side GPU app can profile and preview fast enough to hold it is a hypothesis that AX-70 measures.

The plan has six parts, in this order:

1. **Fix the path that exists** (AX-0): carry known roles into forms, let the catalog read prepared datasets, replace fixed geographic defaults, style catalog results.
2. **One descriptor per operation** (AX-1): add questions, field roles with measurement level, checkable requirements, data-derived default policies and result presentation to the binding. This one record drives suggestions, forms, template slots, workflow steps and assistant tools.
3. **Profile and suggest** (AX-2): a dataset profile on load, a rule-based suggestion engine with reasons, and greyed-out options that say what is missing.
4. **Curated workflows** (AX-3): 27 question-named templates, twelve of them first, built from the existing recipes, each with typed slots, a field-mapping step, three to five exposed parameters and a licence-clean sample.
5. **One analysis sheet and self-explaining results** (AX-4): a single live-updating page in place of two form systems; results arrive styled, with a legend, a plain-language summary, the parameters that were chosen and why, cautions, and suggested next steps.
6. **Stack, workflows, assistant** (AX-5, AX-6): the analysis stack and the KE-WF document and executor, then a command palette and an assistant that proposes a plan over the same descriptors.

A node canvas is not in the plan's first five phases. The research found no evidence that a canvas beats a step list for analysts, and the pattern the best canvases converge on ("offer the compatible next step from here") works in a list.

## 2. Existing state

### 2.1 Surfaces and counts

| Surface | What exists (verified unless marked) | Source |
| --- | --- | --- |
| Kepler host, hand-written analysis | 24 operation options including "off", 11 configuration panels. Work on prepared, GPU-resident data. Results get colours, legends and plain-language summaries | [host journey](./notes/internal-kepler-host-journey.md) |
| Kepler host, generated catalog | 130 bindings (64 geometry, 20 network, 20 analysis, 11 aggregation, 5 clustering, 4 raster, 3 density, 3 interpolation), 531 parameters, all with a constant default. 17 bindings have a host sample | [binding metadata](./notes/internal-kepler-binding-metadata.md) |
| Kepler host, samples | 8 gallery samples (H3 cells, points through time, districts, connections, trips, trip observations, numeric grid, directed network). Each has prose and publication roles; none has an analysis | host journey |
| Kepler host, assistant | A structured-operation panel: seven named analyses, built-in observations only | host journey |
| Analysis library, ontology | 15 data kinds; 529 operator records; 151 of them have empty input and output kinds | [ontology and recipes](./notes/internal-analysis-ontology-recipes.md) |
| Analysis library, recipes | 41 `add*Recipe` functions (24 under `gpu-spatial-analysis/recipes`) | ontology and recipes |
| Lenses | 417 registered scenarios (254 spatial, 41 network, 51 raster, 46 terrain, 25 dataframe). 20 dataframe scenarios accept a user table through named numeric roles | [lenses and website](./notes/internal-lenses-website.md) |
| Website | 532 catalog entries, each with explanation, parameter and output docs; 12 data types, 12 use cases, 10 domains; "What do you have?" home cards; "Before" and "Next" hints; a workflows page with five recipes | lenses and website |
| Showcase | 173 stories in 17 chapters, 7 curated routes, step cards with evidence and limits, pipeline chips that reveal intermediates, parameter sweeps, `param` versus `compile` badges | [deck examples](./notes/internal-deck-examples.md) |
| Explorer | 66 modes in 10 categories, a Recipes mode with 14 choices, sliders that update per frame | deck examples |

### 2.2 Findings

F1. **Two analysis paths that do not meet.** The catalog panel is given only non-resident datasets (`examples/deck/kepler-compute/src/app.tsx:498-502`), and native datasets are normally auto-prepared into resident ones. A catalog input also needs a detected GeoJSON field; longitude and latitude columns are not enough (`analysis-catalog-input.ts:23,43`). The 130 bindings therefore cannot be reached from the data most users will have loaded. Verified.

F2. **Known roles are not carried forward.** Load-time code already infers GeoJSON, GeoArrow, H3 and point-pair roles, trip timestamp encodings and WKT (`auto-prepare-plan.ts:140,175,235`). The point analysis forms then start with blank longitude and latitude fields, in each form separately (`analysis-point-coordinate-settings.ts:83`). Verified.

F3. **Defaults are constants, some of them wrong for the data.** `getCatalogDefaultParameters` copies `defaultValue` (`catalog-bindings.ts:522`). The distance band is 1,000 in whatever unit the CRS has. IDW and kriging start from fixed New York bounds (`interpolation-panel.tsx:50`). The numeric-grid sample's metadata holds its shape, and the user is still told to type 16 by 16. Data-derived choices exist in a few places only: Scott and Silverman bandwidth in `GPUKernelDensity`, variogram fitting, AICc bandwidth search in GWR. Verified.

F4. **Bindings describe geometry and nothing else.** An input declares `accepts: point | line | polygon` and two capacities. Required columns are ordinary string parameters; whether one is needed can depend on another parameter (choropleth `count` needs no value column) with nothing to express that. There is no notion of count versus rate, no declared precondition, and spatial weights are not a port. Verified. The reviewer's conclusion: the present fields can list candidates by geometry but cannot rank them or say whether one will run.

F5. **Three type vocabularies.** Binding geometry kinds (3 plus `none`), ontology data kinds (15) and website data types (12). Raster, terrain, network, trajectories, cells, weights and tables have no binding kind. No code translates between them. Verified.

F6. **Enablement is discovered by failing.** The catalog Run button is disabled only while running; geometry and capacity rejections come back from execution. Verified.

F7. **Catalog results are not presented.** A published catalog result is a fixed cyan layer with raw summary keys (`publication.ts:173-193`). The styles module can do quantile, threshold and ordinal scales and legends, but it is not told which column or scale suits an output. Results without geometry (OLS coefficients, global statistics) get a table and no map. Legends omit OLS, cartogram and interpolation. Verified, last sentence included.

F8. **Curated material is not reusable on user data.** Recipes live in the library, scenarios in lenses, stories in the showcase, explanations on the website, samples in the host. The host can load and run lens scenarios from the catalog panel (`analysis-catalog-panel.tsx:194-215,304-338`), on their embedded data; 20 dataframe scenarios accept a mapped user table. Four catalog recipe bindings exist (hot spot, choropleth, catchments, zonal distribution). No gallery sample carries an analysis. The website's workflow pictures come from independent runs, not from chain intermediates (roadmap WEB-HIW-03). Verified.

F9. **Same name, different statistic.** Catalog local Moran uses row-standardised weights and permutations without correction; the hand-written panel uses binary weights and offers corrections. Catalog kriging has no fitting or cross-validation. Verified. A user choosing between them is not told.

F10. **The patterns the Kepler host lacks are already written elsewhere in this repository.** Guided step cards, a clickable strip of intermediate stages, parameter sweeps, rebuild-cost badges, a bottom sheet for small screens, `GraphSwap` (keep the old graph live while a new one compiles) and `SummaryReader` (one read in flight) are in the showcase and `examples/deck/spatial-analysis-shared`. Inferred from the two reviews.

Counted from code, a hand-written analysis takes 1 to 9 actions after choosing the dataset (4 for most point statistics, 9 for a cartogram). A catalog run is unavailable for auto-prepared datasets; for an eligible non-resident dataset it takes about 5 actions plus 3 per input, ending in a separate stage, check and publish.

## 3. What other tools do

| Theme | What was found | Lesson here | Note |
| --- | --- | --- | --- |
| Suggesting from the data | Tableau Show Me (the 2007 paper) gives each of 14 alternatives an availability rule in counts of field classes; six are ranked, some with a stricter condition for being the default. Unavailable commands stay visible and greyed, and hovering states the minimum requirement. CompassQL enumerates, filters by constraints, ranks and groups. Draco filters with hard constraints and scores with a weighted sum of broken soft constraints, each of which is a reason that can be shown. Lux scores candidates on a sample and recomputes only the top 15 | Hard rules for "can run", soft weighted rules for "ranks first", reasons shown, greyed options that say what to add | [type-driven](./notes/type-driven-recommendation.md) |
| Field roles | ArcGIS Insights typed every field as string, number, rate/ratio, date/time or location, offered tools by role, and grouped them under four questions: how is it distributed, how is it related, what is nearby, how has it changed. A rate/ratio field maps as a choropleth with no statistic applied to the shading (it is still summed in charts). Insights is retired | A role per column, guessed and user-correctable, is the unit suggestions work on | [GIS UI](./notes/gis-analysis-ui.md), type-driven |
| Counts and rates | Chrisman's measurement levels separate additive totals from derived densities and rates. The core-concept data type ontology (Scheider et al. 2020) separates field, object and event data and mentions extensive versus intensive attributes; the follow-on work (Kruiger et al. 2021) adds them as types. No product reviewed warns when raw counts are mapped as a choropleth or tested for hot spots. Values alone cannot settle whether a column is a count or a rate | Guess, show the guess as a chip, let the user change it once, and warn (not block) when an operation gets the wrong kind | type-driven, [synthesis](./notes/geo-workflow-synthesis.md) |
| Does typing pay | Kruiger et al. 2021 synthesised workflows from tool signatures. With geometry-only types 14 of 100 were correct; with concept, layer and measurement types 71 of 72. 70% of the correct ones still had redundant steps, and no ranking beyond enumeration order was documented | Richer types are what make suggestions valid. Automatic composition is not ready to be the main route; curated chains are | synthesis |
| Do recommendations help | In the Voyager studies, recommendations exposed users to 3 to 4 times more field combinations, and the number of bookmarked findings did not change. Users judge a suggestion by whether the columns they care about are in it | Keep the suggested row short (5 to 8), put the user's columns in the card, no more than a third of slots to one family | type-driven |
| Question-led menus | ArcGIS Online groups tools as Summarize data, Find locations, Analyze patterns, Use proximity, Enrich data, Manage data. CARTO names its 61 templates as questions with a place ("Commercial hotspot analysis for a new pizza place") | Organise by question first, by technique second | GIS UI, [templates](./notes/curated-templates-and-guidance.md) |
| Templates on my data | Tableau Accelerators map each required field to a user field of the same type, grey out wrong types, show a mapped counter and warn about what breaks. Insights reran a recorded analysis on new data through placeholder cards and "Replace fields". CARTO has no mapping step: you swap source nodes. OpenRefine's reuse of saved steps reportedly broke on different column names (forum evidence) | A template declares typed slots; opening one is a mapping step that the app pre-fills | templates, [platforms](./notes/web-map-platforms.md), [step lists](./notes/step-lists-and-provenance.md) |
| Gallery size | CARTO 61 templates in 13 categories; ArcGIS Solutions over 100; n8n thousands, mostly community. No study was found comparing a small curated set with a marketplace | Start small and official; the 41 recipes are already in CARTO's range | templates |
| Parameters chosen from the data | Esri's Optimized Hot Spot Analysis drops bad records, flags locational outliers, sets cell size from nearest-neighbour distances, sets the distance band from the first peak of incremental spatial autocorrelation (fallback: average distance to K neighbours), applies FDR correction, and writes every choice to the run messages. GeoDa's default distance band is the largest nearest-neighbour distance, so every feature has a neighbour | A default is a named rule over a data statistic, and the result says which rule fired and what value it gave | GIS UI, templates |
| Wizard or page | Nielsen Norman Group: wizards suit novices and rare tasks and fail for repeated work and interdependent steps. jamovi uses one options page with output that recalculates as options change. Liu and Heer 2014: 500 ms of added latency reduced exploration and the effect persisted. Zgraggen et al. 2017: progressive results did about as well as instant ones and much better than blocking | One live page, not a modal wizard. Preview within 500 ms or show a progressive result | templates |
| Parameter exploration | Design Galleries (Marks et al. 1997) show a spread of distinct outputs side by side. Sedlmair et al. 2014 name sensitivity and partitioning among six parameter-space tasks. Felt previews before Apply and can scope to features on screen | Small multiples across one parameter are cheap on a GPU; preview before publish | templates, platforms |
| Explaining results | Esri hot spot output uses fixed bins (99, 95, 90 percent, hot and cold) and states reliability conditions (at least 30 features, about 8 neighbours). GeoDa keeps inference settings on the result (999 permutations, cut-offs, Bonferroni or FDR) and Anselin cautions that pseudo p-values are not ordinary significance and that 0.05 gives many false positives | Fixed legends per statistic, a summary sentence, cautions that travel with the result, inference controls on the result | templates, GIS UI |
| Weights as an object | GeoDa builds weights once in a Weights Manager, lets the user inspect them (connectivity histogram, map), and every analysis picks them from a list | The two weights caches in the host are the start of this | GIS UI |
| Filter the toolbox by the layer | QGIS "Edit Features In-Place" narrows the toolbox to algorithms that can run on the active layer. ArcGIS Pro's "Suggestions" list is based on past runs, not on the data | Filtering by the active dataset is established; ranking by the data is not | GIS UI |
| Step lists | Power Query's Applied Steps: select a step to see the data at that point, rename, insert, reorder, "Delete until end", and "Extract Previous" to turn the steps above into a reusable query. A missing column fails at the first step that uses it. Dagster labels an output "Unsynced" with the reason on hover and re-runs only what is out of date | The step-list editor in KE-WF has a well-tested model to copy, including what happens when the source changes | step lists |
| History to workflow | ArcGIS Pro and QGIS reopen a past run with its settings and copy it as Python or JSON; QGIS can paste that JSON back into a dialog | Every run is a record that can be reopened, copied and promoted | step lists, GIS UI |
| Canvases | KNIME, ComfyUI and Orange converged on "drag from a port, get a type-filtered list of next nodes" (Blender's link-drag search is the same idea; not confirmed on a primary page). Observable added canvases in 2025 and kept the linear notebook. No study was found comparing a step list with a canvas for novice analysts, and no vendor was found removing a canvas. React Flow 12 is MIT and admits React 19; typed-port feedback has to be built | "Compatible next step" belongs in the list. A canvas is a later, optional view | [node builders](./notes/node-builders.md) |
| Peers | kepler.gl infers layers from column names and H3 validity and suggests no analysis. Its assistant (3.1 onward) has GeoDa tools, and the 3.3 alpha line adds suitability analysis, geometry operations and an experimental MCP package. Felt has an 11-operation panel whose results do not update when inputs change. Foursquare's agent shows its reasoning per step. SQLRooms (MIT) ships kepler.gl, AI chat and experimental canvas packages | No suggestion of analyses from the data was documented in the sources fetched. Results that stay live are a difference worth keeping | platforms |
| Assistants | GeoBenchX: a plain agent with 23 tools solved about 55% of multi-step tasks at best; failures were invented thresholds and attempting unsolvable tasks. Snowflake reports 57% text-to-SQL accuracy without a semantic model and 78% with one. Anthropic says tool choice degrades past 30 to 50 tools. No vendor documents an editable multi-step plan approved before running. `@openassistant/kepler-assistant` uses list, describe and run meta-tools and lets a host add tools and renderers | The assistant should choose among curated workflows and typed operations, have its plan checked by the same validator as a hand-built one, and show it as a step list. Send schema and profile, not rows | [assistants](./notes/ai-assistants-and-tool-catalogs.md) |
| Workflow documents | openEO process graphs: named nodes with `process_id` and `arguments`, arguments that are literals, `from_node` or `from_parameter`, one result node, child graphs as user-defined processes, JSON Schema types with a `subtype`. No standard reviewed has a vocabulary for statistical preconditions | The KE-WF draft already has this shape; `subtype` is the place for data kinds | synthesis |

Evidence that was looked for and not found: a study of automation bias for analysis (as opposed to chart) recommendations; a measure of how well a sampled preview of a spatial statistic matches the full result; a comparison of step list and canvas for novices; vendor policy on template versioning.

## 4. Design

### 4.1 Principles

1. **The data speaks first.** Loading a dataset produces a profile and a short list of analyses that fit it. Browsing the catalog is the fallback, not the entry.
2. **Questions before techniques.** Every operation and workflow is filed under the question it answers.
3. **One descriptor, many surfaces.** The suggestion list, the form, the template slot, the workflow step and the assistant tool are generated from one record per operation.
4. **Defaults come from the data and say so.** A parameter either has a rule ("largest nearest-neighbour distance") or is a declared constant. The result reports the rule and the value.
5. **Check before running.** Requirements are declared and evaluated against the profile, so the UI can say "needs a population column" before Run, and warn about counts used as rates.
6. **Live, on one page.** Parameters change the map as they move wherever the cost allows; a rebuild is marked as one.
7. **A result explains itself** and offers the next step.
8. **Curated first, composed second.** Templates built by people who know the statistics come before automatic composition and before a canvas.

### 4.2 Five ways in, one descriptor

| Way in | For whom | Built from |
| --- | --- | --- |
| Suggested for this data | Someone who has just loaded a dataset | Profile, descriptors, suggestion rules (4.3 to 4.5) |
| Workflows gallery, by question | Someone with a goal and no method | Templates (4.6) |
| Browse and search all operations; command palette | Someone who knows the operation | Descriptors; the catalog filtered by the active dataset |
| Result card "Next" | Someone looking at a result | `next` on the output, plus compatible-step rules |
| Assistant | Someone who would rather ask | The same descriptors and templates as tools (4.10) |

### 4.3 Dataset profile

A `DatasetProfile` is computed once per dataset revision and cached. Most of it exists already and is not collected in one place.

| Part | Content | Already available from |
| --- | --- | --- |
| Shape | Data kind (points, lines, polygons, cells, trajectories, network, surface, table), row count, vertex count, extent | `makeAutoPreparePlan`, `inferLayerRolesFromFields`, prepared tables |
| Coordinates | Geographic or projected (declared or guessed from the extent), unit | Point coordinate settings; new guess |
| Columns | Type, null count, range, distinct estimate, dictionary size, time unit | Arrow normalisation; `gpu-column-statistics.ts`, `gpu-table-inspector.ts` (computed on request; a revision-keyed domain cache exists in the host, `resident-channel-domain.ts:37-55`, but no unified dataset profile) |
| Roles (new) | Per numeric column: identifier, count, amount, rate or proportion, cyclic; per pair: likely numerator and denominator. Guessed from type, range, integrality and name; shown as a chip; user-correctable; saved with the dataset | New. No product reviewed does this reliably, so the guess must be visible and cheap to change |
| What the rows are (new) | Events (each row is an occurrence), objects with attributes, or samples of a continuous surface; and for area attributes, extensive (a total) or intensive (a rate or density). "Points with a number" is a crime count, a shop's revenue or a rain gauge reading, and the three call for different analyses | New. Cannot be read from values; asked once when a suggestion depends on it, then saved |
| Spacing (new) | Centre, standard distance and median distance from the centre; nearest-neighbour distance (minimum, median, maximum); the sorted k-distance curve; for polygons, neighbour counts under queen contiguity | New GPU statistics on published `GPUGridIndex`; feeds the default rules in 4.7. Cost not measured |

The profile is three tiers of cost, after Lux: metadata only (free), column statistics on the GPU (`GPUColumnProfile` runs one pass per column per product, plus merge nodes), spacing statistics (indexed passes, cost unmeasured). Suggestions appear after the first tier and sharpen as the others arrive.

A coordinate system or unit that the profile guessed is a suggestion. An analysis that depends on metric distance asks the user to confirm it before running, and the contributor still resolves its declared `spatialContext` at construction (contributor contract, section 6).

### 4.4 Descriptor additions

Additive, optional fields on `CatalogOperationBinding` and its specs. The vocabulary (questions, kinds, measurement levels, requirement ids, default rules) belongs in `@luma-contributors/analysis` beside the ontology so the website and examples share it; the binding fields stay in `@luma-contributors/kepler`.

```ts
type AnalysisQuestion =
  | 'where-concentrated' | 'where-significant' | 'how-much-per-area' | 'what-is-near-or-reachable'
  | 'how-related' | 'value-where-unmeasured' | 'what-changed' | 'how-it-moves'
  | 'shape-of-the-surface' | 'which-groups-or-regions' | 'prepare-data';

type FieldRoleSpec = {
  role: string;                 // 'value', 'cases', 'population', 'time', 'category', 'response', 'predictors'
  label: string;
  accepts: readonly ('number' | 'integer' | 'category' | 'time' | 'boolean')[];
  measurement?: readonly ('count' | 'amount' | 'rate' | 'ordinal' | 'nominal' | 'cyclic')[];
  required: boolean | {when: {parameter: string; equals: CatalogParameterValue}};
  multiple?: {minimum: number; maximum: number};
};

type RequirementSpec = {
  id: string;                   // 'minimum-rows', 'projected-coordinates', 'positive-values',
                                // 'every-row-has-a-neighbour', 'capacity', ...
  severity: 'block' | 'warn';
  limit?: number;
  message: string;              // contributor id, prop and limit, as the contributor contract requires
  remedy?: string;              // 'aggregate to cells first', 'choose a population column'
};

type ParameterDefaultPolicy =
  | {from: 'constant'}
  | {from: 'data'; rule: string; fallback: CatalogParameterValue};

// CatalogInputSpec  += kind (ontology data kind), fields: FieldRoleSpec[]
// CatalogParameterSpec += defaultPolicy, apply: 'live' | 'rebuild', level: 'basic' | 'advanced'
// CatalogOutputSpec += kind, presentation {column, scale, classes?, layer}, interpretation {summary, cautions}, next
// CatalogOperationBinding += questions, requirements
```

Requirements are evaluated against the profile and the bound roles, once, before a run. That keeps the contract rule: no per-row or per-frame validation on the CPU. Each evaluation returns `passed`, `failed` or `unknown`, with where the evidence came from (profile statistic, user declaration, GPU validator, not checkable). Many preconditions cannot be established from a profile: valid or connected weights, disjoint coverage, uniqueness of a dasymetric partition. Those are `unknown` until a GPU validator or the user's declaration settles them, and the sheet says so. The checker is an early warning in the host. It does not replace the contributor's own construction-time checks with the named-limit helpers, which stay as they are. A spec enforces completeness: every binding has at least one question, every string parameter that names a column is replaced by a field role, every default states its policy.

One vocabulary replaces three: the ontology's data kinds become the port type, with a published mapping to website types and to the three geometry kinds, and the 151 operator records with empty kinds are filled.

### 4.5 Suggestion engine

A pure function, tested on fixture profiles, with no model in it.

```ts
suggestAnalyses(profiles, descriptors, templates): Suggestion[]
// Suggestion = {target, status: 'ready' | 'needs-mapping' | 'blocked',
//               bindings: role -> column, reasons: string[], missing?: {what, remedy}, score}
```

- **Hard rules** (Show Me's availability): input kinds present, required field roles satisfiable from the profile, capacity. A failure makes the item `blocked` with the unmet requirement as text; it stays visible and greyed. `ready` means every mandatory requirement passed; an item with an `unknown` mandatory requirement is `needs-mapping`, with the question to answer.
- **Soft rules** (Draco's weighted costs): templates above single operations; uses more of what the dataset has (a time column, a category, a second dataset); measurement level matches without a warning; cheaper first; question diversity, with no more than a third of slots to one family.
- **Reasons**: each satisfied rule is a phrase. "Points with a time column, 48,210 rows: how do hot spots change over time?"
- **Budget**: 5 to 8 suggestions. The rest is behind "Browse all", filtered to what the dataset can run.
- **Two datasets**: when a second dataset is loaded, pair rules fire (points and polygons, facilities and a network, a raster and zones).

The first test fixture is the eight gallery samples: each must produce an expected analysis in its top three.

Automatic composition by path search over types is left out of this engine. With today's port types it would reproduce the 14% result; with the richer descriptors it becomes a validator for assistant plans (4.10) and a source of "compatible next step" lists.

### 4.6 Curated workflows

A template is a workflow document (the KE-WF format) plus what makes it reusable:

- `question` and a title phrased as a question; a one-paragraph method note; a thumbnail from a real run.
- `slots`: dataset slots by kind, each with field roles. This generalises `CatalogLensFieldRole`, which already maps user columns to named numeric roles for 20 scenarios.
- `exposed`: three to five parameters, each with a default policy. Everything else is under "All settings".
- `sample`: a bundled dataset with a recorded licence that permits it, so the template opens with a result in one action.
- `reads`: the result explanation and cautions (4.8).

Opening a template on user data is a mapping step in the style of Tableau's Data Mapper: slots pre-filled from the profile, columns of the wrong type greyed, a "3 of 3 mapped" counter, then an immediate run with derived defaults.

The first templates are the existing fused recipes run as one step. They need the workflow document types and parser (KE-WF-03, moved into AX-30), a Kepler binding for each recipe that lacks one (input preparation, output allocation, a publication adapter; most of the rows below lack one), a single-step execution path, the descriptor, and the mapping step. They do not need the multi-step executor, which is why templates move ahead of KE-WF-04 in the order of work.

Starting set. Every `add*Recipe` named exists under `modules/analysis/src` (checked by the reviewer). The last column lists direct operation bindings and forms found by searching `modules/kepler/src/analysis` and `examples/deck/kepler-compute/src`; lens scenarios that run on embedded data are not counted. Wave 1 is the first twelve.

| Wave | You have | Question | Template | Built from | Direct bindings and forms in the Kepler host today |
| --- | --- | --- | --- | --- | --- |
| 1 | Points | Where are they concentrated? | Density surface | `GPUKernelDensity` | `kernel-density-grid` binding, with a fixed bandwidth |
| 1 | Points | Where are the significant hot and cold spots? | Hot spot map | `addHotSpotAnalysisRecipe` | Catalog recipe binding; point hot spot panel |
| 1 | Points | Which points form groups, and what area does each cover? | Clusters and footprints | `addClusterAndOutlineRecipe` | DBSCAN panel; `density-clustering` binding; a grouped hull binding separately; no fused recipe binding |
| 1 | Points with time | How do hot spots change over time? | Emerging hot spots | `addSpaceTimeHotSpotsRecipe` | None |
| 2 | Points with time | What changed between two periods? | Period comparison | `addPeriodComparisonRecipe` | Partial: `cell-compare` compares two point sets; no period recipe |
| 2 | Points with a measured value | What is the value where nothing was measured? | IDW surface | IDW contributor | IDW panel; `idw-grid` binding |
| 1 | Points with a measured value | What is the value where nothing was measured, and how sure is it? | Kriging prediction, variance and diagnostics | Kriging, `addInterpolationDiagnosticsRecipe` (kriging only) | Kriging panel; `kriging-grid` binding without fitting |
| 1 | Points and polygons | How many, or how much, in each area? | Count and classify by area | `addPointsInPolygonsChoroplethRecipe` | Catalog recipe binding |
| 1 | Points and polygons | How do areas differ in what they contain? | Distribution by zone | `addZonalDistributionRecipe` | Catalog recipe binding |
| 2 | Facilities and a surface | What does each facility serve, in a straight line? | Straight-line catchments | `addStraightLineCatchmentsRecipe` | Catalog recipe binding |
| 1 | Polygons with a value | Where do similar values cluster, and where are the outliers? | Cluster and outlier map | Queen weights, local Moran | Polygon panel; `local-moran-statistics` binding |
| 1 | Polygons with cases and population | Where do rates cluster? | Rate cluster map | `addRateClusterMapRecipe` | Partial: spatial-rate panels; separate empirical Bayes and local Moran bindings; no fused chain |
| 2 | Polygons with cases and population | Where is risk above expectation? | Excess risk and smoothed rates | `addExcessRiskRecipe`, `addSpatialRateRecipe` | Partial: excess-risk mode, spatial-rate panels, empirical Bayes binding |
| 1 | Polygons with several columns | How does one variable relate to others, and does it vary over space? | Regression with spatial diagnostics | `addSpatialRegressionRecipe` | OLS panel; OLS binding; a point-only GWR binding |
| 2 | Polygons with a value | At what neighbourhood scale is there association? | Contiguity correlogram | `addContiguityOrderCorrelogramRecipe` | None |
| 2 | Two polygon layers, one with totals | How do I move values from one set of boundaries to another? | Change of support | `addChangeOfSupportRecipe` | Runnable sampled `areal-interpolation` binding; the recipe registry is empty |
| 2 | Source zones with totals, target zones, an ancillary raster | Where within each zone is the total likely to be? | Dasymetric redistribution | `addDasymetricRedistributionRecipe` (needs a grid, extensive totals, target-zone ownership and an ancillary band) | None |
| 1 | Elevation raster | Where does water go? | Drainage network and catchments | `addDrainageNetworkRecipe` | None |
| 2 | Raster with gaps | Which gaps can be filled, and which remain? | Surface preparation | `addSurfacePreparationRecipe` | None |
| 2 | Raster and zones | What is in each zone, and who is exposed? | Zonal distribution; exposure by zone | `addRasterZonalDistributionRecipe`, `addExposureByZoneRecipe` | Partial: the zonal panel computes zonal statistics, not either recipe |
| 2 | Two rasters and zones | Where did it change? | Regional raster change | `addRegionalRasterChangeRecipe` | None |
| 2 | Land-cover raster | How fragmented is it, and how isolated are the patches? | Landscape metrics; patch proximity | `addLandscapeMetricsRecipe`, `addPatchProximityRecipe` | None |
| 1 | Network with facilities and demand | Who is within each travel-time band? | Drive-time catchments | `addDriveTimeCatchmentRecipe` | Partial: service-area bindings, not the demand and band recipe |
| 2 | Network | What does a closure cut off? | Network disruption | `addNetworkDisruptionRecipe` | None |
| 2 | Trajectories and zones | Where do vehicles stop, and for how long? | Fleet dwell | `addFleetDwellRecipe` | None |
| 2 | Trajectories | Which observations need checking? | Track quality | `addTrackQualityRecipe` | None |
| 2 | Flows between cells | Which neighbouring cells exchange flow? | Cell flow topology | `addCellFlowTopologyRecipe` | None |

Twenty-seven rows, in CARTO's range. Wave 1 covers each input kind once or more. Eight of its twelve have a direct binding or form to start from, two are partial (rate cluster map, drive-time catchments) and two have none (emerging hot spots, drainage). The urban morphology, resistance connectivity and segregation recipes are a second tier for specialists. Sample data must be licence-clean: the lens fixtures for several of these use non-commercial observations, while the website already runs some on permissive substitutes (earthquakes for the nature points). `THIRD-PARTY-DATA.md` decides; the conflict it records for `s1-chicago-tracts` needs resolving first.

### 4.7 The analysis sheet and its defaults

One generated page replaces the catalog panel and, operation by operation, the hand-written forms (which keep their bespoke result views):

- **Data**: slots and field roles, pre-filled, each a dropdown restricted to compatible columns.
- **Key settings**: `basic` parameters. `live` ones update the map as they move; `rebuild` ones carry a badge and apply on release, with the old result kept on screen (`GraphSwap`).
- **Checks**: requirement results before Run. Blocks name the remedy; warnings can be dismissed.
- **All settings**: `advanced` parameters, folded.
- **Preview scope**: features on screen or a sample when the full run is slow; the sheet says which. A sample or a viewport changes neighbour structure and significance, and no measure was found of how well such a preview matches the full result. So a sampled or viewport preview shows no significance classes unless the operation has its own fidelity check, the preview population is recorded in the result's provenance, and an inferential result is recomputed on the full population before it can be kept.

Default rules to implement, each a named function over the profile (sources in the [templates note](./notes/curated-templates-and-guidance.md), section c2; rows marked * are not confirmed on a primary page):

| Parameter | Rule | Needs |
| --- | --- | --- |
| Distance band | Largest nearest-neighbour distance, so every feature has a neighbour (GeoDa; Esri uses the same floor) | Spacing statistics |
| Neighbour count | 8 (Esri's recommendation; GeoDa's default is 4) | Constant, stated |
| Kernel density bandwidth | Two distinct policies. `normal-reference`: Scott or Silverman as `GPUKernelDensity` already implements them (SciPy's rules; in two dimensions both scale with n^(-1/6)). `esri-spatial`: Esri's variant built from standard distance and median distance from the centre* (the exact formula is published only as an image) | Count and per-axis spread for the first; centre, standard distance and median distance for the second. The `kernel-density-grid` binding passes a fixed bandwidth today |
| Grid cell size | Shorter side of the extent divided by 250. This is Esri's documented default for kernel density and point-to-raster only; using it for hot spot cells is this plan's choice, not Esri's optimized hot spot rule (which works from nearest-neighbour distances) | Extent |
| DBSCAN minimum points | 4 in two dimensions (Ester et al. 1996; Schubert et al. 2017) | Constant, stated |
| DBSCAN radius | Knee of the sorted k-distance plot (k = 3 in two dimensions). Schubert et al. note the plot often has no clear knee: then fall back to a stated constant and say so. Flag results with noise outside 1 to 30%, or with the largest cluster holding more than 20 to 50% of clustered points | The k-distance curve; show the plot |
| Interpolation bounds and grid | The dataset's extent | Extent (today: fixed New York bounds) |
| Variogram maximum lag | One third of the data diagonal | Already in `kriging-variogram-fit.ts` |
| Permutations and correction | 999 permutations (GeoDa's default). FDR correction on by default is this plan's policy, not a documented vendor default: GeoDa offers it as an option. Name the method (Benjamini-Hochberg) and warn when the permutation count cannot resolve the corrected threshold | Constants, stated |
| Class breaks | Fixed classes for significance results; quantile or natural breaks otherwise* | Column distribution |
| Lattice shape | From the sample's or dataset's metadata when present | Metadata |

A sensitivity strip, opened from any `live` parameter, shows four to six small maps across its range, after Design Galleries. The showcase's slider sweeps are the animated form of the same idea. The strip is labelled exploratory: looking at more parameter settings raises the chance of a spurious finding (the type-driven note, section 7), and FDR within one map does not control error across a parameter search. The parameter value a kept result was chosen at, and that a sweep was used, go into its provenance.

Target latency: a `live` change shows within 500 ms on the reference configuration. A `rebuild` change shows progress, can be cancelled, and keeps the previous result on screen until the new one is published.

Live rebuilds and several maps at once need the safeguards the KE-WF report puts in the executor (its section 5.3): dataset-revision checks, rejection of stale results, memory reservation, cleanup of superseded graphs and a visible cancel. `GraphSwap` discards superseded builds; it does not check dataset revisions or admit memory. The sheet depends on those safeguards and must not ship before them.

### 4.8 Results that explain themselves

Each output's `presentation` and `interpretation` produce a result card:

- **Style and legend, automatically**: a fixed diverging class scheme for hot and cold spots, the four LISA classes plus not significant, sequential for densities and predictions, categorical for cluster labels. Table-only results get a chart or a report, not a blank map.
- **One sentence**: "212 of 1,840 cells are significant hot spots at 95% after FDR correction; they hold 31% of events."
- **What was chosen**: "Distance band 1,240 m: the largest nearest-neighbour distance. Changed from the default? No."
- **Cautions that travel with the result**: fewer than 30 features; counts used where a rate was expected; pseudo p-values; islands excluded. The website's 532 explanations and the lens packs' assumptions and guards are the source text; the catalog schema gains an interpretation field.
- **Inference controls on the result**, as in GeoDa: significance cut-off and correction change the display without a rerun where the statistic allows.
- **Next**: two or three follow-on steps ("Outline these clusters", "Compare with last year", "Explain with regression"), from `next` and from kind compatibility.
- **Keep**: "Add to stack", "Save as workflow", "Copy as JSON".

Spatial weights become a named, reusable object with its own inspector (neighbour-count histogram, islands), built on the two weights caches the host already has.

### 4.9 Stack, history and workflows

This is the KE-WF design, with two changes of emphasis from this research.

- Every run from the sheet or a template is appended to an **analysis stack** in the side panel: a list of steps over the dataset, each with status, a visibility toggle and its settings behind a gear. It sits beside Kepler's layer and filter lists and behaves like them. The stack is the "ordered step list" of KE-WF, reached without the user deciding to build a workflow.
- Step behaviour follows Power Query and Dagster: select a step to see the map at that point; rename, disable, reorder where types allow; "Run stale" re-runs only what is out of date, and a stale step says why. When a source loses a column, the first step that uses it fails with the column named and a remapping offer.
- Edits to the stack (add, remove, reorder, change a setting) can be undone and redone. No product was found that undoes a run as such; undo restores the step list and marks affected steps stale.
- "Save as workflow" turns a stack into a template: the user picks which datasets become slots and which parameters are exposed. This is KE-WF's "promote history to workflow".
- KE-WF-01 to KE-WF-08 (adapter, port types, document, executor, persistence, panel, pre-flight, resident-native form) are unchanged in content. KE-WF-02's `ontologyKind` is the `kind` field of 4.4. KE-WF-03's document types and parser move into AX-30. KE-WF-10's templates move earlier as AX-3. KE-WF-11's TypeScript export and its reproduction test stay as KE-WF-11; AX-51 delivers only the JSON half. The canvas stays optional and last.

### 4.10 Assistant and command palette

- **Palette first.** A Cmd-K palette over questions, templates and operations, scoped to the active dataset, needs no model and no key. Typing "hot" on a point dataset offers the hot spot template with its slots already filled.
- **Assistant as a planner over curated tools.** Expose three meta-tools (search, describe, propose) and the templates, not 130 separate tools. The assistant returns a workflow document; the same validator that checks a hand-built stack checks it (kinds, field roles, measurement level, requirements, units); the user sees it as a step list and approves it whole. Whole-plan approval with machine checks is preferred to per-step prompts.
- **Refusal is a feature.** "This cannot be answered with your data: no population column" comes from the requirement checker, which is the failure the benchmarks show models handle worst.
- **Privacy.** The model receives the profile and the descriptors, never rows.
- **Which assistant.** The host has its own router and `@kepler.gl/ai-assistant` is not installed. Kepler's assistant accepts host tools and renderers, so registering this repository's templates as tools there is possible; that is a maintainer decision (section 7).

### 4.11 Sketches

After loading a point dataset with a timestamp:

```
┌ taxi_pickups ─────────────────────────────────────────┐
│ Points · 48,210 rows · lon/lat · time · 3 numeric     │
│ [fare: amount ▾] [passengers: count ▾] [tip_rate: rate ▾]│
├ Suggested ────────────────────────────────────────────┤
│ ▸ Where are the significant hot spots?                │
│   Hot spot map · points → cells → Gi* · ready         │
│ ▸ How do hot spots change over time?                  │
│   Emerging hot spots · uses pickup_time · ready       │
│ ▸ Which points form groups?                           │
│   Clusters and footprints · ready                     │
│ ▸ Where do rates cluster?                             │
│   Needs polygons with a population column  (greyed)   │
│ Browse all that fit · Workflows · Search (⌘K)         │
└───────────────────────────────────────────────────────┘
```

The sheet and the result card (all numbers in both sketches are illustrative):

```
┌ Hot spot map ─────────────────────────────────────────┐
│ Data      points: taxi_pickups ▾   weight: (count) ▾  │
│ Cell size      [──●─────] 180 m   extent ÷ 250        │
│ Distance band  [────●───] 540 m   largest NN distance │
│ Correction     (•) FDR  ( ) none                      │
│ ✓ 48,210 points   ⚠ 3 locational outliers excluded    │
│ ▸ All settings (9)          Preview: full data, 41 ms │
├ Result ───────────────────────────────────────────────┤
│ 212 of 1,840 cells are hot spots at 95% (FDR).        │
│ ■ hot 99 ■ 95 ■ 90 □ not significant ■ cold 90 ■ 95 ■ 99│
│ Next: Outline the hot areas · Compare two periods     │
│ [Add to stack] [Save as workflow] [Sensitivity]       │
└───────────────────────────────────────────────────────┘
```

## 5. Plan

IDs are AX for the analysis experience; KE-WF ids refer to the earlier report. Each gate is a check that can fail. "Fixture profiles" are the eight gallery samples plus one licence-clean dataset per template. "Wave 1" is the twelve templates marked 1 in section 4.6. The "named operations" are the operations behind wave 1 plus the eleven hand-written panels. The "reference configuration" is headless WebGPU Chromium on the machine used for the Kepler acceptance baseline (roadmap KE-UI); AX-70 writes its manifest (hardware, browser version, fixtures) before any latency gate is used. Latency thresholds are taken from the literature, not from measurements here: 500 ms for a live change (Liu and Heer), 1 s and 10 s for staged work (Nielsen).

| ID | Work | Gate |
| --- | --- | --- |
| AX-01 | Carry publication roles (coordinates, geometry, time) and the selected value column into every analysis form; one shared role state instead of one per form | Headless: after loading the points-time sample, each point form opens with coordinates filled; point statistics run in 2 actions (4 today) |
| AX-02 | Catalog reads prepared (resident) datasets and longitude/latitude roles through an adapter, within each binding's caps; caps shown before Run | Headless: a catalog operation runs on an auto-prepared CSV with no GeoJSON column; KE-WF-01 reuses the adapter |
| AX-03 | Data-derived defaults from what is already known without new statistics: interpolation bounds from extent, lattice shape from metadata. The distance band from spacing waits for AX-20 | `rg` finds no fixed geographic bounds in default settings; node spec per rule against a hand-computed fixture |
| AX-04 | Catalog results published with a column, scale and legend. Interim: a hand-written presentation table for the wave 1 outputs, replaced by AX-15 metadata; table-only results get a report view | Headless: hot spot, local Moran and choropleth catalog results publish with the class legend of the matching hand-written panel (legend entries compared in a spec); screenshots reviewed independently |
| AX-05 | Compatibility before Run from the fields bindings have today (geometry kind, feature and vertex capacity): disabled state with the unmet requirement and the remedy. Declared requirements arrive with AX-13 | Node spec: each binding returns a reason for each of three incompatible fixture profiles; no geometry or capacity rejection first surfaces at execution for those |
| AX-10 | One data-kind vocabulary: mapping table between ontology kinds, website types and binding geometry kinds; fill the 151 empty operator records | Spec fails on an operator record with empty kinds; website inventory regenerated; `yarn test-node` |
| AX-11 | Question taxonomy (eleven questions) in the ontology; tag bindings, recipes and website entries | Spec: every binding and recipe has at least one question; website facet built from the same list |
| AX-12 | Field roles and measurement levels on input specs, with conditional requirement; replace column-name string parameters | Spec: no `string` parameter names a column; `yarn typecheck`; `yarn check-contributor-contract` unchanged |
| AX-13 | Declared requirements and a checker over the profile | Node spec: the checker reproduces the runtime rejections of ten bindings without executing them |
| AX-14 | Default policies and `live`/`rebuild`, `basic`/`advanced` flags on parameters; default-rule functions in the analysis package | Spec: every parameter states a policy; rule functions match reference values (GeoDa or Esri documented examples, SciPy where available) |
| AX-15 | Output presentation and interpretation metadata; interpretation field in the website catalog schema | Spec: every output with geometry names a column and scale; the named operations have summary and cautions |
| AX-20 | `DatasetProfile` with cache per revision, role guesses, user overrides saved with the dataset, spacing statistics on the published grid index | Node spec on fixture profiles; headless on the reference configuration with 1,000,000 points and 8 columns: suggestions from metadata in the same frame as the dataset card, column statistics within 1 s, spacing statistics within 10 s with progress shown; overrides survive a sandbox save and replay |
| AX-21 | `suggestAnalyses`: hard and soft rules, reasons, blocked items with remedies, two-dataset rules | Spec: for each fixture profile the expected analysis is in the top three; no family holds more than a third of slots; deterministic order |
| AX-22 | "Suggested for this data" on the dataset card, greyed items with reasons, "Browse all that fit". A suggestion opens the existing form or binding until AX-31 and AX-40 land | Headless: load a sample, one action opens a suggested analysis with its roles filled; with AX-31, one action gives a result; screenshot reviewed independently |
| AX-30 | The workflow document types, strict parser and validator (KE-WF-03) and the template format over them: slots, exposed parameters, question, sample, reads | KE-WF-03's gate; node specs for valid, missing-slot and unknown-operation templates |
| AX-31 | Wave 1: a Kepler binding for each recipe that lacks one (input preparation, output allocation, publication adapter), a single-step execution path, and twelve templates with licence-clean samples. Then wave 2 | Each template runs on its sample in headless WebGPU and on a second dataset through mapping, and its result matches the recipe's existing spec fixture; licence recorded in `THIRD-PARTY-DATA.md` |
| AX-32 | Field-mapping step and Workflows gallery by question; gallery samples paired with a template | Headless: sample to result in 1 action; template on user data in mapping plus 1; wrong-type columns not selectable |
| AX-33 | Website workflows page generated from the same template records, with real chain intermediates | A spec shows each step picture is rendered from the buffer the chained run produced (the recorded intermediate's identity matches the picture's source); closes WEB-HIW-03 |
| AX-40 | Generated analysis sheet with live preview, replacing the catalog panel; hand-written forms migrate one operation at a time; duplicate operations either merged or labelled with their difference | Revision checks, stale-result rejection, memory reservation, cleanup and cancel are in place first (spec: a dataset revision bump during a rebuild publishes nothing). Headless on the reference configuration: a `live` parameter change shows within 500 ms for each wave 1 template on its sample; F9 pairs carry a stated difference or are one operation |
| AX-41 | Result card: sentence, chosen-parameter report, cautions, inference controls, next steps | Spec: sentence templates produce the documented text for fixtures; independent review of 12 result cards against the reference method |
| AX-42 | Sensitivity strip for `live` parameters, labelled exploratory, with the sweep recorded in provenance | Headless: six small maps for one parameter render from one compiled graph within the memory reservation; no significance classes in the strip for sampled previews |
| AX-43 | Weights as a named reusable object with an inspector | Two analyses share one weights object; changing it marks both stale |
| AX-50 | Analysis stack with status, visibility, reopen, run stale, undo and redo of edits; KE-WF-01, KE-WF-02 and KE-WF-04 to KE-WF-08 behind it | Gates of KE-WF-04 to KE-WF-07; stack survives sandbox save and replay; undo after a removed step restores it and marks dependents stale |
| AX-51 | Save a stack as a template; JSON export (the JSON half of KE-WF-11; TypeScript export stays KE-WF-11) | A saved template passes the AX-30 parser and reruns on a second dataset |
| AX-60 | Command palette over questions, templates and operations, scoped to the active dataset | Headless: three keystrokes and Enter open a template with slots filled |
| AX-61 | Assistant: search, describe and propose tools; plan validated by the template parser and requirement checker; whole-plan approval (KE-WF-09) | Spec: invalid plans rejected with the requirement text; a fixture of unanswerable requests is refused, not attempted |
| AX-70 | Measure: the reference-configuration manifest, then scripted journeys for the eight samples before and after | Baseline recorded before AX-01 lands; the section 6 action targets are met for all eight samples when AX-32 lands; results recorded in this folder per release |

Order and dependencies:

- AX-70's baseline and manifest come first; they cost a script and a run.
- AX-01 and AX-02 depend on nothing else. AX-03, AX-04 and AX-05 are scoped above to what can be done before the descriptors; their full forms need AX-20 (spacing), AX-15 (presentation) and AX-13 (requirements).
- AX-10 to AX-15 are one contract change and should be agreed with the binding effort before fanning out by binding family.
- AX-20 needs AX-12 for roles. AX-21 needs AX-12, AX-13 and AX-20. AX-22 needs AX-21, and gives a one-action result only once AX-31 or AX-40 provides something to open.
- AX-30 carries KE-WF-03. AX-31 needs AX-30, AX-12, AX-14 and the new recipe bindings. AX-32 needs AX-20 for automatic mapping.
- AX-40 and AX-42 need the descriptors and the lifecycle safeguards named in section 4.7; if those arrive with KE-WF-04, AX-40 follows it.
- AX-50 is the largest piece and can start once AX-02 has produced the adapter.

Not in the plan: a node canvas (revisit after AX-50 is in use; React Flow is the candidate), automatic workflow synthesis, bringing user data to the website or the showcase (the shared vocabulary makes it possible later), community-contributed templates.

## 6. Measures

| Measure | Today (from code; measured values are in the next table) | Target |
| --- | --- | --- |
| Actions from a loaded sample to a first analysis result | 4 for point statistics, 9 for a cartogram; the catalog is unavailable for auto-prepared samples and takes about 8 for an eligible non-resident single-input dataset | 1 from a suggestion or a paired sample; 2 on user data with mapping |
| Bindings reachable from an auto-prepared dataset | 0 of 130 through the catalog | All whose kinds and caps fit |
| Parameters whose default is derived from the data | A handful (bandwidth, variogram) | Every scale parameter (distances, cell sizes, bounds) |
| Rejections first seen at execution | Geometry and capacity for every catalog run | None for declared requirements |
| Results published with a legend and a sentence | Hand-written panels only | Every template and the named operations |
| Suggestion quality | No suggestions | Expected analysis in the top three for every fixture profile |

### Measured baseline (2026-11-14)

Recorded by `examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs` in headless Chromium with a real WebGPU adapter, on the working tree at `fbb78ec5` plus uncommitted changes; details, the manifest and screenshots are in [baseline/](./baseline/README.md). Times are about one second each and are a script's lower bound, not a person's.

| Journey | Predicted from code | Measured actions |
| --- | --- | --- |
| Gallery sample to a first analysis result (eight samples) | 5 to 6 | 8 to 9: 6 or 7 to load and publish the sample, then 2 |
| Point hot spots from a loaded sample | 4 | 11 with the default field, 12 with a meaningful one |
| Catalog operation on its built-in sample to a published layer | 5 | 6 |

Also measured: the analysis panel's default numeric column is the first column, which is `longitude` for three samples, so the default analysis is of a coordinate; publishing a sample does not select it in the analysis panel; the catalog's dataset list never contains a prepared dataset (tried through the gallery and through Add Data); a published catalog result is one cyan colour with a legend that names no field; IDW and kriging open with the fixed box west -74.1, south 40.65, east -73.9, north 40.85, which happens to contain all four gallery samples that carry coordinates. Not driven: cartogram, lattice hot spots, OLS, DBSCAN and polygon statistics.

## 7. Decisions for the maintainer

1. **Where the shared vocabulary and the suggestion rules live.** Recommended: questions, kinds, measurement levels, requirement ids and default rules in `@luma-contributors/analysis` beside the ontology; profile, bindings, templates and engine wiring in `@luma-contributors/kepler`; UI in the host.
2. **Converge the two form systems?** Recommended: yes, onto the generated sheet, one operation at a time, keeping bespoke result views. The alternative is to keep both and label the differences (F9).
3. **Role guessing.** Recommended: guess counts, amounts and rates, show the guess, let the user change it, warn and do not block. The alternative is to ask once per numeric column on load.
4. **Template scope.** Recommended: the twelve of wave 1 first, the other fifteen of section 4.6 next, specialist recipes as a third tier.
5. **Sample data for templates.** Which datasets may ship (this is decision 11 of the KE-WF report); resolve the `s1-chicago-tracts` record conflict.
6. **Assistant route.** The host's own router, or Kepler's assistant with this repository's templates registered as tools. Kepler 3.3 is adding its own analysis tools and an MCP package, so the choice affects how the two catalogs appear together.
7. **Order against KE-WF.** Recommended: AX-0 to AX-3 before the executor, with KE-WF-03 (document and parser) pulled forward into AX-30, because suggestions and single-step templates need the document but not the executor.
8. **Canvas.** Recommended: defer, decide after the stack has been used.
9. **Website.** Whether the website's "What do you have?" should later accept a dropped file and reuse the profile and suggestion engine.
10. **What the rows are.** Whether to ask the user "events, objects or samples of a surface" when a suggestion depends on it (recommended), or to leave it to the choice of template.
11. **Preview policy.** Recommended: no significance classes from sampled or viewport previews, and full-population recomputation before an inferential result is kept.

The KE-WF report's decisions 1 (hand-off between steps) and 2 (binding contract) are answered in part here: AX-02 is the v1 adapter, and AX-12 to AX-15 are the contract extension to agree with the binding effort.

## 8. Limits of this research

- No browser, GPU, build or test run was made. Action counts and latencies are from reading code.
- The checkout was mid-edit by two other sessions. Counts (130 bindings, 529 operators, 417 scenarios) describe the working tree on 2026-11-14 and will move.
- Which recipes have Kepler bindings (section 4.6, last column) is from a grep of binding ids and panel names, not from running them.
- External notes vary in depth. About a third of the node-builder note is unverified. Esri dialog details, the Tableau Show Me rule table as shipped today, and several live-preview and onboarding examples were not confirmed on primary pages. Template gallery evidence is descriptive; no study compares gallery sizes.
- The default rules in 4.7 are documented vendor and literature defaults, not tested here against this library's operators. Esri's spatial bandwidth formula was not confirmed.
- A profile cannot establish every precondition. The plan relies on `unknown` being shown honestly; how often that leaves a suggestion unconfirmed is not known.
- An independent Codex review of this report (2026-11-14) checked the internal counts and the 27 recipe names against source and the external claims against the notes. Its corrections are applied: the template table's host column, the split of interpolation and of change of support, the dependency order, the requirement evidence states, the preview and parameter-search cautions, and several claims in section 3 that had been stated more strongly than the notes support.
- User research was not done. The measures in section 6 are proxies until AX-70 records real journeys.

## 9. Implementation status (2026-11-14)

The library layer landed in the working tree after the Kepler port (KE-PORT wave 3). It is plain CPU TypeScript in new directories: `modules/analysis/src/gpu-spatial-analysis/ontology/analysis-experience/` (sections 4.3 to 4.5 and 4.8: questions, kinds, measurement levels, roles, requirements, spacing statistics, default rules, classes, sentences, cautions) and `modules/kepler/src/profile/`, `descriptors/`, `presentation/` and `curated/`. The remaining rows are in the AX table of [the roadmap](../../roadmap.md): all host forms and wiring, the workflow document, the gallery and the website page.

| Fact | Value |
| --- | --- |
| Catalog bindings with a hand-written descriptor | 136 of 217; the 81 ported the same day get a derived descriptor and are not suggested |
| Curated analyses | 27, of which 16 run and 11 wait for a binding |
| Bindings added | Six CPU-reference bindings (`curated-*`) that fuse existing steps |
| Tests | 30 spec files, about 2,490 tests, in the node project |

How it was checked. Implementors worked in a private copy from contract type files written first. Three independent review rounds followed, each reviewer writing its own probes against real CPU runs: eight reviewers, then seven, then three verifiers re-running the earlier findings. Four fix rounds answered them. Each round found fewer and narrower defects; the last verifier probe that still fails expects a run to be refused when its coordinate system contradicts the layer's declaration, and the layer instead runs with a warning (the choice is the user's most recent statement, and a host may declare a default the user has to be able to overrule). Defects found in existing catalog bindings are in [binding-defects.md](./binding-defects.md); the session that owns the bindings fixed them the next day, and that file's status section says what changed in the layer as a result.

What the layer cannot do, measured where a number is given:

- Before the binding fixes, `recipe-hot-spot-lattice` flagged at least one cell in 39 of 220 pattern-free datasets (17.7 percent) with Benjamini-Hochberg on. The fixed binding uses conservative binomial tails; its owner measured 2 of 220 (0.91 percent). The other numbers in this list predate the fixes and have not been repeated.
- The rate cluster map recovered a lone planted outlier in 12 of 20 runs.
- The default clustering radius (k-distance knee, else three times the median nearest-neighbour distance) fragments irregular concentrations and merges uniformly repeated locations.
- A requirement that rests on an estimated distinct count is always `unknown`; edge and node capacity of a network is estimated, not exact.
- Row semantics (events, objects, surface samples) are always a low-confidence guess and therefore always a question.
- `network-k-function` works in length mode only; the fixed binding rejects property costs.
- Nothing was run on a GPU or in a browser. The fused bindings are CPU references.

## 10. Implementation pass (2026-11-15)

The host and workbench wiring, the remaining rows of section 5 and a whole-diff review were done on 2026-11-15 (AX-PASS). The per-row status, the provisional decisions and what remains open are in the AX section of [the roadmap](../../roadmap.md). Measurements from the earlier sections (for example the rate cluster outlier recovery) were not repeated.

## Notes index

| Note | Holds |
| --- | --- |
| [internal-kepler-host-journey](./notes/internal-kepler-host-journey.md) | Every route from a loaded dataset to a mapped result in the Kepler host, action counts, what is inferred today, friction list |
| [internal-kepler-binding-metadata](./notes/internal-kepler-binding-metadata.md) | Binding census from source, what the metadata can and cannot drive, saved recipes versus templates, result presentation |
| [internal-analysis-ontology-recipes](./notes/internal-analysis-ontology-recipes.md) | Data kinds and operator counts, the 41 recipes with steps and questions, preconditions, data-driven parameters |
| [internal-lenses-website](./notes/internal-lenses-website.md) | Scenario and pack structure, licences by dataset family, the three vocabularies, the website visitor's route, candidate curated chains |
| [internal-deck-examples](./notes/internal-deck-examples.md) | Explorer and showcase interaction patterns, scheduling code, what the Kepler host lacks |
| [gis-analysis-ui](./notes/gis-analysis-ui.md) | ArcGIS Pro, Map Viewer, Insights, Esri optimized tools, QGIS, GeoDa |
| [web-map-platforms](./notes/web-map-platforms.md) | kepler.gl, CARTO, Felt, Foursquare, Fused, Earth Engine, openEO, SQLRooms, Mosaic and others |
| [type-driven-recommendation](./notes/type-driven-recommendation.md) | Show Me, Voyager, Draco, Lux, SeeDB, semantic type detection, measurement levels |
| [geo-workflow-synthesis](./notes/geo-workflow-synthesis.md) | Core-concept data types, APE and loose programming, LLM benchmarks, question taxonomies, openEO and other workflow standards |
| [node-builders](./notes/node-builders.md) | KNIME, Alteryx, Orange, ComfyUI, Blender, Observable canvases, evidence on visual programming, canvas libraries and licences |
| [step-lists-and-provenance](./notes/step-lists-and-provenance.md) | Power Query, Tableau Prep, OpenRefine, Shortcuts, notebooks, Dagster, geoprocessing history, Trrack |
| [curated-templates-and-guidance](./notes/curated-templates-and-guidance.md) | Template galleries, field mapping, wizards, the defaults table, latency evidence, result explanation, onboarding |
| [ai-assistants-and-tool-catalogs](./notes/ai-assistants-and-tool-catalogs.md) | Kepler and vendor assistants, benchmarks, tool-catalog engineering, plan approval |
