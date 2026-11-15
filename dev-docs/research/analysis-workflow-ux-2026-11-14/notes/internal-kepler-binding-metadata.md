# B. Metadata for suggestions, forms and templates

**Verified:** The current source registry contains **130 bindings and 531 parameter occurrences**. It supports geometry-based candidate filtering and generated parameter controls. Scientific field requirements are mostly strings, prose and runtime checks.

This is a working-tree review, following the earlier study at `dev-docs/research/kepler-workflows-2026-11-14/README.md:1`. Recipe, session and explorer files were modified during this review; findings describe their current contents. The empty change-of-support registry looks like an implementation placeholder, not a reported defect (`modules/kepler/src/analysis/catalog-platform5-recipes-change-of-support.ts:7`, **inferred**).

## Source counts

**Verified; command run:** `node -e "$catalogCensus"`. The inline program transpiled source into memory, evaluated binding declarations and factories, expanded imported registries, and aggregated `KEPLER_CATALOG_BINDINGS`. GPU/runtime dependencies were stubbed; metadata values were checked for unresolved stubs. Nothing imported repository `dist`. A repeated census returned the same totals.

Registry assembly: `modules/kepler/src/analysis/catalog-bindings.ts:488`; nested recipe registration: `modules/kepler/src/analysis/catalog-platform5-recipes.ts:887`.

| Measure | Verified count |
|---|---:|
| Bindings / unique binding IDs | 130 / 130 |
| Input specifications / output specifications | 173 / 144 |
| Parameter occurrences, counting shared specifications once per binding | 531 |

| Domain | Verified bindings |
|---|---:|
| geometry | 64 |
| network | 20 |
| analysis | 20 |
| aggregation | 11 |
| clustering | 5 |
| raster | 4 |
| density | 3 |
| interpolation | 3 |

**Verified:** The following count each binding once per geometry kind. Totals overlap across kinds and outputs; these are declared specifications, not observed execution results. Fields: `modules/kepler/src/analysis/catalog-binding-types.ts:47` and `:55`.

| Geometry kind | Bindings accepting it on any input | Bindings declaring it on any output |
|---|---:|---:|
| point | 83 | 33 |
| line | 57 | 31 |
| polygon | 52 | 51 |
| none | — | 25 |

**Verified:** Parameter-count distribution, expressed as **parameters → bindings**:

`0→5, 1→20, 2→26, 3→24, 4→10, 5→13, 6→4, 7→6, 8→7, 9→3, 10→6, 11→1, 12→3, 13→2`.

The schema is at `modules/kepler/src/analysis/catalog-binding-types.ts:34`; the registry is at `modules/kepler/src/analysis/catalog-bindings.ts:516`.

| Parameter metadata | Verified occurrences |
|---|---:|
| `unit` | 105 |
| `minimum` | 213 |
| `maximum` | 213 |
| `defaultValue` | 531 |
| `options` | 182 |

**Verified:** Bindings have no sample field. The example host supplies samples for **17 bindings**, covering every named input of those bindings; **113 have none** in that switch. This counts sample availability, not successful runs with defaults. Sources: `modules/kepler/src/analysis/catalog-binding-types.ts:64`; `examples/deck/kepler-compute/src/analysis-catalog-input.ts:206`.

## Input requirements and defaults

| Finding | Evidence and status |
|---|---|
| Geometry and capacity are machine-readable: `inputs[].name`, `accepts`, `maximumFeatures`, `maximumVertices`; outputs have `maximumRows`. | **Verified:** `modules/kepler/src/analysis/catalog-binding-types.ts:47`. |
| Preflight requires nonempty collections and checks feature count, vertex count and accepted geometry. Parameter normalization checks kinds, ranges, options and lengths. | **Verified:** `modules/kepler/src/analysis/catalog-bindings.ts:529`, `:602`. |
| There is no input-schema declaration for required numeric/categorical/time columns, field multiplicity, optional field roles or conditional requirements. Property choices generally appear as ordinary string parameters. | **Verified:** `modules/kepler/src/analysis/catalog-binding-types.ts:34`; regression strings at `modules/kepler/src/analysis/catalog-platform4-regression.ts:312`. |
| A named value property can be required only when execution reads it; queen/rook neighborhoods additionally require polygons. | **Verified:** `modules/kepler/src/analysis/catalog-platform3-b-bindings.ts:133`, `:140`. |
| Requirements can depend on another parameter: choropleth `count` needs no value property; other statistics do. No conditional form rule expresses this. | **Verified:** `modules/kepler/src/analysis/catalog-platform5-recipes.ts:254`, `:280`. |
| Regression checks response/predictor names, 1–15 predictors, finite Float32-compatible complete cases, and enough retained rows at runtime. | **Verified:** `modules/kepler/src/analysis/catalog-platform4-regression.ts:46`. |
| Spatial weights are not a catalog input-port kind. Bindings expose neighborhood controls and build weights internally. Recipe types separately express point/contiguity weights, coordinate declarations and capacities. | **Verified:** `modules/kepler/src/analysis/catalog-platform3-b-bindings.ts:101`; `modules/kepler/src/recipes/analysis-recipe.ts:35`, `:66`. |
| CRS is a parameter choosing `EPSG:4326` or `planar`, not an input CRS compatibility requirement or transformation contract. | **Verified:** `modules/kepler/src/analysis/catalog-binding-types.ts:124`. |
| Recipe execution has additional admission metadata: prepared formats, row/chunk counts, estimated bytes, available memory and backend availability. These are separate from catalog input specifications. | **Verified:** `modules/kepler/src/execution/execution-router.ts:21`, `:99`. |

**Verified:** Catalog defaults are dataset-independent values. `getCatalogDefaultParameters` simply copies `defaultValue`; normalization substitutes the same value. For example, distance-band defaults to **1,000**, rather than deriving a distance from the dataset. Sources: `modules/kepler/src/analysis/catalog-bindings.ts:522`, `:538`; `modules/kepler/src/analysis/catalog-platform3-b-bindings.ts:113`.

**Verified:** Constant defaults can invoke automatic implementation behavior: `chunkRows=0` requests automatic GPU chunk sizing. That is resource sizing, not a recommended scientific scale (`modules/kepler/src/analysis/catalog-platform4-weights.ts:333`, `:431`).

**Verified:** Outside the catalog-default mechanism, data-derived configuration already exists: empirical variograms default maximum lag to one-third of the geographic data diagonal, and automatic fitting selects model parameters from empirical bins. Recipes can retain fit provenance. Sources: `modules/kepler/src/analysis/kriging-variogram-fit.ts:111`, `:244`; `modules/kepler/src/recipes/analysis-recipe.ts:146`.

## Saved analyses and template expressiveness

- **Verified:** `KeplerAnalysisRecipe` saves a versioned operation, dataset ID/revision, all/selected population, selection revision, backend policy, named columns and operation-specific scientific settings. Display metadata currently covers cartogram declarations. Restore checks source, selection and applicable external weights revisions. Sources: `modules/kepler/src/recipes/analysis-recipe.ts:220`, `:325`, `:970`.
- **Verified:** Sandbox state can package several sources and analyses, loader references and parameters, schemas, normalization, semantic selections, native configuration and opaque host extensions. Explicit rebinding updates revisions while checking source compatibility. Sources: `modules/kepler/src/recipes/sandbox-state.ts:21`, `:39`, `:161`.
- **Inferred:** These support saved, preconfigured analyses for concrete datasets. They do not directly express reusable dataset slots, abstract field roles, parameter expressions such as “distance from extent,” or a template’s exposed controls. Recipe operations contain concrete column names and values; sandbox rebinding does not remap those names. Sources: `modules/kepler/src/recipes/analysis-recipe.ts:174`, `:325`; `modules/kepler/src/recipes/sandbox-state.ts:186`.
- **Verified:** An adjacent role mechanism already exists for lenses: `CatalogLensFieldRole` supplies a numeric role and example value, with role-to-user-field mapping. It is separate from binding metadata and limited by scenario eligibility. Source: `modules/kepler/src/analysis/catalog-lens.ts:34`, `:59`, `:81`.
- **Verified:** Catalog invocations save operation ID, normalized parameters and input content references, rather than dataset-independent roles. Source: `modules/kepler/src/analysis/catalog-bindings.ts:735`.

## Result presentation

| Behavior | Evidence and status |
|---|---|
| Styles execute a supplied plan: color scales include linear, sqrt, log, quantile, equal-interval, ordinal and threshold; scalar mappings support size-like channels. Palette and column are explicit. | **Verified:** `modules/kepler/src/styles/style-plan.ts:20`. |
| Domains can be fixed, source-derived or selection-derived. Legends return domains, palette/range, labels/codes, finite/missing counts and invalid-log-domain status. | **Verified:** `modules/kepler/src/styles/style-plan.ts:7`; `modules/kepler/src/styles/resident-style.ts:59`, `:360`. |
| The styles module does not choose a layer or result field from a binding’s output specification. Its plan requires those presentation decisions from the caller. | **Verified:** `modules/kepler/src/styles/style-plan.ts:20`; `modules/kepler/src/styles/resident-style.ts:92`. |
| Catalog publication passes the returned table’s `geometryColumn` to Sources. Host publication creates a GeoJSON layer for that column, or point/flow/cell/trip layers for other explicitly supplied roles. The initial layer has a fixed cyan color. | **Verified:** `examples/deck/kepler-compute/src/app.tsx:503`; `examples/deck/kepler-compute/src/publication.ts:173`. |
| Tables without geometry or coordinate roles receive no default map layer. Examples include OLS coefficients, global spatial statistics, spatial-weights tables and Knox summaries. They still receive a table preview. | **Verified:** `modules/kepler/src/analysis/catalog-platform4-regression.ts:206`; `modules/kepler/src/analysis/catalog-platform3-b-statistics.ts:206`, `:308`, `:833`; `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:548`. |

**Verified:** Declared output geometry `none` cannot identify all table-only results: empirical-Bayes and multiple-testing bindings declare `none` while retaining source geometry. Source: `modules/kepler/src/analysis/catalog-platform3-b-bindings.ts:1120`, `:1162`.

**Inferred:** Catalog choropleth and hot-spot outputs therefore do not automatically receive meaningful statistic/bin coloring or a scientific legend through this publication path. Sources: `modules/kepler/src/analysis/catalog-platform5-recipes.ts:266`; `modules/kepler/src/analysis/catalog-platform5-recipes-hot-spot.ts:319`; `examples/deck/kepler-compute/src/publication.ts:193`.

## Dataset information available to suggestions

| Information | Availability without new GPU work |
|---|---|
| Names, source/prepared types, scalar formats, row counts, normalization conversions | **Verified:** retained in source/prepared tables and vectors: `modules/kepler/src/session/compute-session.ts:55`; `modules/analysis/src/analysis-core/graph-support/arrow-analytics-normalization.ts:64`. |
| Categories | **Verified:** prepared dictionaries retain labels. Dictionary size is available; it is not necessarily observed cardinality within a selection. `modules/kepler/src/session/arrow-stream-snapshot.ts:149`. |
| Nulls | **Verified:** retained/adopted Arrow paths carry per-batch null counts and validity. Resident-only analytics initialize null counts to zero, while lazy CPU cells start null until loaded; neither should be treated as a measured resident profile. `modules/kepler/src/session/adopted-arrow-dataset.ts:93`; `modules/kepler/src/session/resident-dataset.ts:98`, `:470`. |
| Ranges | **Verified:** resident datasets retain producer-set domains; explorer summaries expose requested domains/histograms. Computing absent resident domains requires GPU work. `modules/kepler/src/session/resident-dataset.ts:227`; `modules/kepler/src/exploration/gpu-resident-explorer.ts:92`; `modules/kepler/src/session/compute-session.ts:444`. |
| Cardinality and descriptive statistics | **Verified:** the inspector can request a GPU profile containing distinct estimate, nulls, range, moments and histogram. This API computes them; it is not an existing dataset-wide profile cache. `modules/kepler/src/exploration/gpu-table-inspector.ts:212`; `modules/kepler/src/exploration/gpu-column-statistics.ts:105`. |
| Time columns | **Verified:** explicit time metadata includes source type, unit, timezone, nullability and validity. `modules/analysis/src/analysis-core/graph-support/arrow-analytics-normalization.ts:40`. |
| ID columns | **Inferred:** no semantic primary-key/ID-column designation is exposed by the reviewed dataset types. Explorer `sourceRowIndices` are physical row identities, not inferred business keys. `modules/kepler/src/session/compute-session.ts:55`; `modules/kepler/src/exploration/gpu-resident-explorer.ts:100`. |

## Applicability signature

**Inferred proposed signature; not an existing API:**

```ts
declare function listApplicableBindings(
  input: {
    geometryKind: 'point' | 'line' | 'polygon';
    numericColumns: readonly string[];
    categoricalColumns: readonly string[];
    timeColumn?: string;
    rowCount: number;
  },
  bindings: readonly CatalogOperationBinding[]
): readonly {
  bindingId: string;
  matchingInputNames: readonly string[];
  otherRequiredInputNames: readonly string[];
  uncheckedRequirements: readonly string[];
}[];
```

- **Verified fields to use:** `binding.id`, `inputs[].name`, `accepts`, `maximumFeatures`; require positive row count. `maximumVertices` remains unchecked because the supplied profile lacks vertex count. Multiple inputs must remain explicit. Source: `modules/kepler/src/analysis/catalog-binding-types.ts:47`.
- **Inferred limit:** This lists candidates, not guaranteed runnable analyses. Numeric, categorical and time columns cannot reliably drive generic filtering because the contract lacks typed field requirements and conditional rules. Sources: `modules/kepler/src/analysis/catalog-binding-types.ts:34`; `modules/kepler/src/analysis/catalog-platform5-recipes.ts:254`.
- **Verified existing ranking hints:** `domain`, `capabilityIds`, `title`, `summary`, `engine`, `limitation`, `declaredBehavior`, outputs and capacities. **Inferred missing ranking metadata:** intended analytical question, required/optional field roles, statistical assumptions, recommended scale/default derivation, expected cost, and preferred result encodings. Source: `modules/kepler/src/analysis/catalog-binding-types.ts:64`.

## Opportunities

All **inferred**, directly from the findings:

1. Add typed and conditional field roles alongside string parameters (`catalog-binding-types.ts:34`).
2. Expose reusable dataset/field slots around concrete recipes (`modules/kepler/src/recipes/analysis-recipe.ts:325`).
3. Distinguish constant defaults from data-derived parameter policies (`modules/kepler/src/analysis/catalog-bindings.ts:522`).
4. Attach preferred result columns, encodings and legends to outputs (`modules/kepler/src/analysis/catalog-binding-types.ts:55`).
5. Expose profile availability and provenance so suggestions reuse measured metadata (`modules/kepler/src/session/resident-dataset.ts:227`).

## What I checked

- **Files:** Earlier study; catalog types/registry; extra, GPU and snap bindings; platform3 A/B bindings and shared helpers; platform4 registry, DGGS, raster, regression, spherical and weights; platform5 registry, LISA, raster, recipes and nested registries, regression, S2 and recipe helpers; catalog GeoJSON, lens and geodesy code.
- **Files, focused ranges:** Both recipe files; both execution implementations; session README, compute-session, resident/adopted datasets and stream snapshot; explorer, inspector and column statistics; all three style implementations; recipe-bindings and variogram fitting; host catalog input/panel, app and publication; Arrow normalization types and Earth constant.
- **Commands:** `cat`, `nl -ba`/`sed`, `rg`/`rg --files`, source-census `node -e` programs, and scoped `git status --short`. No files edited.

## Not checked

- GPU execution, sample-run success, browser appearance, tests, builds or performance.
- Every contributor implementation, full host replay behavior, complete lens registry or website metadata.
- `dist` contents or changes made after the final census.