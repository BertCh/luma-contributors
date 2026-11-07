# Dataframe stories and auditable functionality coverage

## Current source and acceptance snapshot

The pinned final wave 22 source audit is [functionality-source-final.md](functionality-source-final.md),
with complete callable, method and option inventories in JSON. It contains 174 published stories
and 892 public callable export entries. Before/after fingerprints match all 4,172 TypeScript inputs:
`7d3648ef59ab450e79eaa37c06749e5f59c832ae8167f8907b3439e3d6ccf24e`.
All 91 anchors across 23 manual paths match exactly. The prior wave 21 inputs are archived with
checksums; runtime acceptance remains separately tied to each tested production bundle.

[Option scope](functionality-option-scope.md) gives 25 bounded decisions and retains every literal
candidate in JSON. [Acceptance](acceptance-ledger.md) records the actual tested bundles and scopes.
County 17 and fire 22 pass on wave 14; OSM probe membership passes one three-state return on wave 15.
The finite domain queues include the wave-19 circuit decision: four numerical acceptances, four
strict publication-withholding cases and recovery. Strong cases are not certified predictions.
The complete wave-21 atlas sweep has 173 passing stories and one mobile legend failure across
174 stories and 994 steps. Focused wave-22 display corrections and full repository tests pass;
[final integration evidence](integration-final.json) preserves all bounds, failures and warnings.
Source checks and broad tours do not establish independent numerical acceptance or an
entire option union.

The entries below record research and implementation history. Earlier counts and queue states are
historical; the linked source snapshot and current acceptance table supersede them.

## Scope and evidence standard

This track owns the dataframe chapter, `scripts/report-showcase-coverage.mjs`, its new
`showcase-functionality-inventory.mjs` companion, inventory acceptance test, and correction of the
USGS archive catalog description. Dataset-wide manifest validation is owned by the integration
track. Existing source changes are preserved; this track does not commit or run root formatting,
builds, or installs.

One later county correction worker violated the formatter scope by passing file arguments to
the root `yarn lint fix` wrapper, which scanned the workspace and reported 24 fixes. Parent was
notified immediately; no shared edits were reverted. The command/log and uncertainty are recorded
in `county-query-semantics.md`. This is not counted as a successful integration lint check.

The previous reporter measured direct constructor/function imports. Its 492 public class/recipe
entries are **not 492 independent algorithms**, because family reexports repeat shared classes.
A constructor call establishes a static reference; it does not establish graph encoding, visible
outputs, a meaningful option demonstration, or numerical agreement. Conversely, fluent dataframe
queries construct implementation classes internally and were incorrectly presented as missing by
an import-only audit.

The enhanced optional audit uses TypeScript-resolved signatures against current graph source.
It inventories all callable exports, public declared methods and object option contracts, including
literal unions and booleans. It follows runtime local imports (including dynamic imports) from
scene definitions and excludes type-only imports, scene snippets, metadata and comments from call
evidence. Every call has a source location and statically reachable scene IDs. It distinguishes:

1. Public declaration with no observed scene compute/helper call.
2. A call in a helper with no statically reachable scene.
3. A type-resolved call reachable through a scene's runtime import graph.
4. Browser execution and meaningful output changes — requires separate browser evidence.
5. Numerical validation and option sensitivity — requires reference checks, beyond static analysis.

Only the first three levels are measured by the script. An explicit literal option is credited as
**static candidate evidence**; dynamic expressions are recorded, with no inferred option coverage.
Defaults and internal recipe composition are deliberately not inferred. An unobserved option is
an investigation queue, not proof of absent behavior. Inherited methods are not yet inventoried.

## Reproducible inventory

```sh
node scripts/report-showcase-coverage.mjs --format=json --functionality
node --test scripts/showcase-functionality-inventory.test.mjs
```

`coverage-baseline.json` is the initial direct-import baseline. `functionality-baseline.json` is the
first type-resolved snapshot captured during parallel implementation, already including an early
version of the new earthquake story; it is explicitly **not** a clean before-change baseline.
`functionality-gaps.json` is the compact domain/role/option review queue from that snapshot.
The first typed snapshot measured 892 callable family export entries, 412 with reachable static
calls, 474 with no static calls, and 49 of 554 declared public methods with static call evidence.
These totals include utilities, infrastructure and compiler-result objects separately tagged by
role; there is no claim of complete user-facing coverage.

The inventory acceptance test uses a self-contained fixture to check aliased constructors,
fluent query methods, explicit option variants, unreferenced helpers, type-only imports, private
methods, exported utility functions, and fake snippets. It passes. Option evidence now uses each
method's own calls, rather than mixing unrelated methods' option names.

## Domain findings delivered to planners

- Geometry and observations: spherical coverage, topology, precision/repair, hull/triangulation,
  sampling and point-process variants deserve distinct interpretation, not more constructor rows.
  Existing `cell-pyramid` invokes `addPeriodComparisonRecipe`, which composes cell comparison;
  existing point-pattern scenes already generate 39 polygon-null replicates manually. Missing
  direct calls to `GPUCellTableCompare` or `GPUPointPatternEnvelope` therefore do not mean these
  behaviors are absent. The geometry planner confirmed this nuance.
- Statistics: rank/local-statistic variants, multiple testing, model diagnostics, ML/panel/regime
  models, nested interpolation validation and segregation counterfactuals remain large review
  areas. An option-aware queue was sent to the statistics planner.
- Networks and movement: disruption, direction/cost modes, unique service catchments, trajectory
  restriction/encounters and routing alternatives need visible consequences and failure states.
- Earth systems: fractional raster support and distribution, conservative remapping, registration,
  multiband/temporal variants, branching flow and hydrological boundary conditions remain broader
  than a single raster or terrain class count. Point-cloud ICP/change/SMRF are separate workflows.
- Dataframe: filtering, null semantics, derived expressions, global versus per-batch ranking,
  histogram domains and unique-key join semantics are meaningful user functions; compiler-generated
  query classes must be connected to the fluent methods that actually execute them.

Each domain planner received the raw gaps early and the complete option-aware machine inventory
when available. This track does not claim those gaps are all implemented.

## Research and dataset decision

Reuse the bundled **USGS ComCat event archive** for a coherent dataframe notebook. This avoids an
unnecessary download, permits offline reproduction, and gives the query operations a scientific
question: which records survive a query, and how does the denominator change?

Primary sources consulted on 2026-10-25:

- [USGS ComCat overview](https://earthquake.usgs.gov/data/comcat/) describes source parameters and
  contributed seismic-network products. Its catalog support varies by network, place and time.
- [USGS FDSN event service](https://earthquake.usgs.gov/fdsnws/event/1/) documents query parameters,
  event types, magnitudes and output formats. The bundled file is a fixed archive selection, not
  a live call to that service and not a claim of exhaustive current completeness.
- [USGS magnitude types](https://www.usgs.gov/programs/earthquake-hazards/magnitude-types) establishes
  that different magnitude estimators have different ranges and physical interpretation. The
  story ranks recorded magnitudes and makes no energy, damage, risk or prediction claim.

Local binary audit (little-endian data as declared in the manifest):

| Measure | Actual bundled value |
| --- | --- |
| All events | 77,231 |
| Earthquake event type | 77,215 |
| Mine collapse / mining explosion / volcanic eruption / landslide | 2 / 2 / 11 / 1 |
| Magnitude range | 4 to approximately 8.2 (float32) |
| Depth range | approximately −1.92 to 681.238 km |
| Negative reported depths | 4 rows; retained, not automatically invalid |
| First / last UTC event | 2020-01-01 00:28:20 / 2024-12-30 23:56:30 |
| Earthquakes of magnitude ≥6 | 626 |

The source's embedded prose says 522,982 events while its window metadata and local bytes say
77,231. The raw embedded provenance is preserved for audit. The dataset catalog now describes
77,231 archived records, the December 30 cutoff and the 16 non-earthquake records, without claiming
“every” event or a complete full year. These numbers were independently sent to both geometry and
dataframe implementers as acceptance references.

## Story design and implementation review

`earthquake-query-lab` is a six-step notebook: bounded archive → filters → magnitude histogram →
global versus year-partitioned top-K → event-type lookup/anti join → exploration. The geographic
view uses an abyss ground and fixed magnitude scale. Query selection and ranking must visibly
change GPU-derived map output, with a dim full-archive context layer. Calendar partitions are
explicit application-created analytical batches; they must not be called original source batch
boundaries. Negative depths stay in the default selection.

The story must distinguish event rows from earthquake-only rows, global K from K in each year,
empty queries from valid zero-valued summaries, and lookup/publication failures from matches.
Charts and map legend must describe the actual buffers and colors. The magnitude histogram is
not a frequency-of-future-events forecast. Mean magnitude is a descriptive mean of reported
logarithmic values, not mean energy. The event-type dimension is an adapter dictionary, not a
second independent observational dataset.

The first implementation review rejected an unchanged context-only map, a categorical legend
with no matching layer, metadata-only lookup readouts, frozen theme state, unguarded stale
readback and needless every-frame recomputation. These were returned to the Luna implementer for
correction. `GPUDataFrameQuery.compile()` internally calls synchronous `graph.compile()` in the
current public API; this is a library API constraint, not an asynchronous scene compilation claim.
The story must expose actual encoded map graphs through the scene inspection interface where
possible; private query graphs must not be exposed through casts.

## Verification status

- Inventory fixture: passed (`node --test scripts/showcase-functionality-inventory.test.mjs`).
- Detailed inventory generation: passed against current public sources and scene helpers.
- Dataset binary count/type/range/date audit: passed as listed above.
- New scene typecheck and browser/numerical verification: in progress in the implementation and
  integration tracks; no completed-browser claim is made here yet.
- Root `nvm use`, `yarn install`, `yarn build`, `yarn test`, `yarn lint fix`, `yarn website:build`
  and `(cd website && yarn build)`: delegated to integration, not run by this track.

## Final track review and handoff

The dataframe implementation now includes `select`, a derived `yearsFromStart` column used by the
filter, magnitude/year/depth/event-type predicates, global count/mean/max/min reduction, fixed-domain
histogram, stable global and explicit year-partitioned top-K in both directions at K=50/250,
archive-wide dictionary lookup and an archive-wide anti join. The filtered selection mask drives
orange map points, and the GPU global permutation drives white ranked stars. Rank readbacks are
bounded; the callback requests a layer refresh, and changing options immediately rejects stale
summaries and clears stale ranked instance counts. The context toggle leaves query output visible.
An explicit categorical legend matches the query and ranking marks.

The query compilation API still exposes no public compiled graph getter, so `getCompiledGraphs()`
is empty for this scene. The query encoders do execute; the private graph is not accessed through
casts. This limits the shared “Under the hood” graph inspection UI and must be recorded rather than
replaced with a dummy graph. Query compilation is also synchronous inside the current library.

The updated inventory separately traces direct function invocations from scene create/lifecycle
callbacks. A dead exported function in an otherwise reachable compute module now receives
`module-only-static-call`; its call sites are retained without being promoted to reachable invocation
candidates. Registered callbacks remain potential execution, not proof. The fixture explicitly
checks this case and rejects misleading annotation IDs when finding scene identity. Built-in
TypeScript array/prototype properties are excluded from option contracts; the fixture checks a
Float32Array function parameter. Dynamic option controls must still be manually checked against
the scene: e.g. NetworkSnapping directions and existing rainfall models are real controls even
when no constructor literal appears in the audit. The statistics and movement planners supplied
these examples, and their matrices remain the behavior-level authority.

`functionality-after.json` is a complete, minified machine-readable integration snapshot from the
updated reporter. At its capture it had 892 callable family entries, 361 potential invocation paths,
468 without any static call, and 45 of 105 non-protocol/lifecycle public operations with static calls.
The remaining declared methods include 386 `getCommandNodes`, 30 `addToGraph` and 11
`getInferenceMetadata` methods, which are graph protocol, not 427 missing story controls. The
legacy direct-call summary at this snapshot was 247/492. These values are **not directly comparable**
to the first typed snapshot because the audit now distinguishes function invocation from module
imports and excludes built-in option properties. Ongoing parallel stories can change the totals;
rerun the documented command for the final integration state.

Final scoped checks:

- Inventory fixture: passed after all reachability and option fixes.
- Scoped Biome: passed for the three TypeScript files; the repository ignores `.mjs` in that check.
- Showcase typecheck: no dataframe diagnostics; latest complete invocation was blocked only by the
  concurrently created `terrain/point-cloud-support.scene.ts` referencing its not-yet-created compute
  file. Log: `/tmp/data-coverage-typecheck.log`.
- Browser/numerical checks: handed to integration with exact reference values 77,215 default
  earthquakes; 77,231 including other types; 626 earthquakes at M≥6; anti join 16; archive lookup
  77,231 found and zero missing. Ranking direction/count must visibly change white-star positions.
- An initial `yarn exec biome` attempt could not resolve the binary; the repository-local
  `./node_modules/.bin/biome` succeeded. No root checks were substituted by these scoped checks.

### Subsequent production validation

`earthquake-query-lab` production retest passed all six story steps with zero reported errors
after join dictionary symmetry and GPUVector column-name corrections. Parent report:
`/tmp/luma-showcase-overhaul-20261025/wave3-fixes-browser/report.json`. Reviewed screenshots
01 and 06 show 77,231 archive rows and 77,215 current earthquakes, matching source bytes.
The separate `check-data-topology-reference.mjs` harness tests the 626 magnitude≥6 case and
all-type switch; its numerical run remains pending. The functionality-inventory fixture
was rerun successfully: one test, zero failures, zero TODOs. Final global inventory is
regenerated by the parent after all concurrent waves rather than claiming current snapshots final.

The targeted production dataframe numerical test now passes: 77,215 earthquake-only rows,
77,231 with other event types, 626 earthquake-only magnitude≥6 rows, 77,231 successful category
lookups with zero missing, and 16 anti-join rows. Report:
`/tmp/luma-showcase-overhaul-20261025/data-topology-reference.json` (21 combined scenarios passed,
including 18 fire topology cases). This is runtime numerical evidence, distinct from static
class/method/option inventory and the earlier default story-step tour.

## Concurrent expanded callable decision audit

`functionality-concurrent-decisions.json` is a separate concurrent snapshot, not a replacement for the final integration inventory. It retains all892 exported callable entries, their declared public methods and option contracts; static reachable calls, module-only calls, composed paths, equivalent output contracts, support APIs and pending behavior decisions remain distinct. The source-scan timestamp is separate from the later decision-reconciliation timestamp. No option union or runtime acceptance is inferred from a literal, dynamic property or constructor reference.

Confirmed composed paths include period-recipe cell comparisons, rate-recipe empirical Bayes, regression diagnostics, ML weight traces, local-permutation multiple-testing adjustment, transition diagnostics, clock encounters, counterfactual decomposition, spectral band math and the dilation subclass. Each record cites implementation evidence and limits. Assuncao-Reis standardized output is provided by the existing empirical-Bayes rate scene; this is an equivalent output contract, not a direct standalone class call.

Meaningful gaps prompted explicit assignments: geometry coverage/sampling/linework/footprint operations; BIXI cell OD flow and interaction/allocation; real population-support/ancillary assumptions; scalar projection conventions; raster quantity totals, contrast/profile/transport. Remaining recipes are not dismissed merely because their primitive components exist: building wall adjacency, boundary-derived tessellation and obstacle visibility expose new outcomes; surface preparation retains fill/filter provenance; resistance and regional zonal recipes require an explicit decision.

Fine-option review preserves uncertainty. Fixed source area pixels/north-first rows/60m cells are geometry contracts. Point-neighborhood serial/cooperative is execution decomposition, not a separate scientific estimator; SMRF onlyGround is a real output-class preservation policy that remains unshown. Natural-breaks exact/approximate, region adaptive merges, raster threshold boundary modes and terrain curvature methods cannot be declared complete from their parent class calls.

The focused AST fixture was rerun and passed(1test,0failures,0TODO). It checks metadata/dead-code exclusion, typed fluent calls, literal candidate limitations and built-in property exclusion. Source calls are not production acceptance: the separate scenario reports contain that evidence.

### Current closure reconciliation

The interim priorities ledger was rewritten as a single current snapshot. H3/site-mesh and
pixel-boundary rebuilt acceptance are retained; assigned spherical, linework, KDE, real
population support, resistance and recipe work no longer appear unassigned. Statistics
also owns the distinct rainfall LOO/variance-calibration recipe and GPU fitting comparison;
movement owns the actual GPU rose extension. Existing CPU rose binning is not GPU contributor
execution.

The current source snapshot contains 892 callable exports, of which 604 have potential direct
static invocation paths. A focused fixture now covers callbacks nested in passed object
arguments (such as GraphSwap build factories), while unpassed callback objects remain dead.
The fixture passed one test with zero failures and zero TODOs. No static callback edge proves
that every conditional branch actually ran.

`functionality-manual-paths.json` reconciles the remaining 21 module-only records using concrete
scene creation, indexed factory or stored callback, selected control, graph compilation and
encode paths. Domain owners confirmed the indirect paths. Fresh runtime evidence is absent
for several legacy branches and is explicitly not manufactured from their source use.
`functionality-concurrent-decisions.json` retains the raw evidence alongside these manual paths.
Every substantive top-level entry now has a behavior/composition or implementation-ownership
decision; option acceptance and pending runtime checks remain explicit, so this is not a
completion claim.

Reproduce a concurrent reconciliation after a raw scan with:

```sh
node scripts/report-showcase-coverage.mjs --format=json --functionality > /tmp/functionality-raw.json
python3 dev-docs/research/showcase-overhaul-2026-10-25/reconcile-functionality.py /tmp/functionality-raw.json
```

The reconciliation script preserves explicit decision rationales and manual path references.
Final integration must update domain acceptance evidence and reassess changed source paths;
this command does not infer browser acceptance or make `functionality-after.json` final.
