## c01-naming
Name public inputs for their role, unit and index space. Reuse spellings and accepted values where all three agree. Use searchable reference-algorithm names and lowercase hyphenated strings for library-defined multiword options.
Check resolved constructor fields and packer settings together. Exercise a nondefault value through sibling entry points and an unknown value through JavaScript. Choose units and IDs whose competing interpretations give different answers.
The three larger changes are to expose `'lowest-row'` for deterministic row ties, migrate the three algorithm exports with their documentation and catalog identities, and add raster placement objects with named `origin` and `cellSize` alongside the existing tuple packers.

## c02-output-shapes
Declare each writable port's format, row domain and order, readable extent, tail policy and missing-value rule. `status.count` bounds the readable compact prefix, `…Rows` indexes a named source, `…Ids` carries application identities, and unqualified `validity` applies to the primary result in its own row domain.
Check declarations against public types. Bind a distinct destination to every advertised optional port and require a writer, a declared pass-through identifying its source, or rejection at construction.
The three larger proposals are producer-specific writable status types starting with the three weights builders, the proposed pre-alpha `GPUPairGather.output.status.count` rename, and row-domain metadata beside compact-pair, fixed-stride, generated-geometry and packed-summary declarations. Generate the API tables from that metadata and verify defined extents on a nonempty frame followed by a smaller frame.

## c03-status-vocabulary
Every producer capable of returning a plausible partial answer should expose retrievable `status.complete` for its declared row, query, stage or batch, with distinct causes for final overflow, candidate loss, invalid input and iteration limits. A caller may omit a destination when equivalent evidence remains available. Consumers must forward supplied loss, suppress the dependent result or reject the input.
1. Repair direct CPU weights and point-cloud handoffs. Pass producer results intact in paired clean/loss fixtures and assert both the downstream numerical result and the surviving cause.
2. Add `decodeGPUAnalysisStatus(layout, readback, row?)`, returning `state: 'complete' | 'incomplete' | 'unknown'` and scoped causes. Describe each layout’s encoding, count units, recognized values or bits and affected outputs. Preserve raw words, boundary results, approximation, fallback and provenance.
3. Make stopping and loss evidence retrievable when destinations are omitted. Reuse existing views or expose graph-owned views, and carry them with the data through recipes.

## c04-units-coordinates
Every spatial or temporal port should declare its logical quantity, physical representation, unit or source of units, and missing-value rule. Give fixed-unit scalars explicit suffixes and name the governing context for native-unit fields.
Check resolved public types against a per-port table. Test adapters with asymmetric fixtures that expose swapped axes, shifted origins, changed units and the difference between a valid zero and a missing value.
Prioritize Larger proposals 1, 2 and 5: carry frame and validity through classified points → DEM → flow → recorded raster, make projected-frame identity origin-aware and metric-result compatibility explicit, and extend the public API checker to inspect unit contracts on resolved props, nested bands and exported packers.

## c05-defaults-required
A minimal typed call must compute the selected mode or identify its unmet requirement before graph construction. Derive counts from exact shapes, make scientific choices explicit, and resolve ordinary CPU and GPU settings consistently.
For each mode, test omitted options against resolved settings, packed words, selected nodes, and a distinguishing result. Generate documented defaults, units, ranges, and mode applicability from those settings.
Prioritize an explicit Quantile LISA tail, executable minimal selections for Imhof shading, class breaks, and envelopes, and public types that encode the six verified input alternatives and 25 terrain output requirements while retaining runtime checks.

## c06-validation-errors
A public producer checks every host-knowable port and prop requirement before construction succeeds. A packer checks its complete input and destination before its first write, and graph-dependent checks precede the first graph mutation.  
Each refusal is an `AnalysisValidationError extends Error` carrying `code`, `contributor`, `id`, `path`, `received`, and `expected`. Its message starts with the instance ID and identifies the expected value or relationship and the value received.  
Adopt named constructor port descriptors for formats, presence, topology, and row relationships. Ship the structured error before alpha and migrate end-anchored assertions before enriching messages. Add single-prop mutation probes to contributor acceptance that verify the refusal phase, fields, absence of side effects, and a corrected same-ID retry.

## c07-ingress-packers
Name host adapters `prepare<Domain>Input` and return `{arrays, formats, counts, metadata}`, distinguishing live rows from allocated capacity. Where binding helps, use public luma.gl uploads in a separate adapter returning `{views, counts, metadata}`.
Validate source shape and row identity before allocation, apply masks before shared bounds or dictionaries, and check the representation actually written. Pair each adapter with a real constructor using empty, singleton, sliced, multi-batch, masked, truncated, narrowing and overflow fixtures.
The three larger proposals are to ship GeoArrow geometry ingress first, add independent raster-stack and spatial-weights adapters with explicit layout and edge policies, and migrate restriction-state preparation to the options-object signature before alpha, subject to ING-20.

## c08-egress-decoders
Export `decode<Producer><Output>(readback, layout?)` beside each producer whose output depends on packing, counts, offsets, scale, residuals or status. Mirror port names and nesting, distinguish bytes from scalar elements, share allocation dimensions with decoding, and return the evidence needed to interpret the result. Submission and readback timing stay with the caller.
Test through the public barrel with an exact logical view, a nonzero byte offset, dirty spare capacity, a short view and, where applicable, an incomplete or missing result.
Prioritize three changes: adopt shared output layouts and pure decoders for the ten byte-ambiguous producers and isolines, using K-shortest paths as the example; introduce the zero-bit one-period layout and sentinel-aware unpacking before alpha with explicit migration paths; keep materialization at the boundary through upstream Arrow and vector readers, a bounded packed-buffer reader, and family-specific domain decoders.

## c09-forms-parity
Give the direct contributor, both graph classes and the pure function one package-qualified operation identity, one vocabulary for semantic inputs and results, and one meaning for every explicit algorithm option. Record differences in device limits, reference algorithms, prepared resources and recipe names on the family page.  
Check registered exports in both directions, compare graph ports and run one named-input fixture through all available forms, checking every requested result and status at its documented tolerance.  
Prioritize extending the four-row forms catalog to family pages with runnable CPU examples, adding common logical result paths through compatible views or decoders, and distinguishing bounded calculations from unbounded references wherever an explicit option changes the scientific result.

## c10-composition
Expose a borrowed result port when allocation size, external IDs, axes, missing values or bounded loss affect interpretation. Carry logical shape, row identity, missing-value rules and status views through the consumer or named adapter to its final result.  
Check each advertised composition with a graph-declaration fixture and an adversarial data fixture covering permuted IDs, asymmetric axes, nonzero padding, an absent key and a lossy frame followed by a clean one.  
Prioritize three larger proposals: publish assigned-feature-row, live noded-topology and axis-labelled matrix ports; preserve stage-specific causes alongside a final completeness signal, starting with CMP-01, CMP-07 and CMP-09; and publish the stop-to-cube recipe with explicit event time, dictionary and weights order.

## c11-types
Public constructors should accept their exported props type. Capture selected ports once, reject unknown fixed-schema keys, and preserve each port’s name, format and presence in declarations, bindings and results. Check the resolved format at runtime wherever the declaration cannot establish it.
Exercise public entries with a valid literal, a widened variable, a bad key or format, and a read of an unrequested output. Pair those type fixtures with host assertions that declared keys match returned or resolved keys.
The reviewer’s first three larger proposals are to migrate the operation bridge to checked `getView` and presence-aware `getOutput`, starting with ensemble scores; give plain and diagnostic variogram fits separately named functions with stable results; and encode transfer-mask formats, derivative outputs, distance-field seeds and class-assignment keys in props while preserving valid alternate outputs and independent chunk layouts.

## c12-entry-points
Give each supported declaration one canonical import by user task. Publish a contributor’s named inputs, outputs, statuses, documented limits and direct preparation/readback helpers together, with explicit aliases where another family consumes the contract.
Extend the manifest-based declaration inventory to check those relationships, documentation import lines and workspace-only paths. Verify cross-format behavior by exchanging real objects between `import` and `require`.
Keep the seven domain paths and generate import guidance from the manifest. Migrate catalog users to the dedicated ontology entry before removing the parent re-export. Make mixed-format vectors, operations and recorded graphs part of packed release verification.

## c13-first-run
Take literal inputs through public imports to a displayed, named value and its completeness evidence. Choose fixtures that distinguish the intended answer from a plausible wrong one, then typecheck the displayed source, run it unchanged on a device, and compare its results with independent expectations.  
Publish five runnable entry modules under one example and display their actual source in the docs. Add `makeFacilitySearchTopologyOnCPU(outgoing: PackedRoadTopology, options: {travel: 'to-facility' | 'from-facility'}): PackedRoadTopology`, borrowing outgoing arrays for `'from-facility'` and returning a newly allocated, weight-preserving transpose for `'to-facility'`.  
Keep execution convenience in the shared example through `runFirstComputationOnDevice(device, id, construct)`, the standalone device-owning wrapper, and `graph.importGPUData(id, data)` for resident inputs.

## c14-docs-structure
Each task mapping should identify a qualified export, the props that select the advertised calculation, the result shape and a canonical section explaining how to read it. Verify the binding, mode, rendered destination and reachability from the family chooser.
Use one task record for navigation, starting with the 15 reviewed journeys, to generate the chooser and both cross-reference views. Use one operation-entry template covering the question, execution forms, definition, inputs, outputs and status, a checkable example, limits and alternatives.
Organize the sidebar by task within each data family and retain an alphabetical export index, existing URLs and fragment aliases.

## c15-lifecycle-reuse
Capture validated construction values and nested port maps while preserving GPU resource identities. Declare once per graph, retain nodes for inspection, replay the compiled graph and submit before overwriting inputs. Document capacity, physical stride, active membership, outputs for inactive rows, status and accumulation.
Bind retained results to their provenance and representation. Specify reset, invalidation and rebuild boundaries, iteration budgets, convergence signals, host readbacks, cleanup owners and teardown order. Check mutation, smaller second runs, discard, resize and allocation failure, including readiness after failure.
Prioritize three larger proposals: publish the cost table’s built cutoff and pin the four-node regression, capture raster stream settings at reset and pin the two-sample oracle, and remove operation-level `reusePreparation` before alpha while retaining it on the direct simulation producer, subject to LIF-09.

## c16-scale-variants
Use `execution` for equivalent routes, `method` or `accuracy` for changes to mathematical guarantees, and a separate preparation class for reusable state. Expose `Contributor.limits` and side-effect-free `getExecutionPlan(shape, deviceLimits)` with route, layouts, inclusive bounds, per-binding bytes, commands and valid alternatives.
Name hard limits `<OPERATION>_<ROUTE>_MAXIMUM_<QUANTITY>`, crossovers `<OPERATION>_AUTO_<ROUTE>_THRESHOLD_<QUANTITY>` and defaults `DEFAULT_<QUANTITY>`. Generate docs and catalog from the same descriptors, test immediately below, at and above each boundary, and name the failed constraint and any accuracy trade-off in routing errors.
Prioritize three changes: correct streaming moments on a real GPU before alpha and then pin the state encoding; introduce the common planner first for visibility, class breaks, terrain flow, raster region graph and prepared joins; deprecate `GPUChunkedSpatialWeightsValidation` in favor of `GPUSpatialWeightsValidation`, treating removal as a separate breaking change affecting five in-repository constructions or example occurrences.

## d01-float32-magnitude
Compute difference-based observables in a shared local frame, taking differences before narrowing wider inputs. Publish exact identities and cardinalities as integers.
Pair original and translated fixtures. Assert the selected row, cell, face, observation or support count exactly, and measure displacement, covariance, inertia or residuals in their own units.
The three priority proposals are to enable centered k-means by default before alpha subject to FLT-04, adopt one signed raster edge-ownership calculation across seeding, sampling and profiling, and retain local working state through integration and reduction until publication.

## d02-completeness-travel
A consumer of a bounded result must accept the producer’s evidence with its data and, on every execution, forward the evidence, suppress the dependent result, or refuse the composition. Keep input completeness, local convergence and final capacity distinct, including finite routing horizons and per-row support flags.  
Preserve evidence on reusable handles, starting with `GPUNetworkCostWeights.outputWeights` and extending to neighborhood CSR and solver-derived labels. Give pair-mode smoothing distinct forms for producer status and explicitly complete imported rows.  
Test contiguity → conditioning → likelihood, reachability → rational access, and watershed → region graph. Hold retained data fixed while toggling evidence across direct and GPU/CPU operation forms, with and without optional status destinations. Include single-chunk loss and a converged search whose horizon crosses the scoring threshold.

## d03-host-packers-borders
Validate each encoded record against its WGSL reader’s integer requirements, sentinels, endpoint comparisons and derived quantities. Publish exactly the declared words in one successful update and express each binding offset in units of its WGSL array element.
Make comparison policy explicit per field, starting with the five network-filter calls. Add exact uint32 active counts and local-origin grid placement alongside compact formats.
Test publication with short, exact and padded nonzero-offset targets, distinct sentinels and an invalid final field. Exercise adjacent floats, `2**24+1` and `2**32`, and record generated WGSL array types and binding offsets for prefixed views.

## d04-writers-reuse
An accepted destination receives its documented logical range on every submission, including empty and overflow routes. Reject unsupported destinations at construction with the producer ID and prop path. Compact outputs name their live count and tail policy, and published descriptors expose only supplied result columns.  
Exercise every legal singleton and branch-changing subset with distinct subviews. Seed storage, submit a nonempty or overflowing case followed by a smaller or empty one, and check values, status words and adjacent guards. A writable binding is only the first check.  
The three larger changes are to declare computed status words per spatial-weights producer while retaining the common CSR consumer descriptor, add numerical GPU tests for poisoned storage reused across submissions, and narrow the picking-only output before publication as proposed in WRT-13.

## d05-statistical-numerics
Each public statistical result should state its admissible sample, estimator, tail construction and undefined-result policy. Check it against an independent reference on an ordinary sample and a boundary that distinguishes plausible formulas.  
Three larger changes advance this convention: carry p-value construction and usable null counts through planners and summaries, specify the Mantel and Atkinson boundary policies in tests and formula tables, and qualify climate probabilities independently for monotonicity across the zero atom and accuracy near a large-shape gamma mean.

## d06-wgsl-hazards
Give every arithmetic boundary that selects a bin, label, match, winner, or dispatch count a tiny fixture with an independently known discrete answer. Read the actual result on the device. Establish a unique writer or retry weak exchange while ownership remains available, and audit generated names, reflected bindings, and unhandled weak exchanges.  
Scale before squaring in planar track similarity, then in z-score distances and variogram terms. Offer validated resident CSR for scatter ownership and capacity-bounded loops for read-only consumers.  
Document float32 margin sensitivity for embedding decisions first. Before promising reproducible near-tie labels, specify summation and tie rules, test across backends, and measure the cost.

## d07-degenerate-inputs
Distinguish supplied rows, usable observations and output capacity. Accept well-formed zero-row batches with terminal offsets `[0]`, publish zero counts and the documented empty or insufficient-support result, and require positive dimensions only where the operation needs them.  
Run the pure function, both named forms and the contributor on the same logical input, including a second submission over populated outputs.  
Prioritize three larger changes: accept logical empty batches, starting with zero queries and compact outputs before shared validators; return NaN for untested Pettitt cells; and use one geographic distinctness and accumulation rule for CPU and GPU movement centers.

## d08-twin-disagreement
For the same representable inputs and named method, CPU and GPU should share the input domain, decision precision, tie order and loss-status precedence. Check discrete outputs against independent exact expectations, apply tolerances separately to continuous outputs, and expose intentional estimator differences through a common public option.
Make discrete parity a pre-alpha contract, starting with quantile ranks, modal breaks, profile prominence, scalar medoids, singleton composites and radiation normalization. Preserve extra precision at envelope exceedance and ambiguous planar-radius comparisons.
Fix native integer raster calibration upstream, document integer recentering immediately, and verify calibrated samples and both parent IDs against the published fix.

## f01-statistics
Every PySAL cross-reference should state the feature order, weights transform, diagonal policy, tested population, tail, correction, and exact result field wherever they affect the number. Each field should retain its meaning across optional methods. Check the displayed value with an asymmetric masked fixture per linked operation and a tied permutation fixture.
1. Ship the aligned `w.id_order` import, result join, and transformed-handle example first. Reuse its fixtures for an optional CPU upload helper that reports the first duplicate, missing ID, or order mismatch.
2. Rename reciprocal averaging to `averageReciprocalWeights` before alpha and document structural symmetrization as a separate support-union operation.
3. Add host readers for `GPUGlobalSpatialStatistics` and `GPUGlobalPermutationTest` over the existing packed layouts. Expose fields such as `moran.statistic`, `moran.expected`, `moran.pRandomization`, and `permutation.zSimulated` with the alternative, correction, and counts. Verify that a reader can extract observed I and the intended p-value from the page alone.

## f02-geometry
Geometry consumers should accept the producer’s hierarchy, logical feature count, and completeness evidence directly. Named results should preserve their meaning under identity grouping, capacity padding, permitted ring closure, and reversal of an undirected segment.
Check those transformations on the same fixture against Shapely, comparing live rows, coordinates, and attributes.
The reviewer’s first three priorities are to separate polygon membership counts from assignment counts, carry overlay live feature counts into joins, and publish label coordinates and clearance that agree.

## f03-points
Give every public contributor an executable example that takes named settings through the actual parameter write to one decoded result and identifies its grid, neighborhood, label domain and completion condition.  
Check encoded values, layout bounds and contributor capacity on the host. Verify the statistic or estimator against a small independent numerical fixture.  
The three larger changes are a safeguarded finite-window Hawkes fit before alpha, reusable region-label compaction that preserves source-row identity, and explicit raster placement and sampled inputs through line-density raster metadata, KDE bandwidth population selection and mode-specific envelope props.

## f04-cells-trajectories
Carry the producer’s encoding, units, active-row count, and available loss evidence through every documented handoff. Declare each example on the null device, run the smallest distinguishing fixture on a CPU twin or GPU oracle, and assert that a supplied recipe destination is the exact view returned.  
Adopt native raster patch labels and propagate completeness into geometric outputs. Standardize recipe destinations on `output` with a compatibility alias and view-identity tests.  
Make movement policies explicit for geographic centers, missing fixes, unknown categories, and open final stops while preserving named reference-package conventions.

## f05-network
Give every public network workflow an executable example that starts with identified, directed edges, states the cost unit and travel direction, and decodes results back to input IDs. Run the CPU example, declare the GPU composition, and check explicit results for a one-way edge, parallel edges, an isolated vertex and an exhausted budget.  
Correct path-edge resolution and publish `predecessorEdges` wherever the producer can supply it. Standardize checked word and byte readback, including a flow summary exposing `cost`, `exactCost` as a `bigint`, `unserved` and `cutCertified`.  
Add one host road-network preparation function that preserves application IDs, parallel-edge identity and directed costs while building matched forward and transposed CSR layouts for OD, closest-facility and centrality workflows.

## f06-raster-terrain
At each raster handoff, identify the grid, band representation, physical units, validity rule and output layout. Keep the published `GPURaster` through spatial compatibility checks, name packed-cube layouts, and distinguish same-format ports by quantity and unit.
Check each documented cross-family handoff with two discriminating fixtures: one that composes and one that must be rejected or produce a distinct result.
The three larger changes are to unify visibility sampling on positive-weight support, provide an exact native-integer nearest-label warp, and rename hydrologic `slope` to `descentTangent`.

## f07-dataframe-pointcloud
At every public boundary, document and preserve what each row identifies, how missingness is represented, and which metadata is required to interpret the result later.
Test documented compositions with permuted rows, independently nullable measures, and serialized captures. Check decoded labels, source rows, and restored values.
The reviewer’s three priorities are a shared Arrow preparation path for keys and measures, a lossless capture codec promoted from lenses into analysis, and a direct pre-alpha rename of `GPUKeyJoin` to `GPUKeyLookup`, pending UDF-14.

## f08-consumers-view
Declare each mode once under `{package, exportName}`, including parameter constraints and resolved units, named port paths and formats, valid lengths, row identities, missingness and status meanings. Hosts supply data mapping, presentation and tighter resource budgets.  
Check each declaration against public constructor props using caller-owned views, then round-trip a valid row, a missing row and an empty result through the binding.  
Prioritize three changes from Larger proposals: declare parameters beside their implementations, publish output plans starting with OLS and line geometry, and publish named port roles, row spaces and missingness for lens capture mappings.
