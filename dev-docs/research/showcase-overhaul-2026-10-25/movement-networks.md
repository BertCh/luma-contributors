# Movement, flows, networks and time: research and delivery

Research date: 25 October 2026. This document distinguishes the initial inventory, implementation
plan, and verified delivery. A class mentioned in narrative or called inside another contributor
does not establish a direct showcase call site.

## Questions selected for implementation

1. **Dutch rail resilience:** what happens to access when service at a major interchange is
   removed or slowed? Compare a matched baseline with a counterfactual on the same directed
   station graph, then separate added travel time, lost destinations within a time budget, and
   destinations that become unreachable. Equal station opportunities are not passengers, jobs,
   or population. The planned scene is `rail-resilience`.
2. **Harbor observation quality:** how much of a vessel track is observation and how much is a
   motion-model assumption? Preserve original NOAA observations, flag speed/gap thresholds,
   compare filtering with retrospective smoothing, and show the consequence of refusing to
   interpolate across gaps. The planned scene is `harbor-track-quality`.

These complement existing transit frequency/reachability and harbor playback/encounter stories.
The rail story uses a quiet paper atlas with directional network evidence; the harbor story uses
an inspection view with observed fixes, estimated motion and uncertainty. Neither requires
inventing measurements or shipping duplicate data.

## Authoritative sources and what they support

- [GTFS Schedule Reference](https://gtfs.org/documentation/schedule/reference/), read 25 October
  2026: schedules have explicit trip stop sequences, arrival/departure times, service calendars
  and transfer semantics. Aggregating arrival(B) minus departure(A) to a median edge is a
  graph model, not a timetable journey planner; it omits waiting and connection feasibility.
- [OVapi national feed directory](https://gtfs.ovapi.nl/nl/), inspected 25 October 2026: upstream
  source location for the existing bundled extract. The live ZIP changes; the scene must retain
  the bundled 9 October 2026 service date and its feed provenance, not claim current operations.
- [NS rail works information](https://www.ns.nl/reisinformatie/werk-aan-het-spoor/), read
  25 October 2026: actual diversions motivate asking about network resilience. These notices
  are contextual evidence, not a source of the simulated closure masks. The simulation is not
  an operational prediction or reproduction of a published closure.
- [NOAA AIS project and license](https://github.com/ocm-marinecadastre/ais-vessel-traffic), read
  25 October 2026: NOAA distributes U.S. Coast Guard NAIS observations for planning, with CC0
  dedication. Its experimental GeoParquet product has additional cleaning; it must not be
  silently substituted for the CSV-derived bundle.
- [NOAA AIS FAQ, May 2026](https://coast.noaa.gov/data/marinecadastre/ais/faq.pdf), read
  25 October 2026: terrestrial receiver coverage and minute subsampling constrain interpretation;
  timestamps are UTC; gaps and GPS-related errors can occur. This supports an observation-quality
  story, without identifying threshold exceedances as proven sensor faults.
- [U.S. Coast Guard: How AIS works](https://www.navcen.uscg.gov/how-ais-works), read
  25 October 2026: reported navigation measurements originate in onboard systems. Derived
  displacement speed and reported speed over ground are distinct quantities.

The Kalman and Rauch–Tung–Striebel computation contract is established by the repository's
`GPUTrajectoryKalman` public API and implementation. Noise parameters are user-selected model
assumptions, not calibrated measurement error estimates. A smoother's use of later observations
must be visible in its label and explanation.

## Data audit

### Dutch rail

Existing `gtfs-nl-rail-graph`: 485 stations, 1,411 directed edge slots, 6,251 rail trips on
9 October 2026. Each edge is a consecutive stopping-station pair and train class; the time is
median scheduled in-vehicle seconds, at least 60 s. Parallel edges are real class distinctions.
Positions are longitude/latitude and analysis uses an azimuthal equidistant meter frame. No
passenger demand or capacity is provided.

Closing all edges incident to a station is a **graph station-removal scenario**. An express
edge whose endpoints bypass that station can remain: this is not closure of every physical
track passing through the station. A slowdown multiplier is likewise a hypothetical edge-cost
experiment. Search convergence is required before absence can be called unreachable.

Builder and provenance: `examples/deck/spatial-analysis-showcase/scripts/data/gtfs-nl-rail-graph/`.

### New York / New Jersey harbor AIS

Existing `ais-vessels`: 897 paths from 471 vessel identities, 124,780 fixes on 12 June 2024.
Coordinates are stored float32 geographic degrees; times are integer seconds since midnight UTC.
The manifest describes one-minute source sampling with additional stationary thinning to about
five minutes, and path splitting around long gaps. This preprocessing is part of the evidence.

A direct audit of bundled consecutive timestamps found a median interval of 71 s, 50,422
intervals above 120 s, 44,301 above 300 s and 952 above 600 s; the maximum was 1,289 s. A gap
threshold must therefore be presented as an analyst's interpolation policy, not a detector of
unexplained outages. Dataset membership on one date is not proof of complete 24-hour coverage.

Two useful real passenger paths contain both moving and near-stationary observations:
path 223 (MMSI 366952890, 265 fixes) spans the Upper Bay; path 640 (MMSI 368040960,
335 fixes) spans the Upper Bay and Hudson. Path identifiers are bundle-local. A selected-path
view should report its sample count and UTC support; full-bundle totals must not be displayed
as though they apply to the selected path.

No new source acquisition is planned: no new download can improve provenance merely by being
newer. The chosen bundles are reproducible local inputs with known sampling and units.

## Initial API coverage matrix

Inventory from `node scripts/report-showcase-coverage.mjs --format=json` before this track's edits.
Direct call coverage does not certify option coverage or numerical correctness.

| API family | Existing direct runtime evidence | Uncovered APIs / useful option gaps | This track's plan |
| --- | --- | --- | --- |
| Network searches and views | Reachability, cost matrix, accessibility, snapping, isochrones, line graph, neighborhood, path extraction | Point routing, time routing, alternative routes, service areas; no baseline-versus-disruption composition | `addNetworkDisruptionRecipe` with live edge mask/replacement costs, cumulative station reach, convergence and explicit unreachable states |
| Network structure | Analytics columns, statistics, subgraph filter, coarsening; graph-algorithm centrality/community implementations elsewhere | `GPUNetworkStructure`, direct `GPUNetworkCentrality`, bounded `GPUNetworkLouvain` | Consider SCC/service-area extension only if graph topology, output and story remain honest; not counted from weight masking alone |
| Network opportunities | Job accessibility, drive-time catchment recipe | `addFacilityScenarioRecipe`, `GPUNetworkCostWeights`, `GPUNetworkCostTable`, facility optimization, rational access | Future extension requires real demand/supply or explicit unit-demand planning assumptions; no invented riders |
| Network point processes | `GPUNetworkKFunction` retail story | Network KDE, inhomogeneous K | Separate event/intensity story needed; no superficial insertion into rail routing |
| Network flow | Flow aggregation, edge bundling, adjacency matrix/order | `GPUNetworkFlow` | Real capacity and demand are absent from the rail source; defer rather than equate timetable frequency with passenger capacity |
| Trajectory observations | Metrics, clock resampling, playhead, route similarity | `addTrackQualityRecipe`, Kalman isotropic/full covariance and smoother, restriction time/geometry/gap policies | Raw quality flags plus filter/smoother and explicit gap policy on observed AIS |
| Trajectory interactions | `addClockEncounters` composes encounters; route Hausdorff/Fréchet | Direct continuous closest approach, similarity search, restriction pairwise predicates; direct constructor absence for encounters is partly composition | Keep continuous versus sampled encounter semantics explicit; deferred pending coherent capacity-aware pair story |
| Temporal patterns | Calendar buckets, Knox/Mantel, spatial/LISA/rank Markov, hot-spot recipe | Space-time variogram; irregular-to-regular support issues | Existing earthquake/election stories retained; no duplicate superficial temporal scene |

## Planned validation

- Static scene/story checks, scoped TypeScript errors and WGSL identifier scan.
- Independent CPU references from the exact bundled rail graph for baseline/scenario station
  counts and cost deltas; these are verification evidence, not runtime substitutes for GPU work.
- Browser/GPU checks for each story transition, parameter updates and compilation variants;
  inspect actual nonuniform map output, matching legend units, charts and pending-state behavior.
- Check faster successive option changes cannot publish older readback over newer parameters.
- Parent task owns root install/build/lint and full verification. Scoped checks are not a substitute.

## Continuing implementation waves

The user requested broad functionality coverage, so the first pair of stories is not a stopping
point. The following groupings keep additional methods connected to a real analytical question.
They are plans until the delivery ledger says otherwise.

### Regional aviation networks

Use the existing BTS `us-airline-flows` bundle, preserving directed arcs and July 2023 reporting
carrier coverage. The [BTS source table description](https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ)
describes reported nonstop domestic flight records, including cancellations. Counts are scheduled
flights, not passengers or available seats. The marketing-network totals in the
[BTS July 2023 release](https://www.bts.gov/newsroom/air-travel-consumer-report-july-2023-numbers)
have different carrier coverage and must not be presented as the bundle's totals.

An induced western subgraph selecting the 24 busiest airports within longitude [-125,-105],
latitude [25,50] has 338 directed arcs and 66,741 scheduled flights. The analogous northeastern
box [-90,35,-65,50] has 398 arcs and 69,735 flights; the southern box [-105,24,-75,38] has
350 arcs and 58,627 flights. Selection is deterministic by the bundle's `airportFlights` field.
These are explicitly regional induced subgraphs; a missing arc outside the box is not missing
national service.

The inquiry compares activity, shortest-path brokerage and communities using direct
`GPUNetworkCentrality` and bounded `GPUNetworkLouvain` calls. Directed versus undirected weights,
resolution, seed, iteration/level caps, metric choice and exact/sample pivot coverage belong in
the controls. `GPUAlternativeRoutes` can reveal route redundancy, with frontier and expansion
caps visible. A flow/min-cut demonstration can use observed counts as abstract link budgets,
but must call them flight-leg tokens and avoid implying feasible connecting itineraries,
aircraft capacity or passenger redistribution.

### Service catchments and bounded planning

The rail graph supports nearest-hub service areas, decay-weighted reach, hypothetical facility
availability and small exact facility selection. Candidate stations are real; staffing and
capacities, if introduced, are scenario parameters. Unit station demand can answer how many
stations receive support without inventing passenger counts. Matrix orientation, directed
travel direction and the population of retained demand points must remain explicit.

Target methods: `GPUNetworkServiceAreas`, `GPUNetworkCostWeights` (all four decay kernels),
`addFacilityScenarioRecipe`, bounded facility optimization and rational access. Different
optimization objectives need correctly shaped input matrices; an incomplete search is an
incumbent, not an optimality certificate. Rational-access travel/supply scales and fractional
versus integral demand materially change interpretation and must be exposed.

### Street route assumptions and network point processes

A bounded connected extract of the real Chicago street graph can demonstrate point routing,
alternate routes, reusable bounded cost tables and restriction/time-routing state models.
Modeled congestion must be labelled as a scenario; median rail edge times cannot substitute
for a departure timetable. The time-routing API's interpolated arrival envelope is not itself
a discrete train departure schedule.

The existing retail pattern story can be extended with branch-normalized network KDE and
inhomogeneous K on a manageable street extract. Quadrature arc lengths, edge splitting for
interior events, positive intensity, unique undirected edges and complete network distances
are prerequisites. These are substantive method requirements; using planar distances or
radius-truncated tables would produce an attractive but incorrect demonstration.

## Delivery ledger

| Artifact | Status | Evidence / remaining check |
| --- | --- | --- |
| `networks/rail-resilience.scene.ts` and `.compute.ts` | Implemented, source reviewed | GPU disruption recipe; filtered directed topology and exact SCC; direct GPU station cost/delta/flag layers. Story check passed; parent actual-GPU browser run passed all five story steps without browser errors. Independent Playwright assertions passed three closure scenarios against CPU references (including 64 newly unreachable at Zwolle). |
| `movement/harbor-track-quality.*` | Three independent GPU scenarios passed, zero browser errors | Raw quality flags, isotropic filter/RTS, closed time restriction and gap breaks. Full covariance is the next wave. |
| `flows/aviation-network-lab.*` | Five-step GPU sweep and six independent regional rankings passed | Directed distance/hop centrality, Louvain resolution, SCC; all region GraphSwap transitions verified. Flow remains planned. |
| `networks/rail-support-planning.*` | GPU assignments matched; boundary-conversion fix awaiting rerun | Inbound/outbound service areas and all four cost-weight decays; exact-minute inclusive boundary fixed after independent discrepancy. |
| `networks/rail-support-allocation.*` | Source reviewed; independent reference and GPU harness ready | Eight named demand stations; all three facility selection objectives; per-objective facility scenarios; all four integral/fractional and fractional/absolute RAAM variants. |
| `movement-network-reference.py` / `.json` | Ran successfully | Exact bundled input checksums, independent CPU Dijkstra scenarios and observed AIS timestamp audit. |

The independent rail reference (all classes, no dwell, unbounded search, counts including origin)
gives Amsterdam → Zwolle interchange closure: 283 stations inside 90 minutes before, 265 after,
64 newly unreachable and 49 with positive additional time. Median positive added time is one
minute; maximum is 53 minutes. The disrupted graph has four directed strongly connected
components with sizes 421, 61, 2 and 1. Amsterdam → Utrecht interchange closure changes access
from 283 to 235, with one newly unreachable station and 172 delayed stations; median positive
delay is 22 minutes. The baseline/after-budget counts and newly unreachable counts now match actual GPU execution in the three-scenario Playwright check. The additional delay distribution remains a CPU reference and visual review target.

Review changed the initial direct-link defaults because three chosen links had no or very small
effect from Utrecht. The final interchange-removal scenarios make the distinction between
detours and disconnection observable. Source claims now distinguish station-incident scheduled
edges from physical rail tracks, and map/tooltip comparisons require convergence.

## Planning-method interpretation sources

- [PySAL Access RAAM API](https://pysal.org/access/generated/access.Access.raam.html), read 25
  October 2026, documents separate travel-time and congestion scales and decaying allocation
  steps. This supports exposing both scales and distinguishing fractional versus absolute shifts.
  The showcase uses hypothetical station-support demand, not estimated travelers or patients.
- [PySAL spopt capacitated p-median comparison](https://pysal.org/spopt/notebooks/p-median_variations.html),
  read 25 October 2026, states the assignment, facility count and hard capacity constraints.
  Unlike rational-access congestion, a capacitated assignment cannot exceed supply. These must
  be separate story modes with distinct readouts, not a single overloaded accessibility score.
- [Luo and Qi (2009), enhanced two-step floating catchment area](https://www.sciencedirect.com/science/article/pii/S1353829209000574),
  read 25 October 2026, motivates travel impedance within a catchment. The planned rail scene
  exposes binary versus decaying weights as assumptions; it does not claim calibrated traveler
  choice or implement a health-access model using station counts.

## Independent support-planning references

`rail-support-reference.py` and `.json` compute directed Dijkstra and exhaustive facility
solutions independently of the GPU scene. Four candidates are Amsterdam Centraal, Utrecht
Centraal, Rotterdam Centraal and Zwolle. Eight named demand stations each carry one scenario
unit for placement; rational allocation has its separately labelled demand multiplier.

At an inbound 60-minute limit, nearest-desk counts are 64, 112, 66 and 77, leaving 166 stations
unassigned. There are 718 eligible station-desk pairs. Outbound routing produces 62, 115, 66
and 78, with 164 unassigned and 722 pairs. The script also calculates 30- and 90-minute cases
and independent sums for binary, exponential, power and Gaussian decay.

With two desks and capacity four each, the eight-demand capacitated median optimum is Utrecht
plus Rotterdam, with 231 station-minutes. Two-desk backup coverage at 60 minutes chooses
Amsterdam plus Utrecht and double-covers seven of eight demand stations; at 30 minutes no pair
covers all eight. Dispersion on the symmetric maximum of two directed travel costs chooses
Rotterdam plus Zwolle, separated by 84 minutes. These remain reference targets until browser
assertions pass.

Reusable browser checks are `check-rail-resilience-reference.mjs`,
`check-aviation-network-reference.mjs`, `check-harbor-quality-reference.mjs` and
`check-rail-support-reference.mjs` in the showcase scripts directory. Checks use the exact local
snapshot and preserve diagnostics on failure. A rendered story without console errors is
necessary but does not establish numerical correctness; option transitions are tested separately.

## Continuous encounters: next movement wave

The [MobilityDB temporal distance reference](https://mobilitydb.github.io/MobilityDB/master/ch08s04.html)
and [official BerlinMOD query examples](https://mobilitydb.github.io/MobilityDB-BerlinMOD/html/ch01s05.html),
read 25 October 2026, distinguish nearest approach over common temporal support from distance
between paths that ignores time. Synchronizing linear segments and testing their closest point
can reveal approaches missed at sampled clock instants. The showcase will make this model
assumption explicit; it is neither an observed minimum separation nor a navigation safety finding.

The documented selected hour is 14:00–15:00 UTC on 12 June 2024. Bundle paths 221, 370, 509,
562, 595, 712, 726 and 735 are eight distinct passenger-vessel identities with 53, 45, 46, 47,
40, 48, 44 and 55 retained observations respectively. Their largest within-window gaps are
128, 200, 80, 252, 147, 247, 267 and 127 seconds. Observed support starts and ends at each
track's actual first and last retained sample; no position is extrapolated to the hour boundary.

`ais-encounter-reference.py` independently projects the same stored float32 coordinates to the
bundle's AEQD meter frame, solves each overlapping segment-pair quadratic, unions encounter
intervals and compares shared clocks. At 100 m, this reference finds two continuous pairs,
one pair on a 60-second clock and none on a 300-second clock. At 250 m, the corresponding
counts are five, three and one. Paths 712 and 735 approach to about 89.55 m at 14:57:33 UTC
within this model, whereas the aligned 60-second clock reaches only about 117.65 m. That is a
concrete sampling example; it does not certify the ships' true closest distance between reports.

## Similarity research and review evidence

The same independent AIS reference now computes discrete Fréchet, accumulated-distance DTW
and normalized LCSS with a 150 m match tolerance by direct dynamic programming, without
calling the production contributor. These methods compare ordered paths without synchronizing
timestamps. They answer a different question from continuous closest approach.

For source path 370, discrete Fréchet ranks path 595 first at about 4,062 m, whereas DTW ranks
path 735 first at about 110,145 accumulated metres. LCSS also favors path 735, with a
normalized score about 0.156. Across all 28 unordered pairs, the smallest Fréchet distance
is already 4 km; a 500 m default match threshold would produce an empty example. The review
therefore requires separate Fréchet and DTW scales and a lower LCSS threshold, with honest
kilometre versus accumulated-kilometre labels. The query and best candidate must use different
colors. Production WebGPU comparison now passes all 24 query/metric cases and a strict-threshold pruning case, with no browser errors.

Source review also found two presentation hazards that must be corrected before verification: the audit
subtracted a relative timestamp from an absolute timestamp, and packed path segments were
indexed using vertex offsets. The interval audit must report the independently verified 267 s
maximum. The reviewed continuous scene must expose clock resolution separately from the restriction gap
threshold, report capacity/invalid flags, and treat gap exclusions as unknown support.

## Numerical verification update

Against the production WebGPU preview after the sixth build,
`check-aviation-network-reference.mjs` passed all six region/metric comparisons with no browser
errors. It verified all three region transitions, graph and flight-leg counts, and independent
Brandes top-five rankings for distance and hop costs. `check-harbor-quality-reference.mjs`
passed all three threshold/model-gap scenarios with no browser errors. Reports are saved in
`/tmp/luma-showcase-overhaul-20261025/aviation-reference-latest.json` and
`harbor-reference-latest.json`.

Rail support assignments matched the reference at the 30-minute inbound cutoff: 175 served,
310 unassigned, with candidate counts 52/55/36/32. Weighted pairs were 204 rather than 217.
The independently counted 13 exact-boundary pairs explain the discrepancy: WebGPU division
by 60 lowered to reciprocal multiplication and produced minutes just above the inclusive limit.
The corrected conversion separates an integer whole-minute quotient from the remaining seconds.
The ninth production build passed the numerical rerun: six inbound/outbound service cases at 30/60/90 minutes, all four decay totals, three exhaustive placement objectives, an infeasible backup threshold, and all four integral/fractional × fraction/absolute rational-allocation modes (18 cases).

The Chicago CLI implementation completed and was reviewed. Review fixed missing output-view
imports, separated static point costs from the hypothetical time-dependent costs, corrected shared
route semantics, and replaced landmarks outside the capped graph with explicit east/north/west
extract-boundary queries. Independent references in `chicago-route-reference.mjs/.json` reproduce
61 nodes, 128 directed links, three certified alternative paths for each query, fractional-point
costs, both time-dependent models, and all-pairs table counts. The first runtime comparison caught the reserved WGSL binding name `shared`; the source now uses `sharedMask`. Runtime numerical acceptance awaits the next build.

### Additional delivered source waves

| Scene | Delivered API behavior | Verification state |
| --- | --- | --- |
| `harbor-continuous-encounters` | Continuous closest approach; sampled clock encounters at 60/300/600 s; pairwise restriction with all four time and both geometry endpoint policies; maximum-gap exclusions | Independent segment-quadratic reference ready; first GPU initialization exposed an oversized pairwise parameter vector. Corrected to exactly seven scalars; retry pending |
| `harbor-track-similarity` | Discrete Fréchet, accumulated DTW, normalized LCSS; threshold bounds, top-three exact matches, all eight query choices; separate query/match overlays | Production WebGPU: all 25 independent dynamic-programming/pruning cases pass |
| `chicago-route-workbench` | Fractional point routing; affine and piecewise FIFO time routing; top-three alternatives and shared links; bounded cost-table reuse | Independent path references ready; reserved shader binding corrected after runtime failure, retry pending |
| `retail-network-intensity` | Network KDE and inhomogeneous K, leave-one-out intensity, edge quadrature, correction/normalization options | Planner review fixes complete; independent Dijkstra/KDE mass/support audit complete; production runtime pending |

### Durable verification reports

The ninth-build reports are `rail-support-reference-wave9.json` (18 passing cases),
`harbor-similarity-reference-wave9.json` (25 passing cases), and the initial failed
`chicago-route-reference-wave9.json` / `harbor-analysis-reference-wave9.json`, all under
`/tmp/luma-showcase-overhaul-20261025/`. Browser console errors alone do not establish acceptance:
all six domain harnesses now record an explicit `passed`, `status`, and `failureReason`, separately
from `consoleErrors`. The original shell exit status and failure entry remain the evidence for
reports created before this schema improvement.

Full-covariance Kalman model comparison is a further source wave under review, with the isotropic
default retained and explicit model-implied uncertainty rather than accuracy claims. See
`harbor-full-covariance-kalman.md`. The firm next implementation wave is capacitated network flow
on the BTS regional graph using explicitly hypothetical link budgets, including min-cost flow and
minimum-cut evidence; flight counts are not interpreted as passenger or seat capacities.

### Fleet-view consolidation decision

`vessel-encounters` remains as a narrowly framed fleet route-screening view, titled “Which routes resemble a selected vessel across the fleet?” It preserves interactive selection among 897 whole-day tracks and direct full-pairwise `GPUTrackSimilarity` Hausdorff/discrete Fréchet evaluation on 64 arc-length samples. Those behaviors are not replaced by threshold-pruned Search over eight audited-hour tracks. Migrating the whole fleet into that smaller story would combine different data support and analytical purposes.

The fleet story now distinguishes geometric resemblance from shared travel or coordination, mean-speed selection from a moored-state classifier, and sampled modeled proximity from observed encounter evidence. The unsupported fixed 10 m AIS accuracy claim is removed. Its limitations direct readers to the audited quality → continuous proximity → shape-search sequence for explicit reporting-gap and clock-sampling questions. No legacy route alias is needed because the existing scene ID is retained.

### Remaining behavior waves assigned after coverage review

- The Chicago time-routing workbench will gain a hypothetical forbidden edge-sequence scenario through `prepareGPUNetworkRestrictionStates`. This is state-expanded route topology, distinct from simply changing link costs; restrictions will not be presented as observed OSM turn rules.
- Existing harbor zone-support work will gain polygon rings/holes and compact interval union through `GPUTrajectoryRestriction`, preserving endpoint inclusion and unknown-gap support. This is separate from the already implemented pairwise predicate.
- The BIXI OD interaction and hypothetical service-location wave is assigned to the reused Luna CLI lane. `GPUSpatialInteraction` will compare four constraints and grid/bounded calibration on explicitly included observed OD support. Source omission below three monthly rides remains missing, not zero; same-station trips are excluded by the analytical definition. `GPULocationAllocation` and `GPUBoundedFacilityAllocation` will distinguish weighted-distance, coverage, center and capacity/order objectives with explicit certificates. Sources: [BIXI official open data](https://bixi.com/en/open-data), [PySAL SpInt](https://github.com/pysal/spint), and [PySAL spopt location model documentation](https://pysal.org/spopt/api.html), read 25 October 2026. These are planned behavior, not delivered or runtime accepted yet.
- `addCellFlowTopologyRecipe` is owned by the atlas team as a separate BIXI cell-flow extension. Existing station flows do not count as execution of this H3 OD topology recipe. It must conserve all included OD totals while showing only immediate-neighbor flow on cell boundaries; nonneighbor movements do not imply routes through intervening cells.

### Restriction semantics research

The [OpenStreetMap restriction relation specification](https://wiki.openstreetmap.org/wiki/Restriction_type), read 25 October 2026, models prohibited travel through a from/via/to sequence. That motivates state expansion for a forbidden edge sequence rather than deleting an entire street. The showcase extract does not supply those relations, so its chosen restriction must remain a hypothetical scenario on real geometry; it is not evidence of a current legal turn rule.

[MobilityDB’s temporal restriction documentation](https://mobilitydb.github.io/MobilityDB/stable-1.2/ch06s02.html), read 25 October 2026, distinguishes restriction to geometry from restriction to time and their complements. The zone-support extension will expose the included geometry and temporal support independently, including holes, endpoint inclusion and gaps, rather than treating every straight connection between reports as observed time in a zone.

### Wave 11: production numerical acceptance (25 October 2026)

Production preview at port 5313 was frozen during these sequential WebGPU checks. Every successful report below records `passed: true`, explicit status, and zero console/page errors; browsers closed after each suite.

| Story | Accepted cases | Evidence and limits |
| --- | ---: | --- |
| Chicago route workbench | 14 | Three point/alternative queries, nine departure/model comparisons, bounded table miss, plus hypothetical sequence 66→38: independent CPU baseline 246.8983 versus restricted 247.7714 minutes after 08:00. Rounded GPU readouts agree to 0.051 minute. |
| Harbor continuous encounters | 20 | Twelve radius/clock cases and eight time/geometry endpoint-policy combinations; continuous minima agree with the independent piecewise-linear oracle. |
| Harbor track similarity | 25 | Repeated all eight query tracks across three metrics and strict threshold pruning; independent dynamic-programming reference. |
| Retail network intensity | 14 | Eight category/bandwidth LOO, mass and withholding cases; three independent uncorrected-K normalizations; three Ang normalization status/identity checks. Ang curve values have not received an independent numeric oracle. |
| Aviation capacity flow | 5 | Maximum flow/cut, fixed-budget delay score, reduced-capacity maximum, budget shortfall, and balanced virtual-terminal demands. Saturated balanced supply gates correctly certify the augmented cut at 1,000; this does not certify a cut using only physical airport links. |

Durable reports are `/tmp/luma-showcase-overhaul-20261025/chicago-route-reference-wave11-turn.json`, `harbor-analysis-reference-wave11.json`, `retail-network-reference-wave11-retry.json`, and `aviation-capacity-flow-wave11-retry.json`. Two test corrections did not change source results: Chicago's arrival model is a radio group rather than a select, and the balanced-flow harness must expect the augmented gate cut. The density oracle was corrected to exclude self-loop edges exactly as production does: 2,509 vertices, 4,066 undirected links, 483,642.2909 m. The numerical tolerance was not relaxed.

The extended quality story failed before computation because one new compute stage declared 14 storage buffers against the device limit of 10. `harbor-quality-reference-wave11.json` explicitly records failure. Full covariance, epoch preparation and polygon/compact union remain pending production acceptance; historical isotropic acceptance is not evidence for this modified bundle.

Fine-option audit: spatial snapping sort is an execution strategy already exercised by existing scenes; accessibility orientation is matrix layout, and its catchment composition uses both row and column cost weights. Directed coarsening, undirected network Louvain, and adjacency count versus weight-sum semantics remain distinct behaviors requiring explicit evidence. The workbench explicitly selects waiting until the first sample, but with first sample zero it does not yet demonstrate a pre-first-sample wait. These distinctions are not inferred from static class-name coverage.

### Finite graph-assumption and movement variants queued for closure

The parent requested that the remaining meaningful variants stay inside existing studies. The aviation lab compares its directed scheduled-count graph with the mean-symmetrized `(W + Wᵀ) / 2` graph for undirected Louvain, preserving the total matrix weight. Undirected coarsening instead supplies summed reciprocal weights on each once-counted nonself edge, preserving the original input total under that operator's edge convention. This changes the graph assumption and modularity null model; it need not change every resulting partition, and neither output is a certified global optimum. Primary research consulted 25 October 2026: [Blondel et al., Fast unfolding of communities in large networks](https://arxiv.org/abs/0803.0476) and [Leicht and Newman, Community structure in directed networks](https://arxiv.org/abs/0709.4500).

The existing flight matrix will compare distinct airport-pair counts with summed OpenFlights airline-route records. Its source is already an undirected aggregate, so no orientation will be invented. Its weighted buffer output must drive the rendered cells, legend and conservation readouts. A texture serialization option is separate from this statistical-measure behavior and will not be claimed unless executed.

The existing taxi-trail rose will use `GPURoseStatistic` directly on active modeled path directions, including directional/axial interpretation and descriptive weighting/grouping. These are routed interpolations between real TLC endpoints, not recorded GPS bearings. Weighted bins do not turn Rayleigh z into a weighted significance test; no p-value will be claimed.

The Chicago workbench's pre-first-sample waiting branch is now source-ready: a 0/15/60-minute shift of both departure/arrival samples preserves FIFO durations and makes initial waiting observable. The default remains zero. Three added oracle cases are pending production; see the workbench research note for the time-translation argument.

### Source review after wave 11: finite closure queue

The table below supersedes earlier prospective assignments; it distinguishes current source from GPU acceptance.

| Extension | Current source status | Numerical gate still required |
| --- | --- | --- |
| Chicago first-sample waiting | Implemented with explicit hypothetical availability times; previous 14 cases remain historically accepted | 17-case suite including 15-minute, 30-minute and zero initial wait |
| Harbor quality / covariance / epoch / polygon support | The over-limit ellipse stage is split into four six-binding model stages; polygon/hole compact union oracle is independent | 10-case suite on a rebuilt production bundle |
| BIXI demand model | Four constraint models × grid/bounded calibration; conditional observed support, not zero-filled missing pairs; full station names and conditional residual tooltips | Eight fit cases |
| BIXI service locations | Dense/sparse weighted-distance, coverage/set-covering, exact center and order-sensitive capacity experiments; actual selected-mask and assignment readback, candidate letters A–H and real station names | Seven siting cases, including selected counts and certificates |
| Flight matrix route-record measure | GPU `weightSums` and `maximumWeightSum`, exact scale 1, selected raster/chart/legend/tooltip/readouts; independent reference reconstructs 133,866 mirrored record units | Five configurations × two measures = 10 cases |
| Taxi route-heading rose | Direct GPU directional/axial × equal/current-segment-length × aggregate/fare-class groups; statistical bin-center correction and precision-aware verifier reviewed | Eight variants, all rendered sectors and resultant lengths |
| Aviation graph assumptions | Directed versus mean-symmetrized Louvain, separate once-counted coarsening weight convention; independent source reconstruction passed | Eight membership-conditioned coarsening cases |

The BIXI monthly ≥3 pair threshold is a **showcase aggregation choice**, not an official source privacy or suppression rule. Same-station exclusions are also analytical choices. The activity-based station tooltips report full cross-station departures plus arrivals; these are deliberately distinguished from the 16-station interaction model's conditional induced margins.

Latest scoped story and strict cartography checks passed for networks, flows and movement. The latest showcase TypeScript check reports only concurrent grid-alignment errors; all frozen movement/network source candidates have no diagnostics. These source checks do not substitute for the pending production numerical gates above. The flight matrix browser harness now waits for the requested configuration's expected totals/maxima/occupancy, avoiding acceptance of a previous option's readback. BIXI invalidated radius results clear their old named-site annotations until the new GPU summary arrives.

The finite queue is now fully source-ready: quality 10 + Chicago 17 + BIXI 15 + flight matrix 10 + taxi rose 8 + aviation community aggregation 8 = **68 pending production numerical cases**. The app Luna and CLI implementation lanes have completed; no additional domain feature wave is planned. Method notes for the last two variants are `nyc-taxi-rose.md` and `aviation-community-model.md`.

### Production sweep review and source repairs

The wave 12b atlas/phone sweep exposed three existing graph-import identity failures. Migration stopovers and gull migration imported the same `lng-lat` buffer repeatedly; flight corridors imported a density output again under its balance-stage name. Their source now uses the shared typed buffer importer with a separate cache for each command graph. These repairs preserve buffer formats and graph lifetimes; runtime acceptance remains pending the next build.

The transit playback speed select expanded beyond the phone viewport. Its mobile width is now bounded while the timeline remains flexible. `scripts/check-time-bar-layout.mjs` instantiates the actual `TimeBar` and option store with production styles in a GPU-disabled browser. At viewport widths 320 and 390 pixels, including the longest speed label, the fixture passes document/bar overflow checks and 44-pixel button/select height checks (`/tmp/time-bar-layout.json`). The production phone retry remains separate.

Screenshot review also corrected source framing for the small Chicago street graph, the BIXI 16-station support, the full harbor quality track and its terminal close-up, and the western-airport capacity network. Bounds are derived from the bundled geometry. BIXI assignment spokes now use categorical assignment colors and transparent unassigned values; previously their highlight mask was combined with an ungrouped uniform color mode and hid every spoke. Candidate names remain in readouts/tooltips while compact letters label the map. These visual repairs require the next production screenshot review; they do not alter the independent numerical references.

The next numerical batch must save the served `build-evidence.json` before testing, verify the same `bundleSha256` afterward, and retain that evidence with all 68 case reports. Historical acceptance above is explicitly tied to earlier snapshots and does not certify these pending changes.

### Wave 13 numerical batch: 37 accepted, 31 pending

All eight captured build-evidence snapshots identify `658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d`. The batch is explicitly **partial**, not passed. Its durable ledger is `/tmp/luma-showcase-overhaul-20261025/movement-wave13/acceptance-summary.json`; all browsers closed before release of the GPU slot.

- Harbor quality: all 10 cases passed, including epoch precision, isotropic/full covariance parity, correlated ellipses and polygon/compact interval counts.
- Chicago: all 17 cases passed, including the three pre-first-sample waiting cases.
- BIXI: all eight interaction cases and both dense/sparse p-median cases passed. The first allocation comparison exposed a CPU input-contract error: its equirectangular shortcut differed from the declared local projection and float32 inputs. Corrected independent exhaustive enumeration gives 274,927.694860816, matching displayed GPU 274,927.69. The next case exposed a second oracle error: maximum coverage selected the minimum rather than maximum enumerated objective. The corrected exact upper bound is 273,329; the actual selector helper now passes a known-result fixture selecting 2 for minimization and 7 for maximization. Both failed reports are retained and tolerances are unchanged. Five allocation cases still require rerun.
- Flight matrix and taxi rose: no complete numerical cases accepted. Their harnesses initially targeted segmented controls as selects; later attempts exposed a hidden chart tab and a decorated switch intercepting pointer clicks. The harnesses now use actual accessible radio controls, the visible Story/All readouts chart, and keyboard switch activation. Ten matrix and eight rose cases remain pending.
- Aviation community aggregation: initialization failed because distinct input and output weight buffers had the same resource ID. The input is now named `symmetric-coarsening-input-edge-weights`; this is not an aliasing workaround. All eight cases await the next build.

The BIXI selected-radio refresh assertion also passed the eleven attempted cases on the second run, resolving a screenshot that briefly showed the previous activity selection alongside newly rendered assignment spokes. The rail resilience chip now says “Hypothetical disruption,” covering both closure and slowdown steps. These bounded source fixes are queued for the next integrated build.

### Wave 15: 58 of 68 finite cases accepted across snapshots

The wave 15 batch held bundle `5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65` unchanged. BIXI passed all 15 cases, including exact displayed p-center distance 2.01 km against independently enumerated 2.0132007599 km; comparing the rounded display with a tighter raw-value tolerance had caused the preserved first attempt to fail. Taxi rose passed all eight variants. Flight matrix passed eight full-window cases with exact integer totals, global maxima and occupancy; its six chart-bin labels are checked at their displayed precision because counts of at least 1,000 are intentionally rounded to whole thousands.

Two flight zoom cases remain pending. Their reference initially used the requested 271-airport extent as the entire input support. The scene's documented whole-cell layout instead has three airports per cell, 91 visible cells spanning 273 positions, and 128 compiled cells spanning 384 positions. Visible totals/occupancy use the first 91 cells; the GPU maximum explicitly includes hidden compiled cells. The independent scalar reference now honors these separate supports and has a known-result 271/128 fixture. Corrected expected values are 60 symmetric pair units, 112 symmetric route-record units, 59 occupied visible cells, and global maxima of 2 pairs and 5 route records. No runtime algorithm or tolerance changed for this correction.

All eight aviation aggregation cases remain pending: the repaired distinct resource IDs exposed a genuine contributor-order error. Directed coarsening was added before directed Louvain, so it read initial zero labels and counted one group while the later membership output contained three. Directed Louvain now precedes its consumer, matching the already correct symmetric ordering. This requires a rebuilt bundle.

Visual review found that BIXI's six-label desktop budget hid later candidate H even when selected. Selected candidates now have explicit higher annotation priority; the next browser verifier asserts that every selected short identifier is visible. The ground remains `paperCity`, with subdued roads and orientation labels, rather than an intentionally blank analytical sheet. Numerical readiness can precede external tiles; the harness now gives visual receipts a bounded network-idle opportunity. This does not make tile availability an algorithm acceptance gate.

`movement-acceptance-wave15.json` is the durable compact acceptance packet: exact report hashes, commands, methods, tolerances and limitations for the 58 accepted unique cases, preserving the earlier wave 13 provenance of quality and Chicago. Ten cases remain pending (flight zoom two, aviation eight); selected-site label visibility is an additional visual gate. Runtime source is frozen after scoped formatting and a successful showcase TypeScript check.

### Wave 16: finite numerical queue complete

Bundle `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf` remained unchanged throughout the final batch. Aviation passed all eight membership-conditioned coarsening checks after the contributor-order fix. Flight matrix passed all ten cases, including the two padded zoom cases with exact visible totals 60/112, occupancy 59 and compiled-buffer maxima 2/5. These comparisons retain exact integer assertions; individual chart labels are checked at displayed precision.

The durable focused verifier `scripts/check-bixi-selected-labels.mjs` passed: selected sites A, B and H were all visible, with no console/page or tile request errors. Manual screenshot review confirms H at the central selected site, matching assignment spokes and legend, plus the intended subdued Montréal roads, river and orientation labels. The earlier blank ground was an early visual receipt before external tile context arrived.

All **68 unique numerical cases** in the finite closure queue are accepted across their recorded wave 13/15/16 snapshots; this does not claim that all suites were rerun on one final bundle. `movement-acceptance-final.json` records exact report hashes, methods, tolerances, reproducible commands and limitations, including the focused desktop visual evidence. Historical failed reports and partial packets remain preserved. All browsers closed and the GPU slot was released; no additional movement/network implementation wave remains.

### Final visual sweep repair: street-density polygon publication

The final wave19 screenshot review found missing polygon fills on first entry to community areas and census tracts despite valid GPU statistics. Source tracing established that `handlePolygonSummary` set `ready` and published readouts, legend and annotations, but never refreshed Deck layers. The pre-readback layer list deliberately omitted the fill. The density callback already performed the required refresh.

The bounded repair requests layers after publishing the active polygon set's summary, guarded by matching set and polygon view. Input geometry, values, classification, numerical computation and styles are unchanged. A CPU check of the actual mesh builder found 36,132 finite triangle vertices for 77 community areas and 24,621 for 791 tracts, with correct inverse projected coordinates. The production callback extraction fixture passes four lifecycle cases: active first summary, inactive set, inactive view and destroyed scene.

Reproduce the CPU fixture with `node examples/deck/spatial-analysis-showcase/scripts/check-street-density-publication.mjs`. The focused visual verifier is `node examples/deck/spatial-analysis-showcase/scripts/check-street-density-polygons-browser.mjs http://127.0.0.1:5313/ OUTPUT_DIRECTORY`. It checks fresh first entry to each polygon set before any option interaction, isolates the actual Deck canvas from basemap and HTML overlays, verifies warm class fill pixels, and uses the public outline toggle as a positive control and explicit redraw comparison. The focused browser verifier subsequently passed both cases on frozen wave20 (`834b1464827bcbbd428b32cfd7950cdb383415c56730decfcdc106c98770fcb9`), verified unchanged before/after. First-entry warm fill counts were 258,126 community-area pixels and 244,481 tract pixels; initial and restored outline-on images were pixel-identical, and outline-off images differed. Both full screenshots were visually reviewed, including visible tract suppression hatching. No console errors. Durable evidence is in `street-density-publication-acceptance.json`. The initial attempt stopped on a duplicated readout locator; its failed report is preserved and the harness-only `.first()` correction is documented. No GPU run was performed during the frozen wave19 sweep.

### Wave21 flight matrix settling follow-up

The full-sweep Europe matrix screenshot showed an em dash because its 1,000 ms delay ended before the 900 ms reorder plus 300 ms settling period and asynchronous GPU readback. No runtime fix was needed. The smoke harness now waits for a numeric occupied count when the active step requests it and resolution is at most 1,024 (higher resolutions intentionally withhold the count). On unchanged wave21 bundle `9a5f744c3e1344cc04be45c9626e293af98d36d77f2e058b27acbfe26c476f82`, all 12 independent-reference cases passed: the previous 10 repeated plus Europe/country / 512 in pair and route-record modes. Europe has 660 airports in 330 × 330 visible cells, 7,169 occupied, 11,778 pair total or 36,258 route-record total, and compiled-grid maxima 4 or 64. The settled seven-step desktop tour and one 390 px mobile layout passed; screenshot review retains a mobile title/badge clipping limitation reported to the atlas lane for wave22. Durable references, commands, hashes, exact integer comparisons and limitations are in `flight-wave21-settled-acceptance.json`.
