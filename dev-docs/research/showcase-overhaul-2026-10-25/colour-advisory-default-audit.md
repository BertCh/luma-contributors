# Colour advisories versus current story use

Source: `/tmp/luma-showcase-overhaul-20261025/integration-check-wave13.log`, 42 advisories,
zero failures. CPU-only source tracing plus saved wave12b screenshots; no runtime changes.
**No additional high-impact default encoding defect was confirmed from these advisories.**
The warning total is not a count of defective story maps.

## Interpretation of the audit

The first 21 advisories are ColorBrewer light-table worst cases over every class count from 3
through 7, not necessarily the count used by a scene. Several actual tables use five classes,
skip a near-white first class, reverse a travel-time table intentionally, or expose classification
as the question being studied. The other 21 are named hue-registry ground variants, including
entries with no runtime consumer. All sequential ramp lightness tests pass; the derived dark
class schemes have zero non-monotone tables. The known dark slope convention is documented as
an avalanche-practice overlay and is not a generic intensity ramp.

## ColorBrewer warning mapping (21)

| Warning scheme | Current story use and default relevance |
| --- | --- |
| Blues | Vulnerability raw-indicator quintiles, segregation tiers, taxi destination fills, and selected choropleth variables. Numeric class boundaries and source values are available; the six-class worst case is not a universal default. |
| BuGn | Spatial-weights membership, similar-places similarity, and polygon compactness. Membership skips the palest light-ground class; five-class compactness has explicit numerical bounds. |
| GnBu | No current scene consumer found. |
| OrRd | Sun/shadow energy classes and migration timing speed cells. Timing uses five visible classes and drops the palest light-table class. |
| PuRd | Routing scenario delay, polygon distortion, terrain diagnostics. Routing's first no-delay class is transparent and the five remaining interval meanings are explicit. |
| Purples | Vulnerability comparison and segregation tier display. Quantile/tier labels retain their numeric meaning. |
| RdPu | Similar-places dissimilarity and variable-dependent choropleth classes; the warning is the seven-class worst case. |
| YlGn | No current scene consumer found. |
| YlGnBu | Routing/isochrones/accessibility, grouped observation rates, point-pattern support, migration timing, flow density and selected terrain products. Counts vary by use; maps have labelled intervals, queried values or support charts. No new polarity contradiction identified. |
| YlOrRd | Vulnerability score, street centrality, retail-network summaries and terrain intensity. Vulnerability uses five labelled score quintiles. |
| BrBG | No current scene class-table consumer found. |
| PiYG | No current scene class-table consumer found. |
| RdYlGn | No current scene consumer found; the very low 0.4 deuteranopia separation is not an active default map. |
| Spectral | Only an unused legacy `CLASS_PALETTES` catalogue label in `b5-palettes.ts`; that palette catalogue/select export has no consumer. No current scene default. |
| Set2 | No current scene consumer found. |
| Dark2 | No current scene consumer found. |
| Paired | No current scene consumer found. |
| Set1 | No current scene consumer found. |
| Set3 | No current scene consumer found. |
| Pastel1 | No current scene consumer found. |
| Accent | No current scene consumer found. |

The observed categorical schemes are explicitly curated category/group colours, not those seven
unused ColorBrewer qualitative tables. This audit does not claim that a categorical palette can
make an arbitrary number of categories uniquely discriminable; existing labels, category
selection, charts and documented limits remain necessary.

## Hue-registry warning mapping (21 ground-specific rows)

| Advisory row | Runtime path and decision |
| --- | --- |
| harm light | No current `getRegistryColors('harm', ...)` consumer. Scenes may use YlOrRd independently; those tables are covered above. |
| harm dark | Same unused registry row; ground-level class warning is not a default-scene finding. |
| fireEffect dark | **No runtime consumer.** Burn-severity and wildfire scenes do not call this entry. Its darkening-high-value warning therefore does not justify changing a shipped map palette. |
| people dark | Global-autocorrelation variable tables and hot-spots classified values. Authored default basemaps are paper; dark is an alternate ground. Numeric values/classes and charts provide independent reading. |
| nature dark | Bird migration timing share/count chart helper. Low-share cells intentionally recede; class labels, chart rows and queried values retain meaning. The initial flyway map instead uses its continuous density ramp plus species tracks. |
| natureWeights dark | Hot-spots classified observation cells; authored opening ground is paperCity, not this dark variant. |
| access dark | No current registry consumer. Travel-time scenes use explicit class tables instead. |
| remoteness dark | No current registry consumer. |
| load dark | US shipping density via `getTrafficDensityClasses`. Actual shipping steps explicitly investigate the way heavy-tailed density and class breaks make lanes appear/disappear. Lowest class has alpha140; legend shows interval counts and a no-track category, and a histogram supports the classification comparison. Saved step03 shows low-class lanes as dark purple, consistent with that stated question. |
| similarity light | No current registry consumer. Track-similarity options use their own palettes; an option named similarity is not this registry entry. |
| similarity dark | Same: no registry consumer. |
| rainfall light | No current registry consumer. The rainfall interpolation story uses its own ramp, not this table. |
| rainfall dark | Same: no registry consumer. |
| elapsedTime light | No current registry consumer. |
| elapsedTime dark | No current registry consumer; prior monotonicity repair is retained, but no scene claim is inferred. |
| waiting dark | Harbor stops/dwell helpers; first harbor playback step uses a uniform vessel colour and white tracks. Zone-dwell opens on paperCity and supplies numerical interval counts, time values, charts and outline styles. Low dark dwell values are an alternate encoding, not a new observed default failure. |
| vegetationChange light | No current registry consumer. Existing change scenes use separate explicit ramps/tables. |
| windSpeed dark | No current registry consumer; the terrain-flow option named windSpeed is unrelated to this hue entry. |
| hurricane light | No current registry consumer; hurricane category helpers use their explicit curated constants. |
| slope dark | Actual terrain slope helper, but explicitly documented decision thresholds (25/30/35/40/45/50°), transparent gentle terrain and alpha184 over relief. Preserve its authored avalanche convention; do not reverse it globally. |
| slopeGentle dark | No current registry consumer. |

## Saved visual cross-check

Inspected `global-autocorrelation-01`, `zone-dwell-01`, `harbor-playback-01`,
`us-shipping-day-01`, `us-shipping-day-03`, `migration-flyways-01` and
`terrain-basics-02`. The shipping classification lesson deliberately contains mostly low-class
cells and explicitly explains why. Zone-dwell supplies numeric classes and count bars. Global
autocorrelation uses five light-ground prevalence classes with named low/high examples. The
terrain decoder step's blank hatched result is an intentional invalid-decoder demonstration,
not a colour failure. `hot-spots-01` in the old wave12b archive failed during analysis startup;
that image cannot be used as evidence of a palette or visibility defect.

Recommendation: retain the existing warnings as design advisories. Do not introduce broad
palette changes based on this log. A future consumer of the currently unused `fireEffect.dark`
entry should resolve its polarity before adopting it.
