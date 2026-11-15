# Type-driven recommendation: systems that infer the visualization or analysis from the input data

Scope: mechanisms (metadata computed, candidate space, rules, ranking), presentation, cost handling and failure modes of
"the input infers the analysis" recommenders, as input to a rule-based suggestion engine for a WebGPU geospatial analysis
library hosted in a Kepler.gl-based app. Researched 2026-11-14. Primary PDFs were read in full text where reachable;
items read only from search snippets or secondary sources are marked.

## 1. Tableau Show Me: rule structure, ranking, unavailable options, later inference features

### Takeaway
Show Me is a small, hand-written rule table: each of 14 commands has an availability condition stated as counts of field
classes (C, Cdate, Qd, Qi), and some commands also have a rank condition; the default is the highest-ranked command whose
condition holds. Unavailable commands stay visible and greyed out, and hovering shows the minimum requirements. That is
close to a design that can be copied directly for operations that declare input kinds.

### Cited Findings
- Tableau fields have three properties: data type (text, date, date&time, numeric, boolean), data role (measure or
  dimension) and interpretation (discrete or continuous). Show Me classifies them as C = categorical (discrete and
  dimension), Cdate = categorical date, Q = quantitative (continuous), Qd = quantitative dependent (measure), and
  Qi = quantitative independent or Qdate (dimension) — [Mackinlay, Hanrahan, Stolte 2007, TVCG 13(6)](https://info290.github.io/papers/show-me.pdf)
- Automatic Marks (the mark default) is keyed on the rightmost fields of the Row and Column shelves (pane type). The table
  is C×C → text (cross-tab); Qd×C → bar; Qd×Cdate → line; Qd×Qd → shape (scatter); Qi×C → Gantt; Qi×Qd → line;
  Qi×Qi → shape (scatter). Specific cases that map the same way as general cases are omitted — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- The Show Me Alternatives menu has only 14 items, "and items are only active when they can build appropriate charts". The
  paper contrasts this with the Excel 2003 chart menu of over 70 items — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Ranked commands, as (availability condition, rank). The paper's convention is that 1 is the lowest rank and the default
  is the highest-ranked command whose condition is met:
  - Text Tables: at least 1 field, rank 1 (lowest, "primary utility is to look up specific values"; always available)
  - Aligned Bars: at least 1 Q, rank 2
  - Stacked Bars: at least 2 C and at least 1 Q; rank 3 only when there are at least 3 C
  - Discrete Lines: at least 1 Cdate and at least 1 Q, rank 4
  - Scatter Plots: 2 to 4 Q; rank 5 only for 2 Q (with more than four Q a lower-ranked command becomes the default)
  - Gantt: at least 1 C, at least 1 Qi, 1 to 2 Q, rank 6 ("highly ranked, the conditions ... are rarely satisfied")
  - [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Unranked commands can be picked but are never the default: Heat Map (at least 1 C, 1 to 2 Q), Highlight Table (at least
  1 C, at least 1 Q), Side-by-side Bars (at least 1 C, at least 1 Q), Measure Bars (at least 1 C, at least 2 Q),
  Continuous Lines (at least 1 Qi, at least 1 Q), Circle Charts (at least 1 C, at least 1 Q), Scatter Matrix (3 to 6 Q),
  Histogram (exactly 1 Q that supports binning). Heat Map was left unranked because ranking it "would have required an
  expensive query to the database to identify when the data would generate a large table" — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Ranking principle: "the ranking prefers views that encode data graphically"; "higher ranked commands typically have more
  specialized conditions that must be met for that particular view type to be effective". Each command has a condition
  that "determines whether the user can explicitly select the command", and the rank condition "is only used for the
  default computation" — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Add to Sheet (adding a single field) uses an "affinity" heuristic: the added field goes next to a field of the same type,
  so a categorical field added on the right creates a break-down into small multiples. Adding fields one at a time in
  different orders specifies a wide range of view types — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Cost rule: Show Me does not query the database while computing the default, because that computation enables the toolbar
  button in the basic UI loop. Queries are allowed only after the user picks an alternative, and they use the result
  cache — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- The paper (2007) did not include maps. It proposes extending the type system with "data types that represent space, such
  as longitude and latitude" — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Evaluation is log-based, from about 10 skilled users, 17 user IDs and 1,062 log files dated 5/30/2006 to 4/28/2007.
  Automatic Marks gave 8,248 shelf changes against 560 mark changes, which the paper reads as a "6.8% error rate". Add to
  Sheet was barely used (43 automatic adds against 3,921 direct adds, 1.1%). Show Me and Show Me Alternatives were used
  "modestly" (220 commands per 3,921 shelf adds, 5.6%). Skilled users found Show Me "less useful because it is hard to
  predict what view will be built" — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Current Tableau: Show Me works from the fields in the view plus the fields selected in the Data pane. Hovering a view
  type shows "a description showing the minimum requirements for each viz type", and the recommended type gets "an orange
  outline". A date dimension plus a measure produces a line suggestion — [Tableau help: Show Me](https://help.tableau.com/current/pro/desktop/en-us/buildauto_showme.htm)
- Explain Data "builds statistical models and proposes possible explanations for individual marks in a viz", including from
  data "that isn't used in the current view". It "can't tell you what is causing the relationships" — [Tableau help: Explain Data](https://help.tableau.com/current/pro/desktop/en-us/explain_data.htm)
- Explain Data requirements: one or more measures aggregated with SUM, AVG, COUNT or COUNTD, and at least one dimension in
  the view. Table calculations and Measure Values are not supported. An ideal data source has "at least 10-20 columns" in
  addition to the measure. Dimensions with more than 500 unique values are excluded, and categorical explanations read best
  with fewer than 20 categories. "Row-level data is necessary." Blended, cube and parameterised sources are not supported.
  A mark in a view with "a very low number of marks" is not explained (no threshold given) — [Tableau help: Explain Data requirements](https://help.tableau.com/current/pro/desktop/en-us/explain_data_requirements.htm)
- Tableau Pulse insight types include Top Drivers, Top Detractors, Unexpected Values (outside the range expected from
  history), Period over Period Change, Top/Bottom Contributors, Correlated Metrics, Current Trend and Trend Change. These
  are computed per defined metric, not per raw dataset. Names and licence tiers vary by release (from search summaries of
  the help page, not read in full) — [Tableau help: Pulse insight types](https://help.tableau.com/current/online/en-us/pulse_insights_platform_insight_types.htm)

### Inferences
- The Show Me pattern maps directly onto operations that declare accepted inputs. Each operation gets (a) an availability
  predicate written as counts of typed inputs, (b) an optional rank predicate that is stricter, and (c) a fixed integer
  rank. Text Tables shows why a cheap, always-available fallback should carry the lowest rank.
- "Expensive to decide, so leave it unranked" is a usable rule. Suggestions whose fitness needs a data scan (for example
  "is this point set dense enough for KDE") can be listed as available but never chosen as the default until a cheap
  statistic exists.
- The 6.8% override rate of Automatic Marks suggests that type-based defaults are usually acceptable. The low uptake by
  skilled users says suggestions should stay optional and predictable.

### Gaps
- The full 2007 rule list for the 14 alternatives depends on PDF columns that pdftotext garbled in places. Side-by-side
  Bars, Measure Bars, Continuous Lines and Circle Charts were read from the second text column and should be checked
  against Fig./Section 5 of the PDF.
- I found no Tableau source for the current Show Me rule table (24 chart types, including symbol and filled maps needing a
  geographic dimension). The help page gives only hover requirements, not the rules.
- Ask Data was not researched. It was retired in favour of Pulse and Tableau AI (from memory, unverified).

## 2. Research recommenders: metadata, candidate space, ranking, presentation, evaluation

### Takeaway
Two families exist. Encoding recommenders (APT, Voyager/CompassQL, Draco) enumerate encodings for chosen fields and rank by
perceptual-effectiveness rules. Data-interestingness recommenders (SeeDB, Foresight, QuickInsights, Power BI, Lux's
Correlation and Distribution tabs) enumerate field subsets or subspaces and rank by a statistic. Learned models (VizML,
Data2Vis, DeepEye) replace hand rules with models trained on corpora. User studies consistently find wider data coverage
with recommendations but no increase in bookmarked or "interesting" findings.

### Cited Findings
**Voyager (Wongsuphasawat et al., InfoVis 2015)**
- Compass computes per-field statistics, including "cardinality, min, max, standard deviation, and skew". It clusters
  visualizations "by data and visual similarity" and ranks them "by perceptual effectiveness". The main gallery shows "the
  top-ranked encoding for each unique set of variables and transformations", and expanding a view shows its alternative
  encodings. Low-cardinality integers are treated as ordinal. Encoding high-cardinality variables with color, shape, row
  or column is penalised — [Voyager paper](https://idl.cs.washington.edu/files/2015-Voyager-InfoVis.pdf)
- Study: 16 participants, about 2 hours each, against PoleStar (a Tableau-like baseline). Voyager users were exposed to 69.0
  more variable sets (baseline 30.6) and interacted with 13.4 more (baseline 27.2), "over 3 times more variable sets" seen,
  χ²(1, N=32)=38.056, p<0.001. Bookmark rate showed no tool effect (χ²=0.060, p=0.807). Of 179 bookmarks in Voyager, 124
  (69%) included a variable added automatically by the recommender — [Voyager paper](https://idl.cs.washington.edu/files/2015-Voyager-InfoVis.pdf)

**Voyager 2 (Wongsuphasawat et al., CHI 2017)**
- Combines manual specification with wildcards (one specification enumerates several charts) and "related views":
  univariate summaries when no focus view exists, "related summaries" (aggregate plots), "field suggestions" (the focus view
  plus one more field) and "alternative encodings" — [Voyager 2 paper](https://idl.cs.washington.edu/files/2017-Voyager2-CHI.pdf)
- Study (16 subjects): 102 unique field sets shown against 23 for PoleStar, and 41 interacted with against 17 (4.4× and
  2.4×). Users were exposed to 98% of fields against 80% (χ²=17.476, p<0.001) and interacted with 93% against 79%. Bookmark
  rate was again unaffected (χ²=0.381, p=0.537). Related views took 46.8% of chart interaction time and supplied 43.1% of
  bookmarks. 15 of 16 found Voyager 2 more valuable for exploration — [Voyager 2 paper](https://idl.cs.washington.edu/files/2017-Voyager2-CHI.pdf)

**CompassQL (Wongsuphasawat et al., HILDA 2016)**
- Recommendation is broken into phases: enumeration (fields, filters, aggregation, sorting, binning, encodings), choosing
  (candidates must respect the user's partial specification and expressiveness constraints), ranking ("perceptual
  effectiveness metrics" for encodings; data metrics such as Pearson correlation or deviation for data queries) and
  grouping (cluster similar candidates and show a top-ranked representative) — [CompassQL paper](https://idl.cs.washington.edu/files/2016-CompassQL-HILDA.pdf)

**Draco and Draco 2 (Moritz et al. 2018; Yang et al. VIS 2023)**
- Visualizations are sets of logical facts, and design knowledge is hard and soft constraints in Answer Set Programming
  solved with Clingo. Hard constraints remove ill-formed or non-expressive designs, for example the shape channel on a
  non-point mark, or mean-aggregating an ordinal field. The Draco cost of a design "is the weighted sum of the costs of all
  violated soft constraints", and the default knowledge base is "adopted from CompassQL rules" — [Draco 2 paper](https://arxiv.org/abs/2308.14247)
- Example constraints. Hard: eliminate a log scale on a field containing zero or negative values. Soft:
  `#const high_cardinality_shape_weight = 10` charges cost 10 when the shape channel encodes a field with more than 8
  distinct values — [Too Many Cooks, Zeng et al. 2023](https://arxiv.org/abs/2308.14241)
- Weights are set by hand or learned by Draco-Learn from ranked pairs of visualizations, for example from
  graphical-perception experiments. The Too Many Cooks study modelled results from 30 papers, found 98 of 147 soft
  constraints covered, and showed that some studies dominate the learned recommendations. Raw learned weights vary widely
  in scale between runs — [Too Many Cooks](https://arxiv.org/abs/2308.14241); [Draco 1 record](https://par.nsf.gov/servlets/purl/10074159)
- Draco 2 adds a nested specification format, a test suite and documentation, and debugging tools: a bar chart of
  constraint weights and a heat map of how often each constraint is violated per specification — [Draco 2 paper](https://arxiv.org/abs/2308.14247); [github.com/cmudig/draco2](https://github.com/cmudig/draco2)

**Lux (Lee et al., VLDB 2021)**
- Metadata per column: the list of unique values, cardinality and min/max. Lux infers the semantic type from the pandas
  dtype plus cardinality, supports nominal, quantitative, geographic and temporal types, and lets users override a wrong
  type — [Lux paper](https://arxiv.org/abs/2105.00121)
- Default actions (shown as tabs): Correlation (scatter for pairs of quantitative columns, ranked by Pearson correlation),
  Distribution (histograms of quantitative columns, ranked by skew), Occurrence (bar charts of categorical columns, most
  uneven first), Temporal (line chart) and Geographic (choropleth). Actions that follow an intent are Enhance (add one
  attribute), Filter (add or change one filter) and Generalize (remove an attribute or filter). Series, Index,
  Pre-aggregate and Pre-filter cover dataframes that are 1-D, indexed, already aggregated or already filtered — [Lux paper](https://arxiv.org/abs/2105.00121); [Lux overview docs](https://lux-api.readthedocs.io/en/latest/source/getting_started/overview.html)
- Geographic detection works from the column name and content, for example a column named state or country holding state
  or country values. Temporal detection uses format, content and name — [Lux data types docs](https://lux-api.readthedocs.io/en/stable/source/advanced/datatype.html) (read via search summary)
- Formative study: 587 public notebooks. Evaluation: 16 early adopters. 9 of 16 said printing and visualizing dataframes
  was most useful, and 8 of 16 found the intent language hard to learn. Task success was 68%, 87% and 71% on three
  exercises. 12 of 16 asked for lower latency on large data, and 13 of 16 signed up for follow-up — [Lux paper](https://arxiv.org/abs/2105.00121)

**VizML (Hu et al., CHI 2019)**
- Trained on 1 million dataset–visualization pairs from the Plotly Community Feed. Each column has 81 single-column
  features in four categories: Dimensions (row count), Types (categorical, temporal or quantitative), Values (statistics,
  sortedness, uniqueness) and Names. Each pair of columns has 30 features, and features are aggregated to 841
  dataset-level features. A neural network predicts design choices such as visualization type and whether a column is on
  the X or Y axis. On a crowdsourced consensus benchmark it performs "comparably to Plotly users and Mechanical Turkers" — [VizML paper](https://arxiv.org/abs/1808.04819)

**Data2Vis (Dibia and Demiralp 2019)**
- A sequence-to-sequence network translates a dataset sample (JSON) into a Vega-Lite specification. It learns "the
  vocabulary and syntax for a valid visualization" from a "relatively small training dataset" — [Data2Vis paper](https://arxiv.org/abs/1804.03126)
  (training set size not confirmed)

**DeepEye (Luo et al., ICDE 2018)**
- Three subproblems. Recognition: a binary classifier decides "good or bad". Ranking: a learning-to-rank model or
  expert-written partial orders. Selection: top-k over a search space that includes transformations (binning, grouping,
  sorting). The partial-order factors are (1) matching quality between data and chart (whether the chart shows the data's
  inherent feature, such as trend or correlation), (2) quality of the transformation ("the smaller the cardinality of the
  transformed data, the better") and (3) column importance. Example rule: for categorical X, pie or bar beats scatter or
  line — [DeepEye paper](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/icde18-deepeye.pdf)

**Foresight (Demiralp et al. 2017)**
- "Insight classes", each with a ranking metric and a visualization: Dispersion (variance, histogram), Skew (standardized
  skewness, histogram), Heavy tails (kurtosis, histogram), Outliers (average standardized distance of detected outliers,
  box plot), Heterogeneous frequencies (relative frequency of the top-k values, Pareto chart) and Linear relationship
  (correlation), among others. Classes are pluggable, and insights are computed approximately from precomputed sketches
  and samples — [Foresight paper](https://arxiv.org/abs/1707.03877)

**SeeDB (Vartak et al., VLDB 2015)**
- Candidates are aggregate views (dimension, measure, aggregate). Utility is the deviation between a target subset and a
  reference, measured with EMD, Euclidean distance, K-L divergence or similar. Speed comes from sharing computation across
  views and pruning low-utility views — [SeeDB paper](http://www.vldb.org/pvldb/vol8/p2182-vartak.pdf)

**QuickInsights (Ding et al., SIGMOD 2019) and Power BI**
- A unified formulation of "insights" plus a mining framework, evaluated on 447 real datasets with expert and non-expert
  user studies, and shipped in Power BI — [MSR page](https://microsoft.com/en-us/research/?p=588988); [camera-ready PDF](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/QuickInsights-camera-ready-final.pdf)
- Power BI insight types: top and bottom performers (category outliers), significant changes over time (change points),
  correlated patterns, consistent values (low variance), dominant contributors (majority), unusual values (outliers),
  upward or downward trends, seasonal patterns, stable proportions (steady share) and unusual dates or times (time-series
  outliers). Up to 10 insight types run on a dashboard tile — [Power BI insight types](https://learn.microsoft.com/en-us/power-bi/explore-reports/end-user-insight-types)
- Quick Insights shows "up to 32 separate insight cards", each "a chart or graph plus a short description". Cards can be
  pinned, and "insights on an insight" (scoped insights) run on a card's data. DirectQuery, streaming, push datasets and RLS
  are not supported — [Power BI Quick Insights](https://learn.microsoft.com/en-us/power-bi/create-reports/service-insights)
- Hidden columns are not searched. Duplicate columns waste search time ("takes valuable time away"). The error "data isn't
  statistically significant" appears for simple models, models with little data, or models without date or numeric
  columns — [Optimize for Quick Insights](https://learn.microsoft.com/en-us/power-bi/create-reports/service-insights-optimize)

**Excel Analyze Data, Google Sheets**
- Excel Analyze Data does not support datasets over 1.5 million cells, with "no workaround". It works best on an Excel
  table with one row of unique, non-blank headers. Users can exclude fields and set how each is summarised. A field marked
  "Not a value" acts as a row label next to a summed field, or as a count of unique values on its own — [Microsoft support: Analyze Data](https://support.microsoft.com/excel/analyze-data-in-excel) (read via search summary)
- Google's current Sheets help describes Gemini (summaries, charts, pivot tables and insights from prompts) rather than
  the older Explore panel. Gemini "may not be able to support your request", and its charts do not update with the
  source data — [Google Docs help](https://support.google.com/docs/answer/6280499)

**ydata-profiling**
- Automatic alerts with configurable thresholds: SKEWED (`skewness_threshold`, default 20), IMBALANCE (categorical and
  boolean `imbalance_threshold`, default 0.5) and HIGH_CORRELATION (per-method `threshold`, default 0.9). Other alerts
  include constant, high cardinality, missing, unique and zeros (names read from snippets) — [ydata-profiling settings 4.6](https://docs.profiling.ydata.ai/4.6/advanced_settings/available_settings/)

**Comparisons of algorithms**
- An evaluation framework treats any recommender as (1) a graph of the design space, (2) a traversal method and (3) an
  oracle that ranks candidates. An empirical comparison of four algorithms found they "behave similarly in terms of user
  performance" — [Zeng et al. 2021](https://arxiv.org/abs/2109.02706)

### Inferences
- For a catalog of about 280 operations, the shape that fits is CompassQL's enumerate → choose (hard constraints) →
  rank → group pipeline, with Draco-style costs: hard constraints are the operation's accepted geometry and column types,
  and soft constraints are weighted penalties (for example "count field without normalization", or "under 30 features").
  A weighted sum keeps the ranking explainable, because each violated soft constraint is a sentence the UI can show.
- Use data-interestingness ranking (Foresight or Lux style: skew, correlation, spatial autocorrelation) only inside a
  family, to order which column to pre-fill. It should not decide which operation is offered.
- The evidence says recommenders broaden coverage (3–4× more field sets, 98% against 80% of fields) without increasing
  useful findings. Claims beyond "helps users notice fields and operations they would miss" are not supported.

### Gaps
- QuickInsights' scoring formulas (significance × impact) were not extracted, because the PDF download failed.
- VizML's exact accuracy numbers per task were not extracted. Data2Vis training-set size is unverified.
- APT (Mackinlay 1986) was not fetched. Its expressiveness and effectiveness criteria and the ranking of perceptual tasks
  by data type are the basis of Show Me and CompassQL, but here they are from memory, unverified.
- Data Wrangler (the VS Code / Microsoft one) suggestion behaviour was not researched.
- SeeDB user study numbers were not extracted.

## 3. Predictive interaction in data preparation: suggesting transforms from a selection

### Takeaway
Wrangler and Trifacta invert the flow: the user selects or highlights data, the system enumerates candidate transforms,
ranks them by a corpus frequency model with simplicity and diversity rules, and previews each in the table before
anything is committed. This "guide/decide loop" is the closest model for "select some features or a column, get suggested
analyses".

### Cited Findings
- Predictive Interaction: the user guides by highlighting data, the system "suggests a variety of possible next steps",
  the user decides, and the loop repeats (the "guide/decide loop"). Users "have reported order-of-magnitude productivity
  gains" (an anecdotal field report) — [Heer, Hellerstein, Kandel, CIDR 2015](http://cidrdb.org/cidr2015/Papers/CIDR15_Paper27.pdf)
- After the second highlight the top-ranked suggested transform changes. Suggestions are a ranked list, and the table
  highlights what the top suggestion would match — [CIDR 2015](http://cidrdb.org/cidr2015/Papers/CIDR15_Paper27.pdf)
- Wrangler ranks transforms in this order: (1) explicit interactions (types matching a menu choice rank higher); (2)
  specification difficulty (transforms that accept "hard" row or text-selection parameters rank higher); (3) corpus
  frequency conditioned on the initiating interaction (for a text selection of 3 or fewer characters, split ranks above
  extract, and the reverse for longer selections). Within a type, transforms sort by corpus frequency and then by
  ascending complexity (number of predicate clauses or regex tokens). Finally, no transform type may fill more than 1/3 of
  the list. The engine considers the top-k (default 4) parameterizations that co-occur most often — [Kandel et al., Wrangler, CHI 2011](https://vis.stanford.edu/files/2011-Wrangler-CHI.pdf)
- Suggestions are not executed on click. The selected one becomes the editable "current working transform", with a natural
  language description and a visual preview (for example, rows to be deleted highlighted in red) — [Wrangler](https://vis.stanford.edu/files/2011-Wrangler-CHI.pdf)
- Wrangler has semantic roles (geographic locations, government codes, currencies, dates). A zip code role checks a 5-digit
  format against a dictionary and can map to a state or a central lat-lon. Roles drive parsing, validation and
  suggestion — [Wrangler](https://vis.stanford.edu/files/2011-Wrangler-CHI.pdf)
- In a study against Excel, median completion time in Wrangler was over twice as fast (main effect of tool
  F(1,54)=23.65, p<0.001), for novices and experts alike. Users who got stuck fared slightly better in Excel, because
  Wrangler lacks manual editing — [Wrangler](https://vis.stanford.edu/files/2011-Wrangler-CHI.pdf)
- Power Query Column from Examples: the user types the desired output for a row, and Power Query finds a matching
  transformation, fills the remaining rows and shows the M formula. It "works only on the top 100 rows of your data
  preview". Supported transforms are a fixed list (text extract, combine, dates, number operations,
  bucketing/ranges, conditional column) — [Microsoft Learn: Column from examples](https://learn.microsoft.com/en-us/power-query/column-from-example)

### Inferences
- Selection-driven suggestion fits a map UI: lasso some polygons, or click a column header, and rank operations
  conditioned on what was selected. A Wrangler-like diversity cap (no family takes more than 1/3 of slots) stops, for
  example, ten interpolation variants from crowding out clustering.
- "Suggest, preview, then commit as an editable step" fits a computation record: an accepted suggestion becomes a
  parameterized workflow step rather than a one-shot result.

### Gaps
- Trifacta/Alteryx Designer Cloud's current suggestion ranking is not documented publicly. Not found.

## 4. Semantic type detection for columns, and how geo products detect geographic roles

### Takeaway
Research models (Sherlock, Sato, Doduo) reach F1 of about 0.89–0.93 on 78 DBpedia types from web tables, but products
detecting geography rely on header-name dictionaries plus value validation. kepler.gl matches lat/lng name pairs on
separator boundaries, validates H3 on the first non-null string, and sniffs WKB and GeoJSON. Felt matches regions when at
least 70% of values match. Tableau and ArcGIS Insights use names plus a sample of values. None of the products found
detects counts versus rates automatically, beyond keyword heuristics.

### Cited Findings
- Sherlock: trained on 686,765 columns from VizNet, labelled by matching 78 DBpedia semantic types to column headers. It
  uses 1,588 features (statistics, character distributions, word embeddings, paragraph vectors), reaches F1 0.89, and beats
  dictionary, regex and crowd baselines — [Sherlock, Hulsebos et al. KDD 2019](https://arxiv.org/abs/1905.10688)
- Sato adds table context (LDA topic features plus a CRF over columns) and reaches support-weighted F1 0.925 and macro F1
  0.735. It notes that products (Power BI, Tableau, Trifacta) detect only "simple, well-structured semantic types" with
  regex or dictionaries — [Sato, Zhang et al. VLDB 2020](https://arxiv.org/abs/1911.06311)
- Doduo (pre-trained language model over the whole table) improves on TURL and Sato by up to 4.0% and 11.9% on the
  WikiTable and VizNet benchmarks (VizNet: 119,360 columns, 78 types) — [Doduo, Suhara et al. SIGMOD 2022](https://arxiv.org/abs/2104.01785)
- kepler.gl point pairs: `TRIP_POINT_FIELDS = [['lat','lng'],['lat','lon'],['lat','long'],['latitude','longitude']]`,
  `ALTITUDE_FIELDS = ['alt','altitude']`, `GEOJSON_FIELDS = ['_geojson','all_points','geojson']`, and arc-endpoint
  prefixes `begintrip` and `dropoff` — [kepler.gl default-settings.ts](https://github.com/keplergl/kepler.gl/blob/master/src/constants/src/default-settings.ts)
- `findPointFieldPairs` lowercases names, matches the lat token with the regex `(^|special)lat(special|$)` where
  separators are `#_&@.- `, looks for a partner field with the same prefix, and names the layer after the remaining
  prefix, or "point". This is name-only matching with no value-range check — [kepler.gl kepler-table.ts](https://github.com/keplergl/kepler.gl/blob/master/src/table/src/kepler-table.ts)
- kepler.gl type analysis runs on a sample (`getSampleForTypeAnalyze`, `sampleCount = 50`). A string column becomes H3 if
  the first non-null string passes `h3IsValid`. A string column becomes GEOMETRY if any value is hex WKB. Arrow geometry
  types map to GEOMETRY directly. The H3 layer auto-creates one layer per detected hex field — [kepler.gl data-type.ts](https://github.com/keplergl/kepler.gl/blob/master/src/common-utils/src/data-type.ts); [h3-hexagon-layer.ts](https://github.com/keplergl/kepler.gl/blob/master/src/layers/src/h3-hexagon-layer/h3-hexagon-layer.ts); [data-processor.ts](https://github.com/keplergl/kepler.gl/blob/master/src/processors/src/data-processor.ts)
- kepler.gl default color field: names containing `_id, id, index, uuid, guid, uid, gid, serial, zip, code, post, region,
  fips, cbgs, h3, s2, a5, geohash, lat, lon, lng, latitude, longitude, _x, _y` are excluded as non-metrics. Preferred
  metric words, in order, are `score, metric, value, sum, count, unique, mean, mode, median, max, min, deviation, variance,
  p99…p05, cnt, val`. Constant fields are rejected by scanning up to 1,000,000 values — [kepler.gl utils dataset-utils.ts](https://github.com/keplergl/kepler.gl/blob/master/src/utils/src/dataset-utils.ts)
  (read from current master. Some lines, such as `a5` and the day-first date heuristic, may be recent additions.)
- kepler.gl date parsing: month-first is the default for slash dates. It switches to D/M/YYYY only when every sample is a
  valid day-first date and at least one day exceeds 12 — [kepler.gl data-type.ts](https://github.com/keplergl/kepler.gl/blob/master/src/common-utils/src/data-type.ts)
- Tableau "recognises a lot of common geographic data, such as state and city names" and assigns roles automatically.
  Secondary sources describe this as using the field name plus a sample of values. Roles: Airport, Area Code, CBSA/MSA,
  City, Congressional District, Country/Region, County, NUTS Europe, Latitude, Longitude (numeric only), State/Province,
  ZIP/Postcode. Assigning a role generates Latitude (generated) and Longitude (generated). Unmatched locations need custom
  geocoding or manual edits — [Tableau geographic roles](https://help.tableau.com/current/pro/desktop/en-us/maps_geographicroles.htm); [Tableau tutorial](https://help.tableau.com/v2020.2/guides/get-started-tutorial/en-us/get-started-tutorial-explore.htm)
- Felt: recognises latitude/longitude, lat/lon, y/x and prefixed names such as `site_lat`, in decimal degrees or DMS, in
  one column or two. Projected coordinates such as UTM are not read. It geocodes `address` or combined address parts free
  for up to 10,000 rows per file. "Geomatching" attaches boundaries when at least 70% of a column matches a region type
  (countries, states, time zones, NUTS/LAU, US counties, CBSAs, ZIPs, tracts, block groups, Australian postcodes), and
  FIPS codes must keep their full length. Columns named `wkt`, `geojson` or `geometry` can hold WKT, hex WKB or GeoJSON.
  H3 indexes are read as hex or integer, one resolution per column — [Felt help: spreadsheets and geocoding](https://help.felt.com/upload-anything/files/spreadsheets-and-geocoding)
- CARTO: geometry column names `geom, Geom, geometry, the_geom, wkt, wkb`; latitude `latitude, lat, Latitude`; longitude
  `longitude, lon, Lon, Longitude, lng, Lng`. Content guessing is on by default in the dashboard, but off in the Import
  API. Street addresses are not geocoded on import. Lat and long combined in one column are not geocoded — [CARTO CLI imports](https://docs.carto.com/carto-for-agents/cli/command-reference/imports); [CARTO importing data](https://docs.carto.com/carto-user-manual/data-explorer/importing-data); [CARTO guessing](https://carto.com/help/working-with-data/guessing/)
- ArcGIS Insights: "Fields named zip, postal, or year automatically apply a string field type even if the underlying data
  is a number type". CSV dates arrive as String. Potential rate fields are "usually added as Number", so rates are not
  detected automatically — [ArcGIS Insights field types](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm)

### Inferences
- A practical detector for this app combines kepler.gl's name rules (already present in the host) with value checks that
  kepler.gl lacks: a lat value in [-90, 90] and a lng value in [-180, 180]; integer-valued, non-negative columns as
  candidate counts; values in [0, 1], or names with rate/pct/per/density/ratio, as candidate rates; high uniqueness ratio
  plus an id-like name as identifier; and ZIP/FIPS as strings, never numbers. Each detection should carry a confidence and
  stay overridable, like Lux's override and ArcGIS's type switching.
- Counts versus rates cannot be detected reliably from values. The honest UI is to ask once ("Is `population` a count or a
  rate?") whenever an operation's soft constraints depend on it.

### Gaps
- Power BI's data categories (Latitude, Longitude, Place, Postal code and others) and its auto-detection behaviour were not
  fetched.
- Tableau's exact detection algorithm (name dictionary against value sampling, sample size) is not in official docs.
  Found only in secondary sources.

## 5. ArcGIS Insights field roles and drop-zone behaviour

### Takeaway
Insights is the closest product precedent for an analysis-capable, type-driven UI. Each field has a type (Location,
String, Number, Rate/Ratio, Date/Time, plus Subtype and Coded value domain), each type has a default visualization and a
default statistic, the chosen visualization follows from the combination of dragged fields, only compatible chart types
are enabled, and map cards expose analysis drop zones (Spatial Aggregation, Spatial Filter). Insights was deprecated, with
retirement in 2026.

### Cited Findings
- Defaults per field type:
  - Location: location map with one location field, link map with two; statistic count
  - String: bar chart, or unique values map; count
  - Number: histogram with one, scatter plot with two, graduated symbol map; sum
  - Rate/Ratio: histogram, scatter plot, or choropleth map; sum in charts, but "no statistic" in choropleth maps, so
    values are not summed for shading
  - Date/Time: time series graph; it cannot style a map
  - Subtype and coded value domain: treated as strings
  - [ArcGIS Insights field types](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm)
- Combination rules: mapping a string, number or rate/ratio needs one location field, and link maps need two. Histograms
  need one number or rate/ratio field, and scatter plots need two. Time series need a date/time field. Number can switch to
  String or Rate/Ratio. Location and Date/Time cannot be changed. A field's type cannot change while a card uses it — [ArcGIS Insights field types](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm)
- Drag and drop: "The style of the map or type of chart created will depend on the type of fields you choose". Dragging
  onto an existing card restyles it or adds a layer. Map cards have Spatial Aggregation and Spatial Filter drop zones
  ("drag a dataset and click Run"). Dragging selected or filtered data to a new map creates a result dataset — [ArcGIS Insights drag-and-drop](https://doc.arcgis.com/en/insights/latest/get-started/get-started-with-drag-and-drop.htm)
- "For the Chart menu, only charts that are compatible with your data selection will be enabled", and the Visualization
  type menu shows only compatible options — [ArcGIS Insights column chart](https://doc.arcgis.com/en/insights/2022.1/create/column-chart.htm) (read via search summary)
- Rate/Ratio fields are typically produced by Calculate Ratio or Calculate % Change, or set by hand — [ArcGIS Insights field types](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm)
- Insights 2025.1 docs state it is deprecated and retiring in 2026 — [ArcGIS Insights create maps 2025.1](https://doc.arcgis.com/en/insights/2025.1/create/create-maps.htm) (via search summary)

### Inferences
- The separate Rate/Ratio type, with its different default map and its rule against summing, is the one measurement
  distinction an analysis UI most needs. It is a cheap, explicit precedent for a "count | rate | other" tag on numeric
  columns that drives which hot-spot, choropleth and aggregation defaults are offered.
- Analysis drop zones on a map card (drop a second dataset to aggregate or filter) suit two-input operations such as
  point-in-polygon, overlay and network snapping.

### Gaps
- No Insights page states the exact chart compatibility matrix, or what is shown when a drop zone rejects a combination.

## 6. Measurement theory for spatial analysis suggestion

### Takeaway
Stevens' four levels are not enough for cartography. Chrisman separates counts, extensive ratios (additive, suited to
proportional symbols), derived ratios or densities (suited to choropleths), absolute scales such as probabilities, and
cyclic measures such as angles and time of day. Tools barely enforce these: ArcGIS Insights encodes the count/rate split
through the Rate/Ratio type, and ArcGIS Pro's Hot Spot docs only warn about lack of variation and point incidents. No
tool found warns automatically on choropleths of raw counts.

### Cited Findings
- Chrisman (1995): cartographers teach Stevens' unified "ratio" level, but "must make a distinction between those
  attributes permissible on choropleth maps (densities and other derived measures) and those permissible on proportional
  symbol maps ('extensive' measures where addition is the underlying mechanism)" — [Chrisman, Beyond Stevens, Auto-Carto 12](https://cartogis.org/docs/proceedings/archive/auto-carto-12/pdf/beyond-stevens-a-revised-approach-to-measurement.pdf)
- Counts aggregated over regions "are discrete ... but a count captures more mathematical structure than the other
  discrete levels". They look like ratios but "cannot be rescaled" because they are tied to the unit counted, and counting
  "depends upon the recognition of objects, a procedure tied to nominal measures". The absolute scale (for example
  probability) fixes both 0 and 1. Cyclical measures (angles, parts of time) repeat — [Chrisman, Beyond Stevens](https://cartogis.org/docs/proceedings/archive/auto-carto-12/pdf/beyond-stevens-a-revised-approach-to-measurement.pdf)
- Chrisman's 1998 paper (Cartography and GIS 25(4)) argues that Stevens' levels "are not adequate" for cartography and
  proposes a broader framework — [Rethinking Levels of Measurement for Cartography](https://scilit.com/publications/882d8a2df62cb35e03a5d5cf546bc655).
  A Penn State course says Chrisman counted nine levels, while slide summaries list about ten (nominal, graded membership,
  ordinal, interval, log-interval, extensive ratio, cyclical ratio, derived ratio, counts, absolute). The exact list is
  unverified — [PSU Nature of Geographic Information](https://courses.ems.psu.edu/natureofgeoinfo/c3_p8.html) (search snippet); [Brunsdon slides](https://www.geog.leeds.ac.uk/ectqg17/presentations/c.brunsdon.pdf) (search snippet)
- ArcGIS Pro Hot Spot Analysis (Gi*): "The input field should contain a variety of values", and it "cannot solve if all
  input values are 1". For incident points, where "your focus is on presence or absence rather than some measured
  attribute", the docs say to aggregate first or use Optimized Hot Spot Analysis. Shapefile nulls may be read as zero — [ArcGIS Pro Hot Spot Analysis](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/hot-spot-analysis.html)
- ArcGIS Insights uses no statistic (no summing) when shading a choropleth by a Rate/Ratio field, and sums Number fields — [ArcGIS Insights field types](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm)
- Draco-style hard constraints already encode level-of-measurement validity in chart recommenders, for example "aggregating
  ordinal fields with mean" is non-expressive — [Draco 2](https://arxiv.org/abs/2308.14247)

### Inferences
- These rules can be encoded directly as constraints in a suggestion engine:
  - Hard: no mean or sum over nominal columns; no linear statistics over cyclic columns (bearing, hour) without circular
    handling; Gi* or Moran's I need variance greater than 0.
  - Soft, with a reason shown: choropleth or hot spot on a count-tagged field without a denominator (suggest "normalize by
    area or population"); interpolation such as IDW or kriging of an extensive count (counts are object or areal data, not
    field samples); summing a rate during aggregation (use a weighted mean). Interpolation suits field data (continuous
    surfaces sampled at points) and density or KDE suits object or event data.
- Field versus object data is mostly given by the input kind (raster means field; points with a measured attribute may be
  either). Where it matters, the UI must ask: "Are these points samples of a surface (temperature) or events (crimes)?"

### Gaps
- I did not find an authoritative product doc that warns automatically on choropleths of raw counts, nor primary text of
  the field-versus-object distinction (Couclelis; Goodchild). Both are background knowledge, not cited here.
- The Axis Maps normalization guide URL returned 404.

## 7. UX patterns for showing recommendations, and evidence on trust and over-reliance

### Takeaway
The recurring patterns are:
- a fixed grid of every option, with unavailable items greyed and a hover showing minimum requirements (Show Me);
- enable only the compatible options (ArcGIS Insights);
- tabs per analytic intent, filled with thumbnails computed on the real data (Lux, Voyager);
- cards that pair a chart with a one-line explanation and can be pinned, plus "insights on an insight" (Power BI);
- univariate overviews on load instead of a blank canvas (Voyager 2, Lux, ydata-profiling);
- preview, then commit as an editable step (Wrangler).

Trust evidence is mixed. Users say they prefer human recommendations but judge by whether their attributes of interest
appear. More recommended views increase exposure but not findings, and more comparisons raise false-discovery risk.

### Cited Findings
- Show Me keeps a "stable grid" of icons so learned positions persist. Icons have saturated colors when active, and hover
  shows minimum requirements. The recommended item gets an orange outline — [Show Me paper](https://info290.github.io/papers/show-me.pdf); [Tableau help](https://help.tableau.com/current/pro/desktop/en-us/buildauto_showme.htm)
- When new users invoke an alternative, "they see how the fields are placed on shelves to specify that type of view, which
  can help them learn". Skilled users find it "hard to predict what view will be built" — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Lux shows recommendations as tabs per action inside the dataframe printout, with a toggle between table and Lux view.
  Users skim the pandas table for an average of 28 seconds before toggling, which Lux uses to compute in the background — [Lux paper](https://arxiv.org/abs/2105.00121)
- Voyager 2: "Rather than starting with a blank screen, univariate summaries help users familiarize themselves with the
  different data fields". Field suggestions appear only when an empty encoding channel exists — [Voyager 2 paper](https://idl.cs.washington.edu/files/2017-Voyager2-CHI.pdf)
- Power BI: up to 32 cards, each a chart plus a short description, pinnable to dashboards. Running insights on a card
  produces related cards — [Power BI Quick Insights](https://learn.microsoft.com/en-us/power-bi/create-reports/service-insights)
- Trust (Vis Ex Machina, crowdsourced): participants "express an a priori preference for recommendations provided by
  humans", but evaluate recommendations "based on the inclusion of data attributes they find most relevant", and are
  "source-agnostic when evaluating visualization recommendations of equal quality". Two behaviours appeared:
  "all-rounders" judge the set as a whole, while "seekers" look for particular attributes. The findings "partially support
  existing assumptions ... that users trust automated visualization recommendation systems" — [Zehrung, Singhal, Correll, Battle, CHI 2021](https://arxiv.org/abs/2101.04251)
- Bookmark rates did not change with recommendations in either Voyager study (p=0.807 and p=0.537), despite 3–4× exposure — [Voyager](https://idl.cs.washington.edu/files/2015-Voyager-InfoVis.pdf); [Voyager 2](https://idl.cs.washington.edu/files/2017-Voyager2-CHI.pdf)
- Multiple comparisons: "As more visualizations are examined and more comparisons are made, the probability of discovering
  spurious insights increases". The study involved about 28 analysts. A confirmatory approach that accounts for all
  comparisons approaches the results of using a validation set — [Zgraggen, Zhao, Zeleznik, Kraska, CHI 2018](https://dspace.mit.edu/handle/1721.1/137892) (abstract only)
- Explain Data explicitly disclaims causality ("You are the expert on your data") — [Tableau Explain Data](https://help.tableau.com/current/pro/desktop/en-us/explain_data.htm)
- Draco 2 makes reasons inspectable through a chart of constraint weights and a heat map of violations per candidate — [Draco 2](https://arxiv.org/abs/2308.14247)

### Inferences
- For this app: (1) a profiling summary on load (geometry kind, CRS, row count, column tags with confidence); (2) a stable
  grid of operation families, each enabled, enabled-but-not-recommended, or greyed with "needs: polygon layer + numeric
  rate column" and a one-click fix where one exists ("normalize by area", "aggregate points to H3"); (3) a short ranked
  "Suggested for this layer" row. Each suggestion gets a "because your data has ..." line built from the matched
  constraints, and a thumbnail computed on a sample; (4) apply as an editable workflow step with defaults visible.
- Showing more suggestions does not yield more findings and raises spurious-pattern risk. Cap the suggested row (about
  5–8), favour operations that carry significance testing (permutation p-values, FDR for LISA), and label exploratory
  results as exploratory.
- Users judge suggestions by whether they include "their" columns. Let the user pin columns of interest and rank
  suggestions that use those columns first.

### Gaps
- No controlled study found that measures automation bias specifically for analysis (not chart) recommendations, or for
  greyed-out-with-reason designs against hidden-option designs.
- The full Zgraggen result numbers were not extracted.

## 8. How recommenders handle cost

### Takeaway
Every system bounds cost in one of five ways: availability is decided only from metadata (Show Me, no queries); scores are
estimated on a sample and only the top-k are recomputed exactly (Lux: k=15, 30k-row sample); the cheapest action is
streamed first (Lux); data is processed in phases with confidence-interval or bandit pruning (SeeDB); or sketches are
precomputed (Foresight). Product limits are blunt caps, such as 1.5 million cells in Excel, 100 preview rows in Power Query,
32 cards in Power BI and 50 sampled rows for kepler.gl type inference.

### Cited Findings
- Show Me computes the default without querying the database. Queries run only when the user picks an alternative, through
  the result cache — [Show Me paper](https://info290.github.io/papers/show-me.pdf)
- Lux prune: a first pass approximates the score of every candidate on a cached sample (for example 30k rows of a 1M-row
  dataframe), then the top-k are recomputed exactly. This only pays when N·t_exact ≫ N·t_approx + k·t_exact. The
  experiments used k=15 and pruned actions with more than k candidates. Sample sizing: "for most actions 10% (5k rows)" of
  a 50k dataset reached over 90% Recall@15, and the 100k Airbnb data needed 20–40% (20–40k rows). The 30k cap gave 90.5%
  recall on Airbnb and at least 95% on Communities — [Lux paper](https://arxiv.org/abs/2105.00121)
- Lux async: estimate each action's cost from per-visualization relational cost models, compute the cheapest action first,
  and stream results to the widget as each action completes. Correlation on wide quantitative data is the typical
  "laggard" — [Lux paper](https://arxiv.org/abs/2105.00121)
- Lux overhead: at most 2 s above pandas for dataframes under 1M rows (21 s at 10M rows in Airbnb), and at most 1.5 s for
  Communities. Notebook print overhead had a median of 2.8 s with a long tail (s.d. 183.4 s) — [Lux paper](https://arxiv.org/abs/2105.00121)
- SeeDB phased execution: the data is split into n phases (for example 10 phases of 10,000 records for 100,000 rows). After
  each phase, views are pruned using worst-case Hoeffding–Serfling confidence intervals (drop a view whose upper bound lies
  below the current top-k lower bounds) or multi-armed bandit allocation. Sharing computation across views also helps.
  Together these give "multiple orders of magnitude speedup", and pruning alone a further 5× — [SeeDB paper](http://www.vldb.org/pvldb/vol8/p2182-vartak.pdf)
- Foresight preprocesses the dataset into sketches, samples and indexes, and exploits sketch composability, for fast
  approximate insight metrics — [Foresight paper](https://arxiv.org/abs/1707.03877)
- Wrangler limits the displayed suggestions to "a relevant handful" from a space that can be "in the hundreds", using top-k
  (default 4) parameterizations and the 1/3-per-type cap — [Wrangler](https://vis.stanford.edu/files/2011-Wrangler-CHI.pdf)
- Product caps: Excel Analyze Data, 1.5 million cells — [Microsoft support](https://support.microsoft.com/excel/analyze-data-in-excel).
  Power Query Column from Examples, top 100 preview rows — [Microsoft Learn](https://learn.microsoft.com/en-us/power-query/column-from-example).
  Power BI Quick Insights, up to 32 cards; hidden and duplicate columns cost search time — [Power BI](https://learn.microsoft.com/en-us/power-bi/create-reports/service-insights); [Optimize](https://learn.microsoft.com/en-us/power-bi/create-reports/service-insights-optimize).
  Tableau Explain Data excludes dimensions with more than 500 unique values and recommends extracts over live sources
  because live sources generate many queries — [Tableau](https://help.tableau.com/current/pro/desktop/en-us/explain_data_requirements.htm).
  kepler.gl type inference, 50 sampled rows — [kepler.gl data-type.ts](https://github.com/keplergl/kepler.gl/blob/master/src/common-utils/src/data-type.ts).
  Felt geocoding, 10,000 rows per file free — [Felt](https://help.felt.com/upload-anything/files/spreadsheets-and-geocoding)

### Inferences
- A three-tier budget fits a GPU library:
  - Tier 0, metadata only, synchronous: geometry kind, column tags, row count. This decides availability and the rule rank
    in under a frame, as Show Me does.
  - Tier 1, cheap GPU statistics on a capped sample, streamed as they finish (Lux-style async): per-column min, max,
    distinct estimate and skew; global Moran's I on a sample; nearest-neighbour ratio. This refines the order and fills
    "because ..." text.
  - Tier 2, thumbnails: run the actual operation on a sample, such as 10–30k features (Lux's recall curve suggests about
    10% or 5–30k rows suffices for top-15 retrieval of simple scores), only for the top few suggestions and only when
    visible.
- Wrangler's diversity cap and Lux's k=15 per tab are reasonable starting numbers for the suggested row and per-family
  lists.

### Gaps
- No source measured sample-based preview fidelity for spatial operations, such as KDE or LISA computed on a subsample
  against the full data. Sample-based thumbnails of spatial statistics may differ from full results, because neighbour
  structure changes with sampling (an inference, unverified).
