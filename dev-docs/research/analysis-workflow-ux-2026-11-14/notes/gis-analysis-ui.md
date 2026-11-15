# Non-builder analysis UX in established GIS software (as of November 2026)

Scope: what the user sees and clicks when picking, configuring, running and reading one analysis in ArcGIS Pro, ArcGIS Online Map Viewer, ArcGIS Insights, QGIS, GeoDa, and (briefly) Whitebox, SAGA and GRASS. Visual model builders are out of scope; an earlier study covered them.

Source note: in 2026 Esri moved its documentation from `pro.arcgis.com` / `doc.arcgis.com` to `doc.esri.com`. Many old URLs now redirect, some to 404s. URLs below are the ones that loaded. Version-pinned pages (for example Pro 3.4) are used where the "latest" page did not load. Everything marked **unverified** comes from the researcher's prior knowledge and was not confirmed on a primary page in this session.

## ArcGIS Pro: Analysis ribbon, Geoprocessing pane, tool dialogs, results

### Takeaway
ArcGIS Pro gives the user several ways in: a ribbon gallery of common tools by category, a pane with Suggestions, Favorites and Recent lists, and search that can be semantic (not generative). The tool dialog has a fixed shape. Red asterisks mark required parameters, an "i" hover gives the description, and a separate Environments tab holds run-wide settings. A status banner after the run leads to View Details and History, and output is added to the map automatically. The patterns worth borrowing are Suggestions from the tools already run, Recent entries that reopen with their earlier parameters, and the capability badges on search results.

### Cited Findings
- The Geoprocessing pane **Favorites** tab has four lists. **Suggestions** shows "tools that may help with your next step", based on tools already run in the current project. **My Favorites** is stored in the user profile and appears in every project. **Project Favorites** is per project and includes several common tools by default. **Recent** lists recently run tools in chronological order — [Find a geoprocessing tool (latest)](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- Each Recent entry shows a status icon and a list of recent runs of that tool. Hovering shows details, and clicking reopens the tool "with the parameter settings that were previously used". A tool that is already a favorite is not repeated in Recent — [Find a geoprocessing tool (3.6)](https://pro.arcgis.com/en/pro-app/3.6/help/analysis/geoprocessing/basics/find-geoprocessing-tools.htm), as summarised by search results; the 3.2 page described separate Project Favorites and Recent categories — [3.2](https://pro.arcgis.com/en/pro-app/3.2/help/analysis/geoprocessing/basics/find-geoprocessing-tools.htm)
- The **Toolboxes** tab separates **System** tools (including extension and industry toolboxes) from custom **Project** toolboxes. The **Portal** tab lists portal-hosted tools, which also appear in a Portal group on the Analysis ribbon — [Find a geoprocessing tool](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- Search accepts a tool name or an operation word such as "proximity". It uses a **semantic search** model when that was enabled at install and the CPU supports AVX2; otherwise it falls back to traditional search. Esri states: "Semantic search is not an interactive AI application or chat interface and does not use generative AI technology." — [same page](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- Search results can be shown as **detailed tiles** or a list. Tiles carry icons for tool type (built-in, model, script), locked or unlicensed, **credit consumption**, portal execution, parallel processing, Spark, GPU and big-data-file-share support — [same page](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- The Analysis ribbon **Tools gallery** "highlights common spatial analysis tools by category". It shows a few by default and can be expanded or scrolled, and projects from templates can ship a customised gallery. A separate **Ready To Use Tools** gallery (Elevation: profile, viewshed; Hydrology: watershed, trace downstream) names outputs automatically, copies them to the project geodatabase and adds them to the map — [same page](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- Users can place any tool on any ribbon tab or on the Quick Access Toolbar. The setting is stored per user and applies to all projects — [same page](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- **Tool dialog** — [Use a geoprocessing tool (3.4)](https://pro.arcgis.com/en/pro-app/3.4/help/analysis/geoprocessing/basics/run-geoprocessing-tools.htm):
  - A **Parameters** tab, with required parameters marked by a red asterisk; optional parameters can be left at their defaults.
  - An **information icon** beside each parameter label shows its description on hover. The pane's Help button shows a tool summary on hover and opens the reference page on click.
  - Input can come from the layer's selection, highlight or filter, from a browse button, or from an **interactive input** button for drawing features.
  - Output defaults to the current or scratch workspace.
  - An **Environments** tab holds settings that apply only to that run.
  - The **Run** menu offers Reset Parameters, Schedule and Copy Python Command.
  - After a run a **status banner** appears with a progress bar, Cancel, a hover pop-up, **View Details** and **Open History**. Its icons are a check mark (success), a warning (completed with warnings) and an error (failure).
  - Successful output layers are added to the open map, and "Show notification when tool completes" is an option.
  - A Back button and an Add button ("Open Another Tool") let several tools stay open in the pane.
- Some tools open in a **floating window** with OK or Apply instead of the pane, and show a progress dialog with "Close upon completion" checked by default — [Find a geoprocessing tool](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geoprocessing/basics/find-geoprocessing-tools.html)
- **Unverified (not on a page fetched in this session):**
  - Parameters are grouped into collapsible **categories**, for example "Aggregation Parameters".
  - Invalid values show a red X icon with a hover message, and warnings show a yellow triangle. Validation runs as soon as a value is entered, through each tool's ToolValidator `updateParameters` and `updateMessages`, so later parameters (field lists, enabled or disabled options) update from earlier ones.
  - Each tool has a **Batch** button (a stacked-layers icon at the top right of the dialog). Batch mode lets the user pick one parameter to batch over and optionally fill it from an iterator or a wildcard.
  - Hot Spot Analysis (Gi*) output is drawn automatically with a seven-class diverging renderer on the Gi_Bin field: "Cold Spot – 99% / 95% / 90% Confidence", "Not Significant", and "Hot Spot – 90% / 95% / 99% Confidence", in blue, cream and red.
  - These need confirming on the [Hot Spot Analysis (Getis-Ord Gi*) page](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/hot-spot-analysis.html) and the "Batch geoprocessing" help page; the obvious doc.esri.com URLs for both returned 404.

### Inferences
- Pro shows suggestions **per project, from what the user has already run**, not from the input data's type. Suggesting from the data type would go further than Esri does in the desktop product.
- A Recent entry that reopens with its earlier parameters works as lightweight history, with no model builder needed. It maps directly onto a "rerun or tweak the last run" control in a catalog panel.
- Capability badges on search results (GPU, credits, big data) are a cheap way to surface cost and runtime. A WebGPU library could badge operations by expected cost or the input size they support.

### Gaps
- The exact validation icon styles, parameter categories, batch UI and Gi* default renderer could not be confirmed on a loaded 2026 page.
- No primary page was found describing how Suggestions are ranked.

## Esri "Optimized" and self-configuring tools: how they inspect the data and report their choices

### Takeaway
Optimized Hot Spot Analysis is the canonical example. It checks the data (bad records, minimum counts, locational outliers at 3 standard deviations), picks an aggregation and a cell size from nearest-neighbour statistics, and picks a distance band from the first peak of Incremental Spatial Autocorrelation, falling back to a K-neighbour distance. It then applies FDR correction, and it writes each decision to the run messages. Other tools tune themselves internally: EBK fits semivariograms by subsetting and simulation, and Forest-based offers an Optimize checkbox. Exploratory Regression reports its choices as summary tables in the messages.

### Cited Findings
- **Optimized Hot Spot Analysis, step 1, initial assessment** — [How Optimized Hot Spot Analysis works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/how-optimized-hot-spot-analysis-works.html):
  - Features with bad or missing geometry, or a null Analysis Field, are listed as bad records and excluded.
  - At least 30 features are needed for polygons, or for points with an Analysis Field.
  - Minimum incident counts before aggregation are 60 for a fishnet or hexagon grid without bounding polygons, 30 with them, 30 for aggregation polygons, and 60 for snap to weighted points. At least 30 features must remain after aggregation.
  - The Analysis Field must vary, and binary fields are rejected as unsuitable for Gi*.
  - **Locational outliers** are features whose distance to the nearest non-coincident neighbour is more than **three standard deviations** from the mean of those distances. Their count is reported.
- **Step 2, aggregation of incidents** — [same page](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/how-optimized-hot-spot-analysis-works.html):
  - Fishnet or hexagon grid:
    1. Collapse coincident points.
    2. Compare the data with random points in its minimum bounding polygon. If twice the random average nearest neighbour (ANN) distance is less than extent ÷ 100, the cell size is extent ÷ 100.
    3. Otherwise the cell size is 2 × max(ANN, median nearest neighbour distance), excluding outliers.
    4. Count incidents per cell. Zero-count cells are removed unless bounding polygons are given.
    5. Fewer than 30 cells, or identical counts, means the data is "unsuitable".
  - Snap to weighted points:
    1. Snap distance = min(ANN, MNN) × (unique locations ÷ N).
    2. Integrate in three passes: 0.10×, 0.25× and 1.0× that distance.
    3. Collapse the snapped points to weighted points.
- **Step 3, scale of analysis** — [same page](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/how-optimized-hot-spot-analysis-works.html):
  - It runs **Incremental Spatial Autocorrelation**, Global Moran's I over increasing distances, and takes the **first peak** in the z-score. Outliers are excluded when setting the start and increment distances.
  - If any feature has 500 or more neighbours, this step is skipped and the fallback is used with 30 neighbours.
  - Fallback: K = 0.05 × N, clamped to 3–30. The distance is the average distance to K neighbours, capped at one standard distance.
- **Step 4, FDR and output** — [same page](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/how-optimized-hot-spot-analysis-works.html):
  - Gi* runs with **FDR correction** for multiple testing and spatial dependence.
  - Messages summarise the number of significant hot and cold spots after FDR.
  - Output features carry a z-score, p-value, Gi Bin and neighbour count.
  - The messages report each decision in turn: bad records, outlier count, cell size and aggregation, the chosen distance, unsuitable-data warnings and FDR counts. The page does not give the exact message wording.
- **Empirical Bayesian Kriging** "automatically calculates these parameters through a process of subsetting and simulations":
  - The data is split into overlapping subsets (default 100 points each).
  - For each subset a semivariogram is estimated by REML, new data is simulated from it and re-estimated, giving a distribution of semivariograms (default 100 simulations).
  - Each prediction location blends the distributions of nearby subsets.
  - The user-facing knobs are subset size, overlap factor and number of simulations, each traded against processing time.
  - EBK does not underestimate standard errors the way a single fitted semivariogram does.
  - Source: [What is Empirical Bayesian Kriging? (10.1)](https://resources.arcgis.com/en/help/main/10.1/0031/0031000000q9000000.htm), an old version; the current Pro equivalent was not loaded.
- **Density-based Clustering**, Self-adjusting (HDBSCAN) option: it "uses varying distances, allowing for clusters with varying densities based on cluster probability (or stability)". The output adds a PROB field (membership likelihood) and an OUTLIER field, and the tool produces messages and charts reached from the Contents pane. Older docs tie the reachability plot to OPTICS, not HDBSCAN — [Density-based Clustering (3.4)](https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-statistics/densitybasedclustering.htm), [3.1](https://pro.arcgis.com/en/pro-app/3.1/tool-reference/spatial-statistics/densitybasedclustering.htm)
- **Forest-based and Boosted Classification and Regression**: the user checks **Optimize** and picks an **Optimization Model** option. The tool uses an adapted Breiman–Cutler random forest or XGBoost; gradient boosting includes regularisation and early stopping — [How Forest-based and Boosted ... works (3.3)](https://pro.arcgis.com/en/pro-app/3.3/tool-reference/spatial-statistics/how-forest-works.htm). The search method and the list of target metrics were not confirmed.
- **Exploratory Regression** writes a report with these parts — [Interpreting Exploratory Regression results](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/interpreting-exploratory-regression-results.html), [Pro 3.6](https://pro.arcgis.com/en/pro-app/3.6/tool-reference/spatial-statistics/interpreting-exploratory-regression-results.htm):
  - A **Summary of Variable Significance**: % of models in which each variable was significant, with % Negative and % Positive columns for a stable sign.
  - A **Summary of Multicollinearity**: how often each variable appeared in high-VIF models, and with which other variables.
  - A **Global Summary**: which diagnostic test is causing models to fail, which is where to start when none pass.
  - The listed passing models.
- **Unverified:**
  - Optimized Outlier Analysis is understood to use the same assessment, aggregation and scale steps, then run Anselin Local Moran's I with FDR. Its default permutation count was not confirmed.
  - The output of both Optimized tools is drawn automatically with confidence-class legends.
  - The page to confirm is [How Optimized Outlier Analysis works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/how-optimized-outlier-analysis-works.html) (not loaded).

### Inferences
- The "Optimized" pattern has a reproducible structure a GPU library can copy:
  1. Validate the data and report what was excluded, with counts.
  2. Pick an aggregation and a scale from cheap summary statistics (ANN, MNN, standard distance).
  3. Pick a neighbourhood from a parameter sweep (incremental Moran's I), with a fallback rule.
  4. Correct for multiple tests (FDR).
  5. Write a **plain-language log of every choice**. On a GPU the sweep in step 3 is cheap, so it could be shown as a chart instead of a single number.
- Esri reports these choices as **messages text**, not as an interactive explanation. A better design would render each chosen value as an editable chip ("distance band 1,240 m, chosen from the first Moran's I peak; change"). That chip turns the "optimized" tool back into the expert tool with those values filled in.

### Gaps
- The exact message strings are not published.
- The Optimized Outlier Analysis details and the Forest-based optimisation method and metrics are unconfirmed.
- The current Pro EBK page was not loaded; only the 10.1 version was.

## ArcGIS Online Map Viewer analysis pane, and ArcGIS Insights

### Takeaway
Map Viewer groups feature tools by the question they answer: Summarize data, Find locations, Analyze patterns, Use proximity, Enrich data and Manage data. It keeps a per-map **History** that can be shared, records parameters, messages and credits, and reopens any run's tool with its settings filled in; users can also **Estimate credits** before running. ArcGIS Insights was the strongest field-role-driven, drop-zone, "Find answers"-by-question design, with a recorded, re-runnable analysis model. **Esri has retired Insights**: 2025.1 docs said it would retire in 2026, and the latest docs say "ArcGIS Insights is retired". It is therefore a design reference only, not a live competitor.

### Cited Findings
- Map Viewer categories (Enterprise docs, which match Online) — [Perform analysis (Enterprise 11.x)](https://enterprise-k8s.arcgis.com/en/11.1/analyze/perform-analysis.htm), [Portal 11.3](https://enterprise.arcgis.com/en/portal/11.3/use/perform-analysis.htm):
  - **Summarize data**: totals, lengths, areas and statistics within areas or near features.
  - **Find locations**: site selection against several criteria, for example Find Existing Locations.
  - **Analyze patterns**: identify, measure and map patterns such as hot spots, for example Find Hot Spots.
  - **Use proximity**: "what is near what", for example buffers.
  - **Enrich data** and **Manage data**.
  - Esri notes the groupings "do not affect how you access or use" the tools.
- Analysis came to the new Map Viewer in February 2023, with 22 feature and 24 raster tools at launch. Users filter by analysis type or search by keyword, and raster tools need ArcGIS Image — [Introducing analysis in Map Viewer (Esri blog)](https://www.esri.com/arcgis-blog/products/arcgis-online/announcements/introducing-analysis-in-arcgis-online-map-viewer), [Perform analysis in Map Viewer](https://doc.arcgis.com/en/arcgis-online/analyze/perform-analysis-mv.htm) (summarised by search results; the page itself redirects to doc.esri.com and was not loaded)
- **Estimate credits** appears in the tool pane before Run. If credit budgeting is on and the estimate exceeds the user's allocation, an error is shown — same sources, from search summaries
- **Analysis History** — [Analysis history (Portal 11.5 help)](https://maps.phoenix.gov/pub/help/en/portal/11.5/use/analysis-history-mv.htm), a mirror of Esri's Portal help:
  - **Opening it:** Settings, then Analysis, then **History**.
  - **What it logs:** every tool or raster-function run, including failures.
  - **Each entry shows:**
    - status
    - parameters, including environments (some shown as raw values)
    - start and end time and duration
    - info, warning and error messages, plus credits used
    - run ID and links to the items the run created
  - **Actions:**
    - **Cancel**.
    - **View details**, a panel with Results, Messages and Parameters tabs. Results lets the user zoom to the output, open its table, add it to the map or open its item. Failed runs open on Messages.
    - **Open tool**, which reopens with the original settings; a new output name is needed, and layer filters are not reapplied.
    - **Remove**, which keeps the outputs.
  - **Saving:** history is saved only when the map is saved.
  - **Sharing:** history travels with the shared map, so viewers can inspect parameters. Viewers without the licence cannot rerun, and some parameters are reset.
- Map Viewer Classic offered **Rerun Analysis** on a result layer, which reopens the tool with its parameters filled in, and **Use current map extent** to limit the input and the credits used — [Use the analysis tools (Enterprise 11.5)](https://enterprise-k8s.arcgis.com/en/11.5/analyze/use-analysis-tools.htm)
- **Insights retirement**: "ArcGIS Insights is retired" (latest); "deprecated and will be retiring in 2026" (2025.1) — [Field types (latest)](https://doc.arcgis.com/en/insights/latest/get-started/field-roles.htm), [Field types (2025.1)](https://doc.arcgis.com/en/insights/2025.1/get-started/field-roles.htm), [Perform analysis (latest)](https://doc.arcgis.com/en/insights/latest/analyze/perform-analysis.htm)
- **Insights field roles** — [Field types](https://doc.arcgis.com/en/insights/2025.1/get-started/field-roles.htm):
  - The roles are string, number, rate/ratio, date/time and location.
  - Fields named zip, postal or year are treated as strings even when numeric.
  - Rate/ratio is assigned automatically to fields made by Calculate Ratio or Calculate % Change; existing percentage fields come in as number and can be switched by hand.
  - Date/time fields are split automatically into subfields usable as strings, and time series need a date/time field.
  - A map needs a location field, and **Enable location** adds one to a non-spatial table.
  - Allowed role changes: number to string or rate/ratio; rate/ratio to number or string; string to date/time. Date/time and location roles are fixed, and a role cannot change while the field is used in a card.
- **Insights drop zones** — [Get started with drag-and-drop (2025.1)](https://doc.arcgis.com/en/insights/2025.1/get-started/get-started-with-drag-and-drop.htm), [Create a map](https://doc.arcgis.com/en/insights/2025.1/create/create-maps.htm), [Learn ArcGIS: Investigate prescribed drugs](https://learn.arcgis.com/en/projects/investigate-prescribed-drugs):
  - Dragging a field to a blank area shows **Map / Chart / Table** drop zones. The resulting map style or chart type "depends on the field types you choose".
  - Dropping on Chart shows a chart-type pop-up, and dropping date plus a category on **Time Series** gives a time chart.
  - Dragging a field onto an existing card changes its style or adds the field.
  - **Map cards carry analysis drop zones**, **Spatial Aggregation** and **Spatial Filter**: drag a second dataset onto one and click Run.
  - Dragging a selection to a new card makes a result dataset.
- **Insights Analytics pane** — opened with the card's **Action** button — [Perform analysis (latest)](https://doc.arcgis.com/en/insights/latest/analyze/perform-analysis.htm):
  - **Spatial analysis** tab, map cards only: Buffer/Drive Times, Spatial Aggregation, Spatial Filter, Enrich Data, Calculate Density, Find Nearest, Calculate Density Ratio, Find K-Means Clusters, Find Spatial Mean.
  - **Find answers** tab, grouped by question:
    - **How is it distributed?** Spatial Aggregation, Calculate Density, View Histogram, Classification, View Heat Chart, View Box Plot, Calculate Z-Score, Find K-Means Clusters.
    - **How is it related?** Enrich Data, View Scatter Plot, Calculate Ratio, View Chord Diagram, View Link Chart, Regression Model, Predict Variable, Calculate Density Ratio.
    - **What's nearby?** Buffer/Drive Times, Attribute Filter, Spatial Filter, Find Nearest.
    - **How has it changed?** Time Series, Calculate % Change, View Data Clock, Temporal Decomposition, Forecast.
  - **Temporal analysis** tab, time series cards only: Moving Average, Temporal Decomposition, Forecast.
- Insights tools are constrained by role. For example, Calculate Ratio needs number or rate/ratio fields for both numerator and denominator, and writes a new rate/ratio field — [Calculate Ratio (latest)](https://doc.arcgis.com/en/insights/latest/analyze/calculate-ratio.htm)
- **Insights analysis model**:
  - The **analysis view** records each step automatically.
  - Publishing the page with **Type = Model** makes a shareable item, shared with the organisation, groups or the public.
  - Opening the model from the **Models** tab creates a workbook of **placeholder cards**. The user picks new datasets in the dataset pane and maps original fields to new ones with **Replace fields**, and the analysis then reruns.
  - Sources: [Rerun an analysis (2025.1)](https://doc.arcgis.com/en/insights/2025.1/analyze/automate-analysis-with-models.htm), [Publish and share a model (2024.1)](https://doc.arcgis.com/en/insights/2024.1/share/share-model.htm), [Get started with sharing (2025.1)](https://doc.arcgis.com/en/insights/2025.1/get-started/get-started-with-sharing.htm)

### Inferences
- Insights is the closest prior art for "suggest analyses from the input data type". Data roles gate which tools are offered, while the **question headings** organise them. Its retirement removes a competitor, not the idea. Its model rerun step (placeholder cards plus a Replace fields mapping) is a ready-made design for "run this curated workflow on my data".
- Map Viewer History is a good model for a per-map run log. It records parameters, messages and cost, survives with the saved map, and reopens a run's tool with its settings filled in. A Kepler-hosted app could store the same record in the map config.
- Question-based categories (Summarize, Find locations, Analyze patterns, Use proximity) suit roughly 20 to 50 tools. With about 280 operations, a second level or a data-type filter would be needed.

### Gaps
- The current (2026) doc.esri.com Map Viewer tool list did not load (blank page), so the full tool-per-category list and any 2025–2026 additions (for example AI assistants in Map Viewer) are unconfirmed.
- The exact Insights rule mapping roles and role combinations to card types was not found on a primary page.

## QGIS Processing Toolbox, algorithm dialogs, batch, history

### Takeaway
QGIS uses one searchable tree grouped by provider. Recently Used and Favorites sit at the top. An **Edit Features In-Place** toggle filters the list to algorithms that can modify the active layer, optionally only its selected features. Every algorithm dialog is generated with a Parameters and Log tab, a description panel on the right, and an **Advanced** menu that can copy the run as Python, qgis_process or JSON and paste JSON settings back. It can switch to batch mode, and after a run it offers **Change Parameters**.

### Cited Findings
- **Toolbox** — [The Toolbox (QGIS 3.44 manual)](https://docs.qgis.org/3.44/en/docs/user_manual/processing/toolbox.html):
  - **Toolbar:** create, open and add models and scripts; History; Results Viewer; the in-place toggle; Options.
  - **Search:** filters algorithms, models and scripts as you type, by name or keyword.
  - **Recently Used** is on by default and tracks runs. **Favorites** is built by hand (right-click, "Add to Favorites") and does not change with use.
  - The right-click menu also offers Execute, **Execute as Batch Process** and "Edit Rendering Styles for Outputs".
  - Algorithms are grouped by **provider**. By default only providers that need no third-party apps are active, and others are enabled in Processing settings.
  - **Edit Features In-Place** mode shows only algorithms that can run on the active layer without creating a new layer, and for vectors can process only the selected features.
- **Algorithm dialog** — [same page](https://docs.qgis.org/3.44/en/docs/user_manual/processing/toolbox.html):
  - **Parameters** and **Log** tabs on the left, the algorithm description on the right, and a Help button to the documentation.
  - Parameter widgets are generated by type: layer, raster, numeric with a **data-defined override**, extent, CRS, field and so on.
  - The **Advanced** menu has Algorithm Settings, Copy as Python Command, Copy as qgis_process Command, Copy as JSON and Paste Settings (JSON).
  - **Run as Batch Process** and **Run as Single Process** switch modes.
  - After a run, **Change Parameters** appears while the Log tab is active, and output file names in the log are clickable.
  - A mismatched CRS between inputs gives a warning, but the algorithm can still run.
- An older manual (2.2) showed recently used and favorite tools at the top of the list — [The Toolbox (2.2)](https://docs.qgis.org/2.2/fi/docs/user_manual/processing/toolbox.html)
- **Unverified (not loaded):**
  - The batch table's per-column **Autofill** menu has Fill Down, Calculate by Expression and Add Values by Expression.
  - The **History** panel lists past runs with their Python or JSON commands, and double-clicking one reopens the dialog with those values.
  - Sources to check: [Batch processing (3.44)](https://docs.qgis.org/3.44/en/docs/user_manual/processing/batch.html), [History (3.44)](https://docs.qgis.org/3.44/en/docs/user_manual/processing/history.html).

### Inferences
- In-place mode is a working example of **filtering by data type**. The active layer's geometry type and editability decide what is listed. The same filter could drive "suggested for this dataset" in a catalog panel.
- Copy as JSON and Paste Settings give a cheap way to save, share and replay a configured run without a model builder. They could hold the curated workflow presets.

### Gaps
- No primary page was found on how search ranks results (tags, short help).
- Batch and History details are unverified, as listed above.

## GeoDa (and web descendants): ESDA organisation

### Takeaway
In GeoDa, spatial weights are a first-class object. The user builds them once in the **Weights Manager** (default queen contiguity, first order), inspects them (connectivity histogram, map and graph), and every spatial analysis then selects them from a dropdown. Each analysis opens linked windows. The **Options** menu on each result holds Randomization (default 999 permutations), a Significance Filter (0.05, 0.01, 0.001, 0.0001, plus Bonferroni and FDR), Save Results and Conditional Map. Selection is brushed across all windows.

### Cited Findings
- **Opening local Moran:** from the Cluster Maps toolbar icon or **Space > Univariate Local Moran's I** — [GeoDa workbook: Local spatial autocorrelation (1)](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html)
- **Variable Settings dialog:** lists the variables, with the **current default weights file** shown at the bottom (for example guerry_85_q). A follow-up window-choice dialog defaults to **Cluster Map** only, with optional Significance Map and Moran Scatter Plot — [same page](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html)
- **Options menu** on the cluster or significance map, in order — [same page](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html):
  1. **Randomization**: 999 permutations by default, up to 99999, with a settable seed for replication.
  2. **Significance Filter**: preset cut-offs of 0.05, 0.01, 0.001 and 0.0001, plus a **Custom Inference** item with Bonferroni and FDR.
  3. **Save Results**: LISA indices, cluster codes and pseudo p-values. Cluster codes are 0 not significant, 1 High-High, 2 Low-Low, 3 Low-High and 4 High-Low; the table must be saved to make them permanent.
  4. **Show As Conditional Map**: for example a 2×2 micromap grid at medians.
- **Legends:** the significance map uses progressively darker greens as significance increases. The cluster map colours are High-High dark red, Low-Low dark blue, Low-High light blue and High-Low light red. The workbook stresses that the pseudo p-value is "not a standard p-value" — [same page](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html). **Unverified:** the cluster legend also lists "Not Significant" (grey) and "Neighborless" categories with counts.
- **Linked brushing:** selecting points in the Moran scatter plot highlights them in the cluster map and the reverse. Clicking a legend category such as High-High selects those cores, and the user can choose to select cores, neighbours or both — [same page](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html)
- **Weights Manager:** opened from the Tools menu; queen contiguity, first order is the default, and rook is in the same dialog — [GeoDa workbook: Contiguity-based spatial weights](https://geodacenter.github.io/workbook/4a_contig_weights/lab4a.html). It has an "Order of contiguity" field and an "include lower orders" checkbox — [Anselin, Intro to Spatial Data Science vol. 1, 10.3](https://lanselin.github.io/introbook_vol1/creating-contiguity-weigths.html). The bottom of the Weights Manager offers **Histogram**, **Connectivity Map** and **Connectivity Graph** to inspect the weights — [same book, 10.4](https://lanselin.github.io/introbook_vol1/weightscharacteristics.html)
- **Web and library descendants** expose the same weights-first API: pygeoda, rgeoda, jsgeoda (WebAssembly) and GeoDaLib from GeoDa AI, each with functions such as `queenWeights` — [pygeoda spatial weights](https://geodacenter.github.io/pygeoda/_sources/spatial_weights.rst.txt), [rgeoda gda_queen_weights](https://search.r-project.org/CRAN/refmans/rgeoda/html/gda_queen_weights.html), [jsgeoda spatial weights](https://xunli.gitbook.io/jsgeoda/spatial-weights), [GeoDaLib queenWeights](https://geodaai.github.io/geodalib/api/core/src/functions/queenWeights.html). The UI of GeoDa's web or AI front ends (for example any chat-driven GeoDa) was not investigated.

### Inferences
- A weights object that the user builds, names and inspects once, then picks from a dropdown with the current default preselected, avoids repeating neighbourhood settings in every form across many spatial-statistics operations. It also makes the neighbour structure visible, through the connectivity histogram, before any statistic depends on it.
- Putting post-hoc controls (significance cut-off, permutation count, FDR) in the result's **Options** menu rather than the input form means the user adjusts inference **after** seeing the map. On a GPU, re-permuting at 9999 is quick enough to be interactive.

### Gaps
- The exact current GeoDa legend labels (Not Significant, Neighborless, counts in parentheses) were not confirmed on a fetched page.
- The bivariate and multivariate menu entries (Space > Bivariate Local Moran's I, Local Geary, Local Join Count) were not individually verified.

## Whitebox, SAGA and GRASS: making hundreds of tools discoverable

### Takeaway
None of the three has a distinctive discovery UX. Each mostly reaches users through QGIS Processing providers or scripting. The one notable 2025–2026 development is a natural-language "Whitebox AI Agent" QGIS plugin that finds tools in the Processing registry.

### Cited Findings
- WhiteboxTools is usually reached from QGIS through Alexander Bruy's **Whitebox for Processing** provider — [WhiteboxTools manual: QGIS plugin](https://jblindsay.github.io/wbt_book/qgis_plugin.html). A separate "WhiteboxTools for QGIS" plugin does not bundle the binary — [QGIS plugins: wbt_for_qgis](https://plugins.qgis.org/plugins/wbt_for_qgis). A "Whitebox Workflows for QGIS" plugin also exists (version 2.1.0 listed) — [QGIS plugins](https://plugins.qgis.org/plugins/whitebox_workflows_for_qgis/version/2.1.0/)
- **WhiteboxTools Runner** is a standalone Python GUI that the project calls an example of a more elaborate interface for the library — [WhiteboxTools project page](https://jblindsay.github.io/ghrg/WhiteboxTools)
- **Whitebox AI Agent** is a QGIS plugin that runs tools from natural language and "automatically discovers algorithms from the QGIS Processing registry" — [QGIS plugins: whitebox_agent](https://plugins.qgis.org/plugins/whitebox_agent). Its maturity and authorship were not evaluated.
- **Unverified:**
  - GRASS has keyword search with `g.search.modules` and module keywords.
  - The GRASS GUI has a searchable Tools tree in its Modules tab.
  - SAGA groups hundreds of tools in a library and module tree.
  - None was confirmed on a primary page in this session.

### Inferences
- With hundreds of tools, the common answer is a provider tree plus text search, and the pain point is that names are written for experts. Esri's semantic search and the Whitebox agent both respond to that by matching the user's intent rather than the tool's name.

### Gaps
- No primary GRASS or SAGA discovery documentation was loaded.

## Published decision aids mapping a question or data type to a tool

### Takeaway
Esri publishes structured decision aids, though fewer than commonly assumed:
- toolset-level "what it does" tables, with question headings only in Measuring Geographic Distributions;
- the **Geostatistical Analyst classification trees** for choosing an interpolator;
- Insights' four **Find answers** questions;
- Map Viewer's question-named categories.

A single "choose the right clustering tool" comparison table could not be found on a primary page.

### Cited Findings
- **Spatial Statistics toolbox overview** — [An overview of the Spatial Statistics toolbox](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/an-overview-of-the-spatial-statistics-toolbox.html):
  - Toolsets, each with a one-line purpose:
    - **Analyzing Patterns**: clustered, dispersed or random.
    - **Assessing Sensitivity**: compare results with simulated data.
    - **Mapping Clusters**: hot spots, cold spots and outliers, and grouping similar features.
    - **Measuring Geographic Distributions**: center, shape, orientation, dispersion.
    - **Modeling Spatial Relationships**: regression and spatial weights.
    - **Spatial Component Utilities (Moran Eigenvectors)**: create components, filter autocorrelation, recommend neighbourhoods.
    - **Utilities**.
  - The only exact question headings are "Where's the center?", "What's the shape and orientation?" and "How dispersed are the features?".
  - There is no question-to-tool table on the page.
- **Interpolation decision structure** — [Classification trees of the interpolation methods offered in Geostatistical Analyst (Pro latest)](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geostatistical-analyst/classification-trees-of-the-interpolation-methods-offered-in-geostatistical-analyst.html), [Pro 3.3](https://pro.arcgis.com/en/pro-app/3.3/help/analysis/geostatistical-analyst/classification-trees-of-the-interpolation-methods-offered-in-geostatistical-analyst.htm):
  - The process: define the objective, pick a criterion, follow its tree to candidate methods, use several trees, then compare the shortlisted methods.
  - The trees cover:
    - type of output (predictions only, or predictions with errors)
    - whether a spatial autocorrelation model is needed (extra parameters and interactive fitting)
    - exact or inexact
    - smoothness (RBF "smooth by construction")
    - uncertainty
    - speed ("relatively fast except when barriers are used")
  - The methods covered: GPI, LPI, IDW, RBF, KSB, DKB, kriging variants (ordinary, simple, universal, indicator, probability, disjunctive, EBK, EBK regression prediction) and Gaussian geostatistical simulation.
  - The branch membership is drawn only in images and was not extracted.
- **Clustering choice:** Esri's guidance is spread across pages, not set out in one table:
  - Multivariate Clustering is attribute-only, with K-means (the default) or K-medoids.
  - Spatially Constrained Multivariate Clustering enforces contiguity and optional size limits, and Esri recommends the unconstrained tool when proximity should not be imposed.
  - Density-based Clustering is for point patterns.
  - Esri's blog advice is to "experiment to see which method best fits your analysis".
  - Sources: [Multivariate Clustering](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/multivariate-clustering.html), [Spatially Constrained Multivariate Clustering](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/spatially-constrained-multivariate-clustering.html), [Mapping Clusters toolset overview](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/an-overview-of-the-mapping-clusters-toolset.html), [Esri blog](https://esri.com/arcgis-blog/?p=82111).
  - The Esri Press book *Spatial Statistics Illustrated* has a section "What's the 'right' statistical clustering method?" — [TOC PDF](https://www.esri.com/content/dam/esrisites/en-us/esri-press/browse/spatial-statistics-toc.pdf); its content was not accessed.
- **Insights Find answers**: How is it distributed? / How is it related? / What's nearby? / How has it changed? (tool lists in the Map Viewer and Insights section above) — [Perform analysis (Insights)](https://doc.arcgis.com/en/insights/latest/analyze/perform-analysis.htm)
- **Map Viewer categories**: Summarize data / Find locations / Analyze patterns / Use proximity / Enrich data / Manage data — [Perform analysis (Enterprise)](https://enterprise-k8s.arcgis.com/en/11.1/analyze/perform-analysis.htm)
- **Exploratory Regression** works as an in-tool decision aid. It tries combinations of candidate variables and reports which pass the OLS diagnostics — [Interpreting Exploratory Regression results](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/interpreting-exploratory-regression-results.html)
- **Unverified:** a "Spatial Statistics: questions you can ask" style page (for example "Modeling spatial relationships: what are the factors?") and an "Analyze patterns: which tool?" table. These may exist in the Spatial Statistics Resources pages or Learn ArcGIS but were not found.

### Inferences
- The decision aids that hold up best combine **a question** (distributed, related, nearby, changed) with **data constraints** (field role, geometry type, whether errors or barriers are needed). Insights joined the two in its UI. Esri's interpolation trees join them in the documentation only, as images.
- A designer could make the interpolation trees an interactive filter ("I need standard errors" / "must honour sample values" / "barriers present") that narrows the interpolation operations.

### Gaps
- No machine-readable version of Esri's interpolation tree branches was found.
- No single primary Esri clustering comparison table was found.
- The 2024–2026 release notes for Pro 3.4–3.7 and for Map Viewer were not reviewed for new analysis UX (for example AI assistants or suggested tools), so any 2026 additions are unconfirmed.
