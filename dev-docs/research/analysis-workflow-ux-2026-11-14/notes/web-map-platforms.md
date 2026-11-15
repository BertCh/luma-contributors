# Analysis UX in web map and geospatial analytics platforms (as of 2026-11-14)

Scope: how peer web products let users run spatial analysis in the browser, with concrete UI details, and which patterns suit a client-side WebGPU analysis library (about 280 operations) hosted in a Kepler.gl 3.x app. Research was run on 2026-11-14 with web search and page fetches. Fetches were summarised by a helper model, so the wording is paraphrased unless shown in quotes. "Unverified" means the claim did not come from a primary page fetched in this session.

## 1. kepler.gl (3.0 to the current 3.3 alpha): native analysis, layer inference, AI assistant, roadmap

### Takeaway
kepler.gl has almost no form-based analysis toolbox. Its analysis comes from four places: automatic layers built from column-name and type heuristics, aggregation layers, filters, a DuckDB SQL panel, and a chat AI assistant whose GeoDa/OpenAssistant tools (buffer, spatial join, weights, LISA, regression, classification, routing and isochrones) save their outputs as new datasets. The 3.3 alpha line, current in November 2026, adds suitability analysis, dataset operations for geometry and an experimental `@kepler.gl/mcp` package. No published roadmap was found.

### Cited Findings
**Layer auto-detection, the exact heuristics (from source on `master`)**
- Point pairs. `findPointFieldPairs(fields)` lowercases every field name. For each suffix pair in `TRIP_POINT_FIELDS` it matches the regex `(^|[#_&@\.\-\ ])lat([#_&@\.\-\ ]|$)`, where the separator class is `[#_&@\.\-\ ]`. It then looks for a partner field by swapping in the second suffix, plus an optional altitude field from `ALTITUDE_FIELDS`. The default layer name is the field name with the suffix removed, or `'point'` if nothing is left. — [kepler-table.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/table/src/kepler-table.ts)
- `TRIP_POINT_FIELDS = [['lat','lng'],['lat','lon'],['lat','long'],['latitude','longitude']]` and `ALTITUDE_FIELDS = ['alt','altitude']`. `GEOJSON_FIELDS = { geojson: ['_geojson','all_points','geojson'] }`, `ICON_FIELDS = { icon: ['icon'] }`, and `TRIP_ARC_FIELDS = { lat0:'begintrip', lng0:'begintrip', lat1:'dropoff', lng1:'dropoff' }`. `DEFAULT_LAYER_COLOR` gives fixed colours to `begintrip_lat`, `dropoff_lat`, `request_lat` and `tripArc`, which are Uber legacy names. — [default-settings.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/constants/src/default-settings.ts)
- `fieldPairs` is computed only in `updateSchema`, which runs when the schema changes, not on every row update. — [kepler-table.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/table/src/kepler-table.ts)
- `PointLayer.findDefaultLayerProps` makes one point layer per lat/lng pair. Only the first is visible by default. It names the layer from the pair's `defaultName`, or from the dataset label without its file extension, and uses `DEFAULT_LAYER_COLOR` when the lat field name matches. It also returns `altProps` for GeoArrow point columns (`isGeoArrowPointField`), and vector-tile datasets skip this logic. — [point-layer.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/layers/src/point-layer/point-layer.ts)
- H3 detection is by type, not by name. `getHexFields` keeps fields where `field.type === ALL_FIELD_TYPES.h3`, and the H3 layer creates one layer per such field, mapped to its `hex_id` column. — [h3-utils.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/common-utils/src/h3-utils.ts), [h3-hexagon-layer.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/layers/src/h3-hexagon-layer/h3-hexagon-layer.ts)
- The H3 type comes from sampling in `validateInputData`. For string fields it checks the first 10 non-empty values with `h3IsValid`, using the expression `sample.every(item => !h3IsValid(item))`. The helper summary of which branch assigns the `h3` type was ambiguous, so whether every sampled value must be valid is unverified. — [dataset-utils.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/utils/src/dataset-utils.ts)
- Colour-by defaults leave out ID and coordinate columns. `EXCLUDED_DEFAULT_FIELDS` holds `'zip','code','post','region','fips','cbgs','h3','s2','a5','geohash','lat','lon','lng','latitude','longitude','_x','_y'`, matched with startsWith or endsWith on the lowercased name, and a `METRIC_DEFAULT_FIELDS` match overrides it. — [dataset-utils.ts](https://raw.githubusercontent.com/keplergl/kepler.gl/master/src/utils/src/dataset-utils.ts)
- Recent detection changes: 3.3.0-alpha.20 (8 Nov) stops duplicate automatic point and icon layers and improves US slash-date inference. 3.3.0-alpha.21 (13 Nov) prefers a detected Flow Field over duplicate automatic points and tightens date inference. — [kepler.gl releases](https://github.com/keplergl/kepler.gl/releases)

**Native analysis and SQL**
- The release notes describe a DuckDB SQL explorer for writing queries and mapping the results, and a Vector Tile layer for MVT and PMTiles. The DuckDB plugin landed in the 3.1.0 alphas (January 2025). — [release notes](https://docs.kepler.gl/release-notes), [changelog](https://docs.kepler.gl/changelog)
- The 3.3.0-alpha series (September to November; the releases page shows no year, so 2026 is inferred):
  - alpha.13: extract rows from a drawn polygon.
  - alpha.14: optional charts panel, and JSON editors for layers and filters.
  - alpha.16: dataset operations as a base for geometry operations, and a layer context menu.
  - alpha.17: suitability analysis; column rename, delete and filter in the data table; multi-dataset filters; changeable column types; export to GeoJSON, GeoParquet, Shapefile, GeoPackage and KML.
  - alpha.19: experimental `@kepler.gl/mcp` package with a WebMCP/MCP bridge in the demo app, and an experimental Zarr raster layer.
  - alpha.20: Voronoi Mosaic layer, and 3D density surfaces for heatmaps.
  - alpha.21: deck.gl upgraded to 9.4, clustered donut layer, 3D contour surfaces, and SQL workspace tests.
  — [kepler.gl releases](https://github.com/keplergl/kepler.gl/releases)

**AI assistant (`@kepler.gl/ai-assistant`, built on OpenAssistant)**
- The README for kepler.gl@3.2.5 lists these tool groups. Tools marked "coming soon" are Vega-Lite plots and "data by rates". — [README 3.2.5](https://app.unpkg.com/kepler.gl@3.2.5/files/src/ai-assistant/README.md)
  - kepler.gl tools: basemap, mapBoundary, addLayer, updateLayerColor, loadData, saveData.
  - eCharts plots: boxplot, bubble chart, histogram, parallel coordinates, scatterplot.
  - Query tools: genericQuery (OpenAssistant DuckDB) and filterDataset.
  - OSM tools: geocoding, isochrone, routing, roads, and US county, state and zipcode lookups.
  - GeoDa geo tools: area, buffer, centroid, dissolve, length, perimeter.
  - Classification: quantile, natural breaks, equal interval, percentile, box, standard deviation, unique values.
  - spatialJoin and spatialFilter.
  - Weights: queen and rook contiguity, k-nearest neighbours, distance band.
  - Global Moran's I, and local Moran, Geary, Gi* and quantile LISA.
  - Regression: OLS with spatial diagnostics, spatial lag, spatial error.
- Results come back as new datasets through `saveData` (for example zipcode areas, spatial join output or Local Moran output) and as layers through `addLayer`. There is no form; the chat is the user interface. — [README 3.2.5](https://app.unpkg.com/kepler.gl@3.2.5/files/src/ai-assistant/README.md)
- The 3.1.8 README listed spatial join only as planned, so tool coverage grew between 3.1 and 3.2. — [README 3.1.8](https://unpkg.com/kepler.gl@3.1.8/src/ai-assistant/README.md)
- The GeoDa tool package also offers local G and G* and quantile LISA. — [@openassistant/geoda README](https://cdn.jsdelivr.net/npm/@openassistant/geoda@1.0.0-alpha.0/README.md), [OpenAssistant tools guide](https://geodaai.github.io/openassistant/guide/tools/geoda.html)
- A tutorial on running spatial analysis through the AI assistant exists, and it was announced in kepler.gl discussion #2843. — [GeoDa AI guide](https://geodaai.github.io/docs-kepler-ai/guide.html)
- Foursquare positioned kepler.gl 3.1 as bringing "enterprise-grade spatial analytics" to the browser. — [Foursquare blog](https://foursquare.com/resources/blog/products/foursquare-brings-enterprise-grade-spatial-analytics-to-your-browser-with-kepler-gl-3-1/)

### Inferences
- kepler.gl's inference is name-regex based for points, type-sniffed for H3, and name or type based for GeoJSON. It suggests layers only, never analyses. A data-type-to-analysis suggester in the host app would be new territory rather than a copy of kepler.gl, but it can reuse kepler.gl's `fieldPairs`, field `type` (h3, geojson, geoarrow, timestamp) and `EXCLUDED_DEFAULT_FIELDS`-style rules as its input signals.
- The AI assistant's pattern, a tool call that writes a new dataset and a layer, matches what the library's operations already produce. Each of the 280 operations could be exposed as an OpenAssistant-style tool, or through the new `@kepler.gl/mcp` bridge, as well as through the catalog form.

### Gaps
- No public kepler.gl roadmap document was found. The releases page has none, and GitHub Projects were not checked.
- The exact branch in `validateInputData` that assigns the `h3` type was not confirmed verbatim.
- What "suitability analysis" (alpha.17) looks like in the UI was not examined.
- The GeoJSON and WKT string-sniffing heuristics in the type analyser were not fetched.

## 2. CARTO Builder and CARTO Workflows

### Takeaway
CARTO splits analysis into three surfaces. Builder offers light SQL analyses attached to a source. Workflows is a left-to-right canvas, reached from a gallery of about 61 templates in 13 categories, with a bottom results panel (Messages, Data, Map, SQL). In 2026 everything is also exposed to agents: a workflow becomes an MCP tool by adding an "MCP Tool Output" node and describing its variables, and Builder AI Agents can call those tools.

### Cited Findings
**Builder analysis**
- In Builder you open a data source's three-dot menu and choose "Add SQL Analysis". You see the analyses compatible with that source, they can be stacked, and "Create table from query" saves the result as a table. — [CARTO tutorial](https://docs.carto.com/~/changes/PgVqO5KPaIsKNkRCprIQ/carto-user-manual/tutorials/analysing-urban-areas-affected-by-earthquakes)
- "Add source from" offers "Custom Query (SQL)", which lets you write SQL or run a template SQL analysis. CARTO says the SQL editor "is not designed for conducting complex analysis" because Builder runs a query for each tile, and recommends Workflows for multi-step work. — [Running queries from Builder](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/guides/running-queries-from-builder)
- SQL Parameters turn user-defined analysis into dashboard controls, and since 2 July 2026 text parameters can be single-value radio lists. — [CARTO Academy](https://academy.carto.com/building-interactive-maps/data-analysis/create-a-dashboard-with-user-defined-analysis-using-sql-parameters), [What's new Q3 2026](https://docs.carto.com/whats-new/q3-2026)
- Since 16 July 2026, clicking a feature filters the map. A clicked polygon, H3 or Quadbin cell can act as a spatial mask across intersecting layers, and the filter propagates to widgets and exports. — [What's new Q3 2026](https://docs.carto.com/whats-new/q3-2026)
- The full list of Builder SQL Analyses could not be read, because the fetched page served the Workflows manual instead. Unverified.

**Workflows canvas**
- The canvas is free-form with snap to grid and a minimap, and runs as a directed graph from left to right. The left panel lists the connection's data sources and the components, with search and scroll. The bottom results panel has Messages, Data (table output of the selected node), Map (the selected node's geometry, with a chooser if there are several) and SQL (the compiled SQL, including procedural code). Intermediate results go to a `workflows_temp` schema so each step can be inspected. — [Workflows manual](https://docs.carto.com/carto-user-manual/workflows)
- Q3 2026 additions:
  - Export in CSV, GeoJSON, Shapefile, GeoParquet, GeoPackage or KML (10 September).
  - "Run MCP Tool test" from the canvas, with streamed steps and JSON results with row counts (24 July).
  - New Import/Export components, Export as File, Export Tables as PDF, and Send by Email (20 July).
  - Analytics on Embeddings components (29 July, private beta).
  — [What's new Q3 2026](https://docs.carto.com/whats-new/q3-2026)

**Templates gallery**
- You open the gallery with "New Workflow" then "From template". After the template is re-created you adapt it by "replacing the data sources and re-configuring the different nodes". — [Workflows manual](https://docs.carto.com/carto-user-manual/workflows), [Academy templates](https://academy.carto.com/creating-workflows/workflow-templates)
- Counts from the Academy list: 61 templates in 13 categories, 52 core and 9 that need extension packages. — [Academy templates](https://academy.carto.com/creating-workflows/workflow-templates)
  - Data Preparation (8): joins, unions, geography filtering, calculated columns, normalisation, ranking, column filtering.
  - Data Enrichment (5).
  - Spatial Indexes (4): H3 polyfill, Quadbin and H3 aggregation, K-rings.
  - Spatial Analysis (5): k-means, point-in-polygon, aggregation into polygons, Voronoi, custom SQL.
  - Generating new spatial data (5): custom geographies, routes, geocoding, lat/long points, isochrones.
  - Statistics (9): hotspots, space-time, time-series clustering, autocorrelation, GWR, composite scores, space-time anomaly detection.
  - Retail and CPG (2), Telco (8), Insurance (5), Out of Home Advertising (1).
  - Extension packages: BigQuery ML (4), Snowflake ML (2), Territory Planning (3).
- The launch post (15 February 2024) says templates are "pre-loaded with data", run as-is or can be modified, and work on any connected warehouse. Direct access from Workspace was added in July 2024. — [CARTO blog](https://carto.com/blog/workflows-templates-pre-built-spatial-analysis/)

**AI Agents and Workflows as MCP tools**
- Making a workflow into an MCP tool:
  1. Use one workflow per task.
  2. Add an **MCP Tool Output** component and choose Sync or Async. Redshift and Oracle are always Async.
  3. Define the inputs as workflow variables, each with a name and description. Geo variables become string parameters that carry a GeoJSON FeatureCollection.
  4. Write the tool description in the MCP Tool dialog and enable it.
  5. Sync after edits.
  6. Optionally turn on "Reuse output for repeated calls", which caches results for identical parameters.
  "Run MCP Tool test" pre-fills inputs from the Variables panel. — [Workflows as MCP tools](https://docs.carto.com/carto-user-manual/workflows/workflows-as-mcp-tools)
- Builder AI Agents can be given Workflow MCP tools and then get extra output tools for fetching results. Admins enable agents under Settings > CARTO AI. The CARTO MCP Server, for external hosts such as Claude, ChatGPT and Gemini, is a separate product surface. — [Working with tools](https://docs.carto.com/carto-user-manual/ai-agents/working-with-tools), [CARTO for Agents](https://docs.carto.com/carto-for-agents/carto-for-agents)
- Since 26 August 2026 the MCP Server exposes every CARTO capability in four groups: Maps (create, validate, render), Workflows (create, run, schedule), Data (SQL, geocode, routes, isochrones) and Workspace. Calls use the user's own credentials, so row-level security still applies. — [What's new Q3 2026](https://docs.carto.com/whats-new/q3-2026)
- Since 21 July 2026, semantic models can be attached to AI Agents. They describe what tables and columns mean, metrics and spatial structure, are based on "Apache Ossie" with a CARTO geospatial extension, and can be generated by an Agent Config Assistant. — [What's new Q3 2026](https://docs.carto.com/whats-new/q3-2026)

### Inferences
- The CARTO template model is the same graph with sample data bound to its sources. "Adapting" means rebinding sources and editing nodes, and there is no guided wizard for matching columns. A client-side app can do better: when a template is opened, offer a column-mapping step that checks the user's dataset against the template's typed input slots.
- Turning a workflow into a tool means declaring its inputs as named, described variables and giving it one output node. That maps directly onto a typed workflow format, and the same file could drive a form, a template card and an LLM tool schema.
- The bottom Data/Map/SQL panel for whichever node is selected is the most borrowable canvas detail. For a GPU library the "SQL" tab would become a "parameters and code" or "provenance" tab.

### Gaps
- The Builder SQL Analyses list and its form layout were not retrieved.
- The total number of Workflows components and their category names were not found. The manual page does not list them.
- What an individual template page contains (screenshots, downloadable files) was not confirmed.

## 3. Other hosted products: Felt, Atlas, Foursquare, Fused, Dekart, Wherobots, Mapbox, Honeycomb, Earth Engine, Planetary Computer, UP42

### Takeaway
Hosted products have split into two camps. Form toolboxes (Felt Analysis, Atlas's 50+ tools, Foursquare Studio) are being overtaken by chat agents that choose operations and write SQL (Felt AI, Atlas Navi, FSQ Spatial Agent, Earth Engine's Gemini panel, Wherobots MCP, Dekart GeoSQL). Canvas-style products (Fused, UP42) are built around code or processing blocks. None of the sources fetched documented analysis suggestions derived from the dataset's columns.

### Cited Findings
**Felt**
- The toolbar **Analysis** tool offers Buffer, Bounds, Centroid, Dissolve, Clip, Intersect, Subtract, Count points, Join (exact key match), Sample raster and Travel time. — [Felt help](https://help.felt.com/layers/spatial-analysis)
- The flow: pick an operation, then an Analysis panel opens where you choose the input layer (and a second layer if needed), set Settings, optionally set "Apply to: Features on screen", and click Apply. The result is a new layer in the legend, and "The panel includes a preview of the operation." — [Felt help](https://help.felt.com/layers/spatial-analysis)
- The tool is also reachable from the Felt menu and with Cmd/Ctrl+K. — [Felt help](https://help.felt.com/layers/spatial-analysis)
- Limits:
  - "Analysis results are created once and don't update when an input layer refreshes."
  - Analysis uses only features that pass the layer's filters.
  - "Features on screen" is the default for layers over 500 MB.
  - Travel time allows up to 120 minutes, 5 travel times per run and 10,000 areas.
  - Analysis needs a Team plan or higher.
  — [Felt help](https://help.felt.com/layers/spatial-analysis)
- Felt AI covers what the toolbox cannot: nearest feature, summing values in polygons, appending layers, and SQL on map layers. — [Felt help](https://help.felt.com/layers/spatial-analysis)
- Felt AI (post of 2 June 2026) works out the spatial operations a question needs ("proximity, overlay, spatial joins, aggregations"), writes SQL against connected warehouses or the Global Library, and keeps context across turns. It outputs live shareable maps, can build extensions and dashboard components, and "Nobody needs to pre-build a workflow or template before the agent can answer a new question." The post does not say whether the generated SQL is shown. — [Felt blog](https://felt.com/blog/a-brand-new-era-of-gis)
- A July 2026 post on multi-criteria site selection claims results "up to six times faster" (vendor claim). — [Felt blog](https://felt.com/blog/multi-criteria-site-selection-used-to-take-days-felt-ai-makes-it-instant)
- Felt and Wherobots announced a partnership in February 2026. — [GeoWeek News](https://www.geoweeknews.com/news/wherobots-and-felt-announce-strategic-partnership-to-deliver-the-ai-ready-spatial-data-stack)

**Atlas (atlas.co)**
- Atlas markets "50+ spatial analysis tools" for vector and raster, aimed at users without GIS training. — [Atlas spatial analysis](https://atlas.co/product/spatial-analysis/)
- Navi is an AI assistant that "builds maps from conversation" and can generate filters, attribute expressions or analysis workflows. — [Atlas Navi blog](https://atlas.co/blog/introducing-navi-ai-that-builds-maps-from-conversation/), [Atlas features](https://atlas.co/features/)
- The tool-panel UI was not examined in detail. Unverified.

**Foursquare**
- FSQ Spatial Agent is a "multi-agent orchestration system". It runs complete workflows from choosing datasets through spatial joins to visualisation, and "exposes its reasoning at every step", showing the datasets it considered, the operations it planned and how it read the results. — [Foursquare blog](https://foursquare.com/resources/blog/ad-tech/introducing-fsq-spatial-agent-your-geospatial-ai-analyst/)
- It launched in FSQ Spatial Desktop, with Spatial Workbench "coming soon". It has DuckDB with the spatial and H3 extensions, the H3 Hub and FSQ Places, for example "Draw a 10-minute driving isochrone around each hospital". — [Spatial Agent docs](https://docs.foursquare.com/analytics-products/docs/ai-assistant), [FSQ Spatial Desktop](https://foursquare.com/products/spatial-desktop/)
- The Desktop page lists a free Basic tier through 2026 and paid Standard ($25/month) and Pro ($100/month) tiers that include agent access, with per-analysis cost estimates. Another Foursquare table gives conflicting prices. — [FSQ Spatial Desktop](https://foursquare.com/products/spatial-desktop/)
- Foursquare Studio is described as a free visualisation tool that turns SQL into maps, linked to kepler.gl. No agent integration was found. — [GetApp listing](https://www.getapp.com/gis-software/a/foursquare-studio) (aggregator, so weak)
- Studio release notes exist, but their 2026 contents were not read. — [Studio release notes](https://docs.foursquare.com/studio/docs/release-notes)

**Fused**
- The UDF Catalog is a searchable library of public user-defined functions, with gallery or list view and sorting. Clicking a card shows its profile or adds it to the UDF Builder. — [UDF Catalog](https://docs.fused.io/workbench/udf-catalog)
- A Canvas groups UDFs into projects. The Canvas Catalog lets you browse, create, load, delete and sort canvases. Sharing creates an `fc_` token that makes each UDF on the canvas callable at `udf.ai/fc_<token>/<udf_name>.<ext>`, and shared canvases open read-only. — [Canvas Catalog](https://docs.fused.io/workbench/canvas-catalog), [Canvas](https://docs.fused.io/workbench/udf-builder/canvas)
- App Builder turns UDFs into shareable serverless Streamlit apps. — [App Builder](https://docs.fused.io/workbench/app-builder/app-overview)
- An AI Chat entry exists in the sidebar, but its behaviour was not documented on the pages fetched. A "Canvas MCP server" appears in changelog snippets. Unverified. — [Python SDK changelog](https://docs.fused.io/python-sdk/changelog)

**Dekart**
- Dekart is a self-hosted backend for kepler.gl, written in Go and React, that turns warehouse SQL (BigQuery, Snowflake, Postgres/PostGIS, Wherobots) into maps and caches the query results. — [GitHub dekart-xyz/dekart](https://github.com/dekart-xyz/dekart)
- Its GeoSQL agent skill lets Claude or Codex find schemas, write spatial SQL, render the map through Dekart, inspect the image and fix geometry errors in one loop. — [PyPI geosql](https://pypi.org/project/geosql/0.19.0/), [EveryDev listing](https://www.everydev.ai/tools/geosql)
- The licence was not checked.

**Wherobots**
- Wherobots offers an MCP server, a CLI and a VS Code extension for agents working on WherobotsDB. Chat produces Jupyter notebooks that run unchanged in Wherobots Cloud. — [Wherobots MCP docs](https://docs.wherobots.com/develop/mcp/mcp-server-automating-geospatial-workflows), [Wherobots blog](https://wherobots.com/blog/wherobots-mcp-server-geoint-spatial-pipelines-ai-agents/)
- A HERE partnership was announced in October 2026, with a "first-party agent coming soon". — [GeoWeek News](https://www.geoweeknews.com/articles/here-technologies-and-wherobots-partner-to-give-ai-agents-geospatial-intelligence/)
- Details come from search snippets only.

**Mapbox**
- Mapbox Studio's Data Workbench adds automatic styling defaults on upload, and there is a data visualisation component. Analysis is done in code with Turf.js; Studio has no analysis toolbox. — [Mapbox blog](https://www.mapbox.com/blog/from-upload-to-styled-data-layers-in-seconds-smarter-styling-defaults-in-mapbox-studio), [Turf tutorial](https://docs.mapbox.com/help/tutorials/analysis-with-turf/)

**Honeycomb Maps**
- Honeycomb Maps appears as a DuckDB-WASM map frontend, for example powering the front end of Fused's ParcelPlanner app, and in an OSM-plus-DuckDB tutorial on honeycombmaps.com. No feature or licence page was found. — [Fused blog](https://docs.fused.io/blog/page/3), [Honeycomb tutorial (mirror)](https://hn.jakechampion.name/reader/https://honeycombmaps.com/blog/tutorial-mapping-parcel-lockers-with-osm-and-duckdb-part-1)

**Google Earth Engine**
- The Code Editor's Gemini "Ask" assistant panel was listed in the 16 June 2026 release notes. It uses the user's own Gemini API key to write, debug, explain and optimise scripts. It is still code-first. — [GEE release notes](https://developers.google.com/earth-engine/docs/code-editor/release-notes), [Google Cloud blog](https://cloud.google.com/blog/products/data-analytics/accelerate-geospatial-coding-with-ai-in-google-earth-engine)
- Earth Engine Apps are shareable no-code interfaces built from Code Editor scripts with the UI API. Their assets must be public, and the source is visible to anyone who can open the App. — [Earth Engine Apps](https://developers.google.com/earth-engine/apps)
- No separate 2026 no-code analysis product was found.

**Microsoft Planetary Computer**
- The Hub (hosted notebooks) was retired on 6 June 2024, while the Explorer continues and is described as "rapid prototyping without coding". — [GitHub discussion #347](https://github.com/microsoft/PlanetaryComputer/discussions/347), [Pulse/InfoWorld summary](https://www.pulse.bot/big-data/news/visualizing-the-world-with-planetary-computer-fe0cf11e-17ea-4719-b9e1-eb88cf2e3bfd)
- Planetary Computer Pro is a separate paid Azure product. — [Azure blog](https://azure.microsoft.com/en-us/blog/microsoft-planetary-computer-pro-unlocking-ai-powered-geospatial-insights-for-enterprises-across-industries/)
- That the Explorer offers preset render expressions (such as NDVI) but no analysis tools comes from memory and was not checked against a primary page. Unverified.

**UP42**
- Partners publish reusable data and processing "blocks" to the marketplace. Users chain them into workflows in the UI or the Python SDK, pay in credits, and partners share revenue. — [GeoAwesome](https://geoawesome.com/open-platform-marketplace-satellite-data-analytics-up42-launch/), [UP42 blog](https://up42.com/blog/the-launch-of-our-100th-block)
- A BCG case study (undated) counts 55+ data blocks and 75+ algorithms. — [BCG](https://www.bcg.com/x/mark-your-moment/machine-learning-geospatial-data)
- The SDK migration guide maps old processing classes to "processing templates", which suggests the open block-workflow model was narrowed. Unverified. — [UP42 migration guide](https://docs.up42.com/sdk/migration-guide)
- The May 2026 newsletter covers the console Marketplace tab for data ordering. — [UP42 newsletter](https://up42.com/blog/35-cm-tasking-true-color-conversion-and-more-the-latest-product-updates-from-up42)

### Inferences
- Felt's toolbox is the closest analogue to the current catalog panel, and it differs in three useful ways: an operation-first menu with Cmd+K search, an inline preview before Apply, and an "Apply to: features on screen" scope. A GPU app can make the preview live and on the full dataset, which Felt has to limit to the viewport for large layers.
- The vendor trend is chat first, with toolboxes kept for common one-step operations. The most credible explanation pattern is Foursquare's visible step-by-step reasoning (datasets considered, operations planned, interpretation), compared with Felt's conversational "why" follow-ups.

### Gaps
- No primary-source detail was found on Atlas's tool forms, Foursquare Studio's 2026 analysis features, Fused AI Chat, or Honeycomb Maps' feature set or licence.
- Whether UP42 still offers user-built multi-block processing workflows in 2026 was not confirmed.

## 4. Client-side frameworks near kepler.gl: SQLRooms, Mosaic, Lonboard, Cosmograph, Evidence, Rill

### Takeaway
SQLRooms (MIT, vis.gl/OpenJS) is the most directly reusable source of analysis-step UI for a React and kepler.gl app. It ships kepler.gl, Mosaic, AI chat, a SQL editor and data table, and experimental `canvas` (React Flow DAG of SQL and chart nodes), `cells` and `notebook` packages. Mosaic (BSD-3-Clause per package metadata) offers linked, DuckDB-backed inputs and charts. Lonboard (MIT) and cosmos.gl (MIT) are visualisation only.

### Cited Findings
- SQLRooms core packages: room-shell (Zustand, panel system), duckdb (WASM), db, layout, ui (Tailwind). — [SQLRooms packages](https://sqlrooms.org/packages.html)
- SQLRooms AI packages: ai, ai-core (chat UI primitives and tool-output streaming), ai-settings. — [SQLRooms packages](https://sqlrooms.org/packages.html)
- SQLRooms feature packages: kepler ("map-first analytics"), deck (JSON map specs over DuckDB and Arrow), mosaic, sql-editor (history and result docking), monaco-editor, data-table, schema-tree, recharts, vega, color-scales, cosmos, documents, dropzone. — [SQLRooms packages](https://sqlrooms.org/packages.html)
- SQLRooms experimental packages: **canvas** ("React Flow-based canvas for DAGs of SQL, Vega, and dataflow nodes"), **cells** (a shared cell model used by notebook and canvas), **notebook** (SQL, text, Markdown, Vega and input cells), pivot, artifacts, app-runtime (sandboxed iframe for generated HTML apps), python (Pyodide), crdt (Loro), discuss (threads anchored to data points), ai-rag. — [SQLRooms packages](https://sqlrooms.org/packages.html)
- The API version shown is v0.29.0. — [sqlrooms.org](https://sqlrooms.org/)
- The SQLRooms repository is MIT licensed and part of the OpenJS Foundation through vis.gl, and Foursquare introduced the project. — [GitHub sqlrooms/sqlrooms](https://github.com/sqlrooms/sqlrooms), [Foursquare blog](https://foursquare.com/resources/blog/products/foursquare-introduces-sqlrooms/)
- kepler.gl 3.3 alpha release notes mention SQLRooms work, and FSQ Spatial Desktop is built with it (inferred from Foursquare's SQLRooms announcement; unverified). — [kepler.gl releases](https://github.com/keplergl/kepler.gl/releases)
- `@uwdata/mosaic-plot` is listed as BSD-3-Clause at version 0.27.0. One aggregator lists the repository licence as "other". — [jsDelivr](https://www.jsdelivr.com/package/npm/@uwdata/mosaic-plot), [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fuwdata%2Fmosaic)
- Mosaic's input widgets (menu, search, slider, table) and its cross-filtering with DuckDB are from memory, not fetched this session. Unverified. — [GitHub uwdata/mosaic](https://github.com/UWData/mosaic)
- Lonboard is MIT and does "Fast, interactive geospatial data visualization in Jupyter". It has no analysis UI. — [GitHub developmentseed/lonboard](https://github.com/developmentseed/lonboard)
- cosmos.gl is MIT and powers Cosmograph. Cosmograph's own licence was not stated. — [GitHub cosmograph-org/cosmos](https://github.com/cosmograph-org/cosmos)
- Evidence (SQL plus Markdown reports, MIT) and Rill (Apache-2.0, DuckDB/ClickHouse dashboards) were not fetched this session. Their licences and features are from memory. Unverified.

### Inferences
- For a workflow builder, SQLRooms' `canvas` and `cells`, together with React Flow (MIT), are the closest permissively licensed starting point that already sits next to kepler.gl. They are marked experimental, so version churn is likely. Each node type would need to become "GPU operation" instead of "SQL".
- SQLRooms' `ai-core` tool-output streaming and `discuss` (threads anchored to data points) are reusable patterns for explaining results.

### Gaps
- Licences and UI details for Evidence, Rill and the Cosmograph React library were not confirmed.
- The maturity of SQLRooms canvas (screens, node types) was not examined beyond its package description.

## 5. openEO Web Editor and OGC API Processes clients

### Takeaway
The openEO Web Editor has the clearest open model for a typed process catalogue. It has a process list, a "Visual Model" builder that wires collections and processes into a JSON process graph with a "Code" tab beside it, and an experimental Wizard for common use cases. It is Apache-2.0. OGC API Processes clients mostly generate an HTML form from each process description's JSON-Schema inputs.

### Cited Findings
- In the openEO Web Editor, the Process Editor has a "Visual Model" tab for adding and connecting collections and processes, and a "Code" tab that shows the generated JSON process graph. A process can be loaded from a URL into the Model Builder. The Wizard ("experimental") simplifies common use cases and can be preset with `wizard~` URL parameters, and a "UDP" wizard runs a user-defined process. It supports openEO API 1.0.x. — [Terrascope docs](https://docs.terrascope.be/Tools/openEOWebEditor/openEOWebEditor.html)
- The source is Apache 2.0, a demo runs at editor.openeo.org, and it can be self-hosted with Docker. — [Docker Hub mundialis/openeo-web-editor](https://hub.docker.com/r/mundialis/openeo-web-editor), [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fopen-eo%2Fopeneo-web-editor)
- The openEO R client has a "Process Graph Building Application" vignette. — [CRAN vignette](https://ftp.fau.de/cran/web/packages/openeo/vignettes/openeo-05-process_graph_building_application.html)
- OGC API Processes defines `GET /processes`, `GET /processes/{id}` and `POST /processes/{id}/execution`, with HTML and JSON encodings. — [OGC workshop](https://ogcapi-workshop.ogc.org/api-deep-dive/processes), [OGC overview](https://ogcapi.ogc.org/processes/overview.html)
- Clients for OGC API Processes:
  - The ZOO-Project/MapMint demo lists processes and builds an HTML input form per process. — [zoo-discuss 2020](https://lists.osgeo.org/pipermail/zoo-discuss/2020-November/001765.html)
  - The OGC Processes Dashboard has a "Process list" tab and an execution form driven by the OGC schema. — [secd.eu dashboard](https://processes.ogc.secd.eu/readme.html)
  - Eozilla ships Python GUI widgets for executing processes and viewing jobs. — [Eozilla](https://eo-tools.github.io/eozilla/notebooks/client-gui)
  - Weaver's HTML view shows `minOccurs`, `maxOccurs` and literal value domains for each input. — [Weaver](https://hirondelle.crim.ca/weaver/processes/echo?f=html)

### Inferences
- The library's operation catalog with generated forms is already shaped like OGC and openEO process descriptions. Adopting a JSON-Schema-like typed parameter description, with geometry, raster or table input kinds and value domains, would let one description drive the form, the canvas node ports, template slots and LLM tool schemas, as CARTO's MCP variables do.
- openEO's side-by-side Visual Model and Code tabs are a proven way to keep a canvas honest and diffable.

### Gaps
- The openEO Web Editor's 2026 release state and its UI for wiring callback child graphs were not checked.

## 6. Cross-product comparison: access modes, data-driven suggestions, live preview, result explanation, complaints

### Takeaway
Every product offers either forms or chat. Canvases are found in enterprise or engineering products (CARTO Workflows, openEO, Fused, UP42), and templates are only well developed in CARTO (61). No product found here suggests an analysis from the dataset's column types; kepler.gl suggests layers only. Live parameter preview is rare: Felt shows an operation preview and CARTO previews each node's output after a run. Explanation is either SQL transparency (CARTO's SQL tab) or narrated agent reasoning (Foursquare).

### Cited Findings
Access-mode matrix: (a) toolbox of forms, (b) canvas, (c) SQL or code, (d) chat, (e) templates, (f) suggestions from the data. It is built from the findings above; a cell is "yes" only where cited.

| Product | (a) forms | (b) canvas | (c) SQL/code | (d) chat | (e) templates | (f) data-driven suggestions |
|---|---|---|---|---|---|---|
| kepler.gl 3.x | limited (filters, aggregation layers, suitability in 3.3 alpha) | no | DuckDB SQL panel | AI assistant (GeoDa tools), MCP (experimental) | no | layers only (name/type heuristics) |
| CARTO | Builder SQL Analyses, SQL Parameters | Workflows | Builder SQL, Workflows SQL tab | Builder AI Agents, MCP Server | 61 Workflows templates | none found |
| Felt | Analysis panel (11 operations) | no | SQL via Felt AI and sources | Felt AI | "no templates needed" | Felt AI "can suggest relevant layers" ([GeoAwesome](https://geoawesome.com/wp-content/uploads/2026/04/FINAL-geoawesome-wherobots-felt.html), unverified on a primary page) |
| Atlas | 50+ tools | unverified | unverified | Navi | unverified | none found |
| Foursquare | Studio (SQL to map) | no | DuckDB SQL | Spatial Agent | no | none found |
| Fused | UDF parameters | Canvas (project grouping) | Python UDFs | AI Chat (unverified) | UDF Catalog | none found |
| Dekart | no | no | SQL | GeoSQL agent skill | no | none |
| Wherobots | no | no | notebooks | MCP agent | no | none |
| Earth Engine | Apps (author-built) | no | JS Code Editor | Gemini panel | example scripts (unverified) | none |
| openEO Web Editor | process forms, Wizard | Visual Model | JSON graph | no | Wizard presets, UDPs | none |
| UP42 | block parameters | block workflow (historic) | SDK | no | processing templates (unverified) | none |

- Live preview evidence: Felt "The panel includes a preview of the operation." — [Felt help](https://help.felt.com/layers/spatial-analysis). CARTO previews the selected node's output (Data and Map tabs) and its intermediate tables after a run. — [Workflows manual](https://docs.carto.com/carto-user-manual/workflows)
- Explanation evidence: CARTO's SQL tab shows the compiled SQL. — [Workflows manual](https://docs.carto.com/carto-user-manual/workflows). Foursquare shows its reasoning at every step. — [Foursquare blog](https://foursquare.com/resources/blog/ad-tech/introducing-fsq-spatial-agent-your-geospatial-ai-analyst/). Felt explains through conversational follow-ups. — [Felt blog](https://felt.com/blog/a-brand-new-era-of-gis)
- Limitations vendors document themselves:
  - Felt results are static snapshots: "created once and don't update when an input layer refreshes". Dissolve cannot group by attribute, and Join matches exact values only. — [Felt help](https://help.felt.com/layers/spatial-analysis)
  - CARTO says the Builder SQL editor is unsuitable for complex analysis. — [CARTO docs](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/guides/running-queries-from-builder)
  - On old CARTO, adding an analysis made the layer's SQL read-only. — [CARTO legacy help](https://carto.com/help/tutorials/how-to-preserve-a-layer-query) (old product)
  - Earth Engine App source is public and its assets must be public. — [Earth Engine Apps](https://developers.google.com/earth-engine/apps)
- kepler.gl AI assistant complaints: a third-party tutorial reports that the model hallucinates column names and suggests starting with a request to list the columns. It also says custom layer extensions written for older versions render blank after the deck.gl 9 upgrade. This is a secondary source. — [qwe.edu.pl tutorial](https://www.qwe.edu.pl/tutorial/ai-geospatial-data-visualization/)

### Inferences (patterns worth borrowing for a client-side GPU app, ranked)
1. **Suggestions from data types (an open niche).** No peer does this. kepler.gl's signals (lat/lng `fieldPairs`, h3, geojson and geoarrow field types, timestamp fields, and numeric fields left after `EXCLUDED_DEFAULT_FIELDS`) are enough to rank applicable operations. For example: points plus a numeric column suggests hotspots, LISA, IDW or kriging; polygons plus a numeric column suggests Moran's I, LISA or classification; points plus a time column suggests space-time clustering; a raster suggests terrain or band math; lines suggest network operations. Because the GPU runs locally, each card can carry a cheap thumbnail preview computed on a sample.
2. **Templates with typed input slots plus a column-mapping step.** Borrow CARTO's gallery structure: generic categories (data preparation, statistics, indexes) alongside domain categories, with sample data loaded. Avoid CARTO's "replace the sources and reconfigure the nodes" adaptation by validating the user's columns against slot types when the template is opened.
3. **One typed operation description for four surfaces.** Like OGC and openEO process descriptions and CARTO MCP variables, a single JSON-Schema-style description can drive the form, canvas ports, template slots and an AI tool or MCP schema. kepler.gl's AI assistant (OpenAssistant tools) and `@kepler.gl/mcp` are natural hosts for that schema.
4. **Felt-style operation panel improvements.** Cmd+K search over operations, preview before Apply, an "Apply to: features on screen" scope, and results as a new layer that leaves the inputs unchanged. A client-side app can go further than Felt by re-running when inputs change, fixing Felt's "created once" limitation.
5. **Canvas only as an optional advanced view.** Use a left-to-right DAG with a bottom panel for the selected node (Data, Map, Parameters/Provenance, Messages) and a Code/JSON tab beside it (openEO, CARTO). SQLRooms `canvas`/`cells` (MIT) and React Flow are the reusable base.
6. **Explaining results.** Show the parameters and run time for each result layer (CARTO-style JSON with row count and timing). For statistics, add a short generated narrative of significant clusters, as Foursquare does with its step reasoning.

### Gaps
- Forum, review and GitHub-issue criticism of CARTO Workflows, Felt Analysis, Atlas and Foursquare Studio was not found. Search for kepler.gl AI assistant issues returned no GitHub threads. Sentiment research is mostly a gap.
- None of the sources fetched confirm live preview that updates as parameters change, as opposed to a static operation preview, in any product.
