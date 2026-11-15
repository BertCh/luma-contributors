# Curated templates, guided analysis, parameter UX and result explanation

Research date: 2026-11-14. About 22 search and fetch calls. Items marked **[unverified]** come from
background knowledge, not from a page fetched in this session. Use them as leads and check them
before citing.

## (a) Template galleries: structure, size, categories, and "use with my data"

### Takeaway
The most useful spatial benchmark is CARTO Workflows. Its gallery is a small curated set of about 61
templates in 13 categories. Technique categories (data preparation, enrichment, spatial indexes,
spatial analysis, statistics) sit beside industry categories (retail, telco, insurance, out-of-home
advertising), and extension-gated categories (BigQuery ML, Snowflake ML, territory planning) come
last. You re-point a CARTO template by swapping its source nodes for your own tables. The clearest
"bind the template to my columns" pattern is the Tableau Accelerator Data Mapper. It maps each
required template field to a user field, allows only fields of the same type, greys out the rest,
shows a count of assigned fields, and warns that dashboards depending on unmapped fields break.

### Cited Findings
**CARTO Workflows**
- The Academy index lists these template categories: Data Preparation, Data Enrichment, Spatial
  Indexes, Spatial Analysis, Generating new spatial data, Statistics, Retail and CPG, Telco,
  Insurance, Out Of Home Advertising, BigQuery ML, Snowflake ML and Territory Planning. The last three
  need extension packages. Users open the gallery in the Workspace with "New Workflow" then "From
  template". CARTO says the gallery covers "simple building blocks" through "more complex,
  industry-specific workflows" — [CARTO Academy, Workflow templates](https://academy.carto.com/creating-workflows/workflow-templates); [CARTO blog, Introducing Workflows Templates](https://carto.com/blog/workflows-templates-pre-built-spatial-analysis/)
- Counts from the fetched index, 61 templates in total:
  - Data Preparation: 8
  - Data Enrichment: 5
  - Spatial Indexes: 4
  - Spatial Analysis: 5. These are K-means customer clusters, points within polygons, aggregate
    points into polygons, Voronoi polygons, and custom SQL with two inputs.
  - Generating new spatial data: 5. These are drawing custom geographies, routes from
    origin–destination data, geocoding, points from latitude and longitude, and isochrones.
  - Statistics: 9. These are hotspots of a point-of-interest type, space-time hotspot analysis,
    space-time hotspot classification, time-series clustering, spatial autocorrelation of
    points of interest, GWR, supervised and unsupervised composite scores, and space-time anomalies.
  - Retail and CPG: 2
  - Telco: 8
  - Insurance: 5
  - Out-of-home advertising: 1
  - BigQuery ML: 4
  - Snowflake ML: 2
  - Territory Planning: 3

  The counts are as of this fetch; the in-app labels may differ — [CARTO Academy, Workflow templates](https://academy.carto.com/creating-workflows/workflow-templates)
- Template names are phrased as a task or question tied to a domain example, for example
  "Commercial hotspot analysis for a new pizza place", "New tower site selection in Denver" and
  "Volcano eruption damage on La Palma" — [CARTO Academy](https://academy.carto.com/creating-workflows/workflow-templates)
- Adapting a template to your data: the template is re-created as an ordinary editable workflow,
  and you "replace the data sources and reconfigure the nodes". The index page documents no
  field-mapping dialog — [CARTO Academy](https://academy.carto.com/creating-workflows/workflow-templates)
- **[unverified]** Each CARTO template uses public sample data, so it runs as soon as it is
  created. Check an individual template page such as [Insurance templates](https://academy.carto.com/creating-workflows/workflow-templates/insurance).

**Tableau Accelerators and the Data Mapper**
- Accelerators are pre-built dashboards for an industry or enterprise application. They ship with
  sample data that you replace. The full set is on Tableau Exchange, where you can filter by
  industry, by data source type, and by "Data Mapping enabled". Only a limited number can be
  created directly in Tableau Cloud — [Tableau Help, Use Accelerators](https://help.tableau.com/current/online/en-us/accelerators.htm)
- In the Data Mapper you choose Get Started, connect a source, and map each Accelerator field to
  one of your fields. The two must have the same data type, so you change a field's type to make
  it mappable. A counter in the lower left shows how many fields are assigned. Replacing the data
  removes all sample data. From 2023.2 you can reopen the mapper later — [Tableau Help](https://help.tableau.com/current/online/en-us/accelerators.htm)
- Fields that don't match in type are greyed out. Unmapped fields break the dashboards that use
  them. When you remap by hand, mismatches show as red exclamation marks, which you fix with
  "Replace References" — [Tableau Help](https://help.tableau.com/current/online/en-us/accelerators.htm); [InterWorks 2023](https://interworks.com/blog/2023/03/28/tableau-accelerator-data-mapper-tableau-20231-new-feature/)
- Data mapping arrived in Tableau 2023.1 for a limited set of Accelerators. In 2023.2 it gained
  saved selections and wider coverage — [Tableau blog, 14 March 2023](https://www.tableau.com/blog/release-accelerator-data-mapping-identity-pools); [phData](https://www.phdata.io/blog/how-to-use-accelerator-data-mapping-in-tableau/)
- One reviewer in 2023 reported that calculated fields with circular references cause mapping
  problems — [phData](https://www.phdata.io/blog/how-to-use-accelerator-data-mapping-in-tableau/)
- The Exchange product page lists each Accelerator's required fields — [InterWorks](https://interworks.com/blog/2023/03/28/tableau-accelerator-data-mapper-tableau-20231-new-feature/)

**Alteryx**
- One-Tool Examples sit under Help > Sample Workflows. A 2017 Alteryx tips document describes them
  as "fully annotated, use case examples for specific tools" — [Scribd, Alteryx Tips & Tricks 2017](https://www.scribd.com/document/434915395/Alteryx-Tips-Tricks-2017-pdf); [Alteryx sample workflows 2021.1](https://help.alteryx.com/2021.1/SamplesByDescription.htm)
- A 2019 community idea complained that the Spatial category had only one One-Tool Example, for the
  Distance tool. Spatial coverage was thin at that point — [Alteryx Community idea](https://community.alteryx.com/t5/Alteryx-Designer-Ideas/One-Tool-Examples-in-Spatial-Category/idi-p/447310)
- Starter Kits install from a .yxi file into Help > Sample Workflows > Starter Kits, and are now
  distributed through the Alteryx Gallery. The Spatial Analytics Starter Kit is built around
  customer value, customer location and a new store location — [Alteryx Spatial Analytics Starter Kit](https://www.alteryx.com/starter-kit/spatial-analytics); [Alteryx Community Gallery](https://community.alteryx.com/t5/Community-Gallery/Spatial-Analytics-Starter-Kit/ta-p/1189850)

**KNIME Hub**
- KNIME's official "Examples" space offers "workflows and verified components ... to use as
  blueprints and building blocks" — [KNIME Hub Examples](https://hub.knime.com/knime/space/Examples)
- Community geospatial spaces from Harvard's Center for Geographic Analysis include choropleth,
  OSM dashboard, POI search, site selection and spatial join examples, plus a Google Earth Engine
  examples space — [Geospatial Analytics Examples](https://hub.knime.com/center%20for%20geographic%20analysis%20at%20harvard%20university/spaces/Geospatial%20Analytics%20Examples/); [GEE Examples](https://hub.knime.com/center%20for%20geographic%20analysis%20at%20harvard%20university/spaces/Google%20Earth%20Engine%20Examples/)
- Official versus community: "Verified components" were a KNIME-curated tier released monthly from
  2019 to 2023 — [KNIME verified components history](https://www.knime.com/verified-components-history)

**ArcGIS Solutions**
- ArcGIS Solutions has more than 100 industry configurations compatible with Enterprise 11.5.
  They are grouped by industry: local and state government, electric, gas, water, telecom,
  emergency management, defense, business and intelligence. The deployment tool's steps are Sign
  In, Deploy, Configure and Load Data. The tool is now in the Mature phase, and solutions are
  reached through the ArcGIS Solutions app — [Esri EA blog, 11.5](https://www.blog.esriea.com/?p=2737); [Esri ArcGIS Solutions](https://www.esri.com/en-us/arcgis/products/arcgis-solutions/); [Esri Canada, deployment tool](https://resources.esri.ca/arcgis/using-the-arcgis-solutions-deployment-tool-for-arcgis-pro)

**Workflow-automation marketplaces, for contrast on size**
- n8n's template library is large and mostly community-made. Third-party counts vary by source and
  date: 7.8K (a stats dashboard), "9,000+" and "over 10,000" (scraper listings), and 13,113 (a
  forum indexer). There is no official figure — [n8n community forum](https://community.n8n.io/t/i-indexed-13-113-n8n-workflow-templates-heres-what-people-actually-automate/323360?tl=fr); [n8n-pulse stats](https://n8n-pulse.gui.do/templates); [Apify listing](https://apify.com/automation-lab/n8n-workflow-scraper)

**Not verified this session [unverified]**

Power BI template apps, the ComfyUI template browser and ArcGIS Online analysis templates are now
covered in the second-pass section "(e2) Template leads".
- Looker Blocks (Looker Marketplace)
- Hex and Observable templates
- Google Earth Engine Apps and the community catalog
- Dataiku Solutions, organised by industry with a data-mapping or "connect your data" step
- QGIS model sharing through the QGIS Hub and the plugin repository
- Felt and Atlas template maps
- Zapier and Make template browsers

### Inferences
- Gallery size: spatial products keep curated galleries small, with CARTO at about 61. General
  automation marketplaces grow to thousands of community templates. A 23-recipe library is already
  in CARTO's range. Growing it toward 40–60, organised by technique plus a few domain stories,
  matches the closest comparable product.
- Templates are organised along two axes in practice: technique (what operation) and domain story
  (what question, in what industry). CARTO names templates by question plus example place, which
  sets user expectations better than operator names.
- For "use with my data", a Tableau-style role-slot mapper fits this library's recipes well:
  required fields with types, greyed-out incompatible columns, an assigned-fields counter, and an
  explicit statement of what breaks if a slot stays empty. Composite recipes already have typed
  inputs, so the slots can come from the recipe declarations. CARTO's model of "swap the source
  node" is simpler but leaves validation to the user.

### Gaps
- Exact current CARTO in-app counts and card anatomy were not confirmed. The fetched page did not
  show whether every template bundles sample data.
- No published evidence was found comparing adoption of a small curated gallery with a large
  marketplace. The size argument above is inference from product practice.
- Versioning and breakage: the only evidence found is the Tableau statement that unmapped fields
  break dashboards, and the 2023 calculated-field mapping issue. No vendor policy on template
  versioning was found.

## (b) Guided analysis and wizard patterns

### Takeaway
NN/g's guidance is clear. Wizards suit novices, infrequent tasks, and tasks that split into
independent steps. They are a poor fit for repeated tasks, for steps that depend on each other, and
for experts. ArcGIS Pro Tasks are the GIS form of an "author once, replay" guided workflow.
Datawrapper's four fixed steps and jamovi's live-updating single-page options are two working
alternatives to a modal wizard.

### Cited Findings
- NN/g (Raluca Budiu, 25 June 2017) defines a wizard as a "step-by-step process that allows users to
  input information in a prescribed order", where later steps can depend on earlier answers.
  - Use wizards for novices, infrequent tasks, and complex tasks broken into simpler pages.
  - Avoid them for frequent tasks, for comparing information across steps, for tasks users need to
    leave and resume, where a modal window would hide needed information, and for professionals who
    need flexibility.
  - Recommendations: show all steps and the current one; use descriptive button labels; let users
    save progress and resume; make steps self-contained; put help beside the fields, not over them;
    reuse previous selections as defaults.

  — [NN/g, Wizards](https://www.nngroup.com/articles/wizards/)
- NN/g on progressive disclosure: show core options first and defer advanced ones. "Staged
  disclosure", as in wizards, works when the steps interact little and fails when users must move
  back and forth between dependent steps. Hiding advanced settings helps novices avoid mistakes —
  [NN/g, Progressive Disclosure](https://nngroup.com/articles/progressive-disclosure/)
- ArcGIS Pro Tasks are "preconfigured steps that guide you and others through a workflow or business
  process".
  - Structure: a task item holds optional task groups, which hold tasks, which hold sequential
    steps. A step can run a tool or command, set the map view, or set layer selection and
    visibility.
  - Authoring: Insert > Tasks > New Task Item, then the Task Designer. Validation messages appear
    under the Tasks pane's Messages tab.
  - Sharing: export to a task file that another project can import.
  - Uses include best-practice workflows and tutorials. Esri says step behaviours reduce mouse
    clicks.

  — [Esri, Design a task](https://doc.esri.com/en/arcgis-pro/latest/help/tasks/design-a-task.html); [Esri, Open and run tasks](https://pro.arcgis.com/en/pro-app/help/tasks/open-and-run-tasks.htm); [Esri Canada](https://resources.esri.ca/news-and-updates/streamline-workflows-with-tasks-in-arcgis-pro)
- Datawrapper uses a fixed four-step path:
  1. Upload: paste, CSV or XLS, Google Sheets or URL.
  2. Check & Describe: data types are colour-coded (dates green, numbers blue, text black) and
     cells can be edited.
  3. Visualize: the chart type defaults to a line chart.
  4. Publish & Embed.

  — [Datawrapper Academy, first chart](https://www.datawrapper.de/academy/how-to-create-your-first-datawrapper-chart); [Datawrapper, first line chart](https://www.datawrapper.de/academy/how-to-create-your-first-line-chart)
- jamovi: the results panel is a live document. Output recalculates as options or data change,
  without re-running. Assumption checks are check boxes inside each analysis's options. For one-way
  ANOVA these are homogeneity, normality and Q-Q plot. Users have complained that assumption output
  sits in different places in different analyses — [Learn Statistics with jamovi](https://lsj.readthedocs.io/en/latest/Ch03/Ch03_jamoviIntro_2/); [TRU one-way ANOVA guide](https://researchmethodsandstats.pressbooks.tru.ca/?p=381); [jamovi forum](https://forum.jamovi.org/viewtopic.php?p=1834); [jamovi user guide](https://jamovi.org/user-manual.html)
- **[unverified]** JASP follows the same pattern as jamovi: an analysis menu, an options panel, live
  output, and APA-style tables.
- **[unverified]** Learn ArcGIS lessons are framed as questions.
- **[unverified]** The GeoDa workbook sequence runs from mapping to weights, then global and local
  autocorrelation, then regression.
- **[unverified]** SPSS and Stata use modal dialogs with OK and Paste-syntax buttons, as a contrast.

### Inferences
- A WebGPU library where parameters update the map live fits the jamovi model best: one page of
  options, results recomputing in place, and assumption checks as toggles. A modal wizard does not
  fit, because NN/g notes wizards fail when users iterate between dependent steps, and parameter
  tuning is exactly that.
- A Datawrapper-style constrained path suits only the data-binding stage: load the data, check the
  detected column types, then pick a recipe. Analysis stays on one live page.
- ArcGIS Pro Tasks show that "author once, replay" needs step-level validation and an exportable
  file. Recipes serialised as computation records would play that role.

### Gaps
- No quantitative studies comparing wizard completion rates with single-page forms for analysis
  tools were found.
- Tableau "Explain Data" counts as a guided entry point, but it is covered under (d).

## (c) Parameter UX: smart defaults, live preview, sweeps, sensitivity

### Takeaway
Esri's "Optimized" tools are the main precedent for deriving defaults from the data. The tool
examines the data, picks the aggregation and scale, applies an FDR correction, and reports the
settings it chose as messages. In the visualisation literature, Sedlmair et al. (2014) set out six
parameter-space tasks: optimisation, partitioning, fitting, outliers, uncertainty and sensitivity.
Design Galleries (Marks et al. 1997) set out the small-multiples answer: show a spread of distinct
outputs and let the user pick, instead of asking for numbers.

### Cited Findings
- Optimized Hot Spot Analysis "interrogates your data" to choose its settings.
  - It aggregates incidents automatically, computing a fishnet or hexagon cell size.
  - It identifies an appropriate scale of analysis.
  - It corrects for multiple testing and spatial dependence with FDR.
  - It reports the chosen settings as messages while it runs.
  - Users who want manual control use the full Hot Spot Analysis tool instead.

  — [Esri, Optimized Hot Spot Analysis](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/optimized-hot-spot-analysis.htm)
- Sedlmair, Heinzl, Bruckner, Piringer and Möller, "Visual Parameter Space Analysis: A Conceptual
  Framework", IEEE TVCG 20(12):2161–2170, 2014, DOI 10.1109/TVCG.2014.2346321. The framework has:
  - a data-flow model
  - four navigation strategies
  - six analysis tasks: optimisation, partitioning, fitting, outliers, uncertainty and sensitivity

  — [Sedlmair et al. 2014 PDF](https://vis.uib.no/wp-content/papercite-data/pdfs/Sedlmair-2014-VPS.pdf)
- Marks et al., "Design Galleries", SIGGRAPH 1997, pp. 389–400, addresses parameter mappings that
  are "multidimensional, nonlinear, and discontinuous". It presents "the broadest selection,
  automatically generated and organized, of perceptually different" outputs. The two technical
  problems it names are dispersion and arrangement. It was demonstrated on light placement,
  volume-rendering transfer functions and animation — [Harvard VCG](https://vcg.seas.harvard.edu/publications/19970101-design-galleries-a-general-approach-to-setting-parameters-for-computer-graphics-and-animation); [MERL TR97-14](https://merl.com/publications/TR97-14)
- Bret Victor's 2011 "Explorable Explanations" essay popularised the term. Tangle is his library
  for "reactive documents" with inline variables. Scrubbable numbers are changed by dragging, and
  he notes granularity matters because some values span thousands in one stroke — [Bret Victor, Scrubbing Calculator](https://worrydream.com/ScrubbingCalculator/); [Wikipedia, Explorable explanation](https://wikipedia.com/wiki/Explorable_explanation)
- Several of the default rules first listed here as unverified are now confirmed from primary pages:
  the Esri spatial Silverman variant, the DBSCAN minPts and k-distance heuristics, and the
  Incremental Spatial Autocorrelation peak. See the second-pass section "(c2) Data-derived defaults".
- **[unverified]** These remain unconfirmed:
  - Jenks natural breaks as ArcGIS's default classification. The Esri classification page fetched
    in the second pass does not state a default.
  - Dream Lens (Matejka et al., CHI 2018). Search did not find it.

### Inferences
- Pattern to adopt: compute a suggested value from the data, for example a bandwidth from the
  spatial spread or eps from the k-distance knee. Show it with its rule and units ("Suggested:
  1.2 km (Silverman)"), offer "reset to suggested", and log it as Esri's messages do.
- Because results are cheap on the GPU, Design-Gallery small multiples become practical in the
  browser. For example, show 4–9 thumbnails of a hot-spot map across distance bands. This serves
  the Sedlmair "sensitivity" and "partitioning" tasks directly.
- Direct manipulation fits distance-type parameters: drag a radius ring on the map, with the number
  scrubbable in the label. Victor's granularity point suggests log-scale scrubbing for distances.

### Gaps
- No controlled study was found showing that live preview improves the quality of analysis
  decisions, as opposed to speed or satisfaction.
- No source was retrieved for named presets or for plain-language statistical labels as a measured
  practice.

## (d) Result explanation and onboarding

### Takeaway
Explanation layers in the products found here are:
- the settings the tool chose plus corrected significance bins (Esri Optimized Hot Spot)
- reports tied to the null hypothesis (Esri Moran's I)
- auto-generated narratives (Power BI Smart Narrative, Tableau Data Stories)
- driver analysis on a clicked mark (Tableau Explain Data, Power BI "Explain the increase")

Legends tied to confidence bins (±3, ±2, ±1 for 99, 95 and 90 percent) are the established way to
encode significance on hot-spot maps.

### Cited Findings
- Optimized Hot Spot output has z-score, p-value, Gi_Bin and NNeighbors fields. Gi_Bin values ±3,
  ±2 and ±1 mean 99, 95 and 90 percent confidence. Only Gi_Bin carries the FDR correction for
  multiple testing and spatial dependence; z and p do not — [Esri, Optimized Hot Spot Analysis](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/optimized-hot-spot-analysis.htm)
- Global Moran's I returns the index, expected index, variance, z-score and p-value. The path to an
  HTML report appears in the tool messages. Esri stresses that the index "can only be interpreted
  within the context of the null hypothesis" of complete spatial randomness. When significant, a
  positive index means clustering and a negative one means dispersion — [Esri, How Global Moran's I works](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm); [ArcMap 10.4 Spatial Autocorrelation](https://desktop.arcgis.com/en/arcmap/10.4/tools/spatial-statistics-toolbox/spatial-autocorrelation.htm)
- An example of the report's plain-language framing, from a student report on ArcGIS 10.1 output:
  I = 0.206, z = 3.47, p = 0.0005, and "less than 1% likelihood that this clustered pattern could be
  the result of random chance". The exact current wording was not verified — [USU tutorial](https://www.usu.edu/geospatial/tutorials/advanced/basic-geostatistics)
- Tableau Explain Data (2019.3): the user selects a mark, and Tableau fits statistical models
  (reported as Bayesian) to test candidate drivers, including fields not in the view. It shows only
  the explanations it scores as most informative — [Tableau blog, Sept 2019](https://www.tableau.com/blog/explain-data-helps-you-deeply-understand-your-data); [Explain Data, Explained](https://tableau.com/blog/explain-data-explained)
- Power BI Smart Narrative writes AI commentary on a report, avoids repeating itself across
  visuals, and lets users add fixed text. Tableau Data Stories is the counterpart. Power BI "Analyze
  > Explain the increase" offers waterfall, scatter and ribbon views. This comes from a vendor
  comparison blog, so it is a secondary source — [PA Consulting](https://blog.pa.com.au/microsoft-power-bi/power-bi-feature-focus-smart-narratives); [Concord comparison](https://www.concordusa.com/blog/ai-showdown-power-bi-vs-tableau)
- Assumption checks as visible toggles next to results: jamovi ANOVA offers homogeneity, normality
  and Q-Q plot — [TRU guide](https://researchmethodsandstats.pressbooks.tru.ca/?p=381)
- **[unverified]** Other explanation practices to check:
  - Esri regression diagnostics reports, such as the OLS report with Koenker, Jarque–Bera and VIF
    warnings.
  - Local Moran (Cluster and Outlier) legends with HH, LL, HL and LH.
  - Tableau Pulse metric digests.
  - JASP and jamovi annotations.
  - NN/g articles on empty states.

### Inferences
- An explanation template per recipe would contain:
  - what the tool chose and why, in Esri-style messages
  - the significance encoding and whether it is FDR-corrected, in the legend
  - one plain sentence anchored to the null hypothesis, following Esri's Moran wording
  - assumption warnings as toggles or badges, as in jamovi
  - a "next analysis" link. For example, from global Moran's I significant to local LISA, then to
    cluster-and-outline.
- Esri's split between corrected bins and uncorrected z and p is a concrete warning about multiple
  testing. Legends should say which value they show.

### Gaps
- No primary source was retrieved on empty-state design (NN/g), interactive tutorials, or paired
  sample-data-plus-analysis onboarding. Datawrapper's sign-in-free creation before publishing is
  the only onboarding datum found ([Datawrapper Academy](https://www.datawrapper.de/academy/how-to-create-your-first-datawrapper-chart)).
- No outcome evidence was found on whether auto-generated narratives improve correct
  interpretation. All sources found are vendor descriptions. The second pass found only partial
  HCI evidence; see (d2).

---

# Second pass (2026-11-14): parameter defaults, live preview, explanation, onboarding

About 40 further search and fetch calls. "Fetched" means the primary page was read in this session.
Where only search snippets were available, the finding is marked "(search snippet)".

## (c2) Data-derived defaults: rule, source, required data statistic

### Takeaway
Most mature spatial tools derive their key parameter from one cheap data statistic:
- the nearest-neighbour distance, which guarantees every feature has at least one neighbour
- the spatial spread (standard distance and median distance from the centre), for kernel bandwidth
- the extent divided by 250, for cell size
- the k-distance curve, for DBSCAN eps

Each of these is a single GPU reduction over the input, so a "suggested value plus rule name" can
be shown before the first run.

### Cited Findings

| Parameter | Rule (default) | Data statistic needed | Primary source |
|---|---|---|---|
| Kernel density search radius (bandwidth) | Esri default is "a spatial variant of Silverman's Rule of Thumb" adapted to two dimensions. It uses standard distance SD, the (weighted) median distance from the (weighted) mean centre Dm, and n (point count or population sum), and takes whichever option gives the smaller value. The formula is shown only as an image | Mean centre, standard distance, median distance to centre, n | [Esri, How Kernel Density works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/how-kernel-density-works.html); [Esri, Kernel Density tool](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/kernel-density.html); original: Silverman, B. W. 1986, *Density Estimation for Statistics and Data Analysis*, Chapman and Hall (cited on the Esri page) |
| Kernel density and Point to Raster output cell size | "the cell size is calculated from the shorter of the width or height of the extent divided by 250" | Extent | [Esri, Kernel Density](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/kernel-density.html); [Esri, Point to Raster](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/conversion/point-to-raster.html) |
| DBSCAN minPts | Keep the default minPts = 4 for 2-D data (Ester et al. 1996). Sander et al. suggest minPts = 2·dim. Increase it for noisy, large, high-dimensional or duplicate-heavy data | Dimensionality, duplicate rate | [Schubert et al. 2017, "DBSCAN Revisited, Revisited", ACM TODS 42(3):19, §4.1](https://www.ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf) (PDF pages 11–12 read) |
| DBSCAN eps | ε "should be chosen as small as possible". Use the sorted k-distance plot (Ester et al. 1996, §4.2) with k = 2·dim − 1. Domain knowledge (for example 1 km for GPS) is an alternative. Red flags: aim for 1–30% noise; if the largest cluster holds more than 20–50% of clustered points, lower ε or use OPTICS or HDBSCAN. The paper notes the plot often has no clear knee | k-NN distances for every point, sorted | [Schubert et al. 2017, §4.1–4.2](https://www.ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf) |
| HDBSCAN | min_samples defaults to min_cluster_size. Set min_cluster_size to the "smallest size grouping that you wish to consider a cluster". cluster_selection_epsilon is in data units (for example 0.5 to avoid splitting clusters closer than 0.5). Default alpha = 1.0, and "In practice it is best not to mess with this parameter" | None. The user supplies a minimum cluster size | [hdbscan docs, Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) |
| Distance band for spatial weights (GeoDa) | The default bandwidth is the max-min distance (the largest nearest-neighbour distance), shown in the "Specify bandwidth" box, so every observation has at least one neighbour | All nearest-neighbour distances, then their maximum | [GeoDa workbook, Distance-band weights](https://geodacenter.github.io/workbook/4b_dist_weights/lab4b.html) |
| k for k-nearest-neighbour weights (GeoDa) | "The default is 4" | None | [GeoDa workbook](https://geodacenter.github.io/workbook/4b_dist_weights/lab4b.html) |
| Distance band (Esri Spatial Autocorrelation) | With a fixed distance band, the default distance "will ensure that each feature has at least one neighbor". Guidance: "almost all features should have at least eight neighbors" | Maximum nearest-neighbour distance | [Esri, Spatial Autocorrelation](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/spatial-autocorrelation.html) |
| Distance band (Esri Hot Spot) | Fixed distance band is recommended and the default. Features should have about 8 neighbours. Calculate Distance Band From Neighbor Count finds the average distance for 8 neighbours. Results are unreliable with fewer than 30 features | Distance to the 8th nearest neighbour, averaged | [Esri, How Hot Spot Analysis works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.html) |
| Distance band by scale (Incremental Spatial Autocorrelation) | The default start is the maximum nearest-neighbour distance. The default increment is the smaller of the average nearest-neighbour distance and (Td − B)/I. Moran's I z-score is computed per band. Peaks mark "distances where spatial processes promoting clustering are most pronounced". The tool outputs First_Peak and Max_Peak, and peaks "are often appropriate values" for Hot Spot's distance band | Nearest-neighbour distances, plus Moran's I at about 10 bands | [Esri, Incremental Spatial Autocorrelation](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/incremental-spatial-autocorrelation.html) |
| Optimized Hot Spot scale and cell size | The tool picks the scale and aggregation cell size automatically and reports them in messages. See (c) above | Point pattern | [Esri, Optimized Hot Spot Analysis](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/optimized-hot-spot-analysis.htm) |
| IDW power (Esri) | Default power 2. Higher power gives more local detail, and 30 is "extremely large and thus of questionable use". The optimal power is where the minimum mean absolute error is lowest, through cross-validation. Search radius is variable (a point count, optionally with a maximum distance) or fixed (a distance plus a minimum count). The page does not state the numeric defaults | Cross-validation error, for optimisation only | [Esri, How IDW works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/how-idw-works.html) |
| IDW (GDAL gdal_grid) | `invdist`: power 2.0, smoothing 0, radius1 and radius2 0 (whole point set), max_points 0 (all). `invdistnn`: power 2.0, radius 1.0, max_points 12, min_points 0 | None | [GDAL, gdal_grid](https://gdal.org/en/stable/programs/gdal_grid.html) |
| Kriging variogram (PyKrige) | Default variogram_model "linear", nlags 6, weight False. Without variogram_parameters, the model is fitted to the data by "soft" L1-norm minimisation | Empirical semivariogram in 6 lag bins | [PyKrige OrdinaryKriging](https://geostat-framework.readthedocs.io/projects/pykrige/en/latest/generated/pykrige.ok.OrdinaryKriging.html) |
| Kriging variogram (Esri Empirical Bayesian Kriging) | EBK "automatically calculates these parameters through a process of subsetting and simulations". The default subset size is 100 points, with 100 simulated semivariograms per subset. The user still picks the model type and transformation | Local subsets, plus repeated REML fits | [Esri, What is EBK](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geostatistical-analyst/what-is-empirical-bayesian-kriging-.html) |
| Class breaks: Jenks (natural breaks) | Breaks "best group similar values together and maximize the differences between classes". The breaks are data-specific, so they are not comparable across maps. Defined interval must yield at least 3 classes | Sorted values | [Esri, Data classification methods](https://doc.esri.com/en/arcgis-pro/latest/help/mapping/layer-properties/data-classification-methods.html) |
| Class breaks: Fisher–Jenks versus Jenks–Caspall | "Fisher-Jenks is guaranteed to produce an optimal result", while Jenks–Caspall may not be | Sorted values | [GeoLinter, arXiv 2310.13707](https://arxiv.org/pdf/2310.13707) (search snippet) |
| Class breaks: head/tail | Split at the mean, then recurse on the head while the head stays a minority (about a 40% threshold). Meant for heavy-tailed data. Jiang, *Professional Geographer* 65(3):482–494, 2013 | Mean of successive heads | [classInt head/tails vignette](https://stat.ethz.ch/CRAN/web/packages/classInt/vignettes/headtailsR.html); [Wikipedia](https://en.wikipedia.org/wiki/Head/tail_breaks) (both secondary, search snippet) |
| Number of classes | "approximately four to six distinct classes", from a cartography textbook. Esri gives no count | None | [Essentials of GIS 6.3](https://saylordotorg.github.io/text_essentials-of-geographic-information-systems/s10-03-data-classification.html) (search snippet) |
| LISA permutations and cut-off (GeoDa) | Default 999 permutations, p = 0.05 | None | [GeoDa workbook, Local spatial autocorrelation](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html) |

### Inferences
- The minimum-neighbour distance (the largest nearest-neighbour distance) appears in GeoDa, Esri
  Spatial Autocorrelation and Esri Incremental Spatial Autocorrelation. It is the single most
  reusable "suggested distance" for any weights-based recipe, and one k-NN pass computes it on the
  GPU. Esri's "about 8 neighbours" distance is a second, larger suggestion.
- Incremental Spatial Autocorrelation is itself a parameter sweep shown as a curve with marked peaks.
  It is a direct precedent for showing small multiples or a sensitivity curve over the distance
  band, with "first peak" as the suggestion.
- Schubert et al.'s red flags (noise outside 1–30%, a largest cluster above 20–50%) can become live
  warnings next to an eps slider.
- Esri's "extent / 250" is a simple, documented cell-size default that can be shown as the
  suggested value.

### Gaps
- Silverman (1986) and Scott (1992) one-dimensional rules (h = 0.9·min(σ, IQR/1.34)·n^−1/5 and
  h = 3.49·σ·n^−1/3) were not fetched from primary sources. Treat them as **[unverified]**. The exact
  Esri formula image was not readable. **[unverified]** Its commonly quoted form is
  0.9·min(SD, √(1/ln2)·Dm)·n^−0.2.
- No primary documentation was found for a PySAL KNN default k (the page returned 404), QGIS IDW
  defaults, a gstat `fit.variogram` auto-fit, or a hexbin size rule. kepler.gl's hexbin doc URL
  returned 404.
- ArcGIS's default classification method and class count were not confirmed. The graduated-colors
  page was not fetched.

## (c3) Live preview, direct manipulation and latency evidence

### Takeaway
The research is consistent: added latency of about 500 ms measurably reduces exploration. Progressive
(approximate-first) results perform about as well as instant ones, and far better than blocking
ones. Nielsen's 0.1 s, 1 s and 10 s limits still frame the feedback each delay needs. A GPU library
that updates in under 100 ms lets users feel they manipulate the data directly. Longer runs need a
progressive or sampled preview.

### Cited Findings
- Liu and Heer, "The Effects of Interactive Latency on Exploratory Visual Analysis", IEEE TVCG
  20(12):2122–2131, 2014: "an additional delay of 500ms incurs significant costs, decreasing user
  activity and data set coverage". Higher latency "reduces the rate at which users make observations,
  draw generalizations and generate hypotheses". Early exposure to high latency lowered later
  performance even at low latency (search snippet of the abstract) — [UW IDL paper page](https://idl.uw.edu/papers/latency); [PDF](https://mail.zcliu.cs.umd.edu/latency/latency-InfoVis.pdf)
- Two secondary claims about that paper (blog, search snippet):
  - The paper compares low and high latency and does not set a 500 ms threshold.
  - Panning is more latency-sensitive than zooming.

  — [samw0627 blog, "How fast is fast enough?"](https://samw0627.github.io/blog/2018/09/interaction-speed.html)
- Zgraggen, Galakatos, Crotty, Fekete and Kraska, "How Progressive Visualizations Affect
  Exploratory Analysis", IEEE TVCG 23(8):1977–1987, 2017. Users "perform equally well with either
  instantaneous or progressive visualizations" on insight rate and coverage. Blocking visualisations
  hurt both (search snippet of the abstract) — [HAL hal-01377896](https://hal.archives-ouvertes.fr/hal-01377896)
- Nielsen (1993, *Usability Engineering* excerpt):
  - 0.1 s feels instantaneous and needs no feedback.
  - 1 s keeps the flow of thought, but the user loses the sense of working directly on the data.
  - 10 s is the limit of attention.
  - Over 10 s, use a percent-done bar. For 2–10 s, a busy cursor with a changing number is enough.
  - If the total is unknown, show running progress, such as naming each item as it is processed.

  — [NN/g, Response Times: The 3 Important Limits](https://www.nngroup.com/articles/response-times-3-important-limits/)
- CARTO Builder widgets come in 7 types: Formula, Category, Pie, Histogram, Range, Time Series and
  Table.
  - In viewport mode, "the data gets updated when the viewport extent changes". Global mode
    ignores the viewport and is the default.
  - Cross-filtering lets one widget filter the map and the other widgets.
  - "Clear" and "Clear all" are offered. SQL parameters have a separate "Reset all".

  — [CARTO docs, Widgets](https://docs.carto.com/carto-user-manual/maps/widgets)
- Power BI narratives update on cross-filter: "A summary is dynamic. It automatically updates the
  generated text and dynamic values when you cross-filter" — [Microsoft Learn, Smart narrative](https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-smart-narrative)
- jamovi recomputes output live as options change. See (b) above.

### Inferences
- Design target: under 100 ms per parameter change for slider and drag interactions, which Nielsen
  calls direct manipulation. When a full run exceeds about 1 s (large rasters, networks), show a
  progressive or sampled preview at once and the full result after. Zgraggen et al. found this
  does not hurt insight.
- Avoid introducing users to slow interactions first. Liu and Heer found that early slowness
  carries over and depresses later behaviour.
- CARTO's split between "Clear filter" and "Reset parameters" is a precedent for separate reset
  affordances: reset to suggested versus clear selection.

### Gaps
- Not fetched or documented with sources this session:
  - kepler.gl radius and aggregation sliders. The docs URL returned 404.
  - Felt analysis controls.
  - Mapbox Studio.
  - Lightroom live sliders.
  - Observable Inputs.
  - Grafana variables.
- No controlled study was found of on-map drag handles (radius rings) versus numeric inputs for
  analysis parameters.

## (d2) Result explanation from primary pages

### Takeaway
Esri explains results in layers:
- per-feature z, p and a confidence bin
- an FDR option for multiple testing and spatial dependence
- explicit reliability preconditions (at least 30 features, at least 1 neighbour, about 8
  neighbours)
- an HTML report linked from the messages

Anselin's GeoDa workbook is the strongest primary source of interpretation cautions. Pseudo
p-values are not standard significance. A 0.05 cut-off gives many false positives. Significant
locations are "cores", not whole clusters. Power BI's narrative caps how many statements it
generates and updates them on cross-filter. Tableau Pulse grounds its LLM summaries in templated
statistical facts.

### Cited Findings
- **Esri Hot Spot (Gi\*):**
  - A larger positive z means more intense clustering of high values (hot spots); a smaller
    negative z means cold spots.
  - The output has z, p and Gi_Bin, and the map is drawn by Gi_Bin.
  - With FDR, "statistical significance is adjusted to account for multiple testing and spatial
    dependency".
  - Cautions: "Results aren't reliable with less than 30 features". "All features should have at
    least one neighbor". "No feature should have all other features as neighbors". "Features
    should have about eight neighbors each".

  — [Esri, How Hot Spot Analysis works](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.html)
- **Esri Spatial Autocorrelation:**
  - It returns five values: index, expected index, variance, z and p.
  - The report is "An HTML file with a graphical summary of results", reached from a clickable
    path in the messages.
  - Default distances ensure each feature has at least one neighbour.

  The report's exact wording was not available — [Esri, Spatial Autocorrelation](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/spatial-autocorrelation.html)
- **GeoDa LISA (Anselin workbook):**
  - Cluster map categories: High-High (dark red), Low-Low (dark blue), Low-High (light blue) and
    High-Low (light red). Non-significant locations are grey.
  - Defaults are 999 permutations and p = 0.05.
  - "this notion of significance is not the standard one, and should not be interpreted that way".
    A mechanical 0.05 cut-off "is definitely not the proper way to proceed". Use sensitivity
    analysis across cut-offs.
  - Bonferroni and FDR need many permutations. With too few, Bonferroni may find nothing.
  - Significant locations are "not the actual clusters, but the cores of a cluster". Select the
    cores and their neighbours to judge extent.
  - A separate significance map shades by p-value.

  The fetched text did not confirm whether counts appear in the legend — [GeoDa workbook, Local spatial autocorrelation](https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html)
- **Power BI Narrative visual** (doc dated 2026-09-25):
  - There are two modes. Copilot mode is AI-generated and needs a Copilot licence. Custom mode is
    auto-generated text plus user-defined dynamic values.
  - It produces "up to four summaries per visual and up to 16 per page" and "avoids generating the
    same kind of summaries for different visuals".
  - A hidden-summary symbol marks values with no result under the current filters.
  - An optional smart-narrative icon on a visual's header gives an on-demand text summary, for
    accessibility.
  - There is a long list of unsupported visuals and sources, including map visuals with
    non-aggregated latitude and longitude.

  — [Microsoft Learn, Smart narrative](https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-smart-narrative)
- **Tableau Pulse:** it detects drivers, trends, contributors and outliers per metric. "Insight
  summaries are generated using natural language grounded in statistical truths" and appear at the
  top of digests (search snippet) — [Tableau Help, Insights platform and insight types](https://help.tableau.com/current/online/en-us/pulse_insights_platform_insight_types.htm)
- Two further claims about Pulse, both from a Salesforce blog (search snippet):
  - The LLM summarises templated insight facts, not raw numbers, to reduce factual errors.
  - Each summary covers the top three insights.

  — [Salesforce blog, Tableau Pulse](https://www.salesforce.com/blog/tableau-pulse/)
- **HCI evidence on text with charts** (search snippets, primary pages not fetched):
  - Kim, Setlur and Agrawala (CHI 2021) found readers take as the takeaway the feature emphasised
    by both chart and caption. When the caption stresses a low-prominence feature, readers rely on
    the chart — [Stanford HCI](https://hci.stanford.edu/publications/paper.php?id=429)
  - A 2022 study found captions help readers remember the features they describe — [arXiv 2205.01263](https://arxiv.org/pdf/2205.01263)
  - A 2024 tool study reports better comprehension, with users reluctant to fully trust automated
    output — [arXiv 2409.17494](https://arxiv.org/html/2409.17494v1)
- **jamovi and JASP annotations:**
  - jamovi has editable results annotations, and keyboard shortcuts for them are "limited" (Dec
    2025) — [jamovi a11y](https://jamovi.org/a11y.html)
  - The jamovi module API has `setNote` for table footnotes — [dev.jamovi.org notices](https://dev.jamovi.org/api/notices)
  - A forum post says JASP's AI-generated annotations look identical to user text inside .jasp
    files (search snippet) — [CogSci forum](https://forum.cogsci.nl/discussion/comment/32164)

### Inferences
- A recipe's explanation can be assembled from deterministic templated facts (counts per class,
  chosen parameters, the share significant after FDR). An optional LLM rewording can sit on top, as
  Pulse does. That keeps numbers correct.
- Caption research suggests the generated sentence should name the same feature the map emphasises.
  For example: "12 high-high cores in the north-east". A secondary detail the map does not show
  should not lead the sentence.
- Legends and text should adopt Anselin's wording that significant cells are cluster cores, not
  clusters. When FDR or Bonferroni is used with few permutations, add a warning.
- Esri's reliability preconditions (n ≥ 30, at least 1 neighbour, about 8 neighbours, no feature
  neighbouring all others) are cheap checks to show as assumption badges, in the style of jamovi.

### Gaps
- Esri's "Interpreting OLS results" page (doc.esri.com) returned empty content twice. GWR and OLS
  diagnostics wording (VIF > 7.5, Koenker, Jarque–Bera) remains **[unverified]**.
- Esri pop-up charts on result layers were not researched.
- No study was found that measures over-trust in generated narratives for statistical maps.

## (d3) Onboarding and empty states

### Takeaway
NN/g's two relevant articles say:
- Empty states should communicate status, teach in context, and give direct paths to key tasks,
  including demo data.
- Deck-of-cards tutorials do not improve performance; contextual tips and brief interactive
  walkthroughs for novel workflows work better.

kepler.gl's own first screen already offers "Try sample data" next to upload and URL. Power BI's
template-app guidance tells authors to ship sample data so users can start "in a click".

### Cited Findings
- Kate Kaplan, NN/g, 19 Sep 2021, "Designing Empty States in Complex Applications: 3 Guidelines":
  - Communicate system status: "Totally empty states cause confusion about how and whether the
    system is working".
  - Provide learning cues: empty states are "an opportunity to provide contextual help".
  - Provide "direct pathways for getting users started with key tasks". Its example is Loggly's
    demo data, which lets users explore before adding their own sources.

  — [NN/g, Empty states](https://www.nngroup.com/articles/empty-state-interface-design/)
- Alita Kendrick, NN/g, 21 Jun 2020, on mobile-app onboarding:
  - "we don't recommend deck-of-cards onboarding". NN/g research found these tutorials did not
    improve task performance.
  - "it's often more effective to train users by showing them tips in context".
  - Instructional overlays are "nice-to-have and not need-to-have".
  - Interactive walkthroughs are for novel workflows and should be brief.

  — [NN/g, Mobile-app onboarding](https://www.nngroup.com/articles/mobile-app-onboarding/)
- kepler.gl's first screen offers Local files, From URL, and "Try sample data" in the top right of
  the initial prompt. Sample maps are recommended for new users to see how configurations work.
  Data stays in the browser — [kepler.gl docs, Add data to the map](https://docs.kepler.gl/docs/user-guides/b-kepler-gl-workflow/a-add-data-to-the-map)
- Microsoft's tips for template-app authors:
  - "Make sure your app includes sample data to get everyone started in a click".
  - Keep the .pbix under about 10 MB.
  - Enable custom connection and parameter configuration by the installer.
  - Consider several apps for separate scenarios.

  — [Microsoft Learn, Template apps overview](https://learn.microsoft.com/en-us/power-bi/connect-data/service-template-apps-overview)
- Datawrapper lets users build without signing in and asks for sign-up only to publish. See (d)
  above.

### Inferences
- Since kepler.gl already has a "Try sample data" entry point, the cheapest onboarding step is to
  pair each sample with a pre-run recipe, so the first map is already an analysis. NN/g's "direct
  pathway" guideline supports this, as does Power BI's "started in a click".
- Prefer in-context tips at the parameter (an info link to the docs for the operation) over an
  upfront tour.

### Gaps
- Not fetched:
  - CARTO demo-data onboarding
  - Felt onboarding
  - Observable templates
  - Tableau Public sample workbooks
- No source was found on in-app tool-to-documentation deep links beyond NN/g's general advice.

## (e2) Template leads (second pass)

### Cited Findings
- **Power BI template apps:**
  - Partners build them as parameterised .pbix files and publish through Partner Center to the
    Power BI Apps marketplace and AppSource.
  - Installers connect their own data through parameters and custom connections, and setup can be
    automated.
  - Limits: exactly one semantic model, at most 20 reports, no live or on-premises sources, and no
    Any, Date or Binary query parameters.
  - Doc dated 2025-11-04.

  — [Microsoft Learn, Template apps overview](https://learn.microsoft.com/en-us/power-bi/connect-data/service-template-apps-overview)
- **ComfyUI:**
  - The Templates browser opens from the sidebar Templates icon or Workflow → Browse Workflow
    Templates.
  - It holds ComfyUI's natively supported model workflows plus example workflows supplied by
    custom nodes, filed under a category named after each node.
  - Templates ship as a separately versioned package, `comfyui-workflow-templates`, maintained in a
    workflow_templates repository that accepts pull requests.
  - Source: search snippets of official docs, not fetched.

  — [ComfyUI docs, Templates](https://docs.comfy.org/interface/features/template); [ComfyUI docs, custom-node workflow templates](https://docs.comfy.org/custom-nodes/workflow_templates); [PyPI comfyui-workflow-templates](https://pypi.org/project/comfyui-workflow-templates/)
- **ArcGIS Online:**
  - No "analysis template" feature for feature analysis was found.
  - The nearest equivalents are Map Viewer ModelBuilder analysis models, which chain data and tools
    and can be saved and shared.
  - Raster function templates are explicitly "used to create a specific visualization or analysis
    that can be repeated" and are listed under My content (search snippet).

  — [ArcGIS Online, Get started with ModelBuilder](https://doc.arcgis.com/en/arcgis-online/analyze/get-started-modelbuilder-mv.htm); [Raster function template](https://doc.esri.com/en/arcgis-online/latest/analyze/raster-function-template-mv-ra.html)

### Inferences
- ComfyUI's model shows how a template gallery can be versioned without breaking. Templates ship as
  their own package, separate from the engine, and extension-supplied examples are grouped by the
  contributing node. This maps onto shipping recipes and scenario packs as separately versioned
  content.

### Gaps
- Earth Engine Apps, Looker Blocks, Hex and Observable templates, Dataiku Solutions, Felt and Atlas
  were not researched for lack of calls.
