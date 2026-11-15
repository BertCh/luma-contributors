# Semantic types for GIS operations and deriving analysis workflows from input types and questions (as of November 2026)

Verification legend. "[verified]" means read in the primary text during this session (the full CCD JOSIS 2020 text, the full Kruiger et al. 2021 text through PMC, the openEO API 1.2.0 specification, and the GeoAnalystBench arXiv abstract). "[abstract/secondary]" means confirmed only from an abstract, publisher record or secondary summary. "[not verified]" means stated from prior knowledge, with a URL or DOI given for checking. The report writer should not present an "[not verified]" item as a confirmed number.

## 1. Type systems used for GIS tools (core concepts, CCD ontology, layer types, measurement levels)

### Takeaway
The best-developed scheme is the Core Concept Data types (CCD) ontology (Scheider, Meerlo, Kasalica and Lamprecht, JOSIS 2020). It types a dataset along three independent axes: core concept (field, object or event quality), geometric layer type (point, line, region, tessellation, vector, raster) and measurement level (nominal, boolean, ordinal, interval, ratio, count). Named combinations such as Lattice, Coverage, PointMeasures and Contour are defined over those axes, and tools are annotated with input→output signatures over these classes. The whole scheme is small enough to hand-encode. It maps closely onto a 15-kind ontology that gains a measurement-level axis and a field/object axis.

### Cited Findings
- Kuhn (2012) proposed 10 core concepts of spatial information for transdisciplinary research: location, neighbourhood, field, object, network, event, granularity, accuracy, meaning and value. Citation: IJGIS 26(12), 2267–2276, DOI 10.1080/13658816.2012.722637. [abstract/secondary] — [UCSB Core Concepts project](https://spatial.ucsb.edu/research/core-concepts-spatial-information); [McGill summary](https://rose.geog.mcgill.ca/wordpress/?p=4848)
- The UCSB project page says the concept list changed after Kuhn 2011/2012, and that "meaning" and "value" were dropped because they lack solid theories that would make them core to spatial analysis. [abstract/secondary] — [UCSB Core Concepts](https://spatial.ucsb.edu/research/core-concepts-spatial-information)
- CCD citation: Scheider, Meerlo, Kasalica and Lamprecht, "Ontology of core concept data types for answering geo-analytical questions", JOSIS 20 (2020) 167–201, DOI 10.5311/JOSIS.2020.20.555, CC BY 3.0. The OWL vocabulary is published at http://geographicknowledge.de/vocab/CoreConceptData.ttl. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125); [DOAJ](https://doaj.org/article/a12e8e49c0ae49bf82ef4d55db463a98)
- The core concept is modelled as an attribute (quality) class, not as a dataset class: `FieldQ ⊑ Attribute; EventQ ⊑ Attribute; ObjectQ ⊑ Attribute` (Axiom 3). A field quality is, for example, temperature. An object quality is, for example, building size or the average income of a municipality. Because the concept axis is orthogonal to geometry, a temperature field can be stored as point data or as region data. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Layer-type axioms: `Raster ⊑ Tessellation ⊑ RegionDataSet` (Axiom 1). A vector tessellation is `Vector ⊓ Tessellation`, and PointDataSet and LineDataSet are disjoint subtypes of Vector. A tessellation is a tiling that jointly covers the plane without overlap, such as Thiessen polygons, TINs or administrative units. Cell-centre point rasters are explicitly not tessellations. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Measurement levels form a subsumption chain: `CountA ⊑ RatioA ⊑ IntervalA ⊑ OrdinalA ⊑ NominalA ⊑ Attribute` (Axiom 4), with `BooleanA ⊑ NominalA` (Axiom 5). Count is treated as a ratio scale on which the meanings of both 0 and 1 are fixed. The paper justifies the chain on the grounds that measurement level "largely determine[s] the type of numerical GIS operation that can be applied". [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Extensive versus intensive is named as a relevant distinction: extensive values depend on the size of the support region and intensive values do not. The CCD paper says it does not focus further on this distinction. The follow-on Kruiger et al. paper adds ERA (extensive region attribute) and IRA (intensive region attribute) classes. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125); [Kruiger et al. 2021, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)
- Free combination of the three axes gives 6 layer types × 3 concepts × 6 measurement levels = 108 types. The authors keep only the "meaningful and practically relevant" combinations, shown as a matrix in Fig. 7 with deprecated cells left empty, and they checked consistency with standard OWL reasoners. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Named composite types (definitions 7–15) [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125):
  - Field side: `PointMeasures ≡ FieldQ ⊓ ∃ofDataSet.PointDataSet`, for example air-quality sensors. `LineMeasures ≡ FieldQ ⊓ ∃ofDataSet.LineDataSet`, for example isolines. `FieldRaster ≡ FieldQ ⊓ ∃ofDataSet.Raster`. `Coverage ≡ FieldQ ⊓ ∃ofDataSet.Tessellation`, for example a land-use polygon map. Contour is a coverage refined by measurement level.
  - Object side: ObjectVector, existence vectors and rasters (Boolean "here is X" layers), and `Lattice ≡ ObjectQ ⊓ ∃ofDataSet.Tessellation`. A Lattice is "statistical data summarized by pre-defined statistical regions", in the sense of lattice data in spatial statistics.
  - The field/object difference on the same tessellation (Coverage versus Lattice) is self-similarity. A coverage value holds at every point inside its polygon. A lattice value, such as average elevation within Amsterdam, does not.
- Tool annotation: signatures link CCD input and output classes to tool identifiers in a tool description file (github.com/simonscheider/SemanticPipelines/blob/master/ToolDescription.ttl), using ArcGIS Pro tools as examples. Table 1 lists operation types (Local Map Algebra, Focal Statistics, Zonal Statistics, Boolean Reclassify, Select by attribute, Raster to Polygon, Create Contours, Polygon to Raster, Overlay, Point Interpolation, …), each with input, optional second input and output types. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Parameters that change the type are modelled as overloaded tools. Focal Statistics is overloaded with Mean, Variety and Sum. Zonal Statistics is overloaded with Mean, Majority and Sum. Zonal Mean requires interval scale, Zonal Median ordinal and Zonal Variety nominal. Spatial Join is overloaded on Join_one_to_one with Count, Mean and Sum. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125); Kruiger et al. call these overloaded "subtools" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)
- Worked type-level derivations from the paper (Fig. 9) [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125):
  - Object points are aggregated to a lattice by Spatial Join.
  - Lattice to lattice is possible only through areal interpolation.
  - Nominal coverage → FieldRaster (nominal) → Reclassify → ExistenceRaster → distance raster (ratio) → Zonal Mean → Lattice.
  - A noise contour map is ordinal, so it "can only be aggregated to a lattice via a Zonal Median".
- Stated limitations: network and event were not yet in the type system, and the paper calls for extending it with those concepts to cover accessibility and trajectories. The type system "does not distinguish different kinds of transformations between the same core concept data types", which motivated the later transformation algebra. The authors expect "a substantial amount of manual annotations is unavoidable". [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Kruiger et al. list the CCD layer types used in synthesis: ObjectDS, LatticeDS, FieldDS, CoverageDS, ContourDS, PointMeasuresDS and FieldRasterDS, plus the geometry types PointDataSet, LineDataSet, RegionDataSet, Tessellation, Raster, Vector and VectorTessellation. Missing class combinations were generated programmatically and reasoned over with OWL2-RL. [verified] — [Kruiger et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)

### Inferences
- A rule-based implementation can adopt CCD's three-axis idea without OWL by attaching facets to each data kind: `concept ∈ {field, object, event, network}`, `geometry ∈ {point, line, region, tessellation, raster}`, and a per-attribute `level ∈ {boolean, nominal, ordinal, interval, ratio, count}` plus `extensive|intensive`. Subsumption can be implemented as plain "accepts any subtype" checks on ordered enumerations, such as `count ≥ ratio ≥ interval ≥ ordinal ≥ nominal`.
- The existing kinds "polygons" and "surface" are each ambiguous between Coverage and Lattice, and the CCD paper shows that the distinction decides whether areal interpolation or zonal statistics applies. This is the single most useful facet to add.
- Overloading tools by parameter (Zonal-Mean versus Zonal-Median) is how CCD handles parameters that change the output type. A JSON spec can express the same thing with per-parameter-value output-type rules, without splitting operators.

### Gaps
- The full text of Kuhn (2012) was not retrieved, so the definitions of each concept are not verified beyond the list.
- "Kuhn and Ballatore" (for example "Designing a language for spatial computing", AGILE 2015) was not retrieved.
- The total count of annotated tools in ToolDescription.ttl was not checked. Neither paper states it: Kruiger et al. show a 22-function excerpt.

## 2. How workflows are synthesised or validated from types (APE, loose programming, CCT, question-to-workflow)

### Takeaway
APE (Automated Pipeline Explorer) encodes tool signatures, a type taxonomy and SLTL (semantic linear-time logic) constraints as a SAT problem bounded by workflow length. It enumerates the n shortest type-correct workflows from given input types to a goal output type. In Kruiger et al. (2021), CCD typing raised the share of meaningful synthesised workflows from 14% to 99%. Redundant steps remained the dominant soft error (70% of correct workflows), and there was no ranking beyond ordering by length. CCT later moved from data types to an algebra of conceptual transformations, used to annotate and retrieve existing workflows from parsed questions.

### Cited Findings
- Kruiger, Kasalica, Meerlo, Lamprecht, Nyamsuren and Scheider, "Loose programming of GIS workflows with geo-analytical concepts", Transactions in GIS 25(1) 424–449, 2021, DOI 10.1111/tgis.12692. Loose programming means synthesising programs from loose specifications. [verified] — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)
- CCD synthesis setup (CCD paper): competency questions were rewritten as SLTL formulas of the form `in <types> ∧ out <type> [∧ use tool:X]`. For example, Q1 is `in ccd:ObjectPoint, ccd:VectorTessellation ∧ out ccd:CountALattice`, and Q3 is `in ccd:CoverageNominalA, ccd:VectorTessellation ∧ out ccd:Lattice ∧ use tool:Distance`. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- To speed search, APE assumes that subclass relations are jointly exhaustive and that leaf classes are disjoint, so `A ⇒ A1 ∨ A2`. This lets a goal be stated at an abstract type such as FieldQ. Up to 50 workflows of at most 4 tool applications were synthesised per question, ordered by size. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Two simplifying assumptions were made: only relevant data sources are present at the start, and input data are unambiguously identifiable by type. Quality was judged by an expert, not by execution. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- The CCD paper says that adding constraints on tool kind or tool order would raise precision, but that the preliminary study prioritised recall over precision. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Kruiger et al. settings: up to n = 20 workflows of maximum length k = 8, because longer or more workflows "only increased redundancy". APE uses SLTL synthesis inspired by PROPHETS with a SAT encoding (APE on GitHub: sanctuuary/APE). No ranking was described beyond enumeration order, and runtime was not reported. [verified] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/); [APE repo](https://github.com/sanctuuary/APE)
- Kruiger et al. compared CCD against a baseline "BDT" (basic data types, geometry only) ontology on 5 Amsterdam elderly-liveability questions. [verified] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)
  - Workflows found: BDT 20 per question (100 in total). CCD found Q1 = 20, Q2 = 8, Q3 = 20, Q4 = 4 and Q5 = 20 (72 in total). Fewer workflows were found where the types left few legal paths.
  - Main study, share correct (no hard error): BDT 14/100 (14%) and CCD 71/72 (99%).
  - Hard errors: BDT had 17 signature errors and 86 semantic-imprecision errors. CCD had 0 signature errors and 1 semantic-imprecision error.
  - Soft errors among correct workflows: BDT had 3 redundancy (21%) and 7 data-quality (50%). CCD had 50 redundancy (70%) and 21 data-quality (30%).
  - Preliminary study: BDT 18/100 correct and CCD 68/100 correct. CCD's errors were 6 signature and 27 imprecision, and 83% of its correct workflows were redundant.
  - Error taxonomy. Hard errors are signature errors (the tool is misapplied or not executable) and semantic imprecision (the answer is meaningless). Soft errors are redundancy (unneeded tool applications) and data quality (needless loss of resolution).
  - The text says the CCD hard-error rate is "below 1%", while the table gives 1/72, which is about 1.4%. Precision is the share of correct workflows retrieved. Recall was not measured, and a single expert judged the workflows at design time.
- APE framework documentation (ape-framework.readthedocs.io): domain models are a tool annotation JSON plus an OWL taxonomy. Constraints come from templates such as "use tool X", "do not use", "X followed by Y" and "use type". Results are the shortest solutions in increasing length. [not verified] — [APE docs](https://ape-framework.readthedocs.io/); Kasalica and Lamprecht, "APE: A command-line tool and API for automated workflow composition", ICCS 2020, DOI 10.1007/978-3-030-50436-6_34 [not verified]
- CCT: Steenbergen, Top, Nyamsuren and Scheider, "Procedural metadata for geographic information using an algebra of core concept transformations", JOSIS 27 (2023), DOI 10.5311/JOSIS.2023.27.268. CCT has "a type hierarchy which models core concepts as relation types" and basic transformations given by function signatures over those types. Type inference "compil[es] algebraic tool descriptions" to produce machine-readable workflow metadata. That metadata is used to retrieve workflows from task descriptions derived from geo-analytical questions, across raster and vector implementations. [abstract/secondary] — [JOSIS](https://josis.org/index.php/josis/article/view/268); [UCC record](https://research.ucc.ie/en/publications/procedural-metadata-for-geographic-information-using-an-algebra-o/)
- Question grammar: Xu, Nyamsuren, Scheider and Top, "A grammar for interpreting geo-analytical questions as concept transformations", IJGIS 37(2) 276–306, 2023, DOI 10.1080/13658816.2022.2077947. A context-free grammar parses questions into core concepts and their functional roles, which indicate how each concept is transformed and used in a workflow. It was developed on 309 GIS-related questions and tested on 134 independent questions with workflows, reporting "high precision and recall" against a gold standard of concept transformations. [abstract/secondary] — [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC9851665); [UCC record](https://research.ucc.ie/en/publications/a-grammar-for-interpreting-geo-analytical-questions-as-concept-tr/)
- Corpora: the GeoAnQu corpus of geo-analytical questions was introduced in Xu et al., AGILE GIScience Series 1 (2020). Nyamsuren, Xu, Top, Scheider and Steenbergen, "Semantic complexity of geographic questions" (AGILE GIScience Series 4, 2023), compared five question corpora with the parser. They found GeoAnQu questions need more, and more diverse, geo-analytical operations than the other four corpora. [abstract/secondary] — [AGILE 2020](https://agile-giss.copernicus.org/articles/1/23/2020/); [AGILE 2023](https://agile-giss.copernicus.org/articles/4/10/2023/)
- Prior semantic composition work cited by the CCD paper includes rule sets that identify tool sources starting from a goal concept, and functional taxonomies of GIS operations. Specific attributions here were not verified. Commonly cited works [not verified]:
  - Lutz (2007), "Ontology-based descriptions for semantic discovery and composition of geoprocessing services", GeoInformatica 11, DOI 10.1007/s10707-006-7635-9.
  - Lemmens et al. (2006), "Integrating semantic and syntactic descriptions to chain geographic services", IEEE Internet Computing, DOI 10.1109/MIC.2006.106.
  - Hofer, Mäs, Brauner and Bernard (2017), "Towards a knowledge base to support geoprocessing workflow development", IJGIS 31(4), DOI 10.1080/13658816.2016.1227441.
  - Source: [JOSIS PDF, related work](https://josis.org/index.php/josis/article/download/125/125)
- Galaxy tool recommendation is a data-driven alternative to type-driven synthesis. Kumar et al. (GigaScience 2021) trained a gated recurrent network on European Galaxy workflows and report about 98% mean top-1 accuracy for next-tool recommendation. Kumar et al. (BMC Bioinformatics 2023, DOI 10.1186/s12859-023-05573-w) use a transformer trained on about 60,000 Galaxy Europe workflows, reduced to about 798,000 linear tool sequences (about 353,000 unique). The transformer converges twice as fast and generalises better than RNN, CNN and DNN baselines. The two papers use different metrics. [abstract/secondary] — [GigaScience 2021, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7786169/); [BMC 2023, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10680333/); [Galaxy news](https://galaxyproject.org/news/2023-11-29-transformer-paper/)

LLM-era results:
- LLM-Geo: Zhenlong Li and Huan Ning, "Autonomous GIS: the next-generation AI-powered GIS", IJDE 16(2) 4668–4686, 2023. The GPT-4-based prototype generated and executed a solution graph for three case studies, and the paper reports correct results (numbers, graphs, maps) in all three. This is a demonstration, not a benchmark. [abstract/secondary] — [arXiv 2305.06453](https://arxiv.org/abs/2305.06453v4); [PSU record](https://pure.psu.edu/en/publications/autonomous-gis-the-next-generation-ai-powered-gis/); code at github.com/gladcolor/LLM-Geo
- GIS Copilot (QGIS): Akinboyewa, Li, Ning and Lessani, IJDE 2025, DOI 10.1080/17538947.2025.2497489, arXiv 2411.03205. The paper reports high success on basic and intermediate tasks and remaining difficulty on advanced ones. [abstract/secondary] — [arXiv](https://arxiv.org/abs/2411.03205v1); [PSU news](https://www.psu.edu/news/earth-and-mineral-sciences/story/mapping-new-frontier-ai-integrated-geographic-information-systems)
  - Secondary summaries disagree on the per-level rates. One gives roughly 92%, 83% and 75% (basic, intermediate, advanced) over 60, 30 and 20 tasks. Another gives more than 90%, 80% and 75%.
  - The PSU press release gives about 86% overall across more than 100 tasks and says human supervision is still needed.
  - Lower rates are attributed to multi-step procedures.
- GeoAnalystBench (Transactions in GIS 2025, arXiv 2509.05881) has 50 expert-validated Python geoprocessing tasks. It scores workflow validity, structural alignment, semantic similarity and CodeBLEU. [verified abstract] — [arXiv](https://arxiv.org/abs/2509.05881v1)
  - GPT-4o-mini scored 95% validity and CodeBLEU 0.39. DeepSeek-R1-7B scored 48.5% validity and CodeBLEU 0.272.
  - Smaller models "often generate incomplete or inconsistent workflows". Spatial relationship detection and optimal site selection were hardest for all models.
- GeoBenchX (Krechetova and Kochedykov, arXiv 2503.18129) runs a tool-calling agent with 23 geospatial tools (GeoPandas, Rasterio, GDAL and others) on solvable and deliberately unsolvable tasks, judged by an LLM panel. [abstract/secondary] — [arXiv](https://arxiv.org/abs/2503.18129v1); figures from [emergentmind summary](https://www.emergentmind.com/papers/2503.18129)
  - v1 aggregator figures (solvable match / unsolvable reject): Sonnet 3.5 53% / 37%, GPT-4o 44% / 59%, Gemini 2.0 43% / 46%, Haiku 3.5 43% / 20%, Sonnet 3.7 50% / 14%.
  - Claude models were stronger on solvable tasks, and OpenAI models were better at rejecting unsolvable ones. v2 (October 2025) may differ.
- GISclaw, a later preprint agent on the GeoAnalystBench tasks, reports up to 100% success and 97% mean success over 3 runs. This is a system-level result, not a model baseline. [abstract/secondary] — [arXiv 2603.26845](https://arxiv.org/html/2603.26845)

### Inferences
- For about 280 operators and 15 kinds, full SAT synthesis is not needed. A bounded breadth-first search over typed ports (depth 3 or 4), filtered by facet compatibility and an optional "must use" or "must end in" constraint, reproduces APE's behaviour for short workflows.
- The Kruiger results show that typing removes almost all non-executable and meaningless chains, but not redundancy. The implementation should therefore deduplicate and prune: reject chains in which a step's output is converted back to an earlier kind, and prefer the shortest chain.
- The data-quality error class (needless loss of resolution, such as raster round-trips) suggests a cost per conversion step.
- The absence of ranking in APE, and the redundancy it produced, argue for curated pre-configured workflows as the primary user experience. Synthesis then serves as a validator and suggester rather than the main source of workflows.
- The LLM benchmarks show multi-step workflows degrade, with advanced tasks at about 75% in GIS Copilot and agents in GeoBenchX matching only 43–53% on solvable tasks. Rejecting unsolvable requests is a separate weak skill. A typed validator that says "this cannot be answered with your data" is a recognised gap that a rule engine can fill.
- The Galaxy approach (learn the next likely tool from real workflow logs) is a cheap complementary ranking signal once usage telemetry or curated workflows exist.

### Gaps
- APE's exact SAT encoding, its solving times for GIS domains, and the most recent QuAnGIS outputs were not read in primary text. The QuAnGIS outputs checked for were the github.com/quangis repositories (transformation-algebra, cct, workflow repository) and any end-to-end question-to-workflow retrieval precision figures.
- The CCT paper body (base types, operator count, SPARQL query evaluation) was not retrieved.
- No primary per-level numbers for GIS Copilot were read.
- GeoGPT and GeoAgent were not checked in this session.

## 3. Taxonomies of spatial-analysis questions and tasks (a "what do you want to know?" menu)

### Takeaway
Formal question corpora exist, notably GeoAnQu and the five corpora compared in Nyamsuren et al. 2023. CCD's functional-role grammar implies a menu built on the shape "measure X of objects Y, from field or objects Z". The classic textbook and visual-analytics task typologies were not re-verified in this session and should be treated as background to check.

### Cited Findings
- The CCD competency questions are a ready template of "amount of X per region", "rate of Y per region", "accessibility of Z per region" and "average of field W per region". Each maps to a distinct operator family: spatial join count, areal interpolation, distance then zonal mean, and zonal median. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- The Xu et al. grammar annotates questions with core concepts and functional roles, such as measure, condition, extent and temporal extent. The role list is from prior knowledge; the abstract confirms only "functional roles". [abstract/secondary for existence; role names not verified] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9851665)
- GeoAnQu questions require more numerous and more diverse operations than four other geographic question corpora. [abstract/secondary] — [AGILE 2023](https://agile-giss.copernicus.org/articles/4/10/2023/)
- Background typologies, none checked in primary text this session [not verified]:
  - Esri groups geoprocessing analysis into six categories: Understand where, Measure size, shape and distribution, Determine how places are related, Find the best locations and paths, Detect and quantify patterns, and Make predictions. [Esri](https://www.esri.com/)
  - de Smith, Goodchild and Longley, Geospatial Analysis, which is freely available. [spatialanalysisonline.com](https://www.spatialanalysisonline.com/)
  - Andrienko and Andrienko (2006), Exploratory Analysis of Spatial and Temporal Data, gives an elementary versus synoptic task typology with lookup, comparison and relation-seeking, DOI 10.1007/3-540-31190-4.
  - Dodge, Weibel and Lautenschütz (2008), "Towards a taxonomy of movement patterns", Information Visualization 7(3–4), DOI 10.1057/palgrave.ivs.9500182.
  - Anselin's ESDA sequence: visualise the distribution, then global autocorrelation (Moran's I), then local indicators (LISA), then regression diagnostics and spatial regression specification. Anselin 1995 LISA, Geographical Analysis 27(2), DOI 10.1111/j.1538-4632.1995.tb00338.x. The GeoDa decision tree for spatial regression (Lagrange multiplier tests choosing lag versus error models) is widely reproduced.
  - O'Sullivan and Unwin, Geographic Information Analysis, 2nd edition, Wiley 2010.

### Inferences
- A practical first menu can follow CCD's output-concept framing: "a value per region" (Lattice), "a continuous surface" (Field), "groups or clusters of objects", "paths or reach over a network" and "change over time" (Event). Each can be refined by measurement level. Esri's six categories can serve as user-facing labels over those output types.
- The question-template slots (measure, support region, source data, optional constraint such as "use distance") map one-to-one onto the draft JSON workflow's workflow inputs plus a "must use" filter.

### Gaps
- No published, machine-readable decision tree for choosing a spatial statistic or interpolator was retrieved in this session. The GeoDa regression decision chart and textbook interpolator guidance (de Smith et al.) are the likely sources to check.

## 4. Declarative preconditions (CRS, extensive/intensive, sample size, connected weights) and existing validators

### Takeaway
Semantic-type research encodes only measurement level, field/object and tessellation-ness as preconditions. Practical libraries check a few data conditions at run time, mostly as warnings: libpysal reports islands and disconnected components, and spdep errors on empty neighbour sets unless `zero.policy` is set. No surveyed standard has a vocabulary for statistical preconditions such as projected CRS, minimum n or stationarity.

### Cited Findings
- CCD preconditions are type-level. Zonal Mean requires an interval-or-higher input, Zonal Median ordinal and Zonal Variety nominal. Areal interpolation is the only lattice-to-lattice route. Tessellation is required for Coverage and Lattice. [verified] — [JOSIS PDF](https://josis.org/index.php/josis/article/download/125/125)
- Kruiger et al. distinguish extensive (ERA) from intensive (IRA) region attributes so that synthesis can pick sum-type versus mean-type aggregation. [verified] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983927/)
- libpysal prints a warning by default when a weights object has disconnected components or islands, reporting the component count and the island IDs. `silence_warnings=True` suppresses it; this flag replaced the older `silent_island_warning` and `silent_connected_components`. Repair helpers include `attach_islands`, `nonplanar_neighbors` and `fuzzy_contiguity`. [abstract/secondary, docs pages] — [libpysal W docs](https://pysal.org/libpysal/generated/libpysal.weights.W.html); [weights source](https://pysal.org/libpysal/_modules/libpysal/weights/weights.html); [user guide](https://pysal.org/libpysal/user-guide/weights/weights.html)
- spdep functions such as `nb2listw` and `moran.test` stop on empty neighbour sets unless `zero.policy = TRUE`. A global `set.ZeroPolicyOption` exists, and recent spdep versions also warn about subgraphs (disconnected components). [not verified] — [spdep docs](https://r-spatial.github.io/spdep/)

### Inferences
- A rule-based engine should encode preconditions as named, declarative predicates on input facets, such as `crs: projected`, `attribute.level >= interval`, `attribute.extent: extensive`, `count >= N`, `weights.islands == 0` and `weights.components == 1`. Each predicate should carry a severity: block, warn or suggest a fixing operator. The libpysal and spdep pattern of warn-by-default with explicit opt-out is the established convention.
- Facet checks belong at workflow validation and binding time, which fits the repository's rule to validate once at construction rather than per row.

### Gaps
- No standard (openEO, OGC, CWL) was found with a vocabulary for statistical preconditions.
- Whether GeoDa surfaces island warnings in its user interface was not verified.

## 5. Workflow and process description standards a JSON workflow spec should learn from

### Takeaway
openEO process graphs are the closest model to the draft JSON workflow:
- named nodes with `process_id` and `arguments`;
- arguments bound to literals, `{"from_node"}` or `{"from_parameter"}`;
- exactly one `result: true` node;
- child process graphs for callbacks and user-defined processes;
- JSON Schema draft-07 typing plus a custom `subtype` keyword.

openEO's rules on reserved keys and validation are worth copying almost verbatim. OGC API Processes Part 3 adds nesting, collection inputs and CWL as a deployable format. CWL contributes explicit `format` (ontology IRIs) and provenance through CWLProv and W3C PROV.

### Cited Findings
- openEO process node fields are `process_id` (required), `arguments` (required), `namespace`, `description` and `result` (default false). A process graph is a map of uniquely keyed nodes whose identifiers are scoped to that graph, and circular references are forbidden. [verified] — [openEO API 1.2.0](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Exactly one node per graph must set `result: true`. That node is not necessarily the last one executed. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Argument values can be any JSON value or one of three special objects. A ResultReference `{"from_node": id}` must point to a node in the same graph. A ParameterReference `{"from_parameter": name}` reads a graph parameter. A UserDefinedProcess is an object with `process_graph`. The keys `from_node`, `from_parameter` and `process_graph` are reserved. References can appear inside nested arrays and objects, so back-ends must traverse those fully. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Callbacks: processes such as `apply` take a child process graph, each with its own result node. In the specification's words, "Each child process is simply a user-defined process again." [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Parameter resolution is lexical outward. The engine uses the most specific value found in the current scope or a parent scope, then the parameter's default, and otherwise raises `ProcessParameterMissing`. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Typing is JSON Schema draft-07 plus custom keywords. `subtype` covers values such as `bounding-box`, `date-time` and `datacube`, and a parameter that accepts a general type should also accept its subtypes. `dimensions` describes datacubes. `parameters` and `returns` are used for `subtype: process-graph`. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- Validation beyond the schema requires five checks: exactly one result node, each `process_id` exists in its namespace, arguments match the process schemas, `from_node` targets exist, and return types are compatible with consumers. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- User-defined process identifiers must be unique per user, and clients should warn if one reuses a predefined identifier. [verified] — [openEO API](https://api.openeo.org/1.2.0/#section/Processes/Process-Graphs)
- OGC API – Processes Part 3 (Workflows and Chaining, still a draft) extends Part 1 with nested process execution, local and remote processes, and OGC API collections as inputs, reusing Part 1 endpoints. One contributor describes a "Deployable workflows" requirement class (deploy a workflow as a process through Part 2) and classes for specific workflow languages such as CWL. The nested JSON form is written leaf-first. [abstract/secondary] — [OGC overview](https://github.blesdmm.dns-dynamic.net/opengeospatial/ogcapi-processes/blob/master/ogcapi-ogc-org-processes-overview.md); [OGC workshop](https://ogcapi-workshop.ogc.org/api-deep-dive/processes); [52°North blog](https://blog.52north.org/?p=8685); [Weaver docs](https://pavics-weaver.readthedocs.io/en/6.15.0/processes.html)
- Further standards, not checked in this session [not verified]:
  - CWL v1.2 has typed `inputs` and `outputs`, a `format` field taking ontology IRIs (for example EDAM), `steps` with `in` and `source` links, `SubworkflowFeatureRequirement`, `scatter`, and conditional `when`. CWLProv builds on W3C PROV and Research Object Crate. [CWL v1.2](https://www.commonwl.org/v1.2/Workflow.html); W3C PROV-O [w3.org/TR/prov-o](https://www.w3.org/TR/prov-o/)
  - WDL has typed inputs and outputs and scatter.
  - Galaxy `.ga` JSON workflows record a tool version per step.
  - QGIS `.model3` is an XML model with typed parameters.
  - Vega and Vega-Lite use declarative JSON with a published JSON Schema per version, a `$schema` URL that carries the version, and signals and params.
  - Kepler.gl saved configs carry a `version` key with migrations between versions.

### Inferences
- Copy from openEO: the reserved-key reference objects, a single result node or explicit outputs map, lexical parameter scoping with defaults, child graphs as the only sub-workflow mechanism, JSON Schema plus a `subtype` extension, and the five-step semantic validation list. The repository's data kinds and facets slot naturally into `subtype`, or into a sibling keyword such as `kind` with `level` and `extent`.
- Copy from CWL: an ontology-IRI-like `format` field per port, which here would be the kind identifier plus facets. Also copy explicit per-step tool versions and an optional PROV-shaped run record (`used`, `wasGeneratedBy`, `wasAssociatedWith`) for computation records.
- Copy from Vega-Lite and Kepler: a `$schema` or `version` field from day one, plus migrations.
- Known pain points to avoid:
  - Leaf-first nested inline JSON, the OGC Part 3 style, is hard to read and diff. Flat named nodes, the openEO style, are easier.
  - Unbounded traversal of references inside nested arrays and objects complicates validators. Restrict references to top-level argument values unless a parameter's schema allows nesting.
  - Name collisions between user-defined and predefined process identifiers. Use a namespace field.
- Synthesis research and the standards are complementary. The standards validate a workflow's structure, while CCD and APE-style facets validate its meaning. A small implementation can run the openEO-style structural checks, then a facet-compatibility check per edge, then precondition predicates.

### Gaps
- The text of OGC API Processes Part 3 (requirement classes, status as of November 2026) was not read in primary form.
- CWL, WDL, QGIS `.model3` and Kepler config details are from prior knowledge, not verified this session.
- No published comparative study of pain points across these formats was found.
