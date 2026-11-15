# Visual dataflow and node-graph builders: interaction patterns, evidence and embeddable canvases (as of 2026-11-14)

Verification legend: claims without a marker were confirmed this session from the cited page (or a search
summary of it). **[unverified]** means the claim and URL come from prior knowledge and were not re-fetched
this session; treat as leads to check. Licence strings marked "(GitHub API)" or "(npm)" were read directly
from `api.github.com/repos/...` and `registry.npmjs.org/<pkg>/latest` on 2026-11-14.

## Data science builders (KNIME, Alteryx, Dataiku, Orange, RapidMiner/Altair, Einblick, Enso)

### Takeaway
The mature analyst builders converge on the same small set of mechanics: drag a port into empty canvas to
get a short, compatibility-filtered, popularity-ranked list of next nodes; a per-node status light
(not configured / configured / executed / failed); a results pane that shows the data at whichever node or
port is clicked; partial execution with caching; and a "collapse into a reusable node" feature. The
recommendation source is almost always aggregated usage telemetry ("wisdom of crowds"), not semantics.

### Cited Findings
**KNIME**
- Quick node adding panel opens by double-clicking empty canvas or by dragging an input or output port into the canvas; shortcut Ctrl+. (Windows/Linux) / Cmd+. (Mac); it shows up to 12 recommended nodes or a search — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Dragging from an output port shows the Workflow Coach, "which will suggest you the compatible nodes and directly connect them" — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Workflow Coach recommendations come from anonymised usage data of users who opted in; the preference is "Node Recommendations by the Community" under KNIME > Workflow Coach — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/); [KNIME blog, Wisdom of the KNIME crowd](https://www.knime.com/blog/the-wisdom-of-the-knime-crowd-the-knime-workflow-coach)
- Organisations can point the Coach at their own internal experts' usage via KNIME Server instead of the global population — [KNIME blog](https://www.knime.com/blog/wisdom-knime-crowd-knime-workflow-coach)
- Node status traffic light: red = not configured, yellow = configured, green = executed, error sign on failure — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Ctrl+G creates a metanode, Ctrl+J a component (components are the reusable, configurable, shareable kind; the guide defers definitions to the Components Guide) — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Node repository offers Grid, List and Tree views with live search (can be made non-instant via `knime.repository.non-instant-search`) — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- K-AI Build mode adds, connects and configures nodes; it can "dynamically expand your workflow from that point" after selecting a data-providing node, and is offered inside the quick-add panel as "Build with K-AI". By default it sees only column names and types; seeing values needs a per-workflow "Fetch Data" permission; running nodes needs a separate permission — [KNIME AP user guide](https://docs.knime.com/latest/analytics_platform_user_guide/); [KNIME 5.8 guide](https://docs.knime.com/ap/5.8/analytics_platform_user_guide/)
- KNIME's own forum post concedes K-AI "can and does make mistakes" when building workflows and solicits failure cases — [KNIME forum](https://forum.knime.com/t/feedback-wanted-help-us-improve-k-ais-workflow-building-capabilities/90281)
- K-AI usage (vendor figure, all modes): approaching 50,000 interactions a month from over 3,000 monthly users (June 2025) — [KNIME blog](https://www.knime.com/blog/how-ai-is-changing-the-knime-forum)

**Alteryx Designer**
- Results window below the canvas: click a tool or any of its anchors to see that point's data plus metadata; non-Browse tools show only a sample (default up to 1 MB per anchor, configurable in User Settings); the Browse tool shows all rows — [Alteryx help, Results window](https://help.alteryx.com/current/en/designer/workflows/results-window.html)
- "Cache and Run Workflow": right-click a tool to cache everything up to it; reruns read a temporary .yxdb; cached tools get a blue indicator — [Data School tips](https://www.thedataschool.co.uk/frederik-egervari/helpful-lesser-known-functions-of-alteryx) (practitioner blog)
- Users have long requested browsing results mid-run and running individual tools — [Alteryx Ideas: Browse anywhere whilst running](https://community.alteryx.com/t5/Alteryx-Designer-Ideas/Browse-anywhere-whilst-workflow-is-running/idi-p/632219); [Ability to run individual tools](https://community.alteryx.com/t5/Alteryx-Designer-Desktop-Ideas/Ability-to-run-individual-tools/idi-p/43097)
- Reported bug: Browse Anywhere showing "no data available" after a run (2022.1) — [Alteryx community thread](https://community.alteryx.com/t5/Alteryx-Designer-Discussions/Browse-Anywhere-not-working-for-me-in-2022-1-anyone-else-having/td-p/944218)
- Tool palette organised in categorised colour-coded tabs; macros (standard, batch, iterative) wrap a workflow as a tool with Interface tools defining its exposed parameters; Assisted Modeling is a guided step-by-step ML wizard rather than a canvas — [Alteryx macros docs](https://help.alteryx.com/current/en/designer/workflows/macros.html) **[unverified]**; [Assisted Modeling](https://help.alteryx.com/current/en/designer/assisted-modeling.html) **[unverified]**

**Orange Data Mining**
- Typed channels with a "default" flag: when a link is drawn, Orange connects to the default channel without asking (e.g. training data), and some outputs are explicitly excluded from automatic default selection — [Orange widget dev docs, Channels and Tokens](https://orange-widget-base.readthedocs.io/en/latest/tutorial-channels.html)
- Orange logged how each widget was added (Click, Drag, Search, or Extend-from-another-widget) and planned to use the data to improve widget suggestions; users add root nodes (File) from the sidebar but leaf nodes (Data Table) by extending from an existing widget — [Orange blog, Orange is getting smarter (2018)](https://orangedatamining.com/blog/orange-is-getting-smarter)
- Link-drag interaction implementation lives in orange-canvas-core `document/interactions` — [orange-canvas-core source](https://orange-canvas-core.readthedocs.io/en/0.1.27/_modules/orangecanvas/document/interactions.html)
- Orange's model: every widget is a live window; selecting rows in one visualisation emits a "Selected Data" signal that downstream widgets recompute on immediately (linked interactive widgets) — [Orange visual programming page](https://orangedatamining.com/home/visual-programming/)

**Dataiku, RapidMiner/Altair, Einblick, Enso** **[all unverified]**
- Dataiku Flow alternates dataset nodes and recipe nodes (bipartite), with visual recipes (Prepare, Join, Group...) as form-based editors opened from a node, and Flow Zones to partition large flows — [Dataiku Flow zones](https://doc.dataiku.com/dss/latest/flow/zones.html)
- RapidMiner Studio's "Wisdom of Crowds" operator recommendations come from anonymised community usage; Auto Model and Turbo Prep are wizard/grid UIs that generate processes — [RapidMiner docs](https://docs.rapidminer.com/latest/studio/) 
- Einblick (collaborative data-science canvas) was acquired by Databricks in early 2024 and its standalone canvas wound down — [Databricks announcement](https://www.databricks.com/company/newsroom/press-releases/databricks-acquires-einblick)
- Enso (dual visual/textual language, every node shows a live visualisation) is Apache-2.0 (GitHub API) with last push 2026-08-10, i.e. slowed compared with peers — [GitHub enso-org/enso](https://github.com/enso-org/enso)

### Inferences
- The highest-value KNIME/Orange pattern for a narrow side panel is not the canvas but "extend from here": select a step's output, get a short compatible-next-operation list ranked by usage. This works identically in a step list.
- Alteryx's "click any anchor to see the data there" maps directly onto "click a step to preview its output on the map"; the analyst builders treat per-node inspection as essential.
- K-AI's permission split (schema-only by default; values and execution behind separate consents) is a reusable template for an assistant that builds analysis chains.

### Gaps
- No published acceptance rates for Workflow Coach or RapidMiner recommendations.
- Dataiku, RapidMiner/Altair AI Studio and Einblick details were not re-verified this session.

## Creative and technical node editors (ComfyUI, Blender, Houdini, Grasshopper, Unreal, TouchDesigner, Substance, Nuke)

### Takeaway
The editors with the best reputations share: link-drag-to-search filtered by socket type, socket colour and
shape that encode type and evaluation mode, an always-available viewer/inspection tap on any socket, and a
"node group with an explicit interface" (group inputs, promoted parameters, digital assets) as the unit of
reuse. ComfyUI's 2025 subgraphs and Blender's node groups are the current reference implementations.

### Cited Findings
**ComfyUI**
- Subgraphs officially released August 2025 (frontend 1.24.3+); subgraphs nest; their boundary is represented by dedicated Inputs and Outputs nodes — [Comfy blog, Subgraph official release](https://blog.comfy.org/p/subgraph-official-release); [Comfy blog, Subgraphs are coming](https://blog.comfy.org/p/subgraphs-are-coming-to-comfyui)
- ComfyUI 0.3.51 added publishing subgraphs as reusable nodes, syncing updates across linked instances, auto-generated subgraph widgets, a new Manager UI and a minimap — [Comfy blog 0.3.51](https://blog.comfy.org/i/171568149/new-manager-ui)
- 0.3.63 (Nov 2025) added publishing subgraphs into the node library and a redesigned selection toolbox — [comfyui.org summary (zh)](https://comfyui.org/zh/comfyui-0363-subgraph-publishing) (secondary)
- Frontend PR #3652 made "Action on link release (No modifier)" default to the search box instead of the context menu, because the search box has fuzzy matching and filters — [ComfyUI_frontend PR #3652 (via mirror)](https://worker-github.zxx930427.workers.dev/Comfy-Org/ComfyUI_frontend/pull/3652); canonical [github.com/Comfy-Org/ComfyUI_frontend/pull/3652](https://github.com/Comfy-Org/ComfyUI_frontend/pull/3652)
- Settings include node search box implementation (default vs `litegraph (legacy)`), showing node frequency, node ID names, categories and previews in search results — [docs.comfy.org settings](https://docs.comfy.org/interface/settings/comfy.md)
- Workflows are embedded as JSON in output PNG metadata so dragging an output image onto the canvas restores the graph; missing custom nodes are flagged on load with install via Manager; a templates browser ships starter workflows — [docs.comfy.org](https://docs.comfy.org/) **[unverified]**
- ComfyUI_frontend licence is GPL-3.0 (GitHub API) — [GitHub](https://github.com/Comfy-Org/ComfyUI_frontend)

**Blender Geometry Nodes**
- Socket shape encodes evaluation: diamond sockets accept/output fields; diamond-with-dot means "can be a field but is currently a single value" — [Blender manual, Fields](https://www.blender.org/manual/en/modeling/geometry_nodes/fields.html)
- Viewer node shows data from anywhere inside a node group in the Spreadsheet editor and the 3D viewport (overlay for fields); Shift-Ctrl-LMB on any node or socket connects it to the active viewer; viewer can be pinned in the Spreadsheet — [Blender manual, Viewer node](https://docs.blender.org/manual/en/dev/modeling/geometry_nodes/output/viewer.html)
- Link-drag-search (drag a socket into empty space, search only compatible nodes/sockets and auto-connect) landed in Blender 3.1 — [Blender 3.1 release notes, Nodes](https://developer.blender.org/docs/release_notes/3.1/nodes_physics/) **[unverified]**
- Node group interface (Group Input/Output, panels) defines what the modifier exposes; Blender 4.0 added "node tools" (geometry node groups as operators in menus) and node-group assets — [Blender 4.0 release notes](https://developer.blender.org/docs/release_notes/4.0/nodes_physics/) **[unverified]**
- Socket colour per data type and node parts are documented in Node Parts — [Blender manual 4.5, Node Parts](https://docs.blender.org/manual/en/4.5/interface/controls/nodes/parts.html)

**Houdini, Grasshopper, Unreal, TouchDesigner, Substance, Nuke** **[all unverified]**
- Houdini Digital Assets package a subnetwork with a curated, promoted parameter interface (Type Properties); display/render/bypass flags on each node select what is shown; middle-click on a node shows its info (cook time, memory, attribute counts) — [SideFX assets docs](https://www.sidefx.com/docs/houdini/assets/index.html); [SideFX network flags](https://www.sidefx.com/docs/houdini/network/flags.html)
- Unreal Blueprints: dragging off a pin opens the palette with "Context Sensitive" filtering to actions valid for that pin type — [Epic, Blueprint editor](https://dev.epicgames.com/documentation/en-us/unreal-engine/blueprint-editor-reference-for-unreal-engine)
- TouchDesigner: every operator tile can show a live viewer of its output (toggle per node) — [TouchDesigner wiki, Viewer](https://docs.derivative.ca/Viewer)
- Grasshopper: clusters wrap sub-definitions; Panel components show values inline; preview toggles per component; "wire sprawl" is the community's common complaint and named-relay/wireless plugins exist to cut it — [McNeel Grasshopper docs](https://www.rhino3d.com/learn/?keyword=kind:%20grasshopper)
- "Blueprints from hell" is a well-known practitioner gallery of unreadable graphs — [blueprintsfromhell.tumblr.com](https://blueprintsfromhell.tumblr.com/)

### Inferences
- Type-filtered search at link release is the single interaction every top editor converged on; ComfyUI explicitly switched its default to it.
- Encoding "per-feature field vs. single value" in socket shape (Blender) is directly analogous to per-row column vs. scalar parameters in an analysis library; a step list can show the same with an icon.
- GPL-3.0 on ComfyUI_frontend makes it unusable as an embedded component in a permissively licensed library; only the MIT litegraph fork is a candidate.

### Gaps
- Could not fetch docs.blender.org English intro page (HTTP 403); link-drag-search version and node tools details not re-verified.
- No primary source re-checked for Houdini, Unreal, TouchDesigner, Grasshopper, Substance Designer or Nuke this session.

## Canvas-meets-notebook tools (Observable, Count, Hex, Deepnote, tldraw computer, Natto, Nodes.io)

### Takeaway
Observable Canvases (April 2025, GA around August 2025) is the clearest current example of a data canvas
for analysts: nodes are SQL, UI and chart blocks; readability is kept by branching, duplicating branches,
rewiring, using charts and summary tables as visual filters, and a minimap. The other tools were not
re-verified.

### Cited Findings
- Observable Canvases launched in early access April 2025 as a collaborative data-exploration whiteboard mixing SQL, UI components and AI — [Observable blog, introducing canvases](https://observablehq.com/blog/introducing-canvases-early-access)
- Features: summary tables and charts act as visual filters (brush/select); branching analysis lets users rewire connections and duplicate branches; a minimap aids navigation as the canvas grows — [Observable blog, eight features](https://observablehq.com/blog/eight-features-supporting-fluid-data-exploration-in-observable-canvases)
- Summer 2025 release ("biggest release yet"): advanced charts, sharing, connectors, workspace management — [Observable blog, summer release](https://observablehq.com/blog/canvases-summer-release-2025)
- 2025 year in review: performance work and a new approach to dashboarding from a canvas — [Observable 2025 year in review](https://observablehq.com/blog/observable-2025-year-in-review)
- Observable Notebooks 2.0 (2025) moved back toward a linear, file-based notebook — [Tom MacWright, Observable Notebooks 2.0](https://macwright.com/2025/07/31/observable-notebooks-2)
- Hex shows cells as a linear notebook with an auxiliary dependency Graph view; Count.co is a canvas of SQL cells; Deepnote is linear with block-level reactivity; tldraw "computer" (Dec 2024) was an experimental AI node canvas — [Hex docs](https://learn.hex.tech/docs/explore-data/projects/graph) **[unverified]**; [Count](https://count.co/) **[unverified]**; [tldraw computer](https://computer.tldraw.com/) **[unverified]**

### Inferences
- Observable kept two products: a linear notebook and a separate canvas. That supports the study's "linear primary, graph view optional/auxiliary" split; Hex's auxiliary graph view is the same pattern.

### Gaps
- No independent usability evidence for Observable Canvases; all claims are vendor framing.
- Count, Hex, Deepnote, Natto, Nodes.io and tldraw computer not verified this session.

## Interaction patterns to extract (with specifics)

### Takeaway
For a 280-operation typed library in a narrow side panel, the patterns that transfer without a canvas are:
type-filtered "add next" from an output, per-step status, click-to-preview output, run-to-here with cache
indicators, errors on the step, grouping into a reusable step with an exposed parameter subset, and
templates. Canvas-only patterns (wire drag feedback, auto-layout, minimap) matter only once a graph view
exists.

### Cited Findings
- Typed ports and compatibility at drag time: KNIME (compatible nodes on port drag) — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/); Orange default channel auto-match — [Orange channels](https://orange-widget-base.readthedocs.io/en/latest/tutorial-channels.html); React Flow `isValidConnection` callback for connection validation — [React Flow v12 migration](https://reactflow.dev/learn/troubleshooting/migrate-to-v12)
- Insert from dangling wire with filtered search: KNIME quick-add panel; ComfyUI link-release search default — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/); [ComfyUI PR #3652](https://github.com/Comfy-Org/ComfyUI_frontend/pull/3652)
- Live preview / inspection per node: Alteryx anchor click — [Alteryx Results window](https://help.alteryx.com/current/en/designer/workflows/results-window.html); Blender viewer tap — [Blender Viewer](https://docs.blender.org/manual/en/dev/modeling/geometry_nodes/output/viewer.html)
- Exposed parameter subset as interface: ComfyUI subgraph Inputs/Outputs and auto-generated widgets — [Comfy 0.3.51](https://blog.comfy.org/i/171568149/new-manager-ui); KNIME components (Ctrl+J) — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Run-to-node and caching: Alteryx Cache and Run — [Data School](https://www.thedataschool.co.uk/frederik-egervari/helpful-lesser-known-functions-of-alteryx); KNIME traffic lights as execution state — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Data shown to assistant: K-AI schema-only by default — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/)
- Auto-layout in React Flow: dagre (simple, fast, trees, known sub-flow bug when sub-flow nodes connect outside), d3-hierarchy (single root, uniform sizes), d3-force (non-trees, expensive per render), elkjs (most capable: dynamic sizes, sub-flows, edge routing, async; xyflow "doesn't often recommend" it due to complexity). Edge routing around nodes (libavoid) and editable edges are Pro examples — [React Flow layouting](https://reactflow.dev/learn/layouting/layouting)
- Minimap: Observable Canvases and ComfyUI both added one as canvases grow — [Observable](https://observablehq.com/blog/eight-features-supporting-fluid-data-exploration-in-observable-canvases); [Comfy 0.3.51](https://blog.comfy.org/i/171568149/new-manager-ui)
- Keyboard-first: KNIME Ctrl+. quick add, Ctrl+G/J grouping — [KNIME guide](https://docs.knime.com/latest/analytics_platform_user_guide/); Blender Shift-Ctrl-click to viewer and number keys to switch viewers — [Blender Viewer](https://docs.blender.org/manual/en/dev/modeling/geometry_nodes/output/viewer.html)
- Recommendation telemetry design: record add method (click/drag/search/extend) and source node — [Orange blog](https://orangedatamining.com/blog/orange-is-getting-smarter)

### Inferences
- "Add next step" filtered by the previous step's output type, ranked by frequency (local usage counts at first), is the shared core of KNIME, Orange, ComfyUI, Blender and Unreal and does not need a canvas.
- Error placement on the failing node plus a status colour (KNIME) translates to an inline error on the step row.

### Gaps
- No product found with a documented narrow-screen or mobile node-editor mode; no evidence collected on mobile behaviour.

## Evidence on when visual programming helps and fails

### Takeaway
The research base is old and small. Its consistent message is that graphs are not inherently easier to read
than text or lists. Readability depends on reader skill and secondary notation. Graphs suffer viscosity and
density limits (the "Deutsch limit"), and progressive evaluation (running partial work) matters most for
novices. No study was found that directly compares a linear step list with a node canvas for novice
geospatial or data analysts. Products tend to keep canvases but surround them with wizards, linear views
or AI, or retire whole products rather than remove the canvas.

### Cited Findings
- Green and Petre (1996), "Usability Analysis of Visual Programming Environments: a 'Cognitive Dimensions' Framework", JVLC 7:131-174, DOI 10.1006/jvlc.1996.0009: applied to LabVIEW, Prograph and Basic; progressive evaluation is essential for novices; LabVIEW is strict about execution while Prograph supports it well; LabVIEW has visibility problems with conditionals and Prograph with its subroutine structure; both are limited in secondary notation; designers must weigh viscosity (resistance to local change) and secondary notation — [It Will Never Work in Theory summary](https://neverworkintheory.org/2012/06/25/usability-analysis-of-visual-programming-environments-a-cognitive-dimensions-framework.html); [researchr](https://researchr.org/publication/GreenP96)
- Deutsch limit: "you can't have more than 50 visual primitives on the screen at the same time"; attribution to L. Peter Deutsch, no primary source found; counter-arguments note modularity sidesteps it and text may have a similar limit — [Wikipedia, Deutsch limit](https://en.wikipedia.org/wiki/Deutsch_limit); catalogue of criticisms — [visual-programming-codex criticisms](https://github.com/eliemichel/visual-programming-codex/blob/master/criticisms.md)
- Anjaneyulu and Anderson (1992): a dataflow-diagram language for beginners removed some syntax obstacles, but "many of the conceptual difficulties that students have are unaffected by the choice of a linear or graphical representation" — [Springer](https://link.springer.com/chapter/10.1007/3-540-55606-0_68)
- Visual dataflow modelling in design tools: fewer nodes linked more linearly are easier to understand; highly complex graphs are very hard for novices (analysis, not experiment) — [eCAADe 2014](https://papers.cumincad.org/data/works/att/ecaade2014_169.content.pdf)
- VisCanvas (arXiv 2607.21886, 2026 preprint): node-based interface vs linear chat for LLM-assisted exploratory visualisation, within-subjects study, 20 participants (results not re-read) — [arXiv](https://arxiv.org/pdf/2607.21886)
- Petre (1995) "Why looking isn't always seeing: readership skills and graphical programming", CACM 38(6): graphics are not inherently clearer; experts exploit secondary notation, novices misread layout cues — [ACM DL](https://dl.acm.org/doi/10.1145/203241.203251) **[unverified]**
- Whitley (1997) "Visual programming languages and the empirical evidence for and against", JVLC 8(1): mixed evidence; benefits are task-specific — [DOI](https://doi.org/10.1006/jvlc.1996.0030) **[unverified]**
- Weintrop and Wilensky (2017), blocks vs text in high-school CS (ACM TOCE): block-based modality produced higher learning gains on some concepts — [ACM DL](https://dl.acm.org/doi/10.1145/3089799) **[unverified]**
- Product retirements/changes: Azure ML Studio (classic, drag-and-drop) retired 31 August 2024 (no new resources after 1 Dec 2021); its successor Designer still exists, but classic prebuilt components are frozen and Microsoft recommends custom (code-defined) components for new work — [Microsoft lifecycle](https://learn.microsoft.com/en-us/lifecycle/products/machine-learning-studio-classic); [Azure ML Designer concept](https://learn.microsoft.com/azure/machine-learning/concept-designer)
- Alteryx adds a wizard (Assisted Modeling) and RapidMiner adds Auto Model/Turbo Prep on top of the canvas, which shows vendors routing novices away from wiring — see sources in the first section **[unverified]**

### Inferences
- The Green and Petre finding that "progressive evaluation is essential for novices" supports per-step preview and run-to-here more strongly than it supports any particular layout.
- Anjaneyulu and Anderson suggest the representation (list vs. graph) matters less than conceptual support (good defaults, type guidance, preview).
- Retirements seen here (Azure ML classic, Einblick by acquisition) are whole-product decisions; none was found where a vendor removed a canvas and kept the product with a list. Vendors more often add a linear or wizard layer beside the canvas.

### Gaps
- No controlled study found comparing a step list or form with a node canvas for novice data or GIS analysts.
- No studies of KNIME, Alteryx or Galaxy usability with analysts were retrieved this session.
- Davis (2013) on Grasshopper model structure and cluster use was not retrieved; numbers are unknown.

## Embeddable canvas libraries for a React 19 app

### Takeaway
React Flow (`@xyflow/react` 12.12.1, MIT, peer `react >=17`, pushed 2026-11-12) is the clear default for a
React 19 host: very active, typed via `isValidConnection` and handle IDs, and has documented dagre and elkjs
layout recipes. Its Pro tier sells examples and support, not core features. Rete.js v2 (MIT, active) is the
main alternative if a framework-agnostic engine with typed sockets is wanted. The tldraw SDK needs a licence
key in production. ComfyUI's frontend is GPL-3.0. Drawflow, the original litegraph.js and Reaflow are
stale.

### Cited Findings
Licence, version, peer dependency and activity, all from GitHub API and npm registry on 2026-11-14 ("unpacked" = npm unpackedSize, not a minified gzip bundle):

| Library | npm latest | Licence (exact) | Peer deps | Last push | Stars | Unpacked |
|---|---|---|---|---|---|---|
| React Flow `@xyflow/react` | 12.12.1 | MIT | react >=17 | 2026-11-12 | 39,017 | 1.2 MB |
| Svelte Flow `@xyflow/svelte` | 1.7.1 | MIT | svelte ^5.25.0 | (same repo) | | 0.33 MB |
| Rete.js `rete` | 2.0.6 | MIT | none | 2026-10-25 | 12,314 | 0.23 MB |
| litegraph.js (jagenjo) | 0.7.18 | MIT | none | 2024-08-01 | 8,187 | 3.2 MB |
| `@comfyorg/litegraph` | 0.17.2 | MIT | none | 2026-01-14 (fork repo) | 256 | 3.8 MB |
| ComfyUI_frontend | n/a | GPL-3.0 | n/a | 2026-11-14 | 2,130 | n/a |
| Flume `flume` | 1.2.0 | MIT | react ^18.2.0 | 2026-04-12 | 1,630 | 1.7 MB |
| BaklavaJS `baklavajs` | 2.8.1 | MIT | none (Vue-based) | 2026-10-23 | 2,098 | 1.1 MB |
| JointJS `@joint/core` | 4.3.4 | MPL-2.0 | none | 2026-11-13 | 5,402 | 6.3 MB |
| maxGraph `@maxgraph/core` | 0.25.1 | Apache-2.0 | none | 2026-11-10 | 1,154 | 7.2 MB |
| Reaflow `reaflow` | 5.4.1 | Apache-2.0 | react >=16 | 2025-06-04 | 2,496 | 4.8 MB |
| tldraw `tldraw` | 5.6.0 | "SEE LICENSE IN LICENSE.md" (GitHub: NOASSERTION) | react ^18.2.0 \|\| ^19.2.1 | 2026-11-14 | 51,422 | 15.6 MB |
| Drawflow `drawflow` | 0.0.60 | MIT | none | 2024-10-19 | 6,173 | 0.07 MB |

Sources: [GitHub xyflow](https://github.com/xyflow/xyflow), [npm @xyflow/react](https://www.npmjs.com/package/@xyflow/react), [GitHub retejs/rete](https://github.com/retejs/rete), [GitHub jagenjo/litegraph.js](https://github.com/jagenjo/litegraph.js), [GitHub Comfy-Org/litegraph.js](https://github.com/Comfy-Org/litegraph.js), [GitHub Comfy-Org/ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend), [GitHub chrisjpatty/flume](https://github.com/chrisjpatty/flume), [GitHub newcat/baklavajs](https://github.com/newcat/baklavajs), [GitHub clientIO/joint](https://github.com/clientIO/joint), [GitHub maxGraph/maxGraph](https://github.com/maxGraph/maxGraph), [GitHub reaviz/reaflow](https://github.com/reaviz/reaflow), [GitHub tldraw/tldraw](https://github.com/tldraw/tldraw), [GitHub jerosoler/Drawflow](https://github.com/jerosoler/Drawflow)

- React Flow v12: package renamed to `@xyflow/react` (named import); SSR support via node width/height/handles; computing-flow hooks (`useHandleConnections`, `useNodesData`, `updateNodeData`); `colorMode` light/dark/system with CSS variables; `isValidConnection`, `onBeforeDelete` — [React Flow v12 migration](https://reactflow.dev/learn/troubleshooting/migrate-to-v12)
- React 19: npm peer range `react >=17` admits React 19 (npm registry); no explicit React 19 statement found on the docs pages fetched.
- React Flow Pro: paid subscription (Starter tier listed at $169/month) that gives Pro examples/templates, prioritised issues and support; core stays MIT; Pro-only examples include libavoid edge routing and editable edges — [React Flow Pro pricing](https://reactflow.dev/pro/pricing); [React Flow layouting](https://reactflow.dev/learn/layouting/layouting)
- React Flow attribution: the MIT core shows a small attribution link by default; removal is requested for Pro subscribers or non-commercial use (see policy) — [React Flow attribution](https://reactflow.dev/attribution)
- Rete.js: MIT applies to "most framework packages" (the docs' wording, so some packages may differ) — [Rete.js licensing](https://retejs.org/docs/licensing)
- tldraw SDK: production use (HTTPS, non-loopback domain, NODE_ENV=production) requires a licence key; without one the editor stops rendering after five seconds; hobby licences are non-commercial, keep the "made with tldraw" watermark and are reviewed case by case; commercial licence removes the watermark; terms forbid hiding the watermark or interfering with key validation — [tldraw license key](https://tldraw.dev/sdk-features/license-key); [tldraw SDK License](https://tldraw.dev/legal/tldraw-license)
- Since tldraw SDK 4.0 (September 2025) the SDK ships under the tldraw licence (blog, secondary); v3.0 terms allowed commercial use with visible watermark, now superseded — [tldraw blog](https://tldraw.dev/blog/20-things-i-wish-ai-chatbots-knew-about-tldraw); [tldraw v3.0.0 release](https://tldraw.dev/releases/v3.0.0)
- Typed ports: Rete sockets are objects whose compatibility the app defines; litegraph slots carry type strings and filter search by type; Flume declares port types with allowed connections; Baklava v2 has interface types plugin — [Rete docs](https://retejs.org/docs) **[unverified]**; [Flume docs](https://flume.dev/docs/port-types) **[unverified]**; [Baklava interface types](https://baklava.tech/plugins/interface-types.html) **[unverified]**
- Comfy-Org/litegraph.js repo last pushed 2026-01-14 while ComfyUI_frontend is pushed daily, consistent with litegraph being folded into the GPL frontend monorepo — [GitHub](https://github.com/Comfy-Org/litegraph.js) (folding is **[unverified]**)

### Inferences
- React Flow is the only option combining React-native rendering (custom node components can reuse the panel's existing React parameter widgets), MIT licence, an explicit `react >=17` peer range and very high activity. Typed-port feedback has to be built with `isValidConnection` plus per-handle styling; it is not built in.
- At a few hundred nodes React Flow is expected to be fine with memoised custom nodes (xyflow publishes a stress example), but this was not measured this session. Layout via dagre is the low-risk start; elkjs only if sub-flows (groups) need correct layout.
- litegraph (canvas-2D, ComfyUI-style) performs well with very large graphs but does not render React components inside nodes; the maintained fork's future is tied to a GPL project.
- Flume's peer range `react ^18.2.0` does not admit React 19; tldraw is a whiteboard SDK with commercial key requirements; Drawflow, jagenjo litegraph and Reaflow show little 2025-2026 activity. JointJS core is MPL-2.0 (file-level copyleft; its richer features are in commercial JointJS+).

### Gaps
- Minified/gzipped bundle sizes were not measured (only npm unpacked sizes above).
- No independent benchmark of React Flow at 200 to 500 nodes was retrieved.
- Rete.js per-package licence exceptions not identified.
