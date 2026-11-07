# Spatial atlas experience: research, design and implementation

Date: 2026-10-25. Scope: the existing spatial-analysis showcase shell, discovery and story framing.
This record distinguishes reviewed guidance, design interpretation, implementation and verification.

## Existing experience audit

The initial inventory contains 112 scene definitions across 17 chapters. Scene registration is
filename-driven; the shell must continue to discover every scene without a second registry. Stories
already support guided steps, named readouts, charts, evidence, caveats, source attribution, live
controls, accessible panel tabs, keyboard navigation and a responsive map sheet. Those mechanisms
are valuable and should be preserved.

The entry page initially repeats a thumbnail card for every scene, with four featured cards and no
search or filters. A visitor must know which method chapter contains their geographic question.
Generated seeded thumbnails encode a chapter motif rather than the scene's observed data. They are
illustrations, not analytical evidence. The story's sources and method are available in a details
tab, while evidence and caveat blocks visually differ only through colour and have no visible label.

## Primary sources reviewed

1. [Esri: Narrative structures for stronger storytelling](https://www.esri.com/arcgis-blog/products/story-maps/constituent-engagement/story-structures),
   Cooper Thomas and Ross Donihue, 2025-03-22; accessed 2026-10-25. The authors describe different
   structures for quick findings, staged analytical explanations and related case studies. They
   emphasize matching structure to the audience and question rather than applying one narrative
   template universally. **Design interpretation:** an atlas should offer short, thematic reading
   routes alongside a complete method index. Individual scene authors retain control of their
   evidence sequence; the shell supplies orientation rather than inventing a conclusion.
2. [W3C: Understanding SC 4.1.3, Status Messages](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html),
   accessed 2026-10-25. Search-result totals and empty-result notices are status updates that should
   be available to assistive technology without moving focus. **Implementation requirement:** a
   persistent result-count region uses `role="status"`; filtering updates the result area while
   keeping the search field mounted and focused.
3. [W3C: Understanding SC 1.4.10, Reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html),
   accessed 2026-10-25. Surrounding navigation and prose should reflow at 320 CSS pixels even when a
   map itself needs a two-dimensional presentation. **Implementation requirement:** the atlas
   directory, search controls, route lists and header remain usable in a narrow viewport; the
   existing story map sheet remains intact.
4. [W3C: Understanding SC 2.5.8, Target Size (Minimum)](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html),
   accessed 2026-10-25. WCAG 2.2 AA specifies a minimum of 24 by 24 CSS pixels, subject to listed
   exceptions. **Design target:** use roughly 44-pixel primary navigation and discovery controls
   where practical, with a visible focus indicator. This is an implementation target, not a claim
   of audited WCAG conformance.

## Design decisions

- Put a geographic question before API names. Short curated routes connect existing scenes across
  chapter boundaries, with an explicit analytical purpose for each stop. Every linked scene must
  exist; route labels must not imply a study finding that the scene does not establish.
- Keep the complete atlas inventory. Search title, summary, contributor names, dataset names/IDs
  and chapter text. Combine chapter and declared-capability filters; show the active result count
  and a useful reset action for an empty result. Capabilities describe available presentation
  mechanics, not scientific quality.
- Replace the repeated thumbnail wall with an editorial directory. Use ordered routes, chapter
  numbers, strong typographic hierarchy and compact entries with dataset/method context. Avoid
  arbitrary generated charts or decorative maps that could be mistaken for data.
- Make provenance discoverable before interpretation. A concise study brief introduces declared
  source datasets and methods. Existing detailed attribution and source links remain authoritative.
  Dataset metadata cannot reliably classify all assets as observed, synthetic, modeled or sampled;
  do not infer those claims from a title or filename.
- Show literal Evidence and Limit labels on the story's interpretation notes. Colour can reinforce
  their meaning but cannot be the only visible distinction.
- Preserve lazy compute loading, route semantics, map controls, URL option state, graph inspection,
  keyboard shortcuts, light/dark themes and the mobile sheet. No shared scene-schema migration is
  required: existing scene and dataset metadata is sufficient.

## Ownership and implementation

An Astra planner audits sources and reviews the design; one delegated Luna implementer owns shell
changes. Other planners own scene content and computation. The shared checkout is preserved; the
experience work does not reset, commit or install dependencies. Parent integration coordinates
formatting and repository-wide validation.

### Implemented experience

The home page now has seven inquiry routes: rail disruption, satellite fire evidence, observation
support, terrain drainage, model validation, survey measurement and query provenance. Each stop has its own analytical
purpose and links to an actual registered scene; concurrent additions appear when their scene
definitions are present. The layout uses a full-width lead story, four map plates and two text
ledgers. A searchable chapter directory retains every registered scene underneath. Query terms are
combined, and chapter, chart, declared timeline and declared comparison filters are independent.
Active filters move the directory immediately beneath the search controls by hiding the routes.
The query and filters survive a reload through the URL.
The terrain route now includes the conditioning experiment between watershed delineation and the
flood scenario. All seven routes resolve to three real stops; a repeated no-WebGPU check at 320 px
and 1440 px found no horizontal overflow. Async home completion also checks that its root remains
connected before changing URL state, protecting fast navigation away during metadata loading.

The parent captured and visually reviewed five genuine initial-step WebGPU previews on 2026-10-25:
`transit-frequency`, `burn-severity`, `h3-aggregation`, `watersheds`, and `rainfall-interpolation`.
They are 730-by-528 PNGs in `public/images/story-previews/`. The shell uses the complete frame,
preserving its embedded attribution, and labels each image as a map preview with its story and
dataset names. These are static orientation images, not measurements from a visitor's current
control settings. A missing preview fails gracefully without removing the route.

Each story now opens with a study brief containing the existing summary, contributor methods and
links to the declared source datasets. Evidence and Limit labels are visible in interpretation
notes. The existing detailed sections are labeled Method & reading guide and Sources & provenance.
A keyboard skip link focuses the main content without changing the hash or scrolling the story
behind the sticky header. Active navigation has `aria-current`; the narrow header and mobile sheet
share an accurate height token. The map credit line is raised from a 7-pixel to a 27-pixel offset
above the map bottom inset to separate it from basemap attribution.

### Files

- `examples/deck/spatial-analysis-showcase/src/shell/home-view.ts`
- `examples/deck/spatial-analysis-showcase/src/shell/atlas-routes.ts`
- `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts`
- `examples/deck/spatial-analysis-showcase/src/shell/story-view.css`
- `examples/deck/spatial-analysis-showcase/src/main.ts`
- `examples/deck/spatial-analysis-showcase/src/styles.css`
- This research record; preview PNG assets were supplied by the parent integration task.

### Verification and limits

- `nvm use` selected repository Node 22.23.4 successfully.
- A separate Vite development server on port 5314 served live source for non-GPU shell checks.
  The WebGPU capability was deliberately disabled for the story-shell checks; these checks do not
  establish GPU computation correctness.
- Browser checks passed for contributor search, input focus retention, empty-result reset,
  chapter filtering, multi-term matching, chart/timeline/comparison filters, URL reload persistence,
  recovery from unknown filters, source links, visible Evidence/Limit labels and skip-link focus
  and route retention. No page errors occurred in that sweep.
- At the time of the sweep, the filename registry contained 116 studies. Filters returned 85 chart
  stories, 10 declared timelines and 21 declared comparison stories; these are metadata counts,
  not coverage or quality claims. Counts are computed dynamically and may change during integration.
- Reflow checks passed at 320, 768 and 1440 CSS pixels. At 320 pixels the shared header reservation
  matched its measured 104-pixel height. The corrected skip link retained scroll position zero
  and a main-content top of 52 pixels, exactly below the desktop header.
- Visual review covered light/dark desktop, narrow mobile and the story's no-WebGPU presentation.
  All five map previews loaded at their natural 730-by-528 size and retained the full aspect ratio.
  The local evidence is `/tmp/showcase-atlas-qa/interaction-report.json` and adjacent screenshots;
  temporary evidence is not a committed artifact.
- `git diff --check` passed for the owned files. Package typechecking during implementation found
  only concurrent scene-edit failures, not errors in the owned shell files. Parent integration
  must rerun it after scene changes settle.
- Repository-wide `yarn install`, `yarn build`, `yarn test`, `yarn lint fix`, `yarn website:build`
  and `(cd website && yarn build)` are coordinated by the parent integration task. This subtask
  did not run those checks. Final formatting, production-build browser checks and live GPU
  attribution placement remain integration verification, not claims made by this shell sweep.


### Accepted measurement route

After production numerical/visual acceptance, the seventh inquiry set includes the measurement
sequence `point-cloud-support → point-cloud-change → survey-frame-sensitivity`. The last stop
uses real repeated surveys and makes hypothetical control-area assumptions visible. This replaces
the earlier introductory lidar stop in that curated sequence; the full directory retains the
original lidar story. All seven curated inquiries still contain three linked investigations.

### Continuation and inventory review

The seven route definitions were extracted unchanged into `shell/atlas-routes.ts`. Its typed
`ATLAS_ROUTES` export and dependency-free `getNextAtlasStop(sceneId)` helper let the parent add a
meaningful onward card to the final story step without loading the registry or computation code.
A scoped check verifies all 21 route IDs against current scene files, first-to-second transitions,
terminal null results and an unknown ID. All pass. The parent owns the completion-card UI and its
production browser validation.

The accompanying `information-architecture.md` reviews concrete consolidation candidates and
their functionality-preservation requirements. The movement lane has already narrowed the old
vessel story around its distinct fleet-wide pairwise comparison. Curated routes remain selective;
new investigations are not featured before production acceptance merely to increase route count.

### Elapsed-time dark sequential ramp

The `elapsedTime` dark table now uses `deriveDarkSequential` from its six light-ground teal-to-violet
classes. The derived dark RGB sequence is `#7c4fac`, `#687cc6`, `#55a6ce`, `#75c8cb`, `#a9dfcc`,
`#d9f0e5` (CIE L*: 42.54, 53.42, 64.69, 75.70, 84.78, 92.90): it reverses the light table,
preserves its hue/chroma progression, and lifts the low class to the dark-ground lightness floor,
so higher elapsed values are brighter. This also removes
the hand-authored violet endpoint that reversed lightness direction at the final class. The
focused `check-colours.mjs` audit passes with zero failures; it still reports the existing palette
warnings, including an elapsed-time CVD adjacent-class warning. No current scene dispatches the
`elapsedTime` registry key (the source search found only its registry entry and guide listing), so
there is no honest scene screenshot target until a consumer exists; the registry audit is the
available visual-order check for this change.

### Final curated-route editorial review

The seven routes remain selective rather than growing with every new algorithm. Their evidence
forms are deliberately different: a night rail network; satellite disturbance and recovery;
light urban observation grids followed by global spherical support; relief and drainage;
interpolated gauges and held-out soil predictions; close-range survey differences and assumed
measurement uncertainty; and linked query/count ledgers. Source names and actual scene titles
remain visible beside each route's purpose. The composite stop now says “Audit cloud-screened
observations” to describe temporal support more precisely than “Combine fire evidence.”

A no-WebGPU production review captured every route plate and the narrow layout in
`/tmp/atlas-routes-review/`. All 21 links resolve to the intended story IDs and the 320-pixel page
has no horizontal overflow. The two text-only routes previously left a wide empty right side;
their desktop title and ordered stops now occupy separate columns, returning to one column below
700 pixels. Injecting the saved stylesheet into the production page for a scoped layout review
passed width checks at 320, 768 and 1440 pixels; this is a presentation check, not a rebuilt
production artifact. Parent integration will rebuild the final source.

The newly accepted harbor quality → continuous encounters → similarity sequence is a strong
candidate for a future rotation, not an eighth feature card. The full directory continues to
include those studies. No existing study is removed merely to reduce the count.

### Primary evidence hierarchy

A final presentation audit reduced each step in the seven new investigations owned by this lane
to at most four primary readouts. Source support, invalid/capacity status and the consequential
result remain beside the prose; complete numeric definitions remain mounted in All readouts or
the diagnostic panel. In particular, BIXI accounting keeps valid mass, both marginal residuals and
the relation chart; CTA clipping keeps both geometric lengths and both completion statuses;
subdivision keeps pieces, area, area conservation and status; survey volume keeps the estimate,
assumed-model uncertainty, observed area and mask comparison; local ICP keeps fit status, target
normal support, residual history and inlier history. This changes hierarchy rather than hiding
failure states or removing output coverage. Owned story-step checks pass. Existing numeric
checkers already read the complete mounted readout set and do not rely on these primary lists.

### Compact-height legend verification

The short-phone/timeline fixture exposed a real failure: a flex-shrunk legend could leave its
44-pixel toggle outside its available stack and over the source credit. StoryView now chooses
normal, compact (duplicate cartouche/insets yield), or panel access from the actual space between
the top-left stack and source credit. When there is insufficient map space, a 44-pixel Legend
button remains in the wrapping panel footer. Opening it lowers the real sheet to peek and opens
the map legend; explicit open/collapsed preference survives layout and breakpoint changes.

Independent `scripts/check-legend-layout.mjs` extracts the production StoryView layout methods and
actual click handlers with TypeScript's parser, then bundles them with the real SheetController
and saved styles. It caught an initial implementation error: Chromium returns computed max-height
as `max(0px, 100% - 479px)`, so parsing it numerically selected panel mode on every phone. The final
implementation measures geometric rectangles instead. The no-WebGPU check passes at 1440×960,
390×844, 320×640 with timeline, and 320×568 with timeline, including actual handle interactions,
footer reopen, body/credit clearance, 44-pixel targets, preference persistence across desktop
resize, no horizontal overflow, and settled ResizeObserver notifications. Review artifacts are
`/tmp/showcase-legend-layout/`. The short-phone opened screenshot was visually reviewed. This is
source-layout verification; root's subsequent production rebuild must carry the corrected
measurement (the frozen wave12b bundle predates this fix).

Readout rows now expose `data-readout-id` and their mounted value/chart exposes
`data-readout-value`, both using the stable specification ID. Narrative `data-readout` behavior is
unchanged. The final-step continuation card was separately reviewed at desktop and 320 pixels:
its onward-study link remains readable and the layout has no horizontal overflow.

### Responsive bounds camera verification

Bounds-based story cameras now refit after viewport resize and sheet-padding settlement, using a
380-millisecond debounce and one animation frame. Explicit coordinate/zoom views are unchanged.
The request captures the active mount, scene and step, checks for reader camera movement again
before fitting, and is cancelled on navigation, explicit camera actions and unmount. Direct map
pointer, wheel or camera-key input interrupts a pending request. An interruption latch prevents
a later event in the same resize burst from rearming it while padding temporarily suppresses the
camera-drift comparison; an explicit step-camera reset restores responsive fitting.

`node scripts/check-responsive-camera.mjs` passes 12 no-WebGPU cases. The fixture extracts the
actual production resize handler, camera fitting, scheduling, input cancellation and cleanup
methods with TypeScript's parser and runs them in Chromium against a mocked map host. It verifies
bounds/pose/extra-padding arguments after phone standing padding settles, same-breakpoint resize
coalescing without repeated fits, explicit views left untouched, preservation of moved cameras,
pointer/wheel/keyboard interruption followed by another resize, reset recovery, and step/mount/
cleanup invalidation. The gesture-followed-by-resize regression failed before the latch and
passes after it. Results: `/tmp/showcase-responsive-camera/report.json`. The existing actual
SheetController legend-layout fixture also passes after this change. Showcase typecheck and
scoped StoryView Biome checks pass; rebuilt production visual verification remains with root.

### Production containing-block correction (wave20)

Wave19 saved phone images exposed a fixture gap: the mobile map filled the viewport from y=0,
placing its cartouche under the sticky header. Production `.story-layout` had no positioned
containing block, while the legend fixture accidentally supplied one through an inline style on
its synthetic main element. The fix adds `position: relative` to `.story-layout`; no camera or
annotation changes are involved.

The fixture now removes that synthetic positioning, uses the production header class, and asserts
header, map and visible-cartouche rectangles before and after breakpoint changes. It passes with
all existing real SheetController/legend interactions. At 390×844, header bottom and map top are
52 pixels, map height is 792 pixels, and cartouche top is 64 pixels. At 320-pixel width the map
clears the production 104-pixel header. Evidence: `/tmp/showcase-legend-layout-wave20/report.json`;
phone image visually reviewed. All DOM fixture browsers closed. This check covers layout and
interaction without GPU scenes; root owns the subsequent rebuilt production sweep.

### Phone chart and status-pill collision correction (wave22)

Wave21 revealed that an inset created on desktop stayed expanded after narrowing, so the
Earthquake Query Lab histogram intercepted the mobile legend. Automatically collapsing the
inset would only defer the collision until the reader reopened it. Active-step map charts now
also render as phone-only readout rows inside the scrolling Story sheet, and their map-overlay
copies are hidden on phones. Both representations receive the same live chart publication;
desktop retains its existing inset and collapse state. This affects the three scenes declaring
map chart placement: earthquake-query-lab, horizon and sun-and-shadow.

Status pills now join the title/legend vertical stack on phones and return to their original
right-side stack on desktop. This removes the observed ground-note overlap with flight-matrix's
title. The existing intentional one-line phone title clamp is unchanged.

`check-map-chart-responsive.mjs` extracts actual StoryView readout creation, chart publication,
inset painting and legend placement methods, and uses the production chart renderer, MapInset,
MapChips and CSS. Four DOM cases pass (top/right and bottom/right insets at 390 and 320 pixels),
including normal legend clicks and hit testing, chart bounds, updated values in both copies,
Show values interaction, nonoverlapping title/status pills and desktop restoration. No GPU is
used. Evidence: `/tmp/showcase-map-chart-responsive-wave22/report.json`; phone screenshot
reviewed. Existing actual SheetController/legend fixture also passes at
`/tmp/showcase-legend-layout-wave22/report.json`. Browsers closed; root owns rebuilt scene smoke
checks and final broad validation.
