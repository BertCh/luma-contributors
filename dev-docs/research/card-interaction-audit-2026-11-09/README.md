# Card header pictures: interaction and picture audit (2026-11-09)

Evidence behind the WEB-INT and WEB-PIC items in [../../roadmap.md](../../roadmap.md#7-plan-card-header-pictures-web-int-web-pic). It records the state **before** the gestures and picture fixes of 2026-11-09; the numbers below are the baseline, not the current site. It covers the 428 operation runs in `website/src/gpu/runs/entries/` at commit `edf9ca10` plus the uncommitted working tree of 2026-11-09. The ten flagship demos in `website/src/gpu/demos/` were reviewed separately and are listed in the roadmap.

## Files

| File | What |
| --- | --- |
| [entries.md](./entries.md) | One row per run: what the card scrubs today, the proposed pointer gesture and binding, the cursor glyph and readout, feasibility, effort and picture issues with fixes. Generated from the JSON below |
| `<group>.json` (12 files) | The audit of one catalog group: `summary`, `sharedChanges` (helper files where one change fixes many runs), `frameworkNeeds`, and `entries[]` |
| [baseline.json](./baseline.json) | Measurements per run from `website/scripts/verify-runs.mjs` on WebGPU (dark theme): pass or fail, warnings, start time, share of non-background pixels, and the share of pixels the current scrub parameter changes over its full range |

## Method

1. **Census.** A script read every `<id>.ts` and `<id>.parameters.ts` and derived the parameter the card scrubs, using the selection rule in `src/gpu/runs/define-run.ts` (`scrubParameter`) and `src/gpu/canvas-pointer.ts`. The census misreads runs built through shared factories (`defineTableModelRun`, `definePanelRun`, the triangulation helper): it reported at least nine of them as rebuild scrubs. The audits corrected these from the code, and the numbers below use the audited values.
2. **Baseline.** `node scripts/verify-runs.mjs` over all runs: 425 of 428 pass. Three fail with "output is not drawn" (`geometry-cleanup`, `line-precision-reduction`, `segment-snap`). 26 sliders do not change the picture. Start time (new device, compile, first frame) has a median of 565 ms and a 90th percentile of 635 ms on an Apple M-class machine with 12 cores; two runs exceed 1 s (`s2-cap-covering` 2.9 s, `offset-curve-components` 1.6 s).
3. **Audit.** One reviewer per group read each run, its parameters, its shared helpers and its catalog text, looked at the dark poster, and filled the schema below. The reviewers did not run the scenes. Feasibility is judged from the run's code and, where unclear, from the operation's source. Treat `pictureQuality` and the issue lists as reviewed judgment, not measurement.

## Schema

Each entry has `id`, `operation`, `currentScrub {parameter, kind, mechanism}`, `pictureQuality`, `issues[] {code, detail, fix}`, `proposal {archetype, secondary?, binding, glyph, readout, feasibility, feasibilityNote, effort}` and `priority`.

**Archetypes** (what the pointer is):

| Code | The pointer | Runs |
| --- | --- | --- |
| `probe` | hovers an element and lights up what the operation links to it | 142 |
| `drag` | carries one piece of the input geometry | 53 |
| `reach` | sets a distance by how far it is from an anchor or from the geometry | 52 |
| `place` | is a spatial input: observer, source, target, pour point, query point | 44 |
| `level` | picks a threshold, break or contour level from the value or axis under it | 40 |
| `swipe` | is a curtain between input and output | 26 |
| `time` | is a playhead over time, iterations or a sequence | 25 |
| `window` | is the centre of a box, mask, kernel footprint or region | 18 |
| `pad` | drives two real props, one per axis | 17 |
| `aim` | sets a direction by its angle around an anchor | 11 |

**Feasibility:** `ready` (113): the values are already written per frame, only the binding changes. `update` (107): a small run change makes the value per-frame. `render` (167): no re-run, the gesture highlights or clips existing outputs. `rebuild` (41): the prop is fixed when the graph is built, so the gesture snaps to a few pre-built variants.

**Effort:** `S` 147, `M` 266, `L` 15.

**Picture quality:** 1: 28 runs. 2: 149. 3: 182. 4: 69. 5: none.

**Issue codes** (827 issues on 407 runs; 21 runs have none):

| Code | Meaning | Count |
| --- | --- | --- |
| `data-headline` | the most prominent control is an input-data knob while a real prop exists | 138 |
| `duplicate` | looks the same as a sibling entry | 89 |
| `slow` | the headline interaction rebuilds the graph, or the scene is heavy | 84 |
| `no-contrast` | input and output cannot be told apart, or the effect is invisible at the defaults | 78 |
| `encoding` | wrong ramp, unreadable categories, a single meaningless bar | 73 |
| `chart-bare` | a chart with no reference line or a lone bar | 67 |
| `other` | suspected defects and mismatches, described per entry | 63 |
| `small` | the subject fills too little of the frame | 50 |
| `auto-range` | the colour range renormalizes after each run and hides the parameter's effect | 50 |
| `weak-range` | the parameter range has dead or saturated ends, or the default hides the effect | 49 |
| `clutter` | too many elements for a 320 px card | 47 |
| `data-fit` | the dataset or scenario does not suit the operation | 39 |

## Headline numbers

- 160 of 428 cards scrub an input-data knob (138 labelled `Data:`, 22 unlabelled). None of them has a real per-frame prop the card could scrub instead without a run change.
- 111 cards recompile the command graph on every scrub step.
- The current scrub changes a median of 10.5% of pixels over its full range; on 67 cards it changes under 2%.
- 135 pictures have under 8% non-background pixels.
- 280 proposals (113 ready, 167 render) need no change to how the operation is built.

## Regenerating

`entries.md` and `baseline.json` are a join of the group JSON and the baseline verify report. For the current state run `node scripts/verify-runs.mjs` and then `node scripts/audit-card-interactions.mjs` from `website/`.
