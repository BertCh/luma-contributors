# luma.gl spatial analysis showcase

An interactive WebGPU atlas of luma.gl's experimental spatial-analysis contributors.

- Public site: <https://bertch.github.io/luma.gl/>
- Local development: `yarn workspace luma.gl-examples-deck-spatial-analysis-showcase start`
- Production build: `yarn workspace luma.gl-examples-deck-spatial-analysis-showcase build:pages`

The `map-contributors` branch deploys the production build to GitHub Pages. The Pages build uses
`/luma.gl/` as its Vite base path; local development continues to use `/`.

## Authoring and verification

Read the [scene builder guide](src/scenes/BUILDER-GUIDE.md),
[cartography guide](src/scenes/CARTOGRAPHY-GUIDE.md), and [dataset format](src/data/README.md).
Scenes and datasets are discovered by filename; new stories do not require a registry edit.

From the repository root:

```sh
yarn workspace luma.gl-examples-deck-spatial-analysis-showcase typecheck
yarn workspace luma.gl-examples-deck-spatial-analysis-showcase check
yarn workspace luma.gl-examples-deck-spatial-analysis-showcase build
node scripts/report-showcase-coverage.mjs --format=markdown
```

The checks validate bundled dataset assets, cartography metadata, color scales, story steps and
WGSL identifiers. A scene that cannot load fails the checks. Keep unfinished scene metadata under
a `.scene.ts.wip` filename until its implementation compiles and passes review.

The coverage report records executable call sites separately from scene metadata and lens imports.
Add `--functionality` to inspect public methods and option variants. This is a static inventory;
GPU execution and numerical correctness require separate browser and reference checks.

The [October 2026 expansion sprint](../../../dev-docs/archive/showcase-sprint-2026-10-24.md)
links the API inventory, dataset research, story designs, implementation ledger, and verification.
Research candidates are kept separate from datasets actually bundled with the showcase.

The [showcase overhaul](../../../dev-docs/research/showcase-overhaul-2026-10-25/README.md) records
the subsequent atlas redesign, domain research, independent numerical references, source-data
provenance and verification results.

To inspect the sprint stories against a running local server:

```sh
node examples/deck/spatial-analysis-showcase/scripts/check-sprint-scenes.mjs --help
```

Use a stable production preview for a multi-story browser sweep while other work is editing source
files. Vite hot reload can interrupt graph compilation and invalidate a readiness check.
Pass `--all` to discover every story from the served atlas, or `--scenes=` for a focused set.
Add `--mobile` to capture and check each story at 390 pixels wide. The JSON report is updated after
each scene so an interrupted long sweep retains its completed results.

`scripts/check-atlas-navigation.mjs` exercises search persistence, loading races, keyboard navigation
and narrow layouts. `scripts/capture-story-previews.mjs` refreshes featured cards from actual maps,
including their legends and attribution. Both accept `--url=` for the production preview.
