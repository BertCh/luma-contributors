# Gridlight: the contributors docs site

A visual-first docs site for `@luma-contributors/analysis`, built with
[TanStack Start](https://tanstack.com/start) (React + Vite) and prerendered to static HTML. Brand,
palette, type and voice are in [BRAND.md](./BRAND.md); routes, ownership and contracts are in
[SITE-PLAN.md](./SITE-PLAN.md).

```bash
yarn website-dev     # http://localhost:3000
yarn website-build   # writes website/dist/client (about 680 static pages)
yarn website-deploy  # builds for the GitHub Pages sub-path and publishes to the gh-pages branch
```

## Deploying

The site is published on GitHub Pages at `https://<owner>.github.io/<repository>/`, served from
the `gh-pages` branch. `yarn website-deploy` (`scripts/deploy-pages.mjs`) builds with
`SITE_BASE_PATH=/<repository>/` and force-pushes the build as a single commit to that branch; it
does not touch the working tree. The site reads `modules/*/dist`, so run `yarn build` first when
package sources changed. To check a sub-path build before publishing:

```bash
cd website
SITE_BASE_PATH=/luma-contributors/ yarn build
node scripts/check-all-pages.mjs --base /luma-contributors
node scripts/deploy-pages.mjs --skip-build
```

After a deploy, the same checker can open the published pages:
`node scripts/check-all-pages.mjs --origin https://<owner>.github.io --base /luma-contributors`.
Pages serves from the `gh-pages` branch; if a push there does not start a build, request one with
`gh api -X POST repos/<owner>/<repository>/pages/builds`.

`SITE_BASE_PATH` becomes Vite's `base`: the router, asset URLs and links in rendered docs follow
it. Markdown site routes such as `/explore/` and `/fn/point-density` receive this base at build time;
external URLs, fragment links and already-prefixed site routes retain their destinations.
A plain `src` or `href` to a file in `public/` must go through `getPublicUrl`
(`src/public-url.ts`). Without the variable the site builds for `/`, as before.

## What is on the site

| Route | Page |
| --- | --- |
| `/` | Live luma.gl hero, browse by data type and use case |
| `/explore` | All operations, filtered by data type, use case and domain, with search |
| `/explore/{data,use,domain}/<id>` | Facet landing pages |
| `/fn/<id>` | One operation: live preview, summary, params, outputs, example, next steps, recorded command graph |
| `/demo-data` | Sources, provider credits and terms for the website's data extracts |
| `/how-it-works/<topic>` | Thirteen explainers in reading order: real code, recorded command graphs, stepped figures, workflows |
| `/docs/<path>` | Long-form reference compiled from `docs/**/*.md` |

## How it is built

- **Live pictures.** Catalog pages and cards start with a theme-matched WebP captured from the
  operation's GPU output. Hover or keyboard focus starts the card's luma.gl scene. Every scene
  declares what the pointer is for its operation: an observer or source to place, a sun to aim, a
  distance to reach, a level to pick, a window to move, a before/after curtain, an element to probe,
  a site to drag, a playhead, or a pad of two props. Glyphs show the handle and a chip reads out the
  real prop values. Arrow keys move the same gesture and Home returns it to the poster state. The
  function page runs the same gesture on its hero, keeps the sliders in step with it, and a click
  pins it. When a compute adapter is unavailable, the poster stays visible.
- **Data.** The compact inputs in `src/gpu/data` ship locally with the scene code. The source notes
  beside the extracts document dates, units, sampling and preparation. The build traces scene inputs
  to the source registry and includes attribution on each function page. See the
  [repository data notice](../THIRD-PARTY-DATA.md#website-demo-extracts).
- **Catalog.** One `CatalogEntry` per operation in `src/catalog/entries/<group>.ts`, organized by
  `src/catalog/taxonomy.ts`. `build/catalog-plugin.ts` serves lightweight summaries for cards and
  filters; function pages load their group's full entries on demand.
- **Lenses.** `src/lens/generated/*.json` are compact view models of `docs/lens-records/*.json`
  (command graphs, buffers, timings, small snapshots).
- **Markdown.** `build/markdown-plugin.ts` compiles `docs/**/*.md` to HTML at build time.

## Maintenance

```bash
cd website
node --experimental-strip-types scripts/extract-inventory.ts   # after analysis API changes
node --experimental-strip-types scripts/check-catalog.ts       # validates every entry against the source
node --experimental-strip-types src/lens/generate-view-models.ts   # after lens records change
for page in building-blocks contract iteration upstream-first; do   # other how-it-works page data
  node --experimental-strip-types src/components/meta/pages/$page/generate-data.ts
done
```

Full lens view-model generation also rewrites `summary.json` and the lenses page's
`lenses-data.json` after its models are written successfully. Run it without scenario arguments
to refresh aggregate byte totals. File sizes count UTF-8 bytes, including non-ASCII text;
the page histogram and median count compute nodes, and featured sizes use the files on disk.
Selected-scenario generation refreshes the lenses page metadata but leaves full summary totals
to a full generation. The workflow page's `WorkflowRecord.nodeCount` instead counts all graph
nodes, including copy nodes, and is displayed as “graph nodes”.

Inventory extraction follows generic constructor constraints before collecting props. For union
contracts such as forest inference, it includes both input arms and keeps shared fields such as
`model` and `rowCount` required; a field optional in any arm is displayed as optional. The API guide
states the relationship between mutually exclusive inputs.

After regenerating lens models, run the read-only parity check from the repository root:
`node_modules/.bin/vitest run --project node --maxWorkers=2 modules/lenses/test/computation-lenses/website-view-model-parity.node.spec.ts`.
It compares scenario membership and graph counts/data against every committed recording, checks
`src/lens/summary.json`, and verifies the nearest-weights capture and row-input contract. A full
generator run adds missing models and removes models whose recordings no longer exist.

Live operation runs (run from `website`; reports land in `website/.reports/`):

```bash
node scripts/verify-runs.mjs [--group <group>]   # verify runs and their pointer gestures on real WebGPU
node scripts/verify-demos.mjs [demo id...]       # verify the flagship demos' gestures (needs the harness server)
node scripts/audit-card-interactions.mjs         # summarize the gesture measures from the verify reports
node scripts/snapshot-runs.mjs                   # refresh public/runs snapshots
node --experimental-strip-types scripts/audit-demo-data.ts [--json]  # trace scene inputs
node scripts/check-all-pages.mjs                 # check every built page (needs yarn build)
node scripts/check-card-interactions.mjs http://localhost:3000  # against a running site
```

`check-catalog.ts` checks symbols, import paths, prop names, facet ids and length limits, and
reports operations that have no entry yet. `src/routeTree.gen.ts` is generated by the router plugin;
commit it but don't edit it. `lens-staging/` holds the old Docusaurus lens plugins, kept for reference.
