# Site plan

The contributors docs site: a bespoke brand and a visual-first catalog of every GPU analysis
operation, organized by data type, use case and domain, plus "how it works" pages that explain the
system with the system's own live output.

## Principles

1. **Lead with the operation.** Every catalog card and function page opens with a theme-matched
   WebP of the operation's GPU output. The same scene runs locally on WebGPU when the reader
   hovers or focuses a card, and the pointer does what a reader expects for that operation (see
   "Pointer interaction" in `src/gpu/runs/README.md`). Concept pages use the diagrams and records
   that explain their topic.
2. **Very concise.** One-sentence summaries, at most three-sentence explanations, at most eight
   parameters, examples of at most 25 lines. Cut anything a reader can see in the picture.
3. **Static first.** Every route is prerendered by TanStack Start. Live GPU canvases are client-only
   islands loaded with dynamic `import()`, so luma.gl never blocks first paint and never runs during
   prerendering.
4. **Meta.** The system explains itself: real command graphs, resources, timings and data snapshots
   from `docs/lens-records/` drawn as diagrams and animations.
5. **Accurate.** Symbols, import paths, prop names and types match the source in `modules/*/src`.
   `scripts/check-catalog.ts` verifies this.

## Routes

| Route | Page |
| --- | --- |
| `/` | Home: live GPU hero, what it is in one line, browse by data type (visual grid), use cases, how-it-works teaser |
| `/explore` | Whole catalog: faceted filter (data type, use case, domain) and text search; filters live in URL search params |
| `/explore/data/$id`, `/explore/use/$id`, `/explore/domain/$id` | Prerendered facet landing pages |
| `/fn/$id` | Function page: live preview, summary, explanation, import line, params table, outputs, example, related, lens graph |
| `/demo-data` | Sources and terms for the compact datasets used in catalog scenes |
| `/how-it-works` and `/how-it-works/$topic` | Meta explainers (see below) |
| `/docs/$` | Existing long-form reference, compiled from `docs/**/*.md` (keep working) |

How-it-works pages, in reading order (slugs in `src/components/meta/topic-slugs.ts`; plan and rules
in `dev-docs/roadmap.md` section 9):

- One run: `anatomy` (the kernel-density call line by line, its recorded graph, its raster),
  `data-on-the-gpu` (buffers, views, offsets, CSR, capacity and overflow, no-data, precision),
  `command-graph` (read and write sets become hazards and an order), `scheduling` (compile once,
  encode every frame, buffer lifetimes, shared passes), `kernels` (invocations, workgroups, the guard).
- Parallel analysis: `building-blocks` (reduce, scan, scatter, sort, grid index, stepped on small data
  beside the recorded kernels), `iteration` (unrolled rounds gated on the GPU: a recorded road-network
  search), `contract` (the contributor contract, with the real validators running in the page).
- In practice: `workflows` (five recipes step by step, with code and each step's picture).
- The system: `lenses` (what a record holds and how one is made), `this-page`, `upstream-first`,
  `glossary`.

Each page is a folder `src/components/meta/pages/<slug>/` with `index.tsx` (exports `Page`), `data.ts`
(lens scenarios and code examples the route loader prepares) and its figures, CSS and data. Pages
compose `src/components/meta/page-kit.tsx`. Code on a page is a `*.example.ts` file the website
typecheck compiles, imported with `?raw`. Numbers come from view models, `src/lens/summary.json` or a
page's `generate-data.ts`; none are typed in.

## File ownership (one owner per path; do not edit others' files, import them)

| Owner | Paths |
| --- | --- |
| Brand and design system | `BRAND.md`, `src/styles/**`, `src/components/ui/**`, `src/components/brand/**`, `src/routes/__root.tsx`, `src/routes/index.tsx`, `public/**`, `src/styles.css` (may replace with `src/styles/`) |
| GPU runtime and previews | `src/gpu/**` except `types.ts` (contract, owned by the lead) |
| Catalog inventory and checks | `scripts/**`, `src/catalog/generated/**` |
| Catalog content | one file each: `src/catalog/entries/<group>.ts` exporting `entries: CatalogEntry[]` |
| Catalog index and pages | `src/catalog/index.ts`, `src/components/catalog/**`, `src/routes/explore*`, `src/routes/explore/**`, `src/routes/fn/**` |
| How it works | `src/routes/how-it-works*`, `src/routes/how-it-works/**`, `src/components/meta/**`, `src/lens/**` |
| Flagship live demos | `src/gpu/demos/**` (real analysis operations run on the GPU in the browser) |
| Live operation runs | `src/gpu/runs/**` (one run per catalog entry; see `src/gpu/runs/README.md`), `public/runs/**` (snapshots) |
| Lead (contracts) | `SITE-PLAN.md`, `src/catalog/schema.ts`, `src/catalog/taxonomy.ts`, `src/gpu/types.ts`, `vite.config.ts`, `package.json` |

Need a dependency or a contract change? Write it in your final report; the lead applies it.
Existing `src/docs.ts`, `src/routes/docs/$.tsx` and `build/markdown-plugin.ts` stay working; the
design-system owner may restyle the docs route.

## Shared interfaces

- Hero order on a function page: flagship demo, then the entry's run (`src/gpu/runs/registry.ts`: `hasRun`, `getRunScene`, `getRunParameters`), then the entry-specific preview, then the generic kind.
- `src/catalog/schema.ts`: `CatalogEntry`, `PreviewSpec`, `PreviewKind`.
- `src/catalog/taxonomy.ts`: `DATA_TYPES`, `USE_CASES`, `DOMAINS`.
- `src/catalog/index.ts` (catalog owner): `catalog: CatalogEntry[]`, `getEntry(id)`,
  `filterEntries({dataType, useCase, domain, query})`. It imports every `src/catalog/entries/*.ts` with
  `import.meta.glob(..., {eager: true})`.
- `src/gpu/types.ts`: `GpuScene`, `GpuSceneLoader`, `GpuSceneContext`. A scene with its own pointer
  gesture sets `gesture` and `resetPointer` on its instance and calls `reportReadout` and
  `reportParameters` from its context.
- `src/gpu/runs/interaction.ts`: `RunInteraction` (the gesture a run declares), `bindAxes`,
  `aimAround`, glyph shapes. `src/gpu/runs/pick.ts`: CPU lookups under the pointer.
  `src/gpu/runs/pointer-driver.ts`: the same gesture behaviour for hand-written demo scenes.
- `src/gpu/gpu-canvas.tsx` (GPU owner): `<GpuCanvas scene={loader} fallback={node} seed label aspectRatio />`.
- `src/gpu/previews/index.ts` (GPU owner): `getPreviewScene(kind): GpuSceneLoader` and
  `<PreviewFallback kind seed />`, a static SVG for every `PreviewKind`.
- `src/components/ui/**` (design owner): `Card`, `Chip`, `CodeBlock`, `ParamTable`, `Section`,
  `PageHeader`, `Callout` and the design tokens in CSS custom properties. Pages use tokens, never raw colors.

## Conventions

- TypeScript strict; SPDX header on every source file; no abbreviated names (see AGENTS.md).
- Light and dark themes from the same tokens (`prefers-color-scheme`, plus a toggle).
- Works at phone width; no horizontal page scroll.
- Respect `prefers-reduced-motion`: render a still frame.
- Accessibility: canvases get `role="img"` and an `aria-label`; text contrast meets WCAG AA.
- Verify with `yarn website-build` (all routes prerender, the build fails on errors) and
  `npx tsc -p website/tsconfig.json`.
