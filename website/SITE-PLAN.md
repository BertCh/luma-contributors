# Site plan

The contributors docs site: a bespoke brand and a visual-first catalog of every GPU analysis
operation, organized by data type, use case and domain, plus "how it works" pages that explain the
system with the system's own live output.

## Principles

1. **luma.gl is a visual medium. Show, don't tell.** Every function page, card and concept page leads
   with a picture produced by luma.gl: live in the browser on WebGPU (WebGL2 fallback), with a static
   SVG fallback prerendered from the same seeded data. Prose comes second.
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
| `/how-it-works` and `/how-it-works/$topic` | Meta explainers (see below) |
| `/docs/$` | Existing long-form reference, compiled from `docs/**/*.md` (keep working) |

How-it-works topics (slugs): `anatomy` (data to GPU vectors to command graph to dispatch to
readback, with a live animated graph), `command-graph` (nodes, resources, hazards, compile and
encode, drawn from a real lens record), `data-on-the-gpu` (columns, chunks, vector formats),
`lenses` (computation records and snapshots: what the lens pages show and how they are recorded),
`upstream-first` (how this library sits on published luma.gl: package diagram).

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
| Lead (contracts) | `SITE-PLAN.md`, `src/catalog/schema.ts`, `src/catalog/taxonomy.ts`, `src/gpu/types.ts`, `vite.config.ts`, `package.json` |

Need a dependency or a contract change? Write it in your final report; the lead applies it.
Existing `src/docs.ts`, `src/routes/docs/$.tsx` and `build/markdown-plugin.ts` stay working; the
design-system owner may restyle the docs route.

## Shared interfaces

- `src/catalog/schema.ts`: `CatalogEntry`, `PreviewSpec`, `PreviewKind`.
- `src/catalog/taxonomy.ts`: `DATA_TYPES`, `USE_CASES`, `DOMAINS`.
- `src/catalog/index.ts` (catalog owner): `catalog: CatalogEntry[]`, `getEntry(id)`,
  `filterEntries({dataType, useCase, domain, query})`. It imports every `src/catalog/entries/*.ts` with
  `import.meta.glob(..., {eager: true})`.
- `src/gpu/types.ts`: `GpuScene`, `GpuSceneLoader`, `GpuSceneContext`.
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
