# Gridlight brand

Site name: **Gridlight** (wordmark `grid` + accent `light`). Packages stay `@luma-contributors/*`.
Tagline: **GPU geospatial analysis on luma.gl.** Hero line: "Spatial analysis, on the GPU."

## Voice

Very concise, precise, second person, present tense. One-sentence summaries, no adjectives that a
picture can replace. Say what the operation takes and returns. No exclamation marks, no emoji.

## Idea

luma.gl is a visual medium, so the site is the demo. Light on a dark map: glowing teal on ink, with
one amber signal color reserved for hotspots and warnings. Light theme is paper and ink with a
deeper teal. A faint 32px grid (`.grid-surface`) echoes GPU tiles and map graticules.

## Logo

`src/components/brand/logo.tsx`: a 3x3 tile grid, a coastline-like path through it, one lit tile
(a lit GPU cell) and one amber hotspot. Monochrome via `currentColor`.

## Palette (tokens in `src/styles/tokens.css`)

| Token | Light | Dark |
| --- | --- | --- |
| `--color-background` | #f5f6f2 | #070a10 |
| `--color-surface` | #ffffff | #0d121b |
| `--color-text` | #0b1220 | #e8eef6 |
| `--color-muted` | #4d5a6b | #93a1b4 |
| `--color-border` | #d5d9d0 | #1f2a3a |
| `--color-accent` | #07745f | #3de0c5 |
| `--color-signal` | #b4570a | #ffb454 |

Theme: `prefers-color-scheme` by default; the header toggle sets `data-theme` on `<html>` and stores
it in `localStorage` (`gridlight-theme`, try/catch). An inline script in `<head>` applies it before
paint. GPU scenes read the theme from `GpuSceneContext.theme`.

## Type (Google Fonts)

- Display: Space Grotesk 500/600/700 (headings, wordmark, numerals)
- Text: Inter 400/500/600
- Mono: JetBrains Mono 400/500 (code, ids, counts)

Scale: display `clamp(2.25rem, ..., 4.25rem)`, title `clamp(1.75rem, ..., 2.5rem)`, large 1.125rem,
base 0.9375rem, small 0.8125rem.

## Icons

`src/components/brand/facet-icon.tsx`: `<FacetIcon id="points" />` for every data type, use case and
domain id (24px grid, 1.6 stroke, `currentColor`). Add a facet: add one entry to `FACET_ICON_PATHS`.

## Components (`src/components/ui`)

`Card` (media, title, text, meta; stretched-link when `interactive`), `Chip` / `getChipClassName`
(for links), `CodeBlock` (pre-highlighted `html` passthrough, plain fallback, copy button),
`ParamTable` (stacks on phones), `Section`, `PageHeader`, `Callout`.
`src/components/brand`: `SiteHeader`, `SiteFooter`, `ThemeToggle`, `LogoMark`, `FacetIcon`.

## Inspiration notes

Research was limited (one site rate-limited, deck.gl's landing page fetch returned only hero copy), so
these ideas combine what was verified with well-known patterns of each site.

1. **deck.gl** (https://deck.gl): one plain sentence under a big title, one primary button, the live
   canvas does the selling. Borrowed: hero = full-bleed live canvas, one line of copy.
2. **Observable Plot** (https://observablehq.com/plot/): every doc entry opens with a rendered chart,
   then a short snippet. Borrowed: picture first, code second, on every function page.
3. **Shadertoy** (https://www.shadertoy.com): dense grid of live-rendered thumbnails with the title
   underneath. Borrowed: facet cards are thumbnails first, label second.
4. **Distill.pub** (https://distill.pub): generous prose measure, figures that break out wider than text.
   Borrowed: 720px prose, wide figures for explainers.
5. **Bartosz Ciechanowski** (https://ciechanow.ski): interactive diagrams inline with sentences, each
   with a single control. Borrowed: how-it-works figures with one slider or toggle.
6. **Stripe docs** (https://docs.stripe.com): code on the right, tinted code surface, copy button on
   hover. Borrowed: tinted `CodeBlock` surface, hover copy button.
7. **Linear / Vercel Geist** (https://linear.app, https://vercel.com/geist): near-monochrome UI with
   one accent, hairline borders, tight display type. Borrowed: 1px borders, one accent plus one signal.
8. **Mapbox GL docs** (https://docs.mapbox.com/mapbox-gl-js/): example gallery with tiny map
   thumbnails and tags. Borrowed: thumbnails with count and facet chips.
9. **CARTO / Felt** (https://carto.com, https://felt.com): dark basemaps with glowing data, light UI
   chrome around them. Borrowed: "light on dark map" palette.
10. **cuSpatial / Turf.js** (https://docs.rapids.ai/api/cuspatial/stable/, https://turfjs.org): function
   lists organized by what the data is. Borrowed: facets by data type first, with function counts.
