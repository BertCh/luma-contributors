# Lens rendering (staging)

Lens rendering pieces moved from the luma.gl `map-contributors` branch website (`website/src`). They
were written for Docusaurus and are not built, typechecked or wired into this site yet. Porting them
means running the remark plugins inside `../build/markdown-plugin.ts` and mounting live lenses from a
React component instead of a Docusaurus client module.

| Path | What it does |
| --- | --- |
| `src/remark/lens-directive.js` | Remark plugin that expands lens directives in docs pages into lens records from `docs/lens-records/` |
| `src/remark/lens-twin-html.js` | Builds the static HTML "twin" of a lens for MDX pages |
| `src/remark/lens-catalog.js` | Remark plugin that renders the lens catalog pages from `docs/table-of-contents.json` |
| `src/remark/lens-how-collapse.js` | Collapses the "how it works" sections of lens cards |
| `src/remark/lens-directive.test.mjs` | `node --test` tests for the directive plugin |
| `src/client-modules/ladder-lens.ts`, `live-lens-adapter.ts` | Docusaurus client modules that mount live lenses in the browser |
| `src/components/docs/lens-html.tsx`, `src/theme/MDXComponents/index.tsx` | `LensHtml` MDX component and its registration |
| `src/css/lens-card.css`, `lens-twin.css` | Lens styles |

## Wiring them into a Docusaurus site

This is how the luma.gl site wired them up (`website/docusaurus.config.js` on that branch):

```js
const remarkLensDirective = require('./src/remark/lens-directive');

// presets[classic].docs.remarkPlugins
[remarkLensDirective, {docsDirectory: path.resolve(__dirname, '../docs')}],
require('./src/remark/lens-how-collapse'),
[require('./src/remark/lens-catalog'), {docsDirectory: path.resolve(__dirname, '../docs')}]

// theme.customCss
'./src/css/lens-twin.css', './src/css/lens-card.css'

// a plugin's getClientModules()
path.resolve(__dirname, 'src/client-modules/ladder-lens.ts')
```

The plugins need `katex`. The site framework, layout and hosting are still to be decided.
