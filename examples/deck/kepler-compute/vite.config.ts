// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fileURLToPath} from 'node:url';
import {createExampleViteConfig} from '../../vite.config.shared';
import {createStandaloneViewerServePlugin} from './vite-inline-single-file-plugin';
import {
  createKeplerComputeDeckCompatPlugin,
  createKeplerComputeFilterPlugin,
  patchDeckCoordinateSystemImport,
  patchEditablePathLayerSource,
  patchKeplerGlobeBackfaceCullSource,
  patchKeplerComputeFilterSource,
  patchTiles3DRemovedExtensionsSource
} from './kepler-compute-transform';

const repositoryDirectory = fileURLToPath(new URL('../../../', import.meta.url));
export default createExampleViteConfig({
  config: {
    plugins: [
      createKeplerComputeFilterPlugin(),
      createKeplerComputeDeckCompatPlugin(),
      createStandaloneViewerServePlugin(
        fileURLToPath(new URL('./dist/standalone-viewer.html', import.meta.url))
      )
    ],
    resolve: {
      alias: [
        // Kepler 3.3 depends on deck.gl 9 and luma.gl 9 packages that luma.gl 10 folded into its
        // public entry points. Resolve them to the one published luma.gl 10 copy.
        {
          find: /^@luma\.gl\/(webgl\/constants|constants)$/,
          replacement: '@luma.gl/core/webgl-constants'
        },
        {find: /^@luma\.gl\/webgl$/, replacement: '@luma.gl/core/webgl'},
        {find: /^@luma\.gl\/effects$/, replacement: '@luma.gl/engine/effects'},
        {find: /^@luma\.gl\/gltf$/, replacement: '@luma.gl/engine/gltf'},
        {find: /^@luma\.gl\/shadertools$/, replacement: '@luma.gl/shaders'},
        {
          find: /^@kepler.gl\/utils\/jenks$/,
          replacement: `${repositoryDirectory}node_modules/@kepler.gl/utils/dist/esm/jenks.js`
        },
        {find: /^process$/, replacement: `${repositoryDirectory}node_modules/process/browser.js`},
        {
          find: /^assert$/,
          replacement: `${repositoryDirectory}node_modules/assert/build/assert.js`
        },
        {
          find: /^@math.gl\/core$/,
          replacement: `${repositoryDirectory}examples/deck/kepler-compute/compat/math-core.js`
        },
        {
          find: /^@loaders.gl\/compression$/,
          replacement: `${repositoryDirectory}examples/deck/kepler-compute/compat/compression.js`
        },
        {
          // Kepler 3.3 bundles apache-arrow 21.2 while the repository root resolves 17.0; two runtimes
          // break Kepler's `instanceof arrow.Table` checks on native Arrow uploads. Resolve one copy.
          find: /^apache-arrow(\/.*)?$/,
          replacement: `${repositoryDirectory}node_modules/@kepler.gl/utils/node_modules/apache-arrow$1`
        }
      ],
      dedupe: [
        'react',
        'react-dom',
        'react-redux',
        'styled-components',
        ...[
          'core',
          'react',
          'layers',
          'extensions',
          'aggregation-layers',
          'geo-layers',
          'mesh-layers',
          'mapbox',
          'widgets'
        ].map(name => `@deck.gl/${name}`),
        '@deck.gl-community/editable-layers',
        '@luma.gl/core',
        '@luma.gl/engine',
        '@luma.gl/graph',
        '@luma.gl/shaders'
      ]
    },
    optimizeDeps: {
      include: [
        '@kepler.gl/utils',
        'd3-dsv',
        'parquet-wasm/esm',
        '@loaders.gl/core/arrow-geometry/geometry-codecs'
      ],
      rolldownOptions: {
        plugins: [
          {
            name: 'kepler-tiles3d-removed-extensions-optimizer',
            transform(source, id) {
              const normalized = id.replaceAll('\\', '/');
              if (
                !normalized.includes('/@loaders.gl/tiles-3d/') ||
                !normalized.endsWith('/tiles-3d-source.js')
              )
                return null;
              const patched = patchTiles3DRemovedExtensionsSource(source);
              return patched === source ? null : {code: patched, map: null};
            }
          },
          {
            name: 'kepler-deck-10-coordinate-system-optimizer',
            transform(source, id) {
              const patched = patchDeckCoordinateSystemImport(source);
              return patched === source ? null : {code: patched, map: null};
            }
          },
          {
            name: 'kepler-globe-label-cull-optimizer',
            transform(source, id) {
              if (
                !id
                  .replaceAll('\\', '/')
                  .endsWith('/deckgl-layers/dist/esm/globe/globe-backface-cull.js')
              )
                return null;
              const patched = patchKeplerGlobeBackfaceCullSource(source);
              return patched === source ? null : {code: patched, map: null};
            }
          },
          {
            name: 'kepler-editable-path-anchor-optimizer',
            transform(source, id) {
              if (
                !id
                  .replaceAll('\\', '/')
                  .endsWith('/editable-layers/dist/editable-layers/editable-path-layer.js')
              )
                return null;
              const patched = patchEditablePathLayerSource(source);
              return patched === source ? null : {code: patched, map: null};
            }
          },
          {
            name: 'kepler-resident-filter-plot-optimizer',
            transform(source, id) {
              if (
                id.replaceAll('\\', '/').includes('/@kepler.gl/utils/') &&
                id.endsWith('/filter-utils.js')
              ) {
                return {code: patchKeplerComputeFilterSource(source), map: null};
              }
              return null;
            }
          }
        ]
      }
    },
    server: {port: 5318, strictPort: true, open: false},
    build: {chunkSizeWarningLimit: 3000}
  }
});
