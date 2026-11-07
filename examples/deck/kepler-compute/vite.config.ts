// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fileURLToPath} from 'node:url';
import {createExampleViteConfig} from '../../vite.config.shared';
import {
  createKeplerComputeFilterPlugin,
  patchKeplerComputeFilterSource
} from './kepler-compute-transform';

const repositoryDirectory = fileURLToPath(new URL('../../../', import.meta.url));
export default createExampleViteConfig({
  config: {
    plugins: [createKeplerComputeFilterPlugin()],
    resolve: {
      alias: [
        // Kepler's nested deck.gl 9 still imports the removed luma.gl 9 `webgl` package.
        {
          find: '@luma.gl/webgl/constants',
          replacement: `${repositoryDirectory}modules/core/src/webgl-constants/index.ts`
        },
        {
          find: '@luma.gl/webgl',
          replacement: `${repositoryDirectory}modules/core/src/webgl/index.ts`
        },
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
        ].map(name => `@deck.gl/${name}`)
      ]
    },
    optimizeDeps: {
      include: [
        '@kepler.gl/utils',
        'd3-dsv',
        'parquet-wasm/esm',
        '@loaders.gl/arrow-geometry/geometry-codecs'
      ],
      rolldownOptions: {
        plugins: [
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
