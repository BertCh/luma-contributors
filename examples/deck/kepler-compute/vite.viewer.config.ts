// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fileURLToPath} from 'node:url';
import {defineConfig, mergeConfig} from 'vite';
import baseConfig from './vite.config';
import {createInlineSingleFilePlugin} from './vite-inline-single-file-plugin';

/**
 * Builds dist/standalone-viewer.html: one HTML file with all JS and CSS inlined and no network
 * dependencies. The export step injects the workspace bundle at the placeholder comment.
 * Documented exclusions: no basemap tiles, remote icons or fonts are bundled (system fonts only),
 * and no wasm or workers are used by this viewer.
 */
export default mergeConfig(
  baseConfig,
  defineConfig({
    root: fileURLToPath(new URL('.', import.meta.url)),
    plugins: [createInlineSingleFilePlugin()],
    build: {
      outDir: 'dist',
      emptyOutDir: false,
      assetsInlineLimit: 100_000_000,
      cssCodeSplit: false,
      chunkSizeWarningLimit: 20_000,
      modulePreload: false,
      rollupOptions: {
        input: fileURLToPath(new URL('./standalone-viewer.html', import.meta.url)),
        output: {inlineDynamicImports: true}
      }
    }
  })
);
