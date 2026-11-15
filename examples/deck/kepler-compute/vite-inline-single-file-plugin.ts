// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile} from 'node:fs/promises';
import type {Plugin} from 'vite';

/** Serve the built viewer verbatim, before Vite transforms the unbuilt HTML entry or adds HMR. */
export function createStandaloneViewerServePlugin(viewerPath: string): Plugin {
  return {
    name: 'luma-standalone-viewer-serve',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const viewerUrl = `${server.config.base}standalone-viewer.html`;
        if (request.url?.split('?')[0] !== viewerUrl) {
          next();
          return;
        }
        try {
          const html = await readFile(viewerPath, 'utf8');
          response.setHeader('Content-Type', 'text/html; charset=utf-8');
          response.setHeader('Cache-Control', 'no-cache');
          response.end(html);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === 'ENOENT') next();
          else next(error);
        }
      });
    }
  };
}

/** Replace script/link references in the emitted HTML with inline content and drop the emitted files. */
export function createInlineSingleFilePlugin(): Plugin {
  return {
    name: 'luma-inline-single-file',
    enforce: 'post',
    // Run after Vite's import analysis, which rewrites the __VITE_PRELOAD__ markers in each chunk.
    generateBundle: {
      order: 'post',
      handler(_options, bundle) {
        for (const [fileName, html] of Object.entries(bundle)) {
          if (!fileName.endsWith('.html') || html.type !== 'asset') continue;
          let source = String(html.source);
          for (const [assetName, asset] of Object.entries(bundle)) {
            if (assetName === fileName) continue;
            const escapedName = assetName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (asset.type === 'chunk' && assetName.endsWith('.js')) {
              const code = asset.code.replace(/<\/script/gi, '<\\/script');
              source = source.replace(
                new RegExp(`<script[^>]*src="[^"]*${escapedName}"[^>]*></script>`),
                () => `<script type="module">${code}</script>`
              );
            } else if (asset.type === 'asset' && assetName.endsWith('.css')) {
              const css = String(asset.source);
              source = source.replace(
                new RegExp(`<link[^>]*href="[^"]*${escapedName}"[^>]*>`),
                () => `<style>${css}</style>`
              );
            } else continue;
            delete bundle[assetName];
          }
          html.source = source;
        }
      }
    }
  };
}
