// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Plugin} from 'vite';

/** Replace script/link references in the emitted HTML with inline content and drop the emitted files. */
export function createInlineSingleFilePlugin(): Plugin {
  return {
    name: 'luma-inline-single-file',
    enforce: 'post',
    generateBundle(_options, bundle) {
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
  };
}
