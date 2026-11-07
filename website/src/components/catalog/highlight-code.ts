// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Syntax highlighting for catalog examples. Called from route loaders, so it runs during
// prerendering (the HTML ships highlighted) and lazily on client navigations.

import type {HighlighterCore} from 'shiki/core';

type Highlighter = HighlighterCore;

let highlighterPromise: Promise<Highlighter> | undefined;

function createCodeHighlighter(): Promise<Highlighter> {
  return (async () => {
    const [{createHighlighterCore}, {createJavaScriptRegexEngine}, language, light, dark] =
      await Promise.all([
        import('shiki/core'),
        import('shiki/engine/javascript'),
        import('shiki/langs/typescript.mjs'),
        import('shiki/themes/github-light.mjs'),
        import('shiki/themes/github-dark.mjs')
      ]);
    return createHighlighterCore({
      themes: [light.default, dark.default],
      langs: [language.default],
      engine: createJavaScriptRegexEngine()
    });
  })();
}

/** Returns highlighted HTML (light and dark through CSS variables), or null if highlighting fails. */
export async function highlightCode(code: string, language = 'typescript'): Promise<string | null> {
  try {
    highlighterPromise ??= createCodeHighlighter();
    const highlighter = await highlighterPromise;
    return highlighter.codeToHtml(code, {
      lang: language,
      themes: {light: 'github-light', dark: 'github-dark'},
      defaultColor: false
    });
  } catch {
    return null;
  }
}
