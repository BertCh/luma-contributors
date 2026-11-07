// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {tanstackStart} from '@tanstack/react-start/plugin/vite';
import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';
import {listDocPagePaths, markdownPlugin} from './build/markdown-plugin';

const websiteDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = path.resolve(websiteDirectory, '..');
const docsDirectory = path.join(repositoryDirectory, 'docs');

// Every route is prerendered to static HTML. Docs pages are listed explicitly so that pages
// reachable only by client-side navigation still get an HTML file.
const pages = ['/', '/docs', ...listDocPagePaths(docsDirectory)].map(pagePath => ({
  path: pagePath
}));

export default defineConfig({
  server: {
    port: 3000,
    fs: {allow: [repositoryDirectory]}
  },
  plugins: [
    markdownPlugin({docsDirectory}),
    tanstackStart({
      prerender: {enabled: true, crawlLinks: false, failOnError: true},
      pages
    }),
    react()
  ]
});
