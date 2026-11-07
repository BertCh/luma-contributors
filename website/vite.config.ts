// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {tanstackStart} from '@tanstack/react-start/plugin/vite';
import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';
import {catalogPlugin} from './build/catalog-plugin';
import {listDocPagePaths, markdownPlugin} from './build/markdown-plugin';
import {listCatalogPagePaths} from './src/catalog/routes';
import {TOPICS} from './src/components/meta/topics';

const websiteDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = path.resolve(websiteDirectory, '..');
const docsDirectory = path.join(repositoryDirectory, 'docs');

// Every route is prerendered to static HTML. Docs pages are listed explicitly so that pages
// reachable only by client-side navigation still get an HTML file.
const pages = [
  '/',
  '/docs',
  ...listDocPagePaths(docsDirectory),
  ...listCatalogPagePaths(),
  '/how-it-works',
  ...TOPICS.map(topic => `/how-it-works/${topic.slug}`)
].map(pagePath => ({path: pagePath}));

export default defineConfig({
  server: {
    port: 3000,
    fs: {allow: [repositoryDirectory]}
  },
  plugins: [
    markdownPlugin({docsDirectory}),
    catalogPlugin({entriesDirectory: path.join(websiteDirectory, 'src/catalog/entries')}),
    tanstackStart({
      prerender: {enabled: true, crawlLinks: false, failOnError: true},
      pages
    }),
    react()
  ]
});
