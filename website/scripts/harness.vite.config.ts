// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dev-only Vite config that serves run-harness.html without the TanStack Start SSR layer, which
// would otherwise answer every URL. scripts/verify-runs.mjs starts it. It is never part of the site
// build, so the harness page stays out of production output and prerendering.

import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {defineConfig} from 'vite';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export default defineConfig({
  root: websiteDirectory,
  // Keep a separate dependency cache so the harness never fights the site's dev server.
  cacheDir: path.join(websiteDirectory, 'node_modules/.vite-harness'),
  server: {fs: {allow: [path.resolve(websiteDirectory, '..')]}},
  optimizeDeps: {entries: ['run-harness.html']}
});
