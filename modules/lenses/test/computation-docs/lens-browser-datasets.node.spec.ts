// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createServer} from 'vite';
import {expect, it} from 'vitest';
import {REPOSITORY_ROOT} from './lens-directives';

it('loads Vite browser JSON modules without native JSON import attributes', async () => {
  const temporary = mkdtempSync(join(tmpdir(), 'lens-browser-json-'));
  // Exercise the client transform, not Vite's Node/SSR module runner. No listening server or GPU.
  const server = await createServer({
    configFile: false,
    root: REPOSITORY_ROOT,
    logLevel: 'silent',
    appType: 'custom',
    optimizeDeps: {noDiscovery: true, include: []},
    server: {middlewareMode: true, hmr: false, watch: null}
  });
  try {
    const sourceRoot = '/modules/lenses/src/computation-lenses';
    const dataRoot = join(REPOSITORY_ROOT, sourceRoot.slice(1), 'data');
    const filenames = readdirSync(dataRoot)
      .filter(file => file.endsWith('.json'))
      .sort();
    const loader = await server.transformRequest(`${sourceRoot}/embedded-dataset-loaders.js`);
    expect(loader).not.toBeNull();
    let code = loader!.code;
    for (const filename of filenames) {
      const url = `${sourceRoot}/data/${filename}?import`;
      const transformed = await server.transformRequest(url);
      expect(transformed, filename).not.toBeNull();
      // Vite serves JSON imports as JavaScript modules. Node's native module loader enforces the
      // same incompatibility as Chromium when type:json is attached to a JavaScript response.
      const moduleFile = join(temporary, `${filename}.mjs`);
      writeFileSync(moduleFile, transformed!.code);
      expect(code, filename).toContain(url);
      code = code.replaceAll(url, pathToFileURL(moduleFile).href);
    }
    const loaderFile = join(temporary, 'loader.mjs');
    writeFileSync(loaderFile, code);
    const output = execFileSync(
      process.execPath,
      [
        '-e',
        `
      const nodeProcess = process;
      // Browser main threads and workers have no native Node process global.
      globalThis.process = undefined;
      (async () => {
        const {EMBEDDED_DATASET_LOADERS} = await import(${JSON.stringify(pathToFileURL(loaderFile).href)});
        const datasets = {};
        for (const [name, load] of Object.entries(EMBEDDED_DATASET_LOADERS)) datasets[name] = (await load()).default;
        nodeProcess.stdout.write(JSON.stringify(datasets));
      })().catch(error => {console.error(error); nodeProcess.exitCode = 1;});
    `
      ],
      {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 4 * 1024 * 1024}
    );
    const datasets = JSON.parse(output);
    expect(Object.keys(datasets).sort()).toEqual(filenames);
    for (const filename of filenames)
      expect(datasets[filename], filename).toEqual(
        JSON.parse(readFileSync(join(dataRoot, filename), 'utf8'))
      );
  } finally {
    await server.close();
    rmSync(temporary, {recursive: true, force: true});
  }
});
