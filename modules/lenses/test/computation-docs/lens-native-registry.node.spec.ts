// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {expect, it} from 'vitest';
import {REPOSITORY_ROOT} from './lens-directives';
import {writeNativeWorkspaceLoader} from './native-workspace-loader';

it('loads network scenarios through the public native ESM registry without a GPU', () => {
  const temporary = mkdtempSync(join(tmpdir(), 'lens-native-registry-'));
  try {
    // Erase TypeScript on demand, without bundling or rewriting JSON imports. Native Node still
    // enforces JSON attributes, unlike the Vitest SSR runner or an esbuild bundle.
    const launcher = join(temporary, 'registry.mjs');
    const workspaceLoader = writeNativeWorkspaceLoader(temporary);
    writeFileSync(
      launcher,
      `
      import {registerHooks} from 'node:module';
      import {existsSync, readFileSync} from 'node:fs';
      import {fileURLToPath, pathToFileURL} from 'node:url';
      import typescript from ${JSON.stringify(pathToFileURL(join(REPOSITORY_ROOT, 'node_modules/typescript/lib/typescript.js')).href)};
      const root = ${JSON.stringify(pathToFileURL(join(REPOSITORY_ROOT, 'modules/lenses/src/')).href)};
      const entries = {
        '@luma-contributors/lenses/computation-lenses': root + 'computation-lenses/index.ts',
        '@luma-contributors/lenses/computation-lenses/scenarios/index': root + 'computation-lenses/scenarios/index.ts'
      };
      registerHooks({
        resolve(specifier, context, nextResolve) {
          if (entries[specifier]) return nextResolve(entries[specifier], context);
          if (context.parentURL?.startsWith(root) && specifier.startsWith('.')) {
            const url = new URL(specifier, context.parentURL);
            const path = fileURLToPath(url);
            for (const suffix of ['', '.ts', '.js', '/index.ts']) {
              if (existsSync(path + suffix)) {
                if (suffix === '' && !/\\.(ts|js|json)$/.test(path)) continue;
                return nextResolve(pathToFileURL(path + suffix).href, context);
              }
            }
          }
          return nextResolve(specifier, context);
        },
        load(url, context, nextLoad) {
          if (url.startsWith(root) && url.endsWith('.ts')) return {
            format: 'module', shortCircuit: true,
            source: typescript.transpileModule(readFileSync(fileURLToPath(url), 'utf8'), {
              compilerOptions: {module: typescript.ModuleKind.ES2022, target: typescript.ScriptTarget.ES2022}
            }).outputText
          };
          return nextLoad(url, context);
        }
      });
      await import('@luma-contributors/lenses/computation-lenses/scenarios/index');
      const {getLensScenario} = await import('@luma-contributors/lenses/computation-lenses');
      const loaded = [];
      for (const scenarioId of ['gpu-network.flow-aggregation', 'gpu-network.network-reachability', 'gpu-network.network-isochrones']) {
        const scenario = await getLensScenario(scenarioId).load();
        if (typeof scenario.run !== 'function') throw new Error(scenarioId + ': missing run');
        loaded.push(scenarioId);
      }
      console.log(JSON.stringify(loaded));
    `
    );
    const output = execFileSync(process.execPath, ['--import', workspaceLoader, launcher], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 20_000
    });
    expect(JSON.parse(output)).toEqual([
      'gpu-network.flow-aggregation',
      'gpu-network.network-reachability',
      'gpu-network.network-isochrones'
    ]);
  } finally {
    rmSync(temporary, {recursive: true, force: true});
  }
});
