// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {REPOSITORY_ROOT} from './lens-directives';

/** Native Node checks use the same current workspace sources as Vitest, with real dependencies. */
export function writeNativeWorkspaceLoader(directory: string): string {
  const loader = join(directory, 'workspace-loader.mjs');
  writeFileSync(
    loader,
    `
import {registerHooks} from 'node:module';
import {readFileSync, existsSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import typescript from ${JSON.stringify(pathToFileURL(join(REPOSITORY_ROOT, 'node_modules/typescript/lib/typescript.js')).href)};
const repository = ${JSON.stringify(REPOSITORY_ROOT)};
const sourceRoot = pathToFileURL(resolve(repository, 'modules') + '/').href;
const {config, error} = typescript.readConfigFile(resolve(repository, 'tsconfig.json'), typescript.sys.readFile);
if (error) throw new Error(typescript.flattenDiagnosticMessageText(error.messageText, '\\n'));
const aliases = Object.entries(config.compilerOptions.paths).sort(([left], [right]) => right.length - left.length);
function resolveSource(path) {
  for (const suffix of ['', '.ts', '.js', '.cts', '/index.ts', '/index.js']) {
    const candidate = path + suffix;
    if (existsSync(candidate) && statSync(candidate).isFile()) return pathToFileURL(candidate).href;
  }
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    for (const [alias, targets] of aliases) {
      const wildcard = alias.endsWith('/*');
      if (wildcard ? specifier.startsWith(alias.slice(0, -1)) : specifier === alias) {
        const target = targets[0].replace('*', wildcard ? specifier.slice(alias.length - 1) : '');
        const found = resolveSource(resolve(repository, config.compilerOptions.baseUrl || '.', target));
        if (found) return {url: found, shortCircuit: true};
      }
    }
    if (context.parentURL?.startsWith(sourceRoot) && specifier.startsWith('.')) {
      const found = resolveSource(fileURLToPath(new URL(specifier, context.parentURL)));
      if (found) return {url: found, shortCircuit: true};
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.startsWith(sourceRoot) && /\\.(ts|cts)$/.test(url)) {
      const commonJS = url.endsWith('.cts');
      return {format: commonJS ? 'commonjs' : 'module', shortCircuit: true,
        source: typescript.transpileModule(readFileSync(fileURLToPath(url), 'utf8'), {
          compilerOptions: {module: commonJS ? typescript.ModuleKind.CommonJS : typescript.ModuleKind.ES2022, target: typescript.ScriptTarget.ES2022}
        }).outputText};
    }
    return nextLoad(url, context);
  }
});
`
  );
  return loader;
}
