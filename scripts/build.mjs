// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Build every package: ESM + declarations with tspc, then a CommonJS copy of each ESM file.
// CommonJS files are transformed one-to-one (not bundled) so every entry point shares one
// module graph and class identity, matching the luma.gl packaging convention.

import {execFileSync} from 'node:child_process';
import {readdir, readFile, rm, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {transform} from 'esbuild';

const MODULES = ['graph', 'lenses', 'kepler'];
const repositoryRoot = new URL('../', import.meta.url);

for (const moduleName of MODULES) {
  await rm(new URL(`modules/${moduleName}/dist/`, repositoryRoot), {recursive: true, force: true});
  await rm(new URL(`modules/${moduleName}/tsconfig.tsbuildinfo`, repositoryRoot), {force: true});
}

execFileSync('npx', ['tspc', '-b', 'tsconfig.build.json'], {
  cwd: fileURLToPath(repositoryRoot),
  stdio: 'inherit'
});

for (const moduleName of MODULES) {
  const distributionDirectory = new URL(`modules/${moduleName}/dist/`, repositoryRoot);
  const count = await writeCommonJSModules(distributionDirectory);
  process.stdout.write(`${moduleName}: wrote ${count} CommonJS modules\n`);
}

async function writeCommonJSModules(directory) {
  const files = (await readdir(directory, {recursive: true})).filter(file => file.endsWith('.js'));
  await Promise.all(
    files.map(async file => {
      const source = await readFile(new URL(file, directory), 'utf8');
      const rewrittenSource = source.replace(
        /((?:from\s+|import\s*(?:\(\s*)?)['"])(\.[^'"]+)\.js(['"])/g,
        '$1$2.cjs$3'
      );
      const {code} = await transform(rewrittenSource, {
        format: 'cjs',
        platform: 'node',
        target: 'node16',
        sourcefile: fileURLToPath(new URL(file, directory))
      });
      await writeFile(new URL(file.replace(/\.js$/, '.cjs'), directory), code);
    })
  );
  return files.length;
}
