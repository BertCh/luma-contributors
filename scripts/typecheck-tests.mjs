// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Typechecks tests in two tsc runs (see vitest.config.ts for the run exclusions):
// 1. everything except the specs that import examples/deck/kepler-compute;
// 2. those example-backed specs, with the example's tsconfig settings (jsx react-jsx, vite/client
//    types) and a larger heap, because one program over the specs and the example sources runs
//    out of memory at 8 GB under the root settings.
import {spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';

const rootDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const keplerComputeExampleSpecLists = JSON.parse(
  readFileSync(join(rootDirectory, 'test/kepler-compute-example-specs.json'), 'utf8')
);

// A test file is example-backed when it mentions examples/deck/kepler-compute or imports (by a
// relative path) another example-backed test file, such as a shared harness. The specs left out of
// runs are example-backed too.
function listTestFiles(directory) {
  const files = [];
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== 'dist') files.push(...listTestFiles(entryPath));
    } else if (/\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith('.d.ts')) {
      files.push(entryPath);
    }
  }
  return files;
}

function listExampleBackedTestFiles() {
  const testDirectories = [join(rootDirectory, 'test')];
  for (const moduleName of readdirSync(join(rootDirectory, 'modules'))) {
    testDirectories.push(join(rootDirectory, 'modules', moduleName, 'test'));
  }
  const sources = new Map();
  for (const directory of testDirectories) {
    if (!existsSync(directory)) continue;
    for (const file of listTestFiles(directory)) sources.set(file, readFileSync(file, 'utf8'));
  }
  const importPattern = /(?:from|import)\s*\(?\s*['"](\.{1,2}\/[^'"]+)['"]/g;
  const marked = new Set(
    [...sources].filter(([, text]) => text.includes('kepler-compute')).map(([file]) => file)
  );
  for (const specPath of Object.keys(keplerComputeExampleSpecLists.excludedFromRuns)) {
    marked.add(join(rootDirectory, specPath));
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const [file, text] of sources) {
      if (marked.has(file)) continue;
      for (const match of text.matchAll(importPattern)) {
        const base = join(dirname(file), match[1].replace(/\.js$/, ''));
        if ([base, `${base}.ts`, `${base}.tsx`, join(base, 'index.ts')].some((candidate) => marked.has(candidate))) {
          marked.add(file);
          changed = true;
          break;
        }
      }
    }
  }
  return [...marked].map((file) => relative(rootDirectory, file)).sort();
}

const exampleBackedFiles = listExampleBackedTestFiles();

const mainTsconfig = {
  extends: '../tsconfig.json',
  include: ['../modules/*/src', '../modules/*/test', '../test/**/*'],
  exclude: [
    '../modules/**/dist',
    '../node_modules',
    ...exampleBackedFiles.map((filePath) => `../${filePath}`),
    '../examples/deck/kepler-compute/**'
  ]
};

const excludedFromTypecheck = Object.keys(keplerComputeExampleSpecLists.excludedFromTypecheck ?? {});

// Packages the example declares but that ship no types.
const untypedModuleShims = `declare module 'd3-dsv';
`;

const exampleSpecsTsconfig = {
  extends: '../examples/deck/kepler-compute/tsconfig.json',
  compilerOptions: {noEmit: true, incremental: false},
  include: [
    'untyped-modules.d.ts',
    ...exampleBackedFiles
      .filter((filePath) => !excludedFromTypecheck.includes(filePath))
      .map((filePath) => `../${filePath}`)
  ],
  exclude: ['../node_modules', '../modules/**/dist']
};

mkdirSync(join(rootDirectory, '.typecheck'), {recursive: true});
writeFileSync(
  join(rootDirectory, '.typecheck/tsconfig.tests.json'),
  `${JSON.stringify(mainTsconfig, null, 2)}\n`
);
writeFileSync(join(rootDirectory, '.typecheck/untyped-modules.d.ts'), untypedModuleShims);
writeFileSync(
  join(rootDirectory, '.typecheck/tsconfig.example-specs.json'),
  `${JSON.stringify(exampleSpecsTsconfig, null, 2)}\n`
);

const tscBinary = join(rootDirectory, 'node_modules/.bin/tsc');
let exitCode = 0;
for (const [label, configuration, heapMegabytes] of [
  ['tests', '.typecheck/tsconfig.tests.json', 16384],
  ['example-backed specs', '.typecheck/tsconfig.example-specs.json', 16384]
]) {
  console.log(`typecheck-tests: ${label}`);
  const result = spawnSync(tscBinary, ['-p', configuration, '--noEmit'], {
    cwd: rootDirectory,
    stdio: 'inherit',
    env: {...process.env, NODE_OPTIONS: `--max-old-space-size=${heapMegabytes}`}
  });
  exitCode ||= result.status ?? 1;
}
process.exit(exitCode);
