// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Typechecks tests sequentially (see vitest.config.ts for the separate runtime exclusions):
// 1. ordinary test roots grouped by package/family, with a separate incremental cache per group;
// 2. those example-backed specs, with the example's tsconfig settings (jsx react-jsx, vite/client
//    types), because one program over the specs and the example sources runs
//    out of memory at 8 GB under the root settings;
// 3. website-backed specs (they import website/src), with the website's settings and its
//    virtual-module declarations, because website sources are not written for the root settings.
import {spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import typeScript from 'typescript';
import {resolveNodeHeapOptions} from './verification-policy.mjs';

const rootDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const commandArguments = process.argv.slice(2);
if (
  commandArguments.length > 1 ||
  commandArguments.some(argument => argument !== '--generate-only')
)
  throw new Error('Usage: node scripts/typecheck-tests.mjs [--generate-only]');
const rootConfiguration = typeScript.getParsedCommandLineOfConfigFile(
  join(rootDirectory, 'tsconfig.json'),
  {},
  {
    ...typeScript.sys,
    onUnRecoverableConfigFileDiagnostic(diagnostic) {
      throw new Error(typeScript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
    }
  }
);
// An empty root is valid for generation; the inventory determines each family's actual roots.
const configurationErrors = rootConfiguration?.errors.filter(error => error.code !== 18003) ?? [];
if (!rootConfiguration || configurationErrors.length)
  throw new Error(
    `Invalid root configuration: ${configurationErrors.map(diagnostic => typeScript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')).join('\n')}`
  );
// Tests execute through Vite, including those importing example source, glob and raw modules.
const testTypes = [...new Set([...(rootConfiguration.options.types ?? []), 'vite/client'])];
const keplerComputeExampleSpecLists = JSON.parse(
  readFileSync(join(rootDirectory, 'test/kepler-compute-example-specs.json'), 'utf8')
);

// A test file is example-backed when it mentions examples/deck/kepler-compute or imports (by a
// relative path) another example-backed test file, such as a shared harness. The specs left out of
// runs are example-backed too.
function listTestFiles(directory, includeJavaScriptAndDeclarations = false) {
  const files = [];
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== 'dist')
        files.push(...listTestFiles(entryPath, includeJavaScriptAndDeclarations));
    } else if (
      includeJavaScriptAndDeclarations
        ? /\.(?:[cm]?[jt]s|[jt]sx)$/.test(entry.name)
        : /\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith('.d.ts')
    ) {
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
        if (
          [base, `${base}.ts`, `${base}.tsx`, join(base, 'index.ts')].some(candidate =>
            marked.has(candidate)
          )
        ) {
          marked.add(file);
          changed = true;
          break;
        }
      }
    }
  }
  return [...marked].map(file => relative(rootDirectory, file)).sort();
}

const exampleBackedFiles = listExampleBackedTestFiles();

function listWebsiteBackedTestFiles() {
  const files = [];
  for (const moduleName of readdirSync(join(rootDirectory, 'modules'))) {
    const directory = join(rootDirectory, 'modules', moduleName, 'test');
    if (!existsSync(directory)) continue;
    for (const file of listTestFiles(directory)) {
      const path = relative(rootDirectory, file);
      if (
        !exampleBackedFiles.includes(path) &&
        readFileSync(file, 'utf8').includes('website/src/')
      ) {
        files.push(path);
      }
    }
  }
  return files.sort();
}

const websiteBackedFiles = listWebsiteBackedTestFiles();

const excludedFromTypecheck = Object.keys(
  keplerComputeExampleSpecLists.excludedFromTypecheck ?? {}
);

function createFamilyStages() {
  const groups = new Map();
  const testDirectories = [
    'test',
    ...readdirSync(join(rootDirectory, 'modules'))
      .sort()
      .map(moduleName => `modules/${moduleName}/test`)
  ];
  const excluded = new Set([
    ...exampleBackedFiles,
    ...websiteBackedFiles,
    ...excludedFromTypecheck
  ]);
  for (const directory of testDirectories) {
    if (!existsSync(join(rootDirectory, directory))) continue;
    for (const file of listTestFiles(join(rootDirectory, directory), true)) {
      const filePath = relative(rootDirectory, file);
      if (excluded.has(filePath)) continue;
      const components = relative(join(rootDirectory, directory), file).split('/');
      const family = components.slice(0, Math.min(2, components.length - 1)).join('-') || 'root';
      const packageName = directory === 'test' ? 'shared' : directory.split('/')[1];
      const key = `${packageName}-${family}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(filePath);
    }
  }
  const packageOrder = ['analysis', 'lenses', 'kepler', 'shared'];
  function sortKey(key) {
    const packageIndex = packageOrder.findIndex(name => key.startsWith(`${name}-`));
    return `${packageIndex < 0 ? 4 : packageIndex}-${key.startsWith('analysis-analysis-core') ? 0 : 1}-${key}`;
  }
  mkdirSync(join(rootDirectory, '.typecheck/test-families'), {recursive: true});
  const stages = [...groups]
    .sort(([left], [right]) => sortKey(left).localeCompare(sortKey(right)))
    .map(([key, files]) => {
      const configuration = `.typecheck/test-families/${key}.json`;
      writeFileSync(
        join(rootDirectory, configuration),
        `${JSON.stringify(
          {
            extends: '../../tsconfig.json',
            compilerOptions: {
              noEmit: true,
              incremental: true,
              types: testTypes,
              tsBuildInfoFile: `./${key}.tsbuildinfo`
            },
            files: files.sort().map(file => `../../${file}`),
            include: [],
            exclude: ['../../node_modules', '../../modules/**/dist']
          },
          null,
          2
        )}\n`
      );
      return {label: `tests: ${key}`, configuration, files};
    });
  writeFileSync(
    join(rootDirectory, '.typecheck/test-family-stages.json'),
    `${JSON.stringify(stages, null, 2)}\n`
  );
  return stages.map(({label, configuration}) => [label, configuration]);
}

// Packages the example declares but that ship no types.
const untypedModuleShims = `declare module 'd3-dsv';
`;

const exampleSpecsTsconfig = {
  extends: '../examples/deck/kepler-compute/tsconfig.json',
  compilerOptions: {
    noEmit: true,
    incremental: true,
    tsBuildInfoFile: './example-specs.tsbuildinfo'
  },
  include: [
    'untyped-modules.d.ts',
    ...exampleBackedFiles
      .filter(filePath => !excludedFromTypecheck.includes(filePath))
      .map(filePath => `../${filePath}`)
  ],
  exclude: ['../node_modules', '../modules/**/dist']
};

// The root settings supply the source path mapping; the website settings follow (bundler
// resolution, verbatim module syntax); two root strictness flags the website does not use are off.
const websiteSpecsTsconfig = {
  extends: ['../tsconfig.json', '../website/tsconfig.json'],
  compilerOptions: {
    noEmit: true,
    incremental: true,
    tsBuildInfoFile: './website-specs.tsbuildinfo',
    noPropertyAccessFromIndexSignature: false,
    noImplicitOverride: false
  },
  include: ['../website/src/env.d.ts', ...websiteBackedFiles.map(filePath => `../${filePath}`)],
  exclude: ['../node_modules', '../modules/**/dist']
};

mkdirSync(join(rootDirectory, '.typecheck'), {recursive: true});
writeFileSync(
  join(rootDirectory, '.typecheck/tsconfig.website-specs.json'),
  `${JSON.stringify(websiteSpecsTsconfig, null, 2)}\n`
);
writeFileSync(join(rootDirectory, '.typecheck/untyped-modules.d.ts'), untypedModuleShims);
writeFileSync(
  join(rootDirectory, '.typecheck/tsconfig.example-specs.json'),
  `${JSON.stringify(exampleSpecsTsconfig, null, 2)}\n`
);

const tscBinary = join(rootDirectory, 'node_modules/.bin/tsc');
// Preserve every caller option. Add one shared default only if no heap limit was supplied.
const callerOptions = process.env.NODE_OPTIONS ?? '';
const nodeOptions = resolveNodeHeapOptions(callerOptions);
const stages = [
  ...createFamilyStages(),
  ['example-backed specs', '.typecheck/tsconfig.example-specs.json'],
  ['website-backed specs', '.typecheck/tsconfig.website-specs.json']
];
if (commandArguments.includes('--generate-only')) {
  process.stdout.write(
    `typecheck-tests: generated ${stages.length} stage configurations; no compiler launched\n`
  );
  process.exit(0);
}
let exitCode = 0;
for (const [label, configuration] of stages) {
  process.stdout.write(`typecheck-tests: ${label}; NODE_OPTIONS=${nodeOptions}\n`);
  const result = spawnSync(tscBinary, ['-p', configuration, '--noEmit'], {
    cwd: rootDirectory,
    stdio: 'inherit',
    env: {...process.env, NODE_OPTIONS: nodeOptions}
  });
  exitCode ||= result.status ?? 1;
  if (exitCode) break;
}
process.exit(exitCode);
