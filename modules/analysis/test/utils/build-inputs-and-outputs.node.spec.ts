// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  renameSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync
} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';
import {CHECKED_TARBALL_BASE64} from '../analysis-core/checked-release-archive-fixture';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];

function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
  const timestamp = new Date('2024-01-02T03:04:05.000Z');
  utimesSync(join(directory, file), timestamp, timestamp);
}

function fixture() {
  const parent = join(
    ROOT,
    '.typecheck/r19/scratch/tooling-tests-release-w5-test-resolution-and-package-gates'
  );
  mkdirSync(parent, {recursive: true});
  const directory = mkdtempSync(join(parent, 'inputs-'));
  directories.push(directory);
  for (const script of [
    'build.mjs',
    'build-commonjs.mjs',
    'build-record.mjs',
    'typescript-project-inputs.mjs',
    'verification-policy.mjs',
    'verify-package-exports.mjs'
  ]) {
    mkdirSync(join(directory, 'scripts'), {recursive: true});
    copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
  }
  write(directory, 'package.json', '{"type":"module"}');
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ES2022',
        moduleResolution: 'bundler',
        types: [],
        lib: ['ES2022'],
        declaration: true
      }
    })
  );
  write(
    directory,
    'tsconfig.build.json',
    JSON.stringify({
      files: [],
      references: ['analysis', 'lenses', 'kepler'].map(name => ({path: `./modules/${name}`}))
    })
  );
  mkdirSync(join(directory, 'node_modules/.bin'), {recursive: true});
  for (const name of ['typescript', 'esbuild'])
    symlinkSync(join(ROOT, 'node_modules', name), join(directory, 'node_modules', name), 'dir');
  write(
    directory,
    'node_modules/.bin/tspc',
    `#!${process.execPath}\nrequire('node:child_process').execFileSync(process.execPath, [${JSON.stringify(join(ROOT, 'node_modules/typescript/lib/tsc.js'))}, ...process.argv.slice(2)], {stdio:'inherit'});`
  );
  chmodSync(join(directory, 'node_modules/.bin/tspc'), 0o755);
  for (const name of ['analysis', 'lenses', 'kepler']) {
    write(
      directory,
      `modules/${name}/src/index.ts`,
      'export function readValue(): number { return 11; }'
    );
    write(
      directory,
      `modules/${name}/tsconfig.json`,
      JSON.stringify({
        extends: '../../tsconfig.json',
        include: ['src/**/*', 'src/**/*.json'],
        compilerOptions: {composite: true, rootDir: 'src', outDir: 'dist'}
      })
    );
    write(
      directory,
      `modules/${name}/package.json`,
      JSON.stringify({
        name: `@luma-contributors/${name}`,
        type: 'module',
        exports: {
          '.': {types: './dist/index.d.ts', import: './dist/index.js', require: './dist/index.cjs'}
        }
      })
    );
    mkdirSync(join(directory, 'node_modules/@luma-contributors'), {recursive: true});
    symlinkSync(
      join(directory, 'modules', name),
      join(directory, 'node_modules/@luma-contributors', name),
      'dir'
    );
  }
  return directory;
}

function run(
  directory: string,
  script = 'build.mjs',
  environment: Record<string, string> = {},
  arguments_: string[] = []
) {
  const result = spawnSync(process.execPath, [join(directory, 'scripts', script), ...arguments_], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 20000,
    env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256', ...environment}
  });
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

function overwritePreservingMetadata(file: string, contents: string) {
  const previous = statSync(file);
  expect(Buffer.byteLength(contents)).toBe(previous.size);
  writeFileSync(file, contents);
  utimesSync(file, previous.atime, previous.mtime);
  expect(statSync(file).mtimeMs).toBe(previous.mtimeMs);
  expect(statSync(file).size).toBe(previous.size);
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function parseCompilerProject(configuration: string) {
  return typescript.getParsedCommandLineOfConfigFile(
    configuration,
    {},
    {
      ...typescript.sys,
      onUnRecoverableConfigFileDiagnostic(diagnostic) {
        throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
      }
    }
  )!;
}

function captureTree(directory: string): Array<[string, string | null]> {
  return readdirSync(directory, {recursive: true})
    .map(String)
    .sort()
    .map(file => [
      file,
      statSync(join(directory, file)).isDirectory()
        ? null
        : readFileSync(join(directory, file)).toString('base64')
    ]);
}

function assertRestoredTree(directory: string, previous: ReturnType<typeof captureTree>) {
  expect(captureTree(directory)).toEqual(previous);
}

it.each([
  'index.cjs',
  'index.d.ts',
  '.build-record.json'
])('review S1 restoration assertion rejects a leftover %s', file => {
  const directory = fixture();
  const distribution = join(directory, 'modules/analysis/dist');
  write(directory, 'modules/analysis/dist/previous.txt', 'old');
  const previous = captureTree(distribution);
  write(directory, `modules/analysis/dist/${file}`, 'promoted leftover');
  expect(() => assertRestoredTree(distribution, previous)).toThrow();
});

it.each([
  'node',
  'default',
  'nested node',
  'custom'
])('review B2 accepts native condition ordering (%s) for both loaders', ordering => {
  const directory = fixture();
  const file = join(directory, 'modules/analysis/package.json');
  const manifest = JSON.parse(readFileSync(file, 'utf8'));
  manifest.exports['.'] =
    ordering === 'node'
      ? {node: './dist/index.js', default: './dist/browser.js', require: './dist/index.cjs'}
      : ordering === 'default'
        ? {default: './dist/browser.js', import: './dist/index.js', require: './dist/index.cjs'}
        : ordering === 'nested node'
          ? {
              node: {import: './dist/index.js', require: './dist/index.cjs'},
              default: './dist/browser.js'
            }
          : {custom: './dist/index.js', default: './dist/browser.js', require: './dist/index.cjs'};
  writeFileSync(file, JSON.stringify(manifest));
  write(directory, 'modules/analysis/src/browser.ts', 'export const browser = 33;');
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  result = run(
    directory,
    'verify-package-exports.mjs',
    ordering === 'custom' ? {NODE_OPTIONS: '--max-old-space-size=256 --conditions=custom'} : {}
  );
  expect(result.status, result.output).toBe(0);
});

it.each([
  'implicit reference',
  'typed JSON require',
  'two-argument JS require',
  'extensionless reference',
  'synthetic helpers',
  'external JS depth zero',
  'external JS depth one'
])('review compiler-selected dependency oracle: %s', async scenario => {
  const {
    getTypeScriptProjectInputs,
    listTypeScriptDependencyInputs,
    getTypeScriptOutputPlan,
    captureTypeScriptProjectInputs
  } = await import('../../../../scripts/typescript-project-inputs.mjs');
  const directory = fixture();
  const sourceName = scenario === 'two-argument JS require' ? 'engine.js' : 'engine.ts';
  const sources: Record<string, string> = {
    'implicit reference': "import type {Value} from 'linked'; export type Result = Value;",
    'typed JSON require':
      "declare function require(name: string): number; export const value = require('./flags.json');",
    'two-argument JS require': `/** @param {string} name @param {number} fallback */
function require(name, fallback) { return fallback; } export const value = require('./optional.js', 3);`,
    'extensionless reference': '/// <reference path="./ambient" />\nexport const value: Unit = 11;',
    'synthetic helpers': 'export const value = 11;',
    'external JS depth zero': "import 'js-only'; export const value = 11;",
    'external JS depth one': "import 'js-only'; export const value = 11;"
  };
  write(directory, `modules/analysis/src/${sourceName}`, sources[scenario]!);
  write(directory, 'modules/analysis/src/flags.json', '{"enabled":true}');
  write(directory, 'modules/analysis/src/optional.js', 'export const value = 33;');
  write(directory, 'modules/analysis/src/ambient.d.ts', 'type Unit = 11;');
  write(directory, 'node_modules/linked/package.json', '{"name":"linked","types":"index.d.ts"}');
  write(
    directory,
    'node_modules/linked/index.d.ts',
    '/// <reference types="palette" />\nexport type Value = 11;'
  );
  write(
    directory,
    'node_modules/palette/package.json',
    JSON.stringify({
      name: 'palette',
      exports: {
        '.': {import: {types: './left.d.mts'}, require: {types: './right.d.cts'}}
      }
    })
  );
  write(directory, 'node_modules/palette/left.d.mts', 'export type Palette = 11;');
  write(directory, 'node_modules/palette/right.d.cts', 'export type Palette = 22;');
  write(directory, 'node_modules/tslib/package.json', '{"name":"tslib","types":"index.d.ts"}');
  write(
    directory,
    'node_modules/tslib/index.d.ts',
    'export declare function __assign(...sources: any[]): any;'
  );
  write(
    directory,
    'node_modules/js-only/package.json',
    '{"name":"js-only","type":"module","main":"index.js"}'
  );
  write(directory, 'node_modules/js-only/index.js', "import './more.js'; export const value = 11;");
  write(directory, 'node_modules/js-only/more.js', 'export const more = 22;');
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  writeFileSync(
    configuration,
    JSON.stringify({
      extends: '../../tsconfig.json',
      compilerOptions: {
        module: 'ESNext',
        moduleResolution: 'Bundler',
        allowJs: true,
        checkJs: true,
        noEmit: true,
        resolveJsonModule: true,
        importHelpers: scenario === 'synthetic helpers',
        ...(scenario === 'external JS depth one' ? {maxNodeModuleJsDepth: 1} : {})
      },
      files: [`src/${sourceName}`]
    })
  );
  // The oracle starts with TypeScript's own config expansion, before planner augmentation.
  const parsed = parseCompilerProject(configuration);
  expect(parsed.errors).toEqual([]);
  const program = typescript.createProgram(parsed.fileNames, parsed.options);
  expect(
    typescript
      .getPreEmitDiagnostics(program)
      .map(
        diagnostic =>
          `TS${diagnostic.code}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
      )
  ).toEqual([]);
  const selected = program
    .getSourceFiles()
    .map(source => source.fileName)
    .filter(file => file.startsWith(directory))
    .sort();
  const project = getTypeScriptProjectInputs(configuration);
  if (scenario === 'typed JSON require') {
    expect(project.files).toEqual(parsed.fileNames);
    expect(
      getTypeScriptOutputPlan(project, join(directory, 'modules/analysis')).flatMap(
        (entry: {compilerOutputs: string[]}) => entry.compilerOutputs
      )
    ).not.toContain('flags.json');
  }
  expect(listTypeScriptDependencyInputs(project)).toEqual(selected);
  if (scenario === 'implicit reference') {
    const before = captureTypeScriptProjectInputs(configuration).fingerprint;
    overwritePreservingMetadata(
      join(directory, 'node_modules/palette/right.d.cts'),
      'export type Palette = 33;'
    );
    expect(captureTypeScriptProjectInputs(configuration).fingerprint).toBe(before);
    overwritePreservingMetadata(
      join(directory, 'node_modules/palette/left.d.mts'),
      'export type Palette = 44;'
    );
    expect(captureTypeScriptProjectInputs(configuration).fingerprint).not.toBe(before);
  }
});

it.each([
  'import',
  'require',
  'implicit selected',
  'implicit unselected',
  'unhashed import'
])('W2-06 measurement tracks %s reference bytes with its hash-tracking host', async scenario => {
  const {default: typescript} = await import('typescript');
  const directory = fixture();
  copyFileSync(
    join(ROOT, 'scripts/measure-typecheck.mjs'),
    join(directory, 'scripts/measure-typecheck.mjs')
  );
  const implicit = scenario.startsWith('implicit');
  const mode = scenario === 'require' ? 'require' : 'import';
  const sourceName = implicit ? 'source.ts' : mode === 'import' ? 'source.cts' : 'source.mts';
  write(
    directory,
    `modules/analysis/src/${sourceName}`,
    implicit
      ? "import type {Value} from 'linked'; export type Result = Value;"
      : `/// <reference types="reference-only" resolution-mode="${mode}" />\nexport const value = 11;`
  );
  write(directory, 'node_modules/linked/package.json', '{"name":"linked","types":"index.d.ts"}');
  write(
    directory,
    'node_modules/linked/index.d.ts',
    '/// <reference types="reference-only" />\nexport type Value = 11;'
  );
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({
      extends: '../../tsconfig.json',
      compilerOptions: {
        module: implicit ? 'ESNext' : 'NodeNext',
        moduleResolution: implicit ? 'Bundler' : 'NodeNext',
        noEmit: true
      },
      files: [`src/${sourceName}`]
    })
  );
  write(
    directory,
    'node_modules/reference-only/package.json',
    JSON.stringify({
      name: 'reference-only',
      exports: {'.': {import: {types: './module.d.mts'}, require: {types: './classic.d.cts'}}}
    })
  );
  write(directory, 'node_modules/reference-only/module.d.mts', 'export type Value = 11;');
  write(directory, 'node_modules/reference-only/classic.d.cts', 'export type Value = 22;');
  const selected = join(
    directory,
    'node_modules/reference-only',
    mode === 'import' ? 'module.d.mts' : 'classic.d.cts'
  );
  const parsed = parseCompilerProject(join(directory, 'modules/analysis/tsconfig.json'));
  const program = typescript.createProgram(parsed.fileNames, parsed.options);
  expect(typescript.getPreEmitDiagnostics(program).map(diagnostic => diagnostic.code)).toEqual([]);
  expect(program.getSourceFiles().map(source => source.fileName)).toContain(selected);
  const edited =
    scenario === 'implicit unselected'
      ? join(directory, 'node_modules/reference-only/classic.d.cts')
      : selected;
  if (scenario === 'implicit unselected')
    expect(program.getSourceFiles().map(source => source.fileName)).not.toContain(edited);
  if (scenario === 'unhashed import') {
    const runner = join(directory, 'scripts/measure-typecheck.mjs');
    const source = readFileSync(runner, 'utf8');
    const expression = "createHash('sha256').update(bytes).digest('hex')";
    expect(source).toContain(expression);
    writeFileSync(runner, source.replace(expression, "'content hashing disabled by test'"));
  }
  // The transcript is a compiler double; the real program above establishes its valid inputs.
  write(
    directory,
    'compiler.mjs',
    `import {readFileSync, writeFileSync, statSync, utimesSync} from 'node:fs';
const file = ${JSON.stringify(edited)};
const previous = statSync(file);
const before = readFileSync(file, 'utf8');
const after = before.replace(/11|22/, before.includes('11') ? '22' : '11');
if (Buffer.byteLength(after) !== previous.size || after === before) throw new Error('invalid content-only edit');
writeFileSync(file, after); utimesSync(file, previous.atime, previous.mtime);
if (statSync(file).mtimeMs !== previous.mtimeMs || statSync(file).size !== previous.size) throw new Error('metadata changed');
console.log('Files: 7');`
  );
  // This tiny compiler double needs no GiB; isolate it from shared-machine disk pressure.
  write(
    directory,
    'fixture-disk.cjs',
    `const fs = require('node:fs');
fs.statfsSync = () => ({bavail: 64 * 1024 ** 3, bsize: 1}); require('node:module').syncBuiltinESMExports();`
  );
  const result = run(
    directory,
    'measure-typecheck.mjs',
    {
      NODE_OPTIONS: `--max-old-space-size=256 --require=${JSON.stringify(join(directory, 'fixture-disk.cjs'))}`
    },
    ['--project', 'modules/analysis/tsconfig.json', '--compiler', 'compiler.mjs', '--timing-only']
  );
  if (scenario === 'implicit unselected' || scenario === 'unhashed import') {
    expect(result.status, result.output).toBe(0);
    expect(result.output).toContain('Report:');
  } else {
    expect(result.status, result.output).toBe(1);
    expect(result.output).toContain('inputs changed during cold measurement');
  }
});

it.each([
  'dual-mode references',
  'bundler JSX',
  'typed require',
  'JavaScript require',
  'JSX script'
])('W2-06 shared inputs match compiler-selected files for %s', async scenario => {
  const {default: typescript} = await import('typescript');
  const {getTypeScriptProjectInputs, listTypeScriptDependencyInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  const sourceName =
    scenario === 'dual-mode references'
      ? 'source.mts'
      : scenario === 'bundler JSX'
        ? 'source.tsx'
        : scenario === 'JavaScript require'
          ? 'source.cjs'
          : 'source.ts';
  const sources: Record<string, string> = {
    'dual-mode references': `/// <reference types="reference-only" resolution-mode="require" />
import type {Value} from 'conditional';
export type Both = [Value, import('conditional', {with: {'resolution-mode': 'require'}}).Value];`,
    'bundler JSX': "export {element} from 'jsx-source';",
    'typed require':
      "declare function require(name: string): number; export const value = require('require-only');",
    'JavaScript require': "const value = require('require-only'); module.exports = value;",
    'JSX script': 'const value = 11;'
  };
  for (const name of ['conditional', 'transitive', 'reference-only', 'require-only', 'react']) {
    const entry = name === 'react' ? './jsx-runtime' : '.';
    write(
      directory,
      `node_modules/${name}/package.json`,
      JSON.stringify({
        name,
        exports: {[entry]: {import: {types: './module.d.mts'}, require: {types: './classic.d.cts'}}}
      })
    );
    for (const [file, value] of [
      ['module.d.mts', '11'],
      ['classic.d.cts', '22']
    ])
      write(
        directory,
        `node_modules/${name}/${file}`,
        name === 'conditional'
          ? "export type {Value} from 'transitive';"
          : name === 'react'
            ? 'export namespace JSX { interface IntrinsicElements { div: {}; } }'
            : `export type Value = ${value};`
      );
  }
  // No package type: bundler's synthetic runtime still uses the import condition.
  write(directory, 'modules/analysis/package.json', '{"name":"@luma-contributors/analysis"}');
  write(
    directory,
    'node_modules/jsx-source/package.json',
    '{"name":"jsx-source","types":"view.tsx"}'
  );
  write(directory, 'node_modules/jsx-source/view.tsx', 'export const element = <div/>;');
  write(directory, `modules/analysis/src/${sourceName}`, sources[scenario]!);
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  writeFileSync(
    configuration,
    JSON.stringify({
      extends: '../../tsconfig.json',
      compilerOptions: {
        noEmit: true,
        allowJs: true,
        module: scenario === 'bundler JSX' || scenario === 'JSX script' ? 'ESNext' : 'NodeNext',
        moduleResolution:
          scenario === 'bundler JSX' || scenario === 'JSX script' ? 'Bundler' : 'NodeNext',
        ...(scenario === 'bundler JSX' || scenario === 'JSX script' ? {jsx: 'react-jsx'} : {})
      },
      files: [`src/${sourceName}`]
    })
  );
  const project = getTypeScriptProjectInputs(configuration);
  const parsed = parseCompilerProject(configuration);
  const program = typescript.createProgram(parsed.fileNames, parsed.options);
  expect(
    typescript
      .getPreEmitDiagnostics(program)
      .map(
        diagnostic =>
          `TS${diagnostic.code}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
      )
  ).toEqual([]);
  // Independent oracle: the installed compiler's selected files, not another lexical walk.
  const selected = program
    .getSourceFiles()
    .map(source => source.fileName)
    .filter(file => file.startsWith(directory))
    .sort();
  expect(listTypeScriptDependencyInputs(project)).toEqual(selected);
});

it.each([
  'ts',
  'mts',
  'cts',
  'tsx',
  'd.ts',
  'd.mts',
  'd.cts',
  'js',
  'mjs',
  'cjs',
  'jsx'
])('W2-06 follows compiler JSDoc dependency selection and content freshness (%s)', async extension => {
  const {
    getTypeScriptProjectInputs,
    listTypeScriptDependencyInputs,
    captureTypeScriptProjectInputs
  } = await import('../../../../scripts/typescript-project-inputs.mjs');
  const directory = fixture();
  const sourceName = `source.${extension}`;
  write(
    directory,
    `modules/analysis/src/${sourceName}`,
    `/** @type {import("./optional.mjs").Value} */
      export ${extension.startsWith('d.') ? 'declare const value: number;' : 'const value = 11;'}
      /** @import {Imported} from "./documented.mjs" */
      `
  );
  write(directory, 'modules/analysis/src/optional.d.mts', 'export type Value = 11;');
  write(directory, 'modules/analysis/src/documented.d.mts', 'export type Imported = 33;');
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  writeFileSync(
    configuration,
    JSON.stringify({
      extends: '../../tsconfig.json',
      compilerOptions: {
        noEmit: true,
        allowJs: true,
        checkJs: true,
        module: 'NodeNext',
        moduleResolution: 'NodeNext',
        jsx: 'preserve'
      },
      files: [`src/${sourceName}`]
    })
  );
  const parsed = parseCompilerProject(configuration);
  const program = typescript.createProgram(parsed.fileNames, parsed.options);
  expect(
    typescript
      .getPreEmitDiagnostics(program)
      .map(
        diagnostic =>
          `TS${diagnostic.code}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
      )
  ).toEqual([]);
  const selected = program
    .getSourceFiles()
    .map(source => source.fileName)
    .filter(file => file.startsWith(directory))
    .sort();
  const project = getTypeScriptProjectInputs(configuration);
  expect(listTypeScriptDependencyInputs(project)).toEqual(selected);
  const before = captureTypeScriptProjectInputs(configuration).fingerprint;
  // Literal type changes with identical timestamp/size isolate content hashing. TypeScript
  // selects JSDoc dependencies only for JavaScript, never TypeScript or declaration files.
  for (const [file, original, replacement] of [
    ['optional.d.mts', '11', '22'],
    ['documented.d.mts', '33', '44']
  ]) {
    const dependency = join(directory, 'modules/analysis/src', file!);
    overwritePreservingMetadata(
      dependency,
      readFileSync(dependency, 'utf8').replace(original!, replacement!)
    );
    const after = captureTypeScriptProjectInputs(configuration).fingerprint;
    if (['js', 'mjs', 'cjs', 'jsx'].includes(extension)) expect(after).not.toBe(before);
    else expect(after).toBe(before);
    overwritePreservingMetadata(
      dependency,
      readFileSync(dependency, 'utf8').replace(replacement!, original!)
    );
  }
});

it.each([
  'external declaration',
  'live resolution failure',
  'misdirected ESM link',
  'misdirected CommonJS link'
])('W2-02 verifies installed outputs through the promotion entry point (%s)', state => {
  const directory = fixture();
  copyFileSync(
    join(ROOT, 'scripts/build-release.mjs'),
    join(directory, 'scripts/build-release.mjs')
  );
  copyFileSync(
    join(ROOT, 'scripts/packed-file-inventory.mjs'),
    join(directory, 'scripts/packed-file-inventory.mjs')
  );
  // Packing is independently covered by the producer tests; this fixture exercises real
  // compilation, records, export loading and the release promotion transaction.
  write(
    directory,
    'scripts/check-packed-consumer.mjs',
    `
      import {writeFileSync} from 'node:fs'; import {join} from 'node:path';
      const file = join(process.env.TMPDIR, 'checked.tgz');
      writeFileSync(file, Buffer.from(${JSON.stringify(CHECKED_TARBALL_BASE64)}, 'base64'));
      console.log('checked tarball: ' + file);
    `
  );
  const metadata = '{"name":"linked","version":"1.0.0","types":"index.d.ts","main":"index.cjs"}';
  const declaration = 'export declare function getMarker(): 11;';
  write(directory, 'node_modules/linked/package.json', metadata);
  write(directory, 'node_modules/linked/index.d.ts', declaration);
  write(directory, 'node_modules/linked/index.cjs', 'exports.getMarker = () => 11;');
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import {getMarker} from 'linked'; export const marker = getMarker();"
  );
  if (state.startsWith('misdirected')) {
    // A healthy second checkout can mask the installed package under native bare resolution.
    // For the CommonJS case only, ESM follows a symlink back to the correct installed file.
    mkdirSync(join(directory, 'decoy'), {recursive: true});
    cpSync(
      join(directory, 'modules/analysis/package.json'),
      join(directory, 'decoy/package.json'),
      {
        recursive: true
      }
    );
    write(directory, 'decoy/dist/index.cjs', 'exports.marker = 11;');
    if (state === 'misdirected CommonJS link')
      symlinkSync(
        join(directory, 'modules/analysis/dist/index.js'),
        join(directory, 'decoy/dist/index.js')
      );
    else write(directory, 'decoy/dist/index.js', 'export const marker = 11;');
    rmSync(join(directory, 'node_modules/@luma-contributors/analysis'));
    symlinkSync(
      join(directory, 'decoy'),
      join(directory, 'node_modules/@luma-contributors/analysis'),
      'dir'
    );
  }
  const liveLoadSentinel = join(directory, 'live-load-sentinel');
  if (state === 'misdirected CommonJS link') {
    write(directory, 'modules/analysis/node_modules/linked/package.json', metadata);
    write(directory, 'modules/analysis/node_modules/linked/index.d.ts', declaration);
    write(
      directory,
      'modules/analysis/node_modules/linked/index.cjs',
      `require('node:fs').writeFileSync(${JSON.stringify(liveLoadSentinel)}, 'ESM initialized'); exports.getMarker = () => 11;`
    );
  }
  if (state === 'live resolution failure') {
    write(directory, 'modules/analysis/node_modules/linked/package.json', metadata);
    write(directory, 'modules/analysis/node_modules/linked/index.d.ts', declaration);
    write(
      directory,
      'modules/analysis/node_modules/linked/index.cjs',
      `
        exports.getMarker = () => 11;
        const fs = require('node:fs');
        const installed = ${JSON.stringify(['analysis', 'lenses', 'kepler'].map(name => join(directory, 'modules', name, 'dist/index.js')))}.every(file => fs.existsSync(file));
        throw new Error('live-only resolution sentinel; all installed: ' + installed);
      `
    );
  }
  const previousLatest = '{"previous":"release"}\n';
  write(directory, '.typecheck/release-build-latest.json', previousLatest);
  for (const name of ['analysis', 'lenses', 'kepler']) {
    write(directory, `modules/${name}/dist/previous.txt`, `previous ${name} output`);
    write(directory, `modules/${name}/tsconfig.tsbuildinfo`, `previous ${name} cache`);
  }
  const previousTrees = Object.fromEntries(
    ['analysis', 'lenses', 'kepler'].map(name => [
      name,
      captureTree(join(directory, 'modules', name, 'dist'))
    ])
  );
  const result = run(directory, 'build-release.mjs', {}, ['--promote']);
  if (state === 'external declaration') {
    expect(result.status, result.output).toBe(0);
    // One verifier run in the snapshot and one against all three installed packages.
    expect(result.output.match(/verified 3 entry points/g)).toHaveLength(2);
    expect(readFileSync(join(directory, 'modules/analysis/dist/index.d.ts'), 'utf8')).toContain(
      'marker: 11'
    );
    const verification = run(directory, 'verify-package-exports.mjs');
    expect(verification.status, verification.output).toBe(0);
    expect(
      JSON.parse(readFileSync(join(directory, '.typecheck/release-build-latest.json'), 'utf8'))
        .status
    ).toBe('promoted');
  } else {
    expect(result.status, result.output).toBe(1);
    expect(result.output).toContain(
      state.startsWith('misdirected')
        ? `${state === 'misdirected ESM link' ? 'ESM' : 'CommonJS'} target mismatch`
        : 'live-only resolution sentinel; all installed: true'
    );
    if (state === 'misdirected CommonJS link') expect(existsSync(liveLoadSentinel)).toBe(false);
    expect(result.output).toContain('rolled-back');
    expect(readFileSync(join(directory, '.typecheck/release-build-latest.json'), 'utf8')).toBe(
      previousLatest
    );
    for (const name of ['analysis', 'lenses', 'kepler']) {
      assertRestoredTree(join(directory, 'modules', name, 'dist'), previousTrees[name]!);
      expect(readFileSync(join(directory, `modules/${name}/dist/previous.txt`), 'utf8')).toBe(
        `previous ${name} output`
      );
      expect(readFileSync(join(directory, `modules/${name}/tsconfig.tsbuildinfo`), 'utf8')).toBe(
        `previous ${name} cache`
      );
      expect(existsSync(join(directory, `modules/${name}/dist/index.js`))).toBe(false);
    }
    const stage = result.output.match(/failed release stage retained for diagnosis: (.+)/)?.[1];
    expect(stage).toBeDefined();
    expect(existsSync(join(stage!, 'build.json'))).toBe(false);
    expect(JSON.parse(readFileSync(join(stage!, 'promotion.json'), 'utf8')).status).toBe(
      'rolled-back'
    );
  }
}, 30000);

it('TOOL-108 rejects a changed linked declaration and rebuilds inferred output with restored metadata', () => {
  const directory = fixture();
  const dependency = join(directory, 'linked');
  write(
    directory,
    'linked/package.json',
    JSON.stringify({
      name: 'linked',
      version: '1.0.0',
      type: 'module',
      exports: {'.': {types: './index.d.ts', import: './index.js', require: './index.cjs'}}
    })
  );
  write(directory, 'linked/index.d.ts', 'export declare function getMarker(): 11;');
  write(directory, 'linked/index.js', 'export function getMarker() { return 11; }');
  write(directory, 'linked/index.cjs', 'exports.getMarker = () => 11;');
  symlinkSync(dependency, join(directory, 'node_modules/linked'), 'dir');
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import {getMarker} from 'linked'; export const marker = getMarker();"
  );
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readFileSync(join(directory, 'modules/analysis/dist/index.d.ts'), 'utf8')).toContain(
    'marker: 11'
  );
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('analysis: wrote 0 CommonJS modules');
  overwritePreservingMetadata(
    join(dependency, 'index.d.ts'),
    'export declare function getMarker(): 22;'
  );
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.output).toContain('stale build record');
  expect(result.status).toBe(1);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  // Independent expectation: the declaration returns literal 22, so inference must emit 22.
  expect(readFileSync(join(directory, 'modules/analysis/dist/index.d.ts'), 'utf8')).toContain(
    'marker: 22'
  );
  expect(existsSync(join(directory, 'modules/analysis/dist/linked/index.d.ts'))).toBe(false);
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
}, 30000);

it('build tools reject a restored-mtime transform edit, regenerate CommonJS and retain the warm path', () => {
  const directory = fixture();
  const transformer = join(directory, 'scripts/build-commonjs.mjs');
  const transform = (literal: number) =>
    `export async function transformCommonJSModule(source) { return {code: source.replace('export function readValue()', 'exports.readValue = function()').replace('return 11', 'return ${literal}')}; }`;
  write(directory, 'scripts/build-commonjs.mjs', transform(11));
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  const output = join(directory, 'modules/analysis/dist/index.cjs');
  const previousTime = statSync(output).mtimeMs;
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('analysis: wrote 0 CommonJS modules');
  expect(statSync(output).mtimeMs).toBe(previousTime);
  overwritePreservingMetadata(transformer, transform(22));
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.output).toContain('stale build record');
  expect(result.status).toBe(1);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  // Hand-written transform changes the returned literal to 22, independent of source inference.
  expect(readFileSync(output, 'utf8')).toContain('return 22');
  expect(result.output).toContain('analysis: wrote 1 CommonJS modules');
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('analysis: wrote 0 CommonJS modules');
}, 30000);

it('output planning follows configured roots, native modules, preserved JSX, JSON and maps', async () => {
  const {getTypeScriptOutputPlan} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  const source = (file: string) => join(directory, 'modules/analysis/src', file);
  const project = {
    files: ['plain.ts', 'native.mts', 'native.cts', 'view.tsx', 'data.json', 'ambient.d.ts'].map(
      source
    ),
    options: {
      rootDir: source(''),
      outDir: join(directory, 'modules/analysis/dist'),
      configFilePath: join(directory, 'modules/analysis/tsconfig.json'),
      declaration: true,
      jsx: 1,
      resolveJsonModule: true,
      sourceMap: true,
      declarationMap: true
    }
  };
  const plan = getTypeScriptOutputPlan(project, join(directory, 'modules/analysis'));
  // Literal expectations follow TypeScript's documented suffixes, not a second planner run.
  expect(
    plan.map((entry: {compilerOutputs: string[]; transformedOutputs: string[]}) => ({
      compiler: entry.compilerOutputs,
      transformed: entry.transformedOutputs
    }))
  ).toEqual([
    {
      compiler: ['plain.js', 'plain.js.map', 'plain.d.ts', 'plain.d.ts.map'],
      transformed: ['plain.cjs', 'plain.cjs.map']
    },
    {
      compiler: ['native.mjs', 'native.mjs.map', 'native.d.mts', 'native.d.mts.map'],
      transformed: []
    },
    {
      compiler: ['native.cjs', 'native.cjs.map', 'native.d.cts', 'native.d.cts.map'],
      transformed: []
    },
    {compiler: ['view.jsx', 'view.jsx.map', 'view.d.ts', 'view.d.ts.map'], transformed: []},
    {compiler: ['data.json'], transformed: []}
  ]);
});

it('output planning repairs native compiler files and maps, and prunes removed sources', () => {
  const directory = fixture();
  write(
    directory,
    'tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        target: 'ES2022',
        module: 'ESNext',
        moduleResolution: 'bundler',
        types: [],
        lib: ['ES2022'],
        declaration: true,
        jsx: 'preserve',
        resolveJsonModule: true,
        sourceMap: true,
        inlineSources: true,
        removeComments: true,
        declarationMap: true
      }
    })
  );
  write(directory, 'modules/analysis/src/native.mts', 'export const nativeModule = 7;');
  write(directory, 'modules/analysis/src/native.cts', 'export const nativeCommonJS = 8;');
  write(directory, 'modules/analysis/src/view.tsx', 'export const view = 9;');
  write(directory, 'modules/analysis/src/data.json', '{"value":10}');
  const source =
    "import data from './data.json' with {type: 'json'}; export const value = data.value;";
  write(directory, 'modules/analysis/src/index.ts', source);
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  const distribution = join(directory, 'modules/analysis/dist');
  for (const file of [
    'native.mjs',
    'native.d.mts',
    'native.cjs',
    'native.d.cts',
    'view.jsx',
    'view.d.ts',
    'data.json',
    'index.js.map',
    'index.cjs.map',
    'index.d.ts.map'
  ])
    expect(existsSync(join(distribution, file)), file).toBe(true);
  const commonJSMap = join(distribution, 'index.cjs.map');
  expect(JSON.parse(readFileSync(commonJSMap, 'utf8')).sourcesContent).toEqual([source]);
  rmSync(join(distribution, 'native.cjs'));
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status).toBe(1);
  expect(result.output).toContain('missing build output dist/native.cjs');
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(existsSync(join(distribution, 'native.cjs'))).toBe(true);
  rmSync(join(distribution, 'index.d.ts.map'));
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status).toBe(1);
  expect(result.output).toContain('missing build output dist/index.d.ts.map');
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(existsSync(join(distribution, 'index.d.ts.map'))).toBe(true);
  rmSync(commonJSMap);
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status).toBe(1);
  expect(result.output).toContain('missing build output dist/index.cjs.map');
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(JSON.parse(readFileSync(commonJSMap, 'utf8')).sourcesContent).toEqual([source]);
  const esm = join(distribution, 'index.js');
  const originalJavaScript = readFileSync(esm, 'utf8');
  const updatedSource = `${source}\n// A source-only change\n`;
  write(directory, 'modules/analysis/src/index.ts', updatedSource);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readFileSync(esm, 'utf8')).toBe(originalJavaScript);
  expect(JSON.parse(readFileSync(commonJSMap, 'utf8')).sourcesContent).toEqual([updatedSource]);
  rmSync(join(directory, 'modules/analysis/src/native.cts'));
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  for (const file of ['native.cjs', 'native.cjs.map', 'native.d.cts', 'native.d.cts.map'])
    expect(existsSync(join(distribution, file)), file).toBe(false);
  expect(existsSync(join(distribution, 'native.mjs'))).toBe(true);
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
}, 30000);

it('dependency traversal includes inherited config, JSON, transitive declarations and JS/JSDoc without emitting dependencies', async () => {
  const {getTypeScriptProjectInputs, listTypeScriptDependencyInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  write(
    directory,
    'base.json',
    JSON.stringify({
      compilerOptions: {
        module: 'ESNext',
        moduleResolution: 'bundler',
        types: [],
        resolveJsonModule: true,
        allowJs: true,
        lib: ['ES2022']
      }
    })
  );
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({
      extends: '../../base.json',
      files: ['src/index.ts', 'src/loader.js'],
      compilerOptions: {rootDir: 'src', outDir: 'dist'}
    })
  );
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import data from './data.json'; import type {Value} from 'linked'; export const value: Value = data.value;"
  );
  write(directory, 'modules/analysis/src/data.json', '{"value":11}');
  write(
    directory,
    'modules/analysis/src/loader.js',
    "/** @type {import('jsdoc-only').Value} */\nexport const value = 11;"
  );
  write(directory, 'node_modules/linked/package.json', '{"types":"index.d.ts"}');
  write(directory, 'node_modules/linked/index.d.ts', "export type {Value} from './nested.js';");
  write(directory, 'node_modules/linked/nested.d.ts', 'export type Value = number;');
  write(directory, 'node_modules/jsdoc-only/package.json', '{"types":"index.d.ts"}');
  write(directory, 'node_modules/jsdoc-only/index.d.ts', 'export type Value = number;');
  const project = getTypeScriptProjectInputs(join(directory, 'modules/analysis/tsconfig.json'));
  expect(project.configFiles).toContain(join(directory, 'base.json'));
  const dependencies = listTypeScriptDependencyInputs(project);
  for (const file of [
    'modules/analysis/src/data.json',
    'node_modules/linked/index.d.ts',
    'node_modules/linked/nested.d.ts',
    'node_modules/jsdoc-only/index.d.ts'
  ])
    expect(dependencies, file).toContain(join(directory, file));
  expect(project.files).not.toContain(join(directory, 'node_modules/linked/nested.d.ts'));
  const {listTypeScriptFreshnessInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const {default: typeScript} = await import('typescript');
  const trackedReads = new Set<string>();
  const broader = listTypeScriptFreshnessInputs(project, {
    includeLibraries: true,
    host: {
      ...typeScript.sys,
      readFile(file: string) {
        trackedReads.add(resolve(file));
        return typeScript.sys.readFile(file);
      }
    }
  });
  for (const file of dependencies.filter(file => !project.files.includes(file)))
    expect(broader, file).toContain(file);
  expect(trackedReads).toContain(join(directory, 'node_modules/linked/package.json'));
  expect(broader.some(file => file.endsWith('lib.es2022.d.ts'))).toBe(true);
});

it('compiler and configured plugin content are tracked independently of versions and mtimes', async () => {
  const {captureBuildInputs} = await import('../../../../scripts/build-record.mjs');
  const {getTypeScriptProjectInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  write(
    directory,
    'node_modules/fixture-plugin/package.json',
    '{"name":"fixture-plugin","version":"1.0.0","main":"index.cjs"}'
  );
  write(directory, 'node_modules/fixture-plugin/index.cjs', 'module.exports = () => 11;');
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  const config = JSON.parse(readFileSync(configuration, 'utf8'));
  config.compilerOptions.plugins = [{transform: 'fixture-plugin'}];
  writeFileSync(configuration, JSON.stringify(config));
  const project = getTypeScriptProjectInputs(configuration);
  const moduleDirectory = join(directory, 'modules/analysis');
  const compiler = join(directory, 'node_modules/.bin/tspc');
  const before = captureBuildInputs(moduleDirectory, project, compiler);
  overwritePreservingMetadata(
    join(directory, 'node_modules/fixture-plugin/index.cjs'),
    'module.exports = () => 22;'
  );
  const afterPlugin = captureBuildInputs(moduleDirectory, project, compiler);
  expect(afterPlugin).not.toEqual(before);
  expect(afterPlugin['@tool:compiler-entry']).toBe(before['@tool:compiler-entry']);
  const executable = readFileSync(compiler, 'utf8');
  overwritePreservingMetadata(compiler, executable.replace("stdio:'inherit'", "stdio: 'ignore'"));
  const afterCompiler = captureBuildInputs(moduleDirectory, project, compiler);
  expect(afterCompiler['@tool:compiler-entry']).not.toBe(afterPlugin['@tool:compiler-entry']);
});

it('release-style snapshots verify with the existing named-script copy list', () => {
  const directory = fixture();
  write(
    directory,
    'node_modules/linked/package.json',
    '{"name":"linked","types":"index.d.ts","main":"index.cjs"}'
  );
  write(directory, 'node_modules/linked/index.d.ts', 'export declare function getMarker(): 11;');
  write(directory, 'node_modules/linked/index.cjs', 'exports.getMarker = () => 11;');
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import {getMarker} from 'linked'; export const marker = getMarker();"
  );
  write(
    directory,
    'modules/lenses/src/index.ts',
    "import {marker} from '@luma-contributors/analysis'; export const inferred = marker;"
  );
  const lensConfiguration = JSON.parse(
    readFileSync(join(directory, 'modules/lenses/tsconfig.json'), 'utf8')
  );
  lensConfiguration.references = [{path: '../analysis'}];
  write(directory, 'modules/lenses/tsconfig.json', JSON.stringify(lensConfiguration));
  const workspace = join(directory, 'stage/workspace');
  for (const file of ['package.json', 'tsconfig.json', 'tsconfig.build.json']) {
    mkdirSync(workspace, {recursive: true});
    copyFileSync(join(directory, file), join(workspace, file));
  }
  for (const name of ['analysis', 'lenses', 'kepler']) {
    for (const file of ['package.json', 'tsconfig.json', 'src']) {
      mkdirSync(join(workspace, 'modules', name), {recursive: true});
      cpSync(join(directory, 'modules', name, file), join(workspace, 'modules', name, file), {
        recursive: true
      });
    }
    mkdirSync(join(workspace, 'node_modules/@luma-contributors'), {recursive: true});
    symlinkSync(
      join(workspace, 'modules', name),
      join(workspace, 'node_modules/@luma-contributors', name),
      'dir'
    );
  }
  mkdirSync(join(workspace, 'scripts'), {recursive: true});
  for (const script of [
    'verify-package-exports.mjs',
    'typescript-project-inputs.mjs',
    'build-record.mjs'
  ])
    copyFileSync(join(directory, 'scripts', script), join(workspace, 'scripts', script));
  write(
    directory,
    'scripts/stage-build.mjs',
    `import {buildPackages} from './build.mjs';
await buildPackages(${JSON.stringify(workspace)}, ${JSON.stringify(join(directory, 'node_modules/.bin/tspc'))});`
  );
  let result = run(directory, 'stage-build.mjs');
  expect(result.status, result.output).toBe(0);
  result = run(workspace, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  // Promotion moves only dist: the same external declaration must have the same identity live.
  for (const name of ['analysis', 'lenses', 'kepler'])
    renameSync(join(workspace, 'modules', name, 'dist'), join(directory, 'modules', name, 'dist'));
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  const record = readFileSync(join(directory, 'modules/analysis/dist/.build-record.json'), 'utf8');
  expect(record).not.toContain(ROOT);
  expect(record).not.toContain(directory);
}, 30000);

it('B1 rebuilds inferred output after an ordinary newer linked declaration edit', () => {
  const directory = fixture();
  write(
    directory,
    'node_modules/linked/package.json',
    '{"name":"linked","version":"1.0.0","types":"index.d.ts","main":"index.cjs"}'
  );
  write(directory, 'node_modules/linked/index.d.ts', 'export declare function getMarker(): 11;');
  write(directory, 'node_modules/linked/index.cjs', 'exports.getMarker = () => 11;');
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import {getMarker} from 'linked'; export const marker = getMarker();"
  );
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  const dependency = join(directory, 'node_modules/linked/index.d.ts');
  writeFileSync(dependency, 'export declare function getMarker(): 22;');
  const newer = new Date(
    statSync(join(directory, 'modules/analysis/tsconfig.tsbuildinfo')).mtimeMs + 5000
  );
  utimesSync(dependency, newer, newer);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readFileSync(join(directory, 'modules/analysis/dist/index.d.ts'), 'utf8')).toContain(
    'marker: 22'
  );
}, 30000);

it('B3 verifies a no-config package using the supported fallback planner', () => {
  const directory = fixture();
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  for (const name of ['analysis', 'lenses', 'kepler']) {
    rmSync(join(directory, 'modules', name, 'tsconfig.json'));
    rmSync(join(directory, 'modules', name, 'dist/.build-record.json'));
    for (const file of ['index.js', 'index.cjs', 'index.d.ts']) {
      const output = join(directory, 'modules', name, 'dist', file);
      const current = new Date(Date.now() + 5000);
      utimesSync(output, current, current);
    }
  }
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('verified 3 entry points');
}, 30000);

it('W3-01 preserves wildcard, registration-only and resolution-only asset entries', () => {
  const directory = fixture();
  for (const file of ['flat', 'nested/deep'])
    write(directory, `modules/analysis/src/features/${file}.ts`, 'export const value = 11;');
  write(directory, 'modules/lenses/src/computation-lenses/scenarios/index.ts', 'export {};');
  write(directory, 'modules/analysis/dist/asset.txt', 'an asset, not JavaScript');
  for (const name of ['analysis', 'lenses']) {
    const manifest = JSON.parse(
      readFileSync(join(directory, 'modules', name, 'package.json'), 'utf8')
    );
    if (name === 'analysis') {
      manifest.exports['./features/*'] = {
        types: './dist/features/*.d.ts',
        import: './dist/features/*.js',
        require: './dist/features/*.cjs'
      };
      manifest.exports['./asset'] = './dist/asset.txt';
    } else {
      manifest.exports['./computation-lenses/scenarios/index'] = {
        import: './dist/computation-lenses/scenarios/index.js',
        require: './dist/computation-lenses/scenarios/index.cjs'
      };
    }
    write(directory, `modules/${name}/package.json`, JSON.stringify(manifest));
  }
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  // Three roots, two expanded wildcard entries, one registration and one opaque asset.
  expect(result.output).toContain('verified 7 entry points');
}, 30000);

it('B5 tracks ESM, require, import-type and type-reference conditional declaration branches', async () => {
  const {
    getTypeScriptProjectInputs,
    listTypeScriptDependencyInputs,
    captureTypeScriptProjectInputs
  } = await import('../../../../scripts/typescript-project-inputs.mjs');
  const directory = fixture();
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({
      compilerOptions: {
        module: 'NodeNext',
        moduleResolution: 'NodeNext',
        types: []
      },
      files: ['src/source.mts']
    })
  );
  write(
    directory,
    'modules/analysis/src/source.mts',
    `/// <reference types="reference-only" resolution-mode="import" />
import type {Value} from 'conditional';
export type Both = [Value, import('conditional', {with: {'resolution-mode': 'require'}}).Value];`
  );
  write(
    directory,
    'node_modules/conditional/package.json',
    JSON.stringify({
      name: 'conditional',
      exports: {
        '.': {import: {types: './module.d.mts'}, require: {types: './classic.d.cts'}}
      }
    })
  );
  write(directory, 'node_modules/conditional/module.d.mts', 'export type Value = 11;');
  write(directory, 'node_modules/conditional/classic.d.cts', 'export type Value = 22;');
  write(
    directory,
    'node_modules/reference-only/package.json',
    JSON.stringify({
      name: 'reference-only',
      exports: {
        '.': {import: {types: './module.d.mts'}, require: {types: './classic.d.cts'}}
      }
    })
  );
  write(directory, 'node_modules/reference-only/module.d.mts', 'export type Reference = 11;');
  write(directory, 'node_modules/reference-only/classic.d.cts', 'export type Reference = 22;');

  const project = getTypeScriptProjectInputs(join(directory, 'modules/analysis/tsconfig.json'));
  const dependencies = listTypeScriptDependencyInputs(project);
  expect(dependencies).toContain(join(directory, 'node_modules/conditional/module.d.mts'));
  expect(dependencies).toContain(join(directory, 'node_modules/conditional/classic.d.cts'));
  expect(dependencies).toContain(join(directory, 'node_modules/reference-only/module.d.mts'));
  expect(dependencies).not.toContain(join(directory, 'node_modules/reference-only/classic.d.cts'));
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  const before = captureTypeScriptProjectInputs(configuration).fingerprint;
  overwritePreservingMetadata(
    join(directory, 'node_modules/conditional/module.d.mts'),
    'export type Value = 33;'
  );
  expect(captureTypeScriptProjectInputs(configuration).fingerprint).not.toBe(before);
});

it('B6 fingerprints JSON and source dependencies reached through declarations without planning their output', async () => {
  const {captureBuildInputs} = await import('../../../../scripts/build-record.mjs');
  const {captureTypeScriptProjectInputs, getTypeScriptProjectInputs, getTypeScriptOutputPlan} =
    await import('../../../../scripts/typescript-project-inputs.mjs');
  const directory = fixture();
  write(directory, 'node_modules/linked/package.json', '{"name":"linked","types":"index.d.ts"}');
  write(
    directory,
    'node_modules/linked/index.d.ts',
    "import names from './names.json'; import type {Extra} from './extra.ts'; export declare const value: keyof typeof names; export declare const extra: Extra;"
  );
  write(directory, 'node_modules/linked/names.json', '{"before":1}');
  write(directory, 'node_modules/linked/extra.ts', 'export type Extra = 11;');
  write(
    directory,
    'modules/analysis/src/index.ts',
    "import {value} from 'linked'; export const key = value;"
  );
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  const configurationContents = JSON.parse(readFileSync(configuration, 'utf8'));
  configurationContents.compilerOptions.resolveJsonModule = true;
  writeFileSync(configuration, JSON.stringify(configurationContents));
  const project = getTypeScriptProjectInputs(configuration);
  const before = captureBuildInputs(join(directory, 'modules/analysis'), project);
  const watched = captureTypeScriptProjectInputs(configuration);
  writeFileSync(join(directory, 'node_modules/linked/names.json'), '{"after_":1}');
  expect(captureBuildInputs(join(directory, 'modules/analysis'), project)).not.toEqual(before);
  expect(captureTypeScriptProjectInputs(configuration).fingerprint).not.toBe(watched.fingerprint);
  const afterJSON = captureBuildInputs(join(directory, 'modules/analysis'), project);
  overwritePreservingMetadata(
    join(directory, 'node_modules/linked/extra.ts'),
    'export type Extra = 22;'
  );
  expect(captureBuildInputs(join(directory, 'modules/analysis'), project)).not.toEqual(afterJSON);

  expect(
    getTypeScriptOutputPlan(project, join(directory, 'modules/analysis')).flatMap(
      (entry: {compilerOutputs: string[]}) => entry.compilerOutputs
    )
  ).not.toContain('names.json');
});

it('B7 clean referenced package exports build and verify without treating producer outputs as installed inputs', () => {
  const directory = fixture();
  write(
    directory,
    'modules/analysis/src/api.ts',
    'export function getMarker() { return 11 as const; }'
  );
  write(
    directory,
    'modules/lenses/src/index.ts',
    "import {getMarker} from '@luma-contributors/analysis/api'; export const marker = getMarker();"
  );
  const manifest = JSON.parse(
    readFileSync(join(directory, 'modules/analysis/package.json'), 'utf8')
  );
  manifest.exports['./api'] = {
    types: './dist/api.d.ts',
    import: './dist/api.js',
    require: './dist/api.cjs'
  };
  write(directory, 'modules/analysis/package.json', JSON.stringify(manifest));
  const configuration = JSON.parse(
    readFileSync(join(directory, 'modules/lenses/tsconfig.json'), 'utf8')
  );
  configuration.references = [{path: '../analysis'}];
  write(directory, 'modules/lenses/tsconfig.json', JSON.stringify(configuration));
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readFileSync(join(directory, 'modules/lenses/dist/index.d.ts'), 'utf8')).toContain(
    'marker: 11'
  );
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('lenses: wrote 0 CommonJS modules');
  write(
    directory,
    'modules/analysis/src/api.ts',
    'export function getMarker() { return 22 as const; }'
  );
  result = run(directory);
  expect(result.status, result.output).toBe(0);
  expect(readFileSync(join(directory, 'modules/lenses/dist/index.d.ts'), 'utf8')).toContain(
    'marker: 22'
  );
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
}, 30000);

it('B8 refuses competing native and transformed CommonJS output owners before writing', () => {
  const directory = fixture();
  write(directory, 'modules/analysis/src/a.ts', 'export const ordinary = 11;');
  write(directory, 'modules/analysis/src/a.cts', 'export const native = 22;');
  const result = run(directory);
  expect(result.status).toBe(1);
  expect(result.output).toContain('competing output owners');
  expect(result.output).toContain('a.cjs');
  expect(existsSync(join(directory, 'modules/analysis/dist/a.cjs'))).toBe(false);
});

it('B9 local transformers fingerprint their runtime closure without hashing the consuming dist tree', async () => {
  const {captureBuildInputs} = await import('../../../../scripts/build-record.mjs');
  const {getTypeScriptProjectInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  write(
    directory,
    'modules/analysis/transformer.cjs',
    "module.exports = require('./transform-helper.cjs');"
  );
  write(directory, 'modules/analysis/transform-helper.cjs', 'module.exports = 11;');
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  const configurationContents = JSON.parse(readFileSync(configuration, 'utf8'));
  configurationContents.compilerOptions.plugins = [{transform: './transformer.cjs'}];
  writeFileSync(configuration, JSON.stringify(configurationContents));
  const project = getTypeScriptProjectInputs(configuration);
  const before = captureBuildInputs(join(directory, 'modules/analysis'), project);
  write(directory, 'modules/analysis/dist/generated.js', 'export const unrelated = 22;');
  expect(captureBuildInputs(join(directory, 'modules/analysis'), project)).toEqual(before);
  write(directory, 'modules/analysis/transform-helper.cjs', 'module.exports = 22;');
  expect(captureBuildInputs(join(directory, 'modules/analysis'), project)).not.toEqual(before);
});

it('B4 fingerprints the executable selected by the actual esbuild API', () => {
  const directory = fixture();
  const probe = `import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import {captureBuildInputs, getEsbuildExecutable} from './build-record.mjs';
import {getTypeScriptProjectInputs} from './typescript-project-inputs.mjs';
const executable = getEsbuildExecutable(import.meta.resolve('esbuild'));
const originalRead = fs.readFileSync;
let altered = false;
fs.readFileSync = function(file, ...args) {
  const contents = originalRead.call(this, file, ...args);
  return altered && String(file) === executable ? Buffer.concat([contents, Buffer.from('virtual change')]) : contents;
};
syncBuiltinESMExports();
const project = getTypeScriptProjectInputs(new URL('../modules/analysis/tsconfig.json', import.meta.url).pathname);
const directory = new URL('../modules/analysis/', import.meta.url).pathname;
const before = captureBuildInputs(directory, project);
altered = true;
const after = captureBuildInputs(directory, project);
if (before['@tool:esbuild-executable'] === after['@tool:esbuild-executable']) throw new Error('selected executable was not fingerprinted');
console.log('selected executable changed fingerprint');`;
  write(directory, 'scripts/executable-probe.mjs', probe);
  const result = run(directory, 'executable-probe.mjs');
  expect(result.status, result.output).toBe(0);
});

it('B10 captureBuildInputs accepts a string compiler path under strict TypeScript inference', async () => {
  const {default: typeScript} = await import('typescript');
  const file = join(ROOT, 'scripts/build-record.mjs');
  const program = typeScript.createProgram([file], {
    allowJs: true,
    strict: true,
    noEmit: true,
    types: [],
    skipLibCheck: true,
    noResolve: true,
    target: typeScript.ScriptTarget.ES2022
  });
  const source = program.getSourceFile(file)!;
  const checker = program.getTypeChecker();
  const exported = checker
    .getExportsOfModule(checker.getSymbolAtLocation(source)!)
    .find(symbol => symbol.name === 'captureBuildInputs')!;
  const signature = checker.getTypeOfSymbolAtLocation(exported, source).getCallSignatures()[0]!;
  const parameter = signature.getParameters()[2]!;
  expect(
    checker.isTypeAssignableTo(
      checker.getStringType(),
      checker.getTypeOfSymbolAtLocation(parameter, source)
    )
  ).toBe(true);
});

it('B4 configured esbuild executable changes invalidate records and regenerate CommonJS', async () => {
  const {getEsbuildExecutable} = await import('../../../../scripts/build-record.mjs');
  const directory = fixture();
  const executable = getEsbuildExecutable();
  const wrapper = join(directory, 'node_modules/.bin/esbuild-override');
  const contents = `#!${process.execPath}
// marker=11
const result = require('node:child_process').spawnSync(${JSON.stringify(executable)}, process.argv.slice(2), {stdio:'inherit'});
process.exit(result.status ?? 1);`;
  write(directory, 'node_modules/.bin/esbuild-override', contents);
  chmodSync(wrapper, 0o755);
  const environment = {ESBUILD_BINARY_PATH: wrapper};
  let result = run(directory, 'build.mjs', environment);
  expect(result.status, result.output).toBe(0);
  result = run(directory, 'build.mjs', environment);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('analysis: wrote 0 CommonJS modules');
  overwritePreservingMetadata(wrapper, contents.replace('marker=11', 'marker=22'));
  result = run(directory, 'verify-package-exports.mjs', environment);
  expect(result.status).toBe(1);
  expect(result.output).toContain('stale build record');
  result = run(directory, 'build.mjs', environment);
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('analysis: wrote 1 CommonJS modules');
  result = run(directory, 'verify-package-exports.mjs', environment);
  expect(result.status, result.output).toBe(0);
}, 30000);

it('B4 selects the build engine esbuild instead of an unrelated consumer-local installation', async () => {
  const {captureBuildInputs, getEsbuildExecutable} = await import(
    '../../../../scripts/build-record.mjs'
  );
  const {createHash} = await import('node:crypto');
  const {getTypeScriptProjectInputs} = await import(
    '../../../../scripts/typescript-project-inputs.mjs'
  );
  const directory = fixture();
  const project = getTypeScriptProjectInputs(join(directory, 'modules/analysis/tsconfig.json'));
  const expected = createHash('sha256').update(readFileSync(getEsbuildExecutable())).digest('hex');
  write(
    directory,
    'modules/analysis/node_modules/esbuild/package.json',
    '{"name":"esbuild","version":"0.0.0","main":"index.cjs"}'
  );
  write(
    directory,
    'modules/analysis/node_modules/esbuild/index.cjs',
    "exports.transform = () => { throw new Error('consumer esbuild must not run'); };"
  );
  const after = captureBuildInputs(join(directory, 'modules/analysis'), project);
  expect(after['@tool:esbuild-executable']).toBe(expected);
});

it('review NOTE accepts in-package target filenames starting with two dots', () => {
  const directory = fixture();
  write(directory, 'modules/analysis/..metrics.js', 'export const marker = 11;');
  write(directory, 'modules/analysis/..metrics.cjs', 'exports.marker = 11;');
  write(directory, 'modules/analysis/..metrics.d.ts', 'export declare const marker: 11;');
  const manifestPath = join(directory, 'modules/analysis/package.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  manifest.exports['./metrics'] = {
    types: './..metrics.d.ts',
    import: './..metrics.js',
    require: './..metrics.cjs'
  };
  writeFileSync(manifestPath, JSON.stringify(manifest));
  let result = run(directory);
  expect(result.status, result.output).toBe(0);
  result = run(directory, 'verify-package-exports.mjs');
  expect(result.status, result.output).toBe(0);
  expect(result.output).toContain('verified 4 entry points');
  // A parent segment is independently invalid; two dots within a filename are legal.
  for (const target of ['./..', './dist/../../escape.js']) {
    manifest.exports['./metrics'] = target;
    writeFileSync(manifestPath, JSON.stringify(manifest));
    const rejected = run(directory, 'verify-package-exports.mjs');
    expect(rejected.status, rejected.output).toBe(1);
    expect(rejected.output).toContain(`invalid package export target ${target}`);
  }
});
