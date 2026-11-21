// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawn, spawnSync, type ChildProcess} from 'node:child_process';
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  utimesSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';

const ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const directories: string[] = [];
const children: ChildProcess[] = [];
function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
  utimesSync(join(directory, file), 1_700_000_000, 1_700_000_000);
}
function overwritePreservingMetadata(file: string, contents: string) {
  const previous = statSync(file);
  expect(Buffer.byteLength(contents)).toBe(previous.size);
  writeFileSync(file, contents);
  utimesSync(file, previous.atimeMs / 1000, previous.mtimeMs / 1000);
  expect(statSync(file).size).toBe(previous.size);
  expect(statSync(file).mtimeMs).toBe(previous.mtimeMs);
}
afterEach(async () => {
  for (const child of children.splice(0)) {
    if (child.exitCode !== null || child.signalCode !== null) continue;
    const exited = new Promise<void>(resolve => child.once('exit', () => resolve()));
    child.kill('SIGTERM');
    await exited;
  }
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});
function status(directory: string) {
  const result = spawnSync(process.execPath, [join(directory, 'scripts/typecheck-status.mjs')], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 10000,
    env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256'}
  });
  return {status: result.status, output: `${result.stdout}${result.stderr}`};
}

it.each([
  [
    'import branch',
    'module.d.mts',
    "import names from './names.json' with {type: 'json'}; import type {Extra} from './extra.js'; export type Value = [keyof typeof names, Extra, 22];"
  ],
  ['transitive JSON', 'names.json', '{"after_":1}'],
  ['transitive source', 'extra.ts', 'export type Extra = 22;']
])(
  'TOOL-108 rejects consumed %s edits after a completed live cycle',
  async (_label, input, replacement) => {
    const directory = realpathSync(mkdtempSync(join(tmpdir(), 'typecheck-cycle-')));
    directories.push(directory);
    mkdirSync(join(directory, 'scripts'));
    for (const script of ['typecheck-status.mjs', 'typescript-project-inputs.mjs'])
      copyFileSync(join(ROOT, 'scripts', script), join(directory, 'scripts', script));
    write(
      directory,
      'node_modules/typescript/index.js',
      `module.exports = require(${JSON.stringify(join(ROOT, 'node_modules/typescript/lib/typescript.js'))});`
    );
    // The compiler double emits one completed cycle and remains alive without observing edits.
    // The actual supervisor and status command must detect consumed bytes independently.
    write(
      directory,
      'node_modules/typescript/bin/tsc',
      `
    console.log('Starting compilation in watch mode');
    console.log('Found 0 errors. Watching for file changes.');
    setInterval(() => {}, 1000);
  `
    );
    const options = {
      module: 'NodeNext',
      moduleResolution: 'NodeNext',
      resolveJsonModule: true,
      types: [],
      target: 'ES2022',
      lib: ['ES2022']
    };
    write(
      directory,
      'tsconfig.check-src.json',
      JSON.stringify({compilerOptions: options, files: ['modules/analysis/src/source.mts']})
    );
    write(
      directory,
      'modules/analysis/src/source.mts',
      "import type {Value} from 'conditional'; export type Result = Value;"
    );
    write(
      directory,
      'node_modules/conditional/package.json',
      JSON.stringify({
        name: 'conditional',
        type: 'module',
        exports: {
          '.': {
            import: {types: './module.d.mts'},
            require: {types: './classic.d.cts'}
          }
        }
      })
    );
    write(
      directory,
      'node_modules/conditional/module.d.mts',
      "import names from './names.json' with {type: 'json'}; import type {Extra} from './extra.js'; export type Value = [keyof typeof names, Extra, 11];"
    );
    write(directory, 'node_modules/conditional/classic.d.cts', 'export type Value = 11;');
    write(directory, 'node_modules/conditional/names.json', '{"before":1}');
    write(directory, 'node_modules/conditional/extra.ts', 'export type Extra = 11;');
    // Independent compiler program establishes which branch and transitive files are consumed.
    const configuration = typescript.readConfigFile(
      join(directory, 'tsconfig.check-src.json'),
      typescript.sys.readFile
    );
    const parsed = typescript.parseJsonConfigFileContent(
      configuration.config,
      typescript.sys,
      directory
    );
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
    const consumed = program.getSourceFiles().map(source => source.fileName);
    expect(consumed).toContain(
      join(dirname(typescript.getDefaultLibFilePath(parsed.options)), 'lib.es2022.d.ts')
    );
    expect(consumed).toContain(join(directory, 'node_modules/conditional', input));
    expect(consumed).not.toContain(join(directory, 'node_modules/conditional/classic.d.cts'));
    const child = spawn(
      process.execPath,
      [join(directory, 'scripts/typecheck-status.mjs'), '--watch'],
      {
        cwd: directory,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {...process.env, NODE_OPTIONS: '--max-old-space-size=256'}
      }
    );
    children.push(child);
    let output = '';
    child.stdout?.on('data', chunk => {
      output += chunk;
    });
    child.stderr?.on('data', chunk => {
      output += chunk;
    });
    let completed = false;
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      try {
        const owner = JSON.parse(
          readFileSync(join(directory, '.typecheck/watch.owner.json'), 'utf8')
        );
        if (owner.completedAt && owner.cycleId === 1) {
          completed = true;
          break;
        }
      } catch {}
      await delay(20);
    }
    expect(completed, output).toBe(true);
    let result = status(directory);
    expect(result.status, result.output).toBe(0);
    expect(result.output).toContain('live cycle 1 completed');
    overwritePreservingMetadata(
      join(directory, 'node_modules/conditional/classic.d.cts'),
      'export type Value = 22;'
    );
    result = status(directory);
    expect(result.status, result.output).toBe(0);
    overwritePreservingMetadata(join(directory, 'node_modules/conditional', input), replacement);
    result = status(directory);
    expect(result.status, result.output).toBe(2);
    expect(result.output).toContain(
      'Completed typecheck cycle is stale relative to current source/config edits'
    );
  },
  15000
);
