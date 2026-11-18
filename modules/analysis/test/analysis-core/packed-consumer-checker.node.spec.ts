// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, describe, expect, it} from 'vitest';

const CHECKER_PATH = fileURLToPath(
  new URL('../../../../scripts/check-packed-consumer.mjs', import.meta.url)
);
const fixtureDirectories: string[] = [];

function writeFixtureFile(directory: string, path: string, text: string): void {
  const target = join(directory, path);
  mkdirSync(dirname(target), {recursive: true});
  writeFileSync(target, text);
}

function createFixture(): string {
  const directory = mkdtempSync(join(tmpdir(), 'packed-consumer-fixture-'));
  fixtureDirectories.push(directory);
  const computeFunction = `function computeVariogramModelFitOnCPU(bins, options) {
    if (bins.distances.length !== 20 || options.models[1] !== 'exponential') throw new Error('wrong documented input');
    return {parameters: Float32Array.of(0, 0, 0, 0, 0.2, 1.5, 2.2, 0, 0, 0, 0, 0), status: Uint32Array.of(0, 0, 0)};
  }`;
  const files: Record<string, string> = {
    'package/package.json': JSON.stringify({
      name: '@luma-contributors/analysis',
      version: '0.0.0',
      type: 'module',
      license: 'MIT',
      files: ['dist', 'src', 'LICENSES', 'THIRD-PARTY-NOTICES.md'],
      scripts: {prepack: 'node -e "throw new Error(\'prepack must not run\')"'},
      dependencies: {'fixture-dependency': '1.0.0'},
      peerDependencies: {'fixture-peer': '1.0.0'},
      exports: {
        '.': {types: './dist/index.d.ts', import: './dist/index.js', require: './dist/index.cjs'},
        './dataframe-analysis': {
          types: './dist/dataframe-analysis/index.d.ts',
          import: './dist/dataframe-analysis/index.js',
          require: './dist/dataframe-analysis/index.cjs'
        },
        './gpu-spatial-analysis': {
          types: './dist/gpu-spatial-analysis/index.d.ts',
          import: './dist/gpu-spatial-analysis/index.js',
          require: './dist/gpu-spatial-analysis/index.cjs'
        }
      }
    }),
    'package/README.md': '[licence](LICENSE)\n[web](https://example.com/docs)\n',
    'package/LICENSE': 'MIT fixture licence',
    'package/LICENSES/Apache-2.0.txt': 'Apache fixture licence',
    'package/LICENSES/BSD-3-Clause.txt': 'BSD fixture licence',
    'package/LICENSES/H3-NOTICE.txt': 'H3 fixture notice',
    'package/THIRD-PARTY-NOTICES.md': 'Default source expression: `MIT`.',
    'package/src/index.ts': '// SPDX-License-Identifier: MIT\nexport const value = 1;',
    'package/dist/index.js':
      "import dependency from 'fixture-dependency'; import peer from 'fixture-peer'; export const value = dependency + peer;",
    'package/dist/index.cjs':
      "exports.value = require('fixture-dependency') + require('fixture-peer');",
    'package/dist/index.d.ts': 'export declare const value: number;',
    'package/dist/gpu-spatial-analysis/index.js':
      'export const GPU_LINE_DENSITY_PARAMETER_SCHEMA = {wordLength: 4};',
    'package/dist/gpu-spatial-analysis/index.cjs':
      'exports.GPU_LINE_DENSITY_PARAMETER_SCHEMA = {wordLength: 4};',
    'package/dist/gpu-spatial-analysis/index.d.ts':
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: {wordLength: number};',
    'package/dist/dataframe-analysis/index.js': `export ${computeFunction}`,
    'package/dist/dataframe-analysis/index.cjs': `${computeFunction}\nexports.computeVariogramModelFitOnCPU = computeVariogramModelFitOnCPU;`,
    'package/dist/dataframe-analysis/index.d.ts':
      'export declare function computeVariogramModelFitOnCPU(...args: unknown[]): unknown;'
  };
  for (const name of ['fixture-dependency', 'fixture-peer']) {
    files[`node_modules/${name}/package.json`] = JSON.stringify({
      name,
      version: '1.0.0',
      main: 'index.cjs'
    });
    files[`node_modules/${name}/index.cjs`] = 'module.exports = 1;';
  }
  for (const [path, text] of Object.entries(files)) writeFixtureFile(directory, path, text);
  return directory;
}

function runChecker(
  directory: string,
  environment: NodeJS.ProcessEnv = {},
  executableArguments: string[] = []
) {
  const result = spawnSync(
    process.execPath,
    [
      ...executableArguments,
      CHECKER_PATH,
      '--package-directory',
      join(directory, 'package'),
      '--dependency-directory',
      join(directory, 'node_modules')
    ],
    {encoding: 'utf8', timeout: 30_000, env: {...process.env, ...environment}}
  );
  return {exitCode: result.status, output: `${result.stdout}${result.stderr}`};
}

type FixtureManifest = {
  files: string[];
  license: string;
  exports: Record<string, {types: string; import: string; require: string}>;
  dependencies: Record<string, string>;
  peerDependencies: Record<string, string>;
};

function editManifest(directory: string, edit: (manifest: FixtureManifest) => void): void {
  const path = join(directory, 'package/package.json');
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  edit(manifest);
  writeFileSync(path, JSON.stringify(manifest));
}

afterEach(() => {
  for (const directory of fixtureDirectories.splice(0)) {
    rmSync(directory, {recursive: true, force: true});
  }
});

describe('packed consumer checker', () => {
  it('final1 F8 preserves a 256 MiB heap in the isolated runtime while clearing module-loading options', () => {
    const directory = createFixture();
    const preload = join(directory, 'preload.cjs');
    writeFileSync(preload, 'globalThis.loadingOptionWasInherited = true;');
    const entry = join(directory, 'package/dist/index.js');
    writeFileSync(
      entry,
      `import {getHeapStatistics} from 'node:v8';\nconst heapMegabytes = getHeapStatistics().heap_size_limit / 1024 ** 2;\nprocess.stdout.write('packed-worker-heap-mib=' + heapMegabytes + '\\n');\nif (globalThis.loadingOptionWasInherited) throw new Error('module-loading option leaked');\nif (heapMegabytes >= 384) throw new Error('caller heap was dropped');\n${readFileSync(entry, 'utf8')}`
    );
    for (const environmentHeap of [256, 512]) {
      const result = runChecker(
        directory,
        {
          NODE_OPTIONS: `--max-old-space-size=${environmentHeap} --require=${preload}`
        },
        environmentHeap === 512 ? ['--max-old-space-size=256'] : []
      );
      expect(result.exitCode, result.output).toBe(0);
      const heapMegabytes = Number(result.output.match(/packed-worker-heap-mib=([\d.]+)/)?.[1]);
      expect(heapMegabytes).toBeGreaterThan(256);
      expect(heapMegabytes).toBeLessThan(384);
    }
  });
  it('rejects an external TMPDIR symlink that resolves into the workspace', () => {
    const directory = createFixture();
    const workspaceTemporaryDirectory = mkdtempSync(
      fileURLToPath(
        new URL('../../../../.typecheck/release-package-real-isolation-', import.meta.url)
      )
    );
    fixtureDirectories.push(workspaceTemporaryDirectory);
    const alias = join(directory, 'temporary-root');
    symlinkSync(workspaceTemporaryDirectory, alias, 'dir');
    const result = runChecker(directory, {TMPDIR: alias});
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('temporary directory must be outside the workspace');
  });

  it('accepts a later stable peer version allowed by the documented prerelease range', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.peerDependencies['fixture-peer'] = '~10.0.0-alpha.16';
    });
    writeFixtureFile(
      directory,
      'node_modules/fixture-peer/package.json',
      JSON.stringify({name: 'fixture-peer', version: '10.0.99', main: 'index.cjs'})
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });
  it('rejects a declaration that erases the schema to any', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/dist/gpu-spatial-analysis/index.d.ts',
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: any;'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('TS2578');
  });
  it('rejects a workspace-local TMPDIR before an undeclared workspace import can resolve', () => {
    const directory = createFixture();
    const workspaceTemporaryDirectory = mkdtempSync(
      fileURLToPath(new URL('../../../../.typecheck/release-package-isolation-', import.meta.url))
    );
    fixtureDirectories.push(workspaceTemporaryDirectory);
    writeFixtureFile(
      directory,
      'package/dist/index.js',
      "import 'typescript'; export const value = 1;"
    );
    writeFixtureFile(
      directory,
      'package/dist/index.cjs',
      "require('typescript'); exports.value = 1;"
    );
    const result = runChecker(directory, {TMPDIR: workspaceTemporaryDirectory});
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('temporary directory must be outside the workspace');
  });

  it.each(['md', 'js'])('checks SPDX headers in shipped .%s files', extension => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      `package/src/guide.${extension}`,
      '// SPDX-License-Identifier: ISC\n'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('unlisted SPDX expression ISC');
  });

  it('checks a reference link defined inside a blockquote', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/README.md',
      '> [guide][reference]\n>\n> [reference]: missing.md\n'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('README link missing');
  });

  it('ignores links inside indented Markdown code blocks', () => {
    const directory = createFixture();
    writeFixtureFile(directory, 'package/README.md', '    [example](missing.md)\n');
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });

  it.each([
    'dependencies',
    'peerDependencies'
  ] as const)('rejects a present installed version outside its %s range', section => {
    const directory = createFixture();
    const name = section === 'dependencies' ? 'fixture-dependency' : 'fixture-peer';
    editManifest(directory, manifest => {
      manifest[section][name] = '>=2.0.0';
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`${name}@1.0.0 does not satisfy >=2.0.0`);
  });
  it('packs offline, links only declared dependencies and peers, and checks both native formats', () => {
    const result = runChecker(createFixture());
    expect(result.exitCode, result.output).toBe(0);
    expect(result.output).toContain('verified 3 packed entry points');
    expect(result.output).toContain('native NodeNext declaration consumer passed');
    expect(result.output).toContain('documented CPU variogram fit passed');
  });

  it.each([
    ['missing declaration', 'package/dist/index.d.ts', '', 'types target'],
    [
      'broken native import',
      'package/dist/index.js',
      "export {value} from './workspace-only.js';",
      'workspace-only.js'
    ],
    [
      'different CJS names with the same count',
      'package/dist/index.cjs',
      'exports.other = 1;',
      'export names differ'
    ],
    [
      'relative README escape',
      'package/README.md',
      '[outside](../outside.md)',
      'README link escapes'
    ],
    [
      'missing README reference target',
      'package/README.md',
      '[guide][guide]\n\n[guide]: missing.md',
      'README link missing'
    ],
    ['missing HTML image', 'package/README.md', '<img src="missing.png">', 'README link missing'],
    [
      'nested Markdown label',
      'package/README.md',
      '[a [nested] label](missing.md)',
      'README link missing'
    ],
    [
      'nested parentheses in target',
      'package/README.md',
      '[guide](missing(a(b)).md)',
      'README link missing'
    ],
    ['unquoted HTML image', 'package/README.md', '<img src=missing.png>', 'README link missing'],
    [
      'missing percent-encoded link',
      'package/README.md',
      '[guide](missing%20guide.md)',
      'README link missing'
    ],
    [
      'unlisted SPDX expression',
      'package/src/index.ts',
      '// SPDX-License-Identifier: ISC',
      'unlisted SPDX expression'
    ],
    [
      'unlisted non-MIT file',
      'package/src/index.ts',
      '// SPDX-License-Identifier: BSD-3-Clause',
      'unlisted source notice'
    ],
    [
      'NaN CPU output',
      'package/dist/dataframe-analysis/index.js',
      'export function computeVariogramModelFitOnCPU() {return {parameters: new Float32Array(12).fill(NaN), status: new Uint32Array(3)}}',
      'CPU variogram parameter'
    ]
  ])('rejects %s', (_name, path, text, message) => {
    const directory = createFixture();
    if (text) writeFixtureFile(directory, path, text);
    else rmSync(join(directory, path));
    if (message === 'unlisted source notice') {
      writeFixtureFile(directory, 'package/THIRD-PARTY-NOTICES.md', '`MIT` and `BSD-3-Clause`');
      editManifest(directory, manifest => {
        manifest.license = 'MIT AND BSD-3-Clause';
      });
    }
    writeFixtureFile(directory, 'outside.md', 'This exists outside the tarball.');
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(message);
  });

  it('refuses an empty exports map', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.exports = {};
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('zero exports');
  });

  it('allows packed destinations with spaces and parentheses and ignores links in code', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.files.push('guide (one).md', 'a(b(c)).md');
    });
    writeFixtureFile(directory, 'package/guide (one).md', 'guide');
    writeFixtureFile(directory, 'package/a(b(c)).md', 'guide');
    writeFixtureFile(
      directory,
      'package/README.md',
      [
        '[guide](<guide (one).md>)',
        '[guide](a(b(c)).md)',
        '[encoded](guide%20%28one%29.md#heading)',
        '`[code](missing.md)`',
        '```md',
        '[code](missing.md)',
        '```'
      ].join('\n')
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });

  it('rejects a declaration target outside the tarball even if the file exists', () => {
    const directory = createFixture();
    writeFixtureFile(directory, 'outside.d.ts', 'export const value: number;');
    editManifest(directory, manifest => {
      manifest.exports['.'].types = join(directory, 'outside.d.ts');
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('types target');
  });

  it('refuses wildcard entries instead of silently skipping them', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.exports['./*'] = {...manifest.exports['.']};
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('unsupported export');
  });

  it('requires every declared peer to be installed in the dependency directory', () => {
    const directory = createFixture();
    rmSync(join(directory, 'node_modules/fixture-peer'), {recursive: true});
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('missing installed dependency fixture-peer');
  });

  it('checks licence files in the tarball rather than their workspace presence', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.files = ['dist', 'src', 'THIRD-PARTY-NOTICES.md'];
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('LICENSES/Apache-2.0.txt');
  });
});
