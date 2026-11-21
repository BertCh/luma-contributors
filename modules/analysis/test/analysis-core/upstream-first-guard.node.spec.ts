// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterAll, beforeAll, describe, expect, it} from 'vitest';
import {
  findLumaReExports,
  findNonPublicLumaImports
} from '../../../../scripts/check-upstream-first.mjs';

// Exercise CLI scans as child processes; importing the origin checks does not run the CLI. `UPSTREAM_FIRST_SCRATCH_FILE` makes it also check one file outside the repository tree.
const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const GUARD_SCRIPT = 'scripts/check-upstream-first.mjs';
const GUARD_TIMEOUT_MILLISECONDS = 120_000;

const PUBLIC_FIXTURE = [
  "import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';",
  "import type {Device} from '@luma.gl/core';",
  "import {dggs} from '@luma.gl/shaders';",
  "// A comment that names '@luma.gl/graph/_internal/arrow' is not an import.",
  "export const DOCUMENTED_INTERNAL_ENTRY = '@luma.gl/graph/_internal/arrow';",
  'export type FixtureDevice = Device;',
  'export const fixtureValues = [GPUCommandGraph, dggs];',
  ''
].join('\n');

const VIOLATING_SPECIFIERS = {
  explicitInternalExportsKey: '@luma.gl/graph/_internal/arrow',
  wildcardInternalExportsKey: '@luma.gl/graph/_internal/gpu-command-graph',
  typeOnlyInternal: '@luma.gl/core/_internal/device',
  dynamicInternal: '@luma.gl/shaders/_internal',
  internalOfUncheckedPackage: '@luma.gl/webgpu/_internal/webgpu-device',
  deepDistributionPath: '@luma.gl/graph/dist/gpu-core/index.js'
} as const;

const VIOLATING_FIXTURE = [
  `import {first} from '${VIOLATING_SPECIFIERS.explicitInternalExportsKey}';`,
  `import {second} from '${VIOLATING_SPECIFIERS.wildcardInternalExportsKey}';`,
  `import type {Third} from '${VIOLATING_SPECIFIERS.typeOnlyInternal}';`,
  `export const fourth = import('${VIOLATING_SPECIFIERS.dynamicInternal}');`,
  `import {fifth} from '${VIOLATING_SPECIFIERS.internalOfUncheckedPackage}';`,
  `import {sixth} from '${VIOLATING_SPECIFIERS.deepDistributionPath}';`,
  "export {GPUCommandGraph} from '@luma.gl/graph/gpu-core';",
  'export type FixtureThird = Third;',
  'export const fixtureValues = [first, second, fifth, sixth];',
  ''
].join('\n');

type GuardRun = {exitCode: number | null; output: string};

let fixtureDirectory = '';

function runGuard(fixtureName: string, fixtureText: string): GuardRun {
  const fixturePath = join(fixtureDirectory, fixtureName);
  writeFileSync(fixturePath, fixtureText);
  const result = spawnSync(process.execPath, [GUARD_SCRIPT], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
    env: {...process.env, UPSTREAM_FIRST_SCRATCH_FILE: fixturePath},
    timeout: GUARD_TIMEOUT_MILLISECONDS
  });
  return {exitCode: result.status, output: `${result.stdout}${result.stderr}`};
}

describe('scripts/check-upstream-first.mjs', () => {
  it.each([
    [
      "import * as luma from '@luma.gl/core'; const graphics = luma; export default graphics.Device;",
      1
    ],
    ["import {Device as DEVICE} from '@luma.gl/core'; export const ANALYSIS_DEVICE = DEVICE;", 1],
    [
      "import {GPU_FFT2D_MAX_DIMENSION as limit} from '@luma.gl/graph/gpu-core'; export const ANALYSIS_LIMIT = (limit);",
      0
    ]
  ] as const)('round 1 [B10] shares originating member and namespace semantics: %s', (text, count) => {
    expect(findLumaReExports(text, 'fixture.ts')).toHaveLength(count);
  });
  it.each([
    'scripts/check-upstream-first.mjs',
    'scripts/luma-binding-origins.mjs',
    'scripts/contributor-contract-catalog.mjs',
    'scripts/check-contributor-contract.mjs'
  ])('imports %s without scanning or running its CLI', script => {
    const directory = mkdtempSync(join(tmpdir(), 'checker-import-'));
    try {
      const result = spawnSync(
        process.execPath,
        [
          '--input-type=module',
          '-e',
          `await import(${JSON.stringify(new URL(script, new URL('../../../../', import.meta.url)).href)});`
        ],
        {cwd: directory, encoding: 'utf8'}
      );
      expect(result.status, result.stdout + result.stderr).toBe(0);
      expect(result.stdout).toBe('');
      expect(result.stderr).toBe('');
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  });
  it('discovers a production JavaScript engine import in ordinary package traversal', () => {
    const directory = mkdtempSync(join(tmpdir(), 'upstream-first-javascript-'));
    try {
      mkdirSync(join(directory, 'modules/analysis/src'), {recursive: true});
      mkdirSync(join(directory, 'modules/lenses/src'), {recursive: true});
      symlinkSync(join(REPOSITORY_ROOT, 'node_modules'), join(directory, 'node_modules'), 'dir');
      writeFileSync(join(directory, 'modules/analysis/src/index.ts'), 'export {};');
      writeFileSync(
        join(directory, 'modules/lenses/src/embedded-dataset-loaders.js'),
        "import {Model} from '@luma.gl/engine';"
      );
      const result = spawnSync(process.execPath, [join(REPOSITORY_ROOT, GUARD_SCRIPT)], {
        cwd: directory,
        encoding: 'utf8'
      });
      expect(result.status, result.stdout + result.stderr).toBe(1);
      expect(result.stdout).toContain('embedded-dataset-loaders.js: engine is restricted');
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  });
  it('rejects installed but disallowed luma.gl packages and production engine imports', () => {
    expect(
      findNonPublicLumaImports(
        "import {WebGPUDevice} from '@luma.gl/webgpu';",
        'modules/analysis/src/analysis-core/fixture.ts'
      )
    ).toHaveLength(1);
    expect(
      findNonPublicLumaImports(
        "import {Model} from '@luma.gl/engine';",
        'modules/analysis/src/analysis-core/fixture.ts'
      )
    ).toHaveLength(1);
    expect(
      findNonPublicLumaImports(
        "import {Model} from '@luma.gl/engine';",
        'modules/analysis/test/fixture.node.spec.ts'
      )
    ).toEqual([]);
    expect(
      findNonPublicLumaImports(
        "import {Model} from '@luma.gl/engine';",
        'examples/deck/fixture/src/scene.ts'
      )
    ).toEqual([]);
  });
  it('rejects a luma.gl binding exported by name, as the default or through a value alias', () => {
    // TOOL-116: only `export ... from '@luma.gl/...'` used to be rejected.
    const file = 'modules/analysis/src/analysis-core/fixture.ts';
    const violations: Record<string, string> = {
      named: "import {Device} from '@luma.gl/core';\nexport {Device};",
      renamed: "import {Device} from '@luma.gl/core';\nexport {Device as AnalysisDevice};",
      typeOnly: "import type {Device} from '@luma.gl/core';\nexport type {Device};",
      defaultExport:
        "import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';\nexport default GPUCommandGraph;",
      localAlias:
        "import {Device} from '@luma.gl/core';\nconst LocalDevice = Device;\nexport {LocalDevice};",
      exportedAlias: "import {Device} from '@luma.gl/core';\nexport const LocalDevice = Device;",
      aliasOfAlias:
        "import {Device} from '@luma.gl/core';\nconst First = Device;\nconst Second = First;\nexport default Second;",
      namespaceMember:
        "import * as core from '@luma.gl/core';\nexport const LocalDevice = core.Device;",
      namespace: "import * as core from '@luma.gl/core';\nexport {core};",
      direct: "export {Device} from '@luma.gl/core';"
    };
    for (const [name, source] of Object.entries(violations)) {
      expect(findLumaReExports(source, file), name).toHaveLength(1);
    }
    const permitted: Record<string, string> = {
      typeAlias: "import type {Device} from '@luma.gl/core';\nexport type FixtureDevice = Device;",
      use: "import {Device} from '@luma.gl/core';\nexport function getDevice(device: Device) { return device; }",
      domainConstant:
        "import {GPU_FFT2D_MAX_DIMENSION} from '@luma.gl/graph/gpu-core';\nexport const MAXIMUM_SPECTRAL_DIMENSION = GPU_FFT2D_MAX_DIMENSION;",
      localExport: 'const value = 1;\nexport {value};\nexport default value;',
      otherPackage: "import {Table} from 'apache-arrow';\nexport {Table};"
    };
    for (const [name, source] of Object.entries(permitted)) {
      expect(findLumaReExports(source, file), name).toEqual([]);
    }
  });
  it('rejects a loose source file while permitting only the root index barrel', () => {
    const directory = mkdtempSync(join(tmpdir(), 'upstream-first-layout-'));
    try {
      mkdirSync(join(directory, 'modules/analysis/src'), {recursive: true});
      symlinkSync(join(REPOSITORY_ROOT, 'node_modules'), join(directory, 'node_modules'), 'dir');
      writeFileSync(join(directory, 'modules/analysis/src/index.ts'), 'export {};');
      writeFileSync(join(directory, 'modules/analysis/src/loose.ts'), 'export const value = 1;');
      const result = spawnSync(process.execPath, [join(REPOSITORY_ROOT, GUARD_SCRIPT)], {
        cwd: directory,
        encoding: 'utf8'
      });
      expect(result.status, result.stdout + result.stderr).toBe(1);
      expect(result.stdout).toContain('modules/analysis/src/loose.ts');
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  });
  beforeAll(() => {
    fixtureDirectory = mkdtempSync(join(tmpdir(), 'upstream-first-guard-'));
  });

  afterAll(() => {
    rmSync(fixtureDirectory, {recursive: true, force: true});
  });

  it(
    'reports no violations for the tree and a fixture that uses only public entry points',
    () => {
      const run = runGuard('public-imports.ts', PUBLIC_FIXTURE);
      expect(run.output).toBe('upstream-first: no violations\n');
      expect(run.exitCode).toBe(0);
    },
    GUARD_TIMEOUT_MILLISECONDS
  );

  it(
    'rejects every _internal specifier, including the explicit exports key of @luma.gl/graph',
    () => {
      const run = runGuard('internal-imports.ts', VIOLATING_FIXTURE);
      const fixturePath = join(fixtureDirectory, 'internal-imports.ts');
      const lines = run.output.trimEnd().split('\n');
      expect(run.exitCode).toBe(1);
      for (const specifier of [
        VIOLATING_SPECIFIERS.explicitInternalExportsKey,
        VIOLATING_SPECIFIERS.wildcardInternalExportsKey,
        VIOLATING_SPECIFIERS.typeOnlyInternal,
        VIOLATING_SPECIFIERS.dynamicInternal,
        VIOLATING_SPECIFIERS.internalOfUncheckedPackage
      ]) {
        expect(lines).toContain(`${fixturePath}: luma.gl internal import '${specifier}'`);
      }
      expect(lines).toContain(
        `${fixturePath}: non-public luma.gl import '${VIOLATING_SPECIFIERS.deepDistributionPath}'`
      );
      expect(lines).toContain(`${fixturePath}: re-export from luma.gl '@luma.gl/graph/gpu-core'`);
      // Six imports and one re-export, and nothing from the repository tree itself.
      expect(lines.at(-1)).toBe('7 upstream-first violations');
      expect(lines).toHaveLength(8);
    },
    GUARD_TIMEOUT_MILLISECONDS
  );
});
