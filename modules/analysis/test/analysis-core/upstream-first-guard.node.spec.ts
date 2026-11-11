// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterAll, beforeAll, describe, expect, it} from 'vitest';

// The guard is a script with side effects (it walks the tree and prints), so it is run as a child
// process. `UPSTREAM_FIRST_SCRATCH_FILE` makes it also check one file outside the repository tree.
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
