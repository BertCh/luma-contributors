// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Typechecks tests with the same exclusions vitest applies (see vitest.config.ts):
// specs that need the parked examples/deck/kepler-compute example are left out.
import {spawnSync} from 'node:child_process';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const rootDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
// Specs excluded from runs, plus specs that run but whose imports reach undeclared packages or
// luma.gl names that are not published (see the reasons in the JSON).
const keplerComputeExampleSpecLists = JSON.parse(
  readFileSync(join(rootDirectory, 'test/kepler-compute-example-specs.json'), 'utf8')
);
const keplerComputeExampleSpecs = [
  ...Object.keys(keplerComputeExampleSpecLists.excludedFromRuns),
  ...Object.keys(keplerComputeExampleSpecLists.excludedFromTypecheck)
];

const tsconfig = {
  extends: '../tsconfig.json',
  include: ['../modules/*/src', '../modules/*/test', '../test/**/*'],
  exclude: [
    '../modules/**/dist',
    '../node_modules',
    ...keplerComputeExampleSpecs.map((specPath) => `../${specPath}`),
    '../examples/deck/kepler-compute/**',
    // Uses engine/arrow geometry APIs (makeGeometryGPUVectors) that published luma.gl lacks.
    '../modules/kepler/test/host-wkb-selection.spec.ts',
    // Fixtures used only by the excluded kepler-compute example specs.
    '../modules/kepler/test/utils/trip-runtime-fixture.ts',
    '../modules/kepler/test/utils/tabular-trip-runtime-fixture.ts'
  ]
};

mkdirSync(join(rootDirectory, '.typecheck'), {recursive: true});
writeFileSync(
  join(rootDirectory, '.typecheck/tsconfig.tests.json'),
  `${JSON.stringify(tsconfig, null, 2)}\n`
);

const tscBinary = join(rootDirectory, 'node_modules/.bin/tsc');
const result = spawnSync(tscBinary, ['-p', '.typecheck/tsconfig.tests.json', '--noEmit'], {
  cwd: rootDirectory,
  stdio: 'inherit'
});
process.exit(result.status ?? 1);
