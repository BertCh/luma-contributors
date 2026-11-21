// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {getVitestConfig} from '@vis.gl/dev-tools';
import {
  BENCHMARK_PATTERNS,
  getOptInBrowserDefines,
  getOptInTestExclusions,
  getVerificationBrowserConfiguration
} from './scripts/verification-policy.mjs';

// Kepler specs that exercise examples/deck/kepler-compute run against the example sources in
// place. Those whose imports reach deck.gl, other undeclared packages or luma.gl APIs that are
// not published stay out of the runs; the JSON gives each one's reason.
// LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS=true lifts the exclusion so the listed specs can be
// retried (for example after an example change) without editing the JSON.
const keplerComputeExampleSpecs: string[] =
  process.env['LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS'] === 'true'
    ? []
    : Object.keys(
        JSON.parse(
          readFileSync(new URL('./test/kepler-compute-example-specs.json', import.meta.url), 'utf8')
        ).excludedFromRuns
      );

const excludePatterns = [
  '**/*.disabled.*',
  ...keplerComputeExampleSpecs,
  ...getOptInTestExclusions()
];

// Benchmarks stay out of correctness runs; LUMA_CONTRIBUTORS_BENCHMARKS=true runs only these.
const browserBenchmarkTestPatterns = BENCHMARK_PATTERNS;
const runBrowserBenchmarks = process.env['LUMA_CONTRIBUTORS_BENCHMARKS'] === 'true';

const browserExcludePatterns = [
  'modules/**/*.node.spec.{ts,js}',
  ...browserBenchmarkTestPatterns,
  ...excludePatterns
];

const browserConfiguration = getVerificationBrowserConfiguration();

const browserProject = {
  define: getOptInBrowserDefines(),
  // Declared up front so the first run that reaches DuckDB does not make Vite re-optimise
  // dependencies and reload mid-test, which can leave two apache-arrow module instances.
  optimizeDeps: {include: ['@duckdb/duckdb-wasm', 'apache-arrow']},
  test: {
    environment: 'node',
    // Concurrent pages contend for one GPU; match CI locally.
    fileParallelism: false,
    setupFiles: [
      './test/utils/browser-process-shim.mjs',
      './test/utils/verification-opt-in-env.mjs'
    ],
    include: runBrowserBenchmarks ? browserBenchmarkTestPatterns : ['modules/**/*.spec.{ts,js}'],
    exclude: runBrowserBenchmarks ? excludePatterns : browserExcludePatterns
  }
};

// A committed or local `.only` must not suppress sibling cases while the run exits 0 (TOOL-111):
// Vitest otherwise allows focus whenever CI is unset. `--allowOnly` still opts in for one run.
const headlessProject = {
  ...browserProject,
  test: {...browserProject.test, allowOnly: false}
};

export default getVitestConfig({
  tsconfigProjects: [fileURLToPath(new URL('./tsconfig.test.json', import.meta.url))],
  excludePatterns,
  launchOptions: {
    channel: browserConfiguration.channel,
    args: browserConfiguration.launchArguments
  },
  projects: {
    node: {
      test: {
        browser: {enabled: false},
        pool: 'threads',
        maxWorkers: 1,
        isolate: false,
        allowOnly: false,
        include: ['modules/**/*.node.spec.{ts,js}'],
        exclude: excludePatterns
      }
    },
    browser: browserProject,
    headless: headlessProject
  },
  coverage: {
    provider: 'istanbul',
    include: ['modules/**/src/**/*.{js,ts,tsx}'],
    exclude: ['**/*.d.ts', '**/{dist,node_modules}/**', 'modules/**/test/**']
  }
});
