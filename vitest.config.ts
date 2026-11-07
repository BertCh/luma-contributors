// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {getVitestConfig} from '@vis.gl/dev-tools';

// Specs that exercise examples/deck/kepler-compute. That example still needs
// @deck.gl-community/gpu-layers and luma.gl engine/arrow APIs that are not published,
// so these stay out of the default runs until the example is ported.
const keplerComputeExampleSpecs: string[] = JSON.parse(
  readFileSync(new URL('./test/kepler-compute-example-specs.json', import.meta.url), 'utf8')
);

const excludePatterns = [
  '**/*.disabled.*',
  ...keplerComputeExampleSpecs,
  // Uses engine/arrow geometry APIs (makeGeometryGPUVectors) that published luma.gl lacks.
  'modules/kepler/test/host-wkb-selection.spec.ts'
];

// Benchmarks stay out of correctness runs; LUMA_CONTRIBUTORS_BENCHMARKS=true runs only these.
const browserBenchmarkTestPatterns = [
  'modules/analysis/test/gpu-spatial-analysis/point-process/point-process-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/urban-ecology/urban-ecology-benchmark.spec.ts',
  'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-benchmark.spec.ts',
  'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts',
  'modules/analysis/test/gpu-network/network-optimization/network-movement-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/variogram-profile-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/conditional-simulation-reuse-benchmark.spec.ts',
  'modules/lenses/test/computation-lenses/raster-terrain/coverage-tranche-one-benchmark.spec.ts',
  'modules/analysis/test/geospatial-expansion/backlog-performance-benchmark.spec.ts',
  'modules/analysis/test/geospatial-expansion/irregular-geospatial-profile-benchmark.spec.ts',
  'modules/analysis/test/geospatial-expansion/neighborhood-strategy-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-analysis/shared-pyramid-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts',
  'modules/analysis/test/gpu-raster/remote-sensing/raster-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis-benchmark.spec.ts',
  'modules/analysis/test/gpu-network/network-reachability/gpu-network-reachability-bench.spec.ts',
  'modules/analysis/test/gpu-network/network-accessibility/gpu-network-accessibility-bench.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/region-statistics/region-statistics-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/relief-visualization/relief-visualization-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-illumination/terrain-illumination-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-join/spatial-join-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/neighbor-search/neighbor-search-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/statistics-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-validation-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-dataframe/pair-statistics/ripley-isotropic-envelope-benchmark.spec.ts'
];
const runBrowserBenchmarks = process.env.LUMA_CONTRIBUTORS_BENCHMARKS === 'true';

const browserExcludePatterns = [
  'modules/**/*.node.spec.{ts,js}',
  ...browserBenchmarkTestPatterns,
  ...excludePatterns
];

const launchArguments = [
  '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist',
  ...(process.env.CI && process.platform === 'linux'
    ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    : [])
];

const browserProject = {
  test: {
    environment: 'node',
    // Concurrent pages contend for one GPU; match CI locally.
    fileParallelism: false,
    setupFiles: ['./test/utils/browser-process-shim.mjs'],
    include: runBrowserBenchmarks ? browserBenchmarkTestPatterns : ['modules/**/*.spec.{ts,js}'],
    exclude: runBrowserBenchmarks ? excludePatterns : browserExcludePatterns
  }
};

export default getVitestConfig({
  excludePatterns,
  launchOptions: {channel: 'chromium', args: launchArguments},
  projects: {
    node: {
      test: {
        browser: {enabled: false},
        pool: 'threads',
        isolate: false,
        include: ['modules/**/*.node.spec.{ts,js}'],
        exclude: excludePatterns
      }
    },
    browser: browserProject,
    headless: browserProject
  },
  coverage: {
    provider: 'istanbul',
    include: ['modules/**/src/**/*.{js,ts,tsx}'],
    exclude: ['**/*.d.ts', '**/{dist,node_modules}/**', 'modules/**/test/**']
  }
});
