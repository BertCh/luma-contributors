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
  'modules/kepler/test/host-wkb-selection.spec.ts',
  'modules/analysis/test/arrow/arrow-column-info.spec.ts',
  'modules/analysis/test/arrow/get-arrow-data.spec.ts'
];

// Pure data, layout and shader-assembly specs that run in Node instead of a browser page.
const nodeOnlyTestPatterns = [
  'modules/analysis/test/gpu-tables/{gpu-table-model,generated-buffer-batches,table-buffer-planner}.spec.{ts,js}',
  'modules/analysis/test/arrow/{arrow-column-info,arrow-shader-layout,arrow-input-schema,arrow-splats,arrow-matrix-gpu-vector,arrow-geometry,arrow-model,arrow-paths,arrow-polygon-renderer,arrow-polygon,plain-gpu-table,analyze-arrow-table,arrow-variable-length-attribute-gpu-vector,get-arrow-data,arrow-fixed-size-list}.spec.{ts,js}',
  'modules/analysis/test/gpu-vector/gpu-data-evaluator-types.spec.{ts,js}',
  'modules/analysis/test/operations/arithmetic-operation.spec.{ts,js}',
  'modules/analysis/test/gpu-core/{gpu-scalar,gpu-matvec,gpu-matmul,gpu-elementwise,gpu-coo-to-csr}.spec.{ts,js}',
  'modules/analysis/test/utils/{expression,webgpu-dispatch}.spec.{ts,js}',
  'modules/analysis/test/geospatial/geospatial-types.spec.{ts,js}'
];

// Plain specs that an existing .node.spec wrapper imports; exclude them so each suite registers once.
const nodeWrapperSourcePatterns = [
  'modules/analysis/test/arrow/{arrow-polygon,arrow-polygon-renderer,arrow-splats}.spec.{ts,js}',
  'modules/analysis/test/gpu-tables/gpu-table-model.spec.{ts,js}'
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
  'modules/analysis/test/geospatial-expansion/resistance-components-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-analysis/shared-pyramid-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts',
  'modules/analysis/test/gpu-raster/remote-sensing/raster-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis-benchmark.spec.ts',
  'modules/analysis/test/gpu-network/network-reachability/gpu-network-reachability-bench.spec.ts',
  'modules/analysis/test/gpu-network/network-accessibility/gpu-network-accessibility-bench.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/region-statistics/region-statistics-benchmark.spec.ts',
  'modules/analysis/test/gpu-tables/residency-arena/residency-arena-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/relief-visualization/relief-visualization-benchmark.spec.ts',
  'modules/analysis/test/gpu-terrain/terrain-illumination/terrain-illumination-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-join/spatial-join-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/neighbor-search/neighbor-search-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/statistics-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-validation-performance-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-jacobi-preconditioner-benchmark.spec.ts',
  'modules/analysis/test/experimental/simulation/scientific-flow-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-spatial-query-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-kernel-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-fragmentation-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-workgroup-reduction-benchmark.spec.ts',
  'modules/analysis/test/gpu-core/gpu-workgroup-scan-benchmark.spec.ts',
  'modules/analysis/test/gpu-graph/gpu-graph-benchmark.spec.ts',
  'modules/analysis/test/gpu-project/projection-benchmark.spec.ts',
  'modules/analysis/test/gpu-project/projection-performance.spec.ts',
  'modules/analysis/test/gpu-project/projection-conic-performance.spec.ts',
  'modules/analysis/test/gpu-project/projection-routing-performance.spec.ts',
  'modules/analysis/test/gpu-project/projection-table-performance.spec.ts',
  'modules/analysis/test/gpu-dataframe/pair-statistics/ripley-isotropic-envelope-benchmark.spec.ts',
  'modules/analysis/test/gpu-project/projection-program-benchmark.spec.ts'
];
const runBrowserBenchmarks = process.env.LUMA_CONTRIBUTORS_BENCHMARKS === 'true';

const browserExcludePatterns = [
  'modules/**/*.node.spec.{ts,js}',
  ...browserBenchmarkTestPatterns,
  ...nodeOnlyTestPatterns,
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
        include: ['modules/**/*.node.spec.{ts,js}', ...nodeOnlyTestPatterns],
        exclude: [...excludePatterns, ...nodeWrapperSourcePatterns]
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
