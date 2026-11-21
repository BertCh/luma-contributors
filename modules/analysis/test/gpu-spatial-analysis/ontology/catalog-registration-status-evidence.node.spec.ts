// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {addStopValueCubeRecipe} from '../../../src/gpu-spatial-analysis/recipes/stop-value-cube-recipe';
import {GPUPointCloudCoverageValidity} from '../../../src/pointcloud-analysis/operations/gpu-pointcloud-coverage-validity';
import {AUDITED_EMPTY_OPERATOR_STATUSES} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-status-audits';
import {
  GPUGeographicalDetector,
  GPU_GEOGRAPHICAL_DETECTOR_LAYOUT
} from '../../../src/gpu-spatial-analysis/geographical-detector/gpu-geographical-detector';
import {GPUSegregation} from '../../../src/gpu-spatial-analysis/segregation/gpu-segregation';
import {getGPUSegregationLayout} from '../../../src/gpu-spatial-analysis/segregation/segregation-layout';
import {
  getGPUSpatialAnalysisOperator,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';

it('B1 excludes unrestricted stop assignment from exact time-series discovery', () => {
  // Face 0, longitude 0: v = tan(latitude). S2 row = floor(st(v) * 2^20).
  // Binary64 reference for -1e-7 degrees gives 524287. Float32 rounds 1-3v to 1,
  // so st becomes .5 and the assigned row is 524288. This host transcription is
  // evidence of boundary sensitivity, not a claim about a device's instruction schedule.
  const vertical = Math.tan((-1e-7 * Math.PI) / 180);
  expect(Math.floor((1 - 0.5 * Math.sqrt(1 - 3 * vertical)) * 2 ** 20)).toBe(524287);
  expect(
    Math.floor(
      Math.fround(
        1 - Math.fround(0.5 * Math.sqrt(Math.fround(1 - Math.fround(3 * Math.fround(vertical)))))
      ) *
        2 ** 20
    )
  ).toBe(524288);
  const record = getGPUAnalysisOperator('addStopValueCubeRecipe');
  expect(queryGPUAnalysisOperators({semantics: 'exact', output: 'time-series'})).not.toContain(
    record
  );
  expect(record?.semantics).toBe('approximate');
  expect(record?.summary).toContain('exact after cell assignment');
});

it('B2 scopes stop-cube suppression to the supplied negative loss flags', () => {
  const summary = getGPUAnalysisOperator('addStopValueCubeRecipe')?.summary;
  expect(summary).toContain('supplied overflow and candidateOverflow');
  expect(summary).toContain('requiredCount alone is not checked');
});

it('B3 discovers the allocation adapter through matrix attributes, not network topology', () => {
  const record = getGPUAnalysisOperator('addNetworkLocationAllocationAdapter');
  expect(queryGPUAnalysisOperators({input: 'attributes'})).toContain(record);
  expect(queryGPUAnalysisOperators({input: 'network'})).not.toContain(record);
  expect(record?.summary).toContain('labelled network cost matrix');
});

it('S3 defines audited-empty status by catalog names while retaining scalar diagnostics', () => {
  const source = readFileSync(
    'modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts',
    'utf8'
  );
  expect(source).toContain('no audited scalar under a catalog status name');
  for (const name of ['GPUGeographicalDetector', 'GPUSegregation'])
    expect(getGPUAnalysisOperator(name)?.status).toEqual([]);
});

it('CCK-01 registers the live stop cube and its distinct stage evidence', () => {
  const record = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'addStopValueCubeRecipe'
  });
  expect(record).toMatchObject({
    primaryCapability: 'movement-analysis',
    inputs: ['points', 'attributes', 'time-series', 'cells', 'spatial-weights', 'parameters'],
    // The dense counts are a cell-period time series; returned weights retain dictionary order.
    outputs: ['time-series', 'spatial-weights', 'summary'],
    dimensionality: '2d+time',
    cardinality: 'fixed-sized',
    status: ['count', 'invalidCount', 'overflow', 'candidateOverflow']
  });
  for (const phrase of [
    'live stop prefix',
    'fixed periods',
    'stable dictionary',
    'cell-major',
    'suppress'
  ])
    expect(record?.summary).toContain(phrase);
  expect(getGPUAnalysisOperatorCatalog().filter(candidate => candidate === record)).toHaveLength(1);
  expect(
    queryGPUAnalysisOperators({primaryCapability: 'movement-analysis', status: 'candidateOverflow'})
  ).toContain(record);

  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let sequence = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `stop-catalog-${sequence++}`, format, length);
    const stopStatus = {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    };
    const sliceStatus = {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)};
    const result = addStopValueCubeRecipe(graph, {
      time: 'finish',
      family: 'quadbin',
      resolution: 1,
      dictionary: view('uint32x2', 2),
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      timestamps: view('uint32x2', 3),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', origin: 0n, width: 10, count: 2},
      stops: {
        centroids: view('float32x2', 3),
        startRows: view('uint32', 3),
        finishRows: view('uint32', 3),
        status: stopStatus
      },
      sliceStatuses: [sliceStatus]
    });
    // Hand-worked shape: two dictionary cells x two fixed periods = four cell-major bins.
    expect(result.outputs.counts.length).toBe(4);
    expect(result.status.stages.map(stage => stage.stage)).toEqual(['stops', 'slice-0', 'cube']);
    expect(result.status.stages[0].status).toEqual(stopStatus);
    expect(result.status.stages[1].status).toEqual(sliceStatus);
    expect(result.status.stages[2].status).toEqual({
      invalidCount: result.status.invalidCount,
      overflow: result.status.overflow,
      candidateOverflow: result.status.candidateOverflow
    });
    expect(result.status.stages[2].status).not.toBe(result.status.stages[0].status);
    for (const name of ['invalidCount', 'overflow', 'candidateOverflow'] as const) {
      expect(result.status[name].format).toBe('uint32');
      expect(result.status[name].length).toBe(1);
    }
    const suppress = getWGSLKernelNodeRecord(graph, 'stop-value-cube-suppress');
    expect(suppress?.bindings.find(binding => binding.name === 'invalid')?.view).toBe(
      result.status.invalidCount
    );
    expect(suppress?.bindings.find(binding => binding.name === 'overflow')?.view).toBe(
      result.status.overflow
    );
    expect(suppress?.bindings.find(binding => binding.name === 'candidate')?.view).toBe(
      result.status.candidateOverflow
    );
    expect(suppress?.bindings.find(binding => binding.name === 'counts')?.view).toBe(
      result.outputs.counts
    );
    expect(suppress?.body).toContain('counts[countsOffset + index] = 0u');
  } finally {
    device.destroy();
  }
});

const CANONICAL_GUIDE_CASES = [
  [
    'gpu-network',
    'GPUMapMatchedEdgeVolumes',
    'gpu-network/map-matched-edge-volumes.md',
    '# Edge volumes of map-matched tracks',
    'output.edgeVolumes'
  ],
  [
    'gpu-network',
    'GPUNetworkPotentialPathArea',
    'gpu-network/network-time-geography.md',
    '## `GPUNetworkPotentialPathArea`',
    'forward[u] + weight + reverse[v] <= budget'
  ],
  [
    'gpu-network',
    'GPUNetworkTravelTimePercentiles',
    'gpu-network/network-time-geography.md',
    '## `GPUNetworkTravelTimePercentiles`',
    'ceil(p / 100 * n)'
  ],
  [
    'gpu-network',
    'GPURadiationFlowModel',
    'gpu-network/origin-destination-flow-models.md',
    '## `GPURadiationFlowModel`',
    'm_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))'
  ],
  [
    'gpu-network',
    'GPUKernelFlowSmoothing',
    'gpu-network/origin-destination-flow-models.md',
    '## `GPUKernelFlowSmoothing`',
    'S_ij = sum_a sum_b K_ia K_jb M_ab'
  ],
  [
    'raster-analysis',
    'GPULagrangianFlowMap',
    'gpu-raster/flow-map-and-particle-diagnostics.md',
    '## `GPULagrangianFlowMap`',
    'output.residenceTimes'
  ],
  [
    'raster-analysis',
    'GPUScalarFieldRidges',
    'gpu-raster/flow-map-and-particle-diagnostics.md',
    '## `GPUScalarFieldRidges`',
    'smallest Hessian eigenvalue'
  ],
  [
    'raster-analysis',
    'GPUFiniteTimeLyapunovExponent',
    'gpu-raster/flow-map-and-particle-diagnostics.md',
    '## `GPUFiniteTimeLyapunovExponent`',
    'FTLE = ln(sqrt(lambdaMax(J^T J))) / |T|'
  ],
  [
    'gpu-spatial-analysis',
    'GPUBrownianBridgeUtilization',
    'gpu-spatial-analysis/home-ranges-and-recursions.md',
    '## `GPUBrownianBridgeUtilization`',
    'T alpha (1 - alpha) sigma^2'
  ],
  [
    'gpu-spatial-analysis',
    'GPUUtilizationVolumeIsopleths',
    'gpu-spatial-analysis/home-ranges-and-recursions.md',
    '## `GPUUtilizationVolumeIsopleths`',
    'inside = [1, 1, 0]'
  ],
  [
    'gpu-spatial-analysis',
    'GPUTrajectoryRecursions',
    'gpu-spatial-analysis/home-ranges-and-recursions.md',
    '## `GPUTrajectoryRecursions`',
    'firstPassageCensored'
  ]
] as const;

function readNamedSection(documentation: string, heading: string): string {
  const lines = documentation.split('\n');
  const start = lines.indexOf(heading);
  expect(start).toBeGreaterThanOrEqual(0);
  const level = heading.match(/^#+/)![0].length;
  const following = lines.findIndex(
    (line, index) => index > start && new RegExp(`^#{1,${level}} `).test(line)
  );
  return lines.slice(start + 1, following < 0 ? undefined : following).join('\n');
}

function assertCanonicalSection(
  documentation: string,
  exportName: string,
  heading: string,
  fact: string
): void {
  expect(documentation.split('\n')).toContain(heading);
  expect(documentation).toContain(`\`${exportName}\``);
  expect(readNamedSection(documentation, heading)).toContain(fact);
}

it.each(
  CANONICAL_GUIDE_CASES
)('W5-03 resolves %s %s to its substantive canonical section', (entryPoint, exportName, page, heading, fact) => {
  const record = getGPUAnalysisOperator({
    package: `@luma-contributors/analysis/${entryPoint}`,
    exportName
  });
  expect(record?.evidence.documentation).toBe(`docs/api/${page}`);
  const documentation = readFileSync(`docs/api/${page}`, 'utf8');
  assertCanonicalSection(documentation, exportName, heading, fact);
});

it.each(
  CANONICAL_GUIDE_CASES
)('S4 rejects an empty canonical %s %s section', (_entryPoint, exportName, page, heading, fact) => {
  const documentation = readFileSync(`docs/api/${page}`, 'utf8');
  const body = readNamedSection(documentation, heading);
  expect(body.trim().length).toBeGreaterThan(0);
  expect(() =>
    assertCanonicalSection(documentation.replace(body, ''), exportName, heading, fact)
  ).toThrow();
});

it('W5-03 retains the five historical guide URLs in navigation', () => {
  const navigation = readFileSync('docs/table-of-contents.json', 'utf8');
  for (const page of [
    'gpu-network/map-matched-edge-volumes-mob05',
    'gpu-network/network-time-geography-mob09',
    'gpu-network/flow-models-mob10',
    'gpu-raster/lagrangian-diagnostics-mob06',
    'gpu-spatial-analysis/home-range-and-recursion-mob07'
  ])
    expect(navigation).toContain(`api/${page}`);
});

it.each([
  [
    'computation-record',
    'GPUSnapshotArena',
    'Copies frames into one bounded readback buffer, shares one in-flight read, caches successful bytes and allows retry after a failed read.'
  ],
  [
    'raster-analysis',
    'GPURasterStreamingAccumulator',
    'Accumulates per-cell, per-slot counts, fixed-point sums and moments in chunks of at most 256 slices per dispatch. Stream encoding headroom defaults to 2^24 slices; valid-observation counts saturate at 2^32 - 1 per cell and slot, with optional sticky loss evidence in state.status until reset.'
  ]
] as const)('W5-02 publishes the current %s %s summary', (entryPoint, exportName, expected) => {
  expect(
    getGPUAnalysisOperator({package: `@luma-contributors/analysis/${entryPoint}`, exportName})
      ?.summary
  ).toBe(expected);
});

it('MRG-03 names the actual entropy random variables in the qualified summary', () => {
  // H(category rows), H(source category,destination category) and
  // I(source category;destination category) are distinct from I(unordered pair;distance class).
  expect(
    getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialEntropy'
    })?.summary
  ).toBe(
    "Shannon entropy of row categories, Leibovici co-occurrence entropy of ordered endpoint-category pairs, and mutual information between source and destination categories of links in the supplied weights graph. Altieri's distance-class and unordered-pair construction is a different statistic and is not computed."
  );
});

it('W5-01 audits exact optional certificate and invalid-unit destinations without inventing query kinds', () => {
  const detectorAudit =
    AUDITED_EMPTY_OPERATOR_STATUSES[
      '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeographicalDetector'
    ];
  const segregationAudit =
    AUDITED_EMPTY_OPERATOR_STATUSES[
      '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegregation'
    ];
  expect(detectorAudit).toContain('output.labelsComplete');
  expect(detectorAudit).toContain('supplied partition');
  expect(segregationAudit).toContain('output.invalidUnitCount');
  expect(segregationAudit).toContain('negative or non-finite');
  expect(segregationAudit).toContain('weight-dependent values only at the affected scale');
  expect(segregationAudit).toContain(
    'Environment-form population diversity and output.invalidUnitCount remain readable'
  );
  for (const name of ['GPUGeographicalDetector', 'GPUSegregation']) {
    expect(
      getGPUAnalysisOperator({
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: name
      })?.status
    ).toEqual([]);
    for (const status of ['complete', 'invalidCount'] as const)
      expect(queryGPUAnalysisOperators({status}).map(record => record.exportName)).not.toContain(
        name
      );
  }
  expect(getGPUSpatialAnalysisOperator('GPUSegregation')?.status).toEqual([]);
  expect(
    queryGPUSpatialAnalysisOperators({status: 'invalidCount'}).map(record => record.exportName)
  ).not.toContain('GPUSegregation');
  const device = createNullWebGPUDevice();
  try {
    for (const supplied of [false, true]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      let sequence = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `audit-${sequence++}`, format, length);
      const labelsComplete = supplied ? view('uint32', 1) : undefined;
      const certificate = supplied ? view('uint32', 1) : undefined;
      new GPUGeographicalDetector({
        id: 'detector',
        values: view('float32', 4),
        labels: view('uint32', 4),
        labelsComplete,
        output: {
          summary: view('float32', GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length),
          labelsComplete: certificate
        }
      }).getCommandNodes(graph);
      const provenance = getWGSLKernelNodeRecord(graph, 'detector-partition-provenance');
      expect(Boolean(provenance)).toBe(supplied);
      if (supplied) {
        expect(
          provenance?.bindings.find(binding => binding.name === 'labelsComplete')?.view ===
            labelsComplete
        ).toBe(true);
        expect(
          provenance?.bindings.find(binding => binding.name === 'complete')?.view === certificate
        ).toBe(true);
        expect(provenance?.body).toContain(
          'complete[completeOffset] = u32(labelsComplete[labelsCompleteOffset] != 0u)'
        );
      }
      const invalidUnitCount = supplied ? view('uint32', 1) : undefined;
      new GPUSegregation({
        id: 'segregation',
        groupCount: 2,
        groupCounts: view('float32', 4),
        output: {indices: view('float32', getGPUSegregationLayout(2).stride), invalidUnitCount}
      }).getCommandNodes(graph);
      const sanitize = getWGSLKernelNodeRecord(graph, 'segregation-sanitize-counts');
      expect(Boolean(sanitize)).toBe(supplied);
      if (supplied) {
        expect(
          sanitize?.bindings.find(binding => binding.name === 'invalidUnitCount')?.view ===
            invalidUnitCount
        ).toBe(true);
        expect(sanitize?.body).toContain(
          'if (invalid) { atomicAdd(&invalidUnitCount[invalidUnitCountOffset], 1u); }'
        );
        expect(getWGSLKernelNodeRecord(graph, 'segregation-clear-invalid-units')?.body).toBe(
          'invalidUnitCount[invalidUnitCountOffset] = 0u;'
        );
      }
    }
  } finally {
    device.destroy();
  }
});

it('CCK-02 discovers coverage validity and binds all three cell-domain outputs', () => {
  const record = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/pointcloud-analysis',
    exportName: 'GPUPointCloudCoverageValidity'
  });
  expect(record).toMatchObject({
    primaryCapability: 'pointcloud-ground',
    inputs: ['surface'],
    outputs: ['attributes'],
    dimensionality: '2d',
    status: 'unknown',
    readback: 'unknown',
    evidence: {
      tests: 'modules/analysis/test/gpu-pointcloud/pointcloud-coverage-validity-w3.spec.ts',
      documentation: 'docs/api/pointcloud-analysis.md'
    }
  });
  expect(
    queryGPUAnalysisOperators({
      package: '@luma-contributors/analysis/pointcloud-analysis',
      input: 'surface',
      output: 'attributes'
    })
  ).toContain(record);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const coverage = createTransientView(graph, 'coverage', 'uint32', 4);
    const output = {
      surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', 4),
      terrainValidity: createTransientView(graph, 'terrain-validity', 'uint32', 4),
      canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', 4)
    };
    new GPUPointCloudCoverageValidity({coverage, output}).getCommandNodes(graph);
    const kernel = getWGSLKernelNodeRecord(graph, 'gpu-pointcloud-coverage-validity');
    expect(kernel).toBeDefined();
    for (const [name, destination] of Object.entries(output)) {
      const binding = kernel?.bindings.find(candidate => candidate.name === name);
      expect(binding?.view === destination, name).toBe(true);
      expect(binding?.access, name).toBe('read_write');
      expect(destination.length).toBe(coverage.length);
    }
    // Coverage 0,1,2,3 means surface 0,1,0,1 and ground/canopy 0,0,1,1.
    // Pin independent bit definitions in the real publisher; device numeric coverage lives in
    // pointcloud-coverage-validity-w3.spec.ts, the catalog's evidence destination.
    expect(kernel?.declarations).toContain('(coverage & 1u) != 0u');
    expect(kernel?.declarations).toContain('(coverage & 2u) != 0u');
    expect(kernel?.declarations).toContain(
      'writeTerrainValidity(index, ground); writeCanopyValidity(index, ground);'
    );
  } finally {
    device.destroy();
  }
});
