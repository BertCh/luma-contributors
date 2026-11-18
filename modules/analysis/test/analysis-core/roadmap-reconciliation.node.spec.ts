// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUClassBreaks} from '../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterLength} from '../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPULocalKriging} from '../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {runCentralityContinuationPhases} from '../../src/gpu-network/network-centrality/network-centrality-continuation';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');

function readRepositoryFile(path: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, path), 'utf8');
}

function readRoadmapRow(identity: string): string {
  const row = readRepositoryFile('dev-docs/roadmap.md')
    .split('\n')
    .find(line => line.startsWith(`| ${identity} |`));
  expect(row, `missing roadmap row ${identity}`).toBeDefined();
  return row!;
}

function withGraph(action: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  try {
    action(new GPUCommandGraph(device));
  } finally {
    device.destroy();
  }
}

it('R1 retains the rejecting class-breaks boundary and distinguishes quantile rank routes', () => {
  for (const rowCount of [16_777_216, 16_777_217]) {
    withGraph(graph => {
      const construct = () =>
        new GPUClassBreaks({
          values: createTransientView(graph, 'values', 'float32', rowCount),
          methods: ['quantile'],
          maximumClassCount: 2,
          parameters: createTransientView(
            graph,
            'parameters',
            'float32',
            getGPUClassBreaksParameterLength(2)
          ),
          output: {
            breaks: createTransientView(graph, 'breaks', 'float32', 3),
            classCount: createTransientView(graph, 'class-count', 'uint32', 1),
            classCounts: createTransientView(graph, 'class-counts', 'uint32', 2)
          }
        });
      if (rowCount === 16_777_216) expect(construct).not.toThrow();
      else expect(construct).toThrow(/moment counts and quantile ranks are held in float32/);
    });
  }
  const row = readRoadmapRow('GPU-13');
  expect(row).toContain('GPUClassBreaks');
  expect(row).toContain('2^24');
  expect(row).toContain('GPUColumnQuantiles');
  expect(row).toContain('rationalProbabilities');
  expect(row).toContain('2^31 - 1');
  expect(row).toContain('filter');
});

it('R2 retains the local-system bound and usable-sample indexing acceptance cases', () => {
  for (const neighborCount of [32, 33]) {
    withGraph(graph => {
      const construct = () =>
        new GPULocalKriging({
          positions: createTransientView(graph, 'positions', 'float32x2', 33),
          observations: createTransientView(graph, 'observations', 'float32x4', 33),
          supports: createTransientView(graph, 'supports', 'float32x2', 1),
          neighborCount,
          mode: 'ordinary',
          covariance: {model: 'exponential', sill: 1, range: 2},
          output: {
            values: createTransientView(graph, 'results', 'float32x4', 1),
            status: createTransientView(graph, 'status', 'uint32', 1)
          }
        });
      if (neighborCount === 32) expect(construct).not.toThrow();
      else expect(construct).toThrow(/neighborCount.*32/);
    });
  }
  const row = readRoadmapRow('IN05 / IN02');
  expect(row).toContain('local kriging');
  expect(row).toContain('32');
  expect(row).toContain('automatic RBF indexing');
  expect(row).toContain('nonfinite');
  expect(row).toContain('usable');
});

it('R3 keeps a fast table preview from closing any AX-40 map gate', () => {
  const script = readRepositoryFile(
    'examples/deck/kepler-compute/scripts/ax-analysis-sheet-qualification.mjs'
  );
  expect(script).toContain('mapLatencyMilliseconds: null');
  expect(script).toContain('Retained table preview measured');
  expect(script).toContain('qualified: false');
  const row = readRoadmapRow('AX-40');
  expect(row).toContain('including kernel density');
  expect(row).toContain('500 ms');
  expect(row).toContain('map update');
  expect(row).toContain('table preview alone');
});

it('R4 assigns the million-row fixture and established thresholds only to AX-20', () => {
  const script = readRepositoryFile(
    'examples/deck/kepler-compute/scripts/ax-profile-qualification.mjs'
  );
  expect(script).toContain("address.searchParams.set('rows', '256')");
  expect(script).toContain('length: 24');
  const fixture = readRepositoryFile('modules/kepler/test/profile/dataset-profile-gpu.spec.ts');
  expect(fixture).toContain('rowCount = 1_000_000');
  expect(fixture).toContain('metadata(rowCount, 8)');
  const row = readRoadmapRow('AX-20');
  for (const phrase of ['1,000,000', '8 columns', 'same frame', '1 s', '10 s', 'progress']) {
    expect(row).toContain(phrase);
  }
  expect(row).toContain('dataset-profile-gpu.spec.ts');
  const residentRow = readRepositoryFile('dev-docs/roadmap.md')
    .split('\n')
    .find(line => line.startsWith('| AX-02 |'));
  if (residentRow) {
    expect(residentRow).not.toMatch(/million|latency|threshold/);
    expect(residentRow).toContain('resident');
  }
});

it('R5 describes simple holed tessellation and budgeted CSR metrics as shipped routes', () => {
  const tessellation = readRoadmapRow('Urban tessellation');
  expect(tessellation).toContain('simple');
  expect(tessellation).toContain('holed');
  expect(tessellation).toContain('32');
  expect(tessellation.toLowerCase()).toContain('dissolved');
  const visibility = readRoadmapRow('Visibility graph');
  for (const phrase of [
    'integration',
    'normalized choice',
    'all-source',
    'budget',
    'approximation'
  ]) {
    expect(visibility).toContain(phrase);
  }
  const backlog = readRoadmapRow('GPU-11');
  expect(backlog).toContain('simple');
  expect(backlog).toContain('budget');
  expect(backlog).not.toContain('lift 64-building/256-point');
});

it('R6 permits convergence in the second host-controlled GPU-gated batch', async () => {
  const executions: {roundCount: number; carry: boolean}[] = [];
  const completed = await runCentralityContinuationPhases({
    maximumIterations: 12,
    roundsPerStep: 4,
    hasBetweenness: false,
    initialize: () => {},
    async execute(_phase, roundCount, carry) {
      executions.push({roundCount, carry});
      return executions.length === 1 ? 1 : 0;
    }
  });
  expect(completed).toEqual([true, true, true]);
  expect(executions).toEqual([
    {roundCount: 4, carry: false},
    {roundCount: 4, carry: true}
  ]);
  const proposal = readRepositoryFile('dev-docs/upstream-proposals.md');
  expect(proposal).toContain('one convergence readback per batch');
  const row = readRoadmapRow('PLAT-04');
  expect(row).toContain('GPU-gated batches');
  expect(row).toContain('one convergence readback per batch');
  expect(row).toContain('host chooses');
  expect(row).not.toContain('without host readback/submission');
});

it('R7 scopes the missing size floor to direct REDCAP and preserves the SSD-route floor', () => {
  const row = readRoadmapRow('STX-08');
  expect(row).toContain('GPURedcapSingleLinkage');
  expect(row).toContain('minimumRegionSize');
  expect(row).toContain('both sides');
  expect(row).toContain('direct contributor');
  expect(row).toContain('row-count');
});

it('R8 preserves the B1/B2 migration contracts in the BL-PASS completion record', () => {
  const completion = readRepositoryFile('dev-docs/roadmap.md')
    .split('## Completed passes')[1]
    .split('\n')
    .find(line => line.includes('BL-PASS:'));
  expect(completion).toBeDefined();
  for (const phrase of [
    'GPUNetworkFacilityOptimization',
    'GPUNetworkRationalAccess',
    'output.status',
    'output.summary',
    'groupOverflow',
    'vertexOverflow',
    'boolean',
    'GPURasterReprojection',
    'destroy()',
    'same-CRS',
    'eight words',
    'GPUMarkedPointStatistics',
    'brute-force'
  ]) {
    expect(completion, `missing migration record: ${phrase}`).toContain(phrase);
  }
});

it('R9 indexes every high-priority domain row including qualified upstream priorities', () => {
  const roadmap = readRepositoryFile('dev-docs/roadmap.md');
  const summary = roadmap.split('## High-priority summary')[1].split('## Earlier')[0];
  const domain = roadmap.split('## Open backlog by domain')[1].split('## Known limits')[0];
  const missing: string[] = [];
  for (const line of domain.split('\n')) {
    const cells = line
      .split('|')
      .slice(1, -1)
      .map(cell => cell.trim());
    if (cells.length !== 5 || !/^H(?:\s|$)/.test(cells[3])) continue;
    const identities = cells[0].split(/\s*\/\s*/);
    if (!identities.some(identity => summary.includes(identity))) missing.push(cells[0]);
  }
  expect(missing).toEqual([]);
});
