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

import {getRenderedDocTableRows} from './doc-heading-sections.test-utils';
import {findRoadmapRows, getIdentityTokens, getRoadmapRole} from './roadmap-rows.test-utils';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');

function readRepositoryFile(path: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, path), 'utf8');
}

// Prose assertions in this file are editorial reconciliation pins. Only the constructor and
// continuation executions qualify runtime behavior; prose alone cannot qualify a release gate.

function readRoadmapRow(identity: string, section: string, role = 'implementation'): string {
  const rows = findRoadmapRows(readRepositoryFile('dev-docs/roadmap.md'), identity, section, role);
  expect(rows, `exact roadmap row ${section}/${role}/${identity}`).toHaveLength(1);
  return rows[0].text;
}

function getMissingPriorityIdentities(markdown: string): string[] {
  const rows = getRenderedDocTableRows(markdown).filter(row => row.headers[0] === 'ID');
  const indexed = new Set(
    rows
      .filter(row => getRoadmapRole(row) === 'summary')
      .flatMap(row => getIdentityTokens(row.cells[0]))
  );
  return rows
    .filter(row => {
      if (getRoadmapRole(row) !== 'implementation') return false;
      const priorityIndex = row.headers.indexOf('Priority');
      // KI is an explicit active pass with no Priority column; all its exact records need indexing.
      return priorityIndex >= 0
        ? /^H(?:\s|$)/.test(row.cells[priorityIndex])
        : row.sections.some(section => /^Known-issues pass .*\(KI\)$/.test(section));
    })
    .flatMap(row => getIdentityTokens(row.cells[0]).filter(identity => !indexed.has(identity)));
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
  const row = readRoadmapRow('GPU-13', 'Contributor contracts and GPU-scale designs');
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
  const row = readRoadmapRow('IN05 / IN02', 'Interpolation and simulation');
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
  const row = readRoadmapRow(
    'AX-40',
    'Analysis experience (AX, implementation pass landed 2026-11-15)'
  );
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
  const row = readRoadmapRow(
    'AX-20',
    'Analysis experience (AX, implementation pass landed 2026-11-15)'
  );
  for (const phrase of ['1,000,000', '8 columns', 'same frame', '1 s', '10 s', 'progress']) {
    expect(row).toContain(phrase);
  }
  expect(row).toContain('dataset-profile-gpu.spec.ts');
  const residentRows = findRoadmapRows(
    readRepositoryFile('dev-docs/roadmap.md'),
    'AX-02',
    'Analysis experience (AX, implementation pass landed 2026-11-15)',
    'implementation'
  );
  expect(residentRows.length).toBeLessThanOrEqual(1);
  const residentRow = residentRows[0]?.text;
  if (residentRow) {
    expect(residentRow).not.toMatch(/million|latency|threshold/);
    expect(residentRow).toContain('resident');
  }
});

it('R5 describes simple holed tessellation and budgeted CSR metrics as shipped routes', () => {
  const tessellation = readRoadmapRow(
    'Urban tessellation',
    'Known limits recorded by finished passes',
    'limit'
  );
  expect(tessellation).toContain('simple');
  expect(tessellation).toContain('holed');
  expect(tessellation).toContain('32');
  expect(tessellation.toLowerCase()).toContain('dissolved');
  const visibility = readRoadmapRow(
    'Visibility graph',
    'Known limits recorded by finished passes',
    'limit'
  );
  for (const phrase of [
    'integration',
    'normalized choice',
    'all-source',
    'budget',
    'approximation'
  ]) {
    expect(visibility).toContain(phrase);
  }
  const backlog = readRoadmapRow('GPU-11', 'Contributor contracts and GPU-scale designs');
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
  const row = readRoadmapRow('PLAT-04', 'Spatiotemporal and platform');
  expect(row).toContain('GPU-gated batches');
  expect(row).toContain('one convergence readback per batch');
  expect(row).toContain('host chooses');
  expect(row).not.toContain('without host readback/submission');
});

it('R7 scopes the missing size floor to direct REDCAP and preserves the SSD-route floor', () => {
  const row = readRoadmapRow('STX-08', 'Spatiotemporal and platform');
  expect(row).toContain('GPURedcapSingleLinkage');
  expect(row).toContain('minimumRegionSize');
  expect(row).toContain('both sides');
  expect(row).toContain('direct contributor');
  expect(row).toContain('row-count');
});

// Editorial prose pins below document intended migration guidance; they are not runtime qualification.
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

it('R9 indexes every high-priority domain and LR row plus each active KI record', () => {
  expect(getMissingPriorityIdentities(readRepositoryFile('dev-docs/roadmap.md'))).toEqual([]);
});

it('ROAD-112 resolves exact compound IDs by rendered section and role, with explicit ranges', () => {
  const markdown = `# Roadmap
## High-priority summary
| ID | Still open |
| --- | --- |
| AX-40, IN05, LR-04, KI-01 to KI-02 | Index |
## Domain
| Name | Item | Priority |
| --- | --- | --- |
| AX-40 | Unrelated table | H |

| ID | Item | Priority |
| --- | --- | --- |
| AX-400 | Prefix collision | H |
| AX-40 | Implementation | H |
| IN05 / IN02 | Both needed | H (upstream) |
### Decisions for the maintainer
| ID | Item | Priority |
| --- | --- | --- |
| AX-40 | Decision | Decision |
## Library review (LR)
| ID | Item | Priority |
| --- | --- | --- |
| LR-04 | Release | H |
## Known-issues pass 2026-11-18 (KI)
| ID | Item | Plan |
| --- | --- | --- |
| KI-01 | Audit | None |
| KI-02 | Fix | None |
## Other
| ID | Item |
| --- | --- |
| LOW-01 | No priority column |
`;
  expect(
    findRoadmapRows(markdown, 'AX-40', 'Domain', 'implementation').map(row => row.cells[1])
  ).toEqual(['Implementation']);
  expect(findRoadmapRows(markdown, 'AX-40', 'Domain', 'decision').map(row => row.cells[1])).toEqual(
    ['Decision']
  );
  expect(findRoadmapRows(markdown, 'IN05 / IN03', 'Domain', 'implementation')).toEqual([]);
  expect(getIdentityTokens('KI-01 to KI-03')).toEqual(['KI-01', 'KI-02', 'KI-03']);
  expect(getMissingPriorityIdentities(markdown)).toEqual(['AX-400', 'IN02']);
  expect(
    getMissingPriorityIdentities(
      markdown.replace('AX-40, IN05, LR-04, KI-01 to KI-02', 'AX-40, IN05')
    )
  ).toEqual(['AX-400', 'IN02', 'LR-04', 'KI-01', 'KI-02']);
  expect(getMissingPriorityIdentities(markdown)).not.toContain('LOW-01');
});

it('REVIEW S2 rejects an extra compound ID member for exact row lookup', () => {
  const markdown =
    '# Roadmap\n\n## Domain\n\n| ID | Item | Priority |\n| --- | --- | --- |\n| IN05 / IN02 / IN99 | Extra member | H |\n';
  expect(findRoadmapRows(markdown, 'IN05 / IN02', 'Domain', 'implementation')).toEqual([]);
  expect(findRoadmapRows(markdown, 'IN99 / IN02 / IN05', 'Domain', 'implementation')).toHaveLength(
    1
  );
});

it.each([
  ['NW06', 'X04'],
  ['raster-fields-05', 'STD-06']
])('review B2 preserves both components of %s / %s', (first, second) => {
  const markdown = `# Roadmap
## Domain
| ID | Item | Priority |
| --- | --- | --- |
| ${first} / ${second} | Compound record | H |
`;
  expect(findRoadmapRows(markdown, first, 'Domain', 'implementation')).toEqual([]);
  expect(findRoadmapRows(markdown, second, 'Domain', 'implementation')).toEqual([]);
  expect(
    findRoadmapRows(markdown, `${second} / ${first}`, 'Domain', 'implementation').map(
      row => row.cells[1]
    )
  ).toEqual(['Compound record']);
});
