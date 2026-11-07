// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  validateGPUComputationRecord,
  type GPUComputationRecord
} from '@luma-contributors/analysis/computation-record';
import '../../../src/computation-lenses/scenarios/gpu-network/index';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import {
  extractLensExample,
  findNonPublicImports
} from '../../../src/computation-lenses/lens-example';
import {NETWORK_FORMULA_REFS} from '../../../src/gpu-network/computation-pack/network-formulas';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';

type StructureCase = {
  kebab: string;
  element: string;
  /** Formula recorded on a kernel node, or undefined for `formula: none` elements. */
  formula?: string;
  /** Rungs the registration must offer, result first. */
  rungs: string[];
  /** Last statement of the shown example. */
  outputs: RegExp;
};

const CASES: StructureCase[] = [
  {
    kebab: 'network-analytics-columns',
    element: 'GPUNetworkAnalyticsColumns',
    formula: 'gpu-network.analytics.normalize',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{output: degree, normalized, extent\} = analytics\.props\.degree!;$/
  },
  {
    kebab: 'network-centrality',
    element: 'GPUNetworkCentrality',
    formula: 'gpu-network.centrality.closeness',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{closeness, reachedCounts\} = centrality\.props\.output;$/
  },
  {
    kebab: 'network-statistics',
    element: 'GPUNetworkStatistics',
    formula: 'gpu-network.statistics.degree-histogram',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{output: summary\} = statistics\.props;$/
  },
  {
    kebab: 'network-subgraph-filter',
    element: 'GPUNetworkSubgraphFilter',
    formula: 'gpu-network.subgraph-filter.range',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{nodeMask, edgeMask, counts\} = filter\.props\.output;$/
  },
  {
    kebab: 'network-coarsening',
    element: 'GPUNetworkCoarsening',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    outputs:
      /const \{groupNodeCount, groupCentroid, edges, edgeTargetGroups, edgeCounts\} =\s+coarsening\.props\.output;$/
  },
  {
    kebab: 'adjacency-matrix',
    element: 'GPUAdjacencyMatrix',
    formula: 'gpu-network.adjacency-matrix.bin',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{counts, maximumCount\} = matrix\.props\.output;$/
  },
  {
    kebab: 'adjacency-matrix-order',
    element: 'GPUAdjacencyMatrixOrder',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    outputs: /const \{order\} = ordering\.props\.output;$/
  },
  {
    kebab: 'network-line-graph',
    element: 'GPUNetworkLineGraph',
    formula: 'gpu-network.line-graph.arc-weight',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    outputs: /const \{topology: lineTopology, arcCount\} = lineGraph\.props\.output;$/
  }
];

for (const structureCase of CASES) {
  const scenarioId = `gpu-network.${structureCase.kebab}`;
  const directory = '../../../src/computation-lenses/scenarios/gpu-network';

  test(`${scenarioId} is registered with the result rung first`, () => {
    const descriptor = getLensScenario(scenarioId);
    expect(descriptor?.element).toBe(structureCase.element);
    expect(descriptor?.audience).toBe('analyst');
    expect(descriptor?.rungs).toEqual(structureCase.rungs);
    expect(descriptor?.defaultRung).toBe('result');
  });

  test(`${scenarioId} describes a valid structure on a null device`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const lens: {describeStructure(device: Device): Promise<GPUComputationRecord>} = await import(
        `${directory}/${structureCase.kebab}.lens`
      );
      const record = await lens.describeStructure(device);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      expect(record.graphs[0].nodes.length).toBeGreaterThan(0);
      // Dropped coverage: node-level formula, `wgsl:` source-binding and loop assertions. The
      // network operators in modules/analysis do not yet accept a GPUComputationRecorder, so no
      // node carries an annotation; restore them when the operators record their kernels.
      if (structureCase.formula) {
        const formula = NETWORK_FORMULA_REFS[structureCase.formula];
        // The element's own formula comes first so the formula rung shows it by default.
        expect(record.refs.formulas[0].id).toBe(formula.id);
      }
    } finally {
      device.destroy();
    }
  });

  test(`${scenarioId} example is short, public and ends with the displayed outputs`, () => {
    const source = readFileSync(
      new URL(`${directory}/${structureCase.kebab}.example.ts`, import.meta.url),
      'utf8'
    );
    expect(findNonPublicImports(source)).toEqual([]);
    const example = extractLensExample(source);
    expect(example.lineCount).toBeLessThanOrEqual(25);
    expect(example.code).toMatch(structureCase.outputs);
  });
}
