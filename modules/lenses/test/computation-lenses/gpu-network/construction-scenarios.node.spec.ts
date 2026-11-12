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

type ConstructionCase = {
  kebab: string;
  element: string;
  /** Formula recorded on a kernel node, or undefined for `formula: none` elements. */
  formula?: string;
  /** Last line of the shown example. */
  outputs: RegExp;
  rungs: string[];
};

const CASES: ConstructionCase[] = [
  {
    kebab: 'network-cost-table',
    element: 'GPUNetworkCostTable',
    formula: 'gpu-network.cost-table.admission',
    outputs: /const \{offsets, targets, costs\} = table\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'network-cost-weights',
    element: 'GPUNetworkCostWeights',
    formula: 'gpu-network.cost-weights.decay',
    outputs: /const \{offsets, neighbors, weights\} = costWeights\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'network-noding',
    element: 'GPUNetworkNoding',
    outputs: /const \{pieces, nodes, edges, topology\} = noding\.props\.output;$/,
    rungs: ['result', 'datum', 'kernel', 'graph']
  },
  {
    kebab: 'map-matching',
    element: 'GPUMapMatching',
    outputs: /const \{matchedEdges, snappedPositions, breaks\} = matching\.props\.output;$/,
    rungs: ['result', 'datum', 'kernel', 'graph']
  },
  {
    kebab: 'network-k-function',
    element: 'GPUNetworkKFunction',
    formula: 'gpu-network.k-function.k',
    outputs: /const \{kValues, pairCounts, snappedEventCount\} = kFunction\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'edge-bundling',
    element: 'GPUEdgeBundling',
    outputs: /const \{paths, lineOffsets\} = bundling\.props\.output;$/,
    rungs: ['result', 'datum', 'iterations', 'kernel', 'graph']
  },
  {
    kebab: 'flow-aggregation',
    element: 'GPUFlowAggregation',
    formula: 'gpu-network.flow-aggregation.pair-sum',
    outputs:
      /const \{flowOriginZoneIds, flowDestinationZoneIds, flowWeights\} = flows\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  }
];

for (const constructionCase of CASES) {
  const scenarioId = `gpu-network.${constructionCase.kebab}`;
  const directory = '../../../src/computation-lenses/scenarios/gpu-network';

  test(`${scenarioId} is registered with the result rung first`, () => {
    const descriptor = getLensScenario(scenarioId);
    expect(descriptor?.element).toBe(constructionCase.element);
    expect(descriptor?.audience).toBe('analyst');
    expect(descriptor?.rungs).toEqual(constructionCase.rungs);
    expect(descriptor?.defaultRung).toBe('result');
  });

  test(`${scenarioId} describes a valid structure on a null device`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const lens: {describeStructure(device: Device): Promise<GPUComputationRecord>} = await import(
        `${directory}/${constructionCase.kebab}.lens`
      );
      const record = await lens.describeStructure(device);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      expect(record.graphs[0].nodes.length).toBeGreaterThan(0);
      // Kernels built by createWGSLKernelNode carry a `wgsl:` source and their bindings even on a
      // null device; the rest are published @luma.gl/graph primitives (GPUScan, GPUGroupAggregation,
      // the gathers, and GPUSort passes, which name no operation). Formula and loop provenance stay
      // pack-level.
      const computeNodes = record.graphs[0].nodes.filter(node => node.kind === 'compute');
      const kernelNodes = computeNodes.filter(node => node.source?.startsWith('wgsl:'));
      expect(kernelNodes.length).toBeGreaterThan(0);
      for (const node of kernelNodes) expect(node.bindings?.length, node.id).toBeGreaterThan(0);
      expect(
        computeNodes
          .filter(
            node =>
              !node.source &&
              node.op !== undefined &&
              !['GPUScan', 'GPUGroupAggregation', 'GPUGather', 'GPUUint32Gather'].includes(node.op)
          )
          .map(node => node.id)
      ).toEqual([]);
      if (constructionCase.formula) {
        const formula = NETWORK_FORMULA_REFS[constructionCase.formula];
        // The element's own formula comes first so the formula rung shows it by default.
        expect(record.refs.formulas[0].id).toBe(formula.id);
      }
    } finally {
      device.destroy();
    }
  });

  test(`${scenarioId} example is short, public and ends with the displayed outputs`, () => {
    const source = readFileSync(
      new URL(`${directory}/${constructionCase.kebab}.example.ts`, import.meta.url),
      'utf8'
    );
    expect(findNonPublicImports(source)).toEqual([]);
    const example = extractLensExample(source);
    expect(example.lineCount).toBeLessThanOrEqual(25);
    expect(example.code).toMatch(constructionCase.outputs);
  });
}
