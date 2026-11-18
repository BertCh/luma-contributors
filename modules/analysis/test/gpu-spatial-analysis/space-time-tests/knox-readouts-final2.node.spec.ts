// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import typeScript from 'typescript';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  getKnoxPoissonPValue,
  GPU_SPACE_TIME_SUMMARY
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {createSpatialWeightsSuppressionNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-suppression';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const CLUSTERS =
  'examples/deck/spatial-analysis-showcase/src/scenes/time/space-time-clusters.compute.ts';
const QUAKES =
  'examples/deck/spatial-analysis-showcase/src/scenes/time/quake-aftershocks.compute.ts';
const EXPLORER = 'examples/deck/spatial-analysis-explorer/modes/knox-mode.ts';

function loadFunction(
  path: string,
  name: string,
  bindings: Record<string, unknown> = {}
): (...args: unknown[]) => unknown {
  const tree = typeScript.createSourceFile(
    path,
    readFileSync(path, 'utf8'),
    typeScript.ScriptTarget.Latest,
    true
  );
  let declaration: typeScript.FunctionDeclaration | undefined;
  const visit = (node: typeScript.Node) => {
    if (typeScript.isFunctionDeclaration(node) && node.name?.text === name) declaration = node;
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  if (!declaration) throw new Error(`Missing ${name} in ${path}`);
  const code = typeScript.transpileModule(
    `return (${declaration.getText(tree).replace(/^export\s+/, '')});`,
    {compilerOptions: {target: typeScript.ScriptTarget.ES2022}}
  ).outputText;
  return new Function(...Object.keys(bindings), code)(...Object.values(bindings));
}

function readSummary(loss: number): Float32Array {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const flag = createTransientView(graph, 'loss', 'uint32', 1);
    const summary = createTransientView(graph, 'summary', 'float32', 12);
    const weights = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
      weights: createTransientView(graph, 'weights', 'float32', 3),
      status: {candidateOverflow: flag}
    };
    const memory = createCPUGraphMemory();
    const values = new Float32Array(12).fill(1);
    memory.write(flag, Uint32Array.of(loss));
    memory.write(summary, values);
    executeCommandNodesOnCPU(
      createSpatialWeightsSuppressionNodes(graph, 'knox', 'GPUKnoxTest', weights, {summary}),
      memory
    );
    return memory.readFloat32(summary).slice();
  } finally {
    device.destroy();
  }
}

it.each([
  CLUSTERS,
  QUAKES,
  EXPLORER
])('G1 %s never renders a suppressed Poisson result as significant', path => {
  const format = loadFunction(path, 'formatProbability');
  const suppressed = readSummary(1);
  expect(
    format(
      getKnoxPoissonPValue(
        suppressed[GPU_SPACE_TIME_SUMMARY.observed],
        suppressed[GPU_SPACE_TIME_SUMMARY.expected]
      )
    )
  ).toBe('unavailable');
  const clean = readSummary(0);
  expect(
    format(
      getKnoxPoissonPValue(
        clean[GPU_SPACE_TIME_SUMMARY.observed],
        clean[GPU_SPACE_TIME_SUMMARY.expected]
      )
    )
  ).toBe('0.632');
});

it('G1 distribution readouts suppress unavailable observations and recover', () => {
  const clusters = loadFunction(CLUSTERS, 'formatDistribution');
  const explorer = loadFunction(EXPLORER, 'formatHistogram', {HISTOGRAM_BINS: 32});
  const suppressed = readSummary(1)[GPU_SPACE_TIME_SUMMARY.observed];
  expect(clusters([0, 1, 2], 3, suppressed)).toBe('unavailable');
  expect(explorer([0, 1, 2], 3, suppressed)).toBe('unavailable');
  expect(clusters([0, 1, 2], 3, 1)).toContain('33.3%');
  expect(explorer([0, 1, 2], 3, 1)).toContain('observed');
});

it('G1 quake readback clears the histogram on loss and restores a finite control', () => {
  const readouts = new Map<string, unknown>();
  const charts = new Map<string, {markers: unknown[]; description: string}>();
  const show = loadFunction(QUAKES, 'showKnox', {
    GPU_SPACE_TIME_SUMMARY_LENGTH: 12,
    SUMMARY: GPU_SPACE_TIME_SUMMARY,
    MAXIMUM_PERMUTATIONS: 3,
    HISTOGRAM_BINS: 32,
    ctx: {
      options: {permutations: 3},
      setReadout: (name: string, value: unknown) => readouts.set(name, value),
      setChart: (name: string, value: {markers: unknown[]; description: string}) =>
        charts.set(name, value),
      requestLayers() {}
    },
    formatNumber: loadFunction(QUAKES, 'formatNumber'),
    formatProbability: loadFunction(QUAKES, 'formatProbability'),
    formatCount: loadFunction('examples/deck/spatial-analysis-shared/resources.ts', 'formatCount'),
    getKnoxPoissonPValue
  });
  for (const loss of [1, 0]) {
    const bytes = new ArrayBuffer((12 + 1 + 4) * 4);
    new Float32Array(bytes, 0, 12).set(readSummary(loss));
    new Uint32Array(bytes, 13 * 4).set([1, 0, 1, 2]);
    show(bytes);
    expect(readouts.get('knoxPoisson')).toBe(loss ? 'unavailable' : '0.632');
    expect(charts.get('knoxHistogram')!.markers.length).toBe(loss ? 0 : 1);
    if (loss) expect(charts.get('knoxHistogram')!.description).toMatch(/unavailable/i);
  }
});
