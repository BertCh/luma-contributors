// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure/gpu-network-structure';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

import {
  WGSL_RESERVED_WORDS,
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createShaderSources() {
  const unique = new Set<string>();
  for (const [nodeCount, edgeCount] of [
    [1, 0],
    [2, 2],
    [33, 64],
    [257, 514],
    [1024, 2048],
    [100000, 400000]
  ]) {
    const modes =
      nodeCount > 1024 ? (['auto', 'parallel'] as const) : (['auto', 'exact', 'parallel'] as const);
    for (const mode of modes) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      let serial = 0;
      const view = (length: number) =>
        createTransientView(graph, `shader-${serial++}`, 'uint32', length);
      const contributor = new GPUNetworkStructure({
        mode,
        strongComponentsMaximumIterations: 1,
        topology: {offsets: view(nodeCount + 1), neighbors: view(edgeCount)},
        reverseEdgeSlots: view(edgeCount),
        strongComponentsStatus: {
          converged: view(1),
          iterationCount: view(1),
          iterationLimitReached: view(1)
        },
        output: {
          strongComponents: view(nodeCount),
          bridges: view(edgeCount),
          articulationPoints: view(nodeCount),
          biconnectedComponents: view(edgeCount)
        }
      });
      for (const {source, constants} of getGPUCommandNodeKernelSources(
        contributor.getCommandNodes(graph)
      )) {
        // Bake the pipeline's supplied override constants for offline backend compilation.
        // The real-device headless checks compile the original override declarations as well.
        const specialized = source.replace(
          /override\s+(\w+)\s*:\s*(\w+)\s*;/g,
          (declaration, name: string, type: string) =>
            Object.hasOwn(constants, name)
              ? `const ${name}: ${type} = ${constants[name]}${type === 'u32' ? 'u' : ''};`
              : declaration
        );
        unique.add(specialized);
      }
    }
  }
  return [...unique];
}

it('every generated network structure shader avoids the complete WGSL reserved-word set', () => {
  for (const source of createShaderSources()) {
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  }
});

// Optional offline compiler, used in the stream's focused check. Headless tests always also
// request real device compilation diagnostics; no compiler binary is required for ordinary Node runs.
const compiler = process.env['SC_NETWORK_STRUCTURE_WGSL_COMPILER'];
it.runIf(Boolean(compiler))(
  'all generated shader variants pass native WGSL compiler validation',
  () => {
    const directory = mkdtempSync(join(tmpdir(), 'network-structure-wgsl-'));
    try {
      const paths = createShaderSources().map((source, index) => {
        const path = join(directory, `shader-${index}.wgsl`);
        writeFileSync(path, source);
        return path;
      });
      const result = spawnSync(compiler!, paths, {encoding: 'utf8'});
      expect({
        status: result.status,
        error: result.error?.message,
        output: result.stdout + result.stderr
      }).toEqual({status: 0, error: undefined, output: ''});
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  },
  120000
);

function createPipelineNodes(nodeCount: number, maximumIterations?: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = (length: number) =>
    createTransientView(graph, `pipeline-${serial++}`, 'uint32', length);
  const edgeCount = nodeCount * 2;
  return new GPUNetworkStructure({
    mode: 'parallel',
    strongComponentsMaximumIterations: maximumIterations,
    topology: {offsets: view(nodeCount + 1), neighbors: view(edgeCount)},
    reverseEdgeSlots: view(edgeCount),
    output: {
      ...(maximumIterations === undefined ? {} : {strongComponents: view(nodeCount)}),
      bridges: view(edgeCount),
      articulationPoints: view(nodeCount),
      biconnectedComponents: view(edgeCount)
    }
  }).getCommandNodes(graph);
}

it('shader and pipeline counts stay constant as resilience levels and rounds grow', () => {
  const sourceCounts: number[] = [];
  const pipelineCounts: number[] = [];
  for (const nodeCount of [513, 2049, 8193]) {
    const nodes = createPipelineNodes(nodeCount);
    sourceCounts.push(
      new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size
    );
    pipelineCounts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
    for (const idPattern of [/resilience-tree-lift-/, /resilience-segments-activate-/]) {
      expect(
        getGPUCommandNodePipelineKeys(nodes, {idPattern, includeShaderLayout: true}).size
      ).toBe(1);
    }
  }
  expect(new Set(sourceCounts).size).toBe(1);
  expect(new Set(pipelineCounts).size).toBe(1);
});

it('shader and pipeline counts stay constant as the composed SCC step budget grows', () => {
  const sourceCounts: number[] = [];
  const pipelineCounts: number[] = [];
  for (const maximumIterations of [1, 3, 32]) {
    const nodes = createPipelineNodes(513, maximumIterations);
    sourceCounts.push(
      new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size
    );
    pipelineCounts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
  }
  expect(new Set(sourceCounts).size).toBe(1);
  expect(new Set(pipelineCounts).size).toBe(1);
});

it('uses the shared WGSL reserved-word table, where sample is a valid identifier', () => {
  expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
  expect(findReservedIdentifiers('let sample = 1u;')).toEqual([]);
  expect(
    findReservedIdentifiers('let target = 1u;').some(finding => finding.check === 'reserved word')
  ).toBe(true);
});
