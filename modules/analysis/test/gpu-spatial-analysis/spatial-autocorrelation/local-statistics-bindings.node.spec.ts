// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPULocalMoran,
  type GPULocalMoranProps
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  type RecordedGPUKernelSource
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  findStatisticsBarrierRisks,
  findStatisticsNonFiniteConstants
} from '../spatial-weights/statistics-wgsl-audit';

function createScene(chunkSizes: readonly number[], storageBindings: number) {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBuffersPerShaderStage: storageBindings});
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `local-binding-${serial++}`, format, length);
  const rows = chunkSizes.reduce((sum, length) => sum + length, 0);
  let rowStart = 0;
  const chunks = chunkSizes.map(rowCount => {
    const chunk = {
      rowStart,
      rowCount,
      weights: {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', rowCount),
        weights: view('float32', rowCount)
      }
    };
    rowStart += rowCount;
    return chunk;
  });
  const weights = chunkSizes.length === 1 ? chunks[0].weights : {rowCount: rows, chunks};
  return {
    device,
    graph,
    view,
    rows,
    props: {
      weights,
      values: view('float32', rows),
      parameters: view('float32', 8)
    }
  };
}

function findUnusedBindings(source: string): string[] {
  // wgsl_reflect treats these legal contextual local names as language keywords.
  const reflection = new WgslReflect(
    source.replace(/\bblock\b/g, 'block_').replace(/(?<!<\s*)\bworkgroup\b(?!\s*>)/g, 'workgroup_')
  );
  const used = new Set(
    reflection.entry.compute.flatMap(entry => entry.resources.map(resource => resource.name))
  );
  return [...reflection.storage, ...reflection.uniforms]
    .filter(binding => !used.has(binding.name))
    .map(binding => binding.name);
}

it('ungated cached Moran quadrants keep every binding in the inferred layout', () => {
  const {device, graph, view, props} = createScene([4], 16);
  try {
    const nodes = new GPULocalMoran({
      ...props,
      quadrantGating: 'none',
      output: {
        zScores: view('float32', 4),
        quadrants: view('uint32', 4),
        neighborCounts: view('uint32', 4)
      }
    }).getCommandNodes(graph);
    const neighbors = getGPUCommandNodeKernelSources(nodes).find(record =>
      record.id.includes('-neighbors')
    )!;
    expect(neighbors.source).toContain('quadrants[quadrantsOffset + index] = quadrant');
    expect(findUnusedBindings(neighbors.source)).toEqual([]);
    expect(neighbors.source).not.toMatch(/var<storage, read> parameters:/);
  } finally {
    device.destroy();
  }
});

const checkedSources = new Set<string>();
function checkBindings(records: readonly RecordedGPUKernelSource[], label: string): Set<string> {
  const pipelines = new Set<string>();
  for (const record of records) {
    const hash = createHash('sha256').update(record.source).digest('hex');
    pipelines.add(`${hash}:${JSON.stringify(record.constants)}`);
    if (checkedSources.has(hash)) continue;
    expect(findUnusedBindings(record.source), `${label}: ${record.id}`).toEqual([]);
    expect(findReservedIdentifiers(stripWGSLComments(record.source)), record.id).toEqual([]);
    expect(findStatisticsNonFiniteConstants(record.source), record.id).toEqual([]);
    expect(findStatisticsBarrierRisks(record.source), record.id).toEqual([]);
    checkedSources.add(hash);
  }
  return pipelines;
}

function checkChunkWork(
  nodes: readonly GPUCommandNode[],
  prefix: string,
  chunkSizes: readonly number[]
) {
  const rowNodes = nodes.filter(node => node.id.startsWith(prefix));
  expect(rowNodes.map(node => node.workload?.maximumInvocationCount)).toEqual(chunkSizes);
  expect(rowNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
    chunkSizes.map(rows => Math.ceil(rows / 256))
  );
}

const chunkLayouts = [[769], [1, 255, 256, 257]];
const inferenceModes = [
  {quadrantGating: 'analytic', multipleTesting: 'none'},
  {quadrantGating: 'none', multipleTesting: 'none'},
  {quadrantGating: 'analytic', multipleTesting: 'benjamini-hochberg'}
] as const;

for (const storageBindings of [8, 16]) {
  it(`all Moran optional-output combinations reference their bindings at limit ${storageBindings}`, () => {
    for (const inference of inferenceModes) {
      // Six independent optional row outputs exercise every fused and separate epilogue.
      for (let outputs = 0; outputs < 64; outputs++) {
        let previousPipelines: Set<string> | undefined;
        for (const chunkSizes of chunkLayouts) {
          const {device, graph, view, rows, props} = createScene(chunkSizes, storageBindings);
          try {
            const output: GPULocalMoranProps['output'] = {
              zScores: view('float32', rows),
              localI: outputs & 1 ? view('float32', rows) : undefined,
              spatialLag: outputs & 2 ? view('float32', rows) : undefined,
              quadrants: outputs & 4 ? view('uint32', rows) : undefined,
              pValues: outputs & 8 ? view('float32', rows) : undefined,
              adjustedPValues: outputs & 16 ? view('float32', rows) : undefined,
              neighborCounts: outputs & 32 ? view('uint32', rows) : undefined,
              globalCounts: view('uint32', 2),
              globalStatistics: view('float32', 5),
              status: {invalidCount: view('uint32', 1), overflow: view('uint32', 1)}
            };
            const nodes = new GPULocalMoran({
              ...props,
              ...inference,
              mask: outputs & 1 ? view('uint32', rows) : undefined,
              islands: outputs & 2 ? 'keep' : 'exclude',
              output
            }).getCommandNodes(graph);
            checkChunkWork(nodes, 'local-moran-neighbors', chunkSizes);
            const pipelines = checkBindings(
              getGPUCommandNodeKernelSources(nodes),
              `Moran outputs ${outputs} ${JSON.stringify(inference)} chunks ${chunkSizes}`
            );
            if (previousPipelines) expect(pipelines).toEqual(previousPipelines);
            previousPipelines = pipelines;
          } finally {
            device.destroy();
          }
        }
      }
    }
  }, 60_000);
}

it('all hot-spot optional row outputs reference their bindings for packed and unequal CSR chunks', () => {
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    for (let outputs = 0; outputs < 16; outputs++) {
      let previousPipelines: Set<string> | undefined;
      for (const chunkSizes of chunkLayouts) {
        const {device, graph, view, rows, props} = createScene(chunkSizes, 8);
        try {
          const nodes = new GPUHotSpotAnalysis({
            ...props,
            multipleTesting,
            mask: outputs & 1 ? view('uint32', rows) : undefined,
            output: {
              zScores: view('float32', rows),
              bins: outputs & 1 ? view('sint32', rows) : undefined,
              pValues: outputs & 2 ? view('float32', rows) : undefined,
              adjustedPValues: outputs & 4 ? view('float32', rows) : undefined,
              neighborCounts: outputs & 8 ? view('uint32', rows) : undefined,
              globalCounts: view('uint32', 2),
              globalStatistics: view('float32', 5),
              status: {invalidCount: view('uint32', 1), overflow: view('uint32', 1)}
            }
          }).getCommandNodes(graph);
          checkChunkWork(nodes, 'hot-spot-analysis-neighbors', chunkSizes);
          const pipelines = checkBindings(
            getGPUCommandNodeKernelSources(nodes),
            `hot-spot outputs ${outputs} ${multipleTesting} chunks ${chunkSizes}`
          );
          if (previousPipelines) expect(pipelines).toEqual(previousPipelines);
          previousPipelines = pipelines;
        } finally {
          device.destroy();
        }
      }
    }
  }
}, 60_000);

it('Geary optional summary and status bindings remain referenced in every CSR layout', () => {
  for (let outputs = 0; outputs < 8; outputs++) {
    let previousPipelines: Set<string> | undefined;
    for (const chunkSizes of chunkLayouts) {
      const {device, graph, view, rows, props} = createScene(chunkSizes, 8);
      try {
        const nodes = new GPULocalGeary({
          ...props,
          mask: outputs & 1 ? view('uint32', rows) : undefined,
          output: {
            statistic: view('float32', rows),
            globalCounts: outputs & 1 ? view('uint32', 2) : undefined,
            globalStatistics: outputs & 2 ? view('float32', 5) : undefined,
            status:
              outputs & 4
                ? {invalidCount: view('uint32', 1), overflow: view('uint32', 1)}
                : undefined
          }
        }).getCommandNodes(graph);
        checkChunkWork(nodes, 'local-geary-statistic', chunkSizes);
        const pipelines = checkBindings(
          getGPUCommandNodeKernelSources(nodes),
          `Geary outputs ${outputs} chunks ${chunkSizes}`
        );
        if (previousPipelines) expect(pipelines).toEqual(previousPipelines);
        previousPipelines = pipelines;
      } finally {
        device.destroy();
      }
    }
  }
});
