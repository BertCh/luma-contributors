// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPULocalSpatialHeteroscedasticity} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {
  computeSpatialWeightsAlgebraOnCPU,
  CPUSpatialWeightsAlgebraOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {computeContiguityWeightsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {decodeGPUAnalysisContiguityStatistics} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics';

function scene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `final1-${serial++}`, format, length);
  const weights = () => ({
    offsets: view('uint32', 5),
    neighbors: view('uint32', 6),
    weights: view('float32', 6)
  });
  const output = () => ({
    ...weights(),
    status: {
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  });
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 16),
    ringOffsets: view('uint32', 5),
    polygonOffsets: view('uint32', 5),
    featureOffsets: view('uint32', 5)
  });
  return {device, graph, view, weights, output, polygons};
}

it('catalog-docs:F2 LOSH reads candidate loss', () => {
  const {device, graph, view, weights} = scene();
  try {
    const nodes = new GPULocalSpatialHeteroscedasticity({
      weights: {...weights(), status: {candidateOverflow: view('uint32', 1)}},
      values: view('float32', 4),
      output: {statistic: view('float32', 4), status: view('uint32', 4)}
    }).getCommandNodes(graph);
    expect(
      getGPUCommandNodeKernelSources(nodes).some(record => record.id.includes('weights-loss'))
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('core:F3 minimum spanning forest suppresses status-bearing weights instead of certifying a missing link', () => {
  const {device, graph, view, weights} = scene();
  try {
    const producer = new GPUSpatialWeightsMinimumSpanningTree({
      weights: {...weights(), status: {candidateOverflow: view('uint32', 1)}},
      values: view('float32', 4),
      output: {treeEdgeFlags: view('uint32', 6), componentLabels: view('uint32', 4)}
    });
    expect(
      getGPUCommandNodeKernelSources(producer.getCommandNodes(graph)).some(record =>
        record.id.includes('incomplete-component-labels')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('spatial-1:F1 chunk transpose refuses chunk-local loss flags', () => {
  const {device, view, weights, output} = scene();
  try {
    expect(
      () =>
        new GPUChunkedSpatialWeightsTranspose({
          weights: {
            rowCount: 4,
            chunks: [
              {
                rowStart: 0,
                rowCount: 4,
                weights: {...weights(), status: {candidateOverflow: view('uint32', 1)}}
              }
            ]
          },
          output: output()
        })
    ).toThrow(/chunk.*status.*outer/);
  } finally {
    device.destroy();
  }
});

it('spatial-1:F2 showcase uses the complete importer on all three statistics paths and clears search candidate status', () => {
  const showcase = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation.compute.ts',
    'utf8'
  );
  const core = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-weights-core.ts',
    'utf8'
  );
  expect([...showcase.matchAll(/importWeightsCoreWeights\(/g)].length).toBe(3);
  expect(core).toContain('importOutput(graph, csr, {overflow, total, candidateOverflow}, true)');
});

for (const inherited of ['own', 'invalidCount', 'candidateOverflow'] as const)
  it(`spatial-2:F6 search rejects ${inherited} writer aliases and accepts separate words`, () => {
    const {device, graph, view, output} = scene();
    try {
      const base = {
        mode: 'radius' as const,
        gridSize: [2, 2] as const,
        positions: view('float32x2', 4),
        parameters: view('float32', 12),
        output: output()
      };
      const shared = base.output.status.candidateOverflow;
      const aliasStatus =
        inherited === 'own'
          ? {...base.output.status, invalidCount: shared}
          : Object.assign(
              Object.create({
                get [inherited]() {
                  return shared;
                }
              }),
              {
                overflow: base.output.status.overflow,
                [inherited === 'invalidCount' ? 'candidateOverflow' : 'invalidCount']: shared
              }
            );
      expect(
        () =>
          new GPUNeighborSearch({
            ...base,
            output: {
              ...base.output,
              status: aliasStatus
            }
          })
      ).toThrow(/overlap|share/);
      expect(
        () =>
          new GPUNeighborSearch({
            ...base,
            output: {
              ...base.output,
              neighborCounts: graph.createDataView(base.output.neighbors.buffer, {
                format: 'uint32',
                length: 4
              })
            }
          })
      ).toThrow(/overlap|share/);
      expect(
        () =>
          new GPUNeighborSearch({
            ...base,
            output: {
              ...base.output,
              neighborCounts: view('uint32', 4),
              status: {...base.output.status, invalidCount: view('uint32', 1)}
            }
          })
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });

it('spatial-4:F3 algebra reads operand candidate loss', () => {
  const {device, graph, view, weights, output} = scene();
  try {
    const producer = new GPUSpatialWeightsAlgebra({
      operation: 'higherOrder',
      order: 1,
      weights: {...weights(), status: {candidateOverflow: view('uint32', 1)}},
      output: output()
    });
    expect(
      getGPUCommandNodeKernelSources(producer.getCommandNodes(graph)).some(record =>
        record.id.includes('inherited-candidate')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('spatial-4:F5 lattice clears a requested candidate flag', () => {
  const {device, graph, output} = scene();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPULatticeWeights({
        width: 2,
        height: 2,
        criterion: 'rook',
        output: output()
      }).getCommandNodes(graph)
    );
    expect(
      sources.some(record =>
        /candidateOverflow\[candidateOverflowOffset(?: \+ index)?\] = 0u/.test(record.source)
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('spatial-4:F5 fuzzy contiguity clears its published candidate flag', () => {
  const {device, graph, view, output, polygons} = scene();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUFuzzyContiguity({
        polygons: polygons(),
        tolerance: 0,
        output: {...output(), featureStatus: view('uint32', 4), added: view('uint32', 6)}
      }).getCommandNodes(graph)
    );
    expect(
      sources.some(record =>
        record.source.includes('candidateOverflow[candidateOverflowOffset] = 0u')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('spatial-4:F3 algebra CPU and operation ports preserve distinct operand loss and clear on reuse', () => {
  const weights = {
    offsets: new Uint32Array([0, 1, 2]),
    neighbors: new Uint32Array([1, 0]),
    weights: new Float32Array([1, 1]),
    status: {overflow: new Uint32Array([0]), candidateOverflow: new Uint32Array([0])}
  };
  for (const [overflow, candidateOverflow] of [
    [0, 1],
    [1, 0],
    [0, 0]
  ]) {
    weights.status.overflow[0] = overflow;
    weights.status.candidateOverflow[0] = candidateOverflow;
    const result = computeSpatialWeightsAlgebraOnCPU(
      {weights},
      {operation: 'selfWeight', selfWeight: 0, capacity: 4}
    );
    expect(Array.from(result.overflow)).toEqual([0]);
    expect(Array.from(result.candidateOverflow)).toEqual([
      Number(Boolean(overflow || candidateOverflow))
    ]);
  }
  const props = {
    operation: 'selfWeight' as const,
    selfWeight: 0,
    rowCount: 2,
    capacity: 4,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'loss', candidateOverflow: 'candidateLoss'}
    },
    output: {
      offsets: 'outOffsets',
      neighbors: 'outNeighbors',
      weights: 'outWeights',
      status: {overflow: 'outLoss', candidateOverflow: 'outCandidateLoss'}
    }
  };
  const gpu = new GPUSpatialWeightsAlgebraOperation(props);
  const cpu = new CPUSpatialWeightsAlgebraOperation(props);
  expect(gpu.inputs['sourceCandidateOverflow']).toBe('candidateLoss');
  expect(cpu.contract.values?.map(({value, access}) => ({value, access}))).toEqual(
    gpu.contract.values?.map(({value, access}) => ({value, access}))
  );
  expect(cpu.contract.outputs).toEqual(gpu.contract.outputs);
});

it('spatial-4:F8 contiguity separates final overflow from pair loss', () => {
  const {device, graph, polygons, output} = scene();
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUContiguityWeights({
        criterion: 'queen',
        pairCapacity: 1,
        polygons: polygons(),
        output: output()
      }).getCommandNodes(graph)
    );
    expect(
      sources.some(record =>
        record.source.includes('total > CAPACITY || pairTotal > PAIR_CAPACITY')
      )
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('spatial-4:F8 explorer block statistics preserve the separate candidate cause across graphs', () => {
  const mode = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/spatial-weights-mode.ts',
    'utf8'
  );
  const kit = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/spatial-weights-layers.ts',
    'utf8'
  );
  expect(mode.includes('candidateOverflow: blockCandidateOverflowBuffer')).toBe(true);
  expect(kit.includes('props.producer.candidateOverflow')).toBe(true);
});

it('spatial-4:F8 polygon coloring readback retains candidate loss in its incomplete warning', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/geometry/polygon-measures.compute.ts',
    'utf8'
  );
  expect(source.includes('weightCandidateOverflow')).toBe(true);
  expect(source.includes('words[featureCount + 6]')).toBe(true);
});

it('spatial-4:F8 the CPU contiguity twin binds an explicit pair budget', () => {
  const result = computeContiguityWeightsOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1]),
      ringOffsets: new Uint32Array([0, 4, 8]),
      polygonOffsets: new Uint32Array([0, 1, 2]),
      featureOffsets: new Uint32Array([0, 1, 2])
    },
    {criterion: 'queen', capacity: 8, pairCapacity: 1} as Parameters<
      typeof computeContiguityWeightsOnCPU
    >[1]
  );
  expect(Array.from(result.candidateOverflow)).toEqual([1]);
  expect(Array.from(result.overflow)).toEqual([0]);
});

it('spatial-4:F13 chunked algebra exposes aggregate status consumable by global Moran', () => {
  const {device, view, weights, output} = scene();
  try {
    const producer = new GPUChunkedSpatialWeightsAlgebra({
      operation: 'selfWeight',
      selfWeight: 0,
      weights: {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights: weights()}]},
      output: {chunks: [{rowStart: 0, rowCount: 4, weights: output()}]}
    });
    expect(producer.outputWeights.status?.overflow).toBeDefined();
    expect(
      () =>
        new GPUGlobalSpatialStatistics({
          weights: producer.outputWeights,
          values: view('float32', 4),
          statistics: ['moran'],
          output: {summary: view('float32', 52)}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('spatial-4:F16 contiguity decoder accepts any nonzero usable validity', () => {
  expect(
    decodeGPUAnalysisContiguityStatistics({
      counts: [2, 0, 1, 1, 2, 0],
      median: [1],
      validity: [2],
      overflow: [0],
      candidateOverflow: [0]
    }).exact
  ).toBe(true);
});
