// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUGraph,
  gpuGraph,
  type GraphDataView,
  type GPUCPUVector
} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {
  computeSkaterRegionsOnCPU,
  CPUSkaterRegionsOperation,
  type CPUSkaterRegionsInput
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getCPUVectorRows, makeCPUVector} from '../../utils/mixed-execution-test-utils';

const CHAIN: CPUSkaterRegionsInput = {
  offsets: Uint32Array.of(0, 1, 3, 4),
  neighbors: Uint32Array.of(1, 0, 2, 1),
  weights: Float32Array.of(1, 1, 1, 1),
  treeEdgeFlags: Uint32Array.of(1, 0, 1, 0),
  componentLabels: Uint32Array.of(0, 0, 0),
  values: Float32Array.of(1, 2, 4),
  parameters: Uint32Array.of(1, 1)
};

for (const flagName of ['overflow', 'candidateOverflow'] as const) {
  it(`SKATER CPU suppresses a lost chain link after ${flagName} and recovers`, () => {
    const status = {[flagName]: Uint32Array.of(0)};
    const options = {maximumRegionCount: 3};
    const complete = computeSkaterRegionsOnCPU(CHAIN, options);
    for (const incomplete of [0, 1, 0]) {
      status[flagName][0] = incomplete;
      const result = computeSkaterRegionsOnCPU(
        {
          ...CHAIN,
          ...(incomplete
            ? {
                offsets: Uint32Array.of(0, 1, 2, 2),
                treeEdgeFlags: Uint32Array.of(1, 0, 0, 0),
                componentLabels: Uint32Array.of(0, 0, 2)
              }
            : {}),
          status
        },
        options
      );
      if (!incomplete) expect(result).toEqual(complete);
      else {
        expect(Array.from(result.labels)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
        expect(Array.from(result.regionCount)).toEqual([0]);
        expect(Array.from(result.cutEdges)).toEqual([0xffffffff, 0xffffffff]);
        expect(Array.from(result.cutGains).every(Number.isNaN)).toBe(true);
        // The GPU forest supplies suppressed labels; these are unavailable rather than malformed.
        expect(
          computeSkaterRegionsOnCPU(
            {...CHAIN, status, componentLabels: new Uint32Array(3).fill(0xffffffff)},
            options
          )
        ).toEqual(result);
      }
    }
  });
}

for (const flagName of ['overflow', 'candidateOverflow'] as const)
  it(`SKATER CPU operation recovers from ${flagName} on one compiled graph`, async () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUGraph({id: `cpu-skater-reuse-${flagName}`});
    const props: SkaterRegionsOperationProps = {
      rowCount: 3,
      edgeCount: 4,
      maximumRegionCount: 3,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {[flagName]: 'loss'}
      },
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      values: 'values',
      parameters: 'parameters',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const arrays = {
      offsets: CHAIN.offsets,
      neighbors: CHAIN.neighbors,
      weights: CHAIN.weights,
      treeEdgeFlags: CHAIN.treeEdgeFlags,
      componentLabels: CHAIN.componentLabels,
      values: CHAIN.values,
      parameters: CHAIN.parameters,
      loss: Uint32Array.of(0)
    };
    for (const [name, values] of Object.entries(arrays))
      graph.addInput(
        name,
        gpuGraph.vector(values instanceof Float32Array ? 'float32' : 'uint32', {
          length: values.length
        })
      );
    graph.add(new CPUSkaterRegionsOperation(props));
    const outputs = ['labels', 'regionCount', 'cutEdges', 'cutGains'];
    const compiled = graph.compile<Record<string, CPUVector>>(device, {
      outputs,
      outputLocations: Object.fromEntries(outputs.map(name => [name, 'cpu' as const]))
    });
    try {
      for (const incomplete of [0, 1, 0]) {
        const current = {
          ...arrays,
          loss: Uint32Array.of(incomplete),
          offsets: Uint32Array.from(incomplete ? [0, 1, 2, 2] : [0, 1, 3, 4]),
          treeEdgeFlags: Uint32Array.from(incomplete ? [0, 0, 0, 0] : [1, 0, 1, 0]),
          componentLabels: Uint32Array.from(
            incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
          )
        };
        const result = await compiled.run({
          inputs: Object.fromEntries(
            Object.entries(current).map(([name, values]) => [
              name,
              makeCPUVector(values instanceof Float32Array ? 'float32' : 'uint32', values)
            ])
          )
        });
        expect(getCPUVectorRows(result['labels'])).toEqual(
          Uint32Array.from(incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0])
        );
        expect(getCPUVectorRows(result['regionCount'])).toEqual(Uint32Array.of(incomplete ? 0 : 1));
        expect(getCPUVectorRows(result['cutEdges'])).toEqual(
          Uint32Array.of(0xffffffff, 0xffffffff)
        );
        expect(getCPUVectorRows(result['cutGains'])).toEqual(
          Float32Array.from(incomplete ? [NaN, NaN] : [0, 0])
        );
      }
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });

it('SKATER operations declare and lower both weights loss ports without dropping them', async () => {
  const props: SkaterRegionsOperationProps = {
    rowCount: 3,
    edgeCount: 4,
    maximumRegionCount: 3,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'final-loss', candidateOverflow: 'candidate-loss'}
    },
    treeEdgeFlags: 'treeEdgeFlags',
    componentLabels: 'componentLabels',
    values: 'values',
    parameters: 'parameters',
    output: {
      labels: 'labels',
      regionCount: 'regionCount',
      cutEdges: 'cutEdges',
      cutGains: 'cutGains'
    }
  };
  const gpu = new GPUSkaterRegionsOperation(props);
  const cpu = new CPUSkaterRegionsOperation(props);
  props.weights.status!.overflow = 'mutated-final-loss';
  props.weights.status!.candidateOverflow = 'mutated-candidate-loss';
  for (const operation of [gpu, cpu]) {
    expect(operation.contract.values?.map(value => value.value)).toContain('final-loss');
    expect(operation.contract.values?.map(value => value.value)).toContain('candidate-loss');
    expect(operation.contract.values?.map(value => value.value)).not.toContain(
      'mutated-final-loss'
    );
  }
  const vector = (values: Uint32Array | Float32Array): GPUCPUVector => ({
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const inputs = Object.fromEntries(
    Object.entries({
      offsets: CHAIN.offsets,
      neighbors: CHAIN.neighbors,
      weights: CHAIN.weights,
      treeEdgeFlags: CHAIN.treeEdgeFlags,
      componentLabels: CHAIN.componentLabels,
      values: CHAIN.values,
      parameters: CHAIN.parameters
    }).map(([port, values]) => [port, vector(values)])
  );
  for (const flagName of ['weightsOverflow', 'weightsCandidateOverflow']) {
    const result = await cpu.execute({
      inputs: {
        ...inputs,
        weightsOverflow: vector(Uint32Array.of(Number(flagName === 'weightsOverflow'))),
        weightsCandidateOverflow: vector(
          Uint32Array.of(Number(flagName === 'weightsCandidateOverflow'))
        )
      }
    });
    expect(result['labels']).toEqual(Uint32Array.of(0xffffffff, 0xffffffff, 0xffffffff));
    expect(result['regionCount']).toEqual(Uint32Array.of(0));
    expect(result['cutEdges']).toEqual(Uint32Array.of(0xffffffff, 0xffffffff));
    expect(result['cutGains']).toEqual(Float32Array.of(NaN, NaN));
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = new Map<string, GraphDataView>();
    const getView = (port: string) => {
      if (!views.has(port))
        views.set(
          port,
          createTransientView(
            graph,
            port,
            ['values', 'weights', 'cutGains'].includes(port) ? 'float32' : 'uint32',
            port === 'offsets' || ['neighbors', 'weights', 'treeEdgeFlags'].includes(port)
              ? 4
              : ['values', 'componentLabels', 'labels'].includes(port)
                ? 3
                : ['parameters', 'cutEdges', 'cutGains'].includes(port)
                  ? 2
                  : 1
          )
        );
      return views.get(port)!;
    };
    const contributor = gpu.createContributor(
      getView as Parameters<typeof gpu.createContributor>[0]
    ) as GPUSkaterRegions;
    expect(contributor.props.weights.status?.overflow).toBe(views.get('weightsOverflow'));
    expect(contributor.props.weights.status?.candidateOverflow).toBe(
      views.get('weightsCandidateOverflow')
    );
    expect(contributor.props.weights.status?.candidateOverflow).toBeDefined();
    contributor.getCommandNodes(graph);
  } finally {
    device.destroy();
  }
});

it('SKATER validates supplied CPU loss word lengths', () => {
  for (const flagName of ['overflow', 'candidateOverflow'] as const)
    expect(() =>
      computeSkaterRegionsOnCPU(
        {...CHAIN, status: {[flagName]: new Uint32Array(0)}},
        {maximumRegionCount: 2}
      )
    ).toThrow(`status.${flagName}`);
});

it('REDCAP forwards original topology and both status words into its forest', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const weights = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
      weights: createTransientView(graph, 'weights', 'float32', 4),
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidate', 'uint32', 1)
      }
    };
    const contributor = new GPURedcapSingleLinkage({
      weights,
      componentIterations: 3,
      standardize: false,
      values: createTransientView(graph, 'values', 'float32', 3),
      parameters: createTransientView(graph, 'parameters', 'uint32', 1),
      output: {labels: createTransientView(graph, 'labels', 'uint32', 3)}
    });
    const nodes = contributor.getCommandNodes(graph);
    const treeBindings = nodes.flatMap(node => {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      return node.id.startsWith('redcap-single-linkage-tree-') ? (record?.bindings ?? []) : [];
    });
    expect(treeBindings.some(binding => binding.view === weights.status.overflow)).toBe(true);
    expect(treeBindings.some(binding => binding.view === weights.status.candidateOverflow)).toBe(
      true
    );
    expect(treeBindings.some(binding => binding.view === weights.offsets)).toBe(true);
    expect(nodes.some(node => node.id.endsWith('-guard-topology'))).toBe(false);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
