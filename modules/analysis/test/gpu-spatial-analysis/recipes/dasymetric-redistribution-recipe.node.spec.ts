// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addDasymetricRedistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {createSpatialWeightsSuppressionNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-suppression';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `dasymetric-test-${serial++}`, format, length);
  const props = {
    grid: {
      width: 6,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
      pixelInterpretation: 'area' as const
    },
    source: {
      kind: 'coverage' as const,
      coverage: {
        zoneIds: view('uint32', 8),
        cellIds: view('uint32', 8),
        fractions: view('float32', 8),
        zoneValidity: view('uint32', 3),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    },
    sourceTotals: view('float32', 3),
    targetZones: view('uint32', 6),
    targetZoneCount: 3,
    ancillary: {
      id: 'ancillary',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: view('float32', 6)}
    },
    mask: view('uint32', 6),
    pairCapacity: 8
  };
  return {device, graph, view, props};
}

it('F1 prepared dasymetric coverage reads candidate exhaustion independently of tuple overflow', () => {
  const {device, graph, view, props} = createContext();
  try {
    const coverage = {
      ...props.source.coverage,
      status: {...props.source.coverage.status, candidateOverflow: view('uint32', 1)}
    };
    addDasymetricRedistributionRecipe(graph, {...props, source: {kind: 'coverage', coverage}});
    const source = getGPUCommandNodeKernelSources(
      (graph as unknown as {nodes: unknown[]}).nodes
    ).find(kernel => kernel.id.endsWith('-coverage-evidence'))!.source;
    expect(source).toContain('candidateOverflow[candidateOverflowOffset]');
  } finally {
    device.destroy();
  }
});

it('dasymetric transfer suppression calls its declared NaN helper with the right arity', () => {
  const {device, graph, props} = createContext();
  try {
    addDasymetricRedistributionRecipe(graph, props);
    const kernels = getGPUCommandNodeKernelSources(
      (graph as unknown as {nodes: unknown[]}).nodes
    ).filter(kernel => kernel.id.includes('-transfer-incomplete-'));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      const definition = /fn getQuietNaN\(([^)]*)\)/.exec(kernel.source)!;
      const parameterCount = definition[1].split(',').filter(parameter => parameter.trim()).length;
      for (const call of kernel.source.matchAll(/\bgetQuietNaN\(([^)]*)\)/g)) {
        const argumentCount = call[1].split(',').filter(argument => argument.trim()).length;
        expect(argumentCount, `${kernel.id}: ${call[0]}`).toBe(parameterCount);
      }
    }
  } finally {
    device.destroy();
  }
});

it('G5 prepared dasymetric coverage reads the first logical word of padded status views', () => {
  const {device, graph, view, props} = createContext();
  try {
    const padded = () =>
      graph.createDataView(view('uint32', 4).buffer, {
        format: 'uint32',
        length: 2,
        byteOffset: 4
      });
    const status = {
      count: padded(),
      requiredCount: padded(),
      overflow: padded(),
      candidateOverflow: padded()
    };
    const coverage = {...props.source.coverage, status};
    const result = addDasymetricRedistributionRecipe(graph, {
      ...props,
      source: {kind: 'coverage', coverage}
    });
    const node = (graph as unknown as {nodes: GPUCommandNode[]}).nodes.find(node =>
      node.id.endsWith('-coverage-evidence')
    )!;
    const evidence = result.status.stages.find(
      stage => stage.stage === 'coverage-validation'
    )!.status;
    if (!('complete' in evidence) || !evidence.complete)
      throw new Error('coverage validation must publish completeness');
    const memory = createCPUGraphMemory();
    memory.write(status.count, Uint32Array.of(1, 0xffffffff));
    memory.write(status.requiredCount, Uint32Array.of(1, 0xffffffff));
    memory.write(status.overflow, Uint32Array.of(0, 1));
    for (const loss of [0, 1, 0]) {
      memory.write(status.candidateOverflow, Uint32Array.of(loss, 1 - loss));
      executeCommandNodesOnCPU([node], memory);
      expect(memory.readUint32(evidence.complete)[0]).toBe(1 - loss);
    }
  } finally {
    device.destroy();
  }
});

it('dasymetric shared suppression preserves clean numbers and invalidates lost support', () => {
  const {device, graph, props, view} = createContext();
  try {
    const result = addDasymetricRedistributionRecipe(graph, props);
    const output = view('float32', 1);
    const validity = view('uint32', 1);
    const nodes = createSpatialWeightsSuppressionNodes(
      graph,
      'suppression-test',
      'suppression-test',
      result.intermediates.weights,
      {output, validity}
    );
    const memory = createCPUGraphMemory();
    for (const loss of [0, 1, 0]) {
      // Normal producers rewrite these values before each suppression stage.
      memory.write(output, Float32Array.of(7));
      memory.write(validity, Uint32Array.of(1));
      memory.write(result.intermediates.weights.status!.overflow!, Uint32Array.of(loss));
      executeCommandNodesOnCPU(nodes, memory);
      if (loss) expect(memory.readFloat32(output)[0]).toBeNaN();
      else expect(memory.readFloat32(output)[0]).toBe(7);
      // The shared helper suppresses numerical ports; validity belongs to its producer.
      expect(memory.readUint32(validity)[0]).toBe(1);
    }
  } finally {
    device.destroy();
  }
});

it('dasymetric transfer reads the pair loss word written by the preceding areal stage', () => {
  const {device, graph, props} = createContext();
  try {
    const result = addDasymetricRedistributionRecipe(graph, props);
    const areal = result.contributors.find(
      contributor => contributor instanceof GPUArealInterpolation
    ) as GPUArealInterpolation;
    expect(result.intermediates.weights.status!.overflow).toBe(areal.props.output.status.overflow);
    expect(result.intermediates.weights.status!.requiredCount).toBe(
      areal.props.output.status.requiredPairCount
    );
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    const writerIndex = nodes.findIndex(node => node.id.endsWith('-areal-offsets'));
    const readerIndex = nodes.findIndex(node => node.id.endsWith('-transfer-incomplete-0'));
    expect(writerIndex).toBeGreaterThanOrEqual(0);
    expect(readerIndex).toBeGreaterThan(writerIndex);
    const writer = nodes[writerIndex];
    const reader = nodes[readerIndex];
    const loss = result.intermediates.weights.status!.overflow!;
    expect(writer.resources).toContainEqual({buffer: loss, usage: 'storage-read-write'});
    expect(reader.resources).toContainEqual({buffer: loss, usage: 'storage-read'});
    const output = reader.resources![1] as {buffer: GraphDataView<'float32'>};
    const memory = createCPUGraphMemory();
    // Poison the transient flag: the real offsets writer must clear it on a complete empty CSR,
    // rather than relying on initial memory or reading an unrelated unwritten status view.
    memory.write(loss, Uint32Array.of(1));
    memory.write(output.buffer, Float32Array.of(7, 0, 0));
    executeCommandNodesOnCPU([writer, reader], memory);
    expect(memory.readUint32(loss)[0]).toBe(0);
    expect(Array.from(memory.readFloat32(output.buffer))).toEqual([7, 0, 0]);
  } finally {
    device.destroy();
  }
});

it.each([
  'zone',
  'overlap'
] as const)('dasymetric %s composition keeps provenance and core binding limits', denominator => {
  const {device, graph, props} = createContext();
  try {
    const result = addDasymetricRedistributionRecipe(graph, {...props, denominator});
    expect(result.intermediates.coverage).toEqual(props.source.coverage);
    expect(result.intermediates.targetZones).toBe(props.targetZones);
    expect(result.intermediates.sourceTotals).toBe(props.sourceTotals);
    expect(result.intermediates.ancillary).toEqual(props.ancillary);
    expect(result.intermediates.mask).toBe(props.mask);
    expect(result.status.stages[0].status).toEqual(props.source.coverage.status);
    expect(result.outputs.sourceAllocatedTotals.length).toBe(3);
    expect(result.outputs.targetTotals.length).toBe(3);
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(10);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    device.destroy();
  }
});

it.each([
  'output alias',
  'target length',
  'source validity length',
  'mask length',
  'shape bound'
])('dasymetric rejects %s', boundary => {
  const {device, graph, view, props} = createContext();
  try {
    expect(() =>
      addDasymetricRedistributionRecipe(graph, {
        ...props,
        ...(boundary === 'output alias' ? {output: {targetTotals: props.sourceTotals}} : {}),
        ...(boundary === 'target length' ? {targetZones: view('uint32', 5)} : {}),
        ...(boundary === 'source validity length'
          ? {
              source: {
                ...props.source,
                coverage: {...props.source.coverage, zoneValidity: view('uint32', 2)}
              }
            }
          : {}),
        ...(boundary === 'mask length' ? {mask: view('uint32', 5)} : {}),
        ...(boundary === 'shape bound' ? {targetZoneCount: 524289} : {})
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('STA-05 publishes completeness and eligible-source evidence with compatibility identities', () => {
  const {device, graph, props} = createContext();
  try {
    const result = addDasymetricRedistributionRecipe(graph, props);
    const conservation = result.status.stages.find(
      stage => stage.stage === 'dasymetric-conservation'
    )!.status;
    const validation = result.status.stages.find(
      stage => stage.stage === 'coverage-validation'
    )!.status;
    if (
      !('complete' in conservation) ||
      !('zeroEligibleCount' in conservation) ||
      !('complete' in validation)
    )
      throw new Error('missing canonical conservation status');
    expect(conservation.complete).toBe(result.status.complete);
    expect(conservation.zeroEligibleCount).toBe(result.status.zeroEligibleCount);
    expect(conservation).not.toHaveProperty('converged');
    expect(conservation).not.toHaveProperty('invalidCount');
    expect(validation).not.toHaveProperty('converged');
  } finally {
    device.destroy();
  }
});
