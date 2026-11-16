// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCostDistance} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {getGPURasterRelaxationContinuationStateLength} from '../../../src/raster-analysis/cost-distance/raster-relaxation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(width = 64, maximumIterations = 4, continued = true) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const grid = {
    width,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const view = <Format extends 'uint32' | 'float32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length);
  const props = {
    grid,
    maximumIterations,
    friction: {
      id: 'friction',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: view('friction', 'float32', width)}
    },
    parameters: view('parameters', 'float32', 8),
    sources: view('sources', 'uint32', 1),
    continuation: continued
      ? {
          state: view('state', 'uint32', getGPURasterRelaxationContinuationStateLength(grid)),
          values: view('working', 'float32', width)
        }
      : undefined,
    output: {
      costs: view('costs', 'float32', width),
      settledAt: view('settled-at', 'uint32', width)
    }
  };
  return {graph, props};
}

it('retains settlement stamps on resume and clears them when the cumulative round count resets', () => {
  for (const continued of [false, true]) {
    const {graph, props} = createProps(64, 4, continued);
    const nodes = new GPUCostDistance(props).getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    const reset = sources.get('cost-distance-settled-at-reset')!;
    expect(reset).toContain('settledAt[settledAtOffset + index] = 0u');
    if (continued) {
      const resetCondition = nodes.find(
        node => node.id === 'cost-distance-settled-at-reset'
      )?.condition;
      expect(resetCondition).toEqual(
        nodes.find(node => node.id === 'cost-distance-initialize')?.condition
      );
      expect(resetCondition).toMatchObject({source: 'gpu', mode: 'indirect'});
    } else {
      expect(reset).not.toContain('statusOffset');
    }
    expect(sources.get('cost-distance-relax-0')).toContain('= iteration + 1u');
  }
});

it('settlement recording keeps shader and pipeline counts constant as continued rounds grow', () => {
  const counts = [8, 20].map(maximumIterations => {
    const {graph, props} = createProps(64, maximumIterations);
    const nodes = new GPUCostDistance(props).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const source of sources) {
      expect(findReservedIdentifiers(stripWGSLComments(source.source)), source.id).toEqual([]);
    }
    expect(
      getGPUCommandNodePipelineKeys(nodes, {idPattern: /^cost-distance-relax-\d+$/}).size
    ).toBe(1);
    return {
      sources: new Set(sources.map(source => source.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
});

it('rejects physical output and continuation aliases at construction', () => {
  for (const field of ['settledAt', 'continuation'] as const) {
    const {graph, props} = createProps();
    const buffer = graph.device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
    const costs = importGraphBuffer(graph, 'first', buffer, 'float32', 64);
    const second = graph.createDataView(costs.buffer, {
      format: field === 'settledAt' ? 'uint32' : 'float32',
      length: 64
    });
    expect(
      () =>
        new GPUCostDistance({
          ...props,
          continuation:
            field === 'continuation'
              ? {...props.continuation!, values: second as typeof costs}
              : props.continuation,
          output: {
            ...props.output,
            costs,
            ...(field === 'settledAt' ? {settledAt: second as typeof props.output.settledAt} : {})
          }
        })
    ).toThrow(/(?:logical|aligned binding).*overlap/);
    buffer.destroy();
  }
});

it('validates calibration and band metadata before graph construction can allocate scratch', () => {
  const {props} = createProps();
  for (const value of [NaN, Infinity, 1e100]) {
    expect(
      () => new GPUCostDistance({...props, friction: {...props.friction, scale: value}})
    ).toThrow(/^cost-distance friction.scale/);
  }
  expect(
    () => new GPUCostDistance({...props, friction: {...props.friction, noDataValue: Infinity}})
  ).toThrow(/nodata/);
});

it('rejects non-uint32 sweep indices before emitting a WGSL literal', () => {
  const {props} = createProps();
  for (const sweepAfterIteration of [NaN, Infinity, -1, 0.5, 0x100000000]) {
    expect(() => new GPUCostDistance({...props, sweepAfterIteration})).toThrow(
      /^cost-distance sweepAfterIteration.*4294967295/
    );
  }
  expect(() => new GPUCostDistance({...props, sweepAfterIteration: 0})).not.toThrow();
});

it('bounds the actual node count with optional tie, band and settlement stages at the round limit', () => {
  for (const continued of [false, true]) {
    const {graph, props} = createProps(1, 1024, continued);
    const contributor = new GPUCostDistance({
      ...props,
      costBreaks: createTransientView(graph, 'breaks', 'float32', 2),
      output: {
        ...props.output,
        backLinks: continued ? undefined : createTransientView(graph, 'links', 'uint32', 1),
        bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2)
      }
    });
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  }
});

it('preflights aligned caller views and derived gate tables before changing the graph', () => {
  for (const kind of ['aligned-view', 'gate-table', 'backing-allocation'] as const) {
    const {graph, props} = createProps(kind === 'aligned-view' ? 64 : 1);
    let parameters = props.parameters;
    if (kind === 'aligned-view') {
      const buffer = graph.createTransientBuffer({
        id: 'offset-buffer',
        byteLength: 512,
        usage: Buffer.STORAGE
      });
      parameters = graph.createDataView(buffer, {format: 'float32', byteOffset: 252, length: 8});
    }
    const contributor = new GPUCostDistance({...props, parameters});
    Object.defineProperty(graph.device, 'limits', {
      value: {
        ...graph.device.limits,
        ...(kind === 'backing-allocation'
          ? {maxBufferSize: 1024}
          : {maxStorageBufferBindingSize: kind === 'gate-table' ? 512 : 256})
      }
    });
    const resources = vi.spyOn(graph, 'createTransientBuffer');
    const views = vi.spyOn(graph, 'createDataView');
    const nodes = vi.spyOn(graph, 'addComputePass');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      kind === 'aligned-view' ? /284 bytes/ : /gate slots/
    );
    expect(resources).not.toHaveBeenCalled();
    expect(views).not.toHaveBeenCalled();
    expect(nodes).not.toHaveBeenCalled();
  }
});

it('rejects caller bindings without STORAGE usage before declaring resources or nodes', () => {
  const {graph, props} = createProps(1, 1, false);
  const buffer = graph.createTransientBuffer({
    id: 'copy-only',
    byteLength: 32,
    usage: Buffer.COPY_DST
  });
  const parameters = graph.createDataView(buffer, {format: 'float32', length: 8});
  const contributor = new GPUCostDistance({...props, parameters});
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  const views = vi.spyOn(graph, 'createDataView');
  const nodes = vi.spyOn(graph, 'addComputePass');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE|storage/);
  expect(resources).not.toHaveBeenCalled();
  expect(views).not.toHaveBeenCalled();
  expect(nodes).not.toHaveBeenCalled();
});

it('accepts packed costs, stamps and status when shared alignment prefixes are used in separate dispatches', () => {
  const {graph, props} = createProps(1, 1);
  const buffer = graph.createTransientBuffer({
    id: 'packed-cost-outputs',
    byteLength: 768,
    usage: Buffer.STORAGE
  });
  const view = <Format extends 'float32' | 'uint32'>(format: Format, byteOffset: number) =>
    graph.createDataView(buffer, {format, byteOffset, length: 1});
  expect(() =>
    graph.add(
      new GPUCostDistance({
        ...props,
        output: {
          costs: view('float32', 0),
          settledAt: view('uint32', 256),
          status: {
            converged: view('uint32', 4),
            iterationCount: view('uint32', 260),
            iterationLimitReached: view('uint32', 512)
          }
        }
      })
    )
  ).not.toThrow();
});
