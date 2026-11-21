// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../src/raster-analysis/index';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowProps
} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowOperation} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow-operation';
import {CPUTerrainFlowOperation} from '../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {getGPUTerrainFlowParameterValues} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainWatershedsOperation} from '../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

function createFlowProps(graph: GPUCommandGraph, secondOffset = 256): GPUTerrainFlowProps {
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const pooled = createTransientView(graph, 'pooled', 'uint32', 65);
  return {
    id: 'review-flow',
    grid,
    fillDepressions: true,
    resolveFlats: true,
    maximumFillIterations: 1,
    maximumFlatIterations: 1,
    elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', 1), grid),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {
      filledElevation: createTransientView(graph, 'filled', 'float32', 1),
      status: {
        flatsConverged: graph.createDataView(pooled.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 0
        }),
        flatsIterationCount: graph.createDataView(pooled.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: secondOffset
        })
      }
    }
  };
}

it('flat finalizer refuses overlapping binding windows and accepts aligned disjoint words', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUTerrainFlow(createFlowProps(graph, 4))).toThrow(
    /flatsConverged.*flatsIterationCount.*256/
  );
  const acceptedGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUTerrainFlow(createFlowProps(acceptedGraph, 256))).not.toThrow();
});

it('flow captures caller-owned output selection before lowering', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createFlowProps(graph);
  const flow = new GPUTerrainFlow(props);
  props.output.status!.flatsConverged = undefined;
  props.output.status!.flatsIterationCount = undefined;
  props.maximumFlatIterations = 2;
  const ids = flow.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('review-flow-flats-finalize');
  expect(ids).not.toContain('review-flow-flats-lower-relax-1');
});

function lowerWatershed() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    id: 'review-watershed',
    grid: makeGPURasterMetadata({width: 4, height: 1}),
    maximumIterations: 1,
    flowDirections: 'directions',
    output: {labels: 'labels', status: {converged: 'ok'}}
  };
  const operation = new GPUTerrainWatershedsOperation(props);
  props.maximumIterations = 2;
  props.output.status.converged = 'mutated';
  const views = Object.fromEntries(
    [...Object.keys(operation.inputs), ...Object.keys(operation.outputs)].map(port => [
      port,
      createTransientView(graph, port, 'uint32', port === 'converged' ? 1 : 4)
    ])
  );
  const contributor = operation.createContributor(
    port => views[port] as never
  ) as GPUCommandNodeProducer;
  return contributor.getCommandNodes(graph);
}

it('watershed lowering captures the validated budget', () => {
  const ids = lowerWatershed().map(node => node.id);
  expect(ids).toContain('review-watershed-pointer-round-0');
  expect(ids).not.toContain('review-watershed-pointer-round-1');
});

it('explicit watershed rounds read a stable input snapshot', () => {
  const nodes = lowerWatershed();
  expect(nodes.map(node => node.id)).toContain('review-watershed-pointer-round-0-snapshot');
  const source = getParsedKernelSources(nodes).get('review-watershed-pointer-round-0');
  expect(source).toContain('previousPointer[previousPointerOffset + current]');
});

it('flow operation captures diagnostic names for declaration and lowering', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    grid: makeGPURasterMetadata({width: 1, height: 1}),
    fillDepressions: true,
    resolveFlats: true,
    maximumFlatIterations: 1,
    maximumFillIterations: 1,
    elevation: {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'elevation'}
    },
    parameters: 'parameters',
    output: {
      filledElevation: 'filled',
      status: {flatsConverged: 'ok', flatsIterationCount: undefined as string | undefined}
    }
  };
  const operation = new GPUTerrainFlowOperation(props);
  props.output.status.flatsConverged = 'changed';
  props.output.status.flatsIterationCount = 'later';
  const views = Object.fromEntries(
    [...Object.keys(operation.inputs), ...Object.keys(operation.outputs)].map(port => [
      port,
      createTransientView(
        graph,
        port,
        port === 'flatsConverged' ? 'uint32' : 'float32',
        port === 'parameters' ? 8 : 1
      )
    ])
  );
  const requested: string[] = [];
  const contributor = operation.createContributor(port => {
    requested.push(port);
    return views[port] as never;
  }) as GPUCommandNodeProducer;
  expect(requested).not.toContain('flatsIterationCount');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain(
    'terrain-flow-flats-finalize'
  );
});

it('CPU flow retains the explicit flat budget captured at construction', async () => {
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const props = {
    grid,
    fillDepressions: true,
    resolveFlats: true,
    maximumFlatIterations: 1,
    elevation: {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'elevation'}
    },
    parameters: 'parameters',
    output: {
      filledElevation: 'filled',
      status: {flatsConverged: 'ok', flatsIterationCount: 'rounds'}
    }
  };
  const operation = new CPUTerrainFlowOperation(props);
  props.maximumFlatIterations = 2;
  props.resolveFlats = false;
  const inputs = Object.fromEntries(
    Object.entries({
      values: new Float32Array(9).fill(10),
      parameters: getGPUTerrainFlowParameterValues({}, grid)
    }).map(([port, values]) => [
      port,
      {
        format: 'float32' as const,
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      }
    ])
  );
  const result = await operation.execute({inputs});
  // The center changes in the first lower round; the captured budget excludes confirmation.
  expect(Array.from(result['ok'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['rounds'] as Uint32Array)).toEqual([3]);
});
