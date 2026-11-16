// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPUKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging';
import {GPUFocalStatistics} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaSpaceTimeLogLikelihood,
  GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPU_VECCHIA_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());

let viewSerial = 0;

function createViews(graph: GPUCommandGraph) {
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${name}-${viewSerial++}`, format, length);
  const backing = view('oversized', 'uint32', 33_554_433);
  return {
    view,
    oversized: graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 33_554_432,
      byteOffset: 4
    })
  };
}

function expectPreflight(
  graph: GPUCommandGraph,
  declare: () => unknown,
  message: RegExp = /134217732 bytes/
) {
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(declare).toThrow(message);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
}

for (const mode of ['spatial', 'space-time', 'fisher'] as const) {
  for (const validation of ['size', 'usage']) {
    it(`prechecks ${mode} Vecchia binding ${validation} before any declaration`, () => {
      const graph = new GPUCommandGraph(device);
      const {view, oversized} = createViews(graph);
      const nonStorage = device.createBuffer({
        byteLength: 8,
        usage: Buffer.COPY_SRC | Buffer.COPY_DST
      });
      const props = {
        positions: view('positions', 'float32x2', 2),
        observations: view('observations', 'float32x4', 2),
        neighborhoodOffsets: view('offsets', 'uint32', 3),
        neighborhoods:
          validation === 'size'
            ? oversized
            : importGraphBuffer(graph, 'non-storage-neighbors', nonStorage, 'uint32', 2),
        neighborCount: 1,
        parameters: view(
          'params',
          'float32',
          mode === 'space-time'
            ? GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
            : GPU_VECCHIA_PARAMETER_LENGTH
        ),
        output: {
          conditionals: view('conditionals', 'float32x4', 2),
          conditionalStatus: view('conditional-status', 'uint32', 2),
          siteTerms: view('terms', 'float32', 2),
          siteStatus: view('site-status', 'uint32', 2),
          logLikelihood: view('total', 'float32', 2),
          status: view('status', 'uint32', 3)
        }
      };
      const contributor =
        mode === 'fisher'
          ? new GPUVecchiaFisherScoring({
              ...props,
              output: {
                siteTerms: view('fisher-terms', 'float32', 20),
                siteStatus: props.output.siteStatus,
                totals: view('fisher-total', 'float32', 20),
                status: props.output.status
              }
            })
          : mode === 'space-time'
            ? new GPUVecchiaSpaceTimeLogLikelihood({
                ...props,
                times: view('times', 'float32', 2),
                temporalContext: {
                  unit: 'second',
                  epoch: 0n,
                  encoding: 'float32-relative',
                  axis: {kind: 'irregular'}
                }
              })
            : new GPUVecchiaLogLikelihood(props);
      try {
        expectPreflight(
          graph,
          () => contributor.getCommandNodes(graph),
          validation === 'size' ? /134217732 bytes/ : /neighborhoods.*Buffer.STORAGE/
        );
      } finally {
        nonStorage.destroy();
      }
    });
  }
}

it('prechecks STORAGE usage in raster kriging and focal bindings before any declaration', () => {
  const graph = new GPUCommandGraph(device);
  const base = createKriging(graph).props;
  const buffer = device.createBuffer({byteLength: 64, usage: Buffer.COPY_SRC | Buffer.COPY_DST});
  try {
    const parameters = importGraphBuffer(graph, 'non-storage-params', buffer, 'float32', 16);
    const contributor = new GPUKriging({...base, parameters});
    expectPreflight(graph, () => contributor.getCommandNodes(graph), /parameters.*Buffer.STORAGE/);
    vi.restoreAllMocks();
    const focal = new GPUFocalStatistics({
      values: base.values,
      parameters,
      maximumRadius: 1,
      grid: makeGPURasterMetadata({width: 2, height: 1}),
      output: {mean: createTransientView(graph, 'focal-mean', 'float32', 2)}
    });
    expectPreflight(graph, () => focal.getCommandNodes(graph), /parameters.*Buffer.STORAGE/);
  } finally {
    buffer.destroy();
  }
});

function createKriging(graph: GPUCommandGraph, invocationTileSize?: number) {
  const {view} = createViews(graph);
  return new GPUKriging({
    positions: view('positions', 'float32x2', 2),
    values: view('samples', 'float32', 2),
    parameters: view('params', 'float32', 16),
    grid: makeGPURasterMetadata({width: 97, height: 1}),
    indexGridSize: [2, 1],
    indexBounds: [-1, -1, 1, 1],
    invocationTileSize,
    output: {values: view('results', 'float32', 97)}
  });
}

it('prechecks raster kriging and focal inputs at their aligned binding size', () => {
  const graph = new GPUCommandGraph(device);
  const {view, oversized} = createViews(graph);
  const base = createKriging(graph).props;
  const kriging = new GPUKriging({
    ...base,
    parameters: graph.createDataView(oversized.buffer, {
      format: 'float32',
      length: oversized.length,
      byteOffset: 4
    })
  });
  expectPreflight(graph, () => kriging.getCommandNodes(graph));
  vi.restoreAllMocks();
  const focal = new GPUFocalStatistics({
    values: base.values,
    grid: makeGPURasterMetadata({width: 2, height: 1}),
    maximumRadius: 1,
    parameters: view('focal-params', 'float32', 4),
    output: {counts: oversized}
  });
  expectPreflight(graph, () => focal.getCommandNodes(graph));
});

it('keeps raster kriging tile shaders/pipelines constant and dispatches each tile at its own size', () => {
  const counts: number[] = [];
  for (const tileSize of [97, 33, 31]) {
    const graph = new GPUCommandGraph(device);
    const contributor = createKriging(graph, tileSize);
    const allNodes = contributor.getCommandNodes(graph);
    expect(allNodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const nodes = allNodes.filter(node => node.workload?.variant === 'local-ordinary');
    counts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
    expect(new Set(getGPUCommandNodeKernelSources(nodes).map(({source}) => source)).size).toBe(1);
    const lengths = Array.from({length: Math.ceil(97 / tileSize)}, (_, index) =>
      Math.min(tileSize, 97 - index * tileSize)
    );
    expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      lengths.map(length => Math.ceil(length / 64))
    );
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(constants).toEqual({});
      expectValidQueryShaderLiterals(source);
      const words =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    }
  }
  expect(counts).toEqual([1, 1, 1]);
});

it('rejects unrepresentable shader literals and excessive raster tile node counts at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createKriging(graph).props;
  expect(() => new GPUKriging({...props, indexBounds: [0, 0, 1e100, 1]})).toThrow(
    /indexBounds\[2\].*finite/
  );
  expect(() => new GPUKriging({...props, invocationTileSize: 0})).toThrow(/invocationTileSize/);
  expect(
    () =>
      new GPUKriging({
        ...props,
        grid: makeGPURasterMetadata({width: 40000, height: 1}),
        invocationTileSize: 1,
        output: {values: createTransientView(graph, 'many-rows', 'float32', 40000)}
      })
  ).toThrow(/estimated command node count/);
  expect(
    () =>
      new GPUFocalStatistics({
        values: props.values,
        parameters: props.parameters,
        maximumRadius: 1,
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        noDataValue: 1e100,
        output: {mean: createTransientView(graph, 'mean', 'float32', 2)}
      })
  ).toThrow(/noDataValue.*finite/);
});
