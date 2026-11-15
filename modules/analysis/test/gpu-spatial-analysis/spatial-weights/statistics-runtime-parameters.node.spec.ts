// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  getGPUSpatialWeightsSummaryParameterValues,
  GPUSpatialWeightsSummary,
  getGPULatticeWeightsParameterValues,
  getGPUSpatialWeightsTransformParameterValues,
  GPULatticeWeights,
  GPUSpatialWeightsTransform,
  GPUChunkedSpatialWeightsTransform
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  getGPUHotSpotAnalysisParameterValues,
  GPUHotSpotAnalysis
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {createSpatialStatisticsParameterNodes} from '../../../src/analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 8),
    weights: view('float32', 8),
    distances: view('float32', 8)
  };
  const output = {...weights, status: {overflow: view('uint32', 1)}};
  return {device, graph, view, weights, output};
}

it('packs runtime scales with float32-aware bounds and preserves target padding', () => {
  expect([...getGPULatticeWeightsParameterValues()]).toEqual([1, 1]);
  expect([...getGPUHotSpotAnalysisParameterValues()]).toEqual([1]);
  const target = new Float32Array([8, 8, 8]);
  expect(getGPUSpatialWeightsTransformParameterValues({bandwidth: 2}, target)).toBe(target);
  expect([...target]).toEqual([2, 8, 8]);
  for (const invalid of [0, -1, NaN, Infinity, 1e40, 1e-50]) {
    expect(() => getGPUSpatialWeightsTransformParameterValues({bandwidth: invalid})).toThrow(
      /bandwidth/
    );
    expect(() => getGPULatticeWeightsParameterValues({cellSize: [1, invalid]})).toThrow(/cellSize/);
  }
  expect(() => getGPUHotSpotAnalysisParameterValues({selfWeight: -1})).toThrow(/selfWeight/);
  expect(() => getGPUHotSpotAnalysisParameterValues({selfWeight: 1e40})).toThrow(/selfWeight/);
  expect([...getGPUHotSpotAnalysisParameterValues({selfWeight: 0})]).toEqual([0]);
  expect(() => getGPULatticeWeightsParameterValues({}, new Float32Array(1))).toThrow(
    /target.length/
  );
  expect(() =>
    getGPUSpatialWeightsTransformParameterValues({bandwidth: 1}, new Float32Array(0))
  ).toThrow(/target.length/);
});

it('numeric defaults and runtime views generate identical compute pipelines', () => {
  const sources = (kind: 'transform' | 'lattice' | 'hotSpot', setting: number | 'runtime') => {
    const {device, graph, view, weights, output} = setup();
    try {
      const producer =
        kind === 'transform'
          ? new GPUSpatialWeightsTransform({
              weights,
              operation: 'kernel',
              bandwidth: setting === 'runtime' ? view('float32', 1) : setting
            })
          : kind === 'lattice'
            ? new GPULatticeWeights({
                width: 2,
                height: 2,
                criterion: 'queen',
                output,
                cellSize: setting === 'runtime' ? view('float32', 2) : [setting, setting]
              })
            : new GPUHotSpotAnalysis({
                weights,
                values: view('float32', 4),
                parameters: view('float32', 8),
                selfWeight: setting === 'runtime' ? view('float32', 1) : setting,
                output: {zScores: view('float32', 4)}
              });
      return getGPUCommandNodeKernelSources(producer.getCommandNodes(graph)).map(
        kernel => kernel.source
      );
    } finally {
      device.destroy();
    }
  };
  for (const kind of ['transform', 'lattice', 'hotSpot'] as const) {
    expect(sources(kind, 1)).toEqual(sources(kind, 3));
    expect(sources(kind, 1)).toEqual(sources(kind, 'runtime'));
  }
  expect(sources('hotSpot', 0)).toEqual(sources('hotSpot', 1));
});

it('lattice capacity plans count uploaded distance parameters only when allocated', () => {
  for (const distances of [false, true]) {
    for (const setting of ['default', 'numeric', 'runtime'] as const) {
      const {device, graph, view, output} = setup();
      try {
        const cellSize =
          setting === 'runtime'
            ? view('float32', 2)
            : setting === 'numeric'
              ? ([2, 3] as const)
              : undefined;
        const lattice = new GPULatticeWeights({
          width: 2,
          height: 2,
          criterion: 'queen',
          output: {...output, distances: distances ? output.distances : undefined},
          cellSize
        });
        const createTransientBuffer = vi.spyOn(graph, 'createTransientBuffer');
        lattice.getCommandNodes(graph);
        const declaredBytes = createTransientBuffer.mock.calls.reduce(
          (total, [descriptor]) =>
            total + (descriptor.id.includes('-scan-') ? 0 : descriptor.byteLength),
          0
        );
        const expectedBytes = distances && setting !== 'runtime' ? 40 : 32;
        expect(declaredBytes).toBe(expectedBytes);
        expect(lattice.getCapacityPlan().estimatedPeakTransientBytes).toBe(declaredBytes);
        expect(lattice.getCapacityPlan({requiredCount: 30}).estimatedPeakTransientBytes).toBe(
          declaredBytes
        );
      } finally {
        device.destroy();
      }
    }
  }
});

it('validates parameter shape, aliasing and graph ownership for whole and chunked weights', () => {
  const {device, graph, weights, output, view} = setup();
  try {
    expect(
      () =>
        new GPULatticeWeights({
          width: 2,
          height: 2,
          criterion: 'rook',
          output,
          cellSize: view('float32', 1)
        })
    ).toThrow(/cellSize.length/);
    expect(
      () =>
        new GPUSpatialWeightsTransform({weights, operation: 'kernel', bandwidth: weights.weights})
    ).toThrow(/share buffers/);
    const wrongFormat = view('uint32', 1) as unknown as GraphDataView<'float32'>;
    expect(
      () => new GPUSpatialWeightsTransform({weights, operation: 'kernel', bandwidth: wrongFormat})
    ).toThrow(/bandwidth/);
    const otherGraph = new GPUCommandGraph(device);
    const foreign = createTransientView(otherGraph, 'foreign', 'float32', 1);
    expect(() =>
      new GPUSpatialWeightsTransform({
        weights,
        operation: 'kernel',
        bandwidth: foreign
      }).getCommandNodes(graph)
    ).toThrow(/target graph/);
    const chunked = {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}]};
    expect(() =>
      new GPUChunkedSpatialWeightsTransform({
        weights: chunked,
        operation: 'kernel',
        bandwidth: foreign
      }).getCommandNodes(graph)
    ).toThrow(/target graph/);
    const runtime = view('float32', 1);
    const transformed = new GPUChunkedSpatialWeightsTransform({
      weights: chunked,
      operation: 'kernel',
      bandwidth: runtime
    });
    expect(transformed.getCommandNodes(graph)).toHaveLength(1);
    expect(transformed.outputWeights.chunks[0].weights.weights).toBe(weights.weights);
  } finally {
    device.destroy();
  }
});

it('compiled default uploads copy every encoding and release their GPUParameterBuffer', () => {
  const {device, graph} = setup();
  try {
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      'defaults',
      new Float32Array([2, 3])
    );
    const node = runtime.nodes[0];
    expect(node.type).toBe('copy');
    const executable = node.compile({device});
    const destination = device.createBuffer({byteLength: 8});
    const copyBufferToBuffer = vi.fn();
    const context = {commandEncoder: {copyBufferToBuffer}, getBuffer: () => destination};
    executable.encode(context as never);
    executable.encode(context as never);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(2);
    const source = copyBufferToBuffer.mock.calls[0][0].sourceBuffer;
    const destroy = vi.spyOn(source, 'destroy');
    executable.destroy?.();
    expect(destroy).toHaveBeenCalledOnce();
    expect(
      createSpatialStatisticsParameterNodes(graph, 'borrowed', runtime.parameters).nodes
    ).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('rejects positive hot-spot self weights that cannot retain the Gi* selector in float32', () => {
  const {device, weights, view} = setup();
  try {
    const values = view('float32', 4);
    const parameters = view('float32', 8);
    const zScores = view('float32', 4);
    for (const selfWeight of [1e-50, 2 ** -149, 1e-40, -1e-50]) {
      const target = new Float32Array([1, 99]);
      expect(() => getGPUHotSpotAnalysisParameterValues({selfWeight}, target)).toThrow(
        /^getGPUHotSpotAnalysisParameterValues .*selfWeight/
      );
      expect([...target]).toEqual([1, 99]);
      expect(
        () =>
          new GPUHotSpotAnalysis({
            id: 'hot-spot-boundary',
            weights,
            values,
            parameters,
            selfWeight,
            output: {zScores}
          })
      ).toThrow(/^hot-spot-boundary .*selfWeight/);
    }
    expect([...getGPUHotSpotAnalysisParameterValues({selfWeight: 0})]).toEqual([0]);
    expect([...getGPUHotSpotAnalysisParameterValues({selfWeight: 2 ** -126})]).toEqual([2 ** -126]);
  } finally {
    device.destroy();
  }
});

it('summary symmetry tolerance is packed and never changes the compute pipelines', () => {
  expect([...getGPUSpatialWeightsSummaryParameterValues()]).toEqual([0]);
  const target = new Float32Array([7, 99]);
  expect(getGPUSpatialWeightsSummaryParameterValues({symmetryTolerance: 0.25}, target)).toBe(
    target
  );
  expect([...target]).toEqual([0.25, 99]);
  for (const invalid of [-1, NaN, Infinity, 1e40]) {
    expect(() =>
      getGPUSpatialWeightsSummaryParameterValues({symmetryTolerance: invalid}, target)
    ).toThrow(/symmetryTolerance/);
    expect([...target]).toEqual([0.25, 99]);
  }
  expect(() => getGPUSpatialWeightsSummaryParameterValues({}, new Float32Array(0))).toThrow(
    /target.length/
  );
  for (const chunked of [false, true]) {
    const sources = (setting: number | 'runtime') => {
      const {device, graph, view, weights} = setup();
      try {
        const contributor = new GPUSpatialWeightsSummary({
          weights: chunked ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}]} : weights,
          symmetryTolerance: setting === 'runtime' ? view('float32', 1) : setting,
          output: {statistics: view('float32', 3), counts: view('uint32', 6)}
        });
        return getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).map(
          kernel => kernel.source
        );
      } finally {
        device.destroy();
      }
    };
    expect(sources(0)).toEqual(sources(0.25));
    expect(sources(0)).toEqual(sources('runtime'));
  }
});

it('summary validates tolerance views before creating nodes', () => {
  const {device, graph, view, weights} = setup();
  try {
    const output = {statistics: view('float32', 3), counts: view('uint32', 6)};
    for (const symmetryTolerance of [
      view('float32', 0),
      view('uint32', 1) as unknown as GraphDataView<'float32'>
    ]) {
      expect(() => new GPUSpatialWeightsSummary({weights, output, symmetryTolerance})).toThrow(
        /symmetryTolerance/
      );
    }
    const aliased = graph.createDataView(output.statistics.buffer, {format: 'float32', length: 1});
    expect(
      () => new GPUSpatialWeightsSummary({weights, output, symmetryTolerance: aliased})
    ).toThrow(/share buffers/);
    const otherGraph = new GPUCommandGraph(device);
    const foreign = createTransientView(otherGraph, 'foreign-summary-tolerance', 'float32', 1);
    for (const chunked of [false, true]) {
      expect(() =>
        new GPUSpatialWeightsSummary({
          weights: chunked ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}]} : weights,
          output,
          symmetryTolerance: foreign
        }).getCommandNodes(graph)
      ).toThrow(/target graph/);
    }
  } finally {
    device.destroy();
  }
});

for (const chunked of [false, true]) {
  for (const offsetView of [false, true]) {
    it(`summary rejects oversized ${offsetView ? 'offset' : 'packed'} tolerance bindings before allocation and permits ${chunked ? 'chunked' : 'whole'} retry`, () => {
      const {device, graph, view, weights} = setup();
      try {
        const toleranceStorage = view('float32', 17);
        const oversizedTolerance = offsetView
          ? graph.createDataView(toleranceStorage.buffer, {
              format: 'float32',
              length: 1,
              byteOffset: 64
            })
          : toleranceStorage;
        const correctedTolerance = graph.createDataView(toleranceStorage.buffer, {
          format: 'float32',
          length: 1,
          // The offset case also proves a binding exactly at the limit remains accepted.
          byteOffset: offsetView ? 60 : 0
        });
        const summaryProps = {
          weights: chunked ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}]} : weights,
          output: {statistics: view('float32', 3), counts: view('uint32', 6)}
        };
        Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        expect(() =>
          graph.add(
            new GPUSpatialWeightsSummary({...summaryProps, symmetryTolerance: oversizedTolerance})
          )
        ).toThrow(/68.*64/);
        expect(allocate).not.toHaveBeenCalled();
        expect(() =>
          graph.add(
            new GPUSpatialWeightsSummary({...summaryProps, symmetryTolerance: correctedTolerance})
          )
        ).not.toThrow();
        expect(allocate).toHaveBeenCalled();
      } finally {
        device.destroy();
      }
    });
  }
}
