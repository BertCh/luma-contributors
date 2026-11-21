// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, test} from 'vitest';
import {
  GPUContiguityWeights,
  GPUNeighborSearch,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsTraces,
  getGPUNeighborSearchParameterValues,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {
  importGraphBuffer,
  submitGraph
} from '../../../examples/deck/spatial-analysis-shared/graph-buffers';
import showcaseSource from '../../../examples/deck/spatial-analysis-showcase/src/scenes/regression/health-regression.compute.ts?raw';
import explorerSource from '../../../examples/deck/spatial-analysis-explorer/modes/regression-mode.ts?raw';
import {addKernelPass} from '../../../examples/deck/spatial-analysis-shared/mode-kernels';
import {readRegressionFloat32Word, readRegressionUint32Word} from './regression-example-readback';

// Execute the shipped construction blocks with small fixture resources. Only pair/slot capacities
// are reduced to exercise loss: every failure word below is written by the real GPU producer.
test('showcase actual weight construction propagates candidate-only loss and recovers across source switches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  const compiledGraphs: {destroy(): void}[] = [];
  const createBuffer = (id: string, value: number | Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      id,
      ...(typeof value === 'number' ? {byteLength: value} : {data: value}),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  const rowCount = 3;
  const vertices = Float32Array.from(
    Array.from({length: 3}, () => [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]).flat()
  );
  const geometry = {
    vertices,
    featureCount: rowCount,
    partRingOffsets: Uint32Array.of(0, 1, 2, 3),
    ringOffsets: Uint32Array.of(0, 5, 10, 15)
  };
  const parameters = createBuffer(
    'parameters',
    getGPUNeighborSearchParameterValues({
      bounds: [-1, -1, 2, 2],
      maximumDistance: Infinity,
      weightType: 'binary',
      rowStandardize: true
    })
  );
  const start = showcaseSource.indexOf('  const importWeights =');
  const end = showcaseSource.indexOf('\n  searchParameters.write(', start);
  const code = transpileModule(showcaseSource.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const construct = new Function(
    'environment',
    `with(environment) { ${code} return {importWeights, getWeightsGraph}; }`
  );
  const overflowBuffer = createBuffer('overflow', 4);
  const candidateOverflowBuffer = createBuffer('candidate', 4);
  const environment = {
    device,
    rowCount,
    capacity: 20,
    geometry,
    projection: {spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'}},
    offsetsBuffer: createBuffer('offsets', 16),
    neighborsBuffer: createBuffer('neighbors', 80),
    weightValuesBuffer: createBuffer('values', 80),
    overflowBuffer,
    candidateOverflowBuffer,
    totalNeighborsBuffer: createBuffer('required', 4),
    verticesBuffer: createBuffer('vertices', vertices),
    featureOffsetsBuffer: createBuffer('features', Uint32Array.of(0, 1, 2, 3)),
    polygonOffsetsBuffer: createBuffer('polygons', geometry.partRingOffsets),
    ringOffsetsBuffer: createBuffer('rings', geometry.ringOffsets),
    positionsBuffer: createBuffer('positions', Float32Array.of(0, 0, 1, 0, 2, 0)),
    resources: {
      track: (graph: {destroy(): void}) => {
        compiledGraphs.push(graph);
        return graph;
      }
    },
    searchParameters: {
      importToGraph: (graph: GPUCommandGraph<void>) =>
        importGraphBuffer(graph, 'parameters', parameters, 'float32')
    },
    importGraphBuffer,
    GPUCommandGraph,
    GPUSpatialWeightsTransform,
    GPUNeighborSearch,
    GPUContiguityWeights: class extends GPUContiguityWeights {
      constructor(props: ConstructorParameters<typeof GPUContiguityWeights>[0]) {
        super({...props, pairCapacity: props.criterion === 'queen' ? 1 : 64});
      }
    },
    NEIGHBOR_COUNTS: {knn4: 2},
    GRID_SIZE: [2, 2]
  };
  try {
    const construction = construct(environment);
    const consumer = new GPUCommandGraph<void>(device, {id: 'example-regression-consumer'});
    const norm = createBuffer('norm', 4);
    consumer.add(
      new GPUSpatialWeightsTraces({
        weights: construction.importWeights(consumer),
        maximumPower: 1,
        mode: 'exact',
        output: {
          infinityNorm: importGraphBuffer(consumer, 'norm', norm, 'float32', 1),
          powerTraces: importGraphBuffer(
            consumer,
            'traces',
            createBuffer('traces', 4),
            'float32',
            1
          ),
          powerSums: importGraphBuffer(consumer, 'sums', createBuffer('sums', 4), 'float32', 1)
        }
      })
    );
    const compiledConsumer = await consumer.compileAsync();
    compiledGraphs.push(compiledConsumer);
    for (const kind of ['queen', 'rook', 'queen', 'knn4']) {
      submitGraph(device, await construction.getWeightsGraph(kind));
      submitGraph(device, compiledConsumer);
      const candidate = readRegressionUint32Word(await candidateOverflowBuffer.readAsync());
      expect(candidate).toBe(kind === 'queen' ? 1 : 0);
      expect(readRegressionUint32Word(await overflowBuffer.readAsync())).toBe(0);
      const value = readRegressionFloat32Word(await norm.readAsync());
      if (kind === 'queen') expect(Number.isNaN(value)).toBe(true);
      else {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeCloseTo(1, 6);
      }
    }
  } finally {
    for (const graph of compiledGraphs) graph.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

test('explorer actual search construction propagates final loss and clean recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  const createBuffer = (id: string, value: number | Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      id,
      ...(typeof value === 'number' ? {byteLength: value} : {data: value}),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph<void>(device, {id: 'explorer-regression-construction'});
  const parameters = createBuffer(
    'parameters',
    getGPUNeighborSearchParameterValues({
      bounds: [-1, -1, 3, 1],
      searchRadius: 1.1,
      weightType: 'binary',
      rowStandardize: true
    })
  );
  const start = explorerSource.indexOf(
    "  const positions = resources.createBuffer('positions', study.positions);",
    explorerSource.indexOf('function createSpatialStage(')
  );
  const end = explorerSource.indexOf('\n  const writeSearch =', start);
  const code = transpileModule(explorerSource.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const construct = new Function('environment', `with(environment) { ${code} return weights; }`);
  const weights = construct({
    graph,
    rowCount: 3,
    capacity: 1,
    kind: 'band',
    neighborCount: 2,
    study: {
      positions: Float32Array.of(0, 0, 1, 0, 2, 0),
      predictors: new Float32Array(9),
      response: Float32Array.of(1, 2, 3),
      rowOfCell: Uint32Array.of(0, 1, 2),
      frame: {spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'}}
    },
    resources: {
      createBuffer,
      createParameterBuffer: () => ({
        importToGraph: () => importGraphBuffer(graph, 'parameters', parameters, 'float32')
      })
    },
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    importGraphBuffer,
    GPUNeighborSearch
  });
  const norm = createBuffer('norm', 4);
  graph.add(
    new GPUSpatialWeightsTraces({
      weights,
      maximumPower: 1,
      mode: 'exact',
      output: {
        infinityNorm: importGraphBuffer(graph, 'norm', norm, 'float32', 1),
        powerTraces: importGraphBuffer(graph, 'traces', createBuffer('traces', 4), 'float32', 1),
        powerSums: importGraphBuffer(graph, 'sums', createBuffer('sums', 4), 'float32', 1)
      }
    })
  );
  const compiled = await graph.compileAsync();
  try {
    for (const radius of [1.1, 0.1, 1.1, 0.1]) {
      parameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: [-1, -1, 3, 1],
          searchRadius: radius,
          weightType: 'binary',
          rowStandardize: true
        })
      );
      submitGraph(device, compiled);
      const value = readRegressionFloat32Word(await norm.readAsync());
      const requiredBuffer = buffers.find(buffer => buffer.id === 'total-neighbors')!;
      const requiredCount = readRegressionUint32Word(await requiredBuffer.readAsync());
      if (radius > 1) expect(Number.isNaN(value)).toBe(true);
      else {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBe(1);
        expect(requiredCount).toBe(0);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

test('health model graph stamps each submission for late readback rejection', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const recipeGraph = new GPUCommandGraph<void>(device, {id: 'health-generation-fixture'});
  const modelGenerationBuffer = device.createBuffer({
    data: Uint32Array.of(0),
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const generation = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  const start = showcaseSource.indexOf('    addKernelPass(recipeGraph,');
  const end = showcaseSource.indexOf('\n    // Estimators', start);
  const code = transpileModule(showcaseSource.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  new Function('environment', `with(environment) { ${code} }`)({
    recipeGraph,
    modelGenerationBuffer,
    buffers: {generation},
    variantKey: 'fixture',
    addKernelPass,
    importGraphBuffer
  });
  const compiled = await recipeGraph.compileAsync();
  try {
    submitGraph(device, compiled);
    const oldRead = generation.readAsync();
    modelGenerationBuffer.write(Uint32Array.of(1));
    submitGraph(device, compiled);
    const currentRead = generation.readAsync();
    expect(readRegressionUint32Word(await oldRead)).toBe(0);
    expect(readRegressionUint32Word(await currentRead)).toBe(1);
  } finally {
    compiled.destroy();
    modelGenerationBuffer.destroy();
    generation.destroy();
  }
});
