// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUNetworkCostWeights,
  getGPUNetworkCostWeightsParameterValues
} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialWeightsAlgebra,
  GPUChunkedSpatialWeightsAlgebra,
  GPUFuzzyContiguity,
  GPUAnalysisContiguityStatistics,
  GPUSpatialWeightsTranspose,
  describeGPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';

function setup(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const port = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createOutputBuffer(device, values.length);
    buffer.write(values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `w3-${serial++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = (rowCount = 2) => {
    const count = port('uint32', Uint32Array.of(99));
    const overflow = port('uint32', Uint32Array.of(99));
    const requiredCount = port('uint32', Uint32Array.of(99));
    const candidateOverflow = port('uint32', Uint32Array.of(99));
    const offsets = port('uint32', new Uint32Array(rowCount + 1).fill(99));
    return {
      count,
      overflow,
      requiredCount,
      candidateOverflow,
      offsets,
      weights: {
        offsets: offsets.view,
        neighbors: port('uint32', Uint32Array.of(99, 99)).view,
        weights: port('float32', Float32Array.of(99, 99)).view,
        status: {
          count: count.view,
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          candidateOverflow: candidateOverflow.view
        }
      }
    };
  };
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: port('float32x2', Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1))
      .view,
    ringOffsets: port('uint32', Uint32Array.of(0, 4, 8)).view,
    polygonOffsets: port('uint32', Uint32Array.of(0, 1, 2)).view,
    featureOffsets: port('uint32', Uint32Array.of(0, 1, 2)).view
  });
  const source = () => ({
    offsets: port('uint32', Uint32Array.of(0, 1, 2)).view,
    neighbors: port('uint32', Uint32Array.of(1, 0)).view,
    weights: port('float32', Float32Array.of(1, 1)).view
  });
  return {
    graph,
    port,
    output,
    polygons,
    source,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

for (const route of ['lattice', 'contiguity', 'algebra', 'chunked', 'fuzzy', 'profile'] as const)
  it(`OUT-04 ${route} overwrites all supported status destinations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = setup(device);
    const {graph, port, output, polygons, source} = scene;
    const destination = output();
    const latticeDistances =
      route === 'lattice' ? port('float32', Float32Array.of(99, 99)) : undefined;
    const destinations = route === 'chunked' ? [output(1), output(1)] : [destination];
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      if (route === 'lattice')
        graph.add(
          new GPULatticeWeights({
            width: 2,
            height: 1,
            criterion: 'rook',
            output: {
              ...destination.weights,
              distances: latticeDistances!.view
            }
          })
        );
      if (route === 'contiguity') {
        const producer = new GPUContiguityWeights({
          criterion: 'rook',
          polygons: polygons(),
          output: destination.weights
        });
        graph.add(producer);
        // Consume the actual descriptor; absence of a distance port is meaningful to transpose.
        expect(producer.outputWeights.distances).toBeUndefined();
        const transposed = output();
        graph.add(
          new GPUSpatialWeightsTranspose({
            weights: producer.outputWeights,
            output: {...transposed.weights, status: undefined}
          })
        );
      }
      if (route === 'algebra') {
        const weights = source();
        graph.add(
          new GPUSpatialWeightsAlgebra({
            operation: 'union',
            left: weights,
            right: weights,
            output: destination.weights
          })
        );
      }
      if (route === 'chunked') {
        const weights = {
          rowCount: 2,
          chunks: [0, 1].map(row => ({
            rowStart: row,
            rowCount: 1,
            weights: {
              offsets: port('uint32', Uint32Array.of(0, 1)).view,
              neighbors: port('uint32', Uint32Array.of(1 - row)).view,
              weights: port('float32', Float32Array.of(1)).view
            }
          }))
        };
        graph.add(
          new GPUChunkedSpatialWeightsAlgebra({
            operation: 'union',
            left: weights,
            right: weights,
            output: {
              chunks: destinations.map((item, row) => ({
                rowStart: row,
                rowCount: 1,
                weights: item.weights
              }))
            }
          })
        );
      }
      if (route === 'fuzzy') {
        const distances = port('float32', Float32Array.of(99, 99));
        graph.add(
          new GPUFuzzyContiguity({
            polygons: polygons(),
            tolerance: 0,
            output: {
              ...destination.weights,
              distances: distances.view,
              added: port('uint32', Uint32Array.of(99, 99)).view,
              featureStatus: port('uint32', Uint32Array.of(99, 99)).view
            }
          })
        );
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        expect(await readFloat32(distances.buffer)).toEqual([0, 0]);
      }
      if (route === 'profile')
        graph.add(
          new GPUAnalysisContiguityStatistics({
            criterion: 'rook',
            polygons: polygons(),
            output: {
              weights: destination.weights,
              counts: port('uint32', new Uint32Array(6).fill(99)).view,
              cardinality: port('uint32', Uint32Array.of(99, 99)).view,
              median: port('float32', Float32Array.of(99)).view,
              validity: port('uint32', Uint32Array.of(99)).view
            }
          })
        );
      if (!compiled) {
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
      }
      if (latticeDistances) expect(await readFloat32(latticeDistances.buffer)).toEqual([1, 1]);
      // Two adjacent cells/squares give two directed links; each one-row chunk retains one.
      for (const item of destinations) {
        expect(await readUint32(item.count.buffer)).toEqual([route === 'chunked' ? 1 : 2]);
        expect(await readUint32(item.requiredCount.buffer)).toEqual([route === 'chunked' ? 1 : 2]);
        expect(await readUint32(item.overflow.buffer)).toEqual([0]);
        expect(await readUint32(item.candidateOverflow.buffer)).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      scene.destroy();
    }
  });

for (const chunked of [false, true])
  it(`CPL-03 inherited failure survives described ${chunked ? 'chunked' : 'whole'} weights clean-incomplete-clean`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = setup(device);
    const {graph, port, output, source} = scene;
    const candidate = port('uint32', Uint32Array.of(0));
    const complete = port('uint32', Uint32Array.of(1));
    const weights = describeGPUSpatialWeights(
      {...source(), status: {candidateOverflow: candidate.view, complete: complete.view}},
      {information: {rowCount: 2, transform: 'binary', symmetric: true}}
    );
    const destination = output();
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      const producer = chunked
        ? new GPUChunkedSpatialWeightsAlgebra({
            operation: 'union',
            left: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            right: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination.weights}]}
          })
        : new GPUSpatialWeightsAlgebra({
            operation: 'union',
            left: weights,
            right: weights,
            output: destination.weights
          });
      graph.add(producer);
      const published = port('uint32', Uint32Array.of(99));
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'observe-published-candidate',
          operation: 'TestObserveCertificate',
          invocationCount: 1,
          bindings: [
            {
              name: 'source',
              view: producer.outputWeights.status!.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'destination', view: published.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = source[sourceOffset];'
        })
      );
      compiled = graph.compile();
      for (const loss of [0, 1, 0]) {
        candidate.buffer.write(Uint32Array.of(0));
        complete.buffer.write(Uint32Array.of(1 - loss));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(published.buffer)).toEqual([loss]);
        expect(await readUint32(destination.candidateOverflow.buffer)).toEqual([
          chunked ? 0 : loss
        ]);
        expect(await readUint32(destination.overflow.buffer)).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      scene.destroy();
    }
  });

it('OUT-01 rejects seeded distance destinations before contiguity dispatch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = setup(device);
  try {
    for (const values of [
      [0, 0],
      [10, 10]
    ])
      expect(
        () =>
          new GPUContiguityWeights({
            criterion: 'rook',
            polygons: scene.polygons(),
            output: {
              ...scene.output().weights,
              distances: scene.port('float32', Float32Array.from(values)).view
            }
          })
      ).toThrow(/output.distances/);
  } finally {
    scene.destroy();
  }
});

it('OUT-01 preserves deliberate measured-distance attachment on an actual contiguity handle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = setup(device);
  let compiled: ReturnType<typeof scene.graph.compile> | undefined;
  try {
    const producer = new GPUContiguityWeights({
      criterion: 'rook',
      polygons: scene.polygons(),
      output: scene.output().weights
    });
    const measured = scene.port('float32', Float32Array.of(2, 3));
    const distances = scene.port('float32', Float32Array.of(99, 99));
    scene.graph.add(producer);
    scene.graph.add(
      new GPUSpatialWeightsTranspose({
        weights: {...producer.outputWeights, distances: measured.view},
        output: {...scene.output().weights, distances: distances.view, status: undefined}
      })
    );
    compiled = scene.graph.compile();
    submitGraph(device, compiled, undefined);
    // Directed slots (0,1) and (1,0) exchange rows under transpose.
    expect(await readFloat32(distances.buffer)).toEqual([3, 2]);
  } finally {
    compiled?.destroy();
    scene.destroy();
  }
});

it('S1 retained count is clamped when two required rook slots exceed capacity one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = setup(device);
  const destination = scene.output();
  const output = {
    ...destination.weights,
    neighbors: scene.port('uint32', Uint32Array.of(99)).view,
    weights: scene.port('float32', Float32Array.of(99)).view
  };
  let compiled: ReturnType<typeof scene.graph.compile> | undefined;
  try {
    scene.graph.add(new GPULatticeWeights({width: 2, height: 1, criterion: 'rook', output}));
    compiled = scene.graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(destination.offsets.buffer)).toEqual([0, 1, 1]);
    expect(await readUint32(destination.count.buffer)).toEqual([1]);
    expect(await readUint32(destination.requiredCount.buffer)).toEqual([2]);
    expect(await readUint32(destination.overflow.buffer)).toEqual([1]);
  } finally {
    compiled?.destroy();
    scene.destroy();
  }
});

for (const chunked of [false, true])
  it(`CPL-03 actual network converter to ${chunked ? 'chunked' : 'whole'} algebra preserves independent completeness`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = setup(device);
    const {graph, port, output} = scene;
    const costsComplete = port('uint32', Uint32Array.of(1));
    const converted = output();
    const complete = port('uint32', Uint32Array.of(99));
    const converter = new GPUNetworkCostWeights({
      rowCount: 2,
      costs: port('float32', Float32Array.of(Infinity, 1, 1, Infinity)).view,
      costsComplete: costsComplete.view,
      parameters: port('float32', getGPUNetworkCostWeightsParameterValues({costLimit: 2})).view,
      output: {
        offsets: converted.weights.offsets,
        neighbors: converted.weights.neighbors,
        weights: converted.weights.weights,
        status: {
          overflow: converted.overflow.view,
          requiredCount: converted.requiredCount.view,
          complete: complete.view
        }
      }
    });
    const destination = output();
    const weights = converter.outputWeights;
    const input = {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]};
    const consumer = chunked
      ? new GPUChunkedSpatialWeightsAlgebra({
          operation: 'union',
          left: input,
          right: input,
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination.weights}]}
        })
      : new GPUSpatialWeightsAlgebra({
          operation: 'union',
          left: weights,
          right: weights,
          output: destination.weights
        });
    const published = port('uint32', Uint32Array.of(99));
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      graph.add(converter);
      graph.add(consumer);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'observe-converter-consumer-certificate',
          operation: 'TestObserveCertificate',
          invocationCount: 1,
          bindings: [
            {
              name: 'source',
              view: consumer.outputWeights.status!.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'destination', view: published.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = source[sourceOffset];'
        })
      );
      compiled = graph.compile();
      for (const value of [1, 0, 1]) {
        costsComplete.buffer.write(Uint32Array.of(value));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(complete.buffer)).toEqual([value]);
        expect(await readUint32(converted.overflow.buffer)).toEqual([0]);
        expect(await readUint32(destination.overflow.buffer)).toEqual([0]);
        expect(await readUint32(published.buffer)).toEqual([1 - value]);
        expect(await readUint32(destination.candidateOverflow.buffer)).toEqual([
          chunked ? 0 : 1 - value
        ]);
      }
    } finally {
      compiled?.destroy();
      scene.destroy();
    }
  });

it('S3 count and requiredCount share disjoint packed destinations across separate dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = setup(device);
  const destination = scene.output();
  const packed = scene.port('uint32', new Uint32Array(128).fill(99));
  const requiredCount = scene.graph.createDataView(packed.view.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const count = scene.graph.createDataView(packed.view.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  let compiled: ReturnType<typeof scene.graph.compile> | undefined;
  try {
    scene.graph.add(
      new GPULatticeWeights({
        width: 2,
        height: 1,
        criterion: 'rook',
        output: {
          ...destination.weights,
          status: {...destination.weights.status, requiredCount, count}
        }
      })
    );
    compiled = scene.graph.compile();
    submitGraph(device, compiled, undefined);
    const words = await readUint32(packed.buffer);
    expect(words[0]).toBe(2);
    expect(words[64]).toBe(2);
  } finally {
    compiled?.destroy();
    scene.destroy();
  }
});
