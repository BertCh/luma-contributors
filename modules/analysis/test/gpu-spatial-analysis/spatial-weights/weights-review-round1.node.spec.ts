// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUNetworkCostWeights} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {validateCompleteRegressionWeights} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-weights';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {
  GPUAnalysisContiguityStatistics,
  decodeGPUAnalysisContiguityStatistics
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {getGPUSpatialWeightsCapacityPlan} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {preparePolygonContiguityTopology} from '../../../src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {
  GPUSpatialTwoStageLeastSquaresOperation,
  type SpatialTwoStageLeastSquaresOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {GPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';

const swap = {
  offsets: Uint32Array.of(0, 1, 2),
  neighbors: Uint32Array.of(1, 0),
  weights: Float32Array.of(1, 1)
};
function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review1-${serial++}`, format, length);
  const output = () => ({
    offsets: view('uint32', 3),
    neighbors: view('uint32', 2),
    weights: view('float32', 2),
    status: {
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  });
  const polygons = {
    kind: 'polygons' as const,
    positions: view('float32x2', 8),
    ringOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 3),
    featureOffsets: view('uint32', 3)
  };
  return {device, graph, view, output, polygons};
}
it('B1 preserves all four named HAC kernel inputs in both operation forms', () => {
  for (const Operation of [
    CPUSpatialTwoStageLeastSquaresOperation,
    GPUSpatialTwoStageLeastSquaresOperation
  ])
    for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const) {
      const props: SpatialTwoStageLeastSquaresOperationProps = {
        id: 'review-hac',
        rowCount: 12,
        predictorCount: 2,
        predictors: 'predictors',
        response: 'response',
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        robust: {
          method: 'HAC',
          kernel: {
            offsets: 'kernelOffsets',
            neighbors: 'kernelNeighbors',
            weights: 'kernelWeights',
            status: {[name]: 'live'}
          },
          output: {covariance: 'covariance', status: {code: 'robustStatus'}}
        },
        output: {table: 'table', summary: 'summary', residuals: 'residuals', status: {code: 'code'}}
      };
      const operation = new Operation(props);
      expect(operation.inputs[`kernel${name[0].toUpperCase()}${name.slice(1)}`]).toBe('live');
    }
});
it('B2 refuses zero completeness and matrix requirements above the sum of chunk terminal offsets', () => {
  const chunks = [0, 1].map(row => ({
    rowStart: row,
    rowCount: 1,
    weights: {
      offsets: Uint32Array.of(0, 1),
      neighbors: Uint32Array.of(1 - row),
      weights: Float32Array.of(1)
    }
  }));
  for (const weights of [
    {...swap, status: {complete: Uint32Array.of(0)}},
    {rowCount: 2, chunks, status: {requiredCount: Uint32Array.of(3)}}
  ])
    expect(() =>
      computeSpatialWeightsTracesOnCPU({weights} as never, {maximumPower: 2, mode: 'exact'})
    ).toThrow(/weights.status.*incomplete/);
  expect([
    ...computeSpatialWeightsTracesOnCPU(
      {weights: {...swap, status: {complete: Uint32Array.of(1)}}} as never,
      {maximumPower: 2}
    ).powerTraces
  ]).toEqual([0, 2]);
});
it('B3 binds independent positive completeness even when the candidate flag stays clear', () => {
  for (const chunked of [false, true]) {
    const {device, graph, view, output} = setup();
    try {
      const complete = view('uint32', 1);
      const source = {
        ...output(),
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1), complete}
      };
      const chunks = {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: source}]};
      const producer = chunked
        ? new GPUChunkedSpatialWeightsAlgebra({
            operation: 'union',
            left: chunks,
            right: chunks,
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: output()}]}
          })
        : new GPUSpatialWeightsAlgebra({
            operation: 'union',
            left: source,
            right: source,
            output: output()
          });
      const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
      expect(
        sources.some(record => /lossInput\d+\[lossInput\d+Offset\] == 0u/.test(record.source))
      ).toBe(true);
    } finally {
      device.destroy();
    }
  }
});
it('B5 preserves separated-coordinate Polygon rows and original float64 equality', () => {
  const coordinate = new arrow.Struct([
    new arrow.Field('x', new arrow.Float64()),
    new arrow.Field('y', new arrow.Float64())
  ]);
  const ring = new arrow.List(new arrow.Field('vertex', coordinate));
  const polygon = new arrow.List(new arrow.Field('ring', ring));
  const triangle = [
    {x: 10, y: 20},
    {x: 30, y: 20},
    {x: 10, y: 35},
    {x: 10, y: 20}
  ];
  const polygons = arrow.vectorFromArray(
    [null, [triangle], [triangle.map(point => ({x: point.x + 2 ** -30, y: point.y}))]],
    polygon
  );
  const prepared = preparePolygonContiguityTopology({
    metadata: {rowCount: 3, invalidRows: []},
    batches: [{id: 'separated', rowCount: 3, sourceRowOffset: 0, polygons}]
  });
  expect([...prepared.geometryMask]).toEqual([0, 1, 1]);
  expect([...prepared.featureOffsets]).toEqual([0, 0, 1, 2]);
  expect(prepared.invalidRows).toEqual([]);
  expect(prepared.uniqueVertexCount).toBe(6);
});
it('B6 captures inherited props before normalizing all affected routes', () => {
  const {device, graph, view, output, polygons} = setup();
  try {
    const source = output();
    const factories = [
      () =>
        new GPUSpatialWeightsAlgebra(
          Object.create({operation: 'union', left: source, right: source, output: output()})
        ),
      () =>
        new GPULatticeWeights(
          Object.create({criterion: 'rook', width: 2, height: 1, output: output()})
        ),
      () =>
        new GPUFuzzyContiguity(
          Object.create({
            polygons,
            tolerance: 0,
            output: {...output(), added: view('uint32', 2), featureStatus: view('uint32', 2)}
          })
        ),
      () =>
        new GPUAnalysisContiguityStatistics(
          Object.create({
            criterion: 'rook',
            polygons,
            output: {
              weights: output(),
              counts: view('uint32', 6),
              cardinality: view('uint32', 2),
              median: view('float32', 1),
              validity: view('uint32', 1)
            }
          })
        ),
      () =>
        new GPURegionStatistics(
          Object.create({
            region: {kind: 'rectangle', bounds: view('float32', 4)},
            positions: polygons.positions,
            values: view('float32', 8),
            output: {summary: view('uint32', 8)}
          })
        )
    ];
    for (const construct of factories) expect(construct).not.toThrow();
    let operationReads = 0,
      outputReads = 0;
    const captured = Object.create({
      get operation() {
        operationReads++;
        return 'union';
      },
      get output() {
        outputReads++;
        return output();
      },
      left: source,
      right: source
    });
    new GPUSpatialWeightsAlgebra(captured);
    expect([operationReads, outputReads]).toEqual([1, 1]);
    expect(graph).toBeDefined();
  } finally {
    device.destroy();
  }
});
it('B7 keeps unobserved capacity estimates without inventing an exact requirement', () => {
  expect(getGPUSpatialWeightsCapacityPlan({rowCount: 3, capacity: 4})).toMatchObject({
    capacity: 4,
    estimatedRequiredCount: 4,
    requirement: {kind: 'unknown'},
    retryStage: 'candidates'
  });
});
it('B8 rejects unwritten raw named CPU contiguity destinations', () => {
  const props = {
    criterion: 'rook' as const,
    featureCount: 2,
    capacity: 2,
    polygons: {
      positions: 'positions',
      ringOffsets: 'rings',
      polygonOffsets: 'polygons',
      featureOffsets: 'features'
    },
    output: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'overflow', candidateOverflow: 'candidate'}
    }
  };
  for (const [name, output] of [
    ['distances', {...props.output, distances: 'distance'}],
    ['status.count', {...props.output, status: {...props.output.status, count: 'count'}}],
    [
      'status.converged',
      {...props.output, status: {...props.output.status, converged: 'converged'}}
    ]
  ] as const)
    expect(() => new CPUContiguityWeightsOperation({...props, output} as never)).toThrow(
      new RegExp(`output.${name}`)
    );
});
it('B9 honors positive loss in either flat or nested decoder alias', () => {
  const base = {
    counts: [2, 0, 0, 1, 1, 0],
    median: [1],
    validity: [1],
    overflow: [0],
    candidateOverflow: [0]
  };
  for (const flag of ['overflow', 'candidateOverflow'] as const)
    for (const nested of [false, true]) {
      const values = {
        ...base,
        [flag]: [Number(!nested)],
        weights: {status: {overflow: [0], candidateOverflow: [0], [flag]: [Number(nested)]}}
      };
      expect(decodeGPUAnalysisContiguityStatistics(values)).toMatchObject({
        exact: false,
        [flag]: true
      });
    }
});
it('S3 allows count and requirement at separate ranges of one buffer, but keeps count separate from offsets', () => {
  const {device, graph, view, output} = setup();
  try {
    const buffer = graph.createTransientBuffer({
      id: 'packed-status',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const requiredCount = graph.createDataView(buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    const count = graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 256});
    const destination = output();
    expect(() =>
      new GPULatticeWeights({
        width: 2,
        height: 1,
        criterion: 'rook',
        output: {...destination, status: {...destination.status, requiredCount, count}}
      }).getCommandNodes(graph)
    ).not.toThrow();
    expect(
      () =>
        new GPULatticeWeights({
          width: 2,
          height: 1,
          criterion: 'rook',
          output: {
            ...destination,
            status: {
              ...destination.status,
              count: graph.createDataView(destination.offsets.buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 0
              })
            }
          }
        })
    ).toThrow();
    expect(view).toBeDefined();
  } finally {
    device.destroy();
  }
});
it('S4 refuses an exact profile with a nonfinite median', () => {
  for (const median of [Number.NaN, Infinity, -Infinity])
    expect(() =>
      decodeGPUAnalysisContiguityStatistics({
        counts: [2, 0, 0, 1, 1, 0],
        median: [median],
        validity: [1],
        overflow: [0],
        candidateOverflow: [0]
      })
    ).toThrow(/median.*finite/);
});
it('S5 ignores unsupported optional named destinations whose value is undefined', () => {
  expect(
    () =>
      new GPUSpatialWeightsAlgebraOperation({
        operation: 'union',
        rowCount: 2,
        capacity: 2,
        left: {offsets: 'leftOffsets', neighbors: 'leftNeighbors', weights: 'leftWeights'},
        right: {offsets: 'rightOffsets', neighbors: 'rightNeighbors', weights: 'rightWeights'},
        output: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          status: {overflow: 'overflow', converged: undefined}
        }
      } as never)
  ).not.toThrow();
});

it('CPL-03 actual converter handles preserve their certificate through both algebra constructors', () => {
  for (const chunked of [false, true]) {
    const {device, graph, view, output} = setup();
    try {
      const destination = output();
      const complete = view('uint32', 1);
      const converter = new GPUNetworkCostWeights({
        rowCount: 2,
        costs: view('float32', 4),
        parameters: view('float32', 4),
        costsComplete: view('uint32', 1),
        output: {
          offsets: destination.offsets,
          neighbors: destination.neighbors,
          weights: destination.weights,
          status: {
            overflow: destination.status.overflow,
            requiredCount: destination.status.requiredCount,
            complete
          }
        }
      });
      const weights = converter.outputWeights;
      expect(weights.status?.complete).toBe(complete);
      const input = {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]};
      const consumer = chunked
        ? new GPUChunkedSpatialWeightsAlgebra({
            operation: 'union',
            left: input,
            right: input,
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: output()}]}
          })
        : new GPUSpatialWeightsAlgebra({
            operation: 'union',
            left: weights,
            right: weights,
            output: output()
          });
      const captured = chunked
        ? (consumer as GPUChunkedSpatialWeightsAlgebra).props
        : (consumer as GPUSpatialWeightsAlgebra).props;
      expect(captured.operation).toBe('union');
      expect(
        getGPUCommandNodeKernelSources(consumer.getCommandNodes(graph)).some(record =>
          /lossInput\d+\[lossInput\d+Offset\] == 0u/.test(record.source)
        )
      ).toBe(true);
    } finally {
      device.destroy();
    }
  }
});
it('B1/B2 preserve qualified migration errors while recognizing named chunks', () => {
  expect(() =>
    validateCompleteRegressionWeights('named-loss', {
      offsets: 'offsets',
      status: {overflow: 'loss'}
    })
  ).toThrow(/named-loss weights.status.*GPU contributor/);
  expect(() =>
    validateCompleteRegressionWeights('cpu-loss', {...swap, status: {overflow: Uint32Array.of(1)}})
  ).toThrow(/cpu-loss weights.status.*incomplete.*GPU contributor/);
  expect(() =>
    validateCompleteRegressionWeights('named-chunks', {
      rowCount: 2,
      chunks: [{weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'}}]
    })
  ).not.toThrow();
});

it('S3 composed count permits packed requirement but rejects logical overlap with every extra output', () => {
  const {device, graph, view, output, polygons} = setup();
  try {
    const buffer = graph.createTransientBuffer({
      id: 'profile-packed-status',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const requiredCount = graph.createDataView(buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    const count = graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 256});
    const weights = output();
    expect(
      () =>
        new GPUAnalysisContiguityStatistics({
          criterion: 'rook',
          polygons,
          output: {
            weights: {...weights, status: {...weights.status, requiredCount, count}},
            counts: view('uint32', 6),
            cardinality: view('uint32', 2),
            median: view('float32', 1),
            validity: view('uint32', 1)
          }
        })
    ).not.toThrow();
    const featureStatus = view('uint32', 2);
    const overlappingCount = graph.createDataView(featureStatus.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    const fuzzyOutput = output();
    expect(
      () =>
        new GPUFuzzyContiguity({
          polygons,
          tolerance: 0,
          output: {
            ...fuzzyOutput,
            status: {...fuzzyOutput.status, count: overlappingCount},
            added: view('uint32', 2),
            featureStatus
          }
        })
    ).toThrow();
  } finally {
    device.destroy();
  }
});
