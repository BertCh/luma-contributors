// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, ComputePipeline, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {afterEach, expect, expectTypeOf, it, vi} from 'vitest';
import {
  GPUAffineTransform,
  type GPUAffineTransformProps
} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {
  GPUGeodesicPairs,
  type GPUGeodesicPairsProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {
  GPUGeodesicDestination,
  type GPUGeodesicDestinationProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {
  GPUDatumGridTransform,
  type GPUDatumGridTransformProps
} from '../../../src/geospatial-analysis/datum-grid-transform/index';
import {
  GPUPointBuffer,
  type GPUPointBufferProps
} from '../../../src/geospatial-analysis/point-buffer/index';
import {
  GPUGaussianPredictiveScores,
  type GPUGaussianPredictiveScoresProps
} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-gaussian-predictive-scores';
import {
  findReservedIdentifiers,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

const CONTRIBUTOR_KINDS = [
  'affine',
  'pairs',
  'destination',
  'datum',
  'point-buffer',
  'gaussian'
] as const;
type ContributorKind = (typeof CONTRIBUTOR_KINDS)[number];

afterEach(() => vi.restoreAllMocks());

function createScenario(
  device: Device,
  kind: ContributorKind,
  lengths: number[],
  overlapOutputFields = false
) {
  const graph = new GPUCommandGraph(device);
  let outputIndex = 0;
  let firstOutputBuffer: GraphDataView['buffer'] | undefined;
  let firstOutputWordCount = 0;
  const rows = <Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32'>(
    name: string,
    format: Format,
    output = false,
    multiplier = 1
  ) => {
    const components = format === 'float32x3' ? 3 : format === 'float32x2' ? 2 : 1;
    const rowCount = lengths.reduce((total, length) => total + length, 0) * multiplier;
    let backing = createTransientView(
      graph,
      `${name}-backing`,
      'uint32',
      1 + rowCount * components * 2 + 64
    ).buffer;
    let wordBase = 1;
    if (output) {
      if (outputIndex === 0) {
        firstOutputBuffer = backing;
        firstOutputWordCount = rowCount * components;
      } else if (outputIndex === 1 && overlapOutputFields) {
        backing = firstOutputBuffer!;
        wordBase += firstOutputWordCount;
      }
      outputIndex++;
    }
    const chunks = lengths.map(length => {
      const view = graph.createDataView(backing, {
        format,
        length: length * multiplier,
        byteOffset: wordBase * 4
      });
      wordBase += length * multiplier * components;
      return view;
    });
    return createVectorView(name, format, chunks);
  };
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
  switch (kind) {
    case 'affine':
      return {
        graph,
        contributor: new GPUAffineTransform({
          positions: rows('positions', 'float32x2'),
          parameters: createTransientView(graph, 'parameters', 'float32', 12),
          output: {positions: rows('output', 'float32x2', true)}
        })
      };
    case 'pairs':
      return {
        graph,
        contributor: new GPUGeodesicPairs({
          spatialContext,
          origins: rows('origins', 'float32x2'),
          targets: rows('targets', 'float32x2'),
          output: {
            distances: rows('distances', 'float32', true),
            initialBearings: rows('initialBearings', 'float32', true),
            finalBearings: rows('finalBearings', 'float32', true),
            midpoints: rows('midpoints', 'float32x2', true),
            status: {
              converged: rows('converged', 'uint32', true),
              approximation: rows('approximation', 'uint32', true),
              iterationLimitReached: rows('iterationLimitReached', 'uint32', true)
            }
          }
        })
      };
    case 'destination':
      return {
        graph,
        contributor: new GPUGeodesicDestination({
          spatialContext,
          origins: rows('origins', 'float32x2'),
          bearings: rows('bearings', 'float32'),
          distances: rows('distances', 'float32'),
          output: {
            destinations: rows('destinations', 'float32x2', true),
            finalBearings: rows('finalBearings', 'float32', true),
            status: {
              converged: rows('converged', 'uint32', true),
              approximation: rows('approximation', 'uint32', true),
              iterationLimitReached: rows('iterationLimitReached', 'uint32', true)
            }
          }
        })
      };
    case 'datum':
      return {
        graph,
        contributor: new GPUDatumGridTransform({
          positions: rows('positions', 'float32x3'),
          positionsLow: rows('positionsLow', 'float32x3'),
          samples: createTransientView(graph, 'samples', 'float32x3', 4),
          mask: createTransientView(graph, 'mask', 'uint32', 4),
          grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
          operation: 'combined',
          sourceVerticalReference: 'source',
          targetVerticalReference: 'target',
          output: {
            positions: rows('output', 'float32x3', true),
            positionsLow: rows('outputLow', 'float32x3', true),
            status: rows('status', 'uint32', true)
          }
        })
      };
    case 'point-buffer':
      return {
        graph,
        contributor: new GPUPointBuffer({
          points: rows('points', 'float32x2'),
          offsets: rows('offsets', 'float32x2', true, 4),
          parameters: createTransientView(graph, 'parameters', 'float32', 2),
          verticesPerRing: 4,
          spatialContext: {coordinateSpace: 'planar', metric: 'native'}
        })
      };
    case 'gaussian':
      return {
        graph,
        contributor: new GPUGaussianPredictiveScores({
          means: rows('means', 'float32'),
          variances: rows('variances', 'float32'),
          observations: rows('observations', 'float32'),
          output: {crps: rows('crps', 'float32', true)}
        })
      };
  }
}

it('uses the repository full WGSL reserved-word list, including references rather than declarations alone', () => {
  for (const identifier of ['target', 'active', 'filter', 'module', 'meta']) {
    expect(WGSL_RESERVED_WORDS.has(identifier)).toBe(true);
    expect(findReservedIdentifiers(`fn main() { let result = ${identifier}; }`)).not.toEqual([]);
  }
  expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
});

for (const kind of CONTRIBUTOR_KINDS) {
  it(`${kind} accepts consecutive output chunks and records each chunk's own bounded dispatch with fixed pipelines`, () => {
    const pipelineKeys = new Set<string>();
    const shaderSources = new Set<string>();
    const outputGroupCount = kind === 'pairs' || kind === 'destination' ? 2 : 1;
    const cases = [
      {lengths: [513], dimensionLimit: 65535},
      {lengths: [513, 1, 257], dimensionLimit: 65535},
      {
        lengths: Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
        dimensionLimit: 65535
      },
      {lengths: [0, 1, 255, 0, 256, 257, 0], dimensionLimit: 65535},
      {lengths: kind === 'point-buffer' ? [1, 65, 257] : [1, 257, 1025], dimensionLimit: 2}
    ];
    for (const {lengths, dimensionLimit} of cases) {
      const device = createNullWebGPUDevice();
      Object.defineProperty(device.limits, 'maxComputeWorkgroupsPerDimension', {
        value: dimensionLimit
      });
      const {graph, contributor} = createScenario(device, kind, lengths);
      const buffers = new Map<GraphDataView['buffer'], Buffer>();
      const pipeline = vi.spyOn(device, 'createComputePipeline').mockImplementation(
        props =>
          Object.assign(Object.create(ComputePipeline.prototype), {
            id: props.id,
            hash: '',
            destroy() {}
          }) as ComputePipeline
      );
      const dispatch = vi.spyOn(Kernel.prototype, 'dispatch').mockImplementation(() => {});
      try {
        const nodes = contributor.getCommandNodes(graph);
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          pipelineKeys.add(key);
        for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
          shaderSources.add(source);
          expect(constants).toEqual({});
          expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        }
        const getBuffer = (view: GraphDataView) => {
          let buffer = buffers.get(view.buffer);
          if (!buffer) {
            buffer = device.createBuffer({
              byteLength: view.buffer.byteLength,
              usage: Buffer.STORAGE
            });
            buffers.set(view.buffer, buffer);
          }
          return buffer;
        };
        const computeNodes = nodes.filter(node => node.type === 'compute');
        for (const node of computeNodes) {
          const compiled = node.compile({device});
          try {
            compiled.encode({computePass: {}, getBuffer} as never);
          } finally {
            compiled.destroy?.();
          }
        }
        const rowCounts = lengths
          .filter(length => length > 0)
          .map(length => length * (kind === 'point-buffer' ? 4 : 1));
        // Closed-form expectations do not call the dispatch-layout implementation under test.
        const singleGroupDimensions =
          dimensionLimit === 2
            ? [
                [1, 1, 1],
                [2, 1, 1],
                [2, 2, 2]
              ]
            : rowCounts.map(length => [Math.ceil(length / 256), 1, 1]);
        const expectedDimensions = Array.from(
          {length: outputGroupCount},
          () => singleGroupDimensions
        ).flat();
        expect(dispatch.mock.calls.map(([, options]) => [options.x, options.y, options.z])).toEqual(
          expectedDimensions
        );
        expect(computeNodes.map(node => node.workload?.maximumInvocationCount)).toEqual(
          Array.from({length: outputGroupCount}, () => rowCounts).flat()
        );
        expect(computeNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
          expectedDimensions.map(([x, y, z]) => x * y * z)
        );
      } finally {
        dispatch.mockRestore();
        pipeline.mockRestore();
        for (const buffer of buffers.values()) buffer.destroy();
        device.destroy();
      }
    }
    expect(pipelineKeys.size).toBe(outputGroupCount);
    expect(shaderSources.size).toBe(outputGroupCount);
  });

  it(`${kind} checks derived parameter scratch before any resource, view or node is declared`, () => {
    const device = createNullWebGPUDevice();
    try {
      const {graph, contributor} = createScenario(device, kind, [1, 3, 5]);
      Object.defineProperty(device.limits, 'maxBufferSize', {value: 512});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const add = vi.spyOn(graph, 'add');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /chunk parameter buffer byteLength.*512/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
}

for (const kind of ['pairs', 'destination', 'datum'] as const) {
  it(`${kind} rejects distinct logical output fields whose aligned bindings overlap in one dispatch`, () => {
    const device = createNullWebGPUDevice();
    try {
      expect(() => createScenario(device, kind, [1], true)).toThrow(
        /aligned binding byte ranges must not overlap/
      );
    } finally {
      device.destroy();
    }
  });
}

it('preserves packed public defaults and infers packed or vector output views from the inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const packedScalar = createTransientView(graph, 'type-scalar', 'float32', 1);
  const packedPair = createTransientView(graph, 'type-pair', 'float32x2', 1);
  const packedTriple = createTransientView(graph, 'type-triple', 'float32x3', 1);
  const packedScalarOutput = createTransientView(graph, 'type-scalar-output', 'float32', 1);
  const packedPairOutput = createTransientView(graph, 'type-pair-output', 'float32x2', 1);
  const packedTripleOutput = createTransientView(graph, 'type-triple-output', 'float32x3', 1);
  const packedStatus = createTransientView(graph, 'type-status', 'uint32', 1);
  const parameters = createTransientView(graph, 'type-parameters', 'float32', 12);
  const packedOffsets = createTransientView(graph, 'type-offsets', 'float32x2', 3);
  const vectorScalar = createVectorView('type-vector-scalar', 'float32', [packedScalar]);
  const vectorPair = createVectorView('type-vector-pair', 'float32x2', [packedPair]);
  const vectorTriple = createVectorView('type-vector-triple', 'float32x3', [packedTriple]);
  const vectorScalarOutput = createVectorView('type-vector-scalar-output', 'float32', [
    packedScalarOutput
  ]);
  const vectorPairOutput = createVectorView('type-vector-pair-output', 'float32x2', [
    packedPairOutput
  ]);
  const vectorTripleOutput = createVectorView('type-vector-triple-output', 'float32x3', [
    packedTripleOutput
  ]);
  const vectorStatus = createVectorView('type-vector-status', 'uint32', [packedStatus]);
  const vectorOffsets = createVectorView('type-vector-offsets', 'float32x2', [packedOffsets]);
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const datum = {
    samples: createTransientView(graph, 'type-samples', 'float32x3', 4),
    mask: createTransientView(graph, 'type-mask', 'uint32', 4),
    grids: [
      {width: 2, height: 2, sampleOffset: 0, origin: [0, 0] as const, spacing: [1, 1] as const}
    ],
    operation: 'horizontal' as const
  };
  try {
    expectTypeOf(
      new GPUAffineTransform({
        positions: packedPair,
        parameters,
        output: {positions: packedPairOutput}
      }).props.output.positions
    ).toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf(
      new GPUGeodesicPairs({
        spatialContext,
        origins: packedPair,
        targets: packedPair,
        output: {distances: packedScalarOutput}
      }).props.output.distances
    ).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
    expectTypeOf(
      new GPUGeodesicDestination({
        spatialContext,
        origins: packedPair,
        bearings: packedScalar,
        distances: packedScalar,
        output: {destinations: packedPairOutput}
      }).props.output.destinations
    ).toEqualTypeOf<GraphDataView<'float32x2'> | undefined>();
    expectTypeOf(
      new GPUDatumGridTransform({
        ...datum,
        positions: packedTriple,
        output: {positions: packedTripleOutput, status: packedStatus}
      }).props.output.positions
    ).toEqualTypeOf<GraphDataView<'float32x3'>>();
    expectTypeOf(
      new GPUPointBuffer({
        points: packedPair,
        parameters,
        offsets: packedOffsets,
        verticesPerRing: 3,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      }).props.offsets
    ).toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf(
      new GPUGaussianPredictiveScores({
        means: packedScalar,
        variances: packedScalar,
        observations: packedScalar,
        output: {crps: packedScalarOutput}
      }).props.output.crps
    ).toEqualTypeOf<GraphDataView<'float32'>>();
    expectTypeOf(
      new GPUAffineTransform({
        positions: vectorPair,
        parameters,
        output: {positions: vectorPairOutput}
      }).props.output.positions
    ).toEqualTypeOf<GraphVectorView<'float32x2'>>();
    expectTypeOf(
      new GPUGeodesicPairs({
        spatialContext,
        origins: vectorPair,
        targets: vectorPair,
        output: {distances: vectorScalarOutput}
      }).props.output.distances
    ).toEqualTypeOf<GraphVectorView<'float32'> | undefined>();
    expectTypeOf(
      new GPUGeodesicDestination({
        spatialContext,
        origins: vectorPair,
        bearings: vectorScalar,
        distances: vectorScalar,
        output: {destinations: vectorPairOutput}
      }).props.output.destinations
    ).toEqualTypeOf<GraphVectorView<'float32x2'> | undefined>();
    expectTypeOf(
      new GPUDatumGridTransform({
        ...datum,
        positions: vectorTriple,
        output: {positions: vectorTripleOutput, status: vectorStatus}
      }).props.output.positions
    ).toEqualTypeOf<GraphVectorView<'float32x3'>>();
    expectTypeOf(
      new GPUPointBuffer({
        points: vectorPair,
        parameters,
        offsets: vectorOffsets,
        verticesPerRing: 3,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      }).props.offsets
    ).toEqualTypeOf<GraphVectorView<'float32x2'>>();
    expectTypeOf(
      new GPUGaussianPredictiveScores({
        means: vectorScalar,
        variances: vectorScalar,
        observations: vectorScalar,
        output: {crps: vectorScalarOutput}
      }).props.output.crps
    ).toEqualTypeOf<GraphVectorView<'float32'>>();
    expectTypeOf<GPUAffineTransformProps['output']['positions']>().toEqualTypeOf<
      GraphDataView<'float32x2'>
    >();
    expectTypeOf<GPUAffineTransform['props']['output']['positions']>().toEqualTypeOf<
      GraphDataView<'float32x2'>
    >();
    expectTypeOf<GPUGeodesicPairsProps['output']['distances']>().toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    expectTypeOf<GPUGeodesicPairs['props']['output']['distances']>().toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    expectTypeOf<GPUGeodesicDestinationProps['output']['destinations']>().toEqualTypeOf<
      GraphDataView<'float32x2'> | undefined
    >();
    expectTypeOf<GPUGeodesicDestination['props']['output']['destinations']>().toEqualTypeOf<
      GraphDataView<'float32x2'> | undefined
    >();
    expectTypeOf<GPUDatumGridTransformProps['output']['positions']>().toEqualTypeOf<
      GraphDataView<'float32x3'>
    >();
    expectTypeOf<GPUDatumGridTransform['props']['output']['positions']>().toEqualTypeOf<
      GraphDataView<'float32x3'>
    >();
    expectTypeOf<GPUPointBufferProps['offsets']>().toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf<GPUPointBuffer['props']['offsets']>().toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf<GPUGaussianPredictiveScoresProps['output']['crps']>().toEqualTypeOf<
      GraphDataView<'float32'>
    >();
    expectTypeOf<GPUGaussianPredictiveScores['props']['output']['crps']>().toEqualTypeOf<
      GraphDataView<'float32'>
    >();
  } finally {
    device.destroy();
  }
});
