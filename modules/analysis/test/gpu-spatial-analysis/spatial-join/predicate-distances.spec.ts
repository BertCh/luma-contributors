// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildOracleArrays, type OracleFeature} from './spatial-predicate-oracle';
import fixture from './predicate-distances-fixture.json';

const featureSets = fixture.features as unknown as Record<string, OracleFeature[]>;
const scenes = fixture.scenes.map(scene => ({
  ...scene,
  left: featureSets[scene.leftKind],
  right: featureSets[scene.rightKind]
}));

function createHarness(
  device: Device,
  leftFeatures: OracleFeature[],
  rightFeatures: OracleFeature[],
  options: {
    predicate?: GPUSpatialPredicate;
    engine?: 'auto' | 'fast' | 'relate';
    pairCapacity?: number;
    candidateCapacity?: number;
    weightsOnly?: boolean;
    spatialContext?: GPUSpatialContext;
  } = {}
) {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device);
  const geometry = (prefix: string, features: OracleFeature[]): GPUSpatialJoinGeometry => {
    const arrays = buildOracleArrays(features[0].kind, features);
    const upload = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      data: Float32Array | Uint32Array,
      format: Format,
      length: number
    ) => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return importGraphBuffer(graph, `${prefix}-${name}`, buffer, format, length);
    };
    // Poison vertex rows outside every feature, on both sides of line/polygon joins.
    const padding = arrays.kind === 'points' ? 0 : 2;
    const positions = Float32Array.from([
      ...new Array(padding * 2).fill(Number.NaN),
      ...arrays.positions,
      ...new Array(padding * 2).fill(1e6)
    ]);
    const positionView = upload('positions', positions, 'float32x2', positions.length / 2);
    const offsets = (name: string, values: Uint32Array, shift = 0) =>
      upload(
        name,
        values.map(value => value + shift),
        'uint32',
        values.length
      );
    if (arrays.kind === 'points') {
      return {kind: 'points', positions: positionView};
    }
    if (arrays.kind === 'lines') {
      return {
        kind: 'lines',
        positions: positionView,
        lineOffsets: offsets('line-offsets', arrays.lineOffsets, padding)
      };
    }
    return {
      kind: 'polygons',
      positions: positionView,
      featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
      polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
      ringOffsets: offsets('ring-offsets', arrays.ringOffsets, padding)
    };
  };
  const capacity = options.pairCapacity ?? leftFeatures.length * rightFeatures.length + 3;
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const offsets = output('offsets', 'uint32', leftFeatures.length + 1);
  const neighbors = output('neighbors', 'uint32', capacity);
  const weights = output('weights', 'float32', capacity);
  const distances = output('distances', 'float32', capacity);
  const leftIds = output('left-ids', 'uint32', capacity);
  const rightIds = output('right-ids', 'uint32', capacity);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const overallOverflow = output('overall-overflow', 'uint32', 1);
  const requiredCount = output('required-count', 'uint32', 1);
  const candidateOverflow = output('candidate-overflow', 'uint32', 1);
  const weightsCount = output('weights-count', 'uint32', 1);
  const weightsRequiredCount = output('weights-required-count', 'uint32', 1);
  const weightsOverflow = output('weights-overflow', 'uint32', 1);
  const weightsCandidateOverflow = output('weights-candidate-overflow', 'uint32', 1);
  const threshold = createInputBuffer(device, Float32Array.from([5]));
  buffers.push(threshold);
  const predicate = options.predicate ?? 'dwithin';
  graph.add(
    new GPUSpatialPredicateJoin({
      left: geometry('left', leftFeatures),
      right: geometry('right', rightFeatures),
      predicate,
      engine: options.engine,
      distance: importGraphBuffer(graph, 'threshold', threshold, 'float32', 1),
      spatialContext: options.spatialContext,
      candidateCapacity: options.candidateCapacity ?? leftFeatures.length * rightFeatures.length,
      output: {
        status: {overflow: overallOverflow.view},
        pairs: options.weightsOnly
          ? undefined
          : {
              leftIds: leftIds.view,
              rightIds: rightIds.view,
              status: {
                count: count.view,
                overflow: overflow.view,
                requiredCount: requiredCount.view,
                candidateOverflow: candidateOverflow.view
              }
            },
        weights: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          distances: distances.view,
          status: {
            count: weightsCount.view,
            requiredCount: weightsRequiredCount.view,
            overflow: weightsOverflow.view,
            candidateOverflow: weightsCandidateOverflow.view
          }
        },
        // Exercise the eight-binding scatter (relate + distance pairs) as well.
        relate: predicate === 'touches' ? output('relate', 'uint32', capacity).view : undefined
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(distance = 5) {
      threshold.write(Float32Array.from([distance]));
      distances.buffer.write(new Float32Array(capacity).fill(-123));
      submitGraph(device, compiled, undefined);
      const rowOffsets = await readUint32(offsets.buffer, leftFeatures.length + 1);
      const neighborRows = await readUint32(neighbors.buffer, capacity);
      const distanceValues = await readFloat32(distances.buffer, capacity);
      const weightValues = await readFloat32(weights.buffer, capacity);
      const pairs: number[][] = [];
      for (let row = 0; row < leftFeatures.length; row++) {
        for (let slot = rowOffsets[row]; slot < rowOffsets[row + 1]; slot++) {
          pairs.push([row, neighborRows[slot], distanceValues[slot]]);
          expect(weightValues[slot]).toBe(1);
        }
      }
      expect(distanceValues.slice(pairs.length).every(Number.isNaN)).toBe(true);
      expect((await readUint32(weightsCount.buffer, 1))[0]).toBe(pairs.length);
      const finalOverflow = (await readUint32(weightsOverflow.buffer, 1))[0];
      const intermediateOverflow = (await readUint32(weightsCandidateOverflow.buffer, 1))[0];
      expect((await readUint32(overallOverflow.buffer, 1))[0]).toBe(
        Math.max(finalOverflow, intermediateOverflow)
      );
      if (!options.weightsOnly) {
        expect(await readUint32(weightsRequiredCount.buffer, 1)).toEqual(
          await readUint32(requiredCount.buffer, 1)
        );
        expect(await readUint32(weightsOverflow.buffer, 1)).toEqual(
          await readUint32(overflow.buffer, 1)
        );
        expect(await readUint32(weightsCandidateOverflow.buffer, 1)).toEqual(
          await readUint32(candidateOverflow.buffer, 1)
        );
        expect((await readUint32(count.buffer, 1))[0]).toBe(pairs.length);
        const leftRows = await readUint32(leftIds.buffer, pairs.length);
        const rightRows = await readUint32(rightIds.buffer, pairs.length);
        expect(pairs.map(pair => pair.slice(0, 2))).toEqual(
          leftRows.map((row, slot) => [row, rightRows[slot]])
        );
      }
      return {
        pairs,
        overflow: finalOverflow,
        requiredCount: (await readUint32(weightsRequiredCount.buffer, 1))[0],
        candidateOverflow: intermediateOverflow
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectPairs(actual: number[][], expected: number[][]) {
  expect(actual.map(pair => pair.slice(0, 2))).toEqual(expected.map(pair => pair.slice(0, 2)));
  actual.forEach((pair, index) => expect(pair[2]).toBeCloseTo(expected[index][2], 5));
}

it('predicate distance CSR matches Shapely for all nine geometry combinations, empty features, holes, multipart containment and padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of scenes) {
    const harness = createHarness(device, scene.left, scene.right);
    for (const threshold of [5, 0, 5]) {
      const actual = await harness.run(threshold);
      expectPairs(
        actual.pairs,
        scene.distances.filter(pair => pair[2] <= threshold)
      );
      expect(actual.overflow).toBe(0);
    }
    harness.destroy();
  }
});

it('topological, relate-engine and weights-only joins emit zero distances for matched closed geometries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = scenes.find(
    scene => scene.leftKind === 'polygons' && scene.rightKind === 'polygons'
  )!;
  for (const predicate of ['intersects', 'touches'] as const) {
    const harness = createHarness(device, scene.left, scene.right, {predicate});
    expectPairs((await harness.run()).pairs, scene[predicate]);
    harness.destroy();
  }
  const harness = createHarness(device, scene.left, scene.right, {weightsOnly: true});
  const expected = scene.distances.filter(pair => pair[2] <= 5);
  const matched = await harness.run();
  expectPairs(matched.pairs, expected);
  expect(matched.requiredCount).toBe(expected.length);
  expect(matched.overflow).toBe(0);
  expect(matched.candidateOverflow).toBe(0);
  const empty = await harness.run(-1);
  expectPairs(empty.pairs, []);
  expect(empty.requiredCount).toBe(0);
  expect(empty.overflow).toBe(0);
  expect(empty.candidateOverflow).toBe(0);
  harness.destroy();
});

it('distance output preserves pair overflow and candidate overflow as separate statuses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = scenes[0];
  for (const options of [{pairCapacity: 2}, {candidateCapacity: 2}]) {
    const harness = createHarness(device, scene.left, scene.right, options);
    const actual = await harness.run(50);
    expect(actual.overflow).toBe('pairCapacity' in options ? 1 : 0);
    expect(actual.candidateOverflow).toBe('candidateCapacity' in options ? 1 : 0);
    expectPairs(actual.pairs, scene.distances.slice(0, 2));
    harness.destroy();
  }
});

it('great-circle dwithin writes meters using a hand-derived equatorial arc', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const radius = 6371008.8;
  const harness = createHarness(
    device,
    [{kind: 'points', vertex: [0, 0]}],
    [{kind: 'points', vertex: [1, 0]}],
    {
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: radius
      }
    }
  );
  const result = await harness.run(120000);
  expect(result.pairs).toHaveLength(1);
  expect(Math.abs(result.pairs[0][2] - (radius * Math.PI) / 180)).toBeLessThan(0.1);
  harness.destroy();
});

it('dwithin engines agree at zero distance when only a later multipolygon component is contained', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely fixture features 2 and 5: geometry distance 0, boundary distance 0.5,
  // neither feature's first vertex is contained in the other feature.
  const [multipart, container] = [featureSets['polygons'][2], featureSets['polygons'][5]];
  for (const engine of ['fast', 'relate', 'auto'] as const) {
    for (const [left, right] of [
      [multipart, container],
      [container, multipart]
    ]) {
      const harness = createHarness(device, [left], [right], {engine});
      const actual = await harness.run(0);
      expect(actual.pairs, engine).toEqual([[0, 0, 0]]);
      expect(actual.requiredCount).toBe(1);
      expect(actual.overflow).toBe(0);
      expect(actual.candidateOverflow).toBe(0);
      harness.destroy();
    }
  }
});

it('dwithin engines emit representable large distances and reject pairs beyond a large threshold', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const large = Math.fround(1e20);
  const scenes: {left: OracleFeature; right: OracleFeature; expected: number}[] = [
    {
      left: {kind: 'points', vertex: [0, 0]},
      right: {kind: 'points', vertex: [large, 0]},
      expected: large
    },
    {
      left: {kind: 'points', vertex: [0, 0]},
      right: {kind: 'points', vertex: [Math.fround(6e19), Math.fround(8e19)]},
      expected: Math.hypot(Math.fround(6e19), Math.fround(8e19))
    },
    {
      left: {kind: 'points', vertex: [0, 0]},
      right: {
        kind: 'lines',
        vertices: [
          [large, -large],
          [large, large]
        ]
      },
      expected: large
    },
    {
      left: {
        kind: 'lines',
        vertices: [
          [0, -large],
          [0, large]
        ]
      },
      right: {
        kind: 'lines',
        vertices: [
          [large, -large],
          [large, large]
        ]
      },
      expected: large
    },
    {
      // Overlapping boxes keep this candidate alive even below the actual distance.
      left: {
        kind: 'lines',
        vertices: [
          [0, 0],
          [large, large]
        ]
      },
      right: {
        kind: 'lines',
        vertices: [
          [0, large],
          [large, 2 * large]
        ]
      },
      expected: large / Math.SQRT2
    },
    {
      left: {kind: 'points', vertex: [0, Math.fround(1e-20)]},
      right: {
        kind: 'lines',
        vertices: [
          [-Math.fround(1e38), 0],
          [Math.fround(1e38), 0]
        ]
      },
      expected: Math.fround(1e-20)
    },
    {
      left: {kind: 'points', vertex: [0, large]},
      right: {
        kind: 'lines',
        vertices: [
          [-Math.fround(1e38), 0],
          [Math.fround(1e38), 0]
        ]
      },
      expected: large
    }
  ];
  for (const engine of ['fast', 'relate'] as const) {
    for (const {left, right, expected} of scenes) {
      const harness = createHarness(device, [left], [right], {engine});
      for (const threshold of [2e20, 0.5e20, 2e20]) {
        const actual = await harness.run(threshold);
        if (threshold < expected) {
          expect(actual.pairs, `${engine} rejects distance above threshold`).toEqual([]);
          expect(actual.requiredCount).toBe(0);
        } else {
          expect(actual.pairs.map(pair => pair.slice(0, 2))).toEqual([[0, 0]]);
          // Closed forms on the uploaded f32 coordinates; allow 1e-5 relative f32 error.
          expect(
            Math.abs(actual.pairs[0][2] / expected - 1),
            `${engine} large distance`
          ).toBeLessThan(1e-5);
          expect(actual.requiredCount).toBe(1);
        }
        expect(actual.overflow).toBe(0);
        expect(actual.candidateOverflow).toBe(0);
      }
      harness.destroy();
    }
  }
});

it('dwithin engines preserve normal gaps when an asymmetric segment projection fraction underflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gap = Math.fround(1e-20);
  const near = Math.fround(1e-10);
  const far = Math.fround(1e38);
  // Shapely and the axis-aligned closed forms give gap for each pair. All inputs and outputs
  // are normal f32 values, although a fraction such as near / far underflows to zero.
  const scenes: {point: OracleFeature; vertices: [number, number][]}[] = [
    {
      point: {kind: 'points', vertex: [0, gap]},
      vertices: [
        [-near, 0],
        [far, 0]
      ]
    },
    {
      point: {kind: 'points', vertex: [gap, 0]},
      vertices: [
        [0, -near],
        [0, far]
      ]
    },
    // The tiny slope itself is unrepresentable too; its endpoint separation is still normal.
    {
      point: {kind: 'points', vertex: [far, 0]},
      vertices: [
        [0, 0],
        [far, gap]
      ]
    }
  ];
  for (const engine of ['fast', 'relate'] as const) {
    for (const scene of scenes) {
      for (const vertices of [scene.vertices, [...scene.vertices].reverse()]) {
        const segment: OracleFeature = {kind: 'lines', vertices};
        for (const [left, right] of [
          [scene.point, segment],
          [segment, scene.point]
        ]) {
          const harness = createHarness(device, [left], [right], {engine});
          const actual = await harness.run(2e20);
          expect(
            actual.pairs.map(pair => pair.slice(0, 2)),
            engine
          ).toEqual([[0, 0]]);
          expect(Math.abs(actual.pairs[0][2] / gap - 1), engine).toBeLessThan(1e-5);
          expect(actual.requiredCount).toBe(1);
          expect(actual.overflow).toBe(0);
          expect(actual.candidateOverflow).toBe(0);
          harness.destroy();
        }
      }
    }
  }
});
