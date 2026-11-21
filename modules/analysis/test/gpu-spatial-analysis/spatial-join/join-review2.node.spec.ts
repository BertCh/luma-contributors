// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslParser} from 'wgsl_reflect';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {adaptMultipartNearestUniforms} from './multipart-nearest-executor';
import {createJoinReview2Cases} from './join-review2-fixture';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  createCPUNearestFeatureJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {
  computeSpatialPredicateJoinOnCPU,
  computeNearestFeatureJoinOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {getGPUCommandNodeKernelSources} from '../../../src/testing/gpu-contributor-test-utils';

useCPUExecutorTestHooks();
for (const entry of createJoinReview2Cases())
  for (const squareRootShift of entry.name.startsWith('R1') && entry.expected['distances'][0] === 13
    ? [-1, 0, 1]
    : [0])
    it(`${entry.name}${squareRootShift ? ` publication sqrt shift=${squareRootShift} ULP` : ''}`, () => {
      const originalSquareRoot = Math.sqrt;
      // Decode sqrt(169 / 256) * 16 = 13. Emulate either adjacent f32 square-root
      // result, independently of the implementation's output, to cover GPU publication rounding.
      const squareRoot = squareRootShift
        ? vi
            .spyOn(Math, 'sqrt')
            .mockImplementation(value =>
              value === 169 / 256 ? 13 / 16 + squareRootShift * 2 ** -24 : originalSquareRoot(value)
            )
        : undefined;
      const device = createNullWebGPUDevice(),
        graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      const input = Object.fromEntries(
        Object.entries(entry.inputs).map(([name, value]) => {
          const view = createTransientView(
            graph,
            `input-${name}`,
            value.format,
            value.values.length / (value.format === 'float32x2' ? 2 : 1)
          );
          memory.write(view, value.values);
          return [name, view];
        })
      );
      const output = Object.fromEntries(
        Object.entries(entry.outputs).map(([name, value]) => {
          const view = createTransientView(graph, `output-${name}`, value.format, value.length);
          memory.write(
            view,
            value.format === 'uint32'
              ? new Uint32Array(value.length).fill(99)
              : new Float32Array(value.length * (value.format === 'float32x2' ? 2 : 1)).fill(NaN)
          );
          return [name, view];
        })
      );
      try {
        const nodes = entry.declare(graph, input, output);
        if (entry.cooperative) {
          const kernels = getGPUCommandNodeKernelSources(nodes);
          // Parse the actual cooperative shaders even though this executor cannot simulate barriers.
          for (const kernel of kernels) new WgslParser().parse(kernel.source);
          if (entry.name.startsWith('R5')) {
            const source = kernels.map(kernel => kernel.source).join('\n');
            expect(source).toContain('dwithinUncertain');
            expect(source).toContain('flag & 2u');
          }
        } else
          for (let submission = 0; submission < 2; submission++) {
            executeCommandNodesOnCPU(adaptMultipartNearestUniforms(nodes), memory);
            for (const [name, expected] of Object.entries(entry.expected)) {
              const actual =
                output[name].format === 'uint32'
                  ? memory.readUint32(output[name])
                  : memory.readFloat32(output[name]);
              if (output[name].format === 'uint32')
                expect(Array.from(actual), name).toEqual(expected);
              else
                expected.forEach((value, row) => {
                  expect(Number.isFinite(actual[row]), name).toBe(true);
                  expect(Math.abs(actual[row] - value), name).toBeLessThanOrEqual(
                    entry.tolerance ?? 0
                  );
                });
            }
          }
        if (squareRoot)
          expect(squareRoot.mock.calls.some(([value]) => value === 169 / 256)).toBe(true);
      } finally {
        squareRoot?.mockRestore();
        device.destroy();
      }
    });
for (const create of [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation])
  for (const key of ['count', 'requiredCount', 'overflow', 'candidateOverflow'])
    for (const absent of [true, false])
      it(`R6 ${create.name} validates ${key} absent=${absent}`, () => {
        const output: Record<string, string | undefined> = {
          leftRows: 'leftRows',
          rightRows: 'rightRows',
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        };
        if (absent) delete output[key];
        else output[key] = undefined;
        expect(() =>
          create({
            left: {kind: 'points', positions: {value: 'l', length: 1}},
            right: {kind: 'points', positions: {value: 'r', length: 1}},
            predicate: 'intersects',
            candidateCapacity: 1,
            pairCapacity: 1,
            output: output as never
          })
        ).toThrow(new RegExp(`output[s]?\\.${key}.*non-empty`));
      });
const vector = (values: Float32Array | Uint32Array, positions = false): GPUCPUVector => {
  const length = values.length / (positions ? 2 : 1);
  return {
    format: positions ? 'float32x2' : 'uint32',
    length,
    chunks: [{values, length}],
    pack: () => values
  };
};
for (const reverse of [false, true])
  it(`R7 exact endpoint classification reverse=${reverse}`, async () => {
    const road = Float32Array.from(reverse ? [0, 0, -(2 ** 54), 0] : [-(2 ** 54), 0, 0, 0]);
    const left = {kind: 'points' as const, positions: Float32Array.of(-1, 1)},
      right = {kind: 'lines' as const, positions: road, lineOffsets: Uint32Array.of(0, 2)};
    expect(
      computeSpatialPredicateJoinOnCPU(left, right, {
        predicate: 'dwithin',
        distance: 1,
        candidateCapacity: 1,
        pairCapacity: 1
      }).status.count
    ).toBe(1);
    const operation = createCPUSpatialPredicateJoinOperation({
      left: {kind: 'points', positions: {value: 'left', length: 1}},
      right: {
        kind: 'lines',
        positions: {value: 'right', length: 2},
        lineOffsets: {value: 'offsets', length: 2}
      },
      predicate: 'dwithin',
      distance: 1,
      candidateCapacity: 1,
      pairCapacity: 1,
      output: {
        leftRows: 'l',
        rightRows: 'r',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    });
    const result = (await operation.execute({
      inputs: {
        leftPositions: vector(left.positions, true),
        rightPositions: vector(road, true),
        rightLineOffsets: vector(right.lineOffsets)
      }
    })) as Record<string, Uint32Array>;
    for (const [name, expected] of Object.entries({
      l: [0],
      r: [0],
      count: [1],
      requiredCount: [1],
      overflow: [0],
      candidateOverflow: [0]
    })) {
      expect(Array.from(result[name]), name).toEqual(expected);
    }
  });
for (const initialOrigin of [undefined, [7, 9] as [number, number]])
  it(`R8 aliased contexts retain a single frame snapshot origin=${initialOrigin}`, () => {
    let ids = 0,
      origins = 0;
    const frame = {
      get id() {
        return ++ids === 1 ? 'crs1' : 'crs2';
      },
      get origin(): [number, number] | undefined {
        return ++origins === 1 ? initialOrigin : [17, 19];
      },
      kind: 'crs' as const
    };
    const context = {coordinateSpace: 'planar' as const, metric: 'native' as const, frame};
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    try {
      const join = new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions},
        features: {kind: 'points', positions},
        spatialContext: context,
        featuresSpatialContext: context,
        output: {
          neighborIds: createTransientView(graph, 'ids', 'uint32', 1),
          neighborCounts: createTransientView(graph, 'counts', 'uint32', 1),
          status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
        }
      });
      expect(ids).toBe(1);
      expect(origins).toBe(1);
      expect(join.props.spatialContext!.frame!.id).toBe('crs1');
      expect(join.props.spatialContext!.frame!.origin).toEqual(initialOrigin);
      expect(join.props.featuresSpatialContext).toBe(join.props.spatialContext);
    } finally {
      device.destroy();
    }
  });

for (const reverse of [false, true])
  it(`R2 fractional tie CPU forms reverse=${reverse}`, async () => {
    const first = reverse ? [Math.fround(1.1), 2.5] : [-5, -10],
      second = reverse ? [-5, -10] : [Math.fround(1.1), 2.5];
    const queries = {kind: 'points' as const, positions: Float32Array.of(0, 0)};
    const features = {
      kind: 'lines' as const,
      positions: Float32Array.from([...first, ...second, ...second, ...first]),
      lineOffsets: Uint32Array.of(0, 2, 4),
      sourceIds: Uint32Array.of(33, 42)
    };
    const expected = {
      neighborRows: [0],
      neighborIds: [33],
      neighborCounts: [1],
      neighborDistances: [Math.fround(0.10784390321559371)]
    };
    const result = computeNearestFeatureJoinOnCPU(queries, features);
    for (const [name, values] of Object.entries(expected))
      expect(Array.from(result[name as keyof typeof expected])).toEqual(values);
    const operation = createCPUNearestFeatureJoinOperation({
      queries: {kind: 'points', positions: {value: 'q', length: 1}},
      features: {
        kind: 'lines',
        positions: {value: 'p', length: 4},
        lineOffsets: {value: 'l', length: 3},
        sourceIds: {value: 's', length: 2}
      },
      output: {
        neighborRows: 'neighborRows',
        neighborIds: 'neighborIds',
        neighborCounts: 'neighborCounts',
        neighborDistances: 'neighborDistances',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    });
    const arrays = (await operation.execute({
      inputs: {
        queriesPositions: vector(queries.positions, true),
        featuresPositions: vector(features.positions, true),
        featuresLineOffsets: vector(features.lineOffsets),
        featuresSourceIds: vector(features.sourceIds)
      }
    })) as Record<string, Uint32Array | Float32Array>;
    for (const [name, values] of Object.entries({
      ...expected,
      overflow: [0],
      candidateOverflow: [0]
    }))
      expect(Array.from(arrays[name])).toEqual(values);
  });

for (const reverse of [false, true])
  it(`R9 original-coordinate gap CPU forms reverse=${reverse}`, async () => {
    const left = {kind: 'points' as const, positions: Float32Array.of(0, 0)};
    const right = {
      kind: 'lines' as const,
      positions: Float32Array.from(
        reverse ? [2 ** -10, 2 ** -9, -65536, -65536] : [-65536, -65536, 2 ** -10, 2 ** -9]
      ),
      lineOffsets: Uint32Array.of(0, 2)
    };
    const options = {
      predicate: 'dwithin' as const,
      distance: 2 ** -11,
      candidateCapacity: 1,
      pairCapacity: 1
    };
    expect(computeSpatialPredicateJoinOnCPU(left, right, options).status.count).toBe(0);
    const operation = createCPUSpatialPredicateJoinOperation({
      ...options,
      left: {kind: 'points', positions: {value: 'l', length: 1}},
      right: {
        kind: 'lines',
        positions: {value: 'r', length: 2},
        lineOffsets: {value: 'o', length: 2}
      },
      output: {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        uncertainCount: 'uncertainCount'
      }
    });
    const arrays = (await operation.execute({
      inputs: {
        leftPositions: vector(left.positions, true),
        rightPositions: vector(right.positions, true),
        rightLineOffsets: vector(right.lineOffsets)
      }
    })) as Record<string, Uint32Array>;
    for (const name of [
      'count',
      'requiredCount',
      'overflow',
      'candidateOverflow',
      'uncertainCount'
    ])
      expect(Array.from(arrays[name])).toEqual([0]);
  });

for (const engine of ['fast', 'relate'] as const)
  for (const grouped of [false, true])
    it(`R5 exact CPU dwithin uncertainty engine=${engine} grouped=${grouped}`, async () => {
      const large = 2 ** 120,
        displacement = 2 ** 98,
        small = 2 ** -100;
      const left = {
        kind: 'lines' as const,
        positions: Float32Array.of(
          large / 4,
          large / 4 + displacement,
          large / 2,
          large / 2 + displacement
        ),
        lineOffsets: Uint32Array.of(0, 2),
        featureOffsets: Uint32Array.of(0, 1)
      };
      const right = {
        kind: 'lines' as const,
        positions: Float32Array.of(small, small, large, large),
        lineOffsets: Uint32Array.of(0, 2),
        ...(grouped ? {featureOffsets: Uint32Array.of(0, 1)} : {})
      };
      const options = {
        predicate: 'dwithin' as const,
        distance: 0,
        engine,
        candidateCapacity: 8,
        pairCapacity: 1
      };
      expect(computeSpatialPredicateJoinOnCPU(left, right, options).status.count).toBe(0);
      const geometry = {
        kind: 'lines' as const,
        positions: {value: 'positions', length: 2},
        lineOffsets: {value: 'lines', length: 2}
      };
      const operation = createCPUSpatialPredicateJoinOperation({
        ...options,
        left: {...geometry, featureOffsets: {value: 'features', length: 2}},
        right: {...geometry, ...(grouped ? {featureOffsets: {value: 'features', length: 2}} : {})},
        output: {
          leftRows: 'leftRows',
          rightRows: 'rightRows',
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow',
          uncertainCount: 'uncertainCount'
        }
      });
      const arrays = (await operation.execute({
        inputs: {
          leftPositions: vector(left.positions, true),
          leftLineOffsets: vector(left.lineOffsets),
          leftFeatureOffsets: vector(left.featureOffsets),
          rightPositions: vector(right.positions, true),
          rightLineOffsets: vector(right.lineOffsets),
          ...(grouped ? {rightFeatureOffsets: vector(right.featureOffsets!)} : {})
        }
      })) as Record<string, Uint32Array>;
      for (const name of [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'uncertainCount'
      ])
        expect(Array.from(arrays[name]), name).toEqual([0]);
    });
