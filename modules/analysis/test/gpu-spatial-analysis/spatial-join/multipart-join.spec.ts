// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation,
  createCPUNearestFeatureJoinOperation,
  type FeatureJoinGeometryValues
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

function describeGeometry(
  prefix: string,
  kind: 'points' | 'lines' | 'polygons',
  arrays: Record<string, Float32Array | Uint32Array>,
  inputs: Record<string, OperationGraphInput>
): FeatureJoinGeometryValues {
  return {
    kind,
    ...Object.fromEntries(
      Object.entries(arrays).map(([field, values]) => {
        const value = prefix + field[0].toUpperCase() + field.slice(1),
          length = values.length / (field === 'positions' ? 2 : 1);
        inputs[value] = {format: field === 'positions' ? 'float32x2' : 'uint32', values};
        return [field, {value, length}];
      })
    )
  } as FeatureJoinGeometryValues;
}

for (const metric of ['planar', 'great-circle'] as const)
  it(`Metal nearest ${metric} compiles with a radius and recovers after invalid values`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        queries: {format: 'float32x2', values: Float32Array.of(0, 0)},
        features: {format: 'float32x2', values: Float32Array.of(0, 0)},
        radius: {format: 'float32', values: Float32Array.of(0)}
      },
      outputs: {
        rows: {format: 'uint32', length: 1},
        ids: {format: 'uint32', length: 1},
        counts: {format: 'uint32', length: 1},
        distances: {format: 'float32', length: 1},
        overflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) =>
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: inputs['queries'] as GraphDataView<'float32x2'>},
          features: {
            kind: 'points',
            positions: inputs['features'] as GraphDataView<'float32x2'>
          },
          spatialContext:
            metric === 'planar'
              ? {coordinateSpace: 'planar', metric: 'native'}
              : {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
          radius: inputs['radius'] as GraphDataView<'float32'>,
          spatialSort: false,
          output: {
            neighborRows: outputs['rows'] as GraphDataView<'uint32'>,
            neighborIds: outputs['ids'] as GraphDataView<'uint32'>,
            neighborCounts: outputs['counts'] as GraphDataView<'uint32'>,
            neighborDistances: outputs['distances'] as GraphDataView<'float32'>,
            status: {overflow: outputs['overflow'] as GraphDataView<'uint32'>}
          }
        })
    });
    try {
      for (const limit of [0, -1, NaN, Infinity, -Infinity, Math.fround(3.402823466e38), 0]) {
        fixture.writeInput('radius', Float32Array.of(limit));
        const actual = await fixture.run();
        const valid = Number.isFinite(limit) && limit >= 0;
        expect(actual['counts']).toEqual([Number(valid)]);
        expect(actual['rows']).toEqual([valid ? 0 : 0xffffffff]);
        expect(actual['ids']).toEqual([valid ? 0 : 0xffffffff]);
        expect(actual['distances']).toEqual([valid ? 0 : -1]);
        expect(actual['overflow']).toEqual([0]);
      }
      fixture.writeInput('features', Float32Array.of(1, 0));
      for (const limit of [0, -0, 0]) {
        fixture.writeInput('radius', Float32Array.of(limit));
        const actual = await fixture.run();
        expect(actual['counts']).toEqual([0]);
        expect(actual['rows']).toEqual([0xffffffff]);
        expect(actual['distances']).toEqual([-1]);
      }
      expect(fixture.getCompileCount()).toBe(0);
    } finally {
      fixture.destroy();
    }
  });

for (const predicate of ['within', 'coveredBy'] as const)
  for (const interior of [true, false])
    it(`x1 X1 ${predicate} oblique polygon boundary with interior=${interior} in both forms`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const inputs: Record<string, OperationGraphInput> = {};
      const left = describeGeometry(
        'left',
        'lines',
        {
          positions: Float32Array.from(interior ? [0, 0, 3, 9, 0, 5, 1, 6] : [0, 0, 3, 9]),
          lineOffsets: Uint32Array.from(interior ? [0, 2, 4] : [0, 2]),
          featureOffsets: Uint32Array.from(interior ? [0, 2] : [0, 1])
        },
        inputs
      );
      const right = describeGeometry(
        'right',
        'polygons',
        {
          positions: Float32Array.of(0, 0, 2, 6, 3, 9, 0, 9),
          featureOffsets: Uint32Array.of(0, 1),
          polygonOffsets: Uint32Array.of(0, 1),
          ringOffsets: Uint32Array.of(0, 4)
        },
        inputs
      );
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      for (const create of [
        createSpatialPredicateJoinOperation,
        createCPUSpatialPredicateJoinOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs,
          operations: [
            create({left, right, predicate, candidateCapacity: 8, pairCapacity: 8, output})
          ],
          outputs: Object.values(output)
        });
        const count = Number(interior || predicate === 'coveredBy');
        expect(Array.from(result['count'])).toEqual([count]);
        if (count) {
          expect(result['leftRows'][0]).toBe(0);
          expect(result['rightRows'][0]).toBe(0);
        }
        expect(Array.from(result['overflow'])).toEqual([0]);
        expect(Array.from(result['candidateOverflow'])).toEqual([0]);
      }
    });

for (const reverse of [false, true])
  for (const hit of [false, true])
    it(`x1 X2 compensated road nearest reverse=${reverse}, coincident target=${hit}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const road = reverse ? [1, 1, -8191, -8190] : [-8191, -8190, 1, 1];
      const fixture = createGeometryFixture(device, {
        inputs: {
          queries: {format: 'float32x2', values: Float32Array.of(0, 0)},
          targets: {
            format: 'float32x2',
            values: Float32Array.from([...road, 0, -2, 1, -2, ...(hit ? [0, 0, 1, 0] : [])])
          },
          lines: {format: 'uint32', values: Uint32Array.from(hit ? [0, 2, 4, 6] : [0, 2, 4])},
          features: {format: 'uint32', values: Uint32Array.from(hit ? [0, 2, 3] : [0, 2])},
          radius: {format: 'float32', values: Float32Array.of(hit ? 1e6 : 0)}
        },
        outputs: {
          rows: {format: 'uint32', length: 1},
          counts: {format: 'uint32', length: 1},
          ids: {format: 'uint32', length: 1},
          distances: {format: 'float32', length: 1},
          overflow: {format: 'uint32', length: 1}
        },
        create: ({inputs, outputs}) =>
          new GPUNearestFeatureJoin({
            queries: {kind: 'points', positions: inputs['queries'] as GraphDataView<'float32x2'>},
            features: {
              kind: 'lines',
              positions: inputs['targets'] as GraphDataView<'float32x2'>,
              lineOffsets: inputs['lines'] as GraphDataView<'uint32'>,
              featureOffsets: inputs['features'] as GraphDataView<'uint32'>
            },
            radius: inputs['radius'] as GraphDataView<'float32'>,
            spatialSort: false,
            output: {
              neighborRows: outputs['rows'] as GraphDataView<'uint32'>,
              neighborIds: outputs['ids'] as GraphDataView<'uint32'>,
              neighborCounts: outputs['counts'] as GraphDataView<'uint32'>,
              neighborDistances: outputs['distances'] as GraphDataView<'float32'>,
              status: {overflow: outputs['overflow'] as GraphDataView<'uint32'>}
            }
          })
      });
      try {
        const actual = await fixture.run();
        expect(actual['rows']).toEqual([hit ? 1 : 0xffffffff]);
        expect(actual['counts']).toEqual([Number(hit)]);
        expect(actual['distances']).toEqual([hit ? 0 : -1]);
        expect(actual['overflow']).toEqual([0]);
      } finally {
        fixture.destroy();
      }
    });

for (const scene of ['small-20', 'small-24', 'sub-unit', 'endpoint', 'reversed-endpoint'] as const)
  it(`x1 X5/X6 nearest ${scene} agrees in both operation forms`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const small = scene.startsWith('small');
    const points = small || scene === 'sub-unit';
    const inputs: Record<string, OperationGraphInput> = {};
    const queries = describeGeometry(
      'queries',
      'points',
      {positions: Float32Array.of(points ? 0 : 1, 0)},
      inputs
    );
    const features = describeGeometry(
      'features',
      points ? 'points' : 'lines',
      {
        positions: Float32Array.from(
          scene === 'sub-unit'
            ? [0.5, 0.00015, 0.5, 0]
            : small
              ? [scene === 'small-20' ? 1e-20 : 1e-24, 0, 0, 0]
              : scene === 'endpoint'
                ? [-1e20, 0, 0, 0]
                : [0, 0, -1e20, 0]
        ),
        ...(!points ? {lineOffsets: Uint32Array.of(0, 2)} : {})
      },
      inputs
    );
    const output = {
      neighborRows: 'rows',
      neighborIds: 'ids',
      neighborCounts: 'counts',
      neighborDistances: 'distances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    for (const create of [
      createNearestFeatureJoinOperation,
      createCPUNearestFeatureJoinOperation
    ]) {
      const actual = await runOperationGraph(device, {
        inputs,
        operations: [create({queries, features, output})],
        outputs: Object.values(output)
      });
      expect(Array.from(actual['rows'])).toEqual([points ? 1 : 0]);
      expect(Array.from(actual['counts'])).toEqual([1]);
      expect(Array.from(actual['distances'])).toEqual([scene === 'sub-unit' ? 0.5 : small ? 0 : 1]);
      expect(Array.from(actual['overflow'])).toEqual([0]);
    }
  });

for (const distance of [1e-20, 1e-24])
  it(`x1 X5 zero radius rejects a small nonzero target at ${distance}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        queries: {format: 'float32x2', values: Float32Array.of(0, 0)},
        targets: {format: 'float32x2', values: Float32Array.of(distance, 0)},
        radius: {format: 'float32', values: Float32Array.of(0)}
      },
      outputs: {
        rows: {format: 'uint32', length: 1},
        counts: {format: 'uint32', length: 1},
        ids: {format: 'uint32', length: 1},
        distances: {format: 'float32', length: 1},
        overflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) =>
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: inputs['queries'] as GraphDataView<'float32x2'>},
          features: {kind: 'points', positions: inputs['targets'] as GraphDataView<'float32x2'>},
          radius: inputs['radius'] as GraphDataView<'float32'>,
          spatialSort: false,
          output: {
            neighborRows: outputs['rows'] as GraphDataView<'uint32'>,
            neighborIds: outputs['ids'] as GraphDataView<'uint32'>,
            neighborCounts: outputs['counts'] as GraphDataView<'uint32'>,
            neighborDistances: outputs['distances'] as GraphDataView<'float32'>,
            status: {overflow: outputs['overflow'] as GraphDataView<'uint32'>}
          }
        })
    });
    try {
      const actual = await fixture.run();
      expect(actual['rows']).toEqual([0xffffffff]);
      expect(actual['counts']).toEqual([0]);
      expect(actual['distances']).toEqual([-1]);
    } finally {
      fixture.destroy();
    }
  });

for (const grouped of [false, true])
  it(`x1 X7 uncertainty survives identity grouping on Metal (grouped=${grouped})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const large = 2 ** 120,
      delta = 2 ** 98,
      small = 2 ** -100;
    const fixture = createGeometryFixture(device, {
      inputs: {
        left: {
          format: 'float32x2',
          values: Float32Array.of(large / 4, large / 4 + delta, large / 2, large / 2 + delta)
        },
        right: {format: 'float32x2', values: Float32Array.of(small, small, large, large)},
        leftLines: {format: 'uint32', values: Uint32Array.of(0, 2)},
        rightLines: {format: 'uint32', values: Uint32Array.of(0, 2)},
        leftFeatures: {format: 'uint32', values: Uint32Array.of(0, 1)},
        ...(grouped
          ? {rightFeatures: {format: 'uint32' as const, values: Uint32Array.of(0, 1)}}
          : {})
      },
      outputs: {
        leftRows: {format: 'uint32', length: 1},
        rightRows: {format: 'uint32', length: 1},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        uncertain: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) =>
        new GPUSpatialPredicateJoin({
          left: {
            kind: 'lines',
            positions: inputs['left'] as GraphDataView<'float32x2'>,
            lineOffsets: inputs['leftLines'] as GraphDataView<'uint32'>,
            featureOffsets: inputs['leftFeatures'] as GraphDataView<'uint32'>
          },
          right: {
            kind: 'lines',
            positions: inputs['right'] as GraphDataView<'float32x2'>,
            lineOffsets: inputs['rightLines'] as GraphDataView<'uint32'>,
            ...(grouped ? {featureOffsets: inputs['rightFeatures'] as GraphDataView<'uint32'>} : {})
          },
          predicate: 'intersects',
          candidateCapacity: 8,
          output: {
            pairs: {
              leftRows: outputs['leftRows'] as GraphDataView<'uint32'>,
              rightRows: outputs['rightRows'] as GraphDataView<'uint32'>,
              status: {
                count: outputs['count'] as GraphDataView<'uint32'>,
                overflow: outputs['overflow'] as GraphDataView<'uint32'>
              }
            },
            status: {uncertainCount: outputs['uncertain'] as GraphDataView<'uint32'>}
          }
        })
    });
    try {
      const actual = await fixture.run();
      expect(actual['count']).toEqual([1]);
      expect(actual['uncertain']).toEqual([1]);
      expect(actual['overflow']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });

for (const form of ['predicate', 'nearest'] as const)
  for (const field of ['positions', 'featureOffsets'] as const)
    it(`review R7 ${form} forms reject oversized ${field} graph values equally`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const inputs: Record<string, OperationGraphInput> = {};
      const left = describeGeometry(
        'left',
        'points',
        {
          positions: field === 'positions' ? Float32Array.of(0, 0, 2, 0) : Float32Array.of(0, 0),
          featureOffsets:
            field === 'featureOffsets' ? Uint32Array.of(0, 1, 1) : Uint32Array.of(0, 1)
        },
        inputs
      );
      const declared = {...left, [field]: {...left[field]!, length: field === 'positions' ? 1 : 2}};
      const right = describeGeometry(
        'right',
        'points',
        {positions: Float32Array.of(0, 0), featureOffsets: Uint32Array.of(0, 1)},
        inputs
      );
      const predicateOutput = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      const nearestOutput = {
        neighborRows: 'neighborRows',
        neighborIds: 'neighborIds',
        neighborCounts: 'neighborCounts',
        neighborDistances: 'neighborDistances',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      const operations =
        form === 'predicate'
          ? [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation].map(
              create =>
                create({
                  left: declared,
                  right,
                  predicate: 'intersects',
                  candidateCapacity: 8,
                  pairCapacity: 8,
                  output: predicateOutput
                })
            )
          : [createNearestFeatureJoinOperation, createCPUNearestFeatureJoinOperation].map(create =>
              create({queries: declared, features: right, output: nearestOutput})
            );
      for (const operation of operations)
        await expect(
          runOperationGraph(device, {
            inputs,
            operations: [operation],
            outputs: Object.values(form === 'predicate' ? predicateOutput : nearestOutput)
          })
        ).rejects.toThrow(/must match its declared length/);
    });

for (const predicate of ['within', 'coveredBy'] as const)
  for (const fixture of ['oblique', 'gap', 'closed-gap'] as const)
    it(`review R1/R2 ${predicate}: ${fixture} uses supplied endpoint intervals in both operation forms`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const inputs: Record<string, OperationGraphInput> = {};
      const left = describeGeometry(
        'left',
        'lines',
        {
          positions: Float32Array.from(fixture === 'oblique' ? [0, 0, 3, 9] : [-1e8, 0, 1e8, 0]),
          lineOffsets: Uint32Array.of(0, 2)
        },
        inputs
      );
      const right = describeGeometry(
        'right',
        'lines',
        {
          positions: Float32Array.from(
            fixture === 'oblique'
              ? [0, 0, 2, 6, 2, 6, 3, 9]
              : [-1e8, 0, 0, 0, fixture === 'gap' ? 1 : 0, 0, 1e8, 0]
          ),
          lineOffsets: Uint32Array.of(0, 2, 4),
          featureOffsets: Uint32Array.of(0, 2)
        },
        inputs
      );
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      for (const create of [
        createSpatialPredicateJoinOperation,
        createCPUSpatialPredicateJoinOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs,
          operations: [
            create({left, right, predicate, candidateCapacity: 8, pairCapacity: 8, output})
          ],
          outputs: Object.values(output)
        });
        const expected = fixture === 'gap' ? 0 : 1;
        expect(Array.from(result['count'])).toEqual([expected]);
        expect(Array.from(result['requiredCount'])).toEqual([expected]);
        if (expected) expect([result['leftRows'][0], result['rightRows'][0]]).toEqual([0, 0]);
        expect(Array.from(result['candidateOverflow'])).toEqual([0]);
        expect(Array.from(result['overflow'])).toEqual([0]);
      }
    });

for (const target of [
  [1, 0.0003, 1, 0],
  [1e20, 0]
])
  it(`review R4/R5 nearest ranks targets ${target} before rounding published distances`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs: Record<string, OperationGraphInput> = {};
    const queries = describeGeometry(
      'queries',
      'points',
      {positions: Float32Array.of(0, 0, -10, 0), featureOffsets: Uint32Array.of(0, 2)},
      inputs
    );
    const features = describeGeometry(
      'features',
      'points',
      {positions: Float32Array.from(target)},
      inputs
    );
    const output = {
      neighborRows: 'neighborRows',
      neighborIds: 'neighborIds',
      neighborCounts: 'neighborCounts',
      neighborDistances: 'neighborDistances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    for (const create of [
      createNearestFeatureJoinOperation,
      createCPUNearestFeatureJoinOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs,
        operations: [create({queries, features, output})],
        outputs: Object.values(output)
      });
      expect(Array.from(result['neighborRows'])).toEqual([target.length === 4 ? 1 : 0]);
      expect(Array.from(result['neighborCounts'])).toEqual([1]);
      expect(Number.isFinite(result['neighborDistances'][0])).toBe(true);
      expect(Array.from(result['neighborDistances'])).toEqual([
        Math.fround(target.length === 4 ? 1 : 1e20)
      ]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['candidateOverflow'])).toEqual([0]);
    }
  });

for (const empty of ['queries', 'features'] as const)
  it(`review R9 nearest forms execute zero ${empty} rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs: Record<string, OperationGraphInput> = {};
    const queries = describeGeometry(
      'queries',
      'points',
      {
        positions: empty === 'queries' ? new Float32Array(0) : Float32Array.of(0, 0),
        featureOffsets: empty === 'queries' ? Uint32Array.of(0) : Uint32Array.of(0, 1)
      },
      inputs
    );
    const features = describeGeometry(
      'features',
      'points',
      {
        positions: empty === 'features' ? new Float32Array(0) : Float32Array.of(0, 0),
        featureOffsets: empty === 'features' ? Uint32Array.of(0) : Uint32Array.of(0, 1)
      },
      inputs
    );
    const output = {
      neighborRows: 'neighborRows',
      neighborIds: 'neighborIds',
      neighborCounts: 'neighborCounts',
      neighborDistances: 'neighborDistances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    for (const create of [
      createNearestFeatureJoinOperation,
      createCPUNearestFeatureJoinOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs,
        operations: [create({queries, features, output})],
        outputs: Object.values(output)
      });
      expect(Array.from(result['neighborRows'])).toEqual(empty === 'queries' ? [] : [0xffffffff]);
      expect(Array.from(result['neighborCounts'])).toEqual(empty === 'queries' ? [] : [0]);
      expect(Array.from(result['neighborDistances'])).toEqual(empty === 'queries' ? [] : [-1]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['candidateOverflow'])).toEqual([0]);
    }
  });

for (const predicate of ['within', 'coveredBy'] as const)
  it(`review R2 ${predicate} closes a real gap on the same compiled submission`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        queryPositions: {format: 'float32x2', values: Float32Array.of(-1e8, 0, 1e8, 0)},
        queryLines: {format: 'uint32', values: Uint32Array.of(0, 2)},
        targetPositions: {
          format: 'float32x2',
          values: Float32Array.of(-1e8, 0, 0, 0, 1, 0, 1e8, 0)
        },
        targetLines: {format: 'uint32', values: Uint32Array.of(0, 2, 4)},
        targetFeatures: {format: 'uint32', values: Uint32Array.of(0, 2)}
      },
      outputs: {
        leftRows: {format: 'uint32', length: 1},
        rightRows: {format: 'uint32', length: 1},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) =>
        new GPUSpatialPredicateJoin({
          left: {
            kind: 'lines',
            positions: inputs['queryPositions'] as GraphDataView<'float32x2'>,
            lineOffsets: inputs['queryLines'] as GraphDataView<'uint32'>
          },
          right: {
            kind: 'lines',
            positions: inputs['targetPositions'] as GraphDataView<'float32x2'>,
            lineOffsets: inputs['targetLines'] as GraphDataView<'uint32'>,
            featureOffsets: inputs['targetFeatures'] as GraphDataView<'uint32'>
          },
          predicate,
          candidateCapacity: 8,
          output: {
            pairs: {
              leftRows: outputs['leftRows'] as GraphDataView<'uint32'>,
              rightRows: outputs['rightRows'] as GraphDataView<'uint32'>,
              status: {
                count: outputs['count'] as GraphDataView<'uint32'>,
                overflow: outputs['overflow'] as GraphDataView<'uint32'>
              }
            }
          }
        })
    });
    try {
      expect((await fixture.run())['count']).toEqual([0]);
      fixture.writeInput('targetPositions', Float32Array.of(-1e8, 0, 0, 0, 0, 0, 1e8, 0));
      const result = await fixture.run();
      expect(result['count']).toEqual([1]);
      expect(result['leftRows']).toEqual([0]);
      expect(result['rightRows']).toEqual([0]);
      expect(result['overflow']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });

for (const kind of ['points', 'lines'] as const)
  for (const predicate of ['intersects', 'within', 'coveredBy'] as const) {
    it(`GPU and CPU operation forms preserve multipart ${kind} feature rows for ${predicate}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const inputs: Record<string, OperationGraphInput> = {};
      const left = describeGeometry(
        'left',
        kind,
        {
          positions: Float32Array.from(
            kind === 'points' ? [-10, 0, 2, 0, 30, 0] : [-10, 0, -9, 0, 2, 0, 3, 0, 30, 0, 31, 0]
          ),
          featureOffsets: Uint32Array.of(0, 2, 3),
          ...(kind === 'lines' ? {lineOffsets: Uint32Array.of(0, 2, 4, 6)} : {})
        },
        inputs
      );
      const right = describeGeometry(
        'right',
        'polygons',
        {
          positions: Float32Array.of(0, -1, 5, -1, 5, 1, 0, 1, 20, -1, 40, -1, 40, 1, 20, 1),
          featureOffsets: Uint32Array.of(0, 1, 2),
          polygonOffsets: Uint32Array.of(0, 1, 2),
          ringOffsets: Uint32Array.of(0, 4, 8)
        },
        inputs
      );
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateTruncated: 'candidateTruncated'
      };
      const expected =
        predicate === 'intersects'
          ? [
              [0, 0],
              [1, 1]
            ]
          : [[1, 1]];
      for (const create of [
        createSpatialPredicateJoinOperation,
        createCPUSpatialPredicateJoinOperation
      ]) {
        const operation = create({
          left,
          right,
          predicate,
          candidateCapacity: 8,
          pairCapacity: 8,
          output
        });
        const result = await runOperationGraph(device, {
          inputs,
          operations: [operation],
          outputs: Object.values(output)
        });
        expect(Array.from(result['count'])).toEqual([expected.length]);
        expect(Array.from(result['requiredCount'])).toEqual([expected.length]);
        expect(
          expected.map((_, index) => [result['leftRows'][index], result['rightRows'][index]])
        ).toEqual(expected);
        expect(Array.from(result['overflow'])).toEqual([0]);
        expect(Array.from(result['candidateOverflow'])).toEqual([0]);
      }
    });
  }

it('both operation forms flag later-part candidate loss at feature granularity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {};
  const left = describeGeometry(
    'left',
    'points',
    {positions: Float32Array.of(-100, 0, 100, 0), featureOffsets: Uint32Array.of(0, 2)},
    inputs
  );
  const right = describeGeometry(
    'right',
    'points',
    {positions: Float32Array.of(-100, 0, 100, 0)},
    inputs
  );
  const output = {
    leftRows: 'leftRows',
    rightRows: 'rightRows',
    count: 'count',
    requiredCount: 'requiredCount',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow',
    candidateTruncated: 'candidateTruncated'
  };
  for (const candidateCapacity of [1, 2])
    for (const create of [
      createSpatialPredicateJoinOperation,
      createCPUSpatialPredicateJoinOperation
    ]) {
      const operation = create({
        left,
        right,
        predicate: 'intersects',
        candidateCapacity,
        pairCapacity: 8,
        output
      });
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(output)
      });
      expect(Array.from(result['count'])).toEqual([candidateCapacity]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['candidateOverflow'])).toEqual([candidateCapacity === 1 ? 1 : 0]);
      expect(Array.from(result['candidateTruncated'])).toEqual([candidateCapacity === 1 ? 1 : 0]);
      expect(Array.from(result['leftRows']).slice(0, candidateCapacity)).toEqual(
        Array(candidateCapacity).fill(0)
      );
      expect(Array.from(result['rightRows']).slice(0, candidateCapacity)).toEqual(
        candidateCapacity === 1 ? [0] : [0, 1]
      );
    }
});

for (const queryKind of ['points', 'lines'] as const)
  for (const targetKind of ['points', 'lines'] as const) {
    it(`nearest operation and CPU twin minimize multipart ${queryKind}/${targetKind} parts`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const inputs: Record<string, OperationGraphInput> = {};
      const queries = describeGeometry(
        'queries',
        queryKind,
        {
          positions: Float32Array.from(
            queryKind === 'points' ? [-100, 0, 2, 0] : [-100, 0, -99, 0, 2, 0, 2.5, 0]
          ),
          featureOffsets: Uint32Array.of(0, 2),
          ...(queryKind === 'lines' ? {lineOffsets: Uint32Array.of(0, 2, 4)} : {})
        },
        inputs
      );
      const features = describeGeometry(
        'features',
        targetKind,
        {
          positions: Float32Array.from(
            targetKind === 'points'
              ? [99, 0, 3, 0, 20, 0]
              : [99, 0, 99, 1, 3, -1, 3, 1, 20, 0, 20, 1]
          ),
          featureOffsets: Uint32Array.of(0, 2, 3),
          sourceIds: Uint32Array.of(42, 42),
          ...(targetKind === 'lines' ? {lineOffsets: Uint32Array.of(0, 2, 4, 6)} : {})
        },
        inputs
      );
      const output = {
        neighborRows: 'neighborRows',
        neighborIds: 'neighborIds',
        neighborCounts: 'neighborCounts',
        neighborDistances: 'neighborDistances',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      for (const create of [
        createNearestFeatureJoinOperation,
        createCPUNearestFeatureJoinOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs,
          operations: [create({queries, features, output})],
          outputs: Object.values(output)
        });
        expect(Array.from(result['neighborRows'])).toEqual([0]);
        expect(Array.from(result['neighborIds'])).toEqual([42]);
        expect(Array.from(result['neighborCounts'])).toEqual([1]);
        expect(Array.from(result['neighborDistances'])).toEqual([queryKind === 'points' ? 1 : 0.5]);
        expect(Array.from(result['overflow'])).toEqual([0]);
        expect(Array.from(result['candidateOverflow'])).toEqual([0]);
      }
    });
  }

it('prepared multipart bounds and owners rebuild after feature offsets change', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const positionsBuffer = createInputBuffer(device, Float32Array.of(-100, 0, 100, 0, 0, 0));
  const offsetsBuffer = createInputBuffer(device, Uint32Array.of(0, 2, 3));
  buffers.push(positionsBuffer, offsetsBuffer);
  const geometry = {
    kind: 'points' as const,
    positions: importGraphBuffer(graph, 'target-positions', positionsBuffer, 'float32x2', 3),
    featureOffsets: importGraphBuffer(graph, 'target-features', offsetsBuffer, 'uint32', 3)
  };
  const queryBuffer = createInputBuffer(device, Float32Array.of(100, 0));
  buffers.push(queryBuffer);
  const queries = {
    kind: 'points' as const,
    positions: importGraphBuffer(graph, 'queries', queryBuffer, 'float32x2', 1)
  };
  const outputs: ReturnType<typeof createOutputBuffer>[] = [];
  const scalar = (name: string): GraphDataView<'uint32'> => {
    const buffer = createOutputBuffer(device, 1);
    outputs.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', 1);
  };
  const rightRows = scalar('right-rows'),
    count = scalar('count');
  const prepared = new GPUSpatialJoinPrepared({geometry, spatialSort: false});
  graph.add(prepared);
  graph.add(
    new GPUSpatialPredicateJoin({
      left: queries,
      right: geometry,
      prepared,
      predicate: 'intersects',
      candidateCapacity: 4,
      output: {
        pairs: {
          leftRows: scalar('left-rows'),
          rightRows,
          status: {
            count,
            overflow: scalar('overflow'),
            candidateOverflow: scalar('candidate-overflow')
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(outputs[0], 1))).toEqual([0]);
    expect(Array.from(await readUint32(outputs[1], 1))).toEqual([1]);
    offsetsBuffer.write(Uint32Array.of(0, 1, 3));
    prepared.invalidate();
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(outputs[0], 1))).toEqual([1]);
    expect(Array.from(await readUint32(outputs[1], 1))).toEqual([1]);
  } finally {
    compiled.destroy();
    prepared.destroy();
    for (const buffer of [...buffers, ...outputs]) buffer.destroy();
  }
});

for (const reverse of [false, true])
  it(`nearest operation forms detect later-part containment in ${reverse ? 'targets' : 'queries'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs: Record<string, OperationGraphInput> = {};
    const multipart = describeGeometry(
      'multipart',
      'points',
      {positions: Float32Array.of(-100, 0, 2, 0), featureOffsets: Uint32Array.of(0, 2)},
      inputs
    );
    const polygon = describeGeometry(
      'polygon',
      'polygons',
      {
        positions: Float32Array.of(1, -1, 3, -1, 3, 1, 1, 1),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 4)
      },
      inputs
    );
    const output = {
      neighborRows: 'neighborRows',
      neighborIds: 'neighborIds',
      neighborCounts: 'neighborCounts',
      neighborDistances: 'neighborDistances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    for (const create of [
      createNearestFeatureJoinOperation,
      createCPUNearestFeatureJoinOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs,
        operations: [
          create({
            queries: reverse ? polygon : multipart,
            features: reverse ? multipart : polygon,
            output
          })
        ],
        outputs: Object.values(output)
      });
      expect(Array.from(result['neighborRows'])).toEqual([0]);
      expect(Array.from(result['neighborCounts'])).toEqual([1]);
      expect(Array.from(result['neighborDistances'])).toEqual([0]);
    }
  });

it('sorted multipart line queries pack topology and keep feature-row nearest outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      queryPositions: {
        format: 'float32x2',
        values: Float32Array.of(-100, 0, -99, 0, 2, 0, 2.5, 0, 19, 0, 19.5, 0)
      },
      featurePositions: {
        format: 'float32x2',
        values: Float32Array.of(99, 0, 99, 1, 3, -1, 3, 1, 20, 0, 20, 1)
      },
      queryLines: {format: 'uint32', values: Uint32Array.of(0, 2, 4, 6)},
      queryFeatures: {format: 'uint32', values: Uint32Array.of(0, 2, 3)},
      targetLines: {format: 'uint32', values: Uint32Array.of(0, 2, 4, 6)},
      targetFeatures: {format: 'uint32', values: Uint32Array.of(0, 2, 3)}
    },
    outputs: {
      rows: {format: 'uint32', length: 2},
      ids: {format: 'uint32', length: 2},
      counts: {format: 'uint32', length: 2},
      distances: {format: 'float32', length: 2},
      overflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'lines',
          positions: inputs['queryPositions'] as GraphDataView<'float32x2'>,
          lineOffsets: inputs['queryLines'] as GraphDataView<'uint32'>,
          featureOffsets: inputs['queryFeatures'] as GraphDataView<'uint32'>
        },
        features: {
          kind: 'lines',
          positions: inputs['featurePositions'] as GraphDataView<'float32x2'>,
          lineOffsets: inputs['targetLines'] as GraphDataView<'uint32'>,
          featureOffsets: inputs['targetFeatures'] as GraphDataView<'uint32'>
        },
        querySort: true,
        spatialSort: false,
        output: {
          neighborRows: outputs['rows'] as GraphDataView<'uint32'>,
          neighborIds: outputs['ids'] as GraphDataView<'uint32'>,
          neighborCounts: outputs['counts'] as GraphDataView<'uint32'>,
          neighborDistances: outputs['distances'] as GraphDataView<'float32'>,
          status: {overflow: outputs['overflow'] as GraphDataView<'uint32'>}
        }
      })
  });
  try {
    const result = await fixture.run();
    expect(result['rows']).toEqual([0, 1]);
    expect(result['counts']).toEqual([1, 1]);
    expect(result['distances']).toEqual([0.5, 0.5]);
    expect(result['overflow']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('standalone candidate stage exposes feature-level bounds spanning disconnected parts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      left: {format: 'float32x2', values: Float32Array.of(-100, 0, 100, 0)},
      features: {format: 'uint32', values: Uint32Array.of(0, 2)},
      right: {format: 'float32x2', values: Float32Array.of(-100, 0, 100, 0)}
    },
    outputs: {
      leftRows: {format: 'uint32', length: 2},
      rightRows: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPUSpatialJoinCandidates({
        left: {
          kind: 'points',
          positions: inputs['left'] as GraphDataView<'float32x2'>,
          featureOffsets: inputs['features'] as GraphDataView<'uint32'>
        },
        right: {kind: 'points', positions: inputs['right'] as GraphDataView<'float32x2'>},
        output: {
          leftRows: outputs['leftRows'] as GraphDataView<'uint32'>,
          rightRows: outputs['rightRows'] as GraphDataView<'uint32'>,
          status: {
            count: outputs['count'] as GraphDataView<'uint32'>,
            overflow: outputs['overflow'] as GraphDataView<'uint32'>
          }
        }
      })
  });
  try {
    const result = await fixture.run();
    expect(result['leftRows']).toEqual([0, 0]);
    expect(result['rightRows']).toEqual([0, 1]);
    expect(result['count']).toEqual([2]);
    expect(result['overflow']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('nearest operation forms retain empty feature rows and the missing sentinel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {};
  const queries = describeGeometry(
    'queries',
    'points',
    {positions: Float32Array.of(0, 0, 2, 0), featureOffsets: Uint32Array.of(0, 0, 2, 2)},
    inputs
  );
  const features = describeGeometry(
    'features',
    'lines',
    {
      positions: Float32Array.of(1, -1, 1, 1),
      lineOffsets: Uint32Array.of(0, 0, 2),
      featureOffsets: Uint32Array.of(0, 0, 2, 2)
    },
    inputs
  );
  const output = {
    neighborRows: 'neighborRows',
    neighborIds: 'neighborIds',
    neighborCounts: 'neighborCounts',
    neighborDistances: 'neighborDistances',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow'
  };
  for (const create of [createNearestFeatureJoinOperation, createCPUNearestFeatureJoinOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [create({queries, features, output})],
      outputs: Object.values(output)
    });
    expect(Array.from(result['neighborRows'])).toEqual([0xffffffff, 1, 0xffffffff]);
    expect(Array.from(result['neighborCounts'])).toEqual([0, 1, 0]);
    expect(Array.from(result['neighborDistances'])).toEqual([-1, 1, -1]);
    expect(Array.from(result['overflow'])).toEqual([0]);
    expect(Array.from(result['candidateOverflow'])).toEqual([0]);
  }
});
