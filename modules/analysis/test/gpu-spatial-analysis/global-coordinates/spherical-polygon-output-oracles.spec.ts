// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUSphericalPolygonOverlay,
  type GPUSphericalPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from './output-oracle-fixture';

// The fixed-pass contract exposes one optional numerical diagnostic, on the result only.
// Removed iterative/invalid/approximation fields are covered by constructor-refusal Node cases.
const OPTIONS = [
  'absent',
  'sources',
  'ringCount',
  'requiredRingCount',
  'nodedSegmentRequiredCount',
  'result.uncertainCount'
];

it.each(
  OPTIONS
)('spherical overlay publishes independent triangle boundary, provenance and status (%s)', async option => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // In the north-pole gnomonic chart the operands are x+y<=2 and x+y<=3, y>=0,
  // with x>=0 and x>=1 respectively. Their intersection has vertices (1,0),(2,0),(1,1).
  // Splitting the two original three-edge rings at (1,0),(2,0),(1,1) needs 10 pieces.
  // An empty first feature makes every boundary owner source row 1, distinct from operand IDs.
  const fixture = createOracleFixture(device);
  const operand = (coordinates: number[][], sourceId: number) => ({
    kind: 'polygons' as const,
    positions: fixture.input(
      'float32x3',
      coordinates.flatMap(([horizontal, vertical]) => {
        const length = Math.hypot(horizontal, vertical, 1);
        return [horizontal / length, vertical / length, 1 / length];
      })
    ),
    ringOffsets: fixture.input('uint32', [0, 3]),
    polygonOffsets: fixture.input('uint32', [0, 1]),
    featureOffsets: fixture.input('uint32', [0, 0, 1]),
    sourceIds: fixture.input('uint32', [999, sourceId])
  });
  const positions = fixture.output('float32x3', 8);
  const ringOffsets = fixture.output('uint32', 3);
  const polygonOffsets = fixture.output('uint32', 3);
  const featureOffsets = fixture.output('uint32', 3);
  const complement = fixture.output('uint32', 1);
  const endpoints = fixture.output('float32x4', 8);
  const operands = fixture.output('uint32', 8);
  const owners = fixture.output('uint32', 8);
  const scalars = new Map<string, ReturnType<typeof fixture.output<'uint32'>>>();
  const scalar = (name: string) => {
    const result = fixture.output('uint32', 1);
    scalars.set(name, result);
    return result.view;
  };
  const status: GPUSphericalPolygonOverlayProps['output']['status'] = {
    count: scalar('count'),
    overflow: scalar('overflow'),
    candidateOverflow: scalar('candidateOverflow')
  };
  const boundaryStatus: GPUSphericalPolygonOverlayProps['output']['boundary']['status'] = {
    count: scalar('boundary.count'),
    requiredCount: scalar('boundary.requiredCount'),
    overflow: scalar('boundary.overflow'),
    candidateOverflow: scalar('boundary.candidateOverflow')
  };
  if (option.startsWith('result.')) Object.assign(status, {[option.slice(7)]: scalar(option)});
  else if (['ringCount', 'requiredRingCount', 'nodedSegmentRequiredCount'].includes(option))
    Object.assign(status, {[option]: scalar(option)});
  const sources =
    option === 'sources'
      ? {
          sourceRows: fixture.output('uint32', 2),
          operandIds: fixture.output('uint32', 2),
          mixed: fixture.output('uint32', 2)
        }
      : undefined;
  fixture.graph.add(
    new GPUSphericalPolygonOverlay({
      left: operand(
        [
          [0, 0],
          [2, 0],
          [0, 2]
        ],
        101
      ),
      right: operand(
        [
          [1, 0],
          [3, 0],
          [1, 2]
        ],
        202
      ),
      chartCenter: [0, 0, 1],
      operation: 'intersection',
      vertexTolerance: 2 ** -16,
      capacity: {intersections: 32, nodedSegments: 32},
      output: {
        geometry: {
          kind: 'polygons',
          positions: positions.view,
          ringOffsets: ringOffsets.view,
          polygonOffsets: polygonOffsets.view,
          featureOffsets: featureOffsets.view
        },
        complement: complement.view,
        status,
        boundary: {
          endpoints: endpoints.view,
          operandIds: operands.view,
          sourceFeatureIds: owners.view,
          status: boundaryStatus
        },
        ...(sources
          ? {
              featureSources: {
                sourceRows: sources.sourceRows.view,
                operandIds: sources.operandIds.view,
                mixed: sources.mixed.view
              }
            }
          : {})
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    // Re-poison and resubmit to expose optional outputs that retain a previous submission.
    for (let submission = 0; submission < 2; submission++) {
      for (const output of scalars.values()) output.poison();
      for (const output of [
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        complement,
        endpoints,
        operands,
        owners,
        ...Object.values(sources ?? {})
      ])
        output.poison();
      submitGraph(device, compiled, undefined);
      for (const [name, output] of scalars) {
        const expected =
          name === 'count' || name === 'ringCount' || name === 'requiredRingCount'
            ? 1
            : name === 'boundary.count' || name === 'boundary.requiredCount'
              ? 3
              : name === 'nodedSegmentRequiredCount'
                ? 10
                : 0;
        expect(await output.read(), `${option}: ${name}`).toEqual([expected]);
      }
      expect(await complement.read()).toEqual([0]);
      // The public polygon geometry omits the closing duplicate of the raw assembly ring.
      expect((await ringOffsets.read()).slice(0, 2)).toEqual([0, 3]);
      expect((await polygonOffsets.read()).slice(0, 2)).toEqual([0, 1]);
      expect((await featureOffsets.read()).slice(0, 2)).toEqual([0, 1]);
      const edges = await endpoints.read();
      const operandIds = await operands.read();
      const sourceRows = await owners.read();
      const actual = Array.from({length: 3}, (_, row) => [
        ...edges.slice(row * 4, row * 4 + 4),
        operandIds[row],
        sourceRows[row]
      ]).sort((first, second) => first.join(',').localeCompare(second.join(',')));
      expect(actual).toEqual([
        [1, 0, 2, 0, 0, 101],
        [1, 1, 1, 0, 1, 202],
        [2, 0, 1, 1, 0, 101]
      ]);
      if (sources) {
        expect(await sources.sourceRows.read()).toEqual([1, 0xffffffff]);
        expect(await sources.operandIds.read()).toEqual([0, 0xffffffff]);
        expect(await sources.mixed.read()).toEqual([1, 0]);
      }
      // Gnomonic inverse: normalize(x,y,1). Float32 projection/lift tolerance is 3e-5.
      const values = await positions.read();
      const vertices = Array.from({length: 3}, (_, row) => values.slice(row * 3, row * 3 + 3)).sort(
        (first, second) => first[0] - second[0]
      );
      const expected = [
        [0.5773502692, 0.5773502692, 0.5773502692],
        [0.7071067812, 0, 0.7071067812],
        [0.894427191, 0, 0.4472135955]
      ];
      for (let row = 0; row < 3; row++)
        for (let component = 0; component < 3; component++)
          expect(Math.abs(vertices[row][component] - expected[row][component])).toBeLessThan(3e-5);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}, 60000);
