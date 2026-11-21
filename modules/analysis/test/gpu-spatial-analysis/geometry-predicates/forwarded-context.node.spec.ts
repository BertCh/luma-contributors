// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('forward F6 retains the validated primary and secondary context snapshots', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const geometry = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 1)
    };
    let metricReads = 0;
    const contributor = new GPUGeometryPredicates({
      geometry,
      other: geometry,
      otherSpatialContext: {
        coordinateSpace: 'planar',
        get metric(): 'native' | 'great-circle' {
          metricReads++;
          return metricReads === 1 ? 'native' : 'great-circle';
        }
      },
      output: {equalsIdentical: createTransientView(graph, 'equal', 'uint32', 1)}
    });
    expect(metricReads).toBe(1);
    expect(contributor.props.otherSpatialContext?.metric).toBe('native');
    expect(contributor.props.spatialContext).toBe(contributor.spatialContext);
    expect(Object.isFrozen(contributor.props.spatialContext)).toBe(true);
    expect(Object.isFrozen(contributor.props.otherSpatialContext)).toBe(true);
    expect(metricReads).toBe(1);
  } finally {
    device.destroy();
  }
});

it.each([
  'great-circle',
  'native'
] as const)('forward T1-F2 rejects longitude-latitude %s before reading geometry or output', metric => {
  let geometryReads = 0;
  let otherReads = 0;
  let outputReads = 0;
  expect(
    () =>
      new GPUGeometryPredicates({
        spatialContext: {coordinateSpace: 'longitude-latitude', metric},
        get geometry(): never {
          geometryReads++;
          throw new Error('geometry was read');
        },
        get other(): never {
          otherReads++;
          throw new Error('other was read');
        },
        get output(): never {
          outputReads++;
          throw new Error('output was read');
        }
      })
  ).toThrow(/longitude-latitude/);
  expect([geometryReads, otherReads, outputReads]).toEqual([0, 0, 0]);
});
