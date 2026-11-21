// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, expectTypeOf} from 'vitest';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPUCommandNodeKernelSources} from '../../../src/testing/gpu-contributor-test-utils';

const require = createRequire(import.meta.url);
const common = require('@luma.gl/graph/gpu-core') as typeof import('@luma.gl/graph/gpu-core');

expectTypeOf<{}>().not.toExtend<GPUZonalStatisticsOutput>();
expectTypeOf<{status: {}}>().not.toExtend<GPUZonalStatisticsOutput>();

function fixture() {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const integers = (length: number) =>
    createTransientView(graph, `integers-${serial++}`, 'uint32', length);
  const positions = (length: number) =>
    createTransientView(graph, `positions-${serial++}`, 'float32x2', length);
  return {device, graph, integers, positions};
}
for (const multipart of [false, true]) {
  it(`GEO-112 graph-owned compact completeness, GEO-114 row-only output multipart=${multipart}`, () => {
    const {device, graph, integers, positions} = fixture();
    try {
      const selection = new GPUBufferSelection({
        points: {
          kind: 'points',
          positions: positions(2),
          ...(multipart ? {featureOffsets: integers(3)} : {})
        },
        features: {kind: 'points', positions: positions(2), sourceIds: integers(2)},
        radius: createTransientView(graph, 'radius', 'float32', 1),
        candidateCapacity: 1,
        spatialSort: false,
        querySort: false,
        output: {}
      });
      const matches = selection.getMatches(graph);
      expect(selection.getMatches(graph)).toBe(matches);
      expect(matches.ids.length).toBe(2);
      expect(matches.status.candidateOverflow.length).toBe(1);
      expect(() => selection.getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}
it('UGE-01 and GEO-114 accept separate row/ID and polygon reduction destinations', () => {
  const {device, graph, integers, positions} = fixture();
  try {
    const join = new GPUPointInPolygonJoin({
      points: {kind: 'points', positions: positions(1)},
      polygons: {
        kind: 'polygons',
        positions: positions(8),
        featureOffsets: integers(3),
        polygonOffsets: integers(3),
        ringOffsets: integers(3),
        sourceIds: integers(2)
      },
      candidateCapacity: 8,
      spatialSort: false,
      querySort: false,
      edgeSlabs: false,
      output: {
        pointFeatureRows: integers(1),
        pointFeatureIds: integers(1),
        polygonAssignmentCounts: integers(2),
        polygonMembershipCounts: integers(2),
        status: {overflow: integers(1)}
      }
    });
    const sources = getGPUCommandNodeKernelSources(join.getCommandNodes(graph))
      .map(record => record.source)
      .join('\n');
    expect(sources).toContain('pointFeatureRows');
    expect(sources).toContain('atomicAdd');
  } finally {
    device.destroy();
  }
});
for (const ties of ['lowest-row', 'all'] as const) {
  it(`NAM-03 accepts ${ties} without changing row ordering`, () => {
    const {device, graph, integers, positions} = fixture();
    try {
      const join = new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: positions(1)},
        features: {kind: 'points', positions: positions(2), sourceIds: integers(2)},
        ties,
        output: {
          neighborIds: integers(1),
          neighborRows: integers(1),
          neighborCounts: integers(1),
          status: {overflow: integers(1)}
        }
      });
      expect(join.ties).toBe(ties);
      expect(() => join.getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}
it('NAM-03 rejects an unknown tie selector with the accepted values', () => {
  const {device, integers, positions} = fixture();
  try {
    expect(
      () =>
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: positions(1)},
          features: {kind: 'points', positions: positions(2)},
          ties: 'lowest-source-id' as never,
          output: {
            neighborIds: integers(1),
            neighborCounts: integers(1),
            status: {overflow: integers(1)}
          }
        })
    ).toThrow(/lowest-row.*all/);
  } finally {
    device.destroy();
  }
});
it('rejects the retired nearest-feature join spellings at construction', () => {
  const {device, integers, positions} = fixture();
  try {
    const props = {
      queries: {kind: 'points', positions: positions(1)},
      features: {kind: 'points', positions: positions(2)},
      neighborCount: 1,
      output: {
        neighborIds: integers(1),
        neighborCounts: integers(1),
        status: {overflow: integers(1)}
      }
    } as const;
    const oldProps = {...props, k: 1};
    expect(() => new GPUNearestFeatureJoin(oldProps)).toThrow(
      'nearest-feature-join: k was renamed; use neighborCount'
    );
    expect(() => new GPUNearestFeatureJoin({...props, ties: 'lowest-id' as never})).toThrow(
      "ties 'lowest-id' was renamed; use 'lowest-row'"
    );
  } finally {
    device.destroy();
  }
});
it('ENT-02 CommonJS-created chunked values select atomic zonal output handling', () => {
  const {device, graph, integers} = fixture();
  try {
    const chunks = [
      createTransientView(graph, 'value-0', 'float32', 1),
      createTransientView(graph, 'value-1', 'float32', 1)
    ];
    const values: GraphVectorView<'float32'> = new common.GraphVectorView({
      id: 'foreign-values',
      name: 'foreign-values',
      format: 'float32',
      length: 2,
      valueLength: 2,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: chunks
    });
    const zonal = new GPUZonalStatistics({
      features: {kind: 'feature-rows', pointFeatureRows: integers(2), featureCount: 1},
      values,
      output: {sums: createTransientView(graph, 'sums', 'float32', 1)}
    });
    expect(zonal.sumOrder).toBe('atomic');
    expect(() => zonal.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
