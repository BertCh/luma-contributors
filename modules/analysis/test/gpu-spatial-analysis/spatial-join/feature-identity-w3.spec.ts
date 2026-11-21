// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function fixture(device: Device) {
  const graph = new GPUCommandGraph(device),
    buffers: Buffer[] = [];
  let serial = 0;
  const integers = (values: number[]) => {
    const buffer = createInputBuffer(device, Uint32Array.from(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, `integers-${serial++}`, buffer, 'uint32', values.length);
  };
  const positions = (values: number[]) => {
    const buffer = createInputBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `positions-${serial++}`,
        buffer,
        'float32x2',
        values.length / 2
      )
    };
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, 'uint32', length)};
  };
  return {
    graph,
    integers,
    positions,
    output,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

for (const multipart of [false, true]) {
  it(`UGE-01 GEO-114 explicit polygon counts and original IDs multipart=${multipart}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const test = fixture(device),
      {graph, integers, positions, output} = test;
    const rows = output(1),
      ids = output(1),
      assignment = output(2),
      membership = output(2);
    graph.add(
      new GPUPointInPolygonJoin({
        points: {
          kind: 'points',
          positions: positions([0.5, 0.5]).view,
          ...(multipart ? {featureOffsets: integers([0, 1])} : {})
        },
        polygons: {
          kind: 'polygons',
          positions: positions([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]).view,
          featureOffsets: integers([0, 1, 2]),
          polygonOffsets: integers([0, 1, 2]),
          ringOffsets: integers([0, 4, 8]),
          sourceIds: integers([99, 1])
        },
        candidateCapacity: 8,
        spatialSort: false,
        querySort: false,
        edgeSlabs: false,
        output: {
          pointFeatureRows: rows.view,
          pointFeatureIds: ids.view,
          polygonAssignmentCounts: assignment.view,
          polygonMembershipCounts: membership.view,
          status: {overflow: output(1).view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(rows.buffer)).toEqual([0]);
      expect(await readUint32(ids.buffer)).toEqual([99]);
      // Hand-count: one lowest-row assignment and one membership in each identical square.
      expect(await readUint32(assignment.buffer)).toEqual([1, 0]);
      expect(await readUint32(membership.buffer)).toEqual([1, 1]);
    } finally {
      compiled.destroy();
      test.destroy();
    }
  });
  it(`GEO-114 nearest rows precede source IDs multipart=${multipart}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const test = fixture(device),
      {graph, integers, positions, output} = test;
    const radius = new GPUParameterBuffer(device, {
      id: 'radius',
      format: 'float32',
      length: 1,
      values: Float32Array.of(2)
    });
    const rows = output(1),
      ids = output(1);
    graph.add(
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'points',
          positions: positions([0, 0]).view,
          ...(multipart ? {featureOffsets: integers([0, 1])} : {})
        },
        features: {
          kind: 'points',
          positions: positions([-1, 0, 1, 0]).view,
          sourceIds: integers([99, 1])
        },
        radius: radius.importToGraph(graph),
        candidateCapacity: 8,
        spatialSort: false,
        querySort: false,
        output: {
          nearestFeatureRows: rows.view,
          nearestFeatureIds: ids.view,
          status: {overflow: output(1).view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(rows.buffer)).toEqual([0]);
      expect(await readUint32(ids.buffer)).toEqual([99]);
    } finally {
      compiled.destroy();
      radius.destroy();
      test.destroy();
    }
  });
}
for (const multipart of [false, true])
  for (const rowOnly of [false, true]) {
    it(`B8 row writers publish row 1 and unmatched sentinel on reuse multipart=${multipart} rowOnly=${rowOnly}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const test = fixture(device),
        {graph, integers, positions, output} = test;
      const queries = positions([2.5, 0.5, 99, 99]);
      const queryGeometry = {
        kind: 'points' as const,
        positions: queries.view,
        ...(multipart ? {featureOffsets: integers([0, 1, 2])} : {})
      };
      const polygonRows = output(2),
        polygonIds = output(2),
        nearestRows = output(2),
        nearestIds = output(2);
      const radius = new GPUParameterBuffer(device, {
        id: 'row-radius',
        format: 'float32',
        length: 1,
        values: Float32Array.of(1)
      });
      graph.add(
        new GPUPointInPolygonJoin({
          points: queryGeometry,
          polygons: {
            kind: 'polygons',
            positions: positions([0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 3, 0, 3, 1, 2, 1]).view,
            featureOffsets: integers([0, 1, 2]),
            polygonOffsets: integers([0, 1, 2]),
            ringOffsets: integers([0, 4, 8]),
            sourceIds: integers([99, 1])
          },
          candidateCapacity: 8,
          spatialSort: false,
          querySort: false,
          edgeSlabs: false,
          output: {
            pointFeatureRows: polygonRows.view,
            ...(!rowOnly ? {pointFeatureIds: polygonIds.view} : {}),
            status: {overflow: output(1).view}
          }
        })
      );
      graph.add(
        new GPUNearestFeatureJoin({
          queries: queryGeometry,
          features: {
            kind: 'points',
            positions: positions([0.5, 0.5, 2.5, 0.5]).view,
            sourceIds: integers([99, 1])
          },
          radius: radius.importToGraph(graph),
          candidateCapacity: 8,
          spatialSort: false,
          querySort: false,
          output: {
            nearestFeatureRows: nearestRows.view,
            ...(!rowOnly ? {nearestFeatureIds: nearestIds.view} : {}),
            status: {overflow: output(1).view}
          }
        })
      );
      const compiled = graph.compile();
      try {
        for (const [coordinates, expectedRows, expectedIds] of [
          [
            [2.5, 0.5, 99, 99],
            [1, 0xffffffff],
            [1, 0xffffffff]
          ],
          [
            [99, 99, 0.5, 0.5],
            [0xffffffff, 0],
            [0xffffffff, 99]
          ]
        ]) {
          queries.buffer.write(Float32Array.from(coordinates));
          for (const destination of [polygonRows, polygonIds, nearestRows, nearestIds])
            destination.buffer.write(Uint32Array.of(123, 456));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(polygonRows.buffer)).toEqual(expectedRows);
          expect(await readUint32(nearestRows.buffer)).toEqual(expectedRows);
          if (!rowOnly) {
            expect(await readUint32(polygonIds.buffer)).toEqual(expectedIds);
            expect(await readUint32(nearestIds.buffer)).toEqual(expectedIds);
          }
        }
      } finally {
        compiled.destroy();
        radius.destroy();
        test.destroy();
      }
    });
  }

for (const ties of ['lowest-row'] as const) {
  it(`NAM-03 equidistant target keeps row 0 ID 99 with ${ties}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const test = fixture(device),
      {graph, positions, integers, output} = test;
    const rows = output(1),
      ids = output(1);
    graph.add(
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: positions([0, 0]).view},
        features: {
          kind: 'points',
          positions: positions([-1, 0, 1, 0]).view,
          sourceIds: integers([99, 1])
        },
        ties,
        output: {
          neighborRows: rows.view,
          neighborIds: ids.view,
          neighborCounts: output(1).view,
          status: {overflow: output(1).view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(rows.buffer)).toEqual([0]);
      expect(await readUint32(ids.buffer)).toEqual([99]);
    } finally {
      compiled.destroy();
      test.destroy();
    }
  });
}
for (const multipart of [false, true]) {
  it(`GEO-112 graph-owned compact matches expose candidate loss multipart=${multipart}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const test = fixture(device),
      {graph, positions, integers, output} = test;
    const targets = positions([0, 0, 0, 0]);
    const radius = new GPUParameterBuffer(device, {
      id: 'radius',
      format: 'float32',
      length: 1,
      values: Float32Array.of(1)
    });
    const selection = new GPUBufferSelection({
      points: {
        kind: 'points',
        positions: positions([0, 0]).view,
        ...(multipart ? {featureOffsets: integers([0, 1])} : {})
      },
      features: {kind: 'points', positions: targets.view, sourceIds: integers([99, 1])},
      radius: radius.importToGraph(graph),
      candidateCapacity: 1,
      leafCapacity: multipart ? 1 : 2,
      spatialSort: false,
      querySort: false,
      output: {}
    });
    const matches = selection.getMatches(graph),
      snapshot = output(4);
    graph.add(selection);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'capture-compact',
        operation: 'test',
        invocationCount: 1,
        bindings: [
          {name: 'ids', view: matches.ids, type: 'u32', access: 'read'},
          {name: 'count', view: matches.status.count, type: 'u32', access: 'read'},
          {name: 'candidate', view: matches.status.candidateOverflow, type: 'u32', access: 'read'},
          {name: 'overflow', view: matches.status.overflow, type: 'u32', access: 'read'},
          {name: 'result', view: snapshot.view, type: 'u32', access: 'read_write'}
        ],
        body: 'result[resultOffset] = count[countOffset]; result[resultOffset + 1u] = candidate[candidateOffset]; result[resultOffset + 2u] = overflow[overflowOffset]; result[resultOffset + 3u] = ids[idsOffset];'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(snapshot.buffer)).toEqual([1, 1, 0, 0]);
      if (!multipart) {
        targets.buffer.write(Float32Array.of(0, 0, 100, 100));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(snapshot.buffer)).toEqual([1, 0, 0, 0]);
      }
    } finally {
      compiled.destroy();
      radius.destroy();
      test.destroy();
    }
  });
}
