// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession, KeplerResidentStyle} from '../src';
import {createNativeStylePlan} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import {GeometryRuntime} from '../../../examples/deck/kepler-compute/src/geometry-runtime';
import {
  prepareKeplerGeometrySource,
  makeKeplerGeometryGPUVectors
} from '../../../examples/deck/kepler-compute/src/geometry-preparation';

test('native polygon fixed Elevation Offset consumes a nullable table column on GPU and reuses original XYZ and source masks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const source = new arrow.Table({
    baseAltitude: arrow.vectorFromArray([999, null], new arrow.Float32()),
    selection: arrow.vectorFromArray([0, 1], new arrow.Uint32())
  }).concat(
    new arrow.Table({
      baseAltitude: arrow.vectorFromArray([-50, 200], new arrow.Float32()),
      selection: arrow.vectorFromArray([1, 1], new arrow.Uint32())
    })
  );
  const dataset = session.importArrowDataset({id: 'polygon-altitude', table: source});
  const polygon = {
    type: 'Polygon',
    coordinates: [
      [
        [0, 0, 7],
        [1, 0, 7],
        [0, 1, 7],
        [0, 0, 7]
      ]
    ]
  };
  const geometry = makeKeplerGeometryGPUVectors(
    device,
    prepareKeplerGeometrySource([
      {id: 'first', rows: [null, polygon]},
      {
        id: 'last',
        rows: [
          polygon,
          {type: 'MultiPolygon', coordinates: [polygon.coordinates, polygon.coordinates]}
        ]
      }
    ])
  );
  const translated = createNativeStylePlan(
    {
      id: 'polygon',
      type: 'geojson',
      config: {
        dataId: dataset.id,
        color: [200, 80, 40],
        elevationOffsetField: {name: 'baseAltitude'},
        elevationOffsetDomain: [-100, 1000],
        visConfig: {filled: true, enable3d: false, fixedElevation: true}
      }
    },
    {
      source: {datasetId: dataset.id, revision: dataset.revision},
      columns: {baseAltitude: {format: 'float32'}},
      zoom: 12,
      geojsonPolygonFeatures: true,
      geojsonProperties: {
        fillColor: 'absent',
        lineColor: 'absent',
        lineWidth: 'absent',
        elevation: 'absent',
        elevationOffset: 'absent'
      }
    }
  );
  if (!translated.supported) throw new Error(translated.reasons.join(' '));
  const channel = translated.plan.channels.elevationOffset;
  if (channel?.kind !== 'resident') throw new Error('Expected the native table altitude channel');
  expect(channel.plan.scale).toBe('identity');
  const style = new KeplerResidentStyle({
    session,
    plan: channel.plan,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const runtime = new GeometryRuntime({
    device,
    geometry,
    column: 'shape',
    selectionMask: dataset.value.columns['selection'] as GPUVector<'uint32'>,
    elevationOffsets: style.values,
    elevations: 0,
    stylesResolved: true
  });
  const buffers = [
    ...geometry.buffers,
    ...dataset.value.columns['baseAltitude'].data.map(chunk => chunk.buffer)
  ];
  const reads = buffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const writes = buffers.map(buffer => vi.spyOn(buffer, 'write'));
  try {
    const output = runtime.binding.polygons!.getElevationOffset;
    if (!(output instanceof GPUVector)) throw new Error('Expected a gathered altitude vector');
    const encode = () => {
      const encoder = device.createCommandEncoder();
      style.encode(encoder);
      runtime.encode(encoder);
      device.submit(encoder.finish());
    };
    encode();
    const readFloats = async () =>
      (
        await Promise.all(
          output.data.map(async chunk => {
            const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
            return [...new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length)];
          })
        )
      ).flat();
    expect(await readFloats()).toEqual([999, 0, -50, 200]);
    expect(runtime.binding.polygons!.positions).toBe(geometry.polygons.positions);
    expect(runtime.binding.polygons!.extrusionRings!.positions).toBe(
      geometry.polygonOutlines.getPath
    );
    expect(runtime.binding.polygons!.resolveSourceRow!(3)).toBe(3);
    const resources = [...runtime.resources];
    encode();
    expect(await readFloats()).toEqual([999, 0, -50, 200]);
    expect(runtime.resources).toEqual(resources);
    const sourceMask = dataset.value.columns['selection'];
    sourceMask.data[0].buffer.write(new Uint32Array([1, 0]));
    sourceMask.data[1].buffer.write(new Uint32Array([1, 0]));
    encode();
    const mask = runtime.binding.polygons!.selectionMask!;
    const maskRows = (
      await Promise.all(
        mask.data.map(async chunk => {
          const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
          return [...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length)];
        })
      )
    ).flat();
    expect(maskRows).toEqual([1, 0, 1, 0]);
    expect(runtime.binding.polygons!.getElevationOffset).toBe(output);
    expect(await readFloats()).toEqual([999, 0, -50, 200]);
    expect(runtime.readbackByteLength).toBe(0);
    for (const spy of [...reads, ...writes]) expect(spy).not.toHaveBeenCalled();
  } finally {
    for (const spy of [...reads, ...writes]) spy.mockRestore();
    const fence = device.createFence();
    try {
      await fence.signaled;
      runtime.destroy();
      await style.destroy(fence.signaled);
      geometry.destroy();
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
