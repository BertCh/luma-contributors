// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMaximumInscribedCircle,
  GPUNormalizePolygons,
  GPUSegmentSnap
} from '../../../src/gpu-spatial-analysis/geometry-refinement';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from '../../oracles/fixtures/geometry-refinement.json';

it('GPU geometry canonicalization and bounded circles agree with direct Shapely 2.1.2 fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.polygons) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = (values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const output = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const geometry = {
      positions: importGraphBuffer(
        graph,
        'positions',
        input(Float32Array.from(fixture.input.positions)),
        'float32x2',
        fixture.input.positions.length / 2
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'rings',
        input(Uint32Array.from(fixture.input.ringOffsets)),
        'uint32',
        fixture.input.ringOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygons',
        input(Uint32Array.from(fixture.input.polygonOffsets)),
        'uint32',
        fixture.input.polygonOffsets.length
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'features',
        input(Uint32Array.from(fixture.input.featureOffsets)),
        'uint32',
        fixture.input.featureOffsets.length
      )
    };
    const normalized = output(fixture.input.positions.length);
    const rings = output(fixture.input.ringOffsets.length);
    const polygons = output(fixture.input.polygonOffsets.length);
    const circles = output(4);
    const witnesses = output(2);
    const status = output(1);
    graph.add(
      new GPUNormalizePolygons({
        ...geometry,
        output: {
          positions: importGraphBuffer(
            graph,
            'normalized',
            normalized,
            'float32x2',
            geometry.positions.length
          ),
          ringOffsets: importGraphBuffer(
            graph,
            'normal-rings',
            rings,
            'uint32',
            geometry.ringOffsets.length
          ),
          polygonOffsets: importGraphBuffer(
            graph,
            'normal-polygons',
            polygons,
            'uint32',
            geometry.polygonOffsets.length
          )
        }
      })
    );
    graph.add(
      new GPUMaximumInscribedCircle({
        ...geometry,
        tolerance: 0.002,
        cellCapacity: 1024,
        maximumIterations: 4096,
        output: {
          circles: importGraphBuffer(graph, 'circles', circles, 'float32x4', 1),
          witnesses: importGraphBuffer(graph, 'witnesses', witnesses, 'float32x2', 1),
          status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(normalized, fixture.input.positions.length)).toEqual(
        fixture.normalized.positions
      );
      expect(await readUint32(rings, fixture.input.ringOffsets.length)).toEqual(
        fixture.normalized.ringOffsets
      );
      expect(await readUint32(polygons, fixture.input.polygonOffsets.length)).toEqual(
        fixture.normalized.polygonOffsets
      );
      const circle = await readFloat32(circles, 4);
      expect((await readUint32(status, 1))[0]).toBe(0);
      expect(circle[3]).toBeLessThanOrEqual(0.002);
      expect(Math.abs(circle[2] - fixture.circle[2])).toBeLessThanOrEqual(circle[3] + 0.0001);
      const witness = await readFloat32(witnesses, 2);
      expect(Math.hypot(witness[0] - circle[0], witness[1] - circle[1])).toBeCloseTo(circle[2], 4);
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }
});

it('GPU segment snapping matches the direct reference insertion fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const positions = output(8);
  const sourceEdges = output(4);
  const counts = output(4);
  graph.add(
    new GPUSegmentSnap({
      positions: importGraphBuffer(
        graph,
        'input',
        input(Float32Array.from(fixtures.snap.input.flat())),
        'float32x2',
        2
      ),
      lineOffsets: importGraphBuffer(
        graph,
        'offsets',
        input(Uint32Array.from([0, 2])),
        'uint32',
        2
      ),
      referencePositions: importGraphBuffer(
        graph,
        'references',
        input(Float32Array.from(fixtures.snap.references.flat())),
        'float32x2',
        4
      ),
      tolerance: fixtures.snap.tolerance,
      maximumVerticesPerLine: 4,
      output: {
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 4),
        sourceEdges: importGraphBuffer(graph, 'sources', sourceEdges, 'uint32', 4),
        counts: importGraphBuffer(graph, 'counts', counts, 'uint32x4', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(positions, 8)).toEqual(fixtures.snap.expected.flat().map(Math.fround));
    expect(await readUint32(counts, 4)).toEqual([4, 4, 0, 0]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
