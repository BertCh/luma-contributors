// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {addRasterJoinMismatchNode} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/joins/raster-join-mismatch';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialPredicateJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../../analysis/src/testing/gpu-contributor-test-utils';

it('topological join exposes candidate-only loss with spare pair capacity and a complete control', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One square intersects both identical right squares: exactly two true pairs, by inspection.
  for (const capacity of [1, 2]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format,
      length = values.length
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    };
    const output = (name: string, length = 1) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const geometry = (name: string, count: number) => ({
      kind: 'polygons' as const,
      positions: input(
        `${name}-positions`,
        Float32Array.from(Array.from({length: count}, () => [0, 0, 2, 0, 2, 2, 0, 2]).flat()),
        'float32x2',
        count * 4
      ),
      featureOffsets: input(
        `${name}-features`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32'
      ),
      polygonOffsets: input(
        `${name}-polygons`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32'
      ),
      ringOffsets: input(
        `${name}-rings`,
        Uint32Array.from({length: count + 1}, (_, index) => index * 4),
        'uint32'
      )
    });
    const count = output('count'),
      overflow = output('overflow'),
      candidateOverflow = output('candidate-overflow');
    graph.add(
      new GPUSpatialPredicateJoin({
        left: geometry('left', 1),
        right: geometry('right', 2),
        predicate: 'intersects',
        candidateCapacity: capacity,
        output: {
          pairs: {
            leftRows: output('left-rows', 2).view,
            rightRows: output('right-rows', 2).view,
            status: {
              count: count.view,
              overflow: overflow.view,
              candidateOverflow: candidateOverflow.view
            }
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([capacity]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([capacity === 1 ? 1 : 0]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('COMMON-W3 RT-137 W3-03 raster loss suppresses joins and zonal reductions, then recovers distinctly from empty', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {GPURasterJoin, GPURasterZonalStatistics} = await import(
    '@luma-contributors/analysis/raster-analysis'
  );
  const {createRasterizationFixture, destroyRasterizationFixture, makeExtentGrid} = await import(
    '../../../analysis/test/gpu-raster/polygon-rasterization/polygon-rasterization-fixture'
  );
  // The two vertical edges of this implicitly closed rectangle cross 100 rows each.
  {
    const capacity = 167;
    const fixture = createRasterizationFixture(device, {
      polygons: {
        positions: Float32Array.of(0, 0, 1, 0, 1, 100, 0, 100),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 4)
      },
      width: 2,
      height: 100,
      extent: [0, 0, 1, 1],
      crossingCapacity: capacity
    });
    const graph = fixture.graph;
    const buffers: Buffer[] = [];
    const output = (name: string, length = 1) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const pointBuffer = createInputBuffer(device, Float32Array.of(0.5, 50.5));
    buffers.push(pointBuffer);
    const points = importGraphBuffer(graph, 'point', pointBuffer, 'float32x2', 1);
    const producerOverflow = fixture.contributor.props.output.status!.overflow!;
    const counts = output('counts'),
      incomplete = output('incomplete'),
      assignments = output('assignments');
    const grid = makeExtentGrid(2, 100, [0, 0, 1, 1]);
    graph.add(
      new GPURasterJoin({
        grid,
        parameters: fixture.extentView,
        points,
        zones: fixture.zonesView,
        zoneCount: 1,
        producerOverflow,
        output: {
          counts: counts.view,
          pointZones: assignments.view,
          status: {incomplete: incomplete.view}
        }
      })
    );
    const reference = output('reference'),
      referenceOverflow = output('reference-overflow'),
      referenceUncertain = output('reference-uncertain'),
      mismatch = output('mismatch');
    addRasterJoinMismatchNode(graph, {
      positions: points,
      parameters: fixture.extentView,
      rasterZones: assignments.view,
      exactZones: reference.view,
      producerOverflow,
      exactOverflow: referenceOverflow.view,
      exactUncertainCount: referenceUncertain.view,
      width: 2,
      height: 100,
      output: mismatch.view
    });
    const reductions = Array.from({length: 5}, (_, index) => {
      const count = output(`zonal-count-${index}`),
        loss = output(`zonal-loss-${index}`);
      const valuesBuffer = createInputBuffer(device, new Float32Array(200).fill(1));
      buffers.push(valuesBuffer);
      const values = importGraphBuffer(
        graph,
        `zonal-values-${index}`,
        valuesBuffer,
        'float32',
        200
      );
      graph.add(
        new GPURasterZonalStatistics({
          id: `zonal-${index}`,
          grid,
          zones: fixture.zonesView,
          zoneCapacity: 1,
          producerOverflow,
          values: {id: `band-${index}`, format: 'float32', storage: {kind: 'buffer', values}},
          output: {cellCounts: count.view, status: {incomplete: loss.view}}
        })
      );
      return {count, loss};
    });
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(incomplete.buffer, 1)).toEqual([1]);
      expect(await readUint32(mismatch.buffer, 1)).toEqual([0]);
      expect(await readUint32(counts.buffer, 1)).toEqual([capacity === 167 ? 0xffffffff : 1]);
      expect(await readUint32(assignments.buffer, 1)).toEqual([capacity === 167 ? 0xffffffff : 0]);
      for (const {count, loss} of reductions) {
        expect(await readUint32(loss.buffer, 1)).toEqual([Number(capacity === 167)]);
        expect(await readUint32(count.buffer, 1)).toEqual([capacity === 167 ? 0xffffffff : 100]);
      }
      // Same graph and output buffers: coarsen the row size so the rectangle needs only
      // 50 crossings per vertical edge, then move the point genuinely outside the polygon.
      fixture.extent.write(Float32Array.of(0, 0, 1, 2));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, 1)).toEqual([1]);
      expect(await readUint32(assignments.buffer, 1)).toEqual([0]);
      expect(await readUint32(incomplete.buffer, 1)).toEqual([0]);
      for (const {count, loss} of reductions) {
        expect(await readUint32(loss.buffer, 1)).toEqual([0]);
        expect(await readUint32(count.buffer, 1)).toEqual([50]);
      }
      expect(await readUint32(mismatch.buffer, 1)).toEqual([0]);
      reference.buffer.write(Uint32Array.of(0xffffffff));
      pointBuffer.write(Float32Array.of(1.5, 50.5));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, 1)).toEqual([0]);
      expect(await readUint32(mismatch.buffer, 1)).toEqual([0]);
      expect(await readUint32(assignments.buffer, 1)).toEqual([0xffffffff]);
      expect(await readUint32(incomplete.buffer, 1)).toEqual([0]);
      // A genuine in-extent disagreement is flagged, but a polygon member outside the
      // viewport is not; reference loss independently clears classification.
      pointBuffer.write(Float32Array.of(0.5, 50.5));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(mismatch.buffer, 1)).toEqual([1]);
      referenceUncertain.buffer.write(Uint32Array.of(1));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(mismatch.buffer, 1)).toEqual([0]);
      referenceUncertain.buffer.write(Uint32Array.of(0));
      pointBuffer.write(Float32Array.of(3, 50.5));
      reference.buffer.write(Uint32Array.of(0));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(mismatch.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
      destroyRasterizationFixture(fixture);
    }
  }
});
