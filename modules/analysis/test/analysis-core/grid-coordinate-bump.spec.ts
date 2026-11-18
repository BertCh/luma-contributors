// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createGridCoordinateFixture,
  createGridCoordinateReaderProbe,
  createRuntimeGridCoordinateDiagnostic,
  getGridCoordinateAgreement,
  getFloat32GridCoordinate,
  GRID_PROBE_COMPLETED,
  GRID_COORDINATE_CASES,
  GRID_COORDINATE_READERS
} from '../utils/grid-coordinate-bump-fixtures';
import {
  getGPUCommandNodeKernelSources,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const reader of GRID_COORDINATE_READERS) {
  it.each(
    GRID_COORDINATE_CASES
  )(`${reader} reads the published grid builder's cell: $name`, async testCase => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGridCoordinateFixture(device, reader, testCase);
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const source = getGPUCommandNodeKernelSources(fixture.readerNodes).find(kernel =>
        kernel.source.includes('fn getCoordinate(')
      )?.source;
      expect(source).toBeDefined();
      fixture.graph.add(createGridCoordinateReaderProbe(fixture, testCase, source!));
      const overflowCase = testCase.name === 'overflowing domain width';
      if (overflowCase) {
        fixture.graph.add(createRuntimeGridCoordinateDiagnostic(fixture, testCase, source!));
      }
      compiled = fixture.graph.compile();
      submitGraph(device, compiled, undefined);
      const [coordinate, completed] = await readUint32(fixture.buffers['coordinate'], 2);
      expect(completed, 'reader-coordinate kernel must execute and write the poisoned output').toBe(
        GRID_PROBE_COMPLETED
      );
      const offsets = await readUint32(fixture.buffers['cell-offsets'], testCase.size + 1);
      let expectedCell: number = testCase.cell;
      if (overflowCase) {
        // Read the occupied cell produced by the published GPUGridIndex on this same device.
        // Both encounter points coincide, so this fixture always has exactly one occupied cell.
        const occupiedCells = offsets
          .slice(0, -1)
          .flatMap((start, cell) => (offsets[cell + 1] > start ? [cell] : []));
        expect(
          occupiedCells,
          'published builder must place the fixture point(s) in exactly one cell'
        ).toHaveLength(1);
        expectedCell = occupiedCells[0];
        const modelCell = getFloat32GridCoordinate(
          testCase.value,
          testCase.minimum,
          testCase.maximum,
          testCase.size
        );
        const agreement = getGridCoordinateAgreement(coordinate, expectedCell, modelCell);
        const diagnostic = await readUint32(fixture.buffers['runtime-coordinate'], 5);
        expect(diagnostic[1], 'runtime diagnostic must execute').toBe(GRID_PROBE_COMPLETED);
        console.log('grid extreme-domain specialization', {
          reader,
          outcome: agreement.outcome,
          builderCell: expectedCell,
          modelCell,
          literalCoordinate: coordinate,
          runtimeCoordinate: diagnostic[0],
          scaledMinimum: new Float32Array(Uint32Array.of(diagnostic[2]).buffer)[0],
          scaledMaximum: new Float32Array(Uint32Array.of(diagnostic[3]).buffer)[0],
          reciprocal: new Float32Array(Uint32Array.of(diagnostic[4]).buffer)[0]
        });
        expect(coordinate, agreement.message).toBe(expectedCell);
      } else {
        expect(coordinate).toBe(testCase.cell);
      }
      const expectedRows = reader === 'encounters' ? 2 : 1;
      expect(offsets[coordinate + 1] - offsets[coordinate]).toBe(expectedRows);
      expect(await readUint32(fixture.buffers['index-count'], 1)).toEqual([expectedRows]);
      expect(await readUint32(fixture.buffers['index-overflow'], 1)).toEqual([0]);
      if (reader === 'interpolation') {
        expect(await readFloat32(fixture.buffers['result'], 1)).toEqual([7]);
        expect(await readUint32(fixture.buffers['neighbor-count'], 1)).toEqual([1]);
      } else {
        expect(await readUint32(fixture.buffers['result'], 1)).toEqual([
          reader === 'density' || reader === 'flow' ? expectedCell : 1
        ]);
        if (reader === 'encounters')
          expect(await readUint32(fixture.buffers['hit-pairs'], 2)).toEqual([0, 1]);
      }
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}
