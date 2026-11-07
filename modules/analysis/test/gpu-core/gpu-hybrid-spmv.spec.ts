// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUAdaptiveSpMV, type GPUAdaptiveSpMVProps} from '@luma.gl/graph/gpu-core';
import {GPUJacobiPCG} from '@luma.gl/graph/gpu-core';
import {selectGPUSpMVStrategy} from '@luma.gl/graph/gpu-core';
import {getGraphVectorData} from '@luma.gl/graph/gpu-core';
import {getGPUComputeDispatchWorkgroups} from '@luma.gl/graph/gpu-core';
import {BatchConformanceFixture} from './batch-conformance-utils';
import {requireTestDevice} from '../utils/require-test-device';

function writeValues(
  fixture: BatchConformanceFixture,
  view: GPUAdaptiveSpMVProps['vector'],
  values: number[]
) {
  let offset = 0;
  for (const chunk of getGraphVectorData(view)) {
    fixture.storage
      .get(chunk.buffer)!
      .write(Float32Array.from(values.slice(offset, offset + chunk.length)), chunk.byteOffset);
    offset += chunk.length;
  }
}

for (const featureLevel of ['core', 'max'] as const) {
  test(`shared SpMV selector preserves defaults and forces supported hybrid ${featureLevel}`, async () => {
    const device = await getWebGPUTestDevice(featureLevel);
    requireTestDevice(device);
    for (const nonZeros of [0, 1, 4095, 4096, 65536]) {
      const workload = {rows: 512, nonZeros};
      const original = selectGPUSpMVStrategy(device, workload);
      expect(selectGPUSpMVStrategy(device, {...workload, storage: 'flat'})).toEqual(original);
      expect(selectGPUSpMVStrategy(device, {...workload, storage: 'chunked'})).toEqual(original);
      const forced = selectGPUSpMVStrategy(device, {...workload, storage: 'flat'}, 'hybrid-row');
      expect(forced.id).toBe('hybrid-row');
      expect(forced.details.merge).toEqual({
        workgroupSize: 256,
        tileSize: 1024,
        maximumScalarRowLength: 256
      });
      expect(() => selectGPUSpMVStrategy(device, workload, 'hybrid-row')).toThrow(/not supported/);
      expect(() =>
        selectGPUSpMVStrategy(device, {...workload, storage: 'chunked'}, 'hybrid-row')
      ).toThrow(/not supported/);
    }
    expect(
      selectGPUSpMVStrategy(device, {rows: 1, nonZeros: 0xfffffffe, storage: 'flat'}, 'hybrid-row')
        .id
    ).toBe('hybrid-row');
    expect(() =>
      selectGPUSpMVStrategy(device, {rows: 2, nonZeros: 0xfffffffe, storage: 'flat'}, 'hybrid-row')
    ).toThrow(/not supported/);
  });

  for (const rowLengths of [
    [],
    [0, 0, 0],
    [1],
    [255, 256, 257],
    [257, 257, 257],
    [0, 1, 255, 256, 257, 8192, 3]
  ]) {
    test(`shared SpMV executes forced hybrid ${featureLevel} rows=${rowLengths.join(',')}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const fixture = new BatchConformanceFixture(device);
      const leadingSlots = rowLengths.some(length => length > 0) ? 3 : 0;
      const rowOffsets = [leadingSlots];
      const columns = Array(leadingSlots).fill(999);
      const matrix = Array(leadingSlots).fill(999);
      const vectorValues = rowLengths.some(length => length > 0) ? [1, 2, -0.5, 4] : [];
      const expected: number[] = [];
      for (const length of rowLengths) {
        let product = 0;
        for (let entry = 0; entry < length; entry++) {
          const column = entry % 101 === 100 ? 999 : entry % 4;
          const value = ((entry % 9) - 4) / 16;
          columns.push(column);
          matrix.push(value);
          if (column < vectorValues.length) product += value * vectorValues[column];
        }
        rowOffsets.push(columns.length);
        expected.push(product);
      }
      if (leadingSlots) {
        columns.push(999, 999);
        matrix.push(999, 999);
      }
      const output = fixture.column(
        'output',
        'float32',
        rowLengths.map(() => 77),
        [rowLengths.length]
      );
      const vector = fixture.column('vector', 'float32', vectorValues, [vectorValues.length]);
      const operation = new GPUAdaptiveSpMV({
        id: 'forced',
        strategy: 'hybrid-row',
        rowOffsets: fixture.column('rows', 'uint32', rowOffsets, [rowOffsets.length]),
        columnIndices: fixture.column('columns', 'uint32', columns, [columns.length]),
        values: fixture.column('values', 'float32', matrix, [matrix.length]),
        vector,
        output,
        columns: vectorValues.length
      });
      expect(operation.getStrategy(fixture.graph).id).toBe('hybrid-row');
      expect(operation.explain(fixture.graph).merge).toEqual({
        workgroupSize: 256,
        tileSize: 1024,
        maximumScalarRowLength: 256
      });
      const nodes = operation.getCommandNodes(fixture.graph);
      if (!rowLengths.length) {
        expect(nodes).toHaveLength(0);
        fixture.destroy();
        return;
      }
      expect(nodes.filter(node => node.id.startsWith('forced-long-rows')).length).toBe(2);
      expect(nodes.every(node => getGPUComputeDispatchWorkgroups(node) !== undefined)).toBe(true);
      fixture.graph.add(nodes);
      const compiled = fixture.graph.compile();
      try {
        for (const multiplier of [1, 2, 1]) {
          writeValues(
            fixture,
            vector,
            vectorValues.map(value => value * multiplier)
          );
          const encoder = device.createCommandEncoder();
          compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          expect(await fixture.read(output)).toEqual(expected.map(value => value * multiplier));
        }
      } finally {
        compiled.destroy();
        expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
        fixture.destroy();
      }
    });
  }

  for (const fragmented of ['rowOffsets', 'columnIndices', 'values', 'vector', 'output'] as const) {
    test(`shared SpMV rejects explicit hybrid fragmented ${fragmented} ${featureLevel}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const fixture = new BatchConformanceFixture(device);
      const partition = (name: string, length: number) =>
        name === fragmented ? [1, 0, length - 1] : [length];
      const props = {
        rowOffsets: fixture.column('rows', 'uint32', [0, 1, 2], partition('rowOffsets', 3)),
        columnIndices: fixture.column('columns', 'uint32', [0, 1], partition('columnIndices', 2)),
        values: fixture.column('values', 'float32', [2, 3], partition('values', 2)),
        vector: fixture.column('vector', 'float32', [4, 5], partition('vector', 2)),
        output: fixture.column('output', 'float32', [77, 77], partition('output', 2)),
        columns: 2
      };
      const operation = new GPUAdaptiveSpMV({...props, strategy: 'hybrid-row'});
      expect(() => operation.getStrategy(fixture.graph)).toThrow(/not supported/);
      expect(() => operation.getCommandNodes(fixture.graph)).toThrow(/not supported/);
      const ordinary = new GPUAdaptiveSpMV(props);
      expect(ordinary.getStrategy(fixture.graph).id).not.toBe('hybrid-row');
      fixture.graph.add(ordinary);
      const compiled = fixture.graph.compile();
      try {
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        expect(await fixture.read(props.output)).toEqual([8, 15]);
      } finally {
        compiled.destroy();
        expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
        fixture.destroy();
      }
    });
  }

  test(`shared SpMV preserves scalar short-row rounding and late negative long terms ${featureLevel}`, async () => {
    const device = await getWebGPUTestDevice(featureLevel);
    requireTestDevice(device);
    const fixture = new BatchConformanceFixture(device);
    const matrix = [1e10, 1, -1e10, ...Array(8192).fill(1), -8192];
    const output = fixture.column('output', 'float32', [77, 77], [2]);
    fixture.graph.add(
      new GPUAdaptiveSpMV({
        strategy: 'hybrid-row',
        rowOffsets: fixture.column('rows', 'uint32', [0, 3, matrix.length], [3]),
        columnIndices: fixture.column(
          'columns',
          'uint32',
          matrix.map(() => 0),
          [matrix.length]
        ),
        values: fixture.column('values', 'float32', matrix, [matrix.length]),
        vector: fixture.column('vector', 'float32', [1], [1]),
        output,
        columns: 1
      })
    );
    const compiled = fixture.graph.compile();
    try {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      expect(await fixture.read(output)).toEqual([0, 0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });

  for (const initialFraction of [0, 0.5, 1]) {
    test(`shared PCG forces mostly-long hybrid and replays changed rhs ${featureLevel} initial=${initialFraction}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const fixture = new BatchConformanceFixture(device);
      const rows = 4;
      const columns: number[] = [];
      const values: number[] = [];
      const offsets = [0];
      for (let row = 0; row < rows; row++) {
        for (let entry = 0; entry < 256; entry++) {
          columns.push(row);
          values.push(1 / 64);
        }
        columns.push(row ^ 1);
        values.push(1);
        offsets.push(values.length);
      }
      const target = [1, 2, 1, 2];
      const rightHandSide = [6, 9, 6, 9];
      const rhs = fixture.column('rhs', 'float32', rightHandSide, [rows]);
      const solution = fixture.column(
        'solution',
        'float32',
        target.map(value => value * initialFraction),
        [rows]
      );
      const solver = new GPUJacobiPCG({
        rowOffsets: fixture.column('rows', 'uint32', offsets, [offsets.length]),
        columnIndices: fixture.column('columns', 'uint32', columns, [columns.length]),
        values: fixture.column('values', 'float32', values, [values.length]),
        rhs,
        solution,
        columns: rows,
        iterations: 8,
        toleranceSquared: 1e-10,
        spmvStrategy: 'hybrid-row'
      });
      fixture.graph.add(solver);
      const result = solver.getResult();
      expect(result.spmvStrategy).toBe('hybrid-row');
      const initialNorm = fixture.capture('initial', result.initialResidualSquared.view);
      const finalNorm = fixture.capture('final', result.residualSquared.view);
      const breakdown = fixture.capture('breakdown', result.breakdown.view);
      const compiled = fixture.graph.compile();
      let previousSolution = target.map(value => value * initialFraction);
      try {
        for (const addition of [0, 0, 2, 0]) {
          const expected = target.map(value => value + addition);
          const nextRhs = rightHandSide.map(value => value + 5 * addition);
          const expectedInitial = nextRhs.reduce(
            (sum, value, row) =>
              sum + (value - 4 * previousSolution[row] - previousSolution[row ^ 1]) ** 2,
            0
          );
          writeValues(fixture, rhs, nextRhs);
          const encoder = device.createCommandEncoder();
          compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          const actual = await fixture.read(solution);
          for (let row = 0; row < rows; row++) {
            expect(Math.abs(actual[row] - expected[row])).toBeLessThanOrEqual(1e-5);
            expect(Math.abs(4 * actual[row] + actual[row ^ 1] - nextRhs[row])).toBeLessThanOrEqual(
              1e-4
            );
          }
          expect(
            Math.abs((await fixture.read(initialNorm))[0] - expectedInitial)
          ).toBeLessThanOrEqual(1e-4);
          expect((await fixture.read(finalNorm))[0]).toBeLessThanOrEqual(1e-10);
          expect(await fixture.read(breakdown)).toEqual([0]);
          previousSolution = actual;
        }
      } finally {
        compiled.destroy();
        expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
        fixture.destroy();
      }
    });
  }

  for (const diagonal of [0, -1]) {
    test(`shared PCG hybrid gates unusable diagonal ${diagonal} ${featureLevel}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const fixture = new BatchConformanceFixture(device);
      const solution = fixture.column('solution', 'float32', [0], [1]);
      const solver = new GPUJacobiPCG({
        rowOffsets: fixture.column('rows', 'uint32', [0, 256], [2]),
        columnIndices: fixture.column('columns', 'uint32', Array(256).fill(0), [256]),
        values: fixture.column('values', 'float32', Array(256).fill(diagonal / 256), [256]),
        rhs: fixture.column('rhs', 'float32', [1], [1]),
        solution,
        columns: 1,
        iterations: 4,
        spmvStrategy: 'hybrid-row'
      });
      fixture.graph.add(solver);
      expect(solver.getResult().spmvStrategy).toBe('hybrid-row');
      const breakdown = fixture.capture('breakdown', solver.getResult().breakdown.view);
      const compiled = fixture.graph.compile();
      try {
        for (let replay = 0; replay < 2; replay++) {
          const encoder = device.createCommandEncoder();
          compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          expect(await fixture.read(solution)).toEqual([0]);
          expect(await fixture.read(breakdown)).toEqual([1]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

import {GPUData} from '@luma.gl/graph/gpu-data';
import {
  GPUGraph,
  GPUGraphCSRMatrix,
  GPUGraphCompiler,
  GPUGraphSpMV,
  GPUGraphScalarLiteral,
  GPUConditionalOperation
} from '@luma.gl/graph/gpu-core';

for (const featureLevel of ['core', 'max'] as const) {
  for (const enabled of [0, 1]) {
    test(`semantic forced hybrid obeys GPU predicate ${enabled} ${featureLevel}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const fixture = new BatchConformanceFixture(device);
      const rowOffsets = fixture.column('rows', 'uint32', [0, 257, 258], [3]);
      const columnIndices = fixture.column(
        'columns',
        'uint32',
        [...Array(256).fill(0), 2, 1],
        [258]
      );
      const values = fixture.column('values', 'float32', [...Array(256).fill(0.5), 3, 4], [258]);
      const vector = fixture.column('vector', 'float32', [10, 20, 30], [3]);
      const output = fixture.column('output', 'float32', [77, 77], [2]);
      const bindings = Object.fromEntries(
        Object.entries({rowOffsets, columnIndices, values, vector, output}).map(([name, view]) => [
          name,
          getGraphVectorData(view).map(
            chunk =>
              new GPUData({
                buffer: fixture.storage.get(chunk.buffer)!,
                format: chunk.format,
                length: chunk.length,
                byteOffset: chunk.byteOffset
              })
          )
        ])
      );
      const program = new GPUGraph();
      const matrix = new GPUGraphCSRMatrix({
        id: 'matrix',
        rows: 2,
        columns: 3,
        rowOffsets: program.vector('rowOffsets', 'uint32', 3, {external: true}),
        columnIndices: program.vector('columnIndices', 'uint32', 258, {external: true}),
        values: program.vector('values', 'float32', 258, {external: true})
      });
      const input = program.vector('vector', 'float32', 3, {external: true});
      const result = program.vector('output', 'float32', 2, {external: true});
      const active = program.scalar('active', 'uint32');
      program.add([
        new GPUGraphScalarLiteral({output: active, value: enabled}),
        new GPUConditionalOperation({
          predicate: {id: 'active', source: 'gpu', value: active},
          body: [new GPUGraphSpMV({matrix, vector: input, output: result, strategy: 'hybrid-row'})]
        })
      ]);
      const compilation = new GPUGraphCompiler(device).compile(program, {vectors: bindings});
      const compiled = compilation.graph.compile();
      expect(compiled.stats.nodeOrder.some(id => id.includes('long-rows'))).toBe(true);
      try {
        for (let replay = 0; replay < 2; replay++) {
          const encoder = device.createCommandEncoder();
          compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          expect(await fixture.read(output)).toEqual(enabled ? [1370, 80] : [77, 77]);
        }
      } finally {
        compiled.destroy();
        expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
        fixture.destroy();
      }
    });
  }

  test(`shared PCG forced hybrid detects negative curvature ${featureLevel}`, async () => {
    const device = await getWebGPUTestDevice(featureLevel);
    requireTestDevice(device);
    const fixture = new BatchConformanceFixture(device);
    const columns = [...Array(256).fill(0), 1, ...Array(256).fill(1), 0];
    const values = [...Array(256).fill(1 / 256), 2, ...Array(256).fill(1 / 256), 2];
    const solution = fixture.column('solution', 'float32', [0, 0], [2]);
    const solver = new GPUJacobiPCG({
      rowOffsets: fixture.column('rows', 'uint32', [0, 257, 514], [3]),
      columnIndices: fixture.column('columns', 'uint32', columns, [514]),
      values: fixture.column('values', 'float32', values, [514]),
      rhs: fixture.column('rhs', 'float32', [1, -1], [2]),
      solution,
      columns: 2,
      iterations: 4,
      spmvStrategy: 'hybrid-row'
    });
    fixture.graph.add(solver);
    expect(solver.getResult().spmvStrategy).toBe('hybrid-row');
    const breakdown = fixture.capture('breakdown', solver.getResult().breakdown.view);
    const compiled = fixture.graph.compile();
    try {
      for (let replay = 0; replay < 2; replay++) {
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        expect(await fixture.read(solution)).toEqual([0, 0]);
        expect(await fixture.read(breakdown)).toEqual([1]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });

  test(`shared PCG rejects unsupported forced storage ${featureLevel}`, async () => {
    const device = await getWebGPUTestDevice(featureLevel);
    requireTestDevice(device);
    const fixture = new BatchConformanceFixture(device);
    const solver = new GPUJacobiPCG({
      rowOffsets: fixture.column('rows', 'uint32', [0, 1, 2], [3]),
      columnIndices: fixture.column('columns', 'uint32', [0, 1], [2]),
      values: fixture.column('values', 'float32', [1, 1], [2]),
      rhs: fixture.column('rhs', 'float32', [1, 1], [2]),
      solution: fixture.column('solution', 'float32', [0, 0], [1, 1]),
      columns: 2,
      iterations: 2,
      spmvStrategy: 'hybrid-row'
    });
    try {
      expect(() => solver.getCommandNodes(fixture.graph)).toThrow(/not supported/);
      expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
    } finally {
      fixture.destroy();
    }
  });
}
