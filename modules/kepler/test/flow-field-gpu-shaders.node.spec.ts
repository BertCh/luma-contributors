// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  FLOW_FIELD_GPU_BINDINGS,
  FLOW_FIELD_GPU_SHADERS,
  FlowFieldGpuGrid,
  makeFlowFieldGpuColumns
} from '../../../examples/deck/kepler-compute/src/flow-field-gpu';
import {getFlowFieldExtent} from '../../../examples/deck/kepler-compute/src/flow-field-gpu-reference';

describe('flow field GPU compute stages', () => {
  for (const [stage, source] of Object.entries(FLOW_FIELD_GPU_SHADERS)) {
    test(`${stage}: shader bindings match the declared layout and stay within device limits`, () => {
      const declared = [...source.matchAll(/@binding\((\d+)\) var<([^>]+)> (\w+)/g)].filter(
        match => match[3] !== 'parameters'
      );
      const layout = FLOW_FIELD_GPU_BINDINGS[stage];
      expect(declared.map(match => match[3])).toEqual(layout.map(binding => binding.name));
      // Locations are binding index + 1 (binding 0 is the parameter uniform), contiguous.
      declared.forEach((match, index) => expect(Number(match[1])).toBe(index + 1));
      declared.forEach((match, index) => {
        const isReadOnly = match[2].includes('read>') || match[2] === 'storage, read';
        expect(layout[index].type).toBe(isReadOnly ? 'read-only-storage' : 'storage');
      });
      // The WebGPU default limit is eight storage buffers per shader stage.
      expect(layout.length).toBeLessThanOrEqual(8);
      // Writable bindings of one stage must be distinct buffers; names are unique.
      expect(new Set(layout.map(binding => binding.name)).size).toBe(layout.length);
    });
  }

  test('every parameter word the shaders read exists in the uniform', () => {
    const struct = FLOW_FIELD_GPU_SHADERS['clear'].match(/array<vec4<u32>, (\d+)>/)!;
    const wordCount = Number(struct[1]) * 4;
    for (const source of Object.values(FLOW_FIELD_GPU_SHADERS))
      for (const match of source.matchAll(/get(?:Word|Float)\((\d+)u\)/g))
        expect(Number(match[1])).toBeLessThan(wordCount);
  });

  test('GPU columns carry altitude as NaN for rows without one and invalid rows as NaN', () => {
    const samples = {
      longitudes: [0, 1, 2, 3, Number.NaN],
      latitudes: [0, 1, 2, 3, 4],
      first: [1, 1, 1, 1, 1],
      second: [1, 1, 1, 1, 1],
      altitude: [10, Number.NaN, 30, 40, 50]
    };
    const extent = getFlowFieldExtent(samples)!;
    expect(extent.altitudeRange).toEqual([10, 40]);
    const columns = makeFlowFieldGpuColumns(samples, extent);
    expect(
      Array.from(columns.altitudes!.slice(0, 4)).map(value => (Number.isNaN(value) ? null : value))
    ).toEqual([10, null, 30, 40]);
    expect(Number.isNaN(columns.longitudes[4])).toBe(true);
    expect(
      makeFlowFieldGpuColumns({...samples, altitude: undefined}, extent).altitudes
    ).toBeUndefined();
  });

  test('the memory estimate grows with altitude', () => {
    const none = FlowFieldGpuGrid.estimateByteLength(1000, 64, 64, 'none');
    expect(FlowFieldGpuGrid.estimateByteLength(1000, 64, 64, 'first')).toBeGreaterThan(none);
    expect(FlowFieldGpuGrid.estimateByteLength(1000, 64, 64, 'column')).toBeGreaterThan(
      FlowFieldGpuGrid.estimateByteLength(1000, 64, 64, 'first')
    );
  });
});
