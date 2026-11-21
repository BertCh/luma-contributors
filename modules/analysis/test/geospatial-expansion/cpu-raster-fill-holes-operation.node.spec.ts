// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import type {GPURasterFillHoles} from '../../src/raster-analysis/gpu-raster-reconstruction';
import * as neighbors from '../../src/raster-analysis/raster-morphology/raster-morphology-operation-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  computeRasterFillHolesOnCPU,
  CPURasterFillHolesOperation
} from '../../src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation';
import {
  GPURasterFillHolesOperation,
  type RasterFillHolesOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import fixtures from './raster-morphology-operation-fixtures.json';

const GRID: GPURasterMetadata = {
  width: 9,
  height: 7,
  affine: [1, 0, 0, 0, -1, 7],
  pixelInterpretation: 'area'
};
const PROPS: RasterFillHolesOperationProps = {
  id: 'fill',
  grid: GRID,
  input: 'input',
  output: {
    values: 'filled',
    status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
  }
};

it('R6 never expands cells beyond the budget frontier on a 101 by 101 empty grid', () => {
  const visit = vi.spyOn(neighbors, 'visitRasterMorphologyNeighbors');
  try {
    const result = computeRasterFillHolesOnCPU(new Uint32Array(101 * 101), {
      grid: {...GRID, width: 101, height: 101},
      maximumIterations: 1
    });
    const deepest = Math.max(
      ...visit.mock.calls.map(([cell]) => {
        const row = Math.floor(cell / 101),
          column = cell % 101;
        return Math.min(row, column, 100 - row, 100 - column);
      })
    );
    expect(deepest).toBe(1);
    expect(visit).toHaveBeenCalledTimes(792);
    expect(result.iterationCount[0]).toBe(2);
    expect(result.converged[0]).toBe(0);
    expect(result.values[50 * 101 + 50]).toBe(1);
  } finally {
    visit.mockRestore();
  }
});

it('matches independent SciPy component filling for both connectivities and nonbinary foreground', () => {
  for (const fixture of fixtures.fillHoles) {
    const result = computeRasterFillHolesOnCPU(Uint32Array.from(fixture.input), {
      grid: {...GRID, width: fixture.width, height: fixture.height},
      connectivity: fixture.connectivity as 4 | 8
    });
    expect(Array.from(result.values)).toEqual(fixture.expected);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});

it('counts shortest boundary-background layers, including the verifying round', () => {
  const empty = computeRasterFillHolesOnCPU(new Uint32Array(63), {grid: GRID});
  expect(empty.values.every(value => value === 0)).toBe(true);
  // The most distant pixel is 3 edges from a boundary of the 9 by 7 rectangle.
  expect(empty.iterationCount[0]).toBe(4);
  const full = computeRasterFillHolesOnCPU(new Uint32Array(63).fill(15), {grid: GRID});
  expect(full.values.every(value => value === 1)).toBe(true);
  expect(full.iterationCount[0]).toBe(1);
  expect(() => computeRasterFillHolesOnCPU(new Uint32Array(62), {grid: GRID})).toThrow(/input/);
});

it('validates both operations at construction and drops only the CPU command budget', () => {
  const inputs = {input: {format: 'uint32' as const, length: 63}};
  expect(() => new CPURasterFillHolesOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^fill .*maximumIterations/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFillHolesOperation({...PROPS, maximumIterations: 0}),
      inputs
    )
  ).toThrow(/^fill .*maximumIterations/);
  for (const Operation of [CPURasterFillHolesOperation, GPURasterFillHolesOperation])
    expect(() => new Operation({...PROPS, grid: {...GRID, width: 0}})).toThrow(/^fill /);
  expect(() => new CPURasterFillHolesOperation({...PROPS, maximumIterations: 30000})).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFillHolesOperation({...PROPS, maximumIterations: 30000}),
      inputs
    )
  ).toThrow(/^fill /);
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation(PROPS), inputs)
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation(PROPS), {
      input: {format: 'uint32', length: 62}
    })
  ).toThrow(/^fill .*input/);
});

it('maps optional status names and the output callback exactly', async () => {
  const operation = new CPURasterFillHolesOperation({...PROPS, maximumIterations: 1});
  const values = new Uint32Array(63);
  const result = await operation.execute({
    inputs: {
      input: {format: 'uint32', length: 63, chunks: [{values, length: 63}], pack: () => values}
    }
  });
  expect(Object.keys(result).sort()).toEqual(['converged', 'filled', 'limit', 'rounds']);
  expect(result['rounds']).toEqual(Uint32Array.of(2));
  expect(result['converged']).toEqual(Uint32Array.of(0));
  expect(result['limit']).toEqual(Uint32Array.of(1));
  const minimal = new CPURasterFillHolesOperation({
    ...PROPS,
    output: {values: 'filled', status: {converged: 'done', iterationCount: undefined}}
  });
  expect(
    Object.keys(
      await minimal.execute({
        inputs: {
          input: {format: 'uint32', length: 63, chunks: [{values, length: 63}], pack: () => values}
        }
      })
    ).sort()
  ).toEqual(['done', 'filled']);
});

it('keeps background beyond the one-round boundary flood marked as unfinished holes', () => {
  const result = computeRasterFillHolesOnCPU(new Uint32Array(63), {
    grid: GRID,
    maximumIterations: 1
  });
  // Distance from the boundary is min(row, column, 6-row, 8-column).
  expect(result.values).toEqual(
    Uint32Array.from({length: 63}, (_, cell) => {
      const row = Math.floor(cell / 9);
      const column = cell % 9;
      return Number(Math.min(row, column, 6 - row, 8 - column) > 1);
    })
  );
  expect(result.converged[0]).toBe(0);
  expect(result.iterationLimitReached[0]).toBe(1);
  expect(result.iterationCount[0]).toBe(2);
});

it('ignores surplus input rows and returns only the unit-grid boundary background', () => {
  const grid = {...GRID, width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1] as const};
  const result = computeRasterFillHolesOnCPU(Uint32Array.of(0, 7), {grid});
  // The sole grid cell is background on the boundary, so it is never filled.
  expect(result.values).toEqual(Uint32Array.of(0));
  expect(result.converged).toEqual(Uint32Array.of(1));
  expect(result.iterationCount).toEqual(Uint32Array.of(1));
  expect(result.iterationLimitReached).toEqual(Uint32Array.of(0));
  expect(() =>
    constructRasterOperationContributor(new GPURasterFillHolesOperation({...PROPS, grid}), {
      input: {format: 'uint32', length: 2}
    })
  ).not.toThrow();
  expect(() => computeRasterFillHolesOnCPU(new Uint32Array(0), {grid})).toThrow(/input/);
});

for (const defect of [
  'empty-result',
  'empty-status',
  'duplicate-status',
  'duplicate-result',
  'input-result',
  'input-status',
  'empty-input'
] as const) {
  it(`FRM-01 CPURasterFillHolesOperation rejects ${defect} before keying outputs`, () => {
    const props = {...PROPS, output: {...PROPS.output, status: {...PROPS.output.status}}};
    if (defect === 'empty-result') props.output.values = '';
    if (defect === 'empty-status') props.output.status.iterationCount = '';
    if (defect === 'duplicate-status')
      props.output.status.iterationCount = props.output.status.converged;
    if (defect === 'duplicate-result') props.output.values = props.output.status.converged;
    if (defect === 'input-result') props.output.values = 'input';
    if (defect === 'input-status') props.output.status.iterationCount = 'input';
    if (defect === 'empty-input') props.input = '';
    for (const Operation of [CPURasterFillHolesOperation, GPURasterFillHolesOperation]) {
      expect(() => new Operation(props)).toThrow(/graph value/);
    }
  });
}

it('B3 captures inherited iterationCount from the documented status map', async () => {
  let destination = 'rounds';
  const status = {...PROPS.output.status};
  Reflect.deleteProperty(status, 'iterationCount');
  Object.setPrototypeOf(
    status,
    new (class {
      get iterationCount() {
        return destination;
      }
    })()
  );
  const operation = new CPURasterFillHolesOperation({
    ...PROPS,
    maximumIterations: 1,
    output: {...PROPS.output, status}
  });
  expect(Object.keys(operation.outputs)).toContain('rounds');
  destination = 'replaced';
  const values = new Uint32Array(63);
  const inputs = {
    input: {
      format: 'uint32' as const,
      length: 63,
      chunks: [{values, length: 63}],
      pack: () => values
    }
  };
  const result = await operation.execute({inputs});
  expect(result['rounds']).toEqual(Uint32Array.of(2));
  expect(Object.keys(result).sort()).toEqual(Object.keys(operation.outputs).sort());
});
it('B3 ignores unrelated shared status fields in both declaration and execution', async () => {
  const status = {converged: 'done', invalidCount: 'invalid'};
  const operation = new CPURasterFillHolesOperation({...PROPS, output: {...PROPS.output, status}});
  const values = new Uint32Array(63);
  const inputs = {
    input: {
      format: 'uint32' as const,
      length: 63,
      chunks: [{values, length: 63}],
      pack: () => values
    }
  };
  const result = await operation.execute({inputs});
  expect(Object.keys(operation.outputs)).not.toContain('invalid');
  expect(Object.keys(result).sort()).toEqual(Object.keys(operation.outputs).sort());
});

it('W5-09 GPU fill holes captures inherited status through lowering', () => {
  let destination = 'rounds';
  let reads = 0;
  const status = new (class {
    get converged() {
      return 'done';
    }
    get iterationCount() {
      reads++;
      return destination;
    }
    get iterationLimitReached() {
      return 'limited';
    }
  })();
  const operation = new GPURasterFillHolesOperation({...PROPS, output: {values: 'filled', status}});
  destination = 'changed';
  expect(reads).toBe(1);
  expect(operation.outputs['iterationCount']?.value).toBe('rounds');
  expect(operation.outputs['iterationLimitReached']?.value).toBe('limited');
  const context = createKernelAuditContext('W5-09-fill');
  try {
    const views = Object.fromEntries(
      Object.entries(operation.outputs).map(([port, descriptor]) => [
        port,
        context.view(descriptor.format, descriptor.length)
      ])
    );
    views['input'] = context.view('uint32', GRID.width * GRID.height);
    const contributor = operation.createContributor(
      port => views[port] as never
    ) as GPURasterFillHoles;
    expect(contributor.props.output.status.iterationCount).toBe(views['iterationCount']);
    expect(contributor.props.output.status.iterationLimitReached).toBe(
      views['iterationLimitReached']
    );
  } finally {
    context.device.destroy();
    operation.destroy();
  }
  expect(reads).toBe(1);
});

for (const Operation of [CPURasterFillHolesOperation, GPURasterFillHolesOperation]) {
  it(`B2 ${Operation.name} ignores unrelated reconstruction fields`, () => {
    const props = {...PROPS, mask: Uint32Array.of(0), marker: Uint32Array.of(0)};
    expect(() => new Operation(props)).not.toThrow();
    Object.defineProperty(props, 'mask', {
      get() {
        throw new Error('unrelated mask was read');
      }
    });
    Object.defineProperty(props, 'marker', {
      get() {
        throw new Error('unrelated marker was read');
      }
    });
    expect(() => new Operation(props)).not.toThrow();
  });
}
