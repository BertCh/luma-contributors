// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUCellCompaction,
  GPUCellSetOutline,
  type GPUCellCompactionInput,
  type GPUCellSetOutlineInput
} from '../../../src/gpu-spatial-analysis';
import {resolveCountedCellSource} from '../../../src/gpu-spatial-analysis/cell-aggregation/counted-cell-source';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('TSC-01 retains caller fields, definite views and the real counted constructor shapes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
    const source = {
      cells: createTransientView(graph, 'cells', 'uint32x2', 1),
      status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
    };
    const diagnosticSource = {
      ...source,
      featureIds: scalar(),
      status: {...source.status, candidateCount: scalar(), uncertainCount: scalar()}
    };
    const diagnosticSnapshot = resolveCountedCellSource('diagnostics', {source: diagnosticSource});
    expect(diagnosticSnapshot.source).toMatchObject(diagnosticSource);
    expect((diagnosticSnapshot.source as typeof diagnosticSource).featureIds).toBe(
      diagnosticSource.featureIds
    );
    expect(
      (diagnosticSnapshot.source.status as typeof diagnosticSource.status).candidateCount
    ).toBe(diagnosticSource.status.candidateCount);
    expect(
      (diagnosticSnapshot.source.status as typeof diagnosticSource.status).uncertainCount
    ).toBe(diagnosticSource.status.uncertainCount);
    const explicitUndefined = resolveCountedCellSource('undefined', {
      source,
      cells: undefined,
      count: undefined,
      custom: 23
    });
    expectTypeOf(explicitUndefined.cells).toEqualTypeOf<GraphDataView<'uint32x2'>>();
    expectTypeOf(explicitUndefined.count).toEqualTypeOf<GraphDataView<'uint32'>>();
    expectTypeOf(explicitUndefined.custom).toEqualTypeOf<number>();
    expect(explicitUndefined.cells).toBe(source.cells);
    expect(explicitUndefined.count).toBe(source.status.count);
    const resolved = resolveCountedCellSource('typed', {
      source,
      family: 'quadbin' as const,
      custom: 17
    });
    expectTypeOf(resolved.cells).toEqualTypeOf<GraphDataView<'uint32x2'>>();
    expectTypeOf(resolved.count).toEqualTypeOf<GraphDataView<'uint32'>>();
    expectTypeOf(resolved.family).toEqualTypeOf<'quadbin'>();
    expectTypeOf(resolved.custom).toEqualTypeOf<number>();
    expect(resolved.custom).toBe(17);
    expect(resolved.cells).toBe(source.cells);
    expect(resolved.count).toBe(source.status.count);
    expect(resolved.source.status.candidateOverflow).toBe(source.status.candidateOverflow);
    expect(Object.isFrozen(resolved)).toBe(true);
    expect(Object.isFrozen(resolved.source)).toBe(true);
    expect(Object.isFrozen(resolved.source.status)).toBe(true);
    const original = {...source.status};
    source.status.count = scalar();
    expect(resolved.source.status.count).toBe(original.count);
    const bare = resolveCountedCellSource('bare', {cells: source.cells, custom: 'kept'});
    expectTypeOf(bare.cells).toEqualTypeOf<GraphDataView<'uint32x2'>>();
    expect(bare.custom).toBe('kept');
    const dual = resolveCountedCellSource('dual', {
      source,
      cells: source.cells,
      count: source.status.count
    });
    expect(dual.count).toBe(source.status.count);
    expect(() => resolveCountedCellSource('conflict', {source, count: scalar()})).toThrow(
      /count must equal/
    );
    expect(() =>
      resolveCountedCellSource('conflict', {
        source,
        cells: createTransientView(graph, 'other-cells', 'uint32x2', 1)
      })
    ).toThrow(/cells must equal/);
    const status = {count: scalar(), overflow: scalar()};
    const compact: GPUCellCompactionInput = {
      family: 'quadbin',
      source,
      operation: {type: 'compact', resolution: 2},
      output: {cells: createTransientView(graph, 'compact', 'uint32x2', 4), status}
    };
    const outline: GPUCellSetOutlineInput = {
      family: 'quadbin',
      source,
      output: {
        cells: createTransientView(graph, 'outline', 'uint32x2', 4),
        rows: createTransientView(graph, 'rows', 'uint32', 4),
        edgeIndices: createTransientView(graph, 'edges', 'uint32', 4),
        endpoints: createTransientView(graph, 'endpoints', 'float32x4', 4),
        status
      }
    };
    const resolvedCompact = resolveCountedCellSource('compact', compact);
    const resolvedOutline = resolveCountedCellSource('outline', outline);
    expectTypeOf(resolvedCompact.operation).toEqualTypeOf<typeof compact.operation>();
    expectTypeOf(resolvedOutline.output).toEqualTypeOf<typeof outline.output>();
    expect(new GPUCellCompaction(compact).props.cells).toBe(source.cells);
    expect(new GPUCellSetOutline(outline).props.cells).toBe(source.cells);
  } finally {
    device.destroy();
  }
});
