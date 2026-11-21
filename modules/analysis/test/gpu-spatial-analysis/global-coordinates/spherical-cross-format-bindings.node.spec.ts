// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createRequire} from 'node:module';
import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  GraphVectorView,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {getCellPackedFields} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-dispatch-validation';
import {
  getTrajectoryBindingViews,
  validateTrajectoryWritableDispatches,
  validateTrajectoryBindings
} from '../../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-binding-validation';
import {validateUrbanRecipeBindings} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-validation';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {addPreparedRecipe} from '../../../src/gpu-spatial-analysis/recipes/prepared-recipe';
import {GPUSphericalConvexContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const foreign = createRequire(import.meta.url)('@luma.gl/graph/gpu-core') as Pick<
  typeof import('@luma.gl/graph/gpu-core'),
  'GraphDataView' | 'GraphVectorView'
>;
function getFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const data = <Format extends VertexFormat>(format: Format, length = 1): GraphDataView<Format> => {
    const source = createTransientView(graph, `data-${serial++}`, format, length);
    return new foreign.GraphDataView<Format>(source.buffer, source);
  };
  const vector = <Format extends VertexFormat>(format: Format): GraphVectorView<Format> => {
    const chunks = [data(format), data(format)];
    return new foreign.GraphVectorView<Format>({
      id: `vector-${serial++}`,
      name: 'rows',
      format,
      length: 2,
      valueLength: (2 * chunks[0].rowByteLength) / 4,
      stride: chunks[0].rowByteLength / 4,
      byteStride: chunks[0].byteStride,
      rowByteLength: chunks[0].rowByteLength,
      data: chunks
    });
  };
  return {device, graph, data, vector};
}

it('ENT-02 collects foreign data/vector ports without reflecting over graph resources', () => {
  const {device, data, vector} = getFixture();
  try {
    const single = data('uint32');
    const rows = vector('uint32');
    expect(single).not.toBeInstanceOf(GraphDataView);
    expect(rows).not.toBeInstanceOf(GraphVectorView);
    expect(getCellPackedFields({status: {count: single}})['status.count']).toBe(single);
    const collected = getTrajectoryBindingViews({
      single,
      nested: {rows},
      invalid: {format: 'uint32'}
    });
    expect(collected['single']).toBe(single);
    expect(collected['nested.rows']).toBe(rows);
    expect(Object.keys(collected)).toEqual(['single', 'nested.rows']);
  } finally {
    device.destroy();
  }
});

it('ENT-02 keeps foreign writable ranges, ownership and urban vector chunks checked', () => {
  const {device, graph, data, vector} = getFixture();
  try {
    const single = data('uint32');
    expect(() =>
      validateTrajectoryWritableDispatches('trajectory', [{first: single}, {second: single}])
    ).toThrow(/overlap/);
    const rows = vector('uint32');
    expect(() => validateUrbanRecipeBindings(graph, 'urban', {}, {rows})).not.toThrow();
    expect(() =>
      validateUrbanRecipeBindings(graph, 'urban', {}, {first: rows, second: rows})
    ).toThrow(/overlap/);
    expect(() =>
      validateTrajectoryBindings(new GPUCommandGraph(device), 'trajectory', {rows})
    ).toThrow(/target graph/);
    expect(() =>
      validateUrbanRecipeBindings(new GPUCommandGraph(device), 'urban', {}, {rows})
    ).toThrow(/target graph/);
    const malformed = {buffer: single.buffer, format: 'uint32', byteOffset: 0};
    expect(getTrajectoryBindingViews({malformed})).toEqual({});
  } finally {
    device.destroy();
  }
});

it('ENT-02 prepares foreign vectors once and commits original buffer/view identities', () => {
  const {device, graph, data, vector} = getFixture();
  try {
    const single = data('uint32');
    const rows = vector('uint32');
    const seen: {single: GraphDataView; rows: GraphVectorView}[] = [];
    addPreparedRecipe(graph, {single, rows}, (target, captured) => {
      seen.push({single: captured.single, rows: captured.rows});
      expect(captured.rows.data).toHaveLength(2);
      for (const chunk of captured.rows.data) expect(chunk.buffer.graph).toBe(target);
    });
    expect(seen).toHaveLength(2);
    expect(seen[0].single).not.toBe(single);
    expect(seen[0].rows).not.toBe(rows);
    expect(seen[1].single).toBe(single);
    expect(seen[1].single.buffer).toBe(single.buffer);
    expect(seen[1].rows).toBe(rows);
    expect(() =>
      addPreparedRecipe(new GPUCommandGraph(device), {rows}, (_target, props) => props.rows)
    ).toThrow(/target graph/);
  } finally {
    device.destroy();
  }
});

it('ENT-02 routes both foreign spherical containment vectors through chunk dispatch', () => {
  const {device, graph, data, vector} = getFixture();
  try {
    const shared = {
      positions: data('float32x3', 3),
      queries: vector('float32x3'),
      polygonIndices: vector('uint32'),
      output: {contains: vector('uint32'), status: vector('uint32')}
    };
    const convex = new GPUSphericalConvexContains({...shared, offsets: data('uint32', 2)});
    const polygon = new GPUSphericalPolygonContains({
      ...shared,
      ringOffsets: data('uint32', 2),
      polygonOffsets: data('uint32', 2)
    });
    expect(convex.estimatedCommandNodeCount).toBe(3); // One parameter pass, two query chunks.
    expect(polygon.estimatedCommandNodeCount).toBe(7); // Three setup passes, two passes per chunk.
    expect(() =>
      Reflect.construct(GPUSphericalConvexContains, [
        {
          ...shared,
          offsets: data('uint32', 2),
          positions: vector('float32x3')
        }
      ])
    ).toThrow(/single|packed/);
    expect(
      () =>
        new GPUSphericalPolygonContains({
          ...shared,
          ringOffsets: data('uint32', 2),
          polygonOffsets: data('uint32', 2),
          polygonIndices: data('uint32', 2)
        })
    ).toThrow(/topology/);
    expect(convex.getCommandNodes(graph)).toHaveLength(3);
    expect(polygon.getCommandNodes(graph)).toHaveLength(7);
  } finally {
    device.destroy();
  }
});

it('B2 captures the consumed public surface-band format once', () => {
  const {device, graph, data} = getFixture();
  try {
    let reads = 0;
    const result = addSurfacePreparationRecipe(graph, {
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      input: {
        id: 'band',
        get format() {
          return ++reads === 1 ? ('float32' as const) : ('uint32' as never);
        },
        storage: {kind: 'buffer', values: data('float32', 3)}
      },
      maximumDistance: 1
    });
    expect(result.outputs.values.format).toBe('float32');
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('B2 leaves unrelated configuration getters unread', () => {
  const {device, graph} = getFixture();
  try {
    let reads = 0;
    const props = {
      ok: 7,
      unused: {
        get format() {
          reads++;
          throw new Error('unused format');
        }
      }
    };
    const result = addPreparedRecipe(graph, props, (_target, captured) => {
      // Access the unused metadata record but consume no field in it.
      expect(captured.unused).toBeDefined();
      return captured.ok;
    });
    expect(result).toBe(7);
    expect(reads).toBe(0);
  } finally {
    device.destroy();
  }
});

it('S1 rejects malformed urban view leaves with contributor and port context', () => {
  const {device, graph, data} = getFixture();
  try {
    const real = data('uint32');
    expect(() =>
      validateUrbanRecipeBindings(
        graph,
        'urban-malformed',
        {},
        {
          nested: {values: {buffer: real.buffer, format: 'uint32', byteOffset: 0}}
        }
      )
    ).toThrow(/urban-malformed.*output.nested.values.*graph.*view/);
  } finally {
    device.destroy();
  }
});
