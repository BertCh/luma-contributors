// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUPolygonOverlay,
  preflightGPUPolygonOverlay,
  planGPUPolygonOverlay
} from '../../../src/gpu-spatial-analysis/polygon-overlay';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const polygon = (vertices: number, rings = 1) => ({
    kind: 'polygons' as const,
    positions: view('float32x2', vertices),
    ringOffsets: view('uint32', rings + 1),
    polygonOffsets: view('uint32', rings + 1),
    featureOffsets: view('uint32', rings + 1)
  });
  return {
    device,
    graph,
    view,
    props: {
      id: 'preflight-review',
      operation: 'dissolve' as const,
      left: polygon(3),
      vertexTolerance: 1,
      capacity: {intersections: 8, nodedSegments: 16},
      output: {
        geometry: polygon(16, 4),
        boundary: {
          endpoints: view('float32x4', 16),
          operandIds: view('uint32', 16),
          sourceFeatureIds: view('uint32', 16),
          status: {
            count: view('uint32', 1),
            requiredCount: view('uint32', 1),
            overflow: view('uint32', 1),
            candidateOverflow: view('uint32', 1)
          }
        },
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        }
      }
    }
  };
}
for (const setting of ['leaf', 'vertices', 'boundary'] as const) {
  it(`B7 preflights the nested ${setting} requirement without graph mutation`, () => {
    const {device, graph, view, props} = fixture();
    const invalid = {
      ...props,
      ...(setting === 'leaf' ? {leafCapacity: 3} : {}),
      output: {
        ...props.output,
        geometry: {
          ...props.output.geometry,
          positions: setting === 'vertices' ? view('float32x2', 0) : props.output.geometry.positions
        },
        boundary:
          setting === 'boundary'
            ? {
                ...props.output.boundary,
                endpoints: view('float32x4', 0),
                operandIds: view('uint32', 0),
                sourceFeatureIds: view('uint32', 0)
              }
            : props.output.boundary
      }
    };
    const allocate = vi.spyOn(graph, 'createTransientBuffer'),
      add = vi.spyOn(graph, 'add');
    let message = '';
    try {
      preflightGPUPolygonOverlay(graph, invalid);
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).not.toBe('');
    expect(() => new GPUPolygonOverlay(invalid).getCommandNodes(graph)).toThrow(message);
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    device.destroy();
  });
}
for (const setting of ['operation', 'fillRule'] as const) {
  it(`earlier-fix note rejects unknown ${setting} during public preflight`, () => {
    const {device, graph, props} = fixture();
    const invalid =
      setting === 'operation'
        ? {...props, operation: 'xor' as never, right: props.left}
        : {...props, fillRule: '' as never};
    expect(() => preflightGPUPolygonOverlay(graph, invalid)).toThrow(
      new RegExp(`${setting}.*(must be|must select)`)
    );
    device.destroy();
  });
}

it('B8 plans dimensions before any graph exists and matches resident declaration planning', () => {
  const planner = planGPUPolygonOverlay;
  expect(typeof planner).toBe('function');
  const device = createNullWebGPUDevice();
  const plan = planner(device, {
    id: 'dimension-plan',
    operation: 'dissolve',
    vertexTolerance: 1,
    left: {vertexCount: 3, ringCount: 1},
    capacity: {intersections: 8, nodedSegments: 16},
    output: {vertexCapacity: 16, ringCapacity: 4, boundaryCapacity: 16}
  });
  expect(plan.expandedVertexCount).toBe(4);
  expect(plan.hotPixelCapacity).toBe(12);
  const resident = fixture();
  expect(preflightGPUPolygonOverlay(resident.graph, resident.props)).toEqual(plan);
  resident.device.destroy();
  device.destroy();
});
