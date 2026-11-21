// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../../../src/gpu-spatial-analysis/segment-intersection/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleScene, type ViewFactory} from './make-valid-output-oracles-fixtures';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length) =>
    createTransientView(graph, `forward-${serial++}`, format, length);
  return {device, graph, view, ...createOracleScene(view)};
}

it('forward N1 oversized make-valid refusal declares no scratch and permits same-ID retry', () => {
  const {device, graph, callerProps} = createScene();
  try {
    const oversized = new GPUMakeValid({
      ...callerProps,
      polygons: {
        ...callerProps.polygons,
        positions: createTransientView(
          graph,
          'oversized',
          'float32x2',
          getGPUSegmentIntersectionMaximumVertexCount(device)
        )
      }
    });
    // Each of the two rings needs an additional closing vertex, exceeding the vertex bound.
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    try {
      expect(() => oversized.getCommandNodes(graph)).toThrow(/lines.positions rows/);
      expect(declarations).not.toHaveBeenCalled();
    } finally {
      declarations.mockRestore();
    }
    expect(() => new GPUMakeValid(callerProps).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'own',
  'inherited',
  'proxy'
] as const)('forward publication N1 refuses final-only requiredRingCount on directed edges: %s', kind => {
  const {device, callerProps, view} = createScene();
  try {
    const requiredRingCount = view('uint32', 1);
    const original = callerProps.output.directedEdges.status;
    const status =
      kind === 'own'
        ? {...original, requiredRingCount}
        : kind === 'inherited'
          ? Object.assign(Object.create({requiredRingCount}), original)
          : new Proxy(original, {
              get(target, name, receiver) {
                return name === 'requiredRingCount'
                  ? requiredRingCount
                  : Reflect.get(target, name, receiver);
              }
            });
    expect(
      () =>
        new GPUMakeValid({
          ...callerProps,
          output: {
            ...callerProps.output,
            directedEdges: {...callerProps.output.directedEdges, status}
          }
        })
    ).toThrow(/output.directedEdges.status.requiredRingCount is final-only/);
  } finally {
    device.destroy();
  }
});
