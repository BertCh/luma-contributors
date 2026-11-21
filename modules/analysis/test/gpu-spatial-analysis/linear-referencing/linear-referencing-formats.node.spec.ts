// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {
  GPULinearReferencing,
  type GPULinearReferencingProps
} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

it('rejects substituted radius formats at construction with the exact prop path', () => {
  const graph = new GPUCommandGraph(device);
  const props: GPULinearReferencingProps = {
    id: 'format-check',
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    queryPositions: createTransientView(graph, 'queries', 'float32x2', 1),
    radius: createTransientView(graph, 'radius', 'uint32', 1) as never,
    candidateCapacity: 4,
    output: {
      measures: createTransientView(graph, 'measures', 'float32', 1),
      status: {candidateOverflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  };
  expect(() => new GPULinearReferencing(props)).toThrow(
    /format-check radius.*float32.*received uint32/
  );
  try {
    new GPULinearReferencing(props);
  } catch (error) {
    expect(error).toMatchObject({path: 'radius', received: 'uint32'});
  }
});

it.each([
  'positions',
  'lineOffsets',
  'queryPositions',
  'output.measures',
  'output.sides',
  'output.status.candidateOverflow'
])('checks the declared format and structured path of %s', path => {
  const graph = new GPUCommandGraph(device);
  const props: GPULinearReferencingProps = {
    id: 'format-columns',
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    queryPositions: createTransientView(graph, 'queries', 'float32x2', 1),
    radius: createTransientView(graph, 'radius', 'float32', 1),
    candidateCapacity: 4,
    output: {
      measures: createTransientView(graph, 'measures', 'float32', 1),
      sides: createTransientView(graph, 'sides', 'sint32', 1),
      status: {candidateOverflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  };
  const fields = path.split('.');
  const container = (fields.length === 3
    ? props.output.status
    : fields.length === 2
      ? props.output
      : props) as unknown as Record<string, {length: number}>;
  const field = fields.at(-1)!;
  const originalLength = container[field].length;
  container[field] = createTransientView(graph, 'wrong', 'unorm8x4', originalLength);
  expect(() => new GPULinearReferencing(props)).toThrow(/received unorm8x4/);
  try {
    new GPULinearReferencing(props);
  } catch (error) {
    expect(error).toMatchObject({path, received: 'unorm8x4'});
  }
});

for (const spherical of [false, true]) {
  for (const radiusRows of [0, 2]) {
    it(`requires one radius row before scratch (${spherical ? 'great-circle' : 'planar'}, ${radiusRows} rows)`, () => {
      const graph = new GPUCommandGraph(device);
      const correctedRadius = createTransientView(graph, 'corrected-radius', 'float32', 1);
      const props: GPULinearReferencingProps = {
        id: 'auditL',
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        queryPositions: createTransientView(graph, 'queries', 'float32x2', 1),
        radius: createTransientView(graph, 'radius', 'float32', radiusRows),
        spatialContext: spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
        candidateCapacity: 4,
        output: {
          measures: createTransientView(graph, 'measures', 'float32', 1),
          status: {candidateOverflow: createTransientView(graph, 'overflow', 'uint32', 1)}
        }
      };
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => new GPULinearReferencing(props)).toThrow(/auditL radius.*1.*received/);
      expect(allocate).not.toHaveBeenCalled();
      const nodes = new GPULinearReferencing({...props, radius: correctedRadius}).getCommandNodes(
        graph
      );
      expect(nodes.length).toBeGreaterThan(0);
      const identifiers = allocate.mock.calls.map(([descriptor]) => descriptor.id);
      expect(identifiers).toContain('auditL-vertex-measures');
      expect(new Set(identifiers).size).toBe(identifiers.length);
    });
  }
}
