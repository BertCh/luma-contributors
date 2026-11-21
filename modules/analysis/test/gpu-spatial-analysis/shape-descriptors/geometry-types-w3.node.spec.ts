// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGroupGeometry,
  type GPUGroupGeometryOutput
} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-geometry';
import {
  GPUShapeDescriptors,
  type GPUShapeDescriptorsOutput
} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors';
import {
  GPUHilbertKeys,
  type GPUHilbertKeysProps
} from '../../../src/gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys';
import {
  GPUGeometryValidity,
  type GPUGeometryValidityProps
} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('DEF-09 requires actual terminal output choices and keeps JavaScript guards', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length = 1) =>
    createTransientView(graph, `output-${sequence++}`, format, length);
  try {
    // @ts-expect-error Empty output selections have no terminal leaf.
    const emptyGroup: GPUGroupGeometryOutput = {};
    // @ts-expect-error Empty nested status does not request a terminal geometry column.
    const emptyNestedGroup: GPUGroupGeometryOutput = {status: {}};
    // @ts-expect-error Empty shape selections have no terminal leaf.
    const emptyShape: GPUShapeDescriptorsOutput = {};
    const group = {
      positions: view('float32x2'),
      labels: view('uint32'),
      groupCount: 1,
      parameters: view('float32', 8)
    };
    expect(() => new GPUGroupGeometry({...group, output: emptyGroup})).toThrow(
      /at least one output/
    );
    expect(() => new GPUGroupGeometry({...group, output: emptyNestedGroup})).toThrow(
      /at least one output/
    );
    for (const output of [
      {counts: view('uint32')},
      {bounds: view('float32x4')},
      {meanCenters: view('float32x2')},
      {weightedCenters: view('float32x2')},
      {weightSums: view('float32')},
      {medoidIndices: view('uint32')},
      {ellipses: view('float32', 3)},
      {standardDistances: view('float32')}
    ] satisfies GPUGroupGeometryOutput[])
      expect(
        () => new GPUGroupGeometry({...group, weights: view('float32'), output})
      ).not.toThrow();
    const shape = {
      positions: view('float32x2', 4),
      ringOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      featureOffsets: view('uint32', 2),
      parameters: view('float32', 4)
    };
    expect(() => new GPUShapeDescriptors({...shape, output: emptyShape})).toThrow(
      /at least one output/
    );
    for (const output of [
      {areas: view('float32')},
      {perimeters: view('float32')},
      {polsbyPopper: view('float32')},
      {schwartzberg: view('float32')},
      {elongation: view('float32')},
      {orientation: view('float32')},
      {convexity: view('float32')},
      {clockwise: view('uint32')},
      {sliver: view('uint32')}
    ] satisfies GPUShapeDescriptorsOutput[])
      expect(() => new GPUShapeDescriptors({...shape, output})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('DEF-10 requires one complete geometry source and preserves JavaScript guards', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length = 1) =>
    createTransientView(graph, `source-${sequence++}`, format, length);
  try {
    const positions = view('float32x2', 4),
      minima = view('float32x2', 4),
      maxima = view('float32x2', 4);
    const output = {keys: view('uint32', 4)};
    const accepted: GPUHilbertKeysProps[] = [
      {positions, output},
      {minima, maxima, output}
    ];
    for (const props of accepted) expect(() => new GPUHilbertKeys(props)).not.toThrow();
    // @ts-expect-error Geometry source is required.
    const empty: GPUHilbertKeysProps = {output};
    // @ts-expect-error A box source requires both minima and maxima.
    const partial: GPUHilbertKeysProps = {minima, output};
    // @ts-expect-error Point and box sources are mutually exclusive.
    const both: GPUHilbertKeysProps = {positions, minima, maxima, output};
    for (const props of [empty, partial, both])
      expect(() => new GPUHilbertKeys(props)).toThrow(/positions/);
    const issueOutput = {issueCode: view('uint32', 1)};
    const points = {kind: 'points' as const, positions: view('float32x2')};
    const lines = {kind: 'lines' as const, positions, lineOffsets: view('uint32', 2)};
    const polygons = {
      kind: 'polygons' as const,
      positions,
      ringOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      featureOffsets: view('uint32', 2)
    };
    const valid: GPUGeometryValidityProps[] = [
      {points, output: issueOutput},
      {lines, output: issueOutput},
      {
        polygons,
        intersectionCapacity: 4,
        output: {...issueOutput, status: {overflow: view('uint32')}}
      }
    ];
    for (const props of valid) expect(() => new GPUGeometryValidity(props)).not.toThrow();
    // @ts-expect-error Exactly one geometry family is required.
    const missing: GPUGeometryValidityProps = {output: issueOutput};
    // @ts-expect-error Geometry families cannot be mixed.
    const mixed: GPUGeometryValidityProps = {points, lines, output: issueOutput};
    for (const props of [missing, mixed])
      expect(() => new GPUGeometryValidity(props)).toThrow(/exactly one/);
  } finally {
    device.destroy();
  }
});
