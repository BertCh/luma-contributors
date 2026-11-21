// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
import {Texture} from '@luma.gl/core';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {expect, it, vi} from 'vitest';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('accepts a published band at the isoband-ring public surface', () => {
  const {graph, grid, view, band} = createRasterContributorFixture(3, 3);
  const props = {
    grid,
    values: band,
    breaks: view('float32', 1),
    parameters: view('float32', 12),
    edgeCapacity: 32,
    output: {
      ringOffsets: view('uint32', 9),
      positions: createTransientView(graph, 'positions', 'float32x2', 32),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    }
  };
  expect(() => Reflect.construct(GPUIsobandRings, [props])).toThrow(
    /status.edgeOverflow is required/
  );
  const contributor = new GPUIsobandRings({
    ...props,
    output: {...props.output, status: {...props.output.status, edgeOverflow: view('uint32', 1)}}
  });
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

function createRingProps() {
  const fixture = createRasterContributorFixture(3, 3);
  const {graph, grid, view, band} = fixture;
  return {
    fixture,
    props: {
      grid,
      values: band,
      breaks: view('float32', 1),
      parameters: view('float32', 12),
      edgeCapacity: 32,
      output: {
        ringOffsets: view('uint32', 9),
        positions: createTransientView(graph, 'positions', 'float32x2', 32),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          edgeOverflow: view('uint32', 1)
        }
      }
    }
  };
}

it.each([
  'overflow',
  'count',
  'ringOffsets'
] as const)('rejects edge evidence overlapping final %s before allocating transients', field => {
  const {fixture, props} = createRingProps();
  const evidence = props.output.status.edgeOverflow;
  props.output.status.edgeOverflow =
    field === 'ringOffsets'
      ? fixture.graph.createDataView(props.output.ringOffsets.buffer, {format: 'uint32', length: 1})
      : props.output.status[field];
  const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
  expect(() => new GPUIsobandRings(props).getCommandNodes(fixture.graph)).toThrow(
    /edgeOverflow.*overlap|overlap.*edgeOverflow/
  );
  expect(allocate).not.toHaveBeenCalled();
  props.output.status.edgeOverflow = evidence;
  expect(() => new GPUIsobandRings(props).getCommandNodes(fixture.graph)).not.toThrow();
});

it('accepts a packed offset edgeOverflow view with a spare word', () => {
  const {fixture, props} = createRingProps();
  const storage = fixture.view('uint32', 66);
  props.output.status.edgeOverflow = fixture.graph.createDataView(storage.buffer, {
    format: 'uint32',
    byteOffset: 256,
    length: 2
  });
  expect(() => new GPUIsobandRings(props).getCommandNodes(fixture.graph)).not.toThrow();
});

it('retains ring status bindings after the caller replaces them', () => {
  const {fixture, props} = createRingProps();
  const original = props.output.status.edgeOverflow;
  const contributor = new GPUIsobandRings(props);
  props.output.status.edgeOverflow = props.output.status.overflow;
  expect(contributor.props.output.status.edgeOverflow).toBe(original);
  expect(() => contributor.getCommandNodes(fixture.graph)).not.toThrow();
});

it('validates named ring ports before snapshots and shader declaration', () => {
  const {fixture, props} = createRingProps();
  const invalid = createTransientView(fixture.graph, 'invalid-ring-port', 'unorm8x4', 32);
  for (const name of ['breaks', 'parameters', 'mask'] as const) {
    expect(() =>
      Reflect.construct(GPUIsobandRings, [{...props, id: 'ring-layer', [name]: invalid}])
    ).toThrow(new RegExp(`ring-layer.*${name}.*format`));
  }
  for (const name of [
    'ringOffsets',
    'positions',
    'ringAreas',
    'ringIsHole',
    'ringGroups',
    'segmentRings',
    'segmentVertices',
    'segmentFlags'
  ] as const) {
    expect(() =>
      Reflect.construct(GPUIsobandRings, [
        {...props, id: 'ring-layer', output: {...props.output, [name]: invalid}}
      ])
    ).toThrow(new RegExp(`ring-layer.*output.${name}.*format`));
  }
  expect(() => Reflect.construct(GPUIsobandRings, [{...props, values: undefined}])).toThrow(
    /isoband-rings.*values.*received undefined/
  );
});

it('preserves published scalar band unions while rejecting normalized storage substitutions', () => {
  const {fixture, props} = createRingProps();
  for (const format of ['float32', 'uint32', 'sint32'] as const) {
    const values = {
      ...props.values,
      format,
      storage: {
        kind: 'buffer',
        values: createTransientView(fixture.graph, `band-${format}`, format, 9)
      }
    };
    expect(() => Reflect.construct(GPUIsobandRings, [{...props, values}])).not.toThrow();
  }
  const values = {
    ...props.values,
    storage: {
      kind: 'buffer',
      values: createTransientView(fixture.graph, 'invalid-band-storage', 'unorm8x4', 9)
    }
  };
  expect(() => Reflect.construct(GPUIsobandRings, [{...props, values}])).toThrow(
    /values.storage.values.*format/
  );
});

it('B1 passes boundary overflow alongside the retained count to ring assembly', () => {
  const {fixture, props} = createRingProps();
  const status = {
    ...props.output.status,
    complete: fixture.view('uint32', 1),
    candidateOverflow: fixture.view('uint32', 1)
  };
  const assembly = vi
    .spyOn(GPUSegmentRingAssembly.prototype, 'getCommandNodes')
    .mockReturnValue([]);
  try {
    new GPUIsobandRings({...props, output: {...props.output, status}}).getCommandNodes(
      fixture.graph
    );
    const instance = assembly.mock.contexts[0] as GPUSegmentRingAssembly;
    expect(instance.props.segmentStatus).toEqual({
      count: instance.props.count,
      overflow: status.edgeOverflow
    });
    assembly.mockRestore();
    const kernels = getGPUCommandNodeKernelSources(
      new GPUIsobandRings({
        ...props,
        id: 'complete-probe',
        output: {...props.output, status}
      }).getCommandNodes(fixture.graph)
    );
    const completeness = kernels.find(
      kernel => kernel.id === 'complete-probe-assembly-input-completeness'
    );
    expect(completeness).toBeDefined();
    expect(completeness!.source).toContain('upstreamOverflow[upstreamOverflowOffset] != 0u');
  } finally {
    assembly.mockRestore();
  }
});

it.each([
  'format',
  'channel',
  'validity'
] as const)('B2 rejects invalid texture %s before allocation and permits a same-ID retry', defect => {
  const {fixture, props} = createRingProps();
  const texture = fixture.graph.createTransientTexture({
    id: 'band-texture',
    format: defect === 'format' ? 'rgba8unorm' : 'rgba32sint',
    width: 3,
    height: 3,
    usage: Texture.SAMPLE
  });
  const values = {
    id: 'signed-texture',
    format: 'sint32',
    storage: {
      kind: 'texture',
      view: fixture.graph.createTextureView(texture),
      channel: defect === 'channel' ? 4 : 0
    },
    validity: defect === 'validity' ? fixture.view('uint32', 1) : fixture.view('uint32', 9)
  };
  const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
  try {
    expect(() => Reflect.construct(GPUIsobandRings, [{...props, id: 'quick', values}])).toThrow(
      new RegExp(`quick.*${defect === 'validity' ? 'validity' : defect}`)
    );
    expect(allocate).not.toHaveBeenCalled();
    const corrected = {
      id: 'signed-texture',
      format: 'sint32' as const,
      storage: {
        kind: 'texture' as const,
        channel: 0 as const,
        view: fixture.graph.createTextureView(
          fixture.graph.createTransientTexture({
            id: 'corrected-texture',
            format: 'rgba32sint',
            width: 3,
            height: 3,
            usage: Texture.SAMPLE
          })
        )
      },
      validity: fixture.view('uint32', 9)
    };
    expect(() =>
      new GPUIsobandRings({...props, id: 'quick', values: corrected}).getCommandNodes(fixture.graph)
    ).not.toThrow();
  } finally {
    allocate.mockRestore();
  }
});
