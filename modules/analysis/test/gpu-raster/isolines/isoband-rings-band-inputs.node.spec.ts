// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
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
