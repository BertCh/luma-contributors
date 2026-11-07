// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {KEPLER_RATE_CAUSES, addKeplerRateCauseCounts} from '../src/analysis/rate-cause-counts';

function createProps(rowCount: number, withSelection: boolean) {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const graph = new GPUCommandGraph(device, {id: 'weights7-rate-cause'});
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  return {
    graph,
    props: {
      events: view('events', 'float32', rowCount),
      populations: view('populations', 'float32', rowCount),
      selection: withSelection ? view('selection', 'uint32', rowCount) : undefined,
      counts: view('counts', 'uint32', KEPLER_RATE_CAUSES.length)
    }
  };
}

test('rate cause counts add a selection binding that skips unselected rows', () => {
  const withSelection = createProps(10, true);
  expect(() => addKeplerRateCauseCounts(withSelection.graph, withSelection.props)).not.toThrow();
  const withoutSelection = createProps(10, false);
  expect(() =>
    addKeplerRateCauseCounts(withoutSelection.graph, withoutSelection.props)
  ).not.toThrow();
});

test('rate cause counts reject a selection of the wrong length', () => {
  const {graph, props} = createProps(10, false);
  expect(() =>
    addKeplerRateCauseCounts(graph, {
      ...props,
      selection: createTransientView(graph, 'short-selection', 'uint32', 9)
    })
  ).toThrow();
});
