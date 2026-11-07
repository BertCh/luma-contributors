// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function makeProps(rows = 2) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    graph,
    props: {
      positions: view('float32x2', rows),
      values: view('float32', rows),
      sourceIds: view('uint32', rows),
      output: {
        positions: view('float32x2', rows),
        radii: view('float32', rows),
        sourceIds: view('uint32', rows),
        metrics: view('float32x2', 1),
        status: view('uint32', 4)
      }
    }
  };
}

it('bounds rows and iteration budget and rejects mismatched or aliased views', () => {
  const {props} = makeProps();
  expect(() => new GPUDorlingCartogram({...props, maximumIterations: 257})).toThrow();
  expect(() => new GPUDorlingCartogram({...props, areaScale: Infinity})).toThrow();
  expect(() => new GPUDorlingCartogram({...props, overlapTolerance: -1})).toThrow();
  expect(
    () => new GPUDorlingCartogram({...props, output: {...props.output, positions: props.positions}})
  ).toThrow();
  const large = makeProps(65537);
  expect(() => new GPUDorlingCartogram(large.props)).toThrow();
});

it('parses all kernels within the eight storage binding device default', () => {
  for (const maximumIterations of [0, 1, 2]) {
    const {graph, props} = makeProps();
    const kernels = getParsedKernelSources(
      new GPUDorlingCartogram({...props, maximumIterations}).getCommandNodes(graph)
    );
    expect(kernels.size).toBe(3 + maximumIterations * 3 + (maximumIterations % 2));
    for (const [name, source] of kernels) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    }
  }
});

it('splits relax and diagnose kernels into row tiles that share one shader per pass', () => {
  const first = makeProps(10);
  const untiled = new GPUDorlingCartogram({...first.props, maximumIterations: 1}).getCommandNodes(
    first.graph
  );
  const second = makeProps(10);
  const tiled = new GPUDorlingCartogram({
    ...second.props,
    maximumIterations: 1,
    rowTileSize: 4
  }).getCommandNodes(second.graph);
  // Two tiled passes (one relax, one final diagnose) each gain two nodes (10 rows in tiles of 4).
  expect(tiled.length).toBe(untiled.length + 2 * 2);
});

it('accepts empty topology and maximum supported row and iteration counts', () => {
  const empty = makeProps(0);
  expect(() =>
    new GPUDorlingCartogram({...empty.props, maximumIterations: 0}).getCommandNodes(empty.graph)
  ).not.toThrow();
  const maximum = makeProps(65536);
  expect(() => new GPUDorlingCartogram({...maximum.props, maximumIterations: 256})).not.toThrow();
});
