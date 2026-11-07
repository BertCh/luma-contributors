// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createViews() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = (length: number) =>
    createTransientView(graph, `capacity-view-${next++}`, 'uint32', length);
  return {graph, view};
}

it('GPUNetworkStructure iterative strong components have no node cap but bounded paths keep theirs', () => {
  const {graph, view} = createViews();
  const nodeCount = 5000;
  const topology = {offsets: view(nodeCount + 1), neighbors: view(8000)};
  const structure = new GPUNetworkStructure({
    topology,
    output: {strongComponents: view(nodeCount), strongComponentsConverged: view(1)}
  });
  expect(structure.getCommandNodes(graph).length).toBeGreaterThan(0);
  expect(
    () =>
      new GPUNetworkStructure({
        topology,
        output: {strongComponents: view(nodeCount)},
        strongComponentsMethod: 'reachability'
      })
  ).toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        topology,
        reverseEdgeSlots: view(8000),
        output: {bridges: view(8000)}
      })
  ).toThrow();
});

it('GPUNetworkStructure validates the strong-component status and budget', () => {
  const {view} = createViews();
  const topology = {offsets: view(5), neighbors: view(6)};
  expect(
    () =>
      new GPUNetworkStructure({
        topology,
        output: {strongComponents: view(4), strongComponentsConverged: view(2)}
      })
  ).toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        topology,
        output: {strongComponents: view(4)},
        strongComponentsIterations: 5000
      })
  ).toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        topology,
        output: {strongComponents: view(4), strongComponentsConverged: view(1)},
        strongComponentsMethod: 'reachability'
      })
  ).toThrow();
});
