// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createViews() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = (length: number) =>
    createTransientView(graph, `capacity-view-${next++}`, 'uint32', length);
  return {graph, view};
}

it('GPUNetworkStructure bounds reachability strong components and resilience outputs', () => {
  const {view} = createViews();
  const nodeCount = 5000;
  const topology = {offsets: view(nodeCount + 1), neighbors: view(8000)};
  expect(
    () => new GPUNetworkStructure({topology, output: {strongComponents: view(nodeCount)}})
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

it('GPUNetworkStructure accepts strong components within the reachability bound', () => {
  const {graph, view} = createViews();
  const topology = {offsets: view(5), neighbors: view(6)};
  const structure = new GPUNetworkStructure({topology, output: {strongComponents: view(4)}});
  expect(structure.getCommandNodes(graph).length).toBeGreaterThan(0);
});

it('GPUNetworkStructure rejects strong components above 1024 vertices or 65536 slots', () => {
  const {view} = createViews();
  expect(
    () =>
      new GPUNetworkStructure({
        topology: {offsets: view(1026), neighbors: view(10)},
        output: {strongComponents: view(1025)}
      })
  ).toThrow();
  expect(
    () =>
      new GPUNetworkStructure({
        topology: {offsets: view(5), neighbors: view(65537)},
        output: {strongComponents: view(4)}
      })
  ).toThrow();
});
