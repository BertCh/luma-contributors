// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, it, onTestFinished} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUContributorOperation} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {constructNetworkOperationContributor} from './network-operation-test-utils';

it('keeps the operation fixture device alive through deferred command construction', () => {
  let device: Device | undefined;
  let contributorGraph: GPUCommandGraph | undefined;
  // Finished hooks run in reverse registration order, after the helper's cleanup below.
  onTestFinished(() => expect(device!.isLost).toBe(true));
  const operation = new GPUContributorOperation({
    id: 'lifecycle-fixture',
    inputs: {values: 'values'},
    outputs: {},
    createContributor(getView) {
      const graph = getView('values').buffer.graph;
      if (!(graph instanceof GPUCommandGraph)) throw new Error('Expected fixture graph');
      contributorGraph = graph;
      device = graph.device;
      return {
        getCommandNodes() {
          expect(graph.device.isLost).toBe(false);
          return [];
        }
      };
    }
  });
  const contributor = constructNetworkOperationContributor(operation, {
    values: {format: 'float32', length: 1}
  });
  expect(device).toBeDefined();
  expect(device!.isLost).toBe(false);
  expect(contributor.getCommandNodes(contributorGraph!)).toEqual([]);
});

it('releases the fixture device immediately when contributor construction fails', () => {
  let device: Device | undefined;
  const operation = new GPUContributorOperation({
    id: 'failing-lifecycle-fixture',
    inputs: {values: 'values'},
    outputs: {},
    createContributor(getView) {
      const graph = getView('values').buffer.graph;
      if (!(graph instanceof GPUCommandGraph)) throw new Error('Expected fixture graph');
      device = graph.device;
      throw new Error('expected construction failure');
    }
  });
  expect(() =>
    constructNetworkOperationContributor(operation, {
      values: {format: 'float32', length: 1}
    })
  ).toThrow('expected construction failure');
  expect(device!.isLost).toBe(true);
});
