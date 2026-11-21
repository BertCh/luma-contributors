// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphPrimitive
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createCopyOperation} from './contributor-output-oracles-fixture';

it('generic lowering resolves each copy port to its literal graph value', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = {
    leftInput: createTransientView(graph, 'left', 'uint32', 3),
    rightInput: createTransientView(graph, 'right', 'uint32', 3),
    poisonInput: createTransientView(graph, 'poison', 'uint32', 3),
    firstResult: createTransientView(graph, 'first', 'uint32', 3),
    secondResult: createTransientView(graph, 'second', 'uint32', 3)
  };
  let contributor:
    | ReturnType<ReturnType<typeof createCopyOperation>['createContributor']>
    | undefined;
  GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(createCopyOperation(), {
    resolveVector: (value: string) => ({data: [values[value as keyof typeof values]]}),
    lower: (value: GPUGraphPrimitive) => {
      contributor = value;
    }
  } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1]);
  const nodes = contributor!.getCommandNodes!(graph);
  expect(
    nodes
      .slice(2)
      .map(node =>
        node.resources?.map(resource => ('buffer' in resource ? resource.buffer : undefined))
      )
  ).toEqual([
    [values.leftInput, values.firstResult],
    [values.rightInput, values.secondResult]
  ]);
  device.destroy();
});
