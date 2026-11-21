// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import * as typescript from 'typescript';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {getGPUSpaceTimeHawkesParameterValues} from '../../../src/gpu-spatial-analysis/space-time-hawkes/space-time-hawkes-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('UPT-09 executes the typed Hawkes example with a parameter-buffer owner write', () => {
  const document = readFileSync('docs/api/gpu-spatial-analysis/space-time-hawkes.md', 'utf8');
  const snippet = [...document.matchAll(/```ts\n([\s\S]*?)```/g)]
    .map(match => match[1])
    .find(source => source.includes('declare const graph:'))!;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'example-parameters',
    format: 'float32',
    length: 9
  });
  const write = vi.spyOn(parameterBuffer, 'write');
  // Execute the actual fence after erasing types/import declarations. No compiler or GPU work.
  const source = snippet.replace(/import\s[\s\S]*?from\s+['"][^'"]+['"];?/g, '');
  const javascript = typescript.transpileModule(source, {
    compilerOptions: {target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.None}
  }).outputText;
  const inputs: Record<string, unknown> = {
    graph,
    parameterBuffer,
    GPUSpaceTimeHawkes,
    getGPUSpaceTimeHawkesParameterValues
  };
  for (const match of snippet.matchAll(/declare const (\w+): GraphDataView<'(\w+)'>;/g)) {
    const name = match[1];
    // Legacy unowned graph parameters are present only so the old fence can reproduce a discarded write.
    const length =
      name === 'positions' || name === 'times' || name === 'backgroundProbability'
        ? 2
        : name === 'estimates'
          ? 4
          : name === 'parameters'
            ? 9
            : 1;
    inputs[name] = createTransientView(graph, `example-${name}`, match[2] as 'float32', length);
  }
  try {
    new Function(...Object.keys(inputs), javascript)(...Object.values(inputs));
    expect(write).toHaveBeenCalledTimes(1);
    expect(Array.from(write.mock.calls[0][0])).toEqual(
      Array.from(
        getGPUSpaceTimeHawkesParameterValues({
          windowArea: 1,
          windowDuration: 400,
          maximumLag: 5,
          maximumDistance: 0.12,
          initialBackground: 1,
          initialBranchingRatio: 0.3,
          initialDecayRate: 1,
          initialSpatialStandardDeviation: 0.05
        })
      )
    );
  } finally {
    parameterBuffer.destroy();
    device.destroy();
  }
});
