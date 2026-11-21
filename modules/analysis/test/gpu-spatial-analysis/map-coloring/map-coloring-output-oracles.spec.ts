// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {
  GPUMapColoringOperation,
  type MapColoringOperationProps
} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture, addOracleOperation} from '../global-coordinates/output-oracle-fixture';

it('one-round hub-last coloring reports partial completion consistently with its literal colors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const props: MapColoringOperationProps = {
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    parameters: 'parameters',
    rowCount: 4,
    maximumIterations: 1,
    output: {
      colors: 'colors',
      colorCount: 'colorCount',
      conflictCount: 'conflictCount',
      status: {converged: 'converged', iterationCount: 'iterationCount'}
    }
  };
  let reads = 0;
  Object.defineProperty(props, 'maximumIterations', {
    configurable: true,
    get: () => (++reads === 1 ? 1 : 2)
  });
  const operation = new GPUMapColoringOperation(props);
  expect(reads).toBe(1);
  Object.defineProperty(props, 'maximumIterations', {value: 2});
  props.output.status!.converged = 'other';
  const outputs = addOracleOperation(fixture, operation, {
    offsets: fixture.input('uint32', [0, 1, 2, 3, 6]),
    neighbors: fixture.input('uint32', [3, 3, 3, 0, 1, 2]),
    weights: fixture.input('float32', [1, 1, 1, 1, 1, 1]),
    parameters: fixture.input('uint32', [19])
  });
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const colors = await outputs['colors'].read();
    // Hub degree 3 always wins zero; a leaf either waits or takes one. GPU scheduling can
    // finish leaves in the same round, so do not assert the CPU sweep's partial membership.
    expect(colors[3]).toBe(0);
    for (const color of colors.slice(0, 3)) expect([1, 0xffffffff]).toContain(color);
    expect(await outputs['converged'].read()).toEqual([Number(!colors.includes(0xffffffff))]);
    expect(await outputs['iterationCount'].read()).toEqual([1]);
    expect(await outputs['colorCount'].read()).toEqual([colors.includes(1) ? 2 : 1]);
    expect(await outputs['conflictCount'].read()).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const operationForm of [false, true])
  it(`${operationForm ? 'operation' : 'contributor'} coloring pins isolate colors/count/rounds without a priority mirror`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Four isolates must each take color zero in the first round, for either priority policy.
    for (const {priority, star} of [
      {priority: 'random', star: false},
      {priority: 'largest-degree-first', star: false},
      {priority: 'largest-degree-first', star: true}
    ] as const)
      for (const optional of (star
        ? ['absent', 'colorCount']
        : ['absent', 'colorCount', 'iterationCount']) as (
        | 'absent'
        | 'colorCount'
        | 'iterationCount'
      )[]) {
        const fixture = createOracleFixture(device);
        const inputs = {
          offsets: fixture.input('uint32', star ? [0, 3, 4, 5, 6] : [0, 0, 0, 0, 0]),
          neighbors: fixture.input('uint32', star ? [1, 2, 3, 0, 0, 0] : [0]),
          weights: fixture.input('float32', star ? [1, 1, 1, 1, 1, 1] : [1]),
          parameters: fixture.input('uint32', [19])
        };
        const operation = new GPUMapColoringOperation({
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          parameters: 'parameters',
          rowCount: 4,
          priority,
          maximumIterations: 4,
          output: {
            colors: 'colors',
            ...(optional === 'colorCount' ? {colorCount: 'colorCount'} : {}),
            ...(optional === 'iterationCount' ? {status: {iterationCount: 'iterationCount'}} : {})
          }
        });
        let outputs: ReturnType<typeof addOracleOperation>;
        if (operationForm) outputs = addOracleOperation(fixture, operation, inputs);
        else {
          const contributorOutputs: Record<string, ReturnType<typeof fixture.output<'uint32'>>> = {
            colors: fixture.output('uint32', 4),
            ...(optional === 'absent' ? {} : {[optional]: fixture.output('uint32', 1)})
          };
          outputs = contributorOutputs;
          const contributorProps = {
            weights: inputs,
            parameters: inputs.parameters,
            priority,
            maximumIterations: 4,
            output: {
              colors: contributorOutputs['colors'].view,
              ...(optional === 'colorCount'
                ? {colorCount: contributorOutputs['colorCount'].view}
                : {}),
              ...(optional === 'iterationCount'
                ? {status: {iterationCount: contributorOutputs['iterationCount'].view}}
                : {})
            }
          };
          const contributor = new GPUMapColoring(contributorProps);
          contributorProps.output.colors = fixture.output('uint32', 1).view;
          if (contributorProps.output.status)
            contributorProps.output.status.iterationCount = fixture.output('uint32', 1).view;
          fixture.graph.add(contributor);
        }
        const compiled = fixture.graph.compile();
        try {
          for (let submission = 0; submission < 2; submission++) {
            for (const output of Object.values(outputs)) output.poison();
            submitGraph(device, compiled, undefined);
            // The degree-3 hub is first, then the degree-1 leaves all choose one. The star's
            // round count depends on scheduling, so only isolates pin that count to one.
            expect(await outputs['colors'].read()).toEqual(star ? [0, 1, 1, 1] : [0, 0, 0, 0]);
            if (optional !== 'absent')
              expect(await outputs[optional].read()).toEqual([star ? 2 : 1]);
          }
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      }
  });
