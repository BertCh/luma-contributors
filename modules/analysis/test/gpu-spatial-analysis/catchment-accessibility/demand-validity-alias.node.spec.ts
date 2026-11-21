// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('documents positive completeness recovery', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/analysis-experience.md', 'utf8');
  expect(guide.includes('Write a nonzero `costsComplete` word once the costs are complete')).toBe(
    true
  );
});

it.each([
  'own field',
  'prototype getter'
])('publishes demand validity from a %s and rejects the retired output', storage => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const floats = (name: string, values: number[]) => {
    const view = createTransientView(graph, name, 'float32', values.length);
    memory.write(view, Float32Array.from(values));
    return view;
  };
  const integers = (name: string, values: number[]) => {
    const view = createTransientView(graph, name, 'uint32', values.length);
    memory.write(view, Uint32Array.from(values));
    return view;
  };
  try {
    const weights = {
      offsets: integers('offsets', [0, 1]),
      neighbors: integers('neighbors', [0]),
      weights: floats('weights', [1])
    };
    const demand = floats('demand', [NaN]);
    const demandValidity = integers('validity', [99]);
    const conflicting = integers('conflicting', [99]);
    const captureOutput = <Output extends object>(output: Output) =>
      storage === 'own field'
        ? {...output, demandValidity}
        : Object.setPrototypeOf(output, {
            get demandValidity() {
              return demandValidity;
            }
          });
    const catchmentProps = {
      method: '2sfca' as const,
      supply: floats('supply', [1]),
      demand,
      facilityWeights: weights,
      output: captureOutput({accessibility: floats('accessibility', [99])})
    };
    const huffProps = {
      attractiveness: floats('attraction', [1]),
      parameters: floats('parameters', [1]),
      demand,
      demandWeights: weights,
      facilityWeights: weights,
      output: captureOutput({
        tradeArea: integers('trade-area', [99]),
        expectedDemand: floats('expected', [99])
      })
    };
    // The sole reachable facility receives probability one regardless of missing demand.
    // Missing demand contributes zero to expected demand and the catchment denominator.
    const huff = new GPUHuffTradeAreas(huffProps);
    executeCommandNodesOnCPU(huff.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(huffProps.output.tradeArea))).toEqual([0]);
    expect(Array.from(memory.readFloat32(huffProps.output.expectedDemand))).toEqual([0]);
    expect(Array.from(memory.readUint32(demandValidity))).toEqual([0]);
    memory.write(demandValidity, Uint32Array.of(99));
    executeCommandNodesOnCPU(
      new GPUCatchmentAccessibility(catchmentProps).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readFloat32(catchmentProps.output.accessibility))).toEqual([0]);
    expect(Array.from(memory.readUint32(demandValidity))).toEqual([0]);
    expect(
      () =>
        new GPUCatchmentAccessibility({
          ...catchmentProps,
          output: {...catchmentProps.output, ...{validity: conflicting}}
        })
    ).toThrow(/output.validity was removed.*output.demandValidity/);
    expect(
      () =>
        new GPUHuffTradeAreas({
          ...huffProps,
          output: {...huffProps.output, ...{validity: demandValidity}}
        })
    ).toThrow(/output.validity was removed.*output.demandValidity/);
  } finally {
    device.destroy();
  }
});
