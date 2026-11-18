// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ComputePipelineProps, Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  countDistinctPipelines,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeShaderModuleKeys,
  recordComputePipelines
} from '../../src/testing/gpu-contributor-test-utils';

const SOURCE =
  '@compute @workgroup_size(1) fn first() {} @compute @workgroup_size(1) fn second() {}';
function createNode(configuration: Omit<ComputePipelineProps, 'shader'>) {
  return {
    compile: ({device}: {device: Device}) => {
      device.createComputePipeline({
        ...configuration,
        shader: {source: SOURCE}
      } as ComputePipelineProps);
    }
  };
}

it('counts two entry points in one module as two pipeline configurations', () => {
  const nodes = [createNode({entryPoint: 'first'}), createNode({entryPoint: 'second'})];
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(2);
  expect(countDistinctPipelines(nodes)).toBe(2);
  expect(getGPUCommandNodeShaderModuleKeys(nodes).size).toBe(1);
  expect(new Set(recordComputePipelines(nodes).map(pipeline => pipeline.source)).size).toBe(1);
});

it('counts layout differences by default in both recorders', () => {
  const nodes = [
    createNode({shaderLayout: {bindings: []}}),
    createNode({
      shaderLayout: {bindings: [{name: 'input', type: 'storage', location: 0, group: 0}]}
    })
  ];
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(2);
  expect(countDistinctPipelines(nodes)).toBe(2);
});

it('matches the factory treatment of default entry points and serialized constants', () => {
  expect(countDistinctPipelines([createNode({}), createNode({entryPoint: 'main'})])).toBe(1);
  const nodes = [
    createNode({constants: {first: 1, second: 2}}),
    createNode({constants: {second: 2, first: 1}})
  ];
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(2);
  expect(countDistinctPipelines(nodes)).toBe(2);
});
