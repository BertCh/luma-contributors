// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUNetworkInhomogeneousK} from '../../../src/gpu-network/gpu-network-inhomogeneous-k';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import fixtures from './network-point-process-fixtures.json';

it('inhomogeneous network K validates the pair budget and stays within eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `network-k-${serial++}`, format, length);
  const props = {
    eventNodeIds: view('uint32', 3),
    intensity: view('float32', 3),
    nodeDistances: view('float32', 12),
    nodeCount: 4,
    edgeNodes: view('uint32x2', 4),
    edgeLengths: view('float32', 4),
    radii: view('float32', 5),
    output: {
      k: view('float32', 5),
      numerator: view('float32', 5),
      normalization: view('float32', 3),
      status: view('uint32', 5)
    }
  };
  expect(() => new GPUNetworkInhomogeneousK({...props, maximumPairCount: 8})).toThrow();
  expect(() => new GPUNetworkInhomogeneousK({...props, nodeCount: 3})).toThrow();
  for (const correction of ['none', 'ang'] as const) {
    const source = getGPUCommandNodeKernelSources(
      new GPUNetworkInhomogeneousK({...props, id: correction, correction}).getCommandNodes(graph)
    );
    expect(source.length).toBe(3);
    for (const kernel of source) {
      expect(new WgslReflect(kernel.source).storage.length).toBeLessThanOrEqual(8);
    }
  }
  device.destroy();
});

it('independent graph fixtures preserve disconnected exclusion and geometric correction', () => {
  expect(fixtures.reference.networkx).toBe('3.5');
  const junction = fixtures.cases[0];
  // The event on the disconnected edge has no finite pair with the other three events.
  expect(junction.shellCounts.slice(12, 15)).toEqual([0, 0, 0]);
  // At distance 2 from the central junction, all three branches contribute a sphere location.
  expect(junction.shellCounts[1]).toBe(3);
  expect(junction.numeratorAng.at(-1)).toBeLessThan(junction.numeratorNone.at(-1)!);
  // Distinct events at the same node contribute at radius zero; the diagonal never contributes.
  const coincident = fixtures.cases[2];
  expect(coincident.numeratorAng[0]).toBe(1);
  expect(coincident.numeratorNone[0]).toBe(1);
});
