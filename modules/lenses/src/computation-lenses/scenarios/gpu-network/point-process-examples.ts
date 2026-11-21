// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  recordNetworkElement,
  describeNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample as runDensity} from './network-kernel-density.example';
import {runExample as runK} from './network-inhomogeneous-k.example';

type Element = 'GPUNetworkKernelDensity' | 'GPUNetworkInhomogeneousK';
function createScenario(element: Element): NetworkElementScenario {
  const slug =
    element === 'GPUNetworkKernelDensity' ? 'network-kernel-density' : 'network-inhomogeneous-k';
  return {
    id: `gpu-network.${slug}`,
    weights: 'lengthMeters',
    withoutRoads: true,
    roadContext: false,
    seed: 0,
    datasetVersion: 'point-process-illustrative/1',
    retain: [],
    formulas: [],
    keySpaces: [{id: 'entry', label: 'Point-process input/output row', indexSpace: '1d'}],
    entityKeySpace: 'entry',
    choices: [
      {
        stage: 'representation',
        name: 'dataset',
        value:
          'Unit-length three-node chain, two endpoint events; supplied distances and intensities.',
        sweepable: false
      }
    ],
    async build({graph, upload}) {
      const views: Record<string, GraphDataView> = {};
      const floats = (name: string, values: number[]) => {
        const view = upload(name, Float32Array.from(values), 'float32');
        views[name] = view;
        return view;
      };
      const integers = (name: string, values: number[]) => {
        const view = upload(name, Uint32Array.from(values), 'uint32');
        views[name] = view;
        return view;
      };
      if (element === 'GPUNetworkKernelDensity') {
        await runDensity(graph, {
          distances: floats('distances', [0, 2, 1, 1, 2, 0]),
          eventCount: 2,
          quadratureLengths: floats('quadratureLengths', [0.5, 1, 0.5]),
          bandwidth: 2,
          output: {
            intensity: floats('intensity', [0, 0, 0]),
            kernelMass: floats('kernelMass', [0, 0]),
            validity: integers('validity', [0, 0, 0])
          }
        });
      } else {
        const edgeWords = integers('edgeWords', [0, 1, 1, 2]);
        const edgeNodes = graph.createDataView(edgeWords.buffer, {format: 'uint32x2', length: 2});
        await runK(graph, {
          eventNodeIds: integers('eventNodeIds', [0, 2]),
          intensity: floats('intensity', [1, 1]),
          nodeDistances: floats('nodeDistances', [0, 1, 2, 2, 1, 0]),
          nodeCount: 3,
          edgeNodes,
          edgeLengths: floats('edgeLengths', [1, 1]),
          radii: floats('radii', [1, 2]),
          correction: 'ang',
          normalization: 'network-length',
          maximumPairCount: 4,
          output: {
            k: floats('k', [0, 0]),
            numerator: floats('numerator', [0, 0]),
            normalization: floats('normalization', [0, 0, 0]),
            status: integers('status', [0, 0])
          }
        });
      }
      return {views};
    },
    capture(final, _context, captured) {
      const isDensity = element === 'GPUNetworkKernelDensity';
      const diagnostic = isDensity ? 'validity' : 'status';
      if (
        Array.from(final(diagnostic).values).some(value => (isDensity ? value === 0 : value !== 0))
      )
        throw new Error(`Point-process example has invalid ${diagnostic}`);
      const outputs = isDensity
        ? ['intensity', 'kernelMass', 'validity']
        : ['k', 'numerator', 'normalization', 'status'];
      return captured.map(capture => ({
        ...capture,
        keySpace: 'entry',
        role: outputs.includes(capture.name ?? '') ? 'output' : 'input'
      }));
    }
  };
}
export function runNetworkPointProcess(device: Device, element: Element) {
  return recordNetworkElement(device, createScenario(element));
}
export function describeNetworkPointProcess(device: Device, element: Element) {
  return describeNetworkElement(device, createScenario(element));
}
