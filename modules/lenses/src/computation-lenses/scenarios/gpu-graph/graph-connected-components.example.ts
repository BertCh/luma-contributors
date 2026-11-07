// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUGraphAlgorithmsConnectedComponents,
  type GPUGraphAlgorithmsTopology
} from '@luma-contributors/graph/gpu-graph-algorithms';

/** Declares the connected-components run used by the result figure, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    railTopology: GPUGraphAlgorithmsTopology;
    stations: {vertexCount: number};
  }
) {
  // #region example
  const {commandGraph, railTopology, stations} = input;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createColumn = (name: string, length: number) =>
    new GPUVector({
      type: 'buffer',
      name,
      format: 'uint32',
      length,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: length * 4, usage})
    });
  const componentLabels = createColumn('component-labels', stations.vertexCount);
  const converged = createColumn('converged', 1);
  const components = new GPUGraphAlgorithmsConnectedComponents({
    topology: railTopology,
    output: componentLabels,
    iterations: 32,
    converged
  });
  components.addToGraph(commandGraph);
  return {componentLabels: components.output, converged: components.converged};
  // #endregion example
}
