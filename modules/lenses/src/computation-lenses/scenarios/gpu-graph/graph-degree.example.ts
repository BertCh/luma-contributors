// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUGraphAlgorithmsDegree,
  type GPUGraphAlgorithmsTopology
} from '@luma.gl/graph/gpu-graph-algorithms';

/** Declares outgoing station degrees from the rail CSR offsets, without submitting work. */
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
  const outDegrees = new GPUVector({
    type: 'buffer',
    name: 'out-degrees',
    format: 'uint32',
    length: stations.vertexCount,
    ownsBuffer: true,
    buffer: device.createBuffer({
      byteLength: stations.vertexCount * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    })
  });
  const degree = new GPUGraphAlgorithmsDegree({
    topology: railTopology,
    output: outDegrees,
    direction: 'outgoing'
  });
  degree.addToGraph(commandGraph);
  return {outDegrees: degree.output};
  // #endregion example
}
