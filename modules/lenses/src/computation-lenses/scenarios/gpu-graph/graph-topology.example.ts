// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUGraphAlgorithmsTopology,
  type GPUGraphAlgorithmsData
} from '@luma.gl/graph/gpu-graph-algorithms';

/** Declares a CSR rebuild of the rail edge list, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    graph: GPUGraphAlgorithmsData;
    edgeCount: number;
    stations: {vertexCount: number};
  }
) {
  // #region example
  const {commandGraph, graph, edgeCount, stations} = input;
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
  const forward = {
    offsets: createColumn('offsets', stations.vertexCount + 1),
    neighbors: createColumn('neighbors', edgeCount),
    edgeIds: createColumn('edge-ids', edgeCount),
    count: createColumn('count', 1),
    overflow: createColumn('overflow', 1)
  };
  const invalidEdgeCount = createColumn('invalid-edge-count', 1);
  const options = {id: 'topology', graph, forward, invalidEdgeCount};
  const topology = new GPUGraphAlgorithmsTopology(options);
  topology.addToGraph(commandGraph);
  return {...topology.forward, invalidEdgeCount};
  // #endregion example
}
