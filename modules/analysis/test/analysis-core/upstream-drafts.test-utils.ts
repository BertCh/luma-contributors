// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUGraphAlgorithmsData,
  GPUGraphAlgorithmsTopology
} from '@luma.gl/graph/gpu-graph-algorithms';

/** Physical, prepopulated CSR for baseline probes; this does not implement the proposed view API. */
export function createTopologyFixture(
  device: Device,
  offsets: readonly number[],
  neighbors: readonly number[]
) {
  const buffers: Buffer[] = [];
  const createVector = (name: string, values: readonly number[]) => {
    const buffer = device.createBuffer({
      id: name,
      data: Uint32Array.from(values.length ? values : [0]),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const vector = new GPUVector({
      type: 'buffer',
      name,
      buffer,
      format: 'uint32',
      length: values.length,
      ownsBuffer: false
    });
    return {buffer, vector};
  };
  const sourceRows = neighbors.map((_, slot) => {
    const nextRow = offsets.findIndex(offset => offset > slot);
    return nextRow - 1;
  });
  const forward = {
    offsets: createVector('offsets', offsets),
    neighbors: createVector('neighbors', neighbors),
    edgeIds: createVector(
      'edgeIds',
      neighbors.map((_, index) => index)
    ),
    count: createVector('count', [neighbors.length]),
    overflow: createVector('overflow', [0])
  };
  const topology = new GPUGraphAlgorithmsTopology({
    graph: new GPUGraphAlgorithmsData({
      vertexCount: offsets.length - 1,
      directed: true,
      sourceVertices: createVector('sourceVertices', sourceRows).vector,
      targetVertices: createVector('targetVertices', neighbors).vector
    }),
    forward: {
      offsets: forward.offsets.vector,
      neighbors: forward.neighbors.vector,
      edgeIds: forward.edgeIds.vector,
      count: forward.count.vector,
      overflow: forward.overflow.vector
    },
    invalidEdgeCount: createVector('invalidEdgeCount', [0]).vector
  });
  return {
    topology,
    forward,
    createVector,
    destroy: () => {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
