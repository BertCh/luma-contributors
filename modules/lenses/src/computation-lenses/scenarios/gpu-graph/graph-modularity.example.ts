// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUGraphAlgorithmsLabelPropagation,
  GPUGraphAlgorithmsModularity,
  type GPUGraphAlgorithmsData,
  type GPUGraphAlgorithmsTopology
} from '@luma-contributors/graph/gpu-graph-algorithms';

/** Declares a GPU label-propagation partition scored by modularity, without submitting work. */
export async function runExample(
  device: Device,
  input: {
    commandGraph: GPUCommandGraph<void>;
    graph: GPUGraphAlgorithmsData;
    railTopology: GPUGraphAlgorithmsTopology;
    stations: {vertexCount: number};
  }
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createColumn = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) =>
    new GPUVector({
      type: 'buffer',
      name,
      format,
      length,
      ownsBuffer: true,
      buffer: device.createBuffer({byteLength: length * 4, usage})
    });
  // #region example
  const {commandGraph, graph, railTopology, stations} = input;
  const communities = createColumn('communities', 'uint32', stations.vertexCount);
  const contributions = createColumn('contributions', 'float32', stations.vertexCount);
  const score = createColumn('score', 'float32', 1);
  const valid = createColumn('valid', 'uint32', 1);
  new GPUGraphAlgorithmsLabelPropagation({
    topology: railTopology,
    output: communities,
    iterations: 16
  }).addToGraph(commandGraph);
  const modularity = new GPUGraphAlgorithmsModularity({
    graph,
    communities,
    output: score,
    communityContributions: contributions,
    valid
  });
  modularity.addToGraph(commandGraph);
  return {communities, contributions: modularity.communityContributions, score, valid};
  // #endregion example
}
