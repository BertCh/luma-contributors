// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {runExample} from './variogram-model-fit.example';
import {
  createDataFrameLensGraph,
  runDataFrameContributorLens,
  type DataFrameContributorLensProps
} from './dataframe-lens-support';

/** Build an executable resident fit with three competing models and captured real diagnostics. */
export async function buildVariogramModelFitLens(
  device: Device,
  graph: GPUCommandGraph<undefined> = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe.variogram-model-fit'
  )
) {
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x4' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {
      format,
      length: values.length / (format === 'float32x4' ? 4 : 1)
    });
  };
  try {
    const lags = Float32Array.from(Array.from({length: 20}, (_, bin) => 0.2 * (bin + 1)));
    const distances = upload('distances', lags, 'float32');
    const semivariances = upload(
      'semivariances',
      lags.map(distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))),
      'float32'
    );
    const pairCounts = upload(
      'pair-counts',
      Uint32Array.from(Array.from({length: 20}, (_, bin) => 10 + bin)),
      'uint32'
    );
    const parameters = upload('parameters', new Float32Array(12), 'float32x4');
    const diagnostics = upload('diagnostics', new Float32Array(12), 'float32x4');
    const status = upload('statuses', new Uint32Array(3), 'uint32');
    await runExample(graph, {
      distances,
      semivariances,
      pairCounts,
      weighting: 'pairs',
      output: {parameters, diagnostics, status}
    });
    const recording: DataFrameContributorLensProps = {
      scenarioId: 'gpu-dataframe.variogram-model-fit',
      element: 'GPUVariogramModelFit',
      graph,
      specs: {
        distances: {view: distances, role: 'input', keySpace: 'bin'},
        semivariances: {view: semivariances, role: 'input', keySpace: 'bin'},
        pairCounts: {view: pairCounts, role: 'input', keySpace: 'bin'},
        parameters: {view: parameters, role: 'output', keySpace: 'model'},
        diagnostics: {view: diagnostics, role: 'output', keySpace: 'model'},
        statuses: {view: status, role: 'output', keySpace: 'model'}
      },
      keySpaces: [
        {id: 'model', label: 'Spherical, exponential, Gaussian model', size: 3, indexSpace: '1d'},
        {id: 'bin', label: 'Positive empirical lag', size: 20, indexSpace: '1d'}
      ],
      datumKeySpace: 'model',
      choices: [
        {stage: 'aggregate', name: 'weighting', value: 'pairs', sweepable: true},
        {stage: 'aggregate', name: 'rangeSteps', value: 200, sweepable: true},
        {stage: 'aggregate', name: 'maximumIterations', value: 32, sweepable: true}
      ],
      datasetVersion: 'synthetic-exponential-variogram-bins-v1'
    };
    return {recording, buffers};
  } catch (error) {
    for (const buffer of buffers) {
      buffer.destroy();
    }
    throw error;
  }
}

/** Fits and captures GPU outputs; observations and models are never replaced by CPU fit results. */
export async function run(device: Device) {
  const {recording, buffers} = await buildVariogramModelFitLens(device);
  try {
    return await runDataFrameContributorLens(device, recording);
  } finally {
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}
