// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPUCoregionalizationFitParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {runExample} from './coregionalization-fit.example';
import {
  createDataFrameLensGraph,
  runDataFrameContributorLens,
  type DataFrameContributorLensProps
} from './dataframe-lens-support';

/** A hand-derived signed cross-variogram case, including one excluded lag matrix. */
export async function buildCoregionalizationFitLens(
  device: Device,
  graph: GPUCommandGraph<undefined> = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe.coregionalization-fit'
  )
) {
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x4' | 'uint32'>(
    identifier: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id: identifier,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: identifier, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {
      format,
      length: values.length / (format === 'float32x4' ? 4 : 1)
    });
  };
  try {
    const distances = upload('distances', new Float32Array([0.5, 1, 2, 3]), 'float32');
    // Spherical shapes at h/range = 1/4, 1/2, 1 are 47/128, 11/16, 1.
    // Each upper triangle is shape * [2, -1, 3]; the zero-pair bin is deliberately unrelated.
    const semivariances = upload(
      'semivariances',
      new Float32Array([
        47 / 64,
        -47 / 128,
        141 / 128,
        11 / 8,
        -11 / 16,
        33 / 16,
        2,
        -1,
        3,
        100,
        100,
        100
      ]),
      'float32'
    );
    const pairCounts = upload('pair-counts', new Uint32Array([10, 20, 30, 0]), 'uint32');
    const parameters = upload(
      'parameters',
      getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 2, sill: 1}]),
      'float32'
    );
    const matrices = upload('matrices', new Float32Array(4), 'float32');
    const loadings = upload('loadings', new Float32Array(4), 'float32');
    const diagnostics = upload('diagnostics', new Float32Array(4), 'float32x4');
    const status = {
      validity: upload('validity', new Uint32Array(1), 'uint32'),
      iterationCount: upload('iteration-count', new Uint32Array(1), 'uint32'),
      converged: upload('converged', new Uint32Array(1), 'uint32'),
      iterationLimitReached: upload('iteration-limit-reached', new Uint32Array(1), 'uint32'),
      invalidCount: upload('invalid-count', new Uint32Array(1), 'uint32')
    };
    await runExample(graph, {
      distances,
      semivariances,
      pairCounts,
      parameters,
      variableCount: 2,
      structureCount: 1,
      maximumIterations: 8,
      output: {matrices, loadings, diagnostics, status}
    });
    const recording: DataFrameContributorLensProps = {
      scenarioId: 'gpu-dataframe.coregionalization-fit',
      element: 'GPUCoregionalizationFit',
      graph,
      specs: {
        distances: {view: distances, role: 'input', keySpace: 'bin'},
        semivariances: {view: semivariances, role: 'input', keySpace: 'triangle-entry'},
        pairCounts: {view: pairCounts, role: 'input', keySpace: 'bin'},
        parameters: {view: parameters, role: 'input', keySpace: 'parameter'},
        matrices: {view: matrices, role: 'output', keySpace: 'coefficient'},
        loadings: {view: loadings, role: 'output', keySpace: 'coefficient'},
        diagnostics: {
          view: diagnostics,
          role: 'output',
          keySpace: 'fit',
          meaning:
            '[weighted objective, relative coefficient change, usable bins, total pair weight]'
        },
        ...Object.fromEntries(
          Object.entries(status).map(([name, view]) => [
            name,
            {view, role: 'output' as const, keySpace: 'fit'}
          ])
        )
      },
      keySpaces: [
        {id: 'bin', label: 'Common lag bin (last has zero pairs)', size: 4, indexSpace: '1d'},
        {
          id: 'triangle-entry',
          label: 'Bin-major upper triangle (00, 01, 11)',
          size: 12,
          indexSpace: '1d'
        },
        {
          id: 'parameter',
          label: 'Tolerance, model code, nugget, sill, range',
          size: 5,
          indexSpace: '1d'
        },
        {
          id: 'coefficient',
          label: 'Structure-major row-major square entry',
          size: 4,
          indexSpace: '1d'
        },
        {id: 'fit', label: 'Coregionalization fit', size: 1, indexSpace: '1d'}
      ],
      datumKeySpace: 'coefficient',
      choices: [
        {stage: 'aggregate', name: 'variableCount', value: 2, sweepable: false},
        {stage: 'aggregate', name: 'structureCount', value: 1, sweepable: false},
        {stage: 'aggregate', name: 'maximumIterations', value: 8, sweepable: true},
        {stage: 'aggregate', name: 'tolerance', value: 1e-5, sweepable: true}
      ],
      datasetVersion: 'synthetic-spherical-coregionalization-bins-v1'
    };
    return {recording, buffers};
  } catch (error) {
    for (const buffer of buffers) {
      buffer.destroy();
    }
    throw error;
  }
}

/** Captures the actual GPU fit and named statuses; no CPU result substitutes for readback. */
export async function run(device: Device) {
  const {recording, buffers} = await buildCoregionalizationFitLens(device);
  try {
    return await runDataFrameContributorLens(device, recording);
  } finally {
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}
