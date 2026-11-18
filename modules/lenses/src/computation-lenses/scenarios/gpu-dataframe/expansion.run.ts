// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {runExample as runCross} from './cross-variogram.example';
import {runExample as runSpaceTime} from './space-time-variogram.example';
import {runExample as runMarked} from './marked-point-statistics.example';
import {
  createDataFrameLensGraph,
  runDataFrameContributorLens,
  type DataFrameContributorLensProps
} from './dataframe-lens-support';
export type ExpansionKind = 'cross-variogram' | 'space-time-variogram' | 'marked-point-statistics';

/** Execute deterministic projected observations; values shown by the lens are GPU readbacks. */
export async function buildExpansion(
  device: Device,
  kind: ExpansionKind,
  graph = createDataFrameLensGraph<undefined>(device, `gpu-dataframe.${kind}`)
) {
  const scenarioId = `gpu-dataframe.${kind}`;
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x2'>(
    id: string,
    data: Float32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {
      format,
      length: data.length / (format === 'float32x2' ? 2 : 1)
    });
  };
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  try {
    const positions = upload('positions', new Float32Array([1, 1, 2, 1, 3, 2, 4, 3]), 'float32x2');
    const values = upload('values', new Float32Array([1, 4, 2, 5]), 'float32');
    const temporal = kind === 'space-time-variogram';
    const marked = kind === 'marked-point-statistics';
    // Space-time: (2 + 1) spatial x (2 + 1) temporal classes, each axis with a zero-lag class.
    const binCount = temporal ? 9 : 2;
    const pairCounts = view('pairCounts', 'uint32', binCount);
    const status = view('status', 'uint32', binCount);
    const result = view(
      marked ? 'statistics' : 'semivariances',
      'float32',
      marked ? binCount * 4 : binCount
    );
    if (marked) {
      await runMarked(device, graph, {
        positions,
        marks: values,
        parameters: upload('parameters', new Float32Array([0, 0, 5, 5, 4, 0.0001]), 'float32'),
        radiusCount: binCount,
        edgeCorrection: 'translation',
        output: {statistics: result, pairCounts, status}
      });
    } else {
      const props = {
        positions,
        values,
        secondValues: upload('secondValues', new Float32Array([5, 1, 4, 2]), 'float32'),
        // Cross: [maximumDistance, unused]. Space-time: [minX, minY, maxX, maxY, maximumDistance,
        // maximumTimeLag, 0, 0].
        parameters: upload(
          'parameters',
          new Float32Array(temporal ? [0, 0, 5, 5, 5, 3, 0, 0] : [5, 3]),
          'float32'
        ),
        lagCount: 2,
        output: {semivariances: result, pairCounts, status}
      };
      if (temporal)
        await runSpaceTime(device, graph, {
          ...props,
          gridSize: [2, 2],
          // Relative times: small integers, so no origin subtraction is needed.
          times: upload('times', new Float32Array([0, 1, 2, 3]), 'float32'),
          temporalLagCount: 2
        });
      else await runCross(device, graph, props);
    }
    const recording: DataFrameContributorLensProps = {
      scenarioId,
      element: marked
        ? 'GPUMarkedPointStatistics'
        : temporal
          ? 'GPUSpaceTimeVariogram'
          : 'GPUCrossVariogram',
      graph,
      specs: {
        positions: {
          view: positions,
          role: 'input',
          keySpace: 'row',
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
        },
        values: {view: values, role: 'input', keySpace: 'row'},
        result: {view: result, role: 'output', keySpace: marked ? 'statistic' : 'lag'},
        pairCounts: {view: pairCounts, role: 'output', keySpace: 'lag'},
        status: {view: status, role: 'output', keySpace: 'lag'}
      },
      keySpaces: [
        {id: 'row', label: 'Synthetic observation', size: 4, indexSpace: '1d'},
        {id: 'lag', label: 'Spatial-major lag bin', size: binCount, indexSpace: '1d'},
        {
          id: 'statistic',
          label: 'Flattened radius × K/L/pair/mark',
          size: binCount * 4,
          indexSpace: '1d'
        }
      ],
      datumKeySpace: 'row',
      choices: [
        {stage: 'filter', name: 'coordinateModel', value: 'projected', sweepable: false},
        {stage: 'group', name: 'lagCount', value: 2, sweepable: false},
        {stage: 'group', name: 'maximumDistance', value: marked ? 4 : 5, sweepable: false},
        ...(temporal
          ? [
              {stage: 'group', name: 'temporalLagCount', value: 2, sweepable: false},
              {stage: 'group', name: 'maximumTimeLag', value: 3, sweepable: false}
            ]
          : [
              {
                stage: 'operator',
                name: 'edgeCorrection',
                value: marked ? 'translation' : 'none',
                sweepable: false
              }
            ])
      ],
      datasetVersion: 'synthetic-four-point-pair-statistics-v1'
    };
    return {recording, buffers};
  } catch (error) {
    buffers.forEach(buffer => buffer.destroy());
    throw error;
  }
}

/** Compile, execute and capture the same graph constructed by buildExpansion. */
export async function runExpansion(device: Device, kind: ExpansionKind) {
  const {recording, buffers} = await buildExpansion(device, kind);
  try {
    return await runDataFrameContributorLens(device, recording);
  } finally {
    buffers.forEach(buffer => buffer.destroy());
  }
}
