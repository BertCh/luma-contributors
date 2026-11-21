// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensScenarioRunOptions} from '../../types';
import type {Device} from '@luma.gl/core';
import {runExample} from './variable-standardization.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

/** Executes the GPU recipe and captures GPU-written normalized values for the lens. */
export async function run(device: Device, options?: LensScenarioRunOptions) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts', options?.datasetResolver);
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-variable-standardization'
  );
  const example = await runExample(device, {
    graph,
    values: Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN)
  });
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: 'gpu-dataframe.variable-standardization',
      element: 'GPUVariableStandardization',
      graph,
      specs: {
        values: {
          view: example.values,
          role: 'input',
          keySpace: 'row',
          meaning: 'Census tract poverty percentage; suppressed estimates are NaN.'
        },
        output: {
          view: example.output,
          role: 'output',
          keySpace: 'row',
          meaning: 'Sample z-score across finite tract poverty percentages.'
        }
      },
      keySpaces: [{id: 'row', label: 'Census tract', size: dataset.rowCount, indexSpace: '1d'}],
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('derive', 'method', 'z-score', {
          alternatives: [
            'deviation-from-mean',
            'z-score',
            'mad',
            'range-adjust',
            'range-standardize'
          ]
        }),
        createDataFrameChoice('summarise', 'variance', 'sample', {
          alternatives: ['sample', 'population']
        })
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
