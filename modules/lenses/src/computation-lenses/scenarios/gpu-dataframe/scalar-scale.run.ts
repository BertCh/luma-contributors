// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_SCALAR_SCALE_CODES} from '@luma-contributors/graph/gpu-dataframe';
import {runExample} from './scalar-scale.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.scalar-scale';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['radius', 'radiusValidity'];

/** Executes the recorded scalar scale once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(device, 'gpu-dataframe-scalar-scale', RETAINED);
  const example = await runExample(device, {graph, tracts: {poverty150Pct: poverty}});
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUScalarScale',
      graph,
      specs: {
        values: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning:
            'Share of residents below 150% of the poverty line; suppressed estimates are NaN.'
        },
        outputValues: {
          view: example.radius,
          role: 'output',
          keySpace: 'row',
          meaning: 'Point radius in pixels from a square-root scale; suppressed tracts get 0.'
        },
        outputValidity: {
          view: example.radiusValidity,
          role: 'output',
          keySpace: 'row',
          meaning: 'One when the radius was mapped, zero for a suppressed tract.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: poverty.length, indexSpace: '1d'},
        {id: 'invocation', label: 'Compute invocation', size: poverty.length, indexSpace: '1d'}
      ],
      relations: [
        {
          id: 'invocation-row',
          from: 'invocation',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ],
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('classify', 'scale', 'sqrt', {
          alternatives: Object.keys(GPU_SCALAR_SCALE_CODES)
        }),
        createDataFrameChoice('classify', 'domainCount', 2),
        createDataFrameChoice('derive', 'nullValue', 0),
        createDataFrameChoice('derive', 'invert', false, {alternatives: [false, true]})
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
