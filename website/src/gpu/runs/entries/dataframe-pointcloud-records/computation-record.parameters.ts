// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// `sources` and `formulas` are createGPUComputationRecord props (how many extra source and formula
// refs are passed); `iterations` only sets how long the recorded graph is.
export const parameters: GpuSceneParameter[] = [
  {id: 'sources', label: 'sources', min: 0, max: 6, step: 1, default: 3, unit: ' extra refs'},
  {id: 'formulas', label: 'formulas', min: 0, max: 4, step: 1, default: 2, unit: ' refs'},
  {id: 'iterations', label: 'Data: iterations', min: 1, max: 4, step: 1, default: 3}
];

export const legend: RunLegendItem[] = [
  {label: 'graphs[0] nodes, colored by source id (grey: no source)', color: '#5aa9ff'},
  {label: 'resources: imported, lifetime first to last node', color: '#4fd1c5'},
  {label: 'resources: transient', color: '#ffb454'},
  {label: 'record valid / invalid (corner square)', color: '#8bd450'},
  {
    label: 'bars (log scale): nodes, edges, resources, sources passed, refs.sources, formulas',
    color: '#c792ea'
  }
];
