// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// `coalesced` and `taps` are GPUComputationEncodingFlags that createGPUComputationMetrics records,
// and the graph is encoded that way for real: coalescing merges consecutive compute nodes into one
// pass, taps are copy nodes that end a pass. `iterations` only sets how long the graph is.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'coalesced',
    label: 'encoding.coalesced',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {id: 'taps', label: 'encoding.taps', min: 0, max: 4, step: 1, default: 2, unit: ' taps'},
  {id: 'iterations', label: 'Data: iterations', min: 1, max: 5, step: 1, default: 4}
];

export const legend: RunLegendItem[] = [
  {
    label: 'rows: nodes in scheduled order; bar = cpuEncodeMs (mean of 400 encodings)',
    color: '#5aa9ff'
  },
  {label: 'color: passIndex of the node (same color = same coalesced pass)', color: '#ffb454'},
  {label: 'grey: copy node (snapshot tap), no pass', color: '#5b6580'},
  {label: 'right: total time per physical pass', color: '#4fd1c5'}
];
