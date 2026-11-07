// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUComputationRecorder's `sourceText` prop and its `withScope` call; `iterations` only
// sets how long the recorded relaxation is.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'sourceText',
    label: 'sourceText',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {
    id: 'scopes',
    label: 'withScope',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['none', 'one scope', 'scope per iteration']
  },
  {id: 'iterations', label: 'Data: iterations', min: 1, max: 4, step: 1, default: 3}
];

export const legend: RunLegendItem[] = [
  {label: 'top: compiled graph, published facts only', color: '#5b6580'},
  {label: 'bottom: nodes by recorded scope namespace', color: '#5aa9ff'},
  {label: 'recorded scope per iteration', color: '#ffb454'},
  {label: 'strip: retained source text, one bar per source', color: '#4fd1c5'}
];
