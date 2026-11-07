// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// `snapshots` (how many of the tapped frames go into the sidecar) and `stamps` (whether the
// settle-round stamp is included) are createGPUComputationCaptures props. `epsilon` decides when a
// tract counts as settled in that stamp; `iterations` is the relaxation that was tapped.
export const parameters: GpuSceneParameter[] = [
  {id: 'snapshots', label: 'snapshots', min: 1, max: 6, step: 1, default: 4, unit: ' frames'},
  {
    id: 'stamps',
    label: 'stamps',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['none', 'settledAt']
  },
  {id: 'epsilon', label: 'Data: settle epsilon', min: 0.01, max: 0.1, step: 0.01, default: 0.03},
  {id: 'iterations', label: 'Data: iterations', min: 3, max: 8, step: 1, default: 6}
];

export const legend: RunLegendItem[] = [
  {label: 'snapshots: state (top row) and residual (bottom row) per frame', color: '#5aa9ff'},
  {label: 'stamps: settledAt round per tract (one color per round, grey: never)', color: '#ff9a3c'},
  {label: 'bars: snapshots, stamps and replays in the sidecar', color: '#c792ea'},
  {label: 'corner square: sidecars valid against the record', color: '#8bd450'}
];
