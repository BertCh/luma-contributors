// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'rotation',
    label: 'destinationGrid rotation',
    min: -45,
    max: 45,
    step: 1,
    default: 15,
    unit: '°'
  },
  {
    id: 'missingTile',
    label: 'sources (tile left out)',
    min: 0,
    max: 6,
    step: 1,
    default: 4,
    options: ['none', '1', '2', '3', '4', '5', '6']
  },
  {
    id: 'tileOffset',
    label: 'Data: tile calibration offset',
    min: 0,
    max: 0.3,
    step: 0.01,
    default: 0.12
  }
];
