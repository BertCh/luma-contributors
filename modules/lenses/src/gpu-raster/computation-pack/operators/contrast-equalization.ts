// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

/** Contrast mapping and compact embedding storage. */
export const RASTER_CONTRAST_EQUALIZATION_OPERATORS: RasterOperatorTable = {
  GPURasterContrastEqualization: {
    family: 'contrast-equalization',
    descriptor: {scope: 'block', scopeSecondary: 'local', reduction: 'histogram'},
    variables: {
      inputValues: 'rate',
      equalizedValues: 'fraction',
      equalizedValidity: 'category',
      approximation: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'mode',
        alternatives: ['clahe', 'match-histograms'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'tileWidth', sweepable: true},
      {stage: 'neighbourhood', name: 'tileHeight', sweepable: true},
      {stage: 'operator', name: 'clipLimit', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'CLAHE clips and redistributes each reflected tile histogram, scans its CDF and bilinearly interpolates neighboring mappings; histogram matching interpolates the inverse reference CDF. The graph shows these distinct stages and explicit validity handling.'
    },
    sources: [
      'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
    ]
  }
};
