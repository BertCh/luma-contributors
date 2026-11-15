// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

/** Whole-domain periodic filtering through the published FFT2D. */
export const RASTER_SPECTRAL_FILTER_OPERATORS: RasterOperatorTable = {
  GPURasterSpectralFilter: {
    family: 'spectral-filter',
    descriptor: {scope: 'global', reduction: 'solve'},
    variables: {
      spectralInput: 'measure',
      filteredValues: 'measure',
      filteredValidity: 'category',
      invalidCount: 'measure'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'filter',
        alternatives: ['low-pass', 'high-pass', 'band-pass', 'gaussian', 'notch'],
        sweepable: false
      },
      {stage: 'operator', name: 'cutoff', sweepable: true},
      {stage: 'operator', name: 'sigma', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Multiply the published two-dimensional Fourier transform by a real conjugate-symmetric frequency response and invert it: y = inverseFFT2D(H * FFT2D(x)). Frequencies are cycles per pixel on a periodic complete domain; any missing input invalidates every output. Ideal low/high/band-pass, Gaussian and paired toroidal notch responses use per-frame settings.'
    },
    sources: ['modules/analysis/src/raster-analysis/spectral-filter/gpu-raster-spectral-filter.ts']
  }
};
