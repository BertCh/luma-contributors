// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseConditionalSimulationOptions} from './meuse-conditional-simulation.compute';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';

export default defineScene<MeuseConditionalSimulationOptions>({
  id: 'meuse-conditional-simulation',
  title: 'How much can plausible soil maps differ?',
  chapter: 'interpolation',
  order: 4,
  summary:
    'Joint conditional Gaussian draws use 128 explicitly selected Meuse zinc samples and 128 in-hull source-grid queries, then compare simulated moments with GPU conditional moments.',
  contributors: ['GPUConditionalGaussianSimulation'],
  datasets: [
    {
      id: 'meuse-validation',
      role: 'measured zinc samples, bounded source-grid window and river outline'
    }
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.4},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'How much can plausible soil maps differ?',
      subtitle: 'Joint conditional log-zinc draws under one stated covariance'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Draws condition on a disclosed maximin subset of 128 of the 155 samples and a compact 128-query window, not the full study area.'
  },
  options: [
    {
      kind: 'select',
      id: 'knownMean',
      label: 'Assumed log-Zn mean',
      group: 'Covariance assumptions',
      apply: 'param',
      default: '5.9',
      help: 'This is a known-mean simple-kriging assumption, not a mean estimated from the selected observations. The 5.9 default follows the gstat teaching example.',
      options: [
        {value: '5.5', label: '5.5'},
        {value: '5.9', label: '5.9 · gstat example'},
        {value: '6.3', label: '6.3'}
      ]
    },
    {
      kind: 'select',
      id: 'observationErrorVariance',
      label: 'Assumed observation-error variance',
      group: 'Covariance assumptions',
      apply: 'param',
      default: '0',
      help: 'Adds a disclosed diagonal measurement-error variance to each conditioning observation, in squared natural-log units. This is a sensitivity assumption, not an estimated property of the Meuse samples.',
      options: [
        {value: '0', label: '0 · exact conditioning'},
        {value: '0.01', label: '0.01'},
        {value: '0.04', label: '0.04'},
        {value: '0.16', label: '0.16'}
      ]
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Reproducible random seed',
      group: 'Simulation',
      apply: 'param',
      min: 0,
      max: 65535,
      step: 1,
      default: 31,
      help: 'Changes the joint pseudo-random stream. The random sequence is reproducible within a GPU backend; it is not promised to match another library bit for bit.'
    },
    {
      kind: 'select',
      id: 'replicateCount',
      label: 'Draw count',
      group: 'Simulation',
      apply: 'param',
      default: '32',
      options: [
        {value: '32', label: '32'},
        {value: '128', label: '128'},
        {value: '256', label: '256'}
      ]
    },
    {
      kind: 'select',
      id: 'replicateOffset',
      label: 'Replicate batch offset',
      group: 'Simulation',
      apply: 'param',
      default: '0',
      help: 'Changes the starting replicate identity while reusing the same conditional preparation. Batch identities are the selected offset through offset + draw count − 1.',
      options: [
        {value: '0', label: '0'},
        {value: '256', label: '256'},
        {value: '512', label: '512'}
      ]
    },
    {
      kind: 'slider',
      id: 'replicateIndex',
      label: 'Mapped replicate',
      group: 'Simulation',
      apply: 'param',
      min: 0,
      max: 255,
      step: 1,
      default: 0,
      help: 'A selected joint draw is mapped when its index is below the requested draw count.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped quantity',
      group: 'Map',
      apply: 'param',
      default: 'realization',
      options: [
        {value: 'realization', label: 'Joint realization'},
        {value: 'mean', label: 'Conditional mean'},
        {value: 'variance', label: 'Conditional variance'}
      ]
    }
  ],
  story: [
    {
      id: 'bounded-design',
      title: 'Choose a bounded but real conditioning problem',
      headline: '128 measured sites condition a compact in-hull query window',
      textAlternative:
        'The selected Meuse sample sites and a compact block of original query locations appear around the river.',
      body: 'The GPU operator factors a dense joint covariance, so this lab uses an explicit 128 × 128 problem. The sample subset is maximin, seeded at the full sample bounding-box midpoint, then returned to source-row order. The 128 queries are the source grid locations nearest the centroid of grid points inside that subset’s convex hull; exact source row indices are kept in the scene helper. The remaining 27 measured sites are not used for conditioning.',
      controls: ['knownMean'],
      readouts: ['selectionDesign', 'sampleCount', 'queryCount'],
      evidence:
        'The simulated concentrations are log zinc at field sites and source-grid locations; no thresholded or synthetic response is introduced.'
    },
    {
      id: 'assumed-covariance',
      title: 'Treat mean and covariance as assumptions',
      headline: 'The covariance is authored in advance, not fit to the full data',
      textAlternative:
        'The mean-control selection changes conditional moments while the map and analytic-versus-draw charts update.',
      body: 'The model uses simple kriging with a known log-Zn mean (5.9 by default) and a spherical covariance with sill 0.59, range 874 m and nugget 0.04, taken from the documented gstat teaching example. **Assumed log-Zn mean** offers 5.5, 5.9 and 6.3 to make sensitivity visible. **Assumed observation-error variance** adds the selected diagonal variance to all conditioning observations, in squared log units. Neither the covariance nor the mean is estimated from all 155 samples. The known mean is shifted by the operator to form conditional means; the error setting changes conditioning and therefore both mean and variance.',
      controls: ['knownMean', 'observationErrorVariance', 'display'],
      readouts: ['simulationStatus', 'momentComparison'],
      caveat:
        'These parameters illustrate conditional simulation under an authored model; they are not a fitted or validated variogram for this field.'
    },
    {
      id: 'joint-draws',
      title: 'Inspect one joint realization',
      headline: 'One realization preserves covariance across query locations',
      textAlternative:
        'A spatial map shows one selected conditional log-zinc realization across the bounded query window.',
      body: '**`GPUConditionalGaussianSimulation`** draws all 128 query values jointly, so one realization retains the model covariance across locations. **Mapped replicate** chooses a draw within the active batch. **Replicate batch offset** selects a different non-overlapping range of replicate identities while reusing the same preparation. Changing the **Philox seed** produces a different reproducible stream within this GPU backend; increasing **Draw count** generates a larger batch.',
      controls: ['display', 'replicateIndex', 'replicateCount', 'replicateOffset', 'seed'],
      camera: {
        bounds: [5.7399651, 50.9694087, 5.7468319, 50.9737023],
        padding: {left: 380, right: 300, top: 65, bottom: 65},
        transitionMs: 1500
      },
      readouts: ['simulationStatus'],
      evidence:
        'The simulation reuses its conditional preparation between draw batches instead of refactoring the same observation geometry.'
    },
    {
      id: 'conditional-moments',
      title: 'Separate mean from conditional variance',
      headline: 'The GPU reports analytic conditional moments separately',
      textAlternative:
        'The map switches between the conditional mean in log concentration and conditional variance in squared log units.',
      body: 'The **Conditional mean** includes the selected known log-Zn mean and the observations’ conditioning information. **Conditional variance** reflects the fixed spherical covariance, the 874 m range, and this sample/query geometry. The variance map omits sample-value markers so its single legend describes only the displayed quantity.',
      options: {display: 'variance'},
      controls: ['knownMean', 'display'],
      camera: {
        bounds: [5.7399651, 50.9694087, 5.7468319, 50.9737023],
        padding: {left: 380, right: 300, top: 65, bottom: 65},
        transitionMs: 1200
      },
      readouts: ['simulationStatus', 'momentComparison'],
      caveat:
        'Variance is conditional on the authored covariance and on the selected 128 observations; it is not a fitted standard error.'
    },
    {
      id: 'monte-carlo-check',
      title: 'Check finite-draw moment error',
      headline: 'Empirical moments move toward the analytic moments as draws accumulate',
      textAlternative:
        'Two bar charts compare analytic and empirical conditional means and variances averaged over the selected query locations.',
      body: 'The charts compare analytic conditional moments with the mean and unbiased sample variance computed from the active draws at each query, then averaged across the 128 locations. Choose **Draw count** (32, 128 or 256), **Replicate batch offset** or **Philox seed** to see Monte Carlo variation. The empirical values should approach the analytic values as independent draw identities accumulate, but they will not match exactly.',
      controls: ['replicateCount', 'replicateOffset', 'seed'],
      camera: {
        bounds: [5.7399651, 50.9694087, 5.7468319, 50.9737023],
        padding: {left: 380, right: 300, top: 65, bottom: 65},
        transitionMs: 1200
      },
      readouts: ['meanComparison', 'varianceComparison', 'momentComparison'],
      caveat:
        'This is a conditional distribution under the stated covariance, known mean, observation-error assumption, subset and query window—not a probability statement about the whole Meuse catchment.'
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title:
        state.display === 'variance'
          ? 'Conditional variance'
          : state.display === 'mean'
            ? 'Conditional log-Zn mean'
            : 'Simulated log-Zn',
      unit: state.display === 'variance' ? 'log units²' : 'log mg/kg',
      ramp: 'cividis',
      extent: state.display === 'variance' ? [0, 0.6] : MEUSE_LOG_RESPONSE_RANGES.zinc
    },
    {
      kind: 'categories',
      title: 'Map features',
      entries:
        state.display === 'variance'
          ? [{color: [49, 113, 151, 230], label: 'Meuse river outline'}]
          : [
              {color: [49, 113, 151, 230], label: 'Meuse river outline'},
              {color: [16, 18, 22, 255], label: 'Conditioning sample outline'}
            ]
    }
  ],
  readouts: [
    {id: 'selectionDesign', label: 'Subset selection rule'},
    {id: 'sampleCount', label: 'Conditioning samples', format: 'integer'},
    {id: 'queryCount', label: 'Joint query locations', format: 'integer'},
    {id: 'simulationStatus', label: 'Simulation status'},
    {id: 'momentComparison', label: 'Across-query analytic / draw moments'},
    {id: 'meanComparison', label: 'Mean comparison', kind: 'chart'},
    {id: 'varianceComparison', label: 'Variance comparison', kind: 'chart'}
  ],
  about: {
    what: 'GPUConditionalGaussianSimulation produces bounded joint conditional Gaussian draws at a documented subset of real Meuse query sites.',
    why: 'It separates covariance assumptions, conditional moments and Monte Carlo realizations, while exposing the dense operator’s explicit size bound.',
    howToRead:
      'The map is in natural-log zinc units except for conditional variance, which is in squared log units. Empirical moment charts average across the bounded query window.'
  },
  create: async ctx =>
    (await import('./meuse-conditional-simulation.compute')).createMeuseConditionalSimulation(ctx)
});
