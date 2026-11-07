// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseLimeModelFormOptions} from './meuse-lime-model-form.compute';

export default defineScene<MeuseLimeModelFormOptions>({
  id: 'meuse-lime-model-form',
  title: 'Meuse measured lime: probit and local logistic models',
  chapter: 'regression',
  order: 8,
  summary:
    'The observed binary lime field compares a global probit association with local Bernoulli logistic fits and leave-one-out bandwidth selection.',
  contributors: ['GPUProbitRegression', 'GPUGeneralizedGeographicallyWeightedRegression'],
  datasets: [
    {id: 'meuse-validation', role: '155 measured topsoil samples and observed binary lime field'}
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.8},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Do observed soil associations vary by site?',
      subtitle: 'Observed binary response · global probit and local logistic fits'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'These are associations in measured topsoil composites. The global probit and local logistic models differ in link and spatial support, so their fit differences cannot be attributed to spatial variation alone.'
  },
  options: [
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped values',
      group: 'Map',
      apply: 'param',
      default: 'local-probability',
      options: [
        {value: 'local-probability', label: 'Local logistic in-sample probability'},
        {value: 'lime-observed', label: 'Observed lime indicator'}
      ]
    }
  ],
  story: [
    {
      id: 'measured-binary-field',
      title: 'Use the observed binary response',
      headline: 'The source lime field records presence at 44 of 155 sites',
      textAlternative:
        'Measured sample markers show the local logistic probability or the source zero-one lime indicator.',
      body: 'The response is the original `lime` field stored as 0/1 in the Meuse data; no threshold is applied to a continuous measurement. Predictors are standardized log zinc, measured elevation and measured organic matter. Organic matter is missing in two source rows, so models use 153 complete cases. The 15 m × 15 m topsoil composites are represented by their sample locations in the Dutch RD metric plane.',
      controls: ['display'],
      readouts: ['sampleCount'],
      evidence:
        'The source contains 44 lime presences and 111 absences; two rows are excluded for missing organic matter.'
    },
    {
      id: 'global-probit',
      title: 'Fit a global probit association',
      headline: 'Fisher scoring estimates a shared global coefficient vector',
      textAlternative:
        'Readouts report model status, log likelihood, iteration count, coefficients and standard errors.',
      body: '**`GPUProbitRegression`** fits a global Bernoulli probit model with an implicit intercept and bounded Fisher scoring. Reported coefficient standard errors are model-based; coefficients are not marginal effects. Non-converged or separated fits are identified by status instead of being treated as successful estimates.',
      controls: ['display'],
      readouts: ['probitStatus', 'probitCoefficients'],
      caveat:
        'This global fit assumes one coefficient vector and does not model a spatial lag. It differs from the local fit in both link and spatial support.'
    },
    {
      id: 'local-logistic',
      title: 'Fit local Bernoulli logistic associations',
      headline: 'Bisquare distance weights allow the local logit to vary by site',
      textAlternative:
        'The map shows in-sample fitted local probabilities after the selected-bandwidth fits succeed, with a readout of converged and failed local fits.',
      body: '**`GPUGeneralizedGeographicallyWeightedRegression`** estimates local Bernoulli logistic models using bisquare weights over Euclidean distances in EPSG:28992 metres. The map shows in-sample fitted probabilities rather than observed class predictions; local status counts identify sites where fits were unsupported or failed.',
      options: {display: 'local-probability'},
      controls: ['display'],
      readouts: ['localStatus'],
      caveat:
        'Local fits reuse overlapping neighborhoods and are exploratory. In-sample fitted probabilities are not accuracy estimates.'
    },
    {
      id: 'bandwidth-selection',
      title: 'Compare bandwidths by leave-one-out deviance',
      headline: 'A finite candidate search exposes the local smoothing choice',
      textAlternative:
        'A bar chart compares leave-one-out mean deviance for five fixed bisquare distances.',
      body: 'The generalized GWR contributor scores five authored fixed bandwidths using exact leave-one-out mean deviance, then the local model uses the smallest-scoring distance. This bandwidth search uses the same 153 observations and does not provide independent spatial validation or selection-uncertainty estimates.',
      controls: ['display'],
      readouts: ['selectionStatus', 'bandwidthScores'],
      caveat:
        'The global probit and local logistic likelihoods are not a controlled comparison of spatial variation because they use different link functions and supports.'
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title:
        state.display === 'local-probability'
          ? 'Local fitted lime probability'
          : 'Observed lime indicator',
      unit: state.display === 'local-probability' ? 'probability' : '0 = absent · 1 = present',
      ramp: state.display === 'local-probability' ? 'cividis' : 'diverging',
      extent: [0, 1]
    }
  ],
  readouts: [
    {id: 'sampleCount', label: 'Complete-case sample'},
    {id: 'probitStatus', label: 'Global probit status'},
    {id: 'probitCoefficients', label: 'Probit coefficients / standard errors'},
    {id: 'localStatus', label: 'Local logistic fit status'},
    {id: 'selectionStatus', label: 'LOO bandwidth choice'},
    {id: 'bandwidthScores', label: 'Leave-one-out deviance', kind: 'chart'}
  ],
  about: {
    what: 'A global binary probit and geographically weighted binary logistic model use the measured Meuse lime indicator.',
    why: 'The scene uses the source’s genuine binary field and reports model status and local smoothing scores rather than inventing a cutoff or presenting fitted values as validation.',
    howToRead:
      'The map shows local fitted probabilities or measured 0/1 values. The bandwidth chart selects among five candidates by leave-one-out deviance within the same complete-case sample.'
  },
  create: async ctx =>
    (await import('./meuse-lime-model-form.compute')).createMeuseLimeModelForm(ctx)
});
