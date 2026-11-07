// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseMetalModelFormOptions} from './meuse-metal-model-form.compute';

export default defineScene<MeuseMetalModelFormOptions>({
  id: 'meuse-metal-model-form',
  title: 'Meuse metals: joint equations and local model form',
  chapter: 'regression',
  order: 9,
  summary:
    'Aligned measured zinc, copper and lead support feasible SUR, local metal-predictor collinearity diagnostics and a fixed-profile multiscale GWR fit.',
  contributors: [
    'GPUSeeminglyUnrelatedRegression',
    'GPUNeighborSearch',
    'GPULocalCollinearity',
    'GPUMultiscaleGeographicallyWeightedRegression'
  ],
  datasets: [
    {id: 'meuse-validation', role: '155 measured soil-metal samples and environmental covariates'}
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.8},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Do measured metal associations vary by place?',
      subtitle: 'Joint log-metal equations · local collinearity · multiscale GWR'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Models are exploratory associations over measured topsoil composites. SUR and MGWR use 153 common complete rows; MGWR bandwidths are authored sensitivities, not selected optima.'
  },
  options: [
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped diagnostic',
      group: 'Map',
      apply: 'param',
      default: 'local-distance-coefficient',
      options: [
        {value: 'local-distance-coefficient', label: 'MGWR river-distance coefficient'},
        {value: 'local-metal-condition', label: 'Local log-metal condition index'}
      ]
    },
    {
      kind: 'select',
      id: 'bandwidthProfile',
      label: 'MGWR bandwidth profile',
      group: 'Model form',
      apply: 'param',
      default: 'compact',
      help: 'These are authored fixed bandwidth sensitivities for the intercept, elevation, organic-matter and river-distance components. They are not chosen by an optimization routine.',
      options: [
        {value: 'compact', label: 'Compact: 3.5 / 1.8 / 1.8 / 2.2 km'},
        {value: 'broad', label: 'Broad: 4.5 / 2.8 / 2.8 / 3.2 km'}
      ]
    }
  ],
  story: [
    {
      id: 'common-sample',
      title: 'Keep the measured metal rows aligned',
      headline: 'Three log-metal responses share the same sample IDs and design',
      textAlternative:
        'The map shows local model diagnostics at each retained Meuse topsoil sample.',
      body: 'The response system uses the observed log zinc, log copper and log lead values from the same Meuse rows. Each equation has a common, standardized predictor set: measured elevation, measured organic matter and measured distance to the river. Two rows have missing organic matter and are omitted from all equations. Coordinates are centered EPSG:28992 positions in metres; the map uses the source WGS84 sample locations.',
      controls: ['display', 'bandwidthProfile'],
      readouts: ['studyCount', 'neighborStatus'],
      evidence:
        'Responses and predictors use 153 common complete observations; no metal values are imputed.'
    },
    {
      id: 'joint-metal-system',
      title: 'Fit aligned equations jointly',
      headline: 'One-step feasible SUR estimates the three-metal system',
      textAlternative:
        'A readout lists equation coefficients and residual correlations from a joint feasible SUR fit.',
      body: '**`GPUSeeminglyUnrelatedRegression`** fits first-stage equations, estimates their residual covariance, then solves one joint GLS system. The common measured predictors describe elevation, soil organic matter and river distance. Because the three equations use the same rows and predictor matrix, SUR mainly adds the joint residual-covariance view here; its coefficient estimates do not claim an efficiency gain over separate OLS with identical designs. Residual correlations do not imply that the metals are independent or that cross-equation coefficients are causal.',
      controls: ['bandwidthProfile'],
      readouts: ['surStatus', 'surResiduals'],
      caveat:
        'This is one-step feasible SUR with a common complete-case sample, not iterated SUR, spatial SUR or small-sample inference.'
    },
    {
      id: 'metal-collinearity',
      title: 'Inspect local metal-predictor collinearity',
      headline: 'Local condition indices measure the geometry of the metal vector',
      textAlternative:
        'The map colors each site by a weighted local correlation condition number computed from its 12 nearest neighbors.',
      body: '**`GPUNeighborSearch`** supplies a 12-nearest-neighbor row-standardized graph. **`GPULocalCollinearity`** measures the square root of the maximum-to-minimum eigenvalue ratio of the centered local log zinc, copper and lead correlation matrix. This map diagnoses how well the three metal variables can be distinguished locally; it is not a significance test or a regression coefficient surface.',
      options: {display: 'local-metal-condition'},
      controls: ['display'],
      readouts: ['collinearityStatus', 'neighborStatus'],
      caveat:
        'This condition map uses the three log-metal values as a separate diagnostic predictor set; MGWR instead uses elevation, organic matter and river distance. The condition index is sensitive to local sample geometry and the chosen 12-neighbor rule. Singular or unsupported neighborhoods are counted and shown as missing.'
    },
    {
      id: 'multiscale-metal-model',
      title: 'Compare fixed spatial scales in a local zinc model',
      headline: 'MGWR gives separate authored bandwidths to the covariate surfaces',
      textAlternative:
        'The map shows the local coefficient on standardized measured river distance; the readout identifies the bandwidth profile and fit status.',
      body: '**`GPUMultiscaleGeographicallyWeightedRegression`** models log zinc using standardized elevation, organic matter and measured river distance. Each coefficient surface has its own fixed distance bandwidth. The compact and broad profiles are intentionally authored sensitivity settings; neither is claimed to be optimal or validated. The mapped river-distance coefficient is measured in log-zinc units per one global standard deviation of that predictor.',
      options: {display: 'local-distance-coefficient'},
      controls: ['display', 'bandwidthProfile'],
      readouts: ['mgwrStatus'],
      caveat:
        'The Gaussian response fit is in-sample and local coefficients can be unstable. The displayed surface is not a causal effect or a held-out prediction.'
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title:
        state.display === 'local-distance-coefficient'
          ? 'MGWR coefficient · river distance'
          : 'Local log-metal condition index',
      unit:
        state.display === 'local-distance-coefficient'
          ? 'log-zinc units per SD'
          : 'sqrt(max / min eigenvalue)',
      ramp: state.display === 'local-distance-coefficient' ? 'diverging' : 'magma',
      extent: state.display === 'local-distance-coefficient' ? [-0.4, 0.4] : [0, 40],
      ...(state.display === 'local-distance-coefficient' ? {midpoint: 0} : {})
    }
  ],
  readouts: [
    {id: 'studyCount', label: 'Common complete-case sample'},
    {id: 'neighborStatus', label: 'Local neighbor graph'},
    {id: 'collinearityStatus', label: 'Local collinearity status'},
    {id: 'surStatus', label: 'Joint SUR coefficients'},
    {id: 'surResiduals', label: 'SUR residual correlations'},
    {id: 'mgwrStatus', label: 'MGWR fit and coefficient range'}
  ],
  about: {
    what: 'The scene compares a joint three-equation log-metal model, local predictor-collinearity diagnostics and a Gaussian multiscale GWR model for log zinc.',
    why: 'Aligned measured metals share residual structure, while local collinearity and coefficient surfaces show distinct aspects of spatial model form.',
    howToRead:
      'SUR coefficient readouts use standardized measured predictors. The local condition map diagnoses the three correlated log metals. MGWR maps one local coefficient under authored fixed bandwidth profiles; those profiles are sensitivity settings rather than optimized choices.'
  },
  create: async ctx =>
    (await import('./meuse-metal-model-form.compute')).createMeuseMetalModelForm(ctx)
});
