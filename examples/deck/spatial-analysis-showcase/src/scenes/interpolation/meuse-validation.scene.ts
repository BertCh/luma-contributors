// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseValidationOptions} from './meuse-validation.compute';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';

export default defineScene<MeuseValidationOptions>({
  id: 'meuse-validation',
  title: 'Meuse topsoil metals: choose a surface on held-out regions',
  chapter: 'interpolation',
  order: 2,
  summary:
    'Nested spatial validation compares fixed local-kriging and RBF settings on 155 composite topsoil samples, then maps a separately labelled all-sample fit on the original 40 m-spaced point grid.',
  contributors: ['GPULocalKriging', 'GPULocalRBF', 'GPUNestedInterpolationValidation'],
  datasets: [
    {id: 'meuse-validation', role: 'measured topsoil metals, prediction grid and river outline'}
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.4},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Meuse topsoil metal interpolation',
      subtitle: 'Nested spatial validation · log concentration'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Each sample is a 15 m × 15 m composite represented by a point. Surface view fits all 155 samples; residual view uses held-out predictions.'
  },
  options: [
    {
      kind: 'select',
      id: 'response',
      label: 'Metal',
      group: 'Data',
      apply: 'param',
      default: 'zinc',
      display: 'segmented',
      help: 'Each positive measured concentration is log-transformed before the spatial fits. Validation and surface units are natural-log concentration.',
      options: [
        {value: 'zinc', label: 'Zinc'},
        {value: 'copper', label: 'Copper'},
        {value: 'lead', label: 'Lead'},
        {value: 'cadmium', label: 'Cadmium'}
      ]
    },
    {
      kind: 'select',
      id: 'estimator',
      label: 'Descriptive surface model',
      group: 'Surface',
      apply: 'compile',
      default: 'ordinary',
      help: 'Rebuilds one full-data display model. It is separate from the GPU nested validation result; candidate selection is restricted to each outer-training sample.',
      options: [
        {
          value: 'ordinary',
          label: 'Ordinary kriging',
          help: 'Unknown locally constant mean with a fixed exponential covariance.'
        },
        {
          value: 'universal',
          label: 'Universal kriging',
          help: 'Locally estimated affine spatial trend with the same fixed covariance.'
        },
        {
          value: 'external-drift',
          label: 'External drift',
          help: 'Uses an intercept and square-root river distance as supplied drift columns; exploratory rather than a causal model.'
        },
        {
          value: 'rbf',
          label: 'Local RBF',
          help: 'Local radial-basis surface using the selected kernel; the secondary diagnostic is its solver pivot ratio, not kriging variance.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'rbfKernel',
      label: 'RBF kernel',
      group: 'Surface',
      apply: 'compile',
      default: 'thin-plate-spline',
      disabledWhen: state => state.estimator !== 'rbf',
      help: 'One selected kernel is compiled for the direct surface. Degree is set to the smallest valid polynomial augmentation for that kernel; epsilon is 1/300 m⁻¹ and smoothing is fixed at 0.02. Quintic local fits need degree-two polynomial terms and may fail where a neighborhood geometry is rank-deficient; those locations are shown as gaps, not zero predictions.',
      options: [
        {value: 'thin-plate-spline', label: 'Thin plate spline'},
        {value: 'cubic', label: 'Cubic'},
        {value: 'gaussian', label: 'Gaussian'},
        {value: 'linear', label: 'Linear'},
        {value: 'quintic', label: 'Quintic'},
        {value: 'multiquadric', label: 'Multiquadric'},
        {value: 'inverse-multiquadric', label: 'Inverse multiquadric'},
        {value: 'inverse-quadratic', label: 'Inverse quadratic'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped quantity',
      group: 'Surface',
      apply: 'param',
      default: 'prediction',
      help: 'Prediction and OOF residual are log concentrations. Kriging variance is in squared log units; RBF shows a numerical pivot diagnostic because it has no predictive variance. Diagnostic maps omit sample markers so the single legend matches the mapped quantity.',
      options: [
        {value: 'prediction', label: 'Log prediction'},
        {value: 'variance', label: 'Variance / solver diagnostic'},
        {value: 'oof-residual', label: 'Out-of-fold residuals'}
      ]
    }
  ],
  story: [
    {
      id: 'observations',
      title: 'A measured field, not a generated surface',
      headline: '155 topsoil samples anchor the prediction problem',
      textAlternative:
        'Measured metal concentrations appear at field sites, over the original Meuse river outline and prediction-grid footprint.',
      body: 'The points represent topsoil composites collected over 15 m × 15 m areas; the grid gives the source’s 40 m-spaced query locations, and the blue line is the documented river outline. The computational coordinates remain in the source Dutch RD plane (EPSG:28992), so distance is measured in metres. **Metal** changes the measured response; concentrations are modelled on a natural-log scale. The published cadmium series replaces recorded zeros with 0.2 mg/kg, so those entries are not exact measured zeros.',
      controls: ['response'],
      readouts: ['sampleCount', 'gridCount'],
      caveat:
        'The source package documents the grid soil and flood-frequency classes as uncertain; this scene uses only its coordinates and distance-to-river field.'
    },
    {
      id: 'nested-validation',
      title: 'Choose settings inside the training region',
      headline: 'Nested spatial folds keep each outer test region out of selection',
      textAlternative:
        'Three outer spatial folds each choose among five fixed kriging and RBF candidates using two inner folds, then report held-out error.',
      body: '**`GPUNestedInterpolationValidation`** uses three spatial outer folds. Within each outer-training set, two smaller spatial folds compare five authored settings: exponential kriging at 300 m and 900 m, spherical kriging at 900 m, thin-plate RBF (degree 1), and Gaussian RBF (constant augmentation). All covariance candidates use sill 0.35 and nugget 0.05 in log-metal units; the RBF settings use epsilon 1/300 m⁻¹ and smoothing 0.02 (thin plate) or 0.08 (Gaussian). The outer test block is scored only after that choice. The RMSE chart is in log-metal units. The fold readout names the selected candidate and the pooled RMSE weights each outer fold by its number of scored samples.',
      controls: ['display'],
      readouts: ['candidateScores', 'validationRmse', 'selectedModels', 'foldErrors'],
      evidence:
        'The candidate constants are fixed in advance; no mean, sill, range or bandwidth is estimated using the complete response before validation.'
    },
    {
      id: 'surface',
      title: 'Map the full-sample descriptive fit',
      headline: 'The mapped surface uses all 155 observations after validation',
      textAlternative:
        'A log-metal surface covers the original 40 m-spaced query locations, with observed field sites and the Meuse river outline shown for reference.',
      body: 'This surface is a descriptive fit using all observations. **Descriptive surface model** chooses ordinary kriging, universal kriging, external drift or a local RBF. Its settings are fixed and are not selected by the outer test scores. External drift uses an intercept and square-root distance to the river as supplied columns. Choose one of the eight **RBF kernel** functions when RBF is active. Switch **Mapped quantity** to inspect conditional kriging variance (in squared log units); the RBF instead displays a solver pivot diagnostic.',
      controls: ['estimator', 'rbfKernel', 'display'],
      readouts: ['surfaceStatus'],
      caveat:
        'Exponentiating a predicted log concentration gives a median under the lognormal interpretation, not the expected concentration on the original scale.'
    },
    {
      id: 'interpretation',
      title: 'Read scores and surfaces on their own terms',
      headline: 'The OOF residual map shows where spatial transfer misses',
      textAlternative:
        'A map of out-of-fold residuals shows measured minus predicted log concentration at held-out sites alongside the fold-wise error chart and a separate full-data fit.',
      body: 'Choose **Out-of-fold residuals** to map one held-out prediction per sample. Hover a point for its observed concentration, held-out prediction median, residual and fold. Positive residuals mean observed log concentration exceeded its held-out prediction. The full-grid map instead uses all observations and is not an independent assessment. Kriging variance is conditional on the selected covariance and local neighborhood. The RBF pivot ratio flags numerical conditioning and is not uncertainty.',
      options: {display: 'oof-residual'},
      controls: ['response', 'estimator', 'display'],
      readouts: ['validationRmse', 'selectedModels', 'surfaceStatus', 'foldErrors'],
      caveat:
        'Three outer folds provide a bounded spatial check, not a universal estimate for other catchments, seasons, soil depths or sampling designs.'
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title:
        state.display === 'prediction'
          ? `${state.response.charAt(0).toUpperCase()}${state.response.slice(1)} · log concentration`
          : state.display === 'oof-residual'
            ? 'OOF residual · log units'
            : state.estimator === 'rbf'
              ? 'RBF pivot ratio'
              : 'Kriging variance',
      unit:
        state.display === 'prediction'
          ? 'log mg/kg'
          : state.display === 'oof-residual'
            ? 'log units'
            : state.estimator === 'rbf'
              ? 'dimensionless'
              : 'log units²',
      ramp:
        state.display === 'oof-residual'
          ? 'diverging'
          : state.display === 'prediction'
            ? 'viridis'
            : 'magma',
      extent:
        state.display === 'prediction'
          ? MEUSE_LOG_RESPONSE_RANGES[state.response]
          : state.display === 'oof-residual'
            ? [-1.5, 1.5]
            : state.estimator === 'rbf'
              ? [0, 1]
              : [0, 0.6]
    },
    {
      kind: 'categories',
      title: 'Map features',
      entries: [{color: [49, 113, 151, 230], label: 'Meuse river outline'}]
    }
  ],
  readouts: [
    {id: 'sampleCount', label: 'Composite field samples', format: 'integer'},
    {id: 'gridCount', label: '40 m-spaced queries', format: 'integer'},
    {id: 'validationDesign', label: 'Nested validation design'},
    {id: 'validationRmse', label: 'Pooled outer RMSE'},
    {id: 'selectedModels', label: 'Inner-selected models'},
    {id: 'candidateScores', label: 'Inner candidate RMSE by fold', kind: 'chart'},
    {id: 'foldErrors', label: 'Outer-fold RMSE', kind: 'chart'},
    {id: 'validationStatus', label: 'Validation status'},
    {
      id: 'surfaceStatus',
      label: 'Surface interpretation',
      help: 'Prediction/variance counts describe the current full-data grid. Out-of-fold residual counts describe finite held-out sample predictions; training rows are intentionally blank. RBF pivot diagnostics are unitless and are not prediction variance.'
    }
  ],
  about: {
    what: 'Nested GPU spatial cross-validation compares a fixed five-candidate interpolation ladder on genuine Meuse topsoil measurements and maps a separate full-sample fit on the source prediction grid.',
    why: 'It makes spatial holdout, model selection and full-data description visible as distinct steps rather than treating the best in-sample-looking map as validated.',
    howToRead:
      'Outer RMSE is measured in natural-log concentration units. The surface is fit to all observations; kriging variance is conditional, while the RBF diagnostic is numerical conditioning.'
  },
  create: async ctx => (await import('./meuse-validation.compute')).createMeuseValidation(ctx)
});
