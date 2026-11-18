// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {PointCloudChangeOptions} from './point-cloud-change.types';
export type {PointCloudChangeOptions} from './point-cloud-change.types';

export default defineScene<PointCloudChangeOptions>({
  id: 'point-cloud-change',
  title: 'When does a moving surface exceed measurement uncertainty?',
  chapter: 'terrain',
  order: 9,
  summary:
    'Compare two real rock-glacier lidar epochs along provided core normals, then inspect point-to-point and point-to-plane local ICP as fits that can absorb movement into alignment.',
  contributors: [
    'GPUPointCloudChange',
    'GPUPointCloudICP',
    'GPUPointCloudNeighborhoods',
    'GPUPointCloudFeatures'
  ],
  datasets: [
    {id: 'ahk-rock-glacier', role: 'real 2017 and 2018A TLS epochs, supplied cores and normals'}
  ],
  initialView: {
    longitude: 11.005414391236211,
    latitude: 46.83836570985789,
    zoom: 20.3,
    pitch: 0,
    bearing: 0
  },
  basemap: ground('night'),
  annotations: [
    {
      kind: 'area',
      coordinate: [11.005414391236211, 46.83836570985789],
      text: 'Äußeres Hochebenkar',
      size: 'small'
    }
  ],
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Map result',
      group: 'Display',
      apply: 'param',
      default: 'epochs',
      display: 'segmented',
      help: 'Switch among the measured 2017 and 2018A return sets, signed comparison-minus-reference displacement, and its interval test.',
      options: [
        {value: 'epochs', label: 'Epoch returns'},
        {value: 'difference', label: 'Signed change'},
        {value: 'detection', label: 'Interval test'}
      ]
    },
    {
      kind: 'select',
      id: 'alignment',
      label: 'Comparison frame',
      group: 'Alignment sensitivity',
      apply: 'compile',
      default: 'surveyed',
      display: 'segmented',
      help: 'The supplied survey transform is the study frame. Local ICP is an exploratory refit over bounded moving-crop samples and may absorb real change.',
      options: [
        {value: 'surveyed', label: 'Supplied survey frame'},
        {value: 'local-icp', label: 'Exploratory local ICP'}
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Cylinder radius',
      group: 'Core measurement',
      apply: 'compile',
      min: 0.8,
      max: 2,
      step: 0.1,
      default: 1.5,
      unit: 'm',
      help: 'Sets the radial support around each supplied core; the sample and radius determine which epoch returns contribute.'
    },
    {
      kind: 'slider',
      id: 'halfDepth',
      label: 'Half depth',
      group: 'Core measurement',
      apply: 'compile',
      min: 1,
      max: 5,
      step: 0.5,
      default: 3,
      unit: 'm',
      help: 'Sets the distance on either side of each core along its supplied normal.'
    },
    {
      kind: 'slider',
      id: 'registrationError',
      label: 'Registration error allowance',
      group: 'Uncertainty model',
      apply: 'compile',
      min: 0.03,
      max: 0.2,
      step: 0.01,
      default: 0.03,
      unit: 'm',
      help: 'Adds a scalar allowance to the sample-based uncertainty. The source subset does not include the full published registration covariance.'
    },
    {
      kind: 'select',
      id: 'uncertaintyCombination',
      label: 'Combine uncertainty',
      group: 'Uncertainty model',
      apply: 'compile',
      default: 'linear',
      display: 'segmented',
      help: 'Linear adds standard errors; quadrature treats sampling and registration terms as independent.',
      options: [
        {value: 'linear', label: 'Linear'},
        {value: 'quadrature', label: 'Quadrature'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'ICP metric',
      group: 'Exploratory ICP',
      apply: 'compile',
      default: 'point-to-point',
      disabledWhen: state => state.alignment !== 'local-icp',
      help: 'Point-to-plane estimates normals from 3 m neighborhoods on the bounded reference sample. Feature support and status are reported; this remains an exploratory local fit.',
      options: [
        {value: 'point-to-point', label: 'Point to point'},
        {value: 'point-to-plane', label: 'Point to plane'}
      ]
    },
    {
      kind: 'select',
      id: 'robust',
      label: 'ICP weighting',
      group: 'Exploratory ICP',
      apply: 'compile',
      default: 'none',
      disabledWhen: state => state.alignment !== 'local-icp',
      help: 'Robustly downweights large residual matches in the 1,024-row local fit.',
      options: [
        {value: 'none', label: 'Unweighted'},
        {value: 'huber', label: 'Huber'},
        {value: 'tukey', label: 'Tukey'}
      ]
    },
    {
      kind: 'slider',
      id: 'maximumDistance',
      label: 'ICP match distance',
      group: 'Exploratory ICP',
      apply: 'compile',
      min: 0.4,
      max: 2,
      step: 0.1,
      default: 1,
      unit: 'm',
      disabledWhen: state => state.alignment !== 'local-icp',
      help: 'Rejects distant correspondences. The fit uses at most 1,024 rows per epoch and 12 iterations.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Core mark radius',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 7,
      step: 0.5,
      default: 3,
      unit: 'px',
      help: 'Changes only the rendered core mark size.'
    }
  ],
  readouts: [
    {
      id: 'validCores',
      label: 'Cores with usable measurements',
      format: 'integer',
      emphasis: 'tile',
      help: 'GPUPointCloudChange results with adequate support in both epoch cylinders.'
    },
    {
      id: 'invalidNormals',
      label: 'Cores with invalid supplied normals',
      format: 'integer',
      help: 'Non-finite source normals are rejected and shown separately from support failures.'
    },
    {
      id: 'insufficientSupport',
      label: 'Cores with insufficient support',
      format: 'integer',
      help: 'Valid normals but too few retained returns in one or both epoch cylinders.'
    },
    {
      id: 'medianReferenceSupport',
      label: 'Median reference support',
      format: 'integer',
      help: 'Median number of retained 2017 returns inside the core cylinders.'
    },
    {
      id: 'medianComparisonSupport',
      label: 'Median comparison support',
      format: 'integer',
      help: 'Median number of retained 2018A returns inside the core cylinders.'
    },
    {
      id: 'icpFinalInliers',
      label: 'Final ICP inliers',
      format: 'integer',
      help: 'Final-transform correspondences from the GPU ICP inlier mask.'
    },
    {
      id: 'icpFinalStep',
      label: 'Final ICP translation step',
      format: 'decimal',
      unit: 'm',
      help: 'Translation update magnitude from the last GPU ICP iteration.'
    },
    {
      id: 'detectableCores',
      label: 'Intervals excluding zero',
      format: 'integer',
      emphasis: 'tile',
      help: 'The computed uncertainty interval lies wholly above or below zero.'
    },
    {
      id: 'medianDifference',
      label: 'Median signed difference',
      format: 'decimal',
      unit: 'm',
      help: 'Median of valid comparison-minus-reference core differences, in metres; no annual velocity is inferred.'
    },
    {
      id: 'medianLimit',
      label: 'Median detection limit',
      format: 'decimal',
      unit: 'm',
      help: 'Median half-width of the GPU uncertainty interval across valid cores.'
    },
    {
      id: 'icpStatus',
      label: 'Local ICP status',
      format: 'text',
      help: 'A converged or iteration-budget-exhausted local fit is shown as exploratory; insufficient-match or degenerate fits are rejected.'
    },
    {
      id: 'icpChart',
      label: 'Local ICP residual and step history',
      kind: 'chart',
      help: 'GPU readback of the selected metric objective and translation-step magnitude over at most 12 iterations. Point-to-point RMSE and point-to-plane residual are different objectives.'
    },
    {
      id: 'icpInlierChart',
      label: 'Local ICP inliers by iteration',
      kind: 'chart',
      help: 'GPU readback of the inlier count at each iteration.'
    },
    {
      id: 'targetNormalsReady',
      label: 'Usable target normals',
      format: 'integer',
      help: 'Reference sample rows with a finite nondegenerate local normal; invalid or nonconverged feature normals are zeroed before point-to-plane ICP.'
    },
    {
      id: 'targetNormalsWeak',
      label: 'Weak target normals',
      format: 'integer',
      help: 'Rows with insufficient or degenerate neighborhood support, or a nonconverged feature eigensolve. Their normals are zeroed before point-to-plane ICP.'
    },
    {
      id: 'targetNormalsTruncated',
      label: 'Truncated target neighborhoods',
      format: 'integer',
      help: 'Rows with more than 64 other points inside the fixed 3 m radius; the closest 64 are used for the normal.'
    },
    {
      id: 'targetMaximumSupport',
      label: 'Maximum 3 m neighborhood support',
      format: 'integer',
      help: 'Maximum number of other sampled reference returns inside the radius before the 64-neighbor feature cap.'
    },
    {
      id: 'differenceChart',
      label: 'Core displacement distribution',
      kind: 'chart',
      help: 'Fixed bins from -1 m to +1 m; values outside the shown range are counted in end bins.'
    },
    {
      id: 'boundChart',
      label: 'Per-core uncertainty bounds',
      kind: 'chart',
      help: 'GPU lower and upper bounds for each supported core, in metres.'
    },
    {
      id: 'intervalChart',
      label: 'Uncertainty interval test',
      kind: 'chart',
      help: 'GPU bounds grouped by whether zero is inside or outside the interval.'
    }
  ],
  story: [
    {
      id: 'surveyed-epochs',
      textAlternative:
        'Blue markers show 2017 measured returns, orange markers show 2018A measured returns, and pale rings show the provided core locations.',
      title: 'Start from two measured epochs',
      headline: 'The supplied alignment keeps the comparison in the survey frame',
      body: 'This is a 37 m patch at the Äußeres Hochebenkar rock glacier. The 2017 and 2018A point sets are real TLS returns, reduced to one original row per occupied 0.6 m voxel. The provider’s supplied transform was applied to 2018A during dataset preparation.',
      evidence:
        '**5,192** reference returns, **5,407** comparison returns and **258** supplied cores with normals enter the study.',
      caveat:
        'The compact sample changes support and roughness. It does not reproduce the published M3C2-EP propagation, and the retained epoch labels do not establish exact elapsed days; results are metres, not annual velocity.',
      options: {view: 'epochs', alignment: 'surveyed'},
      controls: ['view', 'alignment', 'pointSize'],
      readouts: ['validCores', 'invalidNormals']
    },
    {
      id: 'measure-along-normal',
      textAlternative:
        'Provided core markers use a fixed diverging scale for negative and positive 2018A-minus-2017 displacement; invalid normals and unsupported cores are neutral.',
      title: 'Measure along each supplied core normal',
      headline: 'A signed difference needs support in both epochs',
      body: '`GPUPointCloudChange` selects returns inside a cylinder around each provided core and normal, then compares the mean signed distances of the two epochs. The map uses a fixed ±1 m diverging scale; neutral marks include invalid normals and unsupported cores.',
      evidence:
        '**{{validCores}}** cores have measurements; **{{insufficientSupport}}** lack enough returns and **{{invalidNormals}}** have invalid supplied normals.',
      caveat:
        'Three provided normals contain non-finite values and are preserved as invalid. Sparse or incomplete support is not interpreted as zero movement.',
      options: {view: 'difference'},
      controls: ['view', 'radius', 'halfDepth'],
      readouts: ['validCores', 'insufficientSupport', 'invalidNormals', 'medianDifference'],
      stage: 'change'
    },
    {
      id: 'test-uncertainty',
      textAlternative:
        'Core colors indicate whether each GPU uncertainty interval includes zero or excludes it on the positive or negative side.',
      title: 'Ask whether the interval excludes zero',
      headline: 'A detection statement depends on the uncertainty model',
      body: 'The contributor reports signed difference, sampling roughness, epoch counts and a detection interval. Change the scalar registration allowance and choose how it combines with sampling uncertainty to see which intervals still exclude zero.',
      evidence:
        '**{{detectableCores}}** valid-core intervals exclude zero; median difference is **{{medianDifference}}** and median interval half-width is **{{medianLimit}}**.',
      caveat:
        'The scalar registration allowance is a sensitivity input, not the source’s complete alignment covariance or sensor error model.',
      options: {view: 'detection'},
      controls: ['view', 'registrationError', 'uncertaintyCombination'],
      readouts: ['detectableCores', 'medianDifference', 'medianLimit', 'intervalChart'],
      stage: 'change'
    },
    {
      id: 'local-fit-sensitivity',
      textAlternative:
        'The core map shows change after bounded exploratory ICP. Choose point-to-point or point-to-plane; the latter estimates reference-sample normals from capped 3 m neighborhoods. Charts retain residual, step and inlier history, including when the fit is rejected.',
      title: 'Refit a local crop as a sensitivity test',
      headline: 'A lower local residual can come from fitting away movement',
      body: 'The optional ICP fit uses no more than 1,024 rows per epoch and 12 iterations. Point-to-point uses nearest positions; point-to-plane estimates normals on the reference sample from 3 m neighborhoods, retaining at most 64 neighbors. It transforms the comparison sample before measuring the cores, and reports residual history, final inliers, translation step, and target-feature support.',
      evidence:
        'Local fit: **{{icpStatus}}**. Compare its core distribution with the supplied survey-frame result.',
      caveat:
        'The crop itself may have moved. A local fit can absorb real displacement into the registration correction, so its output is exploratory and is not evidence of improved alignment. Point-to-plane normals describe the bounded reference sample, not the provided core normals or full-resolution scanner surface; a 64-neighbor cap can affect them.',
      options: {alignment: 'local-icp'},
      controls: ['alignment', 'metric', 'robust', 'maximumDistance'],
      readouts: ['icpStatus', 'targetNormalsReady', 'icpChart', 'icpInlierChart'],
      stage: 'change'
    }
  ],
  pipeline: [
    {
      id: 'change',
      label: 'Measure core change',
      detail:
        'Apply cylindrical support, compute signed differences and uncertainty bounds; optionally refit a bounded local ICP transform.',
      show: {option: 'view', value: 'difference'}
    }
  ],
  legends: state =>
    state.view === 'epochs'
      ? [
          {
            id: 'epoch-returns',
            title: 'Measured epoch returns',
            kind: 'categories',
            entries: [
              {label: '2017 reference', color: [98, 164, 207]},
              {label: '2018A comparison', color: [225, 147, 112]},
              {label: 'Provided core locations', color: [244, 203, 122]}
            ]
          }
        ]
      : state.view === 'difference'
        ? [
            {
              id: 'point-cloud-change',
              title: '2018A − 2017 · metres',
              kind: 'ramp',
              ramp: 'vik',
              extent: [-1, 1],
              unit: 'm',
              note: 'Grey marks indicate invalid normals or insufficient support.'
            }
          ]
        : [
            {
              id: 'point-cloud-detection',
              title: 'Interval status',
              kind: 'categories',
              entries: [
                {label: 'Includes zero', color: [75, 83, 91]},
                {label: 'Excludes zero · positive', color: [113, 195, 157]},
                {label: 'Excludes zero · negative', color: [225, 133, 105]},
                {label: 'Invalid normal / insufficient support', color: [174, 180, 185]}
              ],
              note: 'Invalid normals and insufficient support use the neutral color.'
            }
          ],
  furniture: {title: true, credit: true, scaleBar: true},
  snippet: `graph.add(new GPUPointCloudChange({reference, comparison, cores, normals, alignment: 'already-aligned', radius: 1.5, halfDepth: 3, minimumSupport: 3, registrationError: 0.03, output: {changes, roughness, counts, status}}));`,
  about: {
    what: 'GPU M3C2-style signed mean differences inside cylinders oriented by the provided core normals, with an explicit uncertainty interval.',
    why: 'A displacement estimate is meaningful only alongside per-epoch support and an uncertainty model.',
    howToRead:
      'The map and summaries are computed from GPU outputs. Invalid normals and insufficient support remain separate statuses. The local ICP option is a sensitivity demonstration over the moving patch.'
  },
  create: async ctx => (await import('./point-cloud-change.compute')).createPointCloudChange(ctx)
});
