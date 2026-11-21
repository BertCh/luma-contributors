// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SpectralTextureOptions} from './spectral-texture.compute';
import {
  getSpectralLayerStyle,
  getSpectralTextureRange,
  type FocalLayer,
  SPECTRAL_STATUS_PALETTE
} from './spectral-texture-style';

const FOCAL_OPTIONS: readonly {value: FocalLayer; label: string}[] = [
  {value: 'covariance', label: 'Population or sample covariance'},
  {value: 'correlation', label: 'Correlation'},
  {value: 'slope', label: 'OLS slope'},
  {value: 'intercept', label: 'OLS intercept'},
  {value: 'rSquared', label: 'R²'},
  {value: 'residualSumSquares', label: 'Residual sum of squares'},
  {value: 'pairedCount', label: 'Paired count'},
  {value: 'status', label: 'Fit status bit mask'}
];

export default defineScene<SpectralTextureOptions>({
  id: 'spectral-texture',
  title: 'Which patterns survive a change of scale?',
  chapter: 'raster',
  order: 8,
  summary:
    'Calibrated Sentinel-2 reflectance shows how smoothing, edges, local texture and paired-band statistics describe different parts of an observed image.',
  contributors: [
    'GPURasterNeighborhood',
    'GPURasterConvolution',
    'GPURasterBoxBlur',
    'GPURasterGaussianBlur',
    'GPURasterGradient',
    'GPURasterSobel',
    'GPURasterScharr',
    'GPURasterGradientMagnitude',
    'GPURasterLaplacian',
    'GPURasterTexture',
    'GPURasterFocalPairs',
    'GPURasterContrast',
    'GPURasterHistogram',
    'GPURasterProfile',
    'GPURasterBufferToTexture',
    'GPURasterTextureToBuffer'
  ],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Calibrated Sentinel-2 reflectance and strict native-pixel QA masks'
    }
  ],
  initialView: {longitude: -121.0005443654, latitude: 40.175922885, zoom: 12.2, pitch: 0},
  basemap: ground('relief'),
  annotations: [
    {
      kind: 'point',
      coordinate: [-120.9511, 40.1396],
      text: 'Greenville',
      rank: 'subject',
      anchor: 'auto'
    }
  ],
  furniture: {
    title: {
      title: 'Dixie Fire spectral texture',
      subtitle: '60 m Sentinel-2 band means near Greenville'
    },
    scaleBar: {units: 'metric'},
    credit:
      'Contains modified Copernicus Sentinel data 2021 · Sentinel-2 L2A via AWS Open Data Registry',
    caveat:
      'July 13 is ignition day. Strict QA reduces uncertainty but cannot guarantee cloud-free observations.'
  },
  options: [
    {
      kind: 'select',
      id: 'date',
      label: 'Acquisition',
      group: 'Source',
      apply: 'compile',
      default: 'before',
      options: [
        {value: 'before', label: '2021-07-13 · ignition day'},
        {value: 'after', label: '2021-09-21 · later scene'}
      ],
      help: 'Rebuilds every mapped and clicked GPU result from the selected acquisition.'
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Raster evidence',
      apply: 'param',
      default: 'source',
      options: [
        {value: 'source', label: 'NIR reflectance'},
        {value: 'display', label: 'Transformed display values'},
        {value: 'transport', label: 'Texture-returned NIR'},
        {value: 'smooth', label: 'Gaussian smooth'},
        {value: 'box', label: 'Box smooth'},
        {value: 'weighted', label: 'Weighted neighborhood'},
        {value: 'convolution', label: 'Signed convolution'},
        {value: 'gradient', label: 'Gradient magnitude'},
        {value: 'gradientX', label: 'Selected-operator x response'},
        {value: 'gradientY', label: 'Selected-operator y response'},
        {value: 'sobelX', label: 'Sobel x response'},
        {value: 'sobelY', label: 'Sobel y response'},
        {value: 'scharrX', label: 'Scharr x response'},
        {value: 'scharrY', label: 'Scharr y response'},
        {value: 'laplacian', label: 'Laplacian'},
        {value: 'texture', label: 'Selected GLCM feature'},
        {value: 'focal', label: 'Focal-pairs output'}
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Neighborhood radius',
      group: 'Filter and pair support',
      apply: 'compile',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      unit: 'pixels',
      help: 'Sets the square smoothing and focal-pairs support window; its width is 2r + 1 cells.'
    },
    {
      kind: 'slider',
      id: 'sigma',
      label: 'Gaussian sigma',
      group: 'Filter and pair support',
      apply: 'compile',
      min: 0.5,
      max: 4,
      step: 0.5,
      default: 1.5,
      unit: 'pixels',
      disabledWhen: state => state.layer !== 'smooth',
      help: 'Gaussian standard deviation in source pixels.'
    },
    {
      kind: 'select',
      id: 'borderMode',
      label: 'Neighborhood border',
      group: 'Filter and pair support',
      apply: 'compile',
      default: 'nodata',
      options: [
        {value: 'clamp', label: 'Clamp edge'},
        {value: 'reflect', label: 'Reflect edge'},
        {value: 'nodata', label: 'Outside is nodata'}
      ],
      help: 'All signed derivatives keep strict nodata propagation regardless of this smoothing policy.'
    },
    {
      kind: 'select',
      id: 'noDataPolicy',
      label: 'Positive-kernel missing data',
      group: 'Filter and pair support',
      apply: 'compile',
      default: 'ignore-renormalize',
      options: [
        {value: 'ignore-renormalize', label: 'Ignore and renormalize'},
        {value: 'propagate', label: 'Propagate'}
      ]
    },
    {
      kind: 'select',
      id: 'gradientOperator',
      label: 'Gradient recipe',
      group: 'Edges',
      apply: 'compile',
      default: 'scharr',
      options: [
        {value: 'sobel', label: 'Sobel'},
        {value: 'scharr', label: 'Scharr'}
      ],
      help: 'The selected x/y and magnitude outputs use this recipe; named Sobel and Scharr x/y outputs are also computed.'
    },
    {
      kind: 'select',
      id: 'convolutionKernel',
      label: 'Signed convolution kernel',
      group: 'Edges',
      apply: 'compile',
      default: 'sobel-x',
      disabledWhen: state => state.layer !== 'convolution',
      options: [
        {value: 'sobel-x', label: '3 × 3 Sobel x'},
        {value: 'east-derivative', label: '5 × 1 eastward derivative'},
        {value: 'south-derivative', label: '1 × 5 southward derivative'}
      ],
      help: 'Rectangular kernels use explicit dimensions and remain signed, unnormalized responses.'
    },
    {
      kind: 'select',
      id: 'convolutionBorder',
      label: 'Convolution outside crop',
      group: 'Edges',
      apply: 'compile',
      default: 'inherit',
      disabledWhen: state => state.layer !== 'convolution',
      options: [
        {value: 'inherit', label: 'Use neighborhood border'},
        {value: 'constant', label: 'Constant extrapolation'}
      ],
      help: 'Constant values are hypothetical extrapolation outside this clipped crop, not observed pixels.'
    },
    {
      kind: 'slider',
      id: 'convolutionBorderValue',
      label: 'Constant outside value',
      group: 'Edges',
      apply: 'compile',
      min: 0,
      max: 0.6,
      step: 0.01,
      default: 0,
      unit: 'reflectance',
      disabledWhen: state =>
        state.layer !== 'convolution' || state.convolutionBorder !== 'constant',
      help: 'Used only with constant extrapolation. Signed convolution propagates nodata and does not normalize.'
    },
    {
      kind: 'select',
      id: 'laplacianConnectivity',
      label: 'Laplacian neighbors',
      group: 'Edges',
      apply: 'compile',
      default: '4',
      options: [
        {value: '4', label: 'Four-connected'},
        {value: '8', label: 'Eight-connected'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Texture feature',
      group: 'Local GLCM',
      apply: 'param',
      default: 'contrast',
      disabledWhen: state => state.layer !== 'texture',
      options: [
        {value: 'contrast', label: 'Contrast'},
        {value: 'dissimilarity', label: 'Dissimilarity'},
        {value: 'homogeneity', label: 'Homogeneity'},
        {value: 'ASM', label: 'Angular second moment'},
        {value: 'energy', label: 'Energy'},
        {value: 'correlation', label: 'Correlation'},
        {value: 'entropy', label: 'Natural-log entropy'}
      ]
    },
    {
      kind: 'slider',
      id: 'textureRadius',
      label: 'GLCM window radius',
      group: 'Local GLCM',
      apply: 'compile',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      unit: 'pixels',
      disabledWhen: state => state.layer !== 'texture'
    },
    {
      kind: 'select',
      id: 'textureDirection',
      label: 'Pair direction',
      group: 'Local GLCM',
      apply: 'compile',
      default: 'E',
      disabledWhen: state => state.layer !== 'texture',
      options: [
        {value: 'E', label: 'East'},
        {value: 'N', label: 'North'},
        {value: 'NE', label: 'Northeast'},
        {value: 'NW', label: 'Northwest'}
      ]
    },
    {
      kind: 'select',
      id: 'textureLevels',
      label: 'Quantization levels',
      group: 'Local GLCM',
      apply: 'compile',
      default: '8',
      disabledWhen: state => state.layer !== 'texture',
      options: [
        {value: '8', label: '8 levels'},
        {value: '16', label: '16 levels'}
      ],
      help: 'Values are clipped and quantized over finite reflectance range [0, 0.6]; this changes metric scale.'
    },
    {
      kind: 'select',
      id: 'textureSymmetry',
      label: 'Count reverse pairs',
      group: 'Local GLCM',
      apply: 'compile',
      default: 'off',
      disabledWhen: state => state.layer !== 'texture',
      options: [
        {value: 'off', label: 'No · directed'},
        {value: 'on', label: 'Yes · symmetric'}
      ]
    },
    {
      kind: 'select',
      id: 'textureDisplayDomain',
      label: 'Texture color range',
      group: 'Local GLCM',
      apply: 'param',
      default: 'detail',
      disabledWhen: state => state.layer !== 'texture',
      options: [
        {value: 'detail', label: 'Detail · display clipping'},
        {value: 'theoretical', label: 'Theoretical metric range'}
      ],
      help: 'Detail clips only the color scale (contrast 0–4, dissimilarity 0–2); it does not alter measured values.'
    },
    {
      kind: 'select',
      id: 'focalLayer',
      label: 'Focal-pairs output',
      group: 'Paired bands',
      apply: 'param',
      default: 'slope',
      disabledWhen: state => state.layer !== 'focal',
      options: FOCAL_OPTIONS
    },
    {
      kind: 'select',
      id: 'covariance',
      label: 'Covariance divisor',
      group: 'Paired bands',
      apply: 'compile',
      default: 'sample',
      disabledWhen: state => state.layer !== 'focal',
      options: [
        {value: 'sample', label: 'Sample · n − 1'},
        {value: 'population', label: 'Population · n'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumCount',
      label: 'Minimum paired count',
      group: 'Paired bands',
      apply: 'compile',
      min: 2,
      max: 169,
      step: 1,
      default: 8,
      disabledWhen: state => state.layer !== 'focal'
    },
    {
      kind: 'select',
      id: 'focalNoData',
      label: 'Focal missing pairs',
      group: 'Paired bands',
      apply: 'compile',
      default: 'ignore',
      disabledWhen: state => state.layer !== 'focal',
      options: [
        {value: 'ignore', label: 'Ignore incomplete pairs'},
        {value: 'propagate', label: 'Propagate any missing pair'}
      ]
    },
    {
      kind: 'toggle',
      id: 'centerPropagation',
      label: 'Require valid center pair',
      group: 'Paired bands',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.layer !== 'focal',
      help: 'When enabled, the center needs both NIR and SWIR₂ values.'
    },
    {
      kind: 'select',
      id: 'displayMode',
      label: 'Display transform',
      group: 'Display-only view',
      apply: 'compile',
      default: 'linear',
      options: [
        {value: 'linear', label: 'Linear midpoint contrast'},
        {value: 'gamma', label: 'Gamma curve'},
        {value: 'equalize', label: 'GPU histogram CDF equalization'}
      ],
      help: 'Transforms valid NIR values for display while retaining the reflectance scalar domain; it does not revise observations or scientific texture/focal outputs.'
    },
    {
      kind: 'select',
      id: 'displayDomain',
      label: 'Transform domain',
      group: 'Display-only view',
      apply: 'compile',
      default: 'calibrated',
      options: [
        {value: 'calibrated', label: 'Literal calibrated [0, 0.6]'},
        {value: 'gpu', label: 'GPU-resident valid min/max'}
      ],
      help: 'Uses the selected acquisition validity mask to choose a literal reflectance domain or a GPU-reduced valid min/max.'
    },
    {
      kind: 'select',
      id: 'displayContrast',
      label: 'Contrast strength',
      group: 'Display-only view',
      apply: 'compile',
      default: '1',
      options: [
        {value: '0.5', label: '0.5'},
        {value: '1', label: '1'},
        {value: '2', label: '2'}
      ],
      help: 'Scales distance from the normalized midpoint before mapping back into the chosen scalar domain.'
    },
    {
      kind: 'select',
      id: 'displayGamma',
      label: 'Gamma',
      group: 'Display-only view',
      apply: 'compile',
      default: '1',
      options: [
        {value: '0.5', label: '0.5'},
        {value: '1', label: '1'},
        {value: '2', label: '2'}
      ],
      disabledWhen: state => state.displayMode === 'linear',
      help: 'Applies pow(normalized, 1 / gamma), including after the histogram CDF in equalization mode. This display curve never becomes measured reflectance.'
    },
    {
      kind: 'select',
      id: 'displayBins',
      label: 'Equalization bins',
      group: 'Display-only view',
      apply: 'compile',
      default: '64',
      options: [
        {value: '64', label: '64 bins'},
        {value: '256', label: '256 bins'}
      ],
      disabledWhen: state => state.displayMode !== 'equalize',
      help: 'The contributor uses GPU histogram counts from this exact acquisition and selected domain.'
    },
    {
      kind: 'select',
      id: 'profileSpacing',
      label: 'Transect spacing',
      group: 'Transect',
      apply: 'compile',
      default: '120',
      options: [
        {value: '60', label: '60 m'},
        {value: '120', label: '120 m'},
        {value: '240', label: '240 m'}
      ],
      help: 'Samples along the fixed two-segment line, with an endpoint sample exactly at its total length.'
    },
    {
      kind: 'select',
      id: 'profileMethod',
      label: 'Profile interpolation',
      group: 'Transect',
      apply: 'compile',
      default: 'nearest',
      options: [
        {value: 'nearest', label: 'Nearest'},
        {value: 'bilinear', label: 'Bilinear'}
      ],
      help: 'Uses the GPU raster profile sampler against row-zero-north calibrated NIR.'
    },
    {
      kind: 'select',
      id: 'profileNoData',
      label: 'Profile QA gaps',
      group: 'Transect',
      apply: 'compile',
      default: 'propagate',
      options: [
        {value: 'propagate', label: 'Strict support'},
        {value: 'renormalize', label: 'Renormalize valid neighbors'}
      ],
      disabledWhen: state => state.profileMethod === 'nearest',
      help: 'For bilinear interpolation, `propagate` requires complete support and renormalize uses valid neighbors only.'
    },
    {
      kind: 'select',
      id: 'transportChannel',
      label: 'Texture channel',
      group: 'Transport inspection',
      apply: 'compile',
      default: 'R',
      options: [
        {value: 'R', label: 'Red'},
        {value: 'G', label: 'Green'},
        {value: 'B', label: 'Blue'},
        {value: 'A', label: 'Alpha'}
      ],
      help: 'Writes the source float32 NIR band to this rgba32float channel and gathers it back with both legs uncalibrated.'
    }
  ],
  story: [
    {
      id: 'observation',
      title: 'Distinguish an observation from its display',
      headline: 'Brightness can change while the measured band stays fixed',
      textAlternative: 'A correctly located 60 metre NIR reflectance crop with strict native QA.',
      body: 'Begin with calibrated Sentinel-2 NIR means on a 60 m north-first UTM grid. Map shows can switch to transformed display values: linear contrast, gamma (exponent 1/gamma), or histogram equalization change appearance while the measured band remains the input to later analyses. The two histograms show that distinction. July 13 is ignition day; strict QA does not prove cloud-free support.',
      evidence: '{{sourceSupport}}',
      caveat:
        'Dates differ in season and valid support, so their means do not estimate fire impact.',
      options: {date: 'before', layer: 'source'},
      controls: ['date', 'layer', 'displayMode', 'displayContrast', 'displayGamma'],
      readouts: ['sourceSupport', 'selectedEvidence', 'displayEvidence', 'displayChart']
    },
    {
      id: 'scale',
      title: 'Change the neighborhood scale',
      headline: 'Smoothing trades pixel detail for local structure',
      textAlternative:
        'Box, weighted and Gaussian neighborhoods are GPU filtered at a selectable radius.',
      body: 'Compare Gaussian, box and weighted surfaces as the radius changes. Positive filters either require every contributing sample or ignore missing samples and renormalize their nonnegative weights. The selected-cell readout reports the actual source-minus-Gaussian difference.',
      evidence: '{{selectedEvidence}}',
      caveat: 'Border handling is explicit; no filter recovers context beyond this crop.',
      options: {layer: 'smooth'},
      controls: ['layer', 'radius', 'sigma', 'borderMode', 'noDataPolicy'],
      readouts: ['selectedEvidence']
    },
    {
      id: 'edges',
      title: 'Inspect image-space edge response',
      headline: 'Oriented derivatives retain their sign',
      textAlternative:
        'Signed x and y Sobel and Scharr outputs and gradient magnitude respond to reflectance transitions.',
      body: 'Raster x increases eastward and raster y increases southward because row zero is north. Derivatives are signed pixel responses; gradient magnitude is nonnegative. The convolution recipe selects 3 × 3 Sobel x, 5 × 1 eastward, or 1 × 5 southward kernels. Constant outside values are hypothetical extrapolation beyond the crop, while missing source samples still propagate through signed kernels.',
      evidence: '{{selectedEvidence}}',
      caveat:
        'Constant extrapolation is an assumed boundary value, not an observed reflectance measurement; signed kernels propagate missing data and are never renormalized.',
      options: {layer: 'gradient'},
      controls: [
        'layer',
        'gradientOperator',
        'convolutionKernel',
        'convolutionBorder',
        'convolutionBorderValue'
      ],
      readouts: ['selectedEvidence']
    },
    {
      id: 'texture',
      title: 'Set the texture measurement',
      headline: 'A GLCM value depends on its bins and direction',
      textAlternative:
        'A selected local GLCM feature map reports metric values and observed pair count.',
      body: 'Select a feature, window radius, direction, quantization level and symmetry. Both endpoints must be valid and inside the clipped window. Finite reflectance is clipped to [0, 0.6] before quantization. Detail color range clips contrast at 4 and dissimilarity at 2 for display; choose theoretical range to see their full possible domains. Neither display range changes measured texture values.',
      evidence: '{{textureEvidence}}',
      caveat: 'Overlapping windows share samples and are not independent tests.',
      options: {layer: 'texture'},
      controls: [
        'metric',
        'textureRadius',
        'textureDirection',
        'textureLevels',
        'textureDisplayDomain'
      ],
      readouts: ['textureEvidence']
    },
    {
      id: 'paired',
      title: 'Compare local NIR and SWIR₂ values',
      headline: 'Paired support shapes a local fit',
      textAlternative: 'Actual source pairs and the GPU OLS fit describe a selected neighborhood.',
      body: 'The scatter uses real jointly valid NIR/SWIR₂ pairs from the same clipped window as the GPU calculation. Explore covariance normalization, minimum support and missing-pair handling, then map each computed statistic. Map paired counts or fit status to locate unsupported windows. The exact status bit mask distinguishes insufficient support, missing pairs, constant bands and non-finite results; crop-edge and QA gaps remain part of the result.',
      evidence: '{{focalEvidence}}',
      caveat: 'This local association is descriptive and supplies no inferential p-value.',
      options: {layer: 'focal'},
      controls: ['focalLayer', 'radius', 'covariance', 'minimumCount', 'focalNoData'],
      readouts: ['focalEvidence', 'pairChart']
    },
    {
      id: 'transect',
      title: 'Follow valid measurements across the crop',
      headline: 'A sampled line makes gaps and local values explicit',
      textAlternative: 'A two-segment map transect is sampled on the GPU and charted by distance.',
      body: 'The cyan line follows three fixed vertices across the north-first crop. **Transect spacing**, **Profile interpolation** and **Profile QA gaps** below control the GPU profile sampler; the chart reports sampled NIR by planar distance and the readout shows emitted and required counts, overflow and finite range. **Texture channel** changes the independent GPU transport roundtrip.',
      evidence: '{{profileEvidence}}',
      caveat:
        'Reflectance increments are not elevation gain or loss; the line samples this acquisition and its QA support only.',
      options: {layer: 'source', profileSpacing: '120', profileMethod: 'bilinear'},
      controls: ['profileSpacing', 'profileMethod', 'profileNoData', 'layer', 'transportChannel'],
      readouts: ['profileEvidence', 'profileChart', 'transportEvidence']
    }
  ],
  legends(state, data) {
    const layer = state.layer === 'focal' ? state.focalLayer : state.layer;
    if (state.layer === 'focal' && state.focalLayer === 'status') {
      return [
        {
          kind: 'categories',
          title: 'Focal fit status class',
          entries: [
            {color: SPECTRAL_STATUS_PALETTE[0], label: 'Supported'},
            {color: SPECTRAL_STATUS_PALETTE[1], label: 'Insufficient pairs (bit 1)'},
            {color: SPECTRAL_STATUS_PALETTE[2], label: 'Missing pair (bit 2)'},
            {color: SPECTRAL_STATUS_PALETTE[3], label: 'Constant predictor (bit 4)'},
            {color: SPECTRAL_STATUS_PALETTE[4], label: 'Constant response (bit 8)'},
            {color: SPECTRAL_STATUS_PALETTE[5], label: 'Non-finite result (bit 16)'},
            {color: SPECTRAL_STATUS_PALETTE[6], label: 'Combined status flags'}
          ],
          note: 'The map groups exact GPU status masks into these classes; the clicked readout reports the original bit mask.'
        }
      ];
    }
    if (layer === 'display') {
      const recordedDisplayDomain = data.displayDomain;
      const displayExtent =
        state.displayDomain === 'gpu' &&
        Array.isArray(recordedDisplayDomain) &&
        recordedDisplayDomain.length === 2 &&
        Number.isFinite(Number(recordedDisplayDomain[0])) &&
        Number.isFinite(Number(recordedDisplayDomain[1]))
          ? ([Number(recordedDisplayDomain[0]), Number(recordedDisplayDomain[1])] as const)
          : ([0, 0.6] as const);
      return [
        {
          kind: 'ramp',
          title: 'Transformed display values · reflectance-domain units',
          ramp: 'cividis',
          extent: displayExtent,
          unit: 'display reflectance'
        }
      ];
    }
    if (layer === 'transport') {
      return [
        {
          kind: 'ramp',
          title: 'Texture-returned NIR · calibrated source values',
          ramp: 'ylorrd',
          extent: [0, 0.6],
          unit: 'reflectance'
        }
      ];
    }
    const style =
      layer === 'texture'
        ? getSpectralTextureRange(
            state.metric,
            Number(state.textureLevels),
            state.textureDisplayDomain
          )
        : getSpectralLayerStyle(layer, Number(state.radius));
    const title =
      layer === 'texture'
        ? `${state.metric} · ${state.textureLevels} levels · ${state.textureDisplayDomain} color range`
        : state.layer === 'focal'
          ? `Focal ${state.focalLayer}`
          : `${layer} · ${style.unit}`;
    return [
      {
        kind: 'ramp',
        title,
        ramp: style.ramp === 'category' ? 'ylorrd' : style.ramp,
        extent: style.extent,
        unit: style.unit
      }
    ];
  },
  readouts: [
    {id: 'sourceSupport', label: 'Strict QA source support'},
    {id: 'selectedEvidence', label: 'Selected output and source difference'},
    {id: 'textureEvidence', label: 'Selected texture and pair count'},
    {id: 'focalEvidence', label: 'Covariance, fit and support'},
    {id: 'pairChart', label: 'Selected paired reflectance samples'},
    {id: 'displayEvidence', label: 'Measured and transformed probe values'},
    {id: 'displayChart', label: 'Measured and transformed distributions', kind: 'chart'},
    {id: 'profileEvidence', label: 'Transect sample count and range'},
    {id: 'profileChart', label: 'Measured NIR along transect', kind: 'chart'},
    {id: 'transportEvidence', label: 'GPU texture roundtrip error and validity'}
  ],
  about: {
    what: 'The graph computes Gaussian, box and weighted smoothing; signed convolution; generic, named Sobel and named Scharr x/y derivatives; Sobel or Scharr gradient magnitude; selectable four/eight-neighbor Laplacian; all seven local GLCM features; focal-pairs statistics; display-only contrast transforms; a GPU raster transect; and a float32 buffer/texture/buffer roundtrip.',
    why: 'The inquiry follows observed reflectance through spatial scale, oriented change, quantized co-occurrence, paired-band association and uneven support.',
    howToRead:
      'The crop contains aggregated 60 m means of native Sentinel-2 10 m and 20 m bands, in unitless reflectance; QA-invalid cells remain nodata. July 13, 2021 is ignition day, not a pre-fire baseline, and strict QA does not remove all observation uncertainty. Derivative x is eastward and y southward in raster coordinates, with signed response per pixel; positive x response marks an eastward increase. The Laplacian is a second difference per pixel squared. The selectable 3 × 3 Sobel x and 5 × 1 eastward / 1 × 5 southward convolution kernels are signed, unnormalized and propagate nodata. Constant convolution borders use a hypothetical reflectance beyond the crop, not an observation. Contrast and gamma apply display transforms in the original calibrated domain; gamma uses pow(normalized, 1 / gamma), and equalization uses this acquisition’s GPU histogram counts. Histogram equalization follows [scikit-image exposure documentation](https://scikit-image.org/docs/stable/api/skimage.exposure.html) as a distribution-based display operation, not observation recovery. These transformed values never enter the GLCM or focal-pairs paths. The fixed transect uses the GPU profile contributor and reports reflectance samples by distance; reflectance change is not elevation ascent. The transport roundtrip uses rgba32float with the selected channel and applyCalibration false on both legs; no synthetic/native DN claim is made. Focal scatter points are bounded CPU source evidence, while the plotted fit is the GPU result. Sentinel-2 L2A source scenes: [July 13](https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/T/FK/2021/7/S2B_10TFK_20210713_0_L2A/) and [September 21](https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/T/FK/2021/9/S2B_10TFK_20210921_0_L2A/). GLCM definitions follow [scikit-image feature documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html).'
  },
  create: async ctx => (await import('./spectral-texture.compute')).createSpectralTexture(ctx)
});
