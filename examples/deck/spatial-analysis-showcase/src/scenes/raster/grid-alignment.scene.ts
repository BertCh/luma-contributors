// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {GridAlignmentOptions} from './grid-alignment.compute';

export default defineScene<GridAlignmentOptions>({
  id: 'grid-alignment',
  title: 'When the same pixels use a different grid',
  chapter: 'raster',
  order: 14,
  summary:
    'Use measured Sentinel-2 reflectance in a controlled shift experiment, then compare destination-grid resampling across UTM and a local Web Mercator frame. The imposed offsets are known experiments, not observed landscape motion.',
  contributors: ['GPURasterRegistration', 'GPURasterReprojection'],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Pinned July and September Sentinel-2 NIR reflectance, validity masks and projected source grid'
    }
  ],
  initialView: {longitude: -121.04153, latitude: 40.22095, zoom: 13.25},
  basemap: ground('relief'),
  annotations: [
    {kind: 'area', coordinate: [-121.04153, 40.22095], text: '64 × 64 source window', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'When the same pixels use a different grid',
      subtitle: 'Sentinel-2 L2A · 60 m UTM source · known-shift and reprojection experiments'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2021',
    caveat:
      'The correlation examples impose a known integer translation on measured July pixels. They test registration behavior; they are not a measured change between acquisitions or a ground displacement estimate. September nodata remains masked during reprojection.'
  },
  options: [
    {
      kind: 'select',
      id: 'product',
      label: 'Map layer',
      group: 'Display',
      apply: 'param',
      default: 'reference',
      help: 'Show the measured July reference, the selected controlled shift, its GPU-aligned result or difference, or an inverse-projection resample.',
      options: [
        {value: 'reference', label: 'Measured July reference'},
        {value: 'moving', label: 'Shifted experiment input'},
        {value: 'aligned', label: 'GPU-estimated alignment'},
        {value: 'difference', label: 'Aligned minus reference'},
        {value: 'reprojected', label: 'Reprojected destination grid'}
      ]
    },
    {
      kind: 'select',
      id: 'experiment',
      label: 'Shift model',
      group: 'Registration experiment',
      apply: 'compile',
      default: 'periodic',
      help: 'Periodic phase correlation treats opposite edges as neighbors. Masked linear correlation uses zero-padded support and rejects unsupported shifts.',
      options: [
        {value: 'periodic', label: 'Periodic phase correlation'},
        {value: 'masked-linear', label: 'Masked linear correlation'}
      ]
    },
    {
      kind: 'select',
      id: 'upsampleFactor',
      label: 'Subpixel search factor',
      group: 'Periodic settings',
      apply: 'compile',
      default: '20',
      help: 'Refines the periodic correlation peak at 1/factor-pixel increments. It is ignored in masked-linear mode.',
      options: [
        {value: '10', label: '1/10 pixel'},
        {value: '20', label: '1/20 pixel'},
        {value: '32', label: '1/32 pixel'}
      ]
    },
    {
      kind: 'select',
      id: 'ambiguityTolerance',
      label: 'Peak separation tolerance',
      group: 'Periodic settings',
      apply: 'compile',
      default: '0.01',
      help: 'A close nonadjacent peak marks the periodic estimate ambiguous. It is ignored in masked-linear mode.',
      options: [
        {value: '0.0001', label: 'Near-tie threshold · 0.01%'},
        {value: '0.01', label: 'Near-tie threshold · 1%'},
        {value: '0.05', label: 'Near-tie threshold · 5%'}
      ]
    },
    {
      kind: 'select',
      id: 'maximumShift',
      label: 'Maximum search shift',
      group: 'Masked settings',
      apply: 'compile',
      default: '8',
      help: 'Bounds the masked-linear integer search in both columns and rows. It is ignored in periodic mode.',
      options: [
        {value: '4', label: '±4 cells'},
        {value: '8', label: '±8 cells'},
        {value: '12', label: '±12 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'minimumOverlapFraction',
      label: 'Minimum overlap fraction',
      group: 'Masked settings',
      apply: 'compile',
      default: '0.7',
      help: 'Rejects masked-linear candidates with too little paired-valid area.',
      options: [
        {value: '0.5', label: '50%'},
        {value: '0.7', label: '70%'},
        {value: '0.85', label: '85%'}
      ]
    },
    {
      kind: 'select',
      id: 'minimumOverlapCount',
      label: 'Minimum paired cells',
      group: 'Masked settings',
      apply: 'compile',
      default: '1024',
      help: 'A second support floor in addition to the overlap fraction; it is ignored in periodic mode.',
      options: [
        {value: '512', label: '512 cells'},
        {value: '1024', label: '1,024 cells'},
        {value: '2048', label: '2,048 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'minimumCorrelation',
      label: 'Minimum correlation',
      group: 'Masked settings',
      apply: 'compile',
      default: '0',
      help: 'Rejects masked-linear peaks below the selected normalized-correlation score.',
      options: [
        {value: '-0.5', label: '−0.5'},
        {value: '0', label: '0'},
        {value: '0.25', label: '0.25'}
      ]
    },
    {
      kind: 'select',
      id: 'reprojectDate',
      label: 'Source acquisition',
      group: 'Reprojection',
      apply: 'param',
      default: 'after',
      help: 'Each date is a real Sentinel-2 NIR raster on the full 250 × 250 UTM tile; only the 64 × 64 destination window is drawn.',
      options: [
        {value: 'before', label: '13 Jul 2021'},
        {value: 'after', label: '21 Sep 2021'}
      ]
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Resampling kernel',
      group: 'Reprojection',
      apply: 'param',
      default: 'bilinear',
      help: 'Nearest selects one source cell. Bilinear blends the four spatial neighbors at the inverse-projected destination center.',
      options: [
        {value: 'nearest', label: 'Nearest neighbor'},
        {value: 'bilinear', label: 'Bilinear'}
      ]
    },
    {
      kind: 'select',
      id: 'noDataPolicy',
      label: 'Bilinear nodata policy',
      group: 'Reprojection',
      apply: 'param',
      default: 'strict',
      help: 'Strict invalidates a pixel if a contributing neighbor is missing. Renormalize rescales weights over valid neighbors. Nearest has no multi-cell support, so these policies coincide.',
      options: [
        {value: 'strict', label: 'Strict support'},
        {value: 'renormalize', label: 'Renormalize valid neighbors'}
      ]
    }
  ],
  readouts: [
    {id: 'registrationResult', label: 'GPU registration result'},
    {id: 'alignmentDifference', label: 'GPU-aligned difference summary'},
    {id: 'selectedPixel', label: 'Selected measured pixel'},
    {id: 'shiftChart', label: 'Imposed and estimated shift', kind: 'chart'},
    {id: 'reprojectionSupport', label: 'Reprojection support'},
    {id: 'supportChart', label: 'Valid destination cells by method', kind: 'chart'}
  ],
  legends: state =>
    state.product === 'difference'
      ? [
          {
            kind: 'ramp',
            title: 'Aligned − reference NIR',
            ramp: 'vik',
            extent: [-0.3, 0.3],
            midpoint: 0,
            unit: 'reflectance'
          }
        ]
      : [
          {
            kind: 'ramp',
            title: 'Sentinel-2 NIR surface reflectance',
            ramp: 'cividis',
            extent: [0, 0.5],
            unit: 'unitless reflectance'
          }
        ],
  story: [
    {
      id: 'source-grid',
      headline: 'One observation already has a measured grid',
      textAlternative:
        'The map begins with a bounded 64 by 64 crop of July Sentinel-2 NIR reflectance.',
      title: 'Start from one source raster',
      body: 'This 64 × 64 window comes from the pinned July 13 Dixie Sentinel-2 scene. Its 60 m UTM grid and strict validity are preserved; the outlined footprint and selected cell locate the measured window on the map. The full 250 × 250 tile remains available as the interpolation source, so the displayed crop does not truncate neighboring support.',
      evidence: '**{{selectedPixel}}**',
      caveat: 'This is a single optical acquisition. No displacement has been measured yet.',
      options: {product: 'reference', experiment: 'periodic'},
      controls: ['product'],
      readouts: ['selectedPixel']
    },
    {
      id: 'periodic-registration',
      headline: 'Periodic correlation joins opposite edges',
      textAlternative:
        'A deliberate wrapped translation is estimated from the measured NIR crop; correction is compared with the known offset.',
      title: 'Test the wrapped model',
      body: 'The moving image is a known integer shift of real measured reflectance, with values wrapped across opposite crop edges. The GPU returns a correction that is applied with periodic bilinear sampling, preserving its subpixel estimate. Change the search factor and ambiguity tolerance to see how the estimate is refined or rejected.',
      evidence: '**{{registrationResult}}**',
      caveat: 'A periodic border is a mathematical assumption, not a property of the landscape.',
      options: {experiment: 'periodic', product: 'moving'},
      controls: ['experiment', 'product', 'upsampleFactor', 'ambiguityTolerance'],
      readouts: ['registrationResult', 'shiftChart']
    },
    {
      id: 'masked-registration',
      headline: 'Masked correlation stops wrapping the crop',
      textAlternative:
        'Zero-padded normalized correlation searches only a bounded translation and reports paired-valid support.',
      title: 'Respect the window boundary',
      body: 'The second experiment applies a known nonwrapped shift and marks the uncovered edge as invalid. The masked-linear mode searches a finite range, requires enough paired cells and reports accepted overlap. Vary its bounds and support thresholds to inspect which estimates remain admissible.',
      evidence: '**{{registrationResult}}**',
      caveat:
        'This is still a controlled transform of one image; it does not estimate real fire motion or sensor displacement.',
      options: {experiment: 'masked-linear', product: 'difference'},
      controls: [
        'experiment',
        'maximumShift',
        'minimumOverlapFraction',
        'minimumOverlapCount',
        'minimumCorrelation'
      ],
      readouts: ['registrationResult', 'alignmentDifference', 'shiftChart']
    },
    {
      id: 'inverse-projection',
      headline: 'A destination grid asks new source locations',
      textAlternative:
        'A local Web Mercator destination grid is inverse-projected into the full UTM tile before sampling.',
      title: 'Change the grid through coordinates',
      body: 'The destination centers use an explicit spherical Web Mercator grid scaled to metres at the source-crop center. An adaptive projection fitted to double precision coordinates maps each center into the source UTM 10N grid. Sampling keeps coordinates relative to the source tile, avoiding precision loss from large absolute eastings and northings. The outlined destination footprint and selected destination cell use a separate display transform; the sampler reads the complete source tile, including interpolation support.',
      evidence: '**{{reprojectionSupport}}**',
      caveat:
        'The adaptive fit checks a 0.001 m tolerance at its validation samples; this is not a proof of accuracy everywhere. The display frame and projection domain are bounded to this landscape window.',
      options: {
        product: 'reprojected',
        reprojectDate: 'after',
        method: 'bilinear',
        noDataPolicy: 'strict'
      },
      controls: ['product', 'reprojectDate', 'method'],
      readouts: ['reprojectionSupport', 'supportChart']
    },
    {
      id: 'nodata-support',
      headline: 'Missing neighbors change what bilinear means',
      textAlternative:
        'Strict and renormalized bilinear sampling are compared over real post-fire validity masks.',
      title: 'Keep missing support visible',
      body: 'The September image has fewer valid NIR cells than July. Strict bilinear sampling requires every contributing neighbor. Renormalization rescales only the valid neighbor weights. Compare their support counts, then switch to nearest neighbor, where a single-cell support makes the two policies coincide.',
      evidence: '**{{reprojectionSupport}}**',
      caveat:
        'A larger valid count is not automatically more accurate; it reflects a different rule for partial neighborhoods.',
      options: {
        product: 'reprojected',
        reprojectDate: 'after',
        method: 'bilinear',
        noDataPolicy: 'renormalize'
      },
      controls: ['reprojectDate', 'method', 'noDataPolicy'],
      readouts: ['reprojectionSupport', 'supportChart']
    }
  ],
  create: async ctx => (await import('./grid-alignment.compute')).createGridAlignment(ctx)
});
