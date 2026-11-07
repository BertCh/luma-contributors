// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MissingObservationsOptions} from './missing-observations.compute';

export default defineScene<MissingObservationsOptions>({
  id: 'missing-observations',
  title: 'What can a missing pixel tell us?',
  chapter: 'raster',
  order: 15,
  summary:
    'Compare bounded GPU interpolation against deliberately withheld Sentinel-2 NIR values, then keep real September gaps separate from grayscale morphology, geodesic reconstruction and reflectance contours.',
  contributors: [
    'GPURasterFillNoData',
    'GPURasterFillHoles',
    'GPURasterMorphology',
    'GPURasterOpening',
    'GPURasterClosing',
    'GPURasterReconstruction',
    'GPURasterContourClassifier',
    'GPURasterContours'
  ],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Sentinel-2 L2A NIR reflectance and date-specific validity masks'
    }
  ],
  initialView: {longitude: -120.9937, latitude: 40.1931, zoom: 12.25},
  basemap: ground('relief'),
  annotations: [
    {kind: 'point', coordinate: [-120.95, 40.14], text: 'Greenville', rank: 'context'},
    {kind: 'area', coordinate: [-120.9937, 40.1931], text: '64 × 64 source window', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'What can a missing pixel tell us?',
      subtitle: 'Dixie landscape · Sentinel-2 L2A NIR · 60 m cells'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2021',
    caveat:
      'July holdouts have known measured truth; September gaps do not. The crop uses the supplied per-date validity masks. NIR isolines are spectral contours, not fire perimeters.'
  },
  options: [
    {
      kind: 'select',
      id: 'experiment',
      label: 'Observation set',
      group: 'Measured support',
      apply: 'compile',
      default: 'held-out',
      help: 'Use July values with artificial holdouts and known truth, or the actual September mask whose missing values have no known truth.',
      options: [
        {value: 'held-out', label: 'July · withheld truth'},
        {value: 'september-gaps', label: 'September · real gaps'}
      ]
    },
    {
      kind: 'select',
      id: 'holdoutPattern',
      label: 'Withheld pattern',
      group: 'Known-truth experiment',
      apply: 'compile',
      default: 'block',
      help: 'Remove either a deterministic set of isolated July cells or a contiguous 9 × 9 central block; compare fills only where those observed values were hidden.',
      options: [
        {value: 'isolated', label: 'Isolated cells'},
        {value: 'block', label: '9 × 9 block'}
      ]
    },
    {
      kind: 'select',
      id: 'mapLayer',
      label: 'Map shows',
      group: 'Evidence layer',
      apply: 'param',
      default: 'mask',
      help: 'Switch between observations and the exact artificial or real gaps, filled values, interpolation footprint, withheld-cell residuals, grayscale structure, reconstruction or reflectance contours.',
      options: [
        {value: 'source', label: 'Observed NIR'},
        {value: 'mask', label: 'Missing-cell mask'},
        {value: 'fill', label: 'Filled NIR surface'},
        {value: 'prepared', label: 'Prepared NIR surface'},
        {value: 'filled-cells', label: 'Cells interpolated'},
        {value: 'error', label: 'Withheld-cell error'},
        {value: 'morphology', label: 'Grayscale morphology'},
        {value: 'reconstruction', label: 'Geodesic reconstruction'},
        {value: 'topology-fill', label: 'Enclosed gaps in support mask'},
        {value: 'contours', label: 'NIR isolines'}
      ]
    },
    {
      kind: 'select',
      id: 'fillMethod',
      label: 'Donor rule',
      group: 'Bounded interpolation',
      apply: 'param',
      default: 'gdal-four-quadrant',
      help: 'Select the nearest original sample, all original donors with inverse-distance weights, or the closest donor in each quadrant. Filled cells never become donors.',
      options: [
        {value: 'nearest', label: 'Nearest original donor'},
        {value: 'inverse-distance', label: 'All-donor inverse distance'},
        {value: 'gdal-four-quadrant', label: 'Four-quadrant inverse distance'}
      ]
    },
    {
      kind: 'select',
      id: 'maximumDistance',
      label: 'Search radius',
      group: 'Bounded interpolation',
      apply: 'compile',
      default: '8',
      help: 'Bounds donor search in Euclidean pixel distance. A cell with no original donor within the selected radius stays unresolved.',
      options: [
        {value: '2', label: '2 cells'},
        {value: '4', label: '4 cells'},
        {value: '8', label: '8 cells'},
        {value: '12', label: '12 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'preparationFilter',
      label: 'Surface filter',
      group: 'Bounded interpolation',
      apply: 'compile',
      default: 'none',
      help: 'Keep the bounded fill as-is or apply a median rank filter to its output. Filtering can change measured valid pixels as well as predictions.',
      options: [
        {value: 'none', label: 'None · fill only'},
        {value: 'median', label: 'Median rank filter'}
      ]
    },
    {
      kind: 'select',
      id: 'preparationRadius',
      label: 'Median radius',
      group: 'Bounded interpolation',
      apply: 'compile',
      default: '1',
      help: 'Sets the square median footprint to 3 × 3 or 5 × 5 cells. Radius applies only when the median filter is selected.',
      options: [
        {value: '1', label: '1 cell'},
        {value: '2', label: '2 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'preparationNoData',
      label: 'Median nodata policy',
      group: 'Bounded interpolation',
      apply: 'compile',
      default: 'ignore',
      help: 'Ignore omits invalid neighbors; propagate invalidates an output if any neighbor is missing. The separate border control determines how outside-grid cells are handled.',
      options: [
        {value: 'ignore', label: 'Ignore invalid neighbors'},
        {value: 'propagate', label: 'Propagate nodata'}
      ]
    },
    {
      kind: 'select',
      id: 'preparationBorderMode',
      label: 'Median border',
      group: 'Bounded interpolation',
      apply: 'compile',
      default: 'clamp',
      help: 'Clamp repeats the nearest edge cell. Nodata treats samples beyond the crop as missing: ignore omits them, while propagate invalidates that edge output.',
      options: [
        {value: 'clamp', label: 'Repeat edge'},
        {value: 'nodata', label: 'Outside is missing'}
      ]
    },
    {
      kind: 'select',
      id: 'morphologyOperation',
      label: 'Grayscale operation',
      group: 'Reflectance structure',
      apply: 'param',
      default: 'opening',
      help: 'Dilation and erosion take local maxima and minima; opening and closing compose both operations to suppress bright or dark features by scale.',
      options: [
        {value: 'dilate', label: 'Dilation'},
        {value: 'erode', label: 'Erosion'},
        {value: 'opening', label: 'Opening'},
        {value: 'closing', label: 'Closing'}
      ]
    },
    {
      kind: 'select',
      id: 'morphologyRadius',
      label: 'Structuring radius',
      group: 'Reflectance structure',
      apply: 'compile',
      default: '2',
      help: 'Sets the neighborhood scale in 60 m cells. Larger radii remove or expand broader grayscale features.',
      options: [
        {value: '1', label: '60 m · 1 cell'},
        {value: '2', label: '120 m · 2 cells'},
        {value: '4', label: '240 m · 4 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'structuringElement',
      label: 'Neighborhood shape',
      group: 'Reflectance structure',
      apply: 'compile',
      default: 'square',
      help: 'A square includes all offsets within the radius in each axis. Cross uses Manhattan distance: row distance plus column distance must be within the radius, forming a diamond for radii above one.',
      options: [
        {value: 'square', label: 'Square'},
        {value: 'cross', label: 'Manhattan cross / diamond'}
      ]
    },
    {
      kind: 'select',
      id: 'reconstructionMode',
      label: 'Reconstruction filter',
      group: 'Geodesic reconstruction',
      apply: 'compile',
      default: 'opening',
      help: 'Opening starts from an erosion and regrows by dilation beneath the measured surface; closing starts from dilation and erodes above it.',
      options: [
        {value: 'opening', label: 'Opening by reconstruction'},
        {value: 'closing', label: 'Closing by reconstruction'}
      ]
    },
    {
      kind: 'select',
      id: 'holeConnectivity',
      label: 'Gap connectivity',
      group: 'Binary support topology',
      apply: 'compile',
      default: '8',
      help: 'Controls which missing-mask cells connect to the crop edge. The GPU labels enclosed gaps in the binary support mask; it never estimates reflectance.',
      options: [
        {value: '4', label: '4-connected background'},
        {value: '8', label: '8-connected background'}
      ]
    },
    {
      kind: 'select',
      id: 'reconstructionConnectivity',
      label: 'Propagation connectivity',
      group: 'Geodesic reconstruction',
      apply: 'compile',
      default: '8',
      help: 'Controls whether reconstruction propagates through edge neighbors only or through edges and diagonals.',
      options: [
        {value: '4', label: '4-connected'},
        {value: '8', label: '8-connected'}
      ]
    },
    {
      kind: 'select',
      id: 'maximumIterations',
      label: 'Iteration budget',
      group: 'Geodesic reconstruction',
      apply: 'compile',
      default: '8',
      help: 'Caps synchronous GPU propagation rounds. The output is published only if an extra check round confirms convergence.',
      options: [
        {value: '8', label: '8 rounds'},
        {value: '32', label: '32 rounds'},
        {value: '128', label: '128 rounds'}
      ]
    },
    {
      kind: 'slider',
      id: 'contourLevel',
      label: 'NIR contour level',
      group: 'Reflectance contours',
      apply: 'param',
      min: 0.05,
      max: 0.35,
      step: 0.01,
      default: 0.2,
      help: 'Changes the GPU-classified NIR reflectance isoline. Exact-level ties count as high; ambiguous cases use the bilinear determinant and a documented deterministic tie rule.'
    }
  ],
  readouts: [
    {id: 'observedSupport', label: 'Measured source support'},
    {id: 'fillSupport', label: 'Fill support and unresolved cells'},
    {id: 'preparationEvidence', label: 'Preparation provenance and holdout check'},
    {id: 'preparationPixel', label: 'Selected preparation provenance'},
    {id: 'holdoutError', label: 'Withheld-truth check'},
    {id: 'selectedPixel', label: 'Selected source and fill values'},
    {id: 'fillErrorChart', label: 'Error on withheld cells', kind: 'chart'},
    {id: 'structureChart', label: 'Selected-pixel structure', kind: 'chart'},
    {id: 'reconstructionStatus', label: 'Reconstruction convergence'},
    {id: 'holeFillStatus', label: 'Enclosed support-mask gaps'},
    {id: 'contourEvidence', label: 'Contour geometry and support'},
    {id: 'contourCases', label: 'Contour cell cases'}
  ],
  legends: state => {
    if (state.mapLayer === 'error') {
      return state.experiment === 'september-gaps'
        ? []
        : [
            {
              kind: 'ramp' as const,
              title: 'Filled − measured July NIR',
              ramp: 'vik' as const,
              extent: [-0.2, 0.2] as const,
              midpoint: 0,
              unit: 'reflectance'
            }
          ];
    }
    if (state.mapLayer === 'filled-cells') {
      return [
        {
          kind: 'ramp' as const,
          title: 'GPU-interpolated cells',
          ramp: 'cividis' as const,
          extent: [0, 1] as const,
          unit: 'binary mask'
        }
      ];
    }
    if (state.mapLayer === 'mask' || state.mapLayer === 'topology-fill') {
      const overlayTitle =
        state.mapLayer === 'topology-fill'
          ? 'Enclosed missing-mask cells'
          : state.experiment === 'september-gaps'
            ? 'September source gaps'
            : 'Deliberately withheld July cells';
      const overlayLabel =
        state.mapLayer === 'topology-fill'
          ? 'Missing cells enclosed by measured support'
          : 'Missing from the displayed observations';
      return [
        {
          kind: 'ramp' as const,
          title: 'Sentinel-2 NIR reflectance',
          ramp: 'cividis' as const,
          extent: [0, 0.4] as const,
          unit: 'unitless reflectance'
        },
        {
          kind: 'categories' as const,
          title: overlayTitle,
          entries: [{color: [224, 151, 66, 235], label: overlayLabel}],
          note:
            state.mapLayer === 'topology-fill'
              ? 'Visible only if the bounded GPU connectivity reconstruction converged.'
              : 'Orange marks withheld or source-missing cells.'
        }
      ];
    }
    if (state.mapLayer === 'contours') {
      return [
        {
          kind: 'ramp' as const,
          title: 'Sentinel-2 NIR reflectance',
          ramp: 'cividis' as const,
          extent: [0, 0.4] as const,
          unit: 'unitless reflectance'
        },
        {
          kind: 'categories' as const,
          title: 'Validity-aware marching-squares isoline',
          entries: [
            {
              color: [255, 246, 210, 245],
              label: `NIR ${state.contourLevel.toFixed(2)}`,
              shape: 'line' as const
            }
          ]
        }
      ];
    }
    return [
      {
        kind: 'ramp' as const,
        title: 'Sentinel-2 NIR reflectance',
        ramp: 'cividis' as const,
        extent: [0, 0.4] as const,
        unit: 'unitless reflectance'
      }
    ];
  },
  story: [
    {
      id: 'measured-support',
      headline: 'A blank pixel is a support statement',
      textAlternative: 'The source mask distinguishes measured NIR values from missing cells.',
      title: 'Start with the observation mask',
      body: 'This 64 × 64 crop uses the supplied validity masks for two Sentinel-2 L2A acquisitions. **Map shows** below switches between observed reflectance and the cells excluded from analysis. Missing values remain missing; they are never converted to zero.',
      evidence: '**{{observedSupport}}**',
      caveat:
        'July 13 is the fire ignition date, not a strictly pre-fire scene. September 21 gaps are real source gaps, not withheld observations.',
      options: {experiment: 'held-out', mapLayer: 'mask'},
      controls: ['experiment', 'mapLayer'],
      readouts: ['observedSupport', 'selectedPixel']
    },
    {
      id: 'known-truth',
      headline: 'Withheld measurements let us test interpolation',
      textAlternative:
        'Observed July values are deliberately hidden in either isolated cells or a central block.',
      title: 'Hide values that are known',
      body: 'The July surface is fully observed in this crop, so we can mask known values and later compare predictions with the originals. Choose **Withheld pattern** below to test isolated gaps or a contiguous 9 × 9 block. The output error is computed only at these artificially hidden cells.',
      evidence: '**{{holdoutError}}**',
      caveat:
        'A holdout experiment measures this crop and mask geometry; its error is not a guarantee for other scenes or larger gaps.',
      options: {holdoutPattern: 'block', mapLayer: 'mask'},
      controls: ['holdoutPattern'],
      readouts: ['holdoutError', 'fillErrorChart']
    },
    {
      id: 'bounded-fill',
      headline: 'A fill rule and search radius make different predictions',
      textAlternative:
        'GPU fill outputs retain measured donors and report cells left unresolved within the chosen radius.',
      title: 'Compare nearby original donors',
      body: '**Donor rule** below selects nearest, all-donor inverse distance, or a closest donor from each quadrant. **Search radius** bounds the fill. **Surface filter** composes an optional median rank filter; its radius and invalid-neighbor policy set the neighborhood support. **Median border** controls the crop edge: clamp repeats edge cells, while outside-as-missing treats them as unavailable. The readout reports the active policy and border. Switch **Map shows** between the original fill and prepared output. The original-valid, filled and filtered flags remain separate.',
      evidence: '**{{fillSupport}}** · **{{holdoutError}}**',
      caveat:
        'The GDAL-style four-quadrant method uses zero smoothing and keeps the first donor at an exact squared-distance tie. Native GDAL 3.9.3 can choose a later tied donor through floating-point roundoff. Neither reconstructs hidden landscape detail.',
      options: {mapLayer: 'fill', fillMethod: 'gdal-four-quadrant', maximumDistance: '8'},
      controls: [
        'fillMethod',
        'maximumDistance',
        'preparationFilter',
        'preparationBorderMode',
        'mapLayer'
      ],
      readouts: ['fillSupport', 'preparationEvidence', 'holdoutError', 'preparationPixel']
    },
    {
      id: 'real-gaps',
      headline: 'Real gaps have no hidden truth to score',
      textAlternative:
        'September validity marks actual unavailable pixels; interpolation support and unresolved cells remain visible.',
      title: 'Separate prediction from validation',
      body: 'Switch **Observation set** below to September. The bounded fill still uses only valid original donors, but the hidden values are unknown. **Gap connectivity** classifies enclosed voids in the binary support mask; **Map shows** can display that topology result. It fills no reflectance and does not score the missing values.',
      evidence: '**{{observedSupport}}** · **{{fillSupport}}** · **{{holeFillStatus}}**',
      caveat:
        'No RMSE or MAE is computed for September because the missing source values are not available as truth.',
      options: {experiment: 'september-gaps', mapLayer: 'topology-fill'},
      controls: ['experiment', 'holeConnectivity', 'mapLayer'],
      readouts: ['observedSupport', 'fillSupport', 'holdoutError', 'holeFillStatus']
    },
    {
      id: 'grayscale-structure',
      headline: 'Morphology changes grayscale structure, not a fire class',
      textAlternative:
        'Dilation, erosion, opening and closing operate directly on the measured reflectance surface.',
      title: 'Look at the scale of reflectance features',
      body: '**Grayscale operation** below selects a local maximum, minimum, opening or closing of measured July NIR. **Structuring radius** and **Neighborhood shape** change the scale and geometry of that neighborhood. Click the map to compare actual GPU outputs at one pixel.',
      evidence: '**{{selectedPixel}}**',
      caveat:
        'These continuous-value operations do not identify burn scars or turn reflectance into a categorical map.',
      options: {experiment: 'held-out', mapLayer: 'morphology', morphologyOperation: 'opening'},
      controls: ['morphologyOperation', 'morphologyRadius', 'structuringElement', 'mapLayer'],
      readouts: ['structureChart', 'selectedPixel']
    },
    {
      id: 'reconstruction-contours',
      headline: 'Convergence and isolines make limits inspectable',
      textAlternative:
        'A geodesic grayscale reconstruction publishes only when its iteration check converges; contours trace NIR levels.',
      title: 'Check the stopping rule and contour geometry',
      body: '**Reconstruction filter** below chooses opening or closing by geodesic propagation. **Propagation connectivity** and **Iteration budget** govern the bounded fit; the scene withholds the reconstruction map unless the GPU confirms convergence. Set **NIR contour level** to trace a spectral isoline.',
      evidence: '**{{reconstructionStatus}}** · **{{contourEvidence}}**',
      caveat:
        'Contour lines are reflectance thresholds with validity-aware marching squares. They are not mapped fire boundaries; exact ties follow the GPU’s stated deterministic policy.',
      options: {
        mapLayer: 'contours',
        reconstructionMode: 'opening',
        maximumIterations: '8',
        contourLevel: 0.2
      },
      controls: [
        'reconstructionMode',
        'reconstructionConnectivity',
        'maximumIterations',
        'contourLevel',
        'mapLayer'
      ],
      readouts: ['reconstructionStatus', 'contourEvidence', 'contourCases']
    }
  ],
  about: {
    what: '`GPURasterFillNoData` preserves original valid cells and estimates only missing locations from original donors. `addSurfacePreparationRecipe` composes that fill with an optional median rank filter and reports original-valid, filled, filtered and unresolved provenance. The grayscale morphology and reconstruction outputs operate on measured reflectance; the contour classifier reports per-cell marching-squares cases and bounded line geometry.',
    why: 'An interpolation can be evaluated where measurements were deliberately withheld, while real gaps require a different kind of evidence: donor support, unresolved cells and explicit uncertainty rather than invented error scores.',
    howToRead:
      'The selected cell marker links the map to the readouts and chart. The mask uses orange for withheld or source-missing cells; the interpolation footprint marks GPU-created values. The error map is defined only over July holdouts. The contour map keeps the NIR raster beneath its isolines.'
  },
  create: async ctx =>
    (await import('./missing-observations.compute')).createMissingObservations(ctx)
});
