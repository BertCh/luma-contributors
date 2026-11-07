// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {defineScene} from '../scene';
import type {NatureKernelDensityOptions} from './nature-kernel-density.compute';

const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan', 'loop', 'jackson-park', 'humboldt-park'], {
  loop: {minZoom: 10.2},
  'jackson-park': {tone: 'muted', minZoom: 10.5},
  'humboldt-park': {tone: 'muted', minZoom: 10.5}
});

export default defineScene<NatureKernelDensityOptions>({
  id: 'nature-kernel-density',
  title: 'How does a fixed-window wildlife KDE change with its assumptions?',
  chapter: 'points',
  order: 2,
  summary:
    'An equal-area, fixed-window kernel density estimate of 2023 Chicago iNaturalist observation records, with visible bandwidth, boundary correction, normalization and approximation choices.',
  contributors: ['GPUKernelDensity'],
  datasets: [{id: 'chicago-nature', role: '2023 iNaturalist observation records'}],
  initialView: {...CITY_FRAMES.chicago, zoom: 11.1},
  basemap: ground('night', {labels: 'above', labelPreset: 'places-only'}),
  annotations: ORIENTATION,
  furniture: {
    title: true,
    credit: true,
    caveat: 'Records measure observer effort, not organism abundance.'
  },
  options: [
    {
      kind: 'select',
      id: 'kernel',
      label: 'Kernel shape',
      group: 'Estimator',
      apply: 'compile',
      default: 'gaussian',
      help: 'Gaussian has a long tail truncated at five bandwidths. Quartic and Epanechnikov have compact radial support; their scale is the library bandwidth parameter, not an exact spatstat sigma conversion.',
      options: [
        {value: 'gaussian', label: 'Gaussian'},
        {value: 'quartic', label: 'Quartic'},
        {value: 'epanechnikov', label: 'Epanechnikov'}
      ]
    },
    {
      kind: 'select',
      id: 'bandwidthRule',
      label: 'Bandwidth rule',
      group: 'Estimator',
      apply: 'compile',
      default: 'fixed',
      help: 'Use an authored ground bandwidth or a Scott/Silverman two-dimensional normal-reference rule. The two reference rules are identical in two dimensions and often oversmooth clustered observations.',
      options: [
        {value: 'fixed', label: 'Authored metres'},
        {value: 'scott', label: 'Scott reference'},
        {value: 'silverman', label: 'Silverman reference'}
      ]
    },
    {
      kind: 'slider',
      id: 'bandwidth',
      label: 'Ground bandwidth',
      group: 'Estimator',
      apply: 'param',
      min: 250,
      max: 2500,
      step: 250,
      default: 750,
      unit: 'm',
      disabledWhen: state => state.bandwidthRule !== 'fixed',
      help: 'Sets the exact radial bandwidth in analysis-frame metres. The output grid stays 64 by 64 cells of 250 m as this changes.'
    },
    {
      kind: 'select',
      id: 'edgeCorrection',
      label: 'Window edge assumption',
      group: 'Study window',
      apply: 'compile',
      default: 'diggle',
      help: 'None lets kernel mass spill beyond the rectangle. Uniform divides each cell by in-window kernel mass; Diggle weights each point by its in-window kernel mass and tends to preserve the point total, subject to raster quadrature.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'uniform', label: 'Uniform (location based)'},
        {value: 'diggle', label: 'Diggle (point based)'}
      ]
    },
    {
      kind: 'select',
      id: 'normalization',
      label: 'Density scale',
      group: 'Study window',
      apply: 'compile',
      default: 'intensity',
      help: 'Intensity is weighted observations per square kilometre. Probability density divides by the included weight and should integrate to about one over the full window when edge correction preserves mass.',
      options: [
        {value: 'intensity', label: 'Intensity (records / km²)'},
        {value: 'probability', label: 'Probability density (/ km²)'}
      ]
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Included records',
      group: 'Study window',
      apply: 'param',
      default: 'observations',
      help: 'Each record contributes one, or only community-confirmed research-grade records contribute one. The GPU total weight is reported; zero-weight rows remain valid inputs but add no weight. These are observation records and observer effort, never organism abundance.',
      options: [
        {value: 'observations', label: 'All observations (unit weight)'},
        {value: 'researchGrade', label: 'Research-grade observations only'}
      ]
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Gaussian evaluation',
      group: 'Computation',
      apply: 'compile',
      default: 'exact',
      disabledWhen: state => state.kernel !== 'gaussian',
      help: 'Exact evaluates the chosen kernel at each output cell. Binned and auto apply only to Gaussian; binned distributes points to nearby grid centers and blurs approximately. Auto uses exact at small and very large bandwidths and binned between those ranges; the slider includes 2,500 m to exercise the binned path.',
      options: [
        {value: 'exact', label: 'Exact'},
        {value: 'binned', label: 'Binned approximation'},
        {value: 'auto', label: 'Automatic exact / binned'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSurface',
      label: 'Show KDE surface',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Starts with the observation records and study-window outline, then reveals the computed raster when enabled.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Surface colors',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'Changes only the surface palette. Peak and integral readouts continue to report the computed GPU raster.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'fire', label: 'Fire'},
        {value: 'mako', label: 'Mako'}
      ]
    }
  ],
  story: [
    {
      id: 'source-window',
      title: 'Which observation records fall inside one declared window?',
      headline: 'The fixed rectangle sets the study population',
      textAlternative:
        'Chicago observation records and their fixed 16 kilometer square analysis window.',
      body: 'The dots are 2023 iNaturalist records, not individual animals. The **16 km by 16 km fixed analysis window** below the map is centered on central Chicago; every supplied row inside it is retained, and the excluded total counts records outside it.\n\nThe displayed points and fixed raster make the study boundary visible before smoothing.',
      evidence:
        '**{{windowRecords}}** records enter the initial unit-weight estimate; **{{excluded}}** source rows lie outside the metric rectangle.',
      caveat:
        'The dataset is a sample of observation activity. Its spatial pattern cannot be read as wildlife abundance.',
      camera: {
        longitude: -87.68,
        latitude: 41.88,
        zoom: 11.1,
        pitch: 0,
        bearing: 0,
        transitionMs: 1000
      },
      controls: [],
      readouts: ['records', 'windowRecords', 'excluded'],
      options: {showSurface: false},
      furniture: {
        title: {
          title: 'Observation records inside a fixed window',
          subtitle: '2023 iNaturalist · fixed 16 km square · source record locations'
        },
        credit: true,
        caveat: 'Records measure observer effort, not organism abundance.'
      }
    },
    {
      id: 'metric-bandwidth',
      title: 'How far should each record influence the surface?',
      headline: 'Bandwidth sets the map’s spatial grain',
      textAlternative: 'A GPU density raster over Chicago with the selected metric bandwidth.',
      body: 'The Lambert equal-area analysis frame puts records and the grid in planar metres. **Show KDE surface** below reveals the raster; **Ground bandwidth** changes the kernel reach while the 250 m cells and study rectangle stay fixed. Choose **Bandwidth rule** below to compare the authored distance with a normal-reference rule.\n\nThe control shows a local smoothing scale, not a biological movement distance.',
      evidence:
        'The selected bandwidth is **{{bandwidth}}**; the raster integral is **{{integral}}**.',
      caveat:
        'The ellipsoidal equal-area frame preserves area. Its planar distance distortion is negligible at this urban scale but is not zero.',
      options: {bandwidth: 750, showSurface: true},
      controls: ['showSurface', 'bandwidthRule', 'bandwidth'],
      readouts: ['bandwidth', 'integral'],
      furniture: {
        title: {
          title: 'A metric smoothing scale',
          subtitle: 'Lambert equal-area · 64 × 64 cells · 250 m cell size'
        },
        credit: true,
        scaleBar: true
      }
    },
    {
      id: 'edge-assumption',
      title: 'What happens where the window cuts through a kernel?',
      headline: 'Edge correction changes estimates near the frame',
      textAlternative:
        'The observation density raster and fixed rectangle under three edge assumptions.',
      body: 'The surface is evaluated on the same rectangle as the analysis window. **Window edge assumption** below selects no correction, location-based uniform correction, or point-based Diggle correction.\n\nCompare the reported integral with the included record count (or probability total) to see how boundary treatment and raster quadrature affect mass.',
      evidence:
        'The GPU raster integral is **{{integral}}** across **{{validCells}}** valid cell centers.',
      caveat:
        'Uniform correction targets locally uniform intensity in expectation; it does not force an exact raster integral. Diggle better preserves point mass, with numerical quadrature error.',
      options: {edgeCorrection: 'diggle', showSurface: true},
      controls: ['edgeCorrection'],
      readouts: ['integral', 'included', 'validCells'],
      furniture: {
        title: {
          title: 'Boundary assumptions change edge cells',
          subtitle: 'Same points, window and grid · edge correction varies'
        },
        credit: true,
        caveat: 'The finite window is part of the estimator.'
      }
    },
    {
      id: 'kernel-and-scale',
      title: 'Does kernel shape change the local pattern?',
      headline: 'Kernel shape changes how nearby records contribute',
      textAlternative:
        'The same Chicago observation density surface colored by the selected kernel result.',
      body: '**Kernel shape** below compares Gaussian, quartic and Epanechnikov radial kernels. Choose **Density scale** below to read the same raster in records per square kilometre or as a probability density per square kilometre. The three fixed cell readouts are GPU values at columns and rows 16, 32 and 48 (row zero is north).\n\nA bright cell means a higher estimated density under the selected assumptions.',
      evidence:
        'GPU values: central cell **{{cell32}}**, northwest **{{cell16}}**, southeast **{{cell48}}**; raster integral **{{integral}}**.',
      caveat:
        'Compact kernels use the library bandwidth parameter; they are not rescaled for exact numerical parity with spatstat’s Gaussian standard deviation.',
      controls: ['kernel', 'normalization'],
      options: {showSurface: true},
      readouts: ['integral', 'cell16', 'cell32', 'cell48'],
      furniture: {
        title: {
          title: 'Kernel shape and output units',
          subtitle: 'Radial kernel evaluation at cell centers'
        },
        credit: true,
        scaleBar: true
      }
    },
    {
      id: 'sensitivity',
      title: 'How much does the answer depend on effort and approximation?',
      headline: 'Record quality and approximation affect the result',
      textAlternative:
        'Chicago observation density with the selected record weights and Gaussian evaluation method.',
      body: '**Included records** below tests all unit-weight observations against the actual research-grade flag. With Gaussian selected, **Gaussian evaluation** can switch to a binned approximation or automatic path; the method and its caveat stay visible in the control.\n\nTry changing bandwidth and edge treatment, then compare the displayed integral and peak.',
      evidence:
        '**{{included}}** weighted records contribute, and **{{zeroWeight}}** in-window rows have zero weight; the raster integral is **{{integral}}**.',
      caveat:
        'Binned Gaussian is an approximation to exact evaluation. It is not enabled for quartic or Epanechnikov kernels.',
      controls: ['weight', 'method', 'ramp'],
      options: {showSurface: true},
      readouts: ['included', 'zeroWeight', 'integral', 'peak'],
      furniture: {
        title: {
          title: 'Sensitivity is part of the result',
          subtitle: 'Observation effort · approximate smoothing is identified in the controls'
        },
        credit: true,
        scaleBar: true
      }
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      id: 'density',
      title: state.normalization === 'intensity' ? 'Observation intensity' : 'Probability density',
      ramp: state.ramp,
      extent: 'gpu',
      range: [0.15, 1],
      unit: state.normalization === 'intensity' ? 'records / km²' : '/ km²',
      note: 'One record is an observation, not an organism.'
    }
  ],
  readouts: [
    {
      id: 'records',
      label: 'Source records',
      help: 'All source rows, before the fixed metric window.'
    },
    {
      id: 'windowRecords',
      label: 'Records in fixed window',
      help: 'All source locations retained inside the disclosed rectangle.'
    },
    {
      id: 'included',
      label: 'Included weight',
      help: 'GPU-reported sum of input weights used by the estimator. Zero-weight rows remain valid inputs but add no weight; their row count is shown separately.'
    },
    {
      id: 'zeroWeight',
      label: 'In-window zero-weight rows',
      help: 'In-window observations excluded by the selected unit or research-grade weighting.'
    },
    {
      id: 'excluded',
      label: 'Outside-window records',
      help: 'Source rows outside the analysis rectangle.'
    },
    {
      id: 'bandwidth',
      label: 'Ground bandwidth',
      help: 'Authored value or GPU-computed reference-rule bandwidth.'
    },
    {
      id: 'integral',
      label: 'Raster integral (density × cell area)',
      help: 'Sum of GPU cell-center density values times the 250 m cell area.'
    },
    {
      id: 'peak',
      label: 'Peak density',
      help: 'Maximum finite GPU output cell, converted to the legend unit.'
    },
    {
      id: 'validCells',
      label: 'Valid output cells',
      help: 'Output cell centers with finite estimated density.'
    },
    {
      id: 'cell16',
      label: 'GPU cell (16, 16)',
      help: 'GPU output at this fixed raster cell; rows are indexed north to south.'
    },
    {
      id: 'cell32',
      label: 'GPU cell (32, 32)',
      help: 'GPU output at this fixed raster cell; rows are indexed north to south.'
    },
    {
      id: 'cell48',
      label: 'GPU cell (48, 48)',
      help: 'GPU output at this fixed raster cell; rows are indexed north to south.'
    },
    {
      id: 'invalidRows',
      label: 'GPU input exclusions',
      help: 'Invalid or out-of-index rows reported by the GPU contributor.'
    },
    {
      id: 'resolvedMethod',
      label: 'Resolved GPU method',
      help: 'The method selected by GPUKernelDensity. Auto resolves from the bandwidth and available padding; non-Gaussian kernels use exact evaluation.'
    },
    {
      id: 'paddingStatus',
      label: 'Binned padding',
      help: 'GPU diagnostic for whether the Gaussian blur had its requested edge margin.'
    }
  ],
  snippet: state =>
    `new GPUKernelDensity({ kernel: '${state.kernel}', edgeCorrection: '${state.edgeCorrection}', normalization: '${state.normalization}', method: '${state.method}', bandwidth: ${state.bandwidth} /* metres */ });`,
  about: {
    what: 'This companion estimates a spatial point-process intensity from Chicago iNaturalist observation locations. The existing nature-density story bins observations into a camera-following grid; this fixed-window scene evaluates a metric kernel at each raster cell.',
    why: 'Bandwidth and edge correction determine how local observation effort is smoothed and how the finite study rectangle is treated.',
    howToRead:
      'The raster colors show relative surface density under the chosen model. Use the units, included count, bandwidth and raster integral together; none of these values estimate organism abundance. The source manifest carries dataset provenance and license.'
  },
  create: async ctx =>
    (await import('./nature-kernel-density.compute')).createNatureKernelDensity(ctx)
});
