// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FireScarTopologyOptions} from './fire-scar-topology.compute';

const PATCH_COLORS = [
  [61, 106, 163, 255],
  [221, 137, 47, 255],
  [113, 158, 96, 255],
  [181, 101, 139, 255],
  [84, 159, 164, 255],
  [195, 177, 82, 255],
  [130, 113, 182, 255],
  [209, 107, 84, 255]
] as const;

export default defineScene<FireScarTopologyOptions>({
  id: 'fire-scar-topology',
  title: 'What makes a fire scar a connected patch?',
  chapter: 'raster',
  order: 9,
  summary:
    'Observed pre/post Sentinel-2 reflectance becomes a valid-pixel dNBR mask, then analyst-selected morphology and connected-region summaries. These are spectral-change patches, not official fire boundaries or field-validated burn severity.',
  contributors: [
    'GPURasterArithmetic',
    'GPURasterHistogram',
    'GPURasterOtsuThreshold',
    'GPURasterThreshold',
    'GPURasterRankFilter',
    'GPURasterOpening',
    'GPURasterClosing',
    'GPURasterConnectedComponents',
    'GPURasterDenseComponents',
    'GPURasterRegionMeasurements',
    'GPURasterRegionGraph',
    'GPURasterSkeleton',
    'GPURasterMedialAxis',
    'GPURasterWatershedSegmentation'
  ],
  datasets: [
    {id: 'dixie-spectral', role: 'Sentinel-2 pre/post bands and strict land-valid support'}
  ],
  initialView: {longitude: -121.0, latitude: 40.17, zoom: 11.5},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'What counts as a connected patch?',
      subtitle: 'Dixie Fire · dNBR sensitivity lab'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2021',
    caveat:
      'Masks show analyst-defined spectral change, not an official fire perimeter or validated soil-burn severity. The pre-fire scene is the ignition date; strict land screening can exclude dark burned pixels classified as shadow. [USGS burn-severity glossary](https://burnseverity.cr.usgs.gov/glossary) · [Sentinel-2 scene classification](https://sentiwiki.copernicus.eu/web/s2-processing).'
  },
  options: [
    {
      kind: 'select',
      id: 'mapLayer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'cleaned',
      options: [
        {value: 'support', label: 'Common land support'},
        {value: 'nbrBefore', label: 'NBR · 13 Jul 2021'},
        {value: 'nbrAfter', label: 'NBR · 21 Sep 2021'},
        {value: 'dnbr', label: 'Observed dNBR'},
        {value: 'threshold', label: 'Threshold mask'},
        {value: 'cleaned', label: 'Cleaned mask'},
        {value: 'regions', label: 'Connected patches'},
        {value: 'classContacts', label: 'Selected-mask contacts'},
        {value: 'focusMask', label: '64×64 focus mask'},
        {value: 'skeleton', label: 'Focus · one-pixel skeleton'},
        {value: 'medialAxis', label: 'Focus · maximal-disk axis'},
        {value: 'watershed', label: 'Focus · dNBR marker watershed'},
        {value: 'watershedMerged', label: 'Focus · merged watershed basins'},
        {value: 'watershedMarkers', label: 'Focus · watershed seeds'}
      ]
    },
    {
      kind: 'select',
      id: 'thresholdMethod',
      label: 'Threshold source',
      group: 'Mask definition',
      apply: 'compile',
      default: 'fixed',
      options: [
        {value: 'fixed', label: 'Analyst threshold'},
        {value: 'otsu', label: 'Otsu split'},
        {value: 'source-probe', label: 'GPU source-cell probe'}
      ],
      disabledWhen: state => state.thresholdOperation === 'range',
      help: 'Otsu selects a histogram split over valid dNBR pixels. The source-cell probe uses the actual GPU dNBR value at source row 26,488; with no median filter, this checks exact equality behavior. Range mode uses its two manual endpoints; the Otsu split remains visible as a reference.'
    },
    {
      kind: 'select',
      id: 'thresholdOperation',
      label: 'Selection rule',
      group: 'Mask definition',
      apply: 'compile',
      default: 'above',
      options: [
        {value: 'above', label: 'Above one boundary'},
        {value: 'below', label: 'Below one boundary'},
        {value: 'range', label: 'Between two endpoints'}
      ],
      help: 'Range mode uses the ordered pair of manual endpoints and applies the selected inclusivity at both ends.'
    },
    {
      kind: 'toggle',
      id: 'thresholdInclusive',
      label: 'Include equal boundary values',
      group: 'Mask definition',
      apply: 'compile',
      default: true,
      help: 'Above and below apply ≥ / ≤ when enabled and > / < when disabled. Range applies the corresponding inclusive or exclusive comparison to both endpoints.'
    },
    {
      kind: 'slider',
      id: 'thresholdValue',
      label: 'Manual cutoff / range endpoint A',
      group: 'Mask definition',
      apply: 'param',
      min: -0.2,
      max: 1,
      step: 0.01,
      default: 0.27,
      disabledWhen: state =>
        state.thresholdMethod !== 'fixed' && state.thresholdOperation !== 'range',
      help: 'Used as the manual scalar cutoff in fixed mode and as endpoint A in range mode. Range endpoints are ordered before the GPU query. This is not a validated fire-boundary or soil-severity threshold.'
    },
    {
      kind: 'slider',
      id: 'rangeEndpoint',
      label: 'Manual range endpoint B',
      group: 'Mask definition',
      apply: 'param',
      min: -0.2,
      max: 1,
      step: 0.01,
      default: 0.66,
      disabledWhen: state => state.thresholdOperation !== 'range',
      help: 'Used only for range mode. The GPU receives [min(A, B), max(A, B)], so changing endpoint order cannot make an invalid range.'
    },
    {
      kind: 'select',
      id: 'cleanup',
      label: 'Mask operation',
      group: 'Topology sensitivity',
      apply: 'compile',
      default: 'closing',
      options: [
        {value: 'none', label: 'None'},
        {value: 'median', label: 'Median dNBR, then threshold'},
        {value: 'opening', label: 'Binary opening'},
        {value: 'closing', label: 'Binary closing'}
      ],
      help: 'Median changes the spectral field before classification; opening removes small foreground islands; closing fills small gaps.'
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Neighborhood radius',
      group: 'Topology sensitivity',
      apply: 'compile',
      min: 1,
      max: 3,
      step: 1,
      default: 1,
      unit: 'cells',
      help: 'A square footprint of this many 60 m cells. Larger radii can merge nearby patches or erase narrow features.'
    },
    {
      kind: 'select',
      id: 'connectivity',
      label: 'Pixel connectivity',
      group: 'Topology sensitivity',
      apply: 'compile',
      default: '8',
      options: [
        {value: '4', label: '4-neighbor'},
        {value: '8', label: '8-neighbor'}
      ],
      help: 'Four-neighbor connects shared edges. Eight-neighbor also connects diagonal touches.'
    },
    {
      kind: 'select',
      id: 'regionMergeMode',
      label: 'Basin merge rule',
      group: 'Focus-region merge',
      apply: 'compile',
      default: 'fixed',
      options: [
        {value: 'fixed', label: 'Fixed single-linkage'},
        {value: 'adaptive', label: 'Adaptive mean update'}
      ],
      help: 'Fixed merges original mean-difference edges by bounded propagation. Adaptive recomputes count-weighted merged means and chooses one closest eligible pair per round.'
    },
    {
      kind: 'slider',
      id: 'regionMergeThreshold',
      label: 'Maximum mean dNBR gap',
      group: 'Focus-region merge',
      apply: 'compile',
      min: 0,
      max: 0.3,
      step: 0.005,
      default: 0.2,
      unit: 'dNBR',
      help: 'Merges adjacent focus basins when their relevant mean dNBR gap is at or below this value.'
    },
    {
      kind: 'slider',
      id: 'regionMergeRounds',
      label: 'Maximum merge rounds',
      group: 'Focus-region merge',
      apply: 'compile',
      min: 1,
      max: 16,
      step: 1,
      default: 16,
      unit: 'rounds',
      help: 'Bounds graph propagation or adaptive pair updates across the at-most 16 watershed marker regions.'
    }
  ],
  legends: state => {
    if (state.mapLayer === 'support') {
      return [
        {
          kind: 'categories',
          title: 'Joint land-valid support',
          entries: [
            {color: [80, 171, 222, 220], label: 'Valid in both scenes'},
            {color: [0, 0, 0, 0], label: 'Outside support'}
          ]
        }
      ];
    }
    if (
      state.mapLayer === 'dnbr' ||
      state.mapLayer === 'nbrBefore' ||
      state.mapLayer === 'nbrAfter'
    ) {
      return [
        {
          id: state.mapLayer,
          title:
            state.mapLayer === 'dnbr'
              ? 'dNBR (pre − post)'
              : state.mapLayer === 'nbrBefore'
                ? 'NBR · 13 Jul 2021'
                : 'NBR · 21 Sep 2021',
          kind: 'ramp',
          ramp: 'diverging',
          extent: state.mapLayer === 'dnbr' ? [-2, 2] : [-1, 1],
          note: 'Outside common strict land-valid support is transparent; the dNBR histogram also spans −2 to 2.'
        }
      ];
    }
    if (state.mapLayer === 'focusMask') {
      return [
        {
          kind: 'categories',
          title: '64 × 64 focus mask',
          entries: [{color: [242, 92, 54, 230], label: 'Selected dNBR mask'}],
          note: 'Transparent cells include unchanged and invalid support; see common-support readout.'
        }
      ];
    }
    if (state.mapLayer === 'skeleton') {
      return [
        {
          kind: 'categories',
          title: 'Zhang–Suen thinning',
          entries: [{color: [75, 28, 113, 255], label: 'Converged skeleton cell'}],
          note: 'One-pixel thinning. If the bounded iteration budget does not converge, output is suppressed.'
        }
      ];
    }
    if (state.mapLayer === 'medialAxis') {
      return [
        {
          kind: 'categories',
          title: 'Maximal-inscribed-disk centers',
          entries: [{color: [66, 214, 177, 255], label: 'Medial-axis center'}],
          note: 'Discrete maximal disks; this is not the one-pixel thinning shown in the skeleton view.'
        }
      ];
    }
    if (state.mapLayer === 'watershed' || state.mapLayer === 'watershedMerged') {
      return [
        {
          kind: 'categories',
          title:
            state.mapLayer === 'watershed'
              ? 'Marker watershed · basin IDs repeat by color'
              : 'Merged marker basins · root IDs repeat by color',
          entries: PATCH_COLORS.map((color, index) => ({color, label: `ID mod 8 = ${index}`})),
          note:
            state.mapLayer === 'watershed'
              ? 'Priority flooding on −dNBR: high dNBR are low basins. Unseeded cells remain transparent; this is not hydrologic watershed delineation.'
              : 'The selected fixed or adaptive mean-difference graph mapping is applied to GPU watershed labels. Unseeded cells remain transparent.'
        }
      ];
    }
    if (state.mapLayer === 'watershedMarkers') {
      return [
        {
          kind: 'categories',
          title: 'Analyst-selected watershed markers',
          entries: [{color: [0, 71, 171, 255], label: 'Regular-lattice valid-cell seed'}],
          note: 'Seed positions are selected deterministically from the accepted focus support.'
        }
      ];
    }
    if (state.mapLayer === 'classContacts') {
      return [
        {
          kind: 'categories',
          title: 'Accepted cleaned-mask classes',
          entries: [
            {color: [85, 102, 116, 235], label: 'Accepted land outside selected mask'},
            {color: [242, 92, 54, 230], label: 'Accepted land inside selected mask'}
          ],
          note: 'Invalid support is transparent. Boundary statistics count only observed 4-neighbor shared edges between these cleaned-mask classes.'
        }
      ];
    }
    if (state.mapLayer === 'regions') {
      return [
        {
          kind: 'categories',
          title: 'Patch ID · colors repeat every 8 IDs',
          entries: [
            {color: [61, 106, 163, 255], label: 'ID ≡ 0'},
            {color: [221, 137, 47, 255], label: 'ID ≡ 1'},
            {color: [113, 158, 96, 255], label: 'ID ≡ 2'},
            {color: [181, 101, 139, 255], label: 'ID ≡ 3'},
            {color: [84, 159, 164, 255], label: 'ID ≡ 4'},
            {color: [195, 177, 82, 255], label: 'ID ≡ 5'},
            {color: [130, 113, 182, 255], label: 'ID ≡ 6'},
            {color: [209, 107, 84, 255], label: 'ID ≡ 7'}
          ],
          note: 'Dense GPU component labels; zero is background.'
        }
      ];
    }
    return [
      {
        kind: 'categories',
        title: 'Analyst-defined mask',
        entries: [
          {color: [242, 92, 54, 230], label: 'Inside mask'},
          {color: [0, 0, 0, 0], label: 'Outside mask or support'}
        ]
      }
    ];
  },
  readouts: [
    {id: 'validPixels', label: 'Common valid pixels', unit: 'cells'},
    {id: 'otsuThreshold', label: 'Otsu dNBR split'},
    {id: 'activeThreshold', label: 'Active threshold'},
    {id: 'thresholdCells', label: 'Threshold-selected support', unit: 'cells'},
    {id: 'componentCount', label: 'Connected components', unit: 'patches', emphasis: 'tile'},
    {id: 'selectedCells', label: 'Cells in mask', unit: 'cells'},
    {id: 'selectedArea', label: 'Mask area on accepted support', unit: 'ha'},
    {id: 'largestPatch', label: 'Largest patch', unit: 'ha'},
    {id: 'topologyStatus', label: 'Topology status'},
    {id: 'classBoundaryLength', label: 'Observed class boundary', unit: 'km'},
    {id: 'classMeanDifference', label: 'Mean dNBR class gap'},
    {id: 'classContactStatus', label: 'Class-contact graph status'},
    {id: 'focusSupport', label: 'Focus accepted support', unit: 'cells'},
    {id: 'focusMarkers', label: 'Watershed markers', unit: 'markers'},
    {id: 'skeletonPixels', label: 'Skeleton cells', unit: 'cells'},
    {id: 'medialRadius', label: 'Maximum inscribed radius'},
    {id: 'mergedBasinCount', label: 'Merged basin roots'},
    {id: 'watershedStatus', label: 'Watershed and merge status'},
    {id: 'focusRegionMapping', label: 'Focus basin counts / mapping', layout: 'block'}
  ],
  story: [
    {
      id: 'spectral-change',
      title: 'Start with two observed dates',
      headline: 'The input is a two-date spectral difference',
      body: 'The map derives NBR for **13 July** and **21 September 2021**, then dNBR = NBR before − NBR after, following the [USGS dNBR convention](https://burnseverity.cr.usgs.gov/glossary). The shared support requires both dates to pass the bundled land-valid screen: all native SCL samples are vegetation or non-vegetated land and source reflectance is valid.',
      textAlternative:
        'Three continuous rasters show joint land-valid support, either date’s NBR, or pre-fire minus post-fire dNBR over the 15 km Dixie Fire window.',
      evidence: '**{{validPixels}}** cells are valid in both scenes at the 60 m analysis grid.',
      caveat:
        'The pre-fire image is the fire ignition date, not a pre-season baseline. A two-date spectral difference can reflect vegetation, soil, moisture, illumination, or registration effects; strict SCL screening can censor dark burns mapped as shadow.',
      controls: ['mapLayer'],
      readouts: ['validPixels'],
      options: {mapLayer: 'dnbr'}
    },
    {
      id: 'threshold',
      title: 'Choose a mask threshold',
      headline: 'Threshold direction and boundary rules define the mask',
      body: 'Choose **above**, **below**, or **range**, then set whether boundary equality is included. The manual mode uses the selected dNBR cutoff; the source-probe mode reads an actual GPU dNBR cell as its scalar cutoff so inclusive and exclusive comparisons can be distinguished on an exact source value. Otsu remains a GPU histogram split for one-sided rules. Range uses two bounded manual endpoints, ordered before dispatch; the Otsu split remains visible as a reference but is not silently substituted for a range endpoint.',
      textAlternative:
        'A binary raster selects valid pixels above, below, or within an interval under inclusive or exclusive boundary rules. Readouts show the active rule, GPU-selected cell count, and Otsu split.',
      evidence:
        'Rule: **{{activeThreshold}}**; **{{thresholdCells}}**; Otsu split: **{{otsuThreshold}}**.',
      caveat:
        'Otsu optimizes a two-class histogram partition, not a fire-specific ecological interpretation. The histogram spans −2 to 2; cells outside common land-valid support are excluded. For the default 0.27 and 0.66 endpoints, no source cells equal either boundary, so inclusive and exclusive counts agree; the source-cell probe supplies an exact-equality check.',
      controls: [
        'thresholdMethod',
        'thresholdOperation',
        'thresholdInclusive',
        'thresholdValue',
        'rangeEndpoint'
      ],
      readouts: ['activeThreshold', 'thresholdCells', 'otsuThreshold'],
      options: {
        thresholdMethod: 'fixed',
        thresholdOperation: 'above',
        cleanup: 'none',
        mapLayer: 'threshold'
      }
    },
    {
      id: 'cleanup',
      title: 'Change the patch rule',
      headline: 'Neighborhood cleanup changes connectivity',
      body: '**Median dNBR** filters the continuous index before thresholding. Binary **opening** and **closing** operate on the classified mask. Set the radius in 60 m cells and compare the resulting mask and patch counts.',
      textAlternative:
        'The selected mask is transformed by a square-neighborhood median, binary opening, binary closing, or no cleanup; the map updates to the resulting GPU raster.',
      evidence:
        '**{{selectedCells}}** cells (**{{selectedArea}}**) in **{{componentCount}}** connected patches under the current rule.',
      caveat:
        'Morphological edits are analyst-selected scale assumptions; they may remove narrow features or join separate spectral-change areas.',
      controls: ['cleanup', 'radius', 'mapLayer'],
      readouts: ['selectedCells', 'selectedArea', 'componentCount'],
      options: {cleanup: 'closing', radius: 1, mapLayer: 'cleaned'}
    },
    {
      id: 'regions',
      title: 'Measure the resulting regions',
      headline: 'A connected patch is defined by a pixel-neighbor rule',
      body: 'Connected components label the cleaned mask using **Pixel connectivity**. Region measurements report selected cells and affine area within paired valid land support.',
      textAlternative:
        'Categorical patch IDs show GPU-connected components, with readouts for selected cells, summed patch area, and largest patch area.',
      evidence:
        '**{{selectedArea}}** are in the current mask; the largest patch is **{{largestPatch}}**.',
      caveat:
        'Component IDs and areas describe this thresholded, accepted-support raster at 60 m. They do not represent official fire polygons or field-validated soil burn severity; [USGS guidance](https://www.usgs.gov/programs/landslide-hazards/hazard-assessment-requirements) calls for field validation for soil-burn severity assessment.',
      controls: ['connectivity', 'mapLayer'],
      readouts: ['selectedCells', 'selectedArea', 'componentCount', 'largestPatch'],
      options: {mapLayer: 'regions', connectivity: '8'}
    },
    {
      id: 'class-contact',
      title: 'Measure contact between valid classes',
      headline: 'This is a class boundary, not a fire perimeter',
      body: 'A GPU relabel pass assigns **outside (0)** and **inside (1)** labels from the selected cleaned mask and excludes invalid support. `GPURasterRegionGraph` counts shared four-neighbor edges and compares mean dNBR across the two classes. The displayed length is shared-edge count × 60 m; its one-round zero-threshold merge check is diagnostic only, and no merged mapping is applied.',
      textAlternative:
        'A two-class map shows accepted land outside or inside the selected cleaned mask; readouts report observed shared-edge length and mean dNBR gap between neighboring classes.',
      evidence:
        '**{{classBoundaryLength}}** observed internal class boundary; mean dNBR gap **{{classMeanDifference}}**.',
      caveat:
        'The edge count excludes invalid support and the raster exterior, so it is not a full perimeter. The two classes reflect the selected threshold and cleanup and do not estimate an official fire boundary or validated severity.',
      controls: ['mapLayer', 'thresholdValue', 'cleanup'],
      readouts: ['classBoundaryLength', 'classMeanDifference', 'classContactStatus'],
      options: {
        mapLayer: 'classContacts',
        thresholdMethod: 'fixed',
        thresholdValue: 0.27,
        cleanup: 'none'
      }
    },
    {
      id: 'bounded-focus',
      title: 'Inspect a bounded focus crop',
      headline: 'Thin structures and inscribed disks answer different questions',
      camera: {longitude: -120.941, latitude: 40.189, zoom: 13.4, transitionMs: 2200},
      body: 'The **64 × 64 cell** focus window is copied from the selected GPU mask at column 176, row 56. Zhang–Suen iteratively thins the selected mask; the medial axis marks centers of maximal inscribed disks and reports their pixel radius in metres. The map can also show the analyst markers and original watershed basins, which flood **−dNBR** over accepted focus support.',
      textAlternative:
        'A 3.84 km square zoom displays the selected-mask skeleton, maximal-disk centers, analyst markers, or original watershed basins. Readouts report accepted support, skeleton convergence and maximum disk radius.',
      evidence:
        '**{{focusSupport}}** accepted focus cells; **{{skeletonPixels}}**; maximum discrete disk radius **{{medialRadius}}**.',
      caveat:
        'These operations run only on 4,096 focus cells. Invalid support and the focus-window exterior are barriers or background; the crop censors disks and cuts features at its edge. Thinning can erase tiny patterns and does not guarantee every component survives; see the [published Zhang–Suen critique](https://www.csie.ntu.edu.tw/~hil/paper/cacm86.pdf). The markers define spectral basins, not drainage areas.',
      controls: ['mapLayer', 'cleanup', 'radius'],
      readouts: ['focusSupport', 'skeletonPixels', 'medialRadius'],
      options: {mapLayer: 'skeleton', cleanup: 'none', connectivity: '8'}
    },
    {
      id: 'basin-merging',
      title: 'Compare basin merge rules',
      headline: 'Merge thresholds change how many spectral basins remain',
      camera: {longitude: -120.941, latitude: 40.189, zoom: 13.4, transitionMs: 1000},
      body: 'The same 16-or-fewer marker basins feed `GPURasterRegionGraph`. **Fixed** mode propagates edges measured from original basin means; **adaptive** mode recomputes count-weighted means and chooses one closest eligible pair per round. The map shows the GPU mapping applied to the watershed labels, with the mean-dNBR gap and round budget set here.',
      textAlternative:
        'The bounded focus crop colors each watershed cell by the GPU-computed merged basin root. Readouts give the number of roots, marker count, accepted support and convergence status; the details panel also lists per-marker counts, label checksum and final mapping.',
      evidence: '**{{mergedBasinCount}}** after the selected merge rule; **{{watershedStatus}}**.',
      caveat:
        'Only the 64 × 64 accepted-support crop is analyzed. Invalid cells and the crop exterior interrupt adjacency. Basin merges summarize dNBR means within analyst-seeded spectral regions; they are not drainage catchments, burn perimeters or severity classes.',
      controls: [
        'mapLayer',
        'connectivity',
        'regionMergeMode',
        'regionMergeThreshold',
        'regionMergeRounds'
      ],
      readouts: ['focusSupport', 'focusMarkers', 'mergedBasinCount', 'watershedStatus'],
      options: {
        mapLayer: 'watershedMerged',
        cleanup: 'none',
        connectivity: '4',
        regionMergeMode: 'fixed',
        regionMergeThreshold: 0.2,
        regionMergeRounds: 16
      }
    }
  ],
  snippet: state => {
    const thresholdValue =
      state.thresholdOperation === 'range'
        ? `[${Math.min(state.thresholdValue, state.rangeEndpoint)}, ${Math.max(state.thresholdValue, state.rangeEndpoint)}]`
        : state.thresholdMethod === 'fixed'
          ? state.thresholdValue
          : state.thresholdMethod === 'source-probe'
            ? 'gpuSourceCellValue'
            : 'otsuThreshold';
    return `const threshold = ${thresholdValue};
const mask = new GPURasterThreshold({
  width,
  height,
  input: dNBRBand,
  threshold,
  operation: '${state.thresholdOperation}',
  inclusive: ${state.thresholdInclusive},
  output: maskView
});
const patches = new GPURasterConnectedComponents({
  width,
  height,
  input: maskBand,
  connectivity: ${state.connectivity},
  output: labels,
  outputValidity,
  converged
});`;
  },
  about: {
    what: 'An observed Sentinel-2 spectral-change mask becomes a configurable raster topology.',
    why: 'Thresholds and neighborhood rules alter whether nearby changed pixels become one connected patch.',
    howToRead:
      'The map and summaries are GPU outputs. Treat thresholds and cleanup as sensitivity choices; areas describe only accepted support and this scene does not validate burn severity or a perimeter.'
  },
  create: async ctx => (await import('./fire-scar-topology.compute')).createFireScarTopology(ctx)
});
