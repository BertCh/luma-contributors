// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {HabitatConnectivityOptions} from './habitat-connectivity.compute';
import {HABITAT_PATCH_COLORS, WORLD_COVER_CLASSES} from './habitat-connectivity-style';

export default defineScene<HabitatConnectivityOptions>({
  id: 'habitat-connectivity',
  title: 'Dixie Fire: land-cover connectivity scenarios',
  chapter: 'raster',
  order: 4,
  summary:
    'ESA WorldCover classes in a bounded Dixie Fire window become connected patches, interior-core candidates and alternative raster walking-cost scenarios. These maps describe land cover and explicit model assumptions; they do not measure habitat suitability or animal movement.',
  contributors: [
    'GPURasterConnectedComponents',
    'GPURasterDenseComponents',
    'GPURasterPatchMetrics',
    'addLandscapeMetricsRecipe',
    'addPatchProximityRecipe',
    'GPURasterErosion',
    'GPURasterWalkingCost',
    'addRasterResistanceConnectivityRecipe',
    'addResistanceConnectivityRecipe',
    'addSparseResistanceConnectivityRecipe'
  ],
  datasets: [{id: 'dixie-fire', role: 'ESA WorldCover and USGS elevation'}],
  initialView: {longitude: -120.9511, latitude: 40.1396, zoom: 14.8},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Land-cover connectivity scenarios',
      subtitle: 'WorldCover · Dixie Fire near Greenville'
    },
    scaleBar: {units: 'metric'},
    credit: 'ESA WorldCover 2021 (CC BY 4.0); USGS elevation',
    caveat:
      'WorldCover is a 10 m classification resampled to the 20 m analysis grid. Connectivity is land-cover topology; walking cost is a chosen scenario, not observed movement or habitat suitability.'
  },
  options: [
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'cover',
      options: [
        {value: 'cover', label: 'WorldCover classes'},
        {value: 'mask', label: 'Selected land-cover mask'},
        {value: 'core', label: 'Interior source cells'},
        {value: 'patches', label: 'Connected patches'},
        {value: 'cost', label: 'Scenario walking cost'},
        {value: 'circuit', label: 'Circuit potential / current'}
      ]
    },
    {
      kind: 'toggle',
      id: 'countRasterBorder',
      label: 'Count outer crop faces',
      group: 'Mask and patch definition',
      apply: 'compile',
      default: true,
      help: 'When enabled, selected patch faces along the edge of this 64 × 64 window count toward perimeter. When disabled, only those outside-window faces are omitted; boundaries against other classes and invalid cells still count.'
    },
    {
      kind: 'select',
      id: 'habitat',
      label: 'Land-cover group',
      group: 'Mask and patch definition',
      apply: 'param',
      default: 'tree',
      help: 'Selects ESA WorldCover classes for a binary map mask. This is a transparent land-cover grouping, not a habitat classification.',
      options: [
        {value: 'tree', label: 'Tree cover + mangroves'},
        {value: 'woody', label: 'Tree, shrub + mangroves'},
        {value: 'broad', label: 'Woody, grass, crops, wetland + moss/lichen'}
      ]
    },
    {
      kind: 'slider',
      id: 'coreCells',
      label: 'Interior inset',
      group: 'Core candidate sources',
      apply: 'param',
      min: 0,
      max: 4,
      step: 1,
      default: 1,
      help: 'Keeps selected cells this many 20 m cells inside the selected-group edge using a square neighborhood. This sensitivity map is separate from the fixed one-cell source mask used by the cost scenario.'
    },
    {
      kind: 'select',
      id: 'resistance',
      label: 'Relative resistance',
      group: 'Walking and circuit assumptions',
      apply: 'param',
      default: 'mild',
      help: 'The class values are analyst-chosen relative resistance/friction scenarios. Strong contrast raises built-up, water, bare and cropland penalties more than tree and shrub penalties. Only the walking-cost recipe adds DEM slope.',
      options: [
        {value: 'mild', label: 'Mild contrast'},
        {value: 'strong', label: 'Strong contrast'}
      ]
    },
    {
      kind: 'select',
      id: 'circuitConnectivity',
      label: 'Circuit neighbors',
      group: 'Circuit scenario',
      apply: 'compile',
      default: '8',
      help: 'Four-neighbor connects shared cell edges; eight-neighbor also permits diagonal edges using their longer physical length.',
      options: [
        {value: '4', label: '4 neighbors'},
        {value: '8', label: '8 neighbors'}
      ]
    },
    {
      kind: 'select',
      id: 'circuitBudget',
      label: 'Solver iteration budget',
      group: 'Circuit scenario',
      apply: 'compile',
      default: '512',
      help: 'Maximum configured iteration budget. A separate current-balance certificate determines whether circuit maps and accepted scalars are published; more iterations do not guarantee acceptance under strong contrast.',
      options: [
        {value: '128', label: '128'},
        {value: '512', label: '512'},
        {value: '1024', label: '1024'}
      ]
    },
    {
      kind: 'select',
      id: 'circuitTolerance',
      label: 'Current balance tolerance',
      group: 'Circuit scenario',
      apply: 'compile',
      default: '0.0001',
      help: 'Maximum absolute unit-current conservation residual accepted when publishing circuit outputs.',
      options: [
        {value: '0.001', label: '1e-3'},
        {value: '0.0001', label: '1e-4'}
      ]
    },
    {
      kind: 'toggle',
      id: 'waterBarrier',
      label: 'Water is a barrier',
      group: 'Circuit scenario',
      apply: 'param',
      default: false,
      help: 'Sensitivity switch replacing permanent-water class resistance with a barrier; default uses the documented weighted water resistance.'
    },
    {
      kind: 'select',
      id: 'circuitLayer',
      label: 'Circuit map',
      group: 'Display',
      apply: 'param',
      default: 'potential',
      options: [
        {value: 'potential', label: 'Potential'},
        {value: 'current', label: 'Current density'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85
    }
  ],
  legends: state => {
    if (state.layer === 'cover') {
      return [
        {
          kind: 'categories',
          title: 'ESA WorldCover class',
          entries: WORLD_COVER_CLASSES.map(({code, label, color}, index) => ({
            label: `${label} (${code})`,
            value: index,
            color
          }))
        }
      ];
    }
    if (state.layer === 'patches') {
      return [
        {
          kind: 'categories',
          title: 'Connected land-cover patch',
          entries: HABITAT_PATCH_COLORS.map((_, index) => ({
            label: `Patch colour ${index + 1} (labels repeat)`,
            value: index + 1,
            color: HABITAT_PATCH_COLORS[(index + 1) % HABITAT_PATCH_COLORS.length]
          }))
        }
      ];
    }
    if (state.layer === 'cost') {
      return [
        {
          kind: 'ramp',
          title: 'Relative accumulated walking cost',
          ramp: 'cividis',
          extent: [0, 25000],
          unit: 'relative cost units'
        },
        {
          kind: 'categories',
          title: 'Cost source cells',
          entries: [{label: 'Fixed one-cell patch interior', value: 1, color: [255, 214, 70, 245]}]
        }
      ];
    }
    if (state.layer === 'circuit') {
      return state.circuitLayer === 'potential'
        ? [
            {
              kind: 'ramp',
              id: 'circuit-potential',
              title: 'Circuit potential',
              ramp: 'cividis',
              extent: 'gpu',
              unit: 'relative resistance·metre'
            },
            {
              kind: 'categories',
              title: 'Scenario terminals',
              entries: [
                {label: 'Source', value: 1, color: [255, 214, 70, 255]},
                {label: 'Sink', value: 2, color: [235, 82, 112, 255]}
              ],
              note: 'Analyst-selected valid cells'
            }
          ]
        : [
            {
              kind: 'ramp',
              id: 'circuit-current',
              title: 'Current density',
              ramp: 'magma',
              extent: 'gpu',
              unit: 'unit injection per m²',
              note: 'max(incoming edge-current sum, outgoing edge-current sum) / cell area'
            },
            {
              kind: 'categories',
              title: 'Scenario terminals',
              entries: [
                {label: 'Source', value: 1, color: [255, 214, 70, 255]},
                {label: 'Sink', value: 2, color: [235, 82, 112, 255]}
              ],
              note: 'Analyst-selected valid cells'
            }
          ];
    }
    return [
      {
        kind: 'categories',
        title: 'Selected mask',
        entries: [
          {
            label: state.layer === 'core' ? 'Interior source cell' : 'Selected class cell',
            value: 1,
            color: state.layer === 'core' ? [255, 214, 70, 245] : [48, 190, 112, 235]
          }
        ]
      }
    ];
  },
  readouts: [
    {
      id: 'interiorCells',
      label: 'Displayed interior cells',
      help: 'GPU-eroded cells that remain after the selected square-neighbourhood inset.'
    },
    {
      id: 'patchCount',
      label: 'Connected patches',
      help: 'Dense connected-component count returned by the GPU, bounded by output capacity.'
    },
    {
      id: 'patchPerimeter',
      label: 'Total patch perimeter',
      help: 'GPU sum of per-patch exposed perimeter in metres under the selected crop-edge rule. Background, invalid-cell, and neighboring-patch edges count under either setting.'
    },
    {
      id: 'landscapeCohesion',
      label: 'Landscape cohesion',
      help: 'GPU landscape cohesion index for the selected binary land-cover mask.'
    },
    {
      id: 'patchSeparation',
      label: 'Nearest patch separation',
      help: 'Nearest same-group patch boundary distance within the 2 km search radius; it can be undefined when no other patch is found.'
    },
    {
      id: 'costStatus',
      label: 'Walking-cost status',
      help: 'Whether bounded GPU relaxation converged before publishing cost cells.'
    },
    {
      id: 'circuitStatus',
      label: 'Circuit solve status',
      help: 'Reachability, invalid edges and unit-current convergence.'
    },
    {
      id: 'circuitResidual',
      label: 'Circuit diagnostics',
      help: 'Effective resistance, current-conservation residual, energy and residual norm.'
    },
    {
      id: 'circuitTerminals',
      label: 'Selected terminals',
      help: 'Deterministically selected valid crop cells and their geographic locations; these are scenario terminals.'
    },
    {
      id: 'circuitOracle',
      label: 'Dense / sparse check',
      help: 'Actual GPU dense and sparse agreement on the same 16 × 16 subset and uploaded edges.'
    }
  ],
  story: [
    {
      id: 'land-cover',
      title: 'Start with the classified land cover',
      headline: 'Classes show cover, not habitat quality',
      textAlternative: 'WorldCover classes in a bounded analysis window.',
      body: 'ESA WorldCover labels the window around Greenville. The analysis uses a fixed **64 × 64 cell** crop on the Dixie dataset’s 20 m grid (about 1.28 km across); the source WorldCover classification is 10 m and has been resampled to that grid. Begin by reading the categorical map before grouping classes.',
      options: {layer: 'cover', habitat: 'tree', coreCells: 1, resistance: 'mild'},
      controls: ['layer']
    },
    {
      id: 'mask',
      title: 'Choose a land-cover mask',
      headline: 'The selected classes define the mask',
      textAlternative: 'A binary map isolates the chosen WorldCover group.',
      body: 'The mask follows the selected class grouping exactly: tree cover plus mangroves; that group plus shrubland; or a broader set that also includes grassland, cropland, wetland and moss/lichen. This 64 × 64 crop contains no shrubland cells, so the tree-only and tree-plus-shrub masks coincide here. Built-up, bare, snow and open water remain outside the broad mask. These are reproducible land-cover groupings, not species-specific habitat rules.',
      options: {layer: 'mask'},
      controls: ['habitat', 'layer']
    },
    {
      id: 'patches',
      title: 'Label connected patches',
      headline: 'Touching mask cells form a patch',
      textAlternative: 'GPU connected-component labels colour touching mask cells.',
      body: '`GPURasterConnectedComponents` labels the mask with **eight-neighbour connectivity**; diagonal contact joins cells. Dense labels feed GPU patch area and exposed-face perimeter measurements. Toggle **Count outer crop faces** to compare the total with those window-edge faces included or omitted. Background, invalid-cell and neighboring-patch faces remain exposed under either rule. This is a bounded 64 × 64 window measurement, not evidence of an ecological barrier beyond its edge.',
      options: {layer: 'patches', countRasterBorder: true},
      controls: ['habitat', 'countRasterBorder', 'layer'],
      readouts: ['patchCount', 'patchPerimeter']
    },
    {
      id: 'core-edge',
      title: 'Move the interior boundary',
      headline: 'A wider inset leaves fewer source cells',
      textAlternative: 'Interior source cells recede from the chosen mask edge.',
      body: '`GPURasterErosion` recomputes the selected binary mask at square radii of one to four cells. The interior layer marks cells whose square neighbourhood remains inside the selected mask; radius zero shows the original mask. This is a geometric sensitivity check, not a measured ecological edge depth.',
      options: {layer: 'core', coreCells: 1},
      controls: ['coreCells', 'layer'],
      readouts: ['interiorCells']
    },
    {
      id: 'separation',
      title: 'Use interior cells as cost sources',
      headline: 'Cost accumulates away from source interiors',
      textAlternative: 'A GPU raster shows accumulated modeled cost from interior source cells.',
      body: '`GPURasterWalkingCost` propagates a least-cost surface from the **fixed one-cell core** produced by `addPatchProximityRecipe`; the gold overlay marks those exact source cells. It traverses the full bounded crop using the DEM and selected class-weight table. The square-radius interior view on the previous step is a separate sensitivity map, so its slider does not change these cost sources. The cost map and source overlay are drawn from GPU output buffers.',
      options: {layer: 'cost'},
      controls: ['layer'],
      readouts: ['patchSeparation', 'costStatus']
    },
    {
      id: 'resistance',
      title: 'Change the resistance assumptions',
      headline: 'A different friction table changes the surface',
      textAlternative: 'Mild and strong resistance scenarios use distinct class weights.',
      body: 'The class friction values are explicit analyst assumptions. Mild contrast assigns tree 1, shrub 1.5, grass 2, cropland 3, built-up 8, bare 4, snow 5, water 20, wetland 4, mangrove 2 and moss/lichen 3. Strong contrast assigns those classes 1, 2, 6, 15, 80, 10, 20, 120, 15, 3 and 4 respectively. Built-up, water, bare, cropland and snow penalties rise more sharply, so route patterns can change rather than simply scaling. Elevation adds directional slope cost. Compare both settings to see whether apparent routes depend on these assumptions. Cost convergence is bounded to 128 iterations across the 64 × 64 grid.',
      options: {layer: 'cost', resistance: 'mild'},
      controls: ['resistance', 'layer'],
      caveat:
        'The resistance table, slope coefficients and source definition are scenarios only. No species, movement observations, habitat quality, barriers beyond the raster classes, or validated dispersal distances are supplied.'
    },
    {
      id: 'circuits',
      title: 'Compare circuit connectivity',
      headline: 'Current can divide among alternate paths',
      textAlternative: 'GPU circuit potential or current density over the WorldCover crop.',
      body: 'The main unit-current solve spans the full 64 × 64 crop. Its two deterministic analyst-selected terminal cells lie within the central 16 × 16 subset; gold and magenta mark the source and sink. The separately labeled dense/sparse comparison solves only that 16 × 16 subset. Edge conductance is `2 / (Rₐ + Rᵦ) / physical edge length`; class values use the same mild or strong relative-resistance tables. Potential and current density show current spreading across alternate paths, unlike the least-cost walking surface, which includes DEM slope. Strong resistance contrast also stresses numerical conditioning: the bounded Float32 solve can fail its independent current-balance certificate. In that case both maps and accepted resistance/energy are withheld; the residual remains diagnostic only. Withheld does not mean zero connectivity. Change **Circuit neighbors**, **Relative resistance**, **Water is a barrier**, or **Circuit map** below to test the assumptions. These are analyst scenarios, not animal movement, species habitat, electrical material measurements, or fire-cause evidence.',
      options: {
        layer: 'circuit',
        circuitLayer: 'potential',
        circuitConnectivity: '8',
        circuitBudget: '512',
        circuitTolerance: '0.0001',
        waterBarrier: false
      },
      controls: ['circuitConnectivity', 'resistance', 'waterBarrier', 'circuitLayer'],
      readouts: ['circuitTerminals', 'circuitStatus', 'circuitResidual', 'circuitOracle'],
      caveat:
        'Terminals are chosen scenario points, not observed origins. No calibrated ohms or amperes are claimed; isotropic circuit edges do not include slope. A failed certificate is a numerical limitation, not evidence of a disconnected landscape. The separately labeled 16 × 16 comparison does not certify the full 64 × 64 solve.'
    }
  ],
  create: async ctx =>
    (await import('./habitat-connectivity.compute')).createHabitatConnectivity(ctx)
});
