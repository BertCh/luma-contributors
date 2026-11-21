// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {
  LIDAR_GROUND_DISPLAY_CATEGORIES,
  LIDAR_FEATURE_DISPLAY_CATEGORIES
} from './point-cloud-ground-support';

export type LidarGroundOptions = {
  view: 'source' | 'outliers' | 'ground' | 'height' | 'features' | 'voxels';
  pointSize: number;
};

export default defineScene<LidarGroundOptions>({
  id: 'lidar-ground',
  title: 'What is ground in a lidar cloud?',
  chapter: 'terrain',
  order: 7,
  summary:
    'Follow one bounded USGS 3DEP point cloud through neighborhood filtering, progressive ground classification, height above ground, local shape features, and a GPU voxel reduction.',
  contributors: [
    'GPUPointCloudNeighborhoods',
    'GPUPointCloudStatisticalFilter',
    'GPUPointCloudSurfaces',
    'GPUPointCloudGround',
    'GPUPointCloudHeight',
    'GPUPointCloudFeatures',
    'GPUPointCloudCoverageValidity',
    'GPUPointCloudVoxelDownsample'
  ],
  datasets: [{id: 'lidar-ground', role: 'classified lidar returns'}],
  initialView: {
    longitude: -122.33977,
    latitude: 47.59589,
    zoom: 18.8,
    pitch: 0,
    bearing: -20
  },
  basemap: ground('relief'),
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Point display',
      group: 'Analysis result',
      apply: 'param',
      default: 'source',
      display: 'chips',
      help: 'Switches the point colour or sample to the named GPU result. The coordinates and camera stay fixed so each stage can be compared in place.',
      options: [
        {value: 'source', label: 'LAS classes'},
        {value: 'outliers', label: 'Outliers'},
        {value: 'ground', label: 'Ground'},
        {value: 'height', label: 'Height'},
        {value: 'features', label: 'Shape support'},
        {value: 'voxels', label: 'Voxel sample'}
      ]
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Point radius',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 5,
      step: 0.5,
      default: 1,
      unit: 'px',
      help: 'Changes only the point mark size. Use smaller marks to see local density and larger marks to inspect class boundaries.'
    }
  ],
  readouts: [
    {id: 'featureSupport', label: 'Shape diagnostics', layout: 'block'},
    {id: 'filterSupport', label: 'Filter support', layout: 'block'},
    {
      id: 'samplePoints',
      label: 'Retained source returns',
      format: 'integer',
      emphasis: 'tile',
      help: 'Original LAS records after clipping and deterministic one-per-cell sampling.'
    },
    {
      id: 'voxelPoints',
      label: 'Occupied 2 m voxels',
      format: 'integer',
      emphasis: 'tile',
      help: 'GPUPointCloudVoxelDownsample writes one representative per occupied 3D voxel.'
    }
  ],
  story: [
    {
      id: 'source-cloud',
      title: 'Start with the measured returns',
      headline: 'Original returns retain their recorded LAS classes',
      textAlternative:
        'Original USGS lidar returns appear as small class-coloured points across a compact King County sample.',
      body: 'These dots are original USGS 3DEP lidar returns from a 180 m square in King County. The colour follows the LAS classification code: ground, unclassified, water and noise remain distinct in the source file.',
      evidence: '**{{samplePoints}}** records remain after clipping and fixed 0.5 m sampling.',
      caveat:
        'One retained return per occupied cell is a bounded display sample; it is not the full 6.37 million point tile.',
      options: {view: 'source'},
      controls: ['view', 'pointSize'],
      readouts: ['samplePoints']
    },
    {
      id: 'sampling-outliers',
      title: 'Ask what has enough neighbours',
      headline: 'Unusually sparse returns separate from the main cloud',
      textAlternative:
        'Statistical filtering highlights returns whose mean distance to nearby points is unusually large.',
      body: '`GPUPointCloudNeighborhoods` finds nearby returns in the bounded 3D grid using a 2.5 m radius and 16-neighbour cap. `GPUPointCloudStatisticalFilter` compares each return’s mean retained-neighbour distance with the cloud-wide distribution. Truncated rows use their nearest 16 retained neighbours; a complete threshold describes that retained sample, not uncapped or certified halo support.',
      evidence:
        'Red marks are sparse outliers among retained neighbours; green marks have typical retained support. Suppressed classifications are hidden while the support readout explains the loss.',
      caveat:
        'Sparse edges and genuine isolated objects can look like outliers. The 2.5 m radius and 16-neighbour cap define this result. Reported invalid or incomplete support suppresses all classifications; the support readout distinguishes this from outliers. The clipped sample halo remains uncertified.',
      options: {view: 'outliers'},
      controls: ['view'],
      readouts: ['filterSupport'],
      stage: 'neighbourhoods'
    },
    {
      id: 'find-ground',
      title: 'Separate the ground from objects',
      headline: 'Class-2 seed returns constrain the progressive ground filter',
      textAlternative:
        'The ground filter separates complete accepted and rejected results from provisional or unsupported returns.',
      body: '`GPUPointCloudSurfaces` first builds a cell minimum from returns already marked as LAS ground. `GPUPointCloudGround` applies a radius-one morphological opening and the slope threshold. Its mask and support status jointly determine the displayed category; only complete accepted rows seed the recovered terrain.',
      evidence:
        'Green points are accepted with complete support; coral points are rejected with complete support. Gold marks provisional incomplete windows and gray marks missing seed support or invalid points.',
      caveat:
        'This is a bounded square-window PMF. It does not replace survey-specific ground classification or preserve every breakline.',
      options: {view: 'ground'},
      controls: ['view'],
      stage: 'ground'
    },
    {
      id: 'relative-height',
      title: 'Measure height from the local ground',
      headline: 'Supported ground cells turn elevation into local height',
      textAlternative:
        'A cividis ramp shows the difference between each return and the local classified-ground cell minimum; gold marks missing or invalid height values.',
      body: '`GPUPointCloudHeight` subtracts the cell terrain estimate from every return. The cividis ramp shows height above ground in metres; the same colour at two places means the same local height even when their absolute elevations differ.',
      evidence:
        'Bright returns rise farther above the accepted ground surface; the ramp clips at 25 m to keep low vegetation visible. Gold marks unsupported, outside-grid or invalid height results.',
      caveat:
        'The terrain surface is seeded only where LAS class-2 returns occur. Cells without classified ground have no terrain estimate, and negative relative heights are preserved by the contributor.',
      options: {view: 'height'},
      controls: ['view'],
      stage: 'height'
    },
    {
      id: 'local-shape',
      title: 'Use neighbourhood shape as evidence',
      headline: 'Quality flags show where local shape estimates need caution',
      textAlternative:
        'Green marks usable local fits, gold incomplete support, orange truncation, gray degeneracy and coral failed eigensolves. Halo completeness is not certified.',
      body: '`GPUPointCloudFeatures` fits the local covariance and estimates a normal plus linearity, planarity and scattering. Both the feature fit and statistical filter receive row-aligned neighborhoodStatus. A usable fit with incomplete support is distinct from truncation, degeneracy and a failed eigensolve. Overlapping causes remain in the diagnostic counts; the map prioritises failed fits, then truncation, then incomplete support.',
      evidence: '{{featureSupport}}',
      caveat:
        'Feature shape depends on the 2.5 m radius and the capped neighbourhood. Halo completeness is not certified: the shipped clipped, one-per-cell sample supplies no authoritative supportBounds. A clear status therefore certifies only the local fit, not the source halo.',
      options: {view: 'features'},
      controls: ['view'],
      readouts: ['featureSupport'],
      stage: 'features'
    },
    {
      id: 'voxel-limit',
      title: 'Choose a scale before sending points onward',
      headline: 'Two-metre voxels trade fine detail for a smaller point set',
      textAlternative:
        'Sparser LAS-class-coloured marks show one representative per occupied 2 m voxel, with the GPU-computed count beside the source sample count.',
      body: '`GPUPointCloudVoxelDownsample` chooses the lowest source row id as one representative in each occupied 2 m cube. The map draws this compact set, and the count comes from the GPU-written indirect draw record.',
      evidence:
        '**{{voxelPoints}}** occupied voxels stand in for **{{samplePoints}}** sampled returns.',
      caveat:
        'A representative preserves neither all surfaces nor a statistical average; voxel size is a deliberate limit on spatial detail.',
      options: {view: 'voxels'},
      controls: ['view'],
      readouts: ['voxelPoints', 'samplePoints'],
      stage: 'voxel'
    }
  ],
  pipeline: [
    {
      id: 'neighbourhoods',
      label: 'Find neighbors',
      detail: 'Build bounded 2.5 m 3D neighbourhoods and flag unusually sparse support.',
      show: {option: 'view', value: 'outliers'}
    },
    {
      id: 'ground',
      label: 'Classify ground',
      detail: 'Seed cell minima from LAS class-2 returns, then apply progressive morphology.',
      show: {option: 'view', value: 'ground'}
    },
    {
      id: 'height',
      label: 'Measure height',
      detail: 'Subtract supported classified-ground elevation from each point.',
      show: {option: 'view', value: 'height'}
    },
    {
      id: 'features',
      label: 'Check shape',
      detail: 'Estimate covariance features and mark weak neighbourhood support.',
      show: {option: 'view', value: 'features'}
    },
    {
      id: 'voxel',
      label: 'Reduce detail',
      detail: 'Keep one GPU-selected point in each occupied 2 m voxel.',
      show: {option: 'view', value: 'voxels'}
    }
  ],
  furniture: {title: true, credit: true, scaleBar: true},
  legends: state => {
    switch (state.view) {
      case 'outliers':
        return [
          {
            id: 'point-status',
            title: 'Retained mean neighbour distance',
            kind: 'categories',
            entries: [
              {label: 'Sparse outlier (retained neighbours)', color: [204, 109, 97]},
              {label: 'Typical retained support', color: [150, 211, 166]}
            ]
          }
        ];
      case 'ground':
        return [
          {
            id: 'ground-class',
            title: 'Progressive ground filter',
            kind: 'categories',
            entries: LIDAR_GROUND_DISPLAY_CATEGORIES.map(({label, color}) => ({label, color}))
          }
        ];
      case 'height':
        return [
          {
            id: 'height-above-ground',
            title: 'Height above ground',
            kind: 'ramp',
            ramp: 'cividis',
            extent: [0, 25],
            range: [0.15, 1],
            unit: 'm',
            note: 'Gold: unsupported, outside-grid or invalid height'
          }
        ];
      case 'features':
        return [
          {
            id: 'feature-support',
            title: 'Local feature-fit status',
            kind: 'categories',
            entries: LIDAR_FEATURE_DISPLAY_CATEGORIES.map(({label, color}) => ({label, color}))
          }
        ];
      case 'source':
      case 'voxels':
        return [
          {
            id: 'las-classes',
            title:
              state.view === 'voxels' ? 'LAS class of each representative' : 'LAS classification',
            kind: 'categories',
            entries: [
              {label: 'Unclassified (1)', color: [232, 174, 94]},
              {label: 'Ground (2)', color: [146, 206, 164]},
              {label: 'Low noise (7)', color: [229, 105, 99]},
              {label: 'Water (9)', color: [92, 156, 215]},
              {label: 'High noise (18)', color: [223, 98, 94]},
              {label: 'Ignored ground (20)', color: [185, 183, 175]},
              {label: 'Other LAS classes', color: [150, 155, 160]}
            ]
          }
        ];
    }
  },
  about: {
    what: 'The scene uses the USGS King County 2021 tile 2282, clipped to 180 m and sampled to one original return per occupied 0.5 m cell. It keeps LAS classification and NAVD88 Geoid18 elevation, converted from US survey feet to metres.',
    why: 'The stages answer practical point-cloud questions: which points have sparse support, which belong to ground, how high objects stand above ground, whether neighbourhood shape is reliable, and how many representatives remain at a chosen voxel scale.',
    howToRead:
      'Ground colors distinguish complete acceptance/rejection, provisional windows and unsupported points. A radius-one opening retains supported terrain on this sparse class-2 seed grid; only complete accepted ground seeds height. The last step draws only the GPU-selected representative row from every occupied 2 m cube. Hover the data attribution for the source catalogue and public-domain terms.'
  },
  snippet: state => `import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {
  GPUPointCloudNeighborhoods, GPUPointCloudStatisticalFilter,
  GPUPointCloudGround, GPUPointCloudHeight, GPUPointCloudFeatures,
  GPUPointCloudSurfaces, GPUPointCloudCoverageValidity, getRasterMetadataFromPointCloudGrid
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {GPUElementwise, GPUCompaction} from '@luma.gl/graph/gpu-core';
import {createSupportedGroundClassesNode, createGroundDisplayClassesNode} from './point-cloud-ground-support';

// One bounded USGS return sample, analysed in local metres.
// No supportBounds: clipping/sampling bounds do not certify a complete source halo.
graph.add(new GPUPointCloudNeighborhoods({positions, radius: 2.5, capacity: 16,
  grid: neighborhoodGrid, output: {indices, squaredDistances, counts, status: neighborhoodStatus}}));
graph.add(new GPUPointCloudStatisticalFilter({squaredDistances, counts, neighborhoodStatus,
  capacity: 16, output: {meanDistances, validity: inliers, thresholdComplete}}));
graph.add(new GPUPointCloudSurfaces({id: 'lidar-ground-seed-surfaces', positions, classifications, grid,
  output: {terrain: minimumSurface, surface: seedSurface, canopy: seedCanopy, coverage: seedCoverage}}));
graph.add(new GPUPointCloudCoverageValidity({id: 'lidar-ground-seed-support', coverage: seedCoverage,
  output: {surfaceValidity: seedSurfaceValidity, terrainValidity: groundSeedCoverage, canopyValidity: seedCanopyValidity}}));
graph.add(new GPUPointCloudGround({positions, classifications, minimumSurface, coverage: groundSeedCoverage, grid, radii: [1], initialThreshold: 0.3, slope: 0.1, maximumThreshold: 2, output: {groundMask, outputClassifications, status: groundStatus}}));
graph.add(createSupportedGroundClassesNode(graph, outputClassifications, groundStatus, supportedClassifications));
graph.add(createGroundDisplayClassesNode(graph, groundMask, groundStatus, groundDisplay));
graph.add(new GPUPointCloudSurfaces({id: 'lidar-ground-classified-surfaces', positions, classifications: supportedClassifications, grid,
  output: {terrain, surface, canopy, coverage}}));
graph.add(new GPUPointCloudCoverageValidity({id: 'lidar-ground-raster-validity', coverage, output: {surfaceValidity, terrainValidity, canopyValidity}}));
const terrainBand = makeGPURasterBand(terrain, getRasterMetadataFromPointCloudGrid(grid), {validity: terrainValidity});
if (!terrainBand.validity) throw new Error('lidar-ground terrain band must carry ground validity');
// Zero metres is valid ground wherever terrainValidity is 1; occupancy alone is insufficient.
graph.add(new GPUPointCloudHeight({positions, grid, terrain: terrainBand.storage.values,
  coverage: terrainBand.validity, coverageBit: 1, output: {heights, status: heightStatus}}));
graph.add(new GPUPointCloudFeatures({positions, indices, counts, neighborhoodStatus, capacity: 16, output: {normals, eigenvalues, shapes, status: featureStatus}}));
graph.add(new GPUPointCloudVoxelDownsample({voxelSize: 2, gridSize: [110, 110, 45], ...voxelBuffers}));
graph.add(new GPUElementwise({input: voxelCounts, inputB: ones, output: occupiedFlags, operation: 'min'}));
graph.add(new GPUCompaction({input: representatives, flags: occupiedFlags, output: compactIndices, count: drawCount}));

// Current point display: ${state.view}; point radius: ${state.pointSize} px.`,
  create: async ctx => (await import('./lidar-ground.compute')).createLidarGround(ctx)
});
