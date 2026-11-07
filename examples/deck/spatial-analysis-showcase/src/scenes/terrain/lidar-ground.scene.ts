// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';

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
      body: '`GPUPointCloudNeighborhoods` finds nearby returns in the bounded 3D grid using a 2.5 m radius and 16-neighbour cap. `GPUPointCloudStatisticalFilter` compares each return’s mean neighbour distance with the cloud-wide distribution; red marks have unusually sparse support.',
      evidence:
        'The red marks are rows rejected by the statistical filter; green marks have typical local support.',
      caveat:
        'Sparse edges and genuine isolated objects can look like outliers. The 2.5 m radius and 16-neighbour cap define this result.',
      options: {view: 'outliers'},
      controls: ['view'],
      stage: 'neighbourhoods'
    },
    {
      id: 'find-ground',
      title: 'Separate the ground from objects',
      headline: 'Class-2 seed returns constrain the progressive ground filter',
      textAlternative:
        'A progressive morphological ground filter marks returns that remain close to its seeded ground surface.',
      body: '`GPUPointCloudSurfaces` first builds a cell minimum from returns already marked as LAS ground. `GPUPointCloudGround` applies progressive morphological openings and the slope threshold, then writes the resulting ground mask and updated class codes.',
      evidence:
        'Green points are accepted by the GPU ground filter; coral points were rejected because they exceeded a local threshold or lacked seeded ground support.',
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
        'Gold points have incomplete, truncated, degenerate or unconverged local covariance estimates; pale green points have complete support.',
      body: '`GPUPointCloudFeatures` fits the local covariance and estimates a normal plus linearity, planarity and scattering. The overlay marks neighbourhoods whose feature estimate is incomplete or truncated, where a shape label would be weak evidence.',
      evidence:
        'Gold indicates a neighborhood status flag; pale green marks rows with a complete feature estimate.',
      caveat:
        'Feature shape depends on the 2.5 m radius and the capped neighbourhood. It describes local geometry, not object identity.',
      options: {view: 'features'},
      controls: ['view'],
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
            title: 'Mean neighbour distance',
            kind: 'categories',
            entries: [
              {label: 'Sparse outlier', color: [204, 109, 97]},
              {label: 'Typical support', color: [150, 211, 166]}
            ]
          }
        ];
      case 'ground':
        return [
          {
            id: 'ground-class',
            title: 'Progressive ground filter',
            kind: 'categories',
            entries: [
              {label: 'Rejected as object / non-ground', color: [204, 109, 97]},
              {label: 'Accepted ground', color: [150, 211, 166]}
            ]
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
            title: 'Feature support',
            kind: 'categories',
            entries: [
              {label: 'Complete neighbourhood', color: [150, 211, 166]},
              {label: 'Incomplete, truncated or failed', color: [233, 181, 101]}
            ]
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
      'Ground and relative-height colours are GPU outputs. The last step draws only the GPU-selected representative row from every occupied 2 m cube. Hover the data attribution for the source catalogue and public-domain terms.'
  },
  snippet:
    state => `import {GPUPointCloudGround, GPUPointCloudHeight, GPUPointCloudFeatures, GPUPointCloudVoxelDownsample} from '@luma-contributors/graph/gpu-pointcloud';
import {GPUElementwise, GPUCompaction} from '@luma-contributors/graph/gpu-core';

// One bounded USGS return sample, analysed in local metres.
graph.add(new GPUPointCloudGround({grid, radii: [1, 2, 4], initialThreshold: 0.3, slope: 0.1, maximumThreshold: 2, ...groundBuffers}));
graph.add(new GPUPointCloudHeight({grid, ...terrainBuffers, ...heightBuffers}));
graph.add(new GPUPointCloudFeatures({capacity: 16, ...neighborhoodBuffers, ...featureBuffers}));
graph.add(new GPUPointCloudVoxelDownsample({voxelSize: 2, gridSize: [110, 110, 45], ...voxelBuffers}));
graph.add(new GPUElementwise({input: voxelCounts, inputB: ones, output: occupiedFlags, operation: 'min'}));
graph.add(new GPUCompaction({input: representatives, flags: occupiedFlags, output: compactIndices, count: drawCount}));

// Current point display: ${state.view}; point radius: ${state.pointSize} px.`,
  create: async ctx => (await import('./lidar-ground.compute')).createLidarGround(ctx)
});
