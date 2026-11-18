// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';

export type PointCloudSupportOptions = {
  view: 'source' | 'radius' | 'smrf' | 'octree' | 'merge';
  radius: number;
  minimumNeighbors: number;
  smrfWindow: number;
  smrfSlope: number;
  maximumDepth: number;
  nodePointLimit: number;
  pointSize: number;
};

export default defineScene<PointCloudSupportOptions>({
  id: 'point-cloud-support',
  title: 'How much support does a point cloud have?',
  chapter: 'terrain',
  order: 8,
  summary:
    'Inspect neighborhood support, SMRF ground sensitivity, additive octree ownership and voxel statistic merging in a deterministic 2,000-return sample from the bundled USGS 3DEP lidar scene.',
  contributors: [
    'GPUPointCloudRadiusFilter',
    'GPUPointCloudSMRF',
    'GPUPointCloudOctree',
    'GPUPointCloudVoxelDownsample',
    'GPUPointCloudVoxelMerge'
  ],
  datasets: [{id: 'lidar-ground', role: 'systematic sample of original classified lidar returns'}],
  initialView: {longitude: -122.33977, latitude: 47.59589, zoom: 18.8, pitch: 0, bearing: -20},
  basemap: ground('night'),
  annotations: [
    {
      kind: 'point',
      coordinate: [-122.33977, 47.59589],
      text: 'King County',
      detail: 'USGS 3DEP sample',
      rank: 'context',
      tone: 'muted'
    }
  ],
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Point display',
      group: 'Analysis result',
      apply: 'param',
      default: 'source',
      display: 'chips',
      help: 'Switches between original LAS classes and the GPU outputs while keeping the same sampled coordinates in view.',
      options: [
        {value: 'source', label: 'Source classes'},
        {value: 'radius', label: 'Radius support'},
        {value: 'smrf', label: 'SMRF ground'},
        {value: 'octree', label: 'Octree ownership'},
        {value: 'merge', label: 'Merged voxels'}
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Support radius',
      group: 'Radius filter',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 0.5,
      default: 6,
      unit: 'm',
      help: 'Sets the exact 3D radius used to count other sampled returns for each point. The all-pairs scan is bounded to 2,000 rows.'
    },
    {
      kind: 'slider',
      id: 'minimumNeighbors',
      label: 'Minimum neighbors',
      group: 'Radius filter',
      apply: 'compile',
      min: 1,
      max: 24,
      step: 1,
      default: 4,
      help: 'Rows with fewer neighbors inside the chosen radius are marked as lacking local support. The row itself is excluded from its count.'
    },
    {
      kind: 'slider',
      id: 'smrfWindow',
      label: 'Maximum morphology window',
      group: 'SMRF ground filter',
      apply: 'compile',
      min: 4,
      max: 24,
      step: 2,
      default: 12,
      unit: 'm',
      help: 'Sets the largest progressive morphological window. Larger windows can bridge broader terrain changes and may absorb low objects.'
    },
    {
      kind: 'slider',
      id: 'smrfSlope',
      label: 'Slope threshold',
      group: 'SMRF ground filter',
      apply: 'compile',
      min: 0.05,
      max: 0.5,
      step: 0.05,
      default: 0.15,
      help: 'Controls the local elevation tolerance used by SMRF when separating ground from elevated returns.'
    },
    {
      kind: 'slider',
      id: 'maximumDepth',
      label: 'Maximum octree depth',
      group: 'Additive octree',
      apply: 'compile',
      min: 3,
      max: 7,
      step: 1,
      default: 5,
      help: 'Adds finer levels to the sparse spatial hierarchy. Points remain owned exactly once across levels; deeper levels mean smaller spatial nodes.'
    },
    {
      kind: 'slider',
      id: 'nodePointLimit',
      label: 'Maximum owned points per node',
      group: 'Additive octree',
      apply: 'compile',
      min: 8,
      max: 64,
      step: 8,
      default: 24,
      help: 'Sets the maximum sampled point count allowed in a nonterminal node before subdivision. Leaves retain their remaining rows; this changes hierarchy structure, not total source rows.'
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
      default: 2.5,
      unit: 'px',
      help: 'Changes only the on-map point mark size.'
    }
  ],
  readouts: [
    {
      id: 'sampleRows',
      label: 'Sampled source rows',
      format: 'integer',
      emphasis: 'tile',
      help: 'A deterministic systematic sample of the already-bounded USGS dataset.'
    },
    {
      id: 'radiusSupported',
      label: 'Rows meeting support rule',
      format: 'integer',
      emphasis: 'tile',
      help: 'GPU-counted source rows whose exact-radius neighbor count meets the selected minimum.'
    },
    {
      id: 'smrfGround',
      label: 'Rows marked ground by SMRF',
      format: 'integer',
      emphasis: 'tile',
      help: 'GPU classifications from this sample and the selected radius support mask.'
    },
    {
      id: 'smrfUnresolved',
      label: 'Rows without terrain or excluded',
      format: 'integer',
      help: 'Rows flagged outside the surface, without terrain anchors, or excluded by the radius selection; edge and inpainted estimates are reported separately in the chart.'
    },
    {
      id: 'octreeOwned',
      label: 'Rows owned across octree levels',
      format: 'integer',
      emphasis: 'tile',
      help: 'Sum of GPU node-owned row counts. It should equal the sample size when all positions are in the declared cube.'
    },
    {
      id: 'octreeOverflow',
      label: 'Octree overflow status',
      format: 'text',
      hood: true,
      help: 'Any nonzero overflow invalidates the node and row ownership outputs.'
    },
    {
      id: 'mergeRows',
      label: 'Rows represented by merged voxels',
      format: 'integer',
      emphasis: 'tile',
      help: 'Sum of GPU merged voxel populations from two disjoint source-row partitions.'
    },
    {
      id: 'mergeCountCheck',
      label: 'Partition merge count check',
      format: 'text',
      help: 'Compares per-cell merged counts against an independent single-cloud GPU voxel reduction.'
    },
    {
      id: 'mergeRepresentativeMismatch',
      label: 'Representative mismatches',
      format: 'decimal',
      help: 'Number of occupied cells whose merged minimum-source-row representative differs from the single-cloud GPU reduction, read back from both GPU results.'
    },
    {
      id: 'mergeStatus',
      label: 'Voxel merge status',
      format: 'text',
      hood: true,
      help: 'Nonzero per-cell flags mean count or representative overflow and invalidate that cell.'
    },
    {
      id: 'radiusChart',
      label: 'Exact-radius neighbor counts',
      kind: 'chart',
      help: 'GPU readback grouped by number of other sample rows inside the selected radius.'
    },
    {
      id: 'smrfChart',
      label: 'SMRF classification support',
      kind: 'chart',
      help: 'GPU output classes and status flags grouped after applying the radius mask.'
    },
    {
      id: 'octreeChart',
      label: 'Additive octree by level',
      kind: 'chart',
      help: 'Node counts and uniquely owned rows are derived from the GPU node records.'
    },
    {
      id: 'mergeChart',
      label: 'Merged and direct voxel totals',
      kind: 'chart',
      help: 'GPU readback compares two-part sufficient statistics with one full-sample GPU reduction.'
    }
  ],
  story: [
    {
      id: 'sampled-returns',
      textAlternative:
        'Class-colored lidar returns mark a compact King County sample on a dark basemap.',
      title: 'Begin with a bounded set of measured returns',
      headline: 'Two thousand original rows keep the all-pairs question tractable',
      body: 'The source is already a bounded USGS 3DEP sample with LAS classifications. This study takes a deterministic, evenly spaced row sample from it; the displayed classes remain the recorded source labels.',
      evidence: '**{{sampleRows}}** rows enter every GPU operation in this study.',
      caveat:
        'The systematic sample preserves original records but is not a spatially uniform sample of the full tile. It does not represent all 116,213 rows in the bundled scene.',
      options: {view: 'source'},
      controls: ['view', 'pointSize'],
      readouts: ['sampleRows']
    },
    {
      id: 'local-support',
      textAlternative:
        'Point colors show which source rows meet the exact 3D radius and minimum-neighbor rule.',
      title: 'Count exact 3D neighbors around each return',
      headline: 'Support depends on the radius and the minimum neighbor count',
      body: '`GPUPointCloudRadiusFilter` checks every other sampled return against the selected inclusive 3D radius. The color marks whether each row meets the minimum; the chart shows the full neighbor-count distribution read back from the GPU.',
      evidence: '**{{radiusSupported}}** rows meet the current support rule.',
      caveat:
        'This exact all-pairs comparison is practical here because the sample is bounded to 2,000 rows. Sparse support alone does not establish that a return is noise.',
      options: {view: 'radius'},
      controls: ['view', 'radius', 'minimumNeighbors'],
      readouts: ['radiusSupported', 'radiusChart'],
      stage: 'radius'
    },
    {
      id: 'ground-sensitivity',
      textAlternative:
        'The map colors SMRF ground classifications separately from other supported returns and amber edge or unresolved rows.',
      title: 'Change the scale of a ground hypothesis',
      headline: 'SMRF classifies only where selected returns support a terrain surface',
      body: '`GPUPointCloudSMRF` builds a terrain surface from the radius-supported rows, then applies progressive morphology. The map distinguishes rows marked ground from other selected rows; amber marks unresolved or excluded support.',
      evidence:
        '**{{smrfGround}}** rows are marked ground; **{{smrfUnresolved}}** have no usable ground support.',
      caveat:
        'Original LAS class labels are observations, not independent ground truth. This bounded GPU implementation and sample do not establish survey-grade ground accuracy.',
      options: {view: 'smrf'},
      controls: ['view', 'smrfWindow', 'smrfSlope'],
      readouts: ['smrfGround', 'smrfUnresolved', 'smrfChart'],
      stage: 'smrf'
    },
    {
      id: 'additive-octree',
      textAlternative:
        'Point color identifies the octree level that owns each row; the adjacent chart counts nodes and uniquely owned rows by level.',
      title: 'Read the octree as an additive partition',
      headline: 'Subdivision changes ownership depth while retaining the sampled rows',
      body: '`GPUPointCloudOctree` sorts rows into a bounded Morton hierarchy. The color follows the level at which each GPU node owns a row, and the chart reports node counts and uniquely owned rows at every level.',
      evidence: '**{{octreeOwned}}** row assignments are reported across the generated levels.',
      caveat:
        'Rows outside the declared cube are rejected and any node-capacity overflow invalidates the hierarchy. Depth and subdivision limit change the hierarchy, not the retained point count.',
      options: {view: 'octree'},
      controls: ['view', 'maximumDepth', 'nodePointLimit'],
      readouts: ['octreeOwned', 'octreeChart', 'octreeOverflow'],
      stage: 'octree'
    },
    {
      id: 'merge-sufficient-statistics',
      textAlternative:
        'Voxel representatives retain LAS class colors while charts compare merged partition counts and representatives with direct GPU reduction results.',
      title: 'Merge two disjoint voxel summaries',
      headline: 'Per-cell counts and representatives should match a one-cloud GPU reduction',
      body: '`GPUPointCloudVoxelDownsample` produces per-cell counts and minimum-source-row representatives for each of two disjoint row ranges. `GPUPointCloudVoxelMerge` combines those summaries. A separate GPU downsample of the full sample provides a per-cell count and representative comparison.',
      evidence:
        '**{{mergeRows}}** source rows are represented; **{{mergeCountCheck}}**. Representative mismatches: **{{mergeRepresentativeMismatch}}**.',
      caveat:
        'The voxel grid fixes a 6 m aggregation scale. A passing count and representative comparison checks this merge against the same GPU reduction method; it does not show that a voxel representative preserves terrain detail.',
      options: {view: 'merge'},
      controls: ['view', 'pointSize'],
      readouts: ['mergeRows', 'mergeCountCheck', 'mergeRepresentativeMismatch', 'mergeStatus'],
      stage: 'merge'
    }
  ],
  pipeline: [
    {
      id: 'radius',
      label: 'Count support',
      detail: 'Compare each sampled row with every other row inside the exact 3D radius.',
      show: {option: 'view', value: 'radius'}
    },
    {
      id: 'smrf',
      label: 'Classify ground',
      detail: 'Build selected terrain support and run progressive morphological filtering.',
      show: {option: 'view', value: 'smrf'}
    },
    {
      id: 'octree',
      label: 'Partition by level',
      detail: 'Build sparse Morton nodes and retain additive source-row ownership.',
      show: {option: 'view', value: 'octree'}
    },
    {
      id: 'merge',
      label: 'Merge voxel statistics',
      detail: 'Merge disjoint row partitions and compare against a direct GPU reduction.',
      show: {option: 'view', value: 'merge'}
    }
  ],
  furniture: {title: true, credit: true, scaleBar: true},
  legends: state => {
    if (state.view === 'radius')
      return [
        {
          id: 'radius-support',
          title: 'Rows meeting support rule',
          kind: 'categories',
          entries: [
            {label: 'Below minimum support', color: [206, 108, 96]},
            {label: 'Meets minimum support', color: [116, 190, 159]}
          ]
        }
      ];
    if (state.view === 'smrf')
      return [
        {
          id: 'smrf-ground',
          title: 'SMRF result and support',
          kind: 'categories',
          entries: [
            {label: 'Other selected return', color: [206, 108, 96]},
            {label: 'Marked ground', color: [116, 190, 159]},
            {label: 'Edge / inpainted or unresolved', color: [239, 183, 91]}
          ]
        }
      ];
    if (state.view === 'octree')
      return [
        {
          id: 'octree-level',
          title: 'GPU node ownership level',
          kind: 'categories',
          entries: Array.from({length: 8}, (_, level) => ({
            label: `Level ${level}`,
            color: OCTREE_COLORS[level]
          }))
        }
      ];
    if (state.view === 'merge')
      return [
        {
          id: 'voxel-source-class',
          title: 'Voxel representative LAS class',
          kind: 'categories',
          entries: [
            {label: 'Class 1 · Unclassified', color: [232, 174, 94]},
            {label: 'Class 2 · Ground', color: [146, 206, 164]},
            {label: 'Class 7 · Low noise', color: [229, 105, 99]},
            {label: 'Class 9 · Water', color: [92, 156, 215]},
            {label: 'Class 18 · High noise', color: [223, 98, 94]},
            {label: 'Class 20 · Ignored ground', color: [185, 183, 175]},
            {label: 'Other class', color: [150, 155, 160]}
          ]
        }
      ];
    return [
      {
        id: 'las-class',
        title: 'Source LAS classification',
        kind: 'categories',
        entries: [
          {label: 'Class 1 · Unclassified', color: [232, 174, 94]},
          {label: 'Class 2 · Ground', color: [146, 206, 164]},
          {label: 'Class 7 · Low noise', color: [229, 105, 99]},
          {label: 'Class 9 · Water', color: [92, 156, 215]},
          {label: 'Class 18 · High noise', color: [223, 98, 94]},
          {label: 'Class 20 · Ignored ground', color: [185, 183, 175]},
          {label: 'Other class', color: [150, 155, 160]}
        ]
      }
    ];
  },
  snippet: `graph.add(new GPUPointCloudRadiusFilter({positions, counts: radiusCounts, mask, radius: 3, minimumNeighbors: 4}));\ngraph.add(new GPUPointCloudSMRF({positions, classifications, selection: mask, grid, window: 12, slope: 0.15, output: {groundSurface, groundMask, outputClassifications, status: smrfStatus}}));\ngraph.add(new GPUPointCloudOctree({positions, origin, size, maximumDepth: 5, nodePointLimit: 24, pointIndices, nodes, status: octreeStatus}));\ngraph.add(new GPUPointCloudVoxelDownsample({positions: partA, origin, voxelSize: 6, gridSize, counts: countsA, representatives: representativesA, output: sampleA}));\ngraph.add(new GPUPointCloudVoxelDownsample({positions: partB, origin, voxelSize: 6, gridSize, counts: countsB, representatives: representativesB, output: sampleB}));\n// meansA/B are uploads of computePointCloudVoxelMeans(ownedPositions, 6, gridSize, origin).\n// Selected sampleA/B are representative positions, not voxel means.\ngraph.add(new GPUPointCloudVoxelMerge({partitionPolicy: 'disjoint-owned-rows', parts: [{counts: countsA, representatives: representativesA, centroids: meansA, sourceRowOffset: 0}, {counts: countsB, representatives: representativesB, centroids: meansB, sourceRowOffset}], output: {counts: mergedCounts, representatives: mergedRepresentatives, centroids: mergedCentroids, status: mergedStatus}}));`,
  about: {
    what: 'The workflow applies exact radius support, a Simple Morphological Filter (SMRF), sparse octree partitioning, and a merge of voxel counts and representatives.',
    why: 'These operations answer different scale questions: which rows have local support, which rows fit a terrain surface, how points are partitioned in space, and whether partial summaries can be merged.',
    howToRead:
      'Map colors show the currently selected GPU output. The charts and counts are derived from asynchronous GPU readback. A nonzero overflow or merge status means the affected result must be rejected.'
  },
  create: async ctx => (await import('./point-cloud-support.compute')).createPointCloudSupport(ctx)
});

const OCTREE_COLORS = [
  [80, 145, 181, 255],
  [104, 171, 174, 255],
  [138, 185, 146, 255],
  [190, 190, 113, 255],
  [222, 165, 103, 255],
  [213, 133, 112, 255],
  [186, 118, 153, 255],
  [151, 127, 184, 255]
] as const;
