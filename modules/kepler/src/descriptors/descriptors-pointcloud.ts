// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';
import {MAXIMUM_POINT_CLOUD_ROWS} from '../analysis/catalog-pointcloud-input';

// XYZ is geometry, not a property-name setting in these bindings. In particular, inventing an
// elevation field role would change their execution ABI. Classification and voxel properties
// are the only column-name parameters. The profile currently describes XY, not XYZ support.
const XYZ_NOTE =
  'Elevation is the third coordinate of each finite XYZ Point, not a scalar property. EPSG:4326 horizontal coordinates become local metres; elevation must already be metres in the same vertical datum. Planar XYZ must share one Cartesian unit. The profile cannot certify finite XYZ, vertical units or datum agreement; input preparation checks coordinates before GPU allocation.';
const STATUS_NOTE =
  'Inspect numeric status columns and warning diagnostics before interpreting measurements. Unsupported measurements are null; retained rows and original source identities are preserved.';
const RADIUS_NOTE =
  'The radius starts at three times median XY nearest-neighbour spacing, or one tenth of the XY extent diagonal when spacing is unavailable. This is an exploratory scale proxy, not a 3D support guarantee. Check actual neighbor_count and neighborhood_status; only the nearest capacity other points within the inclusive 3D radius are retained, while counts include all matches.';
const GRID_NOTE =
  'The starting cellSize is the XY extent diagonal divided by 32, a conservative coarse grid for the 4,096-cell catalog limit, rather than the general shorter-side/250 rule. The grid includes the maximum and has at least two cells on each axis. Increase cellSize if preparation rejects the grid; the profile cannot certify its exact Float32 dimensions.';
const TERRAIN_NOTE =
  'Terrain requires explicit LAS class 2 support in each relevant cell. Missing classification defaults to class 1; missing terrain stays null. Run ground classification first when needed; the profile does not certify class-2 coverage.';

function describeConstant(
  name: string,
  level: 'basic' | 'advanced' = 'advanced',
  apply: 'live' | 'rebuild' = 'rebuild'
): KeplerOperationParameterDescriptor {
  return {name, defaultPolicy: {from: 'constant'}, level, apply};
}

function describeDistance(name: string, input: string): KeplerOperationParameterDescriptor {
  return {
    name,
    defaultPolicy: {
      from: 'data',
      input,
      rule: 'median-nearest-neighbor-distance',
      factor: 3,
      fallback: {rule: 'extent-diagonal', factor: 0.1}
    },
    level: 'basic',
    apply: 'rebuild'
  };
}

function describeCellSize(): KeplerOperationParameterDescriptor {
  return {
    name: 'cellSize',
    defaultPolicy: {from: 'data', input: 'points', rule: 'extent-diagonal', factor: 1 / 32},
    level: 'basic',
    apply: 'rebuild'
  };
}

function describeClassification(required: boolean): KeplerOperationFieldDescriptor[] {
  return [
    {
      role: 'classification',
      label: 'LAS classification (integer 0–255)',
      parameter: 'classificationProperty',
      accepts: ['integer'],
      measurement: ['nominal'],
      required
    }
  ];
}

function describeClassificationRequirements(): KeplerOperationRequirement[] {
  return [
    {
      input: 'points',
      role: 'classification',
      id: 'numeric-range',
      minimum: 0,
      maximum: 255,
      integer: true,
      severity: 'block',
      remedy: 'Use LAS classification integers in [0,255]; replace codes outside this range'
    }
  ];
}

function describeConfirmation(
  name: string,
  message: string,
  remedy: string
): KeplerOperationParameterDescriptor {
  return {...describeConstant(name, 'basic'), confirmation: {message, remedy}};
}

function describeRequirements(input: string): KeplerOperationRequirement[] {
  return [
    {
      input,
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Choose a nonempty XYZ point cloud'
    },
    {
      input,
      id: 'maximum-rows',
      severity: 'block',
      limit: MAXIMUM_POINT_CLOUD_ROWS,
      remedy:
        'Split the cloud into at most 2,048 owned rows per input; retain support halos separately'
    },
    {
      input,
      id: 'maximum-vertices',
      severity: 'block',
      limit: MAXIMUM_POINT_CLOUD_ROWS,
      remedy: 'Choose single XYZ Points, at most 2,048 vertices per input'
    },
    {
      input,
      id: 'declared-coordinates',
      severity: 'block',
      remedy:
        'Declare geographic or planar horizontal coordinates and confirm the XYZ units and vertical datum'
    }
  ];
}

function describePointOutput(
  column: string,
  scale: 'sequential' | 'diverging' | 'categorical',
  next: KeplerOperationOutputDescriptor['next'] = ['shape-of-the-surface']
): KeplerOperationOutputDescriptor {
  return {
    tableId: 'points',
    kind: 'points',
    presentation: {column, scale, ...(scale === 'diverging' ? {center: 0} : {})},
    interpretation: {
      summary: 'value-range',
      cautions: ['guessed-coordinates', 'default-from-fallback']
    },
    next
  };
}

const VOXEL_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    role: 'voxel',
    label: 'Unique voxel ID on the shared grid',
    parameter: 'voxelProperty',
    accepts: ['integer', 'number', 'category', 'text'],
    measurement: ['identifier', 'nominal'],
    required: true
  },
  {
    role: 'count',
    label: 'Owned source count (uint32)',
    parameter: 'countProperty',
    accepts: ['integer'],
    measurement: ['count'],
    required: true
  },
  {
    role: 'representative',
    label: 'Representative source row (uint32)',
    parameter: 'representativeProperty',
    accepts: ['integer'],
    measurement: ['identifier', 'nominal'],
    required: true
  }
];

/** User-facing semantics for all eleven bounded point-cloud catalog bindings. */
export const POINT_CLOUD_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'pointcloud-ground',
    questions: ['shape-of-the-surface', 'prepare-data'],
    action: 'Classify ground by morphology',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: describeClassification(false)}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeCellSize(),
      describeConstant('radii', 'basic'),
      describeConstant('initialThreshold', 'basic'),
      describeConstant('slope', 'basic'),
      describeConstant('maximumThreshold', 'advanced')
    ],
    requirements: [...describeRequirements('points'), ...describeClassificationRequirements()],
    outputs: [
      describePointOutput('classification', 'categorical', ['shape-of-the-surface', 'prepare-data'])
    ],
    notes: [
      XYZ_NOTE,
      GRID_NOTE,
      STATUS_NOTE,
      'Ground starts from unclassified cell minima over all points, independent of LAS classes. Morphology radii are strictly increasing integer cell radii (1–32), at most eight. Thresholds use radius increments, not the Zhang full-window-width convention. Elevation thresholds are declared constants because the profile has no vertical roughness; maximumThreshold must be at least initialThreshold.'
    ]
  },
  {
    bindingId: 'pointcloud-smrf',
    questions: ['shape-of-the-surface', 'prepare-data'],
    action: 'Classify ground with SMRF',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: describeClassification(false)}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeCellSize(),
      describeConstant('slope', 'basic'),
      describeConstant('threshold', 'basic'),
      describeConstant('scalar'),
      describeConstant('windowCells', 'basic')
    ],
    requirements: [...describeRequirements('points'), ...describeClassificationRequirements()],
    outputs: [
      describePointOutput('classification', 'categorical', ['shape-of-the-surface', 'prepare-data'])
    ],
    notes: [
      XYZ_NOTE,
      GRID_NOTE,
      STATUS_NOTE,
      'SMRF uses windowCells times cellSize as its window radius; changing either rebuilds the morphology graph. Slope, scalar and elevation threshold retain stated constants because the profile has no terrain calibration. Classification values must be integers in [0,255]; absent values start as class 1.'
    ]
  },
  {
    bindingId: 'pointcloud-surfaces',
    questions: ['shape-of-the-surface'],
    action: 'Measure terrain and canopy',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: describeClassification(true)}],
    parameters: [describeConstant('coordinateSystem'), describeCellSize()],
    requirements: [
      ...describeRequirements('points'),
      ...describeClassificationRequirements(),
      {
        input: 'points',
        id: 'no-missing-values',
        role: 'classification',
        severity: 'warn',
        remedy: 'Supply LAS classes; missing values become class 1'
      }
    ],
    outputs: [
      describePointOutput('canopy', 'sequential'),
      {...describePointOutput('canopy', 'sequential'), tableId: 'grid'}
    ],
    notes: [
      XYZ_NOTE,
      GRID_NOTE,
      TERRAIN_NOTE,
      STATUS_NOTE,
      'Terrain is the ground-class cell minimum; surface is the all-point cell maximum; canopy is their difference. Grid rows are cell centres with coverage bits; the points table samples those cells. These are point-published grid summaries, not a continuous interpolated raster.'
    ]
  },
  {
    bindingId: 'pointcloud-height',
    questions: ['shape-of-the-surface', 'prepare-data'],
    action: 'Measure height above ground',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: describeClassification(true)}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeCellSize(),
      describeConstant('interpolation', 'basic')
    ],
    requirements: [...describeRequirements('points'), ...describeClassificationRequirements()],
    outputs: [describePointOutput('height_above_ground', 'sequential')],
    notes: [
      XYZ_NOTE,
      GRID_NOTE,
      TERRAIN_NOTE,
      STATUS_NOTE,
      'Height subtracts cell or bilinear terrain from original elevation. Negative heights are retained. Unsupported terrain samples publish null and a nonzero height_status; bilinear sampling requires its surrounding terrain support.'
    ]
  },
  {
    bindingId: 'pointcloud-neighborhoods',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Inspect 3D point neighbours',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeDistance('radius', 'points'),
      describeConstant('capacity', 'basic')
    ],
    requirements: describeRequirements('points'),
    outputs: [describePointOutput('neighbor_count', 'sequential')],
    notes: [
      XYZ_NOTE,
      RADIUS_NOTE,
      STATUS_NOTE,
      'Capacity is a constant 32, adjustable from 1 to 64; it changes buffer shapes and requires rebuilding. Neighbor slots publish source rows and distances, with null slots for missing neighbours.'
    ]
  },
  {
    bindingId: 'pointcloud-features',
    questions: ['shape-of-the-surface'],
    action: 'Measure local 3D shape',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeDistance('radius', 'points'),
      describeConstant('capacity', 'basic')
    ],
    requirements: describeRequirements('points'),
    outputs: [describePointOutput('planarity', 'sequential')],
    notes: [
      XYZ_NOTE,
      RADIUS_NOTE,
      STATUS_NOTE,
      'Covariance eigenvalues describe linearity, planarity and scattering, not significance. Normals have positive-z orientation. Fewer than three retained neighbours or degenerate support can leave normals and shape null; inspect feature_status and truncation. Capacity stays 32 (1–64), not a promise of complete support.'
    ]
  },
  {
    bindingId: 'pointcloud-statistical-filter',
    questions: ['prepare-data'],
    action: 'Flag statistical point outliers',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeDistance('radius', 'points'),
      describeConstant('capacity', 'basic'),
      describeConstant('standardDeviations', 'basic')
    ],
    requirements: describeRequirements('points'),
    outputs: [
      describePointOutput('inlier', 'categorical', ['prepare-data', 'shape-of-the-surface'])
    ],
    notes: [
      XYZ_NOTE,
      RADIUS_NOTE,
      STATUS_NOTE,
      'Inlier 1 keeps a row; 0 flags an outlier or unsupported row without deleting it. The threshold is the population mean of retained mean distances plus the stated standardDeviations multiplier (default 2). This is not a hypothesis test. The current binding passes the multiplier as a shader constant, so it rebuilds; capacity stays 32 (1–64).'
    ]
  },
  {
    bindingId: 'pointcloud-normal-scale',
    questions: ['shape-of-the-surface'],
    action: 'Choose a stable normal scale',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('coordinateSystem'),
      describeConstant('radii', 'basic'),
      describeConstant('capacity', 'basic'),
      describeConstant('minimumSupport', 'basic', 'live')
    ],
    requirements: describeRequirements('points'),
    outputs: [describePointOutput('normal_scale', 'sequential')],
    notes: [
      XYZ_NOTE,
      STATUS_NOTE,
      'Radii are a declared list of increasing 3D distances, at most eight values in (0,32]; the default [1,2] must be reviewed in the XYZ unit. XY profiles cannot derive a list of 3D scales. Capacity 32 (1–64) retains support; only complete nondegenerate converged scales participate. minimumSupport is a runtime eligibility parameter, at least 3 and no greater than useful retained support. Ties choose the smaller radius; no eligible scale yields null.'
    ]
  },
  {
    bindingId: 'pointcloud-icp',
    questions: ['prepare-data', 'what-changed'],
    action: 'Align two point clouds',
    prominence: 3,
    inputs: [
      {name: 'source', kind: 'points', fields: []},
      {name: 'target', kind: 'points', fields: []}
    ],
    parameters: [
      describeConstant('coordinateSystem'),
      describeDistance('maximumDistance', 'target'),
      describeConstant('maximumIterations', 'basic')
    ],
    requirements: [...describeRequirements('source'), ...describeRequirements('target')],
    outputs: [
      describePointOutput('inlier', 'categorical', ['what-changed']),
      {
        tableId: 'registration',
        kind: 'summary',
        interpretation: {summary: 'row-count', cautions: ['poor-model-fit']},
        next: ['what-changed']
      }
    ],
    notes: [
      XYZ_NOTE,
      STATUS_NOTE,
      'maximumIterations is the basic iteration budget (default 10; Kepler range 1–30). Point-to-point ICP starts at identity and may converge to a local alignment. maximumDistance uses target XY spacing times three (extent diagonal/10 fallback), an exploratory correspondence gate that cannot estimate the initial 3D displacement. Clouds need overlapping matched structure, common units and datum. Inspect registration_status, matched_count and root_mean_square_error; reaching the iteration limit does not certify convergence. The transform is source-to-target in the shared local frame, not a global geographic affine transform.'
    ]
  },
  {
    bindingId: 'pointcloud-change',
    questions: ['what-changed', 'shape-of-the-surface'],
    action: 'Measure normal-direction change',
    prominence: 3,
    inputs: [
      {name: 'reference', kind: 'points', fields: []},
      {name: 'comparison', kind: 'points', fields: []}
    ],
    parameters: [
      describeConstant('coordinateSystem'),
      describeDistance('radius', 'reference'),
      describeConstant('capacity', 'basic'),
      describeDistance('halfDepth', 'reference'),
      describeConstant('minimumSupport', 'basic'),
      describeConstant('registrationError', 'basic'),
      describeConfirmation(
        'alreadyAligned',
        'pointcloud-change: alreadyAligned must certify a shared aligned frame',
        'Confirm that both epochs are already aligned in one frame and set alreadyAligned to true'
      )
    ],
    requirements: [...describeRequirements('reference'), ...describeRequirements('comparison')],
    outputs: [describePointOutput('change', 'diverging', ['what-changed'])],
    notes: [
      XYZ_NOTE,
      RADIUS_NOTE,
      STATUS_NOTE,
      'alreadyAligned must explicitly certify a shared aligned frame; its default false refuses execution. The profile cannot verify alignment, datum or overlap. Reference rows are cores and normals point toward positive z; change is comparison minus reference along that normal. halfDepth uses the same exploratory XY scale proxy as radius and needs review for 3D relief. Capacity 32 (1–64) controls normal support, not cylinder epoch counts. Inspect reference_count, comparison_count, change_status, lower_bound, upper_bound and detection_limit. registrationError is a stated constant 0 until supplied; this assumes no registration uncertainty and is not evidence of exact alignment or spatially corrected significance.'
    ]
  },
  {
    bindingId: 'pointcloud-voxel-merge',
    questions: ['prepare-data'],
    action: 'Merge owned voxel statistics',
    prominence: 3,
    inputs: [
      {name: 'left', kind: 'points', fields: VOXEL_FIELDS},
      {name: 'right', kind: 'points', fields: VOXEL_FIELDS}
    ],
    parameters: [
      describeConstant('coordinateSystem'),
      describeConstant('rightRowOffset', 'basic'),
      describeConfirmation(
        'disjointOwnedRows',
        'pointcloud-voxel-merge: disjointOwnedRows must certify disjoint source rows and one identical voxel grid',
        'Exclude halo replicas, confirm both parts own disjoint rows on one identical voxel grid, and set disjointOwnedRows to true'
      )
    ],
    requirements: ['left', 'right'].flatMap(input => [
      ...describeRequirements(input),
      {
        input,
        id: 'planar-coordinates' as const,
        severity: 'block' as const,
        remedy: 'Use arithmetic Cartesian centroids in one planar voxel grid'
      },
      ...VOXEL_FIELDS.map(field => ({
        input,
        id: 'no-missing-values' as const,
        role: field.role,
        severity: 'block' as const,
        remedy: `Supply every ${field.role} value in both voxel parts`
      })),
      {
        input,
        id: 'non-negative-values' as const,
        role: 'count',
        severity: 'block' as const,
        remedy: 'Use uint32 owned-row counts, including zero for empty voxels'
      }
    ]),
    outputs: [describePointOutput('count', 'sequential', ['prepare-data'])],
    notes: [
      XYZ_NOTE,
      STATUS_NOTE,
      'Inputs must be arithmetic per-voxel centroids, counts and representative source rows on one identical grid, not raw points or downsample representatives. Both parts must list identical unique voxel IDs in the same order and use the same property names. Counts and representatives must fit uint32; occupied representatives must be below the missing-key sentinel. disjointOwnedRows must certify disjoint ownership with halo replicas excluded; default false refuses execution. The profile cannot verify ownership, matching voxel order or centroid provenance. rightRowOffset adjusts right representative row identities. Empty or failed merged voxels have null coordinates and representatives; inspect merge_status.'
    ]
  }
];
