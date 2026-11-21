// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/**
 * VAL-12/13: explicit refusal boundaries and structured-field fixtures. Integer maxima and
 * received counts are the rejection builders' supplied dimensions/products, not GPU results.
 * Keep the original independent message/limit tables as a second check of these expectations.
 * REQUIREMENT's legacy boolean/English details are pinned until producers supply richer fields.
 */
export const KERNEL_AUDIT_REJECTION_CONTRACTS = {
  'GPUSpatialWeightsTraces exact gram false next batch above the command limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 32775,
      expected: {maximum: 32768},
      id: 'spatial-weights-traces',
      contributor: 'spatial-weights-traces'
    }
  },
  'GPUSpatialWeightsTraces exact gram false binding-limited rows exceed the command limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 262407,
      expected: {maximum: 32768},
      id: 'spatial-weights-traces',
      contributor: 'spatial-weights-traces'
    }
  },
  'GPUSpatialWeightsTraces exact gram true next batch above the command limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 32862,
      expected: {maximum: 32768},
      id: 'spatial-weights-traces',
      contributor: 'spatial-weights-traces'
    }
  },
  'GPUSpatialWeightsTraces exact gram true binding-limited rows exceed the command limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 270088,
      expected: {maximum: 32768},
      id: 'spatial-weights-traces',
      contributor: 'spatial-weights-traces'
    }
  },
  'GPUColumnProfile numeric rows one above the storage binding maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'columns[0].values chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'column-profile',
      contributor: 'column-profile'
    }
  },
  'GPUColumnProfile row metadata one above uint32 addressing': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'columns[0]',
      received: false,
      expected: 'columns[0] length must be at most 4294967295: row indices and counts use uint32',
      id: 'column-profile',
      contributor: 'column-profile'
    }
  },
  'GPUColumnProfile category total one above the maximum, 16 columns': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'category columns categoryCount total',
      received: 33554436,
      expected: {
        maximum: 33554432
      },
      id: 'column-profile',
      contributor: 'column-profile'
    }
  },
  'GPUTemporalReduction word times, slots one cell row above the binding limit': {
    phase: 'declaration'
  },
  'GPUInequality zone count one above the maximum, per-zone views only': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCount',
      received: 3050403,
      expected: {
        minimum: 1,
        maximum: 3050402
      },
      id: 'inequality',
      contributor: 'inequality'
    }
  },
  'GPUInequality zones x 1025 knots one zone above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCount x lorenzKnotCount',
      received: 33555425,
      expected: {
        maximum: 33554432
      },
      id: 'inequality',
      contributor: 'inequality'
    }
  },
  'GPUGroupStatistics uint32x2 keys, groups one above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.keys rows',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'group-statistics',
      contributor: 'group-statistics'
    }
  },
  'GPUColumnQuantiles one row above the maximum (2^24 + 1), float32 counts': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'values',
      received: false,
      expected:
        'values length must be at most 16777216 (2^24): row counts and ranks are held in float32 and are exact only up to 2^24',
      id: 'column-quantiles',
      contributor: 'column-quantiles'
    }
  },
  'GPUClassBreaks one row above the maximum (2^24 + 1), float32 counts': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'values',
      received: false,
      expected:
        'values length must be in [1, 16777216] (2^24): moment counts and quantile ranks are held in float32 and are exact only up to 2^24',
      id: 'class-breaks',
      contributor: 'class-breaks'
    }
  },
  'GPUVariogram maximum rows and lattice, 409 bins with robust channel (points above the maximum)':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'positions length',
        received: 11184811,
        expected: {
          minimum: 1,
          maximum: 11184810
        },
        id: 'variogram',
        contributor: 'variogram'
      }
    },
  'GPUSpatialCorrelogram cumulative bands, maximum rows, lattice and bands (points above the maximum)':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'positions length',
        received: 11184811,
        expected: {
          minimum: 1,
          maximum: 11184810
        },
        id: 'spatial-correlogram',
        contributor: 'spatial-correlogram'
      }
    },
  'GPUSpatialCorrelogram annulus bands, maximum rows, lattice and bands (points above the maximum)':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'positions length',
        received: 11184811,
        expected: {
          minimum: 1,
          maximum: 11184810
        },
        id: 'spatial-correlogram',
        contributor: 'spatial-correlogram'
      }
    },
  'GPURipley maximum rows and lattice, 256 radii, every output (points above the maximum)': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions length',
      received: 11184811,
      expected: {
        minimum: 1,
        maximum: 11184810
      },
      id: 'ripley',
      contributor: 'ripley'
    }
  },
  'GPURipleyEnvelope maximum rows and lattice, 199 simulations, 256 radii (points above the maximum)':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'positions length',
        received: 11184811,
        expected: {
          minimum: 1,
          maximum: 11184810
        },
        id: 'ripley-envelope',
        contributor: 'ripley-envelope'
      }
    },
  'GPUPointPatternIndices maximum rows and lattice, every output (points above the maximum)': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions length',
      received: 11184811,
      expected: {
        minimum: 1,
        maximum: 11184810
      },
      id: 'point-pattern-indices',
      contributor: 'point-pattern-indices'
    }
  },
  'GPURipleyDistanceFunctions maximum rows, lattice and reference grid (2^24) (points above the maximum)':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'positions length',
        received: 11184811,
        expected: {
          minimum: 1,
          maximum: 11184810
        },
        id: 'ripley-distance-functions',
        contributor: 'ripley-distance-functions'
      }
    },
  'GPUCoregionalizationFit variableCount-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'variableCount',
      received: 5,
      expected: {
        minimum: 2,
        maximum: 4
      },
      id: 'coregionalization-fit',
      contributor: 'coregionalization-fit'
    }
  },
  'GPUCoregionalizationFit structureCount-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'structureCount',
      received: 5,
      expected: {
        minimum: 1,
        maximum: 4
      },
      id: 'coregionalization-fit',
      contributor: 'coregionalization-fit'
    }
  },
  'GPUCoregionalizationFit maximumIterations-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'maximumIterations',
      received: 1025,
      expected: {
        minimum: 0,
        maximum: 1024
      },
      id: 'coregionalization-fit',
      contributor: 'coregionalization-fit'
    }
  },
  'GPUCoregionalizationFit distances.length-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'distances.length',
      received: 257,
      expected: {
        minimum: 0,
        maximum: 256
      },
      id: 'coregionalization-fit',
      contributor: 'coregionalization-fit'
    }
  },
  'GPULocationChangePoints one cell above the cube binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'values chunk 0',
      received: 134219776,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'location-change-points',
      contributor: 'location-change-points'
    }
  },
  'GPUPettittTest one cell above the cube binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'values chunk 0',
      received: 134219776,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'pettitt-test',
      contributor: 'pettitt-test'
    }
  },
  'GPUExpectationBasedSpaceTimeScan one zone above the count table limit': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'zones',
      received: false,
      expected:
        'zones * maximumWindowLength received 16777248; must be at most 16777216; reduce zones or maximumWindowLength',
      id: 'expectation-based-scan',
      contributor: 'expectation-based-scan'
    }
  },
  'GPUSpaceTimeKernelDensity one slice above the voxel limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'voxels (grid.width * grid.height * sliceCount)',
      received: 33685504,
      expected: {
        maximum: 33554432
      },
      id: 'space-time-kernel-density',
      contributor: 'space-time-kernel-density'
    }
  },
  'GPUSpaceTimeKernelDensity one point above the row limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions length',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'space-time-kernel-density',
      contributor: 'space-time-kernel-density'
    }
  },
  'GPUSpaceTimeKFunction one row above the row limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions length',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'space-time-k-function',
      contributor: 'space-time-k-function'
    }
  },
  'GPUSpaceTimeKFunction local statistics one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.local rows',
      received: 33554496,
      expected: {
        maximum: 33554432
      },
      id: 'space-time-k-function',
      contributor: 'space-time-k-function'
    }
  },
  'GPUSpaceTimeDiggleTest one permutation above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutationCount',
      received: 1000,
      expected: {
        minimum: 1,
        maximum: 999
      },
      id: 'space-time-diggle-test',
      contributor: 'space-time-diggle-test'
    }
  },
  'GPUSpaceTimeHawkes one event above the row limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions length',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'space-time-hawkes',
      contributor: 'space-time-hawkes'
    }
  },
  'GPUSinkhornDivergence one cell above the grid side limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid.width and grid.height',
      received: 2049,
      expected: {
        maximum: 2048
      },
      id: 'sinkhorn-divergence',
      contributor: 'sinkhorn-divergence'
    }
  },
  'GPUPeriodIndex calendar hour step, one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'calendarYear chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'period-index',
      contributor: 'period-index'
    }
  },
  'GPUPeriodIndex fixed width, one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'timestamps chunk 0',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'period-index',
      contributor: 'period-index'
    }
  },
  'GPUCubeLayoutTranspose cube one period above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'values chunk 0',
      received: 134234112,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'cube-layout-transpose',
      contributor: 'cube-layout-transpose'
    }
  },
  'GPUTimeWordsRelative one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'timestamps chunk 0',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'time-words-relative',
      contributor: 'time-words-relative'
    }
  },
  'GPUValueCube one bin above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'fixed-point sums',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce cellCount * periodCount'
      },
      id: 'value-cube',
      contributor: 'value-cube'
    }
  },
  'GPUCellPeriodKeys one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'cells chunk 0',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'cell-period-keys',
      contributor: 'cell-period-keys'
    }
  },
  'GPUTrajectoryStaypoints great-circle words, one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-staypoints',
      contributor: 'trajectory-staypoints'
    }
  },
  'GPUTrajectoryLocations one staypoint above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'centroids',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'use fewer staypoints per graph'
      },
      id: 'trajectory-locations',
      contributor: 'trajectory-locations'
    }
  },
  'GPUTrajectorySegments triplegs and trips, one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-segments',
      contributor: 'trajectory-segments'
    }
  },
  'GPUMobilityIndicators all indicators, one staypoint above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'centroids',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'use fewer staypoints per graph'
      },
      id: 'mobility-indicators',
      contributor: 'mobility-indicators'
    }
  },
  'GPUModalSplit count metric, one track above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.modalSplit',
      received: 134217744,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce modeCount or the track count'
      },
      id: 'modal-split',
      contributor: 'modal-split'
    }
  },
  'GPUTrajectoryOutlierFlags one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-outlier-flags',
      contributor: 'trajectory-outlier-flags'
    }
  },
  'GPUTrajectoryHampel one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'values',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-hampel',
      contributor: 'trajectory-hampel'
    }
  },
  'GPUTrajectorySplit one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-split',
      contributor: 'trajectory-split'
    }
  },
  'GPUTrajectoryThinning one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-thinning',
      contributor: 'trajectory-thinning'
    }
  },
  'GPUTrajectoryTrackRelativeTimes one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'timestamps',
      received: 134217736,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the tracks over several graphs'
      },
      id: 'trajectory-track-relative-times',
      contributor: 'trajectory-track-relative-times'
    }
  },
  'GPUIsolines maximum grid, levels and capacities with polylines': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'owned cells * levels.length * 2 required segments',
      received: 8585740800,
      expected: {
        maximum: 4294967295
      },
      id: 'isolines',
      contributor: 'isolines'
    }
  },
  'GPUIsolines maximum grid, segments only': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'owned cells * levels.length * 2 required segments',
      received: 8585740800,
      expected: {
        maximum: 4294967295
      },
      id: 'isolines',
      contributor: 'isolines'
    }
  },
  'GPUIsobandRings edge capacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'edgeCapacity',
      received: 8388609,
      expected: {
        minimum: 1,
        maximum: 8388608
      },
      id: 'isoband-rings',
      contributor: 'isoband-rings'
    }
  },
  'GPUIsobandRings ring capacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'ring capacity (output.ringOffsets.length - 1)',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'isoband-rings',
      contributor: 'isoband-rings'
    }
  },
  'GPURasterJoin zones above the maximum with sums, every output': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCount',
      received: 16777216,
      expected: {
        minimum: 1,
        maximum: 16777215
      },
      id: 'raster-join',
      contributor: 'raster-join'
    }
  },
  'GPURasterJoin zones above the maximum, counts only': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCount',
      received: 33554433,
      expected: {
        minimum: 1,
        maximum: 33554432
      },
      id: 'raster-join',
      contributor: 'raster-join'
    }
  },
  'GPURasterProfile paths above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'line count (lineOffsets.length - 1)',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'raster-profile',
      contributor: 'raster-profile'
    }
  },
  'GPURasterZonalStatistics zones above the maximum, atomic sums, uniform cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 33554433,
      expected: {
        minimum: 1,
        maximum: 33554432
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum without moments, atomic sums, uniform cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 33554433,
      expected: {
        minimum: 1,
        maximum: 33554432
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum, atomic sums, geographic cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 33554433,
      expected: {
        minimum: 1,
        maximum: 33554432
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum without moments, atomic sums, geographic cells':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'zoneCapacity',
        received: 33554433,
        expected: {
          minimum: 1,
          maximum: 33554432
        },
        id: 'raster-zonal-statistics',
        contributor: 'raster-zonal-statistics'
      }
    },
  'GPURasterZonalStatistics zones above the maximum, sorted sums, uniform cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 16777216,
      expected: {
        minimum: 1,
        maximum: 16777215
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum without moments, sorted sums, uniform cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 16777216,
      expected: {
        minimum: 1,
        maximum: 16777215
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum, sorted sums, geographic cells': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCapacity',
      received: 16777216,
      expected: {
        minimum: 1,
        maximum: 16777215
      },
      id: 'raster-zonal-statistics',
      contributor: 'raster-zonal-statistics'
    }
  },
  'GPURasterZonalStatistics zones above the maximum without moments, sorted sums, geographic cells':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'zoneCapacity',
        received: 16777216,
        expected: {
          minimum: 1,
          maximum: 16777215
        },
        id: 'raster-zonal-statistics',
        contributor: 'raster-zonal-statistics'
      }
    },
  'GPURasterStreamingAccumulator tc07-chunk-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'chunkSliceCount',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'raster-streaming-accumulator',
      contributor: 'raster-streaming-accumulator'
    }
  },
  'GPURasterRollingMean rts06-window-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'windowLength',
      received: 129,
      expected: {
        minimum: 1,
        maximum: 128
      },
      id: 'raster-rolling-mean',
      contributor: 'raster-rolling-mean'
    }
  },
  'GPURasterWhittakerSmoother rts02-slices-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sliceCount',
      received: 257,
      expected: {
        minimum: 2,
        maximum: 256
      },
      id: 'raster-whittaker-smoother',
      contributor: 'raster-whittaker-smoother'
    }
  },
  'GPURasterHANTS rts02-memory-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'per-thread',
      received: false,
      expected:
        'per-thread QR working memory (sliceCount 256, 7 coefficients) must fit 2048 floats',
      id: 'raster-hants',
      contributor: 'raster-hants'
    }
  },
  'GPURasterTemporalResample rts02-grid-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'gridCount',
      received: 4097,
      expected: {
        minimum: 1,
        maximum: 4096
      },
      id: 'raster-temporal-resample',
      contributor: 'raster-temporal-resample'
    }
  },
  'GPURasterStandardizedIndexCalibration rts06-calibration-chunk-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'chunkSliceCount',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'raster-standardized-index-calibration',
      contributor: 'raster-standardized-index-calibration'
    }
  },
  'GPURasterStandardizedIndex rts06-index-chunk-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'chunkSliceCount',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'raster-standardized-index',
      contributor: 'raster-standardized-index'
    }
  },
  'GPURasterClimateRunAccumulator rts06-run-chunk-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'chunkSliceCount',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'raster-climate-run-accumulator',
      contributor: 'raster-climate-run-accumulator'
    }
  },
  'GPURasterHarmonicRegression rts04-memory-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'per-thread',
      received: false,
      expected:
        'per-thread QR working memory (sliceCount 256, 6 coefficients) must fit 2048 floats',
      id: 'raster-harmonic-regression',
      contributor: 'raster-harmonic-regression'
    }
  },
  'GPURasterHarmonicRegression rts04-storage-slices-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sliceCount',
      received: 4097,
      expected: {
        minimum: 1,
        maximum: 4096
      },
      id: 'raster-harmonic-regression',
      contributor: 'raster-harmonic-regression'
    }
  },
  'GPURasterCOLDMonitor rts04-cold-memory-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'per-thread',
      received: false,
      expected:
        'per-thread working memory (sliceCount 256, 8 coefficients, 5 bands) must fit 2048 floats',
      id: 'raster-cold-monitor',
      contributor: 'raster-cold-monitor'
    }
  },
  'GPURasterCOLDMonitor rts04-cold-fit-budget-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'maximumFitCount',
      received: 1024,
      expected: {
        minimum: 1,
        maximum: 163
      },
      id: 'raster-cold-monitor',
      contributor: 'raster-cold-monitor'
    }
  },
  'GPURasterBFASTMonitor rts04-bfast-memory-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'per-thread',
      received: false,
      expected:
        'per-thread QR working memory (historySliceCount 200, 8 coefficients) must fit 2048 floats',
      id: 'raster-bfast-monitor',
      contributor: 'raster-bfast-monitor'
    }
  },
  'GPURasterSnapshotCovariance rts07-invocations-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'accumulation',
      received: false,
      expected:
        'accumulation invocations (cellCount 524288, sliceCount 256) must be at most 16776960',
      id: 'raster-snapshot-covariance',
      contributor: 'raster-snapshot-covariance'
    }
  },
  'GPURasterEOFProjection rts07-modes-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'modeCount',
      received: 33,
      expected: {
        minimum: 1,
        maximum: 32
      },
      id: 'raster-eof-projection',
      contributor: 'raster-eof-projection'
    }
  },
  'GPURasterLaggedCorrelation rts07-lags-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lags length',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'raster-lagged-correlation',
      contributor: 'raster-lagged-correlation'
    }
  },
  'GPURasterMedoidComposite rts05-slices-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sliceCount',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'raster-medoid-composite',
      contributor: 'raster-medoid-composite'
    }
  },
  'GPURasterGeometricMedianComposite rts05-iterations-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'iterationCount',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'raster-geometric-median-composite',
      contributor: 'raster-geometric-median-composite'
    }
  },
  'GPURasterBestAvailablePixel rts05-slices-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sliceCount',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'raster-best-available-pixel',
      contributor: 'raster-best-available-pixel'
    }
  },
  'GPURasterPhenologyMetrics rts05-slices-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sliceCount',
      received: 257,
      expected: {
        minimum: 2,
        maximum: 256
      },
      id: 'raster-phenology-metrics',
      contributor: 'raster-phenology-metrics'
    }
  },
  'GPURasterFalseDiscoveryRateMask rts01-histogram-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.histogram bin count',
      received: 65537,
      expected: {
        minimum: 1,
        maximum: 65536
      },
      id: 'raster-false-discovery-rate-mask',
      contributor: 'raster-false-discovery-rate-mask'
    }
  },
  'GPURasterEmbeddingSimilarity rts03-queries-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'queryCount',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'raster-embedding-similarity',
      contributor: 'raster-embedding-similarity'
    }
  },
  'GPURasterEmbeddingKNearestClassifier rts03-training-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'trainingCount',
      received: 4097,
      expected: {
        minimum: 1,
        maximum: 4096
      },
      id: 'raster-embedding-knn-classifier',
      contributor: 'raster-embedding-knn-classifier'
    }
  },
  'GPURasterEmbeddingCovariance rts03-ranges-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'rangeCount',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'raster-embedding-covariance',
      contributor: 'raster-embedding-covariance'
    }
  },
  'GPURasterEmbeddingProjection rts03-components-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'componentCount',
      received: 17,
      expected: {
        minimum: 3,
        maximum: 16
      },
      id: 'raster-embedding-projection',
      contributor: 'raster-embedding-projection'
    }
  },
  'GPURasterEmbeddingZoneMean rts03-zones-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'zoneCount',
      received: 65537,
      expected: {
        minimum: 1,
        maximum: 65536
      },
      id: 'raster-embedding-zone-mean',
      contributor: 'raster-embedding-zone-mean'
    }
  },
  'GPURasterEmbeddingPyramidLevel rts03-scale-above-maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'horizontal scale',
      received: 17,
      expected: {
        minimum: 1,
        maximum: 16
      },
      id: 'raster-embedding-pyramid-level',
      contributor: 'raster-embedding-pyramid-level'
    }
  },
  'GPUTerrainSpikeRepair grid above the maximum cell count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'terrain-spike-repair',
      contributor: 'terrain-spike-repair'
    }
  },
  'GPUTerrainPeakProminence grid above the maximum cell count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells',
      received: 3355444,
      expected: {
        maximum: 3355443
      },
      id: 'terrain-peak-prominence',
      contributor: 'terrain-peak-prominence'
    }
  },
  'GPUTerrainPeakSnap candidates above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'candidates rows',
      received: 6710887,
      expected: {
        maximum: 6710886
      },
      id: 'terrain-peak-snap',
      contributor: 'terrain-peak-snap'
    }
  },
  'GPUTerrainTopographicPosition grid above the maximum cell count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells',
      received: 11184811,
      expected: {
        maximum: 11184810
      },
      id: 'terrain-topographic-position',
      contributor: 'terrain-topographic-position'
    }
  },
  'GPUTerrainTopographicPosition grid x scales above the maximum plane count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells x scales',
      received: 33587200,
      expected: {
        maximum: 33554432
      },
      id: 'terrain-topographic-position',
      contributor: 'terrain-topographic-position'
    }
  },
  'GPUTerrainWeissLandforms global, grid above the maximum cell count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells',
      received: 11184811,
      expected: {
        maximum: 11184810
      },
      id: 'terrain-weiss-landforms',
      contributor: 'terrain-weiss-landforms'
    }
  },
  'GPUTerrainWeissLandforms local, grid above the maximum cell count': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells',
      received: 11184811,
      expected: {
        maximum: 11184810
      },
      id: 'terrain-weiss-landforms',
      contributor: 'terrain-weiss-landforms'
    }
  },
  'GPUFlowAggregation maximum pairCapacity 2^23 with weights and sorted sums, rows at the binding limit':
    {
      phase: 'construction',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'rows x maximumProbeCount',
        received: 140737488355328,
        expected: {
          maximum: 4294967295
        },
        id: 'flow-aggregation',
        contributor: 'flow-aggregation'
      }
    },
  'GPUMapMatching entryCapacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'entryCapacity',
      received: 33554433,
      expected: {
        minimum: 1,
        maximum: 33554432
      },
      id: 'map-matching',
      contributor: 'map-matching'
    }
  },
  'GPUMapMatching edges above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'topology edges',
      received: 11184811,
      expected: {
        maximum: 11184810
      },
      id: 'map-matching',
      contributor: 'map-matching'
    }
  },
  'GPUNetworkSnapping candidate search, edges above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'topology edges with candidateCapacity',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'network-snapping',
      contributor: 'network-snapping'
    }
  },
  'GPUNetworkSnapping exact scan, points times edges above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions times topology edges without candidateCapacity',
      received: 134217736,
      expected: {
        maximum: 134217728
      },
      id: 'network-snapping',
      contributor: 'network-snapping'
    }
  },
  'GPUNetworkCentrality local iterations 1, maximum rounds, four lanes, over the command node cap':
    {
      phase: 'construction',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'estimated command node count',
        received: 192581,
        expected: {
          maximum: 32768
        },
        id: 'network-centrality',
        contributor: 'network-centrality'
      }
    },
  'GPUNetworkReachability nodes above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.costs rows',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'network-reachability',
      contributor: 'network-reachability'
    }
  },
  'GPUNetworkKFunction events at the pair-count limit, row blocks past the invocation limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'rowsPerBlock x events',
      received: 4294976562,
      expected: {
        maximum: 4294967295
      },
      id: 'network-k-function',
      contributor: 'network-k-function'
    }
  },
  'GPUNetworkKFunction events above the pair-count limit': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'eventCount',
      received: 92683,
      expected: {
        maximum: 92682
      },
      id: 'network-k-function',
      contributor: 'network-k-function'
    }
  },
  'GPUNetworkCoarsening weighted superedges above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.edges.ids rows with output.edgeWeights',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'network-coarsening',
      contributor: 'network-coarsening'
    }
  },
  'GPUNetworkLineGraph planar, base edges above the maximum, arcs and banned turns at the binding limit':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'topology edges',
        received: 16777217,
        expected: {
          maximum: 16777216
        },
        id: 'network-line-graph',
        contributor: 'network-line-graph'
      }
    },
  'GPUNetworkLineGraph great-circle, base edges above the maximum, arcs and banned turns at the binding limit':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'topology edges with a great-circle spatialContext',
        received: 8388609,
        expected: {
          maximum: 8388608
        },
        id: 'network-line-graph',
        contributor: 'network-line-graph'
      }
    },
  'GPUNetworkNoding intersection capacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'intersectionCapacity',
      received: 1048577,
      expected: {
        minimum: 1,
        maximum: 1048576
      },
      id: 'network-noding',
      contributor: 'network-noding'
    }
  },
  'GPUNetworkKFunction events at the pair-count limit, one-row blocks, over the command node budget':
    {
      phase: 'construction',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'estimated command node count',
        received: 1112384,
        expected: {
          maximum: 32768
        },
        id: 'network-k-function',
        contributor: 'network-k-function'
      }
    },
  'GPUNetworkCostMatrix lane count 1, 40 rows, maximum rounds, over the command node budget': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 41041,
      expected: {
        maximum: 32768
      },
      id: 'network-cost-matrix',
      contributor: 'network-cost-matrix'
    }
  },
  'GPUNetworkCostTable lane count 1, 600 nodes, default rounds, over the command node budget': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 42002,
      expected: {
        maximum: 32768
      },
      id: 'network-cost-table',
      contributor: 'network-cost-table'
    }
  },
  'GPULocationAllocation p-median, 11,000 greedy steps, over the command node budget': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'estimated command node count',
      received: 33005,
      expected: {
        maximum: 32768
      },
      id: 'location-allocation',
      contributor: 'location-allocation'
    }
  },
  'GPUPointCloudNormalScale scale count one above 32': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'radii.length',
      received: 33,
      expected: {
        minimum: 1,
        maximum: 32
      },
      id: 'gpu-pointcloud-normal-scale',
      contributor: 'gpu-pointcloud-normal-scale'
    }
  },
  'GPUPointCloudChange unindexed scan one core above the work limit': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'cores',
      received: false,
      expected:
        'cores x (reference + comparison) is 134234112, above the 134217728 limit of a scan without an index; pass grid',
      id: 'gpu-pointcloud-change',
      contributor: 'gpu-pointcloud-change'
    }
  },
  'GPUPointCloudChange sampling covariance one point above 32768 per cloud': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'samplingCovariance points per cloud',
      received: 32769,
      expected: {
        maximum: 32768
      },
      id: 'gpu-pointcloud-change',
      contributor: 'gpu-pointcloud-change'
    }
  },
  'GPUPointCloudNeighborhoods capacity one above 64': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'capacity',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'gpu-pointcloud-neighborhoods',
      contributor: 'gpu-pointcloud-neighborhoods'
    }
  },
  'GPUPointCloudNeighborhoods unindexed search one query above the work limit': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'queries',
      received: false,
      expected:
        'queries x positions is 134283264, above the 134217728 limit of a search without an index; pass grid',
      id: 'gpu-pointcloud-neighborhoods',
      contributor: 'gpu-pointcloud-neighborhoods'
    }
  },
  'GPUPointCloudNeighborhoods grid axis one above 16777216 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid.dimensions[0]',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'gpu-pointcloud-neighborhoods',
      contributor: 'gpu-pointcloud-neighborhoods'
    }
  },
  'GPUPointCloudFeatures capacity one above 64': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'capacity',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'gpu-pointcloud-features',
      contributor: 'gpu-pointcloud-features'
    }
  },
  'GPUPointCloudStatisticalFilter capacity one above 64': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'capacity',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'gpu-pointcloud-statistical-filter',
      contributor: 'gpu-pointcloud-statistical-filter'
    }
  },
  'GPUPointCloudGround thirty-three radii': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'radii.length',
      received: 33,
      expected: {
        minimum: 1,
        maximum: 32
      },
      id: 'gpu-pointcloud-ground',
      contributor: 'gpu-pointcloud-ground'
    }
  },
  'GPUPointCloudGround radius one above 32 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'radii[0]',
      received: 33,
      expected: {
        minimum: 1,
        maximum: 32
      },
      id: 'gpu-pointcloud-ground',
      contributor: 'gpu-pointcloud-ground'
    }
  },
  'GPUPointCloudSMRF window one above 64 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'window in cells',
      received: 65,
      expected: {
        minimum: 1,
        maximum: 64
      },
      id: 'gpu-pointcloud-smrf',
      contributor: 'gpu-pointcloud-smrf'
    }
  },
  'GPUPointCloudSMRF cut one above 32 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'cut in cells',
      received: 33,
      expected: {
        minimum: 0,
        maximum: 32
      },
      id: 'gpu-pointcloud-smrf',
      contributor: 'gpu-pointcloud-smrf'
    }
  },
  'GPUPointCloudICP maximumIterations one above 100': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'maximumIterations',
      received: 101,
      expected: {
        minimum: 1,
        maximum: 100
      },
      id: 'gpu-pointcloud-icp',
      contributor: 'gpu-pointcloud-icp'
    }
  },
  'GPUPointCloudICP unindexed search one source point above the work limit': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'source',
      received: false,
      expected:
        'source x target is 134225920, above the 134217728 limit of a search without an index; pass targetGrid',
      id: 'gpu-pointcloud-icp',
      contributor: 'gpu-pointcloud-icp'
    }
  },
  'GPUPointCloudHeight inverse-distance power one above 8': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'interpolation.power',
      received: false,
      expected:
        'interpolation.power must be positive in float32 and at most GPU_POINT_CLOUD_MAXIMUM_IDW_POWER (8)',
      id: 'gpu-pointcloud-height',
      contributor: 'gpu-pointcloud-height'
    }
  },
  'GPUPointCloudChange indexed cylinder visiting 102 cells per axis': {
    phase: 'construction',
    validation: {
      code: 'REQUIREMENT',
      path: 'each',
      received: false,
      expected:
        'each core would visit 1061208 grid cells, above the 1048576 limit; enlarge the grid cellSize',
      id: 'gpu-pointcloud-change',
      contributor: 'gpu-pointcloud-change'
    }
  },
  'GPUPointCloudNeighborhoods grid of more than 4294967294 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cell count',
      received: 4294967296,
      expected: {
        maximum: 4294967294
      },
      id: 'gpu-pointcloud-neighborhoods',
      contributor: 'gpu-pointcloud-neighborhoods'
    }
  },
  'GPUPointCloudHeight inverse-distance reach one above 64 cells': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'interpolation.radius in cells',
      received: 65,
      expected: {
        maximum: 64
      },
      id: 'gpu-pointcloud-height',
      contributor: 'gpu-pointcloud-height'
    }
  },
  'GPUMultipleTestingAdjustment none one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'binding 0 chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'multiple-testing-adjustment',
      contributor: 'multiple-testing-adjustment'
    }
  },
  'GPUMultipleTestingAdjustment bonferroni one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'binding 0 chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'multiple-testing-adjustment',
      contributor: 'multiple-testing-adjustment'
    }
  },
  'GPUMultipleTestingAdjustment holm one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'binding 0 chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'multiple-testing-adjustment',
      contributor: 'multiple-testing-adjustment'
    }
  },
  'GPUMultipleTestingAdjustment benjamini-hochberg one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'binding 0 chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'multiple-testing-adjustment',
      contributor: 'multiple-testing-adjustment'
    }
  },
  'GPUMultipleTestingAdjustment benjamini-yekutieli one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'binding 0 chunk 0',
      received: 134217732,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce the chunk row count or output capacity'
      },
      id: 'multiple-testing-adjustment',
      contributor: 'multiple-testing-adjustment'
    }
  },
  'GPUGlobalPermutationTest moran permutations x row blocks above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutations x row blocks',
      received: 16777232,
      expected: {
        maximum: 16777215
      },
      id: 'global-moran',
      contributor: 'global-moran'
    }
  },
  'GPUGlobalPermutationTest geary permutations x row blocks above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutations x row blocks',
      received: 16777232,
      expected: {
        maximum: 16777215
      },
      id: 'global-geary',
      contributor: 'global-geary'
    }
  },
  'GPUGlobalPermutationTest getisOrdG permutations x row blocks above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutations x row blocks',
      received: 16777232,
      expected: {
        maximum: 16777215
      },
      id: 'global-getisOrdG',
      contributor: 'global-getisOrdG'
    }
  },
  'GPUGlobalPermutationTest bivariateMoran permutations x row blocks above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutations x row blocks',
      received: 16777232,
      expected: {
        maximum: 16777215
      },
      id: 'global-bivariateMoran',
      contributor: 'global-bivariateMoran'
    }
  },
  'GPUGlobalPermutationTest leesL permutations x row blocks above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'permutations x row blocks',
      received: 16777232,
      expected: {
        maximum: 16777215
      },
      id: 'global-leesL',
      contributor: 'global-leesL'
    }
  },
  'GPUSpatialRegressionDiagnostics restandardize false rows above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'rows x (predictorCount + 3)',
      received: 33554448,
      expected: {
        maximum: 33554432
      },
      id: 'spatial-regression-diagnostics',
      contributor: 'spatial-regression-diagnostics'
    }
  },
  'GPUSpatialRegressionDiagnostics restandardize true rows above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'rows x (predictorCount + 3)',
      received: 33554448,
      expected: {
        maximum: 33554432
      },
      id: 'spatial-regression-diagnostics',
      contributor: 'spatial-regression-diagnostics'
    }
  },
  'GPUSegmentIntersection vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'left.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'segment-intersection',
      contributor: 'segment-intersection'
    }
  },
  'GPUGeometryPredicates polygons above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'geometry.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'geometry-predicates',
      contributor: 'geometry-predicates'
    }
  },
  'GPULineNoding vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'line-noding',
      contributor: 'line-noding'
    }
  },
  'GPULineNoding piece bound above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines + 4 x intersectionCapacity',
      received: 4194308,
      expected: {
        maximum: 4194304
      },
      id: 'line-noding',
      contributor: 'line-noding'
    }
  },
  'GPUPolygonize vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'polygonize',
      contributor: 'polygonize'
    }
  },
  'GPUMakeValid vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'make-valid',
      contributor: 'make-valid'
    }
  },
  'GPUGeometryValidity polygons above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'polygons.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'geometry-validity',
      contributor: 'geometry-validity'
    }
  },
  'GPULineClipByPolygon line vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows + 2 x intersectionCapacity',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'line-clip-by-polygon',
      contributor: 'line-clip-by-polygon'
    }
  },
  'GPULineClipByPolygon polygon vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'polygons.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'line-clip-by-polygon',
      contributor: 'line-clip-by-polygon'
    }
  },
  'GPULineClipByPolygon sub-piece bound above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows + 2 x intersectionCapacity',
      received: 4194306,
      expected: {
        maximum: 4194304
      },
      id: 'line-clip-by-polygon',
      contributor: 'line-clip-by-polygon'
    }
  },
  'GPUSharedPaths vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'left.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'shared-paths',
      contributor: 'shared-paths'
    }
  },
  'GPUSharedPaths intersections above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'intersectionCapacity',
      received: 2796203,
      expected: {
        minimum: 1,
        maximum: 2796202
      },
      id: 'shared-paths',
      contributor: 'shared-paths'
    }
  },
  'GPULineSplit vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'line-split',
      contributor: 'line-split'
    }
  },
  'GPULineSplit piece bound above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'lines + 4 x intersectionCapacity',
      received: 4194308,
      expected: {
        maximum: 4194304
      },
      id: 'line-split',
      contributor: 'line-split'
    }
  },
  'GPUPolygonOverlay vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'hotPixelSegmentTables',
      received: 600871040,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce capacities or split input features'
      },
      id: 'polygon-overlay',
      contributor: 'polygon-overlay'
    }
  },
  'GPUPolygonOverlay hot pixels above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'hotPixelSegmentTables',
      received: 134217856,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce capacities or split input features'
      },
      id: 'polygon-overlay',
      contributor: 'polygon-overlay'
    }
  },
  'GPUCoverageSimplification vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'coverage-simplification',
      contributor: 'coverage-simplification'
    }
  },
  'GPUCoverageValidity vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'segmentTable',
      received: 134217760,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'split the input or reduce capacity'
      },
      id: 'coverage-validity',
      contributor: 'coverage-validity'
    }
  },
  'GPULineDensity grid above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'columns x rows',
      received: 16777216,
      expected: {
        maximum: 16777215
      },
      id: 'line-density',
      contributor: 'line-density'
    }
  },
  'GPULineLengthPerPolygon positions above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions rows',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'line-length-per-polygon',
      contributor: 'line-length-per-polygon'
    }
  },
  'GPULinearReferencing planar queries above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'queryPositions chunk rows',
      received: 3728271,
      expected: {
        maximum: 3728270
      },
      id: 'linear-referencing',
      contributor: 'linear-referencing'
    }
  },
  'GPULineLocate events above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'eventLineIndices chunk rows',
      received: 5592406,
      expected: {
        maximum: 5592405
      },
      id: 'line-locate',
      contributor: 'line-locate'
    }
  },
  'GPUBufferSurface polygons above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'hotPixelSegmentTables',
      received: 134221952,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce capacities or split input features'
      },
      id: 'buffer-surface',
      contributor: 'buffer-surface'
    }
  },
  'GPUPolygonTriangulation vertices above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'scratch nodes',
      received: 4793497,
      expected: {
        maximum: 4793490
      },
      id: 'polygon-triangulation',
      contributor: 'polygon-triangulation'
    }
  },
  'GPUMinimumClearance polygons above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'geometry.positions rows',
      received: 4194305,
      expected: {
        maximum: 4194304
      },
      id: 'minimum-clearance',
      contributor: 'minimum-clearance'
    }
  },
  'GPULineLengthPerPolygon every output, maximum crossings 1025': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'maximumCrossingCount',
      received: 1025,
      expected: {
        minimum: 1,
        maximum: 1024
      },
      id: 'line-length-per-polygon',
      contributor: 'line-length-per-polygon'
    }
  },
  'GPULineLengthPerPolygon every output, maximum crossings 4294967295': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'maximumCrossingCount',
      received: 4294967295,
      expected: {
        minimum: 1,
        maximum: 1024
      },
      id: 'line-length-per-polygon',
      contributor: 'line-length-per-polygon'
    }
  },
  'GPUOffsetCurve one vertex with quadSegments above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'quadSegments',
      received: 257,
      expected: {
        minimum: 1,
        maximum: 256
      },
      id: 'offset-curve',
      contributor: 'offset-curve'
    }
  },
  'GPURandomPointsInPolygon triangulation scratch above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'scratch nodes',
      received: 4793504,
      expected: {
        maximum: 4793490
      },
      id: 'random-points-in-polygon',
      contributor: 'random-points-in-polygon'
    }
  },
  'GPULineLengthPerPolygon pairs above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'pairCapacity',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'line-length-per-polygon',
      contributor: 'line-length-per-polygon'
    }
  },
  'GPULinearReferencing great-circle positions above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions rows',
      received: 2097153,
      expected: {
        maximum: 2097152
      },
      id: 'linear-referencing',
      contributor: 'linear-referencing'
    }
  },
  'GPUBufferSurface lines above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'hotPixelSegmentTables',
      received: 134219520,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce capacities or split input features'
      },
      id: 'buffer-surface',
      contributor: 'buffer-surface'
    }
  },
  'GPUPolygonTriangulation vertices above the maximum without z-order hash': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'scratch nodes',
      received: 8388610,
      expected: {
        maximum: 8388608
      },
      id: 'polygon-triangulation',
      contributor: 'polygon-triangulation'
    }
  },
  'GPULinearReferencing planar positions above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions rows',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'linear-referencing',
      contributor: 'linear-referencing'
    }
  },
  'GPUCellCover quadbin candidateCapacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'candidateCapacity',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'cell-cover',
      contributor: 'cell-cover'
    }
  },
  'GPUCellCover quadbin edge slab entries above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'positions rows + edgeSlabEntryCapacity',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'cell-cover',
      contributor: 'cell-cover'
    }
  },
  'GPUNeighborSearch radius gridSize above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'REQUIREMENT',
      path: 'gridSize',
      received: false,
      expected:
        'gridSize cells needs 134217732 bytes, above the device limit of 134217728; reduce rows, grid size or neighborCount, or use smaller independent queries',
      id: 'neighbor-search',
      contributor: 'neighbor-search'
    }
  },
  'GPUNearestFeatureJoin neighbors node table above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'node table words (leafCapacity, queries)',
      received: 41943037,
      expected: {
        maximum: 33554432
      },
      id: 'nearest-feature-join',
      contributor: 'nearest-feature-join'
    }
  },
  'GPUSpatialJoinCandidates polygons pair capacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.leftRows.length',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'spatial-join-candidates',
      contributor: 'spatial-join-candidates'
    }
  },
  'GPUTrackSimilarity pairs above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'pairA.length',
      received: 16777216,
      expected: {
        minimum: 0,
        maximum: 16777215
      },
      id: 'track-similarity',
      contributor: 'track-similarity'
    }
  },
  'GPUTrajectoryEncounters lattice above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'columns x rows x bucketCount',
      received: 33556232,
      expected: {
        maximum: 33554431
      },
      id: 'trajectory-encounters',
      contributor: 'trajectory-encounters'
    }
  },
  'GPUTrajectoryPlayhead tracks above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'tracks',
      received: 6710887,
      expected: {
        maximum: 6710886
      },
      id: 'trajectory-playhead',
      contributor: 'trajectory-playhead'
    }
  },
  'GPUHuffTradeAreas normalization scratch one row above the binding limit': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'normalization scratch',
      received: 134217744,
      expected: {
        maximum: 134217728,
        source: 'device.limits.maxStorageBufferBindingSize',
        hint: 'reduce row or group counts'
      },
      id: 'huff-trade-areas',
      contributor: 'huff-trade-areas'
    }
  },
  'GPUKernelDensity index grid above the maximum gaussian': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'indexGridSize cells',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'kernel-density',
      contributor: 'kernel-density'
    }
  },
  'GPUKernelDensity index grid above the maximum epanechnikov': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'indexGridSize cells',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'kernel-density',
      contributor: 'kernel-density'
    }
  },
  'GPUArealInterpolation cells above the maximum, pairs at the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sourceZones rows',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'areal-interpolation',
      contributor: 'areal-interpolation'
    }
  },
  'GPUArealInterpolation cells at the maximum, pairs above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'pair slots (output.weights.neighbors rows, cells)',
      received: 33554431,
      expected: {
        maximum: 16777215
      },
      id: 'areal-interpolation',
      contributor: 'areal-interpolation'
    }
  },
  'GPUDotDensity capacity above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'output.positions rows',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'dot-density',
      contributor: 'dot-density'
    }
  },
  'GPULocationAllocation facilities above the maximum set-covering': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'facilityCount',
      received: 16777217,
      expected: {
        minimum: 1,
        maximum: 16777216
      },
      id: 'location-allocation',
      contributor: 'location-allocation'
    }
  },
  'GPULocationAllocation sparse entries above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'sparseCosts entries',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'location-allocation',
      contributor: 'location-allocation'
    }
  },
  'GPUSimilarLocations columns above the maximum': {
    phase: 'construction',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'columnCount',
      received: 16777216,
      expected: {
        minimum: 1,
        maximum: 16777215
      },
      id: 'similar-locations',
      contributor: 'similar-locations'
    }
  },
  'GPUDBSCAN rows and clusters, index grid above the maximum atomic': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'gridSize cells',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'spatial-clustering',
      contributor: 'spatial-clustering'
    }
  },
  'GPUDBSCAN rows and clusters, index grid at the maximum sorted, clusters above the sorted maximum':
    {
      phase: 'construction',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'clusters.ids rows with sorted centroid sums',
        received: 16777216,
        expected: {
          maximum: 16777215
        },
        id: 'spatial-clustering',
        contributor: 'spatial-clustering'
      }
    },
  'GPUDBSCAN rows and clusters, index grid above the maximum sorted, clusters above the sorted maximum':
    {
      phase: 'construction',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'clusters.ids rows with sorted centroid sums',
        received: 16777216,
        expected: {
          maximum: 16777215
        },
        id: 'spatial-clustering',
        contributor: 'spatial-clustering'
      }
    },
  'GPUDBSCAN rows and clusters, index grid above the maximum sorted below the segment-sum dispatch limit':
    {
      phase: 'declaration',
      validation: {
        code: 'INTEGER_RANGE',
        path: 'gridSize cells',
        received: 33554432,
        expected: {
          maximum: 33554431
        },
        id: 'spatial-clustering',
        contributor: 'spatial-clustering'
      }
    },
  'GPUInverseDistanceWeighting index grid above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'indexGridSize cells',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'inverse-distance-weighting',
      contributor: 'inverse-distance-weighting'
    }
  },
  'GPUKriging index grid above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'indexGridSize cells',
      received: 33554432,
      expected: {
        maximum: 33554431
      },
      id: 'kriging',
      contributor: 'kriging'
    }
  },
  'GPUSkaterRegions slots 16 columns above the maximum 1024 regions': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'weights.neighbors rows x (2 + columnCount)',
      received: 33554448,
      expected: {
        maximum: 33554432
      },
      id: 'skater-regions',
      contributor: 'skater-regions'
    }
  },
  'GPURegionPartitionEvaluation rows above the maximum 1 columns': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'labels rows',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'region-partition-evaluation',
      contributor: 'region-partition-evaluation'
    }
  },
  'GPUAZPRegions slots above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'weights.neighbors rows',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'azp-regions',
      contributor: 'azp-regions'
    }
  },
  'GPUWardRegions slots above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'weights.neighbors rows',
      received: 16777217,
      expected: {
        maximum: 16777216
      },
      id: 'ward-regions',
      contributor: 'ward-regions'
    }
  },
  'GPUMaxPRegions slots above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'weights.neighbors rows',
      received: 8388609,
      expected: {
        maximum: 8388608
      },
      id: 'max-p-regions',
      contributor: 'max-p-regions'
    }
  },
  'addSparseResistanceConnectivityRecipe vertices above the maximum': {
    phase: 'declaration'
  },
  'addSparseResistanceConnectivityRecipe edges above the maximum': {
    phase: 'declaration'
  },
  'addRasterResistanceConnectivityRecipe grid cells above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'grid cells (width * height)',
      received: 1049600,
      expected: {
        maximum: 1048576
      },
      id: 'raster-resistance-connectivity',
      contributor: 'raster-resistance-connectivity'
    }
  },
  'addRasterResistanceConnectivityRecipe edge slots above the maximum': {
    phase: 'declaration',
    validation: {
      code: 'INTEGER_RANGE',
      path: 'edge slots (grid cells * 4 for neighborhood 8)',
      received: 2101248,
      expected: {
        maximum: 2097152
      },
      id: 'raster-resistance-connectivity',
      contributor: 'raster-resistance-connectivity'
    }
  }
} satisfies Record<string, Pick<KernelAuditRejection, 'phase' | 'validation'>>;
