// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

export const RASTER_EXPANSION_OPERATORS: RasterOperatorTable = {
  GPURasterFillNoData: {
    family: 'expansion',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {
      input: 'elevation',
      values: 'elevation',
      validity: 'category',
      filled: 'category',
      invalidCount: 'measure'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterFillNoData.rule',
      derivation: 'curated',
      tex: String.raw`v(c)=\frac{\sum_{d\in D(c)}w(c,d)v(d)}{\sum_{d\in D(c)}w(c,d)}`,
      ascii:
        'Disk search uses original valid donors only; nearest, disk inverse-distance or four-quadrant interpolation preserves observed samples.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-fill-nodata.ts']
  },
  GPURasterRankFilter: {
    family: 'expansion',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {input: 'measure', values: 'measure', validity: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterRankFilter.rule',
      derivation: 'curated',
      tex: String.raw`q(c)=s_{\min(\lfloor pn\rfloor,n-1)}`,
      ascii:
        'Select sorted[min(floor(q*n),n-1)] from the finite square neighborhood; invalid centers remain invalid.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-rank-filter.ts']
  },
  GPURasterFractionalCoverage: {
    family: 'expansion',
    descriptor: {scope: 'pairwise', reduction: 'sum'},
    variables: {
      zoneCoordinates: 'measure',
      zoneIds: 'category',
      cellIds: 'category',
      fractions: 'fraction',
      count: 'measure',
      requiredCount: 'measure',
      overflow: 'category',
      zoneValidity: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'zone',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterFractionalCoverage.rule',
      derivation: 'curated',
      tex: String.raw`f_{zc}=\frac{\operatorname{area}(z\cap c)}{|\det A|}`,
      ascii:
        'Coverage is polygon-cell intersection area divided by affine cell area, assembled from clipped triangles.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts']
  },
  GPURasterFractionalZonalStatistics: {
    family: 'expansion',
    descriptor: {scope: 'zonal', reduction: 'weighted-mean'},
    variables: {
      input: 'measure',
      fractions: 'fraction',
      totals: 'measure',
      means: 'rate',
      coveredArea: 'measure',
      validArea: 'measure',
      zoneAggregateValidity: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'zone',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterFractionalZonalStatistics.rule',
      derivation: 'curated',
      tex: String.raw`T_z=\sum_c v_c f_{zc}|\det A|,\qquad \bar v_z=\frac{\sum_{c\;valid}v_cf_{zc}}{\sum_{c\;valid}f_{zc}}`,
      ascii:
        'Extensive totals sum value times fraction; density integrals additionally multiply cell area. Means divide weighted values by observed fractions.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts']
  },
  GPURasterReconstruction: {
    family: 'expansion',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      marker: 'measure',
      mask: 'measure',
      values: 'measure',
      validity: 'category',
      converged: 'category'
    },
    stages: ['representation', 'operator', 'iteration'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterReconstruction.rule',
      derivation: 'curated',
      tex: String.raw`R_{t+1}(c)=\min(M(c),\max_{d\in N(c)}R_t(d))`,
      ascii:
        'Geodesic dilation iterates min(mask, neighborhood maximum) until a verified fixed point; erosion is the dual. N(c) is c and its 8 neighbours by default (connectivity 8, so regions touching at a corner are connected); connectivity 4 uses the 4 edge neighbours.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts']
  },
  GPURasterFillHoles: {
    family: 'expansion',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {foreground: 'category', values: 'category', converged: 'category'},
    stages: ['representation', 'operator', 'iteration'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Reconstruct boundary-connected background and complement it to fill enclosed background components.'
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts']
  },
  GPURasterReprojection: {
    family: 'expansion',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {input: 'measure', values: 'measure', validity: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Map destination centers through the inverse projection and source affine, then sample the supplied source support.'
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-reprojection.ts']
  },
  GPURasterTileReprojection: {
    family: 'expansion',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {input: 'measure', values: 'measure', validity: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Map destination centers through the inverse projection and source affine, then sample the supplied source support.'
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts']
  },
  GPURasterWalkingCost: {
    family: 'expansion',
    descriptor: {scope: 'solver', reduction: 'argmin'},
    variables: {
      input: 'elevation',
      friction: 'rate',
      sources: 'category',
      values: 'measure',
      validity: 'category',
      converged: 'category'
    },
    stages: ['representation', 'operator', 'iteration'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterWalkingCost.rule',
      derivation: 'curated',
      tex: String.raw`D_{t+1}(v)=\min(D_t(v),\min_{u\in N(v)}(D_t(u)+w(u,v)))`,
      ascii:
        'Directed least costs obey D(v)=min(D(v),min_u(D(u)+cost(u,v))); Tobler speed is 6 exp(-3.5 abs(slope+0.05)) km/hour.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-walking-cost.ts']
  },
  GPURasterWatershedSegmentation: {
    family: 'expansion',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {input: 'elevation', markers: 'category', labels: 'category', converged: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Priority flood selects the smallest path maximum elevation, breaking ties by FIFO discovery age.'
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-watershed-segmentation.ts']
  },
  GPURasterSkeleton: {
    family: 'expansion',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {foreground: 'category', skeleton: 'category', converged: 'category'},
    stages: ['representation', 'operator', 'iteration'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Zhang-Suen alternates two deletion phases preserving connectivity until no foreground cell changes.'
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-skeleton.ts']
  },
  GPURasterMedialAxis: {
    family: 'expansion',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {foreground: 'category', medialAxis: 'category', radius: 'measure'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'window',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.GPURasterMedialAxis.rule',
      derivation: 'curated',
      tex: String.raw`r(c)=\min_{b\in background}\|c-b\|_2`,
      ascii:
        'Retain Euclidean foreground disks that are not contained in another foreground disk, including exterior background.',
      symbols: []
    },
    sources: ['modules/analysis/src/raster-analysis/gpu-raster-skeleton.ts']
  }
};
