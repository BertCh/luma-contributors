// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Our own organization of the catalog: by data type, by use case and by domain. Ids are stable URL
// slugs (/explore/data/<id>, /explore/use/<id>, /explore/domain/<id>).

export type Facet = {id: string; label: string; description: string};

export const DATA_TYPES = [
  {
    id: 'points',
    label: 'Points',
    description: 'Locations with attributes: events, stations, samples.'
  },
  {id: 'lines', label: 'Lines', description: 'Paths and segments: roads, rivers, tracks.'},
  {id: 'polygons', label: 'Polygons', description: 'Areas: parcels, tracts, zones, footprints.'},
  {id: 'raster', label: 'Rasters', description: 'Gridded values: imagery, land cover, surfaces.'},
  {id: 'terrain', label: 'Terrain', description: 'Elevation models and their derivatives.'},
  {id: 'network', label: 'Networks', description: 'Nodes and edges: streets, flights, flows.'},
  {id: 'table', label: 'Tables', description: 'Columns of attributes, with or without geometry.'},
  {id: 'time-series', label: 'Time series', description: 'Values indexed by time.'},
  {
    id: 'trajectories',
    label: 'Trajectories',
    description: 'Moving objects: tracks with timestamps.'
  },
  {
    id: 'point-cloud',
    label: 'Point clouds',
    description: '3D points from lidar or photogrammetry.'
  },
  {
    id: 'cells',
    label: 'Grid cells',
    description: 'H3, S2, A5 and other discrete global grid cells.'
  },
  {
    id: 'spatial-weights',
    label: 'Spatial weights',
    description: 'Neighbor relationships between features.'
  }
] as const satisfies readonly Facet[];

export const USE_CASES = [
  {
    id: 'measure',
    label: 'Measure',
    description: 'Lengths, areas, shapes, distances and descriptors.'
  },
  {
    id: 'summarize',
    label: 'Aggregate and summarize',
    description: 'Roll values up by zone, cell or group.'
  },
  {
    id: 'find-patterns',
    label: 'Find patterns',
    description: 'Clusters, hot spots, outliers and autocorrelation.'
  },
  {
    id: 'relate',
    label: 'Relate and join',
    description: 'Spatial joins, neighbors, predicates and overlays.'
  },
  {
    id: 'interpolate',
    label: 'Interpolate and predict',
    description: 'Surfaces and estimates from samples.'
  },
  {
    id: 'model',
    label: 'Model and test',
    description: 'Regression, inference and significance testing.'
  },
  {
    id: 'route',
    label: 'Route and reach',
    description: 'Paths, catchments, accessibility and allocation.'
  },
  {
    id: 'transform',
    label: 'Transform geometry',
    description: 'Buffer, clip, simplify, tessellate and repair.'
  },
  {
    id: 'classify',
    label: 'Classify and map',
    description: 'Breaks, scales, colors and cartographic layout.'
  },
  {
    id: 'simulate',
    label: 'Simulate flow',
    description: 'Water, movement and diffusion over surfaces and networks.'
  },
  {id: 'detect-change', label: 'Detect change', description: 'Compare epochs, surfaces and scans.'},
  {
    id: 'index',
    label: 'Index and search',
    description: 'Spatial keys, grids and nearest-neighbor search.'
  }
] as const satisfies readonly Facet[];

export const DOMAINS = [
  {
    id: 'spatial-statistics',
    label: 'Spatial statistics',
    description: 'Autocorrelation, regression and point processes.'
  },
  {
    id: 'urban',
    label: 'Cities and mobility',
    description: 'Urban form, access, segregation and movement.'
  },
  {
    id: 'ecology',
    label: 'Ecology and environment',
    description: 'Habitat, connectivity and landscape metrics.'
  },
  {
    id: 'hydrology',
    label: 'Hydrology and terrain',
    description: 'Flow, watersheds, slope and relief.'
  },
  {
    id: 'remote-sensing',
    label: 'Remote sensing',
    description: 'Spectral indices, imagery and change.'
  },
  {
    id: 'transport',
    label: 'Transport networks',
    description: 'Routing, centrality and network structure.'
  },
  {
    id: 'demographics',
    label: 'People and equity',
    description: 'Rates, small areas and fair access.'
  },
  {
    id: 'cartography',
    label: 'Geometry and cartography',
    description: 'Shapes, topology, labels and map making.'
  },
  {
    id: 'data-engineering',
    label: 'Data engineering',
    description: 'Tables, joins, indexing and preparation.'
  },
  {id: 'three-d', label: '3D and point clouds', description: 'Lidar, surfaces and 3D measurement.'}
] as const satisfies readonly Facet[];

export type DataTypeId = (typeof DATA_TYPES)[number]['id'];
export type UseCaseId = (typeof USE_CASES)[number]['id'];
export type DomainId = (typeof DOMAINS)[number]['id'];
