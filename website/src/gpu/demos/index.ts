// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Flagship live demos. Each one runs real @luma-contributors/analysis operations on the GPU (the
// command graph is compiled and encoded in the browser) and draws the operation's output buffers
// directly with luma.gl. Scenes throw on a non-WebGPU device, so <GpuCanvas> keeps its fallback.
//
// Look up by catalog entry id (getDemoScene) or by exported symbol (getDemoSceneForSymbol).

import type {GpuSceneLoader, GpuSceneParameters} from '../types';
import {
  DBSCAN_PARAMETERS,
  DELAUNAY_VORONOI_PARAMETERS,
  HOT_SPOTS_PARAMETERS,
  IDW_PARAMETERS,
  KERNEL_DENSITY_PARAMETERS,
  K_MEANS_PARAMETERS,
  LOCAL_MORANS_I_PARAMETERS,
  NETWORK_ISOCHRONES_PARAMETERS,
  SHORTEST_PATH_PARAMETERS,
  TERRAIN_HYDROLOGY_PARAMETERS
} from './parameters';

export type DemoDescriptor = {
  /** Stable demo id, also the file name in this folder. */
  id: string;
  /** Short name for captions. */
  title: string;
  /** One line: what the picture shows. */
  caption: string;
  /** Analysis symbols the demo runs. */
  symbols: string[];
  /** Catalog entry ids that show this demo (extra aliases are harmless). */
  entryIds: string[];
  /** True when the pointer steers the analysis. */
  interactive: boolean;
  /** Slider parameters, known without loading the scene (match the module's `parameters`). */
  parameters: GpuSceneParameters;
  /** Preferred canvas aspect ratio (width / height). */
  aspectRatio: number;
  loader: GpuSceneLoader;
};

export const DEMOS: readonly DemoDescriptor[] = [
  {
    id: 'kernel-density',
    title: 'Kernel density',
    caption:
      'Move across the map to size the kernel: the ring is the bandwidth of Japanese earthquake density.',
    symbols: ['GPUKernelDensity'],
    entryIds: ['kernel-density'],
    interactive: true,
    parameters: KERNEL_DENSITY_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./kernel-density')
  },
  {
    id: 'local-morans-i',
    title: "Local Moran's I",
    caption:
      'Local Moran clusters of wet and dry gauges around the Carolinas on September 27, 2024. Hover a gauge to see its neighbors and its class.',
    symbols: ['GPULocalMoran', 'GPUNeighborSearch'],
    entryIds: ['local-morans-i'],
    interactive: true,
    parameters: LOCAL_MORANS_I_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./local-morans-i')
  },
  {
    id: 'dbscan',
    title: 'DBSCAN',
    caption:
      'Hover an epicenter: the ring is epsilon, and the neighbors inside it decide core, border or noise.',
    symbols: ['GPUSpatialClustering'],
    entryIds: ['dbscan-clustering'],
    interactive: true,
    parameters: DBSCAN_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./dbscan')
  },
  {
    id: 'k-means',
    title: 'K-means',
    caption:
      'Five centers converge on 820 earthquake epicenters around Japan. Hover one to see its cluster and its distance to the center.',
    symbols: ['GPUKMeans'],
    entryIds: ['k-means-clustering'],
    interactive: true,
    parameters: K_MEANS_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./k-means')
  },
  {
    id: 'idw',
    title: 'Inverse distance weighting',
    caption:
      'Point at a place: the rain gauges that carry its IDW estimate link to it. Hurricane Helene rainfall in millimeters.',
    symbols: ['GPUInverseDistanceWeighting'],
    entryIds: ['inverse-distance-weighting'],
    interactive: true,
    parameters: IDW_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./idw')
  },
  {
    id: 'delaunay-voronoi',
    title: 'Delaunay and Voronoi',
    caption:
      'Rain gauge sites define the triangles and Voronoi cells. Move one site to test a relocation.',
    symbols: ['GPUDelaunayTessellation', 'GPUVoronoiDiagram'],
    entryIds: ['delaunay-tessellation', 'voronoi-diagram'],
    interactive: true,
    parameters: DELAUNAY_VORONOI_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./delaunay-voronoi')
  },
  {
    id: 'terrain-hydrology',
    title: 'Hillshade and streams',
    caption:
      'Aim the sun: Horn hillshade follows it, over a stream network from depression-filled D8 flow.',
    symbols: ['GPUTerrainDerivatives', 'GPUTerrainFlow'],
    entryIds: ['terrain-flow'],
    interactive: true,
    parameters: TERRAIN_HYDROLOGY_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./terrain-hydrology')
  },
  {
    id: 'network-isochrones',
    title: 'Network reachability',
    caption:
      'Reachability: least travel cost over a street grid from a moving origin; the reachable area grows.',
    symbols: ['GPUNetworkReachability'],
    entryIds: ['network-reachability'],
    interactive: true,
    parameters: NETWORK_ISOCHRONES_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./network-isochrones')
  },
  {
    id: 'hot-spots',
    title: 'Hot spots (Getis-Ord Gi*)',
    caption:
      'Compare rainfall clusters on September 26 and 27, 2024. Hover a gauge to see the neighborhood behind its Gi* score.',
    symbols: ['GPUHotSpotAnalysis'],
    entryIds: ['hot-spot-analysis-gi-star', 'hot-spot-analysis'],
    interactive: true,
    parameters: HOT_SPOTS_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./hot-spots')
  },
  {
    id: 'shortest-path',
    title: 'Shortest path',
    caption: 'Point at a destination; predecessors become a route. Congestion makes it detour.',
    symbols: ['GPUNetworkPathExtraction'],
    entryIds: ['network-path-extraction', 'shortest-path'],
    interactive: true,
    parameters: SHORTEST_PATH_PARAMETERS,
    aspectRatio: 16 / 9,
    loader: () => import('./shortest-path')
  }
];

/** Every catalog entry id that has a live demo. */
export const DEMO_ENTRY_IDS: readonly string[] = DEMOS.flatMap(demo => demo.entryIds);

/** Every analysis symbol that has a live demo. */
export const DEMO_SYMBOLS: readonly string[] = DEMOS.flatMap(demo => demo.symbols);

/** The demo for a catalog entry id such as `local-morans-i`, if there is one. */
export function getDemo(entryId: string): DemoDescriptor | undefined {
  return DEMOS.find(demo => demo.entryIds.includes(entryId));
}

/** The demo that runs an exported symbol such as `GPULocalMoran`, if there is one. */
export function getDemoForSymbol(symbol: string): DemoDescriptor | undefined {
  return DEMOS.find(demo => demo.symbols.includes(symbol));
}

/** Scene loader for a catalog entry id. Pass it to `<GpuCanvas scene={...}>`. */
export function getDemoScene(entryId: string): GpuSceneLoader | undefined {
  return getDemo(entryId)?.loader;
}

/** Scene loader for an exported symbol. */
export function getDemoSceneForSymbol(symbol: string): GpuSceneLoader | undefined {
  return getDemoForSymbol(symbol)?.loader;
}

/** Slider parameters for a catalog entry id or an analysis symbol, without loading the scene. */
export function getDemoParameters(entryIdOrSymbol: string): GpuSceneParameters | undefined {
  return (getDemo(entryIdOrSymbol) ?? getDemoForSymbol(entryIdOrSymbol))?.parameters;
}
