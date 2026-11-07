// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Flagship live demos. Each one runs real @luma-contributors/analysis operations on the GPU (the
// command graph is compiled and encoded in the browser) and draws the operation's output buffers
// directly with luma.gl. Scenes throw on a non-WebGPU device, so <GpuCanvas> keeps its fallback.
//
// Look up by catalog entry id (getDemoScene) or by exported symbol (getDemoSceneForSymbol).

import type {GpuSceneLoader} from '../types';

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
  /** Preferred canvas aspect ratio (width / height). */
  aspectRatio: number;
  loader: GpuSceneLoader;
};

export const DEMOS: readonly DemoDescriptor[] = [
  {
    id: 'kernel-density',
    title: 'Kernel density',
    caption: 'Bandwidth sweeps; GPUKernelDensity re-evaluates the surface every frame.',
    symbols: ['GPUKernelDensity'],
    entryIds: ['kernel-density'],
    interactive: false,
    aspectRatio: 16 / 9,
    loader: () => import('./kernel-density')
  },
  {
    id: 'local-morans-i',
    title: "Local Moran's I",
    caption: 'Neighbor search, then hot (red) and cold (blue) clusters of a drifting field.',
    symbols: ['GPULocalMoran', 'GPUNeighborSearch'],
    entryIds: ['local-morans-i'],
    interactive: false,
    aspectRatio: 16 / 9,
    loader: () => import('./local-morans-i')
  },
  {
    id: 'dbscan',
    title: 'DBSCAN',
    caption: 'Epsilon sweeps; clusters merge, split and shed noise on the GPU.',
    symbols: ['GPUSpatialClustering'],
    entryIds: ['dbscan-clustering'],
    interactive: false,
    aspectRatio: 16 / 9,
    loader: () => import('./dbscan')
  },
  {
    id: 'k-means',
    title: 'K-means',
    caption: 'Lloyd iterations replayed one to twelve, then a new seed.',
    symbols: ['GPUKMeans'],
    entryIds: ['k-means-clustering'],
    interactive: false,
    aspectRatio: 16 / 9,
    loader: () => import('./k-means')
  },
  {
    id: 'idw',
    title: 'Inverse distance weighting',
    caption: 'Samples become a surface; power sweeps, and one sample follows the pointer.',
    symbols: ['GPUInverseDistanceWeighting'],
    entryIds: ['inverse-distance-weighting'],
    interactive: true,
    aspectRatio: 16 / 9,
    loader: () => import('./idw')
  },
  {
    id: 'delaunay-voronoi',
    title: 'Delaunay and Voronoi',
    caption:
      'Drifting sites are re-triangulated every frame; Voronoi cells come from the triangles.',
    symbols: ['GPUDelaunayTessellation', 'GPUVoronoiDiagram'],
    entryIds: ['delaunay-tessellation', 'voronoi-diagram'],
    interactive: true,
    aspectRatio: 16 / 9,
    loader: () => import('./delaunay-voronoi')
  },
  {
    id: 'terrain-hydrology',
    title: 'Hillshade and streams',
    caption: 'A circling sun (Horn hillshade) and a stream network from depression-filled D8 flow.',
    symbols: ['GPUTerrainDerivatives', 'GPUTerrainFlow'],
    entryIds: ['terrain-flow', 'slope-aspect-hillshade', 'hillshade'],
    interactive: false,
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
    aspectRatio: 16 / 9,
    loader: () => import('./network-isochrones')
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
