// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  type GraphChunkGroupKernelProps
} from '../../analysis-core/kernels/graph-chunk-groups';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';
import {
  appendTerrainAlternatingSchedule,
  type TerrainKernelSchedule
} from './terrain-tiled-sequence';
import {
  TERRAIN_TILED_WIDE_SUM_WGSL,
  TERRAIN_TILED_LOAD_SUM_WGSL,
  TERRAIN_TILED_STORE_SUM_WGSL
} from './terrain-tiled-wide-sum';

/** Shared by every tile, and separately instantiated for the coarse boundary forest. @internal */
export function createTerrainForestStorage<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: number
) {
  const words = (name: string, length = count) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const sumPlanes = [0, 1].map(
    side =>
      [
        words(`sums-${side}-0`, count * 8),
        words(`sums-${side}-1`, count * 8),
        words(`sums-${side}-2`, count * 8)
      ] as const
  );
  return {
    parents: words('parents'),
    firstChildren: words('first-children'),
    nextSiblings: words('next-siblings'),
    weights: createTransientView(graph, `${id}-weights`, 'float32', count),
    states: words('states'),
    successors: [words('successors-0', count * 2), words('successors-1', count * 2)] as const,
    sumPlanes,
    values: createTransientView(graph, `${id}-values`, 'float32', count),
    resolved: words('resolved')
  };
}
export type TerrainForestStorage = ReturnType<typeof createTerrainForestStorage>;

/** Borrow only this tile's rows while sharing maximum-sized backing allocations. @internal */
export function sliceTerrainForestStorage<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  storage: TerrainForestStorage,
  count: number
): TerrainForestStorage {
  const slice = <Format extends 'float32' | 'uint32'>(
    view: GraphDataView<Format>,
    length = count
  ) => graph.createDataView(view.buffer, {format: view.format, length});
  return {
    parents: slice(storage.parents),
    firstChildren: slice(storage.firstChildren),
    nextSiblings: slice(storage.nextSiblings),
    weights: slice(storage.weights),
    states: slice(storage.states),
    successors: [slice(storage.successors[0], count * 2), slice(storage.successors[1], count * 2)],
    sumPlanes: storage.sumPlanes.map(
      planes =>
        [
          slice(planes[0], count * 8),
          slice(planes[1], count * 8),
          slice(planes[2], count * 8)
        ] as const
    ),
    values: slice(storage.values),
    resolved: slice(storage.resolved)
  };
}

/** Exact 320-bit tour suffixes preserve every float32 runoff bit before subtraction. @internal */
export function appendTerrainForestSchedule<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  schedule: TerrainKernelSchedule<Parameters>,
  id: string,
  storage: TerrainForestStorage
): void {
  const count = storage.parents.length;
  const node = (
    suffix: string,
    invocationCount: number,
    bindings: GraphChunkGroupKernelProps<Parameters>['bindings'],
    body: string,
    declarations = ''
  ) =>
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-${suffix}`,
      pipelineId: `terrain-tiled-forest-${suffix}`,
      operation: 'GPUTerrainTiledFlow',
      invocationCount,
      bindings,
      body,
      declarations
    });
  const readPlanes = (side: number) =>
    ['firstPlane', 'secondPlane', 'thirdPlane'].map((name, index) =>
      terrainBinding(name, storage.sumPlanes[side][index])
    );
  const writePlanes = (side: number) =>
    ['firstOutput', 'secondOutput', 'thirdOutput'].map((name, index) =>
      terrainBinding(name, storage.sumPlanes[side][index], 'read_write')
    );
  schedule.push({
    repetitions: 1,
    nodes: [
      node(
        'link',
        count,
        [
          terrainBinding('parents', storage.parents),
          terrainBinding('children', storage.firstChildren),
          terrainBinding('siblings', storage.nextSiblings),
          terrainBinding('successors', storage.successors[0], 'read_write')
        ],
        `
let parent = parents[parentsOffset + index]; let child = children[childrenOffset + index]; let sibling = siblings[siblingsOffset + index];
let enter = index * 2u; let exitEdge = enter + 1u;
successors[successorsOffset + enter] = select(child * 2u, exitEdge, child == 0xffffffffu);
var next = exitEdge; if (parent != 0xffffffffu) { next = select(parent * 2u + 1u, sibling * 2u, sibling != 0xffffffffu); }
successors[successorsOffset + exitEdge] = next;`
      ),
      node(
        'weights',
        count * 2,
        [
          terrainBinding('weights', storage.weights),
          terrainBinding('states', storage.states),
          ...writePlanes(0)
        ],
        `
var value = terrainZeroSum(); if ((index & 1u) == 0u) { value = terrainEncodeRunoff(weights[weightsOffset + index / 2u], states[statesOffset + index / 2u]); }
terrainStoreRunoff(index, value);`,
        `${TERRAIN_TILED_WIDE_SUM_WGSL}\n${TERRAIN_TILED_STORE_SUM_WGSL}`
      )
    ]
  });
  const rounds = Math.ceil(Math.log2(Math.max(count * 2, 2))) + 1;
  const jumps = [0, 1].map(side =>
    node(
      'jump',
      count * 2,
      [
        terrainBinding('previous', storage.successors[side]),
        ...readPlanes(side),
        terrainBinding('next', storage.successors[1 - side], 'read_write'),
        ...writePlanes(1 - side)
      ],
      `
let following = previous[previousOffset + index]; var value = terrainLoadRunoff(index);
if (following != index) { value = terrainAddRunoff(value, terrainLoadRunoff(following)); }
next[nextOffset + index] = previous[previousOffset + following]; terrainStoreRunoff(index, value);`,
      `${TERRAIN_TILED_WIDE_SUM_WGSL}\n${TERRAIN_TILED_LOAD_SUM_WGSL}\n${TERRAIN_TILED_STORE_SUM_WGSL}`
    )
  ) as [ReturnType<typeof node>, ReturnType<typeof node>];
  appendTerrainAlternatingSchedule(schedule, jumps, rounds);
  schedule.push({
    repetitions: 1,
    nodes: [
      node(
        'resolve',
        count,
        [
          terrainBinding('parents', storage.parents),
          terrainBinding('successors', storage.successors[rounds % 2]),
          ...readPlanes(rounds % 2),
          terrainBinding('states', storage.states),
          terrainBinding('values', storage.values, 'read_write'),
          terrainBinding('resolved', storage.resolved, 'read_write')
        ],
        `
let terminal = successors[successorsOffset + index * 2u];
let finiteTree = (terminal & 1u) != 0u && parents[parentsOffset + terminal / 2u] == 0xffffffffu;
let total = terrainSubtractRunoff(terrainLoadRunoff(index * 2u), terrainLoadRunoff(index * 2u + 1u));
let value = terrainDecodeRunoff(total); let original = states[statesOffset + index];
let state = select(select(2u, 1u, finiteTree && total[10] == 0u && isFiniteValue(value)), 0u, original == 0u);
values[valuesOffset + index] = select(getNaN(), value, state == 1u); resolved[resolvedOffset + index] = state;`,
        `${TERRAIN_TILED_WIDE_SUM_WGSL}\n${TERRAIN_TILED_LOAD_SUM_WGSL}\n${TERRAIN_FINITE_WGSL}`
      )
    ]
  });
}

/**
 * The image of parent^(2^ceil(log2(n))) contains exactly roots and cycle vertices. Mark that image
 * and cut only its non-root vertices. Trees feeding a cycle keep their finite upstream sums;
 * marking the whole rootless component unresolved would incorrectly discard those trees.
 * @internal
 */
export function appendTerrainCycleCutSchedule<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  schedule: TerrainKernelSchedule<Parameters>,
  id: string,
  storage: TerrainForestStorage
): void {
  const count = storage.parents.length;
  const pointers = storage.successors.map(view =>
    graph.createDataView(view.buffer, {format: 'uint32', length: count})
  );
  const node = (
    suffix: string,
    bindings: GraphChunkGroupKernelProps<Parameters>['bindings'],
    body: string
  ) =>
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-${suffix}`,
      pipelineId: `terrain-tiled-cycle-${suffix}`,
      operation: 'GPUTerrainTiledFlow',
      invocationCount: count,
      bindings,
      body
    });
  schedule.push({
    repetitions: 1,
    nodes: [
      node(
        'initialize',
        [
          terrainBinding('parents', storage.parents),
          terrainBinding('pointers', pointers[0], 'read_write'),
          terrainBinding('flags', storage.firstChildren, 'read_write')
        ],
        'let parent = parents[parentsOffset + index]; pointers[pointersOffset + index] = select(parent, index, parent == 0xffffffffu); flags[flagsOffset + index] = 0u;'
      )
    ]
  });
  const passes = [0, 1].map(side =>
    node(
      'jump',
      [
        terrainBinding('previous', pointers[side]),
        terrainBinding('next', pointers[1 - side], 'read_write')
      ],
      'next[nextOffset + index] = previous[previousOffset + previous[previousOffset + index]];'
    )
  ) as [ReturnType<typeof node>, ReturnType<typeof node>];
  const rounds = Math.ceil(Math.log2(Math.max(count, 1)));
  appendTerrainAlternatingSchedule(schedule, passes, rounds);
  schedule.push({
    repetitions: 1,
    nodes: [
      node(
        'mark',
        [
          terrainBinding('parents', storage.parents),
          terrainBinding('pointers', pointers[rounds % 2]),
          {name: 'flags', view: storage.firstChildren, type: 'atomic<u32>', access: 'read_write'}
        ],
        'let image = pointers[pointersOffset + index]; if (parents[parentsOffset + image] != 0xffffffffu) { atomicOr(&flags[flagsOffset + image], 1u); }'
      ),
      node(
        'cut',
        [
          terrainBinding('flags', storage.firstChildren),
          terrainBinding('parents', storage.parents, 'read_write'),
          terrainBinding('states', storage.states, 'read_write')
        ],
        'if (flags[flagsOffset + index] != 0u) { parents[parentsOffset + index] = 0xffffffffu; if (states[statesOffset + index] != 0u) { states[statesOffset + index] = 2u; } }'
      )
    ]
  });
}
