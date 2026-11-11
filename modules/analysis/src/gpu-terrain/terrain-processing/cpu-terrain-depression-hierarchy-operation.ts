// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE
} from '../hydrology/cpu-terrain-drainage-utils';
import {
  getCPUTerrainGroundMeasures,
  TERRAIN_OPERATION_MAXIMUM_WORD_COUNT,
  validateCPUTerrainOperationGrid
} from './cpu-terrain-operation-grid';
import {
  GPUTerrainDepressionHierarchyOperation,
  type TerrainDepressionHierarchyOperationProps
} from './gpu-terrain-depression-hierarchy-operation';

export type CPUTerrainDepressionHierarchyInput = {
  elevation: Float32Array;
  outlets: Uint32Array;
  stages: Float32Array;
  validity?: Uint32Array;
};
export type CPUTerrainDepressionHierarchyOptions = Pick<
  TerrainDepressionHierarchyOperationProps,
  'grid' | 'cellSizeMode' | 'nodeCapacity'
>;
export type CPUTerrainDepressionHierarchyResult = {
  nodes: Uint32Array;
  levels: Float32Array;
  cellNodes: Uint32Array;
  curves: Float32Array;
  status: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
};

/**
 * Builds the stable sublevel-set merge tree (elevation then cell ID), retaining each cell's
 * activation node. New nodes represent isolated births or merges of distinct active components;
 * a single adjacent component absorbs the cell. Curves include a node's descendants at every
 * query stage, without clipping to spill levels. Outlets are explicit; invalid cells are barriers.
 *
 * What differs from the GPU contributor:
 * - Ground areas, depths, products and curve sums use float64, rounded once at output, instead
 *   of fixed-point sums of float32 terms. Compare nonzero finite curves with relative tolerance
 *   2e-5. Node IDs, stable ties, birth/spill levels, ownership and status words agree exactly.
 * - The CPU has no GPU watchdog or storage-binding scratch limit. Overflow still invalidates
 *   every published node/curve, while cellNodes retains the full internal ownership IDs.
 * - The pure function accepts an empty stages array. The operation form requires stageCount
 *   above zero because GPUContributorOperation requires nonempty output vectors.
 * - Subnormal elevations are retained; GPU arithmetic may flush them to zero.
 *
 * @throws If elevation, outlets or validity does not hold exactly one value per grid cell.
 */
export function computeTerrainDepressionHierarchyOnCPU(
  input: CPUTerrainDepressionHierarchyInput,
  options: CPUTerrainDepressionHierarchyOptions
): CPUTerrainDepressionHierarchyResult {
  const count = validateHierarchyOptions('computeTerrainDepressionHierarchyOnCPU', options);
  const {elevation, outlets, stages, validity} = input;
  if (elevation.length !== count) throw new Error('elevation must contain one value per grid cell');
  if (outlets.length !== count) throw new Error('outlets must contain one word per grid cell');
  if (validity && validity.length !== count)
    throw new Error('validity must contain one word per grid cell');
  validateMaximumCount(
    'computeTerrainDepressionHierarchyOnCPU',
    'nodeCapacity * stages.length * 2',
    options.nodeCapacity * stages.length * 2,
    TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
  );
  const order = Array.from({length: count}, (_, cell) => cell)
    .filter(cell => Number.isFinite(elevation[cell]) && (validity?.[cell] ?? 1) !== 0)
    .sort((first, second) => elevation[first] - elevation[second] || first - second);
  const labels = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const links: number[] = [];
  const tree: {parent: number; minimum: number; outlet: number; birth: number; spill: number}[] =
    [];
  const cellNodes = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const findRoot = (node: number): number => {
    let root = node;
    while (links[root] !== root) root = links[root];
    while (node !== root) {
      const next = links[node];
      links[node] = root;
      node = next;
    }
    return root;
  };
  const {grid, nodeCapacity} = options;
  for (const selected of order) {
    const neighbors = new Set<number>();
    for (let direction = 0; direction < 8; direction++) {
      const cell = getTerrainDrainageReceiver(selected, 1 << direction, grid.width, grid.height);
      if (cell === TERRAIN_DRAINAGE_NONE || labels[cell] === TERRAIN_DRAINAGE_NONE) continue;
      labels[cell] = findRoot(labels[cell]);
      neighbors.add(labels[cell]);
    }
    let destination: number;
    if (neighbors.size === 1) destination = neighbors.values().next().value!;
    else {
      destination = tree.length;
      links.push(destination);
      const record = {
        parent: TERRAIN_DRAINAGE_NONE,
        minimum: selected,
        outlet: TERRAIN_DRAINAGE_NONE,
        birth: elevation[selected],
        spill: Infinity
      };
      tree.push(record);
      for (const child of neighbors) {
        links[child] = destination;
        tree[child].parent = destination;
        tree[child].spill = elevation[selected];
        const childMinimum = tree[child].minimum;
        if (
          elevation[childMinimum] < elevation[record.minimum] ||
          (elevation[childMinimum] === elevation[record.minimum] && childMinimum < record.minimum)
        )
          record.minimum = childMinimum;
        record.outlet = Math.min(record.outlet, tree[child].outlet);
      }
    }
    labels[selected] = destination;
    cellNodes[selected] = destination;
    if (outlets[selected] !== 0)
      tree[destination].outlet = Math.min(tree[destination].outlet, selected);
  }
  const required = tree.length,
    stored = Math.min(required, nodeCapacity),
    overflowed = required > nodeCapacity;
  const nodes = new Uint32Array(nodeCapacity * 4).fill(TERRAIN_DRAINAGE_NONE);
  const levels = new Float32Array(nodeCapacity * 2).fill(NaN);
  const curves = new Float32Array(nodeCapacity * stages.length * 2).fill(NaN);
  for (let node = 0; node < nodeCapacity; node++) nodes[node * 4 + 3] = 0;
  if (!overflowed) {
    for (let node = 0; node < required; node++) {
      const record = tree[node];
      nodes.set([record.parent, record.minimum, record.outlet, 1], node * 4);
      levels.set([record.birth, record.spill], node * 2);
    }
    const measures = getCPUTerrainGroundMeasures(grid, options.cellSizeMode);
    for (let stageIndex = 0; stageIndex < stages.length; stageIndex++) {
      const stage = stages[stageIndex];
      if (!Number.isFinite(stage)) continue;
      const areas = new Float64Array(required),
        volumes = new Float64Array(required);
      for (const cell of order) {
        const depth = stage - elevation[cell];
        if (depth < 0) continue;
        const owner = cellNodes[cell],
          area = measures.cellArea(cell);
        areas[owner] += area;
        volumes[owner] += depth * area;
      }
      // Children precede parents, so each exclusive contribution travels once per tree edge.
      for (let node = 0; node < required; node++) {
        const parent = tree[node].parent;
        if (parent !== TERRAIN_DRAINAGE_NONE) {
          areas[parent] += areas[node];
          volumes[parent] += volumes[node];
        }
        curves.set([areas[node], volumes[node]], (node * stages.length + stageIndex) * 2);
      }
    }
  }
  return {
    nodes,
    levels,
    cellNodes,
    curves,
    status: Uint32Array.of(stored, required, Number(overflowed)),
    requiredCount: Uint32Array.of(required),
    overflow: Uint32Array.of(Number(overflowed))
  };
}

function validateHierarchyOptions(
  id: string,
  options: CPUTerrainDepressionHierarchyOptions
): number {
  const count = validateCPUTerrainOperationGrid(id, options.grid, options.cellSizeMode);
  validateCapacity(id, 'nodeCapacity', options.nodeCapacity);
  validateMaximumCount(
    id,
    'nodeCapacity * 4',
    options.nodeCapacity * 4,
    TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
  );
  return count;
}

/** CPU hierarchy twin with stable cell-ID ties and the same whole-hierarchy overflow rule. */
export class CPUTerrainDepressionHierarchyOperation extends GPUCPUOperation {
  constructor(props: TerrainDepressionHierarchyOperationProps) {
    const id = props.id ?? 'terrain-depression-hierarchy';
    validateHierarchyOptions(id, props);
    validateCapacity(id, 'stageCount', props.stageCount);
    validateMaximumCount(
      id,
      'nodeCapacity * stageCount * 2',
      props.nodeCapacity * props.stageCount * 2,
      TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
    );
    const operation = new GPUTerrainDepressionHierarchyOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const stages = getCPUOperationFloat32Rows(id, inputs, 'stages');
        if (stages.length !== props.stageCount)
          throw new Error(`${id} stages length must equal stageCount`);
        const result = computeTerrainDepressionHierarchyOnCPU(
          {
            elevation: getCPUOperationFloat32Rows(id, inputs, 'elevation'),
            outlets: getCPUOperationUint32Rows(id, inputs, 'outlets'),
            stages,
            validity:
              props.validity !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'validity')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTerrainDepressionHierarchyResult]
          ])
        );
      }
    });
  }
}
