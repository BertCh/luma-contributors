// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from './raster-grid/raster-construction-snapshot';
import {
  getGraphOutputViews,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUStatusPort} from '../analysis-core/contracts/status';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBufferBandNodes,
  getRasterBandViews,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph,
  snapshotRasterBand
} from './raster-band-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {getRasterWatershedForestNodes} from './raster-watershed-forest';
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionPorts,
  validateRasterExpansionBand,
  validateRasterExpansionGraph
} from './raster-expansion-utils';

/** FIFO queue cells: one thread runs a binary heap over every cell, so larger grids are rejected. */
const MAXIMUM_FIFO_CELLS = 16384;

/** Forest edge IDs use four canonical edges per cell below the reserved uint32 sentinel. */
const MAXIMUM_FOREST_CELLS = 0x3fffffff;

/** Forest contraction rounds; each round at least halves the unseeded components. */
const MAXIMUM_FOREST_ITERATIONS = 32;

export type GPURasterWatershedSegmentationProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Buffer or sampled-texture surface; raw nodata and validity are checked before calibration. */
  input: GPURasterBand;
  /** Basin IDs 1..0xfffffffd; zero unmarked, larger reserved words act as barriers. */
  markers: GraphDataView<'uint32'>;
  connectivity?: 4 | 8;
  /**
   * Default FIFO priority flood. The scalable minimum-spanning-forest method orders grid edges
   * by (maximum endpoint elevation, canonical edge ID), so plateau ownership can differ from FIFO.
   * Markers remain fixed roots; invalid pixels and reserved marker words remain barriers.
   */
  method?: 'fifo' | 'minimum-spanning-forest';
  /** FIFO-only maximum popped cells. */
  maximumSteps?: number;
  /** Forest-only Boruvka round budget, 1..32; default ceil(log2(pixel count)) + 1. */
  maximumIterations?: number;
  output: {
    labels: GraphDataView<'uint32'>;
    /**
     * `converged` is 1 when every reachable cell was labelled within the budget. The optional
     * `iterationLimitReached` is `1 - converged`: `maximumSteps` (FIFO) or `maximumIterations`
     * (forest) stopped the flood first.
     */
    status: GPURasterWatershedSegmentationStatus;
  };
};

/** Named status words of {@link GPURasterWatershedSegmentation} (`GPUIterationStatusPort` names). */
export type GPURasterWatershedSegmentationStatus = {
  converged: GraphDataView<'uint32'>;
  iterationLimitReached?: GraphDataView<'uint32'>;
};
/**
 * Marker-controlled segmentation with explicit FIFO or rooted minimum-spanning-forest semantics.
 * The default priority flood accepts complete domains of <=16384 cells.
 * Flood ordering is (maximum elevation along path, FIFO discovery age); row-major markers
 * initialize the queue. This is not the queue order of `skimage.segmentation.watershed`, which
 * orders by the cell's own value: the two agree where every basin is entered over its lowest
 * pass, and differ where fronts from higher markers meet on lower ground. For the row
 * `5, 2, 2, 2, 2, 2, 5` with markers on the ends this class labels `1 1 1 1 2 2 2` (the fronts
 * advance in turn) and the scikit-image order gives `1 1 1 1 1 1 2`. No parity with scikit-image
 * is claimed. No watershed-line pixels are removed. Invalid input pixels are barriers.
 * A binary heap preserves the deterministic ordering with O(cells * log(cells)) work
 * and O(cells) queue storage. This remains a bounded segmentation path, not large-image tiling.
 * Queue storage uses workgroup memory when it fits the device limit, otherwise graph scratch.
 * Exhaustion labels every non-marker valid cell unresolved.
 *
 * `minimum-spanning-forest` uses parallel rooted Boruvka contraction with linear scratch storage.
 * It supports larger domains subject to device storage limits and fewer than 2^30 pixels.
 * Canonical edge IDs are 4 * row-major source + direction (right, down-left, down, down-right);
 * absent directions are skipped. Equal elevations, including signed zero, use that static order.
 * This is a distinct deterministic watershed, not a claim of FIFO/skimage plateau parity.
 * Unseeded components retain label zero. An exhausted round budget publishes converged=0 and
 * unresolved non-marker valid cells, never a plausible partial partition.
 */
export class GPURasterWatershedSegmentation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterWatershedSegmentationProps;
  constructor(props: GPURasterWatershedSegmentationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'input',
      'markers',
      'connectivity',
      'method',
      'maximumSteps',
      'maximumIterations',
      'output'
    ]);
    const output = snapshotConstructionFields(props.output, ['labels', 'status']);
    props = Object.freeze({
      ...props,
      grid: snapshotRasterGrid(props.grid),
      input: Object.freeze(snapshotRasterBand(props.input)) as GPURasterBand,
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, ['converged', 'iterationLimitReached'])
      })
    });
    this.id = props.id ?? 'raster-watershed-segmentation';
    this.props = props;
    validateRasterTextureBand(this.id, props.input, props.grid);
    validateRasterExpansionBand(this.id, 'input', props.input);
    validateRasterExpansionPorts(
      this.id,
      props,
      {
        markers: ['uint32'],
        'output.labels': ['uint32'],
        'output.status.converged': ['uint32'],
        'output.status.iterationLimitReached': ['uint32']
      },
      ['output.status.iterationLimitReached']
    );
    validateRequirement(
      this.id,
      Boolean(props.output.status.converged),
      'output.status.converged is required; provide a uint32 convergence view'
    );
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [...getRasterBandViews(props.input), props.markers],
      getGraphOutputViews(props.output)
    );
    // Method-specific work budgets must not silently change their units.
    validateRequirement(
      this.id,
      props.method === undefined || ['fifo', 'minimum-spanning-forest'].includes(props.method),
      "method must be 'fifo' or 'minimum-spanning-forest'"
    );
    validateRequirement(
      this.id,
      props.connectivity === undefined || props.connectivity === 4 || props.connectivity === 8,
      'connectivity must be 4 or 8'
    );
    if (props.method === 'minimum-spanning-forest') {
      validateMaximumCount(this.id, 'grid cells', count, MAXIMUM_FOREST_CELLS);
      validateRequirement(
        this.id,
        props.maximumSteps === undefined,
        'maximumSteps applies to the fifo method only'
      );
      const iterations = props.maximumIterations ?? Math.ceil(Math.log2(count)) + 1;
      validateCapacity(this.id, 'maximumIterations', iterations, 1, MAXIMUM_FOREST_ITERATIONS);
    } else {
      validateMaximumCount(this.id, 'fifo grid cells', count, MAXIMUM_FIFO_CELLS);
      validateRequirement(
        this.id,
        props.maximumIterations === undefined,
        'maximumIterations applies to the minimum-spanning-forest method only'
      );
    }
    requireRasterLength(props.markers, count, this.id, 'markers');
    requireRasterLength(props.output.labels, count, this.id, 'output.labels');
    requireRasterLength(props.output.status.converged, 1, this.id, 'output.status.converged');
    requireRasterLength(
      props.output.status.iterationLimitReached,
      1,
      this.id,
      'output.status.iterationLimitReached'
    );
    validateGPUStatusPort(this.id, props.output.status);
    validateRequirement(
      this.id,
      Number.isInteger(props.maximumSteps ?? count) && (props.maximumSteps ?? count) > 0,
      'maximumSteps must be a positive integer'
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateRasterBandBelongsToGraph(this.id, graph, this.props.input, []);
    const storageProps = this.props;
    const storageCellCount = storageProps.grid.width * storageProps.grid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...getRasterBandViews(storageProps.input),
        storageProps.markers,
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'canonical band and forest planes': storageCellCount * 4,
        'forest gate': storageProps.method === 'minimum-spanning-forest' ? 12 : 0,
        'fifo queue':
          storageProps.method !== 'minimum-spanning-forest' &&
          Math.ceil(storageCellCount / 4) * 16 + Math.ceil((storageCellCount * 3) / 4) * 16 >
            graph.device.limits.maxComputeWorkgroupStorageSize
            ? storageCellCount * 12
            : 0
      }
    );

    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      props.markers,
      ...getGraphOutputViews(output)
    ]);
    const band = getRasterBufferBandNodes(graph, id, props.input, grid, true);
    const statusNodes = this.getIterationLimitNodes(graph);
    if (props.method === 'minimum-spanning-forest') {
      return [
        ...band.nodes,
        ...getRasterWatershedForestNodes(graph, id, props, band.band),
        ...statusNodes
      ];
    }
    // Workgroup allocations are individually rounded to 16 bytes. Only invocation zero uses
    // the queue, so local storage preserves the serial flood order without synchronization.
    const workgroupBytes = Math.ceil(count / 4) * 16 + Math.ceil((count * 3) / 4) * 16;
    const useWorkgroupQueue = workgroupBytes <= graph.device.limits.maxComputeWorkgroupStorageSize;
    const queueBindings: WGSLKernelBinding[] = useWorkgroupQueue
      ? []
      : [
          {
            name: 'costs',
            view: createTransientView(graph, `${id}-costs`, 'float32', count),
            type: 'f32',
            access: 'read_write'
          },
          // Visited flags, discovery ages and heap cell IDs share one binding.
          {
            name: 'queueState',
            view: createTransientView(graph, `${id}-queue`, 'uint32', count * 3),
            type: 'u32',
            access: 'read_write'
          }
        ];
    return [
      ...band.nodes,
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPURasterWatershedSegmentation',
        invocationCount: 1,
        bindings: [
          {name: 'elevation', view: band.band.storage.values, type: 'f32', access: 'read'},
          {name: 'validity', view: band.band.validity!, type: 'u32', access: 'read'},
          {name: 'markers', view: props.markers, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          ...queueBindings,
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        declarations: `
${
  useWorkgroupQueue
    ? `var<workgroup> costs: array<f32, ${count}>;
var<workgroup> queueState: array<u32, ${count * 3}>;
const costsOffset: u32 = 0u;
const queueStateOffset: u32 = 0u;`
    : ''
}
const AGE_START: u32 = ${count}u;
const HEAP_START: u32 = ${count * 2}u;
var<private> pendingCount: u32;
fn heapCell(slot: u32) -> u32 { return queueState[queueStateOffset + HEAP_START + slot]; }
fn setHeapCell(slot: u32, cell: u32) {
  queueState[queueStateOffset + HEAP_START + slot] = cell;
}
fn precedes(first: u32, second: u32) -> bool {
  let firstCost = costs[costsOffset + first];
  let secondCost = costs[costsOffset + second];
  return firstCost < secondCost || (firstCost == secondCost && queueState[queueStateOffset + AGE_START + first] < queueState[queueStateOffset + AGE_START + second]);
}
fn insertCell(cell: u32) {
  // Extracted costs never decrease. The first discovery has cost max(parent cost, elevation),
  // so no later parent can strictly improve it. Each cell enters the queue only once.
  var slot = pendingCount;
  pendingCount++;
  loop {
    if (slot == 0u) { break; }
    let parent = (slot - 1u) / 2u;
    let parentCell = heapCell(parent);
    if (!precedes(cell, parentCell)) { break; }
    setHeapCell(slot, parentCell);
    slot = parent;
  }
  setHeapCell(slot, cell);
}
fn removeMinimum() -> u32 {
  let result = heapCell(0u);
  pendingCount--;
  if (pendingCount > 0u) {
    let replacement = heapCell(pendingCount);
    var slot = 0u;
    loop {
      var child = slot * 2u + 1u;
      if (child >= pendingCount) { break; }
      if (child + 1u < pendingCount && precedes(heapCell(child + 1u), heapCell(child))) { child++; }
      let childCell = heapCell(child);
      if (!precedes(childCell, replacement)) { break; }
      setHeapCell(slot, childCell);
      slot = child;
    }
    setHeapCell(slot, replacement);
  }
  return result;
}`,
        body: `
pendingCount = 0u;
var age = 0u;
for (var cell = 0u; cell < ${count}u; cell++) {
  let marker = markers[markersOffset + cell];
  let valid = validity[validityOffset + cell] != 0u;
  let seeded = valid && marker > 0u && marker < 0xfffffffeu;
  labels[labelsOffset + cell] = select(0u, marker, seeded);
  costs[costsOffset + cell] = select(bitcast<f32>(0x7f800000u | (index & 0u)), elevation[elevationOffset + cell], seeded);
  queueState[queueStateOffset + cell] = u32(!valid);
  queueState[queueStateOffset + AGE_START + cell] = select(0xffffffffu, age, seeded);
  if (seeded) { age++; insertCell(cell); }
}
var complete = false;
for (var step = 0u; step <= ${Math.min(props.maximumSteps ?? count, count)}u; step++) {
  if (pendingCount == 0u) { complete = true; break; }
  if (step == ${Math.min(props.maximumSteps ?? count, count)}u) { break; }
  let best = removeMinimum();
  let bestCost = costs[costsOffset + best];
  queueState[queueStateOffset + best] = 1u;
  for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
    for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
      if ((columnOffset == 0 && rowOffset == 0)${props.connectivity === 8 ? '' : ' || abs(columnOffset) + abs(rowOffset) > 1'}) { continue; }
      let column = i32(best % ${grid.width}u) + columnOffset;
      let row = i32(best / ${grid.width}u) + rowOffset;
      if (column < 0 || row < 0 || column >= ${grid.width} || row >= ${grid.height}) { continue; }
      let neighbor = u32(row) * ${grid.width}u + u32(column);
      if (queueState[queueStateOffset + neighbor] != 0u || markers[markersOffset + neighbor] != 0u) { continue; }
      let candidate = max(bestCost, elevation[elevationOffset + neighbor]);
      if (candidate < costs[costsOffset + neighbor]) {
        costs[costsOffset + neighbor] = candidate;
        labels[labelsOffset + neighbor] = labels[labelsOffset + best];
        queueState[queueStateOffset + AGE_START + neighbor] = age;
        age++;
        insertCell(neighbor);
      }
    }
  }
}
converged[convergedOffset] = u32(complete);
if (!complete) {
  for (var cell = 0u; cell < ${count}u; cell++) {
    if (validity[validityOffset + cell] != 0u && markers[markersOffset + cell] == 0u) { labels[labelsOffset + cell] = 0xfffffffeu; }
  }
}`
      }),
      ...statusNodes
    ];
  }

  /** One-thread node that publishes the optional `iterationLimitReached` as `1 - converged`. */
  private getIterationLimitNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {status} = this.props.output;
    if (!status.iterationLimitReached) {
      return [];
    }
    return [
      createWGSLKernelNode(graph, {
        id: `${this.id}-iteration-status`,
        operation: 'GPURasterWatershedSegmentation',
        variant: 'iteration-status',
        invocationCount: 1,
        bindings: [
          {name: 'converged', view: status.converged, type: 'u32', access: 'read'},
          {
            name: 'iterationLimitReached',
            view: status.iterationLimitReached,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'iterationLimitReached[iterationLimitReachedOffset] = 1u - min(converged[convergedOffset], 1u);'
      })
    ];
  }
}
