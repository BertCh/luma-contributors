// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {validateChunkViews} from '../gpu-core/gpu-chunk-utils';
import {createPointCloudGridIndex} from './pointcloud-grid-index';
import {
  flattenPointCloudAttribute,
  readPointCloudAttribute,
  writePointCloudAttribute
} from './pointcloud-attribute';
import {getSpatialOperationNodes} from '../utils/gpu-spatial-operation';

export type GPUPointCloudVoxelDownsampleProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  origin: readonly [number, number, number];
  voxelSize: number;
  gridSize: readonly [number, number, number];
  /** One source row ID per voxel, or 0xffffffff for an empty voxel. */
  representatives: GraphDataView<'uint32'>;
  /** One population count per voxel. */
  counts: GraphDataView<'uint32'>;
  /** Representative positions in voxel order. Empty voxels contain zero; use counts as validity. */
  output: GraphDataView<'float32x3'>;
  /** Optional arithmetic mean, separate from sampled output. Empty voxels contain zero. */
  centroids?: GraphDataView<'float32x3'>;
  /** Optional scalar or vector averaging alongside centroids; nonfinite values are excluded and counted separately. */
  attribute?: {
    values: GraphDataView<'float32' | 'float32x2' | 'float32x3' | 'float32x4'>;
    means: GraphDataView<'float32' | 'float32x2' | 'float32x3' | 'float32x4'>;
    counts: GraphDataView<'uint32'>;
  };
};

/** Deterministic minimum-source-ID voxel sampling, O(points + voxels), with explicit bounded grid. */
export class GPUPointCloudVoxelDownsample {
  readonly id: string;
  constructor(readonly props: GPUPointCloudVoxelDownsampleProps) {
    this.id = props.id ?? 'gpu-pointcloud-voxel-downsample';
    const cells = props.gridSize.reduce((product, size) => product * size, 1);
    // Fixed grid capacities must fit uint32 and match all output views.
    assert(props.gridSize.every(size => Number.isSafeInteger(size) && size > 0));
    assert(Number.isSafeInteger(cells) && cells < 0xffffffff);
    assert(
      props.counts.length === cells &&
        props.representatives.length === cells &&
        props.output.length === cells
    );
    assert(!props.centroids || props.centroids.length === cells);
    // Attribute aggregation accompanies centroid averaging and keeps its own finite-value support.
    assert(
      !props.attribute ||
        (props.centroids &&
          props.attribute.values.length === props.positions.length &&
          props.attribute.means.length === cells &&
          props.attribute.means.format === props.attribute.values.format &&
          props.attribute.counts.length === cells)
    );
    // Grid arithmetic uses finite float32 coordinates and a positive representable cell size.
    assert(Number.isFinite(Math.fround(props.voxelSize)) && Math.fround(props.voxelSize) > 0);
    assert(props.origin.every(value => Number.isFinite(Math.fround(value))));
    assert(props.positions.length < 0xffffffff);
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {
      positions,
      representatives,
      counts,
      output,
      centroids,
      attribute,
      gridSize,
      origin,
      voxelSize
    } = this.props;
    validateChunkViews(graph, [positions], [representatives, counts, output]);
    const dimensions = gridSize.map(size => `${size}u`).join(', ');
    const nodes = getSpatialOperationNodes(graph, {
      id: `${this.id}-clear`,
      length: counts.length,
      inputs: {},
      outputs: {representatives, counts},
      atomicOutputs: true,
      body: 'writeRepresentatives(index, 0xffffffffu); writeCounts(index, 0u);'
    });
    nodes.push(
      ...getSpatialOperationNodes(graph, {
        id: `${this.id}-select`,
        length: positions.length,
        inputs: {positions},
        outputs: {representatives, counts},
        atomicOutputs: true,
        body: `let coordinate = (readPositions(index) - vec3f(${origin.map(value => Math.fround(value)).join(', ')})) / ${Math.fround(voxelSize)};
if (!all(coordinate >= vec3f(0.0)) || !all(coordinate < vec3f(vec3u(${dimensions})))) { return; }
let cell = vec3u(floor(coordinate));
let voxel = (cell.z * ${gridSize[1]}u + cell.y) * ${gridSize[0]}u + cell.x;
minRepresentatives(voxel, index); addCounts(voxel, 1u);`
      })
    );
    nodes.push(
      ...getSpatialOperationNodes(graph, {
        id: `${this.id}-gather`,
        length: output.length,
        inputs: {positions, representatives},
        outputs: {output},
        body: `let representative = readRepresentatives(index);
var position = vec3f(0.0);
if (representative < ${positions.length}u) { position = readPositions(representative); }
writeOutput(index, position);`
      })
    );
    if (centroids) {
      const indexed = createPointCloudGridIndex(graph, `${this.id}-centroid-index`, positions, {
        origin,
        cellSize: voxelSize,
        dimensions: gridSize
      });
      nodes.push(...indexed.nodes);
      const components = Number(attribute?.values.format.match(/x([234])$/)?.[1] ?? 1);
      const attributeZero = components === 1 ? '0.0' : `vec${components}f(0.0)`;
      const attributeFinite =
        components === 1
          ? 'abs(value) <= 3.402823e38'
          : `all(abs(value) <= vec${components}f(3.402823e38))`;
      nodes.push(
        ...getSpatialOperationNodes(graph, {
          id: `${this.id}-centroids`,
          length: centroids.length,
          inputs: {
            positions,
            sortedIndices: indexed.sortedIndices,
            offsets: indexed.offsets,
            ...(attribute ? {values: flattenPointCloudAttribute(graph, attribute.values)} : {})
          },
          outputs: {
            centroids,
            ...(attribute
              ? {
                  attributeMeans: flattenPointCloudAttribute(graph, attribute.means),
                  attributeCounts: attribute.counts
                }
              : {})
          },
          body: `
let start = readOffsets(index); let end = readOffsets(index+1u);
var result = vec3f(0.0);
var attributeMean = ${attributeZero}; var attributeCount = 0u;
if (start < end) {
  let anchor = readPositions(readSortedIndices(start));
  var differenceSum = vec3f(0.0);
  for (var slot = start; slot < end; slot++) {
    let point = readSortedIndices(slot);
    differenceSum += readPositions(point) - anchor;
    ${
      attribute
        ? `let value = ${readPointCloudAttribute('Values', 'point', components)};
      if (${attributeFinite}) { attributeCount++; attributeMean += (value - attributeMean) / f32(attributeCount); }`
        : ''
    }
  }
  result = anchor + differenceSum / f32(end-start);
}
writeCentroids(index, result);
${attribute ? `${writePointCloudAttribute('AttributeMeans', 'index', 'attributeMean', components)} writeAttributeCounts(index, attributeCount);` : ''}`
        })
      );
    }
    return nodes;
  }
}
