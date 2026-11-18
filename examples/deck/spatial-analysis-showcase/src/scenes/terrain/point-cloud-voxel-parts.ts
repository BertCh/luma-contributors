// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {computePointCloudVoxelMeans} from '@luma-contributors/analysis/pointcloud-analysis';

/** The showcase's actual partition builder: means are uploaded separately from selected points. */
export function addPointCloudSupportVoxelPart(
  graph: GPUCommandGraph,
  props: {
    id: string;
    positions: GraphDataView<'float32x3'>;
    sourcePositions: Float32Array;
    origin: readonly [number, number, number];
    voxelSize: number;
    gridSize: readonly [number, number, number];
    counts: GraphDataView<'uint32'>;
    representatives: GraphDataView<'uint32'>;
    selectedPositions: GraphDataView<'float32x3'>;
    uploadMeans: (means: Float32Array) => GraphDataView<'float32x3'>;
    sourceRowOffset: number;
  }
) {
  graph.add(
    new GPUPointCloudVoxelDownsample({
      id: props.id,
      positions: props.positions,
      origin: props.origin,
      voxelSize: props.voxelSize,
      gridSize: props.gridSize,
      counts: props.counts,
      representatives: props.representatives,
      output: props.selectedPositions
    })
  );
  return {
    counts: props.counts,
    representatives: props.representatives,
    sourceRowOffset: props.sourceRowOffset,
    centroids: props.uploadMeans(
      computePointCloudVoxelMeans(
        props.sourcePositions,
        props.voxelSize,
        props.gridSize,
        props.origin
      )
    )
  };
}
