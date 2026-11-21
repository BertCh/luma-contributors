// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateGraphPorts,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPointCloudKernelNodes, validatePointCloudPackedViews} from './pointcloud-kernel-nodes';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

/**
 * Convert ground-coverage bit 2 to uint32 seed flags (0 or 1), in the coverage cell-row domain.
 * Writes every row, including zero-elevation ground. Only the caller's single output is needed;
 * use GPUPointCloudCoverageValidity when all three raster-validity outputs are wanted.
 */
export function createGroundSeedCoverageNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  coverage: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  const id = 'pointcloud-ground-seed-coverage';
  validateGraphPorts(
    id,
    {coverage, output},
    {
      coverage: {required: true, viewType: 'data', formats: ['uint32']},
      output: {required: true, viewType: 'data', formats: ['uint32']}
    }
  );
  validateViewLength(id, 'output', output, coverage.length);
  validatePointCloudPackedViews(id, {coverage}, {output});
  validateGraphChunkBindingRanges(graph, id, {coverage, output});
  return createWGSLKernelNode(graph, {
    id: `${id}-${output.buffer.id}-${output.byteOffset}`,
    operation: 'GPUPointCloudCoverageValidity',
    invocationCount: coverage.length,
    bindings: [
      {name: 'coverage', view: coverage, type: 'u32', access: 'read'},
      {name: 'output', view: output, type: 'u32', access: 'read_write'}
    ],
    body: 'output[outputOffset + index] = select(0u, 1u, (coverage[coverageOffset + index] & 2u) != 0u);'
  });
}

export type GPUPointCloudCoverageValidityOutput = {
  surfaceValidity: GraphDataView<'uint32'>;
  terrainValidity: GraphDataView<'uint32'>;
  canopyValidity: GraphDataView<'uint32'>;
};
export type GPUPointCloudCoverageValidityProps = {
  id?: string;
  coverage: GraphDataView<'uint32'>;
  output: GPUPointCloudCoverageValidityOutput;
};

/** Convert coverage bits to per-cell raster validity: surface uses bit 1; terrain/canopy use bit 2.
 * Every output has the coverage row domain. Zero ground elevation remains valid. All rows are written.
 */
export class GPUPointCloudCoverageValidity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPointCloudCoverageValidityProps;
  constructor(props: GPUPointCloudCoverageValidityProps) {
    props = snapshotConstructionFields(props, ['id', 'coverage', 'output']);
    this.id = props.id ?? 'gpu-pointcloud-coverage-validity';
    validateGraphPorts(this.id, props, {
      coverage: {required: true, viewType: 'data', formats: ['uint32']},
      'output.surfaceValidity': {required: true, viewType: 'data', formats: ['uint32']},
      'output.terrainValidity': {required: true, viewType: 'data', formats: ['uint32']},
      'output.canopyValidity': {required: true, viewType: 'data', formats: ['uint32']}
    });
    this.props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output, [
        'surfaceValidity',
        'terrainValidity',
        'canopyValidity'
      ])
    });
    for (const [name, view] of Object.entries(this.props.output))
      validateViewLength(this.id, name, view, props.coverage.length);
    validatePointCloudPackedViews(this.id, {coverage: props.coverage}, this.props.output);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getPointCloudKernelNodes(graph, {
      id: this.id,
      length: this.props.coverage.length,
      inputs: {coverage: this.props.coverage},
      outputs: this.props.output,
      body: 'let coverage = readCoverage(index); writeSurfaceValidity(index, select(0u, 1u, (coverage & 1u) != 0u)); let ground = select(0u, 1u, (coverage & 2u) != 0u); writeTerrainValidity(index, ground); writeCanopyValidity(index, ground);'
    });
  }
}
