// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUPointCloudCoverageValidity,
  type GPUPointCloudCoverageValidityOutput,
  type GPUPointCloudCoverageValidityProps,
  getRasterMetadataFromPointCloudGrid
} from '../../src/pointcloud-analysis/index';
import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator
} from '../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('CCK-01: public coverage values, types and qualified catalog registration agree', () => {
  const identity = {
    package: '@luma-contributors/analysis/pointcloud-analysis',
    exportName: 'GPUPointCloudCoverageValidity'
  } as const;
  const capability = getGPUAnalysisCapabilityCatalog().find(
    entry => entry.id === 'pointcloud-ground'
  );
  expect(capability?.operators).toContain(identity.exportName);
  expect(capability?.operatorIdentities).toContainEqual(identity);
  const operator = getGPUAnalysisOperator(identity);
  expect(operator).toMatchObject({
    ...identity,
    inputs: ['surface'],
    outputs: ['attributes'],
    dimensionality: '2d',
    evidence: {
      tests: 'modules/analysis/test/gpu-pointcloud/pointcloud-coverage-validity-w3.spec.ts',
      documentation: 'docs/api/pointcloud-analysis.md'
    }
  });
  expect(operator?.summary).toMatch(/coverage.*validity/);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output: GPUPointCloudCoverageValidityOutput = {
      surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', 4),
      terrainValidity: createTransientView(graph, 'terrain-validity', 'uint32', 4),
      canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', 4)
    };
    const props: GPUPointCloudCoverageValidityProps = {
      coverage: createTransientView(graph, 'coverage', 'uint32', 4),
      output
    };
    const contributor = new GPUPointCloudCoverageValidity(props);
    expect(contributor.props.output).toEqual(output);
    expect(contributor.getCommandNodes(graph)).not.toHaveLength(0);
    // Two 10-unit cells: southward rows start at the top y=200+2*10=220.
    expect(
      getRasterMetadataFromPointCloudGrid({
        origin: [100, 200],
        cellSize: 10,
        width: 2,
        height: 2
      }).affine
    ).toEqual([10, 0, 100, 0, -10, 220]);
  } finally {
    device.destroy();
  }
});
