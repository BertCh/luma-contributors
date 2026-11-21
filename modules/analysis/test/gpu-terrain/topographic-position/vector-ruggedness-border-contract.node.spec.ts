// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessProps
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('DEF-16 exposes only clamp and nodata while retaining the default and JavaScript guard', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const props: GPUTerrainVectorRuggednessProps = {
    grid,
    elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', 9), grid),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {vectorRuggedness: createTransientView(graph, 'ruggedness', 'float32', 9)}
  };
  // @ts-expect-error reflect is unsupported by the vector-ruggedness route.
  const reflect: GPUTerrainVectorRuggednessProps['borderMode'] = 'reflect';
  // @ts-expect-error constant is unsupported by the vector-ruggedness route.
  const constant: GPUTerrainVectorRuggednessProps['borderMode'] = 'constant';
  try {
    const defaultContributor = new GPUTerrainVectorRuggedness(props);
    const assertDefaultClamp = (nodes: readonly GPUCommandNode[]) => {
      const normals = nodes.find(node => node.id.endsWith('-normals'));
      expect(normals?.workload?.variant).toBe('normals-uniform-clamp');
    };
    assertDefaultClamp(defaultContributor.getCommandNodes(graph));
    // The control must fail the omitted-option assertion if the default ever becomes nodata.
    expect(() =>
      assertDefaultClamp(
        new GPUTerrainVectorRuggedness({...props, borderMode: 'nodata'}).getCommandNodes(graph)
      )
    ).toThrow();
    for (const borderMode of ['clamp', 'nodata'] as const)
      expect(() => new GPUTerrainVectorRuggedness({...props, borderMode})).not.toThrow();
    for (const borderMode of [reflect, constant])
      expect(() => new GPUTerrainVectorRuggedness({...props, borderMode})).toThrow(
        /borderMode must be clamp or nodata/
      );
  } finally {
    device.destroy();
  }
});
