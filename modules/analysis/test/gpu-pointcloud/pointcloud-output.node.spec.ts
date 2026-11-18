// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudChange,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudHeight,
  GPUPointCloudICP,
  GPUPointCloudNeighborhoods,
  GPUPointCloudNormalScale,
  GPUPointCloudSMRF,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces,
  GPUPointCloudVoxelMerge
} from '@luma-contributors/analysis/pointcloud-analysis';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

// Wrong implementation: read a former top-level output instead of the caller's output port.
// Every case supplies only the new port, so even one unmigrated validation/binding fails.
const contributors = {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces,
  GPUPointCloudHeight,
  GPUPointCloudGround,
  GPUPointCloudSMRF,
  GPUPointCloudNormalScale,
  GPUPointCloudICP,
  GPUPointCloudChange,
  GPUPointCloudVoxelMerge
} as const;

function createDeclarations(graph: GPUCommandGraph) {
  let sequence = 0;
  const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(format: Format, length = 4) =>
    createTransientView(graph, `output-contract-${sequence++}`, format, length);
  const positions = view('float32x3');
  const classifications = view('uint32');
  const grid = {origin: [0, 0] as const, width: 3, height: 3, cellSize: 1};
  const declarations = {
    GPUPointCloudNeighborhoods: {
      positions,
      radius: 1,
      capacity: 3,
      output: {
        indices: view('uint32', 12),
        squaredDistances: view('float32', 12),
        counts: view('uint32'),
        status: view('uint32')
      }
    },
    GPUPointCloudFeatures: {
      positions,
      indices: view('uint32', 12),
      counts: view('uint32'),
      capacity: 3,
      output: {
        normals: view('float32x3'),
        eigenvalues: view('float32x3'),
        covariance: view('float32', 24),
        shapes: view('float32x3'),
        status: view('uint32')
      }
    },
    GPUPointCloudStatisticalFilter: {
      counts: view('uint32'),
      squaredDistances: view('float32', 12),
      capacity: 3,
      output: {meanDistances: view('float32'), validity: view('uint32')}
    },
    GPUPointCloudSurfaces: {
      positions,
      classifications,
      grid,
      output: {
        terrain: view('float32', 9),
        surface: view('float32', 9),
        canopy: view('float32', 9),
        coverage: view('uint32', 9)
      }
    },
    GPUPointCloudHeight: {
      positions,
      grid,
      terrain: view('float32', 9),
      coverage: view('uint32', 9),
      output: {heights: view('float32'), status: view('uint32')}
    },
    GPUPointCloudGround: {
      positions,
      classifications,
      grid,
      radii: [1],
      initialThreshold: 0.1,
      slope: 0.1,
      maximumThreshold: 1,
      minimumSurface: view('float32', 9),
      coverage: view('uint32', 9),
      output: {
        groundMask: view('uint32'),
        outputClassifications: view('uint32'),
        status: view('uint32')
      }
    },
    GPUPointCloudSMRF: {
      positions,
      classifications,
      grid,
      window: 1,
      output: {
        groundSurface: view('float32', 9),
        groundMask: view('uint32'),
        outputClassifications: view('uint32'),
        status: view('uint32')
      }
    },
    GPUPointCloudNormalScale: {
      positions,
      radii: [1],
      capacity: 3,
      parameters: view('float32', 5),
      output: {
        normals: view('float32x3'),
        normalScales: view('float32'),
        normalVariances: view('float32'),
        status: view('uint32')
      }
    },
    GPUPointCloudICP: {
      source: positions,
      target: positions,
      iterations: 1,
      maximumDistance: 1,
      output: {
        transform: view('float32', 16),
        history: view('float32x3', 1),
        inlierMask: view('uint32'),
        status: view('uint32', 1)
      }
    },
    GPUPointCloudChange: {
      reference: positions,
      comparison: positions,
      cores: positions,
      normals: positions,
      alignment: 'already-aligned',
      radius: 1,
      halfDepth: 1,
      output: {
        changes: view('float32x3'),
        roughness: view('float32x3'),
        counts: view('uint32', 8),
        status: view('uint32')
      }
    },
    GPUPointCloudVoxelMerge: {
      partitionPolicy: 'disjoint-owned-rows',
      parts: [
        {
          counts: view('uint32'),
          centroids: view('float32x3'),
          representatives: view('uint32'),
          sourceRowOffset: 0,
          attribute: {means: view('float32'), counts: view('uint32')}
        }
      ],
      output: {
        counts: view('uint32'),
        centroids: view('float32x3'),
        representatives: view('uint32'),
        attribute: {means: view('float32'), counts: view('uint32')},
        status: view('uint32')
      }
    }
  } satisfies {
    [Name in keyof typeof contributors]: ConstructorParameters<(typeof contributors)[Name]>[0];
  };
  return {declarations, view};
}

function getOutputViews(output: object): import('@luma.gl/graph/gpu-core').GraphDataView[] {
  return Object.values(output).flatMap(value =>
    value && typeof value === 'object' && 'buffer' in value
      ? [value]
      : value && typeof value === 'object'
        ? getOutputViews(value)
        : []
  );
}

function expectWriters(
  nodes: readonly import('@luma.gl/graph/gpu-core').GPUCommandNode[],
  views: ReturnType<typeof getOutputViews>
) {
  for (const view of views) {
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            'buffer' in resource.buffer &&
            resource.usage === 'storage-read-write' &&
            resource.buffer.buffer === view.buffer &&
            resource.buffer.byteOffset === view.byteOffset &&
            resource.buffer.length * resource.buffer.rowByteLength ===
              view.length * view.rowByteLength
        )
      ),
      `writer for ${view.buffer.id}`
    ).toBe(true);
  }
}

for (const name of Object.keys(contributors) as (keyof typeof contributors)[]) {
  const Contributor = contributors[name];
  for (const inherited of [false, true]) {
    it(`${name} retains getter props and output bindings, inherited=${inherited}`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const {declarations} = createDeclarations(graph);
      const props = declarations[name];
      const getterFields = (fields: object): object => {
        const descriptors = Object.fromEntries(
          Object.entries(fields).map(([key, value]) => [key, {get: () => value, enumerable: false}])
        );
        return inherited
          ? Object.create(Object.defineProperties({}, descriptors))
          : Object.defineProperties({}, descriptors);
      };
      const output = getterFields(
        Object.fromEntries(
          Object.entries(props.output).map(([key, value]) => [
            key,
            key === 'attribute' ? getterFields(value) : value
          ])
        )
      );
      const contributor = new Contributor(getterFields({...props, output}) as never);
      expectWriters(contributor.getCommandNodes(graph), getOutputViews(props.output));
    });
  }
  it(`${name} writes every caller-owned output view`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {declarations} = createDeclarations(graph);
    const props = declarations[name];
    const contributor = new Contributor(props as never);
    expectWriters(contributor.getCommandNodes(graph), getOutputViews(props.output));
  });

  it(`${name} snapshots output maps before caller mutation`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {declarations, view} = createDeclarations(graph);
    const props = declarations[name];
    const expected = getOutputViews(props.output);
    const contributor = new Contributor(props as never);
    const mutate = (output: object) => {
      for (const [key, value] of Object.entries(output)) {
        if (value && 'buffer' in value) {
          (output as Record<string, unknown>)[key] = view(value.format, value.length);
        } else if (value && typeof value === 'object') mutate(value);
      }
    };
    mutate(props.output);
    expectWriters(contributor.getCommandNodes(graph), expected);
  });

  it(`${name} names the required output port in migration errors`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {declarations} = createDeclarations(graph);
    const {output, ...inputs} = declarations[name];
    expect(() => new Contributor({...inputs, ...output, id: 'my-cloud'} as never)).toThrow(
      /^my-cloud.*output/
    );
  });
}

it('statistical filter cannot be redirected into counts after validation', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {declarations} = createDeclarations(graph);
  const props = declarations.GPUPointCloudStatisticalFilter;
  const original = props.output.validity;
  const contributor = new GPUPointCloudStatisticalFilter(props);
  props.output.validity = props.counts;
  expectWriters(contributor.getCommandNodes(graph), [original]);
});

it('statistical filter retains an inherited zero standard-deviation multiplier', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {declarations} = createDeclarations(graph);
  const props = Object.assign(
    Object.create({
      get standardDeviations() {
        return 0;
      }
    }),
    declarations.GPUPointCloudStatisticalFilter
  );
  const contributor = new GPUPointCloudStatisticalFilter(props);
  expect(contributor.props.standardDeviations).toBe(0);
});

for (const name of [
  'GPUPointCloudGround',
  'GPUPointCloudHeight',
  'GPUPointCloudSurfaces'
] as const) {
  it(`${name} snapshots nested grid settings and radii before caller mutation`, () => {
    const lower = (mutate: boolean) => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const {declarations} = createDeclarations(graph);
      const props = declarations[name];
      const origin: [number, number] = [0, 0];
      let originReads = 0;
      const grid = Object.create({
        get origin() {
          originReads++;
          return origin;
        }
      });
      Object.assign(grid, {width: 3, height: 3, cellSize: 1});
      props.grid = grid;
      const contributor =
        name === 'GPUPointCloudGround'
          ? new GPUPointCloudGround(declarations.GPUPointCloudGround)
          : name === 'GPUPointCloudHeight'
            ? new GPUPointCloudHeight(declarations.GPUPointCloudHeight)
            : new GPUPointCloudSurfaces(declarations.GPUPointCloudSurfaces);
      if (mutate) {
        origin[0] = 100;
        grid.width = 100;
        grid.cellSize = 100;
        if (name === 'GPUPointCloudGround') declarations.GPUPointCloudGround.radii.length = 0;
      }
      const nodes = contributor.getCommandNodes(graph);
      expectWriters(nodes, getOutputViews(props.output));
      expect(originReads).toBe(1);
      if (name === 'GPUPointCloudGround')
        expect(nodes.some(node => node.id.includes('-classify-0'))).toBe(true);
      return getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source);
    };
    expect(lower(true)).toEqual(lower(false));
  });
}

it('Height snapshots getter-backed interpolation settings once before mutation', () => {
  const lower = (mutate: boolean) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {declarations} = createDeclarations(graph);
    let radius = 1;
    let reads = 0;
    const interpolation = Object.create({
      method: 'idw',
      get radius() {
        reads++;
        return radius;
      },
      power: 2,
      minimumSupport: 1
    });
    const contributor = new GPUPointCloudHeight({
      ...declarations.GPUPointCloudHeight,
      interpolation
    });
    if (mutate) radius = Infinity;
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).map(
      kernel => kernel.source
    );
    expect(reads).toBe(1);
    return sources;
  };
  expect(lower(true)).toEqual(lower(false));
});
