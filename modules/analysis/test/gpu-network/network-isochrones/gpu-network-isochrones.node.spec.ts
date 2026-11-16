// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  type GPUNetworkIsochronesProps
} from '../../../src/gpu-network/network-isochrones/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkIsochronesProps> = {}
): GPUNetworkIsochronesProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    topology: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 6),
      weights: view('float32', 6),
      nodePositions: view('float32x2', 4)
    },
    costs: view('float32', 4),
    costBreaks: view('float32', 3),
    parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
    raster: {
      width: 8,
      height: 8,
      output: {
        triangles: view('float32x2', 3 * 64),
        triangleBands: view('uint32', 64),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    },
    ...overrides
  };
}

it('GPUNetworkIsochrones declares search, splat and isobands nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-node'});
  const plain = new GPUNetworkIsochrones(createProps(graph, {id: 'plain'})).getCommandNodes(graph);
  const withSearch = new GPUNetworkIsochrones(
    createProps(graph, {
      id: 'searching',
      startNodes: createTransientView(graph, 'search-sources', 'uint32', 2)
    })
  ).getCommandNodes(graph);
  expect(plain.length).toBeGreaterThanOrEqual(6);
  expect(withSearch.length).toBeGreaterThan(plain.length);
});

it('GPUNetworkIsochrones validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-validate'});
  const props = createProps(graph);
  expect(() => new GPUNetworkIsochrones({...props, raster: undefined})).toThrow(
    /raster or cellOutline/
  );
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {...props.topology, offsets: props.topology.weights as never}
      })
  ).toThrow(/topology.offsets/);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {...props.topology, nodePositions: undefined as never}
      })
  ).toThrow(/topology.nodePositions is required/);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {
          ...props.topology,
          weights: createTransientView(
            new GPUCommandGraph(createNullWebGPUDevice()),
            'short-weights',
            'float32',
            5
          )
        }
      })
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkIsochrones({...props, raster: {...props.raster!, width: 1}})).toThrow(
    /raster.width/
  );
  expect(
    () => new GPUNetworkIsochrones({...props, raster: {...props.raster!, maximumBufferPixels: 40}})
  ).toThrow(/maximumBufferPixels/);
  expect(() => new GPUNetworkIsochrones({...props, startCount: props.topology.neighbors})).toThrow(
    /needs startNodes/
  );
});

// Review B-05: the wrapped search's convergence was invisible, so a high-diameter network was
// silently truncated into smaller polygons.
it('GPUNetworkIsochrones forwards converged and localIterations to its search', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-converged'});
  for (const withAssignments of [false, true]) {
    const converged = createTransientView(graph, `converged-${serial++}`, 'uint32', 1);
    const nodes = new GPUNetworkIsochrones(
      createProps(graph, {
        id: `search-${withAssignments}`,
        startNodes: createTransientView(graph, `sources-${serial++}`, 'uint32', 2),
        assignments: withAssignments
          ? createTransientView(graph, `assignments-${serial++}`, 'uint32', 4)
          : undefined,
        maximumIterations: 2,
        localIterations: 4,
        output: {status: {converged}}
      })
    ).getCommandNodes(graph);
    const writers = nodes.filter(node =>
      node.resources?.some(
        resource =>
          getResourceBuffer(resource) === converged && resource.usage === 'storage-read-write'
      )
    );
    expect(writers.length).toBeGreaterThan(0);
  }
  const props = createProps(graph);
  const flag = createTransientView(graph, `converged-${serial++}`, 'uint32', 1);
  expect(() => new GPUNetworkIsochrones({...props, output: {status: {converged: flag}}})).toThrow(
    /needs startNodes/
  );
  expect(() => new GPUNetworkIsochrones({...props, localIterations: 4})).toThrow(
    /needs startNodes/
  );
  const startNodes = createTransientView(graph, `sources-${serial++}`, 'uint32', 2);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        startNodes,
        output: {
          status: {converged: createTransientView(graph, `converged-${serial++}`, 'uint32', 2)}
        }
      })
  ).toThrow(/output\.status\.converged must contain exactly one row/);
  // Optional, as on GPUNetworkReachability and GPUNetworkServiceAreas.
  expect(() => new GPUNetworkIsochrones({...props, startNodes})).not.toThrow();
});

it('getGPUNetworkIsochronesParameterValues packs and validates', () => {
  const values = getGPUNetworkIsochronesParameterValues({
    breakCount: 3,
    extent: [0, 1, 10, 21],
    bufferRadius: 2,
    walkCostPerUnit: 0.25
  });
  expect(Array.from(values)).toEqual([
    3,
    0,
    1,
    10,
    21,
    2,
    0.25,
    0,
    2,
    3.0e38 > 3.4e38 ? 0 : Math.fround(3.0e38),
    0,
    0
  ]);
  expect(() =>
    getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 0, 1]})
  ).toThrow(/GPUNetworkIsochrones extent/);
  expect(() =>
    getGPUNetworkIsochronesParameterValues({breakCount: -1, extent: [0, 0, 1, 1]})
  ).toThrow(/GPUNetworkIsochrones breakCount/);
});

it('GPUNetworkIsochrones bakes a numeric startCount without a binding', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-count'});
  const startNodes = createTransientView(graph, `sources-${serial++}`, 'uint32', 3);
  const getSeedSource = (startCount: number | ReturnType<typeof createTransientView>) => {
    const nodes = new GPUNetworkIsochrones(
      createProps(graph, {id: `count-${serial++}`, startNodes, startCount: startCount as never})
    ).getCommandNodes(graph);
    // Only the search kernel is parsed; the isobands kernels belong to another contributor.
    return getParsedKernelSources(nodes.filter(node => node.id.endsWith('-search-seed')));
  };
  const constant = getSeedSource(2);
  const seedSources = [...constant].filter(([id]) => id.endsWith('-search-seed'));
  expect(seedSources.length).toBeGreaterThan(0);
  for (const [, source] of seedSources) {
    expect(source).toContain('if (index >= 2u) { return; }');
    expect(source).not.toContain('sourceCount[');
  }
  expect(() => new GPUNetworkIsochrones(createProps(graph, {startNodes, startCount: 4}))).toThrow(
    /startCount must be an integer in \[0, 3\]/
  );
});

it('forwards iterationLimitReached without converged through both isochrone search paths', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const withAssignments of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const iterationLimitReached = createTransientView(graph, 'limit-reached', 'uint32', 1);
      const status = {converged: undefined, iterationLimitReached};
      const nodes = new GPUNetworkIsochrones(
        createProps(graph, {
          id: 'stopping',
          startNodes: createTransientView(graph, 'sources', 'uint32', 1),
          assignments: withAssignments
            ? createTransientView(graph, 'assignments', 'uint32', 4)
            : undefined,
          output: {status}
        })
      ).getCommandNodes(graph);
      const writers = nodes.filter(node =>
        node.resources?.some(
          resource =>
            getResourceBuffer(resource) === iterationLimitReached &&
            resource.usage === 'storage-read-write'
        )
      );
      expect(writers).toHaveLength(1);
    }
  } finally {
    device.destroy();
  }
});

it('validates search status scalars, graph ownership and separation from geometry and inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const base = createProps(graph, {
      id: 'stopping',
      startNodes: createTransientView(graph, 'sources', 'uint32', 1)
    });
    const flag = createTransientView(graph, 'flag', 'uint32', 1);
    for (const name of ['converged', 'iterationLimitReached'] as const) {
      const withFlag = (view: typeof flag) =>
        new GPUNetworkIsochrones({
          ...base,
          output: {status: {[name]: view}}
        });
      for (const length of [0, 2]) {
        expect(() =>
          withFlag(createTransientView(graph, `bad-${name}-${length}`, 'uint32', length))
        ).toThrow(new RegExp(`^stopping output.status.${name} must contain exactly one row`));
      }
      expect(() =>
        withFlag(createTransientView(graph, `float-${name}`, 'float32', 1) as never)
      ).toThrow(new RegExp(`^stopping output.status.${name}`));
      expect(() => withFlag(base.startNodes!)).toThrow(/stopping.*share buffers/);
      expect(() => withFlag(base.raster!.output.status!.count!)).toThrow(/stopping.*share buffers/);
      const foreign = createTransientView(
        new GPUCommandGraph(device),
        `foreign-${name}`,
        'uint32',
        1
      );
      expect(() => withFlag(foreign).getCommandNodes(graph)).toThrow(/stopping.*target graph/);
      expect(
        () =>
          new GPUNetworkIsochrones({
            ...base,
            startNodes: undefined,
            output: {status: {[name]: flag}}
          })
      ).toThrow(/stopping output needs startNodes/);
    }
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...base,
          output: {
            status: {
              converged: flag,
              iterationLimitReached: flag
            }
          }
        })
    ).toThrow(/stopping output.status views must use separate buffers/);
  } finally {
    device.destroy();
  }
});

for (const withAssignments of [false, true]) {
  for (const alias of ['status', 'input', 'geometry'] as const) {
    it(`rejects physical isochrone status aliases through duplicate imports (${alias}, assignments: ${withAssignments})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const physicalBuffer = device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
      try {
        const firstHandle = graph.importBuffer(
          {id: 'first-import', byteLength: 256, usage: Buffer.STORAGE},
          physicalBuffer
        );
        const secondHandle = graph.importBuffer(
          {id: 'second-import', byteLength: 256, usage: Buffer.STORAGE},
          physicalBuffer
        );
        expect(firstHandle).not.toBe(secondHandle);
        const firstView = graph.createDataView(firstHandle, {
          format: 'uint32',
          byteOffset: 0,
          length: 1
        });
        const secondView = graph.createDataView(secondHandle, {
          format: 'uint32',
          byteOffset: 0,
          length: 1
        });
        const base = createProps(graph, {
          id: 'physical-status',
          startNodes:
            alias === 'input' ? firstView : createTransientView(graph, 'sources', 'uint32', 1),
          assignments: withAssignments
            ? createTransientView(graph, 'assignments', 'uint32', 4)
            : undefined
        });
        expect(
          () =>
            new GPUNetworkIsochrones({
              ...base,
              raster:
                alias === 'geometry'
                  ? {
                      ...base.raster!,
                      output: {
                        ...base.raster!.output,
                        status: {...base.raster!.output.status, count: firstView}
                      }
                    }
                  : base.raster,
              output: {
                status: {
                  converged: alias === 'status' ? firstView : undefined,
                  iterationLimitReached: secondView
                }
              }
            })
        ).toThrow(/^physical-status.*(?:separate buffers|share buffers)/);
      } finally {
        physicalBuffer.destroy();
        device.destroy();
      }
    });
  }
}

it('GPUNetworkIsochrones validates additional snapped-source path inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'source-path-validation'});
  const props = createProps(graph);
  const sourcePaths = {topology: props.topology, costs: props.costs};
  try {
    expect(
      () =>
        new GPUNetworkIsochrones({...props, raster: {...props.raster!, sourcePaths, mode: 'max'}})
    ).toThrow(/sourcePaths.*min/);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          raster: {
            ...props.raster!,
            sourcePaths: {
              ...sourcePaths,
              costs: createTransientView(graph, 'short-source-path-costs', 'float32', 3)
            }
          }
        })
    ).toThrow(/sourcePaths.costs.*one row/);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          assignments: createTransientView(graph, 'source-labels', 'uint32', 4),
          raster: {
            ...props.raster!,
            sourcePaths,
            output: {
              ...props.raster!.output,
              pixelFacilities: createTransientView(graph, 'source-pixel-labels', 'uint32', 64)
            }
          }
        })
    ).toThrow(/sourcePaths.assignments.*required/);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          raster: {
            ...props.raster!,
            sourcePaths,
            output: {...props.raster!.output, values: props.costs}
          }
        })
    ).toThrow(/values.*width/);
  } finally {
    device.destroy();
  }
});
