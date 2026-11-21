// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('terrain roadmap kernels parse within portable binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `terrain-${identifier++}`, format, length);
  const grid = {
    width: 3,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const contributors = [
    ...(['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const).map(
      routing =>
        new GPUTerrainFlowFractions({
          id: routing,
          grid,
          elevation: view('float32', 3),
          flowDirections: view('uint32', 3),
          parameters: view('float32', 8),
          routing,
          output: {fractions: view('float32', 24), edgeLengths: view('float32', 24)}
        })
    ),
    ...(
      [
        'transport',
        'expected-distance',
        'minimum-distance',
        'maximum-distance',
        'fractional-catchment'
      ] as const
    ).map(
      mode =>
        new GPUTerrainBranchingFlow({
          id: mode,
          grid,
          fractions: view('float32', 24),
          values: view('float32', 3),
          mode,
          edgeLengths: view('float32', 24),
          maximumIterations: 2,
          output: {
            values: view('float32', 3),
            status: view('uint32', 3),
            unresolvedCount: view('uint32', 1)
          }
        })
    ),
    new GPUTerrainBreach({
      grid,
      elevation: view('float32', 3),
      pitCells: view('uint32', 1),
      maximumSteps: 2,
      maximumCutDepth: 3,
      elevationDecrement: 0.1,
      burnDepth: view('float32', 3),
      output: {
        elevation: view('float32', 3),
        cutDepth: view('float32', 3),
        modificationMask: view('uint32', 3),
        pathCells: view('uint32', 3),
        pathLengths: view('uint32', 1),
        pitStatus: view('uint32', 1),
        unresolvedCount: view('uint32', 1)
      }
    }),
    new GPUTerrainCatchmentSummary({
      grid,
      elevation: view('float32', 3),
      basinIds: view('uint32', 3),
      basinCount: 1,
      streams: view('uint32', 3),
      flowDirections: view('uint32', 3),
      histogramBins: 2,
      output: {
        statistics: view('float32', 6),
        hypsometry: view('float32', 2),
        profiles: view('float32', 9),
        profileStatus: view('uint32', 3)
      }
    }),
    new GPUTerrainDEMCoregistration({
      grid,
      reference: view('float32', 3),
      moving: view('float32', 3),
      stableMask: view('uint32', 3),
      differenceVariance: view('float32', 3),
      correlationLength: 2,
      output: {
        fit: view('float32', 4),
        differences: view('float32', 6),
        validity: view('uint32', 3),
        volume: view('float32', 3)
      }
    }),
    new GPUTerrainDepressionHierarchy({
      grid,
      elevation: view('float32', 3),
      outlets: view('uint32', 3),
      stages: view('float32', 2),
      nodeCapacity: 3,
      output: {
        nodes: view('uint32', 12),
        levels: view('float32', 6),
        cellNodes: view('uint32', 3),
        curves: view('float32', 12),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    }),
    new GPUTerrainTiledFlow({
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      maximumIterations: 2,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 3,
          height: 1,
          owned: [0, 0, 3, 1],
          flowDirections: view('uint32', 3),
          runoff: view('float32', 3),
          streams: view('uint32', 3),
          output: {
            accumulation: view('float32', 3),
            status: view('uint32', 3),
            reachIds: view('uint32', 3)
          }
        }
      ]
    })
  ];
  contributors.push(
    new GPUTerrainTiledFlow({
      id: 'branching-tiles',
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      routing: 'branching',
      maximumIterations: 2,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 3,
          height: 1,
          owned: [0, 0, 3, 1],
          flowDirections: view('uint32', 3),
          fractions: view('float32', 24),
          runoff: view('float32', 3),
          streams: view('uint32', 3),
          output: {
            accumulation: view('float32', 3),
            status: view('uint32', 3),
            reachIds: view('uint32', 3)
          }
        }
      ]
    })
  );
  for (const contributor of contributors) {
    const commandNodes = contributor.getCommandNodes(graph);
    if (contributor instanceof GPUTerrainBreach) {
      expect(commandNodes.map(node => node.id)).toContain(`${contributor.id}-verify-paths`);
      expect(commandNodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(commandNodes.map(node => node.id)).not.toContain(`${contributor.id}-search`);
      const relaxations = commandNodes.filter(node => /-search-relax-/.test(node.id));
      expect(relaxations).toHaveLength(contributor.props.maximumSteps);
      for (const node of relaxations) expect(node.condition).toBeDefined();
    }
    if (contributor instanceof GPUTerrainDepressionHierarchy) {
      expect(commandNodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(commandNodes.map(node => node.id)).not.toContain(`${contributor.id}-build`);
      expect(commandNodes.map(node => node.id)).toContain(
        `${contributor.id}-activation-splice-round-0`
      );
      expect(commandNodes.map(node => node.id)).toContain(`${contributor.id}-activation-status`);
      expect(commandNodes.map(node => node.id)).not.toContain(`${contributor.id}-status-ports`);
    }
    const sources = getParsedKernelSources(commandNodes);
    expect(sources.size).toBeGreaterThan(0);
    for (const {id: name, source, constants} of sources.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    }
  }
});

it('tile ownership rejects missing domains and overlapping interiors before GPU work', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `ownership-${identifier++}`, format, length);
  const tile = (column: number) => ({
    column,
    row: 0,
    width: 2,
    height: 1,
    owned: [0, 0, 2, 1] as const,
    flowDirections: view('uint32', 2),
    runoff: view('float32', 2),
    streams: view('uint32', 2),
    output: {
      accumulation: view('float32', 2),
      status: view('uint32', 2),
      reachIds: view('uint32', 2)
    }
  });
  expect(
    () =>
      new GPUTerrainTiledFlow({
        grid: {width: 4, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        tiles: [tile(0)]
      })
  ).toThrow();
  expect(
    () =>
      new GPUTerrainTiledFlow({
        grid: {width: 4, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        tiles: [tile(0), tile(1)]
      })
  ).toThrow();
});

it('terrain outputs reject aliases and malformed capacity', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `capacity-${identifier++}`, format, length);
  const values = view('float32', 2);
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  expect(
    () =>
      new GPUTerrainBranchingFlow({
        grid,
        fractions: view('float32', 16),
        values,
        mode: 'transport',
        output: {values, status: view('uint32', 2), unresolvedCount: view('uint32', 1)}
      })
  ).toThrow();
  expect(
    () =>
      new GPUTerrainDepressionHierarchy({
        grid,
        elevation: values,
        outlets: view('uint32', 2),
        stages: view('float32', 2),
        nodeCapacity: 2,
        output: {
          nodes: view('uint32', 7),
          levels: view('float32', 4),
          cellNodes: view('uint32', 2),
          curves: view('float32', 8),
          status: {
            count: view('uint32', 1),
            requiredCount: view('uint32', 1),
            overflow: view('uint32', 1)
          }
        }
      })
  ).toThrow();
});

it('terrain processing validation names the contributor, the prop and the accepted range', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `messages-${identifier++}`, format, length);
  const grid = {
    width: 4,
    height: 4,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const createBreach = (
    overrides: Partial<ConstructorParameters<typeof GPUTerrainBreach>[0]> = {},
    pitCount = 1,
    maximumSteps = 2
  ) =>
    new GPUTerrainBreach({
      id: 'breach-under-test',
      grid,
      elevation: view('float32', 16),
      pitCells: view('uint32', pitCount),
      maximumSteps,
      maximumCutDepth: 3,
      elevationDecrement: 0.1,
      output: {
        elevation: view('float32', 16),
        cutDepth: view('float32', 16),
        modificationMask: view('uint32', 16),
        pathCells: view('uint32', pitCount * (maximumSteps + 1)),
        pathLengths: view('uint32', pitCount),
        pitStatus: view('uint32', pitCount),
        unresolvedCount: view('uint32', 1)
      },
      ...overrides
    });
  expect(() => createBreach({maximumSteps: 0})).toThrow(
    'breach-under-test maximumSteps must be an integer in [1, 4294967295]'
  );
  expect(() => createBreach({elevationDecrement: 0})).toThrow(
    'breach-under-test elevationDecrement must be a positive finite number'
  );
  expect(() => createBreach({validity: view('uint32', 3)})).toThrow(
    'breach-under-test validity must hold one uint32 word per grid cell'
  );
  expect(
    () =>
      new GPUTerrainBranchingFlow({
        id: 'branching-under-test',
        grid,
        fractions: view('float32', 128),
        values: view('float32', 16),
        mode: 'transport',
        maximumWalkLength: 2000,
        output: {
          values: view('float32', 16),
          status: view('uint32', 16),
          unresolvedCount: view('uint32', 1)
        }
      })
  ).toThrow('branching-under-test maximumWalkLength must be an integer in [1, 1024]');
  expect(
    () =>
      new GPUTerrainBranchingFlow({
        id: 'geographic-under-test',
        grid: {...grid, affine: [1, 0.5, 0, 0, 1, 0]},
        cellSizeMode: 'geographic',
        fractions: view('float32', 128),
        values: view('float32', 16),
        mode: 'expected-distance',
        output: {
          values: view('float32', 16),
          status: view('uint32', 16),
          unresolvedCount: view('uint32', 1)
        }
      })
  ).toThrow('geographic cell sizes need an axis-aligned grid');
});

it('breach rejects a search scratch above the storage binding limit with a named message', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `scratch-${identifier++}`, format, length);
  // 300 pits x 33 steps x (65 x 65) window cells x 2 words x 4 bytes is about 335 MiB.
  const pitCount = 300;
  const breach = new GPUTerrainBreach({
    id: 'scratch-under-test',
    grid: {width: 100, height: 100, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    elevation: view('float32', 10000),
    pitCells: view('uint32', pitCount),
    maximumSteps: 32,
    maximumCutDepth: 3,
    elevationDecrement: 0.1,
    output: {
      elevation: view('float32', 10000),
      cutDepth: view('float32', 10000),
      modificationMask: view('uint32', 10000),
      pathCells: view('uint32', pitCount * 33),
      pathLengths: view('uint32', pitCount),
      pitStatus: view('uint32', pitCount),
      unresolvedCount: view('uint32', 1)
    }
  });
  expect(() => breach.getCommandNodes(graph)).toThrow(
    /scratch-under-test search scratch needs \d+ bytes in one binding \(limit \d+\); reduce pitCells.length or maximumSteps/
  );
});

it('flow distance and stream network declare their command node count and stay within binding limits', () => {
  for (const [width, height, maximumSteps] of [
    [3, 1, undefined],
    [64, 64, undefined],
    [64, 64, 9]
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let identifier = 0;
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `jump-${identifier++}`, format, length);
    const count = width * height;
    const grid = {
      width,
      height,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const distance = new GPUTerrainFlowDistance({
      grid,
      flowDirections: view('uint32', count),
      streams: view('uint32', count),
      transitionCosts: view('float32', count),
      maximumSteps,
      output: {
        distance: view('float32', count),
        validity: view('uint32', count),
        terminalCells: view('uint32', count)
      }
    });
    const network = new GPUTerrainStreamNetwork({
      grid,
      flowDirections: view('uint32', count),
      streams: view('uint32', count),
      maximumSteps,
      output: {
        sourceCells: view('uint32', 8),
        targetCells: view('uint32', 8),
        vertexOffsets: view('uint32', 9),
        positions: view('float32x2', 32),
        lengths: view('float32', 8),
        reachCount: view('uint32', 1),
        vertexCount: view('uint32', 1),
        status: {overflow: view('uint32', 1)},
        unresolvedCount: view('uint32', 1)
      }
    });
    const distanceNodes = distance.getCommandNodes(graph);
    expect(distanceNodes.length).toBe(distance.estimatedCommandNodeCount);
    const networkNodes = network.getCommandNodes(graph);
    expect(networkNodes.length).toBeLessThanOrEqual(network.estimatedCommandNodeCount);
    for (const {id: name, source, constants} of getParsedKernelSources([
      ...distanceNodes,
      ...networkNodes
    ]).configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    }
  }
});

it('flow fractions reject a sheared grid, whose two diagonal steps differ, and accept a rotated one', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `fractions-${identifier++}`, format, length);
  const create = (affine: readonly [number, number, number, number, number, number]) =>
    new GPUTerrainFlowFractions({
      grid: {width: 3, height: 1, affine, pixelInterpretation: 'area'},
      elevation: view('float32', 3),
      flowDirections: view('uint32', 3),
      parameters: view('float32', 8),
      routing: 'd-infinity',
      output: {fractions: view('float32', 24), edgeLengths: view('float32', 24)}
    });
  expect(() => create([1, 0.5, 0, 0, -1, 0])).toThrow(
    /^terrain-flow-fractions grid\.affine must have orthogonal pixel axes/
  );
  expect(() => create([0.6, 0.8, 0, 0.8, -0.6, 0])).not.toThrow();
  expect(() => create([1, 0, 0, 0, 1, 0])).not.toThrow();
});

it('breach rejects an excessive step graph before creating command nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `budget-${identifier++}`, format, length);
  expect(
    () =>
      new GPUTerrainBreach({
        id: 'breach-budget',
        grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        elevation: view('float32', 1),
        pitCells: view('uint32', 1),
        maximumSteps: 16384,
        maximumCutDepth: 1,
        elevationDecrement: 0.5,
        output: {
          elevation: view('float32', 1),
          cutDepth: view('float32', 1),
          modificationMask: view('uint32', 1),
          pathCells: view('uint32', 16385),
          pathLengths: view('uint32', 1),
          pitStatus: view('uint32', 1),
          unresolvedCount: view('uint32', 1)
        }
      })
  ).toThrow(/breach-budget.*estimated command node count.*32768/);
});

it('hierarchy requires three one-row status views and rejects the legacy output shape', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let identifier = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length = 1) =>
      createTransientView(graph, `hierarchy-status-${identifier++}`, format, length);
    const props = {
      id: 'hierarchy-status',
      grid: {
        width: 1,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      elevation: view('float32'),
      outlets: view('uint32'),
      stages: view('float32'),
      nodeCapacity: 1,
      output: {
        nodes: view('uint32', 4),
        levels: view('float32', 2),
        cellNodes: view('uint32'),
        curves: view('float32', 2),
        status: {count: view('uint32'), requiredCount: view('uint32'), overflow: view('uint32')}
      }
    };
    expect(
      () =>
        new GPUTerrainDepressionHierarchy({
          ...props,
          // @ts-expect-error Supply a map of one-row views instead of a packed status view.
          output: {...props.output, status: view('uint32', 3)}
        })
    ).toThrow(/^hierarchy-status output\.status.*output\.status\.count.*requiredCount.*overflow/);
    expect(
      () =>
        new GPUTerrainDepressionHierarchy({
          ...props,
          // @ts-expect-error The former optional status map has been removed.
          output: {...props.output, ports: {status: {overflow: view('uint32')}}}
        })
    ).toThrow(/^hierarchy-status output\.ports.*output\.status\.count.*requiredCount.*overflow/);
    for (const port of ['count', 'requiredCount', 'overflow'] as const) {
      expect(
        () =>
          new GPUTerrainDepressionHierarchy({
            ...props,
            output: {
              ...props.output,
              status: {...props.output.status, [port]: view('uint32', 2)}
            }
          })
      ).toThrow(new RegExp(`^hierarchy-status output\\.status\\.${port} must have 1 rows`));
    }
  } finally {
    device.destroy();
  }
});

it('hierarchy rejects reconstruction scratch above the binding limit before allocating nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `activation-storage-${identifier++}`, format, length);
  const count = Math.floor(graph.device.limits.maxStorageBufferBindingSize / 72) + 1;
  const hierarchy = new GPUTerrainDepressionHierarchy({
    id: 'hierarchy-storage',
    grid: {width: count, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    elevation: view('float32', count),
    outlets: view('uint32', count),
    stages: view('float32', 1),
    nodeCapacity: 1,
    output: {
      nodes: view('uint32', 4),
      levels: view('float32', 2),
      cellNodes: view('uint32', count),
      curves: view('float32', 2),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    }
  });
  expect(() => hierarchy.getCommandNodes(graph)).toThrow(
    /hierarchy-storage activation scratch needs.*limit.*reduce grid dimensions/
  );
});

it('parallel terrain node estimates cover single cells, padding and 262144-cell activation graphs', () => {
  for (const count of [1, 257, 262144]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let identifier = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `parallel-${identifier++}`, format, length);
    const grid = {
      width: count,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const hierarchy = new GPUTerrainDepressionHierarchy({
      id: 'parallel-hierarchy',
      grid,
      elevation: view('float32', count),
      outlets: view('uint32', count),
      stages: view('float32', 1),
      nodeCapacity: 1,
      output: {
        nodes: view('uint32', 4),
        levels: view('float32', 2),
        cellNodes: view('uint32', count),
        curves: view('float32', 2),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    });
    const hierarchyNodes = hierarchy.getCommandNodes(graph);
    expect(hierarchyNodes.length).toBeLessThanOrEqual(hierarchy.estimatedCommandNodeCount);
    expect(
      hierarchyNodes.find(node => node.id === 'parallel-hierarchy-activation-splice-round-0')
        ?.workload?.maximumInvocationCount
    ).toBe(count * 8);
    for (const pitCount of [0, 33]) {
      const breach = new GPUTerrainBreach({
        id: `parallel-breach-${pitCount}`,
        grid,
        elevation: view('float32', count),
        validity: view('uint32', count),
        pitCells: view('uint32', pitCount),
        maximumSteps: 3,
        maximumCutDepth: 1,
        elevationDecrement: 0.5,
        output: {
          elevation: view('float32', count),
          cutDepth: view('float32', count),
          modificationMask: view('uint32', count),
          pathCells: view('uint32', pitCount * 4),
          pathLengths: view('uint32', pitCount),
          pitStatus: view('uint32', pitCount),
          unresolvedCount: view('uint32', 1)
        }
      });
      const breachNodes = breach.getCommandNodes(graph);
      expect(breachNodes.length).toBeLessThanOrEqual(breach.estimatedCommandNodeCount);
      const relaxations = breachNodes.filter(node => /-search-relax-/.test(node.id));
      expect(relaxations).toHaveLength(pitCount === 0 ? 0 : 3);
      for (const node of relaxations)
        expect(node.workload?.maximumInvocationCount).toBe(pitCount * Math.min(count, 7));
    }
  }
});
