// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCPUOperation,
  GPUGraphOperation,
  GPUOperationContract
} from '@luma.gl/graph/gpu-core';
import {
  GPUProfilePeaksOperation,
  CPUProfilePeaksOperation,
  GPUTerrainBranchingFlowOperation,
  CPUTerrainBranchingFlowOperation,
  GPUTerrainBreachOperation,
  CPUTerrainBreachOperation,
  GPUTerrainDepressionHierarchyOperation,
  CPUTerrainDepressionHierarchyOperation,
  GPUTerrainCatchmentSummaryOperation,
  CPUTerrainCatchmentSummaryOperation,
  GPUTerrainFlowOperation,
  CPUTerrainFlowOperation,
  GPUTerrainStreamOrderOperation,
  CPUTerrainStreamOrderOperation,
  GPUTerrainWatershedsOperation,
  CPUTerrainWatershedsOperation,
  GPUTerrainPeakProminenceOperation,
  CPUTerrainPeakProminenceOperation,
  GPUSolarPositionOperation,
  CPUSolarPositionOperation,
  getGPUProfilePeaksParameterValues,
  getGPUTerrainFlowParameterValues,
  type ProfilePeaksOperationProps,
  type SolarPositionOperationProps
} from '../../src/gpu-terrain/index';
import {makeGPURasterMetadata} from '../../src/raster-analysis/index';
import type {OperationGraphInput} from '../utils/mixed-execution-test-utils';
import {POSITION_FIXTURE} from './terrain-illumination/parity-wp5-solar-fixtures';

export type TerrainOperationOracle = {
  name: string;
  inputs: Record<string, OperationGraphInput>;
  gpu(): GPUGraphOperation & {readonly contract: GPUOperationContract};
  cpu(): GPUCPUOperation;
  expected: Record<string, number[]>;
  gpuExpected?: Record<string, number[]>;
  angleBounds?: number[];
};

const NONE = 0xffffffff;
const grid = (width: number, height = 1) => makeGPURasterMetadata({width, height});
const floats = (values: ArrayLike<number>): OperationGraphInput => ({
  format: 'float32',
  values: Float32Array.from(values)
});
const words = (values: ArrayLike<number>): OperationGraphInput => ({
  format: 'uint32',
  values: Uint32Array.from(values)
});

/** Literal expectations below come from the derivations in each fixture, never a CPU/GPU run. */
export function getTerrainOperationOracles(): TerrainOperationOracle[] {
  const fixtures: TerrainOperationOracle[] = [];
  // Two copies of y=4-(x-2.25)^2, the second translated vertically by ten.
  // At sample 2: prominence=3, vertex=(2.25,4); the second vertex is (2.25,14).
  // Capacity one stores ID 2, while the independent accepted-peak requirement is two.
  for (const selection of ['all', 'prominence', 'refinedValue', 'peakMask', 'peaks'] as const) {
    const dense = selection === 'all';
    const compact = dense || selection === 'peaks';
    const peaks = {
      ids: 'ids',
      capacity: 1,
      status: {
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    };
    const output: ProfilePeaksOperationProps['output'] = dense
      ? {
          prominence: 'prominence',
          refinedIndex: 'refinedIndex',
          refinedValue: 'refinedValue',
          peakMask: 'peakMask',
          peaks,
          status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
        }
      : selection === 'prominence'
        ? {prominence: 'prominence'}
        : selection === 'refinedValue'
          ? {refinedValue: 'refinedValue'}
          : selection === 'peakMask'
            ? {peakMask: 'peakMask'}
            : {peaks};
    const props: ProfilePeaksOperationProps = {
      values: 'values',
      offsets: 'offsets',
      parameters: 'parameters',
      sampleCount: 10,
      window: 2,
      minimumSide: 2,
      suppressionRadius: 0,
      output
    };
    const expected: Record<string, number[]> = {};
    const prominence = [NaN, NaN, 3, NaN, NaN, NaN, NaN, 3, NaN, NaN];
    const refinedValue = [NaN, NaN, 4, NaN, NaN, NaN, NaN, 14, NaN, NaN];
    const peakMask = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0];
    if (props.output.prominence) expected['prominence'] = prominence;
    if (props.output.refinedValue) expected['refinedValue'] = refinedValue;
    if (props.output.peakMask) expected['peakMask'] = peakMask;
    if (dense)
      Object.assign(expected, {
        refinedIndex: [NaN, NaN, 2.25, NaN, NaN, NaN, NaN, 2.25, NaN, NaN],
        converged: [1],
        iterationLimitReached: [0]
      });
    if (compact)
      Object.assign(expected, {
        ids: [2],
        count: [1],
        requiredCount: [2],
        overflow: [1],
        candidateOverflow: [0]
      });
    fixtures.push({
      name: `profile-parabolas/${selection}`,
      inputs: {
        values: floats([
          -1.0625, 2.4375, 3.9375, 3.4375, 0.9375, 8.9375, 12.4375, 13.9375, 13.4375, 10.9375
        ]),
        offsets: words([0, 5, 10]),
        parameters: floats(getGPUProfilePeaksParameterValues({minimumProminence: 1}))
      },
      gpu: () => new GPUProfilePeaksOperation(props),
      cpu: () => new CPUProfilePeaksOperation(props),
      expected
    });
  }

  // Diamond: 0 sends 1/4 east and 3/4 south; both receivers flow to 3.
  // The two paths have lengths 2 and 4, so their mean length is 3.5.
  const fractions = new Float32Array(32);
  fractions[0] = 0.25;
  fractions[2] = 0.75;
  fractions[10] = 1;
  fractions[16] = 1;
  const lengths = new Float32Array(32);
  lengths[0] = 1;
  lengths[2] = 3;
  lengths[10] = 1;
  lengths[16] = 1;
  for (const [mode, values, expectedValues] of [
    ['transport', [8, 0, 0, 0], [8, 2, 6, 8]],
    ['expected-distance', [0, 0, 0, 0], [3.5, 1, 1, 0]],
    ['minimum-distance', [0, 0, 0, 0], [2, 1, 1, 0]],
    ['maximum-distance', [0, 0, 0, 0], [4, 1, 1, 0]],
    ['fractional-catchment', [0, 0, 0, 1], [1, 1, 1, 1]]
  ] as const) {
    const props = {
      grid: grid(2, 2),
      mode,
      fractions: 'fractions',
      values: 'values',
      edgeLengths: 'lengths',
      output: {values: 'result', status: 'status', unresolvedCount: 'unresolvedCount'}
    };
    fixtures.push({
      name: `branching-diamond/${mode}`,
      inputs: {fractions: floats(fractions), values: floats(values), lengths: floats(lengths)},
      gpu: () => new GPUTerrainBranchingFlowOperation(props),
      cpu: () => new CPUTerrainBranchingFlowOperation(props),
      expected: {result: [...expectedValues], status: [1, 1, 1, 1], unresolvedCount: [0]}
    });
  }

  // One outlet reachable over the embankment: 0->1->2, lowered by 0.5 per step.
  const breachProps = {
    grid: grid(3),
    elevation: 'elevation',
    pitCells: 'pits',
    pitCount: 1,
    maximumSteps: 2,
    maximumCutDepth: 4,
    elevationDecrement: 0.5,
    output: {
      elevation: 'elevationOut',
      cutDepth: 'cutDepth',
      modificationMask: 'modificationMask',
      pathCells: 'pathCells',
      pathLengths: 'pathLengths',
      pitStatus: 'pitStatus',
      unresolvedCount: 'unresolvedCount'
    }
  };
  fixtures.push({
    name: 'breach-embankment',
    inputs: {elevation: floats([2, 5, 0]), pits: words([0])},
    gpu: () => new GPUTerrainBreachOperation(breachProps),
    cpu: () => new CPUTerrainBreachOperation(breachProps),
    expected: {
      elevationOut: [2, 1.5, 0],
      cutDepth: [0, 3.5, 0],
      modificationMask: [0, 1, 0],
      pathCells: [0, 1, 2],
      pathLengths: [3],
      pitStatus: [1],
      unresolvedCount: [0]
    }
  });
  // The only outlet path needs a 5-1.5=3.5 cut. A cut budget of three is infeasible:
  // no proposals or path are published, and the pit must remain unresolved.
  const infeasibleBreachProps = {...breachProps, maximumCutDepth: 3};
  fixtures.push({
    name: 'breach-embankment/infeasible-cut',
    inputs: {elevation: floats([2, 5, 0]), pits: words([0])},
    gpu: () => new GPUTerrainBreachOperation(infeasibleBreachProps),
    cpu: () => new CPUTerrainBreachOperation(infeasibleBreachProps),
    expected: {
      elevationOut: [2, 5, 0],
      cutDepth: [0, 0, 0],
      modificationMask: [0, 0, 0],
      pathCells: [NONE, NONE, NONE],
      pathLengths: [0],
      pitStatus: [2],
      unresolvedCount: [1]
    }
  });

  // 0--2--1 has two leaves merging at height 2. Activation ownership is [leaf0,root,leaf1].
  // At stages 1 and 3 each leaf has area one; the root has areas two/three and volumes one/six.
  const hierarchyProps = {
    grid: grid(3),
    elevation: 'elevation',
    outlets: 'outlets',
    stages: 'stages',
    stageCount: 2,
    nodeCapacity: 3,
    output: {
      nodes: 'nodes',
      levels: 'levels',
      cellNodes: 'cellNodes',
      curves: 'curves',
      status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
    }
  };
  fixtures.push({
    name: 'hierarchy-three-cells',
    inputs: {elevation: floats([0, 2, 1]), outlets: words([0, 0, 0]), stages: floats([1, 3])},
    gpu: () => new GPUTerrainDepressionHierarchyOperation(hierarchyProps),
    cpu: () => new CPUTerrainDepressionHierarchyOperation(hierarchyProps),
    expected: {
      nodes: [2, 0, NONE, 1, 2, 2, NONE, 1, NONE, 0, NONE, 1],
      levels: [0, 2, 1, 2, 2, Infinity],
      cellNodes: [0, 2, 1],
      curves: [1, 1, 1, 3, 1, 0, 1, 2, 2, 1, 3, 6],
      count: [3],
      requiredCount: [3],
      overflow: [0]
    }
  });
  // The same three-node requirement cannot fit one node; only status is usable.
  const hierarchyOverflowProps = {...hierarchyProps, nodeCapacity: 1};
  fixtures.push({
    name: 'hierarchy-three-cells/overflow',
    inputs: {elevation: floats([0, 2, 1]), outlets: words([0, 0, 0]), stages: floats([1, 3])},
    gpu: () => new GPUTerrainDepressionHierarchyOperation(hierarchyOverflowProps),
    cpu: () => new CPUTerrainDepressionHierarchyOperation(hierarchyOverflowProps),
    expected: {count: [1], requiredCount: [3], overflow: [1]}
  });

  // Three unit-area cells descend 3,2,1: cumulative hypsometry is [2,3], downstream
  // distances are [2,1,0], outgoing gradients [1,1,0], all profile states resolved.
  const summaryProps = {
    grid: grid(3),
    elevation: 'elevation',
    basinIds: 'basins',
    streams: 'streams',
    flowDirections: 'directions',
    basinCount: 1,
    histogramBins: 2,
    output: {
      statistics: 'statistics',
      hypsometry: 'hypsometry',
      profiles: 'profiles',
      profileStatus: 'profileStatus'
    }
  };
  fixtures.push({
    name: 'catchment-three-cells',
    inputs: {
      elevation: floats([3, 2, 1]),
      basins: words([0, 0, 0]),
      streams: words([1, 1, 1]),
      directions: words([1, 1, 0])
    },
    gpu: () => new GPUTerrainCatchmentSummaryOperation(summaryProps),
    cpu: () => new CPUTerrainCatchmentSummaryOperation(summaryProps),
    expected: {
      statistics: [3, 2, 1, 3, 2, 2 / 3],
      hypsometry: [2, 3],
      profiles: [2, 3, 1, 1, 2, 1, 0, 1, 0],
      profileStatus: [1, 1, 1]
    }
  });

  // A singleton is already an outlet/fill/flat fixpoint. CPU executes no GPU rounds;
  // GPU loops have one confirming round each (flat resolution has three loops).
  const flowGrid = grid(1);
  const flowProps = {
    grid: flowGrid,
    fillDepressions: true,
    resolveFlats: true,
    maximumFillIterations: 1,
    maximumFlatIterations: 1,
    maximumAccumulationIterations: 1,
    elevation: {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'elevation'}
    },
    parameters: 'parameters',
    output: {
      filledElevation: 'filled',
      flowDirections: 'directions',
      cellClasses: 'classes',
      accumulation: 'accumulation',
      streams: 'streams',
      status: {
        fillConverged: 'fillConverged',
        flatsConverged: 'flatsConverged',
        accumulationConverged: 'accumulationConverged',
        fillIterationCount: 'fillRounds',
        flatsIterationCount: 'flatRounds',
        accumulationIterationCount: 'accumulationRounds'
      }
    }
  };
  fixtures.push({
    name: 'flow-singleton',
    inputs: {
      elevation: floats([7]),
      parameters: floats(
        getGPUTerrainFlowParameterValues({fillEpsilon: 0.25, streamThreshold: 0.5}, flowGrid)
      )
    },
    gpu: () => new GPUTerrainFlowOperation(flowProps),
    cpu: () => new CPUTerrainFlowOperation(flowProps),
    expected: {
      filled: [7],
      directions: [0],
      classes: [3],
      accumulation: [1],
      streams: [1],
      fillConverged: [1],
      flatsConverged: [1],
      accumulationConverged: [1],
      fillRounds: [1],
      flatRounds: [3],
      accumulationRounds: [0]
    },
    gpuExpected: {accumulationRounds: [1]}
  });

  const filledOnlyProps = {
    ...flowProps,
    output: {
      filledElevation: 'filled',
      status: {flatsConverged: 'flatsConverged', flatsIterationCount: 'flatRounds'}
    }
  };
  fixtures.push({
    name: 'flow-singleton/filled-only-flat-diagnostics',
    inputs: {
      elevation: floats([7]),
      parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, flowGrid))
    },
    gpu: () => new GPUTerrainFlowOperation(filledOnlyProps),
    cpu: () => new CPUTerrainFlowOperation(filledOnlyProps),
    expected: {filled: [7], flatsConverged: [1], flatRounds: [3]},
    gpuExpected: {flatRounds: [3]}
  });
  // Eight low-edge seeds surround one flat cell. The lower stage changes infinity
  // to 2 in round one and cannot confirm convergence; all three stages execute once.
  const flatBudgetProps = {...filledOnlyProps, grid: grid(3, 3)};
  fixtures.push({
    name: 'flow-nine-tens/one-round-flat-budget',
    inputs: {
      elevation: floats(new Float32Array(9).fill(10)),
      parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0}, flatBudgetProps.grid))
    },
    gpu: () => new GPUTerrainFlowOperation(flatBudgetProps),
    cpu: () => new CPUTerrainFlowOperation(flatBudgetProps),
    expected: {filled: [10, 10, 10, 10, 10, 10, 10, 10, 10], flatsConverged: [0], flatRounds: [3]}
  });
  const confirmingFlatProps = {...flatBudgetProps, maximumFlatIterations: 2};
  fixtures.push({
    name: 'flow-nine-tens/two-round-flat-confirmation',
    inputs: {
      elevation: floats(new Float32Array(9).fill(10)),
      parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0}, flatBudgetProps.grid))
    },
    gpu: () => new GPUTerrainFlowOperation(confirmingFlatProps),
    cpu: () => new CPUTerrainFlowOperation(confirmingFlatProps),
    // Lower changes then confirms (2), higher and maximum confirm immediately (1 each).
    expected: {filled: [10, 10, 10, 10, 10, 10, 10, 10, 10], flatsConverged: [1], flatRounds: [4]}
  });
  const incompleteFlatProps = {
    ...flatBudgetProps,
    output: {
      ...flatBudgetProps.output,
      flowDirections: 'directions',
      cellClasses: 'classes',
      accumulation: 'accumulation',
      streams: 'streams'
    }
  };
  fixtures.push({
    name: 'flow-nine-tens/incomplete-flat-suppresses-routing',
    inputs: {
      elevation: floats(new Float32Array(9).fill(10)),
      parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0}, flatBudgetProps.grid))
    },
    gpu: () => new GPUTerrainFlowOperation(incompleteFlatProps),
    cpu: () => new CPUTerrainFlowOperation(incompleteFlatProps),
    expected: {
      filled: [10, 10, 10, 10, 10, 10, 10, 10, 10],
      flatsConverged: [0],
      flatRounds: [3],
      directions: new Array(9).fill(0xfffffffe),
      classes: new Array(9).fill(5),
      accumulation: new Array(9).fill(NaN),
      streams: new Array(9).fill(NONE)
    }
  });
  const twoHopFlatProps = {...confirmingFlatProps, grid: grid(5, 5)};
  fixtures.push({
    name: 'flow-twenty-five-tens/tiled-two-hop-confirmation',
    inputs: {
      elevation: floats(new Float32Array(25).fill(10)),
      parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0}, twoHopFlatProps.grid))
    },
    gpu: () => new GPUTerrainFlowOperation(twoHopFlatProps),
    cpu: () => new CPUTerrainFlowOperation(twoHopFlatProps),
    // One tile: ring and center update in two inner steps of global round one;
    // global round two confirms. This rejects a one-hop-per-global-round CPU substitute.
    expected: {filled: new Array(25).fill(10), flatsConverged: [1], flatRounds: [4]}
  });
  for (const diagnostic of ['flatsConverged', 'flatsIterationCount'] as const) {
    const props = {
      ...filledOnlyProps,
      output: {
        filledElevation: 'filled',
        status: {
          flatsConverged: diagnostic === 'flatsConverged' ? 'flatsConverged' : undefined,
          flatsIterationCount: diagnostic === 'flatsIterationCount' ? 'flatRounds' : undefined
        }
      }
    };
    fixtures.push({
      name: `flow-singleton/only-${diagnostic}-explicit-undefined`,
      inputs: {
        elevation: floats([7]),
        parameters: floats(getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, flowGrid))
      },
      gpu: () => new GPUTerrainFlowOperation(props),
      cpu: () => new CPUTerrainFlowOperation(props),
      expected: {
        filled: [7],
        ...(diagnostic === 'flatsConverged' ? {flatsConverged: [1]} : {flatRounds: [3]})
      }
    });
  }

  // One stream outlet has order one and basin label zero; both GPU solvers confirm immediately.
  const orderProps = {
    grid: grid(1),
    flowDirections: 'directions',
    streams: 'streams',
    output: {
      streamOrder: 'order',
      status: {converged: 'converged', iterationLimitReached: 'limit', iterationCount: 'rounds'}
    }
  };
  fixtures.push({
    name: 'order-singleton',
    inputs: {directions: words([0]), streams: words([1])},
    gpu: () => new GPUTerrainStreamOrderOperation(orderProps),
    cpu: () => new CPUTerrainStreamOrderOperation(orderProps),
    expected: {order: [1], converged: [1], limit: [0], rounds: [0]},
    gpuExpected: {rounds: [1]}
  });
  const watershedProps = {
    grid: grid(1),
    flowDirections: 'directions',
    output: {
      labels: 'labels',
      status: {converged: 'converged', iterationLimitReached: 'limit', iterationCount: 'rounds'}
    }
  };
  fixtures.push({
    name: 'watershed-singleton',
    inputs: {directions: words([0])},
    gpu: () => new GPUTerrainWatershedsOperation(watershedProps),
    cpu: () => new CPUTerrainWatershedsOperation(watershedProps),
    expected: {labels: [0], converged: [1], limit: [0], rounds: [0]},
    gpuExpected: {rounds: [1]}
  });
  // Initial pointers [1,2,2] become [2,2,2] in round one. Labels are resolved,
  // but convergence needs an unchanged second round, outside the explicit budget.
  const limitedWatershedProps = {...watershedProps, grid: grid(3), maximumIterations: 1};
  fixtures.push({
    name: 'watershed-three-cells/one-round-budget',
    inputs: {directions: words([1, 1, 0])},
    gpu: () => new GPUTerrainWatershedsOperation(limitedWatershedProps),
    cpu: () => new CPUTerrainWatershedsOperation(limitedWatershedProps),
    expected: {labels: [2, 2, 2], converged: [0], limit: [1], rounds: [1]}
  });
  const confirmingWatershedProps = {...limitedWatershedProps, maximumIterations: 2};
  fixtures.push({
    name: 'watershed-three-cells/two-round-confirmation',
    inputs: {directions: words([1, 1, 0])},
    gpu: () => new GPUTerrainWatershedsOperation(confirmingWatershedProps),
    cpu: () => new CPUTerrainWatershedsOperation(confirmingWatershedProps),
    expected: {labels: [2, 2, 2], converged: [1], limit: [0], rounds: [2]}
  });
  const fourCellWatershedProps = {...limitedWatershedProps, grid: grid(4)};
  fixtures.push({
    name: 'watershed-four-cells/snapshot-one-round',
    inputs: {directions: words([1, 1, 1, 0])},
    gpu: () => new GPUTerrainWatershedsOperation(fourCellWatershedProps),
    cpu: () => new CPUTerrainWatershedsOperation(fourCellWatershedProps),
    // [1,2,3,3] -> [2,3,3,3]; cell 0 has not reached a root after one snapshot round.
    expected: {labels: [0xfffffffe, 3, 3, 3], converged: [0], limit: [1], rounds: [1]}
  });

  // The 8m peak joins the 10m elder at height 2: prominence 6, while the elder is unbounded.
  const prominenceProps = {
    grid: grid(3),
    elevation: {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'elevation'}
    },
    output: {prominence: 'prominence', keySaddle: 'saddle', parentPeak: 'parent'}
  };
  fixtures.push({
    name: 'prominence-two-peaks',
    inputs: {elevation: floats([10, 2, 8])},
    gpu: () => new GPUTerrainPeakProminenceOperation(prominenceProps),
    cpu: () => new CPUTerrainPeakProminenceOperation(prominenceProps),
    expected: {prominence: [Infinity, 0, 6], saddle: [-Infinity, 2, 2], parent: [NONE, 0, 0]}
  });
  const prominenceOnlyProps = {...prominenceProps, output: {prominence: 'prominence'}};
  fixtures.push({
    name: 'prominence-two-peaks/without-secondary-outputs',
    inputs: {elevation: floats([10, 2, 8])},
    gpu: () => new GPUTerrainPeakProminenceOperation(prominenceOnlyProps),
    cpu: () => new CPUTerrainPeakProminenceOperation(prominenceOnlyProps),
    expected: {prominence: [Infinity, 0, 6]}
  });

  // Independently generated published-NOAA fixture, with represented float32 time parameters.
  const solar = POSITION_FIXTURE.cases[0];
  // Keep the original daylight rows and row 27 at night. A constant-one daylight
  // implementation must fail even when daylight is the sole requested output.
  const solarRows = [0, 1, 2, 27];
  const solarPositions = solarRows.flatMap(row => solar.positions.slice(row * 2, row * 2 + 2));
  const solarProps: SolarPositionOperationProps = {
    positions: 'positions',
    parameters: 'parameters',
    rowCount: solarRows.length,
    refraction: false,
    output: {azimuth: 'azimuth', altitude: 'altitude', daylight: 'daylight'}
  };
  fixtures.push({
    name: 'solar-NOAA-reference',
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from(solarPositions)
      },
      parameters: floats(solar.parameters)
    },
    gpu: () => new GPUSolarPositionOperation(solarProps),
    cpu: () => new CPUSolarPositionOperation(solarProps),
    expected: {
      azimuth: solarRows.map(row => solar.expected.azimuth[row]),
      altitude: solarRows.map(row => solar.expected.altitude[row]),
      daylight: solarRows.map(row => solar.expected.daylight[row])
    },
    angleBounds: solarRows.map(row => solar.angleBounds[row])
  });
  for (const port of ['azimuth', 'altitude', 'daylight'] as const) {
    const output: SolarPositionOperationProps['output'] =
      port === 'azimuth'
        ? {azimuth: port}
        : port === 'altitude'
          ? {altitude: port}
          : {daylight: port};
    const singleOutputProps: SolarPositionOperationProps = {...solarProps, output};
    fixtures.push({
      name: `solar-NOAA-reference/only-${port}`,
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.from(solarPositions)
        },
        parameters: floats(solar.parameters)
      },
      gpu: () => new GPUSolarPositionOperation(singleOutputProps),
      cpu: () => new CPUSolarPositionOperation(singleOutputProps),
      expected: {[port]: solarRows.map(row => solar.expected[port][row])},
      angleBounds: solarRows.map(row => solar.angleBounds[row])
    });
  }
  return fixtures;
}
