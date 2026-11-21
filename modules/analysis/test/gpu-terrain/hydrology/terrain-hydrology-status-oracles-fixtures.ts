// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUTerrainFlowOperation} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-operation';
import {CPUTerrainFlowOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import type {TerrainStatusOracle} from '../terrain-status-oracles-test-utils';
import {GPUTerrainStreamOrderOperation} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order-operation';
import {CPUTerrainStreamOrderOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation';
import {GPUTerrainWatershedsOperation} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';
import {CPUTerrainWatershedsOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation';

export function getTerrainHydrologyStatusOracles(): TerrainStatusOracle[] {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, -1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const props = {
    grid,
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
      accumulation: 'accumulation',
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
  // A singleton is a boundary outlet, with no flat dependency or donor. Thus all three
  // convergence words are 1 by inspection at any positive budget. GPU confirmation counts
  // are 1 fill + 3 independent flat loops + 1 Euler-tour loop. Explicit fill/flat budgets
  // execute the CPU twin's same confirming loops. Its exact DAG
  // algorithm executes no relaxation rounds, so the CPU counters are 1/3/0.
  const oracles: TerrainStatusOracle[] = [
    {
      name: 'flow-singleton-independent-flags-and-confirmations',
      gpu: () => new GPUTerrainFlowOperation(props),
      cpu: () => new CPUTerrainFlowOperation(props),
      frames: [7, 9].map(value => ({
        inputs: {
          elevation: {format: 'float32', values: new Float32Array([value])},
          parameters: {
            format: 'float32',
            values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, grid)
          }
        },
        expected: {
          fillConverged: [1],
          flatsConverged: [1],
          accumulationConverged: [1],
          fillRounds: [1],
          flatRounds: [3],
          accumulationRounds: [0]
        },
        gpuExpected: {accumulationRounds: [1]}
      }))
    }
  ];
  // The sole interior flat (cell 4) gains its outlet hop in round one, which
  // changed the field. Two rounds include the unchanged confirmation. Only the
  // eastward tie-first receiver 4→5 adds a donor to another cell on completion.
  // Bind accumulationConverged alone: consumers need not request the upstream flag.
  for (const maximumFlatIterations of [1, 2]) {
    const flatProps = {
      ...props,
      grid: {...grid, width: 3, height: 3},
      fillDepressions: false,
      maximumFlatIterations,
      output: {
        accumulation: 'accumulation',
        status: {accumulationConverged: 'accumulationConverged'}
      }
    };
    oracles.push({
      name: `flow-accumulation-propagates-flat-budget-${maximumFlatIterations}`,
      gpu: () => new GPUTerrainFlowOperation(flatProps),
      cpu: () => new CPUTerrainFlowOperation(flatProps),
      frames: [true, false, true].map(plateau => ({
        inputs: {
          elevation: {
            format: 'float32',
            values: new Float32Array(
              plateau ? [10, 10, 10, 10, 10, 10, 10, 10, 10] : [3, 2, 1, 3, 2, 1, 3, 2, 1]
            )
          },
          parameters: {
            format: 'float32',
            values: getGPUTerrainFlowParameterValues({fillEpsilon: 0}, flatProps.grid)
          }
        },
        expected: {accumulationConverged: [plateau && maximumFlatIterations === 1 ? 0 : 1]},
        floatExpected: {
          accumulation: !plateau
            ? [1, 2, 3, 1, 2, 3, 1, 2, 3]
            : maximumFlatIterations === 1
              ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
              : [1, 1, 1, 1, 1, 2, 1, 1, 1]
        }
      }))
    });
  }
  // Boundary cells are seeded at 2; the centre starts at infinity. Round one
  // lowers it to max(0,2+0.25)=2.25, so only round two can confirm no change.
  for (const maximumFillIterations of [1, 2]) {
    const fillProps = {
      ...props,
      grid: {...grid, width: 3, height: 3},
      maximumFillIterations,
      resolveFlats: false,
      output: {
        filledElevation: 'filled',
        accumulation: 'accumulation',
        status: {
          fillConverged: 'fillConverged',
          fillIterationCount: 'fillRounds',
          accumulationConverged: 'accumulationConverged'
        }
      }
    };
    oracles.push({
      name: `flow-bowl-fill-budget-${maximumFillIterations}`,
      gpu: () => new GPUTerrainFlowOperation(fillProps),
      cpu: () => new CPUTerrainFlowOperation(fillProps),
      frames: [true, false, true].map(bowl => ({
        inputs: {
          elevation: {
            format: 'float32',
            values: new Float32Array([2, 2, 2, 2, bowl ? 0 : NaN, 2, 2, 2, 2])
          },
          parameters: {
            format: 'float32',
            values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, fillProps.grid)
          }
        },
        expected: {
          fillConverged: [bowl && maximumFillIterations === 1 ? 0 : 1],
          fillRounds: [bowl ? maximumFillIterations : 1],
          accumulationConverged: [bowl && maximumFillIterations === 1 ? 0 : 1]
        },
        floatExpected: {
          filled: !bowl
            ? [2, 2, 2, 2, NaN, 2, 2, 2, 2]
            : maximumFillIterations === 1
              ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
              : [2, 2, 2, 2, 2.25, 2, 2, 2, 2],
          accumulation: !bowl
            ? [1, 1, 1, 1, NaN, 1, 1, 1, 1]
            : maximumFillIterations === 1
              ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
              : [1, 1, 1, 1, 1, 2, 1, 1, 1]
        }
      }))
    });
  }
  // One tile row, 101 tiles wide. The only cheap escape for middle-row cell
  // c is c unit-cost steps left to column 0. All other exits have height 10000,
  // so its exact fill is c (except the right boundary's fixed 10000).
  // These end-to-end cases pin filled elevations and a sufficient budget, not the first
  // round of convergence: earlier line visits can advance vertical corridors too.
  // terrain-downward-sweep.node.spec.ts isolates down-phase visibility from a controlled state.
  for (const [direction, length, maximumFillIterations] of [
    ['right', 1616, 99],
    ['left', 1664, 100],
    ['down', 1664, 101],
    ['up', 1664, 102]
  ] as const) {
    const vertical = direction === 'down' || direction === 'up';
    const reverse = direction === 'left' || direction === 'up';
    const corridorGrid = {...grid, width: vertical ? 3 : length, height: vertical ? length : 3};
    const corridorProps = {
      ...props,
      grid: corridorGrid,
      maximumFillIterations,
      resolveFlats: false,
      output: {filledElevation: 'filled', status: {fillConverged: 'fillConverged'}}
    };
    // Rotations/reflections have the same unit-step escape cost. The longer
    // variants have 104 tiles, so the other three sweep phases are also necessary
    // at their caps: plain frozen-halo propagation would need 105 rounds.
    const getCorridorHeight = (cell: number, filled: boolean) => {
      const column = cell % corridorGrid.width,
        row = Math.floor(cell / corridorGrid.width);
      const coordinate = vertical ? row : column;
      const distance = reverse ? length - 1 - coordinate : coordinate;
      return (vertical ? column : row) === 1 && distance < length - 1
        ? filled
          ? distance
          : 0
        : 10000;
    };
    oracles.push({
      name: `flow-positive-epsilon-corridor-${direction}-cap-${maximumFillIterations}`,
      gpu: () => new GPUTerrainFlowOperation(corridorProps),
      cpu: () => new CPUTerrainFlowOperation(corridorProps),
      frames: [
        {
          inputs: {
            elevation: {
              format: 'float32',
              values: Float32Array.from({length: length * 3}, (_, cell) =>
                getCorridorHeight(cell, false)
              )
            },
            parameters: {
              format: 'float32',
              values: getGPUTerrainFlowParameterValues({fillEpsilon: 1}, corridorProps.grid)
            }
          },
          expected: {fillConverged: [1]},
          floatExpected: {
            filled: Array.from({length: length * 3}, (_, cell) => getCorridorHeight(cell, true))
          }
        }
      ]
    });
  }
  // Routing is unusable when filling has not converged, even if all three
  // subsequent flat relaxations happen to be locally stable. Bind no fill flag.
  for (const maximumFillIterations of [1, 2]) {
    const flatAfterFillProps = {
      ...props,
      grid: {...grid, width: 3, height: 3},
      maximumFillIterations,
      maximumFlatIterations: 2,
      output: {flowDirections: 'directions', status: {flatsConverged: 'flatsConverged'}}
    };
    oracles.push({
      name: `flow-flat-routing-propagates-fill-budget-${maximumFillIterations}`,
      gpu: () => new GPUTerrainFlowOperation(flatAfterFillProps),
      cpu: () => new CPUTerrainFlowOperation(flatAfterFillProps),
      frames: [true, false, true].map(bowl => ({
        inputs: {
          elevation: {
            format: 'float32',
            values: new Float32Array([2, 2, 2, 2, bowl ? 0 : NaN, 2, 2, 2, 2])
          },
          parameters: {
            format: 'float32',
            values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, flatAfterFillProps.grid)
          }
        },
        expected: {
          flatsConverged: [bowl && maximumFillIterations === 1 ? 0 : 1],
          directions: !bowl
            ? [0, 0, 0, 0, 0xffffffff, 0, 0, 0, 0]
            : maximumFillIterations === 1
              ? [
                  0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe,
                  0xfffffffe, 0xfffffffe, 0xfffffffe
                ]
              : [0, 0, 0, 0, 1, 0, 0, 0, 0]
        }
      }))
    });
  }
  const orderProps = {
    grid: {...grid, width: 4},
    flowDirections: 'directions',
    streams: 'streams',
    output: {streamOrder: 'order', status: {converged: 'converged', iterationLimitReached: 'limit'}}
  };
  // Cells 0/1 are a rootless receiver cycle, cell 2 is non-stream and cell 3 is an
  // outlet. Removing the two cycle edges resolves every stream independently.
  // Both the default GPU tour and the exact CPU queue have no iteration-limit condition.
  oracles.push({
    name: 'stream-order-cycle-outlets-and-limit-reset',
    gpu: () => new GPUTerrainStreamOrderOperation(orderProps),
    cpu: () => new CPUTerrainStreamOrderOperation(orderProps),
    frames: [true, false, true].map(cycle => ({
      inputs: {
        directions: {
          format: 'uint32',
          values: new Uint32Array(cycle ? [1, 16, 0, 0] : [0, 0, 0, 0])
        },
        streams: {format: 'uint32', values: new Uint32Array([1, 1, 0, 1])}
      },
      expected: {converged: [cycle ? 0 : 1], limit: [0]}
    }))
  });
  const watershedProps = {
    grid: {...grid, width: 3},
    flowDirections: 'directions',
    maximumIterations: 1,
    output: {
      labels: 'labels',
      status: {converged: 'converged', iterationLimitReached: 'limit', iterationCount: 'rounds'}
    }
  };
  // In 0→1→2, pointer 0 must change from 1 to 2 in round one. Labels already reach
  // terminal 2 but there has been no unchanged confirmation, so flags are 0/1.
  // Three independent outlets have no changing pointer and certify 1/0 in one round.
  oracles.push({
    name: 'watershed-one-round-change-versus-outlet-confirmation',
    gpu: () => new GPUTerrainWatershedsOperation(watershedProps),
    cpu: () => new CPUTerrainWatershedsOperation(watershedProps),
    frames: [true, false, true].map(chain => ({
      inputs: {
        directions: {format: 'uint32', values: new Uint32Array(chain ? [1, 1, 0] : [0, 0, 0])}
      },
      expected: {
        labels: chain ? [2, 2, 2] : [0, 1, 2],
        converged: [chain ? 0 : 1],
        limit: [chain ? 1 : 0],
        rounds: [1]
      }
    }))
  });
  // 0↔1 has no stop, although doubling [1,0,2] produces the false fixed points
  // [0,1,2]. Round two confirms those pointers unchanged, not drainage convergence.
  for (const [directionEncoding, cycleDirections] of [
    ['esri', [1, 16, 0]],
    ['whitebox', [2, 32, 0]],
    ['taudem', [1, 5, 0]]
  ] as const) {
    const cycleProps = {...watershedProps, maximumIterations: 2, directionEncoding};
    oracles.push({
      name: `watershed-two-cell-cycle-is-not-a-terminal-${directionEncoding}`,
      gpu: () => new GPUTerrainWatershedsOperation(cycleProps),
      cpu: () => new CPUTerrainWatershedsOperation(cycleProps),
      frames: [true, false, true].map(cycle => ({
        inputs: {
          directions: {
            format: 'uint32',
            values: new Uint32Array(cycle ? cycleDirections : [0, 0, 0])
          }
        },
        expected: {
          labels: cycle ? [0xfffffffe, 0xfffffffe, 2] : [0, 1, 2],
          converged: [cycle ? 0 : 1],
          limit: [0],
          rounds: [cycle ? 2 : 1]
        }
      }))
    });
    // Marking cell 0 cuts the same receiver cycle at a genuine stop. Cell 2
    // reaches no marker and remains NONE, which is complete rather than unresolved.
    const markedProps = {...cycleProps, pourPoints: 'pourPoints'};
    oracles.push({
      name: `watershed-pour-point-cuts-two-cell-cycle-${directionEncoding}`,
      gpu: () => new GPUTerrainWatershedsOperation(markedProps),
      cpu: () => new CPUTerrainWatershedsOperation(markedProps),
      frames: [
        {
          inputs: {
            directions: {format: 'uint32', values: new Uint32Array(cycleDirections)},
            pourPoints: {format: 'uint32', values: new Uint32Array([0])}
          },
          expected: {labels: [0, 0, 0xffffffff], converged: [1], limit: [0], rounds: [1]}
        }
      ]
    });
  }
  for (const maximumIterations of [undefined, 2]) {
    const unresolvedProps = {
      ...watershedProps,
      grid: {...grid, width: 5},
      maximumIterations
    };
    // 0→1 ends at an unknown direction, not an outlet. Invalid cell 2 stays
    // NONE; 3→4 is independently complete. All pointers are stable in round one.
    oracles.push({
      name: `watershed-unresolved-direction-clears-convergence-${maximumIterations ?? 'default'}`,
      gpu: () => new GPUTerrainWatershedsOperation(unresolvedProps),
      cpu: () => new CPUTerrainWatershedsOperation(unresolvedProps),
      frames: [true, false, true].map(unknown => ({
        inputs: {
          directions: {
            format: 'uint32',
            values: new Uint32Array([1, unknown ? 0xfffffffe : 0, 0xffffffff, 1, 0])
          }
        },
        expected: {
          labels: unknown ? [0xfffffffe, 0xfffffffe, 0xffffffff, 4, 4] : [1, 1, 0xffffffff, 4, 4],
          converged: [unknown ? 0 : 1],
          limit: [0],
          rounds: [maximumIterations === undefined ? 0 : 1]
        },
        gpuExpected: {rounds: [1]}
      }))
    });
    const interceptedProps = {...unresolvedProps, pourPoints: 'points'};
    oracles.push({
      name: `watershed-pour-point-intercepts-unresolved-direction-${maximumIterations ?? 'default'}`,
      gpu: () => new GPUTerrainWatershedsOperation(interceptedProps),
      cpu: () => new CPUTerrainWatershedsOperation(interceptedProps),
      frames: [false, true, false].map(intercept => ({
        inputs: {
          directions: {
            format: 'uint32',
            values: new Uint32Array([1, 0xfffffffe, 0xffffffff, 1, 0])
          },
          points: {format: 'uint32', values: new Uint32Array([intercept ? 1 : 99])}
        },
        expected: {
          labels: intercept
            ? [0, 0, 0xffffffff, 0xffffffff, 0xffffffff]
            : [0xfffffffe, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff],
          converged: [intercept ? 1 : 0],
          limit: [0],
          rounds: [maximumIterations === undefined ? 0 : 1]
        },
        gpuExpected: {rounds: [1]}
      }))
    });
  }
  return oracles;
}
