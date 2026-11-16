// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  GPUAdaptiveBandwidthKernelDensity,
  GPUExpectationBasedSpaceTimeScan,
  GPUJacquezTest,
  GPULocalKnox,
  GPULocationChangePoints,
  GPUNearRepeatKnox,
  GPUPettittTest,
  GPUSpaceTimeCrossCorrelation,
  GPUSpaceTimeHawkes,
  GPUSpaceTimeKernelDensity,
  GPUSpaceTimeLocalMoran,
  GPUSpaceTimeRelativeRisk,
  GPU_LOCAL_KNOX_SUMMARY,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_NEAR_REPEAT_CELL_SUMMARY,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  getGPUAdaptiveBandwidthKernelDensityParameterValues,
  getGPUExpectationBasedScanParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpaceTimeHawkesParameterValues,
  getGPUSpaceTimeKernelDensityParameterValues,
  getGPUSpaceTimeParameterValues,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUSpaceTimeDiggleTest,
  GPUSpaceTimeKFunction,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {SPACETIME_BINDINGS} from '../src/analysis/catalog-spacetime-bindings';
import {getKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {planKeplerResultPresentation} from '../src/presentation/result-presentation';
import {withCatalogGpuSession, type CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {
  SPACETIME_SUBSECOND_TEST_POINTS,
  SPACETIME_TEST_PARAMETERS,
  SPACETIME_TEST_POINTS
} from './catalog-spacetime-fixture';

// Independent host packing for this fixture: six locations, eight one-minute periods.
// Direct contributors below never invoke the binding dispatchers or their preparation helpers.
const LOCATION_POSITIONS = Float32Array.from([0, 0, 2, 0, 4, 0, 0, 2, 2, 2, 4, 2]);
const EVENT_POSITIONS = Float32Array.from(
  Array.from({length: 6}, (_, cell) =>
    Array.from({length: 8}, () => [
      LOCATION_POSITIONS[cell * 2],
      LOCATION_POSITIONS[cell * 2 + 1]
    ]).flat()
  ).flat()
);
const TIMES = Float32Array.from(Array.from({length: 48}, (_, row) => (row % 8) * 60));
const FIRST = Float32Array.from(
  Array.from(
    {length: 48},
    (_, row) => (Math.floor(row / 8) + 1) * (row % 8 < 4 ? 1 : 3) + (row % 2)
  )
);
const SECOND = Float32Array.from(
  Array.from(
    {length: 48},
    (_, row) => (Math.floor(row / 8) + 1) * (row % 8 < 3 ? 1 : 3) + ((row + 1) % 2)
  )
);
const BASELINE = Float32Array.from(Array.from({length: 48}, (_, row) => Math.floor(row / 8) + 1));
const GRID = makeGPURasterMetadata({
  width: 2,
  height: 2,
  cellSize: [4, 3],
  origin: [-2, -2],
  rowDirection: 'north'
});
const expand = (values: ArrayLike<number>): number[] =>
  Array.from({length: 48}, (_, row) => values[Math.floor(row / 8)]);

type DirectResult = {
  tableId: string;
  columns: Record<string, ArrayLike<number>>;
  summary?: Record<string, number>;
};

function directWeights(
  session: CatalogGpuSession,
  positions: Float32Array,
  radius = Infinity,
  nearest?: number
): GPUSpatialWeights {
  const rows = positions.length / 2;
  const offsets = [0];
  const entries: number[] = [];
  const distances: number[] = [];
  for (let row = 0; row < rows; row++) {
    const others: {row: number; distance: number}[] = [];
    for (let other = 0; other < rows; other++)
      if (other !== row) {
        const distance = Math.sqrt(
          (positions[row * 2] - positions[other * 2]) ** 2 +
            (positions[row * 2 + 1] - positions[other * 2 + 1]) ** 2
        );
        if (distance <= radius) others.push({row: other, distance});
      }
    others.sort((first, second) => first.distance - second.distance || first.row - second.row);
    const selected = others
      .slice(0, nearest ?? others.length)
      .sort((first, second) => first.row - second.row);
    for (const other of selected) {
      entries.push(other.row);
      distances.push(other.distance);
    }
    offsets.push(entries.length);
  }
  return {
    offsets: session.input('offsets', Uint32Array.from(offsets), 'uint32'),
    neighbors: session.input('neighbors', Uint32Array.from(entries), 'uint32'),
    distances: session.input('distances', Float32Array.from(distances), 'float32'),
    weights: session.input('weights', new Float32Array(entries.length).fill(1), 'float32')
  };
}

async function runDirect(device: Device, operationId: string): Promise<DirectResult> {
  return withCatalogGpuSession(
    device,
    `direct-${operationId}`,
    async (session): Promise<DirectResult> => {
      if (
        [
          'space-time-kernel-density',
          'space-time-relative-risk',
          'adaptive-bandwidth-kernel-density'
        ].includes(operationId)
      ) {
        const adaptive = operationId === 'adaptive-bandwidth-kernel-density';
        const output = session.output('values', 'float32', adaptive ? 4 : 8);
        const common = {
          grid: GRID,
          indexGridSize: [4, 4] as const,
          indexBounds: [-2, -2, 6, 4] as const
        };
        if (adaptive) {
          const bandwidths = session.output('bandwidths', 'float32', 48);
          const pilot = session.output('pilot', 'float32', 48);
          session.graph.add(
            new GPUAdaptiveBandwidthKernelDensity({
              ...common,
              positions: session.input('positions', EVENT_POSITIONS, 'float32x2'),
              parameters: session.input(
                'parameters',
                getGPUAdaptiveBandwidthKernelDensityParameterValues(
                  {bandwidth: 2, sensitivity: 0.5},
                  GRID
                ),
                'float32'
              ),
              bandwidthFactorLimit: 4,
              normalization: 'intensity',
              output: {
                values: output.view,
                pointBandwidths: bandwidths.view,
                pilotDensities: pilot.view
              }
            })
          );
          session.run();
          const directDensity = await session.readFloat32(output);
          // Check both result tables; attach the grid values as separately checked summary scalars.
          return {
            tableId: 'points',
            columns: {
              bandwidth: await session.readFloat32(bandwidths),
              pilot_density: await session.readFloat32(pilot)
            },
            summary: Object.fromEntries(
              Array.from(directDensity, (value, index) => [`grid_${index}`, value])
            )
          };
        }
        const parameters = session.input(
          'parameters',
          getGPUSpaceTimeKernelDensityParameterValues(
            {
              bandwidth: 2,
              temporalBandwidth: 120,
              firstSliceTime: 0,
              sliceInterval: 420,
              timeWindow: [-1, 421]
            },
            GRID,
            2
          ),
          'float32'
        );
        const temporal = {
          ...common,
          parameters,
          sliceCount: 2,
          spatialEdgeCorrection: 'none' as const,
          temporalEdgeCorrection: 'none' as const
        };
        if (operationId === 'space-time-relative-risk') {
          const cases: number[] = [];
          const controls: number[] = [];
          for (let row = 0; row < 48; row++)
            ((Math.floor(row / 8) + (row % 8)) % 2 ? cases : controls).push(row);
          const pickPositions = (indices: number[]) =>
            Float32Array.from(
              indices.flatMap(row => [EVENT_POSITIONS[row * 2], EVENT_POSITIONS[row * 2 + 1]])
            );
          const caseDensity = session.output('case-density', 'float32', 8);
          const controlDensity = session.output('control-density', 'float32', 8);
          session.graph.add(
            new GPUSpaceTimeRelativeRisk({
              ...temporal,
              casePositions: session.input('case-positions', pickPositions(cases), 'float32x2'),
              caseTimes: session.input(
                'case-times',
                Float32Array.from(cases.map(row => TIMES[row])),
                'float32'
              ),
              controlPositions: session.input(
                'control-positions',
                pickPositions(controls),
                'float32x2'
              ),
              controlTimes: session.input(
                'control-times',
                Float32Array.from(controls.map(row => TIMES[row])),
                'float32'
              ),
              scale: 'ratio',
              output: {
                values: output.view,
                caseDensity: caseDensity.view,
                controlDensity: controlDensity.view
              }
            })
          );
          session.run();
          return {
            tableId: 'grid',
            columns: {
              value: await session.readFloat32(output),
              case_density: await session.readFloat32(caseDensity),
              control_density: await session.readFloat32(controlDensity)
            }
          };
        }
        session.graph.add(
          new GPUSpaceTimeKernelDensity({
            ...temporal,
            positions: session.input('positions', EVENT_POSITIONS, 'float32x2'),
            times: session.input('times', TIMES, 'float32'),
            normalization: 'intensity',
            output: {values: output.view}
          })
        );
        session.run();
        return {tableId: 'grid', columns: {value: await session.readFloat32(output)}};
      }
      if (operationId === 'space-time-hawkes') {
        const estimates = session.output('estimates', 'float32', 4);
        const probability = session.output('probability', 'float32', 48);
        session.graph.add(
          new GPUSpaceTimeHawkes({
            positions: session.input('positions', EVENT_POSITIONS, 'float32x2'),
            times: session.input('times', TIMES, 'float32'),
            parameters: session.input(
              'parameters',
              getGPUSpaceTimeHawkesParameterValues({
                windowArea: 80,
                windowDuration: 420,
                maximumLag: 120,
                maximumDistance: 3,
                initialBackground: 48 / (2 * 80 * 420),
                initialBranchingRatio: 0.5,
                initialDecayRate: 1 / 120,
                initialSpatialStandardDeviation: 3
              }),
              'float32'
            ),
            maximumIterations: 3,
            maximumParents: 8,
            output: {estimates: estimates.view, backgroundProbability: probability.view}
          })
        );
        session.run();
        const fitted = await session.readFloat32(estimates);
        return {
          tableId: 'points',
          columns: {background_probability: await session.readFloat32(probability)},
          summary: {
            backgroundRate: fitted[0],
            branchingRatio: fitted[1],
            decayRate: fitted[2],
            spatialStandardDeviation: fitted[3]
          }
        };
      }
      if (
        operationId === 'jacquez-test' ||
        operationId === 'local-knox' ||
        operationId === 'near-repeat-knox'
      ) {
        const local = operationId === 'local-knox';
        const nearRepeat = operationId === 'near-repeat-knox';
        const pairs = directWeights(
          session,
          EVENT_POSITIONS,
          local || nearRepeat ? 3 : Infinity,
          local || nearRepeat ? undefined : 2
        );
        const common = {
          times: session.input('times', TIMES, 'float32'),
          parameters: session.input(
            'parameters',
            getGPUSpaceTimeParameterValues({
              seed: 7,
              permutationCount: 3,
              timeThreshold: local || nearRepeat ? 120 : 0
            }),
            'uint32'
          ),
          maximumPermutations: 3
        };
        const output = session.output(
          'summary',
          'float32',
          local ? 48 * 6 : nearRepeat ? 4 * 8 : 12
        );
        if (local)
          session.graph.add(new GPULocalKnox({...common, pairs, output: {summary: output.view}}));
        else if (nearRepeat)
          session.graph.add(
            new GPUNearRepeatKnox({
              ...common,
              pairs,
              distanceEdges: [0, 1.5, 3],
              timeEdges: [0, 60, 120],
              output: {summary: output.view, counts: session.output('counts', 'uint32', 16).view}
            })
          );
        else
          session.graph.add(
            new GPUJacquezTest({
              ...common,
              neighbors: pairs,
              k: 2,
              output: {
                summary: output.view,
                statistics: session.output('statistics', 'uint32', 4).view
              }
            })
          );
        session.run();
        const result = await session.readFloat32(output);
        if (!local && !nearRepeat)
          return {
            tableId: 'statistics',
            columns: {},
            summary: Object.fromEntries(
              Object.entries(GPU_SPACE_TIME_SUMMARY).map(([name, index]) => [name, result[index]])
            )
          };
        const fields = local ? GPU_LOCAL_KNOX_SUMMARY : GPU_NEAR_REPEAT_CELL_SUMMARY;
        const stride = local ? GPU_LOCAL_KNOX_SUMMARY_LENGTH : GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH;
        return {
          tableId: local ? 'points' : 'bands',
          columns: Object.fromEntries(
            Object.entries(fields).map(([name, offset]) => [
              name,
              Float32Array.from({length: local ? 48 : 4}, (_, row) => result[row * stride + offset])
            ])
          )
        };
      }
      if (operationId === 'space-time-k-function' || operationId === 'space-time-diggle-test') {
        const common = {
          positions: session.input('positions', EVENT_POSITIONS, 'float32x2'),
          times: session.input('times', TIMES, 'float32'),
          parameters: session.input(
            'parameters',
            getGPUSpaceTimeKFunctionParameterValues({
              bounds: [-2, -2, 6, 4],
              maximumDistance: 3,
              maximumTimeLag: 120,
              timeWindow: [-1, 421]
            }),
            'float32'
          ),
          gridSize: [4, 4] as const,
          distanceClassCount: 2,
          timeClassCount: 2,
          edgeCorrection: 'none' as const
        };
        const outputK = session.output('k', 'float32', 4);
        const outputD = session.output('d', 'float32', 4);
        const summary = session.output('summary', 'float32', 11);
        if (operationId === 'space-time-diggle-test')
          session.graph.add(
            new GPUSpaceTimeDiggleTest({
              ...common,
              permutationCount: 3,
              randomParameters: session.input(
                'random',
                getGPUSpaceTimeDiggleRandomParameterValues(7),
                'uint32'
              ),
              output: {
                k: outputK.view,
                d: outputD.view,
                statistics: session.output('statistics', 'float32', 4).view,
                summary: summary.view
              }
            })
          );
        else
          session.graph.add(
            new GPUSpaceTimeKFunction({...common, output: {k: outputK.view, d: outputD.view}})
          );
        session.run();
        const summaryValues =
          operationId === 'space-time-diggle-test' ? await session.readFloat32(summary) : undefined;
        return {
          tableId: 'bands',
          columns: {k: await session.readFloat32(outputK), d: await session.readFloat32(outputD)},
          ...(summaryValues
            ? {
                summary: Object.fromEntries(
                  Object.entries(GPU_SPACE_TIME_DIGGLE_SUMMARY).map(([name, index]) => [
                    name,
                    summaryValues[index]
                  ])
                )
              }
            : {})
        };
      }
      if (operationId === 'pettitt-test') {
        const statistic = session.output('statistic', 'uint32', 6);
        const change = session.output('change', 'uint32', 6);
        const pValues = session.output('p-values', 'float32', 6);
        session.graph.add(
          new GPUPettittTest({
            values: session.input('values', FIRST, 'float32'),
            layout: 'cell-major',
            cellCount: 6,
            sliceCount: 8,
            output: {statistic: statistic.view, changeSlice: change.view, pValues: pValues.view}
          })
        );
        session.run();
        return {
          tableId: 'points',
          columns: {
            statistic: expand(await session.readUint32(statistic)),
            change_slice: expand(await session.readUint32(change)),
            p_value: expand(await session.readFloat32(pValues))
          }
        };
      }
      if (operationId === 'location-change-points') {
        const count = session.output('count', 'uint32', 6);
        const changes = session.output('changes', 'uint32', 6 * 32);
        const cost = session.output('cost', 'float32', 6);
        session.graph.add(
          new GPULocationChangePoints({
            values: session.input('values', FIRST, 'float32'),
            layout: 'cell-major',
            cellCount: 6,
            sliceCount: 8,
            cost: 'mean',
            maximumChangePoints: 32,
            parameters: session.input('parameters', Float32Array.from([10, 2]), 'float32'),
            output: {
              changePointCount: count.view,
              changePoints: changes.view,
              optimalCost: cost.view
            }
          })
        );
        session.run();
        const changeCount = await session.readUint32(count);
        const changePoints = await session.readUint32(changes);
        const segment = Array.from(
          {length: 48},
          (_, row) =>
            Array.from(
              changePoints.slice(
                Math.floor(row / 8) * 32,
                Math.floor(row / 8) * 32 + changeCount[Math.floor(row / 8)]
              )
            ).filter(period => period <= row % 8).length
        );
        const flags = segment.map((value, row) =>
          Number(row % 8 > 0 && value !== segment[row - 1])
        );
        return {
          tableId: 'points',
          columns: {
            change_count: expand(changeCount),
            optimal_cost: expand(await session.readFloat32(cost)),
            segment,
            is_change_point: flags
          }
        };
      }
      if (operationId === 'space-time-cross-correlation') {
        const correlations = session.output('correlations', 'float32', 6);
        const pValues = session.output('p-values', 'float32', 6);
        session.graph.add(
          new GPUSpaceTimeCrossCorrelation({
            first: session.input('first', FIRST, 'float32'),
            second: session.input('second', SECOND, 'float32'),
            layout: 'cell-major',
            cellCount: 6,
            periodCount: 8,
            lags: [1],
            multipleTesting: 'none',
            output: {correlations: correlations.view, pValues: pValues.view}
          })
        );
        session.run();
        return {
          tableId: 'points',
          columns: {
            correlation: expand(await session.readFloat32(correlations)),
            p_value: expand(await session.readFloat32(pValues))
          }
        };
      }
      if (operationId === 'space-time-local-moran') {
        const observed = session.output('observed', 'float32', 6);
        const pValues = session.output('p-values', 'float32', 6);
        session.graph.add(
          new GPUSpaceTimeLocalMoran({
            weights: directWeights(session, LOCATION_POSITIONS, Infinity, 2),
            cube: session.input('cube', FIRST, 'float32'),
            layout: 'cell-major',
            periodCount: 8,
            period: 7,
            lag: 1,
            variant: 'lagged',
            direction: 'inward',
            parameters: session.input(
              'parameters',
              getGPUPermutationParameterValues(
                {seed: 7, permutationCount: 3, significanceLevel: 0.05},
                {maximumPermutations: 3}
              ),
              'uint32'
            ),
            maximumPermutations: 3,
            maximumNeighbors: 32,
            alternative: 'two-sided',
            multipleTesting: 'none',
            output: {
              observed: observed.view,
              pseudoPValues: pValues.view,
              exceedances: session.output('exceedances', 'uint32', 6).view,
              status: {overflow: session.output('overflow', 'uint32', 1).view}
            }
          })
        );
        session.run();
        const statistics = await session.readFloat32(observed);
        const probabilities = await session.readFloat32(pValues);
        return {
          tableId: 'points',
          columns: {
            observed: Float32Array.from({length: 48}, (_, row) =>
              row % 8 === 7 ? statistics[Math.floor(row / 8)] : Number.NaN
            ),
            p_value: Float32Array.from({length: 48}, (_, row) =>
              row % 8 === 7 ? probabilities[Math.floor(row / 8)] : Number.NaN
            )
          }
        };
      }
      if (operationId === 'expectation-based-space-time-scan') {
        const zones = session.output('zones', 'uint32', 2);
        const statistics = session.output('cluster', 'float32', 4);
        const summary = session.output('summary', 'uint32', 5);
        session.graph.add(
          new GPUExpectationBasedSpaceTimeScan({
            positions: session.input('positions', LOCATION_POSITIONS, 'float32x2'),
            cases: session.input('cases', Uint32Array.from(FIRST), 'uint32'),
            baseline: session.input('baseline', BASELINE, 'float32'),
            timeBuckets: 8,
            maximumNeighbors: 2,
            maximumWindowLength: 3,
            maximumReplicates: 3,
            parameters: session.input(
              'parameters',
              getGPUExpectationBasedScanParameterValues({
                seed: 7,
                replicateCount: 3,
                maximumWindowLength: 3
              }),
              'uint32'
            ),
            output: {
              clusterZones: zones.view,
              clusterStatistics: statistics.view,
              summary: summary.view,
              statistics: session.output('statistics', 'float32', 4).view
            }
          })
        );
        session.run();
        const selected = new Set(await session.readUint32(zones));
        const scanSummary = await session.readUint32(summary);
        const cluster = await session.readFloat32(statistics);
        return {
          tableId: 'points',
          columns: {
            cluster_member: Array.from({length: 48}, (_, row) =>
              Number(selected.has(Math.floor(row / 8)) && row % 8 >= 8 - scanSummary[2])
            )
          },
          summary: {
            logLikelihoodRatio: cluster[0],
            observedCases: cluster[1],
            expectedCases: cluster[2],
            pValue: cluster[3],
            windowLength: scanSummary[2]
          }
        };
      }
      throw new Error(`Missing direct contributor oracle for ${operationId}`);
    }
  );
}

function compareValue(actual: unknown, expected: number, label: string): void {
  if (!Number.isFinite(expected) || expected === 0xffffffff) expect(actual, label).toBeNull();
  else {
    expect(typeof actual, label).toBe('number');
    expect(Math.abs(Number(actual) - expected), label).toBeLessThanOrEqual(
      Math.max(1e-6, Math.abs(expected) * 2e-5)
    );
  }
}

for (const binding of SPACETIME_BINDINGS)
  test(`${binding.id}: GPU binding matches ${binding.contributorNames[0]} called directly`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const raw = Object.fromEntries(
      binding.parameters.map(parameter => [
        parameter.name,
        SPACETIME_TEST_PARAMETERS[parameter.name] ?? parameter.defaultValue
      ])
    );
    const parameters = normalizeCatalogParameters(binding, raw);
    const actual = await runCatalogOperationAsync(
      binding,
      parameters,
      {points: SPACETIME_TEST_POINTS},
      {device}
    );
    expect(actual.engine).toBe('gpu');
    expect(actual.dispatchedContributors).toEqual(binding.gpuContributorNames);
    const expected = await runDirect(device, binding.id);
    const table = actual.tables.find(candidate => candidate.id === expected.tableId)!;
    expect(table).toBeDefined();
    for (const [name, expectedValues] of Object.entries(expected.columns)) {
      expect(table.columns[name].length, name).toBe(expectedValues.length);
      Array.from(expectedValues).forEach((value, row) =>
        compareValue(table.columns[name][row], value, `${name} row ${row}`)
      );
    }
    for (const [name, value] of Object.entries(expected.summary ?? {})) {
      if (name.startsWith('grid_'))
        compareValue(
          actual.tables.find(candidate => candidate.id === 'grid')!.columns['value'][
            Number(name.slice(5))
          ],
          value,
          name
        );
      else if (!Number.isFinite(value)) expect(actual.summary[name]).toBe('undefined');
      else compareValue(actual.summary[name], value, name);
    }
    for (const output of binding.outputs) {
      const result = actual.tables.find(candidate => candidate.id === output.tableId)!;
      expect(result).toBeDefined();
      expect(result.rowCount).toBeLessThanOrEqual(output.maximumRows);
      for (const identity of output.identityColumns) expect(result.columns[identity]).toBeDefined();
    }
    const descriptor = getKeplerOperationDescriptor(binding.id);
    if (!descriptor) throw new Error(`${binding.id}: missing hand-written space-time descriptor`);
    for (const output of descriptor.outputs) {
      const result = actual.tables.find(candidate => candidate.id === output.tableId);
      if (!result) throw new Error(`${binding.id}: missing output table ${output.tableId}`);
      if (!output.presentation) continue;
      const column = output.presentation.column;
      if (!result.columns[column])
        throw new Error(
          `${binding.id}.${output.tableId}: presentation column ${column} is absent; available columns: ${Object.keys(result.columns).join(', ')}`
        );
      const presentation = planKeplerResultPresentation(result, output);
      if (
        !presentation ||
        presentation.column !== column ||
        (!presentation.legend.length && !presentation.ranges.length)
      )
        throw new Error(
          `${binding.id}.${output.tableId}: no usable presentation/legend for ${column}`
        );
      expect(presentation.rowCount).toBe(result.rowCount);
      const classes = presentation.legend.length ? presentation.legend : presentation.ranges;
      expect(
        classes.reduce((count, entry) => count + entry.count, 0) + presentation.missingCount,
        `${binding.id}.${column}: legend accounts for all rows`
      ).toBe(result.rowCount);
    }
    const pointTable = actual.tables.find(candidate => candidate.id === 'points');
    if (pointTable) expect(pointTable.geometryColumn).toBe('geometry');
    if (pointTable)
      expect(pointTable.columns['source_id']).toEqual(
        SPACETIME_TEST_POINTS.features.map(feature => String(feature.id))
      );
  });

for (const operationId of ['space-time-kernel-density', 'space-time-relative-risk'])
  test(`${operationId}: subsecond slices match the contributor at the observed endpoints`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const binding = SPACETIME_BINDINGS.find(candidate => candidate.id === operationId)!;
    const parameters = normalizeCatalogParameters(binding, {
      coordinateSystem: 'planar',
      gridSize: 1,
      sliceCount: 2,
      spatialBandwidth: 1,
      temporalBandwidth: 0.05
    });
    const actual = await runCatalogOperationAsync(
      binding,
      parameters,
      {points: SPACETIME_SUBSECOND_TEST_POINTS},
      {device}
    );
    expect(actual.tables[0].columns['time']).toEqual([1700000000000, 1700000000100]);
    const expected = await withCatalogGpuSession(
      device,
      `direct-subsecond-${operationId}`,
      async session => {
        const grid = makeGPURasterMetadata({
          width: 1,
          height: 1,
          cellSize: [3, 3],
          origin: [-1, -1],
          rowDirection: 'north'
        });
        const output = session.output('values', 'float32', 2);
        const common = {
          grid,
          indexGridSize: [4, 4] as const,
          indexBounds: [-1, -1, 2, 2] as const,
          sliceCount: 2,
          spatialEdgeCorrection: 'none' as const,
          temporalEdgeCorrection: 'none' as const,
          parameters: session.input(
            'parameters',
            getGPUSpaceTimeKernelDensityParameterValues(
              {
                bandwidth: 1,
                temporalBandwidth: 0.05,
                firstSliceTime: 0,
                sliceInterval: 0.1,
                timeWindow: [-1, 1.1]
              },
              grid,
              2
            ),
            'float32'
          ),
          output: {values: output.view}
        };
        if (operationId === 'space-time-kernel-density')
          session.graph.add(
            new GPUSpaceTimeKernelDensity({
              ...common,
              positions: session.input('positions', Float32Array.from([0, 0, 1, 1]), 'float32x2'),
              times: session.input('times', Float32Array.from([0, 0.1]), 'float32'),
              normalization: 'intensity'
            })
          );
        else
          session.graph.add(
            new GPUSpaceTimeRelativeRisk({
              ...common,
              casePositions: session.input(
                'case-positions',
                Float32Array.from([1, 1]),
                'float32x2'
              ),
              caseTimes: session.input('case-times', Float32Array.from([0.1]), 'float32'),
              controlPositions: session.input(
                'control-positions',
                Float32Array.from([0, 0]),
                'float32x2'
              ),
              controlTimes: session.input('control-times', Float32Array.from([0]), 'float32'),
              scale: 'ratio'
            })
          );
        session.run();
        return session.readFloat32(output);
      }
    );
    Array.from(expected).forEach((value, index) =>
      compareValue(actual.tables[0].columns['value'][index], value, `subsecond slice ${index}`)
    );
  });

test('near-repeat Knox retains the closed Float32 radius boundary and matches the direct contributor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = structuredClone(SPACETIME_SUBSECOND_TEST_POINTS);
  points.features[1].geometry = {type: 'Point', coordinates: [0.6, 0.8]};
  const binding = SPACETIME_BINDINGS.find(candidate => candidate.id === 'near-repeat-knox')!;
  const parameters = normalizeCatalogParameters(binding, {
    coordinateSystem: 'planar',
    spatialThreshold: 1,
    temporalThreshold: 1,
    permutations: 3,
    seed: 1,
    bandCount: 1
  });
  const actual = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  expect(actual.tables[0].columns['observed']).toEqual([1]);
  const expected = await withCatalogGpuSession(device, 'direct-closed-radius', async session => {
    const output = session.output('summary', 'float32', GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH);
    session.graph.add(
      new GPUNearRepeatKnox({
        pairs: {
          offsets: session.input('offsets', Uint32Array.from([0, 1, 2]), 'uint32'),
          neighbors: session.input('neighbors', Uint32Array.from([1, 0]), 'uint32'),
          weights: session.input('weights', Float32Array.from([1, 1]), 'float32'),
          distances: session.input('distances', Float32Array.from([1, 1]), 'float32')
        },
        times: session.input('times', Float32Array.from([0, 0.1]), 'float32'),
        parameters: session.input(
          'parameters',
          getGPUSpaceTimeParameterValues({seed: 1, permutationCount: 3, timeThreshold: 1}),
          'uint32'
        ),
        maximumPermutations: 3,
        distanceEdges: [0, 1],
        timeEdges: [0, 1],
        output: {summary: output.view, counts: session.output('counts', 'uint32', 4).view}
      })
    );
    session.run();
    return session.readFloat32(output);
  });
  for (const [name, offset] of Object.entries(GPU_NEAR_REPEAT_CELL_SUMMARY))
    compareValue(actual.tables[0].columns[name][0], expected[offset], name);
});
