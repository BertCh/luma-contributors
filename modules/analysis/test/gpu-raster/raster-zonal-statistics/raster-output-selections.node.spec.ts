// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterJoin,
  type GPURasterJoinOutput
} from '../../../src/raster-analysis/polygon-rasterization/gpu-raster-join';
import {
  GPURasterZonalStatistics,
  type GPURasterZonalStatisticsOutput,
  type GPURasterZonalStatisticsProps
} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('W3-12 accepts minimal terminal choices and rejects empty nested selections in types', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const integer = createTransientView(graph, 'integer', 'uint32', 1);
  const float = createTransientView(graph, 'float', 'float32', 1);
  const joins: GPURasterJoinOutput[] = [
    {counts: integer},
    {sums: float},
    {boundaryCounts: integer},
    {unassignedBoundaryCount: integer},
    {outsideCount: integer},
    {pointZones: integer},
    {pointBoundaryMask: integer},
    {status: {incomplete: integer}}
  ];
  const zonal: GPURasterZonalStatisticsProps['output'][] = [
    {cellCounts: integer},
    {valueCounts: integer},
    {sums: float},
    {means: float},
    {minimums: float},
    {maximums: float},
    {weightSums: float},
    {variances: float},
    {populationVariances: float},
    {standardDeviations: float}
  ];
  // @ts-expect-error At least one terminal join output is required.
  const emptyJoin: GPURasterJoinOutput = {};
  // @ts-expect-error An empty nested status is not a terminal join output.
  const emptyJoinStatus: GPURasterJoinOutput = {status: {}};
  // @ts-expect-error At least one statistic is required.
  const emptyZonal: GPURasterZonalStatisticsProps['output'] = {};
  // @ts-expect-error Zonal status alone does not select a statistic.
  const statusOnlyZonal: GPURasterZonalStatisticsProps['output'] = {status: {incomplete: integer}};
  // @ts-expect-error Dictionary metadata alone does not select a statistic.
  const dictionaryOnlyZonal: GPURasterZonalStatisticsProps['output'] = {sourceZoneIds: integer};
  // Preserve the pre-existing six-column base contract used by recipe Partial/Required mappings.
  const partialRecipeOverride: GPURasterZonalStatisticsOutput = {};
  // @ts-expect-error The six-column recipe contract does not promise population moments.
  const unsupportedRecipePopulation: GPURasterZonalStatisticsOutput = {populationVariances: float};
  void partialRecipeOverride;
  void unsupportedRecipePopulation;
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const zones = createTransientView(graph, 'zones', 'uint32', 1);
  const joinProps = {
    grid,
    zones,
    zoneCount: 1,
    points: createTransientView(graph, 'points', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4)
  };
  for (const output of [emptyJoin, emptyJoinStatus])
    expect(() => new GPURasterJoin({...joinProps, output})).toThrow(/at least one output/);
  expect(
    () => new GPURasterJoin({...joinProps, output: {status: {incomplete: integer}}})
  ).not.toThrow();
  const zonalProps = {
    grid,
    zones,
    zoneCapacity: 1,
    values: {
      id: 'values',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: float}
    }
  };
  for (const output of [emptyZonal, statusOnlyZonal])
    expect(() => new GPURasterZonalStatistics({...zonalProps, output})).toThrow(
      /at least one output/
    );
  expect(joins).toHaveLength(8);
  expect(zonal).toHaveLength(10);
  expect([
    emptyJoin,
    emptyJoinStatus,
    emptyZonal,
    statusOnlyZonal,
    dictionaryOnlyZonal
  ]).toHaveLength(5);
});
