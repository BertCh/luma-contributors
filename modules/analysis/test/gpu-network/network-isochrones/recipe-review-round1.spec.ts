// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createIsochroneOracleHarness, expectOracleValues} from './isochrones-output-oracle-utils';
import {createRecipeCellReviewProps} from './recipe-cell-review-fixture';

// The retired `outputs` spelling is rejected at construction (decision: one name, no aliases).
it.each([
  'output'
] as const)('recovers cell stages, polygon destinations and demand matches after table loss (%s, STA-08)', async spelling => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const props = createRecipeCellReviewProps(harness);
  props.isochrones!.joinDemand = {candidateCapacity: 8};
  props[spelling] = {
    isochroneDemand: {
      pointFeatureIds: harness.output('demandMatches', 'uint32', 1),
      status: {
        overflow: harness.output('joinOverflow', 'uint32', 1),
        uncertainCount: harness.output('joinUncertain', 'uint32', 1),
        candidateOverflow: harness.output('joinCandidateOverflow', 'uint32', 1)
      }
    }
  };
  // B4: the poisoned polygonCount destination is inherited, yet must have the actual writer.
  const ringStatus = props.isochrones!.cellOutline!.rings!.output.status;
  const polygonCount = ringStatus.polygonCount!;
  delete ringStatus.polygonCount;
  Object.setPrototypeOf(ringStatus, {
    get polygonCount() {
      return polygonCount;
    }
  });
  addDriveTimeCatchmentRecipe(harness.graph, props);
  const result = await harness.run(undefined, first => {
    expectOracleValues(first['tableOverflow'], [1]);
    expectOracleValues(first['tableCandidateOverflow'], [0]);
    expectOracleValues(first['outlineOverflow'], [0]);
    expectOracleValues(first['outlineCandidateOverflow'], [1]);
    expectOracleValues(first['ringOverflow'], [0]);
    expectOracleValues(first['ringCandidateOverflow'], [1]);
    expectOracleValues(first['ringComplete'], [0]);
    expectOracleValues(first['polygonCount'], [0]);
    expectOracleValues(first['tableSumOverflow'], [0]);
    expectOracleValues(first['demandMatches'], [0xffffffff]);
    expectOracleValues(first['joinOverflow'], [0]);
    expectOracleValues(first['joinUncertain'], [0]);
    expectOracleValues(first['joinCandidateOverflow'], [1]);
    harness.write(
      props.isochrones!.parameters,
      Array.from(
        getGPUNetworkIsochronesParameterValues({
          breakCount: 1,
          extent: [-180, -80, 180, 80],
          cellCostLimit: 0
        })
      )
    );
  });
  // Demand (-60,30) is strictly inside the first Quadbin tile. The zero-cost source cell
  // now fits alone; one polygon is complete and match 0 is authoritative again.
  for (const name of [
    'tableOverflow',
    'tableCandidateOverflow',
    'outlineOverflow',
    'outlineCandidateOverflow',
    'ringOverflow',
    'ringCandidateOverflow',
    'tableSumOverflow',
    'joinOverflow',
    'joinUncertain',
    'joinCandidateOverflow'
  ])
    expectOracleValues(result[name], [0]);
  expectOracleValues(result['ringComplete'], [1]);
  expectOracleValues(result['polygonCount'], [1]);
  expectOracleValues(result['demandMatches'], [0]);
});

it('S2 reports join capacity loss independently over complete ring geometry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const props = createRecipeCellReviewProps(harness);
  props.demand = harness.input('float32x2', [-60, 30, -65, 25]);
  props.isochrones!.joinDemand = {candidateCapacity: 1};
  props.isochrones!.parameters = harness.input(
    'float32',
    Array.from(
      getGPUNetworkIsochronesParameterValues({
        breakCount: 1,
        extent: [-180, -80, 180, 80],
        cellCostLimit: 0
      })
    )
  );
  props.output = {
    isochroneDemand: {
      status: {
        overflow: harness.output('limitedJoinOverflow', 'uint32', 1),
        candidateOverflow: harness.output('limitedJoinSourceLoss', 'uint32', 1),
        uncertainCount: harness.output('limitedJoinUncertain', 'uint32', 1)
      }
    }
  };
  addDriveTimeCatchmentRecipe(harness.graph, props);
  const result = await harness.run();
  // Both demand points are inside the one complete z2 polygon. Two bounding-box pairs
  // exceed candidateCapacity=1, independently of zero inherited loss and ring complete=1.
  expectOracleValues(result['ringComplete'], [1]);
  expectOracleValues(result['limitedJoinSourceLoss'], [0]);
  expectOracleValues(result['limitedJoinOverflow'], [1]);
  expectOracleValues(result['limitedJoinUncertain'], [0]);
});
