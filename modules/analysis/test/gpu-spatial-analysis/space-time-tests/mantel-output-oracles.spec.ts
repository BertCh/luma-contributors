// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {
  expectMantelOracle,
  getMantelLateAnchorInputs,
  expectMantelLateAnchorOracle,
  getMantelRescaledTieInputs,
  getMantelInputs,
  getMantelProps,
  expectMantelTieOracle,
  getMantelTieInputs,
  getMantelTieProps,
  getMantelIdenticalInputs,
  expectMantelIdenticalOracle,
  getMantelThreePairInputs,
  expectMantelThreePairOracle,
  expectMantelEqualCorrelationOracle
} from './mantel-output-oracles-fixture';

it('GPU Mantel reduces late-row counts and selects a nonempty anchor past row256', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expectMantelLateAnchorOracle(
    await runPoisonedOperation(
      device,
      new GPUMantelTestOperation({...getMantelProps(false), rowCount: 300}),
      getMantelLateAnchorInputs(),
      ['statistics', 'summary']
    )
  );
});

it('GPU Mantel preserves two-pair signs when centered variance cancels in float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true])
    expectMantelTieOracle(
      await runPoisonedOperation(
        device,
        new GPUMantelTestOperation(getMantelTieProps(stopping)),
        getMantelRescaledTieInputs(),
        ['statistics', 'summary', ...(stopping ? ['used'] : [])]
      ),
      stopping
    );
});

it('GPU Mantel preserves exact three-pair moment ties for inference and stopping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true])
    expectMantelThreePairOracle(
      await runPoisonedOperation(
        device,
        new GPUMantelTestOperation(getMantelTieProps(stopping)),
        getMantelThreePairInputs(),
        ['statistics', 'summary', ...(stopping ? ['used'] : [])]
      ),
      stopping
    );
});

it('GPU Mantel assigns zero variance to different shuffles with equal correlations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expectMantelEqualCorrelationOracle(
    await runPoisonedOperation(
      device,
      new GPUMantelTestOperation(getMantelProps(false)),
      getMantelThreePairInputs(1, 2),
      ['statistics', 'summary']
    )
  );
});

it('GPU Mantel publishes independent Pearson permutation literals and undefined-observation suppression into poisoned outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true])
    expectMantelOracle(
      await runPoisonedOperation(
        device,
        new GPUMantelTestOperation(getMantelProps(stopping)),
        getMantelInputs(stopping),
        ['statistics', 'summary', ...(stopping ? ['used'] : [])]
      ),
      stopping
    );
});

it('GPU Mantel counts exact two-pair ties and stops at the second draw', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const stopping of [false, true])
    expectMantelTieOracle(
      await runPoisonedOperation(
        device,
        new GPUMantelTestOperation(getMantelTieProps(stopping)),
        getMantelTieInputs(),
        ['statistics', 'summary', ...(stopping ? ['used'] : [])]
      ),
      stopping
    );
});

it('GPU Mantel preserves zero variance for identical nonzero null statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expectMantelIdenticalOracle(
    await runPoisonedOperation(
      device,
      new GPUMantelTestOperation(getMantelProps(false)),
      getMantelIdenticalInputs(),
      ['statistics', 'summary']
    )
  );
});
