// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {CPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
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

it('CPU Mantel matches independent late-row anchor and count reduction literals', async () => {
  const device = createNullWebGPUDevice();
  try {
    expectMantelLateAnchorOracle(
      await runOperationGraph(device, {
        inputs: getMantelLateAnchorInputs(),
        operations: [new CPUMantelTestOperation({...getMantelProps(false), rowCount: 300})],
        outputs: ['statistics', 'summary']
      })
    );
  } finally {
    device.destroy();
  }
});

it('CPU Mantel preserves the independently ordered signs on the rescaled two-pair case', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const stopping of [false, true])
      expectMantelTieOracle(
        await runOperationGraph(device, {
          inputs: getMantelRescaledTieInputs(),
          operations: [new CPUMantelTestOperation(getMantelTieProps(stopping))],
          outputs: ['statistics', 'summary', ...(stopping ? ['used'] : [])]
        }),
        stopping
      );
  } finally {
    device.destroy();
  }
});

it('CPU Mantel preserves exact three-pair moment ties for inference and stopping', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const stopping of [false, true]) {
      const props = getMantelTieProps(stopping);
      expectMantelThreePairOracle(
        await runOperationGraph(device, {
          inputs: getMantelThreePairInputs(),
          operations: [new CPUMantelTestOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        stopping
      );
    }
  } finally {
    device.destroy();
  }
});

it('CPU Mantel assigns zero variance to different shuffles with equal correlations', async () => {
  const device = createNullWebGPUDevice();
  try {
    expectMantelEqualCorrelationOracle(
      await runOperationGraph(device, {
        inputs: getMantelThreePairInputs(1, 2),
        operations: [new CPUMantelTestOperation(getMantelProps(false))],
        outputs: ['statistics', 'summary']
      })
    );
  } finally {
    device.destroy();
  }
});

it('CPU Mantel matches independent Pearson permutation literals and undefined-observation suppression', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const stopping of [false, true]) {
      const props = getMantelProps(stopping);
      expectMantelOracle(
        await runOperationGraph(device, {
          inputs: getMantelInputs(stopping),
          operations: [new CPUMantelTestOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        stopping
      );
    }
  } finally {
    device.destroy();
  }
});

it('CPU Mantel counts exact two-pair ties and stops at the second draw', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const stopping of [false, true]) {
      const props = getMantelTieProps(stopping);
      expectMantelTieOracle(
        await runOperationGraph(device, {
          inputs: getMantelTieInputs(),
          operations: [new CPUMantelTestOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        stopping
      );
    }
  } finally {
    device.destroy();
  }
});

it('CPU Mantel preserves zero variance for identical nonzero null statistics', async () => {
  const device = createNullWebGPUDevice();
  try {
    expectMantelIdenticalOracle(
      await runOperationGraph(device, {
        inputs: getMantelIdenticalInputs(),
        operations: [new CPUMantelTestOperation(getMantelProps(false))],
        outputs: ['statistics', 'summary']
      })
    );
  } finally {
    device.destroy();
  }
});
