// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {toRecordedLensFile} from './registry';
import {getLensScenarioPresentation} from './scenario-presentations';
import type {LensScenarioDescriptor, LensScenarioResult, RecordedLensFile} from './types';

/** Describes the adapter a record was captured on, for the recorded envelope. */
export function getLensAdapterLabel(device: Device): string {
  const {renderer, vendor, type, gpuBackend} = device.info;
  return `${renderer || vendor || type}${gpuBackend ? ` (${gpuBackend})` : ''}`;
}

/** Loads and runs one scenario. The single entry point for specs, capture and hosts. */
export async function runLensScenario(
  scenario: Pick<LensScenarioDescriptor, 'load'>,
  device: Device,
  options?: {signal?: AbortSignal}
): Promise<LensScenarioResult> {
  options?.signal?.throwIfAborted();
  const loaded = await scenario.load();
  return loaded.run(device, options);
}

/**
 * Runs a scenario and returns the encoded file written to docs/lens-records. Decode it with
 * fromRecordedLensFile before accessing graphs or numeric capture values.
 */
export async function captureLensScenarioRecord(
  scenario: Pick<LensScenarioDescriptor, 'id' | 'load' | 'presentation'>,
  device: Device,
  options?: {signal?: AbortSignal; recordedAt?: string}
): Promise<RecordedLensFile> {
  const result = await runLensScenario(scenario, device, options);
  return toRecordedLensFile(result, {
    scenarioId: scenario.id,
    recordedAt: options?.recordedAt ?? new Date().toISOString(),
    adapterLabel: getLensAdapterLabel(device),
    variant: result.adapter.variant,
    presentation: scenario.presentation ?? getLensScenarioPresentation(scenario.id)
  });
}
