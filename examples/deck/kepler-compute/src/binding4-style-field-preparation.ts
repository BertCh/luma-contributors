// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {FieldPreparationCoordinator} from './binding3-field-preparation';

type NamedField = {name: string} | null | undefined;

/** Native layer config fields whose columns feed color, size, height and related scales/domains. */
export type StyleFieldConfig = {
  colorField?: NamedField;
  strokeColorField?: NamedField;
  sizeField?: NamedField;
  radiusField?: NamedField;
  coverageField?: NamedField;
  heightField?: NamedField;
  elevationOffsetField?: NamedField;
};

export type StyleFieldRequirement = {channel: keyof StyleFieldConfig; name: string};

/** Fields a layer style reads, in a stable order, one entry per channel. */
export function listStyleFieldRequirements(config: StyleFieldConfig): StyleFieldRequirement[] {
  const channels: (keyof StyleFieldConfig)[] = [
    'colorField',
    'strokeColorField',
    'sizeField',
    'radiusField',
    'coverageField',
    'heightField',
    'elevationOffsetField'
  ];
  const requirements: StyleFieldRequirement[] = [];
  for (const channel of channels) {
    const field = config[channel];
    if (field?.name) requirements.push({channel, name: field.name});
  }
  return requirements;
}

/**
 * Asks the coordinator to prepare each style field the resident copy lacks (one republish covers
 * all of them) and returns one human-readable sentence per missing field. An empty list means
 * every style field is already resident.
 */
export function requestStyleFieldPreparation(
  datasetId: string,
  config: StyleFieldConfig,
  residentColumns: Readonly<Record<string, unknown>>,
  coordinator: Pick<FieldPreparationCoordinator, 'request'>
): string[] {
  const messages: string[] = [];
  const seen = new Set<string>();
  for (const {channel, name} of listStyleFieldRequirements(config)) {
    if (residentColumns[name] || seen.has(name)) continue;
    seen.add(name);
    const outcome = coordinator.request(datasetId, name);
    const detail =
      outcome.status === 'requested' || outcome.status === 'pending'
        ? 'Preparing it now.'
        : outcome.reason;
    messages.push(`${channel.replace(/Field$/, '')} field "${name}" is not on the GPU. ${detail}`);
  }
  return messages;
}
