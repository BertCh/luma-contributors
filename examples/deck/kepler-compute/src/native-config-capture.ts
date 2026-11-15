// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Saved-format Kepler visState slice that survives replacing a native dataset in place. */
import type {SampleMapProvenance} from './sample-map-provenance';

export type NativeDatasetConfig = {
  sampleMap?: SampleMapProvenance;
  /** Original dataset color, kept so the replaced dataset looks the same in Kepler's dataset list. */
  datasetColor?: readonly number[];
  /** True when at least one saved layer targets the dataset; false means generate a default layer. */
  hasLayers: boolean;
  config: {
    version: 'v1';
    config: {
      visState: {
        layers: readonly unknown[];
        filters: readonly unknown[];
        splitMaps?: readonly unknown[];
        interactionConfig?: Record<string, unknown>;
      };
    };
  };
};

type SavedRecord = Record<string, unknown>;

function isRecord(value: unknown): value is SavedRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Selects the layers, filters and tooltip fields of one dataset from a saved Kepler configuration
 * (`KeplerGlSchema.getConfigToSave`). Used to re-attach a native dataset's presentation after its
 * data is replaced by the equivalent resident dataset under the same ID.
 */
export function selectNativeDatasetConfig(
  saved: unknown,
  datasetId: string
): NativeDatasetConfig | undefined {
  if (!isRecord(saved) || !isRecord(saved['config'])) return undefined;
  const visState = saved['config']['visState'];
  if (!isRecord(visState)) return undefined;
  const layers = Array.isArray(visState['layers'])
    ? visState['layers'].filter(
        layer =>
          isRecord(layer) && isRecord(layer['config']) && layer['config']['dataId'] === datasetId
      )
    : [];
  const filters = Array.isArray(visState['filters'])
    ? visState['filters'].filter(
        filter =>
          isRecord(filter) &&
          (filter['dataId'] === datasetId ||
            (Array.isArray(filter['dataId']) && filter['dataId'].includes(datasetId)))
      )
    : [];
  const interaction = visState['interactionConfig'];
  const tooltip = isRecord(interaction) ? interaction['tooltip'] : undefined;
  const fieldsToShow = isRecord(tooltip) ? tooltip['fieldsToShow'] : undefined;
  const datasetFields = isRecord(fieldsToShow) ? fieldsToShow[datasetId] : undefined;
  return {
    hasLayers: layers.length > 0,
    config: {
      version: 'v1',
      config: {
        visState: {
          layers,
          filters,
          // Removing a table also removes its layer IDs from every map panel. Without
          // these masks, re-registration adds each visible layer to every panel.
          ...(Array.isArray(visState['splitMaps']) ? {splitMaps: visState['splitMaps']} : {}),
          ...(isRecord(tooltip) && datasetFields
            ? {
                interactionConfig: {
                  tooltip: {...tooltip, fieldsToShow: {[datasetId]: datasetFields}}
                }
              }
            : {})
        }
      }
    }
  };
}

/**
 * Field names a dataset's saved layers, visual channels, text labels, filters and tooltips refer to.
 * An in-place replacement must provide every one of them or the saved presentation would silently break.
 */
export function listNativeConfigFieldNames(
  config: NativeDatasetConfig,
  includeSoftReferences = true
): string[] {
  const names = new Set<string>();
  const visState = config.config.config.visState;
  for (const layer of visState.layers) {
    if (!isRecord(layer)) continue;
    const layerConfig = layer['config'];
    if (isRecord(layerConfig)) {
      const columns = layerConfig['columns'];
      if (isRecord(columns))
        for (const column of Object.values(columns)) {
          const name = isRecord(column) ? column['value'] : column;
          if (typeof name === 'string' && name) names.add(name);
        }
      const labels = includeSoftReferences ? layerConfig['textLabel'] : undefined;
      if (Array.isArray(labels))
        for (const label of labels) {
          const field = isRecord(label) ? label['field'] : undefined;
          if (isRecord(field) && typeof field['name'] === 'string') names.add(field['name']);
        }
    }
    const channels = layer['visualChannels'];
    if (isRecord(channels))
      for (const [key, field] of Object.entries(channels))
        if (key.endsWith('Field') && isRecord(field) && typeof field['name'] === 'string')
          names.add(field['name']);
  }
  for (const filter of visState.filters) {
    if (!isRecord(filter)) continue;
    for (const name of Array.isArray(filter['name']) ? filter['name'] : [filter['name']])
      if (typeof name === 'string' && name) names.add(name);
  }
  const tooltip = includeSoftReferences ? visState.interactionConfig?.['tooltip'] : undefined;
  const fieldsToShow = isRecord(tooltip) ? tooltip['fieldsToShow'] : undefined;
  if (isRecord(fieldsToShow))
    for (const fields of Object.values(fieldsToShow))
      if (Array.isArray(fields))
        for (const field of fields)
          if (isRecord(field) && typeof field['name'] === 'string') names.add(field['name']);
  return [...names];
}

/**
 * Hard references (layer columns, visual channels, filters) the replacement lacks; empty means safe to
 * replace. Tooltip and text-label fields are soft: Kepler drops a missing one without breaking the layer.
 */
export function findMissingNativeConfigFields(
  config: NativeDatasetConfig,
  availableFieldNames: readonly string[]
): string[] {
  return listNativeConfigFieldNames(config, false).filter(
    name => !availableFieldNames.includes(name)
  );
}
