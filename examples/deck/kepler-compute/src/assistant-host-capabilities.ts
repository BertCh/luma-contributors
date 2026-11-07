// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Host capability handlers for the assistant operations that the manual UI runs from panels:
 * SQL, charts and map boundary. Like export and source loading they stage a notice naming the
 * approved, already guard-checked request; the user completes it from the matching panel.
 */

type CapabilityHandler = (request: Readonly<Record<string, unknown>>) => void;

export function createAssistantNoticeCapabilities(
  stageNotice: (message: string) => void
): Readonly<Record<'sql-query' | 'chart-widget' | 'map-boundary', CapabilityHandler>> {
  return {
    'sql-query': request => {
      const target = request['createsDataset'] ? 'creating a new dataset' : 'read-only';
      stageNotice(
        `Assistant requested a ${target} SQL query (at most ${String(request['maximumRows'])} rows): ${String(request['query'])}. Run it from Add Data -> SQL.`
      );
    },
    'chart-widget': request => {
      stageNotice(
        `Assistant requested a ${String(request['chartType'])} chart of ${String(request['datasetId'])} (${(request['fields'] as string[]).join(', ')}). Add it from the Charts panel.`
      );
    },
    'map-boundary': request => {
      stageNotice(
        request['datasetId']
          ? `Assistant requested the map bounds of ${String(request['datasetId'])}. Use Fit to data on the layer panel.`
          : 'Assistant requested the current map bounds. They are shown in the map status line.'
      );
    }
  };
}
