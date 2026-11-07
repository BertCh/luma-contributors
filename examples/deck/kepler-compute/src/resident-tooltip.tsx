// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
export {ResidentHoverIdentity, getResidentHoverIdentity} from './resident-hover-identity';
import {ResidentSelectGeometryControl} from './resident-select-geometry-control';
import {getConsumerStateMessage, type ResidentConsumerState} from './resident-consumer-state';
import {
  formatTooltipValue,
  getTooltipComputedProvider,
  readTooltipValues,
  resolveTooltipFields,
  type ResidentTooltipFieldConfig,
  type ResidentTooltipValue
} from './resident-tooltip-fields';
import {
  compareTooltipValues,
  formatTooltipDifference,
  type ResidentTooltipCompareType
} from './resident-tooltip-compare';

export type ResidentTooltipReader = (
  datasetId: string,
  layerId: string,
  sourceRowIndex: number,
  fieldNames: readonly string[]
) => Promise<Record<string, string | number | boolean | null> | null>;

export function ResidentTooltip({
  datasetId,
  layerId,
  sourceRowIndex,
  readTooltip,
  fieldNames,
  fields,
  revisionKey = '',
  compare,
  geometrySelectionUnavailable
}: {
  datasetId: string;
  layerId: string;
  sourceRowIndex: number;
  readTooltip: ResidentTooltipReader;
  fieldNames: readonly string[];
  /** Native fieldsToShow entries with formats; takes precedence over fieldNames. */
  fields?: readonly ResidentTooltipFieldConfig[];
  /** Changes whenever the displayed dataset/selection revision changes. */
  revisionKey?: string;
  /** Native compare mode: the click-pinned row to compare the hovered row against. */
  compare?: {sourceRowIndex: number; type: ResidentTooltipCompareType};
  geometrySelectionUnavailable?: boolean;
}) {
  const resolved = resolveTooltipFields(fields ?? fieldNames.map(name => ({name})));
  const fieldKey = JSON.stringify(resolved.fields);
  const compareKey = compare ? `${compare.sourceRowIndex}:${compare.type}` : '';
  const [state, setState] = useState<ResidentConsumerState<
    Record<string, ResidentTooltipValue>
  > | null>(null);
  const [pinned, setPinned] = useState<Record<string, ResidentTooltipValue> | null>(null);
  useEffect(() => {
    let current = true;
    // Never keep the previous row's values visible as current.
    setState(null);
    setPinned(null);
    const names = resolved.fields.map(field => field.name);
    const readRow = (row: number) => readTooltip(datasetId, layerId, row, names);
    const computedProvider = getTooltipComputedProvider(datasetId);
    void readTooltipValues({
      getRevision: () => revisionKey,
      readOriginal: () => readRow(sourceRowIndex),
      readComputed: computedProvider
        ? () => computedProvider(datasetId, layerId, sourceRowIndex)
        : undefined
    }).then(result => {
      if (current) setState(result);
    });
    if (compare)
      void readRow(compare.sourceRowIndex)
        .then(result => {
          if (current) setPinned(result);
        })
        .catch(() => {
          if (current) setPinned(null);
        });
    return () => {
      current = false;
    };
  }, [datasetId, layerId, sourceRowIndex, readTooltip, fieldKey, revisionKey, compareKey]);
  const formats = new Map(resolved.fields.map(field => [field.name, field.format]));
  const comparisons =
    state?.status === 'ready' && compare && pinned
      ? new Map(
          // Identity entries such as Row are not configured fields, so they never get a delta.
          compareTooltipValues(pinned, state.value, compare.type)
            .filter(entry => formats.has(entry.name))
            .map(entry => [entry.name, entry])
        )
      : null;
  return (
    <section aria-label="Data inspection" data-resident-tooltip>
      {state === null ? (
        <p role="status" data-resident-consumer-state="pending">
          Reading selected row…
        </p>
      ) : state.status === 'ready' ? (
        <dl>
          {Object.entries(state.value).map(([name, value]) => {
            const comparison = comparisons?.get(name);
            return (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  {formatTooltipValue(value, formats.get(name))}
                  {comparison && compare && typeof value === 'number' && (
                    <span data-resident-tooltip-difference>
                      {' '}
                      (
                      {formatTooltipDifference(
                        comparison.difference,
                        compare.type,
                        formats.get(name)
                      )}
                      )
                    </span>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p
          role={state.status === 'error' ? 'alert' : 'status'}
          data-resident-consumer-state={state.status}
        >
          {getConsumerStateMessage(state)}
        </p>
      )}
      {resolved.truncated && <p>Showing the first 8 selected fields.</p>}
      {geometrySelectionUnavailable && <ResidentSelectGeometryControl />}
    </section>
  );
}
