// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useRef, useState} from 'react';
import type * as arrow from 'apache-arrow';
import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/analysis/analysis-core';
import {previewArrowSource, type ArrowSourcePreview} from './source-preview';

export type SourcePreviewPanelProps = {
  source: arrow.Table;
  label: string;
  /** Retained source roles (geometry, H3, etc.) handled outside scalar normalization. */
  sourceOnlyColumns?: readonly string[];
  validateSource?(): string;
  onPublish(
    source: arrow.Table,
    normalization: PrepareArrowTableForGPUAnalyticsOptions
  ): Promise<void>;
};

type CheckedSource = {
  source: arrow.Table;
  normalization: PrepareArrowTableForGPUAnalyticsOptions;
  preview: ArrowSourcePreview;
};

/** Explicit preparation and conversion choices shared by file, native-dataset and SQL imports. */
export function SourcePreviewPanel({
  source,
  label,
  sourceOnlyColumns = [],
  validateSource,
  onPublish
}: SourcePreviewPanelProps) {
  const identifier = useId();
  const sourceOnlyKey = JSON.stringify(sourceOnlyColumns);
  const scalarFields = source.schema.fields.filter(
    field => !sourceOnlyColumns.includes(field.name)
  );
  const [columns, setColumns] = useState(scalarFields.map(field => field.name));
  const [precision, setPrecision] = useState<'reject' | 'exact' | 'round'>('reject');
  const [absoluteError, setAbsoluteError] = useState('0.000001');
  const [relativeError, setRelativeError] = useState('0');
  const [checked, setChecked] = useState<CheckedSource | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const generation = useRef(0);
  useEffect(() => {
    generation.current++;
    setColumns(scalarFields.map(field => field.name));
    setChecked(null);
    setStatus('');
    return () => {
      generation.current++;
    };
  }, [source, sourceOnlyKey]);
  const invalidate = () => {
    generation.current++;
    setChecked(null);
    setStatus('');
  };
  const inspect = () => {
    try {
      const normalization: PrepareArrowTableForGPUAnalyticsOptions = {
        columns,
        ...(precision === 'exact'
          ? {float64: {mode: 'exact' as const}}
          : precision === 'round'
            ? {
                float64: {
                  mode: 'round' as const,
                  maximumAbsoluteError: Number(absoluteError),
                  maximumRelativeError: Number(relativeError)
                }
              }
            : {})
      };
      if (precision === 'round' && (!absoluteError.trim() || !relativeError.trim()))
        throw new Error('Enter both conversion error budgets.');
      const preview = previewArrowSource(source, normalization, sourceOnlyColumns);
      const details = preview.canUpload ? validateSource?.() : undefined;
      setChecked({source, normalization, preview});
      setStatus(
        preview.reason ?? details ?? 'Schema and precision checked. Ready for explicit GPU upload.'
      );
    } catch (error) {
      setChecked(null);
      setStatus(error instanceof Error ? error.message : String(error));
    }
  };
  const publish = async () => {
    if (!checked || checked.source !== source || !checked.preview.canUpload || busy) return;
    const revision = generation.current;
    setBusy(true);
    setStatus('Uploading selected fields…');
    try {
      await onPublish(source, checked.normalization);
      if (revision === generation.current)
        setStatus('Selected fields published to the GPU workspace.');
    } catch (error) {
      if (revision === generation.current)
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section aria-label={`Prepare ${label}`}>
      <h3>Prepare {label}</h3>
      <p>
        {source.numRows.toLocaleString()} rows · {source.batches.length} source batches. Original
        field names and row order are preserved.
      </p>
      <fieldset disabled={busy}>
        <legend>Fields to upload</legend>
        {!scalarFields.length && (
          <p>No scalar fields. Original source rows and role data are retained.</p>
        )}
        {scalarFields.map(field => (
          <label key={field.name} style={{display: 'block'}}>
            <input
              type="checkbox"
              checked={columns.includes(field.name)}
              onChange={event => {
                invalidate();
                setColumns(current =>
                  event.target.checked
                    ? [...current, field.name]
                    : current.filter(name => name !== field.name)
                );
              }}
            />{' '}
            {field.name} ({String(field.type)})
          </label>
        ))}
      </fieldset>
      <label htmlFor={`${identifier}-precision`}>Float64 conversion</label>
      <select
        id={`${identifier}-precision`}
        value={precision}
        disabled={busy}
        onChange={event => {
          invalidate();
          const value = event.target.value;
          if (value === 'reject' || value === 'exact' || value === 'round') setPrecision(value);
        }}
      >
        <option value="reject">Require a conversion choice</option>
        <option value="exact">Only exactly representable Float32 values</option>
        <option value="round">Allow a measured rounding error</option>
      </select>
      {precision === 'round' && (
        <fieldset disabled={busy}>
          <legend>Allowed error: absolute + relative × source magnitude</legend>
          <label htmlFor={`${identifier}-absolute`}>Maximum absolute error</label>
          <input
            id={`${identifier}-absolute`}
            type="number"
            min="0"
            step="any"
            value={absoluteError}
            onChange={event => {
              invalidate();
              setAbsoluteError(event.target.value);
            }}
          />
          <label htmlFor={`${identifier}-relative`}>Maximum relative error</label>
          <input
            id={`${identifier}-relative`}
            type="number"
            min="0"
            step="any"
            value={relativeError}
            onChange={event => {
              invalidate();
              setRelativeError(event.target.value);
            }}
          />
        </fieldset>
      )}
      <div>
        <button type="button" disabled={busy} onClick={inspect}>
          Check schema and precision
        </button>
        <button
          type="button"
          disabled={busy || !checked?.preview.canUpload || checked.source !== source}
          onClick={() => void publish()}
        >
          Publish selected fields
        </button>
      </div>
      {checked?.source === source && (
        <div role="region" aria-label="Source conversion preview" style={{overflow: 'auto'}}>
          <table aria-label="Source field conversion">
            <thead>
              <tr>
                <th scope="col">Field</th>
                <th scope="col">GPU representation</th>
                <th scope="col">Conversion details</th>
              </tr>
            </thead>
            <tbody>
              {checked.preview.fields.map(field => (
                <tr key={field.name}>
                  <th scope="row">
                    {field.name}
                    {field.selected ? '' : ' (excluded)'}
                  </th>
                  <td>{field.targetType ?? 'Unsupported'}</td>
                  <td>
                    {field.reason ??
                      (field.kind === 'time'
                        ? `${field.timeUnit} · ${field.timeZone ?? 'no time zone'}`
                        : field.kind === 'category'
                          ? `${field.categoryCount} category labels`
                          : `Maximum error: absolute ${field.maximumAbsoluteError}, relative ${field.maximumRelativeError}`)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p role="status">{status}</p>
      <p>
        Checking uses retained CPU Arrow data. GPU allocation starts only when you publish. Excluded
        fields remain in the source and are not exposed as resident columns.
      </p>
    </section>
  );
}
