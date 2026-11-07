// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {ComputeRuntime} from './compute-runtime';
import {listTrackedOperations, subscribeOperations} from './operation-tracker';

const CATEGORY_LABELS = [
  ['source', 'Source reservations'],
  ['output', 'Output reservations'],
  ['scratch', 'Scratch reservations'],
  ['readback', 'Readback reservations']
] as const;

function formatBytes(byteLength: number): string {
  if (byteLength >= 1024 * 1024 * 1024) return `${(byteLength / 1024 ** 3).toFixed(2)} GiB`;
  if (byteLength >= 1024 * 1024) return `${(byteLength / 1024 ** 2).toFixed(1)} MiB`;
  if (byteLength >= 1024) return `${(byteLength / 1024).toFixed(1)} KiB`;
  return `${byteLength} B`;
}

/**
 * Live view of the session memory registry: allocated resident data plus tagged reservations,
 * with the active requests and host operations that can be cancelled. Untagged reservations are
 * reported as scratch.
 */
export function MemoryBudgetPanel({runtime}: {runtime: ComputeRuntime | null}) {
  const [, setVersion] = useState(0);
  const registry = runtime?.session.registry;
  useEffect(() => {
    const refresh = () => setVersion(value => value + 1);
    const stopOperations = subscribeOperations(refresh);
    const stopRegistry = registry?.subscribeMemory(refresh);
    return () => {
      stopOperations();
      stopRegistry?.();
    };
  }, [registry]);
  if (!registry) return null;
  const usage = registry.memoryUsage;
  const breakdown = registry.getReservationBreakdown();
  const requests = registry.getActiveRequests();
  const operations = listTrackedOperations();
  const budget = registry.memoryBudgetBytes;
  const allocated = usage.allocatedBytes.cpu + usage.allocatedBytes.gpu;
  return (
    <section aria-label="Memory budget" data-testid="memory-budget-panel">
      <h2>Memory budget</h2>
      <meter
        aria-label="Memory used"
        min={0}
        max={budget}
        value={Math.min(budget, usage.totalBytes)}
        data-testid="memory-total-meter"
      />
      <p className="hint" data-testid="memory-total">
        Used {formatBytes(usage.totalBytes)} of {formatBytes(budget)}; available{' '}
        {formatBytes(usage.availableBytes)}
      </p>
      <ul data-testid="memory-categories">
        <li data-category="resident">
          Resident data (CPU {formatBytes(usage.allocatedBytes.cpu)}, GPU{' '}
          {formatBytes(usage.allocatedBytes.gpu)}): {formatBytes(allocated)}
        </li>
        {CATEGORY_LABELS.map(([category, label]) => (
          <li key={category} data-category={category} data-bytes={breakdown[category]}>
            {label}: {formatBytes(breakdown[category])}
          </li>
        ))}
      </ul>
      <ul data-testid="memory-operations">
        {requests.map(request => (
          <li key={`request-${request.id}`} data-operation="request">
            {request.slot} (inputs:{' '}
            {request.inputs.map(input => input.datasetId).join(', ') || 'none'})
            <button type="button" onClick={() => request.cancel()}>
              Cancel
            </button>
          </li>
        ))}
        {operations.map(operation => (
          <li key={`operation-${operation.id}`} data-operation="host">
            {operation.label}
            {operation.detail ? ` - ${operation.detail}` : ''}
            {operation.fraction !== undefined ? (
              <progress
                aria-label={`${operation.label} progress`}
                max={1}
                value={operation.fraction}
              />
            ) : null}
            <button type="button" onClick={() => operation.cancel()}>
              Cancel
            </button>
          </li>
        ))}
        {!requests.length && !operations.length ? (
          <li className="hint">No active operations.</li>
        ) : null}
      </ul>
    </section>
  );
}
