// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import fleet from '../../../../website/src/gpu/runs/entries/recipes-trajectories/fleet-dwell-zone-events-recipe';
import events from '../../../../website/src/gpu/runs/entries/recipes-trajectories/zone-events';
import type {OperationRunSpec} from '../../../../website/src/gpu/runs/define-run';

vi.mock('../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (spec: unknown) => spec
}));

it.each([
  'count',
  'totals',
  'both'
] as const)('H6 a rejected %s refresh replaces prior clean fleet readbacks', async failure => {
  // Execute the host's actual refresh task; only scheduling and rendering are stubbed.
  const source = readFileSync('website/src/gpu/runs/define-run.ts', 'utf8');
  const start = source.indexOf('const task = (async () => {', source.indexOf('const refreshPeeks'));
  const end = source.indexOf('const tracked: Promise<void> = task', start);
  expect(start).toBeGreaterThan(0);
  expect(end).toBeGreaterThan(start);
  const refresh = new Function(
    'peeks',
    'applyPointer',
    `const isDestroyed = false; const epoch = 0; const peekEpoch = 0; const pointerOwnsValues = true; let framesToDraw = 0; const scheduler = {request() {}}; ${source.slice(start, end)} return task;`
  ) as (peeks: Map<unknown, unknown>, apply: () => void) => Promise<void>;
  const failed = new Set<string>();
  const values = {
    positions: Float32Array.of(0, 0),
    zones: Uint32Array.of(0),
    count: Uint32Array.of(1),
    totals: Float32Array.of(12),
    clean: Uint32Array.of(0)
  };
  const buffers = Object.fromEntries(
    Object.entries(values).map(([name, value]) => [
      name,
      {
        readAsync: async () => {
          if (failed.has(name)) throw new Error('read failed');
          return value;
        }
      }
    ])
  );
  const peeks = new Map(
    Object.entries(values).map(([name, value]) => [
      buffers[name],
      {
        format: value instanceof Float32Array ? 'float32' : 'uint32',
        data: null as Float32Array | Uint32Array | null
      }
    ])
  );
  const spec = fleet as unknown as OperationRunSpec<unknown, unknown>;
  let readout: string | undefined;
  const apply = () => {
    readout = spec.interaction!.resolve!({x: 0, y: 0, unitX: 0, unitY: 0}, {
      inputs: {
        homes: [
          [0, 0],
          [1, 1],
          [2, 2]
        ],
        world: {width: 10, height: 10}
      },
      outputs: {
        eventPositions: buffers['positions'],
        eventZones: buffers['zones'],
        eventCount: buffers['count'],
        zoneTotals: buffers['totals'],
        eventOverflow: buffers['clean'],
        eventCandidateOverflow: buffers['clean'],
        scratchCandidateOverflow: buffers['clean']
      },
      values: {zoneSize: 1},
      peek: (buffer: unknown) => peeks.get(buffer as (typeof buffers)[string])?.data
    } as never).readout;
  };
  await refresh(peeks, apply);
  expect(readout).toBe('zone A · 1 crossings · dwell 12 s');
  if (failure !== 'totals') failed.add('count');
  if (failure !== 'count') failed.add('totals');
  await refresh(peeks, apply);
  if (failure !== 'totals') expect(readout).toContain('crossings pending');
  else expect(readout).toContain('1 crossings');
  if (failure !== 'count') expect(readout).toContain('dwell unavailable');
  else expect(readout).toContain('dwell 12 s');
  failed.clear();
  await refresh(peeks, apply);
  expect(readout).toBe('zone A · 1 crossings · dwell 12 s');
});

for (const [name, entry] of [
  ['fleet', fleet],
  ['events', events]
] as const) {
  it.each([
    'positions',
    'count',
    'zones',
    'totals'
  ] as const)(`G7 ${name} resolver handles missing %s independently`, missing => {
    const outputs = {
      eventPositions: 'positions',
      eventZones: 'zones',
      eventCount: 'count',
      count: 'count',
      zoneTotals: 'totals',
      eventOverflow: 'clean',
      eventCandidateOverflow: 'clean',
      scratchCandidateOverflow: 'clean'
    };
    const readbacks: Record<string, Float32Array | Uint32Array | undefined> = {
      positions: Float32Array.of(0, 0),
      zones: Uint32Array.of(0),
      count: Uint32Array.of(1),
      totals: Float32Array.of(12),
      clean: Uint32Array.of(0)
    };
    delete readbacks[missing];
    const spec = entry as unknown as OperationRunSpec<unknown, unknown>;
    const result = spec.interaction!.resolve!({x: 0, y: 0, unitX: 0, unitY: 0}, {
      inputs: {
        homes: [
          [0, 0],
          [1, 1],
          [2, 2]
        ],
        world: {width: 10, height: 10}
      },
      outputs,
      values: {zoneSize: 1},
      peek: (output: string) => readbacks[output]
    } as never);
    if (missing === 'count' || missing === 'zones')
      expect(result.readout).toMatch(/crossings (pending|unavailable)/);
    else expect(result.readout).toContain('1 crossings');
    if (name === 'fleet') {
      if (missing === 'totals') expect(result.readout).toMatch(/dwell (pending|unavailable)/);
      else expect(result.readout).toContain('dwell 12 s');
    }
  });
}
