// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {getZoneEventLossCaptures} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/zone-event-loss-captures';
import {formatZoneEventReadout} from '../../../../website/src/gpu/runs/entries/recipes-trajectories/lib/zone-event-readout';

it('lenses-consumers:F3 captures all event-loss flags and qualifies crossings and dwell', () => {
  for (const path of [
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/fleet-dwell-zone-events-recipe.lens.ts',
    'website/src/gpu/runs/entries/recipes-trajectories/fleet-dwell-zone-events-recipe.ts',
    'website/src/gpu/runs/entries/recipes-trajectories/zone-events.ts'
  ]) {
    const source = readFileSync(path, 'utf8');
    expect(source, path).toContain(
      path.endsWith('.lens.ts') ? 'getZoneEventLossCaptures' : 'eventCandidateOverflow'
    );
    expect(source, path).toContain('scratchCandidateOverflow');
    expect(source, path).not.toContain('0 means every event fit and dwell totals are complete');
  }
});

it.each([
  [0, 1, 0, '1 crossings (incomplete) · dwell 10 s'],
  [0, 1, 1, '1 crossings (incomplete) · dwell 10 s (incomplete)'],
  [1, 0, 0, '1 crossings (incomplete) · dwell 10 s'],
  [0, 0, 0, '1 crossings · dwell 10 s']
])('separates final=%i candidate=%i scratch=%i in the lens captures and website readout', (overflow, candidateOverflow, scratch, expected) => {
  const captures = getZoneEventLossCaptures({overflow, candidateOverflow}, scratch);
  expect(captures.eventOverflow.view).toBe(overflow);
  expect(captures.eventCandidateOverflow.view).toBe(candidateOverflow);
  expect(captures.scratchCandidateOverflow.view).toBe(scratch);
  expect(captures.scratchCandidateOverflow.meaning).toContain('dwell totals may all be incomplete');
  expect(
    formatZoneEventReadout(
      'A',
      1,
      {
        eventOverflow: captures.eventOverflow.view,
        eventCandidateOverflow: captures.eventCandidateOverflow.view,
        scratchCandidateOverflow: captures.scratchCandidateOverflow.view
      },
      10
    )
  ).toBe(`zone A · ${expected}`);
});

it('does not certify a pending status readback', () => {
  expect(formatZoneEventReadout('A', 0, {}, 0)).toBe(
    'zone A · 0 crossings (status pending) · dwell 0 s (status pending)'
  );
});
