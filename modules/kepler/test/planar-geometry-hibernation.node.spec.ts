// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {PlanarGeometryHibernation} from '../../../examples/deck/kepler-compute/src/planar-geometry-hibernation';

function createHarness(options: {steps?: number; failInstall?: boolean} = {}) {
  const tasks: (() => void)[] = [];
  const events: string[] = [];
  const hibernation = new PlanarGeometryHibernation({
    releasePlanar: () => events.push('release'),
    createPreparation: function* () {
      events.push('prepare');
      for (let step = 0; step < (options.steps ?? 2); step++) yield;
      return 'prepared';
    },
    installPlanar: prepared => {
      if (options.failInstall) throw new Error('over budget');
      events.push(`install:${String(prepared)}`);
    },
    requestUpdate: () => events.push('update'),
    schedule: callback => {
      tasks.push(callback);
      return tasks.length;
    },
    cancel: handle => {
      tasks[(handle as number) - 1] = () => {};
    },
    now: () => 0,
    sliceMilliseconds: 0
  });
  const flush = () => {
    while (tasks.length) tasks.shift()!();
  };
  return {hibernation, events, flush};
}

describe('PlanarGeometryHibernation', () => {
  it('releases the planar copy after the grace period once the globe variant is ready', () => {
    const {hibernation, events, flush} = createHarness();
    hibernation.noteGlobeFrame(false);
    expect(hibernation.status.state).toBe('resident');
    hibernation.noteGlobeFrame(true);
    hibernation.noteGlobeFrame(true);
    flush();
    expect(events).toEqual(['release', 'update']);
    expect(hibernation.status.state).toBe('released');
  });

  it('keeps the planar copy when a flat frame arrives within the grace period', () => {
    const {hibernation, events, flush} = createHarness();
    hibernation.noteGlobeFrame(true);
    expect(hibernation.noteFlatFrame()).toBe('resident');
    flush();
    expect(events).toEqual([]);
  });

  it('rebuilds in slices on exit and reports restoring until installed', () => {
    const {hibernation, events, flush} = createHarness({steps: 3});
    hibernation.noteGlobeFrame(true);
    flush();
    events.length = 0;
    expect(hibernation.noteFlatFrame()).toBe('restoring');
    expect(hibernation.noteFlatFrame()).toBe('restoring');
    flush();
    expect(events).toEqual(['prepare', 'install:prepared', 'update']);
    expect(hibernation.status.state).toBe('resident');
  });

  it('drops the rebuild when the globe returns with its variant ready', () => {
    const {hibernation, events, flush} = createHarness();
    hibernation.noteGlobeFrame(true);
    flush();
    hibernation.noteFlatFrame();
    hibernation.noteGlobeFrame(true);
    flush();
    expect(events).not.toContain('install:prepared');
    expect(hibernation.status.state).toBe('released');
  });

  it('restores planar when the globe variant disappeared', () => {
    const {hibernation, events, flush} = createHarness();
    hibernation.noteGlobeFrame(true);
    flush();
    hibernation.noteGlobeFrame(false);
    flush();
    expect(events).toContain('install:prepared');
    expect(hibernation.status.state).toBe('resident');
  });

  it('does not retry a failed restore and exposes a notice', () => {
    const {hibernation, events, flush} = createHarness({failInstall: true});
    hibernation.noteGlobeFrame(true);
    flush();
    hibernation.noteFlatFrame();
    flush();
    expect(hibernation.status.state).toBe('released');
    expect(hibernation.status.notice).toMatch(/over budget/);
    events.length = 0;
    expect(hibernation.noteFlatFrame()).toBe('released');
    flush();
    expect(events).toEqual([]);
  });
});
