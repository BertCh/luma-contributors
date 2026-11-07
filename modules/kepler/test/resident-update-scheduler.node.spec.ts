// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {ResidentUpdateScheduler} from '../../../examples/deck/kepler-compute/src/resident-update-scheduler';

describe('resident redraw transaction boundary', () => {
  it('renders only the final native filter state after a synchronous transaction', async () => {
    let field = 'old';
    let minimum = 0;
    const rendered: string[] = [];
    const scheduler = new ResidentUpdateScheduler(
      () => true,
      () => rendered.push(`${field}:${minimum}`)
    );
    field = 'value';
    scheduler.request();
    minimum = 50;
    scheduler.request();
    expect(rendered).toEqual([]);
    await Promise.resolve();
    expect(rendered).toEqual(['value:50']);
  });

  it('suppresses disposed or device-lost owners and preserves a later generation', async () => {
    let current = true;
    let draws = 0;
    const scheduler = new ResidentUpdateScheduler(
      () => current,
      () => draws++
    );
    scheduler.request();
    current = false;
    await Promise.resolve();
    expect(draws).toBe(0);
    current = true;
    scheduler.request();
    scheduler.cancel();
    scheduler.request();
    await Promise.resolve();
    expect(draws).toBe(1);
    scheduler.request();
    scheduler.cancel();
    await Promise.resolve();
    expect(draws).toBe(1);
  });

  it('retains invalidation requested during a draw for the next microtask', async () => {
    let draws = 0;
    const scheduler = new ResidentUpdateScheduler(
      () => true,
      () => {
        draws++;
        if (draws === 1) scheduler.request();
      }
    );
    scheduler.request();
    await Promise.resolve();
    expect(draws).toBe(1);
    await Promise.resolve();
    expect(draws).toBe(2);
  });
});
