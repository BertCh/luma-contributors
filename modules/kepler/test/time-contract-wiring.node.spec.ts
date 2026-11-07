// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, it} from 'vitest';
import {makeResidentTimeHistogramPlan} from '../../../examples/deck/kepler-compute/src/resident-time-histogram';
import {makeZonedTimeHistogramEdges} from '../../../examples/deck/kepler-compute/src/time-contract-histogram-plan';
import {
  ExactPlaybackDriver,
  getKeplerValueFromPlayback,
  getPlaybackModeFromKeplerWindow,
  makePlaybackStateFromKeplerFilter,
  type ExactPlaybackHost
} from '../../../examples/deck/kepler-compute/src/time-contract-native-bridge';
import {
  clearTimeIntervalDeclarations,
  declareTimeInterval,
  getTimeIntervalDeclaration,
  setTimeContractSettings
} from '../../../examples/deck/kepler-compute/src/time-contract-settings';

afterEach(() => {
  setTimeContractSettings({timeZone: 'UTC'});
  clearTimeIntervalDeclarations();
});

const HOUR = 3_600_000;
const DAY = 24 * HOUR;

describe('zone-aware time histogram plan', () => {
  it('UTC keeps the native edges', () => {
    const domain: [number, number] = [Date.UTC(2026, 2, 1, 5), Date.UTC(2026, 2, 3, 5)];
    const plan = makeResidentTimeHistogramPlan(domain, '1-day');
    expect(plan.thresholds[0]).toBe(Date.UTC(2026, 2, 1));
  });

  it('New York days are local midnights with a 23 hour day at spring forward', () => {
    // 2026-03-08 is the US spring-forward day.
    const domain: [number, number] = [Date.UTC(2026, 2, 7, 12), Date.UTC(2026, 2, 10, 12)];
    const edges = makeZonedTimeHistogramEdges(domain, 'day', 'America/New_York');
    expect(edges[0]).toBeLessThanOrEqual(domain[0]);
    expect(edges[edges.length - 1]).toBeGreaterThanOrEqual(domain[1]);
    const widths = edges.slice(1).map((edge, index) => edge - edges[index]);
    expect(widths).toContain(23 * HOUR);
    for (const edge of edges) {
      const label = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        hourCycle: 'h23',
        hour: 'numeric',
        minute: 'numeric'
      }).format(edge);
      expect(label).toBe('00:00');
    }
  });

  it('fall back has a 25 hour day and a repeated local hour gets two edges', () => {
    // 2026-11-01 is the US fall-back day.
    const domain: [number, number] = [Date.UTC(2026, 10, 1, 0), Date.UTC(2026, 10, 2, 12)];
    const days = makeZonedTimeHistogramEdges(domain, 'day', 'America/New_York');
    expect(days.slice(1).map((edge, index) => edge - days[index])).toContain(25 * HOUR);
    const hours = makeZonedTimeHistogramEdges(
      [Date.UTC(2026, 10, 1, 4), Date.UTC(2026, 10, 1, 9)],
      'hour',
      'America/New_York'
    );
    // Hourly UTC instants stay one hour apart; the repeated 01:00 appears twice.
    expect(hours.slice(1).every((edge, index) => edge - hours[index] === HOUR)).toBe(true);
  });

  it('feeds makeResidentTimeHistogramPlan through the shared settings, only for local intervals', () => {
    setTimeContractSettings({timeZone: '+05:30'});
    const domain: [number, number] = [Date.UTC(2026, 0, 1, 3), Date.UTC(2026, 0, 3, 3)];
    const zoned = makeResidentTimeHistogramPlan(domain, '1-day');
    expect(zoned.thresholds[0] % DAY).toBe(DAY - 5.5 * HOUR);
    const weekly = makeResidentTimeHistogramPlan(domain, '1-week');
    expect(weekly.thresholds[0] % DAY).toBe(0);
  });
});

describe('native playback bridge', () => {
  const filter = {
    domain: [0, 600_000],
    value: [0, 60_000],
    speed: 1,
    isAnimating: true,
    animationWindow: 'free'
  };

  it('maps Kepler animation windows to contract modes', () => {
    expect(getPlaybackModeFromKeplerWindow('interval')).toBe('snap');
    expect(getPlaybackModeFromKeplerWindow('incremental')).toBe('incremental');
    expect(getPlaybackModeFromKeplerWindow('point')).toBe('point');
    expect(getPlaybackModeFromKeplerWindow('free')).toBe('free');
  });

  it('snap needs steps; Kepler overhang starts free playback with an empty window at the domain start', () => {
    expect(makePlaybackStateFromKeplerFilter(filter, 'snap')).toBeNull();
    const state = makePlaybackStateFromKeplerFilter(filter, 'free')!;
    expect(getKeplerValueFromPlayback(state)).toEqual([0, 0]);
  });

  it('drives the host value from exact elapsed time and pauses native animation', () => {
    const written: [number, number][] = [];
    let nativePaused = 0;
    const host: ExactPlaybackHost = {
      getFilter: () => filter,
      setFilterValue: value => void written.push(value),
      pauseNativeAnimation: () => void nativePaused++,
      requestFrame: () => 1,
      cancelFrame: () => {}
    };
    const driver = new ExactPlaybackDriver(host);
    expect(driver.start('free')).toBe(true);
    expect(nativePaused).toBe(1);
    const first = written.length;
    driver.tick(16_667n);
    expect(written.length).toBe(first + 1);
    const [start, end] = written[written.length - 1];
    // The window grows into the domain from its start, clipped to the domain.
    expect(start).toBe(0);
    expect(end).toBeGreaterThan(0);
    expect(end).toBeLessThanOrEqual(60_000);
    driver.pause();
    expect(driver.isRunning).toBe(false);
  });
});

describe('interval declarations', () => {
  it('are keyed by dataset and start column', () => {
    declareTimeInterval('a', 'start', {endColumn: 'end'});
    expect(getTimeIntervalDeclaration('a', 'start')?.endColumn).toBe('end');
    expect(getTimeIntervalDeclaration('b', 'start')).toBeUndefined();
  });
});
