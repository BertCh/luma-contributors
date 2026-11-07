// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState, useSyncExternalStore} from 'react';
import {
  ExactPlaybackDriver,
  getPlaybackModeFromKeplerWindow,
  type ExactPlaybackHost,
  type KeplerTimeFilterLike
} from './time-contract-native-bridge';
import {parseTimeZoneSpecification} from './time-contract-zone';
import {
  getTimeContractRevision,
  getTimeContractSettings,
  setTimeContractSettings,
  subscribeTimeContractSettings
} from './time-contract-settings';
import type {PlaybackMode} from './time-contract-playback';

export type TimeContractFilterRecord = Readonly<{
  index: number;
  label: string;
  filter: KeplerTimeFilterLike;
  /** Histogram thresholds in milliseconds, for snap playback; empty when unavailable. */
  steps: readonly number[];
}>;

export type TimeContractPanelProps = Readonly<{
  filters: readonly TimeContractFilterRecord[];
  getHost(index: number): ExactPlaybackHost;
  /** Re-plans histograms after the zone changes. */
  onZoneChanged(): void;
}>;

/**
 * Companion to Kepler's native time widget. Zone: local-wall-clock day/hour/minute histogram
 * edges (native buckets by UTC; Kepler applies a timezone only to labels). Exact playback: drives
 * the native time filter value with the bigint playback machine through the same action the native
 * AnimationController dispatches, so the native slider and every resident consumer follow it.
 */
export function TimeContractPanel({filters, getHost, onZoneChanged}: TimeContractPanelProps) {
  useSyncExternalStore(subscribeTimeContractSettings, getTimeContractRevision);
  const settings = getTimeContractSettings();
  const [zoneText, setZoneText] = useState(settings.timeZone);
  const [zoneError, setZoneError] = useState('');
  const [selected, setSelected] = useState(0);
  const [mode, setMode] = useState<PlaybackMode | 'native'>('native');
  const [overhang, setOverhang] = useState<'kepler' | 'contain'>('kepler');
  const [loop, setLoop] = useState(true);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('');
  const driverRef = useRef<ExactPlaybackDriver | null>(null);
  useEffect(() => () => driverRef.current?.destroy(), []);
  const record = filters[Math.min(selected, filters.length - 1)];

  function applyZone(): void {
    try {
      parseTimeZoneSpecification(zoneText);
      setZoneError('');
      setTimeContractSettings({timeZone: zoneText});
      onZoneChanged();
    } catch {
      setZoneError('Unknown time zone');
    }
  }

  function startPlayback(): void {
    if (!record) return;
    driverRef.current?.destroy();
    const driver = new ExactPlaybackDriver(getHost(record.index));
    driverRef.current = driver;
    const chosen: PlaybackMode =
      mode === 'native' ? getPlaybackModeFromKeplerWindow(record.filter.animationWindow) : mode;
    const started = driver.start(chosen, {overhang, loop, steps: record.steps});
    setRunning(started);
    setMessage(started ? '' : 'This filter has no finite domain or histogram steps to play.');
  }

  function stopPlayback(): void {
    driverRef.current?.pause();
    setRunning(false);
  }

  return (
    <section aria-label="Time contract" data-testid="time-contract-panel">
      <h2>Time contract</h2>
      <label>
        Histogram time zone{' '}
        <input
          value={zoneText}
          data-testid="time-contract-zone-input"
          onChange={event => setZoneText(event.target.value)}
          placeholder="UTC, +05:30 or America/New_York"
        />
      </label>{' '}
      <button type="button" data-testid="time-contract-zone-apply" onClick={applyZone}>
        Apply zone
      </button>
      {zoneError ? <p role="alert">{zoneError}</p> : null}
      <p className="hint">
        1-day, 1-hour and 1-minute histograms bucket by local wall-clock time in this zone; native
        Kepler buckets by UTC. Other intervals stay UTC.
      </p>
      {filters.length ? (
        <fieldset>
          <legend>Exact playback</legend>
          <label>
            Time filter{' '}
            <select
              value={selected}
              data-testid="time-contract-filter-select"
              onChange={event => setSelected(Number(event.target.value))}
            >
              {filters.map((candidate, index) => (
                <option key={candidate.index} value={index}>
                  {candidate.label}
                </option>
              ))}
            </select>
          </label>{' '}
          <label>
            Mode{' '}
            <select
              value={mode}
              data-testid="time-contract-mode-select"
              onChange={event => setMode(event.target.value as PlaybackMode | 'native')}
            >
              <option value="native">Follow the native window setting</option>
              <option value="free">Free</option>
              <option value="incremental">Incremental</option>
              <option value="point">Point</option>
              <option value="snap">Snap to histogram bins</option>
            </select>
          </label>{' '}
          <label>
            Overhang{' '}
            <select
              value={overhang}
              onChange={event => setOverhang(event.target.value as 'kepler' | 'contain')}
            >
              <option value="kepler">Kepler (window may extend past the end)</option>
              <option value="contain">Contain (clamp to the domain)</option>
            </select>
          </label>{' '}
          <label>
            <input
              type="checkbox"
              checked={loop}
              onChange={event => setLoop(event.target.checked)}
            />{' '}
            Loop
          </label>{' '}
          {running ? (
            <button type="button" data-testid="time-contract-pause" onClick={stopPlayback}>
              Pause
            </button>
          ) : (
            <button type="button" data-testid="time-contract-play" onClick={startPlayback}>
              Play exactly
            </button>
          )}
          {message ? <p role="status">{message}</p> : null}
        </fieldset>
      ) : (
        <p className="hint">Add a time filter to use exact playback.</p>
      )}
    </section>
  );
}
