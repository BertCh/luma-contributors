// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  advancePlaybackTime,
  createPlaybackState,
  getEffectiveWindow,
  pausePlayback,
  playPlayback,
  resetPlayback,
  seekPlayback,
  setPlaybackSpeed,
  type PlaybackMode,
  type PlaybackOverhang,
  type PlaybackState
} from './time-contract-playback';

/** The subset of a pinned Kepler time or range filter the exact playback driver reads. */
export type KeplerTimeFilterLike = {
  domain?: readonly number[] | null;
  value?: unknown;
  speed?: number;
  isAnimating?: boolean;
  animationWindow?: string;
  plotType?: {interval?: string};
};

export type NativePlaybackOptions = {
  overhang?: PlaybackOverhang;
  loop?: boolean;
  /** Histogram thresholds in the filter unit (milliseconds); required for snap (Kepler interval window). */
  steps?: readonly number[];
};

const MAXIMUM_SPEED_MILLI = 1_000_000n;

/** Kepler animationWindow names to contract modes: free, incremental, point, interval -> snap. */
export function getPlaybackModeFromKeplerWindow(animationWindow: string | undefined): PlaybackMode {
  if (animationWindow === 'incremental') return 'incremental';
  if (animationWindow === 'point') return 'point';
  if (animationWindow === 'interval') return 'snap';
  return 'free';
}

/** Kepler speed is a floating multiplier; the contract uses exact thousandths. */
export function getSpeedMilliFromKeplerSpeed(speed: number | undefined): bigint {
  if (speed === undefined || !Number.isFinite(speed) || speed < 0) return 1000n;
  const milli = BigInt(Math.round(speed * 1000));
  return milli > MAXIMUM_SPEED_MILLI ? MAXIMUM_SPEED_MILLI : milli;
}

/** Builds a playback state from the native filter. Returns null when the filter cannot animate. */
export function makePlaybackStateFromKeplerFilter(
  filter: KeplerTimeFilterLike,
  mode: PlaybackMode,
  options: NativePlaybackOptions = {}
): PlaybackState | null {
  const domain = filter.domain;
  if (!domain || domain.length !== 2 || !domain.every(Number.isSafeInteger)) return null;
  const value = filter.value as readonly number[] | undefined;
  const width =
    value && value.length === 2 && value.every(Number.isSafeInteger)
      ? BigInt(value[1]) - BigInt(value[0])
      : 0n;
  if (mode === 'snap' && (!options.steps || options.steps.length === 0)) return null;
  return createPlaybackState({
    mode,
    domainStart: BigInt(domain[0]),
    domainEnd: BigInt(domain[1]),
    windowWidth: width > 0n ? width : undefined,
    speedMilli: getSpeedMilliFromKeplerSpeed(filter.speed),
    overhang: options.overhang ?? 'kepler',
    loop: options.loop ?? true,
    steps: options.steps?.map(step => BigInt(step))
  });
}

/** The [start, end] milliseconds a native time filter value takes for the effective window. */
export function getKeplerValueFromPlayback(state: PlaybackState): [number, number] {
  const window = getEffectiveWindow(state);
  return [Number(window.start), Number(window.end)];
}

export type ExactPlaybackHost = {
  getFilter(): KeplerTimeFilterLike | undefined;
  /** Dispatches the same native action the Kepler AnimationController uses (setFilterAnimationTime value). */
  setFilterValue(value: [number, number]): void;
  /** Stops the native controller so the two do not both drive the filter. */
  pauseNativeAnimation(): void;
  requestFrame(callback: (timestampMilliseconds: number) => void): number;
  cancelFrame(handle: number): void;
};

/**
 * Drives a native Kepler time filter from the exact bigint playback machine. Every frame writes the
 * native filter value through the host, so the native slider, histogram, resident masks and
 * labels follow it. Elapsed time is exact microseconds from the frame timestamps.
 */
export class ExactPlaybackDriver {
  private state: PlaybackState | null = null;
  private handle = 0;
  private lastTimestamp: number | null = null;
  private running = false;
  frameCount = 0;

  constructor(private readonly host: ExactPlaybackHost) {}

  get isRunning(): boolean {
    return this.running;
  }

  get currentState(): PlaybackState | null {
    return this.state;
  }

  start(mode: PlaybackMode, options: NativePlaybackOptions = {}): boolean {
    const filter = this.host.getFilter();
    if (!filter) return false;
    const created = makePlaybackStateFromKeplerFilter(filter, mode, options);
    if (!created) return false;
    this.stop();
    if (filter.isAnimating) this.host.pauseNativeAnimation();
    this.state = playPlayback(created);
    this.running = true;
    this.lastTimestamp = null;
    this.publish();
    this.schedule();
    return true;
  }

  pause(): void {
    if (this.state) this.state = pausePlayback(this.state);
    this.stop();
  }

  reset(): void {
    if (!this.state) return;
    this.state = resetPlayback(this.state);
    this.publish();
  }

  seek(time: number): void {
    if (!this.state) return;
    this.state = seekPlayback(this.state, BigInt(time));
    this.publish();
  }

  setSpeed(speed: number): void {
    if (this.state) this.state = setPlaybackSpeed(this.state, getSpeedMilliFromKeplerSpeed(speed));
  }

  /** Advance by elapsed wall time without a scheduler; used by specs and by the frame callback. */
  tick(elapsedMicroseconds: bigint): void {
    if (!this.state) return;
    this.state = advancePlaybackTime(this.state, elapsedMicroseconds);
    this.frameCount++;
    this.publish();
    if (!this.state.isPlaying) this.stop();
  }

  destroy(): void {
    this.stop();
    this.state = null;
  }

  private stop(): void {
    if (this.running) this.host.cancelFrame(this.handle);
    this.running = false;
  }

  private schedule(): void {
    this.handle = this.host.requestFrame(timestamp => {
      if (!this.running) return;
      const previous = this.lastTimestamp;
      this.lastTimestamp = timestamp;
      if (previous !== null)
        this.tick(BigInt(Math.max(0, Math.round((timestamp - previous) * 1000))));
      if (this.running) this.schedule();
    });
  }

  private publish(): void {
    if (this.state) this.host.setFilterValue(getKeplerValueFromPlayback(this.state));
  }
}
