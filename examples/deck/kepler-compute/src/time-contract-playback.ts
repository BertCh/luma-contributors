// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Deterministic playback / seek / loop state machine over exact bigint time ticks (any unit, for
 * example microseconds). Mirrors Kepler's AnimationController (components/.../animation-control/
 * animation-controller.js) frame rules, with these deliberate differences:
 * - All arithmetic is bigint. The per-frame delta is domainWidth * speed / 600 with the fractional
 *   remainder CARRIED (never rounded), so 600 frames at speed 1 advance exactly one domain width.
 *   Kepler uses floating point deltas, which agree whenever the delta is integral.
 * - Wall time is explicit (advanceTime); Kepler runs one step per requestAnimationFrame.
 * - Zero-width domains and speed 0 are defined (stalled), where Kepler silently stays still.
 *
 * Modes: 'free' (fixed-width moving window), 'incremental' (fixed start, growing end), 'point'
 * (zero-width cursor) and 'snap' (Kepler's `interval`: jump through sorted steps).
 * `overhang: 'kepler'` keeps Kepler's raw windows, which begin at domainStart - width and end past
 * the domain end on the last frame; `getEffectiveWindow` always clips to the domain.
 * `overhang: 'contain'` keeps the window itself inside the domain: the last frame clamps to
 * [domainEnd - width, domainEnd] before wrapping.
 */

export type PlaybackMode = 'free' | 'incremental' | 'point' | 'snap';
export type PlaybackOverhang = 'kepler' | 'contain';
export type FrameGapPolicy = 'coalesce' | 'catch-up';

export const PLAYBACK_BASE_FRAMES = 600n;
export const PLAYBACK_FRAMES_PER_SECOND = 60n;
const MICROSECONDS_PER_SECOND = 1_000_000n;
const SNAP_SWEEP_MICROSECONDS = 10_000_000n;
const SPEED_SCALE = 1000n;

export type PlaybackConfiguration = {
  mode: PlaybackMode;
  domainStart: bigint;
  domainEnd: bigint;
  /** Window width in ticks (ignored for point and snap; incremental starts at this width). */
  windowWidth?: bigint;
  /** Speed multiplier in thousandths: 1000n is 1x. */
  speedMilli?: bigint;
  overhang?: PlaybackOverhang;
  loop?: boolean;
  /** Sorted ascending steps for 'snap' (and the bin end extrapolation of 'incremental'). */
  steps?: readonly bigint[];
  /** Kepler minimumIncrement: lower bound for the delta and reset width of incremental. */
  minimumIncrement?: bigint;
  frameGapPolicy?: FrameGapPolicy;
  maximumCatchUpFrames?: number;
};

export type PlaybackState = {
  readonly configuration: Required<Omit<PlaybackConfiguration, 'steps' | 'minimumIncrement'>> & {
    steps: readonly bigint[];
    minimumIncrement: bigint | null;
  };
  readonly windowStart: bigint;
  readonly windowEnd: bigint;
  readonly isPlaying: boolean;
  readonly hasEnded: boolean;
  readonly loopCount: number;
  /** Fractional delta carry numerator, in units of 1/(600*1000) tick. */
  readonly deltaCarry: bigint;
  /** Wall-time accumulator numerator, see advanceTime. */
  readonly timeCarry: bigint;
};

export function createPlaybackState(configuration: PlaybackConfiguration): PlaybackState {
  // Domain and speed invariants are caller errors, not runtime-validated user input.
  if (configuration.domainEnd < configuration.domainStart)
    throw new Error('Domain end precedes start');
  const resolved = {
    mode: configuration.mode,
    domainStart: configuration.domainStart,
    domainEnd: configuration.domainEnd,
    windowWidth:
      configuration.mode === 'point' || configuration.mode === 'snap'
        ? 0n
        : (configuration.windowWidth ?? 0n),
    speedMilli: configuration.speedMilli ?? SPEED_SCALE,
    overhang: configuration.overhang ?? 'kepler',
    loop: configuration.loop ?? true,
    steps: configuration.steps ?? [],
    minimumIncrement: configuration.minimumIncrement ?? null,
    frameGapPolicy: configuration.frameGapPolicy ?? 'coalesce',
    maximumCatchUpFrames: configuration.maximumCatchUpFrames ?? 600
  };
  if (resolved.windowWidth < 0n || resolved.speedMilli < 0n)
    throw new Error('Negative width or speed');
  const state: PlaybackState = {
    configuration: resolved,
    windowStart: 0n,
    windowEnd: 0n,
    isPlaying: false,
    hasEnded: false,
    loopCount: 0,
    deltaCarry: 0n,
    timeCarry: 0n
  };
  return resetPlayback(state);
}

function getClampedWidth(state: PlaybackState): bigint {
  const {windowWidth, domainStart, domainEnd, overhang} = state.configuration;
  const domainWidth = domainEnd - domainStart;
  return overhang === 'contain' && windowWidth > domainWidth ? domainWidth : windowWidth;
}

/** Kepler _resetAnimationByDomain. Playing state is preserved. */
export function resetPlayback(state: PlaybackState): PlaybackState {
  const {mode, domainStart, overhang, steps, minimumIncrement} = state.configuration;
  const width = getClampedWidth(state);
  let windowStart = domainStart;
  let windowEnd = domainStart;
  if (mode === 'free') {
    if (overhang === 'kepler') {
      windowStart = domainStart - width;
      windowEnd = domainStart;
    } else {
      windowEnd = domainStart + width;
    }
  } else if (mode === 'incremental') {
    const increment = minimumIncrement === null ? 1n : minimumIncrement;
    windowEnd =
      minimumIncrement === null
        ? domainStart + 1n
        : minOf(state.configuration.domainEnd, domainStart + increment);
  } else if (mode === 'snap' && steps.length) {
    windowStart = steps[0];
    windowEnd = steps[0];
  }
  return {
    ...state,
    windowStart,
    windowEnd,
    hasEnded: false,
    deltaCarry: 0n,
    timeCarry: 0n
  };
}

export function playPlayback(state: PlaybackState): PlaybackState {
  // Kepler only starts when speed > 0; a snap needs steps. A zero-width domain cannot advance.
  const {speedMilli, mode, steps} = state.configuration;
  if (speedMilli <= 0n || (mode === 'snap' && !steps.length)) return {...state, isPlaying: false};
  return {...state, isPlaying: true, hasEnded: false};
}

export function pausePlayback(state: PlaybackState): PlaybackState {
  return {...state, isPlaying: false, timeCarry: 0n};
}

/** Speed change mid-play keeps the position and the wall-time phase but drops the old-speed fraction. */
export function setPlaybackSpeed(state: PlaybackState, speedMilli: bigint): PlaybackState {
  const configuration = {...state.configuration, speedMilli};
  return {
    ...state,
    configuration,
    deltaCarry: 0n,
    isPlaying: state.isPlaying && speedMilli > 0n
  };
}

export function setPlaybackWindowWidth(state: PlaybackState, windowWidth: bigint): PlaybackState {
  if (state.configuration.mode === 'point' || state.configuration.mode === 'snap') return state;
  const configuration = {...state.configuration, windowWidth};
  const next = {...state, configuration};
  const width = getClampedWidth(next);
  const {domainStart, domainEnd, overhang} = configuration;
  let windowStart = state.windowStart;
  if (configuration.mode === 'incremental') {
    return {...next, windowEnd: minOf(domainEnd, windowStart + width)};
  }
  if (overhang === 'contain') {
    windowStart = clamp(windowStart, domainStart, domainEnd - width);
  }
  return {...next, windowStart, windowEnd: windowStart + width};
}

/** Seek the window start (free), end (incremental), cursor (point) or step (snap) to a time. */
export function seekPlayback(state: PlaybackState, time: bigint): PlaybackState {
  const {mode, domainStart, domainEnd, overhang, steps} = state.configuration;
  const width = getClampedWidth(state);
  const base = {...state, deltaCarry: 0n, hasEnded: false};
  if (mode === 'free') {
    const windowStart =
      overhang === 'contain'
        ? clamp(time, domainStart, domainEnd - width)
        : clamp(time, domainStart - width, domainEnd);
    return {...base, windowStart, windowEnd: windowStart + width};
  }
  if (mode === 'incremental') {
    return {...base, windowEnd: clamp(time, state.windowStart, domainEnd)};
  }
  if (mode === 'point') {
    const cursor = clamp(time, domainStart, domainEnd);
    return {...base, windowStart: cursor, windowEnd: cursor};
  }
  if (!steps.length) return base;
  const index = Math.max(0, findLastStepIndexAtOrBefore(steps, time));
  return {...base, windowStart: steps[index], windowEnd: steps[index]};
}

/** Exact per-frame delta with remainder carry. */
function takeDelta(state: PlaybackState): {delta: bigint; deltaCarry: bigint} {
  const {domainStart, domainEnd, speedMilli, minimumIncrement} = state.configuration;
  const denominator = PLAYBACK_BASE_FRAMES * SPEED_SCALE;
  const numerator = state.deltaCarry + (domainEnd - domainStart) * speedMilli;
  let delta = numerator / denominator;
  const deltaCarry = numerator % denominator;
  // Kepler: delta = max(rawDelta, minimumIncrement); the floor does not consume the carry.
  if (minimumIncrement !== null && delta < minimumIncrement) delta = minimumIncrement;
  return {delta, deltaCarry};
}

/** The delta is zero for a zero-width domain or speed 0: playback cannot make progress. */
export function isPlaybackStalled(state: PlaybackState): boolean {
  const {mode, steps, domainStart, domainEnd, speedMilli, minimumIncrement} = state.configuration;
  if (mode === 'snap') return steps.length < 2 || speedMilli <= 0n;
  if (speedMilli <= 0n) return true;
  return domainEnd === domainStart && (minimumIncrement === null || minimumIncrement <= 0n);
}

/** One Kepler frame (_nextFrameByDomain / _nextFrameByTimeStep). */
export function stepPlaybackFrame(state: PlaybackState): PlaybackState {
  const {mode, domainStart, domainEnd, overhang, loop, steps, minimumIncrement} =
    state.configuration;
  if (isPlaybackStalled(state)) return state;
  if (mode === 'snap') {
    const index = findFirstStepIndexAtOrAfter(steps, state.windowStart);
    const isLast = index >= steps.length - 1;
    if (isLast && !loop) return {...state, isPlaying: false, hasEnded: true};
    const nextIndex = isLast ? 0 : index + 1;
    return {
      ...state,
      windowStart: steps[nextIndex],
      windowEnd: steps[nextIndex],
      loopCount: state.loopCount + (isLast ? 1 : 0)
    };
  }
  const {delta, deltaCarry} = takeDelta(state);
  const width = getClampedWidth(state);

  if (mode === 'point') {
    const next = state.windowStart + delta;
    if (next > domainEnd) {
      if (!loop)
        return {
          ...state,
          windowStart: domainEnd,
          windowEnd: domainEnd,
          isPlaying: false,
          hasEnded: true,
          deltaCarry
        };
      return {
        ...state,
        windowStart: domainStart,
        windowEnd: domainStart,
        loopCount: state.loopCount + 1,
        deltaCarry: 0n
      };
    }
    return {...state, windowStart: next, windowEnd: next, deltaCarry};
  }

  if (mode === 'incremental') {
    let endOfSweep = domainEnd;
    if (minimumIncrement === null && steps.length > 1) {
      const last = steps[steps.length - 1];
      const endOfLastBin = last + (last - steps[steps.length - 2]);
      if (endOfLastBin > endOfSweep) endOfSweep = endOfLastBin;
    }
    if (state.windowEnd >= endOfSweep) {
      if (!loop) return {...state, isPlaying: false, hasEnded: true, deltaCarry};
      const resetEnd =
        minimumIncrement === null
          ? state.windowStart + 1n
          : minOf(domainEnd, state.windowStart + minimumIncrement);
      return {...state, windowEnd: resetEnd, loopCount: state.loopCount + 1, deltaCarry: 0n};
    }
    return {...state, windowEnd: minOf(state.windowEnd + delta, endOfSweep), deltaCarry};
  }

  // free
  if (overhang === 'kepler') {
    const isLastFrame = state.windowStart + delta > domainEnd;
    if (isLastFrame) {
      if (!loop) return {...state, isPlaying: false, hasEnded: true, deltaCarry};
      const windowStart = domainStart - width;
      return {
        ...state,
        windowStart,
        windowEnd: windowStart + width,
        loopCount: state.loopCount + 1,
        deltaCarry: 0n
      };
    }
    const windowStart = state.windowStart + delta;
    return {...state, windowStart, windowEnd: windowStart + width, deltaCarry};
  }
  // contain: the last in-domain window is emitted (clamped) before the next frame wraps.
  const maximumStart = domainEnd - width;
  if (state.windowStart >= maximumStart) {
    if (!loop) return {...state, isPlaying: false, hasEnded: true, deltaCarry};
    return {
      ...state,
      windowStart: domainStart,
      windowEnd: domainStart + width,
      loopCount: state.loopCount + 1,
      deltaCarry: 0n
    };
  }
  const windowStart = minOf(state.windowStart + delta, maximumStart);
  return {...state, windowStart, windowEnd: windowStart + width, deltaCarry};
}

/** Apply a number of frames (or snap steps) while playing. Stops early when a non-looping run ends. */
export function advancePlaybackFrames(state: PlaybackState, frameCount: number): PlaybackState {
  let current = state;
  for (let frame = 0; frame < frameCount && current.isPlaying; frame++) {
    current = stepPlaybackFrame(current);
  }
  return current;
}

/**
 * Advance by elapsed wall time in microseconds. Free/incremental/point run at 60 frames per second;
 * snap runs one step every 10 s / (stepCount * speed). The accumulator is exact (no float drift).
 * 'coalesce' (Kepler: one step per animation frame regardless of the gap) applies at most one frame
 * and discards the surplus; 'catch-up' applies every due frame up to maximumCatchUpFrames.
 */
export function advancePlaybackTime(
  state: PlaybackState,
  elapsedMicroseconds: bigint
): PlaybackState {
  if (!state.isPlaying || elapsedMicroseconds <= 0n) return state;
  const {mode, steps, speedMilli, frameGapPolicy, maximumCatchUpFrames} = state.configuration;
  let numeratorPerMicrosecond: bigint;
  let denominator: bigint;
  if (mode === 'snap') {
    numeratorPerMicrosecond = BigInt(steps.length) * speedMilli;
    denominator = SNAP_SWEEP_MICROSECONDS * SPEED_SCALE;
  } else {
    numeratorPerMicrosecond = PLAYBACK_FRAMES_PER_SECOND;
    denominator = MICROSECONDS_PER_SECOND;
  }
  const accumulated = state.timeCarry + elapsedMicroseconds * numeratorPerMicrosecond;
  const dueFrames = accumulated / denominator;
  let timeCarry = accumulated % denominator;
  let frames = dueFrames;
  if (frameGapPolicy === 'coalesce') {
    if (frames > 1n) timeCarry = 0n;
    frames = frames > 1n ? 1n : frames;
  } else if (frames > BigInt(maximumCatchUpFrames)) {
    frames = BigInt(maximumCatchUpFrames);
    timeCarry = 0n;
  }
  const advanced = advancePlaybackFrames({...state, timeCarry}, Number(frames));
  return {...advanced, timeCarry: advanced.isPlaying ? timeCarry : 0n};
}

/** The window clipped to the domain: the interval a time filter should actually apply. */
export function getEffectiveWindow(state: PlaybackState): {start: bigint; end: bigint} {
  const {domainStart, domainEnd} = state.configuration;
  return {
    start: clamp(state.windowStart, domainStart, domainEnd),
    end: clamp(state.windowEnd, domainStart, domainEnd)
  };
}

function minOf(a: bigint, b: bigint): bigint {
  return a < b ? a : b;
}

function clamp(value: bigint, minimum: bigint, maximum: bigint): bigint {
  return value < minimum ? minimum : value > maximum ? maximum : value;
}

/** d3 bisectLeft on bigint: first index whose step >= value (steps.length when none). */
export function findFirstStepIndexAtOrAfter(steps: readonly bigint[], value: bigint): number {
  let low = 0;
  let high = steps.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (steps[middle] < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function findLastStepIndexAtOrBefore(steps: readonly bigint[], value: bigint): number {
  let low = 0;
  let high = steps.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (steps[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}
