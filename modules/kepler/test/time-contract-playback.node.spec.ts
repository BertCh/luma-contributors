// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  advancePlaybackFrames,
  advancePlaybackTime,
  createPlaybackState,
  getEffectiveWindow,
  isPlaybackStalled,
  pausePlayback,
  playPlayback,
  resetPlayback,
  seekPlayback,
  setPlaybackSpeed,
  setPlaybackWindowWidth,
  stepPlaybackFrame,
  type PlaybackConfiguration,
  type PlaybackState
} from '../../../examples/deck/kepler-compute/src/time-contract-playback';

type KeplerProps = Record<string, unknown>;
type KeplerController = {_nextFrameByDomain: (this: {props: KeplerProps}) => unknown};

async function loadKeplerNextFrame(): Promise<null | ((props: KeplerProps) => unknown)> {
  try {
    const path = new URL(
      '../../../node_modules/@kepler.gl/components/dist/esm/common/animation-control/animation-controller.js',
      import.meta.url
    ).pathname;
    const module = await import(/* @vite-ignore */ path);
    const Controller = module.default() as unknown as {prototype: KeplerController};
    return props => Controller.prototype._nextFrameByDomain.call({props});
  } catch {
    return null;
  }
}

const keplerNextFrame = await loadKeplerNextFrame();

function playing(configuration: PlaybackConfiguration): PlaybackState {
  return playPlayback(createPlaybackState(configuration));
}

describe('exact frame arithmetic', () => {
  it('600 frames at 1x advance exactly one domain width for any bigint microsecond domain', () => {
    const domainStart = 1_700_000_000_123_457n;
    const domainEnd = domainStart + 86_400_000_000n * 7n + 13n;
    let state = playing({mode: 'point', domainStart, domainEnd});
    state = advancePlaybackFrames(state, 599);
    // Frames have not yet reached the end: 599 * (width/600) rounded down plus carry.
    expect(state.windowStart).toBeLessThan(domainEnd);
    state = advancePlaybackFrames(state, 1);
    expect(state.windowStart).toBe(domainEnd);
    expect(state.loopCount).toBe(0);
    state = stepPlaybackFrame(state);
    expect(state.windowStart).toBe(domainStart);
    expect(state.loopCount).toBe(1);
  });

  it('carries fractional deltas exactly across a long run (no drift)', () => {
    const domainEnd = 1_000_003n;
    let state = playing({mode: 'point', domainStart: 0n, domainEnd, speedMilli: 1500n});
    let expected = 0n;
    for (let frame = 1n; frame < 400n; frame++) {
      state = stepPlaybackFrame(state);
      expected = (frame * domainEnd * 1500n) / 600_000n;
      expect(state.windowStart).toBe(expected);
    }
  });

  it('speed 0 does not start and zero-width domains stall', () => {
    const idle = playing({mode: 'free', domainStart: 0n, domainEnd: 100n, speedMilli: 0n});
    expect(idle.isPlaying).toBe(false);
    const zero = playing({mode: 'point', domainStart: 5n, domainEnd: 5n});
    expect(isPlaybackStalled(zero)).toBe(true);
    expect(advancePlaybackFrames(zero, 50)).toEqual(zero);
  });
});

describe('free mode', () => {
  const base: PlaybackConfiguration = {
    mode: 'free',
    domainStart: 0n,
    domainEnd: 6000n,
    windowWidth: 1000n
  };

  it('kepler overhang starts before the domain, overhangs past the end, then loops', () => {
    let state = createPlaybackState(base);
    expect([state.windowStart, state.windowEnd]).toEqual([-1000n, 0n]);
    state = playPlayback(state);
    const starts: bigint[] = [];
    for (let frame = 0; frame < 702; frame++) {
      state = stepPlaybackFrame(state);
      starts.push(state.windowStart);
    }
    expect(starts[0]).toBe(-990n);
    // delta is 10: start reaches 6000 on frame 700; the next frame (6010 > 6000) loops.
    expect(starts[699]).toBe(6000n);
    expect(starts[700]).toBe(-1000n);
    expect(getEffectiveWindow(state)).toEqual({start: 0n, end: 10n});
  });

  it('contain overhang never leaves the domain and emits the final window before wrapping', () => {
    let state = playing({...base, overhang: 'contain'});
    expect([state.windowStart, state.windowEnd]).toEqual([0n, 1000n]);
    let wrapped = false;
    for (let frame = 0; frame < 1500; frame++) {
      state = stepPlaybackFrame(state);
      expect(state.windowStart).toBeGreaterThanOrEqual(0n);
      expect(state.windowEnd).toBeLessThanOrEqual(6000n);
      expect(state.windowEnd - state.windowStart).toBe(1000n);
      if (state.windowEnd === 6000n) wrapped = true;
    }
    expect(wrapped).toBe(true);
    expect(state.loopCount).toBeGreaterThan(0);
  });

  it('contain: window wider than the domain clamps to the domain and stays valid', () => {
    let state = playing({...base, overhang: 'contain', windowWidth: 99_999n});
    state = advancePlaybackFrames(state, 25);
    expect([state.windowStart, state.windowEnd]).toEqual([0n, 6000n]);
  });

  it('non-looping playback clamps at the end and stops', () => {
    let state = playing({...base, overhang: 'contain', loop: false});
    state = advancePlaybackFrames(state, 5000);
    expect(state.isPlaying).toBe(false);
    expect(state.hasEnded).toBe(true);
    expect([state.windowStart, state.windowEnd]).toEqual([5000n, 6000n]);
    expect(advancePlaybackFrames(state, 10)).toEqual(state);
  });

  it('seek clamps per overhang and keeps the width; seeking is idempotent', () => {
    const kepler = createPlaybackState(base);
    expect(seekPlayback(kepler, -5000n).windowStart).toBe(-1000n);
    expect(seekPlayback(kepler, 99_999n).windowStart).toBe(6000n);
    const contain = createPlaybackState({...base, overhang: 'contain'});
    const sought = seekPlayback(contain, 99_999n);
    expect([sought.windowStart, sought.windowEnd]).toEqual([5000n, 6000n]);
    expect(seekPlayback(sought, 99_999n)).toEqual(sought);
    expect(seekPlayback(contain, 2500n).windowEnd).toBe(3500n);
  });

  it('window width changes keep the start and clamp in contain mode', () => {
    let state = seekPlayback(createPlaybackState({...base, overhang: 'contain'}), 5000n);
    state = setPlaybackWindowWidth(state, 3000n);
    expect([state.windowStart, state.windowEnd]).toEqual([3000n, 6000n]);
    state = setPlaybackWindowWidth(state, 0n);
    expect(state.windowEnd).toBe(state.windowStart);
  });

  it('speed change mid-play preserves position and uses the new delta immediately', () => {
    let state = playing({...base, overhang: 'contain'});
    state = advancePlaybackFrames(state, 10);
    expect(state.windowStart).toBe(100n);
    state = setPlaybackSpeed(state, 2000n);
    state = stepPlaybackFrame(state);
    expect(state.windowStart).toBe(120n);
    state = setPlaybackSpeed(state, 0n);
    expect(state.isPlaying).toBe(false);
    expect(advancePlaybackFrames(state, 5).windowStart).toBe(120n);
  });
});

describe('incremental, point and snap', () => {
  it('incremental grows the end to the domain end, then resets to start + 1', () => {
    let state = playing({mode: 'incremental', domainStart: 0n, domainEnd: 600n, windowWidth: 0n});
    expect([state.windowStart, state.windowEnd]).toEqual([0n, 1n]);
    state = advancePlaybackFrames(state, 1);
    expect(state.windowEnd).toBe(2n);
    state = advancePlaybackFrames(state, 1000);
    expect(state.windowStart).toBe(0n);
    expect(state.loopCount).toBeGreaterThan(0);
    // The window never exceeds the domain.
    let current = state;
    for (let frame = 0; frame < 3000; frame++) {
      current = stepPlaybackFrame(current);
      expect(current.windowEnd).toBeLessThanOrEqual(600n);
    }
  });

  it('incremental honors the extrapolated last bin like Kepler and the minimum increment', () => {
    let state = playing({
      mode: 'incremental',
      domainStart: 0n,
      domainEnd: 600n,
      steps: [0n, 200n, 400n]
    });
    state = seekPlayback(state, 600n);
    state = stepPlaybackFrame(state);
    expect(state.windowEnd).toBe(1n); // last bin ends at 600 = domain end: sweep done, reset to start + 1
    const extended = playing({
      mode: 'incremental',
      domainStart: 0n,
      domainEnd: 590_000n,
      steps: [0n, 200_000n, 400_000n, 550_000n]
    });
    let current = seekPlayback(extended, 590_000n);
    current = stepPlaybackFrame(current);
    // Last bin ends at 700000 > domain end: Kepler keeps sweeping past the domain end.
    expect(current.windowEnd).toBe(590_983n);
    const increment = playing({
      mode: 'incremental',
      domainStart: 0n,
      domainEnd: 600n,
      minimumIncrement: 50n
    });
    expect(increment.windowEnd).toBe(50n);
    expect(stepPlaybackFrame(increment).windowEnd).toBe(100n);
  });

  it('point reaches the domain end exactly then loops; seek clamps', () => {
    const state = createPlaybackState({mode: 'point', domainStart: 100n, domainEnd: 700n});
    expect(seekPlayback(state, 5000n).windowStart).toBe(700n);
    expect(seekPlayback(state, -5n).windowStart).toBe(100n);
    const looped = stepPlaybackFrame(playPlayback(seekPlayback(state, 700n)));
    expect(looped.windowStart).toBe(100n);
  });

  it('snap walks sorted steps with wrap and nearest-previous seek', () => {
    const steps = [10n, 20n, 40n, 80n];
    let state = playing({mode: 'snap', domainStart: 10n, domainEnd: 80n, steps});
    const visited: bigint[] = [];
    for (let frame = 0; frame < 6; frame++) {
      state = stepPlaybackFrame(state);
      visited.push(state.windowStart);
    }
    expect(visited).toEqual([20n, 40n, 80n, 10n, 20n, 40n]);
    expect(seekPlayback(state, 55n).windowStart).toBe(40n);
    expect(seekPlayback(state, 5n).windowStart).toBe(10n);
    expect(seekPlayback(state, 80n).windowStart).toBe(80n);
  });

  it('snap wall time: one step every 10 s / (steps * speed)', () => {
    const state = playing({
      mode: 'snap',
      domainStart: 10n,
      domainEnd: 80n,
      steps: [10n, 20n, 40n, 80n]
    });
    // 4 steps at 1x: 2.5 s per step.
    expect(advancePlaybackTime(state, 2_499_999n).windowStart).toBe(10n);
    const stepped = advancePlaybackTime(state, 2_500_000n);
    expect(stepped.windowStart).toBe(20n);
    const doubled = advancePlaybackTime(setPlaybackSpeed(state, 2000n), 1_250_000n);
    expect(doubled.windowStart).toBe(20n);
  });
});

describe('wall time and frame gaps', () => {
  const configuration: PlaybackConfiguration = {
    mode: 'point',
    domainStart: 0n,
    domainEnd: 600_000n
  };

  it('60 fps cadence is exact: 1 second of 1/60 s ticks gives exactly 60 frames', () => {
    let state = playing(configuration);
    for (let tick = 0; tick < 60; tick++) {
      // 1/60 s is 16666.666... us; alternate 16666/16667 so a second sums to exactly 1e6.
      state = advancePlaybackTime(state, tick % 3 === 2 ? 16_667n : 16_666n);
    }
    // 60 ticks sum to 999_990 us: 59 frames due, plus carry.
    expect(state.windowStart).toBe(59n * 1000n);
    state = advancePlaybackTime(state, 20n);
    expect(state.windowStart).toBe(60n * 1000n);
  });

  it('coalesce (Kepler) turns a long gap into one frame; catch-up applies due frames', () => {
    const coalesced = advancePlaybackTime(playing(configuration), 5_000_000n);
    expect(coalesced.windowStart).toBe(1000n);
    const catchUp = advancePlaybackTime(
      playing({...configuration, frameGapPolicy: 'catch-up'}),
      1_000_000n
    );
    expect(catchUp.windowStart).toBe(60_000n);
    const capped = advancePlaybackTime(
      playing({...configuration, frameGapPolicy: 'catch-up', maximumCatchUpFrames: 10}),
      1_000_000n
    );
    expect(capped.windowStart).toBe(10_000n);
  });

  it('paused state ignores time; pause resets the time phase', () => {
    const state = playing(configuration);
    const paused = pausePlayback(state);
    expect(advancePlaybackTime(paused, 10_000_000n)).toEqual(paused);
    expect(resetPlayback(state).windowStart).toBe(0n);
  });
});

describe('property style invariants', () => {
  function makeRandom(seed: number): () => number {
    let value = seed >>> 0;
    return () => {
      value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
      return value / 2 ** 32;
    };
  }

  it('random operation sequences keep contain windows inside the domain and widths constant', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const random = makeRandom(seed);
      const domainStart = BigInt(Math.floor(random() * 1e6)) - 500_000n;
      const domainEnd = domainStart + BigInt(Math.floor(random() * 5e9));
      const width = BigInt(Math.floor(random() * 2e9));
      let state = playing({
        mode: 'free',
        overhang: 'contain',
        domainStart,
        domainEnd,
        windowWidth: width
      });
      const effectiveWidth = width > domainEnd - domainStart ? domainEnd - domainStart : width;
      for (let operation = 0; operation < 200; operation++) {
        const choice = random();
        if (choice < 0.6) state = advancePlaybackFrames(state, Math.floor(random() * 50));
        else if (choice < 0.7)
          state = seekPlayback(state, domainStart - 100n + BigInt(Math.floor(random() * 6e9)));
        else if (choice < 0.8) state = setPlaybackSpeed(state, BigInt(Math.floor(random() * 9000)));
        else if (choice < 0.9) state = playPlayback(state);
        else state = advancePlaybackTime(state, BigInt(Math.floor(random() * 3e6)));
        expect(state.windowStart).toBeGreaterThanOrEqual(domainStart);
        expect(state.windowEnd).toBeLessThanOrEqual(domainEnd);
        expect(state.windowEnd - state.windowStart).toBe(effectiveWidth);
      }
    }
  });

  it('pure and deterministic: replaying the same operations gives identical states', () => {
    const run = () => {
      let state = playing({mode: 'incremental', domainStart: 0n, domainEnd: 123_456_789n});
      for (let step = 0; step < 300; step++) {
        state =
          step % 97 === 0 ? seekPlayback(state, BigInt(step) * 1000n) : stepPlaybackFrame(state);
      }
      return state;
    };
    expect(run()).toEqual(run());
  });

  it('input states are never mutated', () => {
    const state = playing({mode: 'free', domainStart: 0n, domainEnd: 1000n, windowWidth: 10n});
    const frozen = JSON.stringify(state, (_key, value) =>
      typeof value === 'bigint' ? value.toString() : value
    );
    stepPlaybackFrame(state);
    seekPlayback(state, 5n);
    setPlaybackWindowWidth(state, 30n);
    expect(
      JSON.stringify(state, (_key, value) => (typeof value === 'bigint' ? value.toString() : value))
    ).toBe(frozen);
  });
});

describe.skipIf(!keplerNextFrame)('agreement with installed Kepler _nextFrameByDomain', () => {
  const domain: [number, number] = [0, 6000];

  it('free mode produces Kepler window sequences while delta is integral', () => {
    let state = playing({mode: 'free', domainStart: 0n, domainEnd: 6000n, windowWidth: 1000n});
    let value: number[] = [Number(state.windowStart), Number(state.windowEnd)];
    for (let frame = 0; frame < 1300; frame++) {
      value = keplerNextFrame!({domain, value, speed: 1, animationWindow: 'free'}) as number[];
      state = stepPlaybackFrame(state);
      expect([Number(state.windowStart), Number(state.windowEnd)]).toEqual(value);
    }
  });

  it('free mode at 2x and incremental/point sequences match', () => {
    for (const mode of ['free', 'incremental', 'point'] as const) {
      let state = playing({
        mode,
        domainStart: 0n,
        domainEnd: 6000n,
        windowWidth: mode === 'free' ? 500n : 0n,
        speedMilli: 2000n
      });
      let value: number | number[] =
        mode === 'point'
          ? Number(state.windowStart)
          : [Number(state.windowStart), Number(state.windowEnd)];
      for (let frame = 0; frame < 700; frame++) {
        value = keplerNextFrame!({
          domain,
          value,
          speed: 2,
          animationWindow: mode
        }) as number | number[];
        state = stepPlaybackFrame(state);
        const actual =
          mode === 'point'
            ? Number(state.windowStart)
            : [Number(state.windowStart), Number(state.windowEnd)];
        expect(actual).toEqual(value);
      }
    }
  });

  it('incremental with an extrapolated last bin matches Kepler', () => {
    const steps = [0, 2000, 4000, 5500];
    let state = playing({
      mode: 'incremental',
      domainStart: 0n,
      domainEnd: 6000n,
      steps: steps.map(BigInt)
    });
    let value = [Number(state.windowStart), Number(state.windowEnd)];
    for (let frame = 0; frame < 1500; frame++) {
      value = keplerNextFrame!({
        domain,
        value,
        speed: 1,
        animationWindow: 'incremental',
        steps
      }) as number[];
      state = stepPlaybackFrame(state);
      expect([Number(state.windowStart), Number(state.windowEnd)]).toEqual(value);
    }
  });
});
