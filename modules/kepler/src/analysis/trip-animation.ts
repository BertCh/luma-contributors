// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type KeplerTripAnimationPlan =
  | {
      supported: true;
      currentTime: number;
      trailLength: number;
      fadeTrail: boolean;
      originMilliseconds: number;
      actualTimeErrorMilliseconds: number;
    }
  | {supported: false; reason: string};

/**
 * Pinned Kepler trip-layer.js:653/666 subtract domain0 from BOTH vertex and current time;
 * line663 converts trail seconds to milliseconds. animationConfig.currentTime is absolute at
 * that boundary. Our vertices already use their prepared origin, so subtract only that origin.
 */
export function planKeplerTripAnimation(
  source: {originMilliseconds: number; maximumTimeErrorMilliseconds: number},
  animation: {
    currentTimeMilliseconds: number;
    domainMilliseconds: readonly [number, number];
    trailLengthSeconds: number;
    fadeTrail?: boolean;
    scenegraphEnabled?: boolean;
    columnMode?: 'geojson' | 'table';
  }
): KeplerTripAnimationPlan {
  // Table-mode trips share the animation math: the caller's prepared grouping adapter already
  // supplies per-trip vertices, so only the prepared origin and trail conversion are needed.
  if (animation.scenegraphEnabled)
    return {supported: false, reason: 'Animated scenegraph objects require a separate renderer'};
  if (
    !Number.isSafeInteger(source.originMilliseconds) ||
    !Number.isFinite(source.maximumTimeErrorMilliseconds) ||
    source.maximumTimeErrorMilliseconds < 0 ||
    !Number.isFinite(animation.currentTimeMilliseconds) ||
    !animation.domainMilliseconds.every(Number.isFinite) ||
    animation.domainMilliseconds[0] > animation.domainMilliseconds[1] ||
    !Number.isFinite(animation.trailLengthSeconds) ||
    animation.trailLengthSeconds < 0
  )
    return {
      supported: false,
      reason:
        'Trip animation requires a finite ordered domain, current time, origin and nonnegative trail length/error budget'
    };
  const relativeTime = animation.currentTimeMilliseconds - source.originMilliseconds;
  const trailMilliseconds = animation.trailLengthSeconds * 1000;
  const currentTime = Math.fround(relativeTime);
  const trailLength = Math.fround(trailMilliseconds);
  const actualTimeErrorMilliseconds = Math.max(
    Math.abs(currentTime - relativeTime),
    Math.abs(trailLength - trailMilliseconds)
  );
  if (
    !Number.isFinite(actualTimeErrorMilliseconds) ||
    actualTimeErrorMilliseconds > source.maximumTimeErrorMilliseconds
  )
    return {
      supported: false,
      reason: 'Animation time or trail length exceeds the prepared Float32 error budget'
    };
  return {
    supported: true,
    currentTime,
    trailLength,
    fadeTrail: animation.fadeTrail ?? true,
    originMilliseconds: source.originMilliseconds,
    actualTimeErrorMilliseconds
  };
}
