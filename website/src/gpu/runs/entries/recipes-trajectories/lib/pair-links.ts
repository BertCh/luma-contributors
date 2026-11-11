// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Adapter kernels that turn the pair table of the trajectory encounter operations into marks on
// the map: one link per pair between the two tracks at the moment the operation reports, and the
// two end points of each link. They read the operation's own output rows on the GPU; rows beyond
// the pair count are parked off screen.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import {addKernel} from '../../geometry-processing/shared/shape-segments';

const PARKED = 1e6;

export type PairLinks = {
  /** One (x0, y0, x1, y1) row per pair row. */
  links: RunBuffer<'float32x4'>;
  /** Two (x, y) rows per pair row: the end of the lower track, then the end of its partner. */
  ends: RunBuffer<'float32x2'>;
};

type PairColumns = {
  ids: RunBuffer<'uint32'>;
  partners: RunBuffer<'uint32'>;
  count: RunBuffer<'uint32'>;
};

/**
 * Links each pair of GPUTrajectoryEncounters or addClockEncounters at the first bucket where the
 * two tracks met: the two resampled positions of that bucket.
 */
export function addBucketLinks(
  builder: RunBuilder,
  options: PairColumns & {
    firstBuckets: RunBuffer<'uint32'>;
    samples: RunBuffer<'float32x2'>;
    bucketCount: number;
    pairCapacity: number;
  }
): PairLinks {
  const links = builder.allocate('bucket-links', 'float32x4', options.pairCapacity);
  addKernel(builder, {
    id: 'bucket-links-kernel',
    invocationCount: options.pairCapacity,
    bindings: [
      {name: 'ids', buffer: options.ids, access: 'read'},
      {name: 'partners', buffer: options.partners, access: 'read'},
      {name: 'firstBuckets', buffer: options.firstBuckets, access: 'read'},
      {name: 'count', buffer: options.count, access: 'read'},
      {name: 'samples', buffer: options.samples, access: 'read'},
      {name: 'links', buffer: links, access: 'read_write'}
    ],
    body: `var lowerX = ${PARKED.toFixed(1)};
  var lowerY = ${PARKED.toFixed(1)};
  var partnerX = ${PARKED.toFixed(1)};
  var partnerY = ${PARKED.toFixed(1)};
  if (index < count[countOffset]) {
    let bucket = firstBuckets[firstBucketsOffset + index];
    let lowerRow = ids[idsOffset + index] * ${options.bucketCount}u + bucket;
    let partnerRow = partners[partnersOffset + index] * ${options.bucketCount}u + bucket;
    lowerX = samples[samplesOffset + lowerRow * 2u];
    lowerY = samples[samplesOffset + lowerRow * 2u + 1u];
    partnerX = samples[samplesOffset + partnerRow * 2u];
    partnerY = samples[samplesOffset + partnerRow * 2u + 1u];
  }
  links[linksOffset + index * 4u] = lowerX;
  links[linksOffset + index * 4u + 1u] = lowerY;
  links[linksOffset + index * 4u + 2u] = partnerX;
  links[linksOffset + index * 4u + 3u] = partnerY;`
  });
  return {links, ends: addLinkEnds(builder, links, options.pairCapacity)};
}

/** The two end points of every link row as (x, y) rows: the lower track's, then its partner's. */
function addLinkEnds(
  builder: RunBuilder,
  links: RunBuffer<'float32x4'>,
  pairCapacity: number
): RunBuffer<'float32x2'> {
  const ends = builder.allocate('link-ends', 'float32x2', pairCapacity * 2);
  addKernel(builder, {
    id: `${links.id}-ends-kernel`,
    invocationCount: pairCapacity,
    bindings: [
      {name: 'links', buffer: links, access: 'read'},
      {name: 'ends', buffer: ends, access: 'read_write'}
    ],
    body: `ends[endsOffset + index * 4u] = links[linksOffset + index * 4u];
  ends[endsOffset + index * 4u + 1u] = links[linksOffset + index * 4u + 1u];
  ends[endsOffset + index * 4u + 2u] = links[linksOffset + index * 4u + 2u];
  ends[endsOffset + index * 4u + 3u] = links[linksOffset + index * 4u + 3u];`
  });
  return ends;
}

/** WGSL that finds a track's position at `time` between its raw fixes (parked when outside). */
function getPositionSnippet(prefix: string, track: string): string {
  return `var ${prefix}X = ${PARKED.toFixed(1)};
    var ${prefix}Y = ${PARKED.toFixed(1)};
    let ${prefix}Start = trackOffsets[trackOffsetsOffset + ${track}];
    let ${prefix}End = trackOffsets[trackOffsetsOffset + ${track} + 1u];
    for (var row = ${prefix}Start; row + 1u < ${prefix}End; row = row + 1u) {
      let before = timestamps[timestampsOffset + row];
      let after = timestamps[timestampsOffset + row + 1u];
      if (time >= before && time <= after) {
        let fraction = select(0.0, (time - before) / (after - before), after > before);
        ${prefix}X = mix(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 2u], fraction);
        ${prefix}Y = mix(positions[positionsOffset + row * 2u + 1u], positions[positionsOffset + row * 2u + 3u], fraction);
        break;
      }
    }`;
}

/**
 * Links each pair of GPUTrajectoryClosestApproach at its closest time: both tracks interpolated
 * between their raw fixes at `closestTimes` (seconds after the first fix of the lower track).
 */
export function addClosestApproachLinks(
  builder: RunBuilder,
  options: PairColumns & {
    closestTimes: RunBuffer<'float32'>;
    positions: RunBuffer<'float32x2'>;
    timestamps: RunBuffer<'float32'>;
    trackOffsets: RunBuffer<'uint32'>;
    pairCapacity: number;
  }
): PairLinks {
  const links = builder.allocate('approach-links', 'float32x4', options.pairCapacity);
  addKernel(builder, {
    id: 'approach-links-kernel',
    invocationCount: options.pairCapacity,
    bindings: [
      {name: 'ids', buffer: options.ids, access: 'read'},
      {name: 'partners', buffer: options.partners, access: 'read'},
      {name: 'closestTimes', buffer: options.closestTimes, access: 'read'},
      {name: 'count', buffer: options.count, access: 'read'},
      {name: 'positions', buffer: options.positions, access: 'read'},
      {name: 'timestamps', buffer: options.timestamps, access: 'read'},
      {name: 'trackOffsets', buffer: options.trackOffsets, access: 'read'},
      {name: 'links', buffer: links, access: 'read_write'}
    ],
    body: `var lowerX = ${PARKED.toFixed(1)};
  var lowerY = ${PARKED.toFixed(1)};
  var partnerX = ${PARKED.toFixed(1)};
  var partnerY = ${PARKED.toFixed(1)};
  if (index < count[countOffset]) {
    let lowerTrack = ids[idsOffset + index];
    let partnerTrack = partners[partnersOffset + index];
    let time = timestamps[timestampsOffset + trackOffsets[trackOffsetsOffset + lowerTrack]] + closestTimes[closestTimesOffset + index];
    ${getPositionSnippet('lowerPosition', 'lowerTrack')}
    ${getPositionSnippet('partnerPosition', 'partnerTrack')}
    lowerX = lowerPositionX;
    lowerY = lowerPositionY;
    partnerX = partnerPositionX;
    partnerY = partnerPositionY;
  }
  links[linksOffset + index * 4u] = lowerX;
  links[linksOffset + index * 4u + 1u] = lowerY;
  links[linksOffset + index * 4u + 2u] = partnerX;
  links[linksOffset + index * 4u + 3u] = partnerY;`
  });
  return {links, ends: addLinkEnds(builder, links, options.pairCapacity)};
}
