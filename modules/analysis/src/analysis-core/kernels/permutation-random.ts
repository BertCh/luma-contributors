// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Private counter-based random numbers for permutation inference: Philox 4x32-10 (Salmon et al.,
// "Parallel random numbers: as easy as 1, 2, 3", SC 2011) and a bounded integer draw. The WGSL and
// TypeScript implementations produce bit-identical streams, so CPU oracles reproduce GPU
// permutations exactly. The WGSL block function is the shared `PHILOX_WGSL` of `utils`.

import {getPhilox4x32, PHILOX_WGSL} from './philox-random';

/** High 32 bits of the 64-bit product of two u32 values, using exact 16-bit partial products. */
function multiplyHigh(a: number, b: number): number {
  const aLow = a & 0xffff;
  const aHigh = a >>> 16;
  const bLow = b & 0xffff;
  const bHigh = b >>> 16;
  const lowLow = aLow * bLow;
  const lowHigh = aLow * bHigh;
  const highLow = aHigh * bLow;
  const highHigh = aHigh * bHigh;
  const middle = (lowLow >>> 16) + (lowHigh & 0xffff) + (highLow & 0xffff);
  return (highHigh + (lowHigh >>> 16) + (highLow >>> 16) + (middle >>> 16)) >>> 0;
}

export {getPhilox4x32};

/**
 * Sequential u32 stream for one `(key, a, b, c)` tuple: words are drawn from Philox blocks with
 * counter `(block, a, b, c)`, four words per block, block 0 first. Mirrors the WGSL
 * `PhiloxStream` exactly.
 *
 * @internal
 */
export class PhiloxStream {
  private readonly key: readonly [number, number];
  private readonly words: readonly [number, number, number];
  private block = 0;
  private lane = 4;
  private buffer: [number, number, number, number] = [0, 0, 0, 0];

  /**
   * @param key Two u32 key words, normally the seed.
   * @param a First caller counter word, for example a row index.
   * @param b Second caller counter word, for example a permutation index.
   * @param c Third caller counter word, a purpose tag.
   */
  constructor(key: readonly [number, number], a: number, b: number, c: number) {
    this.key = [key[0] >>> 0, key[1] >>> 0];
    this.words = [a >>> 0, b >>> 0, c >>> 0];
  }

  /** Returns the next u32 word of the stream. */
  nextUint32(): number {
    if (this.lane === 4) {
      this.buffer = getPhilox4x32([this.block, ...this.words], this.key);
      this.block = (this.block + 1) >>> 0;
      this.lane = 0;
    }
    return this.buffer[this.lane++];
  }

  /**
   * Returns an unbiased integer in `[0, bound)` with Lemire's multiply-shift and rejection.
   *
   * @param bound Exclusive upper bound, `1 <= bound < 2^32`.
   */
  nextBelow(bound: number): number {
    const range = bound >>> 0;
    let value = this.nextUint32();
    let low = Math.imul(value, range) >>> 0;
    if (low < range) {
      const threshold = (0x100000000 - range) % range;
      while (low < threshold) {
        value = this.nextUint32();
        low = Math.imul(value, range) >>> 0;
      }
    }
    return multiplyHigh(value, range);
  }
}

/**
 * WGSL twin of this module: `getPhilox4x32`, `PhiloxStream` with `createPhiloxStream`,
 * `nextPhiloxUint32` and `nextPhiloxBelow`. Bit-identical to the TypeScript functions. It embeds
 * `PHILOX_WGSL`, so a kernel must not include that separately.
 *
 * @internal
 */
export const PERMUTATION_RANDOM_WGSL = /* wgsl */ `
${PHILOX_WGSL}

fn getPhiloxMultiplyHigh(a: u32, b: u32) -> u32 {
  return philoxMulHiLo(a, b).x;
}

fn getPhilox4x32(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  return philox4x32(counter, key);
}

struct PhiloxStream {
  key: vec2<u32>,
  words: vec3<u32>,
  block: u32,
  lane: u32,
  buffer: vec4<u32>
}

fn createPhiloxStream(key: vec2<u32>, a: u32, b: u32, c: u32) -> PhiloxStream {
  var stream: PhiloxStream;
  stream.key = key;
  stream.words = vec3<u32>(a, b, c);
  stream.block = 0u;
  stream.lane = 4u;
  stream.buffer = vec4<u32>(0u);
  return stream;
}

fn nextPhiloxUint32(stream: ptr<function, PhiloxStream>) -> u32 {
  if ((*stream).lane == 4u) {
    (*stream).buffer = getPhilox4x32(
      vec4<u32>((*stream).block, (*stream).words.x, (*stream).words.y, (*stream).words.z),
      (*stream).key
    );
    (*stream).block = (*stream).block + 1u;
    (*stream).lane = 0u;
  }
  let word = (*stream).buffer[(*stream).lane];
  (*stream).lane = (*stream).lane + 1u;
  return word;
}

fn nextPhiloxBelow(stream: ptr<function, PhiloxStream>, bound: u32) -> u32 {
  var value = nextPhiloxUint32(stream);
  var low = value * bound;
  if (low < bound) {
    let threshold = (0u - bound) % bound;
    while (low < threshold) {
      value = nextPhiloxUint32(stream);
      low = value * bound;
    }
  }
  return getPhiloxMultiplyHigh(value, bound);
}
`;
