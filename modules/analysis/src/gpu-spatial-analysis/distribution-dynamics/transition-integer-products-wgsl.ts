// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Exact count products and their signed residual, before conversion to float. @internal */
export const TRANSITION_INTEGER_PRODUCTS_WGSL = /* wgsl */ `
struct TransitionCount { high: u32, low: u32 }
struct TransitionProduct { first: u32, second: u32, third: u32, fourth: u32 }
struct TransitionWordDifference { word: u32, borrow: u32 }
fn addTransitionCounts(left: TransitionCount, right: TransitionCount) -> TransitionCount {
  let low: u32 = left.low + right.low;
  return TransitionCount(left.high + right.high + select(0u, 1u, low < left.low), low);
}
fn transitionCountToFloat(value: TransitionCount) -> f32 {
  return f32(value.high) * 4294967296.0 + f32(value.low);
}
// Exact 32 x 32 -> 64 bits, using four 16-bit products.
fn multiplyTransitionWords(left: u32, right: u32) -> TransitionCount {
  let leftLow: u32 = left & 65535u;
  let leftHigh: u32 = (left >> 16u) & 65535u;
  let rightLow: u32 = right & 65535u;
  let rightHigh: u32 = (right >> 16u) & 65535u;
  let lowLow: u32 = leftLow * rightLow;
  let lowHigh: u32 = leftLow * rightHigh;
  let highLow: u32 = leftHigh * rightLow;
  let middle: u32 = ((lowLow >> 16u) & 65535u) + (lowHigh & 65535u) + (highLow & 65535u);
  return TransitionCount(leftHigh * rightHigh + ((lowHigh >> 16u) & 65535u) + ((highLow >> 16u) & 65535u) + (middle >> 16u),
    u32((lowLow & 65535u) | (middle << 16u)));
}
fn addTransitionProducts(left: TransitionProduct, right: TransitionProduct) -> TransitionProduct {
  let first = addTransitionCounts(TransitionCount(0u, left.first), TransitionCount(0u, right.first));
  let second = addTransitionCounts(addTransitionCounts(TransitionCount(0u, left.second), TransitionCount(0u, right.second)), TransitionCount(0u, first.high));
  let third = addTransitionCounts(addTransitionCounts(TransitionCount(0u, left.third), TransitionCount(0u, right.third)), TransitionCount(0u, second.high));
  return TransitionProduct(first.low, second.low, third.low, left.fourth + right.fourth + third.high);
}
fn multiplyTransitionCounts(left: TransitionCount, right: TransitionCount) -> TransitionProduct {
  let low = multiplyTransitionWords(left.low, right.low);
  let crossFirst = multiplyTransitionWords(left.high, right.low);
  let crossSecond = multiplyTransitionWords(left.low, right.high);
  let high = multiplyTransitionWords(left.high, right.high);
  return addTransitionProducts(addTransitionProducts(TransitionProduct(low.low, low.high, 0u, 0u),
    TransitionProduct(0u, crossFirst.low, crossFirst.high, 0u)),
    addTransitionProducts(TransitionProduct(0u, crossSecond.low, crossSecond.high, 0u), TransitionProduct(0u, 0u, high.low, high.high)));
}
fn subtractTransitionWord(larger: u32, smaller: u32, borrow: u32) -> TransitionWordDifference {
  let partial: u32 = larger - smaller;
  return TransitionWordDifference(partial - borrow, select(0u, 1u, larger < smaller || partial < borrow));
}
fn getTransitionProductDifference(observed: u32, pooledTotal: TransitionCount, regimeTotal: TransitionCount, pooledCount: TransitionCount) -> f32 {
  let observedProduct = multiplyTransitionCounts(TransitionCount(0u, observed), pooledTotal);
  let expectedProduct = multiplyTransitionCounts(regimeTotal, pooledCount);
  let negative = observedProduct.fourth < expectedProduct.fourth ||
    (observedProduct.fourth == expectedProduct.fourth && (observedProduct.third < expectedProduct.third ||
    (observedProduct.third == expectedProduct.third && (observedProduct.second < expectedProduct.second ||
    (observedProduct.second == expectedProduct.second && observedProduct.first < expectedProduct.first)))));
  var larger = observedProduct;
  var smaller = expectedProduct;
  if (negative) { larger = expectedProduct; smaller = observedProduct; }
  let first = subtractTransitionWord(larger.first, smaller.first, 0u);
  let second = subtractTransitionWord(larger.second, smaller.second, first.borrow);
  let third = subtractTransitionWord(larger.third, smaller.third, second.borrow);
  let fourth = subtractTransitionWord(larger.fourth, smaller.fourth, third.borrow);
  let value = ((f32(fourth.word) * 4294967296.0 + f32(third.word)) * 4294967296.0 + f32(second.word)) * 4294967296.0 + f32(first.word);
  return select(value, -value, negative);
}
`;
