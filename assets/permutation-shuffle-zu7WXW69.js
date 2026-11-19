import{t as e}from"./hash-uint32-wL4xFYeV.js";function t(t,n=150){return`
${e}
const PERMUTATION_SHUFFLE_STREAM: u32 = ${t>>>0}u;
const PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS: u32 = ${n}u;
var<workgroup> swapOrNotKeys: array<u32, ${n}>;
var<workgroup> swapOrNotSalts: array<u32, ${n}>;

fn getSwapOrNotRoundCount(count: u32) -> u32 {
  return select(6u * (32u - countLeadingZeros(count - 1u)) + 6u, 0u, count <= 1u);
}

fn loadSwapOrNotRounds(key: vec2<u32>, permutation: u32, count: u32, local: u32, size: u32) {
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = local; round < roundCount; round += size) {
    var stream = createPhiloxStream(key, permutation, round, PERMUTATION_SHUFFLE_STREAM);
    swapOrNotKeys[round] = nextPhiloxBelow(&stream, count);
    swapOrNotSalts[round] = nextPhiloxUint32(&stream);
  }
}

fn getSwapOrNotPermutationIndex(index: u32, count: u32) -> u32 {
  var position = index;
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = 0u; round < roundCount; round++) {
    // (key - position) mod count. Both operands are below count, so neither branch can wrap in
    // u32; the sum key + count - position does for a count above 2^31.
    let key = swapOrNotKeys[round];
    let partner = select(key + (count - position), key - position, key >= position);
    let canonical = max(position, partner);
    if ((hashUint32(canonical ^ swapOrNotSalts[round]) >> 31u) != 0u) {
      position = partner;
    }
  }
  return position;
}
`}export{t};