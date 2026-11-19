function e(e,t={}){let n=e.features?.has(`subgroups`),r=!t.requiresSubgroupId||e.wgslLanguageFeatures?.has(`subgroup_id`);return n&&r?`subgroups`:`portable`}function t(){return`
fn getFirstBallotLane(mask: vec4<u32>) -> u32 {
  if (mask.x != 0u) { return firstTrailingBit(mask.x); }
  if (mask.y != 0u) { return 32u + firstTrailingBit(mask.y); }
  if (mask.z != 0u) { return 64u + firstTrailingBit(mask.z); }
  if (mask.w != 0u) { return 96u + firstTrailingBit(mask.w); }
  return 0u;
}

fn getBallotLaneCount(mask: vec4<u32>) -> u32 {
  return countOneBits(mask.x) + countOneBits(mask.y) +
    countOneBits(mask.z) + countOneBits(mask.w);
}

fn getBallotPrefixLaneCount(mask: vec4<u32>, invocation: u32) -> u32 {
  let word = invocation >> 5u;
  let bit = invocation & 31u;
  var count = 0u;
  if (word > 0u) { count += countOneBits(mask.x); }
  if (word > 1u) { count += countOneBits(mask.y); }
  if (word > 2u) { count += countOneBits(mask.z); }
  let lowerBits = (1u << bit) - 1u;
  return count + countOneBits(mask[word] & lowerBits);
}`}function n(e,t,n,r){return`
  var subgroupPending = ${e};
  for (var subgroupGroup = 0u; subgroupGroup < ${r}u; subgroupGroup++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderKey = subgroupShuffle(${t}, leaderInvocation);
    let matchingKey = hasPending && subgroupPending && ${t} == leaderKey;
    let matchingBallot = subgroupBallot(matchingKey);
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      atomicAdd(&${n}[leaderKey], getBallotLaneCount(matchingBallot));
    }
    subgroupPending = subgroupPending && !matchingKey;
  }`}export{t as n,n as r,e as t};