var e=4294967294,t=`
// The maximum is a sentinel: counts beyond uint32 must not wrap back into an output prefix.
const TRAJECTORY_COUNT_MAXIMUM: u32 = 0xffffffffu;
fn reserveTrajectoryCount() -> u32 {
  var currentCount = atomicLoad(&state[stateOffset]);
  loop {
    if (currentCount == TRAJECTORY_COUNT_MAXIMUM) { return currentCount; }
    let comparison = atomicCompareExchangeWeak(&state[stateOffset], currentCount, currentCount + 1u);
    if (comparison.exchanged) { return currentCount; }
    currentCount = comparison.old_value;
  }
}
`;export{t as n,e as t};