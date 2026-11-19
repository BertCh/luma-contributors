var e=`
fn getRoundedDifference(minuend: f32, subtrahend: f32, runtimeZero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(minuend - subtrahend) ^ runtimeZero);
}
`;export{e as t};