var e=`
fn getCenteredRegressionPrediction(
  responseMean: f32, centeredContribution: f32, meanCorrection: f32, runtimeZero: u32
) -> f32 {
  let correctedContribution = getRoundedDifference(centeredContribution + meanCorrection, 0.0, runtimeZero);
  return responseMean + correctedContribution;
}
`;function t(e){return`var centeredContribution = 0.0;
for (var column = 0u; column < ${e.predictorCount}u; column++) {
  centeredContribution += (${e.slope}) * getRoundedDifference(${e.predictor}, ${e.mean}, ${e.runtimeZero});
}`}export{t as n,e as t};