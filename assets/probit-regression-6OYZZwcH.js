import{r as e}from"./kernel-C3tfQOn4.js";import{a as t,c as n,o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,f as a,p as o,u as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{c as u,n as d,r as f,u as p}from"./wgsl-common-DfyXiYB0.js";import{a as m,c as h}from"./spatial-weights-CqWysstN.js";import{n as g,r as ee}from"./spatial-regression-solve-Vf-sQz9O.js";import{t as te}from"./rounded-difference-wgsl-DgXXQtNj.js";import{i as ne,n as re,r as _,t as v}from"./regression-row-tiles-7Xsw3Utq.js";import{t as y}from"./define-run-B-brkCe4.js";import{c as b}from"./text-DYSgVV-n.js";import{Nn as x}from"./entry-preview-D70TdSbK.js";import{_ as S,a as C,c as w,d as T,h as E,l as D,m as O,p as k,t as A}from"./regression-CXg5hXjb.js";var j=`GPUProbitRegression`,M=11,N=256,P=64,F=1e-5,I=24,L=1e-4,ie=14,R=3,z=6,B=5;function ae(e){return(e+5)*2**-23}var V=5,H=class{props;id;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`probit-regression`;let t=e.response.length,n=e.predictorCount;if(e.spatialDiagnostics){let n=e.spatialDiagnostics;if(h(this.id,n.weights)!==t||n.output.tests.length!==6||n.output.status.length!==1)throw Error(`${this.id}: diagnostic shape mismatch`);i(n.output.tests,[`float32`],this.id),i(n.output.status,[`uint32`],this.id),l(this.id,[...Object.values(e.output),...Object.values(n.output)],[e.predictors,e.response,e.mask,...m(n.weights)])}a(this.id,`rows`,t,1,v),a(this.id,`predictorCount`,n,1,M);let r=e.maximumIterations??P;if(a(this.id,`maximumIterations`,r,1,N),!Number.isFinite(e.tolerance??F)||(e.tolerance??F)<=0)throw Error(`${this.id}: invalid tolerance`);for(let[r,a,o]of[[e.predictors,`float32`,t*n],[e.response,`float32`,t],[e.mask,`uint32`,t],[e.output.coefficients,`float32`,n+1],[e.output.standardErrors,`float32`,n+1],[e.output.fitted,`float32`,t],[e.output.marginalEffects,`float32`,n],[e.output.summary,`float32`,4],[e.output.status,`uint32`,1]])if(r&&(i(r,[a],this.id),r.length!==o))throw Error(`${this.id}: shape mismatch`);l(this.id,Object.values(e.output),[e.predictors,e.response,e.mask]),this.estimatedCommandNodeCount=7+6*r+3+5+(e.spatialDiagnostics?4:0),o(this.id,this.estimatedCommandNodeCount)}getCommandNodes(i){let{id:o,props:l}=this,h=l.response.length,y=l.predictorCount,b=y+1,x=l.maximumIterations??P;a(o,`rows`,h,1,Math.min(v,s(i.device,4*Math.max(y,l.spatialDiagnostics?V:1)))),c(o,i,[l.predictors,l.response,l.mask,...Object.values(l.output)]);let S=(e,t,n=`read`,r=`f32`)=>({name:e,view:t,access:n,type:r}),C=[S(`predictors`,l.predictors),S(`response`,l.response),...l.mask?[S(`mask`,l.mask,`read`,`u32`)]:[]],w=oe(y),T=re(h),E=e(i,`${o}-state`,`float32`,w.length),D=b+b*(b+1)/2,[O,k]=_(T,D+2),A=t(i,{id:`${o}-iteration-dispatch`,operation:j,shapes:[O,k,O,[62]]}),M=(t,n)=>e(i,`${o}-${t}-partials`,`float32`,T.tileCount*n),N=M(`information`,D+2),H=M(`line-search`,62),U=`fn getOrderKey(value: f32) -> f32 { return bitcast<f32>(0x7f7fffffu - (bitcast<u32>(value) & 0x7fffffffu)); }
fn getOrderKeyMagnitude(key: f32) -> f32 { return bitcast<f32>(0x7f7fffffu - bitcast<u32>(key)); }`,W=`${f}\n${u}\n${d}
fn included(row: u32) -> bool {
 if (${l.mask?`mask[maskOffset + row] == 0u || `:``}!isFiniteBits(response[responseOffset + row])) { return false; }
 for (var column = 0u; column < ${y}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${y}u + column])) { return false; } } return true;
}
fn probability(value: f32) -> f32 { return clamp(0.5 * erfcValue(-value * 0.7071067811865475), 1e-7, 1.0 - 1e-7); }
fn density(value: f32) -> f32 { return 0.3989422804014327 * exp(-0.5 * value * value); }
// Phi(value) through erfc, so the lower tail keeps its relative accuracy down to the smallest
// normal f32 (value near -13), where it is floored. Called with +eta for a response of 1 and
// -eta for a response of 0 it is the probability of the observed outcome.
fn tailProbability(value: f32) -> f32 { return max(0.5 * erfcValue(-value * 0.7071067811865475), 1.1754944e-38); }
${U}
`,G=`const C: u32 = ${y}u;
const P: u32 = ${b}u;
${Object.entries(w).map(([e,t])=>`const S_${e.replace(/[A-Z]/g,e=>`_${e}`).toUpperCase()}: u32 = ${t}u;`).join(`
`)}
fn stateAt(slot: u32) -> f32 { return state[stateOffset + slot]; }`,K=`${W}\n${G}\n${`${te}
fn getDesign(row: u32) -> array<f32, ${b}> { var design: array<f32, ${b}>; design[0] = 1.0; let runtimeZero = bitcast<u32>(stateAt(S_RUNTIME_ZERO));
 for (var column = 0u; column < C; column++) { design[column + 1u] = getRoundedDifference(predictors[predictorsOffset + row * C + column], stateAt(S_MEANS + column), runtimeZero) * stateAt(S_SCALES + column); } return design; }`}`,q=[...C,S(`state`,E)],J=e=>2*e,Y=e=>`${K}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; }
 let design = getDesign(row); let side = 2.0 * response[responseOffset + row] - 1.0;
 var estimate: array<f32, ${b}>; var direction: array<f32, ${b}>;
 for (var column = 0u; column < P; column++) { estimate[column] = stateAt(S_ESTIMATE + column); direction[column] = stateAt(S_DIRECTION + column); }
 var rowMove = 0.0; var rowScale = 0.0; var standsStill = true;
 for (var column = 0u; column < P; column++) {
  let candidate = stateAt(S_SEPARATION_DIRECTION + column); let term = design[column] * candidate; rowMove += term; rowScale += abs(term);
  if (candidate != 0.0) { if (column == 0u) { standsStill = false; } else if (predictors[predictorsOffset + row * C + column - 1u] != stateAt(S_MEANS + column - 1u)) { standsStill = false; } }
 }
 if (!standsStill) {
  let signedMove = side * rowMove; let errorBound = ${n(ae(y))} * rowScale;
  let decided = isFiniteBits(rowScale) && isFiniteBits(rowMove) && rowScale >= 1.0e-30;
  if (decided && signedMove > errorBound) { (*terms)[${J(e)+2}u] = 1.0; }
  else if (decided && signedMove < -errorBound) { (*terms)[${J(e)}u] = 1.0; }
  else { (*terms)[${J(e)+1}u] = 1.0; }
 }
 for (var threshold = 0u; threshold < ${R}u; threshold++) {
  let thresholdSign = stateAt(S_THRESHOLDS + 3u * threshold + 2u);
  if (thresholdSign == 0.0) { continue; }
  let value = predictors[predictorsOffset + row * C + u32(stateAt(S_THRESHOLDS + 3u * threshold))];
  let thresholdValue = stateAt(S_THRESHOLDS + 3u * threshold + 1u);
  // Exact: a comparison of two float32 values.
  let thresholdMove = thresholdSign * side * select(select(0.0, 1.0, value > thresholdValue), -1.0, value < thresholdValue);
  if (thresholdMove < 0.0) { (*terms)[${J(e)+3}u + 2u * threshold] = 1.0; }
  if (thresholdMove > 0.0) { (*terms)[${J(e)+4}u + 2u * threshold] = 1.0; }
  if (threshold == 0u) {
   if (value <= thresholdValue) { if (value >= 0.0) { (*terms)[${J(e)+9}u] = max((*terms)[${J(e)+9}u], value); (*terms)[${J(e)+10}u] = 1.0; } else { (*terms)[${J(e)+11}u] = max((*terms)[${J(e)+11}u], getOrderKey(value)); } }
   if (value >= thresholdValue) { if (value < 0.0) { (*terms)[${J(e)+12}u] = max((*terms)[${J(e)+12}u], -value); } else { (*terms)[${J(e)+13}u] = max((*terms)[${J(e)+13}u], getOrderKey(value)); } }
  }
 }
${e===1?` for (var column = 0u; column < C; column++) {
  let extremes = ${J(e)+ie}u + ${z}u * column; let value = predictors[predictorsOffset + row * C + column];
  (*terms)[extremes] = max((*terms)[extremes], design[column + 1u]);
  (*terms)[extremes + 1u] = max((*terms)[extremes + 1u], -design[column + 1u]);
  // Exact comparisons with the lowest and the highest value of the column: is there a row
  // above the lowest value with a response of 1, of 0; below the highest value with 1, with 0.
  if (value > stateAt(S_RAW_MINIMUM + column)) { (*terms)[extremes + select(3u, 2u, side > 0.0)] = 1.0; }
  if (value < stateAt(S_RAW_MAXIMUM + column)) { (*terms)[extremes + select(5u, 4u, side > 0.0)] = 1.0; }
 }`:``}
 var scale = 1.0;
 for (var step = 0u; step < ${e}u; step++) {
  var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * (estimate[column] + scale * direction[column]); }
  let signedLinear = side * linear;
  (*terms)[${e}u + step] = max((*terms)[${e}u + step], select(0.0, abs(linear), signedLinear < 0.0 || !isFiniteBits(linear)));
  (*terms)[step] += log(tailProbability(signedLinear));
  scale *= 0.5;
 }
}`,X=(e,t,n,r,a,s,c=q,l=!1)=>ne(i,{id:`${o}-${e}`,operation:j,variant:e.replace(/-\d+/g,``),rowCount:h,layout:T,sumCount:t,maximumCount:n,bindings:c,declarations:r,partial:s,output:E,outputSlot:a,...l?{tileGate:A.getGate(e.startsWith(`iteration`)&&e.endsWith(`information`)?0:2),mergeGate:A.getGate(e.endsWith(`information`)?1:3)}:{}}),Z=(e,t,n,a)=>r(i,{id:`${o}-${e}`,operation:j,variant:e.replace(/iteration-\d+-/,``),invocationCount:1,bindings:t,declarations:n,body:a}),Q=n(l.tolerance??F),$=[...X(`prepare`,y+2,1,`${W}
const C: u32 = ${y}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let outcome = response[responseOffset + row];
 if (outcome != 0.0 && outcome != 1.0) { (*terms)[SUM_COUNT] = 1.0; }
 (*terms)[0] += 1.0; (*terms)[1] += outcome;
 for (var column = 0u; column < C; column++) { (*terms)[2u + column] += predictors[predictorsOffset + row * C + column]; }
}`,w.prepare,M(`prepare`,y+3),C),...X(`variance`,y,B*y,`${K}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let count = stateAt(S_PREPARE);
 for (var column = 0u; column < C; column++) {
  let value = predictors[predictorsOffset + row * C + column]; let difference = value - stateAt(S_PREPARE + 2u + column) / count; (*terms)[column] += difference * difference;
  let extremes = C + ${B}u * column;
  if (value >= 0.0) { (*terms)[extremes] = max((*terms)[extremes], value); (*terms)[extremes + 1u] = 1.0; (*terms)[extremes + 4u] = max((*terms)[extremes + 4u], getOrderKey(value)); }
  else { (*terms)[extremes + 2u] = max((*terms)[extremes + 2u], getOrderKey(value)); (*terms)[extremes + 3u] = max((*terms)[extremes + 3u], -value); }
 }
}`,w.variance,M(`variance`,6*y),q),Z(`initialize`,[S(`state`,E,`read_write`),A.binding],`${G}\n${U}`,`${A.disarmWGSL}
state[stateOffset + S_DONE] = 1.0; state[stateOffset + S_CODE] = 1.0; state[stateOffset + S_COUNTED] = 0.0;
state[stateOffset + S_ITERATIONS] = 0.0; state[stateOffset + S_ITERATION] = 0.0; state[stateOffset + S_LARGEST] = 0.0;
// No direction has been tested for separation yet (no certificate, no undecided direction, no threshold); and a zero the compiler cannot see (the runtime zero of getRoundedDifference).
state[stateOffset + S_SEPARATING] = 0.0; state[stateOffset + S_UNDECIDED] = 0.0; state[stateOffset + S_RUNTIME_ZERO] = 0.0;
for (var slot = 0u; slot < 9u; slot++) { state[stateOffset + S_THRESHOLDS + slot] = 0.0; }
// The lowest and highest raw value of every column, from the order-key maxima of the variance pass.
for (var column = 0u; column < C; column++) {
 let extremes = S_VARIANCE + C + ${B}u * column;
 var highest = stateAt(extremes); if (stateAt(extremes + 1u) == 0.0) { highest = -getOrderKeyMagnitude(stateAt(extremes + 2u)); }
 var lowest = -stateAt(extremes + 3u); if (stateAt(extremes + 3u) == 0.0) { lowest = getOrderKeyMagnitude(stateAt(extremes + 4u)); }
 state[stateOffset + S_RAW_MINIMUM + column] = lowest; state[stateOffset + S_RAW_MAXIMUM + column] = highest;
}
for (var column = 0u; column < P; column++) { state[stateOffset + S_ESTIMATE + column] = 0.0; state[stateOffset + S_DIRECTION + column] = 0.0; state[stateOffset + S_SEPARATION_DIRECTION + column] = 0.0; }
let count = stateAt(S_PREPARE); let successes = stateAt(S_PREPARE + 1u);
for (var column = 0u; column < C; column++) { state[stateOffset + S_MEANS + column] = stateAt(S_PREPARE + 2u + column) / count; state[stateOffset + S_SCALES + column] = 0.0; }
if (stateAt(S_PREPARE + C + 2u) != 0.0) { state[stateOffset + S_CODE] = 4.0; return; }
state[stateOffset + S_COUNTED] = 1.0;
if (count <= f32(P)) { return; } if (successes == 0.0 || successes == count) { state[stateOffset + S_CODE] = 4.0; return; }
for (var column = 0u; column < C; column++) { let variance = stateAt(S_VARIANCE + column); if (!(variance > 0.0)) { state[stateOffset + S_CODE] = 2.0; return; } state[stateOffset + S_SCALES + column] = sqrt(count / variance); }
// Likelihood noise floor: the tiled pairwise sum is good to a few eps * |loglik|, the per-row log/erfc errors add roughly sqrt(count) of that.
state[stateOffset + S_NOISE] = 4.0 * 1.1920929e-7 * max(1.0, sqrt(count));
state[stateOffset + S_CODE] = 3.0; state[stateOffset + S_DONE] = 0.0;
${A.armWGSL}`),...X(`initial-likelihood`,1,15+z*y,Y(1),w.current,M(`initial-likelihood`,16+z*y))];for(let e=0;e<=x;e++){let t=`iteration-${e}`;if($.push(...X(`${t}-information`,D,2,`${K}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let design = getDesign(row);
 var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
 if (!isFiniteBits(linear)) { (*terms)[SUM_COUNT + 1u] = 1.0; return; }
 (*terms)[SUM_COUNT] = max((*terms)[SUM_COUNT], abs(linear));
 // Score (y - p) phi / (p (1 - p)) and weight phi^2 / (p (1 - p)) as ratios of the density to
 // the two tail probabilities, which stay accurate where p or 1 - p is below float32 rounding.
 let side = 2.0 * response[responseOffset + row] - 1.0; let derivative = density(linear);
 let observedRatio = derivative / tailProbability(side * linear); let score = side * observedRatio; let weight = observedRatio * (derivative / tailProbability(-side * linear));
 var slot = P;
 for (var first = 0u; first < P; first++) { (*terms)[first] += design[first] * score; for (var second = first; second < P; second++) { (*terms)[slot] += design[first] * design[second] * weight; slot++; } }
}`,w.information,N,q,!0),Z(`${t}-solve`,[S(`state`,E,`read_write`),A.binding],`${f}\n${u}\n${G}\n${p}\n${U}\n${g(b)}\n${ee(b)}`,`if (stateAt(S_DONE) != 0.0) { return; }
let iteration = u32(stateAt(S_ITERATION)); state[stateOffset + S_ITERATION] = f32(iteration + 1u);
let informationSums = ${D}u;
let extremes = S_CURRENT + 16u;
// A column separates at its lowest (or highest) value when the rows above (below) that value all have the same response: the direction that moves exactly those rows moves none of them towards the wrong side, and the rows at the value itself stand still. The four flags of the initial pass are exact comparisons of raw values, so this is a certificate, and it does not depend on the fit (an indicator of a group of rows that all have a response of 1 is the usual case).
for (var column = 0u; column < C; column++) { let flags = extremes + ${z}u * column + 2u; if ((stateAt(flags) != 0.0) != (stateAt(flags + 1u) != 0.0) || (stateAt(flags + 2u) != 0.0) != (stateAt(flags + 3u) != 0.0)) { state[stateOffset + S_SEPARATING] = 1.0; } }
// A linear predictor that overflowed: the fit cannot go on. Not converged (status 3, the iterate is written); separated only with a certificate.
if (stateAt(S_INFORMATION + informationSums + 1u) != 0.0) { state[stateOffset + S_CODE] = select(3.0, 4.0, stateAt(S_SEPARATING) == 1.0); state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
state[stateOffset + S_LARGEST] = stateAt(S_INFORMATION + informationSums);
var information: array<f32, ${b*b}>; var gradient: array<f32, ${b}>; var slot = P;
for (var first = 0u; first < P; first++) { gradient[first] = stateAt(S_INFORMATION + first); for (var second = first; second < P; second++) { let value = stateAt(S_INFORMATION + slot); slot++; information[first * P + second] = value; information[second * P + first] = value; } }
// A singular information matrix at the start is a collinear design (status 2). With a certificate of separation it means the rows that still carry information no longer span the design: the fit is running away (status 4).
if (!choleskyFactorLimited_${b}(&information)) { state[stateOffset + S_CODE] = select(2.0, 4.0, stateAt(S_SEPARATING) == 1.0); state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
for (var entry = 0u; entry < ${b*b}u; entry++) { state[stateOffset + S_FACTOR + entry] = information[entry]; }
if (stateAt(S_CODE) == 0.0 || iteration == ${x}u) { state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
choleskySolve_${b}(&information, &gradient);
for (var column = 0u; column < P; column++) { state[stateOffset + S_DIRECTION + column] = gradient[column]; }
// The candidates of the separation tests, from the Fisher direction with its rounding taken out. Any direction that provably moves no row towards the wrong side certifies separation, however it was found, so nothing done here can produce a false certificate; it only decides whether a true one is seen.
// (a) A coefficient that has settled has a direction that is rounding of the score sums: a column whose largest contribution to any row is below ${L} of the largest contribution of any column is dropped.
var contributions: array<f32, ${b}>; var largestContribution = 0.0;
for (var column = 1u; column < P; column++) { let extent = max(stateAt(extremes + ${z}u * (column - 1u)), stateAt(extremes + ${z}u * (column - 1u) + 1u)); contributions[column] = abs(gradient[column]) * extent; largestContribution = max(largestContribution, contributions[column]); }
let noiseLevel = ${n(L)} * largestContribution;
var candidate: array<f32, ${b}>; candidate[0] = gradient[0]; var lowCorner = gradient[0]; var highCorner = gradient[0]; var keptCount = 0u; var keptColumn = 1u;
for (var column = 1u; column < P; column++) { if (contributions[column] >= noiseLevel && largestContribution > 0.0) { candidate[column] = gradient[column]; keptCount++; keptColumn = column; lowCorner += gradient[column] * -stateAt(extremes + ${z}u * (column - 1u) + 1u); highCorner += gradient[column] * stateAt(extremes + ${z}u * (column - 1u)); } }
// (b) The intercept of the centered model compensates the kept columns for the rows they do not touch (the rows outside a small group that an indicator separates): for those rows the move is a difference of two equal terms, each a fraction of the group's share of the rows, and its rounding grows against them as the group shrinks. When the move of a row at the low (or the high) end of every kept column is itself below the noise level, the intercept is set so that such a row does not move at all.
if (abs(lowCorner) <= noiseLevel && abs(lowCorner) <= abs(highCorner)) { candidate[0] = gradient[0] - lowCorner; } else if (abs(highCorner) <= noiseLevel) { candidate[0] = gradient[0] - highCorner; }
for (var column = 0u; column < P; column++) { state[stateOffset + S_SEPARATION_DIRECTION + column] = candidate[column]; }
// (c) A candidate with one column is the direction sign * (x - value) of that predictor: the value is where the candidate changes sign, and the move of a row is then decided by comparing its predictor with the value, which is exact, where the float32 sum of two terms that cancel is not. The line-search pass also reports the data values next to it, which are the thresholds of the following pass.
var thresholdSign = 0.0; var thresholdValue = 0.0;
if (keptCount == 1u) {
 let predictor = keptColumn - 1u; let slope = candidate[keptColumn] * stateAt(S_SCALES + predictor);
 thresholdSign = select(select(0.0, 1.0, slope > 0.0), -1.0, slope < 0.0);
 thresholdValue = stateAt(S_MEANS + predictor) - candidate[0] / slope;
 if (!isFiniteValue(thresholdValue)) { thresholdSign = 0.0; thresholdValue = 0.0; }
 state[stateOffset + S_THRESHOLDS] = f32(predictor);
}
state[stateOffset + S_THRESHOLDS + 1u] = thresholdValue; state[stateOffset + S_THRESHOLDS + 2u] = thresholdSign;`)),e===x)break;$.push(...X(`${t}-line-search`,I,38,Y(I),w.lineSearch,H,q,!0),Z(`${t}-decide`,[S(`state`,E,`read_write`),A.binding],`${u}\n${G}\n${U}`,`if (stateAt(S_DONE) != 0.0) { return; }
// Separation is certified by a direction that provably moves some row towards the right side and no row towards the wrong side: the candidate direction when the sign of every row's move was decided, or a single-column threshold (exact comparisons). Data that are not separated have no such direction. The certificate is kept once found, because the directions of a separated fit end as rounding (the score of the separated rows falls below the float32 noise of the score sums near a linear predictor of 6).
let separation = S_LINE_SEARCH + 48u;
let movesWrong = stateAt(separation) != 0.0; let undecided = stateAt(separation + 1u) != 0.0; let movesRight = stateAt(separation + 2u) != 0.0;
var certified = movesRight && !movesWrong && !undecided;
for (var threshold = 0u; threshold < ${R}u; threshold++) { if (stateAt(S_THRESHOLDS + 3u * threshold + 2u) != 0.0 && stateAt(separation + 3u + 2u * threshold) == 0.0 && stateAt(separation + 4u + 2u * threshold) != 0.0) { certified = true; } }
if (certified) { state[stateOffset + S_SEPARATING] = 1.0; }
// A candidate that moves no row towards the wrong side as far as float32 can tell, but leaves the sign of some row undecided, is not a certificate. It is remembered, because a step that later falls below the tolerance on such data may be rounding and not a maximum: the fit is then status 3, not 0.
if (movesRight && !movesWrong && undecided) { state[stateOffset + S_UNDECIDED] = 1.0; }
// The data values next to the current threshold become the thresholds of the next pass: a column that separates at a value where both outcomes occur needs that value exactly.
let thresholdSign = stateAt(S_THRESHOLDS + 2u); var lowerSign = 0.0; var lowerValue = 0.0; var upperSign = 0.0; var upperValue = 0.0;
if (thresholdSign != 0.0) {
 if (stateAt(separation + 10u) != 0.0) { lowerValue = stateAt(separation + 9u); lowerSign = thresholdSign; } else if (stateAt(separation + 11u) > 0.0) { lowerValue = -getOrderKeyMagnitude(stateAt(separation + 11u)); lowerSign = thresholdSign; }
 if (stateAt(separation + 12u) > 0.0) { upperValue = -stateAt(separation + 12u); upperSign = thresholdSign; } else if (stateAt(separation + 13u) > 0.0) { upperValue = getOrderKeyMagnitude(stateAt(separation + 13u)); upperSign = thresholdSign; }
}
state[stateOffset + S_THRESHOLDS + 3u] = stateAt(S_THRESHOLDS); state[stateOffset + S_THRESHOLDS + 4u] = lowerValue; state[stateOffset + S_THRESHOLDS + 5u] = lowerSign;
state[stateOffset + S_THRESHOLDS + 6u] = stateAt(S_THRESHOLDS); state[stateOffset + S_THRESHOLDS + 7u] = upperValue; state[stateOffset + S_THRESHOLDS + 8u] = upperSign;
let current = stateAt(S_CURRENT); let noiseScale = stateAt(S_NOISE);
var scale = 1.0; var accepted = false; var chosen = 0u; var limitedByBound = false;
for (var attempt = 0u; attempt < ${I}u; attempt++) {
 let candidateLikelihood = stateAt(S_LINE_SEARCH + attempt); let lastLargestLinear = stateAt(S_LINE_SEARCH + ${I}u + attempt);
 // Halve the step until the likelihood does not decrease beyond its noise and every row the step puts on the wrong side stays bounded (|eta| <= 12, the largest value the line-search pass reports); rows on the right side are not bounded.
 let likelihoodAccepts = candidateLikelihood >= current - noiseScale * (abs(current) + 1.0) && isFiniteBits(candidateLikelihood);
 if (likelihoodAccepts && lastLargestLinear <= 12.0) { accepted = true; chosen = attempt; break; }
 // A step the likelihood would take but the bound refuses: the accepted step is shortened by the bound, not by the likelihood.
 if (likelihoodAccepts) { limitedByBound = true; } scale *= 0.5; }
if (!accepted) { state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
var largest = 0.0; var magnitude = 0.0; var largestDirection = 0.0;
for (var column = 0u; column < P; column++) { let direction = stateAt(S_DIRECTION + column); let step = scale * direction; let candidate = stateAt(S_ESTIMATE + column) + step; largest = max(largest, abs(step)); largestDirection = max(largestDirection, abs(direction)); magnitude = max(magnitude, abs(candidate)); state[stateOffset + S_ESTIMATE + column] = candidate; }
state[stateOffset + S_CURRENT] = stateAt(S_LINE_SEARCH + chosen); state[stateOffset + S_ITERATIONS] = stateAt(S_ITERATIONS) + 1.0;
if (largest <= ${Q} * (1.0 + magnitude)) {
 // A small accepted step means convergence only when the full Fisher step is small too. A step that is small because the bound shortened it is a fit whose maximum needs a misclassified row beyond the bound, which float32 cannot evaluate: not converged (the code stays 3 and the iterate is written), and no further step can do better.
 if (limitedByBound && !(largestDirection <= ${Q} * (1.0 + magnitude))) { state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
 // Separated data have no maximum to converge to: a small step there is a step along the separating direction below the tolerance, or rounding. Status 4, not 0.
 if (stateAt(S_SEPARATING) == 1.0) { state[stateOffset + S_CODE] = 4.0; state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
 // A direction of this fit could not be told from a separating one: the small step is not accepted as a maximum (the code stays 3).
 if (stateAt(S_UNDECIDED) == 1.0) { state[stateOffset + S_DONE] = 1.0; ${A.disarmWGSL} return; }
 state[stateOffset + S_CODE] = 0.0; }`))}if($.push(Z(`finish`,[S(`state`,E),S(`coefficients`,l.output.coefficients,`read_write`),S(`errors`,l.output.standardErrors,`read_write`),S(`summary`,l.output.summary,`read_write`),S(`status`,l.output.status,`read_write`,`u32`)],`${f}\n${G}\n${p}\n${g(b)}`,`for (var column = 0u; column < P; column++) { coefficients[coefficientsOffset + column] = getNaN(); errors[errorsOffset + column] = getNaN(); }
for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); }
let code = u32(stateAt(S_CODE)); status[statusOffset] = code;
if (stateAt(S_COUNTED) == 0.0) { return; }
summary[summaryOffset] = stateAt(S_PREPARE); summary[summaryOffset + 3u] = 0.0;
if (code == 1u || code == 2u || code == 4u) { return; }
// Separation: a direction that provably moves no row towards the wrong side was found, so there is no maximum, however the fit ended. An unfinished fit of data without a certificate stays status 3 and publishes its iterate, however large its linear predictors are.
if (stateAt(S_SEPARATING) == 1.0) { status[statusOffset] = 4u; return; }
var information: array<f32, ${b*b}>; for (var entry = 0u; entry < ${b*b}u; entry++) { information[entry] = stateAt(S_FACTOR + entry); }
var intercept = stateAt(S_ESTIMATE); var interceptTransform: array<f32, ${b}>; interceptTransform[0] = 1.0;
for (var column = 0u; column < C; column++) { let scale = stateAt(S_SCALES + column); let mean = stateAt(S_MEANS + column); let coefficient = stateAt(S_ESTIMATE + column + 1u) * scale; coefficients[coefficientsOffset + column + 1u] = coefficient; intercept -= mean * coefficient; interceptTransform[column + 1u] = -mean * scale; if (code == 0u) { errors[errorsOffset + column + 1u] = scale * sqrt(choleskyInverseDiagonal_${b}(&information, column + 1u)); } }
coefficients[coefficientsOffset] = intercept;
if (code == 0u) { var solved = interceptTransform; choleskySolve_${b}(&information, &solved); var variance = 0.0; for (var column = 0u; column < P; column++) { variance += solved[column] * interceptTransform[column]; } errors[errorsOffset] = sqrt(max(variance, 0.0)); }
let current = stateAt(S_CURRENT);
summary[summaryOffset + 1u] = current; summary[summaryOffset + 2u] = -2.0 * current; summary[summaryOffset + 3u] = stateAt(S_ITERATIONS);`),r(i,{id:`${o}-predict`,operation:j,variant:`predict`,invocationCount:h,bindings:[...q,S(`status`,l.output.status,`read`,`u32`),S(`fitted`,l.output.fitted,`read_write`)],declarations:K,body:`var prediction = getNaN(); let code = status[statusOffset]; if ((code == 0u || code == 3u) && included(index)) { let design = getDesign(index); var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); } if (isFiniteBits(linear)) { prediction = probability(linear); } } fitted[fittedOffset + index] = prediction;`}),...X(`effects`,2,0,`${K}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let design = getDesign(row); var linear = 0.0;
 for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
 (*terms)[0] += density(linear); (*terms)[1] += 1.0;
}`,w.effects,M(`effects`,2),q),r(i,{id:`${o}-effects`,operation:j,variant:`average-marginal-effects`,invocationCount:y,bindings:[S(`state`,E),S(`coefficients`,l.output.coefficients),S(`effects`,l.output.marginalEffects,`read_write`)],declarations:G,body:`effects[effectsOffset + index] = stateAt(S_EFFECTS) / stateAt(S_EFFECTS + 1u) * coefficients[coefficientsOffset + index + 1u];`})),l.spatialDiagnostics){let t=l.spatialDiagnostics,n=t.weights.neighbors.length,a=e(i,`${o}-diagnostic-residuals`,`float32`,h*V);c(o,i,[...Object.values(t.output),...m(t.weights)]),$.push(r(i,{id:`${o}-diagnostic-residuals`,operation:j,variant:`generalized-residuals`,invocationCount:h,bindings:[...q,S(`fitted`,l.output.fitted),S(`residuals`,a,`read_write`)],declarations:K,body:`for (var slot = 0u; slot < 5u; slot++) { residuals[residualsOffset + index * 5u + slot] = getNaN(); }
let fittedProbability = fitted[fittedOffset + index]; let observed = response[responseOffset + index];
if (!isFiniteBits(observed) || !isFiniteBits(fittedProbability) || !(fittedProbability > 0.0 && fittedProbability < 1.0) ${l.mask?`|| mask[maskOffset + index] == 0u`:``}) { return; }
let design = getDesign(index); var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
let rowDensity = density(linear); let variance = fittedProbability * (1.0 - fittedProbability); let naive = observed - fittedProbability;
residuals[residualsOffset + index * 5u] = naive; residuals[residualsOffset + index * 5u + 1u] = rowDensity * naive / variance; residuals[residualsOffset + index * 5u + 2u] = naive / sqrt(variance); residuals[residualsOffset + index * 5u + 3u] = variance; residuals[residualsOffset + index * 5u + 4u] = rowDensity * rowDensity / variance;`}),...X(`diagnostics`,7,1,`${u}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!isFiniteBits(residuals[residualsOffset + row * 5u])) { return; }
 (*terms)[0] += 1.0; (*terms)[1] += residuals[residualsOffset + row * 5u + 4u];
 let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u]; if (end < begin || end > ${n}u) { (*terms)[SUM_COUNT] = 1.0; return; }
 for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ROW_COUNT || neighbor == row || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { (*terms)[SUM_COUNT] = 1.0; return; }
 if (!isFiniteBits(residuals[residualsOffset + neighbor * 5u])) { continue; }
 var reciprocal = 0.0; for (var other = offsets[offsetsOffset + neighbor]; other < min(offsets[offsetsOffset + neighbor + 1u], ${n}u); other++) { if (neighbors[neighborsOffset + other] == row) { reciprocal = weights[weightsOffset + other]; } }
 let contribution = weight * (weight + reciprocal); (*terms)[2] += contribution; (*terms)[3] += contribution * residuals[residualsOffset + row * 5u + 3u] * residuals[residualsOffset + neighbor * 5u + 3u];
 (*terms)[4] += weight * residuals[residualsOffset + row * 5u] * residuals[residualsOffset + neighbor * 5u]; (*terms)[5] += weight * residuals[residualsOffset + row * 5u + 1u] * residuals[residualsOffset + neighbor * 5u + 1u]; (*terms)[6] += weight * residuals[residualsOffset + row * 5u + 2u] * residuals[residualsOffset + neighbor * 5u + 2u]; }
}`,w.diagnostics,M(`diagnostics`,8),[S(`offsets`,t.weights.offsets,`read`,`u32`),S(`neighbors`,t.weights.neighbors,`read`,`u32`),S(`weights`,t.weights.weights),S(`residuals`,a)]),r(i,{id:`${o}-spatial-diagnostics`,operation:j,variant:`pinkse-kp-pinkse-slade`,invocationCount:1,bindings:[S(`state`,E),...t.weights.status?.overflow?[S(`overflow`,t.weights.status.overflow,`read`,`u32`)]:[],S(`modelStatus`,l.output.status,`read`,`u32`),S(`tests`,t.output.tests,`read_write`),S(`diagnosticStatus`,t.output.status,`read_write`,`u32`)],declarations:`${f}\n${d}\n${G}`,body:`diagnosticStatus[diagnosticStatusOffset] = 1u; for (var slot = 0u; slot < 6u; slot++) { tests[testsOffset + slot] = getNaN(); } if (modelStatus[modelStatusOffset] != 0u) { return; }
${t.weights.status?.overflow?`if (overflow[overflowOffset] != 0u) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }`:``}
if (stateAt(S_DIAGNOSTICS + 7u) != 0.0) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }
let count = stateAt(S_DIAGNOSTICS); var information = stateAt(S_DIAGNOSTICS + 1u); let trace = stateAt(S_DIAGNOSTICS + 2u); let varianceTrace = stateAt(S_DIAGNOSTICS + 3u);
let naiveProduct = stateAt(S_DIAGNOSTICS + 4u); let generalizedProduct = stateAt(S_DIAGNOSTICS + 5u); let standardizedProduct = stateAt(S_DIAGNOSTICS + 6u);
if (!(count > 0.0)) { return; } diagnosticStatus[diagnosticStatusOffset] = 3u; information /= count; if (!(trace > 0.0 && varianceTrace > 0.0 && information > 0.0)) { return; }
let pinkse = generalizedProduct * generalizedProduct / (information * information * trace); let kelejian = naiveProduct / sqrt(varianceTrace); let slade = standardizedProduct * standardizedProduct / trace;
tests[testsOffset] = pinkse; tests[testsOffset + 1u] = erfcValue(sqrt(pinkse * 0.5)); tests[testsOffset + 2u] = kelejian; tests[testsOffset + 3u] = erfcValue(abs(kelejian) * 0.7071067811865476); tests[testsOffset + 4u] = slade; tests[testsOffset + 5u] = erfcValue(sqrt(slade * 0.5)); diagnosticStatus[diagnosticStatusOffset] = 0u;`}))}return $}};function oe(e){let t=e+1,n=0,r=e=>{let t=n;return n+=e,t};return{prepare:r(e+3),variance:r(6*e),rawMinimum:r(e),rawMaximum:r(e),means:r(e),scales:r(e),code:r(1),done:r(1),counted:r(1),iterations:r(1),iteration:r(1),current:r(16+z*e),noise:r(1),largest:r(1),separating:r(1),undecided:r(1),thresholds:r(9),runtimeZero:r(1),estimate:r(t),direction:r(t),separationDirection:r(t),factor:r(t*t),information:r(t+t*(t+1)/2+2),lineSearch:r(62),effects:r(2),diagnostics:r(8),length:n}}var U=[-1.2,2.4,-1.8],W=[[0,1],[.3,2],[.55,3],[.8,8]],G=[C+.08,.08,.97,.34],K=y({parameters:x,caption:`Fitted probabilities (left) and probit coefficients against the truth (right).`,createInputs:({random:e,world:t})=>D(e,t),build(e,t,n){let r=t.tracts.count,i=e.upload(`predictors`,`float32`,t.predictors),a=e.allocate(`response`,`float32`,r),o=e.allocate(`coefficients`,`float32`,3),s=e.allocate(`fitted`,`float32`,r),c=e.allocate(`marginal-effects`,`float32`,2);e.graph.add(new H({predictors:i.view,predictorCount:2,response:a.view,maximumIterations:n.maximumIterations,output:{coefficients:o.view,standardErrors:e.allocate(`standard-errors`,`float32`,3).view,fitted:s.view,marginalEffects:c.view,summary:e.allocate(`summary`,`float32`,4).view,status:e.allocate(`status`,`uint32`,1).view}}));let l=w(e,o,1,U,[`int`,`x1`,`x2`]);return{outputs:{mesh:S(e,t),fitted:s,response:a,marginalEffects:c,panel:l},update:e=>{a.write(Float32Array.from({length:r},(n,r)=>+(U[0]+U[1]*t.predictors[r*2]+U[2]*t.predictors[r*2+1]+t.normals[r]*e.noise>0))),l.truth.write(Float32Array.from(U,t=>t/e.noise))}}},rebuildOn:[`maximumIterations`],buildCache:5,interaction:{kind:`time`,hint:`step the iterations`,rest:[.9,.5],resolve(e,{outputs:t,peek:n,values:r}){let i=W.reduce((t,[n,r])=>e.unitX>=n?r:t,1),a=n(t.panel.estimates),o=e.x;return{settle:{maximumIterations:i},glyphs:[{shape:`curtain`,x:o}],readout:`maximumIterations ${i} · x1 ${a?a[1].toFixed(2):`-`} (true ${(U[1]/r.noise).toFixed(2)})`}}},draw:({inputs:e,outputs:t,world:n,theme:r})=>{let i=b({labels:[{text:`fitted probability of 1`,at:[e.mapWidth/2,n.height],offset:[0,-12],role:`label`,priority:4},{text:`coefficients (dot: latent truth)`,at:[(C+.08)*n.width,.9*n.height],anchor:`bottom-left`,offset:[0,3],role:`label`,priority:4},{text:`marginal effects`,at:[(C+.08)*n.width,G[3]*n.height],anchor:`bottom-left`,offset:[0,3],role:`label`,priority:4}]});return[...O(e,t.mesh,t.fitted,{ramp:`sequential`,range:[0,1]}),{layer:`dots`,positions:t.mesh.centers,values:t.response,ramp:`sequential`,range:[0,1],radius:3.2},...T(t.panel,{rectangle:[C+.08,.42,.97,.9],world:n,ink:r.ink}),...k(t.marginalEffects,G,A[1]/4),E(t.panel.zeros,G),{layer:`text`,...i}]}});export{K as default};