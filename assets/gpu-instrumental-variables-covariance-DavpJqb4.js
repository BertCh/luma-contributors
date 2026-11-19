import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{E as t,T as n,f as r,p as i,u as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{i as c}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as u}from"./kernel-C3tfQOn4.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as f,r as p,u as m}from"./wgsl-common-DfyXiYB0.js";import{o as h}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as g}from"./spatial-regression-solve-Vf-sQz9O.js";import{i as _,n as v,t as y}from"./regression-row-tiles-7Xsw3Utq.js";import{c as b}from"./spatial-weights-CqWysstN.js";var x=1e-6,S=1e-5;function C(e){return`code`in e?e.code:e}function w(e,t,i){let a=i.method??`HC0`;n(e,[`HC0`,`HC1`,`HC2`,`HC3`,`HAC`,`CLUSTER`].includes(a),`method must be HC0, HC1, HC2, HC3, HAC or CLUSTER`),n(e,a===`HAC`==!!i.kernel,`kernel must be present exactly for method HAC`),n(e,a===`CLUSTER`==!!i.clusters,`clusters must be present exactly for method CLUSTER`),n(e,a===`CLUSTER`==(i.clusterCount!==void 0),`clusterCount must be present exactly for method CLUSTER`),i.clusterCount!==void 0&&r(e,`clusterCount`,i.clusterCount,1,t)}var T=256,E=16;function D(e,t){return t?T+4*(e+2):E}function O(e,t,n,r={}){let i=t+1,a=Math.ceil(e/Math.max(64,Math.ceil(e/4096))),o=r.clusterCount??e,s=Math.ceil(o/Math.max(64,Math.ceil(o/4096))),c=n*(n+1)/2+n*t+t,l=i**2;return{"covariance workspace":t+n+1+c+l+3,"covariance projection":2*t*n,"covariance score rows":e*i,"covariance row status/contributions/sort scratch":e,"covariance sum partials":a*(t+n+1),"covariance moment partials":a*c,"covariance leverage partials":a,"covariance meat partials":s*(l+ +!!r.kernel),...r.clusterCount?{"cluster scores":o*i,"cluster totals":o,"cluster segment offsets":o+1,"cluster occupied partials":s}:{}}}var k=class{props;id;status;estimatedCommandNodeCount;constructor(a){this.props=a,this.id=a.id??`instrumental-variables-covariance`,this.status=C(a.output.status);let o=a.residuals.length;r(this.id,`rows`,o,1,y),r(this.id,`coefficientCount`,a.coefficientCount,1,15),r(this.id,`instrumentCount`,a.instrumentCount,a.coefficientCount,16),w(this.id,o,a),a.kernel&&n(this.id,b(this.id,a.kernel)===o,`kernel row count must equal residuals.length`),this.estimatedCommandNodeCount=D(a.coefficientCount,!!a.clusters),i(this.id,this.estimatedCommandNodeCount);for(let[e,r,i,s]of[[`design`,a.design,`float32`,o*a.coefficientCount],[`instruments`,a.instruments,`float32`,o*a.instrumentCount],[`residuals`,a.residuals,`float32`,o],[`output.covariance`,a.output.covariance,`float32`,(a.coefficientCount+1)**2],[`output.status.code`,this.status,`uint32`,1],...a.clusters?[[`clusters`,a.clusters,`uint32`,o]]:[]])t(r,[i],this.id),n(this.id,r.length===s,`${e} length must equal ${s}`);e(this.id,{covariance:a.output.covariance,status:this.status},[[`covariance`,`status`]]),s(this.id,[a.output.covariance,this.status],[a.design,a.instruments,a.residuals,a.clusters,a.kernel?.offsets,a.kernel?.neighbors,a.kernel?.weights,a.kernel?.status?.overflow])}getCommandNodes(e){let{id:t,props:n}=this,i=n.residuals.length,s=n.coefficientCount,b=n.instrumentCount,C=s+1;l(e,t,{design:n.design,instruments:n.instruments,residuals:n.residuals,clusters:n.clusters,kernelOffsets:n.kernel?.offsets,kernelNeighbors:n.kernel?.neighbors,kernelWeights:n.kernel?.weights,kernelOverflow:n.kernel?.status?.overflow,covariance:n.output.covariance,status:this.status}),c(e.device,t,O(i,s,b,n)),r(t,`rows`,i,1,Math.min(y,a(e.device,4*Math.max(b,C))));let w=v(i),T=b*(b+1)/2+b*s+s,E=C**2,D=s+b,k=D+1,A=k+T,j=u(e,`${t}-workspace`,`float32`,A+E+3),M=u(e,`${t}-projection`,`float32`,2*s*b),N=u(e,`${t}-scores`,`float32`,i*C),P=u(e,`${t}-row-status`,`float32`,i),F=n.method===`HC2`||n.method===`HC3`,I=u(e,`${t}-kernel-status`,`uint32`,i),L=n.kernel?u(e,`${t}-smoothed-scores`,`float32`,i*C):void 0,R=(e,t,n=`read`,r=`f32`)=>({name:e,view:t,access:n,type:r}),z=(n,r,i,a,o=``)=>d(e,{id:`${t}-${n}`,operation:`GPUInstrumentalVariablesCovariance`,variant:n,invocationCount:r,bindings:i,declarations:`${p}\n${f}\n${o}`,body:a});o(t,e,[n.design,n.instruments,n.residuals,n.clusters,n.kernel?.offsets,n.kernel?.neighbors,n.kernel?.weights,n.kernel?.status?.overflow,n.output.covariance,this.status]);let B=`${p}\n${f}
fn getColumn(row: u32, column: u32) -> f32 {
  if (column < ${s}u) { return design[designOffset + row * ${s}u + column]; }
  return instruments[instrumentsOffset + row * ${b}u + column - ${s}u];
}`,V=[..._(e,{id:`${t}-sums`,operation:`GPUInstrumentalVariablesCovariance`,variant:`sums`,rowCount:i,layout:w,sumCount:s+b,maximumCount:1,bindings:[R(`design`,n.design),R(`instruments`,n.instruments)],declarations:`${B}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var column = 0u; column < SUM_COUNT; column++) {
    let value = getColumn(row, column);
    (*terms)[column] += value;
    if (!isFiniteBits(value)) { (*terms)[SUM_COUNT] = 1.0; }
  }
}`,partial:u(e,`${t}-sum-partials`,`float32`,w.tileCount*(D+1)),output:j}),z(`means`,s+b,[R(`workspace`,j,`read_write`)],`workspace[workspaceOffset + index] = workspace[workspaceOffset + index] / ${i}.0;`),..._(e,{id:`${t}-moments`,operation:`GPUInstrumentalVariablesCovariance`,variant:`moments`,rowCount:i,layout:w,sumCount:T,bindings:[R(`design`,n.design),R(`instruments`,n.instruments),R(`workspace`,j)],declarations:`${B}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  var centered: array<f32, ${s+b}>;
  for (var column = 0u; column < ${s+b}u; column++) {
    centered[column] = getColumn(row, column) - workspace[workspaceOffset + column];
  }
  var slot = 0u;
  for (var first = 0u; first < ${b}u; first++) {
    for (var second = first; second < ${b}u; second++) {
      (*terms)[slot] += centered[${s}u + first] * centered[${s}u + second];
      slot++;
    }
  }
  for (var first = 0u; first < ${b}u; first++) {
    for (var column = 0u; column < ${s}u; column++) {
      (*terms)[slot] += centered[${s}u + first] * centered[column];
      slot++;
    }
  }
  for (var column = 0u; column < ${s}u; column++) {
    (*terms)[slot] += centered[column] * centered[column];
    slot++;
  }
}`,partial:u(e,`${t}-moment-partials`,`float32`,w.tileCount*T),output:j,outputSlot:k}),z(`projection`,1,[R(`workspace`,j),R(`projection`,M,`read_write`),R(`status`,this.status,`read_write`,`u32`)],`status[statusOffset] = 0u;
for (var slot = 0u; slot < ${s*b}u; slot++) { projection[projectionOffset + slot] = getNaN(); }
if (${i}u <= ${C}u) { status[statusOffset] = 2u; return; }
var nonfinite = workspace[workspaceOffset + ${D}u] != 0.0;
for (var column = 0u; column < ${s+b}u; column++) { nonfinite = nonfinite || !isFiniteBits(workspace[workspaceOffset + column]); }
if (nonfinite) { status[statusOffset] = 3u; return; }
var instrumentGram: array<f32, ${b*b}>;
var crossGram: array<f32, ${b*s}>;
var designScale: array<f32, ${s}>;
var slot = ${k}u;
for (var first = 0u; first < ${b}u; first++) { for (var second = first; second < ${b}u; second++) {
 let value = workspace[workspaceOffset + slot]; slot++;
 instrumentGram[first * ${b}u + second] = value; instrumentGram[second * ${b}u + first] = value;
} }
for (var first = 0u; first < ${b}u; first++) { for (var column = 0u; column < ${s}u; column++) { crossGram[first * ${s}u + column] = workspace[workspaceOffset + slot]; slot++; } }
for (var column = 0u; column < ${s}u; column++) { designScale[column] = workspace[workspaceOffset + slot]; slot++; }
var instrumentScale: array<f32, ${b}>;
for (var instrument = 0u; instrument < ${b}u; instrument++) { let diagonal = instrumentGram[instrument * ${b+1}u]; if (!(diagonal > 0.0)) { status[statusOffset] = 1u; return; } instrumentScale[instrument] = inverseSqrt(diagonal); }
for (var column = 0u; column < ${s}u; column++) { if (!(designScale[column] > 0.0)) { status[statusOffset] = 1u; return; } designScale[column] = inverseSqrt(designScale[column]); }
for (var first = 0u; first < ${b}u; first++) { for (var second = 0u; second < ${b}u; second++) { instrumentGram[first * ${b}u + second] *= instrumentScale[first] * instrumentScale[second]; } }
if (!choleskyFactor_${b}(&instrumentGram)) { status[statusOffset] = 1u; return; }
var projected: array<f32, ${b*s}>;
for (var column = 0u; column < ${s}u; column++) {
 var right: array<f32, ${b}>;
 for (var instrument = 0u; instrument < ${b}u; instrument++) { right[instrument] = crossGram[instrument * ${s}u + column] * instrumentScale[instrument] * designScale[column]; }
 choleskySolve_${b}(&instrumentGram, &right);
 for (var instrument = 0u; instrument < ${b}u; instrument++) { projected[instrument * ${s}u + column] = right[instrument];
 projection[projectionOffset + ${s*b}u + column * ${b}u + instrument] = right[instrument] * instrumentScale[instrument] / designScale[column]; }
}
var normal: array<f32, ${s*s}>;
for (var first = 0u; first < ${s}u; first++) { for (var second = 0u; second < ${s}u; second++) {
 var sum = 0.0; for (var instrument = 0u; instrument < ${b}u; instrument++) { sum += crossGram[instrument * ${s}u + first] * instrumentScale[instrument] * designScale[first] * projected[instrument * ${s}u + second]; }
 normal[first * ${s}u + second] = sum;
} }
if (!choleskyFactor_${s}(&normal)) { status[statusOffset] = 1u; return; }
for (var instrument = 0u; instrument < ${b}u; instrument++) { var right: array<f32, ${s}>;
 for (var column = 0u; column < ${s}u; column++) { right[column] = projected[instrument * ${s}u + column]; }
 choleskySolve_${s}(&normal, &right);
 for (var column = 0u; column < ${s}u; column++) { projection[projectionOffset + column * ${b}u + instrument] = right[column] * designScale[column] * instrumentScale[instrument]; }
}`,`${m}\n${g(b)}\n${s===b?``:g(s)}`),z(`scores`,i,[R(`instruments`,n.instruments),R(`workspace`,j),R(`projection`,M),R(`residuals`,n.residuals),R(`scores`,N,`read_write`),R(`rowStatus`,P,`read_write`),...n.clusters?[R(`clusters`,n.clusters,`read`,`u32`)]:[]],`var residual = residuals[residualsOffset + index]; var intercept = 1.0 / ${i}.0;
var influences: array<f32, ${s}>;
var leverage = 1.0 / ${i}.0;
for (var column = 0u; column < ${s}u; column++) { var influence = 0.0;
 var fitted = 0.0;
 for (var instrument = 0u; instrument < ${b}u; instrument++) {
  let centered = instruments[instrumentsOffset + index * ${b}u + instrument] - workspace[workspaceOffset + ${s}u + instrument];
  influence += projection[projectionOffset + column * ${b}u + instrument] * centered;
  ${F?`fitted += projection[projectionOffset + ${s*b}u + column * ${b}u + instrument] * centered;`:``}
 }
 influences[column] = influence;
 leverage += fitted * influence;
 intercept -= workspace[workspaceOffset + column] * influence;
}
var code = 0.0;
${F?`let denominator = 1.0 - leverage;
if (!isFiniteBits(leverage) || leverage < -${x} || denominator <= ${x}) { code = 5.0; }
residual /= ${n.method===`HC2`?`sqrt(denominator)`:`denominator`};`:``}
${n.clusters?`if (clusters[clustersOffset + index] >= ${n.clusterCount}u) { code = 6.0; }`:``}
rowStatus[rowStatusOffset + index] = code;
for (var column = 0u; column < ${s}u; column++) { scores[scoresOffset + index * ${C}u + column + 1u] = residual * influences[column]; }
scores[scoresOffset + index * ${C}u] = residual * intercept;`)];V.push(..._(e,{id:`${t}-row-status`,operation:`GPUInstrumentalVariablesCovariance`,variant:`row-status`,rowCount:i,layout:w,sumCount:0,maximumCount:1,bindings:[R(`rowStatus`,P)],declarations:`fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
        (*terms)[0] = max((*terms)[0], rowStatus[rowStatusOffset + row]);
      }`,partial:u(e,`${t}-row-status-partials`,`float32`,w.tileCount),output:j,outputSlot:A+E+1}));let H=N,U=i;if(n.clusters){U=n.clusterCount,H=u(e,`${t}-cluster-scores`,`float32`,U*C);let r=Array.from({length:C+1},(n,r)=>u(e,`${t}-contribution-${r}`,`float32`,i)),a=r.map((n,r)=>u(e,`${t}-cluster-total-${r}`,`float32`,U));for(let e=0;e<=C;e++)V.push(z(`cluster-contribution-${e}`,i,[R(`scores`,N),R(`contribution`,r[e],`read_write`)],`contribution[contributionOffset + index] = ${e===C?`1.0`:`scores[scoresOffset + index * ${C}u + ${e}u]`};`));V.push(...h(e,{id:`${t}-clusters`,operation:`GPUInstrumentalVariablesCovariance`,segmentCount:U,segmentKeys:n.clusters,reductions:r.map((e,t)=>({name:`score-${t}`,contributions:e,output:a[t]}))}));for(let e=0;e<C;e++)V.push(z(`cluster-pack-${e}`,U,[R(`total`,a[e]),R(`scores`,H,`read_write`)],`scores[scoresOffset + index * ${C}u + ${e}u] = total[totalOffset + index];`));let o=v(U);V.push(..._(e,{id:`${t}-occupied-clusters`,operation:`GPUInstrumentalVariablesCovariance`,variant:`occupied-clusters`,rowCount:U,layout:o,sumCount:1,bindings:[R(`counts`,a[C])],declarations:`fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
          (*terms)[0] += select(0.0, 1.0, counts[countsOffset + row] > 0.0);
        }`,partial:u(e,`${t}-occupied-partials`,`float32`,o.tileCount),output:j,outputSlot:A+E+2}))}if(n.kernel&&L){let e=n.kernel;V.push(z(`validate-kernel`,i,[R(`offsets`,e.offsets,`read`,`u32`),R(`neighbors`,e.neighbors,`read`,`u32`),R(`weights`,e.weights),R(`kernelStatus`,I,`read_write`,`u32`),...e.status?.overflow?[R(`overflow`,e.status.overflow,`read`,`u32`)]:[]],`let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u]; var invalid = begin > end || end > ${e.neighbors.length}u ${e.status?.overflow?`|| overflow[overflowOffset] != 0u`:``}; var diagonal = 0.0;
if (!invalid) { for (var slot = begin; slot < end; slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
 if (neighbor >= ${i}u || !isFiniteBits(weight) || weight < 0.0) { invalid = true; continue; }
 if (neighbor == index) { diagonal += weight; }
 let reverseBegin = offsets[offsetsOffset + neighbor]; let reverseEnd = offsets[offsetsOffset + neighbor + 1u];
 if (reverseBegin > reverseEnd || reverseEnd > ${e.neighbors.length}u) { invalid = true; continue; }
 var reverse = 0.0; var forward = 0.0;
 for (var other = reverseBegin; other < reverseEnd; other++) { if (neighbors[neighborsOffset + other] == index) { reverse += weights[weightsOffset + other]; } }
 for (var other = begin; other < end; other++) { if (neighbors[neighborsOffset + other] == neighbor) { forward += weights[weightsOffset + other]; } }
 invalid = invalid || abs(reverse - forward) > 1e-5 * max(1.0, abs(forward));
} }
invalid = invalid || abs(diagonal - 1.0) > 1e-5;
kernelStatus[kernelStatusOffset + index] = select(0u, 4u, invalid);`),z(`smooth-scores`,i,[R(`scores`,N),R(`offsets`,e.offsets,`read`,`u32`),R(`neighbors`,e.neighbors,`read`,`u32`),R(`weights`,e.weights),R(`smoothed`,L,`read_write`)],`var right: array<f32, ${C}>;
let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (begin <= end && end <= ${e.neighbors.length}u) { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${i}u) {
 let weight = weights[weightsOffset + slot];
 for (var second = 0u; second < ${C}u; second++) { right[second] += weight * scores[scoresOffset + neighbor * ${C}u + second]; } } } }
for (var second = 0u; second < ${C}u; second++) { smoothed[smoothedOffset + index * ${C}u + second] = right[second]; }`))}return V.push(..._(e,{id:`${t}-meat`,operation:`GPUInstrumentalVariablesCovariance`,variant:`meat`,rowCount:U,layout:v(U),sumCount:E,maximumCount:+!!n.kernel,bindings:[R(`scores`,H),...L?[R(`smoothed`,L),R(`kernelStatus`,I,`read`,`u32`)]:[]],declarations:`${f}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var first = 0u; first < ${C}u; first++) {
    let left = scores[scoresOffset + row * ${C}u + first];
    for (var second = 0u; second < ${C}u; second++) {
      (*terms)[first * ${C}u + second] += left * ${L?`smoothed[smoothedOffset`:`scores[scoresOffset`} + row * ${C}u + second];
    }
  }
  ${L?`if (kernelStatus[kernelStatusOffset + row] != 0u) { (*terms)[SUM_COUNT] = 1.0; }`:``}
}`,partial:u(e,`${t}-meat-partials`,`float32`,v(U).tileCount*(E+ +!!n.kernel)),output:j,outputSlot:A}),z(`finish`,1,[R(`workspace`,j),R(`covariance`,n.output.covariance,`read_write`),R(`status`,this.status,`read_write`,`u32`)],`var code = status[statusOffset];
${n.kernel?`if (workspace[workspaceOffset + ${A+E}u] != 0.0) { code = 4u; }`:``}
if (code == 0u) { code = u32(workspace[workspaceOffset + ${A+E+1}u]); }
${n.clusters?`let groups = workspace[workspaceOffset + ${A+E+2}u];
if (code == 0u && groups < 2.0) { code = 6u; }`:``}
let scale = ${n.method===`HC1`?`${i}.0 / ${Math.max(i-C,1)}.0`:n.clusters?`(groups / max(groups - 1.0, 1.0)) * ${i-1}.0 / ${Math.max(i-C,1)}.0`:`1.0`};
var matrix: array<f32, ${E}>;
var magnitude = 0.0;
for (var first = 0u; first < ${C}u; first++) { for (var second = 0u; second < ${C}u; second++) {
 let value = (0.5 * workspace[workspaceOffset + ${A}u + first * ${C}u + second] + 0.5 * workspace[workspaceOffset + ${A}u + second * ${C}u + first]) * scale;
 matrix[first * ${C}u + second] = value;
 if (code == 0u && !isFiniteBits(value)) { code = 3u; }
 magnitude = max(magnitude, abs(value));
} }
if (code == 0u && magnitude > 0.0) {
 var shifted = matrix;
 for (var first = 0u; first < ${C}u; first++) { for (var second = 0u; second < ${C}u; second++) {
   shifted[first * ${C}u + second] /= magnitude;
 } shifted[first * ${C+1}u] += ${S}; }
 if (!choleskyFactor_${C}(&shifted)) { code = 7u; }
}
for (var slot = 0u; slot < ${E}u; slot++) { covariance[covarianceOffset + slot] = select(getNaN(), matrix[slot], code == 0u); }
status[statusOffset] = code;`,`${m}\n${g(C)}`)),V}};export{w as a,C as i,O as n,D as r,k as t};