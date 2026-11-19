import{a as e,n as t,r as n}from"./trajectory-binding-validation-C89oJIAZ.js";import{r}from"./kernel-C3tfQOn4.js";import{r as i,u as a}from"./wgsl-common-DfyXiYB0.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as c}from"./scaled-quotient-wgsl-UDyhQag6.js";import{D as l,E as u,f as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{r as m,t as h}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{t as g}from"./define-run-B-brkCe4.js";import{n as _,o as v,r as y,t as b}from"./flash-cube-DKpwlO-P.js";import{n as x,r as S,t as C}from"./inset-CADbSeau.js";import{Xt as w}from"./entry-preview-D70TdSbK.js";var T=`GPUSpaceTimeCrossCorrelation`,E=2147483647,D=class{id;props;rowCount;constructor(t){this.id=t.id??`space-time-cross-correlation`,this.props=t;let r=this.id,{cellCount:i,periodCount:a,lags:o,output:s}=t;if(d(r,`cellCount`,i),d(r,`periodCount`,a),d(r,`lags length`,o.length,1,64),t.layout!==`period-major`&&t.layout!==`cell-major`)throw Error(`${r} layout must be 'period-major' or 'cell-major'`);for(let e of o)d(r,`lag`,Math.abs(e),0,a-4);if(new Set(o).size!==o.length)throw Error(`${r} lags must be distinct`);this.rowCount=i*o.length,d(r,`cube rows`,i*a),d(r,`correlation rows`,this.rowCount),d(r,`periodCount`,a,1,E);for(let[e,n]of[[`first`,t.first],[`second`,t.second]])u(n,[`float32`],`${r} ${e}`),l(r,e,n,i*a);u(s.correlations,[`float32`],`${r} correlations`),l(r,`correlations`,s.correlations,this.rowCount),u(s.pValues,[`float32`],`${r} pValues`),l(r,`pValues`,s.pValues,this.rowCount),s.adjustedPValues&&(u(s.adjustedPValues,[`float32`],`${r} adjustedPValues`),l(r,`adjustedPValues`,s.adjustedPValues,this.rowCount)),s.pairCounts&&(u(s.pairCounts,[`uint32`],`${r} pairCounts`),l(r,`pairCounts`,s.pairCounts,this.rowCount)),s.testedCount&&(u(s.testedCount,[`uint32`],`${r} output.testedCount`),l(r,`output.testedCount`,s.testedCount,1)),m(r,t.multipleTesting),n(this.id,s),e(this.id,[{correlations:s.correlations,pValues:s.pValues,pairCounts:s.pairCounts}]),p(r,[s.correlations,s.pValues,s.adjustedPValues,s.pairCounts,s.testedCount],[t.first,t.second])}getCommandNodes(e){t(e,this.id,this.props);let{props:n,id:l,rowCount:u}=this,{first:d,second:p,layout:m,cellCount:g,periodCount:_,lags:v,output:y}=n;f(l,e,[d,p,y.correlations,y.pValues,y.adjustedPValues,y.pairCounts,y.testedCount]);let b=(e,t)=>m===`period-major`?`(${t}) * CELLS + ${e}`:`${e} * PERIODS + (${t})`,x=[{name:`first`,view:d,type:`f32`,access:`read`},{name:`second`,view:p,type:`f32`,access:`read`},{name:`correlations`,view:y.correlations,type:`f32`,access:`read_write`},{name:`pValues`,view:y.pValues,type:`f32`,access:`read_write`},...y.pairCounts?[{name:`pairCounts`,view:y.pairCounts,type:`u32`,access:`read_write`}]:[]],S=[o(e,{id:`${l}-correlate`,operation:T,variant:`correlate`,bindings:x,invocationCount:u,declarations:`const CELLS: u32 = ${g}u;
const PERIODS: u32 = ${_}u;
const LAG_COUNT: u32 = ${v.length}u;
const MINIMUM_PAIRS: u32 = 4u;
${i}
${a}
${c}
// Numerical Recipes erfcc, relative error below 1.2e-7 for x >= 0.
fn erfcPositive(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.5 * x);
  let polynomial = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return t * exp(polynomial);
}`,body:`let cell = index / LAG_COUNT;
  let lagIndex = index % LAG_COUNT;
  var lagTable = array<i32, ${v.length}>(${v.map(e=>`${e}`).join(`, `)});
  let lag = lagTable[lagIndex];
  let start = max(0, -lag);
  let stop = i32(PERIODS) - max(lag, 0);
  let pairs = u32(max(stop - start, 0));
  ${y.pairCounts?`pairCounts[pairCountsOffset + index] = pairs;`:``}
  correlations[correlationsOffset + index] = getNaN();
  pValues[pValuesOffset + index] = getNaN();
  if (pairs < MINIMUM_PAIRS) {
    return;
  }
  var firstSum = 0.0;
  var secondSum = 0.0;
  var firstMinimum = 3.4e38;
  var firstMaximum = -3.4e38;
  var secondMinimum = 3.4e38;
  var secondMaximum = -3.4e38;
  for (var period = start; period < stop; period++) {
    let a = first[firstOffset + ${b(`cell`,`u32(period)`)}];
    let b = second[secondOffset + ${b(`cell`,`u32(period + lag)`)}];
    if (!isFiniteValue(a) || !isFiniteValue(b)) {
      return;
    }
    firstMinimum = min(firstMinimum, a);
    firstMaximum = max(firstMaximum, a);
    secondMinimum = min(secondMinimum, b);
    secondMaximum = max(secondMaximum, b);

  }
  // A constant series has no correlation; testing the range avoids a rounding residue in the mean.
  if (firstMinimum == firstMaximum || secondMinimum == secondMaximum) {
    return;
  }
  // Normalize before summing or squaring: finite observations can otherwise overflow the
  // mean, covariance or variance product even for a perfectly correlated series.
  let firstScale = max(abs(firstMinimum), abs(firstMaximum));
  let secondScale = max(abs(secondMinimum), abs(secondMaximum));
  let firstAnchor = getScaledQuotient(firstMinimum, firstScale, 1.0, 0);
  let secondAnchor = getScaledQuotient(secondMinimum, secondScale, 1.0, 0);
  for (var period = start; period < stop; period++) {
    firstSum += getScaledQuotient(first[firstOffset + ${b(`cell`,`u32(period)`)}], firstScale, 1.0, 0) - firstAnchor;
    secondSum += getScaledQuotient(second[secondOffset + ${b(`cell`,`u32(period + lag)`)}], secondScale, 1.0, 0) - secondAnchor;
  }
  let firstMean = firstSum / f32(pairs);
  let secondMean = secondSum / f32(pairs);
  var firstSquares = 0.0;
  var secondSquares = 0.0;
  var products = 0.0;
  for (var period = start; period < stop; period++) {
    let a = getScaledQuotient(first[firstOffset + ${b(`cell`,`u32(period)`)}], firstScale, 1.0, 0) - firstAnchor - firstMean;
    let b = getScaledQuotient(second[secondOffset + ${b(`cell`,`u32(period + lag)`)}], secondScale, 1.0, 0) - secondAnchor - secondMean;
    firstSquares += a * a;
    secondSquares += b * b;
    products += a * b;
  }
  let denominator = sqrt(firstSquares * secondSquares);
  if (!(denominator > 0.0)) {
    return;
  }
  let correlation = clamp(products / denominator, -1.0, 1.0);
  correlations[correlationsOffset + index] = correlation;
  if (abs(correlation) >= 1.0) {
    pValues[pValuesOffset + index] = 0.0;
    return;
  }
  let z = abs(0.5 * log((1.0 + correlation) / (1.0 - correlation))) * sqrt(f32(pairs) - 3.0);
  pValues[pValuesOffset + index] = clamp(erfcPositive(z * 0.70710678), 0.0, 1.0);`})],C=y.testedCount?r(e,`${l}-tested-count`,`uint32`,1):void 0;return y.adjustedPValues?S.push(...new h({id:`${l}-adjust`,pValues:y.pValues,method:n.multipleTesting??`benjamini-hochberg`,output:{adjustedPValues:y.adjustedPValues,testedCount:C}}).getCommandNodes(e)):C&&S.push(s(e,{id:`${l}-clear-tested-count`,operation:T,view:C,type:`u32`,value:`0u`}),o(e,{id:`${l}-tested-count`,operation:T,variant:`tested-count`,bindings:[{name:`pValues`,view:y.pValues,type:`f32`,access:`read`},{name:`testedCount`,view:C,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,declarations:a,body:`if (isFiniteValue(pValues[pValuesOffset + index])) {
  atomicAdd(&testedCount[testedCountOffset], 1u);
}`})),C&&y.testedCount&&S.push(o(e,{id:`${l}-publish-tested-count`,operation:T,variant:`publish-tested-count`,bindings:[{name:`count`,view:C,type:`u32`,access:`read`},{name:`testedCount`,view:y.testedCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`testedCount[testedCountOffset] = count[countOffset];`})),S}},O=6,k=9,A=[`none`,`bonferroni`,`holm`,`benjamini-hochberg`,`benjamini-yekutieli`],j=[.04,.06,.46,.34],M=[.54,.06,.96,.34];function N(e){return Array.from({length:2*e+1},(t,n)=>n-e)}function P(e,t,n){let r=-1,i=-1;for(let a=0;a<n;a++){let o=e[t*n+a];Number.isFinite(o)&&Math.abs(o)>i&&(i=Math.abs(o),r=a)}return r}function F(e,t){let n=t?M:j;return{x0:n[0]*e.width,y0:n[1]*e.height,x1:n[2]*e.width,y1:n[3]*e.height}}var I=g({parameters:w,caption:`GOES lightning, May 21–22 2024, in coarse lattice cells and 9 periods: flash count against mean flash area (empty periods take the cell mean); best lag where significant.`,createInputs:({world:e})=>{let t=b(e,O,k),n=new Float32Array(t.cellCount);for(let e=0;e<t.periodCount;e++)for(let r=0;r<t.cellCount;r++)n[r]+=t.counts[e*t.cellCount+r];return{cube:t,totals:n,focus:-1}},build(e,{cube:t,totals:n},r){let i=N(r.lags),a=t.cellCount*i.length,o={totals:e.upload(`totals`,`float32`,n),correlations:e.allocate(`correlations`,`float32`,a),adjustedPValues:e.allocate(`adjusted-p-values`,`float32`,a),pairCounts:e.allocate(`pair-counts`,`uint32`,a),backing:C(e,`inset`,F({width:t.width,height:1},!1))};return e.graph.add(new D({first:e.upload(`first`,`float32`,t.counts).view,second:e.upload(`second`,`float32`,t.meanArea).view,layout:`period-major`,cellCount:t.cellCount,periodCount:t.periodCount,lags:i,multipleTesting:A[r.multipleTesting],output:{correlations:o.correlations.view,pValues:e.allocate(`p-values`,`float32`,a).view,adjustedPValues:o.adjustedPValues.view,pairCounts:o.pairCounts.view}})),{outputs:o}},rebuildOn:[`lags`,`multipleTesting`],async afterRun({inputs:e,outputs:t,values:n,upload:r}){let{cube:i}=e,a=N(n.lags),o=await v(t.correlations),s=await v(t.adjustedPValues),c=new Float32Array(i.cellCount);for(let e=0;e<i.cellCount;e++){let t=P(o,e,a.length);c[e]=t>=0&&s[e*a.length+t]<n.significanceLevel?a[t]:NaN}return{mapped:r(`mapped`,`float32`,c)}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.41,.33],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let{cube:o}=t,s=N(r.lags),c=_(o,e.x,e.y).index;t.focus=c;let l=F(i,e.unitX<.4&&e.unitY>.6);n.backing.positions.write(S(l));let u=[{shape:`box`,...y(o,c)},{shape:`box`,minimum:[l.x0,l.y0],maximum:[l.x1,l.y1]}],d=.025,f=l.x1-l.x0-2*d,p=l.y1-l.y0-2*d,m=f/s.length,h=e=>l.x0+d+(e+.5)*m,g=l.y0+d+p/2,v=(e,t)=>[h(e),g+t*p/2];u.push({shape:`line`,from:[l.x0+d,g],to:[l.x1-d,g],dashed:!0});let b=a(n.correlations),x=a(n.adjustedPValues),C=a(n.pairCounts);if(!b||!x||!C)return{glyphs:u,spotlight:null,readout:`…`};let w=s.indexOf(0);u.push({shape:`line`,from:v(w,-1),to:v(w,-.82),dashed:!0});for(let e=0;e<s.length;e++){let t=b[c*s.length+e];Number.isFinite(t)&&u.push({shape:`line`,from:v(e,0),to:v(e,t)})}let T=P(b,c,s.length);if(T<0)return{glyphs:u,spotlight:null,readout:`untestable · flat series`};let E=c*s.length+T;u.push({shape:`ring`,at:v(T,b[E]),radius:.018});let D=s[T];return{glyphs:u,spotlight:null,readout:`best lag ${D>0?`+`:``}${D} · r ${b[E].toFixed(2)} · adj p ${x[E].toFixed(2)}`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>[{layer:`raster`,context:!0,values:t.totals,columns:e.cube.columns,rows:e.cube.rows,ramp:`grey`,range:[0,12]},...n?[{layer:`raster`,values:n.mapped,columns:e.cube.columns,rows:e.cube.rows,ramp:`diverging`,range:[-5,5]}]:[],x(t.backing,r)]});export{I as default};