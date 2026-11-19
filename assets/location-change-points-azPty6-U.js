import{t as e}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{r as t,u as n}from"./wgsl-common-DfyXiYB0.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as i}from"./wgsl-poisson-observed-term-DLwa4Kd3.js";import{D as a,E as o,T as s,f as c,g as l,v as u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p,c as m,d as h,f as g,i as _,l as v,n as y,o as b,p as x,r as S,s as C,t as w,u as T}from"./dixie-cube-CZf2ZI2K.js";import{t as E}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as D}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as O}from"./define-run-B-brkCe4.js";import{n as k}from"./readback-cfnHpL6G.js";import{jl as A}from"./entry-preview-D70TdSbK.js";var j=`GPULocationChangePoints`,M=4294967295,N={mean:1,poisson:1,variance:2,"mean-variance":2,slope:3},P=class{id;props;constructor(e){let t=e.id??`location-change-points`;this.id=t,this.props=e;let{values:n,cellCount:r,sliceCount:i,cost:l,maximumChangePoints:u,parameters:d,output:p}=e;s(t,l in N,`cost is not supported`),c(t,`cellCount`,r),c(t,`sliceCount`,i,1,512),l===`slope`&&c(t,`sliceCount of the 'slope' cost`,i,1,256),c(t,`maximumChangePoints`,u,1,128),o(n,[`float32`],`${t} values`),a(t,`values`,n,r*i),o(d,[`float32`],`${t} parameters`),s(t,d.length>=2,`parameters needs 2 rows`),o(p.changePointCount,[`uint32`],`${t} changePointCount`),a(t,`changePointCount`,p.changePointCount,r),o(p.changePoints,[`uint32`],`${t} changePoints`),a(t,`changePoints`,p.changePoints,r*u),p.requiredChangePointCounts&&(o(p.requiredChangePointCounts,[`uint32`],`${t} requiredChangePointCounts`),a(t,`requiredChangePointCounts`,p.requiredChangePointCounts,r)),p.optimalCost&&(o(p.optimalCost,[`float32`],`${t} optimalCost`),a(t,`optimalCost`,p.optimalCost,r)),p.status?.overflow&&o(p.status.overflow,[`uint32`],`${t} status.overflow`),D(t,{...p,status:void 0,...p.status}),f(t,[p.changePointCount,p.changePoints,p.requiredChangePointCounts,p.optimalCost,p.status?.overflow],[n,d])}getParameterValues(e,t=new Float32Array(2)){let{id:n,props:r}=this;l(n,`penalty`,e.penalty,0);let i=N[r.cost],a=e.minimumSegmentLength??i;return c(n,`minimumSegmentLength`,a,i,r.sliceCount),t.set([e.penalty,a]),t}getCommandNodes(e){let{id:t,props:n}=this;E(e,t,n);let{values:i,layout:a,cellCount:o,sliceCount:s,cost:c,maximumChangePoints:l,parameters:f,output:p}=n;return d(t,e,[i,f,p.changePointCount,p.changePoints,p.requiredChangePointCounts,p.optimalCost,p.status?.overflow]),u(e.device,t,`values`,o*s*4,`reduce cellCount * sliceCount`),u(e.device,t,`changePoints`,o*l*4,`lower maximumChangePoints or split the cells`),[...p.status?.overflow?[r(e,{id:`${t}-clear-status`,operation:j,variant:`clear-status`,bindings:[{name:`overflowStatus`,view:p.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`overflowStatus[overflowStatusOffset] = 0u;`})]:[],r(e,{id:`${t}-dynamic-program`,operation:j,variant:c,bindings:[{name:`values`,view:i,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`changePointCount`,view:p.changePointCount,type:`u32`,access:`read_write`},{name:`changePoints`,view:p.changePoints,type:`u32`,access:`read_write`},...p.requiredChangePointCounts?[{name:`requiredChangePointCounts`,view:p.requiredChangePointCounts,type:`u32`,access:`read_write`}]:[],...p.optimalCost?[{name:`optimalCost`,view:p.optimalCost,type:`f32`,access:`read_write`}]:[],...p.status?.overflow?[{name:`overflowStatus`,view:p.status.overflow,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:o*2,workgroupSize:2,declarations:F(c,a,o,s,l),body:L(c,p)})]}};function F(r,a,o,s,c){let l=s+1;return`
${n}
${t}
${e}
${r===`poisson`?i:``}
var<workgroup> position: array<u32, ${s}>;
var<workgroup> series: array<f32, ${s}>;
var<workgroup> bestCost: array<f32, ${l}>;
var<workgroup> previousEnd: array<u32, ${l}>;
const CELL_COUNT: u32 = ${o}u;
const SLICE_COUNT: u32 = ${s}u;
const MAXIMUM_CHANGE_POINTS: u32 = ${c}u;
const NO_CHANGE_POINT: u32 = ${M}u;
const TAU: f32 = 6.283185307179586;
// Smallest segment variance as a fraction of the series variance (segment variances are computed
// from the standardized values, so the comparison is in standardized units and does not depend on
// the unit of the data): a segment with a zero variance would otherwise cost minus infinity. The
// changepoint package substitutes 1e-11 in data units when a variance is not positive.
const VARIANCE_FLOOR: f32 = 1e-11;
${x(a)}
// Compensated sum of the whole series: (sum, accumulated rounding error). The error comes from the
// integer significands: the float form (sum - next) + value is algebraically zero and is folded away.
fn addCompensated(accumulator: vec2<f32>, value: f32) -> vec2<f32> {
  let sum = accumulator.x + value;
  return vec2<f32>(sum, accumulator.y + getAdditionRoundingError(accumulator.x, value, sum));
}`}function I(e){switch(e){case`mean`:return`segmentCost = max(squares - total * total / segmentLength, 0.0);`;case`variance`:return`let variance = max(squares / segmentLength, VARIANCE_FLOOR);
      segmentCost = segmentLength * log(variance);`;case`mean-variance`:return`let variance = max((squares - total * total / segmentLength) / segmentLength, VARIANCE_FLOOR);
      segmentCost = segmentLength * log(variance);`;case`poisson`:return`segmentCost = -2.0 * getObservedTerm(max(total, 0.0), segmentLength * overallRate);`;case`slope`:return`let centeredSquares = squares - total * total / segmentLength;
      // Exact in u32: positions are below 256, so n * sum(t^2) stays below 2^32.
      let timeVariance = f32((end - start) * timeSecond - timeFirst * timeFirst) / segmentLength;
      let covariance = timeValue - f32(timeFirst) * total / segmentLength;
      var residual = centeredSquares;
      if (timeVariance > 0.0) {
        residual -= covariance * covariance / timeVariance;
      }
      segmentCost = max(residual, 0.0);`;default:return``}}function L(e,t){let n=e===`poisson`,r=e===`slope`,i=e===`mean`||r,a=e!==`variance`&&!n,o=t.requiredChangePointCounts?`requiredChangePointCounts[requiredChangePointCountsOffset + cellIndex] = found;`:``,s=t.status?.overflow?`if (found > MAXIMUM_CHANGE_POINTS) {
    atomicOr(&overflowStatus[overflowStatusOffset], 1u);
  }`:``,c=t.optimalCost?`optimalCost[optimalCostOffset + cellIndex] = select(getNaN(), ${V(e)}, previousEnd[seriesLength] != 0xffffffffu);`:``,l=t.optimalCost?`optimalCost[optimalCostOffset + cellIndex] = getNaN();`:``;return`if (localInvocationIndex != 0u) {
    return;
  }
  let cellIndex = index / 2u;
  var seriesLength = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = readSeriesValue(cellIndex, slice);
    if (isFiniteValue(value)) {
      position[seriesLength] = slice;
      seriesLength++;
    }
  }
  var sumAccumulator = vec2<f32>(0.0, 0.0);
  for (var row = 0u; row < seriesLength; row++) {
    sumAccumulator = addCompensated(sumAccumulator, readSeriesValue(cellIndex, position[row]));
  }
  let count = f32(seriesLength);
  let overallSum = sumAccumulator.x + sumAccumulator.y;
  let center = ${n?`0.0`:`select(0.0, overallSum / count, seriesLength > 0u)`};
  var squareAccumulator = vec2<f32>(0.0, 0.0);
  for (var row = 0u; row < seriesLength; row++) {
    let deviation = readSeriesValue(cellIndex, position[row]) - center;
    squareAccumulator = addCompensated(squareAccumulator, deviation * deviation);
  }
  let scale = select(0.0, (squareAccumulator.x + squareAccumulator.y) / count, seriesLength > 0u);
  let overallRate = select(0.0, overallSum / count, seriesLength > 0u);
  var found = 0u;
  let hasStructure = seriesLength >= 2u && ${n?`overallRate > 0.0`:`scale > 0.0`};
  if (hasStructure) {
    let deviationScale = ${n?`1.0`:`sqrt(scale)`};
    for (var row = 0u; row < seriesLength; row++) {
      let observed = readSeriesValue(cellIndex, position[row]);
      series[row] = ${n?`observed`:`(observed - center) / deviationScale`};
    }
    let penalty = parameters[parametersOffset]${i?` / scale`:``};
    let minimumLength = max(u32(parameters[parametersOffset + 1u]), 1u);
    bestCost[0] = -penalty;
    for (var end = 1u; end <= seriesLength; end++) {
      bestCost[end] = 0x1.fffffep+127f;
      previousEnd[end] = 0xffffffffu;
    }
    // Forward form of F(end) = min over start of F(start) + C(start, end) + penalty: starts are
    // visited in ascending order, so F(start) is final when its segments are relaxed, and the
    // strict comparison keeps the smallest start among equal candidates. Each segment's moments
    // are running sums that begin at its own start, about its own first value: no difference of
    // two long prefix sums is taken, so nothing has to survive a cancellation (a shader compiler
    // that reassociates float sums also reorders the difference of two compensated prefixes).
    for (var start = 0u; start < seriesLength; start++) {
      if (start > 0u && previousEnd[start] == 0xffffffffu) {
        continue;
      }
      let origin = ${a?`series[start]`:`0.0`};
      var total = 0.0;
      var squares = 0.0;
      ${r?`let timeOrigin = position[start];
      var timeValue = 0.0;
      var timeFirst = 0u;
      var timeSecond = 0u;`:``}
      for (var end = start + 1u; end <= seriesLength; end++) {
        let offsetValue = series[end - 1u] - origin;
        total += offsetValue;
        squares += offsetValue * offsetValue;
        ${r?`let relativeTime = position[end - 1u] - timeOrigin;
        timeValue += f32(relativeTime) * offsetValue;
        timeFirst += relativeTime;
        timeSecond += relativeTime * relativeTime;`:``}
        if (end - start < minimumLength) {
          continue;
        }
        let segmentLength = f32(end - start);
        var segmentCost = 0.0;
        {
          ${I(e)}
        }
        let candidate = bestCost[start] + segmentCost + penalty;
        if (candidate < bestCost[end]) {
          bestCost[end] = candidate;
          previousEnd[end] = start;
        }
      }
    }
    // Backtracking visits the partition from the end; the first walk counts the change points.
    var cursor = seriesLength;
    loop {
      if (cursor == 0u || previousEnd[cursor] == 0xffffffffu || previousEnd[cursor] == 0u) {
        break;
      }
      found++;
      cursor = previousEnd[cursor];
    }
    ${o}
    ${s}
    let kept = min(found, MAXIMUM_CHANGE_POINTS);
    changePointCount[changePointCountOffset + cellIndex] = kept;
    for (var slot = 0u; slot < MAXIMUM_CHANGE_POINTS; slot++) {
      changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = NO_CHANGE_POINT;
    }
    // The second walk writes the earliest change points: the j-th from the end is slot found - 1 - j.
    cursor = seriesLength;
    for (var walked = 0u; walked < found; walked++) {
      let segmentStart = previousEnd[cursor];
      let slot = found - 1u - walked;
      if (slot < kept) {
        changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = position[segmentStart];
      }
      cursor = segmentStart;
    }
    ${c}
  } else {
    changePointCount[changePointCountOffset + cellIndex] = 0u;
    for (var slot = 0u; slot < MAXIMUM_CHANGE_POINTS; slot++) {
      changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = NO_CHANGE_POINT;
    }
    ${o}
    ${R(t,e)}
    ${z(l)}
  }`}function R(e,t){return e.optimalCost?`if (seriesLength >= 1u) {
      optimalCost[optimalCostOffset + cellIndex] = ${B(t)};
    }`:``}function z(e){return e?`if (seriesLength == 0u) {
      ${e}
    }`:``}function B(e){switch(e){case`mean`:case`slope`:return`0.0`;case`poisson`:return`0.0`;default:return`count * (log(TAU) + log(VARIANCE_FLOOR) + 1.0)`}}function V(e){switch(e){case`mean`:case`slope`:return`bestCost[seriesLength] * scale`;case`poisson`:return`bestCost[seriesLength] - 2.0 * overallSum * log(overallRate)`;default:return`bestCost[seriesLength] + count * (log(TAU) + log(scale) + 1.0)`}}var H=16,U=8,W=[`mean`,`slope`],G=[1,3],K=O({parameters:A,caption:`Dixie fire NDVI, 16 real Sentinel-2 scenes (2017-2024), segmented per cell by a penalized cost.`,createInputs:()=>({cube:b(H),focus:{cell:0}}),build(e,t,n){let r=e.allocate(`parameters`,`float32`,2),i=e.allocate(`change-point-count`,`uint32`,w),a=e.allocate(`change-points`,`uint32`,w*U),o=new P({values:e.upload(`cube`,`float32`,t.cube).view,layout:`cell-major`,cellCount:w,sliceCount:H,cost:W[n.cost],maximumChangePoints:U,parameters:r.view,output:{changePointCount:i.view,changePoints:a.view}});e.graph.add(o);let s=Float32Array.from({length:w},(e,n)=>t.cube[n*H]);return{outputs:{changePointCount:i,changePoints:a,fractions:e.upload(`fractions`,`float32`,v(H)),series:e.allocate(`series`,`float32`,H),context:e.upload(`context`,`float32`,g(s))},update:e=>r.write(o.getParameterValues({penalty:e.penalty,minimumSegmentLength:Math.max(e.minimumSegmentLength,G[e.cost])}))}},rebuildOn:[`cost`],buildCache:2,async afterRun({outputs:e,upload:t}){let n=await k(e.changePointCount);return{counts:t(`counts`,`float32`,g(Float32Array.from(n)))}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.27,.45],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=m(e.x,e.y,i);o>=0&&(t.focus.cell=o);let s=t.focus.cell,c=t.cube.slice(s*H,(s+1)*H);n.series.write(c);let l=[{shape:`box`,...C(s,i)}],u=a(n.changePointCount)?.[s]??0,d=a(n.changePoints),f=[];if(d)for(let e=0;e<Math.min(u,U);e++){let t=d[s*U+e];t!==4294967295&&f.push(t)}let g=[0,...f,H],v=[];for(let e=0;e<g.length-1;e++){let t=0,n=0;for(let r=g[e];r<g[e+1];r++)Number.isFinite(c[r])&&(t+=c[r],n++);if(!n)continue;let r=h(Math.min(y[1],Math.max(y[0],t/n)),i);v.push(T(_[g[e]],H,i),r,T(_[g[e+1]-1],H,i),r)}for(let e=0;r.cost===0&&e<v.length;e+=4)l.push({shape:`line`,from:[v[e],v[e+1]],to:[v[e+2],v[e+3]]});for(let e of f){let t=T((_[e-1]+_[e])/2,H,i);l.push({shape:`line`,from:[t,p[1]*i.height],to:[t,p[3]*i.height],dashed:!0})}return{glyphs:l,spotlight:null,readout:`${u} change points${f.length?`: ${f.map(e=>S[e]).join(`, `)}`:``} · cell [${s%48}, ${Math.floor(s/48)}]`}}},draw:({outputs:e,extra:t})=>[{layer:`raster`,context:!0,values:e.context,columns:80,rows:30,ramp:`grey`,range:[0,1]},{layer:`raster`,values:t.counts,columns:80,rows:30,ramp:`sequential`,range:[0,3],alpha:.85},{layer:`chart`,kind:`scatter`,values:e.series,xValues:e.fractions,xRange:[0,1],ramp:`sequential`,range:[y[0],y[1]],rectangle:p,radius:4}]});export{K as default};