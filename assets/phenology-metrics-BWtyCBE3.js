import{a as e,i as t,n,r,t as i}from"./raster-time-series-chunks-b-CPab_8L5.js";import{T as a,f as o,g as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{a as l,i as u,o as d,r as f,t as p}from"./raster-irregular-date-axis-B315UxmE.js";import{n as m,t as h}from"./gpu-raster-whittaker-smoother-DaVr5pK3.js";import{t as g}from"./define-run-B-brkCe4.js";import{c as _}from"./pointer-overlay-Qh8qflcA.js";import{a as v,b as y,c as b,d as x,g as S,n as C,o as w,p as T,r as E,s as D,t as O,u as k,v as A,x as j}from"./charts-Br0w6lsw.js";import{n as M,r as N,t as P}from"./probe-DLtMtDEc.js";import{n as F,r as I,t as L}from"./readback-C6WbEPkA.js";import{Gr as R}from"./entry-preview-D70TdSbK.js";var z=Object.freeze({startTime:0,endTime:1,peakTime:2,length:3,peakValue:4,baseValue:5,amplitude:6,leftSlope:7,rightSlope:8,integral:9});Object.freeze({startAtSeriesStart:1,endAtSeriesEnd:2,startAtOtherSeason:4,endAtOtherSeason:8});function B(e,t=new Float32Array(4)){let n=`getGPURasterPhenologyMetricsParameterValues`;s(n,`startFraction`,e.startFraction,Number.MIN_VALUE,1),s(n,`endFraction`,e.endFraction,Number.MIN_VALUE,1),s(n,`minimumAmplitude`,e.minimumAmplitude,0);let r=Math.max(e.startFraction,e.endFraction),i=e.secondSeasonMinimumFraction??r;return s(n,`secondSeasonMinimumFraction`,i,r,1),t.set([e.startFraction,e.endFraction,e.minimumAmplitude,i]),t}var V=class{id;props;chunkPlan;estimatedCommandNodeCount;ticksPerYear;constructor(t){this.props=t;let r=t.id??`raster-phenology-metrics`;this.id=r,o(r,`cellCount`,t.cellCount),o(r,`sliceCount`,t.sliceCount,2,256),o(r,`maximumSeasonCount`,t.maximumSeasonCount,1,2);let i=t.cellCount*t.sliceCount;o(r,`cellCount times sliceCount`,i);let s=t.cellCount*t.maximumSeasonCount;a(r,t.smoothed.length>=i,`smoothed must hold cellCount times sliceCount entries`),a(r,t.output.metrics.length>=s*10&&t.output.flags.length>=s,`output.metrics must hold 10 entries and output.flags one per cell and season`),a(r,t.output.seasonCount.length>=t.cellCount&&t.output.status.length>=t.cellCount,`output.seasonCount and output.status must hold one entry per cell`),a(r,t.parameters.length>=4,`parameters must hold 4 entries`),this.ticksPerYear=t.dateAxis?l(r,t.dateAxis,t.sliceCount):0,this.chunkPlan=n(this.id,t.cellCount,{smoothed:{view:t.smoothed,entriesPerCell:t.sliceCount},"output.metrics":{view:t.output.metrics,entriesPerCell:t.maximumSeasonCount*10},"output.flags":{view:t.output.flags,entriesPerCell:t.maximumSeasonCount},"output.seasonCount":{view:t.output.seasonCount,entriesPerCell:1},"output.status":{view:t.output.status,entriesPerCell:1}}),this.estimatedCommandNodeCount=this.chunkPlan.estimatedCommandNodeCount,e(this.id,this.getInputs(),this.getOutputs(),this.chunkPlan)}getInputs(){return[...this.props.dateAxis?[this.props.dateAxis.acquisitionTimes]:[],this.props.smoothed,this.props.parameters]}getOutputs(){let{metrics:e,flags:t,seasonCount:n,status:r}=this.props.output;return[e,t,n,r]}getCommandNodes(e){let{props:n,id:a}=this,{sliceCount:o,maximumSeasonCount:s,smoothed:l,parameters:p,output:m}=n,h=n.dateAxis;t(a,e,this.getInputs(),this.getOutputs()),h&&d(a,e,h);let g=e=>h?`times[${e}]`:`f32(${e})`;return i(e,this.chunkPlan,{id:a,operation:`GPURasterPhenologyMetrics`,invocationCount:n.cellCount,workgroupSize:64,bindings:[r(`smoothed`,l),r(`parameters`,p),...h?[f(h)]:[],r(`metrics`,m.metrics,!0),r(`flags`,m.flags,!0),r(`seasonCount`,m.seasonCount,!0),r(`status`,m.status,!0)],declarations:`${c}
${h?u(this.ticksPerYear,o):``}
fn curve(cell: u32, slice: u32) -> f32 { return smoothed[smoothedOffset + cell * ${o}u + slice]; }`,body:`let startFraction = parameters[parametersOffset];
let endFraction = parameters[parametersOffset + 1u];
let minimumAmplitude = parameters[parametersOffset + 2u];
let secondFraction = parameters[parametersOffset + 3u];
${h?`var times: array<f32, ${o}>;`:``}
var lowest = curve(index, 0u);
var highest = lowest;
var allFinite = true;
for (var slice = 0u; slice < ${o}u; slice++) {
  ${h?`times[slice] = rasterSliceYears(slice);`:``}
  let value = curve(index, slice);
  if (!finite(value)) { allFinite = false; }
  lowest = min(lowest, value);
  highest = max(highest, value);
}
let amplitude = highest - lowest;
var code = 0u;
if (!allFinite) { code = 1u; }
else if (!(amplitude > 0.0) || amplitude < minimumAmplitude) { code = 2u; }
var found = 0u;
var season: array<array<f32, 10>, ${s}>;
var seasonFlags: array<u32, ${s}>;
if (code == 0u) {
  let startLevel = lowest + startFraction * amplitude;
  let endLevel = lowest + endFraction * amplitude;
  var excludedLow = 1u;
  var excludedHigh = 0u;
  var hasExcluded = false;
  for (var candidateSeason = 0u; candidateSeason < ${s}u; candidateSeason++) {
    var peak = 0xffffffffu;
    var peakValue = 0.0;
    for (var slice = 0u; slice < ${o}u; slice++) {
      if (hasExcluded && slice >= excludedLow && slice <= excludedHigh) { continue; }
      let value = curve(index, slice);
      if (peak == 0xffffffffu || value > peakValue) { peak = slice; peakValue = value; }
    }
    if (peak == 0xffffffffu) { break; }
    if (candidateSeason > 0u && peakValue < lowest + secondFraction * amplitude) { break; }
    var lower = 0u;
    var upper = ${o-1}u;
    if (hasExcluded) {
      if (peak < excludedLow) { upper = excludedLow - 1u; } else { lower = excludedHigh + 1u; }
    }
    var boundaryFlags = 0u;
    var startTime = ${g(`lower`)};
    var startValue = curve(index, lower);
    var leftIndex = lower;
    var crossing = peak;
    var crossed = false;
    while (crossing > lower) {
      crossing--;
      if (curve(index, crossing) < startLevel) { crossed = true; break; }
    }
    if (crossed) {
      let below = curve(index, crossing);
      let above = curve(index, crossing + 1u);
      startValue = startLevel;
      startTime = ${g(`crossing`)} + (startLevel - below) / (above - below) * (${g(`crossing + 1u`)} - ${g(`crossing`)});
      leftIndex = crossing + 1u;
    } else {
      boundaryFlags |= select(4u, 1u, lower == 0u);
    }
    var endTime = ${g(`upper`)};
    var endValue = curve(index, upper);
    var rightIndex = upper;
    crossing = peak;
    crossed = false;
    while (crossing < upper) {
      crossing++;
      if (curve(index, crossing) < endLevel) { crossed = true; break; }
    }
    if (crossed) {
      let above = curve(index, crossing - 1u);
      let below = curve(index, crossing);
      endValue = endLevel;
      endTime = ${g(`crossing - 1u`)} + (above - endLevel) / (above - below) * (${g(`crossing`)} - ${g(`crossing - 1u`)});
      rightIndex = crossing - 1u;
    } else {
      boundaryFlags |= select(8u, 2u, upper == ${o-1}u);
    }
    var integral = 0.0;
    var previousTime = startTime;
    var previousHeight = startValue - lowest;
    for (var slice = leftIndex; slice <= rightIndex; slice++) {
      let height = curve(index, slice) - lowest;
      integral += 0.5 * (height + previousHeight) * (${g(`slice`)} - previousTime);
      previousTime = ${g(`slice`)};
      previousHeight = height;
    }
    integral += 0.5 * ((endValue - lowest) + previousHeight) * (endTime - previousTime);
    let peakTime = ${g(`peak`)};
    var entry: array<f32, 10>;
    entry[0] = startTime;
    entry[1] = endTime;
    entry[2] = peakTime;
    entry[3] = endTime - startTime;
    entry[4] = peakValue;
    entry[5] = lowest;
    entry[6] = peakValue - lowest;
    entry[7] = select(missing(), (peakValue - startValue) / (peakTime - startTime), peakTime > startTime);
    entry[8] = select(missing(), (endValue - peakValue) / (endTime - peakTime), endTime > peakTime);
    entry[9] = integral;
    season[candidateSeason] = entry;
    seasonFlags[candidateSeason] = boundaryFlags;
    hasExcluded = true;
    excludedLow = leftIndex;
    excludedHigh = rightIndex;
    found++;
  }
  ${s===2?`  if (found == 2u && season[1][2] < season[0][2]) {
    let swapped = season[0];
    season[0] = season[1];
    season[1] = swapped;
    let swappedFlags = seasonFlags[0];
    seasonFlags[0] = seasonFlags[1];
    seasonFlags[1] = swappedFlags;
  }`:``}
}
status[statusOffset + index] = code;
seasonCount[seasonCountOffset + index] = found;
for (var slot = 0u; slot < ${s}u; slot++) {
  flags[flagsOffset + index * ${s}u + slot] = select(0u, seasonFlags[slot], slot < found);
  for (var field = 0u; field < 10u; field++) {
    metrics[metricsOffset + (index * ${s}u + slot) * 10u + field] = select(missing(), season[slot][field], slot < found);
  }
}`})}},H=52,U=2,W=10,[G]=E(1),K=[-.2,1],q=g({parameters:R,caption:`Synthetic one-year cycle on measured Dixie NDVI levels, two seasons in the east; Whittaker-smoothed first.`,createInputs:({aspect:e,random:t})=>{let n=A(e,t,{sliceCount:H,yearCount:1,phaseDrift:.1,doubleSeasonFrom:.62});return{...n,observed:n.observe(.1,.05)}},build(e,t){let{grid:n,sliceCount:r}=t,i=e.upload(`series`,`float32`,t.observed),a=e.allocate(`whittaker-parameters`,`float32`,2);a.write(m({smoothingParameter:3}));let o=e.allocate(`phenology-parameters`,`float32`,4),s=j(e,t.times),c=e.allocate(`smoothed`,`float32`,n.cellCount*r),l={metrics:e.allocate(`metrics`,`float32`,n.cellCount*U*W),flags:e.allocate(`flags`,`uint32`,n.cellCount*U),seasonCount:e.allocate(`season-count`,`uint32`,n.cellCount),status:e.allocate(`status`,`uint32`,n.cellCount)};return e.graph.add(new h({cellCount:n.cellCount,sliceCount:r,values:i.view,parameters:a.view,reweightFunction:`upper-envelope`,dateAxis:s,output:{smoothed:c.view,status:e.allocate(`whittaker-status`,`uint32`,n.cellCount).view}})),e.graph.add(new V({cellCount:n.cellCount,sliceCount:r,maximumSeasonCount:U,smoothed:c.view,parameters:o.view,dateAxis:s,output:{metrics:l.metrics.view,flags:l.flags.view,seasonCount:l.seasonCount.view,status:l.status.view}})),{outputs:{...l,smoothed:c,blank:L(e,`blank`,n.cellCount),level:e.upload(`level`,`float32`,t.level),observations:O(e,`observations`,r),curve:O(e,`curve`,r),peaks:O(e,`peaks`,U)},update:e=>o.write(B({startFraction:e.startFraction,endFraction:e.endFraction,minimumAmplitude:e.minimumAmplitude,secondSeasonMinimumFraction:Math.max(e.secondSeasonMinimumFraction,e.startFraction,e.endFraction)}))}},async afterRun({outputs:e,inputs:t,upload:n}){let[r,i]=await Promise.all([F(e.metrics),I(e.seasonCount)]);return{map:n(`start-map`,`float32`,Float32Array.from({length:t.grid.cellCount},(e,t)=>i[t]>0?r[t*U*W+z.startTime]:NaN))}},interaction:{kind:`probe`,hint:`probe a pixel`,rest:[.72,.5],resolve:(e,{inputs:t,outputs:n,values:r,world:i,peek:a})=>{let{grid:o,sliceCount:s}=t,c=N(o,i,e),l=t.observed.subarray(c.index*s,(c.index+1)*s),u=a(n.smoothed),d=a(n.metrics),f=a(n.seasonCount),m=u?u.subarray(c.index*s,(c.index+1)*s):[],h=Array.from(t.years);S(n.observations,h,l),S(n.curve,h,m);let g=[t.years[0],t.years[s-1]],v=x(t.times),b=f?f[c.index]:0,C=(e,t)=>d?d[(c.index*U+e)*W+z[t]]:NaN,E=[],O=[],k=[M(o,i,c),w(i,G)];for(let e=0;e<b;e++)E.push(C(e,`peakTime`)),O.push(C(e,`peakValue`)),k.push(T(i,G,g,C(e,`startTime`),!0),T(i,G,g,C(e,`endTime`),!0));if(b>0){let e=C(0,`baseValue`),t=C(0,`amplitude`);k.push(D(i,G,K,e+r.startFraction*t),D(i,G,K,e+r.endFraction*t))}S(n.peaks,E,O);let A=e=>y(BigInt(Math.round(v+e*p))).slice(5);return{spotlight:null,glyphs:k,readout:f?b===0?`no season · amplitude under ${_(r.minimumAmplitude)} · ${P(o,c)}`:`${b} season${b>1?`s`:``} · ${A(C(0,`startTime`))} to ${A(C(0,`endTime`))} · ${P(o,c)}`:`reading the curve`}}},draw:({inputs:e,outputs:t,extra:n,theme:r})=>{let i=[e.years[0],e.years[e.sliceCount-1]];return[k(t.level,e.grid,e.grid.mapLeft,{context:!0,ramp:`sequential`,range:[0,1]}),k(n?.map??t.blank,e.grid,e.grid.mapLeft,{ramp:`sequential`,range:[-.5,.3]}),v(t.observations,G,K,i,b(r),4),v(t.curve,G,K,i,C(r,0),3),v(t.peaks,G,K,i,C(r,1),7)]}});export{q as default};