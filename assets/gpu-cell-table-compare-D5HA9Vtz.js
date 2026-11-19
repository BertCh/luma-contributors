import{t as e}from"./common-CshkX-d_.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,T as o,f as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as f}from"./scaled-quotient-wgsl-UDyhQag6.js";import{u as p}from"./wgsl-common-DfyXiYB0.js";import{i as m,t as h}from"./cell-dispatch-validation-PIkQyAK_.js";import{a as g,o as _,r as v,s as y}from"./cell-table-OQfP6eNV.js";import{i as b,n as x,r as S}from"./cell-key-wgsl-Cdgu1snt.js";import{t as C}from"./construction-snapshot-DdkX9Pc0.js";var w=`GPUCellTableCompare`,T=2147483647,E=256,D=256,O=class{id;props;measure;sumScale;zScoreKind;stageMeasures;constructor(e){let t=C(e,[`id`,`before`,`after`,`measure`,`sumScale`,`zScore`,`output`]),n=C(t.output,[`cells`,`presence`,`mask`,`before`,`after`,`delta`,`ratio`,`percentChange`,`zScore`,`status`]),r=g(t.before);e=Object.freeze({...t,before:r,after:t.after===t.before?r:g(t.after),output:Object.freeze({...n,status:_(n.status)})}),this.id=e.id??`cell-table-compare`,this.props=e;let i=this.id;if(this.measure=e.measure??`count`,this.measure!==`count`&&this.measure!==`sum`)throw Error(`${i} measure must be 'count' or 'sum'`);if(this.sumScale=e.sumScale??65536,b(i,this.sumScale),this.zScoreKind=e.zScore??(this.measure===`sum`?`standardized`:`poisson`),this.zScoreKind!==`poisson`&&this.zScoreKind!==`standardized`)throw Error(`${i} zScore must be 'poisson' or 'standardized'`);if(y(i,`before`,e.before),y(i,`after`,e.after),s(i,`before.cells.length`,e.before.cells.length,1,T),s(i,`after.cells.length`,e.after.cells.length,1,T),this.measure===`sum`&&(!e.before.sums||!e.after.sums))throw Error(`${i} measure 'sum' needs sums on both before and after`);let{output:c}=e;o(i,!!c.status.candidateOverflow,`output.status.candidateOverflow is required to report source loss`),a(c.cells,[`uint32x2`],`${i} output.cells`);let l=c.cells.length;s(i,`output.cells.length`,l,1,T),c.presence&&a(c.presence,[`uint32`],`${i} output.presence`),c.mask&&a(c.mask,[`uint32`],`${i} output.mask`);for(let e of[`before`,`after`,`delta`,`ratio`,`percentChange`,`zScore`]){let t=c[e];t&&a(t,[`float32`],`${i} output.${e}`)}for(let e of[`presence`,`mask`,`before`,`after`,`delta`,`ratio`,`percentChange`,`zScore`]){let t=c[e];if(t&&t.length!==l)throw Error(`${i} output.${e} needs output.cells.length rows`)}for(let e of[`count`,`overflow`,`candidateOverflow`,`requiredCount`]){let t=c.status[e];if(t&&(a(t,[`uint32`],`${i} output.status.${e}`),t.length<1))throw Error(`${i} output.status.${e} needs one row`)}let{status:d,...f}=c,p=new Set([c.before,c.after,c.delta].flatMap(e=>e?[h(e)]:[]));this.stageMeasures=[c.ratio,c.percentChange,c.zScore].some(e=>e&&p.has(h(e))),m(i,{...f,...d},[{cells:c.cells,presence:c.presence,mask:c.mask},this.stageMeasures?{}:{before:c.before,after:c.after,delta:c.delta},{ratio:c.ratio,percentChange:c.percentChange,zScore:c.zScore},{count:d.count,overflow:d.overflow,requiredCount:d.requiredCount},{candidateOverflow:d.candidateOverflow}]),u(i,k(c),[...v(e.before),...v(e.after)])}getCommandNodes(a){let{id:o,props:u,measure:m}=this,{before:h,after:g,output:_}=u,y={};for(let[e,t]of[[`before`,h],[`after`,g]])y[`${e}.cells`]=t.cells,y[`${e}.measure`]=m===`sum`?t.sums:t.counts,y[`${e}.status.count`]=t.status.count,y[`${e}.status.overflow`]=t.status.overflow,y[`${e}.status.candidateOverflow`]=t.status.candidateOverflow;for(let[e,t]of Object.entries(k(_)))y[`output ${e}`]=t;d(a,o,y);let b=[[`match bounds scratch`,(h.cells.length+3)*Uint32Array.BYTES_PER_ELEMENT],[`output columns scratch`,_.cells.length*Float32Array.BYTES_PER_ELEMENT]];for(let[e,t]of b)c(a.device,o,e,t,`reduce table capacities or use a device with larger bindings`),s(o,`${e} byteLength`,t,0,a.device.limits.maxBufferSize);if(_.zScore&&this.zScoreKind===`standardized`){let t=2*e(Math.ceil(_.cells.length/E),1,D)*Float32Array.BYTES_PER_ELEMENT;c(a.device,o,`statistics partials scratch`,t,`reduce output.cells.length or use a device with larger bindings`),s(o,`statistics partials scratch byteLength`,t,0,a.device.limits.maxBufferSize)}l(o,a,[...v(h),...v(g),...k(_)]);let C=_.cells.length,T=h.cells.length,O=g.cells.length,A=(e,n,r)=>t(a,`${o}-${e}`,n,r),j=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),M=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),N=[],P=A(`total`,`uint32`,1),F=A(`before-rows`,`uint32`,C),I=A(`after-rows`,`uint32`,C),L=!this.stageMeasures&&_.before||A(`before-values`,`float32`,C),R=!this.stageMeasures&&_.after||A(`after-values`,`float32`,C),z=!this.stageMeasures&&_.delta||A(`delta-values`,`float32`,C),B=A(`matched-flags`,`uint32`,T+1),V=A(`matched-prefix`,`uint32`,T+1),H=A(`after-bounds`,`uint32`,T+3),U=`const CAPACITY: u32 = ${C}u;
const CAPACITY_BEFORE: u32 = ${T}u;
const CAPACITY_AFTER: u32 = ${O}u;
const NO_ROW: u32 = 0xffffffffu;
fn getBeforeKey(row: u32) -> vec2u {
  return vec2u(beforeCells[beforeCellsOffset + 2u * row + 1u], beforeCells[beforeCellsOffset + 2u * row]);
}
fn getAfterKey(row: u32) -> vec2u {
  return vec2u(afterCells[afterCellsOffset + 2u * row + 1u], afterCells[afterCellsOffset + 2u * row]);
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`,W=(e,t)=>`fn ${e}(key: vec2u, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isKeyLess(get${t}Key(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;N.push(i(a,{id:`${o}-flags`,operation:w,variant:`matches`,bindings:[j(`beforeCells`,h.cells),j(`beforeCount`,h.status.count),j(`afterCells`,g.cells),j(`afterCount`,g.status.count),M(`flags`,B),M(`bounds`,H)],invocationCount:T+1,declarations:`${U}
${W(`lowerBoundAfter`,`After`)}`,body:`let countBefore = min(beforeCount[beforeCountOffset], CAPACITY_BEFORE);
  let countAfter = min(afterCount[afterCountOffset], CAPACITY_AFTER);
  if (index == 0u) {
    bounds[boundsOffset + CAPACITY_BEFORE + 1u] = countBefore;
    bounds[boundsOffset + CAPACITY_BEFORE + 2u] = countAfter;
  }
  var isMatch = false;
  if (index < countBefore) {
    let key = getBeforeKey(index);
    let bound = lowerBoundAfter(key, countAfter);
    bounds[boundsOffset + index] = bound;
    isMatch = bound < countAfter && all(getAfterKey(bound) == key);
  }
  flags[flagsOffset + index] = select(0u, 1u, isMatch);`}),...new r({id:`${o}-scan`,input:B,output:V,mode:`exclusive`}).getCommandNodes(a)),N.push(i(a,{id:`${o}-merge`,operation:w,variant:`merge-path`,bindings:[j(`beforeCells`,h.cells),j(`afterCells`,g.cells),j(`bounds`,H),j(`prefix`,V),M(`cellsOut`,_.cells),M(`beforeRows`,F),M(`afterRows`,I),M(`total`,P)],invocationCount:T+O,declarations:`${U}
${W(`lowerBoundBefore`,`Before`)}`,body:`let countBefore = bounds[boundsOffset + CAPACITY_BEFORE + 1u];
  let countAfter = bounds[boundsOffset + CAPACITY_BEFORE + 2u];
  if (index == 0u) {
    total[totalOffset] = countBefore + countAfter - prefix[prefixOffset + CAPACITY_BEFORE];
  }
  var key = vec2u(0u);
  var position = 0u;
  var beforeRow = NO_ROW;
  var afterRow = NO_ROW;
  if (index < CAPACITY_BEFORE) {
    if (index >= countBefore) {
      return;
    }
    key = getBeforeKey(index);
    let bound = bounds[boundsOffset + index];
    position = index + bound - prefix[prefixOffset + index];
    beforeRow = index;
    if (bound < countAfter && all(getAfterKey(bound) == key)) {
      afterRow = bound;
    }
  } else {
    let row = index - CAPACITY_BEFORE;
    if (row >= countAfter) {
      return;
    }
    key = getAfterKey(row);
    let bound = lowerBoundBefore(key, countBefore);
    if (bound < countBefore && all(getBeforeKey(bound) == key)) {
      return;
    }
    position = row + bound - prefix[prefixOffset + bound];
    afterRow = row;
  }
  if (position < CAPACITY) {
    cellsOut[cellsOutOffset + 2u * position] = key.y;
    cellsOut[cellsOutOffset + 2u * position + 1u] = key.x;
    beforeRows[beforeRowsOffset + position] = beforeRow;
    afterRows[afterRowsOffset + position] = afterRow;
  }`}));let G=m===`sum`,K=G?h.sums:h.counts,q=G?g.sums:g.counts,J=(e,t)=>G?`vec2u(${e}Measure[${e}MeasureOffset + 2u * ${t} + 1u], ${e}Measure[${e}MeasureOffset + 2u * ${t}])`:`vec2u(0u, ${e}Measure[${e}MeasureOffset + ${t}])`,Y=e=>G?`getScaledQuotient(cellI64ToF32(${e}), SUM_SCALE, 1.0, 0)`:`cellI64ToF32(${e})`;N.push(i(a,{id:`${o}-measures`,operation:w,variant:G?`sum-measures`:`count-measures`,bindings:[j(`beforeRows`,F),j(`afterRows`,I),j(`beforeMeasure`,K),j(`afterMeasure`,q),j(`total`,P),M(`beforeOut`,L,`f32`),M(`afterOut`,R,`f32`),M(`deltaOut`,z,`f32`)],invocationCount:C,declarations:`const CAPACITY: u32 = ${C}u;
const NO_ROW: u32 = 0xffffffffu;
${x}
${S(this.sumScale)}
${f}`,body:`if (index >= min(total[totalOffset], CAPACITY)) {
    beforeOut[beforeOutOffset + index] = 0.0;
    afterOut[afterOutOffset + index] = 0.0;
    deltaOut[deltaOutOffset + index] = getNaN();
    return;
  }
  let beforeRow = beforeRows[beforeRowsOffset + index];
  let afterRow = afterRows[afterRowsOffset + index];
  var beforeValue = vec2u(0u);
  var afterValue = vec2u(0u);
  if (beforeRow != NO_ROW) {
    beforeValue = ${J(`before`,`beforeRow`)};
  }
  if (afterRow != NO_ROW) {
    afterValue = ${J(`after`,`afterRow`)};
  }
  // Compare signed inputs before subtraction: the magnitude can use all 64 unsigned bits.
  let beforeOrderedHigh = beforeValue.x ^ 0x80000000u;
  let afterOrderedHigh = afterValue.x ^ 0x80000000u;
  let isNegative = afterOrderedHigh < beforeOrderedHigh ||
    (afterOrderedHigh == beforeOrderedHigh && afterValue.y < beforeValue.y);
  let larger = select(afterValue, beforeValue, isNegative);
  let smaller = select(beforeValue, afterValue, isNegative);
  let magnitude = cellAddI64(larger, cellNegateI64(smaller));
  let magnitudeFloat = cellU64ToF32(magnitude);
  let difference = select(magnitudeFloat, -magnitudeFloat, isNegative);
  beforeOut[beforeOutOffset + index] = ${Y(`beforeValue`)};
  afterOut[afterOutOffset + index] = ${Y(`afterValue`)};
  deltaOut[deltaOutOffset + index] = ${G?`getScaledQuotient(difference, SUM_SCALE, 1.0, 0)`:`difference`};`}));let X=_.zScore&&this.zScoreKind===`standardized`?A(`statistics`,`float32`,3):void 0;if(X){let t=A(`statistics-extent`,`float32`,2);N.push(...new n({id:`${o}-statistics-extent`,input:z,output:t,operation:`extent`}).getCommandNodes(a));let r=`let normalizationExponent = frexp(max(abs(extent[extentOffset]), abs(extent[extentOffset + 1u]))).exp;`,s=e(Math.ceil(C/E),1,D),c=A(`statistics-partials`,`float32`,2*s),l=e=>`partials[localInvocationIndex] = ${e};
  workgroupBarrier();
  for (var stride = ${E/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,u=e=>`partials[localInvocationIndex] = 0.0;
  if (localInvocationIndex < GROUPS) {
    partials[localInvocationIndex] = groupPartials[groupPartialsOffset + ${e} + localInvocationIndex];
  }
  workgroupBarrier();
  for (var stride = ${E/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,d=`const CAPACITY: u32 = ${C}u;
const GROUPS: u32 = ${s}u;
const STRIDE: u32 = ${s*E}u;
var<workgroup> partials: array<f32, ${E}>;`;N.push(i(a,{id:`${o}-statistics-sums`,operation:w,variant:`statistics-sums`,bindings:[j(`deltas`,z,`f32`),j(`total`,P),j(`extent`,t,`f32`),M(`groupPartials`,c,`f32`)],workgroupSize:E,invocationCount:s*E,guardIndex:!1,declarations:d,body:`let rowCount = min(total[totalOffset], CAPACITY);
  ${r}
  var partialSum = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    partialSum += ldexp(deltas[deltasOffset + row], -normalizationExponent);
  }
  ${l(`partialSum`)}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + index / ${E}u] = partials[0];
  }`}),i(a,{id:`${o}-statistics-squares`,operation:w,variant:`statistics-squares`,bindings:[j(`deltas`,z,`f32`),j(`total`,P),j(`extent`,t,`f32`),M(`groupPartials`,c,`f32`)],workgroupSize:E,invocationCount:s*E,guardIndex:!1,declarations:d,body:`let rowCount = min(total[totalOffset], CAPACITY);
  ${r}
  ${u(`0u`)}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  var partialSquares = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    let centered = ldexp(deltas[deltasOffset + row], -normalizationExponent) - mean;
    partialSquares += centered * centered;
  }
  ${l(`partialSquares`)}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + GROUPS + index / ${E}u] = partials[0];
  }`}),i(a,{id:`${o}-statistics`,operation:w,variant:`statistics`,bindings:[j(`groupPartials`,c,`f32`),j(`total`,P),j(`extent`,t,`f32`),M(`statistics`,X,`f32`)],workgroupSize:E,invocationCount:E,guardIndex:!1,declarations:d,body:`let rowCount = min(total[totalOffset], CAPACITY);
  ${r}
  ${u(`0u`)}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  ${u(`GROUPS`)}
  if (localInvocationIndex == 0u) {
    statistics[statisticsOffset] = mean;
    statistics[statisticsOffset + 1u] = select(0.0, sqrt(partials[0] / f32(rowCount)), rowCount > 0u);
    statistics[statisticsOffset + 2u] = f32(normalizationExponent);
  }`}))}if((_.ratio||_.percentChange||_.zScore)&&N.push(i(a,{id:`${o}-derived`,operation:w,variant:`derived`,bindings:[j(`beforeValues`,L,`f32`),j(`afterValues`,R,`f32`),j(`deltaValues`,z,`f32`),j(`total`,P),..._.ratio?[M(`ratioOut`,_.ratio,`f32`)]:[],..._.percentChange?[M(`percentOut`,_.percentChange,`f32`)]:[],..._.zScore?[M(`zScoreOut`,_.zScore,`f32`)]:[],...X?[j(`statistics`,X,`f32`)]:[]],invocationCount:C,declarations:`const CAPACITY: u32 = ${C}u;
${x}
${p}
${f}
fn getCompareQuotient(value: f32, divisor: f32) -> f32 {
  if (isFiniteValue(value) && isFiniteValue(divisor) && divisor != 0.0) {
    return getScaledQuotient(value, divisor, 1.0, 0);
  }
  return value / divisor;
}
fn getComparePercent(value: f32, divisor: f32) -> f32 {
  if (isFiniteValue(value) && isFiniteValue(divisor) && divisor != 0.0) {
    let parts = frexp(value);
    return getScaledQuotient(parts.fract * 100.0, divisor, 1.0, parts.exp);
  }
  return (100.0 * value) / divisor;
}`,body:`let isOccupied = index < min(total[totalOffset], CAPACITY);
  let nan = getNaN();
  let beforeValue = beforeValues[beforeValuesOffset + index];
  let afterValue = afterValues[afterValuesOffset + index];
  let delta = deltaValues[deltaValuesOffset + index];
  ${_.ratio?`ratioOut[ratioOutOffset + index] = select(nan, getCompareQuotient(afterValue, beforeValue), isOccupied && beforeValue != 0.0);`:``}
  ${_.percentChange?`percentOut[percentOutOffset + index] = select(nan, getComparePercent(delta, beforeValue), isOccupied && beforeValue != 0.0);`:``}
  ${_.zScore?this.zScoreKind===`poisson`?`// A count statistic: defined for before + after > 0, and 0 when both are 0. Sums can make
  // the denominator zero or negative with a non-zero delta; that has no value (never sqrt of a
  // negative number, which WGSL leaves indeterminate).
  let denominator = beforeValue + afterValue;
  let undefinedScore = select(nan, 0.0, beforeValue == 0.0 && afterValue == 0.0);
  let score = select(undefinedScore, getCompareQuotient(delta, sqrt(max(denominator, 0.0))), denominator > 0.0);
  zScoreOut[zScoreOutOffset + index] = select(nan, score, isOccupied);`:`let mean = statistics[statisticsOffset];
  let deviation = statistics[statisticsOffset + 1u];
  let normalizedDelta = ldexp(delta, -i32(statistics[statisticsOffset + 2u]));
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, getCompareQuotient(normalizedDelta - mean, deviation), deviation != 0.0), isOccupied);`:``}`})),this.stageMeasures)for(let[e,t,n]of[[`before`,L,_.before],[`after`,R,_.after],[`delta`,z,_.delta]])n&&N.push(i(a,{id:`${o}-publish-${e}`,operation:w,variant:`publish-measures`,bindings:[j(`sourceValues`,t,`f32`),M(`destinationValues`,n,`f32`)],invocationCount:C,body:`destinationValues[destinationValuesOffset + index] = sourceValues[sourceValuesOffset + index];`}));N.push(i(a,{id:`${o}-tail`,operation:w,variant:`tail`,bindings:[j(`beforeRows`,F),j(`afterRows`,I),j(`total`,P),M(`cellsOut`,_.cells),..._.presence?[M(`presenceOut`,_.presence)]:[],..._.mask?[M(`maskOut`,_.mask)]:[]],invocationCount:C,declarations:`const CAPACITY: u32 = ${C}u;
const NO_ROW: u32 = 0xffffffffu;`,body:`if (index >= min(total[totalOffset], CAPACITY)) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    ${_.presence?`presenceOut[presenceOutOffset + index] = 0u;`:``}
    ${_.mask?`maskOut[maskOutOffset + index] = 0u;`:``}
    return;
  }
  ${_.mask?`maskOut[maskOutOffset + index] = 1u;`:``}
  ${_.presence?`presenceOut[presenceOutOffset + index] =
    select(0u, 1u, beforeRows[beforeRowsOffset + index] != NO_ROW) |
    select(0u, 2u, afterRows[afterRowsOffset + index] != NO_ROW);`:``}`}));let Z=[h.status.overflow,h.status.candidateOverflow,g.status.overflow,g.status.candidateOverflow].filter(e=>!!e);return N.push(i(a,{id:`${o}-candidate-status`,operation:w,variant:`candidate-status`,invocationCount:1,bindings:[...Z.map((e,t)=>({name:`candidateIn${t}`,view:e,type:`u32`,access:`read`})),{name:`candidateOverflowOut`,view:_.status.candidateOverflow,type:`u32`,access:`read_write`}],body:`candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, ${Z.map((e,t)=>`candidateIn${t}[candidateIn${t}Offset] != 0u`).join(` || `)});`})),N.push(i(a,{id:`${o}-publish`,operation:w,variant:`publish`,bindings:[{name:`totalIn`,view:P,type:`u32`,access:`read`},{name:`countOut`,view:_.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:_.status.overflow,type:`u32`,access:`read_write`},..._.status.requiredCount?[{name:`totalOut`,view:_.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const CAPACITY: u32 = ${_.cells.length}u;`,body:`let required = totalIn[totalInOffset];
  countOut[countOutOffset] = min(required, CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, required > CAPACITY);
  ${_.status.requiredCount?`totalOut[totalOutOffset] = required;`:``}`})),N}};function k(e){return[e.cells,e.presence,e.mask,e.before,e.after,e.delta,e.ratio,e.percentChange,e.zScore,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount]}export{O as t};