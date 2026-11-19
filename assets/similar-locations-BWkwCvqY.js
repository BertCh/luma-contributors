import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{t as n}from"./statistics-kernel-node-lacKNERm.js";import{E as r,S as i,f as a,u as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{r as l,u}from"./wgsl-common-DfyXiYB0.js";import{t as d}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as p}from"./define-run-B-brkCe4.js";import{u as m}from"./inputs-CJKKIRKK.js";import{a as h,i as g,t as _}from"./tract-rates-BlUiSczm.js";import{wt as v}from"./entry-preview-D70TdSbK.js";function y(e){return 4+e}function b(e,t,n=new Float32Array(y(t))){let r=y(t);if(n.length<r)throw Error(`Similar locations parameter target must hold ${r} elements`);if(!Number.isSafeInteger(e.resultCount)||e.resultCount<0)throw Error(`Similar locations resultCount must be a non-negative integer`);if(e.weights&&e.weights.length!==t)throw Error(`Similar locations weights must have one entry per attribute`);n[0]=e.resultCount,n[1]=+(e.direction===`least`),n[2]=e.excludeReference===!1?0:1,n[3]=0;for(let r=0;r<t;r++){let t=e.weights?e.weights[r]:1;if(!Number.isFinite(t)||t<0)throw Error(`Similar locations weights must be finite and non-negative`);n[4+r]=t}return n}var x=`GPUSimilarLocations`,S=256,C=Math.floor(4294967295/S);function w(e){return o(e,4)}var T=4096,E=`
const LANES: u32 = ${S}u;
var<workgroup> floatScratch: array<f32, ${S}>;
var<workgroup> countScratch: array<u32, ${S}>;
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  floatScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      floatScratch[lane] += floatScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return floatScratch[0];
}
fn reduceCount(value: u32, lane: u32) -> u32 {
  workgroupBarrier();
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      countScratch[lane] += countScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return countScratch[0];
}`,D=`
const INVALID_KEY: u32 = 0xffffffffu;
fn getOrderedKey(value: f32) -> u32 {
  // Integer test: a float compare of -0 against 0 can be folded away by the shader compiler.
  let raw = bitcast<u32>(value);
  let bits = select(raw, 0u, (raw & 0x7fffffffu) == 0u);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}`,O=class{id;props;rowCount;resultCapacity;constructor(e){this.id=e.id??`similar-locations`,this.props=e;let{id:t}=this;if(a(t,`columnCount`,e.columnCount,1,C),r(e.values,[`float32`],`${t} values`),e.values.length%e.columnCount!==0)throw Error(`${t} values length must be a multiple of columnCount`);if(this.rowCount=e.values.length/e.columnCount,this.rowCount<1)throw Error(`${t} needs at least one row`);if(e.standardization&&![`zscore`,`rank`].includes(e.standardization))throw Error(`${t} standardization must be 'zscore' or 'rank'`);if(this.resultCapacity=e.resultCapacity??Math.min(this.rowCount,32),a(t,`resultCapacity`,this.resultCapacity,1,this.rowCount),e.mask&&(r(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);if(r(e.referenceMask,[`uint32`],`${t} referenceMask`),e.referenceMask.length!==this.rowCount)throw Error(`${t} referenceMask length must equal the row count`);r(e.parameters,[`float32`],`${t} parameters`);let n=y(e.columnCount);if(e.parameters.length<n)throw Error(`${t} parameters must hold ${n} float32 values`);if(r(e.output.ranks,[`uint32`],`${t} output.ranks`),r(e.output.distances,[`float32`],`${t} output.distances`),r(e.output.topIds,[`uint32`],`${t} output.topIds`),r(e.output.status.count,[`uint32`],`${t} output.status.count`),e.output.ranks.length<this.rowCount||e.output.distances.length<this.rowCount)throw Error(`${t} output.ranks and output.distances must hold one row per input row`);if(e.output.topIds.length<this.resultCapacity)throw Error(`${t} output.topIds must hold resultCapacity rows`);if(e.output.status.count.length<1)throw Error(`${t} output.status.count must hold one row`);f(t,{...e.output,status:void 0,...e.output.status}),c(t,[e.output.ranks,e.output.distances,e.output.topIds,e.output.status.count],[e.values,e.mask,e.referenceMask,e.parameters])}getCommandNodes(r){let{id:a,props:o,rowCount:c,resultCapacity:f}=this;d(r,a,o,{"standardized matrix":o.values.length*4,"column moments":o.columnCount*8,"sort keys and IDs":c*4});let{output:p,columnCount:m}=o;s(a,r,[o.values,o.mask,o.referenceMask,o.parameters,p.ranks,p.distances,p.topIds,p.status.count]),i(a,`values rows`,o.values.length,w(r.device));let h=o.standardization===`rank`,g=c*m,_=e(r,`${a}-validity`,`uint32`,c),v=e(r,`${a}-standardized`,`float32`,g),y=e(r,`${a}-reference`,`float32`,m),b=e(r,`${a}-keys`,`uint32`,c),C=e(r,`${a}-row-ids`,`uint32`,c),O=e(r,`${a}-sorted-keys`,`uint32`,c),k=e(r,`${a}-sorted-ids`,`uint32`,c),A=`const ROW_COUNT: u32 = ${c}u;
const ATTRIBUTE_COUNT: u32 = ${m}u;
const MAX_RESULTS: u32 = ${f}u;
const NO_RANK: u32 = 0xffffffffu;
${u}
${l}`,j=[],M=(e,t,i,o,s={})=>j.push(n(r,{id:`${a}-${e}`,operation:x,variant:e,bindings:t,invocationCount:s.reduction?i*S:i,workgroupSize:s.reduction?S:64,guardIndex:!s.reduction&&void 0,declarations:`${A}\n${s.reduction?E:``}\n${s.declarations??``}`,body:o})),N=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),P=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`});if(M(`validate`,[N(`attributes`,o.values,`f32`),...o.mask?[N(`rowMask`,o.mask,`u32`)]:[],P(`validity`,_,`u32`)],c,`var isValid = true;
  ${o.mask?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    isValid = isValid && isFiniteValue(attributes[attributesOffset + index * ATTRIBUTE_COUNT + channel]);
  }
  validity[validityOffset + index] = select(0u, 1u, isValid);`),h&&c<T)M(`standardize-rank`,[N(`attributes`,o.values,`f32`),N(`validity`,_,`u32`),P(`standardized`,v,`f32`)],g,`let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  let value = attributes[attributesOffset + index];
  var less = 0u;
  var equal = 0u;
  var valid = 0u;
  for (var other = 0u; other < ROW_COUNT; other++) {
    if (validity[validityOffset + other] == 0u) {
      continue;
    }
    valid += 1u;
    let otherValue = attributes[attributesOffset + other * ATTRIBUTE_COUNT + channel];
    less += select(0u, 1u, otherValue < value);
    equal += select(0u, 1u, otherValue == value);
  }
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + index] = (f32(less) + 0.5 * (f32(equal) - 1.0)) / denominator;`);else if(h){let i=e(r,`${a}-rank-keys`,`uint32`,c),s=e(r,`${a}-rank-row-ids`,`uint32`,c),l=e(r,`${a}-rank-sorted-keys`,`uint32`,c),u=e(r,`${a}-rank-sorted-ids`,`uint32`,c),d=e(r,`${a}-rank-column`,`uint32`,1);M(`rank-column-clear`,[P(`rankColumn`,d,`u32`)],1,`rankColumn[rankColumnOffset] = 0xffffffffu;`);for(let e=0;e<m;e++)j.push(n(r,{id:`${a}-rank-column-${e}`,operation:x,variant:`rank-column`,bindings:[P(`rankColumn`,d,`u32`)],invocationCount:1,body:`rankColumn[rankColumnOffset] = rankColumn[rankColumnOffset] + 1u;`})),M(`rank-keys-${e}`,[N(`attributes`,o.values,`f32`),N(`validity`,_,`u32`),P(`rankKeys`,i,`u32`),P(`rankRowIds`,s,`u32`),N(`rankColumn`,d,`u32`)],c,`let value = attributes[attributesOffset + index * ATTRIBUTE_COUNT + rankColumn[rankColumnOffset]];
  rankKeys[rankKeysOffset + index] =
    select(INVALID_KEY, getOrderedKey(value), validity[validityOffset + index] != 0u);
  rankRowIds[rankRowIdsOffset + index] = index;`,{declarations:D}),j.push(...new t({id:`${a}-rank-sort-${e}`,keys:i,values:s,outputKeys:l,outputValues:u,keyBits:32}).getCommandNodes(r)),M(`rank-assign-${e}`,[N(`sortedKeys`,l,`u32`),N(`sortedIds`,u,`u32`),P(`standardized`,v,`f32`),N(`rankColumn`,d,`u32`)],c,`let key = sortedKeys[sortedKeysOffset + index];
  let row = sortedIds[sortedIdsOffset + index];
  let column = rankColumn[rankColumnOffset];
  if (key == INVALID_KEY) {
    standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] = getNaN();
    return;
  }
  let below = lowerBound(key);
  let notAbove = lowerBound(key + 1u);
  let valid = lowerBound(INVALID_KEY);
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] =
    (f32(below) + 0.5 * (f32(notAbove - below) - 1.0)) / denominator;`,{declarations:`${D}
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`})}else{let t=e(r,`${a}-stats`,`float32`,2*m);M(`column-statistics`,[N(`attributes`,o.values,`f32`),N(`validity`,_,`u32`),P(`stats`,t,`f32`)],m,`let column = index / LANES;
  let lane = localInvocationIndex;
  var validPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      validPartial += 1u;
      sumPartial += attributes[attributesOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let valid = reduceCount(validPartial, lane);
  let mean = select(0.0, reduceSum(sumPartial, lane) / f32(valid), valid > 0u);
  var squaresPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      let delta = attributes[attributesOffset + row * ATTRIBUTE_COUNT + column] - mean;
      squaresPartial += delta * delta;
    }
  }
  let squares = reduceSum(squaresPartial, lane);
  if (lane == 0u) {
    let deviation = select(0.0, sqrt(squares / f32(valid)), valid > 0u);
    stats[statsOffset + 2u * column] = mean;
    stats[statsOffset + 2u * column + 1u] = select(0.0, 1.0 / deviation, deviation > 0.0);
  }`,{reduction:!0}),M(`standardize-zscore`,[N(`attributes`,o.values,`f32`),N(`validity`,_,`u32`),N(`stats`,t,`f32`),P(`standardized`,v,`f32`)],g,`let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  standardized[standardizedOffset + index] =
    (attributes[attributesOffset + index] - stats[statsOffset + 2u * channel]) *
    stats[statsOffset + 2u * channel + 1u];`)}return M(`reference`,[N(`standardized`,v,`f32`),N(`validity`,_,`u32`),N(`selection`,o.referenceMask,`u32`),P(`reference`,y,`f32`)],m,`let column = index / LANES;
  let lane = localInvocationIndex;
  var countPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u && selection[selectionOffset + row] != 0u) {
      countPartial += 1u;
      sumPartial += standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let count = reduceCount(countPartial, lane);
  let sum = reduceSum(sumPartial, lane);
  if (lane == 0u) {
    reference[referenceOffset + column] = select(getNaN(), sum / f32(count), count > 0u);
  }`,{reduction:!0}),M(`distance`,[N(`standardized`,v,`f32`),N(`validity`,_,`u32`),N(`selection`,o.referenceMask,`u32`),N(`reference`,y,`f32`),N(`params`,o.parameters,`f32`),P(`keys`,b,`u32`),P(`rowIds`,C,`u32`),P(`distances`,p.distances,`f32`)],c,`rowIds[rowIdsOffset + index] = index;
  let isLeast = params[paramsOffset + 1u] > 0.5;
  let excludeReference = params[paramsOffset + 2u] > 0.5;
  var eligible = validity[validityOffset + index] != 0u &&
    !(excludeReference && selection[selectionOffset + index] != 0u);
  var distanceSquared = 0.0;
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    let weight = params[paramsOffset + 4u + channel];
    let delta = standardized[standardizedOffset + index * ATTRIBUTE_COUNT + channel] -
      reference[referenceOffset + channel];
    distanceSquared += weight * delta * delta;
  }
  eligible = eligible && isFiniteValue(distanceSquared);
  let distance = sqrt(distanceSquared);
  // Non-negative f32 bit patterns order like the values; the sentinel sorts after every key.
  let bits = bitcast<u32>(distance);
  keys[keysOffset + index] = select(NO_RANK, select(bits, 0xfffffffeu - bits, isLeast), eligible);
  distances[distancesOffset + index] = select(getNaN(), distance, eligible);`),j.push(...new t({id:`${a}-sort`,keys:b,values:C,outputKeys:O,outputValues:k,keyBits:32}).getCommandNodes(r)),M(`rank`,[N(`sortedKeys`,O,`u32`),N(`sortedIds`,k,`u32`),N(`params`,o.parameters,`f32`),P(`ranks`,p.ranks,`u32`),P(`topIds`,p.topIds,`u32`),P(`resultCount`,p.status.count,`u32`)],c,`let key = sortedKeys[sortedKeysOffset + index];
  let isEligible = key != NO_RANK;
  let row = sortedIds[sortedIdsOffset + index];
  ranks[ranksOffset + row] = select(NO_RANK, index, isEligible);
  let requested = params[paramsOffset];
  let limit = min(u32(clamp(select(0.0, requested, isFiniteValue(requested)), 0.0, f32(MAX_RESULTS))), MAX_RESULTS);
  // Eligible rows sort first, so the boundary row owns the eligible count.
  let isLast = index + 1u == ROW_COUNT || sortedKeys[sortedKeysOffset + index + 1u] == NO_RANK;
  if (isEligible && isLast) {
    resultCount[resultCountOffset] = min(index + 1u, limit);
  }
  if (index == 0u && !isEligible) {
    resultCount[resultCountOffset] = 0u;
  }
  if (index < MAX_RESULTS) {
    topIds[topIdsOffset + index] = select(NO_RANK, row, isEligible && index < limit);
  }`),j}},k=4,A=40,j=12,M=[`zscore`,`rank`],N=p({parameters:v,caption:`Distance of every tract from the reference tract in attribute space.`,createInputs:({random:e,world:t})=>{let n=m(e,t),r=new Float32Array(n.count*k);for(let e=0;e<n.count;e++)r.set([n.povertyShare[e],n.unemploymentShare[e],n.age65[e]/n.population[e],n.population[e]],e*k);return{tracts:n,attributes:r}},build(e,{tracts:t,attributes:n},r){let i=e.upload(`attributes`,`float32`,n),a=e.allocate(`reference-mask`,`uint32`,t.count),o=e.allocate(`parameters`,`float32`,y(k)),s=e.allocate(`ranks`,`uint32`,t.count),c=e.allocate(`distances`,`float32`,t.count),l=e.allocate(`top-ids`,`uint32`,A),u=e.allocate(`count`,`uint32`,1);e.graph.add(new O({values:i.view,columnCount:k,referenceMask:a.view,parameters:o.view,standardization:M[r.standardization],resultCapacity:A,output:{ranks:s.view,distances:c.view,topIds:l.view,status:{count:u.view}}}));let d=new Uint32Array(t.count);return{outputs:{ranks:s,distances:c,topIds:l,referenceMask:a,...h(e,t)},update:e=>{let n=Math.min(Math.round(e.referenceTract),t.count-1);d.fill(0),d[n]=1,a.write(d),o.write(b({resultCount:j,weights:[e.firstWeight,e.secondWeight,1,1]},k))}}},rebuildOn:[`standardization`],interaction:{kind:`place`,hint:`pick the reference tract`,rest:({inputs:e,toUnit:t})=>t([e.tracts.centers[80],e.tracts.centers[81]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=g(t.tracts,e,r.referenceTract),o=i(n.topIds),s=i(n.distances),c=[_(t.tracts,a)],l=NaN;if(o)for(let e=0;e<j;e++){let n=o[e];n>=t.tracts.count||n===a||(c.push(_(t.tracts,n)),s&&(l=s[n]))}return{values:{referenceTract:a},glyphs:c,readout:`reference tract ${a} · ${j} most similar${Number.isFinite(l)?` within ${l.toFixed(1)}`:``} · weights ${r.firstWeight}, ${r.secondWeight}`}}},draw:({inputs:e,outputs:t})=>[{layer:`polygons`,positions:t.vertices,indices:t.meshIndices,owners:t.owners,values:t.ranks,ramp:`sequential`,range:[0,e.tracts.count],triangleCount:e.tracts.triangleCount},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount}]});export{N as default};