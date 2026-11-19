import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./gpu-uint32-gather-C7ioZBaz.js";import{l as i,o as a}from"./gpu-command-graph-BdzdmV6u.js";import{t as o}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,S as d,c as f,f as p,m,p as h,v as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{a as v,i as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as b,n as x,r as S}from"./cell-key-wgsl-Cdgu1snt.js";import{o as C,t as ee}from"./fixed-point-accumulator-Ce6W3HVY.js";var w=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),T=(e,t)=>({name:e,view:t,type:`u32`,access:`read_write`});function te(e,n){let{id:r,column:i,sortedRows:a,rowUnique:s,leftUnique:u,output:d}=n,f=a.length,p=n=>t(e,`${r}-${n}`,`uint32`,f),m=p(`groups`),h=p(`value-keys`),g=p(`indices`),_=p(`counts`),v=[c(e,{id:`${r}-keys`,operation:`GPUKeyJoin`,variant:`distinct-keys`,invocationCount:f,bindings:[w(`column`,i),w(`sortedRows`,a),w(`rowUnique`,s),T(`groups`,m),T(`valueKeys`,h),T(`indices`,g)],body:`let bits = column[columnOffset + sortedRows[sortedRowsOffset + index]];
  let finite = (bits & 0x7fffffffu) < 0x7f800000u;
  groups[groupsOffset + index] = select(0xffffffffu, rowUnique[rowUniqueOffset + index], finite);
  valueKeys[valueKeysOffset + index] = select(bits, 0u, (bits & 0x7fffffffu) == 0u);
  indices[indicesOffset + index] = index;`})],y=o(e,{id:`${r}-sort`,operation:`GPUKeyJoin`,indices:g,indicesAreIdentity:!0,keys:[{view:m},{view:h}]});return v.push(...y.nodes,l(e,{id:`${r}-clear`,operation:`GPUKeyJoin`,view:_,type:`u32`,value:`0u`})),v.push(c(e,{id:`${r}-heads`,operation:`GPUKeyJoin`,variant:`distinct-heads`,invocationCount:f,bindings:[w(`sortedGroups`,y.sortedPrimaryKeys),w(`sortedIndices`,y.sortedIndices),w(`valueKeys`,h),{name:`counts`,view:_,type:`atomic<u32>`,access:`read_write`}],body:`let group = sortedGroups[sortedGroupsOffset + index];
  if (group == 0xffffffffu) { return; }
  let key = valueKeys[valueKeysOffset + sortedIndices[sortedIndicesOffset + index]];
  var head = index == 0u;
  if (index > 0u) {
    head = group != sortedGroups[sortedGroupsOffset + index - 1u] || key != valueKeys[valueKeysOffset + sortedIndices[sortedIndicesOffset + index - 1u]];
  }
  if (head) { atomicAdd(&counts[countsOffset + group], 1u); }`}),c(e,{id:`${r}-finish`,operation:`GPUKeyJoin`,variant:`distinct-finish`,invocationCount:u.length,bindings:[w(`leftUnique`,u),w(`counts`,_),{name:`output`,view:d,type:`f32`,access:`read_write`}],body:`let group = leftUnique[leftUniqueOffset + index];
  var count = 0u;
  if (group != 0xffffffffu) { count = counts[countsOffset + group]; }
  output[outputOffset + index] = f32(count);`})),v}var E=`GPUKeyJoin`,D=2048,O=1056,k=8,A=2147483648,j=4294967295,M=3,ne=256,N=class{id;props;kind;sumScale;estimatedCommandNodeCount;constructor(t){t=e(t,[`id`,`leftKeys`,`rightKeys`,`leftMask`,`rightMask`,`kind`,`gather`,`aggregates`,`sumScale`,`output`]);let n=e(t.output,[`rightRows`,`matchCounts`,`matched`,`rightMatched`,`rows`]),r=n.rows?e(n.rows,[`ids`,`status`]):void 0;t=Object.freeze({...t,gather:t.gather?Object.freeze(t.gather.map(t=>e(t,[`column`,`output`]))):void 0,aggregates:t.aggregates?Object.freeze(t.aggregates.map(t=>e(t,[`column`,`operation`,`output`,`sums`]))):void 0,output:Object.freeze({...n,rows:r?Object.freeze({...r,status:e(r.status,[`count`,`overflow`,`candidateOverflow`,`requiredCount`])}):void 0})}),this.id=t.id??`key-join`,this.props=t,this.kind=t.kind??`left`,this.sumScale=t.sumScale??65536;let o=this.id;if(b(o,this.sumScale),this.kind!==`left`&&this.kind!==`inner`)throw Error(`${o} kind must be 'left' or 'inner'`);let{output:s}=t,c=t.gather??[],l=t.aggregates??[],d=t.leftKeys.format;if(d!==`uint32`&&d!==`uint32x2`)throw Error(`${o} leftKeys must be uint32 or uint32x2`);u(t.leftKeys,[d],`${o} leftKeys`),u(t.rightKeys,[d],`${o} rightKeys`);let g=t.leftKeys.length,v=t.rightKeys.length;p(o,`leftKeys length`,g,1,j),p(o,`rightKeys length`,v,1,A);for(let[e,n,r]of[[`leftMask`,t.leftMask,g],[`rightMask`,t.rightMask,v]])if(n&&(u(n,[`uint32`],`${o} ${e}`),n.length!==r))throw Error(`${o} ${e} length must equal the ${e.slice(0,-4)} row count`);for(let[e,t]of c.entries()){let n=`gather[${e}]`,r=t.column.format;if(u(t.column,[`float32`,`uint32`],`${o} ${n}.column`),u(t.output,[r],`${o} ${n}.output`),t.column.length!==v)throw Error(`${o} ${n}.column length must equal the right row count`);if(t.output.length!==g)throw Error(`${o} ${n}.output length must equal the left row count`)}for(let[e,t]of l.entries()){let n=`aggregates[${e}]`;if(![`count`,`sum`,`mean`,`minimum`,`maximum`,`count-distinct`].includes(t.operation))throw Error(`${o} ${n}.operation is not supported`);if(t.operation!==`count`&&!t.column)throw Error(`${o} ${n} needs a column for ${t.operation}`);if(t.column&&(u(t.column,[`float32`],`${o} ${n}.column`),t.column.length!==v))throw Error(`${o} ${n}.column length must equal the right row count`);if(u(t.output,[`float32`],`${o} ${n}.output`),t.output.length!==g)throw Error(`${o} ${n}.output length must equal the left row count`);if(t.sums){if(t.operation!==`sum`&&t.operation!==`mean`)throw Error(`${o} ${n}.sums needs the sum or mean operation`);if(u(t.sums,[`uint32x2`],`${o} ${n}.sums`),t.sums.length!==g)throw Error(`${o} ${n}.sums length must equal the left row count`)}}for(let e of[`rightRows`,`matchCounts`,`matched`]){let t=s[e];if(t&&(u(t,[`uint32`],`${o} output.${e}`),t.length!==g))throw Error(`${o} output.${e} length must equal the left row count`)}if(s.rightMatched&&(u(s.rightMatched,[`uint32`],`${o} output.rightMatched`),s.rightMatched.length!==v))throw Error(`${o} output.rightMatched length must equal the right row count`);if(this.kind===`inner`){if(!s.rows)throw Error(`${o} output.rows is required for the inner join`);if(m(o,s.rows),s.rows.ids.length<1)throw Error(`${o} output.rows.ids must hold at least one row`)}else if(s.rows)throw Error(`${o} output.rows is only supported for the inner join`);let x=[t.leftKeys,t.rightKeys,t.leftMask,t.rightMask,...c.map(e=>e.column),...l.map(e=>e.column)],S=[s.rightRows,s.matchCounts,s.matched,s.rightMatched,s.rows?.ids,s.rows?.status.count,s.rows?.status.overflow,s.rows?.status.candidateOverflow,s.rows?.status.requiredCount,...c.map(e=>e.output),...l.flatMap(e=>[e.output,e.sums])],C=[[s.rightRows,s.matchCounts,s.matched],[s.rows?.status.count,s.rows?.status.overflow,s.rows?.status.candidateOverflow,s.rows?.status.requiredCount],...l.map(e=>[e.output,e.sums])];for(let e=0;e<c.length;e+=M)C.push([s.rightRows,...c.slice(e,e+M).map(e=>e.output)]);s.rows&&C.push([s.matched,s.rows.ids]);{let e=Object.fromEntries(S.map((e,t)=>[`output[${t}]`,e])),t=C,n=Object.values(e).flatMap(e=>e?f(e):[]).map(e=>new a(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),r=n.reduce((e,t)=>e+t.length,0);y(o,{outputs:new i({id:`${o}-writable-ranges`,name:`outputs`,format:`uint32`,length:r,valueLength:r,stride:1,byteStride:4,rowByteLength:4,data:n})});for(let e of t)y(o,Object.fromEntries(e.map((e,t)=>[String(t),e])))}this.estimatedCommandNodeCount=D+Math.ceil(c.length/M)+l.reduce((e,t)=>e+(t.operation===`count-distinct`?O:k),0),h(o,this.estimatedCommandNodeCount),_(o,S,x)}getCommandNodes(e){let{id:i,props:a,kind:u}=this,f=a.output,p=a.gather??[],m=a.aggregates??[],h=[a.leftKeys,a.rightKeys,a.leftMask,a.rightMask,...p.flatMap(e=>[e.column,e.output]),...m.flatMap(e=>[e.column,e.output,e.sums]),f.rightRows,f.matchCounts,f.matched,f.rightMatched,f.rows?.ids,f.rows?.status.count,f.rows?.status.overflow,f.rows?.status.candidateOverflow,f.rows?.status.requiredCount];v(e,i,Object.fromEntries(h.map((e,t)=>[`view[${t}]`,e])));let _=a.rightKeys.length,y={leftRows:a.leftKeys.length,uniqueStart:_+1,sumWords:m.some(e=>e.operation===`sum`||e.operation===`mean`)?2*_:0};for(let[t,n]of Object.entries(y))n>0&&(d(i,`${t} scratch byteLength`,n*4,e.device.limits.maxBufferSize),g(e.device,i,`${t} scratch`,n*4,`reduce the left or right row count`));let b=a.leftKeys.format===`uint32x2`,w=a.leftKeys.length,T=a.rightKeys.length,D=(n,r)=>t(e,`${i}-${n}`,`uint32`,r),O=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),k=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),A=(e,t)=>({name:e,view:t,type:`atomic<u32>`,access:`read_write`}),j=(t,n,r,a,o=``)=>c(e,{id:`${i}-${t}`,operation:E,variant:t,bindings:n,invocationCount:r,declarations:o,body:a}),N=[],P=`
const NO_KEY: u32 = 0xffffffffu;
fn isNoKey(key: vec2u) -> bool {
  return key.y == NO_KEY && ${b?`key.x == NO_KEY`:`true`};
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`,F=(e,t)=>b?`vec2u(${e}[${e}Offset + 2u * (${t}) + 1u], ${e}[${e}Offset + 2u * (${t})])`:`vec2u(0u, ${e}[${e}Offset + (${t})])`,I=D(`key-low`,T),L=b?D(`key-high`,T):void 0,R=D(`row-ids`,T);N.push(j(`right-keys`,[O(`rightKeys`,a.rightKeys),...a.rightMask?[O(`rightMask`,a.rightMask)]:[],k(`keyLow`,I),...L?[k(`keyHigh`,L)]:[],k(`rowIds`,R)],T,`var key = ${F(`rightKeys`,`index`)};
  ${a.rightMask?`if (rightMask[rightMaskOffset + index] == 0u) {
    key = vec2u(NO_KEY, NO_KEY);
  }`:``}
  if (isNoKey(key)) {
    key = vec2u(NO_KEY, NO_KEY);
  }
  keyLow[keyLowOffset + index] = key.y;
  ${L?`keyHigh[keyHighOffset + index] = key.x;`:``}
  rowIds[rowIdsOffset + index] = index;`,P));let z=o(e,{id:`${i}-sort`,operation:E,indices:R,indicesAreIdentity:!0,keys:[...L?[{view:L}]:[],{view:I}]});N.push(...z.nodes);let B=z.sortedIndices,V=L?z.sortedPrimaryKeys:void 0,H=L?D(`sorted-low`,T):z.sortedPrimaryKeys;L&&N.push(...new r({id:`${i}-gather-low`,source:I,indices:B,output:H}).getCommandNodes(e));let re=()=>[O(`sortedLow`,H),...V?[O(`sortedHigh`,V)]:[]],U=e=>b?`vec2u(sortedHigh[sortedHighOffset + ${e}], sortedLow[sortedLowOffset + ${e}])`:`vec2u(0u, sortedLow[sortedLowOffset + ${e}])`,W=D(`heads`,T),G=D(`valid-count`,1);N.push(j(`heads`,[...re(),k(`heads`,W),k(`validCount`,G)],T,`let key = ${U(`index`)};
  let isValid = !isNoKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${U(`index - 1u`)});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isNoKey(${U(`index + 1u`)}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`,`const ROW_COUNT: u32 = ${T}u;\n${P}`));let K=D(`unique-indices`,T);N.push(...new n({id:`${i}-scan`,input:W,output:K,mode:`exclusive`}).getCommandNodes(e));let ie=D(`unique-low`,T),q=b?D(`unique-high`,T):void 0,J=D(`unique-start`,T+1);N.push(j(`unique-keys`,[...re(),O(`heads`,W),O(`uniqueIndices`,K),O(`validCount`,G),k(`uniqueLow`,ie),...q?[k(`uniqueHigh`,q)]:[],k(`uniqueStart`,J)],T,`if (heads[headsOffset + index] != 0u) {
    let unique = uniqueIndices[uniqueIndicesOffset + index];
    let key = ${U(`index`)};
    uniqueLow[uniqueLowOffset + unique] = key.y;
    ${q?`uniqueHigh[uniqueHighOffset + unique] = key.x;`:``}
    uniqueStart[uniqueStartOffset + unique] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let uniqueTotal = uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
    uniqueStart[uniqueStartOffset + uniqueTotal] = validCount[validCountOffset];
  }`,`const ROW_COUNT: u32 = ${T}u;`));let ae=D(`unique-total`,1),Y=D(`row-unique`,T);N.push(j(`row-unique`,[O(`heads`,W),O(`uniqueIndices`,K),O(`validCount`,G),k(`rowUnique`,Y),k(`uniqueTotal`,ae)],T,`let isValid = index < validCount[validCountOffset];
  rowUnique[rowUniqueOffset + index] = select(
    0xffffffffu,
    uniqueIndices[uniqueIndicesOffset + index] + heads[headsOffset + index] - 1u,
    isValid
  );
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    uniqueTotal[uniqueTotalOffset] =
      uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
  }`,`const ROW_COUNT: u32 = ${T}u;`));let X=D(`left-unique`,w),Z=f.rightMatched?D(`unique-matched`,T):void 0;Z&&N.push(l(e,{id:`${i}-clear-unique-matched`,operation:E,view:Z,type:`u32`,value:`0u`})),N.push(j(`probe`,[O(`leftKeys`,a.leftKeys),...a.leftMask?[O(`leftMask`,a.leftMask)]:[],O(`uniqueLow`,ie),...q?[O(`uniqueHigh`,q)]:[],O(`uniqueTotal`,ae),k(`leftUnique`,X),...Z?[A(`uniqueMatched`,Z)]:[]],w,`var key = ${F(`leftKeys`,`index`)};
  ${a.leftMask?`if (leftMask[leftMaskOffset + index] == 0u) {
    key = vec2u(NO_KEY, NO_KEY);
  }`:``}
  var found = 0xffffffffu;
  if (!isNoKey(key)) {
    let total = uniqueTotal[uniqueTotalOffset];
    var low = 0u;
    var high = total;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let candidate = ${b?`vec2u(uniqueHigh[uniqueHighOffset + middle], uniqueLow[uniqueLowOffset + middle])`:`vec2u(0u, uniqueLow[uniqueLowOffset + middle])`};
      if (isKeyLess(candidate, key)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    if (low < total) {
      let candidate = ${b?`vec2u(uniqueHigh[uniqueHighOffset + low], uniqueLow[uniqueLowOffset + low])`:`vec2u(0u, uniqueLow[uniqueLowOffset + low])`};
      if (all(candidate == key)) {
        found = low;
      }
    }
  }
  leftUnique[leftUniqueOffset + index] = found;
  ${Z?`if (found != 0xffffffffu) {
    atomicMax(&uniqueMatched[uniqueMatchedOffset + found], 1u);
  }`:``}`,P));let oe=!!f.matched||u===`inner`,se=!!f.rightRows||p.length>0,Q=f.matched??(oe?D(`matched`,w):void 0),$=f.rightRows??(se?D(`first-rows`,w):void 0);(Q||$||f.matchCounts)&&N.push(j(`left-rows`,[O(`leftUnique`,X),O(`uniqueStart`,J),...$?[O(`sortedRows`,B)]:[],...Q?[k(`matchedOut`,Q)]:[],...$?[k(`firstOut`,$)]:[],...f.matchCounts?[k(`countsOut`,f.matchCounts)]:[]],w,`let unique = leftUnique[leftUniqueOffset + index];
  let isMatched = unique != 0xffffffffu;
  ${Q?`matchedOut[matchedOutOffset + index] = select(0u, 1u, isMatched);`:``}
  var begin = 0u;
  var end = 0u;
  if (isMatched) {
    begin = uniqueStart[uniqueStartOffset + unique];
    end = uniqueStart[uniqueStartOffset + unique + 1u];
  }
  ${$?`firstOut[firstOutOffset + index] = select(0xffffffffu, sortedRows[sortedRowsOffset + begin], isMatched);`:``}
  ${f.matchCounts?`countsOut[countsOutOffset + index] = end - begin;`:``}`));for(let e=0;e<p.length;e+=M){let t=p.slice(e,e+M);N.push(j(`gather-${e}`,[k(`firstRows`,$),...t.flatMap((e,t)=>[O(`column${t}`,e.column),k(`out${t}`,e.output)])],w,`let row = firstRows[firstRowsOffset + index];
  let isMatched = row != 0xffffffffu;
  ${t.map((e,t)=>`out${t}[out${t}Offset + index] = select(${e.column.format===`float32`?`0x7fc00000u`:`0xffffffffu`}, column${t}[column${t}Offset + select(0u, row, isMatched)], isMatched);`).join(`
  `)}`))}for(let[n,r]of m.entries()){let a=`aggregate-${n}`,{operation:o,column:s}=r;if(o===`count-distinct`){N.push(...te(e,{id:`${i}-${a}-distinct`,column:s,sortedRows:B,rowUnique:Y,leftUnique:X,output:r.output}));continue}let l=o===`sum`||o===`mean`,u=o===`mean`||o===`count`&&!!s,d=o===`minimum`,f=o===`maximum`,p=l?D(`${a}-sum-words`,2*T):void 0,m=l?D(`${a}-sum-high-words`,T+1):void 0,h=l&&r.sums?t(e,`${i}-${a}-sums`,`uint32x2`,T):void 0,g=u?D(`${a}-counts`,T):void 0,_=d||f?D(`${a}-extremes`,T):void 0,v=`
const SUM_SCALE_INDEX: u32 = ${T}u;
const FIXED_POINT_HEADROOM_BITS: u32 = ${C(T)}u;
${ee}`;(p||h||g||_)&&(N.push(j(`${a}-clear`,[...p?[k(`sumWords`,p)]:[],...m?[k(`sumHighWords`,m)]:[],...h?[k(`sums`,h)]:[],...g?[k(`counts`,g)]:[],..._?[k(`extremes`,_)]:[]],T,`${p?`sumWords[sumWordsOffset + 2u * index] = 0u;
  sumWords[sumWordsOffset + 2u * index + 1u] = 0u;
  sumHighWords[sumHighWordsOffset + index] = 0u;
  if (index == 0u) {
    sumHighWords[sumHighWordsOffset + ${T}u] = 0u;
  }`:``}
  ${h?`sums[sumsOffset + 2u * index] = 0u;
  sums[sumsOffset + 2u * index + 1u] = 0u;`:``}
  ${g?`counts[countsOffset + index] = 0u;`:``}
  ${_?`extremes[extremesOffset + index] = ${d?`0xffffffffu`:`0u`};`:``}`)),p&&N.push(c(e,{id:`${i}-${a}-sum-scale`,operation:E,variant:`${a}-sum-scale`,bindings:[O(`sortedRows`,B),O(`rowUnique`,Y),O(`values`,s,`f32`),A(`sumHighWords`,m)],invocationCount:T,guardIndex:!1,workgroupSize:ne,declarations:`${v}
var<workgroup> workgroupExponentBits: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && rowUnique[rowUniqueOffset + index] != 0xffffffffu) {
    // 0 for zero and for non-finite values, which the accumulate pass skips.
    let exponentBits = getFixedPointExponentBits(
      values[valuesOffset + sortedRows[sortedRowsOffset + index]]
    );
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX], workgroupMaximum);
    }
  }`})),N.push(j(`${a}-accumulate`,[O(`sortedRows`,B),O(`rowUnique`,Y),O(`values`,s,`f32`),...p?[A(`sumWords`,p)]:[],...m?[A(`sumHighWords`,m)]:[],...h?[A(`sums`,h)]:[],...g?[A(`counts`,g)]:[],..._?[A(`extremes`,_)]:[]],T,`let unique = rowUnique[rowUniqueOffset + index];
  if (unique == 0xffffffffu) {
    return;
  }
  let value = values[valuesOffset + sortedRows[sortedRowsOffset + index]];
  if ((bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u) {
    return;
  }
  ${g?`atomicAdd(&counts[countsOffset + unique], 1u);`:``}
  ${p?`let unitExponent = getFixedPointUnitExponent(
    atomicLoad(&sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX]),
    FIXED_POINT_HEADROOM_BITS
  );
  addFixedPointToSumWords(unique, quantizeFixedPoint(value, unitExponent));`:``}
  ${h?`let scaled = cellScaleValue(value);
  let previous = atomicAdd(&sums[sumsOffset + 2u * unique], scaled.y);
  let carried = scaled.x + select(0u, 1u, previous + scaled.y < previous);
  if (carried != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * unique + 1u], carried);
  }`:``}
  ${d?`atomicMin(&extremes[extremesOffset + unique], getOrderedKey(value));`:``}
  ${f?`atomicMax(&extremes[extremesOffset + unique], getOrderedKey(value));`:``}`,`${S(this.sumScale)}\n${x}${p?`\n${v}\n
fn addFixedPointToSumWords(cell: u32, value: vec3<u32>) {
  let base = sumWordsOffset + cell * 2u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&sumWords[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&sumWords[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&sumHighWords[sumHighWordsOffset + cell], highAddend);
  }
}`:``}`)));let y=`let base = sumWordsOffset + 2u * unique;
  let floatTotal = getFixedPointFloat(
    vec3<u32>(sumWords[base], sumWords[base + 1u], sumHighWords[sumHighWordsOffset + unique]),
    getFixedPointUnitExponent(
      sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX],
      FIXED_POINT_HEADROOM_BITS
    )
  );`,b;b=o===`count`?g?`let value = f32(counts[countsOffset + unique]);`:`let value = f32(uniqueStart[uniqueStartOffset + unique + 1u] - uniqueStart[uniqueStartOffset + unique]);`:o===`sum`?`${y}
  let value = floatTotal;`:o===`mean`?`${y}
  let n = counts[countsOffset + unique];
  let value = select(getNaN(), floatTotal / f32(n), n != 0u);`:`let key = extremes[extremesOffset + unique];
    let value = select(decodeOrderedKey(key), getNaN(), key == ${d?`0xffffffffu`:`0u`});`;let M=o===`count`&&!g;N.push(j(`${a}-finish`,[O(`leftUnique`,X),...M?[O(`uniqueStart`,J)]:[],...p?[O(`sumWords`,p)]:[],...m?[O(`sumHighWords`,m)]:[],...h?[O(`sums`,h)]:[],...g?[O(`counts`,g)]:[],..._?[O(`extremes`,_)]:[],k(`valueOut`,r.output,`f32`),...r.sums?[k(`sumsOut`,r.sums)]:[]],w,`let unique = leftUnique[leftUniqueOffset + index];
  if (unique == 0xffffffffu) {
    valueOut[valueOutOffset + index] = ${o===`count`?`0.0`:`getNaN()`};
    ${r.sums?`sumsOut[sumsOutOffset + 2u * index] = 0u;
    sumsOut[sumsOutOffset + 2u * index + 1u] = 0u;`:``}
    return;
  }
  ${h?`let total = vec2u(sums[sumsOffset + 2u * unique + 1u], sums[sumsOffset + 2u * unique]);`:``}
  ${b}
  valueOut[valueOutOffset + index] = value;
  ${r.sums?`sumsOut[sumsOutOffset + 2u * index] = total.y;
  sumsOut[sumsOutOffset + 2u * index + 1u] = total.x;`:``}`,`${x}${p?`\n${v}`:``}`))}if(f.rightMatched&&Z&&N.push(j(`right-matched`,[O(`sortedRows`,B),O(`rowUnique`,Y),O(`uniqueMatched`,Z),k(`rightMatchedOut`,f.rightMatched)],T,`let unique = rowUnique[rowUniqueOffset + index];
  let isMatched = unique != 0xffffffffu && uniqueMatched[uniqueMatchedOffset + unique] != 0u;
  rightMatchedOut[rightMatchedOutOffset + sortedRows[sortedRowsOffset + index]] =
    select(0u, 1u, isMatched);`)),u===`inner`&&f.rows&&Q){let t=D(`positions`,w),r=D(`matched-total`,1),a=f.rows.ids.length;N.push(...new n({id:`${i}-scan-matched`,input:Q,output:t,mode:`exclusive`}).getCommandNodes(e)),N.push(j(`compact`,[k(`matched`,Q),O(`positions`,t),k(`idsOut`,f.rows.ids),k(`totalOut`,r)],Math.max(w,a),`let last = LEFT_COUNT - 1u;
  let total = positions[positionsOffset + last] + matched[matchedOffset + last];
  if (index == 0u) {
    totalOut[totalOutOffset] = total;
  }
  if (index < LEFT_COUNT && matched[matchedOffset + index] != 0u) {
    let position = positions[positionsOffset + index];
    if (position < CAPACITY) {
      idsOut[idsOutOffset + position] = index;
    }
  }
  if (index < CAPACITY && index >= total) {
    idsOut[idsOutOffset + index] = 0xffffffffu;
  }`,`const LEFT_COUNT: u32 = ${w}u;\nconst CAPACITY: u32 = ${a}u;`)),N.push(s(e,{id:`${i}-publish`,operation:E,requiredCount:r,output:f.rows}))}return N}};export{N as t};