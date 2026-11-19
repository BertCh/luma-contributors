import{d as e,r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-group-aggregation-CEYb6GJw.js";import{t as i}from"./gpu-gather-BJIYd708.js";import{l as a,o}from"./gpu-command-graph-BdzdmV6u.js";import{t as s}from"./gpu-visibility-workflow-CkjKbbbA.js";import{t as c}from"./construction-snapshot-DdkX9Pc0.js";import{n as l,o as u,r as d,t as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{S as p,T as m,b as h,c as g,f as _,m as v,v as y,y as b}from"./gpu-contributor-utils-BIwLQON5.js";import{n as x,r as S,t as C}from"./time-window-classify-node-DtxclKaU.js";import{t as w}from"./time-words-BFl8KCxq.js";import{a as T,i as E}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as D}from"./define-run-B-brkCe4.js";import{d as O,o as k,s as A}from"./pointer-overlay-Qh8qflcA.js";import{t as j}from"./lightning-C0gvbE9a.js";import{lc as M}from"./entry-preview-D70TdSbK.js";function N(e,t){let n=t.timestamps.length,r=t.timestamps.format===`uint32x2`,i=[{name:`params`,view:t.window,type:r?`u32`:`f32`,access:`read`}];n>0&&(i.push({name:`rowTimes`,view:t.timestamps,type:r?`u32`:`f32`,access:`read`}),t.timestampsLow&&i.push({name:`rowTimesLow`,view:t.timestampsLow,type:`f32`,access:`read`})),i.push({name:`rangeOut`,view:t.range,type:`u32`,access:`read_write`},{name:`totalOut`,view:t.total,type:`u32`,access:`read_write`}),t.firstInstance&&i.push({name:`firstInstanceOut`,view:t.firstInstance,type:`u32`,access:`read_write`});let a=r?`${w}
override ROW_COUNT: u32;

fn isRowBeforeStart(row: u32, windowStart: vec2<u32>, windowStartFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(time, 0.0, windowStart, windowStartFraction);
}

fn isRowAfterEnd(row: u32, windowEnd: vec2<u32>, windowEndFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(windowEnd, windowEndFraction, time, 0.0);
}`:`${C}
override ROW_COUNT: u32;

fn getRowLow(row: u32) -> f32 {
  return ${t.timestampsLow?`rowTimesLow[rowTimesLowOffset + row]`:`0.0`};
}

fn isRowBeforeStart(row: u32, windowStartHigh: f32, windowStartLow: f32) -> bool {
  return !isTimeAtLeast(rowTimes[rowTimesOffset + row], getRowLow(row), windowStartHigh, windowStartLow);
}

fn isRowAfterEnd(row: u32, windowEndHigh: f32, windowEndLow: f32) -> bool {
  return !isTimeAtLeast(windowEndHigh, windowEndLow, rowTimes[rowTimesOffset + row], getRowLow(row));
}`,o=r?`windowStart, windowStartFraction`:`windowStartHigh, windowStartLow`,s=r?`windowEnd, windowEndFraction`:`windowEndHigh, windowEndLow`;return u(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:r?`sorted-range-words`:`sorted-range`,bindings:i,invocationCount:1,workgroupSize:32,declarations:n===0?void 0:a,constants:n===0?void 0:{ROW_COUNT:n},body:n===0?`rangeOut[rangeOutOffset] = 0u;
  rangeOut[rangeOutOffset + 1u] = 0u;
  totalOut[totalOutOffset] = 0u;
  ${t.firstInstance?`firstInstanceOut[firstInstanceOutOffset] = 0u;`:``}`:`
  var isWindowValid = false;
  ${r?`
  let windowStart = vec2<u32>(params[paramsOffset + 0u], params[paramsOffset + 1u]);
  let windowEnd = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let windowStartFraction = bitcast<f32>(params[paramsOffset + 4u]);
  let windowEndFraction = bitcast<f32>(params[paramsOffset + 5u]);
  isWindowValid = isTimeWordsAtLeast(windowEnd, windowEndFraction, windowStart, windowStartFraction);`:`
  let windowStartHigh = params[paramsOffset + 0u];
  let windowStartLow = params[paramsOffset + 1u];
  let windowEndHigh = params[paramsOffset + 2u];
  let windowEndLow = params[paramsOffset + 3u];
  isWindowValid = isTimeAtLeast(windowEndHigh, windowEndLow, windowStartHigh, windowStartLow);`}
  var firstRow = 0u;
  var endRow = 0u;
  if (isWindowValid) {
    // Lower bound: the first row that is not before the window start.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowBeforeStart(middle, ${o})) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row after the window end, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowAfterEnd(middle, ${s})) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  let rowCount = endRow - firstRow;
  rangeOut[rangeOutOffset] = firstRow;
  rangeOut[rangeOutOffset + 1u] = rowCount;
  totalOut[totalOutOffset] = rowCount;
  ${t.firstInstance?`firstInstanceOut[firstInstanceOutOffset] = firstRow;`:``}`})}function P(e,t){let n=[{name:`range`,view:t.range,type:`u32`,access:`read_write`},{name:`rowsOut`,view:t.rows,type:`u32`,access:`read_write`}];return u(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:`sorted-ids`,bindings:n,invocationCount:t.rows.length,body:`let firstRow = range[rangeOffset];
  // The invocation count is the capacity.
  let count = min(range[rangeOffset + 1u], INVOCATION_COUNT);
  if (index >= count) {
    return;
  }
  rowsOut[rowsOutOffset + index] = firstRow + index;`})}function F(e,t){let n=t.timestamps.format===`uint32x2`,r=[{name:`rowTimes`,view:t.timestamps,type:n?`u32`:`f32`,access:`read`}];return t.timestampsLow&&r.push({name:`rowTimesLow`,view:t.timestampsLow,type:`f32`,access:`read`}),r.push({name:`violations`,view:t.violations,type:`atomic<u32>`,access:`read_write`}),u(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:n?`sorted-order-check-words`:`sorted-order-check`,bindings:r,invocationCount:Math.max(t.timestamps.length-1,1),declarations:n?w:C,body:n?`if (index + 1u >= ${t.timestamps.length}u) {
    return;
  }
  let previous = vec2<u32>(rowTimes[rowTimesOffset + 2u * index], rowTimes[rowTimesOffset + 2u * index + 1u]);
  let next = vec2<u32>(rowTimes[rowTimesOffset + 2u * index + 2u], rowTimes[rowTimesOffset + 2u * index + 3u]);
  if (!isTimeWordsAtLeast(next, 0.0, previous, 0.0)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`:`if (index + 1u >= ${t.timestamps.length}u) {
    return;
  }
  ${t.timestampsLow?`let previousLow = rowTimesLow[rowTimesLowOffset + index];
  let nextLow = rowTimesLow[rowTimesLowOffset + index + 1u];`:`let previousLow = 0.0;
  let nextLow = 0.0;`}
  if (!isTimeAtLeast(rowTimes[rowTimesOffset + index + 1u], nextLow, rowTimes[rowTimesOffset + index], previousLow)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`})}var I=4294967295,L=class{id;props;constructor(t){t=c(t,[`id`,`timestamps`,`timestampsLow`,`endTimestamps`,`endTimestampsLow`,`window`,`additionalPredicates`,`sourceIds`,`trackIds`,`output`,`sorted`,`validateOrder`]);let r=c(t.output,[`rows`,`ids`,`status`,`outputMask`,`fadeWeights`,`clipFractions`,`trackVisibleCounts`,`drawInstanceCount`,`rowRange`,`drawFirstInstance`,`orderViolations`]);t=Object.freeze({...t,additionalPredicates:t.additionalPredicates?Object.freeze(t.additionalPredicates.map(e=>{let t=c(e,[`kind`,`mask`]);return Object.freeze({...t,kind:Array.isArray(t.kind)?Object.freeze([...t.kind]):t.kind})})):void 0,output:Object.freeze({...r,status:c(r.status,[`count`,`overflow`,`candidateOverflow`,`requiredCount`])})}),this.id=t.id??`time-window-filter`,this.props=t;let i=this.id,s=t.timestamps.length;_(i,`timestamps length`,s,0,I);let{output:l}=t,u=t.timestamps.format===`uint32x2`;if(t.endTimestamps&&t.endTimestamps.format===`uint32x2`!==u)throw Error(`${i} endTimestamps must have the same format as timestamps`);if(u&&(t.timestampsLow||t.endTimestampsLow))throw Error(`${i} timestampsLow and endTimestampsLow require float32 timestamps`);for(let[n,r]of[[`timestamps`,t.timestamps],[`timestampsLow`,t.timestampsLow],[`endTimestamps`,t.endTimestamps],[`endTimestampsLow`,t.endTimestampsLow],[`output.fadeWeights`,l.fadeWeights]]){let a=u&&(n===`timestamps`||n===`endTimestamps`);for(let t of r?g(r):[])e(t,[a?`uint32x2`:`float32`],`${i} ${n}`);n!==`timestamps`&&h(i,n,t.timestamps,r)}for(let t of l.clipFractions?g(l.clipFractions):[])e(t,[`float32x2`],`${i} output.clipFractions`);h(i,`output.clipFractions`,t.timestamps,l.clipFractions);for(let e of l.outputMask?g(l.outputMask):[])n(e,`${i} output.outputMask`);if(h(i,`output.outputMask`,t.timestamps,l.outputMask),u){if(t.window.format!==`uint32`)throw Error(`${i} uint32x2 word timestamps require a packed uint32 word window`);if(e(t.window,[`uint32`],`${i} window`),t.window.length<8)throw Error(`${i} window must hold 8 uint32 values`)}else{if(t.window.format!==`float32`)throw Error(`${i} float32 timestamps require a float32 window`);if(e(t.window,[`float32`],`${i} window`),t.window.length<8)throw Error(`${i} window must hold 8 float32 values`)}for(let[e,r]of[[`sourceIds`,t.sourceIds],[`trackIds`,t.trackIds],...(t.additionalPredicates??[]).map((e,t)=>[`additionalPredicates[${t}]`,e.mask])]){if(r&&r.length!==s)throw Error(`${i} ${e} length must equal timestamps length`);for(let t of r?g(r):[])n(t,`${i} ${e}`)}if(t.endTimestampsLow&&!t.endTimestamps)throw Error(`${i} endTimestampsLow requires endTimestamps`);if(l.clipFractions&&!t.endTimestamps)throw Error(`${i} output.clipFractions requires endTimestamps`);if(!!t.trackIds!=!!l.trackVisibleCounts)throw Error(`${i} trackIds and output.trackVisibleCounts must be given together`);if(l.trackVisibleCounts&&(n(l.trackVisibleCounts,`${i} output.trackVisibleCounts`),l.trackVisibleCounts.length<1))throw Error(`${i} output.trackVisibleCounts must contain at least one track`);if(m(i,!!l.rows,`output.rows is required for accepted source-row indices`),v(i,{ids:l.rows,status:l.status}),m(i,!!t.sourceIds==!!l.ids,`sourceIds and output.ids must be supplied together; use output.rows for row indices`),l.ids&&(n(l.ids,`${i} output.ids`),m(i,l.ids.length===l.rows.length,`output.ids length must equal output.rows length`),b(i,[l.ids],[l.rows])),this.validateSorted(),l.drawInstanceCount&&(n(l.drawInstanceCount,`${i} output.drawInstanceCount`),l.drawInstanceCount.length<1))throw Error(`${i} output.drawInstanceCount must contain one uint32 row`);let d=s>0&&l.rows.length>=s;t.sorted||(d&&b(i,[l.rows],[l.outputMask]),l.trackVisibleCounts&&b(i,[l.trackVisibleCounts],[l.outputMask]));let f=!!t.additionalPredicates?.length;{let e={rows:l.rows,ids:l.ids,count:l.status.count,overflow:l.status.overflow,candidateOverflow:l.status.candidateOverflow,requiredCount:l.status.requiredCount,outputMask:l.outputMask,fadeWeights:l.fadeWeights,clipFractions:l.clipFractions,trackVisibleCounts:l.trackVisibleCounts,drawInstanceCount:l.drawInstanceCount,rowRange:l.rowRange,drawFirstInstance:l.drawFirstInstance,orderViolations:l.orderViolations},n=[[f?void 0:l.outputMask,l.fadeWeights,l.clipFractions],[l.status.count,l.status.overflow,l.status.candidateOverflow,l.status.requiredCount,l.drawInstanceCount,...!t.sorted&&!d?[l.rows]:[]],[l.rowRange,l.drawFirstInstance],...t.sorted?[[l.rows,l.rowRange]]:d?[[l.rows,l.outputMask]]:[],...l.trackVisibleCounts?[[l.trackVisibleCounts,l.outputMask]]:[]],r=Object.values(e).flatMap(e=>e?g(e):[]).map(e=>new o(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),s=r.reduce((e,t)=>e+t.length,0);E(i,{outputs:new a({id:`${i}-writable-ranges`,name:`outputs`,format:`uint32`,length:s,valueLength:s,stride:1,byteStride:4,rowByteLength:4,data:r})});for(let e of n)E(i,Object.fromEntries(e.map((e,t)=>[String(t),e])))}b(i,[t.output.ids,t.output.rows,t.output.status.count,t.output.status.overflow,t.output.status.candidateOverflow,t.output.status.requiredCount,l.outputMask,l.fadeWeights,l.clipFractions,l.trackVisibleCounts,l.drawInstanceCount,l.rowRange,l.drawFirstInstance,l.orderViolations],[t.timestamps,t.timestampsLow,t.endTimestamps,t.endTimestampsLow,t.window,t.sourceIds,t.trackIds,...(t.additionalPredicates??[]).map(e=>e.mask)])}validateSorted(){let{props:e,id:t}=this,{output:r}=e,i=[[`output.rowRange`,r.rowRange],[`output.drawFirstInstance`,r.drawFirstInstance],[`validateOrder`,e.validateOrder]];if(e.sorted){if(e.endTimestamps)throw Error(`${t} sorted requires instant timestamps`);if(e.additionalPredicates?.length||e.trackIds)throw Error(`${t} sorted does not support additionalPredicates or trackIds`);for(let[n,r]of[[`timestamps`,e.timestamps],[`timestampsLow`,e.timestampsLow],[`sourceIds`,e.sourceIds]])if(r instanceof a)throw Error(`${t} sorted requires a single packed ${n} view`);if(r.rowRange&&(n(r.rowRange,`${t} output.rowRange`),r.rowRange.length<2))throw Error(`${t} output.rowRange must contain two uint32 rows`);if(r.drawFirstInstance&&(n(r.drawFirstInstance,`${t} output.drawFirstInstance`),r.drawFirstInstance.length<1))throw Error(`${t} output.drawFirstInstance must contain one uint32 row`);if(!!e.validateOrder!=!!r.orderViolations)throw Error(`${t} validateOrder and output.orderViolations must be given together`);if(r.orderViolations&&(n(r.orderViolations,`${t} output.orderViolations`),r.orderViolations.length<1))throw Error(`${t} output.orderViolations must contain one uint32 row`)}else{for(let[e,n]of i)if(n)throw Error(`${t} ${e} requires sorted`);if(r.orderViolations)throw Error(`${t} output.orderViolations requires sorted and validateOrder`)}}getCommandNodes(e){let{props:n,id:i}=this,{output:a,trackIds:o}=n,{outputMask:c,trackVisibleCounts:u,drawInstanceCount:f}=a,m=n.additionalPredicates??[];T(e,i,Object.fromEntries([n.timestamps,n.timestampsLow,n.endTimestamps,n.endTimestampsLow,n.window,n.sourceIds,c,a.fadeWeights,a.clipFractions,o,u,f,a.rowRange,a.drawFirstInstance,a.orderViolations,a.ids,a.rows,a.status.count,a.status.overflow,a.status.candidateOverflow,a.status.requiredCount,...m.map(e=>e.mask)].map((e,t)=>[`view[${t}]`,e])));let h=n.timestamps.length,g=n.sorted?2:Math.max(h,2);if(y(e.device,i,`visibility scratch`,g*4,`reduce the total timestamp rows`),p(i,`visibility scratch byteLength`,g*4,e.device.limits.maxBufferSize),n.sorted)return this.getSortedCommandNodes(e);let _=m.length>0,v=[],b=!_&&c?c:d(e,`${i}-time-mask`,n.timestamps);v.push(...x(e,{id:`${i}-classify`,timestamps:n.timestamps,timestampsLow:n.timestampsLow,endTimestamps:n.endTimestamps,endTimestampsLow:n.endTimestampsLow,window:n.window,mask:b,fadeWeights:a.fadeWeights,clipFractions:a.clipFractions}));let S=b;_&&(S=c??(u?d(e,`${i}-mask`,n.timestamps):void 0));let C=h>0&&a.rows.length>=h,w=t(e,`${i}-total`,`uint32`,1),E=C?a.rows:t(e,`${i}-compact-ids`,`uint32`,h);return v.push(...new s({id:`${i}-visibility`,predicates:[{kind:`time-range`,mask:b},...m],output:E,count:w,outputMask:S}).getCommandNodes(e)),o&&u&&S&&v.push(...new r({id:`${i}-track-counts`,keys:o,mask:S,output:u}).getCommandNodes(e)),v.push(l(e,{id:`${i}-publish`,operation:`GPUTimeWindowFilter`,requiredCount:w,compactIds:C?void 0:E,output:{ids:a.rows,status:a.status},extraCounts:f?[f]:[]})),[...v,...this.getIdentityNodes(e)]}getSortedCommandNodes(e){let{props:n,id:r}=this,{output:i}=n,a=n.timestamps,o=n.timestampsLow,s=[];i.orderViolations&&(s.push(f(e,{id:`${r}-order-fill`,operation:`GPUTimeWindowFilter`,view:i.orderViolations,type:`u32`,value:`0u`,componentCount:1})),a.length>=2&&s.push(F(e,{id:`${r}-order-check`,timestamps:a,timestampsLow:o,violations:i.orderViolations})));let c=i.rowRange??t(e,`${r}-range`,`uint32`,2),u=t(e,`${r}-total`,`uint32`,1);return s.push(N(e,{id:`${r}-range-search`,timestamps:a,timestampsLow:o,window:n.window,range:c,total:u,firstInstance:i.drawFirstInstance})),(i.outputMask||i.fadeWeights)&&s.push(...x(e,{id:`${r}-classify`,timestamps:n.timestamps,timestampsLow:n.timestampsLow,window:n.window,mask:i.outputMask??d(e,`${r}-time-mask`,a),fadeWeights:i.fadeWeights})),i.rows.length>0&&a.length>0&&s.push(P(e,{id:`${r}-ids`,range:c,rows:i.rows})),s.push(l(e,{id:`${r}-publish`,operation:`GPUTimeWindowFilter`,requiredCount:u,output:{ids:i.rows,status:i.status},extraCounts:i.drawInstanceCount?[i.drawInstanceCount]:[]})),[...s,...this.getIdentityNodes(e)]}getIdentityNodes(e){let{output:t,sourceIds:n,timestamps:r}=this.props;return!t.ids||t.rows.length===0||r.length===0?[]:new i({id:`${this.id}-identities`,source:n,indices:t.rows,output:t.ids}).getCommandNodes(e)}},R=2500,z=18,B=[.07,.05,.95,.24];function V({world:e}){let t=j({width:e.width,height:.71},R);for(let e=0;e<R;e++)t.positions[e*2+1]+=.29;let n=new Float32Array(z);for(let e of t.hours)n[Math.min(17,Math.floor(e))]++;return{...t,world:e,hourly:n}}function H(e,t){return(B[0]+t/z*(B[2]-B[0]))*e.world.width}var U=D({parameters:M,caption:`Midwest flashes inside the window, in hours since May 21, 2024, 12:00 UTC; the strip is the hourly flash count with the window marked.`,createInputs:V,build(e,t){let n=e.upload(`timestamps`,`float32`,t.hours),r=e.upload(`positions`,`float32x2`,t.positions),i=e.allocate(`window`,`float32`,8),a=e.allocate(`fade-weights`,`float32`,R);e.graph.add(new L({timestamps:n.view,window:i.view,output:{rows:e.allocate(`rows`,`uint32`,R).view,status:{count:e.allocate(`count`,`uint32`,1).view,overflow:e.allocate(`overflow`,`uint32`,1).view},fadeWeights:a.view}}));let o=e.upload(`hourly`,`float32`,t.hourly),s=e.allocate(`hourly-inside`,`float32`,z);return{outputs:{positions:r,timestamps:n,fadeWeights:a,hourly:o,hourlyInside:s},update:e=>{let n=Math.max(e.end,e.start);i.write(S({start:e.start,end:n,startFadeDuration:e.startFadeDuration})),s.write(Float32Array.from(t.hourly,(t,r)=>r+1>e.start&&r<n?t:0))}}},rebuildOn:[],interaction:{kind:`window`,hint:`center and length of the window`,rest:[.55,.5],resolve(e,{inputs:t,values:n}){let r=A(M,`start`),i=A(M,`end`),a=k((e.unitX-.05)/.9,0,1)*z,o=2+12*k((.95-e.unitY)/.9,0,1),s=O(r,a-o/2),c=O(i,a+o/2),l=0;for(let e of t.hours)e>=s&&e<=c&&l++;return{values:{start:s,end:c},glyphs:[{shape:`box`,minimum:[H(t,s),B[1]*t.world.height],maximum:[H(t,c),B[3]*t.world.height]},{shape:`line`,from:[H(t,s),B[3]*t.world.height],to:[H(t,Math.min(z,s+n.startFadeDuration)),B[3]*t.world.height*.55]}],readout:`window ${s}-${c} h · fade ${n.startFadeDuration} h · ${l} flashes`}}},draw:({inputs:e,outputs:t})=>[{layer:`dots`,context:!0,positions:t.positions,color:[.4,.45,.55,.45],radius:2.4},{layer:`dots`,positions:t.positions,values:t.fadeWeights,ramp:`sequential`,range:[0,1],fade:!0,radius:3.6},{layer:`chart`,kind:`bars`,context:!0,values:t.hourly,range:[0,Math.max(...e.hourly)],rectangle:B},{layer:`chart`,kind:`bars`,color:[.2,.55,.9,1],values:t.hourlyInside,range:[0,Math.max(...e.hourly)],rectangle:B}]});export{U as default};