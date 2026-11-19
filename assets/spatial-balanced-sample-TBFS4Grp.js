import{i as e,r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{v as i}from"./spatial-context-DwZEd4EZ.js";import{T as a,f as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as d}from"./spatial-output-validation-nw5lOmVL.js";import{t as f}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as p,t as m}from"./philox-random-DfVxWNPn.js";import{l as h}from"./wgsl-common-DfyXiYB0.js";import{n as g}from"./point-process-utils-BVnRkMJj.js";import{t as _}from"./define-run-B-brkCe4.js";import{f as v}from"./inputs-CJKKIRKK.js";import{c as y}from"./text-DYSgVV-n.js";import{Ds as b}from"./entry-preview-D70TdSbK.js";import{n as x}from"./gather-BaJ2HB8H.js";var S=4096,C=256,w=16777216,T=256,E=16,D=class{id;props;spatialContext;provenance;constructor(t){let n=t.id??`spatial-balanced-sample`;this.id=n,this.props=t,this.spatialContext=i(n,t.spatialContext);let r=t.positions.length;o(n,`positions rows`,r,1,S),o(n,`allocations`,t.allocations.length,1,C);let s=new Set,c=0;for(let e of t.allocations)a(n,Number.isInteger(e.stratumId)&&e.stratumId>=0&&e.stratumId<w&&!s.has(e.stratumId),`allocation stratumId must be a unique integer in [0, 16777215]`),o(n,`allocation sampleCount`,e.sampleCount,1,r),s.add(e.stratumId),c+=e.sampleCount;a(n,c<=r,`allocations must not sample more than rows`),t.strata||a(n,t.allocations.length===1&&t.allocations[0].stratumId===0,`without strata, allocations must be the single stratum 0`),a(n,t.bounds.every(e=>Number.isFinite(Math.fround(e))),`bounds must be finite in float32`),a(n,Math.fround(t.bounds[2])>Math.fround(t.bounds[0])&&Math.fround(t.bounds[3])>Math.fround(t.bounds[1]),`bounds must have a positive extent on both axes`),o(n,`seed`,t.seed,0,4294967295),o(n,`replicate`,t.replicate??0,0,4294967295),t.parameters&&g(t.parameters,`uint32`,2,n),g(t.positions,`float32x2`,r,n),g(t.frameIds,`uint32`,r,n),t.strata&&g(t.strata,`uint32`,r,n),t.inclusionTickets&&g(t.inclusionTickets,`uint32`,r,n),g(t.output.frameIds,`uint32`,c,n),g(t.output.inclusionProbabilities,`float32`,c,n),g(t.output.designWeights,`float32`,c,n),g(t.output.status,`uint32`,t.allocations.length*2,n),l(this.id,Object.values(t.output),[t.positions,t.frameIds,t.strata,t.inclusionTickets,t.parameters]);let u=Object.values(t.output);d(n,t.output,[[t.output.frameIds,t.output.inclusionProbabilities,t.output.designWeights]]);for(let[t,r]of u.entries())a(n,!u.slice(0,t).some(t=>e(t,r)),`output views must not overlap`);this.provenance=Object.freeze({generator:`Philox4x32-10`,purpose:m.spatialBalancedSampling.id,seed:t.seed,replicate:t.replicate??0,levels:16,design:`hierarchical-circular-systematic`,parameterSource:t.parameters?`resident-view`:`constructor`})}getCommandNodes(e){let{id:i,props:a}=this,o=a.positions.length;c(i,e,[a.positions,a.frameIds,a.strata,a.inclusionTickets,a.parameters,...Object.values(a.output)]),u(e,i,{positions:a.positions,frameIds:a.frameIds,strata:a.strata,inclusionTickets:a.inclusionTickets,parameters:a.parameters,...Object.fromEntries(Object.entries(a.output).map(([e,t])=>[`output.${e}`,t]))});for(let[t,n]of[[`row and selection scratch`,o*4],[`sort histogram`,o>T?Math.ceil(o/T)*E*4:0]])s(e.device,i,t,n,`reduce positions rows`);let l=t(e,`${i}-codes`,`uint32`,o),d=t(e,`${i}-ranks`,`uint32`,o),m=t(e,`${i}-selected-rows`,`uint32`,a.output.frameIds.length),g=t(e,`${i}-selected-probabilities`,`float32`,a.output.frameIds.length),_=a.inclusionTickets?[{name:`tickets`,view:a.inclusionTickets,type:`u32`,access:`read`}]:[],v=`fn ticket(row: u32) -> u32 { return ${a.inclusionTickets?`tickets[ticketsOffset + row]`:`1u`}; }`,y=t(e,`${i}-validity`,`uint32`,o),b=a.strata?[{name:`strata`,view:a.strata,type:`u32`,access:`read`}]:[],x=`fn stratum(row: u32) -> u32 { return ${a.strata?`strata[strataOffset + row]`:`0u`}; }`,S=a.parameters?[{name:`parameters`,view:a.parameters,type:`u32`,access:`read`}]:[],C=a.parameters?`parameters[parametersOffset + 1u]`:`${a.replicate??0}u`,w=`${p}
fn random(counter: vec4<u32>) -> u32 { return philox4x32(counter, vec2<u32>(${a.parameters?`parameters[parametersOffset]`:`${a.seed}u`}, ${this.provenance.purpose}u)).x; }`,D=a.bounds.slice(0,2).map(e=>`${Math.fround(e)}f`).join(`, `),O=a.bounds.slice(2).map(e=>`${Math.fround(e)}f`).join(`, `),k=[0,1].map(e=>{let t=Math.max(Math.abs(a.bounds[e]),Math.abs(a.bounds[e+2]));return`${2**-(t>=2**64?Math.min(126,Math.floor(Math.log2(t))):0)}f`}).join(`, `),A=[r(e,{id:`${i}-hierarchy`,operation:`GPUSpatialBalancedSample`,variant:`randomized-quadtree`,invocationCount:o,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},...S,...b,{name:`codes`,view:l,type:`u32`,access:`read_write`},{name:`validity`,view:y,type:`u32`,access:`read_write`}],declarations:`${h}\n${x}\n${w}`,body:`let point = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let minimum = vec2<f32>(${D});
  let maximum = vec2<f32>(${O});
  let valid = isFiniteFloat(point.x) && isFiniteFloat(point.y) && all(point >= minimum) && all(point <= maximum) && (${a.allocations.map(e=>`stratum(index) == ${e.stratumId}u`).join(` || `)});
  validity[validityOffset + index] = select(0u, 1u, valid);
  var code = 0u;
  if (valid) {
    let normalizationScale = vec2<f32>(${k});
    let coordinate = vec2<u32>(clamp((point * normalizationScale - minimum * normalizationScale) / (maximum * normalizationScale - minimum * normalizationScale), vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0);
    var prefix = 0u;
    for (var level = 0u; level < 16u; level++) {
      let shift = 15u - level;
      let child = ((coordinate.x >> shift) & 1u) | (((coordinate.y >> shift) & 1u) << 1u);
      let rotation = random(vec4<u32>(prefix, (stratum(index) << 4u) | level, ${C}, 0u)) & 3u;
      code = (code << 2u) | ((child + rotation) & 3u);
      prefix = (prefix << 2u) | child;
    }
  }
  codes[codesOffset + index] = code;`})],j=t(e,`${i}-identity-rows`,`uint32`,o);A.push(r(e,{id:`${i}-row-indices`,operation:`GPUSpatialBalancedSample`,variant:`row-indices`,invocationCount:o,bindings:[{name:`rowIndices`,view:j,type:`u32`,access:`read_write`}],body:`rowIndices[rowIndicesOffset + index] = index;`}));let M=f(e,{id:`${i}-order`,operation:`GPUSpatialBalancedSample`,indices:j,keys:[...a.strata?[{view:a.strata}]:[],{view:l}]}),N=t(e,`${i}-ticket-values`,`uint32`,o),P=t(e,`${i}-segment-flags`,`uint32`,o),F=t(e,`${i}-sorted-ranks`,`uint32`,o);A.push(...M.nodes,r(e,{id:`${i}-sorted-tickets`,operation:`GPUSpatialBalancedSample`,variant:`sorted-tickets`,invocationCount:o,bindings:[{name:`sortedIndices`,view:M.sortedIndices,type:`u32`,access:`read`},...b,..._,{name:`sortedTickets`,view:N,type:`u32`,access:`read_write`},{name:`segmentFlags`,view:P,type:`u32`,access:`read_write`}],declarations:`${x}\n${v}`,body:`let row = sortedIndices[sortedIndicesOffset + index];
  sortedTickets[sortedTicketsOffset + index] = ticket(row);
  var flag = 1u;
  if (index > 0u) { flag = select(0u, 1u, stratum(row) != stratum(sortedIndices[sortedIndicesOffset + index - 1u])); }
  segmentFlags[segmentFlagsOffset + index] = flag;`}),...new n({id:`${i}-rank-scan`,input:N,output:F,segmentFlags:P}).getCommandNodes(e),r(e,{id:`${i}-rank`,operation:`GPUSpatialBalancedSample`,variant:`spatial-ranks`,invocationCount:o,bindings:[{name:`sortedIndices`,view:M.sortedIndices,type:`u32`,access:`read`},{name:`sortedRanks`,view:F,type:`u32`,access:`read`},{name:`ranks`,view:d,type:`u32`,access:`read_write`}],body:`ranks[ranksOffset + sortedIndices[sortedIndicesOffset + index]] = sortedRanks[sortedRanksOffset + index];`}));let I=0;for(let[t,n]of a.allocations.entries())A.push(r(e,{id:`${i}-select-${t}`,operation:`GPUSpatialBalancedSample`,variant:`circular-systematic`,invocationCount:1,bindings:[{name:`ranks`,view:d,type:`u32`,access:`read`},{name:`validity`,view:y,type:`u32`,access:`read`},...b,..._,...S,{name:`selected`,view:m,type:`u32`,access:`read_write`},{name:`probabilities`,view:g,type:`f32`,access:`read_write`},{name:`status`,view:a.output.status,type:`u32`,access:`read_write`}],declarations:`${x}\n${v}\n${w}`,body:`var frameCount = 0u;
  var ticketTotal = 0u;
  var maximumTicket = 0u;
  var valid = true;
  for (var row = 0u; row < ${o}u; row++) {
    valid = valid && validity[validityOffset + row] == 1u;
    if (stratum(row) == ${n.stratumId}u) {
      frameCount++;
      let value = ticket(row);
      valid = valid && value > 0u && value <= 16777216u - min(ticketTotal, 16777216u);
      ticketTotal = min(16777217u, ticketTotal + min(value, 16777217u));
      maximumTicket = max(maximumTicket, value);
    }
  }
  valid = valid && frameCount >= ${n.sampleCount}u;
  ${a.inclusionTickets?`let step = ticketTotal / ${n.sampleCount}u;
  valid = valid && ticketTotal % ${n.sampleCount}u == 0u && maximumTicket <= step && step > 0u;
  let drawLimit = step;`:`let drawLimit = frameCount;`}
  var shift = 0u;
  var found = false;
  if (valid) {
    let threshold = (0u - drawLimit) % drawLimit;
    for (var attempt = 0u; attempt < 128u; attempt++) {
      let draw = random(vec4<u32>(${n.stratumId}u, 0x80000000u | attempt, ${C}, 0u));
      if (draw >= threshold) { shift = draw % drawLimit; found = true; break; }
    }
  }
  for (var sample = 0u; sample < ${n.sampleCount}u; sample++) {
    let slot = ${I}u + sample;
    selected[selectedOffset + slot] = 0xffffffffu;
    probabilities[probabilitiesOffset + slot] = bitcast<f32>(0x7fc00000u | (index & 0u));
    if (!valid || !found) { continue; }
    let targetRank = ${a.inclusionTickets?`shift + sample * step`:`(shift + sample * frameCount / ${n.sampleCount}u) % frameCount`};
    for (var row = 0u; row < ${o}u; row++) {
      let first = ranks[ranksOffset + row];
      if (stratum(row) == ${n.stratumId}u && first <= targetRank && targetRank < first + ticket(row)) {
        selected[selectedOffset + slot] = row;
        probabilities[probabilitiesOffset + slot] = ${a.inclusionTickets?`f32(ticket(row)) / f32(step)`:`${n.sampleCount}.0 / f32(frameCount)`};
      }
    }
  }
  status[statusOffset + ${2*t}u] = select(0u, ${n.sampleCount}u, valid && found);
  status[statusOffset + ${2*t+1}u] = select(1u, select(2u, 0u, found), valid);`})),I+=n.sampleCount;return A.push(r(e,{id:`${i}-publish`,operation:`GPUSpatialBalancedSample`,variant:`design-records`,invocationCount:a.output.frameIds.length,bindings:[{name:`selected`,view:m,type:`u32`,access:`read`},{name:`probabilities`,view:g,type:`f32`,access:`read`},{name:`frameIds`,view:a.frameIds,type:`u32`,access:`read`},{name:`ids`,view:a.output.frameIds,type:`u32`,access:`read_write`},{name:`inclusion`,view:a.output.inclusionProbabilities,type:`f32`,access:`read_write`},{name:`weights`,view:a.output.designWeights,type:`f32`,access:`read_write`}],body:`let row = selected[selectedOffset + index];
  var frameId = 0xffffffffu;
  if (row != 0xffffffffu) { frameId = frameIds[frameIdsOffset + row]; }
  let probability = probabilities[probabilitiesOffset + index];
  ids[idsOffset + index] = frameId;
  inclusion[inclusionOffset + index] = probability;
  weights[weightsOffset + index] = 1.0 / probability;`})),A}},O=_({parameters:b,caption:`Selected units (dark or white) are evenly spread inside each stratum; each stratum is sampled at its own rate.`,createInputs:({random:e,world:t})=>{let n=v(e,t,900,0);return{world:t,frame:n,sortedX:Float32Array.from({length:n.count},(e,t)=>n.positions[t*2]).sort(),split:{x:t.width/2}}},build(e,{world:t,frame:n,split:r},i){let a=[i.westCount,i.eastCount],o=a[0]+a[1],s=e.allocate(`strata`,`uint32`,n.count),c=e.upload(`positions`,`float32x2`,n.positions),l=e.allocate(`selected`,`uint32`,o),u=e.allocate(`status`,`uint32`,4);return e.graph.add(new D({positions:c.view,frameIds:e.upload(`frame-ids`,`uint32`,Uint32Array.from({length:n.count},(e,t)=>t)).view,strata:s.view,allocations:[{stratumId:0,sampleCount:a[0]},{stratumId:1,sampleCount:a[1]}],bounds:[0,0,t.width,t.height],seed:i.seed,output:{frameIds:l.view,inclusionProbabilities:e.allocate(`inclusion`,`float32`,o).view,designWeights:e.allocate(`design-weights`,`float32`,o).view,status:u.view}})),{outputs:{status:u,strata:s,positions:c,chosen:x(e,l,c,`chosen`)},update:()=>s.write(Uint32Array.from({length:n.count},(e,t)=>n.positions[t*2]<r.x?0:1))}},async afterRun({outputs:e,inputs:t}){let n=await e.chosen.buffer.readAsync(),r=new Float32Array(n.buffer,n.byteOffset,e.chosen.length*2),i=[0,0];for(let n=0;n<e.chosen.length;n++){let e=r[n*2];Number.isFinite(e)&&i[e<t.split.x?0:1]++}let a=await e.status.buffer.readAsync(),o=new Uint32Array(a.buffer,a.byteOffset,4);return{sampled:i,rejected:[o[1]!==0,o[3]!==0]}},rebuildOn:[`westCount`,`eastCount`,`seed`],interaction:{kind:`place`,hint:`move the stratum split`,rest:[.5,.5],resolve(e,{inputs:t,values:n}){let r=t.sortedX.length,i=t.sortedX[Math.min(r-1,n.westCount)],a=t.sortedX[Math.max(0,r-n.eastCount-1)],o=Math.min(1,Math.max(0,e.x/t.world.width));t.split.x=i+o*Math.max(0,a-i);let s=k(t.frame.positions,t.split.x);return{rerun:!0,glyphs:[{shape:`curtain`,x:t.split.x}],readout:`west ${s} units, ${n.westCount} asked · east ${t.frame.count-s} units, ${n.eastCount} asked · split ${t.split.x.toFixed(2)}`}}},draw:({inputs:e,outputs:t,extra:n,theme:r,values:i,world:a})=>{let o=k(e.frame.positions,e.split.x);return[{layer:`dots`,positions:t.positions,values:t.strata,radius:3.2,alpha:.85},{layer:`dots`,positions:t.chosen,radius:5.5,color:[...r.ink,1]},{layer:`text`,...y({labels:[{text:n.rejected[0]?`west: rejected, ${i.westCount} asked of ${o}`:`west: ${n.sampled[0]} sampled of ${o}`,at:[e.split.x/2,a.height],anchor:`top`,offset:[0,-4],role:`title`,swatch:0},{text:n.rejected[1]?`east: rejected, ${i.eastCount} asked of ${e.frame.count-o}`:`east: ${n.sampled[1]} sampled of ${e.frame.count-o}`,at:[(e.split.x+a.width)/2,a.height],anchor:`top`,offset:[0,-4],role:`title`,swatch:1}]})}]}});function k(e,t){let n=0;for(let r=0;r<e.length/2;r++)e[r*2]<t&&n++;return n}export{O as default};