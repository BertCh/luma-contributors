import{a as e,i as t,r as n}from"./dot-density-sampling-a0mhTtRS.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{n as a,o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,T as l,m as u,v as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{s as p,v as m}from"./spatial-context-DwZEd4EZ.js";import{n as h,t as g}from"./philox-random-DfVxWNPn.js";import{l as _}from"./wgsl-common-DfyXiYB0.js";import{i as v}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{a as y,i as b}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as x}from"./define-run-B-brkCe4.js";import{r as S}from"./pick-Bo3t3hjv.js";import{Fs as C}from"./entry-preview-D70TdSbK.js";import{t as w}from"./geometry-Dz9W1tyx.js";var T=`GPURandomPointsOnLine`,E=`0xffffffffu`,D=Object.freeze({...p,measure:`distance`}),O=g.randomPointsOnLine.id,k=class{id;props;spatialContext;lineCount;constructor(t){this.id=t.id??`random-points-on-line`,this.props=t;let{id:r}=this;this.spatialContext=m(r,t.spatialContext,D);let{output:i}=t;if(c(t.positions,[`float32x2`],`${r} positions`),c(t.lineOffsets,[`uint32`],`${r} lineOffsets`),c(t.counts,[`uint32`],`${r} counts`),c(t.parameters,[`uint32`],`${r} parameters`),t.lineOffsets.length<2)throw Error(`${r} lineOffsets needs at least one line`);if(this.lineCount=t.lineOffsets.length-1,t.counts.length!==this.lineCount)throw Error(`${r} counts must have one row per line`);if(t.parameters.length<8)throw Error(`${r} parameters must contain 8 uint32 rows`);c(i.positions,[`float32x2`],`${r} output.positions`),u(r,i.points);let a=i.positions.length;if(l(r,e(this.lineCount,a)>a||!!i.points.status.candidateOverflow,`output.points.status.candidateOverflow is required when the per-line limit can hide loss below final capacity`),a<1||i.points.ids.length!==a)throw Error(`${r} output.positions and output.points.ids must share a positive capacity`);if(i.fractions&&(c(i.fractions,[`float32`],`${r} output.fractions`),i.fractions.length!==a))throw Error(`${r} output.fractions must have capacity rows`);n(r,Object.fromEntries(j(t).map((e,t)=>[`output ${t}`,e]))),b(r,{positions:i.positions,fractions:i.fractions}),b(r,{count:i.points.status.count,overflow:i.points.status.overflow,candidateOverflow:i.points.status.candidateOverflow,requiredCount:i.points.status.requiredCount}),f(r,j(t),A(t))}getCommandNodes(t){let{id:n,props:c,lineCount:l}=this,{output:u}=c;y(t,n,Object.fromEntries([...A(c).map((e,t)=>[`input ${t}`,e]),...j(c).map((e,t)=>[`output ${t}`,e])]));for(let[e,r]of[[`line prefix scratch`,Math.max(c.positions.length,l,1)*4],[`point slot scratch`,u.positions.length*8]])d(t.device,n,e,r,`reduce line rows or output capacity`);let f=u.positions.length,p=e(l,f),m=r(t,`${n}-lengths`,`float32`,l),g=r(t,`${n}-arc-lengths`,`float32`,Math.max(c.positions.length,1)),b=r(t,`${n}-line-counts`,`uint32`,l),x=r(t,`${n}-line-offsets`,`uint32`,l),S=r(t,`${n}-clamped`,`uint32`,1),C=r(t,`${n}-total`,`uint32`,1),w=r(t,`${n}-point-slots`,`uint32`,f*2),D=`
${v(`readVertex`,`positions`)}
fn segmentLength(vertex: u32) -> f32 {
  let delta = readVertex(vertex + 1u) - readVertex(vertex);
  return sqrt(delta.x * delta.x + delta.y * delta.y);
}`;return[o(t,{id:`${n}-lengths`,operation:T,variant:`lengths`,bindings:[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`inputLineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`lengths`,view:m,type:`f32`,access:`read_write`},{name:`arcLengths`,view:g,type:`f32`,access:`read_write`}],invocationCount:l,declarations:D,body:`let first = inputLineOffsets[inputLineOffsetsOffset + index];
  let last = inputLineOffsets[inputLineOffsetsOffset + index + 1u];
  var total = 0.0;
  if (last > first) {
    arcLengths[arcLengthsOffset + first] = 0.0;
  }
  for (var vertex = first; vertex + 1u < last; vertex = vertex + 1u) {
    total = total + segmentLength(vertex);
    arcLengths[arcLengthsOffset + vertex + 1u] = total;
  }
  lengths[lengthsOffset + index] = total;`}),s(t,{id:`${n}-clear-clamped`,operation:T,view:S,type:`u32`,value:`0u`}),o(t,{id:`${n}-counts`,operation:T,variant:`counts`,bindings:[{name:`sourceCounts`,view:c.counts,type:`u32`,access:`read`},{name:`lengths`,view:m,type:`f32`,access:`read`},{name:`lineCounts`,view:b,type:`u32`,access:`read_write`},{name:`clampFlag`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,declarations:`const LINE_LIMIT: u32 = ${p}u;
${_}`,body:`let lineLength = lengths[lengthsOffset + index];
  var count = select(0u, sourceCounts[sourceCountsOffset + index], isFiniteFloat(lineLength) && lineLength > 0.0);
  if (count > LINE_LIMIT) {
    count = LINE_LIMIT;
    atomicStore(&clampFlag[clampFlagOffset], 1u);
  }
  lineCounts[lineCountsOffset + index] = count;`}),...new i({id:`${n}-line-scan`,input:b,output:x,mode:`exclusive`}).getCommandNodes(t),o(t,{id:`${n}-total`,operation:T,variant:`total`,bindings:[{name:`lineCounts`,view:b,type:`u32`,access:`read`},{name:`lineOffsets`,view:x,type:`u32`,access:`read`},{name:`totalOut`,view:C,type:`u32`,access:`read_write`}],invocationCount:1,body:`let last = ${l-1}u;
  totalOut[totalOutOffset] = lineOffsets[lineOffsetsOffset + last] + lineCounts[lineCountsOffset + last];`}),o(t,{id:`${n}-locate`,operation:T,variant:`locate`,bindings:[{name:`lineCounts`,view:b,type:`u32`,access:`read`},{name:`lineOffsets`,view:x,type:`u32`,access:`read`},{name:`total`,view:C,type:`u32`,access:`read`},{name:`pointSlots`,view:w,type:`u32`,access:`read_write`},{name:`lineIds`,view:u.points.ids,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const LINE_COUNT: u32 = ${l}u;`,body:`let count = min(total[totalOffset], ${f}u);
  if (index >= count) {
    pointSlots[pointSlotsOffset + 2u * index] = ${E};
    pointSlots[pointSlotsOffset + 2u * index + 1u] = 0u;
    lineIds[lineIdsOffset + index] = ${E};
    return;
  }
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] + lineCounts[lineCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  pointSlots[pointSlotsOffset + 2u * index] = low;
  pointSlots[pointSlotsOffset + 2u * index + 1u] = index - lineOffsets[lineOffsetsOffset + low];
  lineIds[lineIdsOffset + index] = low;`}),o(t,{id:`${n}-sample`,operation:T,variant:u.fractions?`sample-fractions`:`sample`,bindings:[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`inputLineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`lengths`,view:m,type:`f32`,access:`read`},{name:`arcLengths`,view:g,type:`f32`,access:`read`},{name:`pointSlots`,view:w,type:`u32`,access:`read`},{name:`parameters`,view:c.parameters,type:`u32`,access:`read`},{name:`pointsOut`,view:u.positions,type:`f32`,access:`read_write`},...u.fractions?[{name:`fractionsOut`,view:u.fractions,type:`f32`,access:`read_write`}]:[]],invocationCount:f,declarations:`${h}${D}`,body:`var nanBits = 0x7fc00000u;
  var point = vec2<f32>(bitcast<f32>(nanBits));
  var fraction = bitcast<f32>(nanBits);
  let line = pointSlots[pointSlotsOffset + 2u * index];
  if (line != ${E}) {
    let rank = pointSlots[pointSlotsOffset + 2u * index + 1u];
    let random = philox4x32(vec4<u32>(line, rank, 0u, 0u), vec2<u32>(parameters[parametersOffset], ${O}u));
    fraction = philoxUnitFloat(random.x);
    let arcTarget = fraction * lengths[lengthsOffset + line];
    let first = inputLineOffsets[inputLineOffsetsOffset + line];
    let last = inputLineOffsets[inputLineOffsetsOffset + line + 1u];
    // First segment whose cumulative end exceeds the target (arc lengths are non-decreasing); zero
    // length segments never qualify. No such segment means rounding put the target at or past the
    // summed length, so the point clamps to the final vertex.
    var low = first;
    var high = last - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (arcLengths[arcLengthsOffset + middle + 1u] > arcTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    if (low + 1u < last && arcLengths[arcLengthsOffset + low + 1u] > arcTarget) {
      let travelled = arcLengths[arcLengthsOffset + low];
      point = mix(readVertex(low), readVertex(low + 1u), (arcTarget - travelled) / segmentLength(low));
    } else {
      point = readVertex(last - 1u);
    }
  }
  pointsOut[pointsOutOffset + 2u * index] = point.x;
  pointsOut[pointsOutOffset + 2u * index + 1u] = point.y;
  ${u.fractions?`fractionsOut[fractionsOutOffset + index] = fraction;`:``}`}),a(t,{id:`${n}-publish`,operation:T,requiredCount:C,output:u.points,candidateOverflowSources:u.points.status.candidateOverflow?[S]:[]})]}};function A(e){return[e.positions,e.lineOffsets,e.counts,e.parameters]}function j(e){let{output:t}=e;return[t.positions,t.points.ids,t.points.status.count,t.points.status.overflow,t.points.status.candidateOverflow,t.points.status.requiredCount,t.fractions]}var M=2400,N=12,P=3,F=x({parameters:C,caption:`Points uniform by length along each line (the hovered line asks for three times as many); color is the fraction along the line.`,createInputs:({random:e,world:t})=>({...w(e,t,N,14),focus:{row:Math.floor(N/2)}}),build(e,n,r){let{focus:i}=n,a=n.offsets.length-1,o=e.allocate(`counts`,`uint32`,a),s=e.allocate(`parameters`,`uint32`,8),c=e.allocate(`samples`,`float32x2`,M),l=e.allocate(`fractions`,`float32`,M),u=e.allocate(`ids`,`uint32`,M),d=e.allocate(`count`,`uint32`,1),f=e.allocate(`overflow`,`uint32`,1);return e.graph.add(new k({positions:e.upload(`positions`,`float32x2`,n.vertices).view,lineOffsets:e.upload(`line-offsets`,`uint32`,n.offsets).view,counts:o.view,parameters:s.view,output:{positions:c.view,fractions:l.view,points:{ids:u.view,status:{count:d.view,overflow:f.view}}}})),{outputs:{samples:c,fractions:l,segments:e.upload(`segments`,`float32x4`,n.segments)},update:e=>{o.write(new Uint32Array(a).map((t,n)=>n===i.row?e.pointsPerLine*P:e.pointsPerLine)),s.write(t({seed:e.seed}))}}},interaction:{kind:`probe`,hint:`hover a line`,rest:({inputs:e,toUnit:t})=>{let n=e.offsets[e.focus.row],r=n+(e.offsets[e.focus.row+1]-n>>1);return t([e.vertices[r*2],e.vertices[r*2+1]])},resolve(e,{inputs:t,values:n}){let r=-1,i=1/0;for(let n=0;n<t.offsets.length-1;n++){let a=t.offsets[n],o=S(t.vertices,e.x,e.y,a,t.offsets[n+1]-a);o&&o.distance<i&&(i=o.distance,r=n)}let a=r!==t.focus.row;if(t.focus.row=r,r<0)return{rerun:a};let o=t.offsets[r];return{rerun:a,glyphs:[{shape:`polyline`,points:t.vertices.subarray(o*2,t.offsets[r+1]*2)}],readout:`line ${r} · counts ${n.pointsPerLine*P} · others ${n.pointsPerLine}`}}},draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},{layer:`dots`,positions:t.samples,values:t.fractions,range:[0,1],radius:4}]});export{F as default};