import{n as e,t}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{n as i}from"./sorted-segment-offsets-ChmOtA9F.js";import{a}from"./sorted-segment-sums-Dvmm6OdR.js";import{l as o,s}from"./wgsl-common-DfyXiYB0.js";import{t as c}from"./scaled-quotient-wgsl-UDyhQag6.js";import{E as l,S as u,f as d,v as f,x as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as h}from"./geographic-distribution-validation-BjMQQJgk.js";import{v as g}from"./spatial-context-DwZEd4EZ.js";import{t as _}from"./define-run-B-brkCe4.js";import{d as v}from"./inputs-CJKKIRKK.js";import{c as y}from"./text-DYSgVV-n.js";import{ks as b}from"./entry-preview-D70TdSbK.js";var x=256,S=360,C=4294967295,w=65535,T=4294967295,E=class{id;props;groupCount;spatialContext;constructor(e){this.id=e.id??`rose-statistic`,this.props=e,this.spatialContext=g(this.id,e.spatialContext),this.groupCount=e.groupCount??1;let{id:t,groupCount:n}=this;if(l(e.starts,[`float32x2`],`${t} starts`),l(e.ends,[`float32x2`],`${t} ends`),e.ends.length!==e.starts.length)throw Error(`${t} ends length must equal starts length`);for(let[n,r]of[[`weights`,e.weights],[`groupIds`,e.groupIds],[`mask`,e.mask]])if(r&&(n===`weights`?l(r,[`float32`],`${t} ${n}`):l(r,[`uint32`],`${t} ${n}`),r.length!==e.starts.length))throw Error(`${t} ${n} length must equal starts length`);if(d(t,`starts.length`,e.starts.length,0,C),d(t,`groupCount`,n),u(t,`groupCount * summary stride`,n*5,T),u(t,`groupCount * circular.binCount`,n*e.circular.binCount,T),!e.groupIds&&n!==1)throw Error(`${t} groupIds are required for multiple groups`);if(e.circular.mode!==`directional`&&e.circular.mode!==`axial`)throw Error(`${t} circular.mode is invalid`);if(![`radians`,`degrees`].includes(e.circular.angleUnit)||![`positive-x-counter-clockwise`,`positive-y-clockwise`].includes(e.circular.origin))throw Error(`${t} circular angle convention is unsupported`);d(t,`circular.binCount`,e.circular.binCount,2,S);let{summary:r,bins:i}=e.output;if(l(r,[`float32`],`${t} output.summary`),l(i,[`float32`],`${t} output.bins`),r.length<n*5)throw Error(`${t} output.summary is too short`);if(i.length<n*e.circular.binCount)throw Error(`${t} output.bins are too short`);let a={starts:e.starts,ends:e.ends,weights:e.weights,groupIds:e.groupIds,mask:e.mask},o=!!e.groupIds&&n>1;h(t,a,e.output,[{reads:{...a,groupIds:o?void 0:e.groupIds,mask:o?void 0:e.mask},writes:e.output}])}getCommandNodes(l){let{id:u,props:d,groupCount:h}=this;e(l,u,d.output),p(u,l,[d.starts,d.ends,d.weights,d.groupIds,d.mask,d.output.summary,d.output.bins]),m(l,u,{starts:d.starts,ends:d.ends,weights:d.weights,groupIds:d.groupIds,mask:d.mask,...d.output}),d.groupIds&&h>1&&(f(l.device,u,`group sort scratch`,d.starts.length*4,`reduce starts length`),f(l.device,u,`group offsets scratch`,(h+1)*4,`reduce groupCount`));let g=d.starts.length,_=!!d.groupIds&&h>1,v=[],y=[];if(_){let e=n(l,`${u}-group-sort-keys`,`uint32`,g),o=n(l,`${u}-group-sort-indices`,`uint32`,g),s=n(l,`${u}-group-sorted-keys`,`uint32`,g),c=n(l,`${u}-group-sorted-rows`,`uint32`,g),f=n(l,`${u}-group-offsets`,`uint32`,h+1);v.push(t(l,{id:`${u}-group-sort-prepare`,operation:`GPURoseStatistic`,variant:`group-sort-prepare`,invocationCount:g,bindings:[{name:`groupIds`,view:d.groupIds,type:`u32`,access:`read`},...d.mask?[{name:`mask`,view:d.mask,type:`u32`,access:`read`}]:[],{name:`sortKeys`,view:e,type:`u32`,access:`read_write`},{name:`sortIndices`,view:o,type:`u32`,access:`read_write`}],declarations:`const GROUP_COUNT: u32 = ${h}u;`,body:`let key = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortKeys[sortKeysOffset + index] = ${d.mask?`select(key, GROUP_COUNT, mask[maskOffset + index] == 0u)`:`key`};
  sortIndices[sortIndicesOffset + index] = index;`}),...new r({id:`${u}-group-sort`,keys:e,values:o,outputKeys:s,outputValues:c,keyBits:a(h)}).getCommandNodes(l),...i(l,{id:`${u}-groups`,operation:`GPURoseStatistic`,groupCount:h,sortedGroupKeys:s,segmentOffsets:f})),y=[{name:`sortedRows`,view:c,type:`u32`,access:`read`},{name:`groupOffsets`,view:f,type:`u32`,access:`read`}]}let b=[{name:`starts`,view:d.starts,type:`f32`,access:`read`},{name:`ends`,view:d.ends,type:`f32`,access:`read`},...d.weights?[{name:`weights`,view:d.weights,type:`f32`,access:`read`}]:[],..._?y:[...d.groupIds?[{name:`groupIds`,view:d.groupIds,type:`u32`,access:`read`}]:[],...d.mask?[{name:`mask`,view:d.mask,type:`u32`,access:`read`}]:[]],{name:`summary`,view:d.output.summary,type:`f32`,access:`read_write`},{name:`bins`,view:d.output.bins,type:`f32`,access:`read_write`}],S=d.circular.mode===`axial`,C=_?`false`:`${d.mask?`mask[maskOffset + row] == 0u || `:``}${d.groupIds?`groupIds[groupIdsOffset + row] != group || `:``}false`,T=_?`if (lane == 0u) {
    segmentBegin = 0u;
    segmentEnd = 0u;
    if (isActiveGroup) {
      segmentBegin = groupOffsets[groupOffsetsOffset + group];
      segmentEnd = groupOffsets[groupOffsetsOffset + group + 1u];
    }
  }
  let beginPosition = workgroupUniformLoad(&segmentBegin);
  let endPosition = workgroupUniformLoad(&segmentEnd);
  for (var tile = beginPosition; tile < endPosition;) {
    let position = tile + lane;
    let hasRow = lane < endPosition - tile;
    var row = 0u;
    if (hasRow) { row = sortedRows[sortedRowsOffset + position]; }`:`for (var tile = 0u; tile < ROWS;) {
    let row = tile + lane;
    let hasRow = isActiveGroup && lane < ROWS - tile;`;return v.push(t(l,{id:`${u}-reduce`,operation:`GPURoseStatistic`,variant:_?`${d.circular.mode}-segmented`:d.circular.mode,workgroupSize:x,invocationCount:Math.min(h,w)*x,guardIndex:!1,bindings:b,declarations:`const ROWS: u32 = ${g}u;\nconst GROUP_COUNT: u32 = ${h}u;\nconst LANES: u32 = ${x}u;\nconst BIN_COUNT: u32 = ${d.circular.binCount}u;\nconst PI: f32 = 3.141592653589793;\nconst PERIOD: f32 = ${S?`PI`:`2.0 * PI`};\n${o}\n${s}
${c}
var<workgroup> tileBin: array<u32, ${x}>;
var<workgroup> tileWeight: array<f32, ${x}>;
var<workgroup> sharedCount: array<u32, ${x}>;
var<workgroup> sharedWeight: array<f32, ${x}>;
var<workgroup> sharedCosine: array<f32, ${x}>;
var<workgroup> sharedSine: array<f32, ${x}>;
var<workgroup> sharedDirectionCosine: array<f32, ${x}>;
var<workgroup> sharedDirectionSine: array<f32, ${x}>;${_?`
var<workgroup> segmentBegin: u32;
var<workgroup> segmentEnd: u32;`:``}`,body:`let lane = localInvocationIndex;
for(var group=workgroupIndex;group<GROUP_COUNT;group+=INVOCATION_COUNT / LANES){
  let isActiveGroup = group < GROUP_COUNT;
  let binBase = binsOffset + group * BIN_COUNT;
  if (isActiveGroup) {
    for (var bin = lane; bin < BIN_COUNT; bin += LANES) { bins[binBase + bin] = 0.0; }
  }
  var count = 0u;
  var weightTotal = 0.0;
  var cosine = 0.0;
  var sine = 0.0;
  var directionCosine = 0.0;
  var directionSine = 0.0;
  ${T}
    var rowBin = 0xffffffffu;
    var rowWeight = 0.0;
    if (hasRow) {
      if (!(${C})) {
        let start = vec2f(starts[startsOffset + row * 2u], starts[startsOffset + row * 2u + 1u]);
        let end = vec2f(ends[endsOffset + row * 2u], ends[endsOffset + row * 2u + 1u]);
        let delta = end - start;
        let lengthSquared = dot(delta, delta);
        let weight = ${d.weights?`weights[weightsOffset + row]`:`1.0`};
        if (lengthSquared > 0.0 && weight > 0.0 && isFiniteFloat(lengthSquared) && isFiniteFloat(weight)) {
          // Transform the vector, rather than a rounded atan2 result. Opposite axial vectors
          // now reach atan2 with identical bits, including signed-zero cardinal components.
          var direction = ${d.circular.origin===`positive-y-clockwise`?`delta.yx`:`delta`};
          ${S?`if(direction.y<0.0||(direction.y==0.0&&direction.x<0.0)){direction=-direction;}`:``}
          direction=select(direction,vec2<f32>(0.0),direction==vec2<f32>(0.0));
          var angle = guardedAtan2(direction.y, direction.x);
          if (angle < 0.0) { angle += 2.0 * PI; }
          // Curated formula binding: phi is the doubled axial angle, with a 2*pi period.
          let circularAngle = ${S?`2.0 * angle`:`angle`};
          let unitDirection=direction/sqrt(lengthSquared);
          let cosineAngle = ${S?`(unitDirection.x-unitDirection.y)*(unitDirection.x+unitDirection.y)`:`unitDirection.x`};
          let sineAngle = ${S?`2.0*unitDirection.x*unitDirection.y`:`unitDirection.y`};
          cosine += weight * cosineAngle;
          sine += weight * sineAngle;
          directionCosine += cosineAngle;
          directionSine += sineAngle;
          weightTotal += weight;
          count++;
          var binPosition = (circularAngle / (2.0 * PI)) * f32(BIN_COUNT);
          let nearestBoundary=round(binPosition);
          // Snap only the f32 uncertainty interval around an integer bin boundary (four ULPs).
          if(abs(binPosition-nearestBoundary)<=4.76837158203125e-7*max(1.0,abs(binPosition))){binPosition=nearestBoundary;}
          rowBin = u32(floor(binPosition)) % BIN_COUNT;
          rowWeight = weight;
        }
      }
    }
    tileBin[lane] = rowBin;
    tileWeight[lane] = rowWeight;
    workgroupBarrier();
    if (isActiveGroup) {
      for (var bin = lane; bin < BIN_COUNT; bin += LANES) {
        var added = 0.0;
        for (var other = 0u; other < LANES; other++) {
          if (tileBin[other] == bin) { added += tileWeight[other]; }
        }
        bins[binBase + bin] += added;
      }
    }
    workgroupBarrier();
    if(${_?`endPosition`:`ROWS`}-tile<=LANES){break;}
    tile+=LANES;
  }
  sharedCount[lane] = count;
  sharedWeight[lane] = weightTotal;
  sharedCosine[lane] = cosine;
  sharedSine[lane] = sine;
  sharedDirectionCosine[lane] = directionCosine;
  sharedDirectionSine[lane] = directionSine;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedCount[lane] += sharedCount[lane + stride];
      sharedWeight[lane] += sharedWeight[lane + stride];
      sharedCosine[lane] += sharedCosine[lane + stride];
      sharedSine[lane] += sharedSine[lane + stride];
      sharedDirectionCosine[lane] += sharedDirectionCosine[lane + stride];
      sharedDirectionSine[lane] += sharedDirectionSine[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveGroup) {
    let base = summaryOffset + group * 5u;
    summary[base] = f32(sharedCount[0]);
    if (sharedWeight[0] > 0.0 && isFiniteFloat(sharedWeight[0])) {
      let totalWeight = sharedWeight[0];
      let meanCosine = getScaledQuotient(sharedCosine[0],totalWeight,1.0,0);
      let meanSine = getScaledQuotient(sharedSine[0],totalWeight,1.0,0);
      var meanAngle = guardedAtan2(meanSine, meanCosine);
      if (meanAngle < 0.0) { meanAngle += 2.0 * PI; }
      ${S?`meanAngle *= 0.5;`:``}
      let resultant = length(vec2<f32>(meanCosine,meanSine));
      let outputMean = ${d.circular.angleUnit===`degrees`?`meanAngle * (180.0 / PI)`:`meanAngle`};
      summary[base + 1u] = select(outputMean,0.0,outputMean>=${d.circular.angleUnit===`degrees`?S?`180.0`:`360.0`:`PERIOD`});
      summary[base + 2u] = resultant;
      summary[base + 3u] = 1.0 - resultant;
      let validCount = f32(sharedCount[0]);
      let rayleighZ = (sharedDirectionCosine[0] * sharedDirectionCosine[0] +
        sharedDirectionSine[0] * sharedDirectionSine[0]) / validCount;
      summary[base + 4u] = rayleighZ;
    } else {
      let nan = bitcast<f32>(0x7fc00000u | (group & 0u));
      for (var slot = 1u; slot < 5u; slot++) { summary[base + slot] = nan; }
    }
  }
  // Lane zero must finish consuming shared moments before the next group reuses them.
  workgroupBarrier();
}`})),v.flat()}},D=.4,O=.06,k=4;function A(e,t,n){return(Math.atan2(e[t*4+3]-e[t*4+1],e[t*4+2]-e[t*4])%n+n)%n}var j=_({parameters:b,caption:`Vessel headings in San Francisco Bay, as a rose of bars by direction; hover a bar for its legs.`,createInputs:({world:e,random:t})=>{let n={width:e.width*.5,height:e.height};return{...v(t,n,16,25),center:[e.width*.75,e.height/2]}},build(e,t,n){let r=t.segments,i=[],a=[],o=[];for(let e=0;e<t.count;e++){let[t,s,c,l]=r.subarray(e*4,e*4+4);Math.hypot(c-t,l-s)<=n.minimumLength||(i.push(t,s),a.push(c,l),o.push(t,s,c,l))}let s=e.allocate(`bins`,`float32`,n.binCount);return e.graph.add(new E({starts:e.upload(`starts`,`float32x2`,Float32Array.from(i)).view,ends:e.upload(`ends`,`float32x2`,Float32Array.from(a)).view,circular:{mode:n.mode===0?`directional`:`axial`,angleUnit:`radians`,origin:`positive-x-counter-clockwise`,binCount:n.binCount},output:{summary:e.allocate(`summary`,`float32`,5).view,bins:s.view}})),{outputs:{bins:s,segments:e.upload(`segments`,`float32x4`,r),highlight:e.upload(`highlight`,`float32x4`,new Float32Array(t.count*4).fill(NaN)),kept:Float32Array.from(o)}}},async afterRun({outputs:e,inputs:t,values:n,upload:r}){let i=await e.bins.buffer.readAsync(),a=new Float32Array(i.buffer,i.byteOffset,n.binCount),o=Math.max(...a,1),s=n.mode===1,c=s?2:1,l=s?Math.PI:Math.PI*2,u=[],d=[],f=[];for(let e=0;e<c;e++)for(let r=0;r<n.binCount;r++){let i=O+.34*(a[r]/o),s=u.length/2;u.push(t.center[0],t.center[1]);for(let a=0;a<=k;a++){let o=e*Math.PI+(r+a/k)/n.binCount*l;u.push(t.center[0]+Math.cos(o)*i,t.center[1]+Math.sin(o)*i)}for(let e=0;e<k;e++)d.push(s,s+1+e,s+2+e),f.push(r)}return{counts:a,vertices:r(`rose-vertices`,`float32x2`,Float32Array.from(u)),indices:r(`rose-indices`,`uint32`,Uint32Array.from(d)),owners:r(`rose-owners`,`uint32`,Uint32Array.from(f)),triangleCount:f.length}},interaction:{kind:`probe`,hint:`point at a direction`,rest:({inputs:e,toUnit:t})=>t([e.center[0]+D*.7,e.center[1]+D*.3]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=r.mode===1,o=a?Math.PI:Math.PI*2,s=(Math.atan2(e.y-t.center[1],e.x-t.center[0])%o+o)%o,c=Math.min(r.binCount-1,Math.floor(s/o*r.binCount)),l=new Float32Array(n.highlight.length*4).fill(NaN),u=0;for(let e=0;e<n.kept.length/4;e++){let t=A(n.kept,e,o);Math.min(r.binCount-1,Math.floor(t/o*r.binCount))===c&&(l.set(n.kept.subarray(e*4,e*4+4),u*4),u++)}n.highlight.write(l);let d=i(n.bins),f=d?O+.34*(d[c]/Math.max(...d,1)):D,p=[t.center[0],t.center[1]];for(let e of a?[0,Math.PI]:[0])for(let n=0;n<=k;n++){let i=e+(c+n/k)/r.binCount*o;p.push(t.center[0]+Math.cos(i)*f,t.center[1]+Math.sin(i)*f)}let m=c*360/r.binCount*(a?.5:1),h=360/r.binCount*(a?.5:1);return{focus:c,glyphs:[{shape:`ring`,at:t.center,radius:D,dashed:!0},{shape:`polyline`,points:p,closed:!0}],readout:`${m.toFixed(0)}° to ${(m+h).toFixed(0)}° · ${u} legs`}}},draw:({outputs:e,inputs:t,extra:n})=>[{layer:`lines`,context:!0,segments:e.segments,count:t.count,color:[.3,.55,.95,.75]},{layer:`lines`,segments:e.highlight,count:t.count,radius:2.2,color:[1,.85,.3,1]},{layer:`polygons`,positions:n.vertices,indices:n.indices,owners:n.owners,values:e.bins,triangleCount:n.triangleCount,ramp:`sequential`},{layer:`text`,...y({labels:[[`E`,1,0],[`N`,0,1],[`W`,-1,0],[`S`,0,-1]].map(([e,n,r])=>({text:e,at:[t.center[0]+n*.44,t.center[1]+r*.44],role:`label`}))})}]});export{j as default};