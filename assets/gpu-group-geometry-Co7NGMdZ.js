import{s as e,v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as o,n as s}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as c}from"./sorted-segment-offsets-ChmOtA9F.js";import{E as l,f as u,l as d,v as f,x as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";import{t as h}from"./gpu-geographic-distribution-BlEGpo45.js";import{a as g}from"./wgsl-common-DfyXiYB0.js";import{a as _}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as v}from"./geographic-distribution-validation-BjMQQJgk.js";var y=`GPUGroupGeometry`,b=Object.freeze({...e,measure:`distance`}),x=2**24-1,S=4096,C=4294967295,w=class{id;props;spatialContext;groupCount;medoidMaximumGroupSize;constructor(e){this.id=e.id??`group-geometry`,this.props=e,this.spatialContext=t(this.id,e.spatialContext,b);let{id:n}=this,{output:r}=e;l(e.positions,[`float32x2`],`${n} positions`);let i=e.positions.length;if(i<1)throw Error(`${n} needs at least one row`);if(i>x)throw Error(`${n} supports fewer than 2^24 rows`);if(l(e.labels,[`uint32`],`${n} labels`),e.labels.length!==i)throw Error(`${n} labels length must equal positions length`);if(e.weights&&(l(e.weights,[`float32`],`${n} weights`),e.weights.length!==i))throw Error(`${n} weights length must equal positions length`);if(l(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<8)throw Error(`${n} parameters must hold 8 float32 values`);this.groupCount=e.groupCount,u(n,`groupCount`,this.groupCount,1,x),e.noiseLabel!==void 0&&u(n,`noiseLabel`,e.noiseLabel,0),this.medoidMaximumGroupSize=e.medoidMaximumGroupSize??S,u(n,`medoidMaximumGroupSize`,this.medoidMaximumGroupSize);let a=this.groupCount,o=[[`counts`,r.counts,[`uint32`],a],[`bounds`,r.bounds,[`float32x4`],a],[`meanCenters`,r.meanCenters,[`float32x2`],a],[`weightedCenters`,r.weightedCenters,[`float32x2`],a],[`weightSums`,r.weightSums,[`float32`],a],[`medoidIndices`,r.medoidIndices,[`uint32`],a],[`ellipses`,r.ellipses,[`float32`],a*3],[`standardDistances`,r.standardDistances,[`float32`],a],[`overflow`,r.status?.overflow,[`uint32`],1]],s=0;for(let[e,t,r,i]of o)if(t&&(s++,l(t,r,`${n} output.${e}`),t.length<i))throw Error(`${n} output.${e} must hold at least ${i} rows`);if(s===0)throw Error(`${n} needs at least one output`);if((r.weightedCenters||r.weightSums)&&!e.weights)throw Error(`${n} output.weightedCenters and output.weightSums require weights`);if(r.status?.overflow&&!r.medoidIndices)throw Error(`${n} output.status.overflow requires output.medoidIndices`);m(n,o.map(([,e])=>e),[e.positions,e.labels,e.weights,e.parameters]),v(n,{},Object.fromEntries(o.map(([e,t])=>[e,t])),[{reads:{},writes:{medoidIndices:r.medoidIndices,overflow:r.status?.overflow}}])}getCommandNodes(e){let{props:t,id:l,groupCount:u,medoidMaximumGroupSize:m}=this,{output:v,positions:b,labels:x,weights:S,parameters:w}=t,{status:T,...E}=v,D=!!(v.ellipses||v.standardDistances),O=!!(v.meanCenters||v.weightedCenters||v.weightSums||D),k=!!(v.weightedCenters||v.weightSums||D);_(e,l,{positions:b,labels:x,weights:k?S:void 0,parameters:O?w:void 0,...E,overflow:T?.overflow});let A=b.length;for(let[t,n]of[[`group row scratch`,4*(A<=256?d(A):A)],[`group offsets scratch`,4*(u+1)],...O?[[`geographic mean scratch`,8*u],...v.ellipses?[[`geographic ellipse scratch`,12*u]]:[],...u<=64?[[`geographic grouped partial scratch`,12*u*Math.min(1024,Math.max(1,Math.ceil(A/256)))]]:[]]:[]])f(e.device,l,t,n,`reduce row or group counts`);p(l,e,[b,x,S,w,...Object.values(E),T?.overflow]);let j=b.length,M=(t,r)=>n(e,`${l}-${t}`,`float32`,r),N=(t,r)=>n(e,`${l}-${t}`,`uint32`,r),P=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),F=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),I=(t,n,r,a,o=``)=>i(e,{id:`${l}-${t}`,operation:y,variant:t,bindings:r,invocationCount:n,declarations:`const GROUP_COUNT: u32 = ${u}u;\n${o}`,body:a}),L=[],R=N(`group-keys`,j),z=N(`row-indices`,j),B=N(`group-counts`,u),V=t.noiseLabel!==void 0&&t.noiseLabel<u?`valid = valid && label != ${t.noiseLabel}u;`:``;L.push(I(`keys`,j,[P(`positions`,b,`f32`),P(`labels`,x,`u32`),F(`groupKeys`,R,`u32`),F(`rowIndices`,z,`u32`)],`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${V}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  rowIndices[rowIndicesOffset + index] = index;`));let H=N(`sorted-keys`,j),U=N(`sorted-indices`,j),W=N(`segment-offsets`,u+1);L.push(...new r({id:`${l}-sort`,keys:R,values:z,outputKeys:H,outputValues:U,keyBits:o(u)}).getCommandNodes(e),...c(e,{id:l,operation:y,groupCount:u,sortedGroupKeys:H,segmentOffsets:W,counts:B}));let G=!!(v.bounds||v.medoidIndices),K=G?M(`sorted-x`,j):void 0,q=G?M(`sorted-y`,j):void 0;if(K&&q&&L.push(I(`gather`,j,[P(`sortedIndices`,U,`u32`),P(`positions`,b,`f32`),F(`sortedX`,K),F(`sortedY`,q)],`let row = sortedIndices[sortedIndicesOffset + index];
  sortedX[sortedXOffset + index] = positions[positionsOffset + row * 2u];
  sortedY[sortedYOffset + index] = positions[positionsOffset + row * 2u + 1u];`)),v.bounds&&K&&q){let t=(t,n,r)=>{let i=M(`bounds-${t}`,u);return L.push(s(e,{id:`${l}-bounds-${t}`,operation:y,segmentCount:u,input:n,segmentOffsets:W,output:i,reduction:r})),i},n=t(`min-x`,K,`min`),r=t(`min-y`,q,`min`),i=t(`max-x`,K,`max`),a=t(`max-y`,q,`max`);L.push(I(`publish-bounds`,u,[P(`counts`,B,`u32`),P(`minX`,n,`f32`),P(`minY`,r,`f32`),P(`maxX`,i,`f32`),P(`maxY`,a,`f32`),F(`bounds`,v.bounds)],`let isEmpty = counts[countsOffset + index] == 0u;
  let nan = getQuietNaN(index);
  bounds[boundsOffset + index * 4u] = select(minX[minXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 1u] = select(minY[minYOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 2u] = select(maxX[maxXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 3u] = select(maxY[maxYOffset + index], nan, isEmpty);`,g))}if(v.counts&&L.push(I(`publish-counts`,u,[P(`counts`,B,`u32`),F(`countsOut`,v.counts,`u32`)],`countsOut[countsOutOffset + index] = counts[countsOffset + index];`)),v.meanCenters||D&&!S){let t=v.meanCenters??n(e,`${l}-mean-centers`,`float32x2`,u);L.push(...new h({id:`${l}-unweighted`,positions:b,groupIds:R,groupCount:u,parameters:w,output:{meanCenters:t,ellipses:S?void 0:v.ellipses,standardDistances:S?void 0:v.standardDistances}}).getCommandNodes(e))}if(S&&(v.weightedCenters||v.weightSums||D)&&L.push(...new h({id:`${l}-weighted`,positions:b,weights:S,groupIds:R,groupCount:u,parameters:w,output:{meanCenters:v.weightedCenters??n(e,`${l}-weighted-centers`,`float32x2`,u),weightSums:v.weightSums,ellipses:v.ellipses,standardDistances:v.standardDistances}}).getCommandNodes(e)),v.medoidIndices&&K&&q){let t=M(`medoid-costs`,j);L.push(I(`medoid-cost`,j,[P(`sortedKeys`,H,`u32`),P(`segmentOffsets`,W,`u32`),P(`sortedX`,K,`f32`),P(`sortedY`,q,`f32`),F(`costs`,t)],`let group = sortedKeys[sortedKeysOffset + index];
  var cost = 0.0;
  if (group < GROUP_COUNT) {
    let begin = segmentOffsets[segmentOffsetsOffset + group];
    let end = segmentOffsets[segmentOffsetsOffset + group + 1u];
    if (end - begin <= MEDOID_MAXIMUM_GROUP_SIZE) {
      let x = sortedX[sortedXOffset + index];
      let y = sortedY[sortedYOffset + index];
      for (var other = begin; other < end; other++) {
        let dx = sortedX[sortedXOffset + other] - x;
        let dy = sortedY[sortedYOffset + other] - y;
        cost += sqrt(dx * dx + dy * dy);
      }
    }
  }
  costs[costsOffset + index] = cost;`,`const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${m}u;`)),v.status?.overflow&&L.push(a(e,{id:`${l}-zero-overflow`,operation:y,view:v.status.overflow,type:`u32`,value:`0u`,componentCount:1}));let n=[P(`segmentOffsets`,W,`u32`),P(`costs`,t,`f32`),P(`sortedIndices`,U,`u32`),F(`medoids`,v.medoidIndices,`u32`)];v.status?.overflow&&n.push(F(`overflow`,v.status.overflow,`atomic<u32>`)),L.push(I(`medoid-select`,u,n,`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  var best = NO_MEDOID;
  if (end - begin > MEDOID_MAXIMUM_GROUP_SIZE) {
    ${v.status?.overflow?`atomicStore(&overflow[overflowOffset], 1u);`:``}
  } else if (end > begin) {
    var bestCost = costs[costsOffset + begin];
    best = sortedIndices[sortedIndicesOffset + begin];
    for (var row = begin + 1u; row < end; row++) {
      let cost = costs[costsOffset + row];
      if (cost < bestCost) {
        bestCost = cost;
        best = sortedIndices[sortedIndicesOffset + row];
      }
    }
  }
  medoids[medoidsOffset + index] = best;`,`const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${m}u;
const NO_MEDOID: u32 = ${C}u;`))}return L}};export{w as t};