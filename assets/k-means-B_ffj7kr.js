import{t as e}from"./spatial-output-validation-nw5lOmVL.js";import{t}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{n,t as r}from"./philox-random-DfVxWNPn.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-compaction-B6IzcF0P.js";import{t as o}from"./gpu-group-aggregation-CEYb6GJw.js";import{a as s,o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as l,f as u,p as d,v as f,x as ee,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as te}from"./spatial-clustering-parameters-BsDsWnge.js";import{a as h,d as g,l as _}from"./wgsl-common-DfyXiYB0.js";import{s as v,v as y}from"./spatial-context-DwZEd4EZ.js";import{n as b}from"./point-observations-CeqP-hIo.js";import{n as x}from"./pick-Bo3t3hjv.js";import{t as S}from"./pointer-driver-Bk4iFPMM.js";import{t as C}from"./japan-coast-Bi3PJqMb.js";import{C as ne,v as w}from"./index-DIfTuwM1.js";import{A as T,C as E,E as D,S as O,T as k,h as A,k as j,m as re,p as M,w as N}from"./pointer-overlay-Qh8qflcA.js";var P=Object.freeze({...v,measure:`distance`}),F=`GPUSpatialKMeans`,I=`0xffffffffu`,L=256,ie=16,ae=256,R=256,z=64,B=5,V=4;function H(e={}){let t=e.seed??0,n=e.tolerance??0;if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`GPUSpatialKMeans seed must be a uint32`);if(!Number.isFinite(n)||n<0)throw Error(`GPUSpatialKMeans tolerance must be a finite number >= 0`);let r=new Uint32Array(2);return r[0]=t,new Float32Array(r.buffer)[1]=n,r}var U=class{id;estimatedCommandNodeCount;props;initialization;restartCount;centerCoordinates;spatialContext;constructor(t){let n=t.id??`kmeans`;this.id=n,this.props=t,this.initialization=t.initialization??`kmeans++`,this.restartCount=t.restartCount??1,this.centerCoordinates=t.centerCoordinates??!1,this.spatialContext=y(n,t.spatialContext,P),l(t.positions,[`float32x2`],`${n} positions`);let r=t.positions.length;if(r<1)throw Error(`${n} positions must hold at least one row`);if(!Number.isInteger(t.k)||t.k<1||t.k>256)throw Error(`${n} k must be an integer in [1, 256]`);if(u(n,`maximumIterations`,t.maximumIterations,1,64),this.initialization!==`first-valid`&&this.initialization!==`kmeans++`)throw Error(`${n} initialization must be 'first-valid' or 'kmeans++'`);if(!Number.isInteger(this.restartCount)||this.restartCount<1||this.restartCount>32)throw Error(`${n} restartCount must be an integer in [1, 32]`);if(this.restartCount>1&&this.initialization!==`kmeans++`)throw Error(`${n} restartCount above 1 needs the 'kmeans++' initialization`);if(l(t.parameters,[`uint32`],`${n} parameters`),t.parameters.length<2)throw Error(`${n} parameters must hold 2 uint32`);let{output:i}=t;for(let[e,t]of[[`inertia`,i.inertia],[`status.iterationCount`,i.status?.iterationCount],[`status.converged`,i.status?.converged]])if(t&&(l(t,e===`inertia`?[`float32`]:[`uint32`],`${n} output.${e}`),t.length!==1))throw Error(`${n} output.${e} must hold one row`);if(l(i.labels,[`uint32`],`${n} output.labels`),i.labels.length!==r)throw Error(`${n} output.labels length must equal positions length`);if(l(i.centers,[`float32x2`],`${n} output.centers`),i.centers.length!==t.k)throw Error(`${n} output.centers length must equal k`);if(i.sizes&&(l(i.sizes,[`uint32`],`${n} output.sizes`),i.sizes.length!==t.k))throw Error(`${n} output.sizes length must equal k`);if(i.squaredDistances&&(l(i.squaredDistances,[`float32`],`${n} output.squaredDistances`),i.squaredDistances.length!==r))throw Error(`${n} output.squaredDistances length must equal positions length`);e(n,Object.fromEntries(W(i).map((e,t)=>[`output[${t}]`,e])),[{read:[this.centerCoordinates?void 0:i.centers],write:[i.labels,i.squaredDistances]},{read:[i.labels],write:[i.sizes]},...(t.restartCount??1)===1?[{read:[i.labels],write:[i.sizes,this.centerCoordinates?void 0:i.centers]}]:[]]),this.estimatedCommandNodeCount=R+this.restartCount*(z+B*t.maximumIterations+(this.initialization===`kmeans++`?V*t.k:0)),d(n,this.estimatedCommandNodeCount),p(n,W(i),[t.positions,t.parameters])}getCommandNodes(e){let{id:l,props:u}=this,{positions:d,k:p,maximumIterations:v,parameters:y,output:b}=u,{labels:x,centers:S}=b;ee(l,e,[d,y,...W(b)]);let C=d.length;m(e,l,{positions:d,parameters:y,...Object.fromEntries(W(b).map((e,t)=>[`output[${t}]`,e]))});let ne=p*Math.min(ae,Math.ceil(C/L));for(let[t,n]of[[`row scratch`,C*4],[`partial reductions`,ne*4],[`cluster sums`,p*8],[`iteration dispatch`,48]])f(e.device,l,t,n,`reduce positions rows or k`);let w=this.restartCount,T=w>1,E=(t,n,r)=>i(e,`${l}-${t}`,n,r),D=this.centerCoordinates,O=D?i(e,`${l}-origin`,`float32`,2):void 0,k=O?[{name:`origin`,view:O,type:`f32`,access:`read`}]:[],A=(e,t)=>`(positions[positionsOffset + ${e} * 2u${t===1?` + 1u`:``}]${D?` - origin[originOffset + ${t}u]`:``})`,j=E(`valid-flags`,`uint32`,C),re=E(`row-ids`,`uint32`,C),M=E(`xs`,`float32`,C),N=E(`ys`,`float32`,C),P=b.sizes??E(`sizes`,`uint32`,p),R=T||D?i(e,`${l}-work-centers`,`float32x2`,p):S,z=T?E(`work-sizes`,`uint32`,p):P,B=b.squaredDistances??(T||b.inertia?E(`squared-distances`,`float32`,C):void 0),V=E(`state`,`uint32`,3),H=E(`seed-cursor`,`uint32`,2),U=Math.min(ae,Math.ceil(C/L)),G=Math.ceil(C/U/L)*L,K=Math.min(ie,Math.floor(L/p)),q=E(`partial-counts`,`uint32`,p*U),J=E(`partial-sums-x`,`float32`,p*U),oe=E(`partial-sums-y`,`float32`,p*U),Y=E(`iteration-distances`,`float32`,C),X=i(e,`${l}-cluster-sums`,`float32x2`,p),Z=s(e,{id:`${l}-iteration-dispatch`,operation:F,shapes:[[C],[U*L,L],[p*L,L],[L,L]]}),Q=`const K: u32 = ${p}u;
const NOISE: u32 = ${te}u;

// Mirrors getGPUSpatialKMeansRestartSeed: restart 0 uses the seed itself, later restarts a murmur3 mix.
// The seed is the per-frame parameters word 0, so the source is the same for every seed.
fn getRestartSeed(seed: u32, restart: u32) -> u32 {
  if (restart == 0u) {
    return seed;
  }
  var hash = seed + restart * 0x9e3779b9u;
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x85ebca6bu;
  hash = hash ^ (hash >> 13u);
  hash = hash * 0xc2b2ae35u;
  return hash ^ (hash >> 16u);
}

${_}
${g}

${h}`,$=[c(e,{id:`${l}-prepare`,operation:F,variant:`prepare`,bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},{name:`validFlags`,view:j,type:`u32`,access:`read_write`},{name:`rowIds`,view:re,type:`u32`,access:`read_write`},{name:`xs`,view:M,type:`f32`,access:`read_write`},{name:`ys`,view:N,type:`f32`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read_write`}],invocationCount:C,declarations:Q,body:`if (index == 0u) {
    seedCursor[seedCursorOffset] = 0xffffffffu;
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isFiniteFloat(x) && isFiniteFloat(y);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  rowIds[rowIdsOffset + index] = index;
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);`})];if(O){let t=E(`origin-row`,`uint32`,1);$.push(c(e,{id:`${l}-origin-reset`,operation:F,variant:`origin-reset`,bindings:[{name:`originRow`,view:t,type:`u32`,access:`read_write`}],invocationCount:1,body:`originRow[originRowOffset] = ${I};`}),c(e,{id:`${l}-origin-row`,operation:F,variant:`origin-row`,bindings:[{name:`validFlags`,view:j,type:`u32`,access:`read`},{name:`originRow`,view:t,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,workgroupSize:L,guardIndex:!1,declarations:`var<workgroup> groupRow: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
      atomicStore(&groupRow, ${I});
    }
    workgroupBarrier();
    if (index < INVOCATION_COUNT && validFlags[validFlagsOffset + index] != 0u) {
      atomicMin(&groupRow, index);
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      atomicMin(&originRow[originRowOffset], atomicLoad(&groupRow));
    }`}),c(e,{id:`${l}-origin`,operation:F,variant:`origin`,bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},{name:`originRow`,view:t,type:`u32`,access:`read`},{name:`origin`,view:O,type:`f32`,access:`read_write`}],invocationCount:1,body:`let row = originRow[originRowOffset];
    let found = row != ${I};
    origin[originOffset] = select(0.0, positions[positionsOffset + row * 2u], found);
    origin[originOffset + 1u] = select(0.0, positions[positionsOffset + row * 2u + 1u], found);`}),c(e,{id:`${l}-center-coordinates`,operation:F,variant:`center-coordinates`,bindings:[{name:`validFlags`,view:j,type:`u32`,access:`read`},{name:`origin`,view:O,type:`f32`,access:`read`},{name:`xs`,view:M,type:`f32`,access:`read_write`},{name:`ys`,view:N,type:`f32`,access:`read_write`}],invocationCount:C,body:`if (validFlags[validFlagsOffset + index] != 0u) {
      xs[xsOffset + index] = xs[xsOffset + index] - origin[originOffset];
      ys[ysOffset + index] = ys[ysOffset + index] - origin[originOffset + 1u];
    }`}))}let se=this.initialization===`kmeans++`,ce=se?E(`min-distance`,`float32`,C):void 0,le=se?E(`keys`,`uint32`,C):void 0,ue=se?E(`best-key`,`uint32`,1):void 0,de=se?E(`best-row`,`uint32`,1):void 0,fe=b.inertia??E(`best-inertia`,`float32`,1),pe=T?i(e,`${l}-best-centers`,`float32x2`,p):void 0,me=T?E(`best-state`,`uint32`,3):void 0,he=(t,n,r,i)=>c(e,{id:t,operation:F,variant:n?`assign-final`:`assign`,...n?{}:Z.getGate(0),bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},...k,{name:`centers`,view:r,type:`f32`,access:`read`},{name:`labels`,view:x,type:`u32`,access:`read_write`},...n?[]:[{name:`state`,view:V,type:`u32`,access:`read`}],...i?[{name:`squaredDistances`,view:i,type:`f32`,access:`read_write`}]:[]],invocationCount:C,declarations:Q,body:`${n?``:`// Converged: centers no longer move, so labels are already final.
  if (state[stateOffset] != 0u) {
    return;
  }
  `}let x = ${A(`index`,0)};
  let y = ${A(`index`,1)};
  var best = NOISE;
  var bestDistance = getQuietNaN(index);
  if (isFiniteFloat(x) && isFiniteFloat(y)) {
    // Strict comparison: the lowest center ID wins ties; NaN centers never win.
    for (var center = 0u; center < K; center++) {
      let deltaX = centers[centersOffset + center * 2u] - x;
      let deltaY = centers[centersOffset + center * 2u + 1u] - y;
      let distance = deltaX * deltaX + deltaY * deltaY;
      if (!isNanBits(distance) && (best == NOISE || distance < bestDistance)) {
        best = center;
        bestDistance = distance;
      }
    }
  }
  labels[labelsOffset + index] = best;
  ${i?`squaredDistances[squaredDistancesOffset + index] = select(getQuietNaN(index), bestDistance, best != NOISE);`:``}`}),ge=t=>{if(this.initialization===`first-valid`){let n=E(`compact-ids`,`uint32`,C),r=E(`valid-count`,`uint32`,1);$.push(...new a({id:`${t}-valid-rows`,input:re,flags:j,output:n,count:r}).getCommandNodes(e),c(e,{id:`${t}-initial-centers`,operation:F,variant:`initial-centers`,bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},...k,{name:`compactIds`,view:n,type:`u32`,access:`read`},{name:`validCount`,view:r,type:`u32`,access:`read`},{name:`centers`,view:R,type:`f32`,access:`read_write`}],invocationCount:p,declarations:Q,body:`let nan = getQuietNaN(index);
    var x = nan;
    var y = nan;
    if (index < validCount[validCountOffset]) {
      let row = compactIds[compactIdsOffset + index];
      x = ${A(`row`,0)};
      y = ${A(`row`,1)};
    }
    centers[centersOffset + index * 2u] = x;
    centers[centersOffset + index * 2u + 1u] = y;`}))}else for(let i=0;i<p;i++){let a=`${t}-seed-${i}`;$.push(c(e,{id:`${a}-begin`,operation:F,variant:`seed-begin`,bindings:[{name:`bestKey`,view:ue,type:`u32`,access:`read_write`},{name:`bestRow`,view:de,type:`u32`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,body:`bestKey[bestKeyOffset] = ${I};
    bestRow[bestRowOffset] = ${I};
    seedCursor[seedCursorOffset + 1u] = seedCursor[seedCursorOffset + 1u] + 1u;`}),c(e,{id:`${a}-key`,operation:F,variant:`seed-key`,bindings:[{name:`validFlags`,view:j,type:`u32`,access:`read`},{name:`minDistance`,view:ce,type:`f32`,access:`read`},{name:`keys`,view:le,type:`u32`,access:`read_write`},{name:`bestKey`,view:ue,type:`atomic<u32>`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read`},{name:`parameters`,view:y,type:`u32`,access:`read`}],invocationCount:C,workgroupSize:L,guardIndex:!1,declarations:`${Q}
  var<workgroup> groupBestKey: atomic<u32>;

  ${n}
  // The registered purpose of this stream, not a seed: the seed is a per-frame parameters word.
  const PHILOX_PURPOSE: u32 = ${r.kMeansPlusPlusSeeding.id}u;`,body:`if (localInvocationIndex == 0u) {
      atomicStore(&groupBestKey, 0xffffffffu);
    }
    workgroupBarrier();
    let round = seedCursor[seedCursorOffset + 1u];
    let restartSeed = getRestartSeed(parameters[parametersOffset], seedCursor[seedCursorOffset]);
    var key = 0xffffffffu;
    if (index < INVOCATION_COUNT && validFlags[validFlagsOffset + index] != 0u) {
      // Exponential race: the minimum of -ln(u) / w samples a row with probability proportional to w.
      // One Philox block per (row, round) under the registered purpose; the uniform is open, so
      // log() never sees 0 and never returns the 0 that would win the race whatever the weight.
      let weight = select(minDistance[minDistanceOffset + index], 1.0, round == 0u);
      let word = philox4x32(vec4<u32>(index, round, 0u, 0u), vec2<u32>(restartSeed, PHILOX_PURPOSE)).x;
      let uniformValue = philoxOpenUnitFloat(word);
      key = select(0x7f800000u, bitcast<u32>(-log(uniformValue) / weight), weight > 0.0);
      atomicMin(&groupBestKey, key);
      keys[keysOffset + index] = key;
    } else if (index < INVOCATION_COUNT) {
      keys[keysOffset + index] = key;
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      atomicMin(&bestKey[bestKeyOffset], atomicLoad(&groupBestKey));
    }`}),c(e,{id:`${a}-row`,operation:F,variant:`seed-row`,bindings:[{name:`keys`,view:le,type:`u32`,access:`read`},{name:`bestKey`,view:ue,type:`u32`,access:`read`},{name:`bestRow`,view:de,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,body:`let best = bestKey[bestKeyOffset];
    if (best != 0xffffffffu && keys[keysOffset + index] == best) {
      atomicMin(&bestRow[bestRowOffset], index);
    }`}),c(e,{id:`${a}-update`,operation:F,variant:`seed-update`,bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},...k,{name:`validFlags`,view:j,type:`u32`,access:`read`},{name:`bestRow`,view:de,type:`u32`,access:`read`},{name:`minDistance`,view:ce,type:`f32`,access:`read_write`},{name:`centers`,view:R,type:`f32`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read`}],invocationCount:C,declarations:Q,body:`let round = seedCursor[seedCursorOffset + 1u];
    let row = bestRow[bestRowOffset];
    let hasCenter = row != 0xffffffffu;
    var centerX = getQuietNaN(index);
    var centerY = centerX;
    if (hasCenter) {
      centerX = ${A(`row`,0)};
      centerY = ${A(`row`,1)};
    }
    if (index == 0u) {
      centers[centersOffset + round * 2u] = centerX;
      centers[centersOffset + round * 2u + 1u] = centerY;
    }
    if (hasCenter && validFlags[validFlagsOffset + index] != 0u) {
      let deltaX = ${A(`index`,0)} - centerX;
      let deltaY = ${A(`index`,1)} - centerY;
      let distance = deltaX * deltaX + deltaY * deltaY;
      minDistance[minDistanceOffset + index] =
        select(min(minDistance[minDistanceOffset + index], distance), distance, round == 0u);
    }`}))}},_e=t=>{for(let n=0;n<v;n++){let r=`${t}-iteration-${n}`;$.push(he(`${r}-assign`,!1,R,Y),c(e,{id:`${r}-partial`,operation:F,variant:`partial`,bindings:[{name:`labels`,view:x,type:`u32`,access:`read`},{name:`xs`,view:M,type:`f32`,access:`read`},{name:`ys`,view:N,type:`f32`,access:`read`},{name:`state`,view:V,type:`u32`,access:`read`},{name:`partialCounts`,view:q,type:`u32`,access:`read_write`},{name:`partialSumsX`,view:J,type:`f32`,access:`read_write`},{name:`partialSumsY`,view:oe,type:`f32`,access:`read_write`}],workgroupSize:L,invocationCount:U*L,guardIndex:!1,...Z.getGate(1),declarations:`${Q}
  const ROWS: u32 = ${C}u;
  const CHUNK_SIZE: u32 = ${G}u;
  const CHUNK_COUNT: u32 = ${U}u;
  const TILE_COUNT: u32 = ${G/L}u;
  const SLICES: u32 = ${K}u;
  const LANES: u32 = ${K*p}u;
  var<workgroup> tileLabels: array<u32, ${L}>;
  var<workgroup> tileXs: array<f32, ${L}>;
  var<workgroup> tileYs: array<f32, ${L}>;
  var<workgroup> laneCounts: array<u32, ${L}>;
  var<workgroup> laneSumsX: array<f32, ${L}>;
  var<workgroup> laneSumsY: array<f32, ${L}>;`,body:`let chunk = index / ${L}u;
    let converged = state[stateOffset] != 0u;
    let begin = chunk * CHUNK_SIZE;
    let end = min(begin + CHUNK_SIZE, ROWS);
    let lane = localInvocationIndex;
    let cluster = lane % K;
    let slice = lane / K;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let row = begin + tile * ${L}u + lane;
      var label = NOISE;
      var x = 0.0;
      var y = 0.0;
      if (!converged && row < end) {
        label = labels[labelsOffset + row];
        x = xs[xsOffset + row];
        y = ys[ysOffset + row];
      }
      tileLabels[lane] = label;
      tileXs[lane] = x;
      tileYs[lane] = y;
      workgroupBarrier();
      if (lane < LANES) {
        // Lane (slice, cluster) owns tile entries slice, slice + SLICES, ...: a fixed order.
        for (var entry = slice; entry < ${L}u; entry += SLICES) {
          if (tileLabels[entry] == cluster) {
            count++;
            sumX += tileXs[entry];
            sumY += tileYs[entry];
          }
        }
      }
      workgroupBarrier();
    }
    laneCounts[lane] = count;
    laneSumsX[lane] = sumX;
    laneSumsY[lane] = sumY;
    workgroupBarrier();
    if (lane < K) {
      var chunkCountTotal = 0u;
      var chunkSumX = 0.0;
      var chunkSumY = 0.0;
      for (var sliceIndex = 0u; sliceIndex < SLICES; sliceIndex++) {
        chunkCountTotal += laneCounts[sliceIndex * K + lane];
        chunkSumX += laneSumsX[sliceIndex * K + lane];
        chunkSumY += laneSumsY[sliceIndex * K + lane];
      }
      let slot = lane * CHUNK_COUNT + chunk;
      partialCounts[partialCountsOffset + slot] = chunkCountTotal;
      partialSumsX[partialSumsXOffset + slot] = chunkSumX;
      partialSumsY[partialSumsYOffset + slot] = chunkSumY;
    }`}),c(e,{id:`${r}-update`,operation:F,variant:`update`,bindings:[{name:`partialCounts`,view:q,type:`u32`,access:`read`},{name:`partialSumsX`,view:J,type:`f32`,access:`read`},{name:`partialSumsY`,view:oe,type:`f32`,access:`read`},{name:`sizes`,view:z,type:`u32`,access:`read_write`},{name:`clusterSums`,view:X,type:`f32`,access:`read_write`},{name:`state`,view:V,type:`u32`,access:`read`}],workgroupSize:L,invocationCount:p*L,guardIndex:!1,...Z.getGate(2),declarations:`const CHUNK_COUNT: u32 = ${U}u;
  var<workgroup> reducedCounts: array<u32, ${L}>;
  var<workgroup> reducedSumsX: array<f32, ${L}>;
  var<workgroup> reducedSumsY: array<f32, ${L}>;`,body:`let cluster = index / ${L}u;
    let lane = localInvocationIndex;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${L}u) {
      let slot = cluster * CHUNK_COUNT + chunk;
      count += partialCounts[partialCountsOffset + slot];
      sumX += partialSumsX[partialSumsXOffset + slot];
      sumY += partialSumsY[partialSumsYOffset + slot];
    }
    reducedCounts[lane] = count;
    reducedSumsX[lane] = sumX;
    reducedSumsY[lane] = sumY;
    workgroupBarrier();
    for (var stride = ${L/2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        reducedCounts[lane] += reducedCounts[lane + stride];
        reducedSumsX[lane] += reducedSumsX[lane + stride];
        reducedSumsY[lane] += reducedSumsY[lane + stride];
      }
      workgroupBarrier();
    }
    // Converged: centers no longer move; the partials are stale and the final sizes come later.
    if (lane == 0u && state[stateOffset] == 0u) {
      sizes[sizesOffset + cluster] = reducedCounts[0];
      clusterSums[clusterSumsOffset + cluster * 2u] = reducedSumsX[0];
      clusterSums[clusterSumsOffset + cluster * 2u + 1u] = reducedSumsY[0];
    }`}),c(e,{id:`${r}-relocate`,operation:F,variant:`relocate`,bindings:[{name:`positions`,view:d,type:`f32`,access:`read`},...k,{name:`labels`,view:x,type:`u32`,access:`read`},{name:`distances`,view:Y,type:`f32`,access:`read`},{name:`clusterSums`,view:X,type:`f32`,access:`read`},{name:`sizes`,view:z,type:`u32`,access:`read_write`},{name:`centers`,view:R,type:`f32`,access:`read_write`},{name:`state`,view:V,type:`atomic<u32>`,access:`read_write`}],workgroupSize:L,invocationCount:L,guardIndex:!1,...Z.getGate(3),declarations:`${Q}
  const ROWS: u32 = ${C}u;
  var<workgroup> activeFlag: u32;
  var<workgroup> stopFlag: u32;
  var<workgroup> emptyCount: u32;
  var<workgroup> pickDistance: f32;
  var<workgroup> pickRow: u32;
  var<workgroup> clusterCounts: array<u32, ${L}>;
  var<workgroup> clusterSumsX: array<f32, ${L}>;
  var<workgroup> clusterSumsY: array<f32, ${L}>;
  var<workgroup> emptyClusters: array<u32, ${L}>;
  var<workgroup> laneDistances: array<f32, ${L}>;
  var<workgroup> laneRows: array<u32, ${L}>;`,body:`let lane = localInvocationIndex;
  if (lane == 0u) {
    activeFlag = select(0u, 1u, atomicLoad(&state[stateOffset]) == 0u);
  }
  // Converged: the partials are stale and the centers no longer move.
  if (workgroupUniformLoad(&activeFlag) == 0u) {
    return;
  }
  if (lane < K) {
    clusterCounts[lane] = sizes[sizesOffset + lane];
    clusterSumsX[lane] = clusterSums[clusterSumsOffset + lane * 2u];
    clusterSumsY[lane] = clusterSums[clusterSumsOffset + lane * 2u + 1u];
  }
  workgroupBarrier();
  if (lane == 0u) {
    // Clusters whose center could not be initialized (NaN) are not empty clusters to relocate.
    var count = 0u;
    for (var cluster = 0u; cluster < K; cluster++) {
      let centerX = centers[centersOffset + cluster * 2u];
      if (clusterCounts[cluster] == 0u && isFiniteFloat(centerX)) {
        emptyClusters[count] = cluster;
        count++;
      }
    }
    emptyCount = count;
    stopFlag = 0u;
    pickDistance = 0.0;
    pickRow = 0u;
  }
  let pickLimit = workgroupUniformLoad(&emptyCount);
  for (var pickIndex = 0u; pickIndex < pickLimit; pickIndex++) {
    // scikit-learn returns without relocating when every point sits on its center.
    if (workgroupUniformLoad(&stopFlag) != 0u) {
      break;
    }
    let previousDistance = pickDistance;
    let previousRow = pickRow;
    var bestDistance = -1.0;
    var bestRow = NOISE;
    for (var row = lane; row < ROWS; row += ${L}u) {
      let distance = distances[distancesOffset + row];
      if (isFiniteFloat(distance) && labels[labelsOffset + row] != NOISE) {
        // Rank (distance descending, row ascending) strictly after the previous winner.
        let ranked = pickIndex == 0u || distance < previousDistance ||
          (distance == previousDistance && row > previousRow);
        if (ranked && distance > bestDistance) {
          bestDistance = distance;
          bestRow = row;
        }
      }
    }
    laneDistances[lane] = bestDistance;
    laneRows[lane] = bestRow;
    workgroupBarrier();
    for (var stride = ${L/2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherDistance = laneDistances[lane + stride];
        let otherRow = laneRows[lane + stride];
        if (otherDistance > laneDistances[lane] ||
          (otherDistance == laneDistances[lane] && otherRow < laneRows[lane])) {
          laneDistances[lane] = otherDistance;
          laneRows[lane] = otherRow;
        }
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let row = laneRows[0];
      let distance = laneDistances[0];
      if (row == NOISE || (pickIndex == 0u && distance == 0.0)) {
        stopFlag = 1u;
      } else {
        pickDistance = distance;
        pickRow = row;
        let emptyCluster = emptyClusters[pickIndex];
        let donor = labels[labelsOffset + row];
        let x = ${A(`row`,0)};
        let y = ${A(`row`,1)};
        clusterSumsX[donor] -= x;
        clusterSumsY[donor] -= y;
        clusterCounts[donor] -= 1u;
        clusterSumsX[emptyCluster] = x;
        clusterSumsY[emptyCluster] = y;
        clusterCounts[emptyCluster] = 1u;
      }
    }
    workgroupBarrier();
  }
  // A cluster still empty (every point on its center, or its only member was taken) moves to the
  // mean of the largest cluster, lowest ID on ties, like scikit-learn's \`_average_centers\`.
  var largest = 0u;
  for (var cluster = 1u; cluster < K; cluster++) {
    if (clusterCounts[cluster] > clusterCounts[largest]) {
      largest = cluster;
    }
  }
  if (lane < K) {
    let size = clusterCounts[lane];
    let previousX = centers[centersOffset + lane * 2u];
    let previousY = centers[centersOffset + lane * 2u + 1u];
    var nextX = previousX;
    var nextY = previousY;
    if (size > 0u) {
      nextX = clusterSumsX[lane] / f32(size);
      nextY = clusterSumsY[lane] / f32(size);
    } else if (isFiniteFloat(previousX) && clusterCounts[largest] > 0u) {
      nextX = clusterSumsX[largest] / f32(clusterCounts[largest]);
      nextY = clusterSumsY[largest] / f32(clusterCounts[largest]);
    }
    sizes[sizesOffset + lane] = size;
    let deltaX = nextX - previousX;
    let deltaY = nextY - previousY;
    let shift = sqrt(deltaX * deltaX + deltaY * deltaY);
    // A never-initialized (NaN) center does not move. Any other non-finite shift (overflowed
    // coordinate sums or squares, or an infinite center) counts as +infinity, so the iteration
    // cannot report convergence.
    if (isFiniteFloat(shift)) {
      atomicMax(&state[stateOffset + 2u], bitcast<u32>(shift));
    } else if (!isNanBits(previousX)) {
      atomicMax(&state[stateOffset + 2u], 0x7f800000u);
    }
    centers[centersOffset + lane * 2u] = nextX;
    centers[centersOffset + lane * 2u + 1u] = nextY;
  }`}),c(e,{id:`${r}-converge`,operation:F,variant:`converge`,bindings:[{name:`state`,view:V,type:`u32`,access:`read_write`},{name:`parameters`,view:y,type:`u32`,access:`read`},Z.binding],invocationCount:1,body:`if (state[stateOffset] != 0u) {
      return;
    }
    state[stateOffset + 1u] = state[stateOffset + 1u] + 1u;
    // The tolerance is the per-frame parameters word 1 (float32 bits).
    if (bitcast<f32>(state[stateOffset + 2u]) <= bitcast<f32>(parameters[parametersOffset + 1u])) {
      state[stateOffset] = 1u;
      ${Z.disarmWGSL}
    }
    state[stateOffset + 2u] = 0u;`}))}},ve=E(`score-partial-sums`,`float32`,U),ye=E(`score-partial-counts`,`uint32`,U),be=n=>{$.push(c(e,{id:`${n}-partial`,operation:F,variant:`score-partial`,bindings:[{name:`squaredDistances`,view:B,type:`f32`,access:`read`},{name:`partialSums`,view:ve,type:`f32`,access:`read_write`},{name:`partialCounts`,view:ye,type:`u32`,access:`read_write`}],workgroupSize:L,invocationCount:U*L,guardIndex:!1,declarations:`${Q}
${t}
const ROWS: u32 = ${C}u;
const CHUNK_SIZE: u32 = ${G}u;
var<workgroup> laneSums: array<f32, ${L}>;
var<workgroup> laneCounts: array<u32, ${L}>;`,body:`let lane = localInvocationIndex;
  let chunk = index / ${L}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  var sum = 0.0;
  // Rounding error of every addition, read off the integer significands. (The float Kahan form
  // (next - sum) - value is algebraically zero and a fast-math shader compiler removes it.)
  var roundingError = 0.0;
  var count = 0u;
  for (var row = begin + lane; row < end; row += ${L}u) {
    let value = squaredDistances[squaredDistancesOffset + row];
    if (isFiniteFloat(value)) {
      let next = sum + value;
      roundingError += getAdditionRoundingError(sum, value, next);
      sum = next;
      count++;
    }
  }
  laneSums[lane] = sum + roundingError;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${L/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    partialSums[partialSumsOffset + chunk] = laneSums[0];
    partialCounts[partialCountsOffset + chunk] = laneCounts[0];
  }`}),c(e,{id:n,operation:F,variant:T?`restart-score`:`inertia`,bindings:[{name:`partialSums`,view:ve,type:`f32`,access:`read`},{name:`partialCounts`,view:ye,type:`u32`,access:`read`},{name:`inertia`,view:fe,type:`f32`,access:`read_write`},...T?[{name:`centers`,view:R,type:`f32`,access:`read`},{name:`bestCenters`,view:pe,type:`f32`,access:`read_write`},{name:`state`,view:V,type:`u32`,access:`read`},{name:`bestState`,view:me,type:`u32`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read`}]:[]],workgroupSize:L,invocationCount:L,guardIndex:!1,declarations:`${Q}
const CHUNK_COUNT: u32 = ${U}u;
var<workgroup> laneSums: array<f32, ${L}>;
var<workgroup> laneCounts: array<u32, ${L}>;
var<workgroup> improves: u32;`,body:`let lane = localInvocationIndex;
  var sum = 0.0;
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${L}u) {
    sum += partialSums[partialSumsOffset + chunk];
    count += partialCounts[partialCountsOffset + chunk];
  }
  laneSums[lane] = sum;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${L/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let total = select(getQuietNaN(0u), laneSums[0], laneCounts[0] > 0u);${T?`
    let previous = inertia[inertiaOffset];
    let firstRestart = seedCursor[seedCursorOffset] == 0u;
    // Strictly lower wins; a finite score also replaces a NaN one.
    let better = firstRestart || total < previous || (!isFiniteFloat(previous) && isFiniteFloat(total));
    improves = select(0u, 1u, better);
    if (better) {
      inertia[inertiaOffset] = total;
      bestState[bestStateOffset] = state[stateOffset];
      bestState[bestStateOffset + 1u] = state[stateOffset + 1u];
    }`:`
    inertia[inertiaOffset] = total;`}
  }${T?`
  if (workgroupUniformLoad(&improves) != 0u) {
    for (var slot = lane; slot < ${2*p}u; slot += ${L}u) {
      bestCenters[bestCentersOffset + slot] = centers[centersOffset + slot];
    }
  }`:``}`}))};for(let t=0;t<w;t++){let n=T?`${l}-restart-${t}`:l;$.push(c(e,{id:`${n}-begin`,operation:F,variant:`restart-begin`,bindings:[{name:`state`,view:V,type:`u32`,access:`read_write`},{name:`seedCursor`,view:H,type:`u32`,access:`read_write`},Z.binding],invocationCount:1,body:`state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
  seedCursor[seedCursorOffset] = seedCursor[seedCursorOffset] + 1u;
  seedCursor[seedCursorOffset + 1u] = 0xffffffffu;
  ${Z.armWGSL}`})),ge(n),_e(n),T&&($.push(he(`${n}-score-assign`,!0,R,B)),be(`${n}-score`))}let xe=T?pe:R;(T||D)&&$.push(c(e,{id:`${l}-publish-centers`,operation:F,variant:D?`publish-centers-absolute`:`publish-centers`,bindings:[{name:`bestCenters`,view:xe,type:`f32`,access:`read`},...k,{name:`centers`,view:S,type:`f32`,access:`read_write`}],invocationCount:2*p,body:`centers[centersOffset + index] = bestCenters[bestCentersOffset + index]${D?` + origin[originOffset + index % 2u]`:``};`})),$.push(he(`${l}-final-assign`,!0,D?xe:S,B),...new o({id:`${l}-final-sizes`,keys:x,output:P}).getCommandNodes(e)),b.inertia&&!T&&be(`${l}-inertia`);for(let[t,n,r]of[[`iteration-count`,b.status?.iterationCount,1],[`converged`,b.status?.converged,0]])n&&$.push(c(e,{id:`${l}-publish-${t}`,operation:F,variant:`publish-state-word`,bindings:[{name:`state`,view:T?me:V,type:`u32`,access:`read`},{name:`stateWord`,view:n,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const WORD: u32 = ${r}u;`,body:`stateWord[stateWordOffset] = state[stateOffset + WORD];`}));return $}};function W(e){return[e.labels,e.centers,e.sizes,e.squaredDistances,e.inertia,e.status?.iterationCount,e.status?.converged]}var G=w,K=5,q=[1,2,3,4,6,12],J=520,oe=1800,Y=async e=>{let{device:t}=e;O(t);let n=D(e.theme),r=ne(G,e.parameters),i=e.width/e.height,a={worldWidth:i,worldHeight:1,width:e.width,height:e.height},o=T(e.seed),{positions:s,count:c}=b(o,{width:i,height:1},820),l=N(t,s),u=E(t,2),d=E(t,c),f=E(t,10),ee=E(t,K),p=e=>A.create(t,`demo-k-means-${e}`,t=>{t.add(new U({id:`demo-k-means-${e}`,positions:j(t,`positions`,l,`float32x2`,c),k:K,maximumIterations:e,parameters:j(t,`parameters`,u,`uint32`,2),output:{labels:j(t,`labels`,d,`uint32`,c),centers:j(t,`centers`,f,`float32x2`,K)}}))}),m=new Map,te=new Set;for(let e of q)m.set(e,await p(e));let h=!1,g=C(t,i),_=new M(t,l,d,c,`uint32`),v=new M(t,f,ee,K),y=E(t,c*2),w=E(t,c),P=new M(t,y,w,c);P.setInstanceCount(0);let F=x(s,c),I=null,L=null,ie=0,ae=async()=>{let e=++ie,[t,n]=await Promise.all([d.readAsync(),f.readAsync()]);h||e!==ie||(I=new Uint32Array(t.buffer,t.byteOffset,c).slice(),L=new Float32Array(n.buffer,n.byteOffset,10).slice(),Z.refresh(),Y.request())},R=q.length*J+oe,z=``,B=!1,V=i=>{let o=Math.round(r.iterations),s=11+Math.round(r.seed)*7919;if(!B){let t=e.reducedMotion?R-1:i,n=Math.floor(t/R),r=t-n*R;o=q[Math.min(q.length-1,Math.floor(r/J))],s=11+n*7919}let c=m.get(o),l=`${o}:${s}:${+!!c}`;l!==z&&(c?(z=l,u.write(H({seed:s})),c.run(),ae()):te.has(o)||(te.add(o),p(o).then(e=>{h?e.destroy():(m.set(o,e),Y.request())}))),_.update(a,n,{mode:2,radius:3.2,alpha:.8}),v.update(a,n,{mode:-1,radius:9,solid:[...n.ink,1],hillshade:1}),g.layer.update(a,n,{radius:.7,solid:[...n.muted,.8]}),P.update(a,n,{mode:-1,radius:4.4,solid:[...n.ink,1],hillshade:1}),k(t,n,[g.layer,_,P,v,...Z.getLayers()])},W=0,Y=new re(()=>V(W)),X=new Float32Array(c*2),Z=S({device:t,view:a,theme:n,reducedMotion:e.reducedMotion,kind:`probe`,hint:`hover an epicenter`,rest:[.5,.5],reportReadout:e.reportReadout,requestDraw:()=>Y.request(),resolve(e,t){if(!I||!L)return P.setInstanceCount(0),{};let n=F.nearest(e.x,e.y);if(n<0)return{};let r=I[n],i=0;for(let e=0;e<c;e++)I[e]===r&&(X[i*2]=s[e*2],X[i*2+1]=s[e*2+1],i++);y.write(X),P.setInstanceCount(i),Y.request();let a=[L[r*2],L[r*2+1]],o=[s[n*2],s[n*2+1]];return{glyphs:[{shape:`line`,from:o,to:a},{shape:`ring`,at:o,radius:t.pixelSize*7},{shape:`crosshair`,at:a}],readout:`cluster ${r+1} of ${K} · ${i} quakes · ${Math.hypot(o[0]-a[0],o[1]-a[1]).toFixed(2)} from its center`}}});return{gesture:Z.gesture,render(e){W=e,Z.advance(e),Y.flush()},setParameters(e){B=!0,Object.assign(r,e),Y.request()},resize(e,t){a.width=e,a.height=t,Z.refresh(),Y.request()},setPointer:Z.setPointer,resetPointer:Z.resetPointer,destroy(){h=!0,Y.cancel(),Z.destroy(),P.destroy(),g.destroy(),_.destroy(),v.destroy();for(let e of m.values())e.destroy();for(let e of[l,u,d,f,ee,y,w])e.destroy()}}};export{Y as default,G as parameters};