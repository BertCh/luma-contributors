import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as t}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{r as n}from"./kernel-C3tfQOn4.js";import{c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,f as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{t as l}from"./regression-output-dispatches-KKVTWkHP.js";import{c as u,r as d}from"./wgsl-common-DfyXiYB0.js";import{v as f}from"./spatial-context-DwZEd4EZ.js";import{t as p}from"./philox-random-DfVxWNPn.js";var m=8,h=16,g=1024,_=64,v=64;function y(e){return!e||e.mode===`groups`?[]:e.mode===`environmental`?[e.features]:[e.predictionPositions,...e.mode===`knndm`&&e.candidates?[e.candidates]:[]]}function b(e,t){let n=t.design;if(!n)return;let r=t.positions.length;if(o(e,`design rows`,r,2,256),n.mode===`environmental`&&(o(e,`featureCount`,n.featureCount,1,m),o(e,`cluster count`,t.foldCount,2,h),o(e,`maximumIterations`,n.maximumIterations??100,1,g),a(n.features,[`float32`],e),n.features.length!==r*n.featureCount))throw Error(`${e}: feature shape mismatch`);if(n.mode===`nndm`||n.mode===`knndm`){if(a(n.predictionPositions,[`float32x2`],e),o(e,`prediction rows`,n.predictionPositions.length,1,256),n.mode===`nndm`){if(r>_||t.foldCount!==r||!Number.isFinite(n.maximumDistance)||n.maximumDistance<0||n.maximumDistance>0xde0b6b3a7640000||!(n.minimumTrainingFraction===void 0||n.minimumTrainingFraction>0&&n.minimumTrainingFraction<1))throw Error(`${e}: invalid NNDM bounds`)}else if(o(e,`candidateCount`,n.candidateCount??8,1,v),n.candidates||o(e,`generated cluster count`,t.foldCount,2,h),n.candidates&&a(n.candidates,[`uint32`],e),n.candidates&&n.candidates.length!==r*(n.candidateCount??8)||!((n.maximumFoldFraction??.8)>1/t.foldCount&&(n.maximumFoldFraction??.8)<1))throw Error(`${e}: invalid kNNDM candidates or balance limit`)}if(n.diagnostics&&(a(n.diagnostics,[`float32`],e),n.diagnostics.length!==4+(n.mode===`nndm`?r:0)))throw Error(`${e}: diagnostics shape mismatch`);n.diagnostics&&l(e,{...t.output,diagnostics:n.diagnostics},[]),c(e,[...Object.values(t.output),n.diagnostics],[...y(n),t.positions,n.mode===`groups`?t.blocks:void 0,t.mask])}function x(e,t,a){let o=t.design,c=t.positions.length;if(o.mode===`knndm`&&!o.candidates){let r=o.candidateCount??8,i=n(e,`${a}-generated-candidates`,`uint32`,c*r),s=e.createDataView(t.positions.buffer,{format:`float32`,length:c*2,byteOffset:t.positions.byteOffset}),l=[];for(let n=0;n<r;n++){let r=e.createDataView(i.buffer,{format:`uint32`,length:c,byteOffset:4*n*c});l.push(...x(e,{...t,seed:(t.seed??0)+n*2654435761>>>0,design:{mode:`environmental`,features:s,featureCount:2,standardize:!1},output:{...t.output,folds:r}},`${a}-candidate-${n}`).nodes)}let u=x(e,{...t,design:{...o,candidates:i,candidateCount:r}},`${a}-match`);return{state:u.state,nodes:[...l,...u.nodes]}}let l=o.diagnostics??n(e,`${a}-design-state`,`float32`,4+(o.mode===`nndm`?c:0));s(a,e,[...y(o),l]);let f=`${d}\n${u}
fn location(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset+2u*row], positions[positionsOffset+2u*row+1u]); }
fn valid(row: u32) -> bool { let point=location(row); return isFiniteBits(point.x)&&isFiniteBits(point.y)${t.mask?`&&mask[maskOffset+row]!=0u`:``}; }
fn rowDistance(first:u32,second:u32)->f32 { return distance(location(first),location(second)); }`,p=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.mask?[{name:`mask`,view:t.mask,type:`u32`,access:`read`}]:[],{name:`folds`,view:t.output.folds,type:`u32`,access:`read_write`},{name:`state`,view:l,type:`f32`,access:`read_write`}],m=`state[stateOffset]=0.0; state[stateOffset+1u]=0.0; state[stateOffset+2u]=getNaN(); state[stateOffset+3u]=getNaN(); for(var row=0u;row<${c}u;row++){folds[foldsOffset+row]=0xffffffffu;}`,h=``;if(o.mode===`groups`)p.push({name:`groups`,view:t.blocks,type:`u32`,access:`read`}),m+=`var ordered:array<u32,${c}>;var count=0u;
for(var row=0u;row<${c}u;row++){if(!valid(row)){continue;}let group=groups[groupsOffset+row];var index=count;loop{if(index==0u){break;}if(ordered[index-1u]<=group){break;}ordered[index]=ordered[index-1u];index--;}ordered[index]=group;count++;}
for(var row=0u;row<${c}u;row++){if(!valid(row)){continue;}let group=groups[groupsOffset+row];var rank=0u;for(var other=0u;other<count;other++){if(ordered[other]>=group){break;}if(other==0u){rank++;}else if(ordered[other]!=ordered[other-1u]){rank++;}}folds[foldsOffset+row]=rank%${t.foldCount}u;}`;else if(o.mode===`environmental`){let e=o.featureCount,n=t.foldCount;p.push({name:`features`,view:o.features,type:`f32`,access:`read`}),h=`fn complete(row:u32)->bool{if(!valid(row)){return false;}for(var column=0u;column<${e}u;column++){if(!isFiniteBits(features[featuresOffset+row*${e}u+column])){return false;}}return true;}`,m+=`var means:array<f32,${e}>;var scales:array<f32,${e}>;var centers:array<f32,${n*e}>;var assignments:array<u32,${c}>;var sampleCount=0u;
for(var row=0u;row<${c}u;row++){assignments[row]=0xffffffffu;if(!complete(row)){continue;}sampleCount++;for(var column=0u;column<${e}u;column++){let value=features[featuresOffset+row*${e}u+column];let difference=value-means[column];means[column]+=difference/f32(sampleCount);scales[column]+=difference*(value-means[column]);}}
if(sampleCount<${n}u){state[stateOffset]=1.0;return;}
for(var column=0u;column<${e}u;column++){scales[column]=${o.standardize===!1?`1.0;means[column]=0.0`:`select(1.0,sqrt(scales[column]/f32(sampleCount-1u)),scales[column]>0.0)`};}
var seedRow=${(t.seed??0)%c}u;for(var offset=0u;offset<${c}u;offset++){let row=(seedRow+offset)%${c}u;if(complete(row)){seedRow=row;break;}}
for(var cluster=0u;cluster<${n}u;cluster++){
 if(cluster>0u){var farthest=-1.0;for(var row=0u;row<${c}u;row++){if(!complete(row)){continue;}var nearest=3.402823e38;for(var previous=0u;previous<cluster;previous++){var squared=0.0;for(var column=0u;column<${e}u;column++){let difference=(features[featuresOffset+row*${e}u+column]-means[column])/scales[column]-centers[previous*${e}u+column];squared+=difference*difference;}if(!isFiniteBits(squared)){state[stateOffset]=1.0;return;}nearest=min(nearest,squared);}if(nearest>farthest){farthest=nearest;seedRow=row;}}if(!(farthest>0.0)){state[stateOffset]=2.0;return;}}
 for(var column=0u;column<${e}u;column++){centers[cluster*${e}u+column]=(features[featuresOffset+seedRow*${e}u+column]-means[column])/scales[column];}}
var converged=false;var objective=0.0;
for(var iteration=0u;iteration<${o.maximumIterations??100}u;iteration++){var sums:array<f32,${n*e}>;var counts:array<u32,${n}>;var changed=false;objective=0.0;
for(var row=0u;row<${c}u;row++){if(!complete(row)){continue;}var nearest=3.402823e38;var selected=0u;for(var cluster=0u;cluster<${n}u;cluster++){var squared=0.0;for(var column=0u;column<${e}u;column++){let difference=(features[featuresOffset+row*${e}u+column]-means[column])/scales[column]-centers[cluster*${e}u+column];squared+=difference*difference;}if(!isFiniteBits(squared)){state[stateOffset]=1.0;return;}if(squared<nearest){nearest=squared;selected=cluster;}}
changed=changed||assignments[row]!=selected;assignments[row]=selected;objective+=nearest;counts[selected]++;for(var column=0u;column<${e}u;column++){sums[selected*${e}u+column]+=(features[featuresOffset+row*${e}u+column]-means[column])/scales[column];}}
for(var cluster=0u;cluster<${n}u;cluster++){if(counts[cluster]==0u){state[stateOffset]=2.0;return;}for(var column=0u;column<${e}u;column++){centers[cluster*${e}u+column]=sums[cluster*${e}u+column]/f32(counts[cluster]);}}
state[stateOffset+1u]=f32(iteration+1u);if(!changed){converged=true;break;}}
if(!isFiniteBits(objective)){state[stateOffset]=1.0;return;}state[stateOffset]=select(3.0,0.0,converged);state[stateOffset+2u]=objective;if(converged){for(var row=0u;row<${c}u;row++){folds[foldsOffset+row]=assignments[row];}}`}else{let i=o.predictionPositions.length;if(p.push({name:`prediction`,view:o.predictionPositions,type:`f32`,access:`read`}),m+=`var predictionDistances:array<f32,${i}>; for(var row=0u;row<${c}u;row++){if(!valid(row)){state[stateOffset]=1.0;return;}}
for(var sample=0u;sample<${i}u;sample++){let point=vec2<f32>(prediction[predictionOffset+sample*2u],prediction[predictionOffset+sample*2u+1u]);if(!isFiniteBits(point.x)||!isFiniteBits(point.y)){state[stateOffset]=1.0;return;}var nearest=3.402823e38;for(var row=0u;row<${c}u;row++){let separation=distance(point,location(row));if(!isFiniteBits(separation)){state[stateOffset]=1.0;return;}nearest=min(nearest,separation);}predictionDistances[sample]=nearest;}`,o.mode===`nndm`){let t=n(e,`${a}-nndm-distances`,`float32`,c*c);p.push({name:`distances`,view:t,type:`f32`,access:`read_write`}),m+=`var nearest:array<f32,${c}>;var neighbors:array<u32,${c}>;var remaining:array<u32,${c}>;
for(var row=0u;row<${c}u;row++){nearest[row]=3.402823e38;remaining[row]=${c-1}u;for(var other=0u;other<${c}u;other++){let value=select(rowDistance(row,other),-1.0,row==other);if(!isFiniteBits(value)){state[stateOffset]=1.0;return;}distances[distancesOffset+row*${c}u+other]=value;if(value>=0.0&&value<nearest[row]){nearest[row]=value;neighbors[row]=other;}}}
var radius=3.402823e38;var chosen=0u;for(var row=0u;row<${c}u;row++){if(nearest[row]<radius){radius=nearest[row];chosen=row;}}
for(var iteration=0u;iteration<${c*c+c}u&&radius<=${r(o.maximumDistance)};iteration++){
var trainingBelow=0u;var predictionBelow=0u;for(var row=0u;row<${c}u;row++){trainingBelow+=u32(nearest[row]<=radius);}for(var sample=0u;sample<${i}u;sample++){predictionBelow+=u32(predictionDistances[sample]<=radius);}
let remove=f32(trainingBelow-1u)/${r(c)}>=f32(predictionBelow)/${r(i)}&&f32(remaining[chosen])/${r(c)}>${r(o.minimumTrainingFraction??.05)};
if(remove){distances[distancesOffset+chosen*${c}u+neighbors[chosen]]=-1.0;remaining[chosen]--;nearest[chosen]=3.402823e38;for(var other=0u;other<${c}u;other++){let value=distances[distancesOffset+chosen*${c}u+other];if(value>=0.0&&value<nearest[chosen]){nearest[chosen]=value;neighbors[chosen]=other;}}}else if(trainingBelow==${c}u){break;}
var next=3.402823e38;for(var row=0u;row<${c}u;row++){if(select(nearest[row]>radius,nearest[row]>=radius,remove)&&nearest[row]<next){next=nearest[row];chosen=row;}}radius=next;state[stateOffset+1u]=f32(iteration+1u);}
for(var row=0u;row<${c}u;row++){folds[foldsOffset+row]=row;state[stateOffset+row+4u]=nearest[row];}`}else p.push({name:`candidates`,view:o.candidates,type:`u32`,access:`read`}),m+=`var nearest:array<f32,${c}>;var best=3.402823e38;var selected=0xffffffffu;
for(var candidate=0u;candidate<${o.candidateCount??8}u;candidate++){var counts:array<u32,${t.foldCount}>;var validCandidate=true;for(var row=0u;row<${c}u;row++){let fold=candidates[candidatesOffset+candidate*${c}u+row];if(fold>=${t.foldCount}u){validCandidate=false;break;}counts[fold]++;var value=3.402823e38;for(var other=0u;other<${c}u;other++){if(candidates[candidatesOffset+candidate*${c}u+other]!=fold){let separation=rowDistance(row,other);if(!isFiniteBits(separation)){state[stateOffset]=1.0;return;}value=min(value,separation);}}nearest[row]=value;}
for(var fold=0u;fold<${t.foldCount}u;fold++){validCandidate=validCandidate&&counts[fold]>0u&&f32(counts[fold])<=${r((o.maximumFoldFraction??.8)*c)};}if(!validCandidate){continue;}
var previous=0.0;var objective=0.0;for(var step=0u;step<${c+i}u;step++){var next=3.402823e38;var trainBelow=0u;var predictionBelow=0u;for(var row=0u;row<${c}u;row++){trainBelow+=u32(nearest[row]<=previous);if(nearest[row]>previous){next=min(next,nearest[row]);}}for(var sample=0u;sample<${i}u;sample++){predictionBelow+=u32(predictionDistances[sample]<=previous);if(predictionDistances[sample]>previous){next=min(next,predictionDistances[sample]);}}if(next==3.402823e38){break;}objective+=abs(f32(trainBelow)/${r(c)}-f32(predictionBelow)/${r(i)})*(next-previous);previous=next;}
if(objective<best){best=objective;selected=candidate;}}
state[stateOffset+1u]=${r(o.candidateCount??8)};if(selected==0xffffffffu){state[stateOffset]=2.0;return;}state[stateOffset+2u]=best;state[stateOffset+3u]=f32(selected);for(var row=0u;row<${c}u;row++){folds[foldsOffset+row]=candidates[candidatesOffset+selected*${c}u+row];}`}return{state:l,nodes:[i(e,{id:`${a}-design`,operation:`GPUSpatialCrossValidation`,variant:o.mode,invocationCount:1,bindings:p,declarations:`${f}\n${h}`,body:m})]}}function S(e){return{positions:e.positions,blocks:!e.design||e.design.mode===`groups`?e.blocks:void 0,mask:e.mask,...e.output,...Object.fromEntries(y(e.design).map((e,t)=>[`design input ${t}`,e])),designDiagnostics:e.design?.diagnostics}}function C(e){let t=e.positions.length,n=e.design;return n?{"design state":4+(n.mode===`nndm`?t:0),...n.mode===`nndm`?{"NNDM distances":t*t}:{},...n.mode===`knndm`&&!n.candidates?{"generated kNNDM candidates":t*(n.candidateCount??8)}:{}}:{"block order":2*t}}function w(n,r,i){e(n,r,S(i)),t(n.device,r,C(i))}var T=65536,E=64,D=class{props;id;spatialContext;constructor(e){this.props=e,this.id=e.id??`spatial-cross-validation`,this.spatialContext=f(this.id,e.spatialContext);let t=e.positions.length;if(b(this.id,e),o(this.id,`rows`,t,1,T),o(this.id,`foldCount`,e.foldCount,2,E),!Number.isInteger(e.seed??0)||(e.seed??0)<0||(e.seed??0)>4294967295||!Number.isFinite(e.bufferDistance??0)||(e.bufferDistance??0)<0||(e.bufferDistance??0)>0xde0b6b3a7640000)throw Error(`${this.id}: invalid seed or distance`);for(let[n,r,i]of[[e.positions,`float32x2`,t],[e.blocks,`uint32`,t],[e.mask,`uint32`,t],[e.output.folds,`uint32`,t],[e.output.training,`uint32`,t*e.foldCount],[e.output.testing,`uint32`,t*e.foldCount],[e.output.counts,`uint32`,3*e.foldCount],[e.output.status,`uint32`,e.foldCount]])if(n&&(a(n,[r],this.id),n.length!==i))throw Error(`${this.id}: shape mismatch`);l(this.id,{...e.output,diagnostics:e.design?.diagnostics},[[`folds`,`diagnostics`],{read:[`folds`,`diagnostics`],write:[`training`,`testing`]},{read:[`folds`,`training`,`testing`],write:[`counts`,`status`]}]),c(this.id,Object.values(e.output),[e.positions,...!e.design||e.design.mode===`groups`?[e.blocks]:[],...e.mask?[e.mask]:[]])}getCommandNodes(e){let{props:t,id:a}=this;w(e,a,t);let o=t.positions.length;s(a,e,[t.positions,!t.design||t.design.mode===`groups`?t.blocks:void 0,t.mask,...Object.values(t.output)]);let c=t.design?x(e,t,a):void 0,l=c?void 0:n(e,`${a}-block-order`,`uint32`,2*o),d=`${u}
fn isUsableRow(row: u32) -> bool {
  return isFiniteBits(positions[positionsOffset + row * 2u]) &&
    isFiniteBits(positions[positionsOffset + row * 2u + 1u])${t.mask?` && mask[maskOffset + row] != 0u`:``};
}`,f=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...t.mask?[{name:`mask`,view:t.mask,type:`u32`,access:`read`}]:[]];return[...c?.nodes??[i(e,{id:`${a}-block-order`,operation:`GPUSpatialCrossValidation`,variant:`block-order`,invocationCount:o,bindings:[...f,{name:`blocks`,view:t.blocks,type:`u32`,access:`read`},{name:`blockOrder`,view:l,type:`u32`,access:`read_write`}],declarations:`${d}
fn philoxMulHiLo(a: u32, b: u32) -> vec2<u32> {
  let aLow = a & 0xffffu;
  let aHigh = a >> 16u;
  let bLow = b & 0xffffu;
  let bHigh = b >> 16u;
  let lowLow = aLow * bLow;
  let highLow = aHigh * bLow;
  let lowHigh = aLow * bHigh;
  let highHigh = aHigh * bHigh;
  let middle = (lowLow >> 16u) + (highLow & 0xffffu) + (lowHigh & 0xffffu);
  let high = highHigh + (highLow >> 16u) + (lowHigh >> 16u) + (middle >> 16u);
  return vec2<u32>(high, a * b);
}

fn philoxRound(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  let product0 = philoxMulHiLo(3528531795u, counter.x);
  let product1 = philoxMulHiLo(3449720151u, counter.z);
  return vec4<u32>(
    product1.x ^ counter.y ^ key.x,
    product1.y,
    product0.x ^ counter.w ^ key.y,
    product0.y
  );
}

fn philox4x32(counterIn: vec4<u32>, keyIn: vec2<u32>) -> vec4<u32> {
  var counter = counterIn;
  var key = keyIn;
  for (var roundIndex = 0u; roundIndex < 9u; roundIndex = roundIndex + 1u) {
    counter = philoxRound(counter, key);
    key = key + vec2<u32>(2654435769u, 3144134277u);
  }
  return philoxRound(counter, key);
}

fn philoxUnitFloat(word: u32) -> f32 {
  return f32(word >> 8u) * (1.0 / 16777216.0);
}

// Uniform in the open interval (0, 1), for log() and inverse-CDF draws. The top 23 bits plus one
// half are exact in f32, so the result lies in [2^-24, 1 - 2^-24] and neither end is reachable.
// (The 24-bit form (f32(word >> 8u) + 0.5) / 16777216.0 rounds to exactly 1.0 for the largest word.)
fn philoxOpenUnitFloat(word: u32) -> f32 {
  return (f32(word >> 9u) + 0.5) * (1.0 / 8388608.0);
}
`,body:`var key = 0u;
var representative = 0u;
if (isUsableRow(index)) {
  let blockId = blocks[blocksOffset + index];
  let random = philox4x32(vec4<u32>(blockId, 0u, 0u, 0u), vec2<u32>(${t.seed??0}u, ${p.spatialCrossValidationBlockOrder.id}u));
  key = random.x;
  representative = 1u;
  for (var other = 0u; other < index; other++) {
    if (blocks[blocksOffset + other] == blockId && isUsableRow(other)) { representative = 0u; break; }
  }
}
blockOrder[blockOrderOffset + 2u * index] = key;
blockOrder[blockOrderOffset + 2u * index + 1u] = representative;`}),i(e,{id:`${a}-assign`,operation:`GPUSpatialCrossValidation`,variant:`assign`,invocationCount:o,bindings:[...f,{name:`blocks`,view:t.blocks,type:`u32`,access:`read`},{name:`blockOrder`,view:l,type:`u32`,access:`read`},{name:`folds`,view:t.output.folds,type:`u32`,access:`read_write`}],declarations:d,body:`var fold = 0xffffffffu;
if (isUsableRow(index)) {
  let blockId = blocks[blocksOffset + index];
  let key = blockOrder[blockOrderOffset + 2u * index];
  // Rank of this block among the blocks that hold a usable row, by (key, block id).
  var rank = 0u;
  for (var other = 0u; other < ${o}u; other++) {
    if (blockOrder[blockOrderOffset + 2u * other + 1u] != 0u) {
      let otherKey = blockOrder[blockOrderOffset + 2u * other];
      if (otherKey < key || (otherKey == key && blocks[blocksOffset + other] < blockId)) { rank++; }
    }
  }
  fold = rank % ${t.foldCount}u;
}
folds[foldsOffset + index] = fold;`})],i(e,{id:`${a}-masks`,operation:`GPUSpatialCrossValidation`,variant:`buffered-masks`,invocationCount:o*t.foldCount,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`folds`,view:t.output.folds,type:`u32`,access:`read`},...c?[{name:`designState`,view:c.state,type:`f32`,access:`read`}]:[],{name:`training`,view:t.output.training,type:`u32`,access:`read_write`},{name:`testing`,view:t.output.testing,type:`u32`,access:`read_write`}],body:`let fold = index / ${o}u;
let row = index % ${o}u;
let assigned = folds[foldsOffset + row];
let test = assigned == fold;
var train = assigned < ${t.foldCount}u && !test;
${c?`train = train && designState[designStateOffset] == 0.0;`:``}
if (train) {
  let position = vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  for (var other = 0u; other < ${o}u; other++) {
    if (folds[foldsOffset + other] == fold) {
      let difference = position - vec2<f32>(positions[positionsOffset + other * 2u], positions[positionsOffset + other * 2u + 1u]);
      if (${c&&t.design?.mode===`nndm`?`length(difference) <= designState[designStateOffset + fold + 4u]`:`dot(difference, difference) <= ${r((t.bufferDistance??0)**2)}`}) { train = false; break; }
    }
  }
}
training[trainingOffset + index] = select(0u, 1u, train);
testing[testingOffset + index] = select(0u, 1u, test);`}),i(e,{id:`${a}-counts`,operation:`GPUSpatialCrossValidation`,variant:`counts`,invocationCount:t.foldCount,bindings:[{name:`folds`,view:t.output.folds,type:`u32`,access:`read`},{name:`training`,view:t.output.training,type:`u32`,access:`read`},{name:`testing`,view:t.output.testing,type:`u32`,access:`read`},{name:`counts`,view:t.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:t.output.status,type:`u32`,access:`read_write`}],body:`var train = 0u; var test = 0u; var valid = 0u;
for (var row = 0u; row < ${o}u; row++) {
 train += training[trainingOffset + index * ${o}u + row]; test += testing[testingOffset + index * ${o}u + row]; valid += select(0u, 1u, folds[foldsOffset + row] < ${t.foldCount}u);
}
counts[countsOffset + 3u * index] = train;
counts[countsOffset + 3u * index + 1u] = test;
counts[countsOffset + 3u * index + 2u] = valid - train - test;
status[statusOffset + index] = select(0u, 1u, train == 0u) | select(0u, 2u, test == 0u);`})]}};export{w as n,y as r,D as t};