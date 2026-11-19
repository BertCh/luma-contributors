import{i as e,s as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as n}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{r,s as i}from"./kernel-C3tfQOn4.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,T as s,f as c,v as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{c as f,r as p}from"./wgsl-common-DfyXiYB0.js";import{t as m}from"./philox-random-DfVxWNPn.js";import{t as h}from"./permutation-random-SiwCiDNI.js";import{r as g,t as _}from"./gpu-multiple-testing-adjustment-Bghjpx33.js";import{i as v,s as y,t as b}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as x}from"./gpu-compaction-B6IzcF0P.js";import{t as S}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{t as C}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{t as w}from"./define-run-B-brkCe4.js";import{m as T}from"./inputs-CJKKIRKK.js";import{a as E,g as D,o as O,r as k,s as A}from"./point-field-CLZ0m0cx.js";import{kt as j}from"./entry-preview-D70TdSbK.js";function M(e,t){let n=t.seed??[0,0];s(e,n.length===2,`seed must contain exactly 2 uint32 words`),c(e,`seed[0]`,n[0],0,4294967295),c(e,`seed[1]`,n[1],0,4294967295);let r=t.significance??.05;s(e,r>0&&r<1,`significance must lie in (0, 1)`)}function N(e={},t=new Uint32Array(3)){let n=`getGPUMultivariateLocalGearyPermutationTestParameterValues`;c(n,`target.length`,t.length,3),M(n,e);let r=e.seed??[0,0],i=new Uint32Array(new Float32Array([e.significance??.05]).buffer)[0];return t.set([r[0],r[1],i]),t}var P=4096;function F(e,t){let{id:n,rows:i,columns:a,permutations:o,maximumNeighbors:s}=t,c=Math.ceil(i/P),l=r(e,`${n}-valid`,`uint32`,i),u=r(e,`${n}-row-ids`,`uint32`,i),d=r(e,`${n}-valid-rows`,`uint32`,i),g=r(e,`${n}-inference-state`,`uint32`,4),_=e.createDataView(g.buffer,{format:`uint32`,length:1}),v=r(e,`${n}-standardized`,`float32`,i*a),y=r(e,`${n}-moment-partials`,`float32`,3*a*c),w=r(e,`${n}-moments`,`float32`,2*a),T=r(e,`${n}-accumulators`,`float32`,4*i),E=(e,t,n=`read`,r=`f32`)=>({name:e,view:t,access:n,type:r}),D=(t,r,i,a,o=``,s)=>S(e,{id:`${n}-${t}`,operation:`GPUMultivariateLocalGearyPermutationTest`,variant:t,invocationCount:r,bindings:i,declarations:`${p}\n${f}
var<private> TILE_ROW_BASE: u32;
var<private> REPLICATE_BASE: u32;
var<private> REPLICATE_COUNT: u32;
var<private> WEIGHTS_ROW_START: u32;
var<private> CAPACITY: u32;
${o}`,parameterValues:[s?.TILE_ROW_BASE??0,s?.REPLICATE_BASE??0,s?.REPLICATE_COUNT??0,s?.WEIGHTS_ROW_START??0],body:`if (index >= INVOCATION_COUNT) { return; }
TILE_ROW_BASE = getChunkParameter(36u); REPLICATE_BASE = getChunkParameter(37u); REPLICATE_COUNT = getChunkParameter(38u); WEIGHTS_ROW_START = getChunkParameter(39u);
${i.some(e=>e.name===`neighbors`)?`CAPACITY = arrayLength(&neighbors) - neighborsOffset;`:``}
${a}`}),O=t.weights.status?.overflow?[E(`overflow`,t.weights.status.overflow,`read`,`u32`)]:[],k=b(t.weights),A=e=>`(row - WEIGHTS_ROW_START)`,j=[D(`parameters`,1,[E(`parameters`,t.parameters,`read`,`u32`),E(`inferenceState`,g,`read_write`,`u32`)],`inferenceState[inferenceStateOffset + 1u] = parameters[parametersOffset];
       inferenceState[inferenceStateOffset + 2u] = parameters[parametersOffset + 1u];
       inferenceState[inferenceStateOffset + 3u] = parameters[parametersOffset + 2u];`),D(`complete`,i,[E(`values`,t.values),...t.mask?[E(`mask`,t.mask,`read`,`u32`)]:[],E(`valid`,l,`read_write`,`u32`),E(`rowIdentifiers`,u,`read_write`,`u32`),...O],`var included = ${t.mask?`mask[maskOffset + index] != 0u`:`true`} ${t.weights.status?.overflow?`&& overflow[overflowOffset] == 0u`:``};
for (var column = 0u; column < ${a}u; column++) { included = included && isFiniteBits(values[valuesOffset + index * ${a}u + column]); }
valid[validOffset + index] = select(0u, 1u, included);
rowIdentifiers[rowIdentifiersOffset + index] = index;`),...new x({id:`${n}-compaction`,input:u,flags:l,output:d,count:_}).getCommandNodes(e),D(`moment-chunks`,a*c,[E(`values`,t.values),E(`valid`,l,`read`,`u32`),E(`partials`,y,`read_write`)],`let column = index / ${c}u; let chunk = index % ${c}u;
var count = 0.0; var mean = 0.0; var squares = 0.0;
let chunkStart = chunk * ${P}u; let chunkEnd = min(chunkStart + ${P}u, ${i}u);
for (var row = chunkStart; row < chunkEnd; row++) { if (valid[validOffset + row] != 0u) { count += 1.0; let value = values[valuesOffset + row * ${a}u + column]; let difference = value - mean; mean += difference / count; squares += difference * (value - mean); } }
let base = (column * ${c}u + chunk) * 3u;
partials[partialsOffset + base] = count; partials[partialsOffset + base + 1u] = mean; partials[partialsOffset + base + 2u] = squares;`),D(`moment-merge`,a,[E(`partials`,y),E(`moments`,w,`read_write`)],`var count = 0u; var mean = 0.0; var squares = 0.0;
for (var chunk = 0u; chunk < ${c}u; chunk++) {
 let base = (index * ${c}u + chunk) * 3u;
 let chunkCountValue = u32(partials[partialsOffset + base]);
 if (chunkCountValue > 0u) {
  let chunkMean = partials[partialsOffset + base + 1u]; let chunkSquares = partials[partialsOffset + base + 2u];
  if (count == 0u) { count = chunkCountValue; mean = chunkMean; squares = chunkSquares; continue; }
  let combined = count + chunkCountValue; let delta = chunkMean - mean;
  squares += chunkSquares + multiplyBySpatialCount(delta * delta * divideSpatialCounts(chunkCountValue, combined), count);
  mean += delta * divideSpatialCounts(chunkCountValue, combined); count = combined;
 }
}
var scale = getNaN();
if (count > 1u && squares > 0.0 && isFiniteBits(squares) && isFiniteBits(mean)) { scale = sqrt(divideSpatialCountByValue(count, squares)); }
moments[momentsOffset + index * 2u] = mean; moments[momentsOffset + index * 2u + 1u] = scale;`,C),D(`standardize`,i*a,[E(`values`,t.values),E(`valid`,l,`read`,`u32`),E(`moments`,w),E(`standardized`,v,`read_write`)],`let row = index / ${a}u; let column = index % ${a}u;
var value = getNaN();
if (valid[validOffset + row] != 0u) { value = (values[valuesOffset + index] - moments[momentsOffset + column * 2u]) * moments[momentsOffset + column * 2u + 1u]; }
standardized[standardizedOffset + index] = value;`)],M=`${h}
const ROWS: u32 = ${i}u;
const COLUMNS: u32 = ${a}u;
const MAXIMUM_NEIGHBORS: u32 = ${s}u;
const PAIR_CAPACITY: u32 = ${2*s}u;

fn isRowValid(row: u32) -> bool { return isFiniteBits(standardized[standardizedOffset + row * COLUMNS]); }
fn getVectorDistance(row: u32, candidate: u32) -> f32 {
 var distance = 0.0;
 for (var column = 0u; column < COLUMNS; column++) { let difference = standardized[standardizedOffset + row * COLUMNS + column] - standardized[standardizedOffset + candidate * COLUMNS + column]; distance += difference * difference; }
 return distance;
}`,N=e=>M;j.push(...k.flatMap(e=>I(e.rowCount,t.rowTileSize).map(n=>{let r=e.rowStart+n;return D(`observed-${r}`,Math.min(t.rowTileSize,e.rowCount-n),[E(`standardized`,v),E(`validCount`,g,`read`,`u32`),E(`offsets`,e.weights.offsets,`read`,`u32`),E(`neighbors`,e.weights.neighbors,`read`,`u32`),E(`weights`,e.weights.weights),E(`observed`,t.output.statistic,`read_write`),E(`accumulators`,T,`read_write`)],`{ let row = index + TILE_ROW_BASE;
for (var slotField = 0u; slotField < 4u; slotField++) { accumulators[accumulatorsOffset + slotField * ROWS + row] = 0.0; }
var value = getNaN();
if (isRowValid(row) && validCount[validCountOffset] > 1u) {
 var total = 0.0; var weightSum = 0.0;
 for (var slot = offsets[offsetsOffset + ${A(e)}]; slot < min(offsets[offsetsOffset + ${A(e)} + 1u], CAPACITY); slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  if (neighbor >= ROWS || neighbor == row || !(weight > 0.0)) { continue; }
  if (!isRowValid(neighbor)) { continue; }
  total += weight * getVectorDistance(row, neighbor) / ${a}.0; weightSum += weight;
 }
 if (weightSum > 0.0) { value = total / weightSum; }
}
observed[observedOffset + row] = value; }`,N(e),{TILE_ROW_BASE:r,WEIGHTS_ROW_START:e.rowStart})})));let F=`${M}

var<private> pairPositions: array<u32, PAIR_CAPACITY>;
var<private> pairValues: array<u32, PAIR_CAPACITY>;
var<private> pairCount: u32;
fn getVirtualCandidate(position: u32, rank: u32) -> u32 {
 for (var pair = 0u; pair < pairCount; pair++) { if (pairPositions[pair] == position) { return pairValues[pair]; } }
 return validRows[validRowsOffset + select(position + 1u, position, position < rank)];
}
fn setVirtualCandidate(position: u32, value: u32) {
 for (var pair = 0u; pair < pairCount; pair++) { if (pairPositions[pair] == position) { pairValues[pair] = value; return; } }
 pairPositions[pairCount] = position; pairValues[pairCount] = value; pairCount++;
}`;for(let e of k)for(let n of I(e.rowCount,t.rowTileSize)){let r=e.rowStart+n;for(let i of I(o,t.permutationTileSize))j.push(D(`simulate-${r}-${i}`,Math.min(t.rowTileSize,e.rowCount-n),[E(`standardized`,v),E(`validRows`,d,`read`,`u32`),E(`validCount`,g,`read`,`u32`),E(`offsets`,e.weights.offsets,`read`,`u32`),E(`neighbors`,e.weights.neighbors,`read`,`u32`),E(`weights`,e.weights.weights),E(`observed`,t.output.statistic),E(`accumulators`,T,`read_write`)],`{ let row = index + TILE_ROW_BASE;
if (isRowValid(row) && validCount[validCountOffset] > 1u) {
 let observedValue = observed[observedOffset + row];
 let othersCount = validCount[validCountOffset] - 1u;
 var rankLow = 0u; var rankHigh = validCount[validCountOffset];
 while (rankLow < rankHigh) { let middle = (rankLow + rankHigh) / 2u; if (validRows[validRowsOffset + middle] < row) { rankLow = middle + 1u; } else { rankHigh = middle; } }
 let rank = rankLow;
 var lower = accumulators[accumulatorsOffset + row]; var upper = accumulators[accumulatorsOffset + ROWS + row];
 var invalid = accumulators[accumulatorsOffset + 2u * ROWS + row]; var average = accumulators[accumulatorsOffset + 3u * ROWS + row];
 for (var tileStep = 0u; tileStep < REPLICATE_COUNT; tileStep++) {
  let replicate = REPLICATE_BASE + tileStep + 1u;
  var stream = createPhiloxStream(vec2<u32>(validCount[validCountOffset + 1u], validCount[validCountOffset + 2u]), row, replicate, ${m.localPermutationTest.id}u);
  pairCount = 0u;
  var chosen = 0u; var total = 0.0; var weightSum = 0.0; var overflowed = false;
  for (var slot = offsets[offsetsOffset + ${A(e)}]; slot < min(offsets[offsetsOffset + ${A(e)} + 1u], CAPACITY); slot++) {
   let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
   if (neighbor >= ROWS || neighbor == row || !(weight > 0.0)) { continue; }
   if (!isRowValid(neighbor)) { continue; }
   if (chosen >= MAXIMUM_NEIGHBORS || chosen >= othersCount) { overflowed = true; break; }
   let selection = chosen + nextPhiloxBelow(&stream, othersCount - chosen);
   let candidate = getVirtualCandidate(selection, rank);
   setVirtualCandidate(selection, getVirtualCandidate(chosen, rank));
   setVirtualCandidate(chosen, candidate);
   chosen++;
   total += weight * getVectorDistance(row, candidate) / ${a}.0; weightSum += weight;
  }
  var value = getNaN();
  if (!overflowed && weightSum > 0.0) { value = total / weightSum; }
  invalid += select(0.0, 1.0, !isFiniteBits(value));
  lower += select(0.0, 1.0, value <= observedValue); upper += select(0.0, 1.0, value >= observedValue);
  average += value / ${o}.0;
 }
 accumulators[accumulatorsOffset + row] = lower; accumulators[accumulatorsOffset + ROWS + row] = upper;
 accumulators[accumulatorsOffset + 2u * ROWS + row] = invalid; accumulators[accumulatorsOffset + 3u * ROWS + row] = average;
}
}`,F,{TILE_ROW_BASE:r,WEIGHTS_ROW_START:e.rowStart,REPLICATE_BASE:i,REPLICATE_COUNT:Math.min(t.permutationTileSize,o-i)}))}return j.push(D(`inference`,i,[E(`observed`,t.output.statistic),E(`accumulators`,T),E(`validCount`,g,`read`,`u32`),E(`pValues`,t.output.pValues,`read_write`),E(`classes`,t.output.classes,`read_write`,`u32`)],`let observedValue = observed[observedOffset + index];
let lower = u32(accumulators[accumulatorsOffset + index]); let upper = u32(accumulators[accumulatorsOffset + ${i}u + index]);
let average = accumulators[accumulatorsOffset + 3u * ${i}u + index];
let finite = isFiniteBits(observedValue) && accumulators[accumulatorsOffset + 2u * ${i}u + index] == 0.0;
var probability = getNaN(); var classification = 0xffffffffu;
if (finite) { probability = ${t.tail===`esda-min-tail`?`f32(min(upper, ${o}u - upper) + 1u) / ${o+1}.0`:`min(1.0, 2.0 * f32(min(lower, upper) + 1u) / ${o+1}.0)`};
${t.correction===`bonferroni`?`probability = min(1.0, probability * f32(validCount[validCountOffset]));`:`probability = min(1.0, probability);`}
classification = 0u;
if (probability <= ${t.classifyDirections?`0.999999`:`bitcast<f32>(validCount[validCountOffset + 3u])`}) { classification = select(2u, 1u, observedValue < average); }
}
pValues[pValuesOffset + index] = probability; classes[classesOffset + index] = classification;`)),j}function I(e,t){let n=[];for(let r=0;r<e;r+=Math.max(1,t))n.push(r);return n}var L=2**24-1,R=9999,z=class x{props;id;parameterValues;constructor(t){this.props=t,this.id=t.id??`multivariate-local-geary-permutation-test`,M(this.id,t),t.parameters?(o(t.parameters,[`uint32`],`${this.id} parameters`),c(this.id,`parameters.length`,t.parameters.length,3)):this.parameterValues=N(t);let n=y(this.id,t.weights),r=this.getStrategy(n)===`tiled`;if(c(this.id,`rows`,n,2,r?L:256),c(this.id,`variableCount`,t.variableCount,1,16),c(this.id,`maximumPermutations`,t.maximumPermutations??999,1,r?R:4096),r){if(t.output.simulations)throw Error(`${this.id}: tiled has no simulations output`);c(this.id,`maximumNeighbors`,t.maximumNeighbors??64,1,128),c(this.id,`rowTileSize`,t.rowTileSize??262144),c(this.id,`permutationTileSize`,t.permutationTileSize??128)}if(g(this.id,t.multipleTesting)!==`none`&&t.correction===`bonferroni`)throw Error(`${this.id}: use either correction or multipleTesting`);if(t.tail&&![`equal-tail`,`esda-min-tail`].includes(t.tail))throw Error(`${this.id}: invalid tail convention`);s(this.id,[`none`,`bonferroni`].includes(t.correction??`none`),`correction must be none or bonferroni`);for(let[e,r,i]of[[t.values,`float32`,n*t.variableCount],[t.mask,`uint32`,n],[t.output.statistic,`float32`,n],[t.output.pValues,`float32`,n],[t.output.classes,`uint32`,n],[t.output.simulations,`float32`,n*(t.maximumPermutations??999)]])if(e&&(o(e,[r],this.id),e.length!==i))throw Error(`${this.id}: shape mismatch`);e(this.id,t.output),d(this.id,Object.values(t.output),[t.values,t.mask,t.parameters,...v(t.weights),t.weights.status?.overflow])}_getRowCount(){let{weights:e}=this.props;return`chunks`in e?e.rowCount:e.offsets.length-1}getStrategy(e=this._getRowCount()){return this.props.strategy??(e>256||(this.props.maximumPermutations??999)>4096?`tiled`:`exact`)}getCommandNodes(e){n(e,this.id,this.props,{"row scratch":this.props.values.length*4,"tiled accumulators":this._getRowCount()*16,"moment partials":Math.ceil(this._getRowCount()/4096)*this.props.variableCount*12});let{props:o,id:s}=this,c=this._getRowCount();u(s,e,[o.values,o.mask,o.parameters,...v(o.weights),o.weights.status?.overflow,...Object.values(o.output)]),o.parameters&&l(e.device,s,`parameters binding`,i(o.parameters).size,`use a smaller packed parameter view`);let d=t(e,`${s}-parameters`,this.parameterValues??o.parameters),f=g(s,o.multipleTesting);if(f!==`none`){let t=r(e,`${s}-raw-p-values`,`float32`,c),n=r(e,`${s}-directions`,`uint32`,c),i=new x({...o,id:`${s}-uncorrected`,multipleTesting:void 0,correction:`none`,significance:.999999,parameters:void 0,output:{...o.output,pValues:t,classes:n}});return[...d.nodes,...i.getInferenceNodes(e,d.parameters,!0),...new _({id:`${s}-multiple-testing`,pValues:t,method:f,output:{adjustedPValues:o.output.pValues}}).getCommandNodes(e),a(e,{id:`${s}-classify-adjusted`,operation:`GPUMultivariateLocalGearyPermutationTest`,variant:`classify-adjusted`,invocationCount:c,bindings:[{name:`directions`,view:n,type:`u32`,access:`read`},{name:`adjusted`,view:o.output.pValues,type:`f32`,access:`read`},{name:`classes`,view:o.output.classes,type:`u32`,access:`read_write`},{name:`parameters`,view:d.parameters,type:`u32`,access:`read`}],body:`let direction = directions[directionsOffset + index];
  classes[classesOffset + index] = select(
    direction,
    select(0u, direction, adjusted[adjustedOffset + index] <= bitcast<f32>(parameters[parametersOffset + 2u])),
    direction != 0xffffffffu
  );`})]}return[...d.nodes,...this.getInferenceNodes(e,d.parameters)]}getInferenceNodes(e,t,n=!1){let{props:i,id:o}=this,s=this._getRowCount(),c=i.variableCount,l=i.maximumPermutations??999;if(this.getStrategy(s)===`tiled`)return u(o,e,[i.values,i.mask,t,...v(i.weights),i.weights.status?.overflow,...Object.values(i.output)]),F(e,{id:o,rows:s,columns:c,permutations:l,parameters:t,classifyDirections:n,correction:i.correction??`none`,tail:i.tail??`equal-tail`,rowTileSize:Math.min(i.rowTileSize??262144,s),permutationTileSize:Math.min(i.permutationTileSize??128,l),maximumNeighbors:i.maximumNeighbors??64,weights:i.weights,values:i.values,mask:i.mask,output:i.output});let d=r(e,`${o}-valid`,`uint32`,s),g=r(e,`${o}-standardized`,`float32`,s*c),_=i.output.simulations??r(e,`${o}-simulations`,`float32`,s*l),y=(e,t,n=`read`,r=`f32`)=>({name:e,view:t,access:n,type:r}),x=(t,n,r,i,s=``)=>a(e,{id:`${o}-${t}`,operation:`GPUMultivariateLocalGearyPermutationTest`,variant:t,invocationCount:n,bindings:r,declarations:`${p}\n${f}\n${s}`,body:i});u(o,e,[i.values,i.mask,t,...v(i.weights),i.weights.status?.overflow,...Object.values(i.output)]);let S=b(i.weights),C=e=>e.isWhole?`row`:`(row - WEIGHTS_ROW_START)`;return[x(`complete`,s,[y(`values`,i.values),...i.mask?[y(`mask`,i.mask,`read`,`u32`)]:[],y(`valid`,d,`read_write`,`u32`),...i.weights.status?.overflow?[y(`overflow`,i.weights.status.overflow,`read`,`u32`)]:[]],`var included = ${i.mask?`mask[maskOffset + index] != 0u`:`true`} ${i.weights.status?.overflow?`&& overflow[overflowOffset] == 0u`:``};
for (var column = 0u; column < ${c}u; column++) { included = included && isFiniteBits(values[valuesOffset + index * ${c}u + column]); }
valid[validOffset + index] = select(0u, 1u, included);`),x(`standardize`,c,[y(`values`,i.values),y(`valid`,d,`read`,`u32`),y(`standardized`,g,`read_write`)],`var count = 0u; var mean = 0.0; var squares = 0.0;
for (var row = 0u; row < ${s}u; row++) { if (valid[validOffset + row] != 0u) { count++; let value = values[valuesOffset + row * ${c}u + index]; let difference = value - mean; mean += difference / f32(count); squares += difference * (value - mean); } }
for (var row = 0u; row < ${s}u; row++) { var value = getNaN(); if (count > 1u && squares > 0.0 && isFiniteBits(squares) && isFiniteBits(mean) && valid[validOffset + row] != 0u) { value = (values[valuesOffset + row * ${c}u + index] - mean) * sqrt(f32(count) / squares); } standardized[standardizedOffset + row * ${c}u + index] = value; }`),...S.map(e=>x(`draws${e.idSuffix}`,e.rowCount*(l+1),[y(`standardized`,g),y(`valid`,d,`read`,`u32`),y(`offsets`,e.weights.offsets,`read`,`u32`),y(`neighbors`,e.weights.neighbors,`read`,`u32`),y(`weights`,e.weights.weights),y(`observed`,i.output.statistic,`read_write`),y(`simulations`,_,`read_write`),y(`parameters`,t,`read`,`u32`)],`let row = index / ${l+1}u${e.isWhole?``:` + WEIGHTS_ROW_START`}; let replicate = index % ${l+1}u;
var candidates: array<u32, ${s}>; var count = 0u;
for (var other = 0u; other < ${s}u; other++) { if (other != row && valid[validOffset + other] != 0u) { candidates[count] = other; count++; } }
var value = getNaN();
if (valid[validOffset + row] != 0u && count > 0u) {
 var total = 0.0; var weightSum = 0.0; var chosen = 0u;
 var stream = createPhiloxStream(vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]), row, replicate, ${m.localPermutationTest.id}u);
 for (var slot = offsets[offsetsOffset + ${C(e)}]; slot < offsets[offsetsOffset + ${C(e)} + 1u]; slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  if (neighbor >= ${s}u || neighbor == row || !(weight > 0.0)) { continue; }
  if (valid[validOffset + neighbor] == 0u) { continue; }
  var candidate = neighbor;
  if (replicate > 0u) { let selection = chosen + nextPhiloxBelow(&stream, count - chosen); candidate = candidates[selection]; candidates[selection] = candidates[chosen]; candidates[chosen] = candidate; chosen++; }
  var distance = 0.0;
  for (var column = 0u; column < ${c}u; column++) { let difference = standardized[standardizedOffset + row * ${c}u + column] - standardized[standardizedOffset + candidate * ${c}u + column]; distance += difference * difference; }
  total += weight * distance / ${c}.0; weightSum += weight;
 }
 if (weightSum > 0.0) { value = total / weightSum; }
}
if (replicate == 0u) { observed[observedOffset + row] = value; } else { simulations[simulationsOffset + row * ${l}u + replicate - 1u] = value; }`,`${h}${e.isWhole?``:`\n${e.constantsWGSL}`}`)),x(`inference`,s,[y(`observed`,i.output.statistic),y(`simulations`,_),y(`valid`,d,`read`,`u32`),y(`pValues`,i.output.pValues,`read_write`),y(`classes`,i.output.classes,`read_write`,`u32`),y(`parameters`,t,`read`,`u32`)],`let observed = observed[observedOffset + index]; var lower = 0u; var upper = 0u; var average = 0.0; var finite = isFiniteBits(observed);
for (var replicate = 0u; replicate < ${l}u; replicate++) { let value = simulations[simulationsOffset + index * ${l}u + replicate]; finite = finite && isFiniteBits(value); lower += select(0u, 1u, value <= observed); upper += select(0u, 1u, value >= observed); average += value / ${l}.0; }
var probability = getNaN(); var classification = 0xffffffffu;
if (finite) { probability = ${i.tail===`esda-min-tail`?`f32(min(upper, ${l}u - upper) + 1u) / ${l+1}.0`:`min(1.0, 2.0 * f32(min(lower, upper) + 1u) / ${l+1}.0)`}; var family = 1u;
${i.correction===`bonferroni`?`family = 0u; for (var row = 0u; row < ${s}u; row++) { family += valid[validOffset + row]; }`:``}
probability = min(1.0, probability * f32(family)); classification = 0u;
if (probability <= ${n?`0.999999`:`bitcast<f32>(parameters[parametersOffset + 2u])`}) { classification = select(2u, 1u, observed < average); }
}
pValues[pValuesOffset + index] = probability; classes[classesOffset + index] = classification;`)]}},B=[`not significant`,`cluster`,`negative association`],V=[`equal-tail`,`esda-min-tail`],H=w({parameters:j,caption:`Multivariate local Geary classes: cluster, negative association, not significant.`,createInputs:({random:e,world:t})=>{let n=O(e,t,240,0);return{...A(n),graph:E(n.positions,6)}},build(e,t,n){let{graph:r}=t,i=n.variableCount,a=e.allocate(`values`,`float32`,t.count*i),o=e.allocate(`statistic`,`float32`,t.count),s=e.allocate(`p-values`,`float32`,t.count),c=e.allocate(`classes`,`uint32`,t.count);return e.graph.add(new z({weights:T(e,r.weights),values:a.view,variableCount:i,maximumPermutations:n.maximumPermutations,seed:[1,2],tail:V[n.tail],output:{statistic:o.view,pValues:s.view,classes:c.view}})),{outputs:{classes:c,statistic:o,pValues:s,positions:e.upload(`positions`,`float32x2`,t.positions)},update:e=>a.write(k(t.field,i,e.noise))}},rebuildOn:[`variableCount`,`tail`,`maximumPermutations`],interaction:{kind:`probe`,hint:`hover a point`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=D(t,e,t.graph.weights);if(a.row<0)return{};let o=i(n.pValues)?.[a.row],s=i(n.classes)?.[a.row];return{glyphs:a.glyphs,readout:`${s===void 0?`…`:B[s]??`class ${s}`} · p ${o===void 0?`…`:o.toFixed(3)} · ${r.variableCount} variables`}}},draw:({outputs:e,theme:t})=>[{layer:`dots`,context:!0,positions:e.positions,radius:2},{layer:`dots`,positions:e.positions,values:e.classes,ramp:`categorical`,categories:[t.muted,t.categories[1],t.categories[0]],radius:6}]});export{H as default};