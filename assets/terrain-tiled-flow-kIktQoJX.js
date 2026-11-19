import{t as e}from"./common-CshkX-d_.js";import{u as t}from"./raster-band-utils-BcTxc8_Y.js";import{D as n,S as r,T as i,f as a,p as o,v as s}from"./gpu-contributor-utils-BIwLQON5.js";import{i as c,t as l}from"./graph-chunk-groups-BB8ACpQO.js";import{i as u,r as d}from"./terrain-flow-tracing-BfyLX2kv.js";import{c as f,l as p,n as m,r as h,t as g}from"./terrain-processing-utils-D-TSF_vl.js";import{r as _}from"./kernel-C3tfQOn4.js";import{o as v}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as y,n as b,r as x}from"./gpu-terrain-branching-flow-BVLHQyy9.js";import{n as S,t as C}from"./terrain-resolution-status-DTEt89JU.js";import{m as w}from"./raster-relaxation-Di9wjtxp.js";import{t as T}from"./binary-search-wgsl-CMmaMYIt.js";import{t as E}from"./gpu-sort-xRm8l5yn.js";import{s as D}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as O}from"./gpu-command-graph-BdzdmV6u.js";import{t as k}from"./define-run-B-brkCe4.js";import{X as A}from"./entry-preview-D70TdSbK.js";import{c as j,d as M,l as N,o as P,p as F,r as I,u as ee}from"./terrain-Bwc0vRX3.js";function te(t,n,r){let i=n.grid.width*n.grid.height,a=_(t,`${r}-fractions`,`float32`,i*8),o=_(t,`${r}-runoff`,`float32`,i),s=_(t,`${r}-streams`,`uint32`,i),c=_(t,`${r}-accumulation`,`float32`,i),u=_(t,`${r}-status`,`uint32`,i),p=_(t,`${r}-unresolved`,`uint32`,1),m=[0,1].map(e=>_(t,`${r}-reach-${e}`,`uint32`,i)),h=`${d(n.grid.width,n.grid.height,n.directionEncoding)}\n${g}
fn isStreamWord(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }`,y=`${h}
fn getTileParameter(word: u32) -> u32 { return getChunkParameter(36u + word); }`,x=[];n.tiles.forEach((e,n)=>{let[i,c,u,d]=e.owned;x.push(l(t,{id:`${r}-exchange-owned-${n}`,pipelineId:`${r}-exchange-owned`,operation:`GPUTerrainTiledFlow`,invocationCount:u*d,declarations:y,parameterValues:[e.width,i,c,u,d,e.column+i,e.row+c],bindings:[f(`directions`,e.flowDirections),...e.fractions?[f(`inputFractions`,e.fractions)]:[],f(`inputRunoff`,e.runoff),f(`inputStreams`,e.streams),f(`fractions`,a,`read_write`),f(`runoff`,o,`read_write`),f(`streams`,s,`read_write`)],body:`let ownedWidth = getTileParameter(3u);
let local = (index / ownedWidth + getTileParameter(2u)) * getTileParameter(0u) + index % ownedWidth + getTileParameter(1u);
let cell = (index / ownedWidth + getTileParameter(6u)) * GRID_WIDTH + index % ownedWidth + getTileParameter(5u);
let stream = inputStreams[inputStreamsOffset + local];
let code = getFlowDirectionCode(directions[directionsOffset + local]);
var valid = stream < UNRESOLVED_DIRECTION;
${e.fractions?`valid = valid && code != NO_CELL;`:`valid = valid && (code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && getFlowReceiver(cell, code) != NO_CELL));`}
runoff[runoffOffset + cell] = select(getNaN(), inputRunoff[inputRunoffOffset + local], valid);
streams[streamsOffset + cell] = stream;
for (var direction = 0u; direction < 8u; direction++) { fractions[fractionsOffset + cell * 8u + direction] = ${e.fractions?`inputFractions[inputFractionsOffset + local * 8u + direction]`:`select(0.0, 1.0, code == (1u << direction))`}; }`}))});let S=n.maximumIterations??e(n.grid.width+n.grid.height,128,1024),w=b(t,{id:`${r}-dag`,grid:n.grid,fractions:a,values:o,mode:`transport`,maximumIterations:S,output:{values:c,status:u,unresolvedCount:p}});x.push(...w.nodes),x.push(v(t,{id:`${r}-reach-seeds`,operation:`GPUTerrainTiledFlow`,invocationCount:i,declarations:h,bindings:[f(`fractions`,a),f(`streams`,s),f(`status`,u),f(`reach`,m[0],`read_write`)],body:`var parent = NO_CELL;
if (status[statusOffset + index] == 1u && isStreamWord(streams[streamsOffset + index])) {
 var donors = 0u; var inherited = index;
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(index, 1u << direction);
  if (neighbor == NO_CELL || !isStreamWord(streams[streamsOffset + neighbor]) || !(fractions[fractionsOffset + neighbor * 8u + (direction + 4u) % 8u] > 0.0)) { continue; }
  donors++;
  var receiverCount = 0u;
  for (var receiverDirection = 0u; receiverDirection < 8u; receiverDirection++) { receiverCount += u32(fractions[fractionsOffset + neighbor * 8u + receiverDirection] > 0.0); }
  inherited = select(neighbor, index, receiverCount > 1u);
 }
 parent = select(index, inherited, donors == 1u);
}
reach[reachOffset + index] = parent;`}));let T=Math.ceil(Math.log2(Math.max(i,1)));for(let e=0;e<T;e++)x.push(v(t,{id:`${r}-reach-jump-${e}`,operation:`GPUTerrainTiledFlow`,invocationCount:i,bindings:[f(`previous`,m[e%2]),f(`next`,m[(e+1)%2],`read_write`)],body:`let parent = previous[previousOffset + index]; var ancestor = parent; if (parent != 0xffffffffu) { ancestor = previous[previousOffset + parent]; } next[nextOffset + index] = ancestor;`}));return n.tiles.forEach((e,n)=>{let[i,a,o,s]=e.owned;x.push(l(t,{id:`${r}-publish-${n}`,pipelineId:`${r}-publish`,operation:`GPUTerrainTiledFlow`,invocationCount:e.width*e.height,declarations:y,parameterValues:[e.width,i,a,o,s,e.column+i,e.row+a],bindings:[f(`values`,c),f(`globalStatus`,u),f(`globalReach`,m[T%2]),f(`accumulation`,e.output.accumulation,`read_write`),f(`status`,e.output.status,`read_write`),f(`reachIds`,e.output.reachIds,`read_write`)],body:`let column = index % getTileParameter(0u); let row = index / getTileParameter(0u);
let ownedColumn = getTileParameter(1u); let ownedRow = getTileParameter(2u);
let owned = column >= ownedColumn && column - ownedColumn < getTileParameter(3u) && row >= ownedRow && row - ownedRow < getTileParameter(4u);
var value = getNaN(); var state = 0u; var reach = NO_CELL;
if (owned) {
 let cell = (row - ownedRow + getTileParameter(6u)) * GRID_WIDTH + column - ownedColumn + getTileParameter(5u);
 value = values[valuesOffset + cell]; state = globalStatus[globalStatusOffset + cell]; reach = globalReach[globalReachOffset + cell];
}
accumulation[accumulationOffset + index] = value; status[statusOffset + index] = state; reachIds[reachIdsOffset + index] = reach;`}))}),x.push(...C(t,{id:r,operation:`GPUTerrainTiledFlow`,states:u,status:n.output?.status,iterationLimit:{source:w.iterationStatus,expression:`u32(source[sourceOffset + 2u] >= ${S}u && source[sourceOffset + 3u] == 0u)`}})),x}var L=536870911,ne=4294967295,re=512,R=5,z=2147483646;function B(e){return e*R+re}function ie(e,t){r(e,`grid.width`,t.grid.width,z),r(e,`grid.height`,t.grid.height,z);let n=0,i=0,o=[],s=t.tiles.map(e=>{let[t,r,a,s]=e.owned,c=a*s,l=a===1||s===1?c:a*2+(s-2)*2,u=n;return n+=l,i=Math.max(i,c),o.push(e.column+t,e.row+r,a,s,u),{column:e.column+t,row:e.row+r,width:a,height:s,boundaryBase:u,boundaryCount:l}});a(e,`tile owned cell count`,i,1,L),a(e,`coarse boundary cell count`,n,1,L);let c=[...new Set(s.flatMap(e=>[e.row,e.row+e.height]))].sort((e,t)=>e-t),l=[],u=[];for(let t of c.slice(0,-1)){let n=s.map((e,t)=>({...e,index:t})).filter(e=>e.row<=t&&t<e.row+e.height).sort((e,t)=>e.column-t.column);l.push(u.length/2,t);for(let e of n)u.push(e.index,e.column);r(e,`tile index words`,u.length,ne)}return{tiles:s,boundaryCount:n,maximumCellCount:i,rectangles:Uint32Array.from(o),rowBands:Uint32Array.from(l),columnBands:Uint32Array.from(u),estimatedCommandNodeCount:B(t.tiles.length)}}function ae(e,t,n){for(let[r,i]of[[`tile tour sums scratch`,n.maximumCellCount*32],[`boundary tour sums scratch`,n.boundaryCount*32],[`tile index rectangles`,n.rectangles.byteLength],[`tile index row bands`,n.rowBands.byteLength],[`tile index column bands`,n.columnBands.byteLength]])s(e.device,t,r,i,`use fewer, larger tiles or reduce the largest tile dimensions`),a(t,`${r} buffer bytes`,i,1,e.device.limits.maxBufferSize)}var oe=`
const TILE_NONE: u32 = 0xffffffffu;
fn tileParameter(word: u32) -> u32 { return getChunkParameter(36u + word); }
fn tileWidth() -> u32 { return tileParameter(4u); }
fn tileHeight() -> u32 { return tileParameter(5u); }
fn localInput(cell: u32) -> u32 { return (cell / tileWidth() + tileParameter(3u)) * tileParameter(0u) + cell % tileWidth() + tileParameter(2u); }
fn globalCell(cell: u32) -> u32 { return (cell / tileWidth() + tileParameter(7u)) * GRID_WIDTH + cell % tileWidth() + tileParameter(6u); }
fn localCell(cell: u32) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 if (column < tileParameter(6u) || row < tileParameter(7u) || column - tileParameter(6u) >= tileWidth() || row - tileParameter(7u) >= tileHeight()) { return TILE_NONE; }
 return (row - tileParameter(7u)) * tileWidth() + column - tileParameter(6u);
}
fn isBoundary(cell: u32) -> bool { let column = cell % tileWidth(); let row = cell / tileWidth(); return column == 0u || row == 0u || column + 1u == tileWidth() || row + 1u == tileHeight(); }
fn boundaryLocal(slot: u32, width: u32, height: u32) -> u32 {
 if (width == 1u || height == 1u) { return slot; }
 if (slot < width) { return slot; }
 let middle = (height - 2u) * 2u;
 if (slot < width + middle) { let edge = slot - width; return (edge / 2u + 1u) * width + (edge % 2u) * (width - 1u); }
 return (height - 1u) * width + slot - width - middle;
}
fn boundarySlot(cell: u32, width: u32, height: u32) -> u32 {
 if (width == 1u || height == 1u) { return cell; }
 let column = cell % width; let row = cell / width;
 if (row == 0u) { return column; }
 if (row + 1u == height) { return width + (height - 2u) * 2u + column; }
 return width + (row - 1u) * 2u + u32(column != 0u);
}
fn localNeighbor(cell: u32, direction: u32) -> u32 {
 let global = getFlowReceiver(globalCell(cell), 1u << direction);
 if (global == NO_CELL) { return TILE_NONE; }
 return localCell(global);
}
fn isStream(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }
`,V=`${T(`rowBands`,`u64`)}
${T(`columnBands`,`u64`)}
fn globalBoundary(cell: u32) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 let bandCount = (arrayLength(&rowBands) - rowBandsOffset) / 2u;
 let band = upperBoundRowBands(0u, bandCount, vec2<u32>(row, TILE_NONE)) - 1u;
 let first = rowBands[rowBandsOffset + band * 2u];
 var last = (arrayLength(&columnBands) - columnBandsOffset) / 2u;
 if (band + 1u < bandCount) { last = rowBands[rowBandsOffset + (band + 1u) * 2u]; }
 let interval = upperBoundColumnBands(first, last, vec2<u32>(column, TILE_NONE)) - 1u;
 let tile = columnBands[columnBandsOffset + interval * 2u] * 5u + rectanglesOffset;
 let width = rectangles[tile + 2u]; let height = rectangles[tile + 3u];
 let local = (row - rectangles[tile + 1u]) * width + column - rectangles[tile];
 return rectangles[tile + 4u] + boundarySlot(local, width, height);
}
`;function H(e,t){let n=new Map,r=0,i=0,a=0;for(let o of t)for(let t of o.nodes){r+=(t.workload?.commandCount??1)*o.repetitions,i+=(t.workload?.maximumInvocationCount??0)*o.repetitions,a+=(t.workload?.maximumWorkgroupCount??0)*o.repetitions;for(let r of t.resources??[]){if(!(`buffer`in r))throw Error(`${e} terrain tile sequences require buffer-only compute passes`);let t=r.buffer,i=t.buffer,a=n.get(i),o=a?.buffer,s=Math.min(t.byteOffset,o?.byteOffset??t.byteOffset),c=Math.max(t.byteOffset+(t.length-1)*t.byteStride+t.rowByteLength,o?o.byteOffset+(o.length-1)*o.byteStride+o.rowByteLength:0);n.set(i,{buffer:i.graph.createDataView(i,{format:`uint32`,byteOffset:s,length:(c-s)/4}),usage:a?.usage===`storage-read-write`||r.usage===`storage-read-write`?`storage-read-write`:`storage-read`})}}return O({id:e,resources:[...n.values()],workload:{operation:`GPUTerrainTiledFlow`,commandCount:r,maximumInvocationCount:i,maximumWorkgroupCount:a},compile:n=>{let r=new Map;try{for(let i of t)for(let t of i.nodes){if(t.type!==`compute`)throw Error(`${e} terrain tile sequences require compute passes`);r.has(t)||r.set(t,t.compile(n))}}catch(e){for(let e of r.values())e.destroy?.();throw e}return{encode:e=>{for(let n of t)for(let t=0;t<n.repetitions;t++)for(let t of n.nodes)r.get(t).encode(e)},destroy:()=>{for(let e of r.values())e.destroy?.()}}}})}function U(e,t,n){n>=2&&e.push({nodes:t,repetitions:Math.floor(n/2)}),n%2&&e.push({nodes:[t[0]],repetitions:1})}var W=`
alias TerrainRunoffSum = array<u32, 11>;
fn terrainZeroSum() -> TerrainRunoffSum { return TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); }
fn terrainEncodeRunoff(value: f32, state: u32) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
 if (state != 1u) { result[10] = 1u; return result; }
 let bits = bitcast<u32>(value); let exponent = (bits >> 23u) & 255u;
 var significand = bits & 0x7fffffu; var shift = 0u;
 if (exponent != 0u) { significand |= 0x800000u; shift = exponent - 1u; }
 let word = shift / 32u; let bit = shift % 32u;
 result[word] = significand << bit; result[word + 1u] = (significand >> 1u) >> (31u - bit);
 return result;
}
fn terrainAddRunoff(first: TerrainRunoffSum, second: TerrainRunoffSum) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var carry = 0u;
 var firstWords = first; var secondWords = second;
 for (var word = 0u; word < 10u; word++) {
  let partial = firstWords[word] + secondWords[word]; let total = partial + carry;
  carry = u32(partial < firstWords[word]) | u32(total < partial); result[word] = total;
 }
 result[10] = first[10] + second[10]; return result;
}
fn terrainSubtractRunoff(first: TerrainRunoffSum, second: TerrainRunoffSum) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var borrow = 0u;
 var firstWords = first; var secondWords = second;
 for (var word = 0u; word < 10u; word++) {
  let partial = firstWords[word] - secondWords[word]; let total = partial - borrow;
  borrow = u32(firstWords[word] < secondWords[word]) | u32(partial < borrow); result[word] = total;
 }
 result[10] = first[10] - second[10]; return result;
}
fn terrainRunoffBitsBelow(value: TerrainRunoffSum, count: u32) -> bool {
 var words = value;
 let whole = count / 32u; let remainder = count % 32u; var found = false;
 for (var word = 0u; word < 10u; word++) {
  if (word < whole) { found = found || words[word] != 0u; }
  else if (word == whole) { found = found || (words[word] & ((1u << remainder) - 1u)) != 0u; }
 }
 return found;
}
fn terrainDecodeRunoff(value: TerrainRunoffSum) -> f32 {
 var words = value;
 var leading = 0u; var found = false;
 for (var reverse = 10u; reverse > 0u; reverse--) {
  let word = reverse - 1u;
  if (!found && words[word] != 0u) { leading = word * 32u + firstLeadingBit(words[word]); found = true; }
 }
 if (!found) { return 0.0; }
 if (leading < 23u) { return bitcast<f32>(value[0]); }
 let shift = leading - 23u; let word = shift / 32u; let bit = shift % 32u;
 var significand = words[word] >> bit;
 if (word + 1u < 10u) { significand |= (words[word + 1u] << 1u) << (31u - bit); }
 if (shift > 0u) {
  let position = shift - 1u; let roundBit = (words[position / 32u] >> (position % 32u)) & 1u;
  if (roundBit != 0u && (terrainRunoffBitsBelow(value, position) || (significand & 1u) != 0u)) { significand++; }
 }
 var exponent = leading - 22u;
 if (significand == 0x1000000u) { significand >>= 1u; exponent++; }
 if (exponent >= 255u) { var infinityBits = 0x7f800000u; return bitcast<f32>(infinityBits); }
 return bitcast<f32>((exponent << 23u) | (significand & 0x7fffffu));
}
`,G=`
fn terrainLoadRunoff(edge: u32) -> TerrainRunoffSum {
 let first = firstPlaneOffset + edge * 4u; let second = secondPlaneOffset + edge * 4u; let third = thirdPlaneOffset + edge * 4u;
 return TerrainRunoffSum(firstPlane[first], firstPlane[first + 1u], firstPlane[first + 2u], firstPlane[first + 3u], secondPlane[second], secondPlane[second + 1u], secondPlane[second + 2u], secondPlane[second + 3u], thirdPlane[third], thirdPlane[third + 1u], thirdPlane[third + 2u]);
}
`,K=`
fn terrainStoreRunoff(edge: u32, value: TerrainRunoffSum) {
 let first = firstOutputOffset + edge * 4u; let second = secondOutputOffset + edge * 4u; let third = thirdOutputOffset + edge * 4u;
 firstOutput[first] = value[0]; firstOutput[first + 1u] = value[1]; firstOutput[first + 2u] = value[2]; firstOutput[first + 3u] = value[3];
 secondOutput[second] = value[4]; secondOutput[second + 1u] = value[5]; secondOutput[second + 2u] = value[6]; secondOutput[second + 3u] = value[7];
 thirdOutput[third] = value[8]; thirdOutput[third + 1u] = value[9]; thirdOutput[third + 2u] = value[10]; thirdOutput[third + 3u] = 0u;
}
`;function q(e,t,n){let r=(r,i=n)=>_(e,`${t}-${r}`,`uint32`,i),i=[0,1].map(e=>[r(`sums-${e}-0`,n*8),r(`sums-${e}-1`,n*8),r(`sums-${e}-2`,n*8)]);return{parents:r(`parents`),firstChildren:r(`first-children`),nextSiblings:r(`next-siblings`),weights:_(e,`${t}-weights`,`float32`,n),states:r(`states`),successors:[r(`successors-0`,n*2),r(`successors-1`,n*2)],sumPlanes:i,values:_(e,`${t}-values`,`float32`,n),resolved:r(`resolved`)}}function se(e,t,n){let r=(t,r=n)=>e.createDataView(t.buffer,{format:t.format,length:r});return{parents:r(t.parents),firstChildren:r(t.firstChildren),nextSiblings:r(t.nextSiblings),weights:r(t.weights),states:r(t.states),successors:[r(t.successors[0],n*2),r(t.successors[1],n*2)],sumPlanes:t.sumPlanes.map(e=>[r(e[0],n*8),r(e[1],n*8),r(e[2],n*8)]),values:r(t.values),resolved:r(t.resolved)}}function J(e,t,n,r){let i=r.parents.length,a=(t,r,i,a,o=``)=>l(e,{id:`${n}-${t}`,pipelineId:`terrain-tiled-forest-${t}`,operation:`GPUTerrainTiledFlow`,invocationCount:r,bindings:i,body:a,declarations:o}),o=e=>[`firstPlane`,`secondPlane`,`thirdPlane`].map((t,n)=>f(t,r.sumPlanes[e][n])),s=e=>[`firstOutput`,`secondOutput`,`thirdOutput`].map((t,n)=>f(t,r.sumPlanes[e][n],`read_write`));t.push({repetitions:1,nodes:[a(`link`,i,[f(`parents`,r.parents),f(`children`,r.firstChildren),f(`siblings`,r.nextSiblings),f(`successors`,r.successors[0],`read_write`)],`
let parent = parents[parentsOffset + index]; let child = children[childrenOffset + index]; let sibling = siblings[siblingsOffset + index];
let enter = index * 2u; let exitEdge = enter + 1u;
successors[successorsOffset + enter] = select(child * 2u, exitEdge, child == 0xffffffffu);
var next = exitEdge; if (parent != 0xffffffffu) { next = select(parent * 2u + 1u, sibling * 2u, sibling != 0xffffffffu); }
successors[successorsOffset + exitEdge] = next;`),a(`weights`,i*2,[f(`weights`,r.weights),f(`states`,r.states),...s(0)],`
var value = terrainZeroSum(); if ((index & 1u) == 0u) { value = terrainEncodeRunoff(weights[weightsOffset + index / 2u], states[statesOffset + index / 2u]); }
terrainStoreRunoff(index, value);`,`${W}\n${K}`)]});let c=Math.ceil(Math.log2(Math.max(i*2,2)))+1;U(t,[0,1].map(e=>a(`jump`,i*2,[f(`previous`,r.successors[e]),...o(e),f(`next`,r.successors[1-e],`read_write`),...s(1-e)],`
let following = previous[previousOffset + index]; var value = terrainLoadRunoff(index);
if (following != index) { value = terrainAddRunoff(value, terrainLoadRunoff(following)); }
next[nextOffset + index] = previous[previousOffset + following]; terrainStoreRunoff(index, value);`,`${W}\n${G}\n${K}`)),c),t.push({repetitions:1,nodes:[a(`resolve`,i,[f(`parents`,r.parents),f(`successors`,r.successors[c%2]),...o(c%2),f(`states`,r.states),f(`values`,r.values,`read_write`),f(`resolved`,r.resolved,`read_write`)],`
let terminal = successors[successorsOffset + index * 2u];
let finiteTree = (terminal & 1u) != 0u && parents[parentsOffset + terminal / 2u] == 0xffffffffu;
let total = terrainSubtractRunoff(terrainLoadRunoff(index * 2u), terrainLoadRunoff(index * 2u + 1u));
let value = terrainDecodeRunoff(total); let original = states[statesOffset + index];
let state = select(select(2u, 1u, finiteTree && total[10] == 0u && isFiniteValue(value)), 0u, original == 0u);
values[valuesOffset + index] = select(getNaN(), value, state == 1u); resolved[resolvedOffset + index] = state;`,`${W}\n${G}\n${g}`)]})}function Y(e,t,n,r){let i=r.parents.length,a=r.successors.map(t=>e.createDataView(t.buffer,{format:`uint32`,length:i})),o=(t,r,a)=>l(e,{id:`${n}-${t}`,pipelineId:`terrain-tiled-cycle-${t}`,operation:`GPUTerrainTiledFlow`,invocationCount:i,bindings:r,body:a});t.push({repetitions:1,nodes:[o(`initialize`,[f(`parents`,r.parents),f(`pointers`,a[0],`read_write`),f(`flags`,r.firstChildren,`read_write`)],`let parent = parents[parentsOffset + index]; pointers[pointersOffset + index] = select(parent, index, parent == 0xffffffffu); flags[flagsOffset + index] = 0u;`)]});let s=[0,1].map(e=>o(`jump`,[f(`previous`,a[e]),f(`next`,a[1-e],`read_write`)],`next[nextOffset + index] = previous[previousOffset + previous[previousOffset + index]];`)),c=Math.ceil(Math.log2(Math.max(i,1)));U(t,s,c),t.push({repetitions:1,nodes:[o(`mark`,[f(`parents`,r.parents),f(`pointers`,a[c%2]),{name:`flags`,view:r.firstChildren,type:`atomic<u32>`,access:`read_write`}],`let image = pointers[pointersOffset + index]; if (parents[parentsOffset + image] != 0xffffffffu) { atomicOr(&flags[flagsOffset + image], 1u); }`),o(`cut`,[f(`flags`,r.firstChildren),f(`parents`,r.parents,`read_write`),f(`states`,r.states,`read_write`)],`if (flags[flagsOffset + index] != 0u) { parents[parentsOffset + index] = 0xffffffffu; if (states[statesOffset + index] != 0u) { states[statesOffset + index] = 2u; } }`)]})}var X=256;function ce(e,t,n,r){let i=[],a=(t,r)=>{let a=D(e,`${n}-${t}`,r);return i.push(...a.nodes),a.parameters},o=a(`tile-rectangles`,r.rectangles),s=a(`tile-row-bands`,r.rowBands),c=a(`tile-column-bands`,r.columnBands),u=[f(`rectangles`,o),f(`rowBands`,s),f(`columnBands`,c)],p=_(e,`${n}-boundary-words`,`uint32`,r.boundaryCount*2),m=[0,1].map(t=>_(e,`${n}-boundary-reach-${t}`,`uint32`,r.boundaryCount*2)),h=q(e,`${n}-boundary-forest`,r.boundaryCount),v=q(e,`${n}-tile-forest`,r.maximumCellCount),y=[0,1].map(t=>_(e,`${n}-tile-pointers-${t}`,`uint32`,r.maximumCellCount)),b=[0,1].map(t=>_(e,`${n}-tile-reach-${t}`,`uint32`,r.maximumCellCount)),x=_(e,`${n}-totals`,`uint32`,2),S=`${d(t.grid.width,t.grid.height,t.directionEncoding)}\n${g}\n${oe}`,C=(t,r,i,a,o=[],s=S)=>l(e,{id:`${n}-${t}`,pipelineId:`terrain-tiled-${t.replace(/-\d+/g,``)}`,operation:`GPUTerrainTiledFlow`,invocationCount:r,bindings:i,body:a,declarations:s,parameterValues:o});i.push(C(`initialize`,1,[f(`totals`,x,`read_write`)],`totals[totalsOffset] = 0u; totals[totalsOffset + 1u] = 0u;`,[],``));let w=(e,n=0)=>{let i=t.tiles[e],a=r.tiles[e];return[i.width,i.height,...i.owned,a.column,a.row,a.boundaryBase,a.boundaryCount,n]};for(let[e,n]of t.tiles.entries()){let t=r.tiles[e],a=w(e);i.push(C(`boundary-words-${e}`,t.boundaryCount,[f(`directions`,n.flowDirections),f(`streams`,n.streams),f(`words`,p,`read_write`)],`
let local = boundaryLocal(index, tileWidth(), tileHeight()); let source = localInput(local);
let destination = (tileParameter(8u) + index) * 2u + wordsOffset;
words[destination] = getFlowDirectionCode(directions[directionsOffset + source]); words[destination + 1u] = streams[streamsOffset + source];`,a))}let O=`${S}\n${V}
fn streamPredecessor(cell: u32) -> vec2<u32> {
 var donor = TILE_NONE; var count = 0u; let global = globalCell(cell);
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(global, 1u << direction); if (neighbor == NO_CELL) { continue; }
  let local = localCell(neighbor); var code = 0u; var stream = 0u;
  if (local != TILE_NONE) { let source = localInput(local); code = getFlowDirectionCode(directions[directionsOffset + source]); stream = streams[streamsOffset + source]; }
  else { let source = wordsOffset + globalBoundary(neighbor) * 2u; code = words[source]; stream = words[source + 1u]; }
  if (isStream(stream) && getFlowReceiver(neighbor, code) == global) { donor = neighbor; count++; }
 }
 return vec2<u32>(donor, count);
}`,k=(a,o)=>{let s=t.tiles[a],c=r.tiles[a],l=c.width*c.height,d=w(a,o),g=se(e,v,l),_=y.map(t=>e.createDataView(t.buffer,{format:`uint32`,length:l})),x=b.map(t=>e.createDataView(t.buffer,{format:`uint32`,length:l})),T=[];T.push({repetitions:1,nodes:[C(`prepare-${o}-${a}`,l,[f(`directions`,s.flowDirections),f(`runoff`,s.runoff),f(`streams`,s.streams),f(`boundaryValues`,h.values),f(`boundaryStates`,h.resolved),f(`parents`,g.parents,`read_write`),f(`weights`,g.weights,`read_write`),f(`states`,g.states,`read_write`)],`
let source = localInput(index); let global = globalCell(index);
let code = getFlowDirectionCode(directions[directionsOffset + source]); let receiver = getFlowReceiver(global, code);
let validCode = code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && receiver != NO_CELL);
var weight = runoff[runoffOffset + source]; var state = u32(validCode && streams[streamsOffset + source] < UNRESOLVED_DIRECTION && isFiniteValue(weight) && weight >= 0.0);
var parent = TILE_NONE; if (receiver != NO_CELL) { parent = localCell(receiver); }
if (tileParameter(10u) == 0u) { if (isBoundary(index)) { parent = TILE_NONE; } }
else {
 if (parent != TILE_NONE && isBoundary(parent)) { parent = TILE_NONE; }
 if (isBoundary(index)) { let slot = tileParameter(8u) + boundarySlot(index, tileWidth(), tileHeight()); weight = boundaryValues[boundaryValuesOffset + slot]; state = boundaryStates[boundaryStatesOffset + slot]; }
}
parents[parentsOffset + index] = parent; weights[weightsOffset + index] = weight; states[statesOffset + index] = state;`,d)]}),o===0&&T.push({repetitions:1,nodes:[C(`downstream-seeds-${a}`,l,[f(`directions`,s.flowDirections),f(`pointers`,_[0],`read_write`)],`
let code = getFlowDirectionCode(directions[directionsOffset + localInput(index)]); let receiver = getFlowReceiver(globalCell(index), code);
var parent = index; if (!isBoundary(index) && receiver != NO_CELL) { let local = localCell(receiver); if (local != TILE_NONE) { parent = local; } }
pointers[pointersOffset + index] = parent;`,d)]}),T.push({repetitions:1,nodes:[C(`reach-seeds-${a}`,l,[f(`directions`,s.flowDirections),f(`streams`,s.streams),f(`words`,p),...u,f(`reach`,x[0],`read_write`)],`
var parent = index;
if (!isBoundary(index) && isStream(streams[streamsOffset + localInput(index)])) {
 let donor = streamPredecessor(index); if (donor.y == 1u) { let local = localCell(donor.x); if (local != TILE_NONE) { parent = local; } }
}
reach[reachOffset + index] = parent;`,d,O)]});let E=Math.ceil(Math.log2(Math.max(l,1))),D=(e,t)=>{let n=[0,1].map(n=>C(`${t}-${a}-${n}`,l,[f(`previous`,e[n]),f(`next`,e[1-n],`read_write`)],`next[nextOffset + index] = previous[previousOffset + previous[previousOffset + index]];`,[],``));return U(T,n,E),e[E%2]},k=o===0?D(_,`pointer-jump`):_[0],A=D(x,`pointer-jump`);return Y(e,T,`${n}-tile-${a}-phase-${o}-cycles`,g),T.push({repetitions:1,nodes:[C(`topology-${a}`,l,[f(`parents`,g.parents),f(`children`,g.firstChildren,`read_write`),f(`siblings`,g.nextSiblings,`read_write`)],`
var child = TILE_NONE;
for (var direction = 0u; direction < 8u; direction++) { let neighbor = localNeighbor(index, direction); if (neighbor != TILE_NONE && parents[parentsOffset + neighbor] == index && child == TILE_NONE) { child = neighbor; } }
children[childrenOffset + index] = child;
let parent = parents[parentsOffset + index]; var sibling = TILE_NONE; var passed = false;
if (parent != TILE_NONE) { for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = localNeighbor(parent, direction); if (neighbor == index) { passed = true; continue; }
 if (neighbor != TILE_NONE && passed && sibling == TILE_NONE && parents[parentsOffset + neighbor] == parent) { sibling = neighbor; }
} }
siblings[siblingsOffset + index] = sibling;`,d)]}),J(e,T,`${n}-tile-${a}-phase-${o}`,g),o===0&&T.push({repetitions:1,nodes:[C(`boundary-seeds-${a}`,c.boundaryCount,[f(`values`,g.values),f(`resolved`,g.resolved),f(`weights`,h.weights,`read_write`),f(`states`,h.states,`read_write`)],`
let local = boundaryLocal(index, tileWidth(), tileHeight()); let destination = tileParameter(8u) + index;
weights[weightsOffset + destination] = values[valuesOffset + local]; states[statesOffset + destination] = resolved[resolvedOffset + local];`,d),C(`boundary-parents-${a}`,c.boundaryCount,[f(`directions`,s.flowDirections),f(`pointers`,k),...u,f(`parents`,h.parents,`read_write`)],`
let local = boundaryLocal(index, tileWidth(), tileHeight()); let receiver = getFlowReceiver(globalCell(local), getFlowDirectionCode(directions[directionsOffset + localInput(local)]));
var parent = TILE_NONE;
if (receiver != NO_CELL) {
 let receiverLocal = localCell(receiver);
 if (receiverLocal == TILE_NONE) { parent = globalBoundary(receiver); }
 else { let terminal = pointers[pointersOffset + receiverLocal]; if (isBoundary(terminal)) { parent = tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight()); } }
}
parents[parentsOffset + tileParameter(8u) + index] = parent;`,d,`${S}\n${V}`),C(`boundary-reach-seeds-${a}`,c.boundaryCount,[f(`directions`,s.flowDirections),f(`streams`,s.streams),f(`words`,p),...u,f(`pointers`,A),f(`reach`,m[0],`read_write`)],`
let local = boundaryLocal(index, tileWidth(), tileHeight()); var parent = TILE_NONE; var source = TILE_NONE;
if (isStream(streams[streamsOffset + localInput(local)])) {
 let donor = streamPredecessor(local);
 if (donor.y != 1u) { source = globalCell(local); }
 else {
  let donorLocal = localCell(donor.x);
  if (donorLocal == TILE_NONE) { parent = globalBoundary(donor.x); }
  else { let terminal = pointers[pointersOffset + donorLocal]; if (isBoundary(terminal)) { parent = tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight()); } else { source = globalCell(terminal); } }
 }
}
let destination = reachOffset + (tileParameter(8u) + index) * 2u; reach[destination] = parent; reach[destination + 1u] = source;`,d,O)]}),i.push(H(`${n}-tile-${a}-phase-${o}`,T)),{local:g,reachRoots:A,parameters:d}};for(let e=0;e<t.tiles.length;e++)k(e,0);let A=[];Y(e,A,`${n}-boundary-cycles`,h),i.push(H(`${n}-boundary-cycles`,A));let j=_(e,`${n}-boundary-identifiers`,`uint32`,r.boundaryCount),M=_(e,`${n}-boundary-sorted-parents`,`uint32`,r.boundaryCount),N=_(e,`${n}-boundary-sorted-children`,`uint32`,r.boundaryCount);i.push(C(`boundary-identifiers`,r.boundaryCount,[f(`identifiers`,j,`read_write`)],`identifiers[identifiersOffset + index] = index;`,[],``),...new E({id:`${n}-boundary-order`,keys:h.parents,values:j,outputKeys:M,outputValues:N}).getCommandNodes(e)),i.push(C(`boundary-topology`,r.boundaryCount,[f(`parents`,h.parents),f(`sortedParents`,M),f(`sortedChildren`,N),f(`children`,h.firstChildren,`read_write`),f(`siblings`,h.nextSiblings,`read_write`)],`
let count = getChunkParameter(1u); let first = lowerBoundSortedParents(0u, count, index);
var child = 0xffffffffu; if (first < count && sortedParents[sortedParentsOffset + first] == index) { child = sortedChildren[sortedChildrenOffset + first]; }
children[childrenOffset + index] = child;
let parent = parents[parentsOffset + index]; var sibling = 0xffffffffu;
if (parent != 0xffffffffu) {
 let firstSibling = lowerBoundSortedParents(0u, count, parent); let end = upperBoundSortedParents(firstSibling, count, parent);
 // Stable radix sorting preserves ascending child IDs within one parent segment.
 let position = lowerBoundSortedChildren(firstSibling, end, index);
 if (position + 1u < end) { sibling = sortedChildren[sortedChildrenOffset + position + 1u]; }
}
siblings[siblingsOffset + index] = sibling;`,[],`${T(`sortedParents`)}\n${T(`sortedChildren`)}`));let P=[];J(e,P,`${n}-boundary`,h),i.push(H(`${n}-boundary-solve`,P));let F=Math.ceil(Math.log2(Math.max(r.boundaryCount,1))),I=[];U(I,[0,1].map(e=>C(`boundary-reach-jump-${e}`,r.boundaryCount,[f(`previous`,m[e]),f(`next`,m[1-e],`read_write`)],`
let base = previousOffset + index * 2u; let parent = previous[base];
var following = parent; var source = previous[base + 1u];
if (parent != 0xffffffffu) { following = previous[previousOffset + parent * 2u]; source = previous[previousOffset + parent * 2u + 1u]; }
next[nextOffset + index * 2u] = following; next[nextOffset + index * 2u + 1u] = source;`,[],``)),F),I.length&&i.push(H(`${n}-boundary-reaches`,I));for(let[e,n]of t.tiles.entries()){let{local:t,reachRoots:r,parameters:a}=k(e,1);i.push(C(`publish-bounded-${e}`,n.width*n.height,[f(`values`,t.values),f(`resolved`,t.resolved),f(`pointers`,r),f(`reach`,m[F%2]),f(`streams`,n.streams),f(`accumulation`,n.output.accumulation,`read_write`),f(`status`,n.output.status,`read_write`),f(`reachIds`,n.output.reachIds,`read_write`)],`
let column = index % tileParameter(0u); let row = index / tileParameter(0u);
var value = getNaN(); var state = 0u; var source = TILE_NONE;
if (column >= tileParameter(2u) && row >= tileParameter(3u) && column - tileParameter(2u) < tileWidth() && row - tileParameter(3u) < tileHeight()) {
 let local = (row - tileParameter(3u)) * tileWidth() + column - tileParameter(2u);
 state = resolved[resolvedOffset + local]; value = values[valuesOffset + local];
 if (state == 1u && isStream(streams[streamsOffset + index])) {
  let terminal = pointers[pointersOffset + local];
  if (isBoundary(terminal)) { source = reach[reachOffset + (tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight())) * 2u + 1u]; }
  else { source = globalCell(terminal); }
 }
}
accumulation[accumulationOffset + index] = value; status[statusOffset + index] = state; reachIds[reachIdsOffset + index] = source;`,a)),i.push(C(`tally-${e}`,Math.ceil(t.parents.length/X),[f(`states`,n.output.status),{name:`totals`,view:x,type:`atomic<u32>`,access:`read_write`}],`
var invalid = 0u; var unresolved = 0u;
for (var position = index * ${X}u; position < min((index + 1u) * ${X}u, tileWidth() * tileHeight()); position++) {
 let state = states[statesOffset + localInput(position)]; invalid += u32(state == 0u); unresolved += u32(state == 2u);
}
atomicAdd(&totals[totalsOffset], invalid); atomicAdd(&totals[totalsOffset + 1u], unresolved);`,a))}for(let[e,n]of[t.output?.status.invalidCount,t.output?.status.converged,t.output?.status.iterationLimitReached].entries())n&&i.push(C(`aggregate-${e}`,1,[f(`totals`,x),f(`port`,n,`read_write`)],`
var value = 0u; let kind = getChunkParameter(36u);
if (kind == 0u) { value = totals[totalsOffset]; } else if (kind == 1u) { value = u32(totals[totalsOffset + 1u] == 0u); }
port[portOffset] = value;`,[e],``));return i}var Z=536870911,le=4294967295,ue=6;function Q(e,t){return e.routing!==`branching`&&e.execution!==`resident`&&(e.execution===`tiled`||e.grid.width*e.grid.height*32>Math.min(t.device.limits.maxStorageBufferBindingSize,t.device.limits.maxBufferSize))}var de=class{props;id;estimatedCommandNodeCount;tiledPlan;constructor(r){this.props=r,this.id=r.id??`terrain-tiled-flow`;let s=t(this.id,r.grid);r.execution===`resident`||r.routing===`branching`?a(this.id,`resident grid cell count`,s,1,Z):a(this.id,`D8 grid cell count`,s,1,le),i(this.id,r.execution===void 0||[`auto`,`resident`,`tiled`].includes(r.execution),`execution must be 'auto', 'resident' or 'tiled'`),i(this.id,r.execution!==`tiled`||r.routing!==`branching`,`execution 'tiled' requires routing 'd8'`),i(this.id,r.routing===void 0||r.routing===`d8`||r.routing===`branching`,`routing must be 'd8' or 'branching'`),u(this.id,r.directionEncoding);let c=r.maximumIterations??e(r.grid.width+r.grid.height,128,1024);w(this.id,`maximumIterations`,c),this.estimatedCommandNodeCount=Math.max(x(c)+r.tiles.length*2+1+Math.ceil(Math.log2(s))+ue,r.routing!==`branching`&&r.execution!==`resident`?B(r.tiles.length):0),o(this.id,this.estimatedCommandNodeCount),p(this.id,{...r.output?.status});let l=0,d=[],f=[],m=[];for(let[e,t]of r.tiles.entries()){let o=`tiles[${e}]`,[s,c,u,p]=t.owned;i(this.id,[t.column,t.row,t.width,t.height,...t.owned].every(Number.isSafeInteger),`${o} column, row, width, height and owned must be integers`),a(this.id,`${o} cell count`,t.width*t.height),i(this.id,t.width>0&&t.height>0&&s>=0&&c>=0&&u>0&&p>0&&s+u<=t.width&&c+p<=t.height,`${o} owned rectangle must be non-empty and inside the tile`);let h=[t.column+s,t.row+c,u,p];i(this.id,h[0]>=0&&h[1]>=0&&h[0]+u<=r.grid.width&&h[1]+p<=r.grid.height,`${o} owned rectangle must be inside the grid`);for(let e of d)i(this.id,h[0]>=e[0]+e[2]||e[0]>=h[0]+u||h[1]>=e[1]+e[3]||e[1]>=h[1]+p,`${o} owned rectangle must not overlap another tile's`);d.push(h),l+=u*p,r.routing===`branching`?(a(this.id,`${o} cell count with routing 'branching'`,t.width*t.height,1,Z),i(this.id,!!t.fractions,`${o} fractions are required with routing 'branching'`),n(this.id,`${o} fractions`,t.fractions,t.width*t.height*8),f.push(t.fractions)):i(this.id,!t.fractions,`${o} fractions need routing 'branching'`),f.push(t.flowDirections,t.runoff,t.streams),m.push(...Object.values(t.output));for(let[e,r]of[[`flowDirections`,t.flowDirections],[`runoff`,t.runoff],[`streams`,t.streams],...Object.entries(t.output).map(([e,t])=>[`output.${e}`,t])])n(this.id,`${o} ${e}`,r,t.width*t.height)}i(this.id,l===r.grid.width*r.grid.height,`owned rectangles must cover every grid cell exactly once`);let g=[...m,...S(r.output?.status)],_=Q(r,r.tiles[0].streams.buffer.graph);h(this.id,f,g,[...r.tiles.map(e=>({reads:_?[e.streams]:[],writes:Object.values(e.output)})),...S(r.output?.status).map(e=>({writes:[e]}))]),r.routing!==`branching`&&r.execution!==`resident`&&(this.tiledPlan=ie(this.id,r))}getCommandNodes(t){let{id:n,props:r}=this;m(n,t,[...r.tiles.flatMap(e=>[e.flowDirections,e.runoff,e.streams,...e.fractions?[e.fractions]:[],...Object.values(e.output)]),...S(r.output?.status)]);let i=r.grid.width*r.grid.height;return this.tiledPlan&&Q(r,t)?(ae(t,n,this.tiledPlan),c(t,n,8),ce(t,r,n,this.tiledPlan)):(s(t.device,n,`resident fractions scratch`,i*8*4,`reduce grid dimensions`),y(t,n,i,r.maximumIterations??e(r.grid.width+r.grid.height,128,1024)),c(t,n,r.routing===`branching`?7:6),te(t,r,n))}},fe=128,pe=72,me=4,he=4294967295;function ge(e,t){return e!==null&&e.length===t.length&&e.every((e,n)=>e===t[n])}function $(e,t,n){let r=new e(n),i=0;for(let e of t)r.set(e,i),i+=e.length;return r}var _e=k({parameters:A,caption:`Streams stay continuous across tile seams.`,createInputs:e=>({...P(e,pe,fe),focus:{reach:-1,key:``,source:null}}),build(e,t,n){let r=Math.round(n.tiles),i=I(e,t),a=e.upload(`runoff`,`float32`,new Float32Array(t.cellCount).fill(1)),o=t.rows/r,s=o*t.columns,c=(t,n)=>e.graph.createDataView(t.view.buffer,{format:t.format,length:s,byteOffset:n*s*4}),l=Array.from({length:r},(n,r)=>({column:0,row:r*o,width:t.columns,height:o,owned:[0,0,t.columns,o],flowDirections:c(i.flowDirections,r),runoff:c(a,r),streams:c(i.streams,r),accumulation:e.allocate(`tile-${r}-accumulation`,`float32`,s),reachIds:e.allocate(`tile-${r}-reach-ids`,`uint32`,s),status:e.allocate(`tile-${r}-status`,`uint32`,s)}));e.graph.add(new de({grid:t.grid,routing:`d8`,tiles:l.map(e=>({...e,output:{accumulation:e.accumulation.view,status:e.status.view,reachIds:e.reachIds.view}}))}));let u=e.allocate(`assembled-accumulation`,`float32`,t.cellCount),d=e.allocate(`assembled-reach-ids`,`uint32`,t.cellCount),f=()=>{let t=e.device.createCommandEncoder();l.forEach((e,n)=>{for(let[r,i]of[[e.accumulation,u],[e.reachIds,d]])t.copyBufferToBuffer({sourceBuffer:r.buffer,destinationBuffer:i.buffer,destinationOffset:n*s*4,size:s*4})}),e.device.submit(t.finish())},p=e.allocate(`seams`,`float32x4`,me),m=e.allocate(`tile-tint`,`uint32`,t.cellCount),h=e.allocate(`focus-mask`,`uint32`,t.cellCount);return{outputs:{filledElevation:i.filledElevation,flowDirections:i.flowDirections,accumulation:u,reachIds:d,seams:p,tileTint:m,focusMask:h,tiles:l,assembleTiles:f},update:e=>{i.writeStreamThreshold(e.streamThreshold),m.write(Uint32Array.from({length:t.cellCount},(e,n)=>+(Math.floor(Math.floor(n/t.columns)/o)%2==1)));let n=new Float32Array(16);for(let e=1;e<r;e++){let i=1-e/r;n.set([0,i,t.columns*t.cellSize,i],(e-1)*4)}p.write(n)}}},rebuildOn:[`tiles`],interaction:{kind:`probe`,hint:`hover a reach`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=n.tiles.map(e=>a(e.reachIds)),s=n.tiles.map(e=>a(e.accumulation));if(o.some(e=>!e)||s.some(e=>!e))return{};let c=$(Uint32Array,o,t.cellCount),l=$(Float32Array,s,t.cellCount),u=N(t,i,e.x,e.y),d=-1,f=6;for(let e=-5;e<=5;e++)for(let n=-5;n<=5;n++){let r=u.column+n,i=u.row+e;if(r<0||i<0||r>=t.columns||i>=t.rows)continue;let a=i*t.columns+r,o=Math.hypot(n,e);c[a]!==he&&o<f&&(d=a,f=o)}let p=Math.hypot(e.unitX-.5,e.unitY-.5)<.003;p&&(d=-1);let m=d<0?-1:c[d],h=Math.round(r.tiles),g=t.rows/h,_=Math.min(h-1,Math.floor(u.row/g)),v=`${m}:${_}:${h}`;t.focus.key===v&&ge(t.focus.source,c)||(n.focusMask.write(Uint32Array.from(c,(e,n)=>m>=0&&e===m?1:!p&&Math.floor(Math.floor(n/t.columns)/g)===_?2:0)),t.focus={reach:m,key:v,source:c});let y=1-_*g/t.rows,b=1-(_+1)*g/t.rows;if(p)return{focus:-1,spotlight:null,readout:`${h} tiles`};let x=[{shape:`box`,minimum:[0,b*i.height],maximum:[i.width,y*i.height],dashed:!0}];if(m<0)return{focus:-1,spotlight:null,glyphs:x,readout:`tile ${_+1} of ${h}`};let S=a(n.flowDirections);if(S){let e=F(t,S,d);x.push({shape:`polyline`,points:j(t,i,e)},{shape:`dot`,at:ee(t,i,d)})}let C=1/0,w=-1/0;return c.forEach((e,n)=>{if(e!==m)return;let r=Math.floor(Math.floor(n/t.columns)/g)+1;C=Math.min(C,r),w=Math.max(w,r)}),{focus:m,spotlight:null,glyphs:x,readout:`reach ${m} · ${C===w?`tile ${C}`:`tiles ${C}-${w}`} · accumulation ${Math.round(l[d])}`}}},draw:({inputs:e,outputs:t,values:n,interaction:r})=>{t.assembleTiles();let i=r.focus!==void 0&&r.focus!==-1;return[M(e,t.filledElevation),{layer:`raster`,values:t.tileTint,columns:e.columns,rows:e.rows,categories:[[0,0,0,0],[1,1,1,.1]]},{layer:`raster`,values:t.accumulation,columns:e.columns,rows:e.rows,ramp:`sequential`,range:[0,150],fade:!0,alpha:.7,smoothing:!0},{layer:`raster`,values:t.reachIds,columns:e.columns,rows:e.rows,alpha:i?.35:.9},{layer:`raster`,values:t.focusMask,columns:e.columns,rows:e.rows,categories:[[0,0,0,0],[1,1,1,1],[1,1,1,.14]]},{layer:`lines`,context:!0,segments:t.seams,count:Math.round(n.tiles)-1,radius:1.5,color:[1,1,1,.8]}]}});export{_e as default};