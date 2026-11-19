import{Q as e}from"./dist-BusamGlx.js";import{r as t}from"./kernel-C3tfQOn4.js";import{c as n,o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,S as a,d as o,f as s,u as c,x as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./masked-weak-components-j3xMrDf7.js";import{i as d,l as f}from"./terrain-processing-utils-D-TSF_vl.js";import{c as p,l as m,n as h,r as ee,t as te,u as g}from"./raster-band-utils-BcTxc8_Y.js";import{t as _}from"./define-run-B-brkCe4.js";import{i as v,m as y,o as b}from"./terrain-Cc9dW1uq.js";import{A as x}from"./entry-preview-D70TdSbK.js";var S={jumpCount:0,repairedPixelCount:1,shiftedComponentCount:2,remainingJumpCount:3,converged:4},C=4294967295;function ne(e){return c(e,32)}var re=127,w=16,T=8,E=0,D=1,O=2,k=4,A=`0xffffffffu`;function j(e,t){let n=Math.round((e??t)*100);return Math.abs((e??t)*100-n)<1e-9?n:void 0}var M=class{id;props;maximumComponentIterations;agreementPercent;maximumComponentPercent;ownedBuffers;constructor(e){this.id=e.id??`terrain-spike-repair`,this.props=e;let{id:t}=this,{output:n}=e,r=g(t,e.grid),a=e.step??256,o=e.jump??200,c=e.tolerance??40;if(!Number.isFinite(a)||a<=0)throw Error(`${t} step must be finite and positive`);if(!Number.isFinite(o)||o<=0)throw Error(`${t} jump must be finite and positive`);if(!Number.isFinite(c)||c<=0||c>=a/2)throw Error(`${t} tolerance must be positive and less than step / 2`);let l=j(e.agreement,.8);if(l===void 0||l<=50||l>100)throw Error(`${t} agreement must be a multiple of 0.01 in (0.5, 1]`);let u=j(e.maximumComponentFraction,.25);if(u===void 0||u<1||u>100)throw Error(`${t} maximumComponentFraction must be a multiple of 0.01 in (0, 1]`);this.agreementPercent=l,this.maximumComponentPercent=u,this.maximumComponentIterations=e.maximumComponentIterations??32,s(t,`maximumComponentIterations`,this.maximumComponentIterations,1,1024);for(let[e,a,o]of[[`values`,n.values,`float32`],[`outputValidity`,n.outputValidity,`uint32`],[`labels`,n.labels,`uint32`]])if(a&&(i(a,[o],`${t} output.${e}`),a.length!==r))throw Error(`${t} output.${e} must contain one value per pixel`);if(n.statistics&&(i(n.statistics,[`uint32`],`${t} output.statistics`),n.statistics.length<5))throw Error(`${t} output.statistics must contain at least 5 rows`);f(t,{converged:n.status?.converged,iterationLimitReached:n.status?.iterationLimitReached}),m(t,[n.values,n.outputValidity,n.labels,n.statistics,n.status?.converged,n.status?.iterationLimitReached],h(e.elevation))}getCommandNodes(i){let{id:s,props:c}=this,{output:f}=c,{width:m,height:h}=c.grid,g=m*h;p(s,i,c.elevation,[]),l(s,i,[f.values,f.outputValidity,f.labels,f.statistics,f.status?.converged,f.status?.iterationLimitReached]),a(s,`grid cells`,g,ne(i.device));let _=T*g,v=`GPUTerrainSpikeRepair`,y=c.step??256,b=c.jump??200,x=c.tolerance??40;if(!this.ownedBuffers){let t=e.STORAGE|e.COPY_SRC|e.COPY_DST,n=(e,n)=>i.device.createBuffer({id:`${s}-${e}`,byteLength:Math.max(n,1)*4,usage:t});this.ownedBuffers={offsets:n(`offsets`,g+1),neighbors:n(`neighbors`,2*g),labels:n(`labels`,g),converged:n(`converged`,1)}}let j=this.ownedBuffers,M=o(i,`${s}-offsets`,j.offsets,`uint32`,g+1),N=o(i,`${s}-neighbors`,j.neighbors,`uint32`,2*g),P=o(i,`${s}-labels`,j.labels,`uint32`,g),F=o(i,`${s}-converged`,j.converged,`uint32`,1),I=ee(i,s,c.elevation,c.grid,!0),L=[...I.nodes],R=I.band.storage.values,z=I.band.validity,B=(e,n)=>t(i,`${s}-${e}`,`uint32`,n),V=B(`component-size`,g),H=B(`seam-count`,g),U=B(`vote-count`,g),W=B(`agree-count`,g),G=B(`bit-count`,_),K=B(`scalars`,k),q=B(`shift-key`,g),J=f.statistics??B(`statistics`,5),Y=S,X=`const WIDTH: u32 = ${m}u;
const HEIGHT: u32 = ${h}u;
const PIXEL_COUNT: u32 = ${g}u;
const NO_VERTEX: u32 = ${A};
const JUMP: f32 = ${n(b)};
const STEP: f32 = ${n(y)};
const INVERSE_STEP: f32 = ${n(1/y)};
const TOLERANCE: f32 = ${n(x)};
const BIT_COUNT: u32 = ${T}u;
${te}`,Z=`
fn isValidPixel(pixel: u32) -> bool {
  return validity[validityOffset + pixel] != 0u && isFiniteValue(values[valuesOffset + pixel]);
}
fn getNeighbor(pixel: u32, direction: u32) -> u32 {
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  if (direction == 0u) { return select(NO_VERTEX, pixel + 1u, column + 1u < WIDTH); }
  if (direction == 1u) { return select(NO_VERTEX, pixel - 1u, column > 0u); }
  if (direction == 2u) { return select(NO_VERTEX, pixel + WIDTH, row + 1u < HEIGHT); }
  return select(NO_VERTEX, pixel - WIDTH, row > 0u);
}`,Q=`
// Biased step multiple k + 128 in 1..255 when the height difference votes, else 0.
fn getVoteKey(difference: f32) -> u32 {
  let multiple = round(difference * INVERSE_STEP);
  if (multiple == 0.0 || abs(multiple) > ${re}.0) { return 0u; }
  if (!(abs(difference - multiple * STEP) < TOLERANCE)) { return 0u; }
  return u32(i32(multiple) + 128);
}`;L.push(r(i,{id:`${s}-clear`,operation:v,variant:`clear`,bindings:[{name:`componentSize`,view:V,type:`u32`,access:`read_write`},{name:`seamCount`,view:H,type:`u32`,access:`read_write`},{name:`voteCount`,view:U,type:`u32`,access:`read_write`},{name:`agreeCount`,view:W,type:`u32`,access:`read_write`},{name:`bitCount`,view:G,type:`u32`,access:`read_write`},{name:`scalars`,view:K,type:`u32`,access:`read_write`},{name:`statistics`,view:J,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`const PIXEL_COUNT: u32 = ${g}u;`,body:`bitCount[bitCountOffset + index] = 0u;
  if (index < PIXEL_COUNT) {
    componentSize[componentSizeOffset + index] = 0u;
    seamCount[seamCountOffset + index] = 0u;
    voteCount[voteCountOffset + index] = 0u;
    agreeCount[agreeCountOffset + index] = 0u;
  }
  if (index < ${k}u) {
    scalars[scalarsOffset + index] = select(0u, 0xffffffffu, index == ${D}u);
  }
  if (index < 5u) {
    statistics[statisticsOffset + index] = 0u;
  }`})),L.push(r(i,{id:`${s}-graph`,operation:v,variant:`graph`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`offsets`,view:M,type:`u32`,access:`read_write`},{name:`neighbors`,view:N,type:`u32`,access:`read_write`}],invocationCount:g+1,declarations:`${X}${Z}
fn getEdge(pixel: u32, direction: u32) -> u32 {
  let neighbor = getNeighbor(pixel, direction);
  if (neighbor == NO_VERTEX || !isValidPixel(pixel) || !isValidPixel(neighbor)) {
    return NO_VERTEX;
  }
  let difference = values[valuesOffset + pixel] - values[valuesOffset + neighbor];
  return select(NO_VERTEX, neighbor, abs(difference) <= JUMP);
}`,body:`offsets[offsetsOffset + index] = 2u * index;
  if (index < PIXEL_COUNT) {
    neighbors[neighborsOffset + 2u * index] = getEdge(index, 0u);
    neighbors[neighborsOffset + 2u * index + 1u] = getEdge(index, 2u);
  }`})),L.push(...u(i,{id:`${s}-components`,operation:v,nodeCount:g,offsets:M,neighbors:N,labels:P,converged:F,iterations:this.maximumComponentIterations})),L.push(r(i,{id:`${s}-sizes`,operation:v,variant:`sizes`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`labels`,view:P,type:`u32`,access:`read`},{name:`componentSize`,view:V,type:`atomic<u32>`,access:`read_write`},{name:`statistics`,view:J,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(g/w),declarations:`${X}${Z}
const RUN_LENGTH: u32 = ${w}u;`,body:`var runLabel = NO_VERTEX;
  var runLength = 0u;
  var jumps = 0u;
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label < PIXEL_COUNT) {
      if (label == runLabel) {
        runLength++;
      } else {
        if (runLength > 0u) {
          atomicAdd(&componentSize[componentSizeOffset + runLabel], runLength);
        }
        runLabel = label;
        runLength = 1u;
      }
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction += 2u) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
          abs(value - values[valuesOffset + neighbor]) > JUMP) {
        jumps++;
      }
    }
  }
  if (runLength > 0u) {
    atomicAdd(&componentSize[componentSizeOffset + runLabel], runLength);
  }
  if (jumps > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${Y.jumpCount}u], jumps);
  }`})),L.push(r(i,{id:`${s}-main-size`,operation:v,variant:`main-size`,bindings:[{name:`componentSize`,view:V,type:`u32`,access:`read`},{name:`scalars`,view:K,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,body:`let size = componentSize[componentSizeOffset + index];
  if (size > 0u) {
    atomicMax(&scalars[scalarsOffset + ${E}u], size);
    // Component sizes partition the valid pixels, so their sum is the valid pixel count: one
    // atomic per component instead of one per pixel on a single address.
    atomicAdd(&scalars[scalarsOffset + ${O}u], size);
  }`}),r(i,{id:`${s}-main-label`,operation:v,variant:`main-label`,bindings:[{name:`componentSize`,view:V,type:`u32`,access:`read`},{name:`scalars`,view:K,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,body:`let size = componentSize[componentSizeOffset + index];
  if (size > 0u && size == atomicLoad(&scalars[scalarsOffset + ${E}u])) {
    atomicMin(&scalars[scalarsOffset + ${D}u], index);
  }`})),L.push(r(i,{id:`${s}-votes`,operation:v,variant:`votes`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`labels`,view:P,type:`u32`,access:`read`},{name:`seamCount`,view:H,type:`atomic<u32>`,access:`read_write`},{name:`voteCount`,view:U,type:`atomic<u32>`,access:`read_write`},{name:`bitCount`,view:G,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(g/w),declarations:`${X}${Z}${Q}
const RUN_LENGTH: u32 = ${w}u;
var<private> runBits: array<u32, ${T}>;
var<private> runSeams: u32;
var<private> runVotes: u32;
// Adds the counters of the finished label run, one atomic per non-zero counter.
fn flushRun(label: u32) {
  if (runSeams > 0u) {
    atomicAdd(&seamCount[seamCountOffset + label], runSeams);
  }
  if (runVotes > 0u) {
    atomicAdd(&voteCount[voteCountOffset + label], runVotes);
    for (var bit = 0u; bit < BIT_COUNT; bit++) {
      if (runBits[bit] > 0u) {
        atomicAdd(&bitCount[bitCountOffset + label * BIT_COUNT + bit], runBits[bit]);
      }
    }
  }
  runSeams = 0u;
  runVotes = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    runBits[bit] = 0u;
  }
}`,body:`var runLabel = NO_VERTEX;
  runSeams = 0u;
  runVotes = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    runBits[bit] = 0u;
  }
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label >= PIXEL_COUNT) {
      continue;
    }
    if (label != runLabel) {
      if (runLabel != NO_VERTEX) {
        flushRun(runLabel);
      }
      runLabel = label;
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction++) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
          labels[labelsOffset + neighbor] == label) {
        continue;
      }
      runSeams++;
      let key = getVoteKey(value - values[valuesOffset + neighbor]);
      if (key == 0u) {
        continue;
      }
      runVotes++;
      for (var bit = 0u; bit < BIT_COUNT; bit++) {
        runBits[bit] += (key >> bit) & 1u;
      }
    }
  }
  if (runLabel != NO_VERTEX) {
    flushRun(runLabel);
  }`}));let $=`
// Majority-bit candidate key of a component, 0 when no bit has a majority.
fn getCandidateKey(label: u32) -> u32 {
  let votes = voteCount[voteCountOffset + label];
  var key = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    if (bitCount[bitCountOffset + label * BIT_COUNT + bit] * 2u > votes) {
      key |= 1u << bit;
    }
  }
  return key;
}`;return L.push(r(i,{id:`${s}-agree`,operation:v,variant:`agree`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`labels`,view:P,type:`u32`,access:`read`},{name:`voteCount`,view:U,type:`u32`,access:`read`},{name:`bitCount`,view:G,type:`u32`,access:`read`},{name:`agreeCount`,view:W,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(g/w),declarations:`${X}${Z}${Q}${$}
const RUN_LENGTH: u32 = ${w}u;`,body:`var runLabel = NO_VERTEX;
  var candidate = 0u;
  var runAgree = 0u;
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label >= PIXEL_COUNT) {
      continue;
    }
    if (label != runLabel) {
      if (runAgree > 0u) {
        atomicAdd(&agreeCount[agreeCountOffset + runLabel], runAgree);
      }
      runAgree = 0u;
      runLabel = label;
      candidate = getCandidateKey(label);
    }
    if (candidate == 0u) {
      continue;
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction++) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
          labels[labelsOffset + neighbor] == label) {
        continue;
      }
      if (getVoteKey(value - values[valuesOffset + neighbor]) == candidate) {
        runAgree++;
      }
    }
  }
  if (runAgree > 0u) {
    atomicAdd(&agreeCount[agreeCountOffset + runLabel], runAgree);
  }`})),L.push(r(i,{id:`${s}-decide`,operation:v,variant:`decide`,bindings:[{name:`componentSize`,view:V,type:`u32`,access:`read`},{name:`seamCount`,view:H,type:`u32`,access:`read`},{name:`voteCount`,view:U,type:`u32`,access:`read`},{name:`agreeCount`,view:W,type:`u32`,access:`read`},{name:`bitCount`,view:G,type:`u32`,access:`read`},{name:`scalars`,view:K,type:`u32`,access:`read`},{name:`converged`,view:F,type:`u32`,access:`read`},{name:`shiftKey`,view:q,type:`u32`,access:`read_write`}],invocationCount:g,declarations:`const PIXEL_COUNT: u32 = ${g}u;
const BIT_COUNT: u32 = ${T}u;
const AGREEMENT_PERCENT: u32 = ${this.agreementPercent}u;
const MAXIMUM_COMPONENT_PERCENT: u32 = ${this.maximumComponentPercent}u;${$}
// Exact 64-bit product of a u32 and a factor below 2^16, as (low, high) words: the percentage
// tests below would wrap in u32 for components or valid counts above about 42.9 million pixels.
fn getWideProduct(value: u32, factor: u32) -> vec2<u32> {
  let low = (value & 0xffffu) * factor;
  let high = (value >> 16u) * factor;
  let lowWord = low + (high << 16u);
  return vec2<u32>(lowWord, (high >> 16u) + select(0u, 1u, lowWord < low));
}
fn isWideLessOrEqual(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.y < right.y || (left.y == right.y && left.x <= right.x);
}`,body:`let size = componentSize[componentSizeOffset + index];
  let seams = seamCount[seamCountOffset + index];
  let key = getCandidateKey(index);
  let validCount = scalars[scalarsOffset + ${O}u];
  let mainLabel = scalars[scalarsOffset + ${D}u];
  let shift = converged[convergedOffset] != 0u && size > 0u && index != mainLabel &&
    isWideLessOrEqual(getWideProduct(size, 100u), getWideProduct(validCount, MAXIMUM_COMPONENT_PERCENT)) &&
    seams > 0u && key != 0u &&
    isWideLessOrEqual(getWideProduct(seams, AGREEMENT_PERCENT), getWideProduct(agreeCount[agreeCountOffset + index], 100u));
  shiftKey[shiftKeyOffset + index] = select(0u, key, shift);`})),L.push(r(i,{id:`${s}-shift`,operation:v,variant:`shift`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`labels`,view:P,type:`u32`,access:`read`},{name:`shiftKey`,view:q,type:`u32`,access:`read`},{name:`converged`,view:F,type:`u32`,access:`read`},{name:`valuesOut`,view:f.values,type:`f32`,access:`read_write`},{name:`validityOut`,view:f.outputValidity,type:`u32`,access:`read_write`},{name:`statistics`,view:J,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:`${X}${Z}`,body:`if (index == 0u) {
    atomicStore(&statistics[statisticsOffset + ${Y.converged}u],
      select(0u, 1u, converged[convergedOffset] != 0u));
  }
  if (!isValidPixel(index)) {
    valuesOut[valuesOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    validityOut[validityOutOffset + index] = 0u;
    return;
  }
  if (converged[convergedOffset] == 0u) {
    // Labelling stopped early: nothing is repaired and the pixel is marked, not copied.
    var infinityBits = 0x7f800000u;
    valuesOut[valuesOutOffset + index] = bitcast<f32>(infinityBits);
    validityOut[validityOutOffset + index] = ${C}u;
    return;
  }
  validityOut[validityOutOffset + index] = 1u;
  var value = values[valuesOffset + index];
  let label = labels[labelsOffset + index];
  var key = 0u;
  if (label < PIXEL_COUNT) {
    key = shiftKey[shiftKeyOffset + label];
  }
  if (key != 0u) {
    value = value - STEP * f32(i32(key) - 128);
    atomicAdd(&statistics[statisticsOffset + ${Y.repairedPixelCount}u], 1u);
    if (label == index) {
      atomicAdd(&statistics[statisticsOffset + ${Y.shiftedComponentCount}u], 1u);
    }
  }
  valuesOut[valuesOutOffset + index] = value;`})),f.labels&&L.push(r(i,{id:`${s}-publish-labels`,operation:v,variant:`labels`,bindings:[{name:`values`,view:R,type:`f32`,access:`read`},{name:`validity`,view:z,type:`u32`,access:`read`},{name:`labels`,view:P,type:`u32`,access:`read`},{name:`converged`,view:F,type:`u32`,access:`read`},{name:`labelsOut`,view:f.labels,type:`u32`,access:`read_write`}],invocationCount:g,declarations:`${X}${Z}`,body:`let unresolved = converged[convergedOffset] == 0u && isValidPixel(index);
  labelsOut[labelsOutOffset + index] =
    select(labels[labelsOffset + index], ${C}u, unresolved);`})),L.push(r(i,{id:`${s}-recount`,operation:v,variant:`recount`,bindings:[{name:`values`,view:f.values,type:`f32`,access:`read`},{name:`validity`,view:f.outputValidity,type:`u32`,access:`read`},{name:`statistics`,view:J,type:`atomic<u32>`,access:`read_write`}],invocationCount:g,declarations:`${X}${Z}`,body:`if (!isValidPixel(index)) {
    return;
  }
  let value = values[valuesOffset + index];
  var count = 0u;
  for (var direction = 0u; direction < 4u; direction += 2u) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
        abs(value - values[valuesOffset + neighbor]) > JUMP) {
      count++;
    }
  }
  if (count > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${Y.remainingJumpCount}u], count);
  }`})),L.push(r(i,{id:`${s}-recount-unresolved`,operation:v,variant:`recount-unresolved`,bindings:[{name:`converged`,view:F,type:`u32`,access:`read`},{name:`statistics`,view:J,type:`u32`,access:`read_write`}],invocationCount:1,body:`if (converged[convergedOffset] == 0u) {
    statistics[statisticsOffset + ${Y.remainingJumpCount}u] =
      statistics[statisticsOffset + ${Y.jumpCount}u];
  }`}),...d(i,{id:s,operation:v,source:F,ports:[{name:`converged`,view:f.status?.converged,expression:`source[sourceOffset]`},{name:`iterationLimitReached`,view:f.status?.iterationLimitReached,expression:`1u - source[sourceOffset]`}]})),L}destroy(){if(this.ownedBuffers){for(let e of Object.values(this.ownedBuffers))e.destroy();this.ownedBuffers=void 0}}},N=30,P=200,F=40,I=_({parameters:x,caption:`A damaged elevation tile in; the repaired tile out. Scenario: each damaged block is offset by one step plus up to 30 m, drawn at random; jump (200 m) and tolerance (40 m) are fixed.`,createInputs:({random:e,world:t})=>{let n=v(t,48),r=Array.from({length:12},()=>({column:Math.floor(e()*(n.columns-10)),row:Math.floor(e()*(n.rows-10)),width:3+Math.floor(e()*6),height:3+Math.floor(e()*6),offset:(e()<.5?-1:1)*(256+(e()*2-1)*N)}));return{...n,blocks:r}},build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=y(e,t),s=e.allocate(`repaired`,`float32`,o),c=e.allocate(`output-validity`,`uint32`,o),l=e.allocate(`statistics`,`uint32`,5);return e.graph.add(new M({grid:r,elevation:i,step:256,jump:P,tolerance:F,maximumComponentIterations:32,output:{values:s.view,outputValidity:c.view,statistics:l.view}})),{outputs:{elevation:a,repaired:s},update:e=>{let n=Float32Array.from(t.elevation);for(let r of t.blocks.slice(0,e.spikeCount))for(let e=r.row;e<r.row+r.height;e++)for(let i=r.column;i<r.column+r.width;i++)n[e*t.columns+i]+=r.offset;a.write(n)}}},interaction:{kind:`swipe`,hint:`swipe damaged and repaired`,rest:[.5,.5],resolve:(e,{values:t})=>({curtain:e.x,glyphs:[{shape:`curtain`,x:e.x}],readout:`damaged | repaired · ${t.spikeCount} blocks shifted`})},draw:({inputs:e,outputs:t,interaction:n,world:r})=>{let i=n.curtain??r.width/2,a={range:[650,2300],hillshade:.6,elevation:60};return[b(e,t.elevation,{...a,clip:{side:`left`,x:i}}),b(e,t.repaired,{...a,clip:{side:`right`,x:i}})]}});export{I as default};