import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{c as t,o as n,r,t as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-reduction-tNgOVQ_C.js";import{D as o,S as s,T as c,f as l,g as u,i as d,p as f}from"./gpu-contributor-utils-BIwLQON5.js";import{n as p,r as m,t as h}from"./pointcloud-grid-index-Bt2ZJi74.js";import{f as g,h as _,i as v,n as y,o as b,r as x,s as S,u as C}from"./raster-utils-CztuVGLY.js";import{a as w,i as T,n as ee}from"./pointcloud-kernel-nodes-YcR7W84x.js";import{n as E,r as D}from"./gpu-pointcloud-surfaces-B2hF70_C.js";import{a as O,r as k}from"./pointcloud-chunk-kernel-nodes-0zmunsXp.js";import{a as A}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as j}from"./define-run-B-brkCe4.js";import{o as M}from"./pointer-overlay-Qh8qflcA.js";import{t as N}from"./convert-CaKqeDF0.js";import{n as P,u as F}from"./helpers-DPAd8TjN.js";import{Sc as I}from"./entry-preview-D70TdSbK.js";import{c as L,t as R}from"./pointcloud-display-run-BsAozfd0.js";var z=class{id;width;height;input;output;outputValidity;mode;operation;radius;requiredHalo;structuringElement;borderMode;borderValue;noDataPolicy;constructor(e){if(this.id=e.id??`gpu-raster-morphology`,this.width=e.width,this.height=e.height,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.mode=e.mode??`grayscale`,this.operation=e.operation,this.radius=e.radius,this.requiredHalo=e.radius,this.structuringElement=e.structuringElement??`square`,this.borderMode=e.borderMode??`clamp`,this.borderValue=e.borderValue??0,this.noDataPolicy=e.noDataPolicy??`propagate`,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count must fit in uint32`);if(this.mode!==`binary`&&this.mode!==`grayscale`)throw Error(`${this.id} morphology mode must be binary or grayscale`);if(this.operation!==`dilate`&&this.operation!==`erode`)throw Error(`${this.id} morphology operation must be dilate or erode`);if(!Number.isSafeInteger(this.radius)||this.radius<0||this.radius>8)throw Error(`${this.id} morphology radius must be an integer from zero through eight`);if(this.structuringElement!==`square`&&this.structuringElement!==`cross`)throw Error(`${this.id} structuring element must be square or cross`);if(![`clamp`,`reflect`,`constant`,`nodata`].includes(this.borderMode))throw Error(`${this.id} border mode must be clamp, reflect, constant, or nodata`);if(!Number.isFinite(this.borderValue))throw Error(`${this.id} constant border value must be finite`);if(v(this.borderValue),this.noDataPolicy!==`propagate`&&this.noDataPolicy!==`ignore`)throw Error(`${this.id} nodata policy must be propagate or ignore`);if(this.input.storage.kind!==`buffer`)throw Error(`${this.id} requires a buffer-backed input band`);let n=C(this.input,this,`${this.id} input`),r=this.mode===`binary`?`uint32`:`float32`;if(this.mode===`binary`&&this.input.format!==`uint32`)throw Error(`${this.id} binary morphology requires uint32 input flags`);if(this.mode===`binary`&&(this.input.scale!==void 0&&this.input.scale!==1||this.input.offset!==void 0&&this.input.offset!==0))throw Error(`${this.id} binary morphology requires identity input calibration`);if(g(this.output,r,t,`${this.id} output`),_(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==n||this.outputValidity.buffer.graph!==n)throw Error(`${this.id} resources must belong to the same graph`);let i=[this.input.storage.values.buffer,...this.input.validity?[this.input.validity.buffer]:[]];if(this.output.buffer===this.outputValidity.buffer||i.includes(this.output.buffer)||i.includes(this.outputValidity.buffer))throw Error(`${this.id} morphology inputs and outputs must use separate buffers`);v(this.input.scale??1),v(this.input.offset??0)}addToGraph(e){let t=U(e,this),r=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.input.validity&&r.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:r,compile:({device:e})=>{let r=[{name:`sourceValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`outputValidity`,type:`storage`,group:0,location:2}];this.input.validity&&r.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let a=new i(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:r}});return{encode:({computePass:e,getBuffer:r})=>{let i={sourceValues:n(this.input.storage.values,r),outputValues:n(this.output,r),outputValidity:n(this.outputValidity,r)};this.input.validity&&(i.sourceValidity=n(this.input.validity,r));let o=i;a.dispatch(e,{bindings:o,x:t[0],y:t[1]})},destroy:()=>a.destroy()}}})}getShaderSource(){let e=8+this.radius*2,n=e*e,r=this.mode===`binary`?`u32`:`f32`,i=this.input.validity?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${t(this.input.validity)}u;`:``,a=[`isFiniteValue(sample)`];this.input.format===`float32`&&a.push(`isFiniteValue(rawSample)`),this.input.validity&&a.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + sourceIndex] != 0u`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&a.push(`rawSample != ${b(this.input.noDataValue,this.input.format)}`);let o=this.mode===`binary`?`select(0.0, 1.0, rawSample != 0u)`:`f32(rawSample) * ${v(this.input.scale??1)} + ${v(this.input.offset??0)}`,s=this.mode===`binary`?this.borderValue===0?0:1:this.borderValue,c=W(this.borderMode,s),l=this.structuringElement===`cross`?`abs(i32(kernelColumn) - RADIUS) + abs(i32(kernelRow) - RADIUS) <= RADIUS`:`true`,u=this.operation===`dilate`?`-3.402823466e+38`:`3.402823466e+38`,d=this.operation===`dilate`?`max(extreme, sample)`:`min(extreme, sample)`,f=this.noDataPolicy===`propagate`?`neighborhoodIsValid = false;`:``,p=this.mode===`binary`?`select(0u, select(0u, 1u, extreme != 0.0), neighborhoodIsValid)`:`select(invalidValue, extreme, neighborhoodIsValid)`,m=this.mode===`grayscale`?`let invalidValue = bitcast<f32>(0x7fc00000u | (pixelIndex & 0u));`:``;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const RADIUS: i32 = ${this.radius}i;
const TILE_WIDTH: u32 = ${e}u;
const TILE_PIXEL_COUNT: u32 = ${n}u;
const SOURCE_OFFSET: u32 = ${t(this.input.storage.values)}u;
const OUTPUT_OFFSET: u32 = ${t(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${t(this.outputValidity)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<${S(this.input.format)}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<${r}>;
@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;
${i}
var<workgroup> neighborhoodValues: array<f32, ${n}>;
var<workgroup> neighborhoodValidity: array<u32, ${n}>;

fn isFiniteValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn reflectCoordinate(value: i32, length: i32) -> i32 {
  if (length <= 1i) { return 0i; }
  let period = (length - 1i) * 2i;
  let reflected = ((value % period) + period) % period;
  return select(reflected, period - reflected, reflected >= length);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localIndex = localId.y * 8u + localId.x;
  let workgroupOrigin = vec2<i32>(
    i32(workgroupId.x * 8u),
    i32(workgroupId.y * 8u)
  );
  for (
    var tileIndex = localIndex;
    tileIndex < TILE_PIXEL_COUNT;
    tileIndex += 64u
  ) {
    var sourceColumn = workgroupOrigin.x + i32(tileIndex % TILE_WIDTH) - RADIUS;
    var sourceRow = workgroupOrigin.y + i32(tileIndex / TILE_WIDTH) - RADIUS;
    var sample = ${v(s)};
    var validSample = false;
    var sampleSource = true;
    if (
      sourceColumn < 0i || sourceColumn >= i32(WIDTH) ||
      sourceRow < 0i || sourceRow >= i32(HEIGHT)
    ) {
      ${c}
    }
    if (sampleSource) {
      let sourceIndex = u32(sourceRow) * WIDTH + u32(sourceColumn);
      let rawSample = sourceValues[SOURCE_OFFSET + sourceIndex];
      sample = ${o};
      validSample = ${a.join(` && `)};
    }
    neighborhoodValues[tileIndex] = sample;
    neighborhoodValidity[tileIndex] = select(0u, 1u, validSample);
  }
  workgroupBarrier();

  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let centerIndex = (localId.y + u32(RADIUS)) * TILE_WIDTH + localId.x + u32(RADIUS);
  var neighborhoodIsValid = neighborhoodValidity[centerIndex] != 0u;
  var extreme = ${u};
  var participatingSamples = 0u;
  for (var kernelRow = 0u; kernelRow < ${this.radius*2+1}u; kernelRow++) {
    for (var kernelColumn = 0u; kernelColumn < ${this.radius*2+1}u; kernelColumn++) {
      if (${l}) {
        let tileIndex = (localId.y + kernelRow) * TILE_WIDTH + localId.x + kernelColumn;
        if (neighborhoodValidity[tileIndex] != 0u) {
          let sample = neighborhoodValues[tileIndex];
          extreme = ${d};
          participatingSamples += 1u;
        } else {
          ${f}
        }
      }
    }
  }
  neighborhoodIsValid = neighborhoodIsValid && participatingSamples != 0u && isFiniteValue(extreme);
  ${m}
  outputValues[OUTPUT_OFFSET + pixelIndex] = ${p};
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, neighborhoodIsValid);
}`}},B=class extends z{constructor(e){let t=e.id??`gpu-raster-dilation`;super(H(e,`dilate`,t))}},V=class extends z{constructor(e){let t=e.id??`gpu-raster-erosion`;super(H(e,`erode`,t))}};function H(e,t,n){return e.mode,{...e,id:n,operation:t}}function U(e,t){let n=[t.input.storage.values,t.output,t.outputValidity,...t.input.validity?[t.input.validity]:[]];for(let r of n){if(r.buffer.graph!==e)throw Error(`${t.id} resources must belong to the target graph`);y(e.device,r,`${t.id} ${r.buffer.id}`)}let r=x(e.device,t.width,t.height,t.id),i=8+t.radius*2;if(i*i*Float32Array.BYTES_PER_ELEMENT*2>e.device.limits.maxComputeWorkgroupStorageSize)throw Error(`${t.id} morphology exceeds the device workgroup storage limit`);if(n.length>e.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${t.id} morphology exceeds the device storage binding count`);return r}function W(e,t){switch(e){case`clamp`:return`sourceColumn = clamp(sourceColumn, 0i, i32(WIDTH) - 1i);
      sourceRow = clamp(sourceRow, 0i, i32(HEIGHT) - 1i);`;case`reflect`:return`sourceColumn = reflectCoordinate(sourceColumn, i32(WIDTH));
      sourceRow = reflectCoordinate(sourceRow, i32(HEIGHT));`;case`constant`:return`sample = ${v(t)};\n      validSample = true;\n      sampleSource = false;`;case`nodata`:return`sampleSource = false;`}}var G=1;function K(e,t,n,i,a,o){let s=[],c=i.width*i.height,l=o===`min`?V:B,u=n,f=a;for(let n=0;f>0;n++){let a=Math.min(f,G),o=r(e,`${t}-${n}`,`float32`,c),p=r(e,`${t}-${n}-validity`,`uint32`,c),m=u;s.push(...d(e,()=>new l({id:`${t}-${n}`,width:i.width,height:i.height,radius:a,structuringElement:`cross`,borderMode:`nodata`,noDataPolicy:`ignore`,input:{id:`${t}-${n}-input`,format:`float32`,storage:{kind:`buffer`,values:m}},output:o,outputValidity:p}).addToGraph(e))),u=o,f-=a}return{nodes:s,output:u}}var q=640,J=10;function Y(e){return 2+e}var X=class{props;id;maximumRadius;cutRadius;estimatedCommandNodeCount;constructor(t){this.props=t,t=e(t,[`parameters`,`id`,`positions`,`classifications`,`selection`,`grid`,`slope`,`scalar`,`threshold`,`window`,`cut`,`groundClass`,`otherClass`,`onlyGround`,`output`]),this.id=t.id??`gpu-pointcloud-smrf`,c(this.id,t.output!==void 0&&t.output!==null,`output must be given`),this.props=t=Object.freeze({...t,output:e(t.output,[`groundSurface`,`groundMask`,`outputClassifications`,`status`])}),D(t.grid,this.id),l(this.id,`grid.width`,t.grid.width,2),l(this.id,`grid.height`,t.grid.height,2),s(this.id,`grid.width squared plus grid.height squared`,t.grid.width**2+t.grid.height**2,2147483646),this.maximumRadius=Math.ceil((t.window??18*t.grid.cellSize)/t.grid.cellSize),this.cutRadius=Math.ceil((t.cut??0)/t.grid.cellSize),l(this.id,`window in cells`,this.maximumRadius,1,64),l(this.id,`cut in cells`,this.cutRadius,0,32),t.parameters&&o(this.id,`parameters`,t.parameters,Y(this.maximumRadius)),u(this.id,`slope`,t.slope??.15,0),u(this.id,`scalar`,t.scalar??1.25,0),u(this.id,`threshold`,t.threshold??.5,0),u(this.id,`cut`,t.cut??0,0),l(this.id,`groundClass`,t.groundClass??2,0),l(this.id,`otherClass`,t.otherClass??1,0),o(this.id,`groundSurface`,t.output.groundSurface,t.grid.width*t.grid.height),o(this.id,`classifications`,t.classifications,t.positions.length),o(this.id,`groundMask`,t.output.groundMask,t.positions.length),o(this.id,`outputClassifications`,t.output.outputClassifications,t.positions.length),o(this.id,`status`,t.output.status,t.positions.length),t.selection&&o(this.id,`selection`,t.selection,t.positions.length),u(this.id,`low-outlier threshold (5 * grid.cellSize)`,5*t.grid.cellSize,0),u(this.id,`maximum slope threshold`,(t.slope??.15)*t.grid.cellSize*this.maximumRadius,0),T(this.id,{positions:t.positions,classifications:t.classifications,selection:t.selection,parameters:t.parameters},{groundSurface:t.output.groundSurface,groundMask:t.output.groundMask,outputClassifications:t.output.outputClassifications,status:t.output.status},[{inputs:{parameters:t.parameters},outputs:{groundSurface:t.output.groundSurface}},{inputs:{positions:t.positions,terrain:t.output.groundSurface,selection:t.selection,parameters:t.parameters},outputs:{groundMask:t.output.groundMask,status:t.output.status}},{inputs:{classifications:t.classifications,groundMask:t.output.groundMask,status:t.output.status,parameters:t.parameters},outputs:{outputClassifications:t.output.outputClassifications}}]);let n=2+4*this.cutRadius+this.maximumRadius+this.maximumRadius*(this.maximumRadius+1)/2;this.estimatedCommandNodeCount=q+n+2*this.maximumRadius,f(this.id,this.estimatedCommandNodeCount)}getCommandNodes(e){let{positions:t,classifications:n,selection:i,grid:o,parameters:s,output:{groundSurface:c,groundMask:l,outputClassifications:u,status:d}}=this.props;A(e,this.id,{positions:t,classifications:n,selection:i,groundSurface:c,groundMask:l,outputClassifications:u,status:d,parameters:s}),m(e,this.id,t,[o.width,o.height,1]),w(e,this.id,{"SMRF raster scratch and morphology validity":o.width*o.height*4,"SMRF reduction totals":4}),O(e,this.id,J,1);let f=o.width*o.height,g=[],_=0,v=(t,n)=>r(e,`${this.id}-${n}-${_++}`,t,f),y=(t,n,r,i,a=f,o)=>{g.push(...(o?k:ee)(e,{...o?{runtimeWords:o}:{},id:`${this.id}-${t}`,length:a,inputs:{...n,...s?{parameters:s}:{}},outputs:r,body:i}))},b=h(e,`${this.id}-raster-index`,t,{origin:[...o.origin,0],dimensions:[o.width,o.height,1],cellSize:o.cellSize},!0,p(o));g.push(...b.nodes);let x=v(`float32`,`minimum`),S=v(`uint32`,`occupied`);y(`minimum`,{positions:t,sortedIndices:b.sortedIndices,offsets:b.offsets,...i?{selection:i}:{}},{minimum:x,occupied:S},`
var elevation = 3.402823e38; var count = 0u;
for (var slot=readOffsets(index);slot<readOffsets(index+1u);slot++) {
  let point=readSortedIndices(slot);
  ${i?`if (readSelection(point) == 0u) { continue; }`:``}
  let position = readPositions(point);
  if (!all(abs(position) <= vec3f(3.402823e38))) { continue; }
  ${E(o)}
  if (inside && cellIndex == index) { elevation = min(elevation,position.z); count++; }
}
writeMinimum(index,select(0.0,elevation,count>0u)); writeOccupied(index,u32(count>0u));`);let C=(t,n,i,s,c)=>{let l=r(e,`${this.id}-${t}-valid-count`,`uint32`,1);g.push(...new a({id:`${this.id}-${t}-valid-count`,input:i,output:l,operation:`sum`}).getCommandNodes(e)),y(t,{surface:n,validity:i,validCount:l},{result:s,availability:c},`
if(readValidCount(0u)==0u){writeResult(index,0.0);writeAvailability(index,0u);return;}
if (readValidity(index) != 0u) { writeResult(index,readSurface(index)); writeAvailability(index,1u); return; }
var ids: array<u32,8>; var distances: array<u32,8>;
for (var slot=0u; slot<8u; slot++) { ids[slot]=0xffffffffu; distances[slot]=0xffffffffu; }
let column = i32(index % ${o.width}u); let row = i32(index / ${o.width}u);
for (var ring=1;ring<${Math.max(o.width,o.height)};ring++) {
 for (var vertical=max(-ring,-row);vertical<=min(ring,${o.height-1}-row);vertical++) {
  for (var horizontal=max(-ring,-column);horizontal<=min(ring,${o.width-1}-column);horizontal++) {
   if(max(abs(horizontal),abs(vertical))!=ring){continue;}
   let candidate=u32(row+vertical)*${o.width}u+u32(column+horizontal);
  if (readValidity(candidate)==0u) { continue; }
  let distance = u32(horizontal*horizontal+vertical*vertical);
  let order = (candidate % ${o.width}u)*${o.height}u+candidate/${o.width}u;
  var insertion = 8u;
  for (var slot=0u; slot<8u; slot++) {
    let previous = ids[slot];
    let previousOrder = (previous % ${o.width}u)*${o.height}u+previous/${o.width}u;
    if (distance<distances[slot] || (distance==distances[slot] && order<previousOrder)) { insertion=slot; break; }
  }
  if (insertion<8u) {
    for (var slot=7u; slot>insertion; slot--) { ids[slot]=ids[slot-1u]; distances[slot]=distances[slot-1u]; }
    ids[insertion]=candidate; distances[insertion]=distance;
  }
  }
 }
 // Every unvisited cell is at least ring+1 away in one coordinate; strict inequality preserves cutoff ties.
 if(ids[7]!=0xffffffffu && u32((ring+1)*(ring+1))>distances[7]){break;}
}
var mean=0.0; var count=0u;
for (var slot=0u; slot<8u; slot++) { if (ids[slot]<${f}u) { count++; mean += (readSurface(ids[slot])-mean)/f32(count); } }
writeResult(index,mean); writeAvailability(index,select(0u,2u,count>0u));`)},T=v(`float32`,`filled`),D=v(`uint32`,`filled-availability`);C(`initial-inpaint`,x,S,T,D);let j=(t,n,r,i)=>{let a=K(e,`${this.id}-${t}`,n,o,r,i);return g.push(...a.nodes),a.output},M=v(`float32`,`inverted`);y(`invert`,{filled:T},{inverted:M},`writeInverted(index,-readFilled(index));`);let N=j(`low-dilate`,j(`low-erode`,M,1,`min`),1,`max`),P=v(`uint32`,`low-mask`);y(`low-mask`,{inverted:M,opening:N},{lowMask:P},`writeLowMask(index,u32(abs(readInverted(index)-readOpening(index))>${Math.fround(5*o.cellSize)}));`);let F=T,I=this.cutRadius?`(index % ${o.width}u) % ${this.cutRadius}u == 0u || (index / ${o.width}u) % ${this.cutRadius}u == 0u`:`false`;if(this.cutRadius){let e=j(`net-dilate`,j(`net-erode`,T,2*this.cutRadius,`min`),2*this.cutRadius,`max`);F=v(`float32`,`net-surface`),y(`cut-net`,{filled:T,opening:e},{result:F},`writeResult(index,select(readFilled(index),readOpening(index),${I}));`)}let L=F,R=F,z=v(`uint32`,`objects`);y(`clear-objects`,{},{objectMask:z},`writeObjectMask(index,0u);`);for(let e=1;e<=this.maximumRadius;e++){R=j(`object-erode-${e}`,R,1,`min`);let t=j(`object-dilate-${e}`,R,e,`max`),n=v(`uint32`,`objects`);y(`object-mask-${e}`,{previous:L,opening:t,objectMask:z},{nextMask:n},`writeNextMask(index,readObjectMask(index)|u32(abs(readPrevious(index)-readOpening(index))>${s?`readParameters(readDispatchParameter(0u))`:`bitcast<f32>(readDispatchParameter(1u))`}));`,f,[e+1,new Uint32Array(new Float32Array([(this.props.slope??.15)*o.cellSize*e]).buffer)[0]]),z=n,L=t}let B=v(`uint32`,`anchors`);y(`anchors`,{objectMask:z,lowMask:P,availability:D},{anchorMask:B},`writeAnchorMask(index,u32(readAvailability(index)>0u && readObjectMask(index)==0u && readLowMask(index)==0u && !(${I})));`);let V=v(`uint32`,`terrain-availability`);C(`terrain-inpaint`,T,B,c,V);let H=v(`float32`,`gradients`);return y(`slope`,{terrain:c},{gradients:H},`
let column = index % ${o.width}u; let row = index / ${o.width}u;
let left = select(column-1u,column,column==0u); let right=min(column+1u,${o.width-1}u);
let bottom=select(row-1u,row,row==0u); let top=min(row+1u,${o.height-1}u);
let horizontal=(readTerrain(row*${o.width}u+right)-readTerrain(row*${o.width}u+left))/(f32(right-left)*${Math.fround(o.cellSize)});
let vertical=(readTerrain(top*${o.width}u+column)-readTerrain(bottom*${o.width}u+column))/(f32(top-bottom)*${Math.fround(o.cellSize)});
writeGradients(index,length(vec2f(horizontal,vertical)));`),y(`classify`,{positions:t,terrain:c,availability:V,gradients:H,...i?{selection:i}:{}},{groundMask:l,status:d},`
writeGroundMask(index,0u);
${i?`if (readSelection(index)==0u) { writeStatus(index,4u); return; }`:``}
let position=readPositions(index);
${E(o)}
if (!inside || !all(abs(position)<=vec3f(3.402823e38))) { writeStatus(index,1u); return; }
if (readAvailability(cellIndex)==0u) { writeStatus(index,2u); return; }
let tolerance=${s?`readParameters(1u)`:Math.fround(this.props.threshold??.5)}+${s?`readParameters(0u)`:Math.fround(this.props.scalar??1.25)}*readGradients(cellIndex);
writeGroundMask(index,u32(abs(position.z-readTerrain(cellIndex))<=tolerance));
let margin=${Math.max(2*this.maximumRadius,4*this.cutRadius,2)}u;
let edge = cell.x<margin || cell.y<margin || cell.x+margin>=${o.width}u || cell.y+margin>=${o.height}u;
writeStatus(index,select(0u,8u,edge || readAvailability(cellIndex)==2u));`,t.length),y(`classes`,{classifications:n,groundMask:l,status:d},{outputClassifications:u},`
let original=readClassifications(index); var result=original;
if (readGroundMask(index)!=0u) { result=${this.props.groundClass??2}u; }
else if ((readStatus(index)&7u)==0u && ${!this.props.onlyGround}) { result=${this.props.otherClass??1}u; }
writeOutputClassifications(index,result);`,t.length),g}},te=.3,Z=9e3,Q=.03,$=[.01,.04,.1,.2,.3];function ne({random:e,world:t}){let n=F(e,t.width,Z),r=Array.from({length:Z},(e,t)=>n.positions[t*3+2]).filter((e,t)=>n.classes[t]===2).sort((e,t)=>e-t);return{cloud:n,width:t.width,surfaceRange:[r[Math.floor(r.length*.02)],r[Math.floor(r.length*.98)]]}}var re=j({parameters:I,caption:`Inpainted ground surface (left of the curtain, hillshaded) and the points classified as ground (orange) or removed (blue) (right). The tile has little intermediate relief, so the window (fixed at 0.3) barely changes the result.`,buildCache:5,createInputs:ne,build(e,t,n){let r=P(t.cloud.width,Q),i=r.width*r.height,a=e.allocate(`ground-surface`,`float32`,i),o=e.allocate(`ground-mask`,`uint32`,Z),s=e.allocate(`status`,`uint32`,Z);e.graph.add(new X({positions:e.upload(`positions`,`float32x3`,t.cloud.positions).view,classifications:e.upload(`classes`,`uint32`,new Uint32Array(Z).fill(1)).view,grid:r,slope:n.slope,window:te,threshold:n.threshold,cut:n.cut,output:{groundSurface:a.view,groundMask:o.view,outputClassifications:e.allocate(`output-classes`,`uint32`,Z).view,status:s.view}}));let c=e.upload(`footprint`,`float32x2`,t.cloud.footprint);return{outputs:{footprint:R(e,`supported-footprint`,c,s,Z,{components:2}),groundMask:o,status:s,surface:N(e,`surface-grid`,a,i)}}},rebuildOn:[`slope`,`threshold`,`cut`],interaction:{kind:`swipe`,hint:`swipe surface and points, height sets threshold`,rest:[.5,.95-.27],resolve(e,{outputs:t,peek:n}){let r=M((.95-e.unitY)/.9,0,.9999),i=Math.floor(r*$.length),a=n(t.groundMask),o=n(t.status),s=L(a,o);return{curtain:e.x,settle:{threshold:$[i]},glyphs:[{shape:`curtain`,x:e.x}],readout:`threshold ${$[i]} · ground ${o?s.ground:`…`} of ${s.supported} supported · ${s.incomplete} incomplete/unsupported (hidden); raster is provisional`}}},draw:({inputs:e,outputs:t,interaction:n,world:r})=>{let i=P(e.width,Q),a=n.curtain??r.width/2;return[{layer:`raster`,values:t.surface,columns:i.width,rows:i.height,range:e.surfaceRange,hillshade:.8,smoothing:!0,clip:{side:`left`,x:a}},{layer:`dots`,positions:t.footprint,values:t.groundMask,radius:2.4,clip:{side:`right`,x:a}}]}});export{re as default};