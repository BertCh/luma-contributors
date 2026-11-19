import{d as e,s as t}from"./kernel-C3tfQOn4.js";import{u as n}from"./wgsl-common-DfyXiYB0.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{S as i,f as a,p as o,x as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c,s as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{o as u}from"./raster-algebra-input-5ZxymrFJ.js";import{n as d,r as f,t as p}from"./raster-fractional-tile-kernel-D2BjD3g1.js";import{c as m,n as h,r as g,u as _}from"./raster-band-utils-BcTxc8_Y.js";var v=-34028234663852886e22,y=34028234663852886e22,b=4,x=16,S=256,C=64,w=18,T=4294967295;function E(e,t={}){_(`GPURasterExtremaPyramid`,e);let{width:n,height:r}=e,o=t.firstBlockSize??4;if(!Number.isInteger(o)||o<1||o>S||o&o-1)throw Error(`GPURasterExtremaPyramid firstBlockSize must be a power of two in [1, 256]`);let s=t.footprint??`bilinear`;if(s!==`cell`&&s!==`bilinear`)throw Error(`GPURasterExtremaPyramid footprint must be 'cell' or 'bilinear'`);let c=t.levelCount??1/0;if(c!==1/0&&(!Number.isInteger(c)||c<1))throw Error(`GPURasterExtremaPyramid levelCount must be an integer >= 1`);let l=[],u=0;for(let e=0;e<c;e++){let t=o*2**e,s=Math.ceil(n/t),c=Math.ceil(r/t);if(a(`GPURasterExtremaPyramid`,`level blockSize`,t,1,T),l.push({level:e,blockSize:t,width:s,height:c,offset:u}),u+=s*c,i(`GPURasterExtremaPyramid`,`packed level cells`,u,T),s===1&&c===1)break}return{width:n,height:r,firstBlockSize:o,footprint:s,levels:l,length:u}}function D(e,t,n){let{layout:r,combined:i}=t,{width:a,height:o}=n;if(!r||!i)throw Error(`${e} pyramid must provide a layout and a combined view`);if(r.width!==a||r.height!==o)throw Error(`${e} pyramid covers ${r.width}x${r.height} pixels but the grid is ${a}x${o}`);if(r.footprint!==`bilinear`)throw Error(`${e} pyramid must use the 'bilinear' footprint`);if(r.levels.length<1)throw Error(`${e} pyramid has no levels`);return A(e,`pyramid combined`,i,2*r.length),r}function O(e){return`${Math.fround(e)}`.replace(/^(-?\d+)$/,`$1.0`)}function k(e){return e.replace(/([a-z0-9])([A-Z])/g,`$1_$2`).toUpperCase()}function A(t,n,r,i){if(e(r,[`float32`],`${t} ${n}`),r.length!==i)throw Error(`${t} ${n} must contain ${i} values`)}function j(e,n,r,i){c(e,n,Object.fromEntries(i.map((e,t)=>[`view.${t}`,e])));for(let r of i)r&&d(e.device,n,r.buffer.id,t(r).size,`reduce the packed pyramid size`,r.buffer.byteLength);d(e.device,n,`canonical band`,r.width*r.height*4,`reduce grid dimensions`);let a=Math.ceil((r.levels.length-1)/b);a>0&&d(e.device,n,`fused parameter slot`,72,`reduce the number of pyramid levels`,a*C*4)}var M=class{estimatedCommandNodeCount;id;props;layout;output;constructor(e){this.id=e.id??`raster-extrema-pyramid`,this.props=e;let{id:t}=this;_(t,e.grid),this.layout=E(e.grid,e),u(t,`input`,e.input,e.grid),this.estimatedCommandNodeCount=10+Math.ceil((this.layout.levels.length-1)/b),o(t,this.estimatedCommandNodeCount);let{maximum:n,minimum:r,combined:i}=e.output;if(i&&(n||r))throw Error(`${t} output.combined cannot be used together with maximum or minimum`);if(!n&&!r&&!i)throw Error(`${t} requires at least one output`);for(let[e,i]of[[`maximum`,n],[`minimum`,r]])i&&A(t,`output.${e}`,i,this.layout.length);i&&A(t,`output.combined`,i,2*this.layout.length);let a=e.input,s=a.storage.kind===`buffer`&&a.format===`float32`&&a.validity&&a.scale===void 0&&a.offset===void 0&&a.noDataValue===void 0?h(a):[];f(t,e.output,h(a),[{writes:[n,r,i],reads:s}]),i&&(this.output={layout:this.layout,combined:i})}getCommandNodes(e){let{id:t,props:n,layout:r}=this;m(t,e,n.input,[]);let{maximum:i,minimum:a,combined:o}=n.output;j(e,t,r,[...h(n.input),i,a,o]),s(t,e,[i,a,o]);let c=g(e,t,n.input,n.grid,`reuse-canonical`);return[...c.nodes,...N(e,{id:t,layout:r,values:c.band.storage.values,validity:c.band.validity,maximum:i,minimum:a,combined:o})]}};function N(e,t){let{id:i,layout:a,values:o,validity:s,maximum:c,minimum:u,combined:d}=t;if(j(e,i,a,[o,s,c,u,d]),f(i,{maximum:c,minimum:u,combined:d},[o,s]),d){if(c||u)throw Error(`${i} combined cannot be used together with maximum or minimum`);A(i,`combined`,d,2*a.length)}else if(!c&&!u)throw Error(`${i} requires at least one output`);for(let[e,t]of[[`maximum`,c],[`minimum`,u]])t&&A(i,e,t,a.length);let m=a.width*a.height;if(o.length!==m||s.length!==m)throw Error(`${i} values and validity must contain one value per pixel`);let h=O(v),g=O(y),_=+(a.footprint===`bilinear`),S=d?[{name:`combinedLevels`,view:d,type:`f32`,access:`read_write`}]:[...c?[{name:`maximumLevels`,view:c,type:`f32`,access:`read_write`}]:[],...u?[{name:`minimumLevels`,view:u,type:`f32`,access:`read_write`}]:[]],T=!!(d||c),E=!!(d||u),D=d?`combinedLevels`:`maximumLevels`,k=d?`combinedLevels`:`minimumLevels`,M=d?`combinedLevelsOffset`:`maximumLevelsOffset`,N=d?`(combinedLevelsOffset + ${a.length}u)`:`minimumLevelsOffset`,P=[];for(let t of a.levels){let c=t.width*t.height,l=`const LEVEL_WIDTH: u32 = ${t.width}u;
const LEVEL_OFFSET: u32 = ${t.offset}u;
const EMPTY_MAXIMUM: f32 = ${h};
const EMPTY_MINIMUM: f32 = ${g};`,u=`${T?`${D}[${M} + LEVEL_OFFSET + index] = maximumValue;\n  `:``}${E?`${k}[${N} + LEVEL_OFFSET + index] = minimumValue;`:``}`;t.level===0&&P.push(r(e,{id:`${i}-level-0`,operation:`GPURasterExtremaPyramid`,variant:`level-0`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},{name:`validity`,view:s,type:`u32`,access:`read`},...S],invocationCount:c,declarations:`const WIDTH: u32 = ${a.width}u;
const HEIGHT: u32 = ${a.height}u;
const BLOCK_SIZE: u32 = ${t.blockSize}u;
const FOOTPRINT_EXTRA: u32 = ${_}u;
${l}
${n}`,body:`let cellColumn = index % LEVEL_WIDTH;
  let cellRow = index / LEVEL_WIDTH;
  let x0 = cellColumn * BLOCK_SIZE;
  let y0 = cellRow * BLOCK_SIZE;
  let x1 = min(x0 + BLOCK_SIZE + FOOTPRINT_EXTRA, WIDTH);
  let y1 = min(y0 + BLOCK_SIZE + FOOTPRINT_EXTRA, HEIGHT);
  var maximumValue = EMPTY_MAXIMUM;
  var minimumValue = EMPTY_MINIMUM;
  for (var y = y0; y < y1; y++) {
    for (var x = x0; x < x1; x++) {
      let pixel = y * WIDTH + x;
      if (validity[validityOffset + pixel] != 0u) {
        let value = values[valuesOffset + pixel];
        if (isFiniteValue(value)) {
          if (value > maximumValue) { maximumValue = value; }
          if (value < minimumValue) { minimumValue = value; }
        }
      }
    }
  }
  ${u}`}))}let F=Math.ceil((a.levels.length-1)/b),I=new Uint32Array(F*C);for(let e=0;e<F;e++){let t=e*b,n=a.levels[t],r=a.levels.slice(t+1,t+1+b),i=Math.ceil(n.width/x),o=Math.ceil(n.height/x),s=e*C;I.set([n.width,n.height,n.offset,i,r.length,i*o],s);for(let[e,t]of r.entries())I.set([t.width,t.height,t.offset],s+6+e*3)}let L=F>0?l(e,`${i}-fused-parameters`,I):void 0;L&&P.push(...L.nodes);let R=Array.from({length:b},(e,t)=>{let n=x>>t,r=n>>1,i=6+t*3,a=(e,t,r)=>`for (var child = 0u; child < 4u; child++) {
      let value = ${e}[(2u * stageRow + (child >> 1u)) * ${n}u + 2u * stageColumn + (child & 1u)];
      if (value ${t} ${r}) { ${r} = value; }
    }`;return`{
    let isStageThread = localInvocationIndex < ${r*r}u;
    let stageColumn = localInvocationIndex % ${r}u;
    let stageRow = localInvocationIndex / ${r}u;
    var maximumValue = EMPTY_MAXIMUM;
    var minimumValue = EMPTY_MINIMUM;
    if (isStageThread) {
      ${T?a(`tileMaximum`,`>`,`maximumValue`):``}
      ${E?a(`tileMinimum`,`<`,`minimumValue`):``}
    }
    workgroupBarrier();
    if (isStageThread) {
      ${T?`tileMaximum[localInvocationIndex] = maximumValue;`:``}
      ${E?`tileMinimum[localInvocationIndex] = minimumValue;`:``}
      let cellColumn = tileColumn * ${r}u + stageColumn;
      let cellRow = tileRow * ${r}u + stageRow;
      let levelWidth = parameters[parametersOffset + ${i}u];
      let levelHeight = parameters[parametersOffset + ${i+1}u];
      if (${t}u < parameters[parametersOffset + 4u] && cellColumn < levelWidth && cellRow < levelHeight) {
        let cellIndex = parameters[parametersOffset + ${i+2}u] + cellRow * levelWidth + cellColumn;
        ${T?`${D}[${M} + cellIndex] = maximumValue;`:``}
        ${E?`${k}[${N} + cellIndex] = minimumValue;`:``}
      }
    }
    workgroupBarrier();
  }`}).join(`
`);for(let t=0;t<F;t++){let n=t*b,r=a.levels[n],o=Math.min(n+b,a.levels.length-1),s=Math.ceil(r.width/x)*Math.ceil(r.height/x),c=e.createDataView(L.parameters.buffer,{format:`uint32`,byteOffset:L.parameters.byteOffset+t*C*4,length:w});P.push(p(e,{id:`${i}-levels-${n+1}-${o}`,operation:`GPURasterExtremaPyramid`,bindings:[...S,{name:`parameters`,view:c,type:`u32`,access:`read`}],invocationCount:s*x*x,rowCount:`0u`,workgroupCount:`parameters[parametersOffset + 5u]`,declarations:`const EMPTY_MAXIMUM: f32 = ${h};
const EMPTY_MINIMUM: f32 = ${g};
${T?`var<workgroup> tileMaximum: array<f32, 256>;`:``}
${E?`var<workgroup> tileMinimum: array<f32, 256>;`:``}`,body:`let tilesX = parameters[parametersOffset + 3u];
  let tileColumn = workgroupIndex % tilesX;
  let tileRow = workgroupIndex / tilesX;
  let baseColumn = tileColumn * ${x}u + localInvocationIndex % ${x}u;
  let baseRow = tileRow * ${x}u + localInvocationIndex / ${x}u;
  var baseMaximum = EMPTY_MAXIMUM;
  var baseMinimum = EMPTY_MINIMUM;
  let baseWidth = parameters[parametersOffset];
  if (baseColumn < baseWidth && baseRow < parameters[parametersOffset + 1u]) {
    let baseCell = parameters[parametersOffset + 2u] + baseRow * baseWidth + baseColumn;
    ${T?`baseMaximum = ${D}[${M} + baseCell];`:``}
    ${E?`baseMinimum = ${k}[${N} + baseCell];`:``}
  }
  ${T?`tileMaximum[localInvocationIndex] = baseMaximum;`:`_ = baseMaximum;`}
  ${E?`tileMinimum[localInvocationIndex] = baseMinimum;`:`_ = baseMinimum;`}
  workgroupBarrier();
  ${R}`}))}return P}function P(e,t=`pyramid`){let n=k(t),r=e.levels.length,i=t=>`array<u32, ${r}>(${e.levels.map(e=>`${t(e)}u`).join(`, `)})`;return`const ${n}_LEVEL_COUNT: u32 = ${r}u;
const ${n}_EMPTY_MAXIMUM: f32 = ${O(v)};
const ${n}_EMPTY_MINIMUM: f32 = ${O(y)};
const ${n}_MINIMUM_OFFSET: u32 = ${e.length}u;
const ${n}_BLOCK_SIZES = ${i(e=>e.blockSize)};
const ${n}_LEVEL_WIDTHS = ${i(e=>e.width)};
const ${n}_LEVEL_OFFSETS = ${i(e=>e.offset)};
fn ${t}LevelBlockSize(level: u32) -> u32 {
  return ${n}_BLOCK_SIZES[level];
}
fn ${t}LevelIndex(level: u32, column: u32, row: u32) -> u32 {
  let blockSize = ${n}_BLOCK_SIZES[level];
  return ${n}_LEVEL_OFFSETS[level] + (row / blockSize) * ${n}_LEVEL_WIDTHS[level] + column / blockSize;
}`}export{D as a,P as i,N as n,E as r,M as t};