import{i as e,r as t}from"./cell-dispatch-validation-PIkQyAK_.js";import{v as n}from"./spatial-context-DwZEd4EZ.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-sort-xRm8l5yn.js";import{E as a,f as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as u}from"./bounds-reduction-6MyKJx-7.js";import{m as d,u as f}from"./wgsl-common-DfyXiYB0.js";import{t as p}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as m}from"./define-run-B-brkCe4.js";import{p as h}from"./kernels-IDOsx_br.js";import{dl as g}from"./entry-preview-D70TdSbK.js";var _=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metrics:Object.freeze([`none`,`native`])}),v=`GPUHilbertKeys`,y=2147483647;function b(e){return e>=16?4294967295:4**e}var x=class{id;props;spatialContext;order;itemCount;constructor(t){let r=t.id??`hilbert-keys`;this.id=r,this.props=t,this.spatialContext=n(this.id,t.spatialContext,_),this.order=t.order??15,o(r,`order`,this.order,1,16);let{positions:i,minima:s,maxima:l,bounds:u}=t;if(i){if(s||l)throw Error(`${r} takes positions or minima and maxima, not both`);a(i,[`float32x2`],`${r} positions`),this.itemCount=i.length}else{if(!s||!l)throw Error(`${r} needs positions, or minima and maxima`);if(a(s,[`float32x2`],`${r} minima`),a(l,[`float32x2`],`${r} maxima`),s.length!==l.length)throw Error(`${r} minima and maxima must have the same length`);this.itemCount=s.length}if(o(r,`itemCount`,this.itemCount,1,y),u&&(a(u,[`float32`],`${r} bounds`),u.length<4))throw Error(`${r} bounds must hold 4 float32 values`);let{keys:d,sortedRows:f,sortedKeys:p}=t.output;for(let[e,t]of[[`keys`,d],[`sortedRows`,f],[`sortedKeys`,p]])if(t&&(a(t,[`uint32`],`${r} output.${e}`),t.length!==this.itemCount))throw Error(`${r} output.${e} must have one row per item`);if(p&&!f)throw Error(`${r} output.sortedKeys needs output.sortedRows`);e(r,{keys:d,sortedRows:f,sortedKeys:p},[]),c(r,[d,f,p],[i,s,l,u])}getCommandNodes(e){let{props:n,id:a,order:o,itemCount:c}=this,{positions:m,minima:h,maxima:g}=n,{keys:_,sortedRows:y,sortedKeys:x}=n.output,S=Math.min(256,Math.max(1,Math.ceil(c/4096)));t(e,a,{positions:m,minima:m?void 0:h,maxima:m?void 0:g,bounds:n.bounds,keys:_,sortedRows:y,sortedKeys:x},[...n.bounds?[]:[[`bounds scratch`,16],[`bounds partial scratch`,S>1?S*16:0]],...y?[[`sort scratch`,Math.max(c,1)*4],[`radix histogram scratch`,c>1?16*Math.ceil(c/256)*4:0]]:[]]),s(a,e,[m,h,g,n.bounds,_,y,x]);let C=[],w=m?[{name:`positions`,view:m,type:`f32`,access:`read`}]:[{name:`minima`,view:h,type:`f32`,access:`read`},{name:`maxima`,view:g,type:`f32`,access:`read`}],T=`${f}
// Returns (x, y, valid, 0) for an item: the point, or the box center (halved before the sum to
// avoid overflow near FLT_MAX).
fn readItem(row: u32) -> vec4f {
${m?`  let point = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  let valid = isFiniteValue(point.x) && isFiniteValue(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);`:`  let minimum = vec2f(minima[minimaOffset + row * 2u], minima[minimaOffset + row * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + row * 2u], maxima[maximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  return vec4f(minimum * 0.5 + maximum * 0.5, select(0.0, 1.0, valid), 0.0);`}
}`,E=n.bounds;E||(E=r(e,`${a}-bounds`,`float32`,4),C.push(...u(e,{id:a,operation:v,variant:m?`bounds-points`:`bounds-boxes`,bindings:w,declarations:T,itemCount:c,output:E})));let D=y?r(e,`${a}-rows`,`uint32`,c):void 0;if(C.push(l(e,{id:`${a}-keys`,operation:v,variant:`${m?`points`:`boxes`}-order-${o}${D?`-rows`:``}`,bindings:[...w,{name:`bounds`,view:E,type:`f32`,access:`read`},{name:`keysOut`,view:_,type:`u32`,access:`read_write`},...D?[{name:`rowsOut`,view:D,type:`u32`,access:`read_write`}]:[]],invocationCount:c,declarations:`${T}
${p}
const ORDER: u32 = ${o}u;
const CELLS: u32 = ${2**o}u;

fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let scale = max(abs(minimum), abs(maximum));
  if (!(maximum > minimum) || !isFiniteValue(scale)) { return 0u; }
  // Power-of-two normalization prevents both overflowing differences and reciprocal flushing
  // under Metal fast math. The shared quotient divides only bounded significands.
  let exponent = frexp(scale).exp;
  let lower = ldexp(minimum, -exponent);
  let upper = ldexp(maximum, -exponent);
  let position = ldexp(clamp(value, minimum, maximum), -exponent);
  let normalized = clamp(getScaledQuotient(position - lower, upper - lower, 1.0, 0), 0.0, 1.0);
  return min(u32(floor(normalized * f32(CELLS))), CELLS - 1u);
}

// Spreads the low 16 bits of a value to the even bit positions.
${d}

// The Hilbert curve index of cell (x, y) on a CELLS x CELLS grid, equal to the classic xy2d
// rotation loop. The loop carries a rotate/flip state from the top bit down, so its 16 steps
// are serial and branch on data. The state transform of each bit level is a small affine map over
// GF(2), so the same result comes from a parallel prefix over the levels (Hacker's Delight 16-5,
// as in rawrunprotected's "Hilbert curves in O(log n)"): four branch-free rounds on 16-bit lanes,
// then one interleave.
fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  let x = cellX << (16u - ORDER);
  let y = cellY << (16u - ORDER);
  let a0 = x ^ y;
  let b0 = 0xffffu ^ a0;
  let c0 = 0xffffu ^ (x | y);
  let d0 = x & (y ^ 0xffffu);
  var transformA = a0 | (b0 >> 1u);
  var transformB = (a0 >> 1u) ^ a0;
  var transformC = ((c0 >> 1u) ^ (b0 & (d0 >> 1u))) ^ c0;
  var transformD = ((a0 & (c0 >> 1u)) ^ (d0 >> 1u)) ^ d0;
  for (var shift = 2u; shift <= 4u; shift = shift << 1u) {
    let a = transformA;
    let b = transformB;
    let c = transformC;
    let d = transformD;
    transformA = (a & (a >> shift)) ^ (b & (b >> shift));
    transformB = (a & (b >> shift)) ^ (b & ((a ^ b) >> shift));
    transformC = transformC ^ ((a & (c >> shift)) ^ (b & (d >> shift)));
    transformD = transformD ^ ((b & (c >> shift)) ^ ((a ^ b) & (d >> shift)));
  }
  let a = transformA;
  let b = transformB;
  let c = transformC;
  let d = transformD;
  transformC = c ^ ((a & (c >> 8u)) ^ (b & (d >> 8u)));
  transformD = d ^ ((b & (c >> 8u)) ^ ((a ^ b) & (d >> 8u)));
  let low = transformC ^ (transformC >> 1u);
  let high = transformD ^ (transformD >> 1u);
  let indexLow = x ^ y;
  let indexHigh = high | (0xffffu ^ (indexLow | low));
  return ((spreadBits(indexHigh) << 1u) | spreadBits(indexLow)) >> (32u - 2u * ORDER);
}`,body:`${D?`rowsOut[rowsOutOffset + index] = index;`:``}
  let item = readItem(index);
  var key = ${b(o)}u;
  if (item.z > 0.5) {
    let cellX = quantizeAxis(item.x, bounds[boundsOffset], bounds[boundsOffset + 2u]);
    let cellY = quantizeAxis(item.y, bounds[boundsOffset + 1u], bounds[boundsOffset + 3u]);
    key = getHilbertIndex(cellX, cellY);
  }
  keysOut[keysOutOffset + index] = key;`})),y&&D){let t=r(e,`${a}-sorted-keys`,`uint32`,c),n=r(e,`${a}-sorted-rows`,`uint32`,c);C.push(...new i({id:`${a}-sort`,keys:_,values:D,outputKeys:t,outputValues:n,algorithm:`radix`,keyBits:Math.min(32,2*o+1)}).getCommandNodes(e));for(let[r,i,o]of[[`rows`,n,y],[`keys`,t,x]])o&&C.push(l(e,{id:`${a}-publish-${r}`,operation:v,invocationCount:c,bindings:[{name:`input`,view:i,type:`u32`,access:`read`},{name:`output`,view:o,type:`u32`,access:`read_write`}],body:`output[outputOffset + index] = input[inputOffset + index];`}))}return C}},S=8;function C(e,t){let n=2**e,r=[];for(let e=0;e<n*n;e++){let i=e,a=0,o=0;for(let e=1;e<n;e*=2){let t=1&i>>1,n=1&(i^t);n===0&&(t===1&&(a=e-1-a,o=e-1-o),[a,o]=[o,a]),a+=e*t,o+=e*n,i>>=2}r.push((a+.5)/n*t.width,(o+.5)/n*t.height)}let i=new Float32Array((r.length/2-1)*4);for(let e=0;e<r.length/2-1;e++)i.set(r.slice(e*2,e*2+4),e*4);return i}var w=m({parameters:g,caption:`Points keyed along a Hilbert curve; the line visits them in key order.`,createInputs:({world:e})=>{let t=new Float32Array(128);for(let n=0;n<S;n++)for(let r=0;r<S;r++)t[(n*S+r)*2]=(r+.5)/S*e.width,t[(n*S+r)*2+1]=(n+.5)/S;return{count:64,positions:t,world:e}},build(e,t,n){let r=e.allocate(`positions`,`float32x2`,t.count),i=e.allocate(`keys`,`uint32`,t.count),a=e.allocate(`sorted-rows`,`uint32`,t.count),o=e.upload(`bounds`,`float32`,Float32Array.of(0,0,t.world.width,t.world.height));return e.graph.add(new x({order:n.order,positions:r.view,bounds:o.view,output:{keys:i.view,sortedRows:a.view}})),{outputs:{positions:r,keys:i,sortedRows:a,keyValues:h(e,`keys`,i),curve:e.upload(`curve`,`float32x4`,C(n.order,t.world))},update:e=>r.write(Float32Array.from(t.positions,(t,n)=>{let r=Math.sin(n*12.9898)*43758.5453;return t+(r-Math.floor(r)-.5)*2*e.jitter}))}},rebuildOn:[`order`],interaction:{kind:`time`,hint:`scrub along the curve`,rest:[.95,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=Math.min(1,Math.max(0,(e.unitX-.05)/.9)),o=Math.max(2,Math.round(a*t.count)),s=i(n.sortedRows),c=i(n.keys),l=i(n.positions),u=4**r.order,d=`${o} of ${t.count} points visited`,f=[];if(s&&c&&l){let e=s[o-1];f.push({shape:`ring`,at:[l[e*2],l[e*2+1]],radius:.03}),d=`key ${c[e]} of ${u} · ${o} of ${t.count} points`}return{focus:o,glyphs:f,readout:d}}},draw:({inputs:e,interaction:t,outputs:n,theme:r,values:i})=>[{layer:`lines`,context:!0,segments:n.curve,radius:1.6,color:[...r.muted,.7]},{layer:`paths`,nodes:n.positions,ids:n.sortedRows,count:typeof t?.focus==`number`?t.focus:e.count,radius:1.4},{layer:`dots`,positions:n.positions,values:n.keyValues,ramp:`cyclic`,range:[0,4**i.order-1],radius:5.5}]});export{w as default};