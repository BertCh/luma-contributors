import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{o as t,t as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-dispatch-utils-CbRzFAgT.js";import{E as i,S as a,T as o,f as s,p as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{i as u,r as d,v as f}from"./spatial-context-DwZEd4EZ.js";import{i as p,n as m,o as h,r as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as _}from"./wgsl-common-DfyXiYB0.js";import{n as v,t as y}from"./cell-keys-Dl1tIQ_Z.js";import{r as b,t as x}from"./coastline-shKuV2bs.js";var S=`
${y}
${v}
// asin(2/3) in degrees: the latitude where the equatorial belt ends.
const CELL_INDEX_HEALPIX_BELT_LATITUDE: f32 = 41.810314895778596;

fn cellIndexHealpixNestedFromLngLat(longitude: f32, latitude: f32, order: u32) -> vec2u {
  let nside = f32(1u << order);
  let sideMask = (1u << order) - 1u;
  let wrapped = longitude - 360.0 * floor(longitude / 360.0);
  var tt = wrapped / 90.0;
  if (tt >= 4.0) {
    tt = 0.0;
  }
  let clamped = clamp(latitude, -90.0, 90.0);
  var face = 0u;
  var ix = 0u;
  var iy = 0u;
  if (abs(clamped) <= CELL_INDEX_HEALPIX_BELT_LATITUDE) {
    let z = sin(radians(clamped));
    let temp1 = nside * (0.5 + tt);
    let temp2 = nside * z * 0.75;
    let jp = u32(max(temp1 - temp2, 0.0));
    let jm = u32(max(temp1 + temp2, 0.0));
    let ifp = jp >> order;
    let ifm = jm >> order;
    if (ifp == ifm) {
      face = ifp | 4u;
    } else if (ifp < ifm) {
      face = ifp;
    } else {
      face = ifm + 8u;
    }
    ix = jm & sideMask;
    iy = sideMask - (jp & sideMask);
  } else {
    var ntt = u32(tt);
    ntt = min(ntt, 3u);
    let tp = tt - f32(ntt);
    let halfColatitude = sin(radians(90.0 - abs(clamped)) * 0.5);
    let tmp = nside * sqrt(6.0 * halfColatitude * halfColatitude);
    let jp = min(u32(tp * tmp), sideMask);
    let jm = min(u32((1.0 - tp) * tmp), sideMask);
    if (clamped >= 0.0) {
      face = ntt;
      ix = sideMask - jm;
      iy = sideMask - jp;
    } else {
      face = ntt + 8u;
      ix = jp;
      iy = jm;
    }
  }
  let path = quadbinGetCompactKey(ix, iy);
  return path | cellShiftLeft(vec2u(0u, face), 2u * order);
}

/** The nested index plus \`4^(order + 1)\`: the UNIQ scheme, never zero and self-describing. */
fn cellIndexHealpixUniq(nested: vec2u, order: u32) -> vec2u {
  let marker = cellShiftLeft(vec2u(0u, 1u), 2u * order + 2u);
  let low = nested.y + marker.y;
  let carry = select(0u, 1u, low < nested.y);
  return vec2u(nested.x + marker.x + carry, low);
}
`,C=256,w=4294967295,T=16777216,E=`GPUHealpixPointToCell`,D=class{id;props;spatialContext;estimatedCommandNodeCount;chunkPlan;constructor(e){this.id=e.id??`healpix-point-to-cell`,this.props=e;let t=this.id;this.spatialContext=f(t,e.spatialContext??d,u),s(t,`order`,e.order,0,29),o(t,e.scheme===void 0||[`nested`,`uniq`].includes(e.scheme),`scheme must be 'nested' or 'uniq'`),this.chunkPlan=g(t,{positions:e.positions,mask:e.mask,outputCells:e.output.cells,validity:e.output.validity}),s(t,`positions row count`,this.chunkPlan.totalLength);for(let e of this.chunkPlan.chunks)e.views.positions&&i(e.views.positions,[`float32x2`],`${t} positions chunk ${e.chunkIndex}`),a(t,`positions chunk components`,e.length*2,w),e.views.mask&&i(e.views.mask,[`uint32`],`${t} mask chunk ${e.chunkIndex}`),a(t,`mask chunk components`,e.length,w),e.views.outputCells&&i(e.views.outputCells,[`uint32x2`],`${t} outputCells chunk ${e.chunkIndex}`),a(t,`outputCells chunk components`,e.length*2,w),e.views.validity&&i(e.views.validity,[`uint32`],`${t} validity chunk ${e.chunkIndex}`),a(t,`validity chunk components`,e.length,w);p(t,e.output),this.estimatedCommandNodeCount=+(this.chunkPlan.chunks.length>0)+this.chunkPlan.nonemptyChunkCount,c(t,this.estimatedCommandNodeCount),l(t,[e.output.cells,e.output.validity],[e.positions,e.mask])}getCommandNodes(i){let{id:o,props:s}=this,{output:c}=s,l=s.scheme===`uniq`;a(o,`storage bindings`,2+Number(!!s.mask)+Object.values(s.output).filter(Boolean).length,i.device.limits.maxStorageBuffersPerShaderStage),h(i,o,this.chunkPlan);let u=this.chunkPlan.chunks.map(e=>r(o,Math.max(e.length*1,1),C,i.device.limits.maxComputeWorkgroupsPerDimension)),{slots:d,nodes:f}=m(i,{id:`${o}-chunks`,plan:this.chunkPlan}),p,g=this.chunkPlan.chunks.filter(e=>e.length>0).map(r=>{let i={id:`${o}-keys`,operation:E,variant:`healpix-${l?`uniq`:`nested`}`,bindings:[{name:`chunkParameters`,view:d[r.chunkIndex],type:`u32`,access:`read`},{name:`positions`,view:r.views.positions,type:`f32`,access:`read`},...s.mask?[{name:`rowMask`,view:r.views.mask,type:`u32`,access:`read`}]:[],{name:`cellsOut`,view:r.views.outputCells,type:`u32`,access:`read_write`},...c.validity?[{name:`validityOut`,view:r.views.validity,type:`u32`,access:`read_write`}]:[]],invocationCount:r.length*1,declarations:`const ORDER: u32 = ${s.order}u;
${S}
${_}`,body:`let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${s.mask?`isValid = isValid && rowMask[rowMaskOffset + index] != 0u;`:``}
  var key = vec2u(0u);
  if (isValid) {
    key = cellIndexHealpixNestedFromLngLat(longitude, latitude, ORDER);
    ${l?`key = cellIndexHealpixUniq(key, ORDER);`:``}
  }
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${c.validity?`validityOut[validityOutOffset + index] = select(0u, 1u, isValid);`:``}`},a=i.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),f={positions:2,rowMask:1,cellsOut:2,validityOut:1},m=i.bindings.filter(e=>e.name!==`chunkParameters`).map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - rowCount * ${f[e.name]}u;`).join(`
  `),h=p??`${a}
${i.declarations??``}
@compute @workgroup_size(${C})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${T}u) { return; }
  let index = workgroupIndex * ${C}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * 1u) { return; }
  ${m}
  ${i.body}
}`;p=h;let g=i.bindings,v=i.operation,y=u[r.chunkIndex];return e({id:`${i.id}-${r.chunkIndex}`,resources:i.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:i.operation,...i.variant?{variant:i.variant}:{},commandCount:1,maximumWorkgroupCount:y.x*y.y*y.z,maximumInvocationCount:r.length*1},compile:({device:e})=>{let r=new n(e,{id:v,source:h,shaderLayout:{bindings:g.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:n})=>r.dispatch(e,{bindings:Object.fromEntries(g.map(e=>[e.name,t(e.view,n)])),...y}),destroy:()=>r.destroy()}}})});return[...f,...g]}},O=[2,2,2,2,3,3,3,3,4,4,4,4],k=[1,3,5,7,0,2,4,6,1,3,5,7];function A(e){let t=0;for(let n=0;n<30;n++)e>>BigInt(2*n)&1n&&(t|=1<<n);return t}function j(e,t,n){let r=O[n]-e-t,i,a;r<1?(i=r,a=1-i*i/3):r>3?(i=4-r,a=i*i/3-1):(i=1,a=(2-r)*2/3);let o=k[n]*i+e-t;return o<0&&(o+=8),o>=8&&(o-=8),[(i<1e-12?NaN:Math.PI/4*(o/i))*180/Math.PI,Math.asin(Math.max(-1,Math.min(1,a)))*180/Math.PI]}function M(e,t){let n=2**t,r=Number(e>>BigInt(2*t)),i=e&(1n<<BigInt(2*t))-1n,a=A(i),o=A(i>>1n),s=(a+.5)/n,c=(o+.5)/n,l=.5/n,u=e=>((e+180)%360+360)%360-180,[d,f]=j(s,c,r),p=[u(d),f];return{center:p,corners:[[-l,-l],[l,-l],[l,l],[-l,l]].map(([e,t])=>{let[n,i]=j(s+e,c+t,r),a=Number.isNaN(n)?p[0]:u(n);for(;a-p[0]>180;)a-=360;for(;a-p[0]<-180;)a+=360;return[a,i]})}}function N(){let e=[];for(let t of b)for(let[n,r]of t)e.push(n,r);return Float32Array.from(e)}function P(e,t=1){let n=e.width*t,r=.96*Math.min(n/360,e.height/180),i=(n-360*r)/2+180*r,a=(e.height-180*r)/2+90*r;return{scale:r,toWorld:(e,t)=>[i+e*r,a+t*r],toLongitudeLatitude:(e,t)=>[(e-i)/r,(t-a)/r],coast:x([r,r],[i,a])}}function F(e,t,n=0n){return Array.from({length:t},(t,r)=>(BigInt(e[r*2+1])<<32n|BigInt(e[r*2]))-n)}function I(e,t,n){let r=e.length,i=new Float32Array(r*8),a=new Float32Array(r*16),o=new Float32Array(r*2),s=new Uint32Array(r*6),c=new Uint32Array(r*2);return e.forEach((e,r)=>{let{corners:l,center:u}=M(e,t);o.set(n.toWorld(u[0],u[1]),r*2);let d=l.map(([e,t])=>n.toWorld(e,t));d.forEach(([e,t],n)=>i.set([e,t],r*8+n*2)),d.forEach(([e,t],n)=>{let[i,o]=d[(n+1)%4];a.set([e,t,i,o],r*16+n*4)}),s.set([0,1,2,0,2,3].map(e=>r*4+e),r*6),c.set([r,r],r*2)}),{count:r,positions:i,centers:o,indices:s,owners:c,triangleCount:r*2,segments:a}}export{M as a,N as i,P as n,D as o,F as r,I as t};