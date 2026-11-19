import{t as e}from"./gpu-period-index-BfJhmJF9.js";import{r as t}from"./kernel-C3tfQOn4.js";import{l as n,o as r}from"./gpu-command-graph-BdzdmV6u.js";import{t as i}from"./status-aOwAAXek.js";import{a,n as o,o as s,r as c,t as l}from"./fixed-point-accumulator-Ce6W3HVY.js";import{D as u,E as d,S as f,T as p,c as m,f as h,v as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{r as v}from"./wgsl-common-DfyXiYB0.js";import{o as y,t as b}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as x,i as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as C,r as w,t as T}from"./flash-events-DkmBpYol.js";var E=`GPUValueCube`,D=4294967294,O=4294967295,k=256,A=class{id;props;rowCount;binCount;constructor(e){this.id=e.id??`value-cube`,this.props=e;let t=this.id,{cellCount:a,periodCount:o,cells:s,periods:c,values:l,mask:g,output:v}=e;h(t,`cellCount`,a,1,D),h(t,`periodCount`,o,1,D),this.binCount=a*o,f(t,`cellCount * periodCount`,this.binCount,D),d(s,[`uint32`],`${t} cells`),d(c,[`uint32`],`${t} periods`),d(l,[`float32`],`${t} values`),this.rowCount=s.length,h(t,`cells length`,this.rowCount,1,O),u(t,`periods`,c,this.rowCount),u(t,`values`,l,this.rowCount),g&&(d(g,[`uint32`],`${t} mask`),u(t,`mask`,g,this.rowCount)),p(t,!!(v.counts||v.sums||v.means),`output needs at least one of counts, sums and means`);for(let[e,n,r]of[[`counts`,v.counts,`uint32`],[`sums`,v.sums,`float32`],[`means`,v.means,`float32`]])n&&(d(n,[r],`${t} output.${e}`),u(t,`output.${e}`,n,this.binCount));i(t,{invalidCount:v.status?.invalidCount});let y=this.getOutputViews();_(t,y,this.getInputViews());{let e={counts:v.counts,sums:v.sums,means:v.means,invalidCount:v.status?.invalidCount},i=[[v.counts,v.status?.invalidCount],...v.means?[[v.counts,v.sums,v.means]]:[]],a=Object.values(e).flatMap(e=>e?m(e):[]).map(e=>new r(e.buffer,{format:`uint32`,length:e.length*e.rowByteLength/4,byteOffset:e.byteOffset,byteStride:4,rowByteLength:4})),o=a.reduce((e,t)=>e+t.length,0);S(t,{outputs:new n({id:`${t}-writable-ranges`,name:`outputs`,format:`uint32`,length:o,valueLength:o,stride:1,byteStride:4,rowByteLength:4,data:a})});for(let e of i)S(t,Object.fromEntries(e.map((e,t)=>[String(t),e])))}}getCommandNodes(e){let{props:n,id:r,rowCount:i,binCount:u}=this,{cells:d,periods:p,values:m,mask:h,output:_,cellCount:S,periodCount:C}=n;x(e,r,Object.fromEntries([...this.getInputViews(),...this.getOutputViews()].map((e,t)=>[`view[${t}]`,e]))),f(r,`fixed-point scratch byteLength`,u*3*4,e.device.limits.maxBufferSize),g(e.device,r,`fixed-point sums`,u*3*4,`reduce cellCount * periodCount`);let w=_.counts??t(e,`${r}-counts`,`uint32`,u),T=_.sums??(_.means?t(e,`${r}-sums`,`float32`,u):void 0),D=t(e,`${r}-scale`,`uint32`,1),O=t(e,`${r}-fixed-point`,`uint32`,u*3),A=_.status?.invalidCount,j={CELL_COUNT:S,PERIOD_COUNT:C},M=`override CELL_COUNT: u32;
override PERIOD_COUNT: u32;
fn isRowBinned(cell: u32, period: u32, value: f32) -> bool {
  return cell < CELL_COUNT && period < PERIOD_COUNT && ((bitcast<u32>(value) >> 23u) & 0xffu) != 0xffu;
}`,N=h?`mask[maskOffset + index] != 0u`:`true`,P=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),F=[P(`cells`,d,`u32`),P(`periods`,p,`u32`),P(`values`,m,`f32`),...h?[P(`mask`,h,`u32`)]:[]],I=[b(e,{id:`${r}-clear-scale`,operation:E,view:D,type:`u32`,value:`0u`}),o(e,{id:`${r}-clear-words`,operation:E,words:O})];return I.push(b(e,{id:`${r}-clear-counts`,operation:E,view:w,type:`u32`,value:`0u`})),A&&I.push(b(e,{id:`${r}-clear-invalid-count`,operation:E,view:A,type:`u32`,value:`0u`})),I.push(y(e,{id:`${r}-scale-pass`,operation:E,variant:`scale`,bindings:[...F,{name:`scale`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,workgroupSize:k,guardIndex:!1,declarations:`${M}
${l}
var<workgroup> workgroupExponentBits: atomic<u32>;`,constants:j,body:`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let period = periods[periodsOffset + index];
    let value = values[valuesOffset + index];
    if (period != 0xffffffffu && ${N} && isRowBinned(cells[cellsOffset + index], period, value)) {
      let exponentBits = getFixedPointExponentBits(value);
      if (exponentBits != 0u) {
        atomicMax(&workgroupExponentBits, exponentBits);
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&scale[scaleOffset], workgroupMaximum);
    }
  }`}),y(e,{id:`${r}-accumulate`,operation:E,variant:`accumulate`,bindings:[...F,P(`scale`,D,`u32`),{name:`countsOut`,view:w,type:`atomic<u32>`,access:`read_write`},{name:`fixedWords`,view:O,type:`atomic<u32>`,access:`read_write`},...A?[{name:`invalidCountOut`,view:A,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:i,declarations:`${M}
override FIXED_POINT_HEADROOM_BITS: u32;
${l}
${a(`fixedWords`)}`,constants:{...j,FIXED_POINT_HEADROOM_BITS:s(i)},body:`let period = periods[periodsOffset + index];
  if (period == 0xffffffffu || !(${N})) {
    return;
  }
  let cell = cells[cellsOffset + index];
  let value = values[valuesOffset + index];
  if (!isRowBinned(cell, period, value)) {${A?`
    atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);`:``}
    return;
  }
  let bin = cell * PERIOD_COUNT + period;
  let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
  atomicAdd(&countsOut[countsOutOffset + bin], 1u);
  addFixedPointToFixedWords(bin, quantizeFixedPoint(value, unitExponent));`})),T&&I.push(c(e,{id:`${r}-decode`,operation:E,words:O,scale:D,maximumTermCount:i,output:T})),_.means&&T&&I.push(y(e,{id:`${r}-means`,operation:E,variant:`means`,bindings:[{name:`countsIn`,view:w,type:`u32`,access:`read_write`},{name:`sumsIn`,view:T,type:`f32`,access:`read_write`},{name:`meansOut`,view:_.means,type:`f32`,access:`read_write`}],invocationCount:u,declarations:v,body:`let count = countsIn[countsInOffset + index];
  meansOut[meansOutOffset + index] = select(getNaN(), sumsIn[sumsInOffset + index] / f32(count), count > 0u);`})),I}getInputViews(){let{cells:e,periods:t,values:n,mask:r}=this.props;return[e,t,n,r].filter(e=>!!e)}getOutputViews(){let{output:e}=this.props;return[e.counts,e.sums,e.means,e.status?.invalidCount].filter(e=>!!e)}};function j(t,n,r,i){let a=t.upload(`timestamps`,`uint32x2`,n.timestampWords),o=t.allocate(`periods`,`uint32`,n.count);t.graph.add(new e({temporalContext:C,periods:{kind:`fixed`,width:BigInt(w),origin:T,count:18},timestamps:a.view,output:{periods:o.view}}));let s=t.allocate(`mask`,`uint32`,n.count),c=t.allocate(`counts`,`uint32`,i*18),l=t.allocate(`sums`,`float32`,i*18),u=t.allocate(`means`,`float32`,i*18),d=t.allocate(`invalid-count`,`uint32`,1);return t.graph.add(new A({cellCount:i,periodCount:18,cells:t.upload(`cells`,`uint32`,r).view,periods:o.view,values:t.upload(`energy`,`float32`,n.energy).view,mask:s.view,output:{counts:c.view,sums:l.view,means:u.view,status:{invalidCount:d.view}}})),{mask:s,periods:o,counts:c,sums:l,means:u,invalidCount:d}}function M(e,t,n,r){for(let e=0;e<n.length;e++)t[e]=+(n[e]>=r);e.mask.write(t)}export{M as n,j as t};