import{t as e}from"./gpu-class-assignment-DmZVEfZd.js";import{r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,f as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{a as u}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as d}from"./spatial-output-validation-nw5lOmVL.js";import{t as f}from"./define-run-B-brkCe4.js";import{tc as p}from"./entry-preview-D70TdSbK.js";import{l as m,n as h,t as g}from"./dynamics-DVM7QJSQ.js";import{c as _,i as v,r as y}from"./geometry-Dz9W1tyx.js";import{t as b}from"./gather-BaJ2HB8H.js";import{l as x,r as S,u as C}from"./regions-CO_uxNyo.js";var w=`GPUClassificationFit`,T=256,E=256,D=256,O=16,k=class{id;props;constructor(e){let t=e.id??`classification-fit`;this.id=t,this.props=e;let{values:r,classes:i,classCount:s,mask:c,output:u}=e;if(a(r,[`float32`],`${t} values`),n(i,`${t} classes`),o(t,`classCount`,s,1,E),r.length<1)throw Error(`${t} values must hold at least one row`);if(i.length!==r.length)throw Error(`${t} classes length must equal values length`);if(c&&(n(c,`${t} mask`),c.length!==r.length))throw Error(`${t} mask length must equal values length`);let f=s+1;n(u.counts,`${t} counts`),a(u.medians,[`float32`],`${t} medians`),a(u.absoluteDeviations,[`float32`],`${t} absoluteDeviations`),a(u.squaredDeviations,[`float32`],`${t} squaredDeviations`);for(let[e,n]of[[`counts`,u.counts],[`medians`,u.medians],[`absoluteDeviations`,u.absoluteDeviations],[`squaredDeviations`,u.squaredDeviations]])if(n.length!==f)throw Error(`${t} ${e} length must equal classCount + 1`);if(u.summary&&(a(u.summary,[`float32`],`${t} summary`),u.summary.length!==5))throw Error(`${t} summary length must equal the summary layout`);l(t,[u.counts,u.medians,u.absoluteDeviations,u.squaredDeviations,u.summary],[r,i,c]),d(t,u,[[u.counts,u.medians,u.absoluteDeviations,u.squaredDeviations],...u.summary?[{read:[u.absoluteDeviations,u.squaredDeviations],write:[u.summary]}]:[]])}getCommandNodes(e){let{id:n,props:a}=this,{values:o,classes:l,classCount:d,mask:f,output:p}=a;c(n,e,[o,l,f,p.counts,p.medians,p.absoluteDeviations,p.squaredDeviations,p.summary]);let m=o.length;u(e,n,{values:o,classes:l,mask:f,...p});for(let[t,r]of[[`sorted row scratch`,m*4],[`sort histogram`,m>D?Math.ceil(m/D)*O*4:0]])s(e.device,n,t,r,`reduce values rows`);let h=t(e,`${n}-value-keys`,`uint32`,m),g=t(e,`${n}-row-ids`,`uint32`,m),_=t(e,`${n}-sorted-value-keys`,`uint32`,m),v=t(e,`${n}-rows-by-value`,`uint32`,m),y=t(e,`${n}-class-keys`,`uint32`,m),b=t(e,`${n}-sorted-class-keys`,`uint32`,m),x=t(e,`${n}-rows-by-class`,`uint32`,m),S=`const ROWS: u32 = ${m}u;
const CLASSES: u32 = ${d}u;
const UNCOUNTED_KEY: u32 = 0xffffffffu;`,C=[i(e,{id:`${n}-keys`,operation:w,variant:`keys`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},{name:`classes`,view:l,type:`u32`,access:`read`},...f?[{name:`mask`,view:f,type:`u32`,access:`read`}]:[],{name:`valueKeys`,view:h,type:`u32`,access:`read_write`},{name:`rowIds`,view:g,type:`u32`,access:`read_write`}],invocationCount:m,declarations:S,body:`let value = values[valuesOffset + index];
  let bits = bitcast<u32>(value);
  // WGSL gives no NaN guarantees to comparisons: counted values have a non-all-ones exponent.
  let counted = (bits & 0x7f800000u) != 0x7f800000u && classes[classesOffset + index] < CLASSES
    && ${f?`mask[maskOffset + index] != 0u`:`true`};
  // Order-preserving key; the all-ones key (above every finite value) marks uncounted rows.
  let key = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
  valueKeys[valueKeysOffset + index] = select(UNCOUNTED_KEY, key, counted);
  rowIds[rowIdsOffset + index] = index;`}),...new r({id:`${n}-value-sort`,keys:h,values:g,outputKeys:_,outputValues:v}).getCommandNodes(e),i(e,{id:`${n}-class-keys`,operation:w,variant:`class-keys`,bindings:[{name:`sortedValueKeys`,view:_,type:`u32`,access:`read`},{name:`rowsByValue`,view:v,type:`u32`,access:`read`},{name:`classes`,view:l,type:`u32`,access:`read`},{name:`classKeys`,view:y,type:`u32`,access:`read_write`}],invocationCount:m,declarations:S,body:`let counted = sortedValueKeys[sortedValueKeysOffset + index] != UNCOUNTED_KEY;
  classKeys[classKeysOffset + index] =
    select(CLASSES, classes[classesOffset + rowsByValue[rowsByValueOffset + index]], counted);`}),...new r({id:`${n}-class-sort`,keys:y,values:v,outputKeys:b,outputValues:x,keyBits:Math.max(1,Math.ceil(Math.log2(d+1)))}).getCommandNodes(e),i(e,{id:`${n}-stats`,operation:w,variant:`stats`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},{name:`sortedClassKeys`,view:b,type:`u32`,access:`read`},{name:`rowsByValue`,view:v,type:`u32`,access:`read`},{name:`rowsByClass`,view:x,type:`u32`,access:`read`},{name:`counts`,view:p.counts,type:`u32`,access:`read_write`},{name:`medians`,view:p.medians,type:`f32`,access:`read_write`},{name:`absoluteDeviations`,view:p.absoluteDeviations,type:`f32`,access:`read_write`},{name:`squaredDeviations`,view:p.squaredDeviations,type:`f32`,access:`read_write`}],invocationCount:(d+1)*T,workgroupSize:T,guardIndex:!1,declarations:`${S}
const LANES: u32 = ${T}u;
var<workgroup> scratch: array<f32, ${T}>;

fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  scratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  return scratch[0];
}

// First position in sortedClassKeys whose key is at least bound. In-kernel on purpose:
// GPUGallopingSearch is a batch node and would add a pass and a buffer.
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedClassKeys[sortedClassKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}

fn valueAt(slot: u32, position: u32) -> f32 {
  // Slot CLASSES (all counted rows) reads the value-ordered rows, a class slot its class segment.
  let row = select(
    rowsByClass[rowsByClassOffset + position],
    rowsByValue[rowsByValueOffset + position],
    slot == CLASSES
  );
  return values[valuesOffset + row];
}`,body:`let slot = index / LANES;
  let lane = localInvocationIndex;
  let start = select(lowerBound(slot), 0u, slot == CLASSES);
  let end = lowerBound(select(slot + 1u, CLASSES, slot == CLASSES));
  let count = end - start;
  var median = 0.0;
  if (count > 0u) {
    let lower = valueAt(slot, start + (count - 1u) / 2u);
    let upper = valueAt(slot, start + count / 2u);
    median = lower + (upper - lower) * 0.5;
  }
  var partialSum = 0.0;
  for (var position = start + lane; position < end; position += LANES) {
    partialSum += valueAt(slot, position);
  }
  let mean = reduceSum(partialSum, lane) / f32(max(count, 1u));
  var partialAbsolute = 0.0;
  var partialSquared = 0.0;
  for (var position = start + lane; position < end; position += LANES) {
    let value = valueAt(slot, position);
    partialAbsolute += abs(value - median);
    let centered = value - mean;
    partialSquared += centered * centered;
  }
  let absolute = reduceSum(partialAbsolute, lane);
  let squared = reduceSum(partialSquared, lane);
  if (lane == 0u) {
    counts[countsOffset + slot] = count;
    medians[mediansOffset + slot] = median;
    absoluteDeviations[absoluteDeviationsOffset + slot] = absolute;
    squaredDeviations[squaredDeviationsOffset + slot] = squared;
  }`})];return p.summary&&C.push(i(e,{id:`${n}-summary`,operation:w,variant:`summary`,bindings:[{name:`absoluteDeviations`,view:p.absoluteDeviations,type:`f32`,access:`read`},{name:`squaredDeviations`,view:p.squaredDeviations,type:`f32`,access:`read`},{name:`summary`,view:p.summary,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const CLASSES: u32 = ${d}u;`,body:`var adcm = 0.0;
  var sdcm = 0.0;
  for (var slot = 0u; slot < CLASSES; slot++) {
    adcm += absoluteDeviations[absoluteDeviationsOffset + slot];
    sdcm += squaredDeviations[squaredDeviationsOffset + slot];
  }
  let adam = absoluteDeviations[absoluteDeviationsOffset + CLASSES];
  let tss = squaredDeviations[squaredDeviationsOffset + CLASSES];
  summary[summaryOffset + 0u] = select(1.0, 1.0 - adcm / adam, adam > 0.0);
  summary[summaryOffset + 1u] = adcm;
  summary[summaryOffset + 2u] = adam;
  summary[summaryOffset + 3u] = select(1.0, 1.0 - sdcm / tss, tss > 0.0);
  summary[summaryOffset + 4u] = sdcm;`})),C}},A=f({parameters:p,caption:`Class medians (top) and summed absolute deviations from them (bottom), over all periods.`,createInputs:({random:e,world:t})=>{let n=h(e,t);return{...n,focus:{tract:y(n.tracts,t)}}},build(t,{tracts:n,values:r},i){let a=i.classCount,o=t.allocate(`classes`,`uint32`,r.length),s=t.upload(`values`,`float32`,r);t.graph.add(new e({values:s.view,breaks:t.upload(`breaks`,`float32`,g(r,a,i.quantile===1)).view,classCount:t.upload(`class-count`,`uint32`,Uint32Array.of(a)).view,output:o.view}));let c=t.allocate(`counts`,`uint32`,a+1),l=t.allocate(`summary`,`float32`,5),u=t.allocate(`medians`,`float32`,a+1),d=t.allocate(`absolute-deviations`,`float32`,a+1);t.graph.add(new k({values:s.view,classes:o.view,classCount:a,output:{counts:c.view,medians:u.view,absoluteDeviations:d.view,squaredDeviations:t.allocate(`squared-deviations`,`float32`,a+1).view,summary:l.view}}));let f=_(t,n);return{outputs:{classes:o,counts:c,summary:l,medians:u,deviations:d,medianBars:b(t,u,a,`median-bars`),deviationBars:b(t,d,a,`deviation-bars`),highlight:S(t,{tracts:n}),...f},update:e=>f.owners.write(m(n.triangleOwners,n.count,e.period))}},rebuildOn:[`classCount`,`quantile`],interaction:{kind:`probe`,hint:`hover a tract`,rest:({inputs:e,toUnit:t})=>t([e.tracts.centers[e.focus.tract*2],e.tracts.centers[e.focus.tract*2+1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=v(t.tracts,e.x,e.y);a>=0&&(t.focus.tract=a);let o=i(n.classes);if(!o)return{};let s=o.subarray(r.period*t.tracts.count,(r.period+1)*t.tracts.count),c=s[t.focus.tract],l=C(t,s,c,n.highlight),u=i(n.medians)?.[c],d=i(n.counts)?.[c],f=i(n.deviations)?.[c],p=f!==void 0&&d?f/d:void 0;return{focus:c,readout:`class ${c+1} · ${l} tracts · median ${u?.toFixed(2)??`...`} · mean deviation ${p?.toFixed(3)??`...`}`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`polygons`,positions:t.vertices,indices:t.indices,owners:t.owners,values:t.classes,triangleCount:e.tracts.triangleCount,ramp:`sequential`,range:[0,n.classCount-1]},{layer:`lines`,context:!0,segments:t.outline,count:e.tracts.outlineCount,alpha:.4},x(e,t.highlight),{layer:`chart`,kind:`bars`,values:t.medianBars,range:[0,1],rectangle:[.58,.55,.97,.92],ramp:`sequential`},{layer:`chart`,kind:`bars`,values:t.deviationBars,range:`auto`,rectangle:[.58,.08,.97,.45],ramp:`sequential`}]});export{A as default};