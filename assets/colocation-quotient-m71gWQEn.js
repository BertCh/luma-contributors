import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,T as i,f as a,v as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{a as l,n as u,o as d,r as f,t as p}from"./fixed-point-accumulator-Ce6W3HVY.js";import{d as m,r as h}from"./wgsl-common-DfyXiYB0.js";import{c as g}from"./spatial-weights-CqWysstN.js";import{t as _}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as v}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as y}from"./japan-coast-Btjlj199.js";import{c as b,l as x,t as S}from"./point-observations-CeqP-hIo.js";import{t as C}from"./define-run-B-brkCe4.js";import{m as w,o as T}from"./inputs-CJKKIRKK.js";import{n as E}from"./pick-Bo3t3hjv.js";import{o as D}from"./tract-map-BETLV_b0.js";import{a as O,r as k}from"./grid-panel-CtEMLsqP.js";import{t as A}from"./readback-cfnHpL6G.js";import{Rl as j}from"./entry-preview-D70TdSbK.js";var M=`GPUColocationQuotient`,N=127,P=class{id;props;rowCount;constructor(e){let t=e.id??`colocation-quotient`;this.id=t,this.props=e,this.rowCount=g(t,e.weights),a(t,`categoryCount`,e.categoryCount,1,32),r(e.categories,[`uint32`],`${t} categories`),i(t,e.categories.length===this.rowCount,`categories length must equal the weights row count`),r(e.output.globalQuotients,[`float32`],`${t} output.globalQuotients`),i(t,e.output.globalQuotients.length>=e.categoryCount**2,`output.globalQuotients must hold categoryCount * categoryCount entries`),e.output.localQuotients&&(r(e.output.localQuotients,[`float32`],`${t} output.localQuotients`),i(t,e.output.localQuotients.length>=this.rowCount*e.categoryCount,`output.localQuotients must hold rows * categoryCount entries`)),e.output.categoryRowCounts&&(r(e.output.categoryRowCounts,[`uint32`],`${t} output.categoryRowCounts`),i(t,e.output.categoryRowCounts.length>=e.categoryCount,`output.categoryRowCounts must hold categoryCount entries`)),v(t,e.output),c(t,[e.output.globalQuotients,e.output.localQuotients,e.output.categoryRowCounts],[e.categories,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(r){let{id:i,props:a,rowCount:c}=this;_(r,i,a,{"category counters":a.categoryCount*4,"share accumulators":a.categoryCount**2*3*4});let{weights:g,output:v,categoryCount:y}=a;s(i,r,[a.categories,g.offsets,g.neighbors,g.weights,v.globalQuotients,v.localQuotients,v.categoryRowCounts]),v.localQuotients&&o(r.device,i,`output.localQuotients`,c*y*4,`reduce the rows or the categoryCount`);let b=g.neighbors.length,x=y**2,S=v.categoryRowCounts??e(r,`${i}-category-row-counts`,`uint32`,y),C=e(r,`${i}-linked-counts`,`uint32`,y),w=e(r,`${i}-share-words`,`uint32`,x*3),T=e(r,`${i}-share-sums`,`float32`,x),E=e(r,`${i}-share-scale`,`uint32`,1),D=v.localQuotients,O={FIXED_POINT_HEADROOM_BITS:d(c)},k=[n(r,{id:`${i}-clear-rows`,operation:M,view:S,type:`u32`,value:`0u`,componentCount:y}),n(r,{id:`${i}-clear-linked`,operation:M,view:C,type:`u32`,value:`0u`}),n(r,{id:`${i}-scale`,operation:M,view:E,type:`u32`,value:`${N}u`}),u(r,{id:`${i}-clear-shares`,operation:M,words:w}),t(r,{id:`${i}-shares`,operation:M,variant:D?`shares-local`:`shares`,bindings:[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read`},{name:`linkWeights`,view:g.weights,type:`f32`,access:`read`},{name:`categories`,view:a.categories,type:`u32`,access:`read`},{name:`rowCounts`,view:S,type:`atomic<u32>`,access:`read_write`},{name:`linkedCounts`,view:C,type:`atomic<u32>`,access:`read_write`},{name:`shareWords`,view:w,type:`atomic<u32>`,access:`read_write`},...D?[{name:`localValues`,view:D,type:`f32`,access:`read_write`}]:[]],invocationCount:c,declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
const ROWS: u32 = ${c}u;
const SLOTS: u32 = ${b}u;
const CATEGORIES: u32 = ${y}u;
${h}
${p}
${l(`shareWords`)}`,constants:O,body:`let own = categories[categoriesOffset + index];
  if (own >= CATEGORIES) {
    ${D?`for (var b = 0u; b < CATEGORIES; b++) { localValues[localValuesOffset + index * CATEGORIES + b] = getNaN(); }`:``}
    return;
  }
  atomicAdd(&rowCounts[rowCountsOffset + own], 1u);
  var shares: array<f32, ${y}>;
  for (var b = 0u; b < CATEGORIES; b++) {
    shares[b] = 0.0;
  }
  var total = 0.0;
  let first = min(offsets[offsetsOffset + index], SLOTS);
  let last = min(offsets[offsetsOffset + index + 1u], SLOTS);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = linkWeights[linkWeightsOffset + slot];
    if (neighbor < ROWS && neighbor != index && weight > 0.0 && weight < 3.0e38) {
      let other = categories[categoriesOffset + neighbor];
      if (other < CATEGORIES) {
        shares[other] += weight;
        total += weight;
      }
    }
  }
  if (!(total > 0.0)) {
    ${D?`for (var b = 0u; b < CATEGORIES; b++) { localValues[localValuesOffset + index * CATEGORIES + b] = getNaN(); }`:``}
    return;
  }
  atomicAdd(&linkedCounts[linkedCountsOffset + own], 1u);
  let unitExponent = getFixedPointUnitExponent(${N}u, FIXED_POINT_HEADROOM_BITS);
  for (var b = 0u; b < CATEGORIES; b++) {
    let share = shares[b] / total;
    addFixedPointToShareWords(own * CATEGORIES + b, quantizeFixedPoint(share, unitExponent));
    ${D?`localValues[localValuesOffset + index * CATEGORIES + b] = share;`:``}
  }`}),f(r,{id:`${i}-decode-shares`,operation:M,words:w,scale:E,maximumTermCount:c,output:T}),t(r,{id:`${i}-global`,operation:M,variant:`global`,bindings:[{name:`sums`,view:T,type:`f32`,access:`read`},{name:`rowCounts`,view:S,type:`u32`,access:`read`},{name:`linkedCounts`,view:C,type:`u32`,access:`read`},{name:`globalQuotients`,view:v.globalQuotients,type:`f32`,access:`read_write`}],invocationCount:x,declarations:`const CATEGORIES: u32 = ${y}u;
${h}`,body:`let own = index / CATEGORIES;
  let other = index % CATEGORIES;
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  let otherRows = rowCounts[rowCountsOffset + other] - select(0u, 1u, own == other);
  let linked = linkedCounts[linkedCountsOffset + own];
  let valid = linked > 0u && total > 1u && rowCounts[rowCountsOffset + other] >= select(0u, 1u, own == other) && otherRows > 0u;
  let chance = f32(otherRows) / f32(total - 1u);
  globalQuotients[globalQuotientsOffset + index] = select(
    getNaN(),
    sums[sumsOffset + index] / f32(linked) / chance,
    valid
  );`})];return D&&k.push(t(r,{id:`${i}-local`,operation:M,variant:`local`,bindings:[{name:`categories`,view:a.categories,type:`u32`,access:`read`},{name:`rowCounts`,view:S,type:`u32`,access:`read`},{name:`localValues`,view:D,type:`f32`,access:`read_write`}],invocationCount:c*y,declarations:`const CATEGORIES: u32 = ${y}u;
${h}${m}`,body:`let row = index / CATEGORIES;
  let other = index % CATEGORIES;
  let own = categories[categoriesOffset + row];
  let share = localValues[localValuesOffset + index];
  if (own >= CATEGORIES || isNanBits(share)) {
    return;
  }
  var total = 0u;
  for (var b = 0u; b < CATEGORIES; b++) {
    total += rowCounts[rowCountsOffset + b];
  }
  let otherRows = rowCounts[rowCountsOffset + other] - select(0u, 1u, own == other);
  let valid = total > 1u && otherRows > 0u && otherRows < 0x80000000u;
  localValues[localValuesOffset + index] = select(getNaN(), share * f32(total - 1u) / f32(otherRows), valid);`})),k}},F;function I(){if(F)return F;let e=Uint8Array.from(atob(x.data),e=>e.charCodeAt(0)),t=new DataView(e.buffer),[n,r,i,a]=x.bounds,[o,s,c,l]=S,u=[],d=[],f=[];for(let p=0;p<e.length/8;p++){let e=n+(i-n)*t.getUint16(p*8,!0)/1e4,m=r+(a-r)*t.getUint16(p*8+2,!0)/1e4;e<o||e>c||m<s||m>l||(u.push(e,m),d.push(t.getUint16(p*8+4,!0)/100),f.push(t.getUint16(p*8+6,!0)))}return F={count:d.length,positions:Float32Array.from(u),magnitude:Float32Array.from(d),day:Float32Array.from(f)},F}var L=3,R=[.03,.62,.3,.95],z=C({parameters:j,caption:`Earthquakes off Honshu, 2020-2024, in three eras of equal size; dots show local colocation quotients.`,createInputs:({world:e})=>{let t=I(),n=e=>Array.from({length:e.length/2},(t,n)=>({longitude:e[n*2],latitude:e[n*2+1]})),r=b(n(t.positions),[...S],e);return{...t,positions:r,categories:D(t.day,L),index:E(r),world:e,focus:{event:-1},coastline:b(n(y.flat()),[...S],e)}},build(e,t,n){let r=T(t.positions,n.neighborCount),i=e.allocate(`global-quotients`,`float32`,L**2),a=e.allocate(`local-quotients`,`float32`,t.count*L);return e.graph.add(new P({weights:w(e,r),categories:e.upload(`categories`,`uint32`,t.categories).view,categoryCount:L,output:{globalQuotients:i.view,localQuotients:a.view}})),{outputs:{global:i,local:a,csr:r,panel:O(e,R,t.world),positions:e.upload(`positions`,`float32x2`,t.positions),coast:e.upload(`coast`,`float32x4`,t.coastline),categories:e.upload(`category-colors`,`uint32`,t.categories)}}},rebuildOn:[`neighborCount`],buildCache:4,async afterRun({inputs:e,outputs:t,values:n,upload:r}){let i=await A(t.local);return{selected:r(`selected`,`float32`,Float32Array.from({length:e.count},(t,r)=>e.categories[r]===n.from?i[r*L+n.to]:NaN))}},interaction:{kind:`probe`,hint:`hover an event`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=t.index.nearest(e.x,e.y);o>=0&&(t.focus.event=o);let s=t.focus.event,c=(R[1]+(R[3]-R[1])/2)*i.height,l=[{shape:`line`,from:[R[0]*i.width,c],to:[R[2]*i.width,c],dashed:!0}];if(s<0)return{glyphs:l,spotlight:null};let{csr:u}=n,d=0,f=u.offsets[s+1]-u.offsets[s];for(let e=u.offsets[s];e<u.offsets[s+1];e++){let n=u.neighbors[e];t.categories[n]===r.to&&d++,l.push({shape:`line`,from:[t.positions[s*2],t.positions[s*2+1]],to:[t.positions[n*2],t.positions[n*2+1]]})}let p=a(n.global),m=t.categories[s],h=a(n.local)?.[s*L+r.to];return{glyphs:l,spotlight:null,readout:`category ${m} · ${d} of ${f} neighbours in ${r.to} · local ${h!==void 0&&Number.isFinite(h)?h.toFixed(2):`–`} · CLQ ${p?p[r.from*L+r.to].toFixed(2):`…`}`}}},draw:({outputs:e,extra:t,theme:n})=>[{layer:`lines`,context:!0,segments:e.coast},{layer:`dots`,context:!0,positions:e.positions,radius:2.4},{layer:`dots`,positions:e.positions,values:t.selected,ramp:`diverging`,range:[0,2],radius:4.2},k(e.panel,n.background),{layer:`chart`,kind:`bars`,values:e.global,ramp:`sequential`,range:[0,2],rectangle:R}]});export{z as default};