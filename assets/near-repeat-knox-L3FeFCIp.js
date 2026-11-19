import{_ as e,a as t,l as n,n as r,o as i,r as a,s as o,v as s}from"./space-time-kernels-a9nSiN0t.js";import{n as c,r as l}from"./trajectory-binding-validation-C89oJIAZ.js";import{t as u}from"./philox-random-DfVxWNPn.js";import{r as d}from"./kernel-C3tfQOn4.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as p,f as m,v as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{c as v}from"./spatial-weights-CqWysstN.js";import{a as y,i as b,r as x,t as S}from"./space-time-bands-Lbr2KxkP.js";import{t as C}from"./define-run-B-brkCe4.js";import{m as w,r as T}from"./inputs-CJKKIRKK.js";import{c as E,o as D,r as O}from"./events-CU2v4wkX.js";import{t as k}from"./kernel-CrpKP5IU.js";import{i as A,n as j,r as M,t as N}from"./table-CZ7xiVf4.js";import{ln as P}from"./entry-preview-D70TdSbK.js";var F=`GPUNearRepeatKnox`,I={observed:0,permutationMean:1,permutationVariance:2,greaterCount:3,lesserCount:4,pseudoPGreater:5,pseudoPLesser:6,knoxRatio:7},L=class{id;props;rows;distanceBands;timeBands;cells;temporalContext;constructor(e){let t=e.id??`near-repeat-knox`;if(this.id=t,this.props=e,this.rows=v(t,e.pairs,`pairs`),!e.pairs.distances)throw Error(`${t} pairs.distances is required`);if(p(e.times,[`float32`],`${t} times`),p(e.parameters,[`uint32`],`${t} parameters`),p(e.output.counts,[`uint32`],`${t} output.counts`),p(e.output.summary,[`float32`],`${t} output.summary`),this.temporalContext=b(t,e.temporalContext),this.rows<2)throw Error(`${t} needs at least two rows`);if(this.rows>a)throw Error(`${t} supports at most ${a} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(y(t,`distanceEdges`,e.distanceEdges),y(t,`timeEdges`,e.timeEdges),this.distanceBands=e.distanceEdges.length-1,this.timeBands=e.timeEdges.length-1,this.cells=this.distanceBands*this.timeBands,this.cells>256)throw Error(`${t} supports at most 256 cells`);if(m(t,`maximumPermutations`,e.maximumPermutations,1,r),e.output.counts.length<(e.maximumPermutations+1)*this.cells)throw Error(`${t} output.counts needs (maximumPermutations + 1) * cells rows`);if(e.output.summary.length<this.cells*8)throw Error(`${t} output.summary needs cells * 8 rows`);if(l(this.id,this.props.output),_(t,[e.output.counts,e.output.summary],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights,e.pairs.distances]),e.output.counts.buffer===e.output.summary.buffer)throw Error(`${t} outputs must use separate buffers`)}getInferenceMetadata(t){return{...e(`knox`,t,this.props.maximumPermutations),testedRowsDefinition:`one test per cell of distance band by time band over all unordered event pairs listed in the spatial pair list`}}getCommandNodes(e){c(e,this.id,this.props);let{id:r,props:a,rows:s,cells:l,timeBands:p}=this,{pairs:m,times:_,parameters:v,maximumPermutations:y}=a,{counts:b,summary:C}=a.output,w=m.distances;g(r,e,[m.offsets,m.neighbors,m.weights,w,_,v,b,C]),h(e.device,r,`pair-band`,m.neighbors.length*4,`reduce the pair capacity`);let T=y+1,[E,D]=i(s,64),O=o(s,y,E,D),k=d(e,`${r}-pair-mask`,`uint32`,m.neighbors.length),A=d(e,`${r}-pair-band`,`uint32`,m.neighbors.length),j=`${x(`Distance`,a.distanceEdges)}${x(`Time`,a.timeEdges)}`;return[t(e,{id:r,operation:F,rows:s,offsets:m.offsets,neighbors:m.neighbors,pairMask:k}),f(e,{id:`${r}-pair-band`,operation:F,variant:`pair-band`,bindings:[{name:`pairMask`,view:k,type:`u32`,access:`read`},{name:`distances`,view:w,type:`f32`,access:`read`},{name:`pairBand`,view:A,type:`u32`,access:`read_write`}],invocationCount:m.neighbors.length,declarations:j,body:`pairBand[pairBandOffset + index] = select(
    ${S},
    getDistanceBand(distances[distancesOffset + index]),
    pairMask[pairMaskOffset + index] != 0u
  );`}),f(e,{id:`${r}-histogram`,operation:F,variant:`histogram`,bindings:[{name:`offsets`,view:m.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:m.neighbors,type:`u32`,access:`read`},{name:`pairBand`,view:A,type:`u32`,access:`read`},{name:`times`,view:_,type:`f32`,access:`read`},{name:`parameters`,view:v,type:`u32`,access:`read`},{name:`counts`,view:b,type:`u32`,access:`read_write`}],workgroupSize:64,invocationCount:T*64,guardIndex:!1,declarations:`${O}
${n(E,u.knoxTestShuffle.id)}
${j}
const CELLS: u32 = ${l}u;
const TIME_BANDS: u32 = ${p}u;
const NO_BAND: u32 = ${S};
var<workgroup> histogram: array<atomic<u32>, ${l}>;`,body:`let slot = index / SPACE_TIME_WORKGROUP_SIZE;
  let runsSlot = slot <= readPermutationCount();
  if (slot > 0u && runsSlot) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  for (var cell = localInvocationIndex; cell < CELLS; cell += SPACE_TIME_WORKGROUP_SIZE) {
    atomicStore(&histogram[cell], 0u);
  }
  workgroupBarrier();
  if (runsSlot) {
    for (var row = localInvocationIndex; row < ROWS; row += SPACE_TIME_WORKGROUP_SIZE) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let distanceBand = pairBand[pairBandOffset + entry];
        if (distanceBand != NO_BAND) {
          let otherTime = times[timesOffset + getPermutedRow(neighbors[neighborsOffset + entry], slot)];
          let timeBand = getTimeBand(abs(time - otherTime));
          if (timeBand != NO_BAND) {
            atomicAdd(&histogram[distanceBand * TIME_BANDS + timeBand], 1u);
          }
        }
      }
    }
  }
  workgroupBarrier();
  for (var cell = localInvocationIndex; cell < CELLS; cell += SPACE_TIME_WORKGROUP_SIZE) {
    counts[countsOffset + slot * CELLS + cell] = select(0u, atomicLoad(&histogram[cell]), runsSlot);
  }`}),f(e,{id:`${r}-summary`,operation:F,variant:`summary`,bindings:[{name:`parameters`,view:v,type:`u32`,access:`read`},{name:`counts`,view:b,type:`u32`,access:`read`},{name:`summary`,view:C,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${O}
const CELLS: u32 = ${l}u;`,body:`let permutations = readPermutationCount();
  let observed = counts[countsOffset + index];
  var greater = 0u;
  var lesser = 0u;
  var sumDeviation = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = counts[countsOffset + slot * CELLS + index];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sumDeviation += f32(i32(value) - i32(observed));
  }
  let permutationCount = f32(permutations);
  let meanDeviation = sumDeviation / permutationCount;
  var sumSquares = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let deviation = f32(i32(counts[countsOffset + slot * CELLS + index]) - i32(observed)) - meanDeviation;
    sumSquares += deviation * deviation;
  }
  let variance = select(0.0, sumSquares / (permutationCount - 1.0), permutations > 1u);
  let mean = f32(observed) + meanDeviation;
  let cellOffset = summaryOffset + index * 8u;
  summary[cellOffset + ${I.observed}u] = f32(observed);
  summary[cellOffset + ${I.permutationMean}u] = mean;
  summary[cellOffset + ${I.permutationVariance}u] = variance;
  summary[cellOffset + ${I.greaterCount}u] = f32(greater);
  summary[cellOffset + ${I.lesserCount}u] = f32(lesser);
  summary[cellOffset + ${I.pseudoPGreater}u] = f32(greater + 1u) / (permutationCount + 1.0);
  summary[cellOffset + ${I.pseudoPLesser}u] = f32(lesser + 1u) / (permutationCount + 1.0);
  summary[cellOffset + ${I.knoxRatio}u] = select(
    bitcast<f32>(parameters[parametersOffset + 3u] | 0x7fc00000u),
    f32(observed) / mean,
    mean > 0.0
  );`})]}},R=5,z=900,B=199,V=.05,H=160;function U(e){return Array.from({length:6},(t,n)=>n*e)}function W(e,t){let n=Math.max(0,Math.ceil(e/t)-1);return n<R?n:-1}var G=C({parameters:P,caption:`GOES lightning, May 21–22 2024 (900 of 4,096 sampled flashes): Knox ratio per distance band (rows, near at top) and time band (columns, short at left); dashed cells p ≤ 0.05, uncorrected.`,createInputs:({world:e})=>({...O([0,0,e.width*.56,e.height],z),table:N(R,R,[e.width*.6,.12,e.width*.96,.88])}),build(e,t,n){let r=U(n.distanceBand/t.kilometersPerUnit),i=U(n.timeBand),a=T(t.positions,r[R]),o=e.allocate(`times`,`float32`,t.count),c=e.allocate(`parameters`,`uint32`,4),l=e.allocate(`summary`,`float32`,200);e.graph.add(new L({pairs:w(e,a),times:o.view,parameters:c.view,distanceEdges:r,timeEdges:i,maximumPermutations:B,output:{counts:e.allocate(`counts`,`uint32`,5e3).view,summary:l.view}}));let u=e.allocate(`ratio`,`float32`,25);k(e,{id:`near-repeat-ratio`,invocationCount:25,bindings:[{name:`summary`,buffer:l,access:`read`},{name:`ratio`,buffer:u,access:`read_write`}],body:`let value = summary[summaryOffset + index * 8u + ${I.knoxRatio}u];
  ratio[ratioOffset + index] = log2(max(value, 0.01));`});let d=A(e,t.table);return{outputs:{positions:e.upload(`positions`,`float32x2`,t.positions),ratio:u,summary:l,pairs:a,tablePositions:d.positions,tableIndices:d.indices,tableOwners:d.owners},update:e=>{o.write(E(t.hours,e.observedShare)),c.write(s({seed:5,permutationCount:B}))}}},rebuildOn:[`distanceBand`,`timeBand`],buildCache:5,interaction:{kind:`probe`,hint:`hover a table cell`,rest:[.66,.2],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let{table:a}=t,o=M(a,e.x,e.y),s=i(n.summary),c=[];if(s){for(let e=0;e<25;e++)if(s[e*8+I.pseudoPGreater]<=V){let t=j(a,Math.floor(e/R),e%R);c.push({shape:`box`,...t,dashed:!0})}}c.push({shape:`box`,...j(a,o.row,o.column)});let l=E(t.hours,r.observedShare),u=r.distanceBand/t.kilometersPerUnit,{pairs:d}=n,f=0;for(let e=0;e<d.rowCount;e++)for(let n=d.offsets[e];n<d.offsets[e+1];n++){let i=d.neighbors[n];i<=e||W(d.distances[n],u)===o.row&&W(Math.abs(l[e]-l[i]),r.timeBand)===o.column&&(f++,f>H||c.push({shape:`line`,from:[t.positions[e*2],t.positions[e*2+1]],to:[t.positions[i*2],t.positions[i*2+1]]}))}let p=o.index*8,m=`${o.row*r.distanceBand}–${(o.row+1)*r.distanceBand} km`,h=`${o.column*r.timeBand}–${(o.column+1)*r.timeBand} h`;return{focus:o.index,glyphs:c,readout:`${D(s?.[p+I.knoxRatio],1)}x expected · ${m}, ${h} · ${f} pairs · p ${D(s?.[p+I.pseudoPGreater],3)}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`dots`,positions:t.positions,radius:2.2,color:[...n.muted,.9]},{layer:`polygons`,positions:t.tablePositions,indices:t.tableIndices,owners:t.tableOwners,values:t.ratio,triangleCount:e.table.triangleCount,ramp:`diverging`,range:[-1.5,1.5]}]});export{G as default};