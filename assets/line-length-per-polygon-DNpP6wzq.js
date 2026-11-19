import{_ as e,a as t,m as n,n as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{r as a}from"./kernel-C3tfQOn4.js";import{E as o,S as s,f as c,u as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{o as f}from"./sorted-segment-sums-Dvmm6OdR.js";import{c as p,o as m,t as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as g}from"./wgsl-common-DfyXiYB0.js";import{n as _,t as v}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as y}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as b}from"./geodesic-wgsl-I0CYDKcz.js";import{t as x}from"./line-segmentize-kernels-J-DWl7up.js";import{t as S}from"./gpu-spatial-join-candidates-DpXPR5vp.js";import{n as C}from"./spatial-join-geometry-DYRzw652.js";import{n as w,r as T}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{t as E}from"./define-run-B-brkCe4.js";import{u as D}from"./inputs-CJKKIRKK.js";import{Ks as O}from"./entry-preview-D70TdSbK.js";import{c as k,o as A,s as j}from"./geometry-Dz9W1tyx.js";var M=`GPULineLengthPerPolygon`,N=Object.freeze({...t,edgeModels:r,measure:`distance`});function P(e){return l(e,16)}function F(e){return l(e,8)}var I=class{id;props;polygonCount;spatialContext;coordinateSystem;radius;pairCapacity;maximumCrossingCount;constructor(t){this.id=t.id??`line-length-per-polygon`,this.props=t;let{id:r}=this;this.spatialContext=i(r,t.spatialContext,N),e(r,`polygons`,t.polygonsSpatialContext,this.spatialContext,N.edgeModels);let a=this.spatialContext.coordinateSpace===`longitude-latitude`;if(this.coordinateSystem=a?`spherical`:`planar`,this.radius=n(this.spatialContext)??1,this.pairCapacity=t.pairCapacity??Math.max(1024,8*t.positions.length),this.maximumCrossingCount=t.maximumCrossingCount??64,c(r,`pairCapacity`,this.pairCapacity),c(r,`maximumCrossingCount`,this.maximumCrossingCount,1,1024),o(t.positions,[`float32x2`],`${r} positions`),t.positions.length<2)throw Error(`${r} needs at least two positions`);if(o(t.lineOffsets,[`uint32`],`${r} lineOffsets`),t.lineOffsets.length<2)throw Error(`${r} lineOffsets needs two rows`);if(t.lineWeights&&(o(t.lineWeights,[`float32`],`${r} lineWeights`),t.lineWeights.length!==t.lineOffsets.length-1))throw Error(`${r} lineWeights must hold one row per line`);if(t.polygons.kind!==`polygons`)throw Error(`${r} polygons must have kind 'polygons'`);for(let[e,n]of[[`featureOffsets`,t.polygons.featureOffsets],[`polygonOffsets`,t.polygons.polygonOffsets],[`ringOffsets`,t.polygons.ringOffsets]])o(n,[`uint32`],`${r} polygons.${e}`);if(o(t.polygons.positions,[`float32x2`],`${r} polygons.positions`),this.polygonCount=t.polygons.featureOffsets.length-1,this.polygonCount<1)throw Error(`${r} polygons must contain at least one feature`);let{lengths:s,weightedLengths:l,segmentCounts:u,status:f}=t.output,{overflow:p}=f;if(o(s,[`float32`],`${r} output.lengths`),s.length!==this.polygonCount)throw Error(`${r} output.lengths must hold ${this.polygonCount} rows`);if(l){if(!t.lineWeights)throw Error(`${r} output.weightedLengths requires lineWeights`);if(o(l,[`float32`],`${r} output.weightedLengths`),l.length!==this.polygonCount)throw Error(`${r} output.weightedLengths must hold ${this.polygonCount} rows`)}if(u&&(o(u,[`uint32`],`${r} output.segmentCounts`),u.length!==this.polygonCount))throw Error(`${r} output.segmentCounts must hold ${this.polygonCount} rows`);if(o(p,[`uint32`],`${r} output.status.overflow`),p.length<1)throw Error(`${r} output.status.overflow must contain one uint32 row`);for(let e of[`antimeridianUncertainCount`,`poleUncertainCount`,`candidateOverflow`]){let t=f[e];if(t&&(o(t,[`uint32`],`${r} output.status.${e}`),t.length<1))throw Error(`${r} output.status.${e} must contain one uint32 row`)}d(r,[s,l,u,p,f.candidateOverflow,f.antimeridianUncertainCount,f.poleUncertainCount],[t.positions,t.lineOffsets,t.lineWeights,t.polygons.positions,t.polygons.featureOffsets,t.polygons.polygonOffsets,t.polygons.ringOffsets])}getCommandNodes(e){let{props:t,id:n,polygonCount:r,maximumCrossingCount:i}=this,{polygons:o,output:l}=t;u(n,e,[t.positions,t.lineOffsets,t.lineWeights,o.positions,o.featureOffsets,o.polygonOffsets,o.ringOffsets,l.lengths,l.weightedLengths,l.segmentCounts,l.status.overflow,l.status.candidateOverflow,l.status.antimeridianUncertainCount,l.status.poleUncertainCount]),s(n,`positions rows`,t.positions.length,P(e.device));let d=F(e.device),E=t.pairCapacity??Math.min(this.pairCapacity,d);c(n,`pairCapacity`,E,1,d);let D=this.coordinateSystem===`spherical`,O=t.positions.length,k=t.lineOffsets.length-1,A=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&T(l.status)){let t=l.status;for(let r of[`antimeridianUncertainCount`,`poleUncertainCount`]){let i=t[r];i&&A.push(h(e,{id:`${n}-clear-${r}`,operation:M,view:i,type:`u32`,value:`0u`,componentCount:1}))}A.push(w(e,{id:`${n}-lon-lat-polygon-check-0`,operation:M,positions:o.positions,ringOffsets:o.ringOffsets,status:t}))}let j=a(e,`${n}-segment-positions`,`float32x2`,2*O),N=a(e,`${n}-segment-offsets`,`uint32`,O+1),I=a(e,`${n}-segment-weights`,`float32`,O);A.push(m(e,{id:`${n}-build-segments`,operation:M,variant:`build-segments`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`lineWeights`,view:t.lineWeights??l.lengths,type:`f32`,access:`read`},{name:`segmentPositions`,view:j,type:`f32`,access:`read_write`},{name:`segmentOffsets`,view:N,type:`u32`,access:`read_write`},{name:`segmentWeights`,view:I,type:`f32`,access:`read_write`}],invocationCount:O+1,declarations:`const LINE_COUNT: u32 = ${k}u;
const ROW_COUNT: u32 = ${O}u;
const HAS_WEIGHTS: bool = ${t.lineWeights?`true`:`false`};
${x}`,body:`segmentOffsets[segmentOffsetsOffset + index] = 2u * index;
  if (index < ROW_COUNT) {
    let line = findLine(index);
    var valid = line != NO_LINE;
    var weight = 1.0;
    if (valid) {
      valid = index + 1u < min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
      if (HAS_WEIGHTS) { weight = lineWeights[lineWeightsOffset + line]; }
    }
    var nanBits = 0x7fc00000u;
    let nan = bitcast<f32>(nanBits);
    var a = vec2f(nan);
    var b = vec2f(nan);
    if (valid) {
      a = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
      b = vec2f(positions[positionsOffset + 2u * index + 2u], positions[positionsOffset + 2u * index + 3u]);
    }
    segmentPositions[segmentPositionsOffset + 4u * index] = a.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 1u] = a.y;
    segmentPositions[segmentPositionsOffset + 4u * index + 2u] = b.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 3u] = b.y;
    segmentWeights[segmentWeightsOffset + index] = select(0.0, weight, valid);
  }`}));let L={leftRows:a(e,`${n}-pair-left`,`uint32`,E),rightRows:a(e,`${n}-pair-right`,`uint32`,E),status:{count:a(e,`${n}-pair-count`,`uint32`,1),overflow:a(e,`${n}-pair-overflow`,`uint32`,1)}};A.push(...new S({id:`${n}-candidates`,left:{kind:`lines`,positions:j,lineOffsets:N},right:o,output:L}).getCommandNodes(e));let R=a(e,`${n}-feature-rings`,`uint32x2`,r);A.push(C(e,`${n}-polygon-rings`,{featureCount:r,geometry:o,featureRings:R}));let z=a(e,`${n}-crossing-overflow`,`uint32`,1),B=a(e,`${n}-inside-lengths`,`float32`,E);A.push(h(e,{id:`${n}-fill-crossing-overflow`,operation:M,view:z,type:`u32`,value:`0u`}),m(e,{id:`${n}-clip`,operation:M,variant:D?`clip-spherical`:`clip-planar`,bindings:[{name:`leftRows`,view:L.leftRows,type:`u32`,access:`read`},{name:`rightRows`,view:L.rightRows,type:`u32`,access:`read`},{name:`segmentPositions`,view:j,type:`f32`,access:`read`},{name:`featureRings`,view:R,type:`u32`,access:`read`},{name:`ringOffsets`,view:o.ringOffsets,type:`u32`,access:`read`},{name:`polygonPositions`,view:o.positions,type:`f32`,access:`read`},{name:`insideLengths`,view:B,type:`f32`,access:`read_write`},{name:`crossingOverflow`,view:z,type:`atomic<u32>`,access:`read_write`}],invocationCount:E,declarations:`const NO_FEATURE: u32 = 0xffffffffu;
const VERTEX_COUNT: u32 = ${o.positions.length}u;
const MAXIMUM_CROSSINGS: u32 = ${i}u;
const RADIUS: f32 = ${p(this.radius)};
const EDGE_TOLERANCE: f32 = 1.0e-6;
${D?b:``}
${g}
${y}
${v}

fn getVertex(row: u32) -> vec2f {
  return vec2f(polygonPositions[polygonPositionsOffset + 2u * row], polygonPositions[polygonPositionsOffset + 2u * row + 1u]);
}

var<private> segmentOrigin: vec2f;

fn getRelativeVertex(row: u32) -> vec2f {
  return getVertex(row) - segmentOrigin;
}

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn getPieceLength(a: vec2f, b: vec2f, t0: f32, t1: f32) -> f32 {
  ${D?`return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;`:`return (t1 - t0) * length(b - a);`}
}

${_(`getRingWinding`,`getRelativeVertex`)}
// Even/odd fill over every ring of the feature, with the shared exact point-in-ring rule
// (utils/exact-geometry-wgsl.ts). Called from one site.
fn isInside(point: vec2f, ringStart: u32, ringEnd: u32) -> bool {
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  var winding = 0;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    winding += getRingWinding(probe, first, last).x;
  }
  return (winding & 1) != 0;
}`,body:`var inside = 0.0;
  let segment = leftRows[leftRowsOffset + index];
  let feature = rightRows[rightRowsOffset + index];
  if (segment != NO_FEATURE && feature != NO_FEATURE) {
    let a = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment], segmentPositions[segmentPositionsOffset + 4u * segment + 1u]);
    let b = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment + 2u], segmentPositions[segmentPositionsOffset + 4u * segment + 3u]);
    let delta = b - a;
    segmentOrigin = a;
    let ringStart = featureRings[featureRingsOffset + 2u * feature];
    let ringEnd = featureRings[featureRingsOffset + 2u * feature + 1u];
    if (!isNanBits(a.x) && !isNanBits(a.y) && !isNanBits(b.x) && !isNanBits(b.y) && (delta.x != 0.0 || delta.y != 0.0)) {
      var crossings = array<f32, ${i}>();
      var crossingCount = 0u;
      var truncated = false;
      for (var ring = ringStart; ring < ringEnd; ring++) {
        let first = ringOffsets[ringOffsetsOffset + ring];
        let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
        if (last < first + 3u) { continue; }
        var previous = getVertex(last - 1u);
        for (var row = first; row < last; row++) {
          let current = getVertex(row);
          let edge = current - previous;
          let denominator = cross2(delta, edge);
          if (denominator != 0.0) {
            let offset = previous - a;
            let t = cross2(offset, edge) / denominator;
            let u = cross2(offset, delta) / denominator;
            // Extra breakpoints are harmless (pieces are tested by midpoint), so edges are
            // inclusive with a small tolerance and a vertex crossing is never dropped.
            if (t > 0.0 && t < 1.0 && u >= -EDGE_TOLERANCE && u <= 1.0 + EDGE_TOLERANCE) {
              if (crossingCount < MAXIMUM_CROSSINGS) {
                var slot = crossingCount;
                crossingCount++;
                loop {
                  if (slot == 0u || crossings[slot - 1u] <= t) { break; }
                  crossings[slot] = crossings[slot - 1u];
                  slot--;
                }
                crossings[slot] = t;
              } else {
                truncated = true;
              }
            }
          }
          previous = current;
        }
      }
      if (truncated) { atomicStore(&crossingOverflow[crossingOverflowOffset], 1u); }
      var start = 0.0;
      for (var piece = 0u; piece <= crossingCount; piece++) {
        let end = select(1.0, crossings[min(piece, MAXIMUM_CROSSINGS - 1u)], piece < crossingCount);
        if (end > start) {
          // Classify in the segment's local frame. Adding the world origin can round a
          // genuine interior midpoint onto a hole edge, losing a contributing segment.
          if (isInside(delta * (0.5 * (start + end)), ringStart, ringEnd)) {
            inside += getPieceLength(a, b, start, end);
          }
          start = end;
        }
      }
    }
  }
  insideLengths[insideLengthsOffset + index] = inside;`}));let V=l.segmentCounts,H=a(e,`${n}-keys`,`uint32`,E),U=a(e,`${n}-weighted`,`float32`,E);return V&&A.push(h(e,{id:`${n}-fill-counts`,operation:M,view:V,type:`u32`,value:`0u`})),A.push(m(e,{id:`${n}-finish-pairs`,operation:M,variant:`finish-pairs`,bindings:[{name:`leftRows`,view:L.leftRows,type:`u32`,access:`read`},{name:`rightRows`,view:L.rightRows,type:`u32`,access:`read`},{name:`insideLengths`,view:B,type:`f32`,access:`read`},{name:`segmentWeights`,view:I,type:`f32`,access:`read`},{name:`keys`,view:H,type:`u32`,access:`read_write`},{name:`weighted`,view:U,type:`f32`,access:`read_write`},...V?[{name:`counts`,view:V,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:E,declarations:`const NO_FEATURE: u32 = 0xffffffffu;
const POLYGON_COUNT: u32 = ${r}u;`,body:`let segment = leftRows[leftRowsOffset + index];
  let feature = rightRows[rightRowsOffset + index];
  let inside = insideLengths[insideLengthsOffset + index];
  var key = POLYGON_COUNT;
  var contribution = 0.0;
  if (segment != NO_FEATURE && feature < POLYGON_COUNT && inside > 0.0) {
    key = feature;
    contribution = inside * segmentWeights[segmentWeightsOffset + segment];
    ${V?`atomicAdd(&counts[countsOffset + feature], 1u);`:``}
  }
  keys[keysOffset + index] = key;
  weighted[weightedOffset + index] = contribution;`}),...f(e,{id:`${n}-sum`,operation:M,segmentCount:r,segmentKeys:H,sumContributions:B,sums:l.lengths,reductions:l.weightedLengths?[{name:`weighted`,contributions:U,output:l.weightedLengths}]:[]}),m(e,{id:`${n}-publish`,operation:M,variant:`publish`,bindings:[{name:`pairOverflow`,view:L.status.overflow,type:`u32`,access:`read`},{name:`crossingOverflow`,view:z,type:`u32`,access:`read`},{name:`overflow`,view:l.status.overflow,type:`u32`,access:`read_write`},...l.status.candidateOverflow?[{name:`candidateOverflow`,view:l.status.candidateOverflow,type:`u32`,access:`read_write`}]:[]],invocationCount:1,body:`overflow[overflowOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || crossingOverflow[crossingOverflowOffset] != 0u);${l.status.candidateOverflow?`
  candidateOverflow[candidateOverflowOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u);`:``}`})),A}},L=[[43,35,2033],[3,35,1279],[39,35,1247],[80,35,1245],[20,35,1176],[78,35,1063],[83,35,1042],[16,35,954],[58,35,905],[109,35,801],[9,35,795],[126,35,791],[91,35,739],[116,35,663],[73,35,662],[63,35,658],[90,35,631],[31,35,619],[139,35,614],[65,35,594],[45,35,538],[92,35,529],[57,35,516],[123,35,509],[119,35,492],[75,35,446],[87,35,439],[108,35,429],[47,35,426],[98,35,420],[121,35,413],[96,35,410],[93,35,402],[71,35,399],[46,35,399],[29,35,391],[133,35,390],[106,35,388],[143,35,361],[83,94,353],[48,35,330],[76,35,315],[104,35,301],[25,35,275],[43,57,272],[51,35,269],[30,35,267],[78,57,266],[21,35,256],[54,35,250],[43,58,245],[56,35,238],[62,35,227],[61,35,223],[43,20,212],[18,35,210],[19,35,203],[20,57,198],[14,35,197],[80,57,196],[39,57,191],[3,16,189],[23,35,183],[49,35,183],[52,35,181],[11,35,179],[10,35,175],[24,35,175],[3,57,173],[1,35,169],[16,20,168],[43,16,164],[28,35,164],[20,16,157],[3,20,155],[2,35,147],[3,0,134],[33,35,134],[39,16,132],[3,9,130],[35,20,128],[8,35,122],[37,35,121],[9,16,118],[13,35,115],[35,16,112],[12,35,112],[31,16,108],[3,29,107],[17,35,105],[0,35,105],[20,9,86],[15,35,85],[5,35,83],[9,20,80],[20,0,62],[9,0,57],[16,9,55],[16,0,51],[11,0,44],[21,9,41],[8,0,39],[16,3,37],[2,0,32],[19,9,31],[21,20,31],[18,9,30],[9,3,29],[23,9,26],[1,16,25],[18,16,24],[19,20,24],[13,0,24],[21,0,23],[19,16,23],[23,16,23],[10,9,23],[12,9,21],[21,16,21],[23,20,20],[11,16,20],[17,16,19],[0,16,19],[11,5,19],[14,0,19],[24,16,19],[1,0,18],[7,0,18],[14,5,16],[6,0,16],[11,9,16],[12,0,16],[3,5,15],[10,0,15],[2,5,14],[15,0,14],[8,3,13],[2,9,13],[8,5,12],[4,0,12],[0,9,11],[10,3,11],[11,3,10],[1,9,10],[13,5,9],[11,8,9],[3,6,9],[8,9,9],[10,5,8],[13,9,8],[14,9,8],[3,10,8],[11,2,7],[2,8,7]],R=7,z=35,B=E({parameters:O,caption:`Length of straight home-to-work flow lines inside each Chicago tract.`,createInputs:({world:e})=>({world:e,destination:{x:0,y:0,isMoved:!1},defaultTracts:D(()=>.5,e,R)}),build(e,{world:t,destination:n},r){let i=D(()=>.5,t,r.tractRows),a=L.filter(([e,t])=>e<i.count&&t<i.count).slice(0,r.lineCount),o=Float32Array.from(a.flatMap(([e,t])=>[i.centers[e*2],i.centers[e*2+1],i.centers[t*2],i.centers[t*2+1]])),s=e.upload(`positions`,`float32x2`,o),c=e.upload(`segments`,`float32x4`,o),l=Uint32Array.from({length:a.length+1},(e,t)=>t*2),u=e.allocate(`lengths`,`float32`,i.count),d=e.allocate(`overflow`,`uint32`,1);return e.graph.add(new I({positions:s.view,lineOffsets:e.upload(`line-offsets`,`uint32`,l).view,polygons:{kind:`polygons`,...j(e,A(i))},output:{lengths:u.view,status:{overflow:d.view}}})),{outputs:{lengths:u,...k(e,i),segments:c,segmentCount:a.length,triangleCount:i.triangleCount,outlineCount:i.outlineCount},update:()=>{let e=Float32Array.from(o);a.forEach(([,t],r)=>{t===z&&n.isMoved&&(e[r*4+2]=n.x,e[r*4+3]=n.y)}),s.write(e),c.write(e)}}},rebuildOn:[`lineCount`,`tractRows`],interaction:{kind:`drag`,hint:`move the workplace`,rest:({inputs:e,toUnit:t})=>t([e.defaultTracts.centers[70],e.defaultTracts.centers[71]]),resolve(e,{inputs:t,outputs:n,peek:r}){t.destination.x=e.x,t.destination.y=e.y,t.destination.isMoved=!0;let i=r(n.lengths);return{rerun:!0,glyphs:[{shape:`crosshair`,at:[e.x,e.y],size:8}],readout:i?`${n.segmentCount} flows · longest in one tract ${Math.max(...i).toFixed(2)}`:`${n.segmentCount} flows`}}},draw:({outputs:e})=>[{layer:`polygons`,positions:e.vertices,indices:e.indices,owners:e.owners,values:e.lengths,triangleCount:e.triangleCount},{layer:`lines`,context:!0,segments:e.outline,count:e.outlineCount,alpha:.5},{layer:`lines`,context:!0,color:[1,1,1,1],segments:e.segments,count:e.segmentCount}]});export{B as default};