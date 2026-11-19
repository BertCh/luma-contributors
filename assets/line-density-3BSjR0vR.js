import{a as e,d as t,g as n,v as r}from"./spatial-context-DwZEd4EZ.js";import{r as i}from"./scan-total-C4QddMmi.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-scan-B4NNTfF9.js";import{E as s,S as c,f as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{o as d}from"./sorted-segment-sums-Dvmm6OdR.js";import{c as f,o as p,t as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as h,t as g}from"./parameter-schema-BEeBYH77.js";import{t as _}from"./geodesic-wgsl-I0CYDKcz.js";import{t as v}from"./line-segmentize-kernels-J-DWl7up.js";import{a as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as b,t as x}from"./geometry-storage-validation-T89ZhmCj.js";import{t as S}from"./define-run-B-brkCe4.js";import{c as C,i as w}from"./pick-Bo3t3hjv.js";import{qs as T}from"./entry-preview-D70TdSbK.js";import{a as E,t as D}from"./geometry-Dz9W1tyx.js";function O(e,t){let n=e.coordinateSystem===`spherical`;return`
const LINE_COUNT: u32 = ${e.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${e.positions.length}u;
const COLUMNS: i32 = ${e.columns};
const ROWS: i32 = ${e.rows};
const CAPACITY: u32 = ${e.capacity}u;
const RADIUS: f32 = ${f(e.radius)};
const INFINITE: f32 = 3.0e38;
const EMITS_PIECES: bool = ${t};
${n?_:``}
${v}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getPieceLength(a: vec2<f32>, b: vec2<f32>, t0: f32, t1: f32) -> f32 {
  ${n?`return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;`:`return (t1 - t0) * length(b - a);`}
}

fn walkSegment(row: u32, slot: u32) -> u32 {
  let line = findLine(row);
  if (line == NO_LINE) {
    return 0u;
  }
  if (row + 1u >= min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT)) {
    return 0u;
  }
  let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let cellSize = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let a = getPosition(row);
  let b = getPosition(row + 1u);
  let ua = (a - origin) / cellSize;
  let delta = (b - origin) / cellSize - ua;
  // Liang-Barsky against [0, COLUMNS] x [0, ROWS].
  var t0 = 0.0;
  var t1 = 1.0;
  var p = array<f32, 4>(-delta.x, delta.x, -delta.y, delta.y);
  var q = array<f32, 4>(ua.x, f32(COLUMNS) - ua.x, ua.y, f32(ROWS) - ua.y);
  for (var side = 0u; side < 4u; side++) {
    if (p[side] == 0.0) {
      if (q[side] < 0.0) {
        return 0u;
      }
    } else {
      let r = q[side] / p[side];
      if (p[side] < 0.0) {
        if (r > t1) {
          return 0u;
        }
        t0 = max(t0, r);
      } else {
        if (r < t0) {
          return 0u;
        }
        t1 = min(t1, r);
      }
    }
  }
  if (t1 <= t0) {
    return 0u;
  }
  let start = ua + delta * t0;
  let end = ua + delta * t1;
  var cx = clamp(i32(floor(start.x)), 0, COLUMNS - 1);
  var cy = clamp(i32(floor(start.y)), 0, ROWS - 1);
  let endX = clamp(i32(floor(end.x)), 0, COLUMNS - 1);
  let endY = clamp(i32(floor(end.y)), 0, ROWS - 1);
  var stepsX = u32(abs(endX - cx));
  var stepsY = u32(abs(endY - cy));
  let pieceCount = stepsX + stepsY + 1u;
  // The counting pass only needs the piece count; walking the cells would be wasted work.
  if (!EMITS_PIECES) {
    return pieceCount;
  }
  let stepX = select(-1, 1, delta.x > 0.0);
  let stepY = select(-1, 1, delta.y > 0.0);
  var tMaxX = INFINITE;
  var tMaxY = INFINITE;
  if (stepsX > 0u) {
    tMaxX = t0 + (select(f32(cx), f32(cx + 1), delta.x > 0.0) - start.x) / delta.x;
  }
  if (stepsY > 0u) {
    tMaxY = t0 + (select(f32(cy), f32(cy + 1), delta.y > 0.0) - start.y) / delta.y;
  }
  let tDeltaX = select(INFINITE, 1.0 / abs(delta.x), delta.x != 0.0);
  let tDeltaY = select(INFINITE, 1.0 / abs(delta.y), delta.y != 0.0);
  var tCurrent = t0;
  for (var piece = 0u; piece < pieceCount; piece++) {
    let moveX = stepsX > 0u && (stepsY == 0u || tMaxX <= tMaxY);
    var tNext = t1;
    if (piece + 1u < pieceCount) {
      tNext = clamp(select(tMaxY, tMaxX, moveX), tCurrent, t1);
    }
    ${t?`if (slot + piece < CAPACITY) {
      keys[keysOffset + slot + piece] = u32(cy * COLUMNS + cx);
      contributions[contributionsOffset + slot + piece] = getPieceLength(a, b, tCurrent, tNext);
    }`:``}
    if (moveX) {
      cx += stepX;
      stepsX -= 1u;
      tMaxX += tDeltaX;
    } else {
      cy += stepY;
      stepsY -= 1u;
      tMaxY += tDeltaY;
    }
    tCurrent = tNext;
  }
  return pieceCount;
}
`}function k(e,t){return p(e,{id:t.id,operation:t.operation,variant:`count`,bindings:[{name:`positions`,view:t.constants.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.constants.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read_write`}],invocationCount:t.constants.positions.length,declarations:O(t.constants,!1),body:`counts[countsOffset + index] = walkSegment(index, 0u);`})}function A(e,t){return p(e,{id:t.id,operation:t.operation,variant:`emit`,bindings:[{name:`positions`,view:t.constants.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.constants.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`},{name:`contributions`,view:t.contributions,type:`f32`,access:`read_write`}],invocationCount:t.constants.positions.length,declarations:O(t.constants,!0),body:`walkSegment(index, starts[startsOffset + index]);`})}function j(e,t){let n=[{name:`scanTotal`,view:t.total,type:`u32`,access:`read`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}];return t.totalRecords&&n.push({name:`totalOut`,view:t.totalRecords,type:`u32`,access:`read_write`}),p(e,{id:t.id,operation:t.operation,variant:`publish`,bindings:n,invocationCount:1,body:`let total = scanTotal[scanTotalOffset];
  overflowOut[overflowOutOffset] = select(0u, 1u, total > ${t.capacity}u);
  ${t.totalRecords?`totalOut[totalOutOffset] = total;`:``}`})}function M(e,t){return p(e,{id:t.id,operation:t.operation,variant:`density`,bindings:[{name:`lengths`,view:t.lengths,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`densities`,view:t.densities,type:`f32`,access:`read_write`}],invocationCount:t.columns*t.rows,declarations:`const COLUMNS: u32 = ${t.columns}u;
const AREA_RADIUS: f32 = ${f(t.areaRadius)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;`,body:`let row = index / COLUMNS;
  let width = parameters[parametersOffset + 2u];
  let height = parameters[parametersOffset + 3u];
  ${t.coordinateSystem===`spherical`?`let south = parameters[parametersOffset + 1u] + f32(row) * height;
  // Exact area of a longitude/latitude cell on the sphere, R^2 dLon (sin(north) - sin(south)).
  // The sine difference is written as the product 2 cos(middle) sin(height / 2): the difference
  // of two float32 sines cancels for small cells (6 percent off at 0.0001 degrees).
  let area = AREA_RADIUS * AREA_RADIUS * width * DEGREES_TO_RADIANS * 2.0 *
    cos((south + 0.5 * height) * DEGREES_TO_RADIANS) * sin(0.5 * height * DEGREES_TO_RADIANS);`:`let area = width * height;`}
  densities[densitiesOffset + index] = lengths[lengthsOffset + index] / area;`})}var N=`GPULineDensity`,P=Object.freeze({...e,measure:`area`}),F=16777215,I=g({id:`line-density`,format:`float32`,wordLength:4,fields:[{name:`minimumX`,format:`float32`,wordOffset:0,defaultValue:0,dynamic:!0},{name:`minimumY`,format:`float32`,wordOffset:1,defaultValue:0,dynamic:!0},{name:`cellWidth`,format:`float32`,wordOffset:2,defaultValue:1,minimum:Number.MIN_VALUE,units:`position-units`,dynamic:!0},{name:`cellHeight`,format:`float32`,wordOffset:3,defaultValue:1,minimum:Number.MIN_VALUE,units:`position-units`,dynamic:!0}]});function L(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Line density target must hold 4 elements`);return t.set(h(I,{minimumX:e.minX,minimumY:e.minY,cellWidth:e.cellWidth,cellHeight:e.cellHeight})),t}var R=class{id;props;spatialContext;cellCount;coordinateSystem;radius;areaRadius;recordCapacity;constructor(e){this.id=e.id??`line-density`,this.props=e,this.spatialContext=r(this.id,e.spatialContext,P);let i=this.spatialContext.coordinateSpace===`longitude-latitude`;this.coordinateSystem=i?`spherical`:`planar`,this.radius=i?n(this.spatialContext):1,this.areaRadius=i?t(this.spatialContext):1,this.recordCapacity=e.recordCapacity??Math.max(1024,4*e.positions.length);let{id:a}=this;if(l(a,`columns`,e.columns),l(a,`rows`,e.rows),this.cellCount=e.columns*e.rows,c(a,`columns x rows`,this.cellCount,F),l(a,`recordCapacity`,this.recordCapacity),s(e.positions,[`float32x2`],`${a} positions`),e.positions.length<2)throw Error(`${a} needs two positions`);if(s(e.lineOffsets,[`uint32`],`${a} lineOffsets`),e.lineOffsets.length<2)throw Error(`${a} lineOffsets needs two rows`);if(s(e.parameters,[`float32`],`${a} parameters`),e.parameters.length<4)throw Error(`${a} parameters must hold 4 float32 values`);let{lengths:o,densities:d,status:f}=e.output,{overflow:p,requiredCount:m}=f;if(s(o,[`float32`],`${a} output.lengths`),o.length!==this.cellCount)throw Error(`${a} output.lengths must hold ${this.cellCount} rows`);if(d&&(s(d,[`float32`],`${a} output.densities`),d.length!==this.cellCount))throw Error(`${a} output.densities must hold ${this.cellCount} rows`);for(let[e,t]of[[`overflow`,p],[`requiredCount`,m]])if(t&&(s(t,[`uint32`],`${a} output.status.${e}`),t.length<1))throw Error(`${a} output.status.${e} must contain one uint32 row`);u(a,[o,d,p,m],[e.positions,e.lineOffsets,e.parameters])}getCommandNodes(e){let{props:t,id:n,cellCount:r,recordCapacity:s}=this,{output:c}=t;y(e,n,Object.fromEntries([t.positions,t.lineOffsets,t.parameters,c.lengths,c.densities,c.status.overflow,c.status.requiredCount].map((e,t)=>[String(t),e]))),b(e,n,{rowCountsAndScan:t.positions.length*4,...x(s),cellOffsets:(r+1)*4});let l=t.positions.length,u={positions:t.positions,lineOffsets:t.lineOffsets,columns:t.columns,rows:t.rows,coordinateSystem:this.coordinateSystem,radius:this.radius,capacity:s},f=a(e,`${n}-counts`,`uint32`,l),p=a(e,`${n}-starts`,`uint32`,l),h=a(e,`${n}-scan-total`,`uint32`,1),g=a(e,`${n}-keys`,`uint32`,s),_=a(e,`${n}-contributions`,`float32`,s);return[k(e,{id:`${n}-count`,operation:N,constants:u,parameters:t.parameters,counts:f}),...new o({id:`${n}-scan`,input:f,output:p,mode:`exclusive`}).getCommandNodes(e),...i(e,{id:n,operation:N,counts:f,offsets:p,rowCount:l,total:h}),m(e,{id:`${n}-fill-keys`,operation:N,view:g,type:`u32`,value:`${r}u`}),A(e,{id:`${n}-emit`,operation:N,constants:u,parameters:t.parameters,starts:p,keys:g,contributions:_}),j(e,{id:`${n}-publish`,operation:N,capacity:s,total:h,overflow:c.status.overflow,totalRecords:c.status.requiredCount}),...d(e,{id:`${n}-sum`,operation:N,segmentCount:r,segmentKeys:g,sumContributions:_,sums:c.lengths}),...c.densities?[M(e,{id:`${n}-density`,operation:N,columns:t.columns,rows:t.rows,coordinateSystem:this.coordinateSystem,areaRadius:this.areaRadius,parameters:t.parameters,lengths:c.lengths,densities:c.densities})]:[]]}},z=S({parameters:T,caption:`Length of lines per cell area; the lines are the input.`,createInputs:({random:e,world:t})=>{let n=D(e,t,40,14),r=n.offsets[18]-18,i=0;for(let e=0;e<r;e++)i+=Math.hypot(n.segments[e*4+2]-n.segments[e*4],n.segments[e*4+3]-n.segments[e*4+1]);return{width:t.width,lines:n,maximumDensity:8*i/t.width}},build(e,t,n){let r=n.rows,i=Math.round(t.width*r),{lines:a}=t,o=n.lineCount,s=a.offsets[o],c=e.upload(`positions`,`float32x2`,E(a.vertices.slice(0,s*2))),l=e.upload(`line-offsets`,`uint32`,a.offsets.slice(0,o+1)),u=e.allocate(`parameters`,`float32`,4);u.write(L({minX:0,minY:0,cellWidth:t.width/i,cellHeight:1/r}));let d=e.allocate(`lengths`,`float32`,i*r),f=e.allocate(`densities`,`float32`,i*r),p=e.allocate(`overflow`,`uint32`,1);return e.graph.add(new R({positions:c.view,lineOffsets:l.view,columns:i,rows:r,parameters:u.view,output:{lengths:d.view,densities:f.view,status:{overflow:p.view}}})),{outputs:{densities:f,columns:i,rows:r,segments:e.upload(`segments`,`float32x4`,a.segments),segmentCount:s-o,crossing:e.upload(`crossing`,`float32x4`,new Float32Array((s-o)*4).fill(NaN))}}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n}){let r=[0,0,t.width,1],i=w(t.lines.segments,e.x,e.y,n.segmentCount),a=C(n.columns,n.rows,r,i?.point[0]??e.x,i?.point[1]??e.y),o=t.width/n.columns,s=1/n.rows,c=[a.column*o,1-(a.row+1)*s],l=[c[0]+o,c[1]+s],u=new Float32Array(n.segmentCount*4).fill(NaN),d=0,f=0;for(let e=0;e<n.segmentCount;e++){let n=B(t.lines.segments.subarray(e*4,e*4+4),c,l);n&&(u.set(n,d*4),f+=Math.hypot(n[2]-n[0],n[3]-n[1]),d++)}return n.crossing.write(u),{focus:a.index,glyphs:[{shape:`box`,minimum:c,maximum:l}],readout:`cell ${a.column}, ${a.row} · length ${f.toFixed(3)} / area ${(o*s).toFixed(4)} = ${(f/(o*s)).toFixed(1)}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.densities,columns:t.columns,rows:t.rows,range:[0,e.maximumDensity]},{layer:`lines`,context:!0,segments:t.segments,count:t.segmentCount},{layer:`lines`,segments:t.crossing,count:t.segmentCount,radius:2.4,color:[1,.9,.35,1]}]});function B(e,t,n){let[r,i,a,o]=e,s=0,c=1,l=[[-(a-r),r-t[0]],[a-r,n[0]-r],[-(o-i),i-t[1]],[o-i,n[1]-i]];for(let[e,t]of l){if(e===0){if(t<0)return null;continue}let n=t/e;if(e<0?s=Math.max(s,n):c=Math.min(c,n),s>c)return null}return[r+(a-r)*s,i+(o-i)*s,r+(a-r)*c,i+(o-i)*c]}export{z as default};