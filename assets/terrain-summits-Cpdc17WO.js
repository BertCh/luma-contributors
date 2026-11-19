import{n as e,r as t}from"./terrain-binding-preflight-laSEObQE.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-compaction-B6IzcF0P.js";import{i,n as a,r as o}from"./gpu-raster-extrema-pyramid-BYXR99nY.js";import{n as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,S as d,m as f,x as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m,f as h,r as g,u as _}from"./raster-grid-VB0AM_GY.js";import{c as v,d as y,h as b,l as x,n as S,o as C,r as w,t as T,u as E}from"./raster-band-utils-BcTxc8_Y.js";import{a as D,i as O,n as k,o as A,r as j,t as M}from"./terrain-features-utils-BLqx9E-Z.js";import{t as N}from"./define-run-B-brkCe4.js";import{l as P,s as F,u as I}from"./pointer-overlay-Qh8qflcA.js";import{a as L,f as R,i as z,m as B,n as V,o as H,u as U}from"./terrain-Cc9dW1uq.js";import{C as W}from"./entry-preview-D70TdSbK.js";var G={cellSizeIndex:2,northEdgeIndex:4};function K(e,t,n=new Float32Array(8)){let r=`getGPUTerrainSummitsParameterValues`;return C(r,n,8),A(r,`radius`,e.radius,0,!0),A(r,`minimumDrop`,e.minimumDrop??0,0),O(r,g(t).cellSize),n[0]=e.radius,n[1]=e.minimumDrop??0,n[6]=0,n[7]=0,h(n,t,G),n}var q=class{id;props;pyramidLayout;constructor(e){this.id=e.id??`terrain-summits`,this.props=e;let{id:t}=this,{output:n}=e,r=n.status?.radiusClamped,i=E(t,e.grid);if(_(t,e.cellSizeMode??`uniform`),y(t,e.grid,e.cellSizeMode),D(t,e.maximumRadiusPixels),e.incompleteNeighborhood!==void 0&&e.incompleteNeighborhood!==`reject`&&e.incompleteNeighborhood!==`ignore`)throw Error(`${t} incompleteNeighborhood must be 'reject' or 'ignore'`);if(!n.summitMask&&!n.drop&&!n.summits)throw Error(`${t} requires at least one output`);if(b(t,e.parameters,8),e.pyramid){if(e.incompleteNeighborhood===void 0||e.incompleteNeighborhood===`reject`)throw Error(`${t} pyramid requires incompleteNeighborhood 'ignore'`);this.pyramidLayout=X(t,e.pyramid,e.grid)}else this.pyramidLayout=o(e.grid,{firstBlockSize:J,footprint:`cell`});if(n.summitMask&&(u(n.summitMask,[`uint32`],`${t} output.summitMask`),n.summitMask.length!==i))throw Error(`${t} output.summitMask must contain one value per pixel`);if(n.drop&&(u(n.drop,[`float32`],`${t} output.drop`),n.drop.length!==i))throw Error(`${t} output.drop must contain one value per pixel`);if(n.summits&&f(t,n.summits),n.summitDrop){if(!n.summits)throw Error(`${t} output.summitDrop requires output.summits`);if(u(n.summitDrop,[`float32`],`${t} output.summitDrop`),n.summitDrop.length<n.summits.ids.length)throw Error(`${t} output.summitDrop must contain at least summits.ids.length rows`)}if(r&&(u(r,[`uint32`],`${t} output.status.radiusClamped`),r.length<1))throw Error(`${t} output.status.radiusClamped must contain one row`);x(t,[n.summitMask,n.drop,r,n.summitDrop,n.summits?.ids,n.summits?.status.count,n.summits?.status.overflow,n.summits?.status.requiredCount],[...S(e.elevation),e.parameters,...e.pyramid?[e.pyramid.combined]:[]])}getCommandNodes(i){t(i,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height,"pyramid fused parameters":e(this.props.pyramid?void 0:this.pyramidLayout),"maximum pyramid":this.props.pyramid?0:4*this.pyramidLayout.length});let{id:o,props:u}=this,{width:d,height:f}=u.grid,m=u.output.summits,h=u.output.status?.radiusClamped;v(o,i,u.elevation,[]),p(o,i,[u.parameters,u.pyramid?.combined,u.output.summitMask,u.output.drop,h,u.output.summitDrop,m?.ids,m?.status.count,m?.status.overflow,m?.status.requiredCount]);let g=d*f,_=u.cellSizeMode??`uniform`,y=D(o,u.maximumRadiusPixels),b=w(i,o,u.elevation,u.grid,!0),x=[...b.nodes],S=b.band.validity,C=u.output.summitMask??(m?n(i,`${o}-summit-mask`,`uint32`,g):void 0),T=u.output.drop??(u.output.summitDrop?n(i,`${o}-drop`,`float32`,g):void 0),{pyramidLayout:E}=this,O=(u.incompleteNeighborhood??`reject`)===`reject`,k;if(u.pyramid)k=u.pyramid.combined;else{k=n(i,`${o}-pyramid-maximum`,`float32`,E.length);let e=b.band.storage.values,t=S;O&&(e=n(i,`${o}-pyramid-values`,`float32`,g),t=n(i,`${o}-pyramid-validity`,`uint32`,g),x.push(c(i,{id:`${o}-pyramid-input`,operation:`GPUTerrainSummits`,variant:`pyramid-input`,bindings:[{name:`sourceValues`,view:b.band.storage.values,type:`f32`,access:`read`},{name:`sourceValidity`,view:S,type:`u32`,access:`read`},{name:`blockValues`,view:e,type:`f32`,access:`read_write`},{name:`blockValidity`,view:t,type:`u32`,access:`read_write`}],invocationCount:g,body:`let isValid = sourceValidity[sourceValidityOffset + index] != 0u;
  blockValues[blockValuesOffset + index] = select(${Y}, sourceValues[sourceValuesOffset + index], isValid);
  blockValidity[blockValidityOffset + index] = 1u;`}))),x.push(...a(i,{id:`${o}-pyramid`,layout:E,values:e,validity:t,maximum:k}))}let A=[{name:`elevationValues`,view:b.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:S,type:`u32`,access:`read`},{name:`settings`,view:u.parameters,type:`f32`,access:`read`},{name:`pyramid`,view:k,type:`f32`,access:`read`},...C?[{name:`maskValues`,view:C,type:`u32`,access:`read_write`}]:[],...T?[{name:`dropValues`,view:T,type:`f32`,access:`read_write`}]:[],...h?[{name:`overflowValues`,view:h,type:`atomic<u32>`,access:`read_write`}]:[]];if(h&&x.push(l(i,{id:`${o}-overflow-clear`,operation:`GPUTerrainSummits`,view:h,type:`u32`,value:`0u`,componentCount:1})),x.push(c(i,{id:`${o}-summits`,operation:`GPUTerrainSummits`,variant:`disc-${_}-${u.incompleteNeighborhood??`reject`}`,bindings:A,invocationCount:g,declarations:Q({width:d,height:f,maximumRadiusPixels:y,cellSizeMode:_,rejectIncomplete:(u.incompleteNeighborhood??`reject`)===`reject`,reportsClamp:!!h,layout:E}),body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  summitDrop = getQuietNaN(index);
  let isSummit = evaluateSummit(column, row, index);
  ${C?`maskValues[maskValuesOffset + index] = select(0u, 1u, isSummit);`:``}
  ${T?`dropValues[dropValuesOffset + index] = select(getQuietNaN(index), summitDrop, isSummit);`:``}`})),m&&C){let e=n(i,`${o}-pixel-ids`,`uint32`,g),t=n(i,`${o}-compact-rows`,`uint32`,g),a=n(i,`${o}-summit-total`,`uint32`,1);x.push(c(i,{id:`${o}-pixel-ids`,operation:`GPUTerrainSummits`,variant:`pixel-ids`,bindings:[{name:`pixelIds`,view:e,type:`u32`,access:`read_write`}],invocationCount:g,body:`pixelIds[pixelIdsOffset + index] = index;`}),...new r({id:`${o}-compaction`,input:e,flags:C,output:t,count:a}).getCommandNodes(i),s(i,{id:`${o}-publish`,operation:`GPUTerrainSummits`,requiredCount:a,compactIds:t,output:m})),u.output.summitDrop&&T&&x.push(c(i,{id:`${o}-output-drop`,operation:`GPUTerrainSummits`,variant:`gather-drop`,bindings:[{name:`idsIn`,view:m.ids,type:`u32`,access:`read`},{name:`countIn`,view:m.status.count,type:`u32`,access:`read`},{name:`dropIn`,view:T,type:`f32`,access:`read`},{name:`dropOut`,view:u.output.summitDrop,type:`f32`,access:`read_write`}],invocationCount:m.ids.length,body:`if (index < countIn[countInOffset]) {
    dropOut[dropOutOffset + index] = dropIn[dropInOffset + idsIn[idsInOffset + index]];
  } else {
    dropOut[dropOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
  }`}))}return x}},J=2,Y=`3.4028234663852886e38`;function X(e,t,n){let{layout:r,combined:i}=t;if(!r||!i)throw Error(`${e} pyramid must provide a layout and a combined view`);if(r.width!==n.width||r.height!==n.height)throw Error(`${e} pyramid covers ${r.width}x${r.height} pixels but the grid is ${n.width}x${n.height}`);if(r.footprint!==`cell`)throw Error(`${e} pyramid must use the 'cell' footprint`);let a=r.levels[r.levels.length-1];if(!a||a.width!==1||a.height!==1)throw Error(`${e} pyramid must reach a single top cell`);if(u(i,[`float32`],`${e} pyramid combined`),i.length!==2*r.length)throw Error(`${e} pyramid combined must contain ${2*r.length} values`);return r}function Z(e){return(3*e.levels.length+4)*3*4}function Q(e){let{width:t,height:n,maximumRadiusPixels:r,cellSizeMode:a,layout:o}=e,s=3*o.levels.length+4;d(`GPUTerrainSummits`,`pyramid stack bytes plus 256 bytes of private words`,Z(o)+256,8191);let c=o.levels.length-1;return`const WIDTH: u32 = ${t}u;
const HEIGHT: u32 = ${n}u;
const MAXIMUM_RADIUS_PIXELS: i32 = ${r};
const REJECT_INCOMPLETE: bool = ${e.rejectIncomplete};
const NO_INDEX: u32 = 0xffffffffu;
const FLOAT_MAXIMUM: f32 = ${Y};
const STACK_SIZE: u32 = ${s}u;
${j}
${T}
${m(a,G)}
${k}
${M}
${i(o,`pyramid`)}
var<private> summitDrop: f32;
var<private> stackLevel: array<u32, ${s}>;
var<private> stackColumn: array<u32, ${s}>;
var<private> stackRow: array<u32, ${s}>;
// Whether a pixel of height \`other\` beats the centre: >= where it has the smaller index, else >.
fn beatsCentre(other: f32, ownHeight: f32, hasSmallerIndex: bool) -> bool {
  return select((other > ownHeight), (other >= ownHeight), hasSmallerIndex);
}
// Returns true when no disc pixel of rows [rowLow, rowHigh] beats the centre pixel.
fn isHalfDiscDominated(column: i32, row: i32, rowLow: i32, rowHigh: i32, ownHeight: f32,
    hasSmallerIndex: bool, exact: bool, cell: vec2<f32>, radiusSquared: f32) -> bool {
  if (rowLow > rowHigh) { return true; }
  var depth = 1u;
  stackLevel[0] = ${c}u;
  stackColumn[0] = 0u;
  stackRow[0] = 0u;
  loop {
    if (depth == 0u) { break; }
    depth--;
    let level = stackLevel[depth];
    let cellColumn = i32(stackColumn[depth]);
    let cellRow = i32(stackRow[depth]);
    let blockSize = i32(pyramidLevelBlockSize(level));
    let x0 = cellColumn * blockSize;
    let y0 = cellRow * blockSize;
    let x1 = min(x0 + blockSize, i32(WIDTH)) - 1;
    let y1 = min(y0 + blockSize, i32(HEIGHT)) - 1;
    let ya = max(y0, rowLow);
    let yb = min(y1, rowHigh);
    if (ya > yb) { continue; }
    let dxLow = x0 - column;
    let dxHigh = x1 - column;
    let dyLow = ya - row;
    let dyHigh = yb - row;
    // Block pixel nearest the centre: disjoint from the disc when it is outside.
    var nearX = 0;
    if (dxLow > 0) { nearX = dxLow; } else if (dxHigh < 0) { nearX = dxHigh; }
    var nearY = 0;
    if (dyLow > 0) { nearY = dyLow; } else if (dyHigh < 0) { nearY = dyHigh; }
    if (!isOffsetInDisc(nearX, nearY, cell, radiusSquared)) { continue; }
    let farX = max(abs(dxLow), abs(dxHigh));
    let farY = max(abs(dyLow), abs(dyHigh));
    if (!exact && ya == y0 && yb == y1 && isOffsetInDisc(farX, farY, cell, radiusSquared)) {
      let blockMaximum = pyramid[pyramidOffset + pyramidLevelIndex(level, u32(x0), u32(y0))];
      if (beatsCentre(blockMaximum, ownHeight, hasSmallerIndex)) { return false; }
      continue;
    }
    if (level == 0u) {
      for (var y = ya; y <= yb; y++) {
        for (var x = x0; x <= x1; x++) {
          if (!isOffsetInDisc(x - column, y - row, cell, radiusSquared)) { continue; }
          let pixelIndex = u32(y) * WIDTH + u32(x);
          if (elevationValidity[elevationValidityOffset + pixelIndex] == 0u) {
            if (REJECT_INCOMPLETE) { return false; }
            continue;
          }
          if (beatsCentre(elevationValues[elevationValuesOffset + pixelIndex], ownHeight, hasSmallerIndex)) {
            return false;
          }
        }
      }
      continue;
    }
    let childSize = i32(pyramidLevelBlockSize(level - 1u));
    for (var child = 0; child < 4; child++) {
      let childColumn = 2 * cellColumn + (child & 1);
      let childRow = 2 * cellRow + (child >> 1);
      if (childColumn * childSize >= i32(WIDTH) || childRow * childSize >= i32(HEIGHT)) { continue; }
      stackLevel[depth] = level - 1u;
      stackColumn[depth] = u32(childColumn);
      stackRow[depth] = u32(childRow);
      depth++;
    }
  }
  return true;
}
// Returns whether the pixel is a summit and leaves its drop in summitDrop.
fn evaluateSummit(column: i32, row: i32, index: u32) -> bool {
  let cell = getGroundCellSize(u32(row));
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0)) {
    return false;
  }
  let requested = settings[settingsOffset];
  let minimumDrop = settings[settingsOffset + 1u];
  if (!(isFiniteValue(requested) && requested > 0.0)) { return false; }
  ${e.reportsClamp?`// The clamp depends only on the row's cell size: one pixel per row reports it, so at most
  // HEIGHT invocations touch the flag word instead of every pixel (review C-17).
  if (column == 0 && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }`:``}
  if (elevationValidity[elevationValidityOffset + index] == 0u) { return false; }
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let centre = vec2<f32>(f32(column), f32(row));
  let ownHeight = elevationValues[elevationValuesOffset + index];
  // Cheap rejection first: almost every pixel has a higher disc pixel among its 8 neighbors, so the
  // search below only runs for 3x3 local maxima. Every rejection here would also be reached by
  // the full search (higher pixel, or an incomplete neighborhood under 'reject'), so results match.
  for (var nearY = -1; nearY <= 1; nearY++) {
    for (var nearX = -1; nearX <= 1; nearX++) {
      let nearPixel = vec2<i32>(column + nearX, row + nearY);
      if ((nearX == 0 && nearY == 0) || !isInDisc(nearPixel, centre, cell, radiusSquared)) { continue; }
      if (nearPixel.x < 0 || nearPixel.y < 0 || nearPixel.x >= i32(WIDTH) || nearPixel.y >= i32(HEIGHT)) { continue; }
      let nearIndex = u32(nearPixel.y) * WIDTH + u32(nearPixel.x);
      if (elevationValidity[elevationValidityOffset + nearIndex] == 0u) { continue; }
      let nearHeight = elevationValues[elevationValuesOffset + nearIndex];
      if (nearHeight > ownHeight || (nearHeight == ownHeight && nearIndex < index)) { return false; }
    }
  }
  let ownRowHalfWidth = getDiscRowHalfWidth(0, cell, radiusSquared);
  let columnHalfHeight = getDiscColumnHalfHeight(cell, radiusSquared);
  // The widest row is dy = 0 and the tallest column is dx = 0, so these decide grid containment.
  if (REJECT_INCOMPLETE &&
      (column - ownRowHalfWidth < 0 || column + ownRowHalfWidth >= i32(WIDTH) ||
       row - columnHalfHeight < 0 || row + columnHalfHeight >= i32(HEIGHT))) {
    return false;
  }
  // Heights of FLT_MAX collide with the empty/invalid sentinels of block maxima.
  let exact = abs(ownHeight) >= FLOAT_MAXIMUM;
  // Own row: every |dx| <= ownRowHalfWidth is in the disc.
  for (var offsetX = -ownRowHalfWidth; offsetX <= ownRowHalfWidth; offsetX++) {
    let pixelColumn = column + offsetX;
    if (offsetX == 0 || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
    let neighborIndex = index + u32(offsetX) ;
    if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) {
      if (REJECT_INCOMPLETE) { return false; }
      continue;
    }
    if (beatsCentre(elevationValues[elevationValuesOffset + neighborIndex], ownHeight, offsetX < 0)) {
      return false;
    }
  }
  let lastRow = i32(HEIGHT) - 1;
  if (!isHalfDiscDominated(column, row, max(row - columnHalfHeight, 0), row - 1, ownHeight, true, exact, cell, radiusSquared) ||
      !isHalfDiscDominated(column, row, row + 1, min(row + columnHalfHeight, lastRow), ownHeight, false, exact, cell, radiusSquared)) {
    return false;
  }
  var ringHeight = 0.0;
  var hasRing = false;
  var previousHalfWidth = -1;
  var currentHalfWidth = getDiscRowHalfWidth(-columnHalfHeight, cell, radiusSquared);
  for (var offsetY = -columnHalfHeight; offsetY <= columnHalfHeight; offsetY++) {
    var nextHalfWidth = -1;
    if (offsetY < columnHalfHeight) { nextHalfWidth = getDiscRowHalfWidth(offsetY + 1, cell, radiusSquared); }
    let pixelRow = row + offsetY;
    if (pixelRow >= 0 && pixelRow <= lastRow) {
      let interiorHalfWidth = min(previousHalfWidth, min(currentHalfWidth, nextHalfWidth));
      // Interior pixels (all eight neighbors in the disc) have |dx| <= interiorHalfWidth - 1.
      let hasInterior = interiorHalfWidth > 0;
      for (var run = 0; run < 2; run++) {
        if (run == 1 && !hasInterior) { break; }
        var runStart = -currentHalfWidth;
        var runEnd = currentHalfWidth;
        if (hasInterior) {
          if (run == 0) { runEnd = -interiorHalfWidth; } else { runStart = interiorHalfWidth; }
        }
        for (var offsetX = runStart; offsetX <= runEnd; offsetX++) {
          let pixelColumn = column + offsetX;
          if ((offsetX == 0 && offsetY == 0) || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
          let neighborIndex = u32(pixelRow) * WIDTH + u32(pixelColumn);
          if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { continue; }
          let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
          if (!hasRing || neighborHeight > ringHeight) {
            ringHeight = neighborHeight;
            hasRing = true;
          }
        }
      }
    }
    previousHalfWidth = currentHalfWidth;
    currentHalfWidth = nextHalfWidth;
  }
  if (hasRing) {
    let dropValue = ownHeight - ringHeight;
    if (dropValue < minimumDrop) { return false; }
    summitDrop = dropValue;
  } else {
    summitDrop = getInfinity(index);
  }
  return true;
}`}var $=N({parameters:W,rebuildOn:[`incompleteNeighborhood`],caption:`Elevation in; the summit mask and each summit's drop out.`,createInputs:({random:e,world:t})=>{let n=z(t,30);return{...n,noise:Float32Array.from(n.elevation,()=>e())}},build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=B(e,t),s=e.allocate(`parameters`,`float32`,8),c=e.allocate(`summit-mask`,`uint32`,o),l=e.allocate(`drop`,`float32`,o);return e.graph.add(new q({grid:r,elevation:i,maximumRadiusPixels:16,incompleteNeighborhood:n.incompleteNeighborhood===1?`ignore`:`reject`,parameters:s.view,output:{summitMask:c.view,drop:l.view}})),{outputs:{elevation:a,drop:l,summitMask:c},update:e=>{a.write(V(t,t.noise,e.noise)),s.write(K({radius:e.radius,minimumDrop:e.minimumDrop},r))}}},interaction:{kind:`reach`,hint:`size the radius`,rest:[.5,1-P(F(W,`radius`),350)],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=I(F(W,`radius`),1-e.unitY),o=a*R(t,r),s=i(n.summitMask),c=[{shape:`ring`,at:[e.x,e.y],radius:o},{shape:`dot`,at:[e.x,e.y]}],l=0;if(s)for(let e=0;e<s.length;e++)s[e]===1&&(l++,l<=24&&c.push({shape:`ring`,at:U(t,r,e%t.columns,Math.floor(e/t.columns)),radius:o,dashed:!0}));return{values:{radius:a},glyphs:c,readout:`radius ${a} m${s?` · ${l} summits`:``}`}}},draw:({inputs:e,outputs:t})=>[L(e,t.elevation),H(e,t.summitMask,{ramp:`categorical`,categories:[[0,0,0,0],[.95,.45,.1,1]]})]});export{$ as default};