import{i as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{u as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,f as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{i as s,n as c}from"./pair-histogram-COfc4qPq.js";import{c as l,i as u,l as d,t as f,u as p}from"./pair-statistics-grid-L8bKgCyF.js";import{t as m}from"./pair-statistics-parameters-BSUD2FEq.js";import{v as h}from"./spatial-context-DwZEd4EZ.js";import{t as g}from"./study-window-BK-B07m5.js";import{t as _}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as v}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as y,n as b,o as x}from"./study-window-wgsl-B3mOUanG.js";import{t as S}from"./pair-statistics-preflight-CaTMfN03.js";var C={none:0,border:1,isotropic:2};function w(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPURipley target must hold 8 elements`);m(`GPURipley`,e.bounds,e.maximumDistance);let n=e.edgeCorrection??`isotropic`;if(!Object.prototype.hasOwnProperty.call(C,n))throw Error(`GPURipley edgeCorrection must be 'none', 'border' or 'isotropic'`);let r=e.seed??0;if(!Number.isInteger(r)||r<0||r>4294967295)throw Error(`GPURipley seed must be a uint32`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,C[n],0,0]),new Uint32Array(t.buffer,t.byteOffset,8)[6]=r,t}var T=`GPURipley`,E=256,D=5,O=131072,k=class{id;props;spatialContext;studyWindow;constructor(n){this.id=n.id??`ripley`,this.props=n;let a=this.id;this.spatialContext=h(a,n.spatialContext),this.studyWindow=g(a,n.studyWindow,this.spatialContext),d(a,{...n,parameterLength:8});let{radiusCount:c,output:l}=n;i(a,`radiusCount`,c,1,E),s(a,c,D);for(let[e,t]of[[`output.k`,l.k],[`output.l`,l.l],[`output.lMinusR`,l.lMinusR],[`output.pairCorrelation`,l.pairCorrelation],[`output.radii`,l.radii]])if(t&&(r(t,[`float32`],`${a} ${e}`),t.length<c))throw Error(`${a} ${e} must hold radiusCount rows`);if(l.pairCounts&&(t(l.pairCounts,`${a} output.pairCounts`),l.pairCounts.length<c))throw Error(`${a} output.pairCounts must hold radiusCount rows`);e(a,l),o(a,[l.k,l.l,l.lMinusR,l.pairCorrelation,l.pairCounts,l.radii],[n.positions,n.parameters,n.mask])}getCommandNodes(e){let{id:t,props:r}=this,{positions:i,parameters:o,gridSize:s,radiusCount:d,output:m}=r;p(t,e.device,i.length),a(t,e,[i,o,r.mask,m.k,m.l,m.lMinusR,m.pairCorrelation,m.pairCounts,m.radii]);let h=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);h&&a(t,e,[h.edges,h.summary,h.slabs.params,h.slabs.slabOffsets,h.slabs.slabEntries,h.sampleCells,h.sampleCellCount,h.packed]),S(e,t,{rows:i.length,gridSize:s,slots:d,channels:D,views:{positions:i,mask:r.mask,parameters:o,window:h?.packed,k:m.k,l:m.l,lMinusR:m.lMinusR,pairCorrelation:m.pairCorrelation,pairCounts:m.pairCounts,radii:m.radii}});let g=h?l(e,{id:t,operation:T,parameters:o,parameterLength:o.length,window:h}):void 0,C=g?.parameters??o,w=u(e,{id:t,operation:T,positions:i,parameters:C,gridSize:s,mask:r.mask,studyWindow:h}),E=[...g?.nodes??[],...w.nodes],k=`
const RADIUS_COUNT: u32 = ${d}u;
const ISOTROPIC_SCALE: f32 = ${O}.0;
const WEIGHT_CAP: f32 = 100.0;
// The raw per-annulus pair counts feed K only for the uncorrected estimator, so the other modes
// tally them only when the caller asked for pairCounts.
const RAW_PAIRS: bool = ${!!m.pairCounts};
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
`,A=h?`${_}
${v}
${y(x(h))}`:``,j=c(e,{id:`${t}-histogram`,operation:T,positions:i,parameters:C,gridSize:s,sortedRows:w.sortedRows,cellOffsets:w.cellOffsets,slotCount:d,channelCount:D,pairOrder:`ordered`,extraBindings:h?[b(h)]:void 0,declarations:`${k}${A}
// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    // A point on the side line: a quarter turn. h can be -0.0 (a coordinate of -0.0 against a
    // bound of 0), and atan2(y, -0.0) is -pi / 2 on some adapters.
    if (h == 0.0) {
      return HALF_PI;
    }
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

// Ripley (1977) weight 1 / (fraction of the circle inside the rectangle), capped.
fn getIsotropicWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}`,focusPrologue:`let mode = u32(readParameter(5u) + 0.5);
    ${h?`var boundaryDistance = lattice.maximumDistance;
    if (mode == 1u || mode == 2u) {
      boundaryDistance = getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance);
    }`:`let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;`}
    var borderCount = 0u;
    // Border estimator: pairs of this focus below its border count, tallied in a register and
    // added once per focus instead of once per pair (integer sums, so the result is identical).
    var borderPairs = 0u;
    if (mode == 1u) {
      let borderDistance = ${h?`min(boundaryDistance, lattice.maximumDistance)`:`min(min(left, right), min(bottom, top))`};
      borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    }`,pairAction:`let scaled = pairDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
            let annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
            if (RAW_PAIRS || mode == 0u) {
              accumulate(annulus, 3u, 1u);
            }
            if (mode == 1u) {
              if (annulus < borderCount) {
                accumulate(annulus, 0u, 1u);
                borderPairs++;
              }
            } else if (mode == 2u) {
              ${h?`var weight = 1.0;
              if (boundaryDistance < pairDistance) {
                weight = 1.0 / max(getStudyWindowCircleFraction(vec2f(x, y), pairDistance), 1.0 / WEIGHT_CAP);
              }
              accumulate(annulus, 4u, quantizePairAmount(weight * ISOTROPIC_SCALE));`:`accumulate(annulus, 4u, quantizePairAmount(getIsotropicWeight(left, bottom, right, top, pairDistance) * ISOTROPIC_SCALE));`}
            }`,focusEpilogue:`if (mode == 1u) {
      accumulate(borderCount, 1u, borderPairs);
      accumulate(borderCount, 2u, 1u);
    }`});E.push(...j.nodes);let M=[...h?[b(h)]:[],{name:`accumulators`,view:j.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:w.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:C,type:`f32`,access:`read`},{name:`k`,view:m.k,type:`f32`,access:`read_write`}];m.pairCounts&&M.push({name:`pairCounts`,view:m.pairCounts,type:`u32`,access:`read_write`}),E.push(n(e,{id:`${t}-finish`,operation:T,variant:`finish`,bindings:M,invocationCount:d,declarations:`${f}${A}
${k}
const CELL_COUNT: u32 = ${s[0]*s[1]}u;

fn readAccumulator(slot: u32, channel: u32) -> vec2<u32> {
  let word = (slot * ${D}u + channel) * 2u;
  return vec2<u32>(accumulators[accumulatorsOffset + word], accumulators[accumulatorsOffset + word + 1u]);
}

fn add64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let low = first.x + second.x;
  return vec2<u32>(low, first.y + second.y + select(0u, 1u, low < first.x));
}

fn subtract64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(first.x - second.x, first.y - second.y - select(0u, 1u, first.x < second.x));
}

fn toFloat64(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// Exact 64-bit prefix sum of one channel over slots [0, last].
fn getPrefix(channel: u32, last: u32) -> vec2<u32> {
  var sum = vec2<u32>(0u, 0u);
  for (var slot = 0u; slot <= last; slot++) {
    sum = add64(sum, readAccumulator(slot, channel));
  }
  return sum;
}`,body:`let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = ${h?`studyWindowSummaryValue(4u)`:`(parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u])`};
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  var kValue = getQuietNaN(index);
  if (n >= 2.0 && isFiniteValue(area) && area > 0.0) {
    if (mode == 1u) {
      let numerator = subtract64(getPrefix(0u, index), getPrefix(1u, index));
      let borderCount = n - toFloat64(getPrefix(2u, index));
      if (borderCount >= 1.0) {
        // spatstat Kest border: lambda = n / A.
        kValue = area * toFloat64(numerator) / (n * borderCount);
      }
    } else if (mode == 2u) {
      kValue = area / (n * (n - 1.0)) * (toFloat64(getPrefix(4u, index)) / ISOTROPIC_SCALE);
    } else {
      kValue = area / (n * (n - 1.0)) * toFloat64(getPrefix(3u, index));
    }
  }
  k[kOffset + index] = kValue;
  ${m.pairCounts?`let annulusCount = readAccumulator(index, 3u);
  pairCounts[pairCountsOffset + index] = select(annulusCount.x, 0xffffffffu, annulusCount.y != 0u);`:``}`}));let N=[{name:`parameters`,view:C,type:`f32`,access:`read`},{name:`k`,view:m.k,type:`f32`,access:`read`}];for(let[e,t]of[[`l`,m.l],[`lMinusR`,m.lMinusR],[`pairCorrelation`,m.pairCorrelation],[`radii`,m.radii]])t&&N.push({name:e,view:t,type:`f32`,access:`read_write`});return N.length>2&&E.push(n(e,{id:`${t}-derive`,operation:T,variant:`derive`,bindings:N,invocationCount:d,declarations:`${f}
${k}`,body:`let nan = getQuietNaN(index);
  let maximumDistance = parameters[parametersOffset + 4u];
  let radius = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);
  let kValue = k[kOffset + index];
  let finite = isFiniteValue(kValue);
  let lValue = select(nan, sqrt(max(kValue, 0.0) / PI), finite);
  ${m.l?`l[lOffset + index] = lValue;`:``}
  ${m.lMinusR?`lMinusR[lMinusROffset + index] = select(nan, lValue - radius, finite);`:``}
  ${m.radii?`radii[radiiOffset + index] = radius;`:``}
  ${m.pairCorrelation?`var previousK = 0.0;
  var previousFinite = true;
  if (index > 0u) {
    previousK = k[kOffset + index - 1u];
    previousFinite = isFiniteValue(previousK);
  }
  let previousRadius = maximumDistance * f32(index) / f32(RADIUS_COUNT);
  let annulusArea = PI * (radius - previousRadius) * (radius + previousRadius);
  pairCorrelation[pairCorrelationOffset + index] =
    select(nan, (kValue - previousK) / annulusArea, finite && previousFinite);`:``}`})),E}};export{w as n,k as t};