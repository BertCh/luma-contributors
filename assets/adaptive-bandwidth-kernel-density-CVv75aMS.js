import{n as e}from"./gpu-raster-grid-C_izEAl0.js";import{r as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{E as r,S as i,T as a,f as o,g as s,o as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{c as d,o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as p,v as m}from"./spatial-context-DwZEd4EZ.js";import{n as h,r as g}from"./gpu-inverse-distance-weighting-KIb4SzJG.js";import{t as _}from"./grid-cell-row-order-DUF05y4e.js";import{n as v}from"./point-neighbor-grid-index-D0wS1y1I.js";import{n as y,r as b}from"./gpu-kernel-density-ZcuIBWnT.js";import{_ as x,b as S,h as C,m as w,v as T,x as E}from"./density-scratch-D0AtmvQz.js";import{n as D}from"./point-observations-CeqP-hIo.js";import{t as O}from"./define-run-B-brkCe4.js";import{n as k,t as A}from"./pick-Bo3t3hjv.js";import{o as j}from"./events-CU2v4wkX.js";import{pn as M}from"./entry-preview-D70TdSbK.js";var N=`GPUAdaptiveBandwidthKernelDensity`,P=10;function F(e,t,n=new Float32Array(P)){if(n.length<P)throw Error(`Adaptive kernel density parameter target must hold ${P} elements`);let r=e.sensitivity??.5;if(!(r>=0&&r<=1))throw Error(`Adaptive kernel density sensitivity must be in [0, 1]`);return y({bandwidth:e.bandwidth},t,n),n[9]=r,n}function I(e){return`const DOMAIN_MIN: vec2f = vec2f(${d(e[0])}, ${d(e[1])});
const DOMAIN_MAX: vec2f = vec2f(${d(e[2])}, ${d(e[3])});`}var L=class{id;props;spatialContext;kernel;normalization;bandwidthFactorLimit;cellCount;constructor(e){let t=e.id??`adaptive-kernel-density`;this.id=t,this.props=e,this.spatialContext=m(t,e.spatialContext,{...p,measure:`area`}),this.kernel=e.kernel===`bisquare`?`quartic`:e.kernel??`gaussian`,this.normalization=e.normalization??`intensity`,this.bandwidthFactorLimit=e.bandwidthFactorLimit??4,this.cellCount=e.grid.width*e.grid.height,a(t,[`gaussian`,`quartic`,`epanechnikov`].includes(this.kernel),`kernel must be gaussian, quartic (alias bisquare) or epanechnikov`),a(t,[`intensity`,`probability`].includes(this.normalization),`normalization must be intensity or probability`),s(t,`bandwidthFactorLimit`,this.bandwidthFactorLimit),a(t,this.bandwidthFactorLimit>=1&&this.bandwidthFactorLimit<=16,`bandwidthFactorLimit must be in [1, 16]`);for(let[n,r]of[[`indexGridSize[0]`,e.indexGridSize[0]],[`indexGridSize[1]`,e.indexGridSize[1]]])o(t,n,r,1,w);for(let[n,r]of e.indexBounds.entries())s(t,`indexBounds[${n}]`,r);let[n,i,c,l]=e.indexBounds;a(t,e.indexBounds.length===4&&e.indexBounds.every(Number.isFinite)&&n<=c&&i<=l,`indexBounds must be finite and ordered`),r(e.positions,[`float32x2`],`${t} positions`),o(t,`positions length`,e.positions.length,1,C);let d=e.positions.length;e.weights&&(r(e.weights,[`float32`],`${t} weights`),a(t,e.weights.length===d,`weights length must equal positions length`)),r(e.parameters,[`float32`],`${t} parameters`),a(t,e.parameters.length>=P,`parameters must hold ${P} float32 values`),r(e.output.values,[`float32`],`${t} output.values`),a(t,e.output.values.length>=this.cellCount,`output.values must hold grid.width * grid.height rows`),e.output.weightSum&&(r(e.output.weightSum,[`float32`],`${t} output.weightSum`),a(t,e.output.weightSum.length===1,`output.weightSum must contain exactly one float32 row`));for(let n of[`pointBandwidths`,`pilotDensities`]){let i=e.output[n];i&&(r(i,[`float32`],`${t} output.${n}`),a(t,i.length>=d,`output.${n} must hold one row per point`))}E(t,{values:e.output.values,weightSum:e.output.weightSum,pointBandwidths:e.output.pointBandwidths,pilotDensities:e.output.pilotDensities}),u(t,[e.output.values,e.output.weightSum,e.output.pointBandwidths,e.output.pilotDensities],[e.positions,e.weights,e.parameters])}getCommandNodes(e){let{id:r,props:a,kernel:o,normalization:s,bandwidthFactorLimit:u,cellCount:p}=this,{output:m,grid:y,indexGridSize:C,indexBounds:w}=a,E=m;l(r,e,[a.positions,a.weights,a.parameters,E.values,E.weightSum,E.pointBandwidths,E.pilotDensities]),i(r,`indexGridSize cells`,C[0]*C[1],c(e.device)),S(e,r,[[`positions`,a.positions],[`weights`,a.weights],[`parameters`,a.parameters],[`output.values`,E.values],[`output.weightSum`,E.weightSum],[`output.pointBandwidths`,E.pointBandwidths],[`output.pilotDensities`,E.pilotDensities]],{dispatchMetadata:!1}),T(e,r,a.positions.length,C[0]*C[1]);let D=x(e,r,N,m,{values:p,pointBandwidths:a.positions.length,pilotDensities:a.positions.length});E=D.views;let{width:O,height:k}=y,A=a.positions.length,j=v(e,{id:r,indexId:`${r}-index`,positions:a.positions,gridSize:C,bounds:w,rowCount:A}),{cellOffsets:M,sortedRows:P,gridCount:F}=j,L=[...j.nodes],R=_(e,{id:`${r}-cell-row-order`,operation:N,cellOffsets:M,objectIds:P,indexCount:F,rowCount:A});L.push(...R.nodes);let z=R.sortedRows,B=t(e,`${r}-point-weights`,`float32`,A),V=E.pilotDensities??t(e,`${r}-pilot`,`float32`,A),H=t(e,`${r}-log-terms`,`float32`,A),U=E.pointBandwidths??t(e,`${r}-bandwidths`,`float32`,A),W=E.weightSum??t(e,`${r}-weight-sum`,`float32`,1),G=t(e,`${r}-log-sum`,`float32`,1),K=I(w),q=d(o===`gaussian`?5:1),J=[{name:`cellOffsets`,view:M,type:`u32`,access:`read`},{name:`sortedIds`,view:z,type:`u32`,access:`read`}],Y=`const INDEX_WIDTH: u32 = ${C[0]}u;
const INDEX_HEIGHT: u32 = ${C[1]}u;
const SUPPORT_FACTOR: f32 = ${q};
${K}
${h}
${b(o)}`,X=e=>`let queryMin = center - vec2f(radius);
    let queryMax = center + vec2f(radius);
    if (all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
      let clampedMin = max(queryMin, DOMAIN_MIN);
      let clampedMax = min(queryMax, DOMAIN_MAX);
      // Widened by one index cell per side to absorb f32 rounding of the cell mapping.
      var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
      var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
      columnLow = columnLow - min(columnLow, 1u);
      rowLow = rowLow - min(rowLow, 1u);
      let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
      let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
      ${g(e)}
    }`;return L.push(f(e,{id:`${r}-point-weights`,operation:N,variant:`point-weights`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},...a.weights?[{name:`inputWeights`,view:a.weights,type:`f32`,access:`read`}]:[],{name:`pointWeights`,view:B,type:`f32`,access:`read_write`}],invocationCount:A,declarations:`${K}
${h}`,body:`let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${a.weights?`inputWeights[inputWeightsOffset + index]`:`1.0`};
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  pointWeights[pointWeightsOffset + index] = select(0.0, weight, included);`}),...new n({id:`${r}-weight-sum`,input:B,output:W,operation:`sum`}).getCommandNodes(e),f(e,{id:`${r}-pilot`,operation:N,variant:`pilot-${o}`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`pointWeights`,view:B,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},...J,{name:`pilot`,view:V,type:`f32`,access:`read_write`},{name:`logTerms`,view:H,type:`f32`,access:`read_write`}],invocationCount:A,workgroupSize:64,declarations:Y,body:`let own = pointWeights[pointWeightsOffset + index];
  pilot[pilotOffset + index] = 0.0;
  logTerms[logTermsOffset + index] = 0.0;
  let bandwidth = params[paramsOffset + 4u];
  if (!(own > 0.0) || !(bandwidth > 0.0) || !isFiniteValue(bandwidth)) {
    return;
  }
  let center = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  var total = 0.0;
  ${X(`let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let delta = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]) - center;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`)}
  pilot[pilotOffset + index] = total;
  logTerms[logTermsOffset + index] = select(0.0, own * log(total), total > 0.0);`}),...new n({id:`${r}-log-sum`,input:H,output:G,operation:`sum`}).getCommandNodes(e),f(e,{id:`${r}-point-bandwidths`,operation:N,variant:`point-bandwidths`,bindings:[{name:`pointWeights`,view:B,type:`f32`,access:`read`},{name:`pilot`,view:V,type:`f32`,access:`read`},{name:`logSum`,view:G,type:`f32`,access:`read`},{name:`weightSum`,view:W,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`bandwidths`,view:U,type:`f32`,access:`read_write`}],invocationCount:A,workgroupSize:64,declarations:`const FACTOR_LIMIT: f32 = ${d(u)};`,body:`let own = pointWeights[pointWeightsOffset + index];
  let pilotValue = pilot[pilotOffset + index];
  let bandwidth = params[paramsOffset + 4u];
  let sensitivity = params[paramsOffset + 9u];
  if (!(own > 0.0) || !(pilotValue > 0.0)) {
    bandwidths[bandwidthsOffset + index] = 0.0;
    return;
  }
  let logMean = logSum[logSumOffset] / weightSum[weightSumOffset];
  // (pilot / g)^(-alpha) = exp(-alpha (ln pilot - ln g)) keeps the ratio out of f32 overflow.
  let factor = clamp(exp(-sensitivity * (log(pilotValue) - logMean)), 1.0 / FACTOR_LIMIT, FACTOR_LIMIT);
  bandwidths[bandwidthsOffset + index] = bandwidth * factor;`}),f(e,{id:`${r}-gather`,operation:N,variant:`gather-${o}-${s}`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`pointWeights`,view:B,type:`f32`,access:`read`},{name:`bandwidths`,view:U,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},...J,...s===`probability`?[{name:`weightSum`,view:W,type:`f32`,access:`read`}]:[],{name:`valuesOut`,view:E.values,type:`f32`,access:`read_write`}],invocationCount:p,workgroupSize:64,declarations:`const WIDTH: u32 = ${O}u;
const HEIGHT: u32 = ${k}u;
const FACTOR_LIMIT: f32 = ${d(u)};
${Y}`,body:`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let globalBandwidth = params[paramsOffset + 4u];
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SUPPORT_FACTOR * globalBandwidth * FACTOR_LIMIT;
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(globalBandwidth) && globalBandwidth > 0.0 && all(gridEnd != gridStart);
  var total = 0.0;
  if (valid) {
    ${X(`let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let pointBandwidth = bandwidths[bandwidthsOffset + sample];
        let pointRadius = SUPPORT_FACTOR * pointBandwidth;
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]) - gridStart) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < pointRadius * pointRadius)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, pointBandwidth);`)}
  }
  var result = select(getNaN(), total, valid);
  ${s===`probability`?`if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    result = select(getNaN(), result / totalWeight, totalWeight > 0.0);
  }`:``}
  valuesOut[valuesOutOffset + index] = result;`})),[...L,...D.publishNodes]}},R=1400,z=60;function B({random:t,world:n}){let r=D(t,n,R,.35),i=Math.round(z*n.width);return{...r,width:n.width,columns:i,index:k(r.positions),focus:{row:-1},grid:e({width:i,height:z,cellSize:[n.width/i,1/z],origin:[0,1]})}}var V=O({parameters:M,caption:`USGS ComCat M4+ earthquakes around Japan, 2020–2024: adaptive kernel density; each event’s bandwidth narrows where events are dense. 35% of the sample is drawn from sparse cells, which adds to the contrast.`,createInputs:B,build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),i=e.allocate(`parameters`,`float32`,P),a=e.allocate(`density`,`float32`,t.columns*z),o=e.allocate(`point-bandwidths`,`float32`,t.count);return e.graph.add(new L({positions:r.view,parameters:i.view,grid:t.grid,indexGridSize:[40,24],indexBounds:[0,0,t.width,1],bandwidthFactorLimit:n.bandwidthFactorLimit,output:{values:a.view,pointBandwidths:o.view}})),{outputs:{positions:r,density:a,pointBandwidths:o},update:e=>i.write(F({bandwidth:e.bandwidth,sensitivity:e.sensitivity},t.grid))}},rebuildOn:[`bandwidthFactorLimit`],buildCache:5,interaction:{kind:`probe`,hint:`hover an event`,rest:[.3,.4],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=A(t.index,e.x,e.y,t.focus.row);if(t.focus.row=a,a<0)return{};let o=[t.positions[a*2],t.positions[a*2+1]],s=i(n.pointBandwidths)?.[a];return{focus:a,glyphs:[{shape:`crosshair`,at:o,size:10},{shape:`ring`,at:o,radius:r.bandwidth,dashed:!0},...s&&Number.isFinite(s)?[{shape:`ring`,at:o,radius:s}]:[]],readout:`bandwidth ${j(s,3)} = h × ${j(s===void 0?void 0:s/r.bandwidth)} · M${t.magnitude[a].toFixed(1)}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.density,columns:e.columns,rows:z,range:[6,15],logarithmic:!0,smoothing:!0},{layer:`dots`,context:!0,positions:t.positions,radius:1.6,alpha:.6}]});export{V as default};