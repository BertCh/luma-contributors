import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{D as n,E as r,T as i,f as a,p as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{v as u}from"./spatial-context-DwZEd4EZ.js";import{a as d,c as f,d as p,f as m,i as h,m as g,n as _,o as v,p as y,r as b,s as x,t as S,u as C}from"./gtwr-data-BPOwieYA.js";import{t as w}from"./regression-output-dispatches-KKVTWkHP.js";import{a as T}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as E,r as D}from"./wgsl-common-DfyXiYB0.js";import{S as O,a as k}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{n as A}from"./temporal-context-xddVhAJD.js";import{t as j}from"./define-run-B-brkCe4.js";import{a as M,c as N}from"./pointer-overlay-Qh8qflcA.js";import{t as P}from"./kernel-DfdJX5rr.js";import{a as F}from"./weights-BRH93xop.js";import{c as I,f as L}from"./charts-Br0w6lsw.js";import{Zt as R}from"./entry-preview-D70TdSbK.js";function z(e){let t=new Float32Array(2+2*e.components.length);return[e.initial,...e.components].forEach((e,n)=>{let r=Math.fround(e.bandwidth),i=Math.fround(e.timeScale);if(!(Number.isFinite(r)&&r>0))throw Error(`getGPUMGTWRParameterValues bandwidth must be a positive finite number`);if(!(Number.isFinite(i)&&i>=0))throw Error(`getGPUMGTWRParameterValues timeScale must be a non-negative finite number`);t[2*n]=r,t[2*n+1]=i}),t}var B=`GPUMultiscaleGeographicallyTemporallyWeightedRegression`,V=O,H=class{props;id;rows;estimatedCommandNodeCount;constructor(e){this.props=e,this.id=e.id??`multiscale-geographically-temporally-weighted-regression`;let{id:t}=this,s=e.predictorCount,c=e;i(t,c.gaussianCutoff===void 0,`gaussianCutoff is not supported by multiscale GTWR`),i(t,c.searchStructure===void 0,`searchStructure is not supported by multiscale GTWR`);let d=g(t,e.bandwidthSearch,e.output.status);if(this.estimatedCommandNodeCount=7+e.iterationCount*(s+2)+(d?y(d,2)+1+e.iterationCount*(s+1)*(1+y(d,2)):0)+Number(!!e.output.selectedParameters),o(t,this.estimatedCommandNodeCount),u(t,e.spatialContext,{coordinateSpaces:[`planar`],metrics:[`native`],measure:`distance`}),a(t,`predictorCount`,s,1,11),a(t,`iterationCount`,e.iterationCount,1,32),A(t,e.temporalContext,{encodings:[`float32-relative`]}),r(e.response,[`float32`],`${t} response`),this.rows=e.response.length,a(t,`rows`,this.rows,2,x),r(e.positions,[`float32x2`],`${t} positions`),n(t,`positions`,e.positions,this.rows),r(e.times,[`float32`],`${t} times`),n(t,`times`,e.times,this.rows),r(e.predictors,[`float32`],`${t} predictors`),n(t,`predictors`,e.predictors,this.rows*s),e.mask&&(r(e.mask,[`uint32`],`${t} mask`),n(t,`mask`,e.mask,this.rows)),r(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<2+2*(s+1))throw Error(`${t} parameters must have ${2+2*(s+1)} values`);let{output:f}=e;for(let[i,a,o,c]of[[`coefficients`,f.coefficients,`float32`,this.rows*(s+1)],[`fitted`,f.fitted,`float32`,this.rows],[`residuals`,f.residuals,`float32`,this.rows],[`localStatus`,f.localStatus,`uint32`,this.rows],[`scores`,f.scores,`float32`,e.iterationCount]])r(a,[o],`${t} output.${i}`),n(t,`output.${i}`,a,c);f.selectedParameters&&(r(f.selectedParameters,[`float32`],`${t} output.selectedParameters`),n(t,`output.selectedParameters`,f.selectedParameters,2+2*(s+1))),l(t,this.getOutputViews(),this.getInputViews()),w(t,{...f,status:void 0,...Object.fromEntries(Object.entries(f.status??{}).map(([e,t])=>[`status.${e}`,t]))},[[`fitted`,`residuals`,`localStatus`],{read:[`coefficients`],write:[`fitted`,`residuals`,`localStatus`]},Object.keys(f.status??{}).map(e=>`status.${e}`)])}getInputViews(){let{props:e}=this;return[e.positions,e.times,e.predictors,e.response,e.mask,e.bandwidthSearch?void 0:e.parameters,e.bandwidthSearch?.parameters].filter(e=>e!==void 0)}getOutputViews(){let{status:e,...t}=this.props.output;return[...Object.values(t),...Object.values(e??{})].filter(e=>!!e)}getCommandNodes(n){let{id:r,props:i,rows:a}=this,{output:o}=i,l=i.predictorCount,u=l+1;c(r,n,[...this.getInputViews(),...this.getOutputViews()]),T(n,r,Object.fromEntries([...this.getInputViews(),...this.getOutputViews()].map((e,t)=>[`input/output[${t}]`,e]))),s(n.device,r,`initial summary scratch`,48,`use a device with a larger storage binding limit`),s(n.device,r,`component scratch`,a*Math.max(3,u)*4,`reduce rows or predictorCount`);let d=(t,i,a)=>e(n,`${r}-${t}`,i,a),h={coefficients:d(`initial-coefficients`,`float32`,a*u),localStatus:d(`initial-status`,`uint32`,a),fitted:d(`initial-fitted`,`float32`,a),hatDiagonal:d(`initial-hat`,`float32`,a),residuals:d(`initial-residuals`,`float32`,a),summary:d(`initial-summary`,`float32`,12)},g=i.bandwidthSearch?d(`selected-parameters`,`float32`,2+2*u):i.parameters,_=f(n,r,i,a),y=[d(`components-a`,`float32`,a*u),d(`components-b`,`float32`,a*u)],b=[d(`errors-a`,`float32`,a),d(`errors-b`,`float32`,a)],x=d(`iteration-state`,`uint32`,1),S=`fn getDesign(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return predictors[predictorsOffset + row * ${l}u + column - 1u];
}`,w=k(),O=[...new v({id:`${r}-initial`,positions:i.positions,times:i.times,temporalContext:i.temporalContext,spatialContext:i.spatialContext,adaptive:i.adaptive,bandwidthSearch:i.bandwidthSearch,predictors:i.predictors,predictorCount:l,response:i.response,mask:i.mask,kernel:i.kernel,searchStructure:`brute-force`,parameters:i.parameters,output:{...h,selectedParameters:i.bandwidthSearch?g:void 0,status:o.status}}).getCommandNodes(n),..._.nodes,t(n,{id:`${r}-start`,operation:B,variant:`initial-components`,invocationCount:a,bindings:[{name:`points`,view:_.points,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},{name:`response`,view:i.response,type:`f32`,access:`read`},{name:`initialCoefficients`,view:h.coefficients,type:`f32`,access:`read`},{name:`components`,view:y[0],type:`f32`,access:`read_write`},{name:`errors`,view:b[0],type:`f32`,access:`read_write`},{name:`iterationState`,view:x,type:`u32`,access:`read_write`}],declarations:`${E}
${S}`,body:`if (index == 0u) { iterationState[iterationStateOffset] = 0u; }
  let included = isFiniteBits(points[pointsOffset + 3u * index]);
  var total = 0.0;
  var finite = included;
  for (var column = 0u; column < ${u}u; column++) {
    let design = getDesign(index, column);
    let value = initialCoefficients[initialCoefficientsOffset + index * ${u}u + column] * design;
    finite = finite && isFiniteBits(value);
  }
  for (var column = 0u; column < ${u}u; column++) {
    let design = getDesign(index, column);
    let value = initialCoefficients[initialCoefficientsOffset + index * ${u}u + column] * design;
    let component = select(0.0, value, finite);
    components[componentsOffset + index * ${u}u + column] = component;
    total += component;
  }
  errors[errorsOffset + index] = select(0.0, response[responseOffset + index] - total, included);`})];for(let e=0;e<i.iterationCount;e++){let s=y[e%2],c=y[(e+1)%2];for(let f=0;f<u;f++){let h=e*u+f;if(i.bandwidthSearch){let c=d(`partial-${e}-${f}`,`float32`,a);O.push(t(n,{id:`${r}-partial-${e}-${f}`,operation:B,invocationCount:a,bindings:[{name:`previous`,view:s,type:`f32`,access:`read`},{name:`errors`,view:b[h%2],type:`f32`,access:`read`},{name:`partial`,view:c,type:`f32`,access:`read_write`}],body:`partial[partialOffset + index] = previous[previousOffset + index * ${u}u + ${f}u] + errors[errorsOffset + index];`})),O.push(...C(n,{id:`${r}-search-${e}-${f}`,rows:a,adaptive:i.adaptive??!1,search:i.bandwidthSearch,parameters:g,parameterOffset:2+2*f,status:o.status,accumulateStatus:!0,evaluate:(e,t,r)=>p(n,{id:r,rows:a,predictorCount:l,component:f,points:_.points,predictors:i.predictors,partial:c,parameters:e,gate:t,adaptive:i.adaptive??!1,bisquare:i.kernel===`bisquare`})}))}O.push(t(n,{id:`${r}-iteration-${e}-component-${f}`,operation:B,variant:`backfit-component`,invocationCount:a,bindings:[{name:`parameters`,view:g,type:`f32`,access:`read`},{name:`points`,view:_.points,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},{name:`previous`,view:s,type:`f32`,access:`read`},{name:`next`,view:c,type:`f32`,access:`read_write`},{name:`errorsIn`,view:b[h%2],type:`f32`,access:`read`},{name:`errorsOut`,view:b[(h+1)%2],type:`f32`,access:`read_write`},{name:`coefficients`,view:o.coefficients,type:`f32`,access:`read_write`}],declarations:`${E}\n${D}\n${w}\n${i.adaptive?m(a):``}`,body:`let timeScale = parameters[parametersOffset + ${3+2*f}u];
  let bandwidth = ${i.adaptive?`getAdaptiveBandwidth(index, parameters[parametersOffset + ${2+2*f}u], timeScale)`:`parameters[parametersOffset + ${2+2*f}u]`};
  let originX = points[pointsOffset + 3u * index];
  if (!isFiniteBits(originX)) {
    next[nextOffset + index * ${u}u + ${f}u] = 0.0;
    errorsOut[errorsOutOffset + index] = 0.0;
    coefficients[coefficientsOffset + index * ${u}u + ${f}u] = getNaN();
  } else {
    let originY = points[pointsOffset + 3u * index + 1u];
    let originTime = points[pointsOffset + 3u * index + 2u];
    var numerator = 0.0;
    var denominator = 0.0;
    for (var row = 0u; row < ${a}u; row++) {
      let rowX = points[pointsOffset + 3u * row];
      if (!isFiniteBits(rowX)) { continue; }
      let deltaX = rowX - originX;
      let deltaY = points[pointsOffset + 3u * row + 1u] - originY;
      let deltaTime = points[pointsOffset + 3u * row + 2u] - originTime;
      let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
      let weight = getLocalKernelWeight(distance, bandwidth, ${i.kernel===`bisquare`?`true`:`false`});
      if (!(weight > 0.0)) { continue; }
      ${f===0?`let design = 1.0;`:`let design = predictors[predictorsOffset + row * ${l}u + ${f-1}u];`}
      let partial = previous[previousOffset + row * ${u}u + ${f}u] + errorsIn[errorsInOffset + row];
      numerator += weight * design * partial;
      denominator += weight * design * design;
    }
    ${f===0?`let ownDesign = 1.0;`:`let ownDesign = predictors[predictorsOffset + index * ${l}u + ${f-1}u];`}
    let ownPartial = previous[previousOffset + index * ${u}u + ${f}u] + errorsIn[errorsInOffset + index];
    let ok = bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale) && denominator > 0.0 && isFiniteBits(numerator) && isFiniteBits(denominator);
    let beta = select(getNaN(), numerator / denominator, ok);
    let smoothed = select(0.0, ownDesign * beta, ok);
    next[nextOffset + index * ${u}u + ${f}u] = smoothed;
    errorsOut[errorsOutOffset + index] = ownPartial - smoothed;
    coefficients[coefficientsOffset + index * ${u}u + ${f}u] = beta;
  }`}))}O.push(t(n,{id:`${r}-iteration-${e}-score`,operation:B,variant:`convergence-score`,invocationCount:1,bindings:[{name:`points`,view:_.points,type:`f32`,access:`read`},{name:`previous`,view:s,type:`f32`,access:`read`},{name:`next`,view:c,type:`f32`,access:`read`},{name:`scores`,view:o.scores,type:`f32`,access:`read_write`},{name:`iterationState`,view:x,type:`u32`,access:`read_write`}],declarations:`${E}`,body:`var count = 0.0;
  var change = 0.0;
  var total = 0.0;
  for (var row = 0u; row < ${a}u; row++) {
    if (!isFiniteBits(points[pointsOffset + 3u * row])) { continue; }
    count += 1.0;
    var sum = 0.0;
    for (var column = 0u; column < ${u}u; column++) {
      let value = next[nextOffset + row * ${u}u + column];
      let difference = value - previous[previousOffset + row * ${u}u + column];
      change += difference * difference;
      sum += value;
    }
    total += sum * sum;
  }
  let scoreIndex = iterationState[iterationStateOffset];
  scores[scoresOffset + scoreIndex] = sqrt(change / count / total);
  iterationState[iterationStateOffset] = scoreIndex + 1u;`}))}let A=y[i.iterationCount%2];return O.push(t(n,{id:`${r}-results`,operation:B,variant:`results`,invocationCount:a,bindings:[{name:`points`,view:_.points,type:`f32`,access:`read`},{name:`response`,view:i.response,type:`f32`,access:`read`},{name:`finalComponents`,view:A,type:`f32`,access:`read`},{name:`coefficients`,view:o.coefficients,type:`f32`,access:`read`},{name:`fitted`,view:o.fitted,type:`f32`,access:`read_write`},{name:`residuals`,view:o.residuals,type:`f32`,access:`read_write`},{name:`localStatus`,view:o.localStatus,type:`u32`,access:`read_write`}],declarations:`${E}\n${D}`,body:`let included = isFiniteBits(points[pointsOffset + 3u * index]);
  var total = 0.0;
  var finite = true;
  for (var column = 0u; column < ${u}u; column++) {
    total += finalComponents[finalComponentsOffset + index * ${u}u + column];
    finite = finite && isFiniteBits(coefficients[coefficientsOffset + index * ${u}u + column]);
  }
  fitted[fittedOffset + index] = select(getNaN(), total, included);
  residuals[residualsOffset + index] = select(getNaN(), response[responseOffset + index] - total, included);
  localStatus[localStatusOffset + index] = select(${V.EXCLUDED}u, select(${V.SINGULAR}u, ${V.OK}u, finite), included);`})),o.selectedParameters&&O.push(t(n,{id:`${r}-publish-selected`,operation:B,invocationCount:2+2*u,bindings:[{name:`selected`,view:g,type:`f32`,access:`read`},{name:`output`,view:o.selectedParameters,type:`f32`,access:`read_write`}],body:`output[outputOffset + index] = selected[selectedOffset + index];`})),O}},U=5,W=.3,G=[.5,2.5],K=.24,q=[.06,.04,.4,.2],J=[-7,0],Y=[0,3.4],X=j({parameters:R,caption:`Intercept (top) and slope (bottom) in four periods, each with its own bandwidth and its own color scale; bottom left: log score of each backfitting iteration; the response is simulated on real tract poverty shares with a slowly varying intercept, a sweeping slope and noise.`,createInputs:({random:e,world:t})=>_(e,t,{tileColumns:4,tileRows:2,periodsFillGrid:!1,bottomMargin:K}),build(e,t){let n=d(e,t),r=e.allocate(`parameters`,`float32`,6),i=e.allocate(`coefficients`,`float32`,t.rowCount*2),a=e.allocate(`scores`,`float32`,U);e.graph.add(new H({positions:n.positions.view,times:n.times.view,temporalContext:S,predictors:n.predictors.view,predictorCount:1,response:n.response.view,kernel:`gaussian`,iterationCount:U,parameters:r.view,output:{coefficients:i.view,fitted:e.allocate(`fitted`,`float32`,t.rowCount).view,residuals:e.allocate(`residuals`,`float32`,t.rowCount).view,localStatus:e.allocate(`local-status`,`uint32`,t.rowCount).view,scores:a.view}}));let o=e.allocate(`log-scores`,`float32`,U);return P(e,{id:`log-scores`,invocationCount:U,bindings:[{name:`scores`,buffer:a,access:`read`},{name:`values`,buffer:o,access:`read_write`}],body:`values[valuesOffset + index] = log(max(scores[scoresOffset + index], 1e-9)) / 2.302585;`}),{outputs:{logScores:o,backing:e.upload(`backing`,`float32`,new Float32Array(U).fill(1)),meshes:[h(e,t,0),h(e,t,1)],intercept:F(e,`intercept`,i,2,0,t.rowCount),slope:F(e,`slope`,i,2,1,t.rowCount),scores:a},update:e=>r.write(z({initial:{bandwidth:W,timeScale:e.timeScale},components:[{bandwidth:e.interceptBandwidth,timeScale:e.timeScale},{bandwidth:e.slopeBandwidth,timeScale:e.timeScale}]}))}},interaction:M(R,{kind:`pad`,hint:`slope and intercept bandwidth`,x:{parameter:`slopeBandwidth`,scale:`logarithmic`},y:{parameter:`interceptBandwidth`,scale:`logarithmic`},decorate:(e,t,{outputs:n,peek:r})=>{let i=r(n.scores),a=i?` · score ${i[4].toExponential(0)} after ${U}`:``;return{glyphs:[{shape:`crosshair`,at:[t.x,t.y]}],readout:`slope h ${N(e.slopeBandwidth)} · intercept h ${N(e.interceptBandwidth)}${a}`}}}),draw:({inputs:e,outputs:t,theme:n})=>[...b(e,0,t.meshes[0],t.intercept,G),...b(e,1,t.meshes[1],t.slope,Y),{layer:`chart`,kind:`bars`,values:t.backing,range:[0,1],color:L(n,.12),rectangle:q},{layer:`chart`,kind:`line`,values:t.logScores,range:J,color:I(n,.95),radius:1.6,rectangle:q}]});export{X as default};