import{n as e,t}from"./interpolation-view-validation-pYo9tqh3.js";import{r as n}from"./kernel-C3tfQOn4.js";import{s as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{E as a,S as o,T as s,f as c,p as l,u,v as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{u as p}from"./wgsl-common-DfyXiYB0.js";import{a as m,o as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as g}from"./raster-band-utils-BcTxc8_Y.js";import{i as _,n as v,r as y,t as b}from"./regression-row-tiles-7Xsw3Utq.js";import{n as x}from"./gpu-raster-grid-C_izEAl0.js";import{t as S}from"./define-run-B-brkCe4.js";import{i as C,t as w}from"./point-observations-CeqP-hIo.js";import{a as T,r as E}from"./grid-panel-CtEMLsqP.js";import{t as D}from"./readback-cfnHpL6G.js";import{Dl as O}from"./entry-preview-D70TdSbK.js";function k(e,t=new Float32Array(6)){let n=`getGPUSinkhornDivergenceParameterValues`;if(t.length<6)throw Error(`${n} target must hold 6 elements`);let r=[[`cellWidth`,e.cellWidth],[`cellHeight`,e.cellHeight],[`epsilon`,e.epsilon],[`tolerance`,e.tolerance??1e-4],[`epsilonStart`,e.epsilonStart??e.epsilon],[`scalingFactor`,e.scalingFactor??1]];for(let[e,t]of r)if(!(Math.fround(t)>0)||!Number.isFinite(Math.fround(t)))throw Error(`${n} ${e} must be positive and finite`);if(!(Math.fround(r[4][1])>=Math.fround(e.epsilon)))throw Error(`${n} epsilonStart must be at least epsilon`);if(!(Math.fround(r[5][1])<=1))throw Error(`${n} scalingFactor must be in (0, 1]`);return t.set(r.map(([,e])=>e)),t}var A={epsilon:0,done:1,iterationCount:2,converged:3,iterationLimitReached:4,marginalError:5,sourceSelfError:6,targetSelfError:7,valid:8},j=-1e30,M=`fn scaleSinkhornMass(mass: f32, maximum: f32) -> f32 {
  let maximumExponent = (bitcast<u32>(maximum) >> 23u) & 255u;
  let inverseExponent = max(254u - maximumExponent, 1u);
  return mass * bitcast<f32>(inverseExponent << 23u);
}`,N=`const MASKED: f32 = ${j};
const MASKED_LIMIT: f32 = ${j/2};
fn accumulateLogSum(value: f32, maximum: ptr<function, f32>, sum: ptr<function, f32>) {
  if (value <= MASKED_LIMIT) {
    return;
  }
  if (value > *maximum) {
    *sum = *sum * exp(*maximum - value) + 1.0;
    *maximum = value;
  } else {
    *sum += exp(value - *maximum);
  }
}`,P=`GPUSinkhornDivergence`,F=1e3,I=2048,L=10,R=7,z=class{id;props;spatialContext;maximumIterations;blurRadius;isExactBlur;estimatedCommandNodeCount;constructor(n){this.id=n.id??`sinkhorn-divergence`,this.props=n;let{id:u}=this;this.spatialContext=i(u,n.spatialContext,{...r,measure:`area`});let d=g(u,n.grid),{width:f,height:p}=n.grid;o(u,`grid.width and grid.height`,Math.max(f,p),I),o(u,`cells`,d,b),this.maximumIterations=n.maximumIterations??100,c(u,`maximumIterations`,this.maximumIterations,1,F);let m=Math.max(f,p)-1;this.blurRadius=n.blurRadius??Math.max(m,1),c(u,`blurRadius`,this.blurRadius,1,I),this.isExactBlur=this.blurRadius>=m,a(n.source,[`float32`],`${u} source`),a(n.target,[`float32`],`${u} target`),a(n.parameters,[`float32`],`${u} parameters`),s(u,n.source.length===d&&n.target.length===d,`source and target must hold grid.width * grid.height rows`),s(u,n.parameters.length>=6,`parameters must hold 6 float32 values`);let{output:h}=n;a(h.divergence,[`float32`],`${u} output.divergence`),s(u,h.divergence.length===1,`output.divergence must contain exactly one row`),h.components&&(a(h.components,[`float32`],`${u} output.components`),s(u,h.components.length===3,`output.components must contain exactly three rows`)),h.marginalError&&(a(h.marginalError,[`float32`],`${u} output.marginalError`),s(u,h.marginalError.length===1,`output.marginalError must contain exactly one row`));for(let[e,t]of Object.entries(h.status??{}))t&&(a(t,[`uint32`],`${u} output.status.${e}`),s(u,t.length===1,`output.status.${e} must contain exactly one row`));let _=this.getOutputViews();t(u,this.getOutputBindings(),[{write:{divergence:h.divergence,components:h.components,marginalError:h.marginalError}},{write:h.status??{}}]),e(u,_,[n.source,n.target,n.parameters]),this.estimatedCommandNodeCount=L+R*this.maximumIterations,l(u,this.estimatedCommandNodeCount)}getOutputViews(){return Object.values(this.getOutputBindings()).filter(e=>e!==void 0)}getOutputBindings(){let{output:e}=this.props;return{"output.divergence":e.divergence,"output.components":e.components,"output.marginalError":e.marginalError,...Object.fromEntries(Object.entries(e.status??{}).map(([e,t])=>[`output.status.${e}`,t]))}}getCommandNodes(e){let{id:t,props:r,maximumIterations:i,blurRadius:a}=this,{output:s}=r,{width:c,height:l}=r.grid,g=c*l;f(e,t,{source:r.source,target:r.target,parameters:r.parameters,...this.getOutputBindings()}),o(t,`cells`,g,u(e.device,16));let b=v(g),[x,S]=y(b,3),C=[[3*g],[3*g],x,S,[g],[g]];for(let[n,r]of Object.entries({state:9,massMaxima:2,masses:4,logSource:g,logTarget:g,potentials:4*g,blurred:3*g,errors:3*g,finalSums:5,massPartials:b.tileCount*4,maximumPartials:b.tileCount*2,errorPartials:b.tileCount*3,finalPartials:b.tileCount*5,iterationDispatch:C.length*3}))d(e.device,t,n,r*Float32Array.BYTES_PER_ELEMENT,`reduce the grid size`);let w=A,T=(r,i)=>n(e,`${t}-${r}`,`float32`,Math.max(i,1)),E=T(`state`,9),D=T(`mass-maxima`,2),O=T(`masses`,4),k=T(`log-source`,g),j=T(`log-target`,g),F=T(`potentials`,4*g),I=T(`blurred`,3*g),L=T(`errors`,3*g),R=T(`final-sums`,5),z=m(e,{id:`${t}-iteration-dispatch`,operation:P,shapes:C}),B=Object.entries(w).map(([e,t])=>`const S_${e.replace(/[A-Z]/g,e=>`_${e}`).toUpperCase()}: u32 = ${t}u;`).join(`
`),V=`const WIDTH: u32 = ${c}u;
const HEIGHT: u32 = ${l}u;
const N: u32 = ${g}u;
const RADIUS: u32 = ${a}u;`,H=`fn getNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,U=(e,t)=>({name:e,view:t,type:`f32`,access:`read`}),W=(e,t)=>({name:e,view:t,type:`f32`,access:`read_write`}),G=[];G.push(..._(e,{id:`${t}-mass-maxima`,operation:P,variant:`mass-maxima`,rowCount:g,layout:b,sumCount:0,maximumCount:2,bindings:[U(`sourceMasses`,r.source),U(`targetMasses`,r.target)],declarations:`${p}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let sourceMass = sourceMasses[sourceMassesOffset + row];
  if (isFiniteValue(sourceMass) && sourceMass > 0.0) { (*terms)[0] = max((*terms)[0], sourceMass); }
  let targetMass = targetMasses[targetMassesOffset + row];
  if (isFiniteValue(targetMass) && targetMass > 0.0) { (*terms)[1] = max((*terms)[1], targetMass); }
}`,partial:T(`maximum-partials`,b.tileCount*2),output:D})),G.push(..._(e,{id:`${t}-masses`,operation:P,variant:`masses`,rowCount:g,layout:b,sumCount:4,bindings:[U(`sourceMasses`,r.source),U(`targetMasses`,r.target),U(`massMaxima`,D)],declarations:`${p}
${M}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let sourceMass = sourceMasses[sourceMassesOffset + row];
  let sourceMaximum = massMaxima[massMaximaOffset];
  if (isFiniteValue(sourceMass) && sourceMass >= 0.0) {
    (*terms)[0] += scaleSinkhornMass(sourceMass, sourceMaximum);
  } else { (*terms)[2] += 1.0; }
  let targetMass = targetMasses[targetMassesOffset + row];
  let targetMaximum = massMaxima[massMaximaOffset + 1u];
  if (isFiniteValue(targetMass) && targetMass >= 0.0) {
    (*terms)[1] += scaleSinkhornMass(targetMass, targetMaximum);
  } else { (*terms)[3] += 1.0; }
}`,partial:T(`mass-partials`,b.tileCount*4),output:O})),G.push(h(e,{id:`${t}-log-masses`,operation:P,variant:`log-masses`,bindings:[U(`sourceMasses`,r.source),U(`targetMasses`,r.target),U(`masses`,O),U(`massMaxima`,D),W(`logSource`,k),W(`logTarget`,j),W(`potentials`,F)],invocationCount:g,declarations:`${p}
${N}
${M}
const N: u32 = ${g}u;`,body:`let sourceMass = sourceMasses[sourceMassesOffset + index];
  let targetMass = targetMasses[targetMassesOffset + index];
  let sourceTotal = masses[massesOffset];
  let targetTotal = masses[massesOffset + 1u];
  let sourceMaximum = massMaxima[massMaximaOffset];
  let targetMaximum = massMaxima[massMaximaOffset + 1u];
  let sourceValid = isFiniteValue(sourceMass) && sourceMass > 0.0 && sourceTotal > 0.0;
  let targetValid = isFiniteValue(targetMass) && targetMass > 0.0 && targetTotal > 0.0;
  logSource[logSourceOffset + index] = select(MASKED, log(scaleSinkhornMass(sourceMass, sourceMaximum) / sourceTotal), sourceValid);
  logTarget[logTargetOffset + index] = select(MASKED, log(scaleSinkhornMass(targetMass, targetMaximum) / targetTotal), targetValid);
  potentials[potentialsOffset + index] = 0.0;
  potentials[potentialsOffset + N + index] = 0.0;
  potentials[potentialsOffset + 2u * N + index] = 0.0;
  potentials[potentialsOffset + 3u * N + index] = 0.0;`})),G.push(h(e,{id:`${t}-initialize`,operation:P,variant:`initialize`,invocationCount:1,bindings:[U(`params`,r.parameters),U(`masses`,O),W(`state`,E),z.binding],declarations:`${B}\n${p}\n${H}`,body:`let cellWidth = params[paramsOffset];
  let cellHeight = params[paramsOffset + 1u];
  let epsilon = params[paramsOffset + 2u];
  let tolerance = params[paramsOffset + 3u];
  let epsilonStart = params[paramsOffset + 4u];
  let scalingFactor = params[paramsOffset + 5u];
  let valid = masses[massesOffset] > 0.0 && masses[massesOffset + 1u] > 0.0 &&
    cellWidth > 0.0 && cellHeight > 0.0 && epsilon > 0.0 && tolerance > 0.0 &&
    epsilonStart >= epsilon && scalingFactor > 0.0 && scalingFactor <= 1.0 &&
    isFiniteValue(cellWidth) && isFiniteValue(cellHeight) && isFiniteValue(epsilon) &&
    isFiniteValue(tolerance) && isFiniteValue(epsilonStart);
  state[stateOffset + S_EPSILON] = select(1.0, max(epsilon, epsilonStart), valid);
  state[stateOffset + S_DONE] = select(1.0, 0.0, valid);
  state[stateOffset + S_VALID] = select(0.0, 1.0, valid);
  state[stateOffset + S_ITERATION_COUNT] = 0.0;
  state[stateOffset + S_CONVERGED] = 0.0;
  state[stateOffset + S_ITERATION_LIMIT_REACHED] = 0.0;
  state[stateOffset + S_MARGINAL_ERROR] = select(getNaN(), 0.0, valid);
  state[stateOffset + S_SOURCE_SELF_ERROR] = select(getNaN(), 0.0, valid);
  state[stateOffset + S_TARGET_SELF_ERROR] = select(getNaN(), 0.0, valid);
  ${z.disarmWGSL}
  if (valid) {
    ${z.armWGSL}
  }`}));let K=`${V}\n${B}\n${N}`,q=e=>{let t=e===1?3:1;return{bindings:[U(`potentials`,F),U(`logSource`,k),U(`logTarget`,j),U(`state`,E),U(`params`,r.parameters),W(`blurred`,I)],invocationCount:t*g,body:`let job = index / N;
  let cell = index % N;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  let epsilon = state[stateOffset + S_EPSILON];
  let cellWidth = params[paramsOffset];
  ${e===1?`let inputBlock = select(select(0u, 2u, job == 1u), 3u, job == 2u);
  let useTarget = job == 2u;`:`let inputBlock = 1u;
  let useTarget = true;`}
  var maximum = MASKED;
  var sum = 0.0;
  let first = column - min(column, RADIUS);
  let last = min(column + RADIUS, WIDTH - 1u);
  for (var other = first; other <= last; other++) {
    let at = row * WIDTH + other;
    let weight = select(logSource[logSourceOffset + at], logTarget[logTargetOffset + at], useTarget);
    if (weight > MASKED_LIMIT) {
      let distance = (f32(other) - f32(column)) * cellWidth;
      accumulateLogSum(potentials[potentialsOffset + inputBlock * N + at] / epsilon + weight - distance * distance / epsilon, &maximum, &sum);
    }
  }
  blurred[blurredOffset + index] = select(MASKED, maximum + log(sum), sum > 0.0);`}},J=e=>{let t=e===1?3:1;return{bindings:[U(`blurred`,I),W(`potentials`,F),...e===1?[U(`logSource`,k),U(`logTarget`,j)]:[],U(`state`,E),U(`params`,r.parameters),...e===1?[W(`errors`,L)]:[]],invocationCount:t*g,body:`let job = index / N;
  let cell = index % N;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  let epsilon = state[stateOffset + S_EPSILON];
  let cellHeight = params[paramsOffset + 1u];
  var maximum = MASKED;
  var sum = 0.0;
  let first = row - min(row, RADIUS);
  let last = min(row + RADIUS, HEIGHT - 1u);
  for (var other = first; other <= last; other++) {
    let distance = (f32(other) - f32(row)) * cellHeight;
    accumulateLogSum(blurred[blurredOffset + job * N + other * WIDTH + column] - distance * distance / epsilon, &maximum, &sum);
  }
  let updated = select(UNREACHABLE, -epsilon * (maximum + log(sum)), sum > 0.0);
  ${e===1?`let outputBlock = select(select(1u, 2u, job == 1u), 3u, job == 2u);
  let slot = potentialsOffset + outputBlock * N + cell;
  let previous = potentials[slot];
  // Each residual is mass * |exp((old - T(old)) / epsilon) - 1|. The symmetric
  // self plans have equal row and column residuals, measured before damping.
  let massLog = select(logTarget[logTargetOffset + cell], logSource[logSourceOffset + cell], job == 1u);
  let mass = select(0.0, exp(massLog), massLog > MASKED_LIMIT);
  // An empty blur delivers no mass, even if both potentials carry UNREACHABLE.
  var residual = mass;
  if (sum > 0.0 && massLog > MASKED_LIMIT) {
    let delta = (previous - updated) / epsilon;
    if (delta > 1.0) {
      // Combine with the log mass before exponentiating: a tiny mass can offset a
      // large delta. Capping delta alone would undercount the marginal residual.
      residual = abs(exp(massLog + delta) - mass);
    } else {
      residual = mass * abs(exp(delta) - 1.0);
    }
  }
  errors[errorsOffset + job * N + cell] = residual;
  if (job == 0u) {
    potentials[slot] = updated;
  } else {
    potentials[slot] = 0.5 * (previous + updated);
  }`:`potentials[potentialsOffset + cell] = updated;`}`}},Y=(n,r,i,a)=>h(e,{id:`${t}-${n}`,operation:P,variant:r,bindings:i.bindings,invocationCount:i.invocationCount,declarations:`${K}\nconst UNREACHABLE: f32 = 1.0e30;`,body:i.body,...z.getGate(a)});for(let n=0;n<i;n++){let a=`iteration-${n}`;G.push(Y(`${a}-blur-columns-first`,`blur-x-batch`,q(1),0),Y(`${a}-blur-rows-first`,`blur-y-batch`,J(1),1),..._(e,{id:`${t}-${a}-marginal-error`,operation:P,variant:`marginal-error`,rowCount:g,layout:b,sumCount:3,bindings:[U(`errors`,L)],declarations:`const N: u32 = ${g}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  (*terms)[0] += errors[errorsOffset + row];
  (*terms)[1] += errors[errorsOffset + N + row];
  (*terms)[2] += errors[errorsOffset + 2u * N + row];
}`,partial:T(`${a}-error-partials`,b.tileCount*3),output:E,outputSlot:w.marginalError,tileGate:z.getGate(2),mergeGate:z.getGate(3)}),h(e,{id:`${t}-${a}-update`,operation:P,variant:`update`,invocationCount:1,bindings:[U(`params`,r.parameters),W(`state`,E),z.binding],declarations:`${B}
const MAXIMUM_ITERATIONS: f32 = ${i}.0;`,body:`if (state[stateOffset + S_DONE] != 0.0) {
    return;
  }
  let iterations = state[stateOffset + S_ITERATION_COUNT] + 1.0;
  state[stateOffset + S_ITERATION_COUNT] = iterations;
  let finalEpsilon = params[paramsOffset + 2u];
  let tolerance = params[paramsOffset + 3u];
  // The error only counts once the regularization has reached its final value.
  let atFinalEpsilon = state[stateOffset + S_EPSILON] == finalEpsilon;
  let converged = atFinalEpsilon && state[stateOffset + S_MARGINAL_ERROR] <= tolerance &&
    state[stateOffset + S_SOURCE_SELF_ERROR] <= tolerance && state[stateOffset + S_TARGET_SELF_ERROR] <= tolerance;
  state[stateOffset + S_EPSILON] = max(finalEpsilon, params[paramsOffset + 4u] * pow(params[paramsOffset + 5u], iterations));
  if (converged) {
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_CONVERGED] = 1.0;
    ${z.disarmWGSL}
  } else if (iterations >= MAXIMUM_ITERATIONS) {
    // The second half of this last iteration still runs: the gate stays armed.
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_ITERATION_LIMIT_REACHED] = 1.0;
  }`}),Y(`${a}-blur-columns-second`,`blur-x-single`,q(2),4),Y(`${a}-blur-rows-second`,`blur-y-single`,J(2),5))}G.push(..._(e,{id:`${t}-final-sums`,operation:P,variant:`final-sums`,rowCount:g,layout:b,sumCount:5,bindings:[U(`potentials`,F),U(`logSource`,k),U(`logTarget`,j)],declarations:`${N}
const N: u32 = ${g}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let sourceLog = logSource[logSourceOffset + row];
  if (sourceLog > MASKED_LIMIT) {
    let mass = exp(sourceLog);
    let cross = potentials[potentialsOffset + row];
    let own = potentials[potentialsOffset + 2u * N + row];
    (*terms)[0] += mass * cross;
    (*terms)[2] += mass * own;
    (*terms)[4] += mass * (cross - own);
  }
  let targetLog = logTarget[logTargetOffset + row];
  if (targetLog > MASKED_LIMIT) {
    let mass = exp(targetLog);
    let cross = potentials[potentialsOffset + N + row];
    let own = potentials[potentialsOffset + 3u * N + row];
    (*terms)[1] += mass * cross;
    (*terms)[3] += mass * own;
    (*terms)[4] += mass * (cross - own);
  }
}`,partial:T(`final-partials`,b.tileCount*5),output:R})),G.push(h(e,{id:`${t}-publish-values`,operation:P,variant:`publish-values`,invocationCount:1,bindings:[U(`state`,E),U(`finalSums`,R),W(`divergence`,s.divergence),...s.components?[W(`components`,s.components)]:[],...s.marginalError?[W(`marginalError`,s.marginalError)]:[]],declarations:`${B}\n${H}`,body:`let valid = state[stateOffset + S_VALID] != 0.0;
  let invalidValue = getNaN();
  divergence[divergenceOffset] = select(invalidValue, finalSums[finalSumsOffset + 4u], valid);
  ${s.components?`components[componentsOffset] = select(invalidValue, finalSums[finalSumsOffset] + finalSums[finalSumsOffset + 1u], valid);
  components[componentsOffset + 1u] = select(invalidValue, 2.0 * finalSums[finalSumsOffset + 2u], valid);
  components[componentsOffset + 2u] = select(invalidValue, 2.0 * finalSums[finalSumsOffset + 3u], valid);`:``}
  ${s.marginalError?`marginalError[marginalErrorOffset] = select(invalidValue, state[stateOffset + S_MARGINAL_ERROR], valid);`:``}`}));let X=s.status??{},Z=[[`converged`,X.converged],[`iterationLimitReached`,X.iterationLimitReached],[`iterationCount`,X.iterationCount],[`invalidCount`,X.invalidCount],[`approximation`,X.approximation]].filter(([,e])=>e!==void 0);return Z.length>0&&G.push(h(e,{id:`${t}-publish-status`,operation:P,variant:`publish-status`,invocationCount:1,bindings:[U(`state`,E),U(`masses`,O),...Z.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],declarations:B,body:`${X.converged?`converged[convergedOffset] = u32(state[stateOffset + S_CONVERGED]);`:``}
  ${X.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = u32(state[stateOffset + S_ITERATION_LIMIT_REACHED]);`:``}
  ${X.iterationCount?`iterationCount[iterationCountOffset] = u32(state[stateOffset + S_ITERATION_COUNT]);`:``}
  ${X.invalidCount?`invalidCount[invalidCountOffset] = u32(masses[massesOffset + 2u] + masses[massesOffset + 3u]);`:``}
  ${X.approximation?`approximation[approximationOffset] = ${this.isExactBlur?`0u`:`1u`};`:``}`})),G}},B=32,V=18,H=[12,7],U=[4,2],W=[.64,.08,.97,.38],G=[.64,.46,.97,.72],K=[.64,.08,.97,.72];function q(e){let[t,n,r,i]=w,a=new Float32Array(576);for(let o=0;o<e.length/2;o++){let s=Math.min(31,Math.floor((e[o*2]-t)/(r-t)*B)),c=Math.min(17,Math.floor((i-e[o*2+1])/(i-n)*V));a[c*B+s]++}return a}function J(e,t,n){let r=new Float32Array(e.length);for(let i=0;i<V;i++)for(let a=0;a<B;a++){let o=a+t,s=i+n;o<0||o>=B||s<0||s>=V||(r[s*B+o]=e[i*B+a])}return r}var Y=S({parameters:O,caption:`Honshu earthquake counts per cell and the same field moved by the pointer. Upper bars: plain difference (total variation) and sqrt(S) in cells (of 16); lower bars: OT(a,b), OT(a,a), OT(b,b), S. Mass moved off the grid is dropped and the target renormalized, and total mass is ignored.`,createInputs:({random:e,world:t})=>({world:t,source:q(C(e,4e3).positions),shift:[...U],distance:0}),build(e,t,n){let r=e.upload(`source`,`float32`,t.source),i=e.allocate(`target`,`float32`,576),a=e.allocate(`parameters`,`float32`,6),o=e.allocate(`divergence`,`float32`,1),s=e.allocate(`components`,`float32`,3),c=e.allocate(`iterationLimitReached`,`uint32`,1);return e.graph.add(new z({grid:x({width:B,height:V}),source:r.view,target:i.view,parameters:a.view,maximumIterations:n.maximumIterations,output:{divergence:o.view,components:s.view,status:{iterationLimitReached:c.view}}})),{outputs:{source:r,target:i,divergence:o,components:s,iterationLimitReached:c,panel:T(e,K,t.world,.015)},update:e=>{i.write(J(t.source,t.shift[0],t.shift[1])),a.write(k({cellWidth:1,cellHeight:1,epsilon:e.epsilon}))}}},rebuildOn:[`maximumIterations`],buildCache:3,async afterRun({inputs:e,outputs:t,upload:n}){let r=await D(t.components),i=await D(t.divergence),a=Float32Array.of(r[0],r[1],r[2],i[0]).map(e=>Number.isFinite(e)?Math.max(e,0):0),o=J(e.source,e.shift[0],e.shift[1]),s=e.source.reduce((e,t)=>e+t,0),c=o.reduce((e,t)=>e+t,0)||1,l=0;for(let t=0;t<o.length;t++)l+=Math.abs(e.source[t]/s-o[t]/c)/2;let u=Number.isFinite(i[0])?Math.sqrt(Math.max(i[0],0)):0;return e.distance=u,{bars:n(`bars`,`float32`,a),compare:n(`compare`,`float32`,Float32Array.of(l,u/16))}},interaction:{kind:`drag`,hint:`drag the target`,rest:[.5+U[0]/(2*H[0]),.5+U[1]/(2*H[1])],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=[Math.round((e.unitX-.5)*2*H[0]),Math.round((e.unitY-.5)*2*H[1])],s=o[0]!==t.shift[0]||o[1]!==t.shift[1];t.shift=o;let c=a(n.divergence)?.[0],l=a(n.iterationLimitReached)?.[0],u=i.width/B,d=i.height/V;return{rerun:s,glyphs:[{shape:`arrow`,from:[i.width/2,i.height/2],to:[i.width/2+o[0]*u,i.height/2-o[1]*d]}],readout:`shift ${o[0]},${o[1]} cells · epsilon ${r.epsilon} · typical distance ${t.distance.toFixed(1)} cells · S ${c===void 0?`…`:Number.isFinite(c)?c.toFixed(2):`–`}${l===void 0?` · status pending`:l?` · iteration limit reached (estimate)`:``}`}}},draw:({outputs:e,extra:t,theme:n})=>[{layer:`raster`,context:!0,values:e.source,columns:B,rows:V,ramp:`grey`,range:[0,10]},{layer:`raster`,values:e.target,columns:B,rows:V,ramp:`sequential`,range:[0,10],alpha:.6},E(e.panel,n.background),{layer:`chart`,kind:`bars`,values:t.bars,ramp:`sequential`,rectangle:W},{layer:`chart`,kind:`bars`,values:t.compare,ramp:`diverging`,range:[0,1],rectangle:G}]});export{Y as default};