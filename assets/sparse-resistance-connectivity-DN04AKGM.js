import{d as e,o as t,u as n}from"./recipe-utils-CICB7qiS.js";import{r,u as i}from"./wgsl-common-DfyXiYB0.js";import{D as a,T as o,f as s,p as c}from"./gpu-contributor-utils-BIwLQON5.js";import{a as l,i as u,t as d}from"./sorted-segment-sums-Dvmm6OdR.js";import{n as f}from"./gpu-command-graph-BdzdmV6u.js";import{c as p,d as m,r as h,u as g}from"./kernel-C3tfQOn4.js";import{r as _}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{r as v,t as y}from"./gpu-chunk-utils-DMYEopiS.js";import{n as b,t as x}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as S}from"./gpu-elementwise-CcQBKyqy.js";import{t as C}from"./gpu-sort-xRm8l5yn.js";import{t as w}from"./masked-weak-components-j3xMrDf7.js";import{a as T}from"./wgsl-kernel-nodes-DagJklH2.js";var E=class{id;props;constructor(e){this.id=e.id??`gpu-coo-to-csr`,this.props=e;for(let t of _(e.rowIndices))g(t,`${this.id} rowIndices`);for(let t of _(e.columnIndices))g(t,`${this.id} columnIndices`);for(let t of _(e.values))m(t,[`float32`],`${this.id} values`);for(let t of _(e.rowOffsets))g(t,`${this.id} rowOffsets`);for(let t of _(e.outputColumnIndices))g(t,`${this.id} outputColumnIndices`);for(let t of _(e.outputValues))m(t,[`float32`],`${this.id} outputValues`);let t=e.rowIndices.length;if(e.columnIndices.length!==t||e.values.length!==t)throw Error(`${this.id} COO arrays must have equal length`);if(!Number.isSafeInteger(e.rows)||e.rows<0||e.rows>=4294967295||t>4294967295)throw Error(`${this.id} rows must be a non-negative integer`);if(e.rowOffsets.length!==e.rows+1)throw Error(`${this.id} rowOffsets length must equal rows + 1`);if(e.outputColumnIndices.length!==t||e.outputValues.length!==t)throw Error(`${this.id} CSR entry outputs must have length nnz`)}getCommandNodes(e){let{rowIndices:t,columnIndices:n,values:r,rowOffsets:i,outputColumnIndices:a,outputValues:o}=this.props;v(e,[t,n,r],[i,a,o]);let s=[...new S({id:`${this.id}-copy-columns`,input:n,output:a,operation:`copy`}).getCommandNodes(e),...new S({id:`${this.id}-copy-values`,input:r,output:o,operation:`copy`}).getCommandNodes(e)],c=_(t).filter(e=>e.length),l=0;for(let[t,n]of _(i).entries()){if(!n.length)continue;let r=x(this.id,n.length,256,e.device.limits.maxComputeWorkgroupsPerDimension);for(let i=0;i<Math.max(1,c.length);i++){let a=c[i];s.push(y(e,{id:`${this.id}-offsets-${t}-${i}`,inputs:a?{rows:a}:{},outputs:{offsets:n},dispatch:r,source:`
${a?`@group(0) @binding(0) var<storage, read> rows: array<u32>;`:``}
@group(0) @binding(${+!!a}) var<storage, read_write> offsets: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${b(r,256)}
  if (index >= ${n.length}u) { return; }
  let targetRow = ${l}u + index;
  var low = 0u;
  ${a?`var high = ${a.length}u;
  loop {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (rows[${p(a)}u + middle] < targetRow) { low = middle + 1u; }
    else { high = middle; }
  }`:``}
  offsets[${p(n)}u + index] ${i?`+=`:`=`} low;
}`}))}l+=n.length}return s}},D=`
const STATE_ACTIVE: u32 = 0u;
const STATE_ITERATIONS: u32 = 1u;
const STATE_REACHABLE: u32 = 2u;
const STATE_INVALID: u32 = 3u;
const STATE_RESIDUAL_DOT_PRECONDITIONED: u32 = 4u;
const STATE_ALPHA: u32 = 5u;
const STATE_BETA: u32 = 6u;
const STATE_RESIDUAL_DOT_RESIDUAL: u32 = 7u;
const STATE_BREAKDOWN: u32 = 8u;
const STATE_COMPONENTS: u32 = 9u;
const STATE_CONVERGED: u32 = 10u;
`,O=`
fn readStateFloat(word: u32) -> f32 { return bitcast<f32>(state[stateOffset + word]); }
`,k=`
fn writeStateFloat(word: u32, value: f32) { state[stateOffset + word] = bitcast<u32>(value); }
`;function A(e,n){let{id:r,operation:a,edges:o,conductances:s,vertexCount:c,connectivityIterations:u}=n,d=o.length*2,p=h(e,`${r}-state`,`uint32`,16),m=h(e,`${r}-keys`,`uint32`,d),g=h(e,`${r}-half-edges`,`uint32`,d),_=h(e,`${r}-sorted-keys`,`uint32`,d),v=h(e,`${r}-sorted-half-edges`,`uint32`,d),y=h(e,`${r}-entry-columns`,`uint32`,d),b=h(e,`${r}-entry-conductances`,`float32`,d),x=h(e,`${r}-row-offsets`,`uint32`,c+1),S=h(e,`${r}-csr-columns`,`uint32`,d),T=h(e,`${r}-csr-conductances`,`float32`,d),O=h(e,`${r}-component-neighbors`,`uint32`,d),k=h(e,`${r}-labels`,`uint32`,c),A=h(e,`${r}-components-converged`,`uint32`,1),j=h(e,`${r}-inverse-diagonal`,`float32`,c);return e.add(t(e,{id:`${r}-reset`,operation:a,invocationCount:1,bindings:[{name:`state`,view:p,type:`u32`,access:`read_write`}],body:`for (var word = 0u; word < 16u; word++) { state[stateOffset + word] = 0u; }`})),e.add(t(e,{id:`${r}-expand`,operation:a,invocationCount:d,declarations:`override VERTEX_COUNT: u32;
${i}${D}`,constants:{VERTEX_COUNT:c},bindings:[{name:`edges`,view:o,type:`u32`,access:`read`},{name:`conductances`,view:s,type:`f32`,access:`read`},{name:`keys`,view:m,type:`u32`,access:`read_write`},{name:`halfEdges`,view:g,type:`u32`,access:`read_write`},{name:`state`,view:p,type:`atomic<u32>`,access:`read_write`}],body:`let edge = index >> 1u;
  let side = index & 1u;
  let first = edges[edgesOffset + edge * 2u];
  let second = edges[edgesOffset + edge * 2u + 1u];
  let conductance = conductances[conductancesOffset + edge];
  let valid = first < VERTEX_COUNT && second < VERTEX_COUNT && isFiniteValue(conductance) && conductance >= 0.0;
  if (!valid && side == 0u) { atomicAdd(&state[stateOffset + STATE_INVALID], 1u); }
  // Invalid half-edges sort after every vertex row, beyond rowOffsets[VERTEX_COUNT].
  keys[keysOffset + index] = select(VERTEX_COUNT, select(first, second, side == 1u), valid);
  halfEdges[halfEdgesOffset + index] = index;`})),f(e,new C({id:`${r}-sort`,keys:m,values:g,outputKeys:_,outputValues:v,keyBits:l(c)}).getCommandNodes(e)),e.add(t(e,{id:`${r}-gather`,operation:a,invocationCount:d,declarations:`override VERTEX_COUNT: u32;`,constants:{VERTEX_COUNT:c},bindings:[{name:`sortedKeys`,view:_,type:`u32`,access:`read`},{name:`sortedHalfEdges`,view:v,type:`u32`,access:`read`},{name:`edges`,view:o,type:`u32`,access:`read`},{name:`conductances`,view:s,type:`f32`,access:`read`},{name:`entryColumns`,view:y,type:`u32`,access:`read_write`},{name:`entryConductances`,view:b,type:`f32`,access:`read_write`}],body:`let halfEdge = sortedHalfEdges[sortedHalfEdgesOffset + index];
  let edge = halfEdge >> 1u;
  let side = halfEdge & 1u;
  let first = edges[edgesOffset + edge * 2u];
  let second = edges[edgesOffset + edge * 2u + 1u];
  let conductance = conductances[conductancesOffset + edge];
  // Self edges and barriers contribute nothing; column 0 keeps every read in range.
  let usable = sortedKeys[sortedKeysOffset + index] < VERTEX_COUNT && first != second && conductance > 0.0;
  entryColumns[entryColumnsOffset + index] = select(0u, select(second, first, side == 1u), usable);
  entryConductances[entryConductancesOffset + index] = select(0.0, conductance, usable);`})),f(e,new E({id:`${r}-csr`,rowIndices:_,columnIndices:y,values:b,rows:c,rowOffsets:x,outputColumnIndices:S,outputValues:T}).getCommandNodes(e)),e.add(t(e,{id:`${r}-diagonal`,operation:a,invocationCount:c,declarations:i,bindings:[{name:`rowOffsets`,view:x,type:`u32`,access:`read`},{name:`csrColumns`,view:S,type:`u32`,access:`read`},{name:`csrConductances`,view:T,type:`f32`,access:`read`},{name:`inverseDiagonal`,view:j,type:`f32`,access:`read_write`},{name:`componentNeighbors`,view:O,type:`u32`,access:`read_write`}],body:`let first = rowOffsets[rowOffsetsOffset + index];
  let last = rowOffsets[rowOffsetsOffset + index + 1u];
  var degree = 0.0;
  for (var slot = first; slot < last; slot++) {
    let conductance = csrConductances[csrConductancesOffset + slot];
    degree += conductance;
    componentNeighbors[componentNeighborsOffset + slot] = select(0xffffffffu, csrColumns[csrColumnsOffset + slot], conductance > 0.0);
  }
  let inverse = 1.0 / degree;
  inverseDiagonal[inverseDiagonalOffset + index] = select(0.0, inverse, degree > 0.0 && isFiniteValue(inverse));`})),f(e,w(e,{id:`${r}-components`,operation:a,nodeCount:c,offsets:x,neighbors:O,labels:k,converged:A,iterations:u})),{rowOffsets:x,csrColumns:S,csrConductances:T,inverseDiagonal:j,labels:k,componentsConverged:A,state:p}}function j(e,n){let{id:r,operation:a,system:o,vertexCount:s,source:c,sink:l,maximumIterations:f,tolerance:p}=n,{rowOffsets:m,csrColumns:g,csrConductances:_,inverseDiagonal:v,labels:y,componentsConverged:b}=o,{state:x}=o,S=Math.ceil(s/256),C=h(e,`${r}-solution`,`float32`,s),w=h(e,`${r}-residual`,`float32`,s),E=h(e,`${r}-search`,`float32`,s),A=h(e,`${r}-product`,`float32`,s),j=h(e,`${r}-partials`,`float32`,3*S),M=T(e,{id:`${r}-dispatch`,operation:a,shapes:[[s,256]]}),N=D,P=d,F=u(`sum`),I={PARTIAL_COUNT:S};e.add(t(e,{id:`${r}-initialize`,operation:a,invocationCount:s,declarations:`override SOURCE_VERTEX: u32;`,constants:{SOURCE_VERTEX:c},bindings:[{name:`inverseDiagonal`,view:v,type:`f32`,access:`read`},{name:`solution`,view:C,type:`f32`,access:`read_write`},{name:`residual`,view:w,type:`f32`,access:`read_write`},{name:`search`,view:E,type:`f32`,access:`read_write`}],body:`let injected = select(0.0, 1.0, index == SOURCE_VERTEX);
  solution[solutionOffset + index] = 0.0;
  residual[residualOffset + index] = injected;
  search[searchOffset + index] = injected * inverseDiagonal[inverseDiagonalOffset + index];`})),e.add(t(e,{id:`${r}-start`,operation:a,invocationCount:1,declarations:`override SOURCE_VERTEX: u32;
override SINK_VERTEX: u32;
${N}${k}`,constants:{SOURCE_VERTEX:c,SINK_VERTEX:l},bindings:[{name:`labels`,view:y,type:`u32`,access:`read`},{name:`componentsConverged`,view:b,type:`u32`,access:`read`},{name:`inverseDiagonal`,view:v,type:`f32`,access:`read`},{name:`state`,view:x,type:`u32`,access:`read_write`},M.binding],body:`let invalid = state[stateOffset + STATE_INVALID];
  let componentsKnown = componentsConverged[componentsConvergedOffset] != 0u;
  let sameComponent = labels[labelsOffset + SOURCE_VERTEX] == labels[labelsOffset + SINK_VERTEX];
  var reachable = select(select(2u, 0u, componentsKnown), 1u, sameComponent);
  if (invalid != 0u) { reachable = 0u; }
  // r0 . z0 for r0 = e_source.
  let preconditioned = inverseDiagonal[inverseDiagonalOffset + SOURCE_VERTEX];
  state[stateOffset + STATE_REACHABLE] = reachable;
  state[stateOffset + STATE_COMPONENTS] = u32(componentsKnown);
  state[stateOffset + STATE_ITERATIONS] = 0u;
  writeStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED, preconditioned);
  writeStateFloat(STATE_RESIDUAL_DOT_RESIDUAL, 1.0);
  if (reachable != 0u && preconditioned > 0.0) {
    state[stateOffset + STATE_ACTIVE] = 1u;
    ${M.armWGSL}
  } else {
    state[stateOffset + STATE_ACTIVE] = 0u;
    ${M.disarmWGSL}
  }`}));let L=(p/2)**2;for(let n=0;n<f;n++){let o=`${r}-curvature-${n}`,c=`${r}-update-${n}`,u=`${r}-search-${n}`,d=e=>{let{condition:t,extraResources:n}=M.getGate(0);return{condition:{...t,id:`${e}-gate`},extraResources:n}};e.add(t(e,{id:o,operation:a,invocationCount:s,workgroupSize:256,guardIndex:!1,...d(o),declarations:`override SINK_VERTEX: u32;
override PARTIAL_COUNT: u32;
${P}`,constants:{SINK_VERTEX:l,...I},bindings:[{name:`rowOffsets`,view:m,type:`u32`,access:`read`},{name:`csrColumns`,view:g,type:`u32`,access:`read`},{name:`csrConductances`,view:_,type:`f32`,access:`read`},{name:`search`,view:E,type:`f32`,access:`read`},{name:`product`,view:A,type:`f32`,access:`read_write`},{name:`partials`,view:j,type:`f32`,access:`read_write`}],body:`var partialSum = 0.0;
  if (index < INVOCATION_COUNT) {
    let first = rowOffsets[rowOffsetsOffset + index];
    let last = rowOffsets[rowOffsetsOffset + index + 1u];
    let center = search[searchOffset + index];
    var flow = 0.0;
    for (var slot = first; slot < last; slot++) {
      let column = csrColumns[csrColumnsOffset + slot];
      flow += csrConductances[csrConductancesOffset + slot] * (center - search[searchOffset + column]);
    }
    // Grounded sink: its row and column leave the system; search[sink] stays 0.
    flow = select(flow, 0.0, index == SINK_VERTEX);
    product[productOffset + index] = flow;
    partialSum = center * flow;
  }
  ${F}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = partialSums[0];
  }`})),e.add(t(e,{id:`${r}-alpha-${n}`,operation:a,invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`override PARTIAL_COUNT: u32;
${P}${i}${N}${O}${k}`,constants:I,bindings:[{name:`partials`,view:j,type:`f32`,access:`read`},{name:`state`,view:x,type:`u32`,access:`read_write`},M.binding],body:`var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum += partials[partialsOffset + partial];
  }
  ${F}
  if (localInvocationIndex == 0u && state[stateOffset + STATE_ACTIVE] != 0u) {
    let curvature = partialSums[0];
    if (!(curvature > 0.0) || !isFiniteValue(curvature)) {
      state[stateOffset + STATE_ACTIVE] = 0u;
      state[stateOffset + STATE_BREAKDOWN] = 1u;
      ${M.disarmWGSL}
    } else {
      writeStateFloat(STATE_ALPHA, readStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED) / curvature);
    }
  }`})),e.add(t(e,{id:c,operation:a,invocationCount:s,workgroupSize:256,guardIndex:!1,...d(c),declarations:`override PARTIAL_COUNT: u32;
${P}${N}`,constants:I,bindings:[{name:`search`,view:E,type:`f32`,access:`read`},{name:`product`,view:A,type:`f32`,access:`read`},{name:`inverseDiagonal`,view:v,type:`f32`,access:`read`},{name:`state`,view:x,type:`u32`,access:`read`},{name:`solution`,view:C,type:`f32`,access:`read_write`},{name:`residual`,view:w,type:`f32`,access:`read_write`},{name:`partials`,view:j,type:`f32`,access:`read_write`}],body:`let alpha = bitcast<f32>(state[stateOffset + STATE_ALPHA]);
  var preconditioned = 0.0;
  var squared = 0.0;
  var signedResidual = 0.0;
  if (index < INVOCATION_COUNT) {
    solution[solutionOffset + index] = solution[solutionOffset + index] + alpha * search[searchOffset + index];
    let nextResidual = residual[residualOffset + index] - alpha * product[productOffset + index];
    residual[residualOffset + index] = nextResidual;
    squared = nextResidual * nextResidual;
    signedResidual = nextResidual;
    preconditioned = squared * inverseDiagonal[inverseDiagonalOffset + index];
  }
  var partialSum = preconditioned;
  ${F}
  let preconditionedTotal = partialSums[0];
  // Every lane has read the first total before the second tree rewrites partialSums.
  workgroupBarrier();
  partialSum = squared;
  ${F}
  let squaredTotal = partialSums[0];
  workgroupBarrier();
  // The sink row is not iterated, so its true residual is minus the signed sum of the others.
  partialSum = signedResidual;
  ${F}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = preconditionedTotal;
    partials[partialsOffset + PARTIAL_COUNT + workgroupIndex] = squaredTotal;
    partials[partialsOffset + 2u * PARTIAL_COUNT + workgroupIndex] = partialSums[0];
  }`})),e.add(t(e,{id:`${r}-beta-${n}`,operation:a,invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`override PARTIAL_COUNT: u32;
override MAXIMUM_ITERATIONS: u32;
override STOP_RESIDUAL_SQUARED: f32;
override TOLERANCE: f32;
${P}${i}${N}${O}${k}`,constants:{...I,MAXIMUM_ITERATIONS:f,STOP_RESIDUAL_SQUARED:L,TOLERANCE:p},bindings:[{name:`partials`,view:j,type:`f32`,access:`read`},{name:`state`,view:x,type:`u32`,access:`read_write`},M.binding],body:`var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum += partials[partialsOffset + partial];
  }
  ${F}
  let preconditionedTotal = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum += partials[partialsOffset + PARTIAL_COUNT + partial];
  }
  ${F}
  let squaredTotal = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum += partials[partialsOffset + 2u * PARTIAL_COUNT + partial];
  }
  ${F}
  if (localInvocationIndex == 0u && state[stateOffset + STATE_ACTIVE] != 0u) {
    let residualSum = partialSums[0];
    let iterations = state[stateOffset + STATE_ITERATIONS] + 1u;
    let beta = preconditionedTotal / readStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED);
    state[stateOffset + STATE_ITERATIONS] = iterations;
    writeStateFloat(STATE_BETA, beta);
    writeStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED, preconditionedTotal);
    writeStateFloat(STATE_RESIDUAL_DOT_RESIDUAL, squaredTotal);
    let converged = squaredTotal <= STOP_RESIDUAL_SQUARED && abs(residualSum) <= TOLERANCE / 2.0;
    // An exactly zero residual divides 0 by 0 but has converged, so it is not a breakdown.
    let failed = !converged && !isFiniteValue(beta);
    if (converged || iterations >= MAXIMUM_ITERATIONS || failed) {
      state[stateOffset + STATE_ACTIVE] = 0u;
      state[stateOffset + STATE_BREAKDOWN] = u32(failed);
      ${M.disarmWGSL}
    }
  }`})),e.add(t(e,{id:u,operation:a,invocationCount:s,workgroupSize:256,...d(u),declarations:N,bindings:[{name:`residual`,view:w,type:`f32`,access:`read`},{name:`inverseDiagonal`,view:v,type:`f32`,access:`read`},{name:`state`,view:x,type:`u32`,access:`read`},{name:`search`,view:E,type:`f32`,access:`read_write`}],body:`let beta = bitcast<f32>(state[stateOffset + STATE_BETA]);
  search[searchOffset + index] = residual[residualOffset + index] * inverseDiagonal[inverseDiagonalOffset + index] + beta * search[searchOffset + index];`}))}return{solution:C,partials:j,partialCount:S}}var M=1048576,N=2097152,P=M,F=N,I=4096,L=1024,R=I,z=L,B=`addSparseResistanceConnectivityRecipe`,V=5,H=2;function U(e,t){return 64+V*e+H*t}function W(l,f){let p=f.id??`sparse-resistance-connectivity`,{edges:m,conductances:h,vertexCount:g,source:_,sink:v}=f,y=m.length,b=f.maximumIterations??1024,x=f.tolerance??1e-4,S=f.connectivityIterations??128;if(!Number.isSafeInteger(g)||g<2||g>P)throw Error(`${p} vertexCount must be an integer in [2, ${P}]`);if(y<1||y>F)throw Error(`${p} edges must have between 1 and ${F} rows`);s(p,`source`,_,0,g-1),s(p,`sink`,v,0,g-1),o(p,_!==v,`source and sink must be different vertices`),a(p,`conductances`,h,y),s(p,`maximumIterations`,b,1,R),s(p,`connectivityIterations`,S,1,z),o(p,Number.isFinite(x)&&x>0,`tolerance must be a positive finite number`);let C=U(b,S);c(p,C);let w=f.output,T=n(l,`${p}-potentials`,`float32`,g,w?.potentials),E=n(l,`${p}-currents`,`float32`,y,w?.currents),O=n(l,`${p}-vertex-currents`,`float32`,g,w?.vertexCurrents),k=n(l,`${p}-status`,`uint32x4`,1,w?.status),M=n(l,`${p}-diagnostics`,`float32x4`,1,w?.diagnostics);e(l,p,[m,h],[T,E,O,M,k]);let N=A(l,{id:p,operation:B,edges:m,conductances:h,vertexCount:g,connectivityIterations:S}),{solution:I,partials:L,partialCount:V}=j(l,{id:p,operation:B,system:N,vertexCount:g,source:_,sink:v,maximumIterations:b,tolerance:x}),{rowOffsets:H,csrColumns:W,csrConductances:G,labels:K,state:q}=N,J=D,Y=u(`sum`),X=u(`max`);return l.add(t(l,{id:`${p}-true-residual`,operation:B,invocationCount:g,workgroupSize:256,guardIndex:!1,declarations:`override SOURCE_VERTEX: u32;
override SINK_VERTEX: u32;
override PARTIAL_COUNT: u32;
${d}${i}`,constants:{SOURCE_VERTEX:_,SINK_VERTEX:v,PARTIAL_COUNT:V},bindings:[{name:`rowOffsets`,view:H,type:`u32`,access:`read`},{name:`csrColumns`,view:W,type:`u32`,access:`read`},{name:`csrConductances`,view:G,type:`f32`,access:`read`},{name:`solution`,view:I,type:`f32`,access:`read`},{name:`partials`,view:L,type:`f32`,access:`read_write`}],body:`var residualMaximum = 0.0;
  var energy = 0.0;
  if (index < INVOCATION_COUNT) {
    let first = rowOffsets[rowOffsetsOffset + index];
    let last = rowOffsets[rowOffsetsOffset + index + 1u];
    let center = solution[solutionOffset + index];
    var flow = 0.0;
    for (var slot = first; slot < last; slot++) {
      let difference = center - solution[solutionOffset + csrColumns[csrColumnsOffset + slot]];
      let current = csrConductances[csrConductancesOffset + slot] * difference;
      flow += current;
      // Every edge is visited from both ends.
      energy += 0.5 * current * difference;
    }
    let expected = select(0.0, 1.0, index == SOURCE_VERTEX) - select(0.0, 1.0, index == SINK_VERTEX);
    let balance = abs(flow - expected);
    // Every vertex counts, including floating ones: their exact residual is 0 unless they hold the source.
    residualMaximum = select(3.0e38, balance, isFiniteValue(balance));
  }
  var partialSum = residualMaximum;
  ${X}
  let maximumTotal = partialSums[0];
  workgroupBarrier();
  partialSum = energy;
  ${Y}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = maximumTotal;
    partials[partialsOffset + PARTIAL_COUNT + workgroupIndex] = partialSums[0];
  }`})),l.add(t(l,{id:`${p}-finish`,operation:B,invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`override PARTIAL_COUNT: u32;
override SOURCE_VERTEX: u32;
override TOLERANCE: f32;
override VERTEX_COUNT: u32;
${d}${i}${r}${J}`,constants:{PARTIAL_COUNT:V,SOURCE_VERTEX:_,TOLERANCE:x,VERTEX_COUNT:g},bindings:[{name:`partials`,view:L,type:`f32`,access:`read`},{name:`solution`,view:I,type:`f32`,access:`read`},{name:`state`,view:q,type:`u32`,access:`read_write`},{name:`status`,view:k,type:`u32`,access:`read_write`},{name:`diagnostics`,view:M,type:`f32`,access:`read_write`}],body:`var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum = max(partialSum, partials[partialsOffset + partial]);
  }
  ${X}
  let maximumResidual = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += 256u) {
    partialSum += partials[partialsOffset + PARTIAL_COUNT + partial];
  }
  ${Y}
  if (localInvocationIndex == 0u) {
    let energyTotal = partialSums[0];
    let reachable = state[stateOffset + STATE_REACHABLE];
    let invalid = state[stateOffset + STATE_INVALID];
    let solved = invalid == 0u && reachable != 0u;
    let resistance = solution[solutionOffset + SOURCE_VERTEX];
    // A maximum residual below 1 / n bounds the summed residual below 1, which proves the source
    // reaches the sink even when the component labels are undetermined (reachable == 2).
    let certified = reachable == 1u || maximumResidual * f32(VERTEX_COUNT) < 1.0;
    let converged = solved && certified && maximumResidual <= TOLERANCE && isFiniteValue(resistance) && isFiniteValue(energyTotal);
    state[stateOffset + STATE_CONVERGED] = u32(converged);
    status[statusOffset + 0u] = u32(converged);
    status[statusOffset + 1u] = state[stateOffset + STATE_ITERATIONS];
    status[statusOffset + 2u] = reachable;
    status[statusOffset + 3u] = invalid;
    let missing = getNaN();
    diagnostics[diagnosticsOffset + 0u] = select(missing, resistance, converged);
    diagnostics[diagnosticsOffset + 1u] = select(missing, maximumResidual, solved);
    diagnostics[diagnosticsOffset + 2u] = select(missing, energyTotal, solved);
    diagnostics[diagnosticsOffset + 3u] = sqrt(bitcast<f32>(state[stateOffset + STATE_RESIDUAL_DOT_RESIDUAL]));
  }`})),l.add(t(l,{id:`${p}-vertex-outputs`,operation:B,invocationCount:g,declarations:`override SINK_VERTEX: u32;
${r}${J}`,constants:{SINK_VERTEX:v},bindings:[{name:`rowOffsets`,view:H,type:`u32`,access:`read`},{name:`csrColumns`,view:W,type:`u32`,access:`read`},{name:`csrConductances`,view:G,type:`f32`,access:`read`},{name:`solution`,view:I,type:`f32`,access:`read`},{name:`labels`,view:K,type:`u32`,access:`read`},{name:`state`,view:q,type:`u32`,access:`read`},{name:`potentials`,view:T,type:`f32`,access:`read_write`},{name:`vertexCurrents`,view:O,type:`f32`,access:`read_write`}],body:`let first = rowOffsets[rowOffsetsOffset + index];
  let last = rowOffsets[rowOffsetsOffset + index + 1u];
  let center = solution[solutionOffset + index];
  var inflow = 0.0;
  var outflow = 0.0;
  for (var slot = first; slot < last; slot++) {
    let current = csrConductances[csrConductancesOffset + slot] * (center - solution[solutionOffset + csrColumns[csrColumnsOffset + slot]]);
    outflow += max(current, 0.0);
    inflow += max(-current, 0.0);
  }
  let componentsKnown = state[stateOffset + STATE_COMPONENTS] != 0u;
  let connected = !componentsKnown || labels[labelsOffset + index] == labels[labelsOffset + SINK_VERTEX];
  let solved = state[stateOffset + STATE_REACHABLE] != 0u && state[stateOffset + STATE_INVALID] == 0u;
  let converged = state[stateOffset + STATE_CONVERGED] != 0u;
  potentials[potentialsOffset + index] = select(getNaN(), center, solved && connected);
  vertexCurrents[vertexCurrentsOffset + index] = select(getNaN(), max(inflow, outflow), converged && connected);`})),l.add(t(l,{id:`${p}-edge-currents`,operation:B,invocationCount:y,declarations:`${r}
fn readEdges(row: u32) -> vec2<u32> {
  return vec2<u32>(edges[edgesOffset + row * 2u + 0u], edges[edgesOffset + row * 2u + 1u]);
}
fn readStatus(row: u32) -> vec4<u32> {
  return vec4<u32>(
    status[statusOffset + row * 4u + 0u],
    status[statusOffset + row * 4u + 1u],
    status[statusOffset + row * 4u + 2u],
    status[statusOffset + row * 4u + 3u]
  );
}
`,bindings:[{name:`edges`,view:m,type:`u32`,access:`read`},{name:`conductances`,view:h,type:`f32`,access:`read`},{name:`potentials`,view:T,type:`f32`,access:`read`},{name:`status`,view:k,type:`u32`,access:`read`},{name:`output`,view:E,type:`f32`,access:`read_write`}],body:`var current = getNaN();
  let endpoints = readEdges(index);
  if (readStatus(0u).x != 0u) {
    let conductance = conductances[conductancesOffset + index];
    current = 0.0;
    if (conductance > 0.0) {
      current = conductance * (potentials[potentialsOffset + endpoints.x] - potentials[potentialsOffset + endpoints.y]);
    }
  }
  output[outputOffset + index] = current;`})),{outputs:{potentials:T,currents:E,vertexCurrents:O,diagnostics:M},status:k,estimatedCommandNodeCount:C,provenance:{edges:m,conductances:h,source:_,sink:v,injection:1,tolerance:x,maximumIterations:b,connectivityIterations:S,solver:`grounded-jacobi-conjugate-gradient`}}}export{W as a,M as i,N as n,U as o,I as r,L as t};