import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,S as i,f as a,s as o,v as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{d as u,r as d}from"./wgsl-common-DfyXiYB0.js";import{t as f}from"./status-aOwAAXek.js";import{t as p}from"./incomplete-gamma-wgsl-BipKCma-.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";var h=`GPUTransitionDiagnostics`,g={Q:0,Q_P_VALUE:1,LR:2,LR_P_VALUE:3,DEGREES_OF_FREEDOM:4},_={SPARSE:0,EMPTY:1},v=class{id;props;conditionCount;chainCount;minimumTransitions;constructor(e){let n=e.id??`transition-diagnostics`;this.id=n,this.props=e,this.conditionCount=e.conditionCount??1,this.chainCount=this.conditionCount>1?this.conditionCount+1:1,this.minimumTransitions=e.minimumTransitions??5;let{classCount:i,output:o}=e;if(a(n,`classCount`,i,1,64),a(n,`conditionCount`,this.conditionCount),a(n,`minimumTransitions`,this.minimumTransitions),t(e.counts,`${n} counts`),e.counts.length<this.conditionCount*i*i)throw Error(`${n} counts is too short`);if(o.homogeneity){if(r(o.homogeneity,[`float32`],`${n} output.homogeneity`),o.homogeneity.length<5)throw Error(`${n} output.homogeneity is too short`);if(this.conditionCount<2)throw Error(`${n} output.homogeneity needs conditionCount of at least 2`)}if(o.steadyStates&&(r(o.steadyStates,[`float32`],`${n} output.steadyStates`),o.steadyStates.length<this.chainCount*i))throw Error(`${n} output.steadyStates is too short`);if(o.stateSparsity&&(t(o.stateSparsity,`${n} output.stateSparsity`),o.stateSparsity.length<2))throw Error(`${n} output.stateSparsity is too short`);if(o.status&&(f(n,o.status),o.status.converged&&!o.steadyStates))throw Error(`${n} output.status.converged needs output.steadyStates`);let s=[o.homogeneity,o.steadyStates,o.stateSparsity,o.status?.converged];l(n,s,[e.counts]);let c=s.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(c).size!==c.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(t){let{id:r,props:i,conditionCount:a,chainCount:l,minimumTransitions:f}=this,{counts:v,classCount:y,output:b}=i,x=b.status?.converged;c(r,t,[v,b.homogeneity,b.steadyStates,b.stateSparsity,x]),m(t,r,{counts:v,...Object.fromEntries(o({...b,status:{converged:b.status?.converged}}).map((e,t)=>[`output[${t}]`,e]))}),b.steadyStates&&s(t.device,r,`chainWork`,l*y*y*4,`reduce conditionCount or classCount`);let S=`const K: u32 = ${y}u;
const C: u32 = ${a}u;
const MINIMUM_TRANSITIONS: u32 = ${f}u;`,C=[];if((b.homogeneity||b.stateSparsity||x)&&C.push(n(t,{id:`${r}-summary`,operation:h,variant:`summary`,bindings:[{name:`counts`,view:v,type:`u32`,access:`read`},...b.homogeneity?[{name:`homogeneity`,view:b.homogeneity,type:`f32`,access:`read_write`}]:[],...b.stateSparsity?[{name:`stateSparsity`,view:b.stateSparsity,type:`u32`,access:`read_write`}]:[],...x?[{name:`converged`,view:x,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`${S}
${d}
${u}
${p}`,body:`${x?`converged[convergedOffset] = 1u;`:``}
  ${b.stateSparsity?`var sparse = 0u;
  var empty = 0u;
  for (var state = 0u; state < C * K; state++) {
    var stateTotal = 0u;
    for (var to = 0u; to < K; to++) {
      stateTotal += counts[countsOffset + state * K + to];
    }
    if (stateTotal == 0u) {
      empty += 1u;
    } else if (stateTotal < MINIMUM_TRANSITIONS) {
      sparse += 1u;
    }
  }
  stateSparsity[stateSparsityOffset + ${_.SPARSE}u] = sparse;
  stateSparsity[stateSparsityOffset + ${_.EMPTY}u] = empty;`:``}
  ${b.homogeneity?`var q = 0.0;
  var likelihoodRatio = 0.0;
  var degreesOfFreedom = 0.0;
  var pooledRow: array<u32, ${y}>;
  for (var origin = 0u; origin < K; origin++) {
    var pooledTotal = 0u;
    var nonZeroCells = 0u;
    for (var to = 0u; to < K; to++) {
      var pooled = 0u;
      for (var condition = 0u; condition < C; condition++) {
        pooled += counts[countsOffset + (condition * K + origin) * K + to];
      }
      pooledRow[to] = pooled;
      pooledTotal += pooled;
      if (pooled > 0u) {
        nonZeroCells += 1u;
      }
    }
    if (pooledTotal == 0u) {
      continue;
    }
    var regimes = 0u;
    for (var condition = 0u; condition < C; condition++) {
      var regimeTotal = 0u;
      for (var to = 0u; to < K; to++) {
        regimeTotal += counts[countsOffset + (condition * K + origin) * K + to];
      }
      if (regimeTotal == 0u) {
        continue;
      }
      regimes += 1u;
      for (var to = 0u; to < K; to++) {
        let observed = f32(counts[countsOffset + (condition * K + origin) * K + to]);
        let expected = f32(regimeTotal) * f32(pooledRow[to]) / f32(pooledTotal);
        if (pooledRow[to] > 0u) {
          q += (observed - expected) * (observed - expected) / expected;
        }
        if (observed > 0.0) {
          likelihoodRatio += observed * log(observed / expected);
        }
      }
    }
    degreesOfFreedom += f32(regimes - 1u) * f32(nonZeroCells - 1u);
  }
  likelihoodRatio = max(2.0 * likelihoodRatio, 0.0);
  let nan = getNaN();
  homogeneity[homogeneityOffset + ${g.Q}u] = q;
  homogeneity[homogeneityOffset + ${g.Q_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(q, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${g.LR}u] = likelihoodRatio;
  homogeneity[homogeneityOffset + ${g.LR_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(likelihoodRatio, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${g.DEGREES_OF_FREEDOM}u] = degreesOfFreedom;`:``}`})),b.steadyStates){let i=e(t,`${r}-chain-work`,`float32`,l*y*y);C.push(n(t,{id:`${r}-steady-states`,operation:h,variant:`steady-states`,bindings:[{name:`counts`,view:v,type:`u32`,access:`read`},{name:`steadyStates`,view:b.steadyStates,type:`f32`,access:`read_write`},{name:`chainWork`,view:i,type:`f32`,access:`read_write`},...x?[{name:`converged`,view:x,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:`${S}
${d}
// Private memory: 2 * K words (at most 512 bytes), far below the 8,192-byte core limit.
var<private> usedStates: array<u32, ${y}>;
var<private> solution: array<f32, ${y}>;

// Transition count of one chain: chain 0 is the pooled chain when there are conditions.
fn getChainCount(chain: u32, origin: u32, to: u32) -> u32 {
  if (C == 1u) {
    return counts[countsOffset + origin * K + to];
  }
  if (chain == 0u) {
    var pooled = 0u;
    for (var condition = 0u; condition < C; condition++) {
      pooled += counts[countsOffset + (condition * K + origin) * K + to];
    }
    return pooled;
  }
  return counts[countsOffset + ((chain - 1u) * K + origin) * K + to];
}`,body:`let chain = index;
  let workBase = chainWorkOffset + chain * K * K;
  let nan = getNaN();
  var activeCount = 0u;
  var solvable = true;
  for (var state = 0u; state < K; state++) {
    var outgoing = 0u;
    var incoming = 0u;
    for (var other = 0u; other < K; other++) {
      outgoing += getChainCount(chain, state, other);
      incoming += getChainCount(chain, other, state);
    }
    if (outgoing > 0u || incoming > 0u) {
      usedStates[activeCount] = state;
      activeCount += 1u;
      if (outgoing == 0u) {
        solvable = false;
      }
    }
  }
  if (activeCount == 0u) {
    // No transitions at all: nothing to solve, and not a failure of the solver.
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    return;
  }
  let m = activeCount;
  if (solvable) {
    for (var a = 0u; a < m; a++) {
      var rowTotal = 0u;
      for (var b = 0u; b < m; b++) {
        rowTotal += getChainCount(chain, usedStates[a], usedStates[b]);
      }
      for (var b = 0u; b < m; b++) {
        chainWork[workBase + a * m + b] = f32(getChainCount(chain, usedStates[a], usedStates[b])) / f32(rowTotal);
      }
    }
    // Grassmann-Taksar-Heyman: eliminate the last state repeatedly. The sum of the remaining
    // outgoing probabilities of a state is zero only when it cannot reach the lower states.
    for (var k = m - 1u; k >= 1u; k--) {
      var escape = 0.0;
      for (var j = 0u; j < k; j++) {
        escape += chainWork[workBase + k * m + j];
      }
      if (!(escape > 0.0)) {
        solvable = false;
        break;
      }
      for (var i = 0u; i < k; i++) {
        chainWork[workBase + i * m + k] = chainWork[workBase + i * m + k] / escape;
      }
      for (var i = 0u; i < k; i++) {
        for (var j = 0u; j < k; j++) {
          chainWork[workBase + i * m + j] = chainWork[workBase + i * m + j] + chainWork[workBase + i * m + k] * chainWork[workBase + k * m + j];
        }
      }
    }
  }
  if (solvable) {
    solution[0] = 1.0;
    var total = 1.0;
    for (var k = 1u; k < m; k++) {
      var weight = 0.0;
      for (var i = 0u; i < k; i++) {
        weight += solution[i] * chainWork[workBase + i * m + k];
      }
      solution[k] = weight;
      total += weight;
    }
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = 0.0;
    }
    for (var a = 0u; a < m; a++) {
      steadyStates[steadyStatesOffset + chain * K + usedStates[a]] = solution[a] / total;
    }
  } else {
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    ${x?`atomicMin(&converged[convergedOffset], 0u);`:``}
  }`}))}return C}},y=`GPUTransitionMatrix`,b=65535,x=4294967295,S=2048,C=16,w=256,T=class{id;props;conditionCount;periodLag;rowCount;constructor(e){let n=e.id??`transition-matrix`;this.id=n,this.props=e,this.periodLag=e.periodLag??1,this.conditionCount=e.conditionCount??1;let{periodCount:o,classCount:s,output:c}=e;if(a(n,`periodLag`,this.periodLag),a(n,`periodCount`,o,this.periodLag+1),a(n,`classCount`,s,1,b),a(n,`conditionCount`,this.conditionCount),!e.conditionClasses&&this.conditionCount!==1)throw Error(`${n} conditionCount needs conditionClasses`);t(e.classes,`${n} classes`);let u=e.classes.length/o;if(!Number.isInteger(u)||u<1)throw Error(`${n} classes length must be a multiple of periodCount`);if(this.rowCount=u,e.conditionClasses&&(t(e.conditionClasses,`${n} conditionClasses`),e.conditionClasses.length!==u*o))throw Error(`${n} conditionClasses length must equal classes length`);if(e.mask&&(t(e.mask,`${n} mask`),e.mask.length!==u))throw Error(`${n} mask length must equal the row count`);let d=this.conditionCount*s*s;if(i(n,`conditionCount * classCount * classCount`,d,x),i(n,`rowCount * (periodCount - periodLag)`,u*(o-this.periodLag),x),i(n,`classes.length`,e.classes.length,x),t(c.counts,`${n} output.counts`),c.counts.length<d)throw Error(`${n} output.counts is too short`);if(c.probabilities&&(r(c.probabilities,[`float32`],`${n} output.probabilities`),c.probabilities.length<d))throw Error(`${n} output.probabilities is too short`);if(c.rowTotals&&(t(c.rowTotals,`${n} output.rowTotals`),c.rowTotals.length<this.conditionCount*s))throw Error(`${n} output.rowTotals is too short`);if(c.ignored&&(t(c.ignored,`${n} output.ignored`),c.ignored.length<1))throw Error(`${n} output.ignored must hold one row`);let f=[c.counts,c.probabilities,c.rowTotals,c.ignored,c.homogeneity,c.steadyStates,c.stateSparsity,c.status?.converged];l(n,f,[e.classes,e.conditionClasses,e.mask]);let p=f.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(p).size!==p.length)throw Error(`${n} outputs must use separate buffers`);this.diagnostics=this.hasDiagnostics()?new v(this.getDiagnosticsProps()):void 0}diagnostics;hasDiagnostics(){let{output:e}=this.props;return!!(e.homogeneity||e.steadyStates||e.stateSparsity||e.status)}getDiagnosticsProps(){let{props:e,conditionCount:t,id:n}=this,{output:r}=e;return{id:`${n}-diagnostics`,counts:r.counts,classCount:e.classCount,conditionCount:t,minimumTransitions:e.minimumTransitions,output:{homogeneity:r.homogeneity,steadyStates:r.steadyStates,stateSparsity:r.stateSparsity,status:r.status}}}getCommandNodes(e){let{id:t,props:r,conditionCount:i,periodLag:a,rowCount:l}=this,{classes:u,conditionClasses:d,mask:f,output:p,periodCount:h,classCount:g}=r;c(t,e,[u,d,f,p.counts,p.probabilities,p.rowTotals,p.ignored,p.homogeneity,p.steadyStates,p.stateSparsity,p.status?.converged]),m(e,t,{classes:u,conditionClasses:d,mask:f,...Object.fromEntries(o({...p,status:{converged:p.status?.converged}}).map((e,t)=>[`output[${t}]`,e]))}),p.steadyStates&&s(e.device,t,`diagnostic chainWork`,(i>1?i+1:1)*g*g*4,`reduce conditionCount or classCount`);let _=i*g*g,v=i*g,b=`const ROWS: u32 = ${l}u;
const CLASSES: u32 = ${g}u;
const CONDITIONS: u32 = ${i}u;
const CELLS: u32 = ${_}u;`,x=l*(h-a),T=_+1<=S,E=[{name:`classes`,view:u,type:`u32`,access:`read`},...d?[{name:`conditions`,view:d,type:`u32`,access:`read`}]:[],...f?[{name:`mask`,view:f,type:`u32`,access:`read`}]:[],{name:`counts`,view:p.counts,type:`atomic<u32>`,access:`read_write`},...p.ignored?[{name:`ignored`,view:p.ignored,type:`atomic<u32>`,access:`read_write`}]:[]],D=()=>n(e,{id:`${t}-count`,operation:y,variant:`count-private`,bindings:E,invocationCount:Math.ceil(x/C),workgroupSize:w,guardIndex:!1,declarations:`${b}
const TRANSITIONS: u32 = ${x}u;
const ITEMS: u32 = ${C}u;
const TILE_SIZE: u32 = ${w}u;
// Slot CELLS counts the transitions that are not counted.
var<workgroup> tile: array<atomic<u32>, ${_+1}>;`,body:`for (var cell = localInvocationIndex; cell <= CELLS; cell += TILE_SIZE) {
    atomicStore(&tile[cell], 0u);
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * ITEMS;
  for (var item = 0u; item < ITEMS; item++) {
    let transition = tileStart + item * TILE_SIZE + localInvocationIndex;
    if (transition < TRANSITIONS) {
      let period = transition / ROWS;
      let row = transition - period * ROWS;
      let fromClass = classes[classesOffset + transition];
      let toClass = classes[classesOffset + transition + ${a}u * ROWS];
      let condition = ${d?`conditions[conditionsOffset + transition]`:`0u`};
      let selected = ${f?`mask[maskOffset + row] != 0u`:`true`};
      if (selected && fromClass < CLASSES && toClass < CLASSES && condition < CONDITIONS) {
        atomicAdd(&tile[(condition * CLASSES + fromClass) * CLASSES + toClass], 1u);
      } else {
        atomicAdd(&tile[CELLS], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var cell = localInvocationIndex; cell <= CELLS; cell += TILE_SIZE) {
    let value = atomicLoad(&tile[cell]);
    if (value > 0u) {
      if (cell < CELLS) {
        atomicAdd(&counts[countsOffset + cell], value);
      } else {
        ${p.ignored?`atomicAdd(&ignored[ignoredOffset], value);`:``}
      }
    }
  }`}),O=[n(e,{id:`${t}-clear`,operation:y,variant:`clear`,bindings:[{name:`counts`,view:p.counts,type:`u32`,access:`read_write`},...p.ignored?[{name:`ignored`,view:p.ignored,type:`u32`,access:`read_write`}]:[]],invocationCount:_,declarations:b,body:`counts[countsOffset + index] = 0u;
  ${p.ignored?`if (index == 0u) { ignored[ignoredOffset] = 0u; }`:``}`}),T?D():n(e,{id:`${t}-count`,operation:y,variant:`count`,bindings:[{name:`classes`,view:u,type:`u32`,access:`read`},...d?[{name:`conditions`,view:d,type:`u32`,access:`read`}]:[],...f?[{name:`mask`,view:f,type:`u32`,access:`read`}]:[],{name:`counts`,view:p.counts,type:`atomic<u32>`,access:`read_write`},...p.ignored?[{name:`ignored`,view:p.ignored,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l*(h-a),declarations:b,body:`let period = index / ROWS;
  let row = index - period * ROWS;
  let fromClass = classes[classesOffset + index];
  let toClass = classes[classesOffset + index + ${a}u * ROWS];
  let condition = ${d?`conditions[conditionsOffset + index]`:`0u`};
  let selected = ${f?`mask[maskOffset + row] != 0u`:`true`};
  if (selected && fromClass < CLASSES && toClass < CLASSES && condition < CONDITIONS) {
    atomicAdd(&counts[countsOffset + (condition * CLASSES + fromClass) * CLASSES + toClass], 1u);
  } else {
    ${p.ignored?`atomicAdd(&ignored[ignoredOffset], 1u);`:``}
  }`})];return(p.probabilities||p.rowTotals)&&O.push(n(e,{id:`${t}-normalize`,operation:y,variant:`normalize`,bindings:[{name:`counts`,view:p.counts,type:`u32`,access:`read`},...p.probabilities?[{name:`probabilities`,view:p.probabilities,type:`f32`,access:`read_write`}]:[],...p.rowTotals?[{name:`rowTotals`,view:p.rowTotals,type:`u32`,access:`read_write`}]:[]],invocationCount:v,declarations:b,body:`var total = 0u;
  for (var to = 0u; to < CLASSES; to++) {
    total += counts[countsOffset + index * CLASSES + to];
  }
  for (var to = 0u; to < CLASSES; to++) {
    ${p.probabilities?`probabilities[probabilitiesOffset + index * CLASSES + to] = select(0.0, f32(counts[countsOffset + index * CLASSES + to]) / f32(total), total > 0u);`:``}
  }
  ${p.rowTotals?`rowTotals[rowTotalsOffset + index] = total;`:``}`})),this.diagnostics&&O.push(...this.diagnostics.getCommandNodes(e)),O}};export{b as n,v as r,T as t};