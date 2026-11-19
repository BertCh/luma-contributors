import{c as e,o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,f as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{n as s,t as c}from"./float32-values-60_XMjYS.js";import{c as l,o as u}from"./spatial-weights-CqWysstN.js";import{t as d}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{t as f}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as m}from"./define-run-B-brkCe4.js";import{f as h,o as g}from"./inputs-CJKKIRKK.js";import{c as _}from"./text-DYSgVV-n.js";import{n as v,t as y}from"./pick-Bo3t3hjv.js";import{In as b}from"./entry-preview-D70TdSbK.js";import{C as x,a as S,g as C,m as w,v as T}from"./weights-BRH93xop.js";var E=`GPUNeighborhoodSummary`,D=4294967295,O=[`count`,`weightSum`,`sum`,`mean`,`min`,`max`,`standardDeviation`,`median`],k=class{id;props;constructor(e){let t=e.id??`neighborhood-summary`;this.id=t,this.props=e;let n=l(t,e.weights),a=e.statistics??[];for(let e of a)if(!O.includes(e))throw Error(`${t} unknown statistic ${e}`);if(new Set(a).size!==a.length)throw Error(`${t} statistics must not repeat`);let{output:c}=e,u=!!(c.modes||c.entropy);if(a.length===0&&!u)throw Error(`${t} requests no statistic`);if(a.length>0){if(!e.values||!c.table)throw Error(`${t} statistics require values and output.table`);if(s(e.values,`${t} values`)!==n)throw Error(`${t} values needs one row per weights row`);if(r(c.table,[`float32`],`${t} output.table`),c.table.length!==n*a.length)throw Error(`${t} output.table needs rows * statistics.length values`)}else if(c.table||e.values)throw Error(`${t} values and output.table require statistics`);let d=c.status?.overflow;if(a.includes(`median`)){if(!d)throw Error(`${t} median requires output.status.overflow`);if(r(d,[`uint32`],`${t} output.status.overflow`),d.length<1)throw Error(`${t} output.status.overflow needs one row`)}if(i(t,`maximumNeighbors`,e.maximumNeighbors??32,1,64),u){if(!e.categories)throw Error(`${t} modes and entropy require categories`);if(r(e.categories,[`uint32`],`${t} categories`),e.categories.length!==n)throw Error(`${t} categories needs one row per weights row`)}if(c.modes&&(r(c.modes,[`uint32`],`${t} output.modes`),c.modes.length!==n))throw Error(`${t} output.modes needs one row per weights row`);if(c.entropy&&(r(c.entropy,[`float32`],`${t} output.entropy`),c.entropy.length!==n))throw Error(`${t} output.entropy needs one row per weights row`);if(e.mask&&(r(e.mask,[`uint32`],`${t} mask`),e.mask.length!==n))throw Error(`${t} mask needs one row per weights row`);e.weights.status?.overflow&&r(e.weights.status.overflow,[`uint32`],`${t} weights.status.overflow`);let f=e.focalWeight??1;if(!Number.isFinite(Math.fround(f))||f<0)throw Error(`${t} focalWeight must round to a finite non-negative float32`);p(t,{...c,status:void 0,...c.status}),o(t,[c.table,d,c.modes,c.entropy],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.status?.overflow,e.values,e.categories,e.mask])}getCommandNodes(r){let{id:i,props:o}=this;f(r,i,o);let{weights:s,mask:l,output:p}=o;a(i,r,[s.offsets,s.neighbors,s.weights,s.status?.overflow,o.values,o.categories,l,p.table,p.status?.overflow,p.modes,p.entropy]);let m=s.offsets.length-1,h=o.statistics??[],g=!!o.includeFocal,_=s.status?.overflow,v=u(s),y=o.maximumNeighbors??32,b=`const ROWS: u32 = ${m}u;
const CAPACITY: u32 = ${s.neighbors.length}u;
const FOCAL_COUNT: u32 = ${+!!g}u;
const FOCAL_WEIGHT: f32 = ${e(o.focalWeight??1)};
${d}
struct Member {
  row: u32,
  weight: f32,
  valid: bool
}
fn getMemberRange(row: u32) -> vec2<u32> {
  let begin = min(offsets[offsetsOffset + row], CAPACITY);
  let end = min(offsets[offsetsOffset + row + 1u], CAPACITY);
  return vec2<u32>(begin, max(end, begin));
}`,x=e=>l?`mask[maskOffset + ${e}] != 0u`:`true`,S=e=>`
fn getMember(row: u32, begin: u32, visit: u32, scale: f32) -> Member {
  var member = Member(row, FOCAL_WEIGHT, false);
  if (visit >= FOCAL_COUNT) {
    let slot = begin + visit - FOCAL_COUNT;
    member.row = neighbors[neighborsOffset + slot];
    member.weight = weights[weightsOffset + slot];
    // Only active slots are members: another row with a positive weight.
    if (member.row >= ROWS || member.row == row || !(member.weight > 0.0)) {
      return member;
    }
    member.weight = member.weight * scale;
  }
  member.valid = ${x(`member.row`)} && ${e(`member.row`)};
  return member;
}
// Reciprocal sum of the valid slot weights of a row-standardized row (1 for weights as given).
fn getRowScale(row: u32, begin: u32, visits: u32) -> f32 {
  ${v?`var sum = 0.0;
  for (var visit = FOCAL_COUNT; visit < visits; visit++) {
    let member = getMember(row, begin, visit, 1.0);
    if (member.valid) {
      sum += member.weight;
    }
  }
  return select(0.0, 1.0 / sum, sum > 0.0);`:`return 1.0;`}
}`,C=[];if(h.length>0){let e=c(r,`${i}-values`,o.values);C.push(...e.nodes);let a=e.view,u=p.table,d=h.length,f=e=>h.includes(e),g=f(`mean`)||f(`standardDeviation`),v=[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`values`,view:a,type:`f32`,access:`read`},...l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[],{name:`output`,view:u,type:`f32`,access:`read_write`},..._?[{name:`truncation`,view:_,type:`u32`,access:`read`}]:[]];f(`median`)&&(v.push({name:`overflow`,view:p.status.overflow,type:`atomic<u32>`,access:`read_write`}),C.push(n(r,{id:`${i}-overflow-clear`,operation:E,view:p.status.overflow,type:`u32`,value:`0u`,componentCount:1})));let w=e=>{switch(e){case`count`:return`f32(count)`;case`weightSum`:return`weightSum`;case`sum`:return`weightedSum`;case`mean`:return`select(nan, mean, weightSum > 0.0)`;case`min`:return`select(nan, minimum, count > 0u)`;case`max`:return`select(nan, maximum, count > 0u)`;case`standardDeviation`:return`select(nan, sqrt(squares / weightSum), weightSum > 0.0)`;default:return`median`}},T=h.map((e,t)=>`output[outputOffset + index * ${d}u + ${t}u] = ${e===`count`?`0.0`:`nan`};`).join(`
    `),D=h.map((e,t)=>`output[outputOffset + index * ${d}u + ${t}u] = nan;`).join(`
    `),O=h.map((e,t)=>`output[outputOffset + index * ${d}u + ${t}u] = ${w(e)};`).join(`
    `);C.push(t(r,{id:`${i}-numeric`,operation:E,variant:h.join(`-`),bindings:v,invocationCount:m,declarations:`${b}
const MAXIMUM_NEIGHBORS: u32 = ${y}u;
${S(e=>`isFiniteFloat(values[valuesOffset + ${e}])`)}`,body:`let nan = getQuietNaN(index);
  if (!(${x(`index`)})) {
    ${T}
    return;
  }
  ${_?`// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${D}
    return;
  }`:``}
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  var count = 0u;
  var weightSum = 0.0;
  var weightedSum = 0.0;
  var minimum = 3.0e38;
  var maximum = -3.0e38;
  ${f(`median`)?`var sorted: array<f32, ${y}>;`:``}
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (!member.valid) {
      continue;
    }
    let value = values[valuesOffset + member.row];
    weightSum += member.weight;
    weightedSum += member.weight * value;
    minimum = min(minimum, value);
    maximum = max(maximum, value);
    ${f(`median`)?`if (count < MAXIMUM_NEIGHBORS) {
      var position = count;
      loop {
        if (position == 0u || sorted[position - 1u] <= value) {
          break;
        }
        sorted[position] = sorted[position - 1u];
        position--;
      }
      sorted[position] = value;
    }`:``}
    count++;
  }
  ${g?`let mean = weightedSum / weightSum;`:``}
  var squares = 0.0;
  ${f(`standardDeviation`)?`for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid) {
      let deviation = values[valuesOffset + member.row] - mean;
      squares += member.weight * deviation * deviation;
    }
  }`:``}
  ${f(`median`)?`var median = nan;
  if (count > MAXIMUM_NEIGHBORS) {
    atomicMax(&overflow[overflowOffset], 1u);
  } else if (count > 0u) {
    let middle = count / 2u;
    median = select(0.5 * (sorted[middle - 1u] + sorted[middle]), sorted[middle], (count & 1u) == 1u);
  }`:``}
  ${O}`}))}if(p.modes||p.entropy){let e=[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`categories`,view:o.categories,type:`u32`,access:`read`},...l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[],...p.modes?[{name:`modes`,view:p.modes,type:`u32`,access:`read_write`}]:[],...p.entropy?[{name:`entropy`,view:p.entropy,type:`f32`,access:`read_write`}]:[],..._?[{name:`truncation`,view:_,type:`u32`,access:`read`}]:[]];C.push(t(r,{id:`${i}-categorical`,operation:E,variant:`categorical${p.modes?`-mode`:``}${p.entropy?`-entropy`:``}`,bindings:e,invocationCount:m,declarations:`${b}
const NO_MODE: u32 = ${D}u;
const MAXIMUM_NEIGHBORS: u32 = ${y}u;
${S(()=>`true`)}`,body:`let nan = getQuietNaN(index);
  if (!(${x(`index`)})) {
    ${p.modes?`modes[modesOffset + index] = NO_MODE;`:``}
    ${p.entropy?`entropy[entropyOffset + index] = nan;`:``}
    return;
  }
  ${_?`// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${p.modes?`modes[modesOffset + index] = NO_MODE;`:``}
    ${p.entropy?`entropy[entropyOffset + index] = nan;`:``}
    return;
  }`:``}
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  // Decode the members once. Up to MAXIMUM_NEIGHBORS of them are kept in private arrays so the
  // quadratic frequency scan below reads registers instead of re-decoding every member (four
  // global loads) for every pair; larger rows take the unbounded decoding scan.
  var memberCategories: array<u32, ${y}>;
  var memberWeights: array<f32, ${y}>;
  var memberCount = 0u;
  var weightSum = 0.0;
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid) {
      weightSum += member.weight;
      if (memberCount < MAXIMUM_NEIGHBORS) {
        memberCategories[memberCount] = categories[categoriesOffset + member.row];
        memberWeights[memberCount] = member.weight;
      }
      memberCount++;
    }
  }
  var bestCategory = NO_MODE;
  var bestFrequency = -1.0;
  var entropyTotal = 0.0;
  if (memberCount <= MAXIMUM_NEIGHBORS) {
    for (var member = 0u; member < memberCount; member++) {
      let category = memberCategories[member];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < memberCount; other++) {
        if (memberCategories[other] == category) {
          if (other < member) {
            firstOccurrence = false;
            break;
          }
          frequency += memberWeights[other];
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  } else {
    for (var visit = 0u; visit < visits; visit++) {
      let member = getMember(index, range.x, visit, scale);
      if (!member.valid) {
        continue;
      }
      let category = categories[categoriesOffset + member.row];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < visits; other++) {
        let candidate = getMember(index, range.x, other, scale);
        if (candidate.valid && categories[categoriesOffset + candidate.row] == category) {
          if (other < visit) {
            firstOccurrence = false;
            break;
          }
          frequency += candidate.weight;
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  }
  ${p.modes?`modes[modesOffset + index] = bestCategory;`:``}
  ${p.entropy?`entropy[entropyOffset + index] = select(nan, entropyTotal, weightSum > 0.0);`:``}`}))}return C}},A=[`count`,`weightSum`,`sum`,`mean`,`min`,`max`,`standardDeviation`,`median`],j=[e=>[0,e+1],e=>[0,e+1],e=>[0,e+1],()=>[0,1],()=>[0,1],()=>[0,1],()=>[0,.5],()=>[0,1]],M=m({parameters:b,caption:`Input values (left) and one statistic of each neighborhood (right).`,createInputs:({random:e,world:t})=>{let n={width:t.width/2,height:t.height},r=T(h(e,n,260,.35),n),i=Array.from(r.values.keys()).sort((e,t)=>r.values[e]-r.values[t]),a=new Float32Array(r.count);return i.forEach((e,t)=>{a[e]=t/(r.count-1)}),{world:t,...r,values:a,index:v(r.positions),focus:{row:-1}}},build(e,t,n){let{world:r}=t,i=Float32Array.from(t.positions,(e,t)=>t%2==0?e+r.width/2:e),a=e.upload(`positions`,`float32x2`,t.positions),o=e.upload(`shifted-positions`,`float32x2`,i),s=e.upload(`values`,`float32`,t.values),c=x(e,`neighbors`,g(t.positions,n.k)),l=e.allocate(`table`,`float32`,t.count*A.length),u=e.allocate(`overflow`,`uint32`,1);return e.graph.add(new k({weights:c.input,values:s.view,statistics:A,includeFocal:n.includeFocal===1,output:{table:l.view,status:{overflow:u.view}}})),{outputs:{positions:a,shifted:o,sourceValues:s,columns:A.map((n,r)=>S(e,`summary`,l,A.length,r,t.count)),weights:c}}},rebuildOn:[`includeFocal`,`k`],interaction:{kind:`probe`,hint:`hover a point`,rest:[.25,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=t.world.width/2,o=e.x>a?e.x-a:e.x,s=y(t.index,o,e.y,t.focus.row);t.focus.row=s;let c=C(i,n.weights,s),l=i(n.columns[r.statistic]);if(s<0||!c||!l)return{};let u=[t.positions[s*2],t.positions[s*2+1]],d=[u[0]+a,u[1]];return{focus:s,glyphs:[{shape:`ring`,at:u,radius:.02},{shape:`ring`,at:d,radius:.02},...w(t.positions,c).flatMap(e=>{let t=[e[0]+a,e[1]];return[{shape:`line`,from:u,to:e},{shape:`dot`,at:e},{shape:`line`,from:d,to:t,dashed:!0},{shape:`dot`,at:t}]})],readout:`${A[r.statistic]} of ${c.length} neighbors ${l[s].toFixed(2)} (own ${t.values[s].toFixed(2)})`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`dots`,positions:t.positions,values:t.sourceValues,ramp:`sequential`,range:[0,1],radius:4.5},{layer:`dots`,positions:t.shifted,values:t.columns[n.statistic],ramp:`sequential`,range:j[n.statistic](n.k),radius:4.5},{layer:`text`,..._({labels:[{text:`input value`,at:[e.world.width/4,e.world.height],offset:[0,-12],role:`title`,priority:5},{text:`${A[n.statistic]} of ${n.k} nearest`,at:[e.world.width*3/4,e.world.height],offset:[0,-12],role:`title`,priority:5}]})}]});export{M as default};