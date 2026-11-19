import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{r as t}from"./scan-total-C4QddMmi.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i}from"./gpu-command-graph-BdzdmV6u.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as s}from"./spatial-weights-parameters-DDN52fQU.js";import{E as c,T as l,f as u,g as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m,i as h,l as g,n as _,t as v}from"./spatial-weights-CqWysstN.js";var y=2,b=2**-149,x=`GPULatticeWeights`,S=class{id;props;outputWeights;cellSizeParameters;constructor(e){let t=e.id??`lattice-weights`;if(this.id=t,this.props=e,e.criterion!==`rook`&&e.criterion!==`queen`)throw Error(`${t} criterion must be 'rook' or 'queen'`);u(t,`width`,e.width),u(t,`height`,e.height),u(t,`width * height`,e.width*e.height,1,2**31-1);let n=e.radius??1;if(u(t,`radius`,n,1,32),e.cellSize instanceof i)c(e.cellSize,[`float32`],`${t} cellSize`),u(t,`cellSize.length`,e.cellSize.length,2),this.cellSizeParameters=e.cellSize;else{let n=e.cellSize??[1,1];l(t,n.length===2,`cellSize must have exactly 2 elements`),d(t,`cellSize[0]`,n[0],b),d(t,`cellSize[1]`,n[1],b),this.cellSizeParameters=s({cellSize:n})}let r=e.width*e.height;if(e.mask&&(c(e.mask,[`uint32`],`${t} mask`),e.mask.length!==r))throw Error(`${t} mask length must equal width * height`);let a=C(e.output);if(g(t,a,[[`offsets`,`overflow`,`requiredCount`],[`neighbors`,`weights`,`distances`]])!==r)throw Error(`${t} output rows must equal width * height`);p(t,[e.output.neighbors,e.output.weights,e.output.distances],[e.output.offsets]),this.outputWeights=v(e.output,{information:{rowCount:r,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:{source:x,criterion:e.criterion,order:n}},status:e.output.status}),p(t,[...m(e.output),...h(a)],[e.mask,e.cellSize instanceof i?e.cellSize:void 0])}getCapacityPlan(e={}){let t=this.props.width*this.props.height,n=this.props.output.distances&&this.cellSizeParameters instanceof Float32Array?this.cellSizeParameters.byteLength:0,r=_({rowCount:t,capacity:this.props.output.neighbors.length,requiredCount:e.requiredCount,distances:!!this.props.output.distances});return{slots:r,estimatedWorkItems:t+r.capacity,estimatedOutputBytes:r.estimatedOutputBytes,estimatedPeakTransientBytes:y*t*Uint32Array.BYTES_PER_ELEMENT+n}}getCommandNodes(s){e(s,this.id,{...this.props,output:C(this.props.output)},{"row scratch":this.props.width*this.props.height*4});let{id:c,props:l}=this,{width:u,height:d,mask:p}=l,g=l.output,{overflow:_,requiredCount:v}=g.status;f(c,s,[p,l.cellSize instanceof i?l.cellSize:void 0,...m(g),...h(C(g))]);let y=u*d,b=g.neighbors.length,S=l.radius??1,w=g.distances?o(s,`${c}-parameters`,this.cellSizeParameters):void 0,T=n(s,`${c}-counts`,`uint32`,y),E=n(s,`${c}-starts`,`uint32`,y),D=p?[{name:`mask`,view:p,type:`u32`,access:`read`}]:[],O=`const WIDTH: i32 = ${u};
const HEIGHT: i32 = ${d};
const RADIUS: i32 = ${S};
const QUEEN: bool = ${l.criterion===`queen`};`,k=e=>`let cellX = i32(index) % WIDTH;
  let cellY = i32(index) / WIDTH;
  if (${p?`mask[maskOffset + index] != 0u`:`true`}) {
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
      let y = cellY + dy;
      if (y < 0 || y >= HEIGHT) {
        continue;
      }
      for (var dx = -RADIUS; dx <= RADIUS; dx++) {
        let x = cellX + dx;
        if (x < 0 || x >= WIDTH || (dx == 0 && dy == 0)) {
          continue;
        }
        if (!QUEEN && abs(dx) + abs(dy) > RADIUS) {
          continue;
        }
        let neighbor = u32(y * WIDTH + x);
        if (${p?`mask[maskOffset + neighbor] != 0u`:`true`}) {
          ${e}
        }
      }
    }
  }`,A=n(s,`${c}-scan-total`,`uint32`,1);return[...g.status.candidateOverflow?[a(s,{id:`${c}-candidate-status`,operation:x,invocationCount:1,bindings:[{name:`candidateOverflow`,view:g.status.candidateOverflow,type:`u32`,access:`read_write`}],body:`candidateOverflow[candidateOverflowOffset] = 0u;`})]:[],...w?.nodes??[],a(s,{id:`${c}-counts`,operation:x,variant:`counts`,bindings:[...D,{name:`counts`,view:T,type:`u32`,access:`read_write`}],invocationCount:y,declarations:O,body:`var count = 0u;
  ${k(`count++;`)}
  counts[countsOffset + index] = count;`}),...new r({id:`${c}-scan`,input:T,output:E,mode:`exclusive`}).getCommandNodes(s),...t(s,{id:c,operation:x,counts:T,offsets:E,rowCount:y,total:A}),a(s,{id:`${c}-offsets`,operation:x,variant:`offsets`,bindings:[{name:`scanTotal`,view:A,type:`u32`,access:`read`},{name:`starts`,view:E,type:`u32`,access:`read`},{name:`offsets`,view:g.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:_,type:`u32`,access:`read_write`},...v?[{name:`requiredCount`,view:v,type:`u32`,access:`read_write`}]:[]],invocationCount:y+1,declarations:`const ROWS: u32 = ${y}u;
const CAPACITY: u32 = ${b}u;`,body:`if (index < ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    offsets[offsetsOffset + ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${v?`requiredCount[requiredCountOffset] = total;`:``}
  }`}),a(s,{id:`${c}-emit`,operation:x,variant:`emit`,bindings:[...w?[{name:`parameters`,view:w.parameters,type:`f32`,access:`read`}]:[],...D,{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:g.weights,type:`f32`,access:`read_write`},...g.distances?[{name:`distances`,view:g.distances,type:`f32`,access:`read_write`}]:[]],invocationCount:y,declarations:O,body:`var next = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${k(`if (next < end) {
            neighbors[neighborsOffset + next] = neighbor;
            weights[weightsOffset + next] = 1.0;
            ${g.distances?`distances[distancesOffset + next] = sqrt(f32(dx * dx) * parameters[parametersOffset] * parameters[parametersOffset] + f32(dy * dy) * parameters[parametersOffset + 1u] * parameters[parametersOffset + 1u]);`:``}
            next++;
          }`)}`})]}};function C(e){return{...e,status:e.status}}export{S as t};