import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{c as n,l as r,n as i,r as a,u as o}from"./raster-band-utils-BcTxc8_Y.js";import{E as s,S as c,T as l,x as u}from"./gpu-contributor-utils-BIwLQON5.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as f}from"./terrain-flow-spill-utils-BvdNovhC.js";import{r as p}from"./parallel-merge-tree-DHFVp61C.js";import{t as m}from"./define-run-B-brkCe4.js";import{a as h,i as g,m as _,n as v,r as y,u as b}from"./terrain-Cc9dW1uq.js";import{H as x}from"./entry-preview-D70TdSbK.js";function S(e,t){let n=t.width*t.height,r=n*5,i=`${f(t.width,t.height)}
const CELL_COUNT: u32 = ${n}u;`,{nodes:a,parents:o,children:s,createWordView:c,appendKernel:l,appendJumpNodes:u}=p(e,{id:`${t.id}-tree`,operation:`GPUTerrainPeakProminence`,nodeCount:r,edgeCount:n*4,edgeBindings:[{name:`keys`,view:t.sortedKeys,type:`u32`,access:`read`},{name:`edges`,view:t.sortedEdges,type:`u32`,access:`read`}],declarations:i,edgeBody:`if (keys[keysOffset + index] == 0u) { return; }
  let edge = edges[edgesOffset + index];
  let firstEndpoint = edge / 4u;
  let secondEndpoint = getSpillNeighbor(firstEndpoint, edge % 4u);
  let node = CELL_COUNT + index;`}),d=c(`elders`,r);return l(`elders-init`,r,[{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`validity`,view:t.validity,type:`u32`,access:`read`},{name:`elders`,view:d,type:`u32`,access:`read_write`}],`var elder = NONE;
  if (index < CELL_COUNT) {
    if (validity[validityOffset + index] != 0u && isFiniteValue(values[valuesOffset + index])) {
      elder = index;
    }
  }
  elders[eldersOffset + index] = elder;`,i),u(`elders`,r,1,Math.ceil(Math.log2(r))+1,o,`
  var ancestor = table[tableOffset + index];
  let candidate = atomicLoad(&elders[eldersOffset + index]);
  if (ancestor != NONE) {
    if (candidate != NONE) {
      var previous = atomicLoad(&elders[eldersOffset + ancestor]);
      loop {
        if (previous != NONE) {
          let previousHeight = values[valuesOffset + previous];
          let candidateHeight = values[valuesOffset + candidate];
          if (previousHeight > candidateHeight || (previousHeight == candidateHeight && previous <= candidate)) { break; }
        }
        let exchange = atomicCompareExchangeWeak(&elders[eldersOffset + ancestor], previous, candidate);
        if (exchange.exchanged) { break; }
        previous = exchange.old_value;
      }
    }
    ancestor = table[tableOffset + ancestor];
  }
  scratch[scratchOffset + index] = ancestor;
  if (ancestor != NONE) { atomicOr(&pending[pendingOffset], 1u); }
  `,``,[{name:`elders`,view:d,type:`atomic<u32>`,access:`read_write`},{name:`values`,view:t.values,type:`f32`,access:`read`}]),l(`record`,n*4,[{name:`children`,view:s,type:`u32`,access:`read`},{name:`elders`,view:d,type:`u32`,access:`read`},{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`keys`,view:t.sortedKeys,type:`u32`,access:`read`},{name:`result`,view:t.result,type:`u32`,access:`read_write`}],`let node = CELL_COUNT + index;
  let first = children[childrenOffset + node * 2u];
  let second = children[childrenOffset + node * 2u + 1u];
  // A cycle edge has one child and creates no death in the join tree.
  if (first == NONE || first == second) { return; }
  let firstPeak = elders[eldersOffset + first];
  let secondPeak = elders[eldersOffset + second];
  let firstHeight = values[valuesOffset + firstPeak];
  let secondHeight = values[valuesOffset + secondPeak];
  let firstWins = firstHeight > secondHeight || (firstHeight == secondHeight && firstPeak < secondPeak);
  let winner = select(secondPeak, firstPeak, firstWins);
  let loser = select(firstPeak, secondPeak, firstWins);
  // Each leaf loses at exactly one binary merge, so these writes are disjoint.
  result[resultOffset + loser] = getSpillValueBits(keys[keysOffset + index]);
  result[resultOffset + CELL_COUNT + loser] = winner;`,i),a}var C=`GPUTerrainPeakProminence`;function w(e){return Math.min(429496729,Math.floor(e.limits.maxStorageBufferBindingSize/40))}var T=class{id;props;constructor(e){this.id=e.id??`terrain-peak-prominence`,this.props=e;let t=o(this.id,e.grid);for(let[n,r,i]of[[`prominence`,e.output.prominence,`float32`],[`keySaddle`,e.output.keySaddle,`float32`],[`parentPeak`,e.output.parentPeak,`uint32`]])r&&(s(r,[i],`${this.id} ${n}`),l(this.id,r.length===t,`${n} must have one element per raster cell (${t})`));r(this.id,Object.values(e.output),i(e.elevation))}getCommandNodes(r){let{id:i,props:o}=this,{width:s,height:l}=o.grid,p=s*l;c(i,`grid cells`,p,w(r.device)),n(i,r,o.elevation,[]),u(i,r,Object.values(o.output));let m=a(r,i,o.elevation,o.grid,!0),h=e(r,`${i}-edge-keys`,`uint32`,p*4),g=e(r,`${i}-edges`,`uint32`,p*4),_=e(r,`${i}-sorted-keys`,`uint32`,p*4),v=e(r,`${i}-sorted-edges`,`uint32`,p*4),y=e(r,`${i}-result`,`uint32`,p*2),b=`${f(s,l)}\nconst CELL_COUNT: u32 = ${p}u;`,x=[...m.nodes,d(r,{id:`${i}-initialize`,operation:C,bindings:[{name:`values`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`validity`,view:m.band.validity,type:`u32`,access:`read`},{name:`result`,view:y,type:`u32`,access:`read_write`}],invocationCount:p,declarations:b,body:`let valid = validity[validityOffset + index] != 0u && isFiniteValue(values[valuesOffset + index]);
          result[resultOffset + index] = select(0x7fc00000u, 0xff800000u, valid);
          result[resultOffset + CELL_COUNT + index] = 0xffffffffu;`}),d(r,{id:`${i}-spill-edges`,operation:C,bindings:[{name:`values`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`validity`,view:m.band.validity,type:`u32`,access:`read`},{name:`keys`,view:h,type:`u32`,access:`read_write`},{name:`edges`,view:g,type:`u32`,access:`read_write`}],invocationCount:p*4,declarations:b,body:`let cell = index / 4u;
          let neighbor = getSpillNeighbor(cell, index % 4u);
          var key = 0u;
          if (neighbor != SPILL_NONE && validity[validityOffset + cell] != 0u && validity[validityOffset + neighbor] != 0u) {
            let first = values[valuesOffset + cell]; let second = values[valuesOffset + neighbor];
            if (isFiniteValue(first) && isFiniteValue(second)) {
              // Normalize signed zero so equal numerical saddles have one stable key.
              let saddle = min(first, second);
              key = getSpillKey(select(saddle, 0.0, saddle == 0.0));
            }
          }
          keys[keysOffset + index] = key; edges[edgesOffset + index] = index;`}),...new t({id:`${i}-sort`,keys:h,values:g,outputKeys:_,outputValues:v,direction:`descending`,algorithm:`radix`}).getCommandNodes(r),...S(r,{id:i,width:s,height:l,values:m.band.storage.values,validity:m.band.validity,sortedKeys:_,sortedEdges:v,result:y})];return x.push(d(r,{id:`${i}-publish`,operation:C,bindings:[{name:`values`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`result`,view:y,type:`u32`,access:`read`},{name:`prominence`,view:o.output.prominence,type:`f32`,access:`read_write`},...o.output.keySaddle?[{name:`saddle`,view:o.output.keySaddle,type:`f32`,access:`read_write`}]:[],...o.output.parentPeak?[{name:`parent`,view:o.output.parentPeak,type:`u32`,access:`read_write`}]:[]],invocationCount:p,declarations:b,body:`let saddleBits = result[resultOffset + index];
        var prominenceBits = 0x7fc00000u;
        if (saddleBits == 0xff800000u) { prominenceBits = 0x7f800000u; }
        else if (saddleBits != 0x7fc00000u) { prominenceBits = bitcast<u32>(values[valuesOffset + index] - bitcast<f32>(saddleBits)); }
        prominence[prominenceOffset + index] = bitcast<f32>(prominenceBits);
        ${o.output.keySaddle?`saddle[saddleOffset + index] = bitcast<f32>(saddleBits);`:``}
        ${o.output.parentPeak?`parent[parentOffset + index] = result[resultOffset + CELL_COUNT + index];`:``}`})),x}},E=m({parameters:x,caption:`Elevation in; prominence and key saddle height at every peak out.`,createInputs:({random:e,world:t})=>{let n=g(t,28);return{...n,noise:Float32Array.from(n.elevation,()=>e()),centers:y(n)}},build(e,t){let{grid:n,band:r,elevation:i,cellCount:a}=_(e,t),o=e.upload(`centers`,`float32x2`,t.centers),s=e.allocate(`prominence`,`float32`,a),c=e.allocate(`key-saddle`,`float32`,a),l=e.allocate(`parent-peak`,`uint32`,a);return e.graph.add(new T({grid:n,elevation:r,output:{prominence:s.view,keySaddle:c.view,parentPeak:l.view}})),{outputs:{elevation:i,prominence:s,keySaddle:c,parentPeak:l,centers:o},update:e=>{let n=v(t,t.noise,e.noise);for(let t=0;t<n.length;t++)n[t]*=e.relief;i.write(n)}}},interaction:{kind:`probe`,hint:`hover a peak`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=i(n.prominence),o=i(n.keySaddle),s=i(n.parentPeak);if(!a||!o||!s)return{readout:`prominence of each peak`};let c=-1,l=1/0;for(let n=0;n<a.length;n++){if(!(a[n]>0))continue;let[i,o]=b(t,r,n%t.columns,Math.floor(n/t.columns)),s=Math.hypot(i-e.x,o-e.y);s<l&&(l=s,c=n)}if(c<0)return{readout:`no peaks`};let u=b(t,r,c%t.columns,Math.floor(c/t.columns)),d=s[c],f=[{shape:`crosshair`,at:u}];return d!==c&&d<a.length&&f.push({shape:`line`,from:u,to:b(t,r,d%t.columns,Math.floor(d/t.columns)),dashed:!0}),{glyphs:f,readout:`prominence ${Math.round(a[c])} m · key saddle ${Math.round(o[c])} m`}}},draw:({outputs:e,inputs:t,values:n})=>[h(t,e.elevation),{layer:`dots`,positions:e.centers,values:n.output===0?e.prominence:e.keySaddle,fade:!0,sizeByValue:!0,range:n.output===0?[0,60]:[650,2300],radius:10}]});export{E as default};