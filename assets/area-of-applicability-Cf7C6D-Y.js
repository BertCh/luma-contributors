import{i as e}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as n}from"./kernel-C3tfQOn4.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as i}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{D as a,E as o,f as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./regression-output-dispatches-KKVTWkHP.js";import{c as d,r as f}from"./wgsl-common-DfyXiYB0.js";import{t as p}from"./define-run-B-brkCe4.js";import{a as m}from"./inputs-CJKKIRKK.js";import{c as h}from"./pointer-overlay-Qh8qflcA.js";import{rn as g}from"./entry-preview-D70TdSbK.js";var _=65536,v=`GPUAreaOfApplicability`,y=class{props;id;rows;queryRows;constructor(e){this.props=e,this.id=e.id??`area-of-applicability`;let{id:t}=this,n=e.predictorCount;s(t,`predictorCount`,n,1,16),o(e.training,[`float32`],`${t} training`),o(e.query,[`float32`],`${t} query`),this.rows=e.training.length/n,this.queryRows=e.query.length/n,s(t,`training rows`,this.rows,2,_),s(t,`query rows`,this.queryRows,1),e.weights&&(o(e.weights,[`float32`],`${t} weights`),a(t,`weights`,e.weights,n)),e.mask&&(o(e.mask,[`uint32`],`${t} mask`),a(t,`mask`,e.mask,this.rows));let r=e.crossValidation;r&&(s(t,`crossValidation.foldCount`,r.foldCount,1,64),o(r.folds,[`uint32`],`${t} crossValidation.folds`),a(t,`crossValidation.folds`,r.folds,this.rows),o(r.training,[`uint32`],`${t} crossValidation.training`),a(t,`crossValidation.training`,r.training,r.foldCount*this.rows));let{output:i}=e;for(let[e,n,r,s]of[[`trainDissimilarity`,i.trainDissimilarity,`float32`,this.rows],[`queryDissimilarity`,i.queryDissimilarity,`float32`,this.queryRows],[`applicable`,i.applicable,`uint32`,this.queryRows],[`summary`,i.summary,`float32`,6],[`status`,i.status,`uint32`,1]])o(n,[r],`${t} output.${e}`),a(t,`output.${e}`,n,s);u(this.id,i,[[`summary`,`status`],{read:[`summary`],write:[`trainDissimilarity`]},{read:[`summary`],write:[`queryDissimilarity`,`applicable`]}]),l(t,Object.values(i),this.getInputViews())}getInputViews(){let{props:e}=this;return[e.training,e.query,e.weights,e.mask,e.crossValidation?.folds,e.crossValidation?.training].filter(e=>e!==void 0)}getCommandNodes(a){let{id:o,props:s,rows:l,queryRows:u}=this,p=s.predictorCount,m=s.crossValidation,{output:h}=s;t(a,o,{...Object.fromEntries(this.getInputViews().map((e,t)=>[`input ${t}`,e])),...h}),e(a.device,o,{"training row scratch, keys and published sort/scan scratch":l,"feature moments":2*p,"scaled training":l*p,"scaled query":u*p}),c(o,a,[...this.getInputViews(),...Object.values(h)]);let g=(e,t,r)=>n(a,`${o}-${e}`,t,r),_=g(`row-valid`,`uint32`,l),y=g(`moments`,`float32`,2*p),b=g(`scaled`,`float32`,l*p),x=g(`scaled-query`,`float32`,u*p),S=g(`nearest`,`float32`,l),C=g(`mean`,`float32`,l),w=g(`keys`,`uint32`,l),T=g(`indices`,`uint32`,l),E=i(a,{id:`${o}-sort`,operation:v,indices:T,keys:[{view:w,keyBits:32}]}),D=s.weights?[{name:`weights`,view:s.weights,type:`f32`,access:`read`}]:[],O=e=>s.weights?`weights[weightsOffset + ${e}]`:`1.0`;return[r(a,{id:`${o}-valid`,operation:v,variant:`usable-rows`,invocationCount:l,bindings:[{name:`training`,view:s.training,type:`f32`,access:`read`},...s.mask?[{name:`mask`,view:s.mask,type:`u32`,access:`read`}]:[],{name:`rowValid`,view:_,type:`u32`,access:`read_write`}],declarations:d,body:`var valid = ${s.mask?`mask[maskOffset + index] != 0u`:`true`};
  for (var column = 0u; column < ${p}u; column++) {
    valid = valid && isFiniteBits(training[trainingOffset + index * ${p}u + column]);
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, valid);`}),r(a,{id:`${o}-moments`,operation:v,variant:`moments`,invocationCount:p,bindings:[{name:`training`,view:s.training,type:`f32`,access:`read`},{name:`rowValid`,view:_,type:`u32`,access:`read`},{name:`moments`,view:y,type:`f32`,access:`read_write`}],body:`var count = 0.0;
  var sum = 0.0;
  for (var row = 0u; row < ${l}u; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      count += 1.0;
      sum += training[trainingOffset + row * ${p}u + index];
    }
  }
  let mean = sum / count;
  var squares = 0.0;
  for (var row = 0u; row < ${l}u; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let deviation = training[trainingOffset + row * ${p}u + index] - mean;
      squares += deviation * deviation;
    }
  }
  moments[momentsOffset + index * 2u] = mean;
  moments[momentsOffset + index * 2u + 1u] = select(0.0, sqrt(squares / (count - 1.0)), count > 1.0);`}),r(a,{id:`${o}-scale`,operation:v,variant:`standardize-training`,invocationCount:l*p,bindings:[{name:`training`,view:s.training,type:`f32`,access:`read`},{name:`moments`,view:y,type:`f32`,access:`read`},...D,{name:`scaled`,view:b,type:`f32`,access:`read_write`}],body:`let column = index % ${p}u;
  let deviation = training[trainingOffset + index] - moments[momentsOffset + column * 2u];
  let standardized = deviation / moments[momentsOffset + column * 2u + 1u];
  scaled[scaledOffset + index] = ${O(`column`)} * standardized;`}),r(a,{id:`${o}-scale-query`,operation:v,variant:`standardize-query`,invocationCount:u*p,bindings:[{name:`query`,view:s.query,type:`f32`,access:`read`},{name:`moments`,view:y,type:`f32`,access:`read`},...D,{name:`scaledQuery`,view:x,type:`f32`,access:`read_write`}],body:`let column = index % ${p}u;
  let deviation = query[queryOffset + index] - moments[momentsOffset + column * 2u];
  let standardized = deviation / moments[momentsOffset + column * 2u + 1u];
  scaledQuery[scaledQueryOffset + index] = ${O(`column`)} * standardized;`}),r(a,{id:`${o}-training-distances`,operation:v,variant:`training-distances`,invocationCount:l,bindings:[{name:`scaled`,view:b,type:`f32`,access:`read`},{name:`rowValid`,view:_,type:`u32`,access:`read`},...m?[{name:`testFolds`,view:m.folds,type:`u32`,access:`read`},{name:`trainingMasks`,view:m.training,type:`u32`,access:`read`}]:[],{name:`nearest`,view:S,type:`f32`,access:`read_write`},{name:`mean`,view:C,type:`f32`,access:`read_write`}],declarations:f,body:`var nearestSquared = 3.4e38;
  var found = false;
  var total = 0.0;
  var others = 0.0;
  let usable = rowValid[rowValidOffset + index] != 0u;
  ${m?`let testFold = testFolds[testFoldsOffset + index];
  let scored = testFold < ${m.foldCount}u;`:`let scored = true;`}
  if (usable) {
    for (var row = 0u; row < ${l}u; row++) {
      if (row == index || rowValid[rowValidOffset + row] == 0u) { continue; }
      var squared = 0.0;
      for (var column = 0u; column < ${p}u; column++) {
        let difference = scaled[scaledOffset + index * ${p}u + column] - scaled[scaledOffset + row * ${p}u + column];
        squared += difference * difference;
      }
      total += sqrt(squared);
      others += 1.0;
      ${m?`if (scored && trainingMasks[trainingMasksOffset + min(testFold, ${m.foldCount-1}u) * ${l}u + row] != 0u && squared < nearestSquared) {`:`if (squared < nearestSquared) {`}
        nearestSquared = squared;
        found = true;
      }
    }
  }
  nearest[nearestOffset + index] = select(getNaN(), sqrt(nearestSquared), usable && scored && found);
  mean[meanOffset + index] = select(getNaN(), total / others, usable && others > 0.0);`}),r(a,{id:`${o}-mean-distance`,operation:v,variant:`mean-distance`,invocationCount:1,bindings:[{name:`mean`,view:C,type:`f32`,access:`read`},{name:`rowValid`,view:_,type:`u32`,access:`read`},{name:`moments`,view:y,type:`f32`,access:`read`},{name:`summary`,view:h.summary,type:`f32`,access:`read_write`},{name:`status`,view:h.status,type:`u32`,access:`read_write`}],declarations:`${d}\n${f}`,body:`var total = 0.0;
  var count = 0.0;
  var usable = 0.0;
  for (var row = 0u; row < ${l}u; row++) {
    usable += f32(rowValid[rowValidOffset + row]);
    let value = mean[meanOffset + row];
    if (isFiniteBits(value)) {
      total += value;
      count += 1.0;
    }
  }
  var code = 0u;
  if (usable < 2.0) { code |= 1u; }
  for (var column = 0u; column < ${p}u; column++) {
    if (!(moments[momentsOffset + column * 2u + 1u] > 0.0)) { code |= 2u; }
  }
  let meanDistance = select(getNaN(), total / count, count > 0.0 && code == 0u);
  for (var element = 0u; element < 6u; element++) {
    summary[summaryOffset + element] = getNaN();
  }
  summary[summaryOffset] = meanDistance;
  summary[summaryOffset + 2u] = usable;
  status[statusOffset] = code;`}),r(a,{id:`${o}-training-index`,operation:v,variant:`training-index`,invocationCount:l,bindings:[{name:`nearest`,view:S,type:`f32`,access:`read`},{name:`summary`,view:h.summary,type:`f32`,access:`read`},{name:`trainDissimilarity`,view:h.trainDissimilarity,type:`f32`,access:`read_write`},{name:`keys`,view:w,type:`u32`,access:`read_write`},{name:`indices`,view:T,type:`u32`,access:`read_write`}],declarations:`${d}\n${f}`,body:`let meanDistance = summary[summaryOffset];
  let value = nearest[nearestOffset + index] / meanDistance;
  let valid = isFiniteBits(value) && meanDistance > 0.0;
  trainDissimilarity[trainDissimilarityOffset + index] = select(getNaN(), value, valid);
  keys[keysOffset + index] = select(0xffffffffu, bitcast<u32>(value), valid);
  indices[indicesOffset + index] = index;`}),...E.nodes,r(a,{id:`${o}-threshold`,operation:v,variant:`threshold`,invocationCount:1,bindings:[{name:`sortedKeys`,view:E.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`summary`,view:h.summary,type:`f32`,access:`read_write`},{name:`status`,view:h.status,type:`u32`,access:`read_write`}],declarations:f,body:`var scoredCount = 0u;
  for (var row = 0u; row < ${l}u; row++) {
    if (sortedKeys[sortedKeysOffset + row] != 0xffffffffu) { scoredCount++; }
  }
  summary[summaryOffset + 3u] = f32(scoredCount);
  if (scoredCount == 0u) {
    status[statusOffset] = status[statusOffset] | 4u;
  } else {
    let span = f32(scoredCount - 1u);
    var quartiles = vec2f(0.0, 0.0);
    for (var which = 0u; which < 2u; which++) {
      let position = span * select(0.25, 0.75, which == 1u);
      let lowerIndex = u32(floor(position));
      let upperIndex = min(lowerIndex + 1u, scoredCount - 1u);
      let lowerValue = bitcast<f32>(sortedKeys[sortedKeysOffset + lowerIndex]);
      let upperValue = bitcast<f32>(sortedKeys[sortedKeysOffset + upperIndex]);
      quartiles[which] = lowerValue + (position - floor(position)) * (upperValue - lowerValue);
    }
    let maximum = bitcast<f32>(sortedKeys[sortedKeysOffset + scoredCount - 1u]);
    summary[summaryOffset + 1u] = min(quartiles.y + 1.5 * (quartiles.y - quartiles.x), maximum);
    summary[summaryOffset + 4u] = quartiles.x;
    summary[summaryOffset + 5u] = quartiles.y;
  }`}),r(a,{id:`${o}-query`,operation:v,variant:`query-index`,invocationCount:u,bindings:[{name:`query`,view:s.query,type:`f32`,access:`read`},{name:`scaled`,view:b,type:`f32`,access:`read`},{name:`scaledQuery`,view:x,type:`f32`,access:`read`},{name:`rowValid`,view:_,type:`u32`,access:`read`},{name:`summary`,view:h.summary,type:`f32`,access:`read`},{name:`queryDissimilarity`,view:h.queryDissimilarity,type:`f32`,access:`read_write`},{name:`applicable`,view:h.applicable,type:`u32`,access:`read_write`}],declarations:`${d}\n${f}`,body:`var complete = true;
  for (var column = 0u; column < ${p}u; column++) {
    complete = complete && isFiniteBits(query[queryOffset + index * ${p}u + column]);
  }
  var nearestSquared = 3.4e38;
  if (complete) {
    for (var row = 0u; row < ${l}u; row++) {
      if (rowValid[rowValidOffset + row] == 0u) { continue; }
      var squared = 0.0;
      for (var column = 0u; column < ${p}u; column++) {
        let difference = scaledQuery[scaledQueryOffset + index * ${p}u + column] - scaled[scaledOffset + row * ${p}u + column];
        squared += difference * difference;
      }
      nearestSquared = min(nearestSquared, squared);
    }
  }
  let meanDistance = summary[summaryOffset];
  let value = sqrt(nearestSquared) / meanDistance;
  let valid = complete && isFiniteBits(value) && meanDistance > 0.0;
  queryDissimilarity[queryDissimilarityOffset + index] = select(getNaN(), value, valid);
  applicable[applicableOffset + index] = select(0u, 1u, valid && value <= summary[summaryOffset + 1u]);`})]}},b=2,x=[0,3],S=1,C=4294967295;function w({world:e}){let t=m(e),n=[];for(let e=0;e<t.rows;e+=b)for(let r=0;r<t.columns;r+=b)n.push(e*t.columns+r);return{...t,world:e,trainingPixels:Uint32Array.from(n),box:{x:e.width*.3,y:.45}}}function T(e){return[e.red,e.nearInfrared,e.shortwaveInfrared,e.green,e.blue,e.shortwaveInfrared2]}function E(e){return e.discRadius/100*.5}function D(e,t){let n=t%e.columns,r=Math.floor(t/e.columns);return[(n+.5)*e.cellSize,e.world.height-(r+.5)*e.cellSize]}function O(e,t){let n=E(t);return Uint32Array.from(e.trainingPixels,t=>{let[r,i]=D(e,t);return+(Math.hypot(r-e.box.x,i-e.box.y)<=n)})}var k=p({parameters:g,caption:`Dissimilarity index and area of applicability of Sentinel-2 bands for training pixels in a disc around the pointer; no cross-validation folds, so the threshold is tight.`,createInputs:w,build(e,t,n){let r=n.predictorCount,i=T(t).slice(0,r),a=t.columns*t.rows,o=new Float32Array(a*r),s=new Float32Array(t.trainingPixels.length*r);for(let e=0;e<a;e++)i.forEach((t,n)=>o[e*r+n]=t[e]);t.trainingPixels.forEach((e,t)=>{s.set(o.subarray(e*r,(e+1)*r),t*r)});let c=e.allocate(`mask`,`uint32`,t.trainingPixels.length),l=e.allocate(`weights`,`float32`,r),u=e.allocate(`query-dissimilarity`,`float32`,a),d=e.allocate(`applicable`,`uint32`,a),f=e.allocate(`summary`,`float32`,6);return e.graph.add(new y({training:e.upload(`training`,`float32`,s).view,query:e.upload(`query`,`float32`,o).view,predictorCount:r,weights:l.view,mask:c.view,output:{trainDissimilarity:e.allocate(`train-dissimilarity`,`float32`,t.trainingPixels.length).view,queryDissimilarity:u.view,applicable:d.view,summary:f.view,status:e.allocate(`status`,`uint32`,1).view}})),{outputs:{queryDissimilarity:u,applicable:d,summary:f,red:e.upload(`red`,`float32`,t.red)},update:e=>{c.write(O(t,e)),l.write(Float32Array.from({length:r},(t,n)=>n===0?e.weights:1))}}},rebuildOn:[`predictorCount`],buildCache:5,async afterRun({inputs:e,outputs:t,upload:n}){let r=await t.applicable.buffer.readAsync(),i=new Uint32Array(r.buffer,r.byteOffset,e.columns*e.rows);return{outside:n(`outside`,`uint32`,Uint32Array.from(i,e=>e===1?C:S))}},interaction:{kind:`place`,hint:`move the fieldwork`,rest:({inputs:e,toUnit:t})=>t([e.box.x,e.box.y]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=E(r);t.box={x:e.x,y:e.y};let o=[{shape:`ring`,at:[e.x,e.y],radius:a}],s=i(n.summary),c=i(n.applicable),l=`training radius ${r.discRadius}%`;if(s&&c){let e=0;for(let t of c)e+=+(t===1);l=`${h(s[2],0)} samples · threshold ${h(s[1],2)} · ${h(e/c.length*100,0)}% inside`}return{rerun:!0,glyphs:o,readout:l,spotlight:null}}},draw:({inputs:e,outputs:t,extra:n})=>[{layer:`raster`,context:!0,ramp:`grey`,values:t.red,columns:e.columns,rows:e.rows},{layer:`raster`,values:t.queryDissimilarity,columns:e.columns,rows:e.rows,ramp:`sequential`,range:x,alpha:.8},...n?[{layer:`raster`,values:n.outside,columns:e.columns,rows:e.rows,alpha:.6,categories:[[.2,.2,.2],[.95,.6,.1]]}]:[]]});export{k as default};