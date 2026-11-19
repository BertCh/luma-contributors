import{t as e}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{l as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-reduction-tNgOVQ_C.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,g as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{c as d,t as f}from"./spatial-weights-CqWysstN.js";import{l as p}from"./wgsl-common-DfyXiYB0.js";import{s as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as h}from"./spatial-weights-parameters-DDN52fQU.js";var g=`GPUSpatialWeightsTransform`,_=2**-149,v=1.0000001,y={row:`row`,binary:`binary`,kernel:`kernel`,symmetrize:`symmetric`,double:`double`,variance:`variance`};function b(e){let t=e.weights.information,n=t?.symmetric===!0,r=e.operation===`binary`||e.operation===`double`||e.operation===`symmetrize`||e.operation===`kernel`&&e.bandwidth!==`adaptive`;return{rowCount:e.weights.offsets.length-1,transform:y[e.operation],symmetric:n&&r?!0:`unknown`,hasSelfWeights:t?.hasSelfWeights,provenance:{...t?.provenance,source:g,...e.operation===`kernel`&&typeof e.bandwidth==`number`?{radius:e.bandwidth}:{}}}}var x=class{id;props;outputWeights;bandwidthValues;constructor(e){let n=e.id??`spatial-weights-transform`;if(this.id=n,this.props=e,![`row`,`binary`,`kernel`,`symmetrize`,`double`,`variance`].includes(e.operation))throw Error(`${n} unknown operation ${e.operation}`);if(d(n,e.weights),e.weights.spatialContext&&e.spatialContext&&t(n,e.weights.spatialContext,e.spatialContext),e.output&&(o(e.output,[`float32`],`${n} output`),e.output.length!==e.weights.neighbors.length))throw Error(`${n} output length must equal the slot capacity`);if(e.operation===`kernel`){if(!e.weights.distances)throw Error(`${n} 'kernel' needs weights.distances`);let{bandwidth:t}=e;if(t===void 0)throw Error(`${n} 'kernel' needs a bandwidth`);if(typeof t==`number`?(c(n,`bandwidth`,t,_),this.bandwidthValues=h({bandwidth:t})):t!==`adaptive`&&(o(t,[`float32`],`${n} bandwidth`),s(n,`bandwidth.length`,t.length,1),u(n,[e.output??e.weights.weights],[t])),t===`adaptive`&&e.weights.information?.provenance?.criterion!==`knn`)throw Error(`${n} bandwidth 'adaptive' needs kNN weights`);let r=e.kernel??`triangular`;if(![`gaussian`,`triangular`,`epanechnikov`,`bisquare`,`uniform`].includes(r))throw Error(`${n} unknown kernel ${r}`)}if(e.doubleSum!==void 0&&![`one`,`rows`].includes(e.doubleSum))throw Error(`${n} doubleSum must be 'one' or 'rows'`);if(e.operation===`symmetrize`&&this.isInPlace())throw Error(`${n} 'symmetrize' needs a separate output`);this.outputWeights=f({...e.weights,weights:e.output??e.weights.weights},{information:b(e),status:e.weights.status})}isInPlace(){let{output:e,weights:t}=this.props;return!e||e.buffer===t.weights.buffer}getCommandNodes(t){e(t,this.id,this.props,{"row scratch":this.props.weights.offsets.length*4,"transpose scratch":this.props.weights.neighbors.length*4});let{id:o,props:s}=this,{weights:c,operation:u}=s;l(o,t,[c.offsets,c.neighbors,c.weights,c.distances,s.output,typeof s.bandwidth==`object`?s.bandwidth:void 0]);let d=c.offsets.length-1,f=c.neighbors.length,h=this.isInPlace(),_=h?`values`:`source`,y=[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},...h?[{name:`values`,view:c.weights,type:`f32`,access:`read_write`}]:[{name:`source`,view:c.weights,type:`f32`,access:`read`},{name:`values`,view:s.output,type:`f32`,access:`read_write`}]],b=e=>`${_}[${_}Offset + ${e}]`,x=(e,t)=>`values[valuesOffset + ${e}] = ${t};`,S=`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];`,C=p;if(u===`row`)return[a(t,{id:`${o}-row`,operation:g,variant:`row`,bindings:y,invocationCount:d,declarations:C,body:`${S}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${b(`slot`)};
  }
  let scale = select(0.0, 1.0 / sum, sum > 0.0 && isFiniteFloat(sum));
  for (var slot = begin; slot < end; slot++) {
    ${x(`slot`,`${b(`slot`)} * scale`)}
  }`})];if(u===`double`||u===`variance`){let e=n(t,`${o}-partials`,`float32`,d),i=n(t,`${o}-total`,`float32`,1),c=u===`variance`,l=`var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${b(`slot`)};
    sum += value;
    squares += value * value;
  }
  let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);`,f=c?`let totalValue = total[totalOffset];
  let scale = select(0.0, f32(ROWS) / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`:`let totalValue = total[totalOffset];
  let scale = select(0.0, ${s.doubleSum===`rows`?`f32(ROWS)`:`1.0`} / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`;return[a(t,{id:`${o}-partials`,operation:g,variant:`${u}-partials`,bindings:[...y,{name:`partials`,view:e,type:`f32`,access:`read_write`}],invocationCount:d,declarations:C,body:`${S}
  ${c?`${l}
  partials[partialsOffset + index] = select(0.0, sum / norm, valid);`:`var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${b(`slot`)};
  }
  partials[partialsOffset + index] = select(0.0, sum, isFiniteFloat(sum));`}`}),...new r({id:`${o}-reduce`,input:e,output:i,operation:`sum`}).getCommandNodes(t),a(t,{id:`${o}-apply`,operation:g,variant:`${u}-apply`,bindings:[...y,{name:`total`,view:i,type:`f32`,access:`read`}],invocationCount:d,declarations:`${C}\nconst ROWS: u32 = ${d}u;`,body:`${S}
  ${f}
  ${c?`${l}
  let rowScale = select(0.0, scale / norm, valid);`:`let rowScale = scale;`}
  for (var slot = begin; slot < end; slot++) {
    let result = ${b(`slot`)} * rowScale;
    ${x(`slot`,`select(0.0, result, isFiniteFloat(result) && result >= 0.0)`)}
  }`})]}if(u===`binary`)return[a(t,{id:`${o}-binary`,operation:g,variant:`binary`,bindings:y,invocationCount:d,body:`${S}
  for (var slot = begin; slot < end; slot++) {
    ${x(`slot`,`select(0.0, 1.0, ${b(`slot`)} > 0.0)`)}
  }`})];if(u===`kernel`){let e=s.kernel??`triangular`,n=s.bandwidth,r=n===`adaptive`?void 0:m(t,`${o}-parameters`,typeof n==`number`?this.bandwidthValues:n),l=[y[0],{name:`distances`,view:c.distances,type:`f32`,access:`read`},{name:`values`,view:s.output??c.weights,type:`f32`,access:`read_write`},...r?[{name:`parameters`,view:r.parameters,type:`f32`,access:`read`}]:[]],u={gaussian:`exp(-0.5 * z * z) * 0.3989422804014327`,triangular:`max(1.0 - z, 0.0)`,epanechnikov:`0.75 * max(1.0 - z * z, 0.0)`,bisquare:`0.9375 * max(1.0 - z * z, 0.0) * max(1.0 - z * z, 0.0)`,uniform:`select(0.0, 0.5, z <= 1.0)`}[e];return[...r?.nodes??[],a(t,{id:`${o}-kernel`,operation:g,variant:`kernel`,bindings:l,invocationCount:d,declarations:C,body:`${S}
  ${n===`adaptive`?`var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  // libpysal Kernel: h_i = (distance of the k-th neighbor) * eps, eps = 1.0000001.
  bandwidth = bandwidth * ${i(v)};`:`let bandwidth = parameters[parametersOffset];`}
  for (var slot = begin; slot < end; slot++) {
    let z = select(0.0, distances[distancesOffset + slot] / bandwidth, bandwidth > 0.0);
    let weight = ${u};
    ${x(`slot`,`select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0)`)}
  }`})]}return[a(t,{id:`${o}-symmetrize`,operation:g,variant:`symmetrize`,bindings:[y[0],{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},y[1],y[2]],invocationCount:f,declarations:`const ROWS: u32 = ${d}u;`,body:`if (index >= offsets[offsetsOffset + ROWS]) {
    return;
  }
  var row = 0u;
  var high = ROWS;
  while (row + 1u < high) {
    let middle = (row + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      row = middle;
    } else {
      high = middle;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  var reverse = 0.0;
  if (neighbor < ROWS) {
    var low = offsets[offsetsOffset + neighbor];
    var upper = offsets[offsetsOffset + neighbor + 1u];
    while (low < upper) {
      let middle = (low + upper) / 2u;
      let candidate = neighbors[neighborsOffset + middle];
      if (candidate == row) {
        reverse = source[sourceOffset + middle];
        break;
      }
      if (candidate < row) {
        low = middle + 1u;
      } else {
        upper = middle;
      }
    }
  }
  values[valuesOffset + index] = 0.5 * (source[sourceOffset + index] + reverse);`})]}};export{v as n,x as t};