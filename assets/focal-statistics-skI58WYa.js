import{t as e}from"./spatial-interpolation-parameters-CfvG78hf.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n,t as r}from"./interpolation-view-validation-pYo9tqh3.js";import{c as i,o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o,f as s,g as c}from"./gpu-contributor-utils-BIwLQON5.js";import{u as l}from"./raster-band-utils-BcTxc8_Y.js";import{r as u,u as d}from"./wgsl-common-DfyXiYB0.js";import{t as f}from"./define-run-B-brkCe4.js";import{i as p}from"./inputs-CJKKIRKK.js";import{l as m,s as h,u as g}from"./pointer-overlay-Qh8qflcA.js";import{c as _,l as v}from"./pick-Bo3t3hjv.js";import{Zs as y}from"./entry-preview-D70TdSbK.js";import{t as b}from"./grid-CuWZ3T2e.js";var x=`GPUFocalStatistics`,S=64,C=class{id;props;requiredHalo;cellCount;constructor(e){this.id=e.id??`focal-statistics`,this.props=e;let{id:t}=this;if(this.cellCount=l(t,e.grid),s(t,`maximumRadius`,e.maximumRadius,0,S),this.requiredHalo=e.maximumRadius,e.noDataValue!==void 0&&!Number.isFinite(e.noDataValue))throw Error(`${t} noDataValue must be finite (NaN cells are always nodata)`);let{output:i}=e,a=w(i);if(a.length===0)throw Error(`${t} needs at least one output`);if(o(e.values,[`float32`],`${t} values`),e.values.length<this.cellCount)throw Error(`${t} values must hold grid.width * grid.height rows`);if(e.validity&&(o(e.validity,[`uint32`],`${t} validity`),e.validity.length<this.cellCount))throw Error(`${t} validity must hold grid.width * grid.height rows`);if(o(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);for(let[e,n]of a)if(e===`counts`?o(n,[`uint32`],`${t} output.counts`):o(n,[`float32`],`${t} output.${e}`),n.length<this.cellCount)throw Error(`${t} output.${e} must hold grid.width * grid.height rows`);e.noDataValue!==void 0&&c(t,`noDataValue`,e.noDataValue),r(t,i,[{write:{mean:i.mean,sum:i.sum,standardDeviation:i.standardDeviation,counts:i.counts}},{write:{min:i.min,max:i.max,range:i.range}}]),n(t,a.map(([,e])=>e),[e.values,e.validity,e.parameters])}getCommandNodes(e){let{id:n,props:r,cellCount:i}=this,{output:o}=r;t(e,n,{values:r.values,validity:r.validity,parameters:r.parameters,...o});let s=[{name:`raster`,view:r.values,type:`f32`,access:`read`},...r.validity?[{name:`validity`,view:r.validity,type:`u32`,access:`read`}]:[],{name:`params`,view:r.parameters,type:`f32`,access:`read`}],c=`const WIDTH: u32 = ${r.grid.width}u;
const HEIGHT: u32 = ${r.grid.height}u;
const MAXIMUM_RADIUS: i32 = ${r.maximumRadius};
${T(!!r.validity,r.noDataValue)}`,l=[],u=(e,t,n=`f32`)=>t?[{name:e,view:t,type:n,access:`read_write`}]:[];return(o.sum||o.mean||o.standardDeviation||o.counts)&&l.push(a(e,{id:`${n}-moments`,operation:x,variant:`moments`,bindings:[...s,...u(`sumOut`,o.sum),...u(`meanOut`,o.mean),...u(`deviationOut`,o.standardDeviation),...u(`countOut`,o.counts,`u32`)],invocationCount:i,workgroupSize:64,declarations:c,body:`${E}
  var count = 0u;
  var sum = 0.0;
  ${D(`count += 1u;
      sum += value;`)}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  let mean = sum / f32(max(count, 1u));
  ${o.sum?`sumOut[sumOutOffset + index] = select(nan, sum, isDefined);`:``}
  ${o.mean?`meanOut[meanOutOffset + index] = select(nan, mean, isDefined);`:``}
  ${o.counts?`countOut[countOutOffset + index] = count;`:``}
  ${o.standardDeviation?`var squaredDeviationSum = 0.0;
  if (isDefined) {
    ${D(`let deviation = value - mean;
      squaredDeviationSum += deviation * deviation;`)}
  }
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`:``}`})),(o.min||o.max||o.range)&&l.push(a(e,{id:`${n}-extremes`,operation:x,variant:`extremes`,bindings:[...s,...u(`minOut`,o.min),...u(`maxOut`,o.max),...u(`rangeOut`,o.range)],invocationCount:i,workgroupSize:64,declarations:c,body:`${E}
  var count = 0u;
  var minimum = 0.0;
  var maximum = 0.0;
  ${D(`if (count == 0u) {
        minimum = value;
        maximum = value;
      } else {
        minimum = min(minimum, value);
        maximum = max(maximum, value);
      }
      count += 1u;`)}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  ${o.min?`minOut[minOutOffset + index] = select(nan, minimum, isDefined);`:``}
  ${o.max?`maxOut[maxOutOffset + index] = select(nan, maximum, isDefined);`:``}
  ${o.range?`rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);`:``}`})),l}};function w(e){return[`mean`,`sum`,`min`,`max`,`range`,`standardDeviation`,`counts`].flatMap(t=>{let n=e[t];return n?[[t,n]]:[]})}function T(e,t){return`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${u}
${d}
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: every non-finite sample (NaN or an infinity) is nodata, the published band rule.
  var isValid = isFiniteValue(sample);
  ${t===void 0?``:`isValid = isValid && sample != ${i(t)};`}
  ${e?`isValid = isValid && validity[validityOffset + cell] != 0u;`:``}
  return isValid;
}`}var E=`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let radiusValue = params[paramsOffset];
  let isCircle = params[paramsOffset + 1u] == 1.0;
  let minimumCountValue = params[paramsOffset + 2u];
  let propagateCenterNoData = params[paramsOffset + 3u] != 0.0;
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  // NaN or negative radii give an empty window.
  let hasWindow = radiusValue >= 0.0;
  let windowRadius = select(-1, i32(min(floor(radiusValue), f32(MAXIMUM_RADIUS))), hasWindow);
  let radiusSquared = radiusValue * radiusValue;
  var centerValue = 0.0;
  let centerValid = readCell(index, &centerValue);`;function D(e){return`for (var dy = -windowRadius; dy <= windowRadius; dy++) {
    let sampleRow = row + dy;
    if (sampleRow < 0 || sampleRow >= i32(HEIGHT)) {
      continue;
    }
    for (var dx = -windowRadius; dx <= windowRadius; dx++) {
      let sampleColumn = column + dx;
      if (sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
        continue;
      }
      if (isCircle && f32(dx * dx + dy * dy) > radiusSquared) {
        continue;
      }
      var value = 0.0;
      if (!readCell(u32(sampleRow) * WIDTH + u32(sampleColumn), &value)) {
        continue;
      }
      ${e}
    }
  }`}var O=[[0,1],[0,1],[0,1],[0,.5],[0,.2]];function k(e,t){return[e.mean,e.min,e.max,e.range,e.standardDeviation][t]}var A=f({parameters:y,caption:`Window statistic of a terrain-like raster; the muted raster is the input.`,createInputs:({world:e})=>{let t=b(e,72),n=p(e,72);return{grid:t,raster:n,windowCells:new Float32Array(t.columns*t.rows),rasterRange:[Math.min(...n.values),Math.max(...n.values)]}},build(t,{grid:n,raster:r}){let i=n.columns*n.rows,a=t.upload(`input`,`float32`,r.values),o=t.allocate(`parameters`,`float32`,8),s={mean:t.allocate(`mean`,`float32`,i),min:t.allocate(`min`,`float32`,i),max:t.allocate(`max`,`float32`,i),range:t.allocate(`range`,`float32`,i),standardDeviation:t.allocate(`standard-deviation`,`float32`,i)};return t.graph.add(new C({values:a.view,grid:n.grid,maximumRadius:8,parameters:o.view,output:{mean:s.mean.view,min:s.min.view,max:s.max.view,range:s.range.view,standardDeviation:s.standardDeviation.view}})),{outputs:{input:a,...s,window:t.upload(`window`,`float32`,new Float32Array(i).fill(NaN))},update:t=>o.write(e({radius:t.radius,shape:t.shape===0?`square`:`circle`}))}},interaction:{kind:`window`,hint:`move and size the window`,rest:[.5,1-m(h(y,`radius`),h(y,`radius`).default)],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let{grid:a}=t,o=g(h(y,`radius`),1-e.unitY),s=_(a.columns,a.rows,a.window,e.x,e.y),c=v(a.columns,a.rows,a.window,s.column,s.row),l=(o+.5)*a.cellSize;j(t,s,o,r.shape===0,t.windowCells),n.window.write(t.windowCells);let u=i(k(n,r.statistic))?.[s.index],d=String(y[2].options?.[r.statistic]);return{values:{radius:o},glyphs:r.shape===0?[{shape:`box`,minimum:[c[0]-l,c[1]-l],maximum:[c[0]+l,c[1]+l]}]:[{shape:`ring`,at:c,radius:l}],readout:`radius ${o} cells · ${d} ${u===void 0?`...`:u.toFixed(2)}`}}},draw:({inputs:e,outputs:t,values:n})=>{let r=k(t,n.statistic);return[{layer:`raster`,context:!0,values:t.input,columns:e.grid.columns,rows:e.grid.rows},{layer:`raster`,values:r,columns:e.grid.columns,rows:e.grid.rows,range:O[n.statistic]},{layer:`raster`,values:t.window,columns:e.grid.columns,rows:e.grid.rows,ramp:`grey`,range:e.rasterRange}]}});function j(e,t,n,r,i){let{columns:a,rows:o}=e.grid;i.fill(NaN);for(let s=Math.max(0,t.row-n);s<=Math.min(o-1,t.row+n);s++)for(let o=Math.max(0,t.column-n);o<=Math.min(a-1,t.column+n);o++)(r||Math.hypot(o-t.column,s-t.row)<=n+.5)&&(i[s*a+o]=e.raster.values[s*a+o])}export{A as default};