import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./construction-snapshot-DdkX9Pc0.js";import{E as i,S as a,T as o,x as s}from"./gpu-contributor-utils-BIwLQON5.js";import{a as c,c as l,d as u,h as d,l as f,n as p,o as m,r as h,t as g,u as _}from"./raster-band-utils-BcTxc8_Y.js";import{o as v,t as y}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{t as b}from"./gpu-terrain-topographic-position-BiCWovPj.js";import{a as x}from"./terrain-summed-area-table-BJldJnOP.js";import{a as S,u as C}from"./raster-grid-VB0AM_GY.js";import{t as w}from"./define-run-B-brkCe4.js";import{r as T}from"./text-DYSgVV-n.js";import{n as E,r as D,t as O}from"./class-highlight-WAHL_zPO.js";import{a as k,d as A,i as j,l as M,m as N,o as P}from"./terrain-Cc9dW1uq.js";import{v as F}from"./entry-preview-D70TdSbK.js";function I(e,t,n=new Float32Array(8)){return m(`getGPUTerrainWeissLandformsParameterValues`,n,8),v(e,t,n),n[5]=e.standardThreshold??1,n[6]=e.slopeThresholdDegrees??5,n[7]=0,n}var L=class{id;props;localRadius;constructor(e){e=r(e,[`id`,`grid`,`elevation`,`parameters`,`smallScale`,`largeScale`,`standardization`,`quantum`,`output`,`cellSizeMode`]);let t=r(e.grid,[`width`,`height`,`affine`,`pixelInterpretation`,`coordinateReferenceSystem`,`level`,`levelZeroOrigin`]),n=r(e.smallScale??{radius:3},[`radius`,`innerRadius`]),a=r(e.largeScale??{radius:15},[`radius`,`innerRadius`]);e=Object.freeze({...e,grid:Object.freeze({...t,affine:Object.freeze([...t.affine])}),smallScale:Object.freeze({radius:n.radius,innerRadius:n.innerRadius??0}),largeScale:Object.freeze({radius:a.radius,innerRadius:a.innerRadius??0}),elevation:c(e.elevation),output:r(e.output,[`landforms`,`standardizedPosition`,`outputValidity`])}),this.id=e.id??`terrain-weiss-landforms`,this.props=e;let{id:o}=this,{output:s}=e,l=_(o,e.grid),m=this.getScales();for(let e of m){let t=e.innerRadius??0;if(!Number.isSafeInteger(e.radius)||e.radius<1||!Number.isSafeInteger(t)||t<0||t>=e.radius)throw Error(`${o} scales need integer radius >= 1 and 0 <= innerRadius < radius`)}if(this.localRadius=Math.max(1,...m.map(e=>e.radius)),![`global`,`local`].includes(e.standardization??`global`))throw Error(`${o} standardization must be global or local`);if(C(o,e.cellSizeMode),u(o,e.grid,e.cellSizeMode),d(o,e.parameters,8),!s.landforms&&!s.standardizedPosition&&!s.outputValidity)throw Error(`${o} requires at least one output`);if(s.standardizedPosition&&(i(s.standardizedPosition,[`float32`],`${o} output.standardizedPosition`),s.standardizedPosition.length!==2*l))throw Error(`${o} output.standardizedPosition must contain two values per pixel`);for(let[e,t]of[[`landforms`,s.landforms],[`outputValidity`,s.outputValidity]])if(t&&(i(t,[`uint32`],`${o} output.${e}`),t.length!==l))throw Error(`${o} output.${e} must contain one value per pixel`);f(o,[s.landforms,s.standardizedPosition,s.outputValidity],[...p(e.elevation),e.parameters])}getHaloStage(){return o(this.id,this.props.standardization===`local`,`standardization global has no local halo bound; use local for halo assembly`),{requiredHalo:this.localRadius}}getCommandNodes(t){let{id:r,props:i}=this,{output:o}=i,{width:c,height:u}=i.grid;l(r,t,i.elevation,[]),s(r,t,[i.parameters,o.landforms,o.standardizedPosition,o.outputValidity]);let d=c*u;a(r,`grid cells`,d,x(t.device));let f=(i.standardization??`global`)===`global`,p=h(t,r,i.elevation,i.grid,!0),m=p.band.validity;if(!m)throw Error(`${r} canonical elevation is missing validity`);let _=[...p.nodes],v=e(t,`${r}-position`,`float32`,2*d),C=e(t,`${r}-position-validity`,`uint32`,d);_.push(...new b({id:`${r}-position`,grid:i.grid,elevation:p.band,scales:this.getScales(),quantum:i.quantum,output:{...f?{topographicPositionIndex:v}:{deviationFromMean:v},outputValidity:C}}).getCommandNodes(t));let w=e(t,`${r}-slope`,`float32`,d),T=e(t,`${r}-mask`,`uint32`,d);_.push(n(t,{id:`${r}-slope`,operation:`GPUTerrainWeissLandforms`,variant:`slope-${i.cellSizeMode??`uniform`}`,bindings:[{name:`elevationValues`,view:p.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:m,type:`u32`,access:`read`},{name:`settings`,view:i.parameters,type:`f32`,access:`read`},{name:`positionValues`,view:v,type:`f32`,access:`read`},{name:`positionValidity`,view:C,type:`u32`,access:`read`},{name:`slopeValues`,view:w,type:`f32`,access:`read_write`},{name:`maskValues`,view:T,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const WIDTH: u32 = ${c}u;
const HEIGHT: u32 = ${u}u;
const PIXEL_COUNT: u32 = ${d}u;
${g}
${S(i.cellSizeMode??`uniform`,y)}
fn readElevation(column: i32, row: i32) -> vec2<f32> {
  let sampleColumn = u32(clamp(column, 0i, i32(WIDTH) - 1i));
  let sampleRow = u32(clamp(row, 0i, i32(HEIGHT) - 1i));
  let sampleIndex = sampleRow * WIDTH + sampleColumn;
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  let isValid = elevationValidity[elevationValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(value, select(0.0, 1.0, isValid));
}`,body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let centre = readElevation(column, row);
  var isValid = centre.y != 0.0;
  // Horn weights over centre-relative differences, which are exact in float32.
  var eastSum = 0.0;
  var southSum = 0.0;
  for (var dy = -1i; dy <= 1i; dy++) {
    for (var dx = -1i; dx <= 1i; dx++) {
      if (dx == 0i && dy == 0i) {
        continue;
      }
      let sample = readElevation(column + dx, row + dy);
      isValid = isValid && sample.y != 0.0;
      let weight = select(1.0, 2.0, dx == 0i || dy == 0i);
      let difference = sample.x - centre.x;
      eastSum += f32(dx) * weight * difference;
      southSum += f32(dy) * weight * difference;
    }
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Only the gradient magnitude matters, so the row direction does not.
  let eastGradient = zFactor * eastSum / (8.0 * groundCell.x);
  let rowGradient = zFactor * southSum / (8.0 * groundCell.y);
  let slope = atan(length(vec2<f32>(eastGradient, rowGradient))) * 57.29577951308232;
  let small = positionValues[positionValuesOffset + index];
  let large = positionValues[positionValuesOffset + PIXEL_COUNT + index];
  isValid = isValid && isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    positionValidity[positionValidityOffset + index] != 0u;
  slopeValues[slopeValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), slope, isValid);
  maskValues[maskValuesOffset + index] = select(0u, 1u, isValid);`}));let E=[{name:`positionValues`,view:v,type:`f32`,access:`read`},{name:`slopeValues`,view:w,type:`f32`,access:`read`},{name:`settings`,view:i.parameters,type:`f32`,access:`read`}];if(f){let e=this.addGlobalStatisticsNodes(t,_,v,T,d);E.push({name:`statisticsValues`,view:e.values,type:`f32`,access:`read`},{name:`statisticsCount`,view:e.count,type:`u32`,access:`read`})}for(let[e,t,n]of[[`landformValues`,o.landforms,`u32`],[`standardizedValues`,o.standardizedPosition,`f32`],[`validityValues`,o.outputValidity,`u32`]])t&&E.push({name:e,view:t,type:n,access:`read_write`});return _.push(n(t,{id:`${r}-classify`,operation:`GPUTerrainWeissLandforms`,variant:f?`classify-global`:`classify-local`,bindings:E,invocationCount:d,declarations:`const PIXEL_COUNT: u32 = ${d}u;
${g}
fn standardize(value: f32, scale: u32) -> f32 {
  ${f?`let count = f32(statisticsCount[statisticsCountOffset]);
  let mean = statisticsValues[statisticsValuesOffset + scale] / count;
  let deviation = sqrt(statisticsValues[statisticsValuesOffset + 2u + scale] / count);
  return select(0.0, (value - mean) / deviation, deviation > 0.0);`:`return value;`}
}
// 0 = low (at or below -threshold), 1 = mid, 2 = high (at or above threshold).
fn getPositionClass(value: f32, threshold: f32) -> u32 {
  if (value <= -threshold) {
    return 0u;
  }
  return select(1u, 2u, value >= threshold);
}`,body:`let slope = slopeValues[slopeValuesOffset + index];
  let threshold = settings[settingsOffset + 5u];
  let slopeThreshold = settings[settingsOffset + 6u];
  let small = standardize(positionValues[positionValuesOffset + index], 0u);
  let large = standardize(positionValues[positionValuesOffset + PIXEL_COUNT + index], 1u);
  let isValid = isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    isFiniteValue(threshold) && threshold > 0.0 && isFiniteValue(slopeThreshold);
  let smallClass = getPositionClass(small, threshold);
  let largeClass = getPositionClass(large, threshold);
  // Low small position: canyon 1, midslope drainage 2, upland drainage 3.
  var landform = 1u + largeClass;
  if (smallClass == 1u) {
    // Mid small position: valley 4, plain 5, open slope 6, upper slope 7.
    landform = select(select(4u, 7u, largeClass == 2u), select(6u, 5u, slope <= slopeThreshold), largeClass == 1u);
  } else if (smallClass == 2u) {
    // High small position: local ridge 8, midslope ridge 9, mountain top 10.
    landform = 8u + largeClass;
  }
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${o.landforms?`landformValues[landformValuesOffset + index] = select(0u, landform, isValid);`:``}
  ${o.standardizedPosition?`standardizedValues[standardizedValuesOffset + index] = select(invalidValue, small, isValid);
  standardizedValues[standardizedValuesOffset + PIXEL_COUNT + index] = select(invalidValue, large, isValid);`:``}
  ${o.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})),_}getScales(){return[this.props.smallScale??{radius:3},this.props.largeScale??{radius:15}]}addGlobalStatisticsNodes(r,i,a,o,s){let{id:c}=this,l=e(r,`${c}-statistics`,`float32`,4),u=e(r,`${c}-statistics-count`,`uint32`,1),d=e=>r.createDataView(a.buffer,{format:`float32`,length:s,byteOffset:a.byteOffset+e*s*4}),f=e=>r.createDataView(l.buffer,{format:`float32`,length:1,byteOffset:l.byteOffset+e*4});i.push(...new t({id:`${c}-count`,input:o,output:u,operation:`sum`}).getCommandNodes(r));for(let e of[0,1])i.push(...new t({id:`${c}-sum-${e}`,input:d(e),mask:o,output:f(e),operation:`sum`}).getCommandNodes(r));let p=e(r,`${c}-squares`,`float32`,2*s);i.push(n(r,{id:`${c}-squares`,operation:`GPUTerrainWeissLandforms`,variant:`centred-squares`,bindings:[{name:`positionValues`,view:a,type:`f32`,access:`read`},{name:`maskValues`,view:o,type:`u32`,access:`read`},{name:`statisticsValues`,view:l,type:`f32`,access:`read`},{name:`statisticsCount`,view:u,type:`u32`,access:`read`},{name:`squareValues`,view:p,type:`f32`,access:`read_write`}],invocationCount:2*s,declarations:`const PIXEL_COUNT: u32 = ${s}u;`,body:`let scale = index / PIXEL_COUNT;
  let pixel = index - scale * PIXEL_COUNT;
  let mean = statisticsValues[statisticsValuesOffset + scale] / f32(statisticsCount[statisticsCountOffset]);
  let difference = positionValues[positionValuesOffset + index] - mean;
  squareValues[squareValuesOffset + index] = select(0.0, difference * difference, maskValues[maskValuesOffset + pixel] != 0u);`}));for(let e of[0,1])i.push(...new t({id:`${c}-square-sum-${e}`,input:r.createDataView(p.buffer,{format:`float32`,length:s,byteOffset:p.byteOffset+e*s*4}),mask:o,output:f(2+e),operation:`sum`}).getCommandNodes(r));return{values:l,count:u}}},R=[`invalid`,`canyon`,`midslope drainage`,`upland drainage`,`u-shaped valley`,`plain`,`open slope`,`upper slope`,`local ridge`,`midslope ridge`,`mountain top`],z=[`canyon`,`midslope drain.`,`upland drain.`,`u-shaped valley`,`plain`,`open slope`,`upper slope`,`local ridge`,`midslope ridge`,`mountain top`],B=[[0,0,0,0],[.1,.1,.35,1],[.15,.35,.8,1],[.55,.8,.9,1],[.2,.65,.6,1],[.8,.8,.78,1],[.95,.93,.7,1],[.98,.7,.2,1],[.95,.4,.15,1],[.8,.2,.15,1],[.55,.05,.1,1]],V=w({parameters:F,rebuildOn:[`smallScale`,`largeScale`,`standardization`],caption:`Elevation in; ten Weiss landform classes out.`,createInputs:({world:e})=>({...j(e),maskState:O()}),build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=N(e,t),s=e.allocate(`parameters`,`float32`,8),c=e.allocate(`landforms`,`uint32`,o);return e.graph.add(new L({grid:r,elevation:i,parameters:s.view,smallScale:{radius:n.smallScale},largeScale:{radius:n.largeScale},standardization:n.standardization===1?`local`:`global`,output:{landforms:c.view}})),{outputs:{elevation:a,landforms:c,highlight:e.allocate(`highlight`,`float32`,o)},update:e=>s.write(I({slopeThresholdDegrees:e.slopeThresholdDegrees},r))}},interaction:{kind:`probe`,hint:`hover a landform`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let[o,s]=M(t,i,e.x,e.y),c=a(n.landforms),l=A(t,i),u=[e.x,e.y],d=[{shape:`ring`,at:u,radius:r.smallScale*l,dashed:!0},{shape:`ring`,at:u,radius:r.largeScale*l},{shape:`dot`,at:u}];if(!c)return{spotlight:null,glyphs:d,readout:`scales ${r.smallScale} and ${r.largeScale} cells`};let f=c[s*t.columns+o];D(n.highlight,c,f===0?-1:f,t.maskState);let p=0;for(let e of c)e===f&&p++;return{spotlight:null,glyphs:d,readout:`${R[f]??`class `+f} · ${Math.round(100*p/c.length)}% of cells`}}},draw:({inputs:e,outputs:t})=>[k(e,t.elevation),P(e,t.landforms,{ramp:`categorical`,categories:B,alpha:.9}),E(e,t.highlight),{layer:`text`,...T({corner:`bottom-left`,entries:B.slice(1).map((e,t)=>({label:z[t],swatch:[e[0],e[1],e[2]]}))})}]});export{V as default};