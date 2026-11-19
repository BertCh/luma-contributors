import{D as e,E as t,T as n,f as r,g as i,v as a,x as o}from"./gpu-contributor-utils-BIwLQON5.js";import{r as s}from"./wgsl-common-DfyXiYB0.js";import{o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{l}from"./raster-band-utils-BcTxc8_Y.js";import{a as u,i as d,n as f,r as p,t as m}from"./wind-6laZ5Fmw.js";import{t as h}from"./define-run-B-brkCe4.js";import{a as g}from"./pointer-overlay-Qh8qflcA.js";import{i as _}from"./kernels-iXfy7fsH.js";import{li as v}from"./entry-preview-D70TdSbK.js";var y=65535,b=`GPUFiniteTimeLyapunovExponent`;function x(e,t=new Float32Array(4)){if(t.length<4)throw Error(`${b} parameter target must hold 4 elements`);return i(b,`seedSpacingX`,e.seedSpacingX,Number.MIN_VALUE),i(b,`seedSpacingY`,e.seedSpacingY,Number.MIN_VALUE),i(b,`integrationTime`,e.integrationTime),n(b,e.integrationTime!==0,`integrationTime must not be 0`),t.set([e.seedSpacingX,e.seedSpacingY,e.integrationTime,0]),t}var S=class{id;props;constructor(i){let a=i.id??`finite-time-lyapunov-exponent`;u(a,i.spatialContext),r(a,`seedWidth`,i.seedWidth,2,y),r(a,`seedHeight`,i.seedHeight,2,y),t(i.positions,[`float32x2`],`${a} positions`),e(a,`positions`,i.positions,i.seedWidth*i.seedHeight),t(i.parameters,[`float32`],`${a} parameters`),n(a,i.parameters.length>=4,`parameters must contain 4 float32 rows`),t(i.output.ftle,[`float32`],`${a} output.ftle`),e(a,`output.ftle`,i.output.ftle,i.seedWidth*i.seedHeight);for(let[n,r]of[[`validity`,i.validity],[`output.validity`,i.output.validity]])r&&(t(r,[`uint32`],`${a} ${n}`),e(a,n,r,i.seedWidth*i.seedHeight));l(a,[i.output.ftle,i.output.validity],[i.positions,i.parameters,i.validity]),this.id=a,this.props=i}getCommandNodes(e){let{id:t,props:n}=this,{seedWidth:r,seedHeight:i}=n;o(t,e,[n.positions,n.validity,n.output.validity,n.parameters,n.output.ftle]),a(e.device,t,`positions`,r*i*8,`use a smaller seed grid`);let l=n.spatialContext?.coordinateSpace===`longitude-latitude`?d(e,t):void 0;return[...l?.nodes??[],c(e,{id:`${t}-exponent`,operation:b,invocationCount:r*i,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`ftle`,view:n.output.ftle,type:`f32`,access:`read_write`},...n.validity?[{name:`validity`,view:n.validity,type:`u32`,access:`read`}]:[],...n.output.validity?[{name:`outputValidity`,view:n.output.validity,type:`u32`,access:`read_write`}]:[],...l?[l.binding]:[]],modules:l?.modules,defines:l?.defines,declarations:`${s}
${l?.declarations??``}
const SEED_WIDTH: u32 = ${r}u;
const SEED_HEIGHT: u32 = ${i}u;
fn readPosition(column: u32, row: u32) -> vec2<f32> {
  let base = positionsOffset + 2u * (row * SEED_WIDTH + column);
  ${n.validity?`if (validity[validityOffset + row * SEED_WIDTH + column] == 0u) { return vec2<f32>(getNaN()); }`:``}
  let position = vec2<f32>(positions[base], positions[base + 1u]);
  ${l?`let projected = ${l.forward}(position, 1u);
  if (projected.valid == 0u) { return vec2<f32>(getNaN()); }
  return projected.position;`:`return position;`}
}
fn isNanPosition(value: vec2<f32>) -> bool {
  return (bitcast<u32>(value.x) & 0x7fffffffu) >= 0x7f800000u ||
    (bitcast<u32>(value.y) & 0x7fffffffu) >= 0x7f800000u;
}`,body:`
  let column = index % SEED_WIDTH;
  let row = index / SEED_WIDTH;
  let column0 = select(column, column - 1u, column > 0u);
  let column1 = select(column, column + 1u, column + 1u < SEED_WIDTH);
  let row0 = select(row, row - 1u, row > 0u);
  let row1 = select(row, row + 1u, row + 1u < SEED_HEIGHT);
  let west = readPosition(column0, row);
  let east = readPosition(column1, row);
  let south = readPosition(column, row0);
  let north = readPosition(column, row1);
  let center = readPosition(column, row);
  var exponent = getNaN();
  if (!(isNanPosition(center) || isNanPosition(west) || isNanPosition(east) || isNanPosition(south) || isNanPosition(north))) {
    let spacingX = parameters[parametersOffset];
    let spacingY = parameters[parametersOffset + 1u];
    let time = abs(parameters[parametersOffset + 2u]);
    let gradientX = (east - west) / (f32(column1 - column0) * spacingX);
    let gradientY = (north - south) / (f32(row1 - row0) * spacingY);
    let a = dot(gradientX, gradientX);
    let b = dot(gradientX, gradientY);
    let d = dot(gradientY, gradientY);
    let largest = 0.5 * (a + d) + sqrt(0.25 * (a - d) * (a - d) + b * b);
    exponent = 0.5 * log(largest) / time;
  }
  ftle[ftleOffset + index] = exponent;
  ${n.output.validity?`outputValidity[outputValidityOffset + index] = select(1u, 0u, (bitcast<u32>(exponent) & 0x7fffffffu) >= 0x7f800000u);`:``}`})]}},C=64,w=64,T=.1,E=[0,1.2],D=h({parameters:v,caption:`FTLE of the Helene wind: left of centre integrates backward (attracting ridges), right forward (repelling). Steady wind; time units are uncalibrated (1 m/s = 0.01 field heights per unit).`,createInputs:({aspect:e})=>{let t=m(e),n=t.columns*t.cellSize,r=Math.round(C*n),i=Float32Array.from({length:r*C*2},(e,t)=>{let i=t>>1;return t&1?(Math.floor(i/r)+.5)/C:(i%r+.5)*n/r});return{...t,width:n,seeds:i,seedColumns:r,seedCount:r*C}},build(e,t){let n=e.upload(`velocities`,`float32x2`,t.velocities),r=e.allocate(`flow-parameters`,`float32`,8),i=e.allocate(`ftle-parameters`,`float32`,4),a=e.allocate(`positions`,`float32x2`,t.seedCount),o=e.allocate(`ftle`,`float32`,t.seedCount);e.graph.add(new f({seeds:e.upload(`seeds`,`float32x2`,t.seeds).view,velocitiesStart:n.view,velocitiesEnd:n.view,fieldWidth:t.columns,fieldHeight:t.rows,stepCount:w,parameters:r.view,output:{positions:a.view}})),e.graph.add(new S({positions:a.view,seedWidth:t.seedColumns,seedHeight:C,parameters:i.view,output:{ftle:o.view}}));let s=e.allocate(`picture`,`float32`,t.seedCount);return _(e,`ftle-flip`,o,t.seedColumns,s),{outputs:{picture:s},update:e=>{let n=Math.sign(e.integrationTime||1)*Math.max(T,Math.abs(e.integrationTime));r.write(p({fieldExtent:[0,0,t.cellSize,t.cellSize],startTime:0,timeStep:n/w,sliceStartTime:0,sliceEndTime:1})),i.write(x({seedSpacingX:t.width/t.seedColumns,seedSpacingY:1/C,integrationTime:n}))}}},interaction:g(v,{kind:`time`,hint:`scrub integration time`,x:`integrationTime`,decorate:e=>({readout:`integrationTime ${e.integrationTime.toFixed(2)} · ${O(e.integrationTime)}`})}),draw:({inputs:e,outputs:t})=>[{layer:`raster`,values:t.picture,columns:e.seedColumns,rows:C,ramp:`sequential`,range:E,smoothing:!0}]});function O(e){return e===0?`instantaneous strain (no direction)`:e<0?`backward · attracting ridges`:`forward · repelling ridges`}export{D as default};