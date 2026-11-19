import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./scaled-quotient-wgsl-UDyhQag6.js";import{E as i,x as a}from"./gpu-contributor-utils-BIwLQON5.js";import{c as o,d as s,h as c,l,n as u,o as d,r as f,t as p,u as m}from"./raster-band-utils-BcTxc8_Y.js";import{a as h,o as g,t as _}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{a as v,c as y,u as b}from"./raster-grid-VB0AM_GY.js";import{t as x}from"./define-run-B-brkCe4.js";import{l as S,s as C,u as w}from"./pointer-overlay-Qh8qflcA.js";import{t as T}from"./pointer-BY8DXNWA.js";import{a as E,d as D,i as O,l as k,m as A,o as j,u as M}from"./terrain-Cc9dW1uq.js";import{b as N}from"./entry-preview-D70TdSbK.js";var P=3,F=5;function I(e,t,n=new Float32Array(8)){if(d(`getGPUTerrainVectorRuggednessParameterValues`,n,8),e.radius!==void 0&&(!Number.isSafeInteger(e.radius)||e.radius<1))throw Error(`getGPUTerrainVectorRuggednessParameterValues radius must be an integer of at least 1`);return n.fill(0,0,8),g(e,t,n),n[F]=e.radius??0,n}var L=class{id;props;requiredHalo;rowDirection;constructor(e){this.id=e.id??`terrain-vector-ruggedness`,this.props=e;let{id:t}=this,{output:n}=e,r=e.maximumRadius??1;if(!Number.isSafeInteger(r)||r<1)throw Error(`${t} maximumRadius must be an integer of at least 1`);this.requiredHalo=r+1;let a=m(t,e.grid);if(!n.vectorRuggedness&&!n.outputValidity)throw Error(`${t} requires at least one output`);if(n.vectorRuggedness&&(i(n.vectorRuggedness,[`float32`],`${t} output.vectorRuggedness`),n.vectorRuggedness.length!==a))throw Error(`${t} output.vectorRuggedness must contain one value per pixel`);if(n.outputValidity&&(i(n.outputValidity,[`uint32`],`${t} output.outputValidity`),n.outputValidity.length!==a))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(c(t,e.parameters,8),b(t,e.cellSizeMode),s(t,e.grid,e.cellSizeMode),this.rowDirection=h(t,e.grid,e.rowDirection),![`clamp`,`nodata`].includes(e.borderMode??`clamp`))throw Error(`${t} borderMode must be clamp or nodata`);l(t,[n.vectorRuggedness,n.outputValidity],[...u(e.elevation),e.parameters])}getCommandNodes(i){e(i,this.id,this.props,{"normal and filter scratch":4*this.props.grid.width*this.props.grid.height});let{id:s,props:c}=this,{output:l}=c,{width:u,height:d}=c.grid;o(s,i,c.elevation,[]),a(s,i,[c.parameters,l.vectorRuggedness,l.outputValidity]);let m=u*d,h=f(i,s,c.elevation,c.grid,!0),g=[`x`,`y`,`z`].map(e=>t(i,`${s}-normal-${e}`,`float32`,m)),b=n(i,{id:`${s}-normals`,operation:`GPUTerrainVectorRuggedness`,variant:`normals-${c.cellSizeMode??`uniform`}-${c.borderMode??`clamp`}`,bindings:[{name:`elevation`,view:h.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:h.band.validity,type:`u32`,access:`read`},{name:`settings`,view:c.parameters,type:`f32`,access:`read`},{name:`normalX`,view:g[0],type:`f32`,access:`read_write`},{name:`normalY`,view:g[1],type:`f32`,access:`read_write`},{name:`normalZ`,view:g[2],type:`f32`,access:`read_write`}],invocationCount:m,declarations:`const WIDTH: u32 = ${u}u;
const HEIGHT: u32 = ${d}u;
${y(this.rowDirection)}
${p}
${r}
${v(c.cellSizeMode??`uniform`,_)}`,body:`let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  let centreElevation = elevation[elevationOffset + index];
  var isValid = true;
  var windowValues: array<f32, 9>;
  for (var k = 0; k < 9; k++) {
    var sampleRow = row + k / 3 - 1;
    var sampleColumn = column + k % 3 - 1;
    ${(c.borderMode??`clamp`)===`nodata`?`if (sampleRow < 0 || sampleRow >= i32(HEIGHT) || sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
      isValid = false;
      continue;
    }`:``}
    sampleRow = clamp(sampleRow, 0, i32(HEIGHT) - 1);
    sampleColumn = clamp(sampleColumn, 0, i32(WIDTH) - 1);
    let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
    let sampleValue = elevation[elevationOffset + sampleIndex];
    isValid = isValid && elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
      isFiniteValue(sampleValue);
    // Difference before the Horn weights are applied: a large common elevation must not
    // round away a representable local slope when the two weighted sides are subtracted.
    windowValues[k] = sampleValue - centreElevation;
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Keep the cell-size denominator and the vertical factor in exponent form. Metal fast math
  // must not form 8 * cellSize or its reciprocal at the extremes of the float32 range.
  let verticalFactor = frexp(zFactor);
  let eastDifference =
    ((windowValues[2] + 2.0 * windowValues[5] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[3] + windowValues[6]));
  let northDifference =
    ((windowValues[6] + 2.0 * windowValues[7] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[1] + windowValues[2]));
  let eastGradient = getScaledQuotient(eastDifference * verticalFactor.fract, groundCell.x, 1.0, verticalFactor.exp - 3);
  let northGradient = ROW_NORTH_SIGN * getScaledQuotient(northDifference * verticalFactor.fract, groundCell.y, 1.0, verticalFactor.exp - 3);
  isValid = isValid && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient);
  // Normalization squares its components. Bound those components first so finite steep
  // gradients do not overflow that square and collapse the normal to zero.
  let normalScale = max(1.0, max(abs(eastGradient), abs(northGradient)));
  let normal = normalize(vec3<f32>(
    getScaledQuotient(-eastGradient, normalScale, 1.0, 0),
    getScaledQuotient(-northGradient, normalScale, 1.0, 0),
    getScaledQuotient(1.0, normalScale, 1.0, 0)));
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  normalX[normalXOffset + index] = select(invalidValue, normal.x, isValid);
  normalY[normalYOffset + index] = select(invalidValue, normal.y, isValid);
  normalZ[normalZOffset + index] = select(invalidValue, normal.z, isValid);`}),x=c.maximumRadius??1,S={name:`settings`,view:c.parameters,type:`f32`,access:`read`},C=[{name:`normalX`,view:g[0],type:`f32`,access:`read`},{name:`normalY`,view:g[1],type:`f32`,access:`read`},{name:`normalZ`,view:g[2],type:`f32`,access:`read`},S];l.vectorRuggedness&&C.push({name:`vrmValues`,view:l.vectorRuggedness,type:`f32`,access:`read_write`}),l.outputValidity&&C.push({name:`validityValues`,view:l.outputValidity,type:`u32`,access:`read_write`});let w=`let radiusWord = settings[settingsOffset + ${F}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);`,T=e=>`let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let isValid = isFiniteValue(normalX[normalXOffset + index]);
  var vectorSum = vec3<f32>(0.0);
  var count = 0.0;
  if (isValid) {
    ${e}
  }
  // count >= 1 whenever the centre normal is valid.
  let ruggedness = clamp(1.0 - length(vectorSum) / max(count, 1.0), 0.0, 1.0);
  ${l.vectorRuggedness?`vrmValues[vrmValuesOffset + index] = select(invalidValue, ruggedness, isValid);`:``}
  ${l.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`;if(x<P){let e=n(i,{id:`${s}-window`,operation:`GPUTerrainVectorRuggedness`,variant:`window`,bindings:C,invocationCount:m,declarations:`const WIDTH: i32 = ${u};
const HEIGHT: i32 = ${d};
const RADIUS: i32 = ${x};
${p}`,body:`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${w}
  ${T(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
        let sampleIndex = u32(sampleRow * WIDTH + sampleColumn);
        let normalXValue = normalX[normalXOffset + sampleIndex];
        if (isFiniteValue(normalXValue)) {
          vectorSum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
          count += 1.0;
        }
      }
    }`)}`});return[...h.nodes,b,e]}let E=[`x`,`y`,`z`].map(e=>t(i,`${s}-row-sum-${e}`,`float32`,m)),D=t(i,`${s}-row-count`,`uint32`,m),O=n(i,{id:`${s}-window-rows`,operation:`GPUTerrainVectorRuggedness`,variant:`window-rows`,bindings:[{name:`normalX`,view:g[0],type:`f32`,access:`read`},{name:`normalY`,view:g[1],type:`f32`,access:`read`},{name:`normalZ`,view:g[2],type:`f32`,access:`read`},{name:`rowSumX`,view:E[0],type:`f32`,access:`read_write`},{name:`rowSumY`,view:E[1],type:`f32`,access:`read_write`},{name:`rowSumZ`,view:E[2],type:`f32`,access:`read_write`},{name:`rowCounts`,view:D,type:`u32`,access:`read_write`},S],invocationCount:m,declarations:`const WIDTH: i32 = ${u};
const RADIUS: i32 = ${x};
${p}`,body:`let column = i32(index) % WIDTH;
  let rowStart = i32(index) - column;
  ${w}
  var sum = vec3<f32>(0.0);
  var count = 0u;
  for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
    let sampleIndex = u32(rowStart + sampleColumn);
    let normalXValue = normalX[normalXOffset + sampleIndex];
    if (isFiniteValue(normalXValue)) {
      sum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
      count += 1u;
    }
  }
  rowSumX[rowSumXOffset + index] = sum.x;
  rowSumY[rowSumYOffset + index] = sum.y;
  rowSumZ[rowSumZOffset + index] = sum.z;
  rowCounts[rowCountsOffset + index] = count;`}),k=n(i,{id:`${s}-window`,operation:`GPUTerrainVectorRuggedness`,variant:`window-columns`,bindings:[{name:`normalX`,view:g[0],type:`f32`,access:`read`},{name:`rowSumX`,view:E[0],type:`f32`,access:`read`},{name:`rowSumY`,view:E[1],type:`f32`,access:`read`},{name:`rowSumZ`,view:E[2],type:`f32`,access:`read`},{name:`rowCounts`,view:D,type:`u32`,access:`read`},...C.slice(3)],invocationCount:m,declarations:`const WIDTH: i32 = ${u};
const HEIGHT: i32 = ${d};
const RADIUS: i32 = ${x};
${p}`,body:`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${w}
  ${T(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      let sampleIndex = u32(sampleRow * WIDTH + column);
      vectorSum += vec3<f32>(rowSumX[rowSumXOffset + sampleIndex], rowSumY[rowSumYOffset + sampleIndex], rowSumZ[rowSumZOffset + sampleIndex]);
      count += f32(rowCounts[rowCountsOffset + sampleIndex]);
    }`)}`});return[...h.nodes,b,O,k]}},R=x({parameters:N,caption:`Elevation in; the vector ruggedness measure out: orientation scatter, not height.`,createInputs:({world:e})=>O(e),build(e,t){let{grid:n,band:r,elevation:i,cellCount:a}=A(e,t),o=e.allocate(`parameters`,`float32`,8),s=e.allocate(`ruggedness`,`float32`,a);return e.graph.add(new L({grid:n,elevation:r,maximumRadius:7,parameters:o.view,output:{vectorRuggedness:s.view}})),{outputs:{elevation:i,ruggedness:s},update:e=>o.write(I({radius:e.radius,zFactor:e.zFactor},n))}},interaction:{kind:`reach`,hint:`size the window`,rest:[.5,1-S(C(N,`radius`),3)],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=w(C(N,`radius`),1-e.unitY),[o,s]=k(t,r,e.x,e.y),c=T(M(t,r,o,s),a,D(t,r)),l=i(n.ruggedness);return{values:{radius:a},glyphs:[{shape:`box`,...c}],readout:`radius ${a} cells (${2*a+1}x${2*a+1})${l?` · VRM ${l[s*t.columns+o].toFixed(3)}`:``}`}}},draw:({inputs:e,outputs:t})=>[E(e,t.elevation),j(e,t.ruggedness,{range:[0,.2],smoothing:!0})]});export{R as default};