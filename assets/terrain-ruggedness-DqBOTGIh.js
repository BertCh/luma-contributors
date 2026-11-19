import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,x as r}from"./gpu-contributor-utils-BIwLQON5.js";import{c as i,l as a,n as o,r as s,t as c,u as l}from"./raster-band-utils-BcTxc8_Y.js";import{t as u}from"./define-run-B-brkCe4.js";import{t as d}from"./pointer-BY8DXNWA.js";import{a as f,d as p,i as m,l as h,m as g,o as _,u as v}from"./terrain-Cc9dW1uq.js";import{w as y}from"./entry-preview-D70TdSbK.js";var b=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-ruggedness`,this.props=e;let{id:t}=this,{output:r}=e,{width:i,height:s}=e.grid,c=l(t,e.grid);if(!r.topographicPositionIndex&&!r.terrainRuggednessIndex&&!r.roughness&&!r.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,i]of[[`topographicPositionIndex`,r.topographicPositionIndex],[`terrainRuggednessIndex`,r.terrainRuggednessIndex],[`roughness`,r.roughness]])if(i&&(n(i,[`float32`],`${t} output.${e}`),i.length!==c))throw Error(`${t} output.${e} must contain one value per pixel`);if(r.outputValidity&&(n(r.outputValidity,[`uint32`],`${t} output.outputValidity`),r.outputValidity.length!==c))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(![`riley`,`wilson`].includes(e.terrainRuggednessAlgorithm??`riley`))throw Error(`${t} terrainRuggednessAlgorithm must be riley or wilson`);let u=e.edgeMode??`nodata`;if(![`nodata`,`extrapolate`].includes(u))throw Error(`${t} edgeMode must be nodata or extrapolate`);if(u===`extrapolate`&&(i<2||s<2))throw Error(`${t} edgeMode extrapolate requires width and height of at least 2`);a(t,[r.topographicPositionIndex,r.terrainRuggednessIndex,r.roughness,r.outputValidity],o(e.elevation))}getCommandNodes(n){e(n,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height});let{id:a,props:o}=this,{output:c}=o,{width:l,height:u}=o.grid;i(a,n,o.elevation,[]),r(a,n,[c.topographicPositionIndex,c.terrainRuggednessIndex,c.roughness,c.outputValidity]);let d=s(n,a,o.elevation,o.grid,!0),f=[{name:`elevation`,view:d.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:d.band.validity,type:`u32`,access:`read`}],p=[[`tpiValues`,c.topographicPositionIndex,`f32`],[`rugValues`,c.terrainRuggednessIndex,`f32`],[`roughnessValues`,c.roughness,`f32`],[`validityValues`,c.outputValidity,`u32`]];for(let[e,t,n]of p)t&&f.push({name:e,view:t,type:n,access:`read_write`});let m=o.edgeMode??`nodata`,h=o.terrainRuggednessAlgorithm??`riley`,g=t(n,{id:`${a}-window`,operation:`GPUTerrainRuggedness`,variant:`${m}-${h}`,bindings:f,invocationCount:l*u,declarations:x(l,u,m),body:S(o,h)});return[...d.nodes,g]}};function x(e,t,n){let r=`const WIDTH: i32 = ${e};
const HEIGHT: i32 = ${t};
${c}
fn isSampleValid(row: i32, column: i32) -> bool {
  let sampleIndex = u32(row * WIDTH + column);
  return elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
    isFiniteValue(elevation[elevationOffset + sampleIndex]);
}
fn getSample(row: i32, column: i32) -> f32 {
  return elevation[elevationOffset + u32(row * WIDTH + column)];
}`;return n===`nodata`?r:`${r}
struct WindowSample { value: f32, valid: bool };
fn getExtrapolated(
  rowA: i32, columnA: i32, rowB: i32, columnB: i32
) -> WindowSample {
  let valid = isSampleValid(rowA, columnA) && isSampleValid(rowB, columnB);
  return WindowSample(2.0 * getSample(rowA, columnA) - getSample(rowB, columnB), valid);
}
fn getWindowSample(row: i32, column: i32, deltaRow: i32, deltaColumn: i32) -> WindowSample {
  if (row == 0 || row == HEIGHT - 1) {
    let sampleColumn = clamp(column + deltaColumn, 0, WIDTH - 1);
    if (row == 0 && deltaRow < 0) {
      return getExtrapolated(0, sampleColumn, 1, sampleColumn);
    }
    if (row == HEIGHT - 1 && deltaRow > 0) {
      return getExtrapolated(HEIGHT - 1, sampleColumn, HEIGHT - 2, sampleColumn);
    }
    let sampleRow = row + deltaRow;
    return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
  }
  let sampleRow = row + deltaRow;
  if (column == 0 && deltaColumn < 0) {
    return getExtrapolated(sampleRow, 0, sampleRow, 1);
  }
  if (column == WIDTH - 1 && deltaColumn > 0) {
    return getExtrapolated(sampleRow, WIDTH - 1, sampleRow, WIDTH - 2);
  }
  let sampleColumn = column + deltaColumn;
  return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
}`}function S(e,t){return`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  var windowValues: array<f32, 9>;
  let centerValue = getSample(row, column);
  ${(e.edgeMode??`nodata`)===`nodata`?`var isValid = row > 0 && row < HEIGHT - 1 && column > 0 && column < WIDTH - 1;
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let sampleRow = row + k / 3 - 1;
      let sampleColumn = column + k % 3 - 1;
      isValid = isValid && isSampleValid(sampleRow, sampleColumn);
      if (isValid) {
        windowValues[k] = getSample(sampleRow, sampleColumn);
      }
    }
  }`:`var isValid = isSampleValid(row, column);
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let windowSample = getWindowSample(row, column, k / 3 - 1, k % 3 - 1);
      // gdaldem replaces invalid window samples by the centre when computing edges.
      windowValues[k] = select(centerValue, windowSample.value, windowSample.valid);
    }
  }`}
  var differenceSum = 0.0;
  var absoluteSum = 0.0;
  var squareSum = 0.0;
  var minimum = windowValues[0];
  var maximum = windowValues[0];
  for (var k = 0; k < 9; k++) {
    // Differences against the centre are formed first so they stay exact near large offsets.
    let difference = windowValues[k] - centerValue;
    differenceSum += difference;
    absoluteSum += abs(difference);
    squareSum += difference * difference;
    minimum = min(minimum, windowValues[k]);
    maximum = max(maximum, windowValues[k]);
  }
  let topographicPosition = -differenceSum / 8.0;
  ${t===`riley`?`let ruggedness = sqrt(squareSum);`:`let ruggedness = absoluteSum / 8.0;`}
  let roughness = maximum - minimum;
  isValid = isValid && isFiniteValue(topographicPosition) && isFiniteValue(ruggedness) &&
    isFiniteValue(roughness);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${e.output.topographicPositionIndex?`tpiValues[tpiValuesOffset + index] = select(invalidValue, topographicPosition, isValid);`:``}
  ${e.output.terrainRuggednessIndex?`rugValues[rugValuesOffset + index] = select(invalidValue, ruggedness, isValid);`:``}
  ${e.output.roughness?`roughnessValues[roughnessValuesOffset + index] = select(invalidValue, roughness, isValid);`:``}
  ${e.output.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`}var C=[[0,300],[0,300],[-40,40]],w=u({parameters:y,caption:`Elevation in; local ruggedness out: bright where the surface is broken up.`,createInputs:({world:e})=>m(e),build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=g(e,t),s=e.allocate(`result`,`float32`,o);return e.graph.add(new b({grid:r,elevation:i,terrainRuggednessAlgorithm:n.terrainRuggednessAlgorithm===1?`wilson`:`riley`,edgeMode:n.edgeMode===1?`extrapolate`:`nodata`,output:{[[`terrainRuggednessIndex`,`roughness`,`topographicPositionIndex`][n.output]]:s.view}})),{outputs:{elevation:a,result:s}}},interaction:{kind:`probe`,hint:`hover a cell`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let[o,s]=h(t,i,e.x,e.y),c=d(v(t,i,o,s),1,p(t,i)),l=a(n.result),u=[`TRI`,`roughness`,`TPI`][r.output];return{glyphs:[{shape:`box`,...c}],readout:`${u} ${l?Math.round(l[s*t.columns+o]):`...`} m · ${[`riley`,`wilson`][r.terrainRuggednessAlgorithm]}`}}},draw:({inputs:e,outputs:t,values:n})=>[f(e,t.elevation),_(e,t.result,{ramp:n.output===2?`diverging`:`sequential`,range:C[n.output],smoothing:!0})]});export{w as default};