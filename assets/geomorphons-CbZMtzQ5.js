import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as n,x as r}from"./gpu-contributor-utils-BIwLQON5.js";import{c as i,d as a,h as o,l as s,n as c,o as l,r as u,t as d,u as f}from"./raster-band-utils-BcTxc8_Y.js";import{a as p,o as m,t as h}from"./terrain-geomorphometry-utils-BhBw8dIY.js";import{a as g,c as _,t as v,u as y}from"./raster-grid-VB0AM_GY.js";import{t as b}from"./define-run-B-brkCe4.js";import{n as x,r as S,t as C}from"./class-highlight-WAHL_zPO.js";import{a as w,i as T,l as E,m as D,o as O,u as k}from"./terrain-Cc9dW1uq.js";import{q as A}from"./entry-preview-D70TdSbK.js";function j(e,t,n=new Float32Array(8)){return l(`getGPUGeomorphonsParameterValues`,n,8),m(e,t,n),n[5]=e.flatThresholdDegrees??1,n[6]=e.flatDistance??0,n[7]=0,n}var M=class{id;props;searchRadius;rowDirection;constructor(e){this.id=e.id??`geomorphons`,this.props=e;let{id:t}=this,{output:r}=e,i=f(t,e.grid),l=e.skipRadius??0;if(!Number.isSafeInteger(e.searchRadius)||e.searchRadius<1)throw Error(`${t} searchRadius must be a positive integer`);if(!Number.isSafeInteger(l)||l<0)throw Error(`${t} skipRadius must be a non-negative integer`);if(e.searchRadius<l+2)throw Error(`${t} searchRadius must be at least skipRadius + 2`);if(![`anglev1`,`anglev2`,`anglev2-distance`].includes(e.comparison??`anglev1`))throw Error(`${t} comparison must be anglev1, anglev2, or anglev2-distance`);if(!r.forms&&!r.ternary&&!r.pattern&&!r.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,a]of[[`forms`,r.forms],[`ternary`,r.ternary],[`pattern`,r.pattern],[`outputValidity`,r.outputValidity]])if(a&&(n(a,[`uint32`],`${t} output.${e}`),a.length!==i))throw Error(`${t} output.${e} must contain one value per pixel`);o(t,e.parameters,8),y(t,e.cellSizeMode),a(t,e.grid,e.cellSizeMode),this.rowDirection=p(t,e.grid,e.rowDirection),s(t,[r.forms,r.ternary,r.pattern,r.outputValidity],[...c(e.elevation),e.parameters]),this.searchRadius=e.searchRadius}getCommandNodes(n){e(n,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height});let{id:a,props:o}=this,{output:s}=o,{width:c,height:l}=o.grid;i(a,n,o.elevation,[]),r(a,n,[o.parameters,s.forms,s.ternary,s.pattern,s.outputValidity]);let d=u(n,a,o.elevation,o.grid,!0),f=d.band.validity;if(!f||d.band.storage.kind!==`buffer`)throw Error(`${a} could not canonicalize elevation`);let p=[{name:`elevation`,view:d.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:f,type:`u32`,access:`read`},{name:`settings`,view:o.parameters,type:`f32`,access:`read`}];for(let[e,t]of[[`formValues`,s.forms],[`ternaryValues`,s.ternary],[`patternValues`,s.pattern],[`validityValues`,s.outputValidity]])t&&p.push({name:e,view:t,type:`u32`,access:`read_write`});let m=o.comparison??`anglev1`,h=o.skipRadius??0,g=o.cellSizeMode??`uniform`,_=this.rowDirection;return[...d.nodes,t(n,{id:`${a}-classify`,operation:`GPUGeomorphons`,variant:`${m}-${g}-skip${h}-radius${o.searchRadius}`,bindings:p,invocationCount:c*l,declarations:N(o,m,h,g,_),body:P(o)})]}};function N(e,t,n,r,i){return`
const WIDTH: u32 = ${e.grid.width}u;
const HEIGHT: u32 = ${e.grid.height}u;
const SEARCH_RADIUS: i32 = ${e.searchRadius};
const SKIP_RADIUS: i32 = ${n};
const COMPARISON: u32 = ${t===`anglev1`?0:t===`anglev2`?1:2}u;
${_(i,`i32`)}
${v}
${d}
${g(r,h)}

// Directions in GRASS order: NE, N, NW, W, SW, S, SE, E.
const DIRECTION_COLUMN = array<i32, 8>(1, 0, -1, -1, -1, 0, 1, 1);
const DIRECTION_NORTH = array<i32, 8>(1, 1, 1, 0, -1, -1, -1, 0);
// Landform codes by minus count (row) and plus count (column); 0 marks impossible pairs.
const FORM_TABLE = array<u32, 81>(
  1u, 1u, 1u, 8u, 8u, 9u, 9u, 9u, 10u,
  1u, 1u, 8u, 8u, 8u, 9u, 9u, 9u, 0u,
  1u, 4u, 6u, 6u, 7u, 7u, 9u, 0u, 0u,
  4u, 4u, 6u, 6u, 6u, 7u, 0u, 0u, 0u,
  4u, 4u, 5u, 6u, 6u, 0u, 0u, 0u, 0u,
  3u, 3u, 5u, 5u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
  2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

fn isElevationValid(column: i32, row: i32) -> bool {
  // Canonical elevation is NaN exactly where a pixel is invalid, so no validity load is needed.
  return isFiniteValue(elevation[elevationOffset + u32(row) * WIDTH + u32(column)]);
}

fn getElevation(column: i32, row: i32) -> f32 {
  return elevation[elevationOffset + u32(row) * WIDTH + u32(column)];
}

// Re-encodes the digits cyclically shifted by shift, optionally mirrored, in base 3.
fn encodeRotation(digits: array<u32, 8>, shift: u32, mirrored: bool) -> u32 {
  var code = 0u;
  var power = 1u;
  for (var position = 0u; position < 8u; position++) {
    let source = (position + 8u - shift) % 8u;
    let digit = select(digits[source], digits[7u - source], mirrored);
    code += digit * power;
    power *= 3u;
  }
  return code;
}

// Reduces one line of sight, given zenith/nadir heights and step counts, to a digit in {-1, 0, 1}.
fn getPatternDigit(
  zenithHeight: f32, zenithCount: i32, nadirHeight: f32, nadirCount: i32,
  tangent: f32, flatDistance: f32, step: f32
) -> i32 {
  let zenithDistance = f32(zenithCount) * step;
  let nadirDistance = f32(nadirCount) * step;
  let zenithReference = select(zenithDistance, flatDistance, flatDistance > 0.0 && flatDistance < zenithDistance);
  let nadirReference = select(nadirDistance, flatDistance, flatDistance > 0.0 && flatDistance < nadirDistance);
  let zenithAbsolute = abs(zenithHeight);
  let nadirAbsolute = abs(nadirHeight);
  let zenithOver = zenithAbsolute > tangent * zenithReference;
  let nadirOver = nadirAbsolute > tangent * nadirReference;
  // Absolute angles compare as |height| / count, cross-multiplied to stay division free.
  let zenithScaled = zenithAbsolute * f32(nadirCount);
  let nadirScaled = nadirAbsolute * f32(zenithCount);
  if (COMPARISON == 0u) {
    if (!(zenithOver || nadirOver)) { return 0; }
    if (nadirScaled < zenithScaled) { return 1; }
    if (nadirScaled > zenithScaled) { return -1; }
    return 0;
  }
  if (!zenithOver && !nadirOver) { return 0; }
  if (zenithOver && !nadirOver) { return 1; }
  if (nadirOver && !zenithOver) { return -1; }
  if (nadirScaled < zenithScaled) { return 1; }
  if (nadirScaled > zenithScaled) { return -1; }
  if (COMPARISON == 2u) {
    if (nadirCount < zenithCount) { return 1; }
    if (nadirCount > zenithCount) { return -1; }
  }
  return 1;
}`}function P(e){return`
  let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  let flatThreshold = settings[settingsOffset + 5u];
  let flatDistance = settings[settingsOffset + 6u];
  let settingsValid = isFiniteValue(groundCell.x) && isFiniteValue(groundCell.y) &&
    groundCell.x > 0.0 && groundCell.y > 0.0 && isFiniteValue(zFactor) &&
    isFiniteValue(flatThreshold) && flatThreshold > 0.0 && flatThreshold < 90.0 &&
    isFiniteValue(flatDistance) && flatDistance >= 0.0;
  let margin = SKIP_RADIUS + 1;
  let insideBand = row >= margin && row < i32(HEIGHT) - margin &&
    column >= margin && column < i32(WIDTH) - margin;
  let isValid = settingsValid && insideBand && isElevationValid(column, row);
  var outputForm = 0u;
  var outputTernary = 0u;
  var outputPattern = 0u;
  if (isValid) {
    let centerElevation = getElevation(column, row);
    let tangent = tan(flatThreshold * DEGREES_TO_RADIANS);
    // The strict search limit is shrunk by 1e-5 so a sample exactly at the limit is excluded even
    // when f32 cell sizes round (for example cos(0) or equal east-west and north-south sizes).
    let searchDistance = f32(SEARCH_RADIUS) * groundCell.y * 0.99999;
    // GRASS r.geomorphon ignores flat_distance at or inside the skip distance and at or beyond
    // the search distance (review C-13).
    let skipDistance = f32(SKIP_RADIUS) * groundCell.y;
    let flatReference = select(
      0.0,
      flatDistance,
      flatDistance > skipDistance && flatDistance < f32(SEARCH_RADIUS) * groundCell.y
    );
    var digits: array<u32, 8>;
    var minusCount = 0u;
    var plusCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      digits[direction] = 1u;
      let columnStep = DIRECTION_COLUMN[direction];
      let northStep = DIRECTION_NORTH[direction];
      let rowStep = northStep * ROW_NORTH_SIGN;
      if (!isElevationValid(column + columnStep, row + rowStep)) { continue; }
      let eastMetres = f32(columnStep) * groundCell.x;
      let northMetres = f32(northStep) * groundCell.y;
      // Axis directions use the exact cell size so the search limit has no square-root rounding.
      var step = sqrt(eastMetres * eastMetres + northMetres * northMetres);
      if (columnStep == 0) { step = groundCell.y; }
      if (northStep == 0) { step = groundCell.x; }
      var zenithHeight = 0.0;
      var zenithCount = 0;
      var nadirHeight = 0.0;
      var nadirCount = 0;
      for (var count = SKIP_RADIUS + 1; count <= i32(max(WIDTH, HEIGHT)); count++) {
        if (!(f32(count) * step < searchDistance)) { break; }
        let sampleColumn = column + count * columnStep;
        let sampleRow = row + count * rowStep;
        if (sampleColumn < 0 || sampleColumn >= i32(WIDTH) || sampleRow < 0 || sampleRow >= i32(HEIGHT)) { break; }
        let sampleElevation = getElevation(sampleColumn, sampleRow);
        if (!isFiniteValue(sampleElevation)) { continue; }
        let height = zFactor * (sampleElevation - centerElevation);
        if (zenithCount == 0 || height * f32(zenithCount) > zenithHeight * f32(count)) {
          zenithHeight = height;
          zenithCount = count;
        }
        if (nadirCount == 0 || height * f32(nadirCount) < nadirHeight * f32(count)) {
          nadirHeight = height;
          nadirCount = count;
        }
      }
      if (zenithCount == 0) { continue; }
      let digit = getPatternDigit(
        zenithHeight, zenithCount, nadirHeight, nadirCount, tangent, flatReference, step
      );
      digits[direction] = u32(digit + 1);
      if (digit > 0) { plusCount++; }
      if (digit < 0) { minusCount++; }
    }
    var code = 0u;
    var power = 1u;
    var rotated = 0xffffffffu;
    for (var position = 0u; position < 8u; position++) {
      code += digits[position] * power;
      power *= 3u;
      rotated = min(rotated, encodeRotation(digits, position, false));
      rotated = min(rotated, encodeRotation(digits, position, true));
    }
    outputPattern = code;
    outputTernary = rotated;
    outputForm = FORM_TABLE[minusCount * 9u + plusCount];
  }
  ${[e.output.forms?`formValues[formValuesOffset + index] = outputForm;`:``,e.output.ternary?`ternaryValues[ternaryValuesOffset + index] = outputTernary;`:``,e.output.pattern?`patternValues[patternValuesOffset + index] = outputPattern;`:``,e.output.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``].join(`
  `)}`}var F=[[.45,.47,.52,.35],[.8,.8,.78,1],[.75,.1,.1,1],[.95,.4,.15,1],[.98,.7,.2,1],[.7,.8,.25,1],[.95,.93,.7,1],[.55,.8,.9,1],[.2,.65,.6,1],[.15,.35,.8,1],[.1,.1,.35,1]];function I(e,t){let{columns:n,rows:r,elevation:i}=e;for(let e=0;e<t;e++){let e=i.slice();for(let t=1;t<r-1;t++)for(let r=1;r<n-1;r++){let a=0;for(let i=0;i<9;i++)a+=e[(t+Math.floor(i/3)-1)*n+r+i%3-1];i[t*n+r]=a/9}}}var L=[`edge`,`flat`,`peak`,`ridge`,`shoulder`,`spur`,`slope`,`hollow`,`footslope`,`valley`,`pit`],R=[[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]],z=[`anglev1`,`anglev2`,`anglev2-distance`],B=b({parameters:A,rebuildOn:[`searchRadius`,`comparison`],caption:`Elevation in; one of ten landform classes per cell out.`,createInputs:({world:e})=>{let t=T(e);return I(t,3),{...t,maskState:C()}},build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=D(e,t),s=e.allocate(`parameters`,`float32`,8),c=e.allocate(`forms`,`uint32`,o);return e.graph.add(new M({grid:r,elevation:i,parameters:s.view,searchRadius:n.searchRadius,skipRadius:1,comparison:z[n.comparison],output:{forms:c.view}})),{outputs:{elevation:a,forms:c,highlight:e.allocate(`highlight`,`float32`,o)},update:e=>s.write(j({flatThresholdDegrees:e.flatThresholdDegrees},r))}},interaction:{kind:`probe`,hint:`hover a landform`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let[o,s]=E(t,i,e.x,e.y),c=[],l=0,u=0,d=t.elevation[s*t.columns+o],f=Math.tan(r.flatThresholdDegrees*Math.PI/180);for(let[e,n]of R){let a=-1/0,p=1/0,m=0;for(let i=2;i<=r.searchRadius;i++){let r=o+e*i,c=s+n*i;if(r<0||c<0||r>=t.columns||c>=t.rows)break;let l=(t.elevation[c*t.columns+r]-d)/(i*t.cellSize*Math.hypot(e,n));a=Math.max(a,l),p=Math.min(p,l),m=i}m>0&&(a>f&&a>=-p?l++:-p>f&&u++,c.push({shape:`line`,from:k(t,i,o,s),to:k(t,i,o+e*m,s+n*m)}))}c.push({shape:`dot`,at:k(t,i,o,s)});let p=a(n.forms);if(!p)return{spotlight:null,glyphs:c,readout:`${l} higher, ${u} lower`};let m=p[s*t.columns+o];return S(n.highlight,p,m===0?-1:m,t.maskState),{spotlight:null,glyphs:c,readout:`${L[m]??`form`} · ${l} up ${u} down · ${r.searchRadius} cells`}}},draw:({inputs:e,outputs:t})=>[w(e,t.elevation),O(e,t.forms,{ramp:`categorical`,categories:F,alpha:.9}),x(e,t.highlight)]});export{B as default};