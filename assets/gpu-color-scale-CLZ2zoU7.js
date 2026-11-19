import{t as e}from"./classification-kernel-node-C62eCEbJ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{E as n,S as r,c as i,f as a,p as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./column-classification-shared-BTIVlxdJ.js";import{t as l}from"./color-scale-wgsl-CvXrUz2H.js";import{a as u,i as d,n as f,r as p,t as m}from"./column-classification-rows-D2hiB8L9.js";var h={linear:0,sqrt:1,pow:2,log:3,symlog:4,quantize:5,threshold:6,quantile:7,ordinal:8};function g(e,t=new Float32Array(9)){if(t.length<9)throw Error(`GPUColorScale parameter target must hold 9 elements`);if(!(e.scale in h))throw Error(`GPUColorScale unknown scale '${String(e.scale)}'`);for(let[t,n]of[[`domainCount`,e.domainCount],[`paletteCount`,e.paletteCount]])if(!Number.isInteger(n)||n<0)throw Error(`GPUColorScale ${t} must be a non-negative integer`);let n=(e.noDataColor??0)>>>0;return t[0]=e.scale===`log`&&e.logMode===`domain-sign`?9:h[e.scale],t[1]=e.domainCount,t[2]=e.paletteCount,t[3]=+(e.interpolation===`linear`),t[4]=e.clamp?1:e.extrapolate?2:0,t[5]=n&65535,t[6]=n>>>16,t[7]=e.logFloor??1e-5,t[8]=e.exponent??1,t}function _(e,t,n,r=255){return(e&255|(t&255)<<8|(n&255)<<16|(r&255)<<24)>>>0}var v=`GPUColorScale`,y=2**32-1,b=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`color-scale`,this.props=e;let t=this.id,{output:c,maximumDomainCount:l,maximumPaletteCount:u}=e;if(a(t,`maximumDomainCount`,l),a(t,`maximumPaletteCount`,u),!c.colors&&!c.classIndices&&!c.classCounts)throw Error(`${t} needs at least one output (colors, classIndices, or classCounts)`);let m=e.values.length;if(c.classCounts&&r(t,`values length with output.classCounts`,m,y),m<1)throw Error(`${t} needs at least one row`);if(d(t,`values`,e.values,[`float32`,`uint32`,`sint32`]),e.integerValues!==void 0&&![`ordinal`,`numeric`].includes(e.integerValues))throw Error(`${t} integerValues must be 'ordinal' or 'numeric'`);if(n(e.domain,[`float32`],`${t} domain`),n(e.palette,[`uint32`],`${t} palette`),n(e.parameters,[`float32`],`${t} parameters`),e.mask&&(d(t,`mask`,e.mask,[`uint32`],e.values),e.mask.length!==m))throw Error(`${t} mask length must equal values length`);if(e.domain.length<l)throw Error(`${t} domain must hold maximumDomainCount rows`);if(e.palette.length<u)throw Error(`${t} palette must hold maximumPaletteCount rows`);if(e.parameters.length<9)throw Error(`${t} parameters must hold 9 float32 values`);if(e.domainCount&&(n(e.domainCount,[`uint32`],`${t} domainCount`),e.domainCount.length<1))throw Error(`${t} domainCount must contain one uint32 row`);for(let n of[`colors`,`classIndices`]){let r=c[n];if(r&&(f(t,`output.${n}`,r,`uint32`,e.values),r.length<m))throw Error(`${t} output.${n} must hold one row per value`)}if(c.classCounts&&(n(c.classCounts,[`uint32`],`${t} output.classCounts`),c.classCounts.length<u))throw Error(`${t} output.classCounts must hold maximumPaletteCount rows`);let h=[c.colors,c.classIndices,c.classCounts];p(t,h),s(t,h,[e.values,e.mask,e.domain,e.domainCount,e.palette,e.parameters]),this.estimatedCommandNodeCount=i(e.values).filter(e=>e.length>0).length*(1+Number(!!(c.colors||c.classCounts)))+Number(!!c.classCounts),o(t,this.estimatedCommandNodeCount)}getCommandNodes(n){let{props:r,id:a}=this,{output:o,maximumDomainCount:s,maximumPaletteCount:d}=r;m(a,n,{values:r.values,mask:r.mask,domain:r.domain,domainCount:r.domainCount,palette:r.palette,parameters:r.parameters,"output.colors":o.colors,"output.classIndices":o.classIndices,"output.classCounts":o.classCounts});let f=i(r.values).reduce((e,t)=>Math.max(e,t.length),0);u(n,a,{"class scratch":f,positions:f});let p=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),h=`const MAXIMUM_DOMAIN_COUNT: u32 = ${s}u;
const MAXIMUM_PALETTE_COUNT: u32 = ${d}u;
${c}
${l}`,g=[];o.classCounts&&g.push(e(n,{id:`${a}-init`,operation:v,variant:`init`,bindings:[{name:`counts`,view:o.classCounts,type:`u32`,access:`read_write`}],invocationCount:d,body:`counts[countsOffset + index] = 0u;`}));let _=i(this.props.values);for(let r=0;r<_.length;r++){if(_[r].length===0)continue;let o={...this.props,values:i(this.props.values)[r],mask:this.props.mask?i(this.props.mask)[r]:void 0,output:{...this.props.output,classIndices:this.props.output.classIndices?i(this.props.output.classIndices)[r]:void 0,colors:this.props.output.colors?i(this.props.output.colors)[r]:void 0}},{output:s}=o,c=_.length===1?a:`${a}-chunk-${r}`,l=o.values.length,u=o.values.format===`uint32`&&o.integerValues!==`numeric`,d=o.values.format===`float32`?void 0:o.values.format===`uint32`?`u32`:`i32`,f=s.classIndices??t(n,`${c}-classes`,`uint32`,l),m=t(n,`${c}-positions`,`float32`,l);g.push(e(n,{id:`${c}-classify`,operation:v,variant:u?`classify-ordinal`:d?`classify-integer`:`classify`,bindings:[p(`values`,o.values,d??`f32`),...o.mask?[p(`rowMask`,o.mask,`u32`)]:[],...u?[]:[p(`domain`,o.domain,`f32`)],...!u&&o.domainCount?[p(`domainCountIn`,o.domainCount,`u32`)]:[],p(`params`,o.parameters,`f32`),{name:`classes`,view:f,type:`u32`,access:`read_write`},{name:`positions`,view:m,type:`f32`,access:`read_write`}],invocationCount:l,declarations:h,body:`classes[classesOffset + index] = NO_CLASS;
  positions[positionsOffset + index] = 0.0;
  ${o.mask?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  if (paletteCount == 0u) {
    return;
  }
    ${u?x:S(o,d!==void 0)}`})),(s.colors||s.classCounts)&&g.push(e(n,{id:`${c}-colorize`,operation:v,variant:`colorize`,bindings:[p(`classes`,f,`u32`),...s.colors?[p(`positions`,m,`f32`),p(`palette`,o.palette,`u32`)]:[],p(`params`,o.parameters,`f32`),...s.colors?[{name:`colors`,view:s.colors,type:`u32`,access:`read_write`}]:[],...s.classCounts?[{name:`counts`,view:s.classCounts,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:`const IS_ORDINAL: bool = ${u};
${h}`,body:`let classIndex = classes[classesOffset + index];
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  let isValid = classIndex != NO_CLASS && classIndex < paletteCount;
  ${s.classCounts?`if (isValid) {
    atomicAdd(&counts[countsOffset + classIndex], 1u);
  }`:``}
  ${s.colors?`let noDataColor = u32(params[paramsOffset + 5u]) | (u32(params[paramsOffset + 6u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classIndex];
    let scale = u32(max(params[paramsOffset], 0.0));
    if (!IS_ORDINAL && (scale <= 4u || scale == 9u) && params[paramsOffset + 3u] == 1.0) {
      let position = positions[positionsOffset + index];
      let lower = u32(clamp(floor(position), 0.0, f32(max(paletteCount, 2u) - 2u)));
      let upper = min(lower + 1u, paletteCount - 1u);
      color = blendColors(
        palette[paletteOffset + lower],
        palette[paletteOffset + upper],
        position - f32(lower)
      );
    }
  }
  colors[colorsOffset + index] = color;`:``}`}))}return g}},x=`let code = values[valuesOffset + index];
  if (code < paletteCount) {
    classes[classesOffset + index] = code;
  }`;function S(e,t){return`var domainCount = u32(max(params[paramsOffset + 1u], 0.0));
  ${e.domainCount?`let classTotal = domainCountIn[domainCountInOffset];
  domainCount = select(0u, min(classTotal, MAXIMUM_DOMAIN_COUNT) + 1u, classTotal > 0u);`:``}
  domainCount = min(domainCount, min(MAXIMUM_DOMAIN_COUNT, arrayLength(&domain) - domainOffset));
  if (domainCount == 0u) {
    return;
  }
  let value = ${t?`f32(values[valuesOffset + index])`:`values[valuesOffset + index]`};
  if (isNanBits(value)) {
    return;
  }
  let scale = u32(max(params[paramsOffset], 0.0));
  let isClamped = params[paramsOffset + 4u] == 1.0;
  let isExtrapolating = params[paramsOffset + 4u] == 2.0;
  let logFloor = params[paramsOffset + 7u];
  let exponent = params[paramsOffset + 8u];
  var classIndex = NO_CLASS;
  var rampPosition = 0.0;
  if (scale == 6u || scale == 7u) {
    // Threshold and quantile: count the inner edges domain[1 .. domainCount - 2] that are <= value.
    if (domainCount < 2u) {
      return;
    }
    let valueKey = getComparisonKey(value);
    var low = 1u;
    var high = domainCount - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getComparisonKey(domain[domainOffset + middle]) <= valueKey) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    classIndex = min(low - 1u, paletteCount - 1u);
  } else if (scale == 5u) {
    if (domainCount < 2u) {
      return;
    }
    let lowEdge = domain[domainOffset];
    let span = domain[domainOffset + domainCount - 1u] - lowEdge;
    var t = select(0.5, (value - lowEdge) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    t = clamp(t, 0.0, 1.0);
    classIndex = min(u32(floor(t * f32(paletteCount))), paletteCount - 1u);
  } else if (scale <= 4u || scale == 9u) {
    if (domainCount < 2u) {
      return;
    }
    let negativeLog = domain[domainOffset] < 0.0;
    let firstStop = transformColorValue(domain[domainOffset], scale, exponent, logFloor, negativeLog);
    let lastStop = transformColorValue(domain[domainOffset + domainCount - 1u], scale, exponent, logFloor, negativeLog);
    let transformed = transformColorValue(value, scale, exponent, logFloor, negativeLog);
    if (isNanBits(firstStop) || isNanBits(lastStop)) { return; }
    // Native finite constant domains normalize to the midpoint, including zero/wrong-sign log inputs.
    let t = normalizeColorValue(transformed, firstStop, lastStop);
    if (isNanBits(t) || (!isClamped && !isExtrapolating && (t < 0.0 || t > 1.0))) {
      return;
    }
    if ((isExtrapolating || scale == 9u) &&
        ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u ||
         (bitcast<u32>(t) & 0x7f800000u) == 0x7f800000u)) { return; }
    if (domainCount > 2u && paletteCount == domainCount) {
      if (!isExtrapolating && t <= 0.0) {
        rampPosition = 0.0;
      } else if (!isExtrapolating && t >= 1.0) {
        rampPosition = f32(paletteCount - 1u);
      } else {
        // Select the first/last segment outside the domain, including descending domains.
        var segment = 0u;
        let descending = lastStop < firstStop;
        for (var stopIndex = 1u; stopIndex + 1u < domainCount; stopIndex++) {
          let stop = transformColorValue(domain[domainOffset + stopIndex], scale, exponent, logFloor, negativeLog);
          if (select(stop <= transformed, stop >= transformed, descending)) {
            segment = stopIndex;
          }
        }
        let segmentStart = transformColorValue(domain[domainOffset + segment], scale, exponent, logFloor, negativeLog);
        let segmentEnd = transformColorValue(domain[domainOffset + segment + 1u], scale, exponent, logFloor, negativeLog);
        var fraction = normalizeColorValue(transformed, segmentStart, segmentEnd);
        if (isNanBits(fraction)) { return; }
        if (!isExtrapolating) {
          fraction = clamp(fraction, 0.0, 1.0);
          // Preserve the legacy duplicate-stop choice unless native log was explicitly requested.
          if (segmentStart == segmentEnd && scale != 9u) { fraction = 0.0; }
        }
        rampPosition = f32(segment) + fraction;
      }
      classIndex = u32(clamp(floor(rampPosition), 0.0, f32(paletteCount - 1u)));
    } else {
      let clampedT = clamp(t, 0.0, 1.0);
      rampPosition = select(clampedT, t, isExtrapolating) * f32(paletteCount - 1u);
      classIndex = min(u32(floor(clampedT * f32(paletteCount))), paletteCount - 1u);
    }
  } else {
    return;
  }
  if ((isExtrapolating || scale == 9u) &&
      (bitcast<u32>(rampPosition) & 0x7f800000u) == 0x7f800000u) { return; }
  classes[classesOffset + index] = classIndex;
  positions[positionsOffset + index] = rampPosition;`}export{g as n,_ as r,b as t};