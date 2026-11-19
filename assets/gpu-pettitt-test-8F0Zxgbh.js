import{r as e,u as t}from"./wgsl-common-DfyXiYB0.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{D as r,E as i,f as a,v as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{p as l}from"./dixie-cube-CZf2ZI2K.js";import{t as u}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{i as d}from"./graph-chunk-dispatch-zBt8k2-o.js";var f=`GPUPettittTest`,p=4294967295,m=class{id;props;constructor(e){let t=e.id??`pettitt-test`;this.id=t,this.props=e;let{values:n,cellCount:o,sliceCount:s,output:l}=e;a(t,`cellCount`,o),a(t,`sliceCount`,s,1,512),i(n,[`float32`],`${t} values`),r(t,`values`,n,o*s),i(l.statistic,[`uint32`],`${t} statistic`),r(t,`statistic`,l.statistic,o),i(l.changeSlice,[`uint32`],`${t} changeSlice`),r(t,`changeSlice`,l.changeSlice,o),i(l.pValues,[`float32`],`${t} pValues`),r(t,`pValues`,l.pValues,o),d(t,l),c(t,[l.statistic,l.changeSlice,l.pValues],[n])}getCommandNodes(r){let{id:i,props:a}=this;u(r,i,a);let{values:c,layout:d,cellCount:m,sliceCount:h,output:g}=a;return s(i,r,[c,g.statistic,g.changeSlice,g.pValues]),o(r.device,i,`values`,m*h*4,`reduce cellCount * sliceCount`),[n(r,{id:`${i}-statistic`,operation:f,bindings:[{name:`values`,view:c,type:`f32`,access:`read`},{name:`statistic`,view:g.statistic,type:`u32`,access:`read_write`},{name:`changeSlice`,view:g.changeSlice,type:`u32`,access:`read_write`},{name:`pValues`,view:g.pValues,type:`f32`,access:`read_write`}],invocationCount:m,declarations:`${t}
${e}
const CELL_COUNT: u32 = ${m}u;
const SLICE_COUNT: u32 = ${h}u;
const NO_CHANGE_SLICE: u32 = ${p}u;
${l(d)}`,body:`var series: array<f32, SLICE_COUNT>;
  var position: array<u32, SLICE_COUNT>;
  var seriesLength = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = readSeriesValue(index, slice);
    if (isFiniteValue(value)) {
      series[seriesLength] = value;
      position[seriesLength] = slice;
      seriesLength++;
    }
  }
  var running = 0;
  var largest = 0;
  var largestRow = 0u;
  if (seriesLength >= 2u) {
    for (var row = 0u; row + 1u < seriesLength; row++) {
      var columnSum = 0;
      for (var other = 0u; other < seriesLength; other++) {
        let difference = series[other] - series[row];
        columnSum += select(select(0, -1, difference < 0.0), 1, difference > 0.0);
      }
      running += columnSum;
      let magnitude = abs(running);
      if (magnitude > largest) {
        largest = magnitude;
        largestRow = row;
      }
    }
  }
  statistic[statisticOffset + index] = u32(largest);
  if (seriesLength >= 2u) {
    let lengthValue = f32(seriesLength);
    let statisticValue = f32(largest);
    changeSlice[changeSliceOffset + index] = position[largestRow + 1u];
    pValues[pValuesOffset + index] = min(1.0, 2.0 * exp(-6.0 * statisticValue * statisticValue / (lengthValue * lengthValue * lengthValue + lengthValue * lengthValue)));
  } else {
    changeSlice[changeSliceOffset + index] = NO_CHANGE_SLICE;
    pValues[pValuesOffset + index] = 1.0;
  }`})]}};export{m as t};