import{t as e}from"./classification-kernel-node-C62eCEbJ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{E as n,S as r,c as i,f as a,p as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./column-classification-shared-BTIVlxdJ.js";import{t as l}from"./color-scale-wgsl-CvXrUz2H.js";import{a as u,i as d,n as f,r as p,t as m}from"./column-classification-rows-D2hiB8L9.js";import{t as h}from"./define-run-B-brkCe4.js";import{t as g}from"./lightning-C0gvbE9a.js";import{n as _,t as v}from"./pick-Bo3t3hjv.js";import{n as y}from"./convert-CaKqeDF0.js";import{p as b,t as x}from"./helpers-DPAd8TjN.js";import{Jc as S}from"./entry-preview-D70TdSbK.js";function C(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUBivariateClassification parameter target must hold 8 elements`);for(let[t,n]of[[`classCountX`,e.classCountX],[`classCountY`,e.classCountY]])if(!Number.isInteger(n)||n<0)throw Error(`GPUBivariateClassification ${t} must be a non-negative integer`);let n=(e.noDataColor??0)>>>0,r=e.valueByAlpha;return t[0]=e.classCountX,t[1]=e.classCountY,t[2]=n&65535,t[3]=n>>>16,t[4]=+!!r,t[5]=r?r.domain[0]:0,t[6]=r?r.domain[1]:1,t[7]=r?r.minimumAlpha:1,t}var w=`GPUBivariateClassification`,T=2**32-1,E=16,D=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`bivariate-classification`,this.props=e;let t=this.id,{output:c,maximumClassCount:l}=e;if(a(t,`maximumClassCount`,l,1,E),!c.classIds&&!c.colors&&!c.classCounts)throw Error(`${t} needs at least one output (classIds, colors, or classCounts)`);let u=e.valuesX.length;if(c.classCounts&&r(t,`values length with output.classCounts`,u,T),u<1)throw Error(`${t} needs at least one row`);d(t,`valuesX`,e.valuesX,[`float32`]),d(t,`valuesY`,e.valuesY,[`float32`],e.valuesX),n(e.breaksX,[`float32`],`${t} breaksX`),n(e.breaksY,[`float32`],`${t} breaksY`),n(e.palette,[`uint32`],`${t} palette`),n(e.parameters,[`float32`],`${t} parameters`),e.alphaValues&&d(t,`alphaValues`,e.alphaValues,[`float32`],e.valuesX),e.mask&&d(t,`mask`,e.mask,[`uint32`],e.valuesX);for(let[n,r]of[[`valuesY`,e.valuesY],[`mask`,e.mask],[`alphaValues`,e.alphaValues]])if(r&&r.length!==u)throw Error(`${t} ${n} length must equal valuesX length`);for(let[n,r]of[[`breaksX`,e.breaksX],[`breaksY`,e.breaksY]])if(r.length<l+1)throw Error(`${t} ${n} must hold maximumClassCount + 1 rows`);let m=l*l;if(e.palette.length<m)throw Error(`${t} palette must hold maximumClassCount * maximumClassCount rows`);if(e.parameters.length<8)throw Error(`${t} parameters must hold 8 float32 values`);for(let n of[`classIds`,`colors`]){let r=c[n];if(r&&(f(t,`output.${n}`,r,`uint32`,e.valuesX),r.length<u))throw Error(`${t} output.${n} must hold one row per value`)}if(c.classCounts&&(n(c.classCounts,[`uint32`],`${t} output.classCounts`),c.classCounts.length<m))throw Error(`${t} output.classCounts must hold maximumClassCount ** 2 rows`);let h=[c.classIds,c.colors,c.classCounts];p(t,h),s(t,h,[e.valuesX,e.valuesY,e.mask,e.breaksX,e.breaksY,e.palette,e.alphaValues,e.parameters]),this.estimatedCommandNodeCount=i(e.valuesX).filter(e=>e.length>0).length*(1+Number(!!(c.colors||c.classCounts)))+Number(!!c.classCounts),o(t,this.estimatedCommandNodeCount)}getCommandNodes(n){let{props:r,id:a}=this,{output:o,maximumClassCount:s}=r;m(a,n,{valuesX:r.valuesX,valuesY:r.valuesY,mask:r.mask,breaksX:r.breaksX,breaksY:r.breaksY,palette:r.palette,alphaValues:r.alphaValues,parameters:r.parameters,"output.classIds":o.classIds,"output.colors":o.colors,"output.classCounts":o.classCounts});let d=i(r.valuesX).reduce((e,t)=>Math.max(e,t.length),0);u(n,a,{"class scratch":d});let f=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),p=`const MAXIMUM_CLASS_COUNT: u32 = ${s}u;
${c}
${l}`,h=[];o.classCounts&&h.push(e(n,{id:`${a}-init`,operation:w,variant:`init`,bindings:[{name:`counts`,view:o.classCounts,type:`u32`,access:`read_write`}],invocationCount:s*s,body:`counts[countsOffset + index] = 0u;`}));let g=i(this.props.valuesX);for(let r=0;r<g.length;r++){if(g[r].length===0)continue;let o={...this.props,valuesX:i(this.props.valuesX)[r],valuesY:i(this.props.valuesY)[r],mask:this.props.mask?i(this.props.mask)[r]:void 0,alphaValues:this.props.alphaValues?i(this.props.alphaValues)[r]:void 0,output:{...this.props.output,classIds:this.props.output.classIds?i(this.props.output.classIds)[r]:void 0,colors:this.props.output.colors?i(this.props.output.colors)[r]:void 0}},{output:s}=o,c=g.length===1?a:`${a}-chunk-${r}`,l=o.valuesX.length,u=s.classIds??t(n,`${c}-classes`,`uint32`,l);h.push(e(n,{id:`${c}-classify`,operation:w,variant:`classify`,bindings:[f(`valuesX`,o.valuesX,`f32`),f(`valuesY`,o.valuesY,`f32`),...o.mask?[f(`rowMask`,o.mask,`u32`)]:[],f(`breaksX`,o.breaksX,`f32`),f(`breaksY`,o.breaksY,`f32`),f(`params`,o.parameters,`f32`),{name:`classes`,view:u,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${p}
// Number of inner edges edges[1 .. classCount - 1] that are <= value, as a u32 key comparison.
fn getAxisClass(valueKey: u32, classCount: u32, isX: bool) -> u32 {
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    var edge = 0.0;
    if (isX) {
      edge = breaksX[breaksXOffset + middle];
    } else {
      edge = breaksY[breaksYOffset + middle];
    }
    if (getComparisonKey(edge) <= valueKey) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low - 1u;
}`,body:`classes[classesOffset + index] = NO_CLASS;
  ${o.mask?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
  let valueX = valuesX[valuesXOffset + index];
  let valueY = valuesY[valuesYOffset + index];
  let classCountX = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_CLASS_COUNT);
  let classCountY = min(u32(max(params[paramsOffset + 1u], 0.0)), MAXIMUM_CLASS_COUNT);
  if (isNanBits(valueX) || isNanBits(valueY) || classCountX == 0u || classCountY == 0u) {
    return;
  }
  let classX = getAxisClass(getComparisonKey(valueX), classCountX, true);
  let classY = getAxisClass(getComparisonKey(valueY), classCountY, false);
  classes[classesOffset + index] = classY * classCountX + classX;`})),(s.colors||s.classCounts)&&h.push(e(n,{id:`${c}-colorize`,operation:w,variant:`colorize`,bindings:[f(`classes`,u,`u32`),...s.colors?[f(`palette`,o.palette,`u32`),f(`params`,o.parameters,`f32`)]:[],...o.alphaValues&&s.colors?[f(`alphaValues`,o.alphaValues,`f32`)]:[],...s.colors?[{name:`colors`,view:s.colors,type:`u32`,access:`read_write`}]:[],...s.classCounts?[{name:`counts`,view:s.classCounts,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:p,body:`let classId = classes[classesOffset + index];
  let isValid = classId != NO_CLASS;
  ${s.classCounts?`if (isValid) {
    atomicAdd(&counts[countsOffset + classId], 1u);
  }`:``}
  ${s.colors?`let noDataColor = u32(params[paramsOffset + 2u]) | (u32(params[paramsOffset + 3u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classId];
    ${o.alphaValues?`if (params[paramsOffset + 4u] != 0.0) {
      let alphaLow = params[paramsOffset + 5u];
      let alphaHigh = params[paramsOffset + 6u];
      let minimumAlpha = clamp(params[paramsOffset + 7u], 0.0, 1.0);
      let alphaValue = alphaValues[alphaValuesOffset + index];
      var ramp = 1.0;
      if (alphaHigh > alphaLow) {
        ramp = clamp((alphaValue - alphaLow) / (alphaHigh - alphaLow), 0.0, 1.0);
      }
      if (isNanBits(alphaValue)) {
        ramp = 0.0;
      }
      let factor = minimumAlpha + (1.0 - minimumAlpha) * ramp;
      let alpha = packChannel(unpackChannel(color, 3u) * factor);
      color = (color & 0x00ffffffu) | (alpha << 24u);
    }`:``}
  }
  colors[colorsOffset + index] = color;`:``}`}))}return h}},O=2e3,k=4,A=k**2;function j(e){return Math.min(Math.max(e,0),1)}function M(e,t){let n=new Float32Array(5).fill(2);n[0]=0;for(let r=1;r<t;r++)n[r]=e[Math.floor(r*e.length/t)];return n[t]=1.001,n}function N(e,t){let n=[];for(let r=0;r<t;r++)for(let i=0;i<e;i++){let a=i/Math.max(e-1,1),o=r/Math.max(t-1,1),s=[.9-.8*a,.92-.55*a,.97-.15*a],c=[.98-.05*o,.9-.55*o,.85-.8*o];n.push([s[0]*c[0],s[1]*c[1],s[2]*c[2]])}return n}function P(e,t,n,r,i,a){let o=new Float32Array(128),s=1;for(let e=0;e<r*i;e++)s=Math.max(s,a?.[e]??0);for(let c=0;c<A;c++){let l=c<r*i,u=a?Math.sqrt((a[c]??0)/s):1,d=l?n*.92*(.18+.82*u):0,f=n*.46,p=e+c%r*n+f-d/2,m=t+Math.floor(c/r)*n+f-d/2;o.set([p,m,p+d,m,p+d,m+d,p,m+d],c*8)}return o}var F=h({parameters:S,caption:`Joint classes of log flash energy and footprint area; the key squares at right grow with their row count.`,createInputs:({world:e})=>{let t={width:e.width*.66,height:e.height},n=g(t,O),r=Float32Array.from(n.energy,e=>j(Math.log1p(e)/8)),i=Float32Array.from(n.area,e=>j((Math.log1p(e)-4)/4)),a=Math.min(e.width*.07,.13);return{world:e,mapWorld:t,positions:n.positions,index:_(n.positions),focus:{row:-1},valuesX:r,valuesY:i,sortedX:Float32Array.from(r).sort(),sortedY:Float32Array.from(i).sort(),cellSize:a,matrixX:e.width*.7,matrixY:.5-2*a}},build(e,t){let{cellSize:n,matrixX:r,matrixY:i}=t,a=e.upload(`values-x`,`float32`,t.valuesX),o=e.upload(`values-y`,`float32`,t.valuesY),s=e.upload(`positions`,`float32x2`,t.positions),c=e.upload(`key-positions`,`float32x2`,P(r,i,n,k,k)),l=e.upload(`cell-indices`,`uint32`,Uint32Array.from({length:96},(e,t)=>{let n=[0,1,2,0,2,3][t%6];return Math.floor(t/6)*4+n})),u=e.upload(`cell-owners`,`uint32`,Uint32Array.from({length:32},(e,t)=>Math.floor(t/2))),d=e.upload(`cell-classes`,`float32`,Float32Array.from({length:A},(e,t)=>t)),f=e.allocate(`breaks-x`,`float32`,5),p=e.allocate(`breaks-y`,`float32`,5),m=e.allocate(`palette`,`uint32`,k**2),h=e.allocate(`mask`,`uint32`,O),g=e.allocate(`parameters`,`float32`,8),_=e.allocate(`class-ids`,`uint32`,O),v=e.allocate(`class-counts`,`uint32`,k**2);return e.graph.add(new D({valuesX:a.view,valuesY:o.view,mask:h.view,breaksX:f.view,breaksY:p.view,palette:m.view,parameters:g.view,maximumClassCount:k,output:{classIds:_.view,classCounts:v.view}})),{outputs:{positions:s,jointClass:y(e,`joint-class`,_),classCounts:v,highlight:e.allocate(`highlight`,`float32x2`,O),keyPositions:c,cellIndices:l,cellOwners:u,cellClasses:d},update:e=>{m.write(new Uint32Array(k**2).fill(4294967295)),f.write(M(t.sortedX,e.classCountX)),p.write(M(t.sortedY,e.classCountY)),h.write(x(t.positions,b(t.mapWorld.width,t.mapWorld.height,e.maskShare))),g.write(C({classCountX:e.classCountX,classCountY:e.classCountY}))}}},rebuildOn:[],interaction:{kind:`probe`,hint:`hover a class`,rest:({inputs:e,values:t})=>{let n=Math.min(1,t.classCountX-1),r=Math.min(1,t.classCountY-1),i=e.matrixX+(n+.5)*e.cellSize,a=e.matrixY+(r+.5)*e.cellSize;return[i/e.world.width,1-a/e.world.height]},resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=r.classCountX,o=r.classCountY,s=i(n.classCounts),c=i(n.jointClass);n.keyPositions.write(P(t.matrixX,t.matrixY,t.cellSize,a,o,s));let l=Math.floor((e.x-t.matrixX)/t.cellSize),u=Math.floor((e.y-t.matrixY)/t.cellSize),d=-1;if(l>=0&&l<a&&u>=0&&u<o)d=u*a+l;else if(c){t.focus.row=v(t.index,e.x,e.y,t.focus.row);let n=c[t.focus.row];t.focus.row>=0&&n<A&&(d=n)}if(d<0||!c)return{focus:-1,readout:`hover a flash or the key`};let f=new Float32Array(O*2).fill(-10),p=0;for(let e=0;e<O;e++)c[e]===d&&(f.set([t.positions[e*2],t.positions[e*2+1]],p*2),p++);n.highlight.write(f);let m=d%a,h=Math.floor(d/a),g=t.matrixX+(m+.46)*t.cellSize,_=t.matrixY+(h+.46)*t.cellSize,y=t.cellSize*.5;return{focus:[d,p],spotlight:null,glyphs:[{shape:`box`,minimum:[g-y,_-y],maximum:[g+y,_+y]}],readout:`class (x ${m+1}, y ${h+1}) · ${p} flashes`}}},draw:({outputs:e,values:t,interaction:n})=>{let r=N(t.classCountX,t.classCountY),i=n.focus,a=Array.isArray(i)&&i[0]>=0,o=Array.isArray(i)?i:[-1,0],s=r[o[0]]??[1,1,1];return[{layer:`dots`,positions:e.positions,values:e.jointClass,ramp:`categorical`,categories:r,alpha:a?.3:1,radius:4},...a?[{layer:`dots`,positions:e.highlight,count:Math.max(1,o[1]),color:[s[0],s[1],s[2],1],radius:5.5}]:[],{layer:`polygons`,positions:e.keyPositions,indices:e.cellIndices,owners:e.cellOwners,values:e.cellClasses,triangleCount:32,ramp:`categorical`,categories:r}]}});export{F as default};