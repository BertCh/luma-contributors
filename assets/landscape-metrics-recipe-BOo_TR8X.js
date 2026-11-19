import{n as e,t}from"./landscape-a3oqafYa.js";import{t as n}from"./construction-snapshot-DdkX9Pc0.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-raster-patch-metrics-CYvJmZ9l.js";import{a,i as o}from"./patch-labels-Bfw_yy99.js";import{t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{d as c,o as l,u}from"./recipe-utils-CICB7qiS.js";import{o as d}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as f}from"./gpu-reduction-tNgOVQ_C.js";import{S as p,T as m,f as h,v as g}from"./gpu-contributor-utils-BIwLQON5.js";import{t as _}from"./define-run-B-brkCe4.js";import{c as v}from"./pick-Bo3t3hjv.js";import{t as y}from"./readout-CAaJ0JAY.js";import{Or as b}from"./entry-preview-D70TdSbK.js";function x(e,t){let{id:n,width:c,height:u,classCount:d,cellSize:f,patches:p,patchClasses:m}=t,h=m.length,g=r(e,`${n}-dense-labels`,`uint32`,c*u),_=r(e,`${n}-pixel-counts`,`uint32`,h),v=r(e,`${n}-perimeters`,`float32`,h),y=r(e,`${n}-patch-like-sides`,`uint32`,h),b=r(e,`${n}-class-cells`,`uint32`,d),x=r(e,`${n}-class-like-sides`,`uint32`,d),S=t.countBackground===!1?r(e,`${n}-perimeter-sides`,`uint32x2`,h):void 0;e.add(l(e,{id:`${n}-dense-labels`,operation:n,invocationCount:c*u,bindings:[{name:`patches`,view:p,type:`u32`,access:`read`},{name:`labels`,view:g,type:`u32`,access:`read_write`}],body:`let owner=patches[patchesOffset+index];
labels[labelsOffset+index]=select(0u,owner+1u,owner<${h}u);`})),e.add(new i({id:`${n}-raster-patches`,grid:{width:c,height:u,pixelInterpretation:`area`,affine:[f[0],0,0,0,f[1],0]},labels:g,countRasterBorder:t.countBoundary,output:{pixelCounts:_,...S?{}:{perimeters:v}}}).getCommandNodes(e));for(let[r,i]of[[`patch-like-sides`,y],[`class-cells`,b],[`class-like-sides`,x],[`class-adjacency`,t.classAdjacency],...S?[[`perimeter-sides`,S]]:[]])e.add(s(e,{id:`${n}-${r}-clear`,operation:n,view:i,type:`u32`,value:`0u`}));return e.add(l(e,{id:`${n}-raster-side-counts`,operation:n,invocationCount:a(c,u),bindings:[{name:`keys`,view:g,type:`u32`,access:`read`},{name:`classes`,view:m,type:`u32`,access:`read`},{name:`patchLikeSides`,view:y,type:`atomic<u32>`,access:`read_write`},{name:`classCells`,view:b,type:`atomic<u32>`,access:`read_write`},{name:`classLikeSides`,view:x,type:`atomic<u32>`,access:`read_write`},{name:`adjacency`,view:t.classAdjacency,type:`atomic<u32>`,access:`read_write`},...S?[{name:`perimeterSides`,view:S,type:`atomic<u32>`,access:`read_write`}]:[]],declarations:`const WIDTH:u32=${c}u;
fn getClass(key:u32)->u32 {
  if(key==0u){return 0xffffffffu;}
  let category=classes[classesOffset+key-1u];
  return select(category,0xffffffffu,category==${t.backgroundClass??4294967295}u);
}`,body:o(c,{declareRunState:`var likeSides=0u;var sides=vec2<u32>(0u);`,onRunStart:`likeSides=0u;sides=vec2<u32>(0u);`,onPixel:`let category=getClass(key);
let directions=array<vec2<i32>,4>(vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1));
for(var direction=0u;direction<4u;direction++){
  let adjacent=vec2<i32>(i32(column),i32(row))+directions[direction];
  let exterior=any(adjacent < vec2<i32>(0))||adjacent.x>=${c}||adjacent.y>=${u};
  var neighbor=0u;
  if(!exterior){neighbor=keys[keysOffset+u32(adjacent.y)*WIDTH+u32(adjacent.x)];}
  let neighborClass=getClass(neighbor);
  likeSides+=u32(neighborClass==category);
  if(category<${d}u&&neighborClass<${d}u&&neighborClass!=category){
    atomicAdd(&adjacency[adjacencyOffset+category*${d}u+neighborClass],1u);
  }
  ${S?`let countContact=select(neighborClass<${d}u,${t.countBoundary!==!1},exterior);
  if(neighbor!=key&&countContact){sides[select(1u,0u,direction<2u)]+=1u;}`:``}
}`,onRun:`let owner=runKey-1u;
atomicAdd(&patchLikeSides[patchLikeSidesOffset+owner],likeSides);
let category=getClass(runKey);
if(category<${d}u){
  atomicAdd(&classCells[classCellsOffset+category],runEnd-runStart);
  atomicAdd(&classLikeSides[classLikeSidesOffset+category],likeSides);
  atomicAdd(&adjacency[adjacencyOffset+category*${d}u+category],likeSides);
}
${S?`atomicAdd(&perimeterSides[perimeterSidesOffset+2u*owner],sides.x);
atomicAdd(&perimeterSides[perimeterSidesOffset+2u*owner+1u],sides.y);`:``}`})})),e.add(l(e,{id:`${n}-patch-summary`,operation:n,invocationCount:h,bindings:[{name:`pixelCounts`,view:_,type:`u32`,access:`read`},{name:`patchLikeSides`,view:y,type:`u32`,access:`read`},...S?[{name:`perimeterSides`,view:S,type:`u32`,access:`read`}]:[{name:`perimeters`,view:v,type:`f32`,access:`read`}],{name:`output`,view:t.patchMetrics,type:`f32`,access:`read_write`}],body:`let cells=pixelCounts[pixelCountsOffset+index];
let perimeter=${S?`f32(perimeterSides[perimeterSidesOffset+2u*index])*${f[1]}+f32(perimeterSides[perimeterSidesOffset+2u*index+1u])*${f[0]}`:`perimeters[perimetersOffset+index]`};
output[outputOffset+4u*index]=f32(cells)*${f[0]*f[1]};
output[outputOffset+4u*index+1u]=perimeter;
output[outputOffset+4u*index+2u]=f32(cells);
output[outputOffset+4u*index+3u]=f32(patchLikeSides[patchLikeSidesOffset+index]);`})),{classCells:b,classLikeSides:x}}var S=1073741823;function C(t,i){i=n(i,[`id`,`patches`,`patchClasses`,`width`,`height`,`classCount`,`cellSize`,`connectivity`,`reconcilePatches`,`countBoundary`,`countBackground`,`backgroundClass`,`contagionWeighting`,`output`]);let a=i.id??`landscape-metrics`,{width:o,height:s,classCount:_,cellSize:v}=i,y=o*s,b=i.patchClasses.length;m(a,Number.isSafeInteger(o)&&o>0&&Number.isSafeInteger(s)&&s>0,`width and height must be positive integers`),m(a,i.patches.length===y,`patches must have width times height rows`),m(a,b>0,`patchClasses must have at least 1 row`),h(a,`classCount`,_,1,256),m(a,v.every(e=>Number.isFinite(Math.fround(e))&&e>0),`cellSize entries must be positive finite float32 numbers`),m(a,i.connectivity===4||i.connectivity===8,`connectivity must be 4 or 8`),p(a,`width times height`,y,S),g(t.device,a,`raster labels`,y*4,`reduce the raster size`),g(t.device,a,`patch metrics`,b*16,`reduce the patch count`);let C=n(i.output??{},[`patchMetrics`,`classMetrics`,`classAdjacency`,`aggregation`,`landscape`,`landscapeAggregation`,`landscapeCohesion`,`reconciledPatches`,`patchParents`]);c(t,a,[i.patches,i.patchClasses],Object.values(C??{}));for(let[e,n,r]of[[`patchMetrics`,`float32x4`,b],[`classMetrics`,`float32x4`,_],[`classAdjacency`,`uint32`,_*_],[`aggregation`,`float32`,_],[`landscape`,`float32x4`,1],[`landscapeAggregation`,`float32`,1],[`landscapeCohesion`,`float32`,1],[`reconciledPatches`,`uint32`,y],[`patchParents`,`uint32`,b]]){let i=C[e];i&&u(t,`${a} output.${e}`,n,r,i)}let w=i.reconcilePatches?e(t,{...i,id:a,output:{patches:C?.reconciledPatches,parents:C?.patchParents}}):void 0,T=w?.patches??i.patches,E=u(t,`${a}-patches`,`float32x4`,b,C?.patchMetrics),D=u(t,`${a}-classes`,`float32x4`,_,C?.classMetrics),O=u(t,`${a}-adjacency`,`uint32`,_*_,C?.classAdjacency),k=u(t,`${a}-aggregation`,`float32`,_,C?.aggregation),A=u(t,`${a}-landscape`,`float32x4`,1,C?.landscape),j=u(t,`${a}-landscape-aggregation`,`float32`,1,C?.landscapeAggregation),M=u(t,`${a}-landscape-cohesion`,`float32`,1,C?.landscapeCohesion),{classCells:N,classLikeSides:P}=x(t,{...i,id:a,patches:T,patchMetrics:E,classAdjacency:O}),F=r(t,`${a}-valid-cell-count`,`uint32`,1);t.add(new f({id:`${a}-valid-cell-count`,input:N,output:F,operation:`sum`}).getCommandNodes(t));let I=r(t,`${a}-class-keys`,`uint32`,b),L=[`perimeter`,`squared-area`,`weighted-perimeter`].map(e=>({name:e,contributions:r(t,`${a}-class-${e}-terms`,`float32`,b),output:r(t,`${a}-class-${e}-sums`,`float32`,_)}));t.add(l(t,{id:`${a}-class-terms`,operation:a,invocationCount:b,bindings:[{name:`patches`,view:E,type:`f32`,access:`read`},{name:`classes`,view:i.patchClasses,type:`u32`,access:`read`},{name:`keys`,view:I,type:`u32`,access:`read_write`},...L.map((e,t)=>({name:`terms${t}`,view:e.contributions,type:`f32`,access:`read_write`}))],body:`let category=classes[classesOffset+index];
keys[keysOffset+index]=select(category,0xffffffffu,category==${i.backgroundClass??4294967295}u);
let area=patches[patchesOffset+4u*index];
let perimeter=patches[patchesOffset+4u*index+1u];
let cells=patches[patchesOffset+4u*index+2u];
terms0[terms0Offset+index]=perimeter;
terms1[terms1Offset+index]=area*area;
terms2[terms2Offset+index]=perimeter*sqrt(cells);`})),t.add(d(t,{id:`${a}-class-sums`,operation:a,segmentCount:_,segmentKeys:I,reductions:L})),t.add(l(t,{id:`${a}-class-summary`,operation:a,invocationCount:_,declarations:`
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,bindings:[{name:`classCells`,view:N,type:`u32`,access:`read`},{name:`classLikeSides`,view:P,type:`u32`,access:`read`},{name:`totalCellCount`,view:F,type:`u32`,access:`read`},...L.map((e,t)=>({name:`sums${t}`,view:e.output,type:`f32`,access:`read`})),{name:`output`,view:D,type:`f32`,access:`read_write`},{name:`aggregation`,view:k,type:`f32`,access:`read_write`}],body:`let cells=classCells[classCellsOffset+index];
let likeSides=classLikeSides[classLikeSidesOffset+index];
let totalCells=totalCellCount[totalCellCountOffset];
let area=f32(cells)*${v[0]*v[1]};
let perimeter=sums0[sums0Offset+index];
let squaredArea=sums1[sums1Offset+index];
let weightedPerimeter=sums2[sums2Offset+index];
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));var cohesion=missing;var mesh=missing;
if(totalCells>0u){mesh=squaredArea/(f32(totalCells)*${v[0]*v[1]});}
if(weightedPerimeter>0.0&&totalCells>1u){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(f32(totalCells)));}
var side=min(u32(sqrt(f32(cells))),65535u);
while(side*side>cells){side-=1u;}
while(side<65535u&&(side+1u)*(side+1u)<=cells){side+=1u;}
let remainder=cells-side*side;
var shortfall=0u;if(remainder>0u){shortfall=select(2u,1u,remainder<=side);}
let maximum=2.0*f32(cells-side)-f32(shortfall);
var aggregate=missing;if(maximum>0.0){aggregate=100.0*f32(likeSides)/(2.0*maximum);}
writeOutput(index,vec4<f32>(area,perimeter,mesh,cohesion));aggregation[aggregationOffset+index]=aggregate;`})),t.add(l(t,{id:`${a}-landscape-summary`,operation:a,invocationCount:1,declarations:`
fn readClasses(row:u32)->vec4<f32>{return vec4<f32>(classes[classesOffset+row*4u+0u],classes[classesOffset+row*4u+1u],classes[classesOffset+row*4u+2u],classes[classesOffset+row*4u+3u]);}
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,bindings:[{name:`classes`,view:D,type:`f32`,access:`read`},{name:`adjacency`,view:O,type:`u32`,access:`read`},{name:`output`,view:A,type:`f32`,access:`read_write`}],body:`var area=0.0;var mesh=0.0;var present=0.0;
for(var category=0u;category<${_}u;category++){let values=readClasses(category);area+=values.x;mesh+=values.z;present+=f32(values.x>0.0);}
var entropy=0.0;var adjacencyComplete=true;var totalAdjacency=0.0;for(var pair=0u;pair<${_*_}u;pair++){totalAdjacency+=f32(adjacency[adjacencyOffset+pair]);}
for(var sourceClass=0u;sourceClass<${_}u;sourceClass++){
 var total=0.0;for(var targetClass=0u;targetClass<${_}u;targetClass++){total+=f32(adjacency[adjacencyOffset+sourceClass*${_}u+targetClass]);}
 let proportion=readClasses(sourceClass).x/area;
 if(proportion>0.0&&total==0.0){adjacencyComplete=false;}
 for(var targetClass=0u;targetClass<${_}u;targetClass++){let probability=${i.contagionWeighting===`adjacency`?`f32(adjacency[adjacencyOffset+sourceClass*`+_+`u+targetClass])/totalAdjacency`:`proportion*f32(adjacency[adjacencyOffset+sourceClass*`+_+`u+targetClass])/total`};if(probability>0.0){entropy+=probability*log(probability);}}
}
var contagion=bitcast<f32>(0x7fc00000u | (index & 0u));if(present>1.0&&adjacencyComplete){contagion=100.0*(1.0+entropy/(2.0*log(present)));}
writeOutput(0u,vec4<f32>(contagion,mesh,area,present));`}));let R=r(t,`${a}-cell-terms`,`float32`,b),z=r(t,`${a}-perimeter-terms`,`float32`,b),B=r(t,`${a}-weighted-terms`,`float32`,b),V=r(t,`${a}-cell-total`,`float32`,1),H=r(t,`${a}-perimeter-total`,`float32`,1),U=r(t,`${a}-weighted-total`,`float32`,1);t.add(l(t,{id:`${a}-landscape-patch-terms`,operation:a,invocationCount:b,bindings:[{name:`patches`,view:E,type:`f32`,access:`read`},{name:`patchClasses`,view:i.patchClasses,type:`u32`,access:`read`},{name:`cellTerms`,view:R,type:`f32`,access:`read_write`},{name:`perimeterTerms`,view:z,type:`f32`,access:`read_write`},{name:`weightedTerms`,view:B,type:`f32`,access:`read_write`}],body:`let category=patchClasses[patchClassesOffset+index];
let counted=category<${_}u&&category!=${i.backgroundClass??4294967295}u;
let patchCells=patches[patchesOffset+4u*index+2u];
let patchPerimeter=patches[patchesOffset+4u*index+1u];
cellTerms[cellTermsOffset+index]=select(0.0,patchCells,counted);
perimeterTerms[perimeterTermsOffset+index]=select(0.0,patchPerimeter,counted);
weightedTerms[weightedTermsOffset+index]=select(0.0,patchPerimeter*sqrt(patchCells),counted);`}));for(let[e,n,r]of[[`cells`,R,V],[`perimeter`,z,H],[`weighted-perimeter`,B,U]])t.add(new f({id:`${a}-landscape-${e}-sum`,input:n,output:r,operation:`sum`}).getCommandNodes(t));return t.add(l(t,{id:`${a}-landscape-aggregation-cohesion`,operation:a,invocationCount:1,bindings:[{name:`cellTotal`,view:V,type:`f32`,access:`read`},{name:`perimeterTotal`,view:H,type:`f32`,access:`read`},{name:`weightedTotal`,view:U,type:`f32`,access:`read`},{name:`classes`,view:D,type:`f32`,access:`read`},{name:`aggregation`,view:k,type:`f32`,access:`read`},{name:`landscapeAggregation`,view:j,type:`f32`,access:`read_write`},{name:`landscapeCohesion`,view:M,type:`f32`,access:`read_write`}],body:`let cells=cellTotal[cellTotalOffset];
let perimeter=perimeterTotal[perimeterTotalOffset];
let weightedPerimeter=weightedTotal[weightedTotalOffset];
var weightedAggregation=0.0;
var area=0.0;
for(var category=0u;category<${_}u;category++){
  let classArea=classes[classesOffset+4u*category];
  let aggregate=aggregation[aggregationOffset+category];
  area+=classArea;
  // The pinned landscapemetrics landscape convention omits undefined single-cell class indices.
  if(classArea>0.0&&aggregate>=0.0){weightedAggregation+=classArea*aggregate;}
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
var aggregate=missing;
var cohesion=missing;
if(area>0.0){aggregate=weightedAggregation/area;}
if(cells>1.0&&weightedPerimeter>0.0){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(cells));}
landscapeAggregation[landscapeAggregationOffset]=aggregate;
landscapeCohesion[landscapeCohesionOffset]=cohesion;`})),{outputs:{patchMetrics:E,classMetrics:D,classAdjacency:O,aggregation:k,landscape:A,landscapeAggregation:j,landscapeCohesion:M},provenance:{patches:i.patches,reconciledPatches:w?.patches,patchParents:w?.parents,patchClasses:i.patchClasses,connectivity:i.connectivity,adjacency:4,cellSize:v,countBoundary:i.countBoundary!==!1,countBackground:i.countBackground!==!1,backgroundClass:i.backgroundClass,contagionWeighting:i.contagionWeighting??`area`}}}var w=64,T=36,E=_({parameters:b,caption:`Class adjacency and aggregation of the Hyde Park WorldCover landscape.`,createInputs:()=>({focus:{buffer:null,className:-1}}),build(e,n,r){let i=t(w,T,r.grain,r.classCount),a=C(e.graph,{patches:e.upload(`patches`,`uint32`,i.patches).view,patchClasses:e.upload(`patch-classes`,`uint32`,i.patchClasses).view,width:w,height:T,classCount:r.classCount,cellSize:[i.cellSizeMeters,i.cellSizeMeters],connectivity:4});return{outputs:{classValues:i.classes,highlight:e.allocate(`highlight`,`uint32`,i.classes.length),classes:e.upload(`classes`,`uint32`,i.classes),adjacency:y(e,`adjacency`,a.outputs.classAdjacency),aggregation:y(e,`aggregation`,a.outputs.aggregation)},update:()=>void 0}},rebuildOn:[`classCount`,`grain`],interaction:{kind:`probe`,hint:`hover a class`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let a=v(w,T,[0,0,r.width,r.height],e.x,e.y),o=n.classValues[a.index];(t.focus.buffer!==n.highlight||t.focus.className!==o)&&(t.focus={buffer:n.highlight,className:o},n.highlight.write(Uint32Array.from(n.classValues,e=>e===o?e:4294967295)));let s=i(n.aggregation);return{glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`class ${o} · aggregation ${s?s[o].toFixed(2):`...`}`}}},draw:({outputs:e})=>[{layer:`raster`,values:e.classes,columns:w,rows:T,alpha:.4},{layer:`raster`,values:e.highlight,columns:w,rows:T},{layer:`chart`,kind:`bars`,values:e.adjacency,ramp:`sequential`,range:[0,60],rectangle:[.04,.05,.48,.34]},{layer:`chart`,kind:`bars`,values:e.aggregation,ramp:`sequential`,rectangle:[.56,.05,.96,.34]}]});export{E as default};