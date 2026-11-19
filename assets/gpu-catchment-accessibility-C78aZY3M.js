import{n as e,t}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{a as r}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as i,t as a}from"./catchment-accessibility-validation-BjMQQJgk.js";import{c as o,r as s,u as c}from"./wgsl-common-DfyXiYB0.js";import{t as l}from"./spatial-analysis-quotient-CbSG4rNM.js";import{E as u,T as d,x as f}from"./gpu-contributor-utils-BIwLQON5.js";import{c as p}from"./spatial-weights-CqWysstN.js";import{l as m}from"./spatial-context-DwZEd4EZ.js";var h=`GPUCatchmentAccessibility`,g=class{id;props;facilityCount;demandCount;constructor(e){let t=e.id??`catchment-accessibility`;if(this.id=t,this.props=e,e.method!==`2sfca`&&e.method!==`3sfca`)throw Error(`${t} method must be '2sfca' or '3sfca'`);if(this.facilityCount=p(t,e.facilityWeights,`facilityWeights`),this.demandCount=e.demandWeights?p(t,e.demandWeights,`demandWeights`):this.facilityCount,!e.demandWeights&&e.supply.length!==e.demand.length)throw Error(`${t} needs demandWeights unless supply and demand have equal length`);e.demandWeights?.spatialContext&&e.facilityWeights?.spatialContext&&m(t,e.demandWeights.spatialContext,e.facilityWeights.spatialContext);let{output:n}=e;for(let[r,i]of[[`supply`,e.supply],[`demand`,e.demand],[`output.accessibility`,n.accessibility],[`output.ratios`,n.ratios]])i&&u(i,[`float32`],`${t} ${r}`);if(n.reachableFacilities&&u(n.reachableFacilities,[`uint32`],`${t} output.reachableFacilities`),e.supply.length!==this.facilityCount)throw Error(`${t} supply length must equal the facilityWeights row count`);if(e.demand.length!==this.demandCount)throw Error(`${t} demand length must equal the demandWeights row count`);if(n.accessibility.length!==this.demandCount)throw Error(`${t} output.accessibility length must equal the demand count`);if(n.ratios&&n.ratios.length!==this.facilityCount)throw Error(`${t} output.ratios length must equal the facility count`);if(n.reachableFacilities&&n.reachableFacilities.length!==this.demandCount)throw Error(`${t} output.reachableFacilities length must equal the demand count`);n.validity&&(u(n.validity,[`uint32`],`${t} output.validity`),d(t,n.validity.length===this.demandCount,`output.validity must have one row per demand location`));let r=_(e),a={supply:e.supply,demand:e.demand,...Object.fromEntries(r.map((e,t)=>[`weights.${t}`,e]))},o={...n},s={offsets:e.facilityWeights.offsets,neighbors:e.facilityWeights.neighbors,weights:e.facilityWeights.weights},c=e.demandWeights??e.facilityWeights,l={offsets:c.offsets,neighbors:c.neighbors,weights:c.weights};i(t,a,o,[{reads:{demand:e.demand,supply:e.supply,...s},writes:{ratios:n.ratios}},{reads:{...l,ratios:n.ratios},writes:{accessibility:n.accessibility}},{reads:{offsets:c.offsets,neighbors:c.neighbors},writes:{reachableFacilities:n.reachableFacilities}},{reads:{demand:e.demand},writes:{validity:n.validity}}])}getCommandNodes(i){let{id:u,props:d,facilityCount:p,demandCount:m}=this,{supply:g,demand:y,facilityWeights:b,output:x}=d,{accessibility:S}=x,C=d.demandWeights??b;e(i,u,x),f(u,i,[g,y,S,x.ratios,x.reachableFacilities,x.validity,..._(d)]),r(i,u,{supply:g,demand:y,...x,...Object.fromEntries(_(d).map((e,t)=>[`weights.${t}`,e]))}),x.ratios||a(i.device,u,`ratios scratch`,p*4),d.method===`3sfca`&&a(i.device,u,`selection scratch`,m*4);let w=d.method===`3sfca`,T=x.ratios??n(i,`${u}-ratios`,`float32`,p),E=w?n(i,`${u}-selection`,`float32`,m):void 0,D=[];E&&D.push(t(i,{id:`${u}-selection`,operation:h,variant:`selection`,bindings:[{name:`offsets`,view:C.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:C.neighbors,type:`u32`,access:`read`},{name:`weights`,view:C.weights,type:`f32`,access:`read`},{name:`selection`,view:E,type:`f32`,access:`read_write`}],invocationCount:m,declarations:`const PARTNERS: u32 = ${p}u;`,body:`var total = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      total += weights[weightsOffset + slot];
    }
  }
  selection[selectionOffset + index] = total;`}));let O=E?[{name:`selection`,view:E,type:`f32`,access:`read`}]:[];D.push(t(i,{id:`${u}-ratios`,operation:h,variant:`ratios`,bindings:[{name:`offsets`,view:b.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:b.neighbors,type:`u32`,access:`read`},{name:`weights`,view:b.weights,type:`f32`,access:`read`},{name:`demand`,view:y,type:`f32`,access:`read`},{name:`supply`,view:g,type:`f32`,access:`read`},...O,{name:`ratios`,view:T,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`const PARTNERS: u32 = ${m}u;
${o}
${c}
${l}`,body:`var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and captures nothing.
      if (isFiniteBits(partnerDemand)) {
        ${w?`let total = selection[selectionOffset + partner];
        if (total > 0.0) {
          // Wan, Zou and Sternberg (2012): selection weight G = w / T times the decay w.
          let weight = weights[weightsOffset + slot];
          captured += getSpatialAnalysisQuotient(partnerDemand * weight * weight,total);
        }`:`captured += partnerDemand * weights[weightsOffset + slot];`}
      }
    }
  }
  ratios[ratiosOffset + index] = select(0.0, getSpatialAnalysisQuotient(supply[supplyOffset + index],captured), captured > 0.0);`}),t(i,{id:`${u}-scores`,operation:h,variant:`scores`,bindings:[{name:`offsets`,view:C.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:C.neighbors,type:`u32`,access:`read`},{name:`weights`,view:C.weights,type:`f32`,access:`read`},{name:`ratios`,view:T,type:`f32`,access:`read`},...O,{name:`accessibility`,view:S,type:`f32`,access:`read_write`}],invocationCount:m,declarations:`const PARTNERS: u32 = ${p}u;
${o}
${c}
${l}`,body:`var score = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let ratio = ratios[ratiosOffset + partner];
      // A non-finite ratio (a non-finite supply) is missing and adds nothing.
      if (isFiniteBits(ratio)) {
        let weight = weights[weightsOffset + slot];
        score += ${w?`weight * weight`:`weight`} * ratio;
      }
    }
  }
  ${w?`let total = selection[selectionOffset + index];
  score = select(0.0, getSpatialAnalysisQuotient(score,total), total > 0.0);`:``}
  accessibility[accessibilityOffset + index] = score;`})),x.reachableFacilities&&D.push(t(i,{id:`${u}-reachable`,operation:h,variant:`reachable`,bindings:[{name:`offsets`,view:C.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:C.neighbors,type:`u32`,access:`read`},{name:`reachable`,view:x.reachableFacilities,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`const PARTNERS: u32 = ${p}u;`,body:`var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      count++;
    }
  }
  reachable[reachableOffset + index] = count;`})),x.validity&&D.push(t(i,{id:`${u}-validity`,operation:h,variant:`validity`,bindings:[{name:`demand`,view:y,type:`f32`,access:`read`},{name:`validity`,view:x.validity,type:`u32`,access:`read_write`}],invocationCount:m,declarations:o,body:`validity[validityOffset + index] = u32(isFiniteBits(demand[demandOffset + index]));`}));let k=v(C),A=v(b);for(let[e,n,r,a,o]of[[`accessibility`,x.accessibility,[...k,...A],`f32`,`getNaN()`],[`ratios`,x.ratios,[...A,...w?k:[]],`f32`,`getNaN()`],[`reachableFacilities`,x.reachableFacilities,k,`u32`,`0xffffffffu`]])n&&r.length!==0&&D.push(t(i,{id:`${u}-suppress-${e}`,operation:h,variant:`incomplete-input`,bindings:[...r.map((e,t)=>({name:`loss${t}`,view:e,type:`u32`,access:`read`})),{name:`result`,view:n,type:a,access:`read_write`}],invocationCount:n.length,declarations:a===`f32`?s:void 0,body:`if (${r.map((e,t)=>`loss${t}[loss${t}Offset] != 0u`).join(` || `)}) {
  result[resultOffset + index] = ${o};
}`}));return D.flat()}};function _(e){let t=[];for(let n of[e.facilityWeights,e.demandWeights])n&&t.push(n.offsets,n.neighbors,n.weights,...v(n));return t}function v(e){return[e.status?.overflow,e.status?.candidateOverflow].filter(e=>e!==void 0)}export{g as t};