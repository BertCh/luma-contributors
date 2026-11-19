import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{a as n,c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as a,r as o,u as s}from"./wgsl-common-DfyXiYB0.js";import{D as c,E as l,S as u,T as d,f,g as p,p as m,v as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{a as v,g as y,v as b}from"./spatial-context-DwZEd4EZ.js";import{D as x,a as S,i as C,l as w,n as T,o as E,r as D,s as O,t as k}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{n as A}from"./spatial-regression-solve-Vf-sQz9O.js";var j=4096,M=1024,N=`2.384185791015625e-7`,P=4294967294,F=16777216,I=65536,L=34028234663852886e22;function R(e){let t=Math.ceil(e/256),n=Math.ceil(t/256),r=t>256;return{tileCount:t,mergedTileCount:n,finalTileCount:r?n:t,usesParallelMerge:r}}var z=4,B=`${o}\n${a}`,V=`GPUGeneralizedGeographicallyWeightedRegression`;function H(t,a){let{id:o,props:s}=a,c=s.bandwidthSelection,l=c.method===`golden-section`,u=l||c.criterion===`aicc`,d=l?c.maximumIterations??24:1,f=l?2*d:c.bandwidths.length,p=s.response.length,m=s.predictorCount+1,h=R(p),{tileCount:_}=h,v=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read`}),y=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`});g(o,t,[...Object.values(c.output).filter(e=>e!==void 0),...Object.values(c.status??{}).filter(e=>e!==void 0),...l?[c.parameters]:[]]);let b=(n,r)=>e(t,`${o}-${n}`,`float32`,r),x=b(`selected-parameters`,3),S=l?void 0:b(`candidate-values`,f),C=b(`selection-fit-state`,Math.min(p,I)*(2*m+4)+1),w=n(t,{id:`${o}-selection-focus-dispatch`,operation:V,shapes:[[Math.min(p,I),256]]}),T=b(`selection-coefficients`,p*m),E=b(`selection-fitted`,p),D=e(t,`${o}-selection-codes`,`uint32`,p),O=e(t,`${o}-selection-iterations`,`uint32`,p),k=e(t,`${o}-selection-eligible`,`uint32`,p),A=b(`selection-tile-scores`,_*4),j=b(`selection-merged-scores`,h.mergedTileCount*4),M=u?b(`selection-errors`,p*m):void 0,N=u?b(`selection-influence`,p):void 0,P=q(t,`${o}-selection`,s,a.sphereRadius,s.family===`poisson`?`responseValue >= 0.0 && floor(responseValue) == responseValue`:`responseValue == 0.0 || responseValue == 1.0`),F={...s,...P.props},L=[...P.nodes];L.push(i(t,{id:`${o}-selection-eligible`,operation:V,invocationCount:p,bindings:[...Y(F),y(`eligible`,k,`u32`)],declarations:X(F,{sphereRadius:a.sphereRadius}),body:`let observed = response[responseOffset + index];
      eligible[eligibleOffset + index] = select(0u, 1u, included(index) &&
        (${s.family===`poisson`?`observed >= 0.0 && floor(observed) == observed`:`observed == 0.0 || observed == 1.0`}));`}));let H=l?b(`search-state`,11):void 0,U=l?n(t,{id:`${o}-search-dispatch`,operation:V,shapes:[[Math.min(p,I)],[1],[_],[h.mergedTileCount]]}):void 0,W=c.output.evaluatedBandwidths?[y(`history`,c.output.evaluatedBandwidths)]:[];L.push(i(t,{id:`${o}-selection-init`,operation:V,invocationCount:f,bindings:[y(`scores`,c.output.scores),y(`parameters`,x),y(`focalDispatch`,w.view,`u32`),...S?[y(`candidateValues`,S)]:[],...W,...l?[v(`bounds`,c.parameters),y(`state`,H),U.binding]:[]],declarations:B,body:`scores[scoresOffset + index] = getNaN();
      ${l?``:`let values = array<f32, ${f}>(${c.bandwidths.map(r).join(`, `)}); candidateValues[candidateValuesOffset + index] = values[index];`}
      ${W.length?`history[historyOffset + index] = getNaN();`:``}
      if (index != 0u) { return; }
      parameters[parametersOffset + 2u] = 0.0;
      // Initialize once, unconditionally, so skipped candidates never see stale transient alias data.
      // Every completed candidate disarms this same record before the next candidate uses it.
      focalDispatch[focalDispatchOffset] = 0u;
      focalDispatch[focalDispatchOffset + 1u] = 1u;
      focalDispatch[focalDispatchOffset + 2u] = 1u;
    ${l?`
    for (var slot = 0u; slot < 11u; slot++) { state[stateOffset + slot] = 0.0; }
    let lower = bounds[boundsOffset]; let upper = bounds[boundsOffset + 1u];
    let tolerance = bounds[boundsOffset + 2u];
    var valid = isFiniteBits(lower) && isFiniteBits(upper) && lower > 0.0 && lower < upper &&
      isFiniteBits(tolerance) && tolerance > 0.0;
    ${s.bandwidthMode===`adaptive`?`valid = valid && floor(lower) == lower && floor(upper) == upper && upper <= ${s.maximumNeighborCount??128}.0;`:``}
    state[stateOffset] = lower; state[stateOffset + 1u] = upper;
    state[stateOffset + 2u] = ${s.bandwidthMode===`adaptive`?`round(`:``}lower + 0.38197 * (upper - lower)${s.bandwidthMode===`adaptive`?`)`:``};
    state[stateOffset + 3u] = ${s.bandwidthMode===`adaptive`?`round(`:``}upper - 0.38197 * (upper - lower)${s.bandwidthMode===`adaptive`?`)`:``};
    ${s.bandwidthMode===`adaptive`?`if (upper - lower <= ${z}.0) {
      state[stateOffset + 9u] = 1.0;
      state[stateOffset + 2u] = lower; state[stateOffset + 3u] = min(lower + 1.0, upper);
    }`:``}
    state[stateOffset + 4u] = getNaN(); state[stateOffset + 5u] = 3.402823e38;
    state[stateOffset + 8u] = select(0.0, 1.0, valid);
    if (valid) { ${U.armWGSL} } else { ${U.disarmWGSL} }
    `:``}`}));let G=(e,n)=>{L.push(i(t,{id:`${o}-candidate-${e}-parameters`,operation:V,invocationCount:1,...n?.getGate(1)??{},bindings:[v(`source`,s.parameters),y(`parameters`,x),...H?[v(`state`,H)]:[v(`candidateValues`,S)],...W],body:`let candidateSlot = u32(parameters[parametersOffset + 2u]);
        parameters[parametersOffset] = ${l?`state[stateOffset + 2u + candidateSlot % 2u]`:`candidateValues[candidateValuesOffset + candidateSlot]`};
        parameters[parametersOffset + 1u] = source[sourceOffset + 1u];
        ${W.length?`history[historyOffset + candidateSlot] = parameters[parametersOffset];`:``}`})),L.push(...K(t,{...F,id:`${o}-candidate-${e}`,bandwidthSelection:void 0,inference:u?{standardErrors:M,influence:N}:void 0,leaveOneOut:!u,predictExcluded:!1,parameters:x,output:{coefficients:T,fitted:E,status:{codes:D,iterationCount:O}}},{fittedScale:`link`,gridSetup:P,gate:n,fits:C,focusGate:w})),L.push(i(t,{id:`${o}-candidate-${e}-deviance`,operation:V,invocationCount:_,...n?.getGate(2)??{},bindings:[v(`response`,s.response),v(`fitted`,E),v(`codes`,D,`u32`),v(`eligible`,k,`u32`),...N?[v(`influence`,N)]:[],y(`tiles`,A)],declarations:B,body:`var deviance = 0.0; var trace = 0.0; var count = 0.0; var failed = 0.0;
        let endRow = min((index + 1u) * 256u, ${p}u);
        for (var row = index * 256u; row < endRow; row++) {
          if (eligible[eligibleOffset + row] == 0u) { continue; }
          count += 1.0;
          if (codes[codesOffset + row] != 0u) { failed += 1.0; continue; }
          let observed = response[responseOffset + row]; let linear = fitted[fittedOffset + row];
          ${s.family===`poisson`?`let score = 2.0 * (select(0.0, observed * (log(max(observed, 1.0e-30)) - linear), observed > 0.0) - observed + exp(linear));`:`let opposed = select(linear, -linear, observed == 1.0);
          let score = 2.0 * (max(opposed, 0.0) + log(1.0 + exp(-abs(opposed))));`}
          if (!isFiniteBits(score)) { failed += 1.0; continue; }
          deviance += max(0.0, score);
          ${N?`let leverage = influence[influenceOffset + row];
          if (!isFiniteBits(leverage)) { failed += 1.0; } else { trace += leverage; }`:``}
        }
        let base = tilesOffset + index * 4u;
        tiles[base] = deviance; tiles[base + 1u] = trace;
        tiles[base + 2u] = count; tiles[base + 3u] = failed;`}));let r=A;h.usesParallelMerge&&(r=j,L.push(i(t,{id:`${o}-candidate-${e}-merge-scores`,operation:V,invocationCount:h.finalTileCount,...n?.getGate(3)??{},bindings:[v(`tiles`,A),y(`totals`,r)],body:`var totalsValue = vec4f(0.0);
        let endTile = min((index + 1u) * 256u, arrayLength(&tiles) / 4u);
        for (var tile = index * 256u; tile < endTile; tile++) {
          let base = tilesOffset + tile * 4u;
          totalsValue += vec4f(tiles[base], tiles[base + 1u], tiles[base + 2u], tiles[base + 3u]);
        }
        for (var component = 0u; component < 4u; component++) {
          totals[totalsOffset + index * 4u + component] = totalsValue[component];
        }`}))),L.push(i(t,{id:`${o}-candidate-${e}-score`,operation:V,invocationCount:1,...n?.getGate(1)??{},bindings:[v(`tiles`,r),y(`scores`,c.output.scores),y(`parameters`,x)],declarations:B,body:`var deviance = 0.0; var trace = 0.0; var count = 0.0; var failed = 0.0;
        // Fixed order: increasing rows within 256-row tiles, increasing adjacent tiles within
        // 256-tile groups when needed, then increasing groups. Each invocation adds at most 256
        // values, avoiding the float32 drift of a serial sum of 4,096 non-integer tile scores.
        let totalCount = (arrayLength(&tiles) - tilesOffset) / 4u;
        for (var tile = 0u; tile < 256u; tile++) {
          if (tile >= totalCount) { break; }
          let base = tilesOffset + tile * 4u;
          deviance += tiles[base]; trace += tiles[base + 1u];
          count += tiles[base + 2u]; failed += tiles[base + 3u];
        }
        var score = getNaN();
        if (failed == 0.0 && count > 0.0 ${u?`&& count > trace + 1.0`:``}) {
          score = ${u?`deviance + 2.0 * trace + 2.0 * trace * (trace + 1.0) / (count - trace - 1.0)`:`deviance / count`};
        }
        let candidateSlot = u32(parameters[parametersOffset + 2u]);
        scores[scoresOffset + candidateSlot] = score;
        parameters[parametersOffset + 2u] += 1.0;`}))};if(l)for(let e=0;e<d;e++)G(2*e,U),G(2*e+1,U),L.push(i(t,{id:`${o}-search-${e}-iterate`,operation:V,invocationCount:1,bindings:[v(`scores`,c.output.scores),v(`parameters`,x),v(`bounds`,c.parameters),y(`state`,H),U.binding],declarations:`${B}
fn getFeasibilityFraction(ordinal: u32) -> f32 {
  let levelSize = 1u << firstLeadingBit(ordinal);
  return f32(2u * (ordinal - levelSize) + 1u) / f32(2u * levelSize);
}`,body:`if (state[stateOffset + 8u] == 0.0) { return; }
          let candidateSlot = u32(parameters[parametersOffset + 2u]);
          let firstScore = scores[scoresOffset + candidateSlot - 2u];
          let secondScore = scores[scoresOffset + candidateSlot - 1u];
          let firstValid = isFiniteBits(firstScore); let secondValid = isFiniteBits(secondScore);
          let chooseFirst = firstValid && (!secondValid || firstScore <= secondScore);
          let winner = select(state[stateOffset + 3u], state[stateOffset + 2u], chooseFirst);
          let winnerScore = select(secondScore, firstScore, chooseFirst);
          if ((firstValid || secondValid) && winnerScore < state[stateOffset + 5u]) {
            state[stateOffset + 4u] = winner; state[stateOffset + 5u] = winnerScore;
          }
          state[stateOffset + 6u] += 1.0;
          var lower = state[stateOffset]; var upper = state[stateOffset + 1u];
          var first = state[stateOffset + 2u]; var second = state[stateOffset + 3u];
          var mode = state[stateOffset + 9u];
          var converged = false; var finished = false;
          if (mode == 1.0) {
            // Every integer in this bracket must be scored before claiming convergence.
            finished = second >= upper;
            converged = finished && isFiniteBits(state[stateOffset + 4u]);
            if (!finished) { first = second + 1.0; second = min(first + 1.0, upper); }
          } else if (mode == 2.0) {
            // Invalid interiors imply no ordering. Keep the bracket, probe both endpoints first,
            // then cover it by a breadth-first dyadic sequence. Report budget exhaustion, not
            // convergence, since feasibility probes do not establish a minimizing bracket.
            let ordinal = u32(state[stateOffset + 10u]) + 1u;
            first = lower + (upper - lower) * getFeasibilityFraction(ordinal);
            second = lower + (upper - lower) * getFeasibilityFraction(ordinal + 1u);
            state[stateOffset + 10u] += 2.0;
          } else if (!firstValid && !secondValid) {
            // Never discard either side of an unordered pair of invalid scores.
            ${s.bandwidthMode===`adaptive`?`mode = 1.0; first = lower; second = min(first + 1.0, upper);`:`mode = 2.0; first = lower; second = upper;`}
          } else {
            let previousFirst = first; let previousSecond = second;
            let closeScores = firstValid && secondValid && first != second &&
              abs(firstScore - secondScore) <= bounds[boundsOffset + 2u];
            if (chooseFirst) {
              upper = previousSecond;
              first = ${s.bandwidthMode===`adaptive`?`round(`:``}lower + 0.38197 * (upper - lower)${s.bandwidthMode===`adaptive`?`)`:``};
              second = previousFirst;
            } else {
              lower = previousFirst;
              first = previousSecond;
              second = ${s.bandwidthMode===`adaptive`?`round(`:``}upper - 0.38197 * (upper - lower)${s.bandwidthMode===`adaptive`?`)`:``};
            }
            ${s.bandwidthMode===`adaptive`?`if (closeScores || upper - lower <= ${z}.0 ||
                first >= second || first <= lower || second >= upper) {
              mode = 1.0; first = lower; second = min(first + 1.0, upper);
            }`:`converged = closeScores || first == second; finished = converged;`}
          }
          state[stateOffset + 9u] = mode;
          state[stateOffset] = lower; state[stateOffset + 1u] = upper;
          state[stateOffset + 2u] = first; state[stateOffset + 3u] = second;
          if (converged) { state[stateOffset + 7u] = 1.0; }
          if (finished || state[stateOffset + 6u] >= ${d}.0) {
            state[stateOffset + 8u] = 0.0; ${U.disarmWGSL}
          }`}));else c.bandwidths.forEach((e,t)=>G(t));let J=Object.entries(c.status??{}).filter(e=>e[1]!==void 0).map(([e,t])=>y(e,t,`u32`));return L.push(i(t,{id:`${o}-select-bandwidth`,operation:V,invocationCount:1,bindings:[v(`scores`,c.output.scores),y(`bandwidth`,c.output.bandwidth),y(`status`,c.output.status,`u32`),y(`parameters`,x),...H?[v(`state`,H)]:[],...J],declarations:B,body:`var chosen = getNaN(); var best = 3.402823e38;
      ${l?`chosen = state[stateOffset + 4u];`:c.bandwidths.map((e,t)=>`if (isFiniteBits(scores[scoresOffset + ${t}u]) && scores[scoresOffset + ${t}u] < best) {
          best = scores[scoresOffset + ${t}u]; chosen = ${r(e)};
        }`).join(`
`)}
      bandwidth[bandwidthOffset] = chosen; parameters[parametersOffset] = chosen;
      status[statusOffset] = select(1u, 0u, isFiniteBits(chosen));
      ${c.status?.iterationCount?`iterationCount[iterationCountOffset] = ${l?`u32(state[stateOffset + 6u])`:`1u`};`:``}
      ${c.status?.converged?`converged[convergedOffset] = ${l?`u32(state[stateOffset + 7u])`:`select(0u, 1u, isFiniteBits(chosen))`};`:``}
      ${c.status?.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = ${l?`select(0u, 1u, state[stateOffset + 6u] >= ${d}.0 && state[stateOffset + 7u] == 0.0)`:`0u`};`:``}`})),L.push(i(t,{id:`${o}-selected-tolerance`,operation:V,invocationCount:1,bindings:[v(`source`,s.parameters),y(`parameters`,x)],body:`parameters[parametersOffset + 1u] = source[sourceOffset + 1u];`})),L.push(...K(t,{...F,id:`${o}-selected-fit`,bandwidthSelection:void 0,parameters:x},{gridSetup:P,fits:C,focusGate:w})),L}var U=65536,W=1048576,G=class{id;spatialContext;estimatedCommandNodeCount;sphereRadius;props;fittedScale;constructor(e,n=`response`){this.fittedScale=n;let r={...e,indexGridSize:e.indexGridSize||void 0};this.id=r.id??`generalized-geographically-weighted-regression`,this.spatialContext=b(this.id,r.spatialContext,{...v,measure:`distance`}),this.sphereRadius=this.spatialContext.metric===`great-circle`?y(this.spatialContext):null,this.sphereRadius!==null&&p(this.id,`spatialContext.sphereRadius`,this.sphereRadius,Number.MIN_VALUE,L),e.indexGridSize===void 0&&r.response.length>65536&&this.sphereRadius===null&&(r.kernel===`bisquare`||r.gaussianCutoff!==void 0)&&(r.indexGridSize=O(r.response.length)),this.props=r;let i=!!r.indexGridSize&&(r.kernel===`bisquare`||r.gaussianCutoff!==void 0);J(this.id,r,this.sphereRadius!==null,i?F:W),d(this.id,!r.bandwidthMode||[`fixed`,`adaptive`].includes(r.bandwidthMode),`bandwidthMode must be fixed or adaptive`);let a=r.maximumNeighborCount??128;f(this.id,`maximumNeighborCount`,a,1,128),d(this.id,[`poisson`,`logistic`].includes(r.family),`family must be poisson or logistic`),r.offset&&(l(r.offset,[`float32`],this.id),c(this.id,`offset`,r.offset,r.response.length));let o=r.response.length,s=r.bandwidthSelection,h=0;if(s){d(this.id,!r.leaveOneOut,`leaveOneOut must be false with bandwidthSelection`),d(this.id,!s.method||[`grid`,`golden-section`].includes(s.method),`bandwidthSelection.method must be grid or golden-section`),s.method===`golden-section`?(f(this.id,`bandwidthSelection.maximumIterations`,s.maximumIterations??24,1,64),l(s.parameters,[`float32`],this.id),c(this.id,`bandwidthSelection.parameters`,s.parameters,3),d(this.id,!s.criterion||s.criterion===`aicc`,`golden-section bandwidthSelection.criterion must be aicc`),h=2*(s.maximumIterations??24)):(f(this.id,`bandwidthSelection.bandwidths.length`,s.bandwidths.length,2,32),d(this.id,!s.criterion||[`loo-deviance`,`aicc`].includes(s.criterion),`bandwidthSelection.criterion must be loo-deviance or aicc`),s.bandwidths.forEach((e,t)=>{p(this.id,`bandwidthSelection.bandwidths[${t}]`,e,Number.MIN_VALUE,L),r.bandwidthMode===`adaptive`&&f(this.id,`bandwidthSelection.bandwidths[${t}]`,e,1,a)}),h=s.bandwidths.length),i||s.method!==`golden-section`&&s.criterion!==`aicc`||f(this.id,`bandwidthSelection AICc rows`,o,1,j);for(let[e,t]of Object.entries(s.output))(e!==`evaluatedBandwidths`||t)&&(l(t,[e===`status`?`uint32`:`float32`],this.id),c(this.id,`bandwidthSelection.output.${e}`,t,e===`scores`||e===`evaluatedBandwidths`?h:1));for(let[e,t]of Object.entries(s.status??{}))t&&(l(t,[`uint32`],this.id),c(this.id,`bandwidthSelection.status.${e}`,t,1))}s&&u(this.id,`bandwidthSelection final score partials`,R(o).finalTileCount,256);let g=Math.ceil(o/I),x=r.indexGridSize?M:0;this.estimatedCommandNodeCount=x+1+3*g+(s?h*(3*g+8)+8:0),m(this.id,this.estimatedCommandNodeCount),r.inference&&(l(r.inference.standardErrors,[`float32`],this.id),l(r.inference.influence,[`float32`],this.id),c(this.id,`inference.standardErrors`,r.inference.standardErrors,o*(r.predictorCount+1)),c(this.id,`inference.influence`,r.inference.influence,o));for(let[e,t]of Object.entries(Z(r.output)))l(t,[e===`coefficients`||e===`fitted`?`float32`:`uint32`],this.id),c(this.id,`output.${e}`,t,o*(e===`coefficients`?r.predictorCount+1:1));let S=[...Object.entries(Z(r.output)).map(([e,t])=>[`output.${e}`,t]),...Object.entries(r.inference??{}).map(([e,t])=>[`inference.${e}`,t]),...Object.entries(s?.output??{}).map(([e,t])=>[`bandwidthSelection.output.${e}`,t]),...Object.entries(s?.status??{}).map(([e,t])=>[`bandwidthSelection.status.${e}`,t])].filter(e=>e[1]!==void 0),C=[[...Object.keys(Z(r.output)).map(e=>`output.${e}`),...Object.keys(r.inference??{}).map(e=>`inference.${e}`)],[`bandwidthSelection.output.scores`,`bandwidthSelection.output.evaluatedBandwidths`],[`bandwidthSelection.output.scores`,`bandwidthSelection.output.bandwidth`,`bandwidthSelection.output.status`,...Object.keys(s?.status??{}).map(e=>`bandwidthSelection.status.${e}`)]];for(let e=0;e<S.length;e++){let[n,r]=S[e];for(let[i,a]of S.slice(e+1)){let e=C.some(e=>e.includes(n)&&e.includes(i)),o=t(r),s=t(a),c=r.buffer===a.buffer&&(e?o.offset<s.offset+s.size&&s.offset<o.offset+o.size:r.byteOffset<a.byteOffset+a.length*4&&a.byteOffset<r.byteOffset+r.length*4);d(this.id,!c,`${n} and ${i} output ranges must not overlap`)}}_(this.id,S.map(([,e])=>e),[...Y(r).map(e=>e.view),...r.bandwidthSelection?.method===`golden-section`?[r.bandwidthSelection.parameters]:[]])}getCommandNodes(r,a={}){let{id:o,props:c}=this,l=c.response.length,u=c.predictorCount+1,d=c.maximumIterations??32,f=u+3+(c.inference?u+1:0),p=Math.min(l,I),m=Math.ceil(l/p),_=c.bandwidthSelection,v=[...Y(c).map(e=>e.view),...Object.values(Z(c.output)),...Object.values(c.inference??{}),...Object.values(_?.output??{}),...Object.values(_?.status??{}),..._?.method===`golden-section`?[_.parameters]:[],a.fits,a.focusGate?.view,a.gridSetup?.grid?.cellOffsets,a.gridSetup?.grid?.sortedIds].filter(e=>e!==void 0);g(o,r,v);for(let e of v)h(r.device,o,`view binding`,t(e).size,`reduce rows or predictorCount`);let y=[[`fit state`,(p*(_?2*u+4:f)+1)*4],[`focus dispatch`,12]];_&&y.push([`selection coefficients/errors`,l*u*4],[`selection fitted/codes/iterations/eligible/influence`,l*4],[`selection tile scores`,R(l).tileCount*16],[`selection merged scores`,R(l).mergedTileCount*16],[`selection parameters/search/gate`,128]),y.push(...x(l,c.indexGridSize??null));for(let[e,t]of y)h(r.device,o,e,t,`reduce rows, predictorCount or indexGridSize`);if(_)return H(r,this);let b=a.fits??e(r,`${o}-fits`,`float32`,p*f+1),S=a.gridSetup??q(r,o,c,this.sphereRadius,c.family===`poisson`?`responseValue >= 0.0 && floor(responseValue) == responseValue`:`responseValue == 0.0 || responseValue == 1.0`),C={...c,response:S.props.response,mask:S.props.mask},E=[...Y(C),...S.grid?[{name:`cellOffsets`,view:S.grid.cellOffsets,type:`u32`,access:`read`},{name:`sortedIds`,view:S.grid.sortedIds,type:`u32`,access:`read`}]:[]],D={sphereRadius:this.sphereRadius,grid:c.indexGridSize},O=c.bandwidthMode===`adaptive`&&!!c.indexGridSize,j=(c.maximumNeighborCount??128)+ +!!c.leaveOneOut,M=e=>`{\n${k(e,!!c.indexGridSize,`included(row) && responseValid(row)`)}\n}`,P=`${X(C,D)}\n${D.grid?``:s}\n${A(u,25)}\n${T(u)}
${O?`const MAXIMUM_NEIGHBORS: u32 = ${j}u;\nconst SENTINEL: f32 = 3.0e38;\n${w()}`:``}
fn responseValid(row: u32) -> bool {
  let responseValue = response[responseOffset + row];
  return ${c.family===`poisson`?`responseValue >= 0.0 && floor(responseValue) == responseValue`:`responseValue == 0.0 || responseValue == 1.0`};
}
// IRLS runs in a design basis centered on the focal row and scaled by the largest in-bandwidth
// deviation, so large mean-to-spread predictors (years, dollars, projected coordinates) keep the
// information matrix well conditioned in float32. Coefficients are mapped back at the end.
var<private> centerValues: array<f32, ${u}>;
var<private> scaleValues: array<f32, ${u}>;
fn designValue(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return (predictor(row, column) - centerValues[column]) / scaleValues[column];
}
fn linearPredictor(row: u32, coefficients: ptr<function, array<f32, ${u}>>) -> f32 {
  var value = ${c.offset?`offsets[offsetsOffset + row]`:`0.0`};
  for (var column = 0u; column < ${u}u; column++) { value += (*coefficients)[column] * designValue(row, column); }
  return value;
}
// Symmetric diagonal equilibration of the lower-triangular information, then Cholesky (shared with GWR).
fn factorInformation(information: ptr<function, array<f32, ${u*u}>>, equilibration: ptr<function, array<f32, ${u}>>) -> bool {
  return factorEquilibrated_${u}(information, equilibration);
}
fn solveInformation(information: ptr<function, array<f32, ${u*u}>>, equilibration: ptr<function, array<f32, ${u}>>, vector: ptr<function, array<f32, ${u}>>) {
  solveEquilibrated_${u}(information, equilibration, vector);
}
fn meanResponse(linear: f32) -> f32 {
  return ${c.family===`poisson`?`exp(linear)`:`select(exp(linear) / (1.0 + exp(linear)), 1.0 / (1.0 + exp(-linear)), linear >= 0.0)`};
}
fn varianceResponse(linear: f32) -> f32 {
  ${c.family===`poisson`?`return exp(linear);`:`let tail = exp(-abs(linear)); return tail / ((1.0 + tail) * (1.0 + tail));`}
}
fn logLikelihood(focus: u32, bandwidth: f32, coefficients: ptr<function, array<f32, ${u}>>) -> f32 {
  var likelihood = 0.0;
  let origin = getPosition(focus);
  ${M(`let weight = spatialWeight(focus, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let linear = linearPredictor(row, coefficients);
    if (!isFiniteBits(linear) || abs(linear) > ${c.family===`poisson`?`30.0`:`80.0`}) { return getNaN(); }
    let responseValue = response[responseOffset + row];
    likelihood += weight * (${c.family===`poisson`?`responseValue * linear - exp(linear)`:`responseValue * linear - max(linear, 0.0) - log(1.0 + exp(-abs(linear)))`});`)}
  return likelihood;
}`,F=a.gridSetup?[]:[...S.nodes],L=a.focusGate??n(r,{id:`${o}-focus-dispatch`,operation:`GPUGeneralizedGeographicallyWeightedRegression`,shapes:[[p,256]]});F.push(i(r,{id:`${o}-focus-init`,operation:`GPUGeneralizedGeographicallyWeightedRegression`,invocationCount:1,...a.gate?.getGate(1)??{},bindings:[{name:`fits`,view:b,type:`f32`,access:`read_write`},L.binding],body:`fits[fitsOffset] = 0.0; ${L.armWGSL}`}));for(let e=0;e<m;e++){let t=m>1?`-tile${e}`:``;F.push(i(r,{id:`${o}-fit${t}`,operation:`GPUGeneralizedGeographicallyWeightedRegression`,variant:c.family,invocationCount:p,workgroupSize:256,...L.getGate(0),bindings:[...E,{name:`fits`,view:b,type:`f32`,access:`read_write`}],declarations:P,body:`let focus = index + u32(fits[fitsOffset]);
if (focus >= ROW_COUNT) { return; }
let base = fitsOffset + 1u + index * ${f}u;
for (var column = 0u; column < ${f}u; column++) { fits[base + column] = getNaN(); }
fits[base + ${u+1}u] = 1.0; fits[base + ${u+2}u] = 0.0;
if (${c.predictExcluded?`!queryValid(focus)`:`!included(focus) || !responseValid(focus)`}) { return; }
var bandwidth = parameters[parametersOffset]; let tolerance = parameters[parametersOffset + 1u];
if (!(bandwidth > 0.0) || !isFiniteBits(bandwidth) || !(tolerance > 0.0) || !isFiniteBits(tolerance)) { fits[base + ${u+1}u] = 4.0; return; }
${c.bandwidthMode===`adaptive`?`
if (floor(bandwidth) != bandwidth || bandwidth > ${c.maximumNeighborCount??128}.0) { fits[base + ${u+1}u] = 4.0; return; }
let neighborCount = u32(bandwidth);
${O?`// The grid holds the eligible rows; with leave-one-out the focus is one of them (distance 0), so
// the kth neighbor excluding it is entry k of the ring list.
let ringCount = neighborCount + select(0u, 1u, ${c.leaveOneOut?`included(focus) && responseValid(focus)`:`false`});
var ringList: array<f32, ${j}>;
selectNeighborDistances(focus, ringCount, &ringList);
if (ringList[ringCount - 1u] >= SENTINEL) { fits[base + ${u+1}u] = 2.0; return; }
bandwidth = sqrt(ringList[ringCount - 1u]) * 1.00001;`:`var distances: array<f32, ${c.maximumNeighborCount??128}>;
for (var neighbor = 0u; neighbor < neighborCount; neighbor++) { distances[neighbor] = 3.402823e38; }
var eligibleCount = 0u;
for (var row = 0u; row < ${l}u; row++) {
  if (!included(row) || !responseValid(row) ${c.leaveOneOut?`|| row == focus`:``}) { continue; }
  let distance = getDistance(getPosition(focus), getPosition(row));
  if (!isFiniteBits(distance)) { fits[base + ${u+1}u] = 4.0; return; }
  eligibleCount++;
  var candidateDistance = distance;
  for (var neighbor = 0u; neighbor < neighborCount; neighbor++) {
    if (candidateDistance < distances[neighbor]) { let previous = distances[neighbor]; distances[neighbor] = candidateDistance; candidateDistance = previous; }
  }
}
if (eligibleCount < neighborCount) { fits[base + ${u+1}u] = 2.0; return; }
bandwidth = distances[neighborCount - 1u] * 1.00001;`}
// A zero kth distance has no positive distance scale; never invent a coordinate-unit floor.
if (!(bandwidth > 0.0)) { fits[base + ${u+1}u] = 2.0; return; }
if (!isFiniteBits(bandwidth)) { fits[base + ${u+1}u] = 4.0; return; }
`:``}
var positiveCount = 0u; var responseSum = 0.0; var zeroResponseCount = 0u;
var minimumZeroValues: array<f32, ${u}>;
var maximumZeroValues: array<f32, ${u}>;
var minimumPositiveValues: array<f32, ${u}>;
var maximumPositiveValues: array<f32, ${u}>;
let origin = getPosition(focus);
for (var column = 1u; column < ${u}u; column++) {
  centerValues[column] = predictor(focus, column); scaleValues[column] = 0.0;
  minimumZeroValues[column] = bitcast<f32>(0x7f7fffffu);
  maximumZeroValues[column] = -bitcast<f32>(0x7f7fffffu);
  minimumPositiveValues[column] = bitcast<f32>(0x7f7fffffu);
  maximumPositiveValues[column] = -bitcast<f32>(0x7f7fffffu);
}
${M(`if (!(spatialWeight(focus, row, bandwidth) > 0.0)) { continue; }
  let observed = response[responseOffset + row];
  positiveCount++; responseSum += observed;
  zeroResponseCount += select(0u, 1u, observed == 0.0);
  for (var column = 1u; column < ${u}u; column++) {
    let value = predictor(row, column);
    scaleValues[column] = max(scaleValues[column], abs(value - centerValues[column]));
    if (observed == 0.0) {
      minimumZeroValues[column] = min(minimumZeroValues[column], value);
      maximumZeroValues[column] = max(maximumZeroValues[column], value);
    } else {
      minimumPositiveValues[column] = min(minimumPositiveValues[column], value);
      maximumPositiveValues[column] = max(maximumPositiveValues[column], value);
    }
  }`)}
for (var column = 1u; column < ${u}u; column++) { if (!(scaleValues[column] > 0.0) || !isFiniteBits(scaleValues[column])) { scaleValues[column] = 1.0; } }
if (positiveCount < ${u}u) { fits[base + ${u+1}u] = 2.0; return; }
if (responseSum == 0.0 ${c.family===`logistic`?`|| responseSum == f32(positiveCount)`:``}) { fits[base + ${u+1}u] = 5.0; return; }
// Raw float32 comparisons certify a single-column threshold without rounded dot products.
// Logistic: the two class intervals touch or are disjoint, with at least one strict move.
// Poisson: every positive-count row is on the threshold; zero-count rows move to one side.
if (zeroResponseCount > 0u) {
  for (var column = 1u; column < ${u}u; column++) {
    let minimumZero = minimumZeroValues[column]; let maximumZero = maximumZeroValues[column];
    let minimumPositive = minimumPositiveValues[column]; let maximumPositive = maximumPositiveValues[column];
    let separatesColumn = ${c.family===`logistic`?`((maximumZero <= minimumPositive) && (minimumZero < minimumPositive || maximumPositive > maximumZero)) ||
      ((maximumPositive <= minimumZero) && (minimumPositive < minimumZero || maximumZero > maximumPositive))`:`(minimumPositive == maximumPositive) &&
      ((minimumZero >= minimumPositive && maximumZero > minimumPositive) ||
       (maximumZero <= minimumPositive && minimumZero < minimumPositive))`};
    if (separatesColumn) { fits[base + ${u+1}u] = 6.0; return; }
  }
}
var coefficients: array<f32, ${u}>;
var state = 3u; var completed = 0u;
for (var iteration = 0u; iteration < ${d}u; iteration++) {
  var information: array<f32, ${u*u}>; var gradient: array<f32, ${u}>;
  var equilibration: array<f32, ${u}>;
  var observations = 0u; var finite = true;
  ${M(`if (!finite) { continue; }
    let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
    let linear = linearPredictor(row, &coefficients);
    if (!isFiniteBits(linear) || abs(linear) > ${c.family===`poisson`?`30.0`:`80.0`}) { finite = false; continue; }
    let mean = meanResponse(linear); let variance = varianceResponse(linear);
    observations++;
    var design: array<f32, ${u}>;
    for (var column = 0u; column < ${u}u; column++) { design[column] = designValue(row, column); }
    for (var column = 0u; column < ${u}u; column++) {
      gradient[column] += spatial * design[column] * (response[responseOffset + row] - mean);
      for (var other = 0u; other <= column; other++) { information[column * ${u}u + other] += spatial * variance * design[column] * design[other]; }
    }`)}
  completed = iteration + 1u;
  if (!finite) { state = 4u; break; }
  if (observations < ${u}u || !factorInformation(&information, &equilibration)) { state = 2u; break; }
  solveInformation(&information, &equilibration, &gradient);
  // Certify only decided moves in the recession cone. A rounding bound rejects ambiguous
  // signs; it never permits an opposing move. Only structurally zero terms establish zero.
  var directionScale = 0.0; var finiteDirection = true;
  for (var column = 0u; column < ${u}u; column++) {
    finiteDirection = finiteDirection && isFiniteBits(gradient[column]);
    directionScale = max(directionScale, abs(gradient[column]));
  }
  var separates = finiteDirection && directionScale > 0.0 && isFiniteBits(directionScale);
  var strictSeparation = false;
  ${M(`if (!(spatialWeight(focus, row, bandwidth) > 0.0) || !separates) { continue; }
    var direction = 0.0; var directionMagnitude = 0.0; var standsStill = true;
    for (var column = 0u; column < ${u}u; column++) {
      let coefficient = gradient[column] / directionScale;
      let term = coefficient * designValue(row, column);
      direction += term; directionMagnitude += abs(term);
      if (coefficient != 0.0) {
        if (column == 0u) { standsStill = false; }
        else if (predictor(row, column) != centerValues[column]) { standsStill = false; }
      }
    }
    let errorBound = ${u+8}.0 * ${N} * directionMagnitude;
    let decided = isFiniteBits(direction) && isFiniteBits(directionMagnitude) && directionMagnitude > 0.0;
    let observed = response[responseOffset + row];
    ${c.family===`logistic`?`let margin = select(-direction, direction, observed == 1.0);
    let movesRight = decided && margin > errorBound;
    separates = standsStill || movesRight;
    strictSeparation = strictSeparation || movesRight;`:`let decreasesZero = decided && direction < -errorBound;
    separates = standsStill || (observed == 0.0 && decreasesZero);
    strictSeparation = strictSeparation || (observed == 0.0 && decreasesZero);`}`)}
  if (separates && strictSeparation) { state = 6u; break; }
  let previousLikelihood = logLikelihood(focus, bandwidth, &coefficients);
  var step = 1.0; var accepted = false; var candidate: array<f32, ${u}>;
  for (var attempt = 0u; attempt < 16u; attempt++) {
    for (var column = 0u; column < ${u}u; column++) { candidate[column] = coefficients[column] + step * gradient[column]; }
    let likelihood = logLikelihood(focus, bandwidth, &candidate);
    if (isFiniteBits(likelihood) && likelihood >= previousLikelihood - 1.0e-6 * max(1.0, abs(previousLikelihood))) { accepted = true; break; }
    step *= 0.5;
  }
  if (!accepted) { state = 4u; break; }
  var change = 0.0;
  for (var column = 0u; column < ${u}u; column++) { change = max(change, abs(candidate[column] - coefficients[column]) / max(1.0, abs(coefficients[column]))); coefficients[column] = candidate[column]; }
  if (change <= tolerance) { state = 0u; break; }
}
if (state == 0u || state == 3u) {
  // Publish the current finite iterate even when the iteration budget is exhausted.
  // Map the centered, scaled coefficients back to the raw predictor basis.
  fits[base + ${u}u] = ${this.fittedScale===`link`?`linearPredictor(focus, &coefficients)`:`meanResponse(linearPredictor(focus, &coefficients))`};
  var intercept = coefficients[0];
  for (var column = 1u; column < ${u}u; column++) {
    fits[base + column] = coefficients[column] / scaleValues[column];
    intercept -= fits[base + column] * centerValues[column];
  }
  fits[base] = intercept;
  ${c.inference?`
  var finalInformation: array<f32, ${u*u}>;
  var finalEquilibration: array<f32, ${u}>;
  ${M(`let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
    let variance = varianceResponse(linearPredictor(row, &coefficients));
    for (var column = 0u; column < ${u}u; column++) {
      for (var other = 0u; other <= column; other++) { finalInformation[column * ${u}u + other] += spatial * variance * designValue(row, column) * designValue(row, other); }
    }`)}
  if (factorInformation(&finalInformation, &finalEquilibration)) {
    // Raw coefficient c is row c of the affine map A from the centered basis, so its variance
    // solves with A^T e_c and projects each row on its centered design vector.
    for (var column = 0u; column < ${u}u; column++) {
      var inverseColumn: array<f32, ${u}>;
      if (column == 0u) {
        inverseColumn[0] = 1.0;
        for (var other = 1u; other < ${u}u; other++) { inverseColumn[other] = -centerValues[other] / scaleValues[other]; }
      } else {
        inverseColumn[column] = 1.0 / scaleValues[column];
      }
      solveInformation(&finalInformation, &finalEquilibration, &inverseColumn);
      var coefficientVariance = 0.0;
      ${M(`let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
        let variance = varianceResponse(linearPredictor(row, &coefficients));
        var projection = 0.0;
        for (var other = 0u; other < ${u}u; other++) { projection += inverseColumn[other] * designValue(row, other); }
        coefficientVariance += spatial * spatial * variance * projection * projection;`)}
      fits[base + ${u+3}u + column] = sqrt(max(0.0, coefficientVariance));
    }
    // The focal design vector is (1, 0, ..., 0), so the leverage is the first inverse entry.
    var focalColumn: array<f32, ${u}>;
    focalColumn[0] = 1.0;
    solveInformation(&finalInformation, &finalEquilibration, &focalColumn);
    fits[base + ${2*u+3}u] = focalColumn[0] * spatialWeight(focus, focus, bandwidth) * varianceResponse(linearPredictor(focus, &coefficients));
  }
  `:``}
}
fits[base + ${u+1}u] = f32(state); fits[base + ${u+2}u] = f32(completed);`}),i(r,{id:`${o}-publish${t}`,operation:`GPUGeneralizedGeographicallyWeightedRegression`,variant:`publish`,invocationCount:p,workgroupSize:256,...L.getGate(0),bindings:[{name:`fits`,view:b,type:`f32`,access:`read`},...Object.entries(Z(c.output)).map(([e,t])=>({name:e,view:t,type:e===`coefficients`||e===`fitted`?`f32`:`u32`,access:`read_write`})),...Object.entries(c.inference??{}).map(([e,t])=>({name:e,view:t,type:`f32`,access:`read_write`}))],body:`let row = index + u32(fits[fitsOffset]);
if (row >= ${l}u) { return; }
let base = fitsOffset + 1u + index * ${f}u;
for (var column = 0u; column < ${u}u; column++) { coefficients[coefficientsOffset + row * ${u}u + column] = fits[base + column]; }
fitted[fittedOffset + row] = fits[base + ${u}u]; status[statusOffset + row] = u32(fits[base + ${u+1}u]); iterations[iterationsOffset + row] = u32(fits[base + ${u+2}u]);
${c.inference?`for (var column = 0u; column < ${u}u; column++) { standardErrors[standardErrorsOffset + row * ${u}u + column] = fits[base + ${u+3}u + column]; } influence[influenceOffset + row] = fits[base + ${2*u+3}u];`:``}`})),F.push(i(r,{id:`${o}-focus-advance${t}`,operation:`GPUGeneralizedGeographicallyWeightedRegression`,invocationCount:1,...a.gate?.getGate(1)??{},bindings:[{name:`fits`,view:b,type:`f32`,access:`read_write`},L.binding],body:`fits[fitsOffset] += ${p}.0;
          let remainingRows = ${l}u - min(${l}u, u32(fits[fitsOffset]));
          dispatch[dispatchOffset] = (min(remainingRows, ${p}u) + 255u) / 256u;`}))}return F}};function K(e,t,n){let r=Reflect.construct(G,[t,n.fittedScale]);return Reflect.apply(r.getCommandNodes,r,[e,n])}function q(t,n,r,a,s=`true`){if(!r.indexGridSize)return{props:r,grid:void 0,nodes:[]};let c=r.response.length,l=[],u=r.response;r.mask&&(u=e(t,`${n}-masked-response`,`float32`,c),l.push(i(t,{id:`${n}-masked-response`,operation:`GPULocalRegressionGrid`,variant:`masked-response`,invocationCount:c,bindings:[{name:`source`,view:r.response,type:`f32`,access:`read`},{name:`mask`,view:r.mask,type:`u32`,access:`read`},{name:`masked`,view:u,type:`f32`,access:`read_write`}],declarations:o,body:`masked[maskedOffset + index] = select(getNaN(), source[sourceOffset + index], mask[maskOffset + index] != 0u);`})));let d={...r,response:u,mask:void 0},f=e(t,`${n}-grid-positions`,`float32x2`,c);l.push(i(t,{id:`${n}-grid-positions`,operation:`GPULocalRegressionGrid`,variant:`grid-positions`,invocationCount:c,bindings:[...Y(d),{name:`gridPositions`,view:f,type:`f32`,access:`read_write`}],declarations:`${X(d,{sphereRadius:a})}
fn isTrainingRow(row: u32) -> bool {
  let responseValue = response[responseOffset + row];
  return included(row) && (${s});
}`,body:`let keep = isTrainingRow(index);
  let invalid = getNaN();
  gridPositions[gridPositionsOffset + 2u * index] = select(invalid, positions[positionsOffset + 2u * index], keep);
  gridPositions[gridPositionsOffset + 2u * index + 1u] = select(invalid, positions[positionsOffset + 2u * index + 1u], keep);`}));let p=E(t,{id:`${n}-grid`,operation:`GPULocalRegressionGrid`,positions:f,rowCount:c,gridSize:r.indexGridSize});return l.push(...p.nodes),{props:d,grid:p,nodes:l}}function J(e,t,n=!1,r=W){let i=t.response.length,a=t.indexGridSize?r:U;if(f(e,`response rows`,i,1,a),f(e,`predictorCount`,t.predictorCount,1,24),t.indexGridSize){let[r,i]=t.indexGridSize;d(e,!n,`indexGridSize requires a planar spatialContext`),f(e,`indexGridSize columns`,r,1,P),f(e,`indexGridSize rows`,i,1,P),f(e,`indexGridSize cells`,r*i,1,P)}let o=t.maximumIterations??32;f(e,`maximumIterations`,o,1,128),d(e,!t.kernel||[`gaussian`,`bisquare`].includes(t.kernel),`kernel must be gaussian or bisquare`),t.gaussianCutoff!==void 0&&p(e,`gaussianCutoff`,t.gaussianCutoff,Number.MIN_VALUE,L),l(t.positions,[`float32x2`],e),l(t.predictors,[`float32`],e),l(t.response,[`float32`],e),l(t.parameters,[`float32`],e),c(e,`positions`,t.positions,i),c(e,`predictors`,t.predictors,i*t.predictorCount),d(e,t.parameters.length>=2,`parameters must contain at least 2 rows`),t.mask&&(l(t.mask,[`uint32`],e),c(e,`mask`,t.mask,i))}function Y(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`predictors`,view:e.predictors,type:`f32`,access:`read`},{name:`response`,view:e.response,type:`f32`,access:`read`},{name:`parameters`,view:e.parameters,type:`f32`,access:`read`},...e.offset?[{name:`offsets`,view:e.offset,type:`f32`,access:`read`}]:[],...e.mask?[{name:`mask`,view:e.mask,type:`u32`,access:`read`}]:[]]}function X(e,t={}){let n=t.sphereRadius??null,r=e=>`isFiniteBits(positions[positionsOffset + 2u * ${e}]) && isFiniteBits(positions[positionsOffset + 2u * ${e} + 1u])${n===null?``:` && abs(positions[positionsOffset + 2u * ${e} + 1u]) <= 90.0`}`,i=`fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
${D(n)}
${S(e.gaussianCutoff??null)}
${t.grid?C(t.grid,e.gaussianCutoff??null):``}
fn spatialWeight(focus: u32, row: u32, bandwidth: f32) -> f32 {
  ${e.leaveOneOut?`if (focus == row) { return 0.0; }`:``}
  return getLocalKernelWeight(getDistance(getPosition(focus), getPosition(row)), bandwidth, getIsBisquare());
}`,s=`${o}\n${a}
const ROW_COUNT: u32 = ${e.response.length}u;
fn getIsBisquare() -> bool { return ${e.kernel===`bisquare`?`true`:`false`}; }`;return t.neighborhoodOnly?`${s}\n${i}`:`${s}
fn queryValid(row: u32) -> bool {
  ${e.offset?`if (!isFiniteBits(offsets[offsetsOffset + row])) { return false; }`:``}
  if (!(${r(`row`)})) { return false; }
  for (var column = 0u; column < ${e.predictorCount}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${e.predictorCount}u + column])) { return false; } }
  return true;
}
fn included(row: u32) -> bool {
  ${e.mask?`if (mask[maskOffset + row] == 0u) { return false; }`:``}
  ${e.offset?`if (!isFiniteBits(offsets[offsetsOffset + row])) { return false; }`:``}
  if (!isFiniteBits(response[responseOffset + row]) || !(${r(`row`)})) { return false; }
  for (var column = 0u; column < ${e.predictorCount}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${e.predictorCount}u + column])) { return false; } }
  return true;
}
fn predictor(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return predictors[predictorsOffset + row * ${e.predictorCount}u + column - 1u];
}
${i}`}function Z(e){let{status:t,...n}=e;return{...n,status:t.codes,iterations:t.iterationCount}}export{X as a,Y as i,U as n,J as o,q as r,G as t};