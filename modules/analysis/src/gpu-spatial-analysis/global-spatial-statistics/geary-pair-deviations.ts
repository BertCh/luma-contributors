// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCompaction,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {
  getGPUSpatialWeightsChunkDispatches,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  isGPUSpatialWeightsRowStandardized,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';

/**
 * Stable Geary moments on complete symmetric support. With S = (W + W')/2,
 * D = S - a*(J-I), the uniform part is permutation invariant and
 * C(W)-1 = ((n-1)*sum(D_ij*(xi-xj)^2)/(2*M2) - sum(D))/sum(W).
 * Evaluate the variance polynomial on D/sum(W), normalizing before squaring to
 * preserve small weight scales. Publish C as 1 + (C-1) when |C-1| < 1/2, which keeps
 * invariant weights at exactly 1, and otherwise from its nonnegative quadratic
 * numerator, which keeps a tiny positive C; z-scores always use the centered C-1.
 * The pass is gated on complete input (finite positive sum(W)), at least four rows,
 * complete support and nonconstant labels, not on the preliminary C: that ratio can
 * underflow to 0/0 at small weight scales (R19 W5-17 review B3).
 * D=0 then gives exact zero; small D never subtracts two order-one moments.
 * Non-complete support keeps the ordinary result written by the caller.
 *
 * Pair sums retain high/low f32 parts so a half-ulp asymmetric perturbation
 * cannot disappear. Each stored direction contributes once; a pair with only
 * one stored direction contributes twice. The exact union count gates the
 * complete-support decision, independently of rounded weight moments.
 * @internal
 */
export function getGearyPairDeviationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
    values: GraphDataView<'float32'>;
    centered: GraphDataView<'float32'>;
    centeredStride: number;
    rowScale: GraphDataView<'float32'>;
    counters: GraphDataView<'uint32'>;
    summary: GraphDataView<'float32'>;
    pValueDeclarations: string;
  }
): GPUCommandNode<Parameters>[] {
  const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights),
    rows = props.values.length;
  const capacity = dispatches.reduce(
    (total, dispatch) => total + dispatch.weights.neighbors.length,
    0
  );
  const id = `${props.id}-geary-deviations`;
  const scratch = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = rows
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const state = createTransientView(
    graph,
    `${id}-state`,
    'uint32',
    64,
    Buffer.STORAGE | Buffer.UNIFORM
  );
  const stateWord = <Format extends 'float32' | 'uint32'>(word: number, format: Format) =>
    graph.createDataView(state.buffer, {format, byteOffset: word * 4, length: 1});
  const scales = scratch('scales', 'float32'),
    flags = scratch('flags', 'uint32'),
    rowIds = scratch('row-ids', 'uint32'),
    validRows = scratch('valid-rows', 'uint32');
  const squareTerms = scratch('squares', 'float32'),
    fourthTerms = scratch('fourths', 'float32'),
    candidates = scratch('candidates', 'float32'),
    unionTerms = scratch('union-terms', 'uint32');
  const outTerms = scratch('out', 'float32'),
    s1Terms = scratch('s1', 'float32'),
    s2Terms = scratch('s2', 'float32'),
    quadTerms = scratch('quadratic', 'float32'),
    observedTerms = scratch('observed', 'float32'),
    incoming = scratch('incoming', 'float32');
  const totals = scratch('totals', 'float32', 5);
  const high = dispatches.map((dispatch, index) =>
    scratch(`high-${index}`, 'float32', dispatch.weights.neighbors.length)
  );
  const low = dispatches.map((dispatch, index) =>
    scratch(`low-${index}`, 'float32', dispatch.weights.neighbors.length)
  );
  const factors = dispatches.map((dispatch, index) =>
    scratch(`factors-${index}`, 'uint32', dispatch.weights.neighbors.length)
  );
  const delta = dispatches.map((dispatch, index) =>
    scratch(`delta-${index}`, 'float32', dispatch.weights.neighbors.length)
  );
  const bind = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, access, type: view.format === 'uint32' ? 'u32' : 'f32'});
  const common = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}\n${ADDITION_ROUNDING_ERROR_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${SCALED_QUOTIENT_WGSL}
fn controlFloat(word:u32)->f32{return bitcast<f32>(getStatisticsControl(word));}
fn pairDelta(high:f32,low:f32)->f32{return getRoundedDifference(high,controlFloat(3u),getStatisticsControl(8u))+getRoundedDifference(low,controlFloat(4u),getStatisticsControl(8u));}`;
  const node = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    parameters: number[] = [],
    declarations = ''
  ) => {
    const writing = bindings.some(
      binding => binding.view === state && binding.access === 'read_write'
    );
    return createSpatialStatisticsCollectiveNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUGlobalSpatialStatistics',
      variant: name,
      invocationCount: count,
      bindings,
      control: writing ? undefined : state,
      parameterValues: parameters,
      declarations: `${writing ? 'fn getStatisticsControl(word:u32)->u32{return controlState[controlStateOffset+word];}' : ''}\n${common}\n${declarations}`,
      body: `if(index>=INVOCATION_COUNT){return;}\n${body}`
    });
  };
  const reduce = <Format extends 'float32' | 'uint32'>(
    name: string,
    input: GraphDataView<Format>,
    output: GraphDataView<Format>,
    operation: 'min' | 'max' | 'sum'
  ) => new GPUReduction({id: `${id}-${name}`, input, output, operation}).getCommandNodes(graph);
  const nodes: GPUCommandNode<Parameters>[] = [
    createFillNode(graph, {
      id: `${id}-clear`,
      operation: 'GPUGlobalSpatialStatistics',
      view: state,
      type: 'u32',
      value: '0u'
    }),
    node(
      'initialize',
      1,
      [
        bind('counters', props.counters),
        bind('summary', props.summary),
        bind('controlState', state, 'read_write')
      ],
      `let n=counters[countersOffset];let pairs=multiplyUint32Wide(n,max(n,1u)-1u);
controlState[controlStateOffset]=n;controlState[controlStateOffset+1u]=counters[countersOffset+5u];controlState[controlStateOffset+2u]=bitcast<u32>(summary[summaryOffset+1u]);
controlState[controlStateOffset+5u]=u32(n>=4u&&pairs.x==0u&&pairs.y<=${capacity * 2}u&&isFiniteFloat(summary[summaryOffset+1u])&&summary[summaryOffset+1u]>0.0);`
    ),
    node(
      'rows',
      rows,
      [
        bind('centered', props.centered),
        ...(isGPUSpatialWeightsRowStandardized(props.weights)
          ? [bind('rowScale', props.rowScale)]
          : []),
        bind('scales', scales, 'read_write'),
        bind('flags', flags, 'read_write'),
        bind('rowIds', rowIds, 'read_write'),
        bind('squareTerms', squareTerms, 'read_write'),
        bind('fourthTerms', fourthTerms, 'read_write')
      ],
      `let value=centered[centeredOffset+index*${props.centeredStride}u];let valid=isFiniteFloat(value);
scales[scalesOffset+index]=select(0.0,${isGPUSpatialWeightsRowStandardized(props.weights) ? 'select(-1.0,rowScale[rowScaleOffset+index],rowScale[rowScaleOffset+index]>0.0)' : '1.0'},valid);flags[flagsOffset+index]=u32(valid);rowIds[rowIdsOffset+index]=index;
let square=value*value;squareTerms[squareTermsOffset+index]=select(0.0,square,valid);fourthTerms[fourthTermsOffset+index]=select(0.0,square*square,valid);`
    ),
    ...new GPUCompaction({
      id: `${id}-compact`,
      input: rowIds,
      flags,
      output: validRows,
      count: stateWord(0, 'uint32')
    }).getCommandNodes(graph),
    ...reduce('squares', squareTerms, stateWord(6, 'float32'), 'sum'),
    ...reduce('fourths', fourthTerms, stateWord(7, 'float32'), 'sum')
  ];
  for (const [firstIndex, first] of dispatches.entries())
    for (const [secondIndex, second] of dispatches.entries())
      for (const field of ['high', 'low', 'factor'] as const) {
        const output =
          field === 'high'
            ? high[firstIndex]
            : field === 'low'
              ? low[firstIndex]
              : factors[firstIndex];
        nodes.push(
          node(
            `pairs-${firstIndex}-${secondIndex}-${field}`,
            first.rowCount,
            [
              bind('offsets', first.weights.offsets),
              bind('neighbors', first.weights.neighbors),
              bind('weights', first.weights.weights),
              bind('targetOffsets', second.weights.offsets),
              bind('targetNeighbors', second.weights.neighbors),
              bind('targetWeights', second.weights.weights),
              bind('scales', scales),
              bind('pairOutput', output, 'read_write')
            ],
            `if(getStatisticsControl(5u)==0u){return;}let row=index+getChunkParameter(36u);let targetStart=getChunkParameter(37u);let targetEnd=targetStart+getChunkParameter(38u);
for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){
 let neighbor=neighbors[neighborsOffset+slot];if(neighbor<targetStart||neighbor>=targetEnd){continue;}
 var firstWeight=0.0;var reverse=0.0;
 let raw=weights[weightsOffset+slot];let valid=row!=neighbor&&scales[scalesOffset+row]!=0.0&&scales[scalesOffset+neighbor]!=0.0&&raw>0.0&&isFiniteFloat(raw);
 if(valid){firstWeight=raw*max(0.0,scales[scalesOffset+row]);var begin=targetOffsets[targetOffsetsOffset+neighbor-targetStart];let end=targetOffsets[targetOffsetsOffset+neighbor-targetStart+1u];var finish=end;
 while(begin<finish){let middle=begin+(finish-begin)/2u;if(targetNeighbors[targetNeighborsOffset+middle]<row){begin=middle+1u;}else{finish=middle;}}
 if(begin<end&&targetNeighbors[targetNeighborsOffset+begin]==row){let weight=targetWeights[targetWeightsOffset+begin];if(weight>0.0&&isFiniteFloat(weight)){reverse=weight*max(0.0,scales[scalesOffset+neighbor]);}}}
 let firstHalf=firstWeight*0.5;let secondHalf=reverse*0.5;let sum=getRoundedDifference(firstHalf+secondHalf,0.0,getStatisticsControl(8u));
 pairOutput[pairOutputOffset+slot]=${field === 'high' ? 'select(0.0,sum,valid)' : field === 'low' ? 'select(0.0,getAdditionRoundingError(firstHalf,secondHalf,sum),valid)' : 'select(0u,select(2u,1u,reverse>0.0),valid)'};
}`,
            [first.rowStart, second.rowStart, second.rowCount]
          )
        );
      }
  for (const [chunk, dispatch] of dispatches.entries())
    nodes.push(
      node(
        `metadata-${chunk}`,
        dispatch.rowCount,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('high', high[chunk]),
          bind('factors', factors[chunk]),
          bind('candidates', candidates, 'read_write'),
          bind('unionTerms', unionTerms, 'read_write')
        ],
        `let row=index+getChunkParameter(36u);var minimum=3.402823466e38;var count=0u;
if(getStatisticsControl(5u)!=0u){for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){let factor=factors[factorsOffset+slot];if(factor!=0u){minimum=min(minimum,high[highOffset+slot]);count+=factor;}}}
candidates[candidatesOffset+row]=select(-3.402823466e38,minimum,count>0u);unionTerms[unionTermsOffset+row]=count;`,
        [dispatch.rowStart]
      )
    );
  nodes.push(
    ...reduce('anchor-high', candidates, stateWord(3, 'float32'), 'max'),
    ...reduce('union', unionTerms, stateWord(9, 'uint32'), 'sum'),
    node(
      'ready',
      1,
      [bind('controlState', state, 'read_write')],
      `let n=getStatisticsControl(0u);let pairs=multiplyUint32Wide(n,max(n,1u)-1u);controlState[controlStateOffset+5u]=u32(getStatisticsControl(5u)!=0u&&pairs.x==0u&&pairs.y==getStatisticsControl(9u)&&controlFloat(6u)>0.0&&isFiniteFloat(controlFloat(6u)));`
    )
  );
  for (const [chunk, dispatch] of dispatches.entries())
    nodes.push(
      node(
        `anchor-low-${chunk}`,
        dispatch.rowCount,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('high', high[chunk]),
          bind('low', low[chunk]),
          bind('factors', factors[chunk]),
          bind('candidates', candidates, 'read_write')
        ],
        `let row=index+getChunkParameter(36u);var minimum=3.402823466e38;var found=false;
if(getStatisticsControl(5u)!=0u){for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){if(factors[factorsOffset+slot]!=0u&&high[highOffset+slot]==controlFloat(3u)){minimum=min(minimum,low[lowOffset+slot]);found=true;}}}
candidates[candidatesOffset+row]=select(-3.402823466e38,minimum,found);`,
        [dispatch.rowStart]
      )
    );
  nodes.push(
    ...reduce('anchor-low', candidates, stateWord(4, 'float32'), 'max'),
    createFillNode(graph, {
      id: `${id}-clear-incoming`,
      operation: 'GPUGlobalSpatialStatistics',
      view: incoming,
      type: 'f32',
      value: '0.0'
    })
  );
  for (const [chunk, dispatch] of dispatches.entries()) {
    nodes.push(
      node(
        `moments-${chunk}`,
        dispatch.rowCount,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('high', high[chunk]),
          bind('low', low[chunk]),
          bind('factors', factors[chunk]),
          bind('delta', delta[chunk], 'read_write'),
          bind('outTerms', outTerms, 'read_write'),
          bind('s1Terms', s1Terms, 'read_write')
        ],
        `let row=index+getChunkParameter(36u);var sum=0.0;var squares=0.0;
for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){var value=0.0;let factor=factors[factorsOffset+slot];if(getStatisticsControl(5u)!=0u&&factor!=0u){value=getScaledQuotient(pairDelta(high[highOffset+slot],low[lowOffset+slot]),controlFloat(2u),1.0,0);}let weighted=f32(factor)*value;delta[deltaOffset+slot]=weighted;sum+=weighted;squares+=2.0*f32(factor)*value*value;}
outTerms[outTermsOffset+row]=sum;s1Terms[s1TermsOffset+row]=squares;`,
        [dispatch.rowStart]
      )
    );
    nodes.push(
      node(
        `quadratic-${chunk}`,
        dispatch.rowCount,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('neighbors', dispatch.weights.neighbors),
          bind('delta', delta[chunk]),
          bind('values', props.values),
          bind('quadTerms', quadTerms, 'read_write')
        ],
        `let row=index+getChunkParameter(36u);var sum=0.0;
if(getStatisticsControl(5u)!=0u){for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){let weighted=delta[deltaOffset+slot];if(weighted==0.0){continue;}let neighbor=neighbors[neighborsOffset+slot];let difference=getRoundedDifference(values[valuesOffset+row]*controlFloat(1u),values[valuesOffset+neighbor]*controlFloat(1u),getStatisticsControl(8u));let normalized=getScaledQuotient(difference,sqrt(controlFloat(6u)/f32(getStatisticsControl(0u))),1.0,0);sum+=weighted*normalized*normalized;}}
quadTerms[quadTermsOffset+row]=sum;`,
        [dispatch.rowStart]
      )
    );
    nodes.push(
      node(
        `observed-${chunk}`,
        dispatch.rowCount,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('neighbors', dispatch.weights.neighbors),
          bind('high', high[chunk]),
          bind('low', low[chunk]),
          bind('factors', factors[chunk]),
          bind('values', props.values),
          bind('observedTerms', observedTerms, 'read_write')
        ],
        `let row=index+getChunkParameter(36u);var sum=0.0;
if(getStatisticsControl(5u)!=0u){for(var slot=offsets[offsetsOffset+index];slot<offsets[offsetsOffset+index+1u];slot++){let factor=factors[factorsOffset+slot];if(factor==0u){continue;}let neighbor=neighbors[neighborsOffset+slot];let difference=getRoundedDifference(values[valuesOffset+row]*controlFloat(1u),values[valuesOffset+neighbor]*controlFloat(1u),getStatisticsControl(8u));let normalized=getScaledQuotient(difference,sqrt(controlFloat(6u)/f32(getStatisticsControl(0u))),1.0,0);let weight=getScaledQuotient(high[highOffset+slot],controlFloat(2u),1.0,0)+getScaledQuotient(low[lowOffset+slot],controlFloat(2u),1.0,0);sum+=f32(factor)*weight*normalized*normalized;}}
observedTerms[observedTermsOffset+row]=sum;`,
        [dispatch.rowStart]
      )
    );
    nodes.push(
      node(
        `incoming-${chunk}`,
        rows,
        [
          bind('offsets', dispatch.weights.offsets),
          bind('neighbors', dispatch.weights.neighbors),
          bind('delta', delta[chunk]),
          bind('validRows', validRows),
          bind('scales', scales),
          bind('incoming', incoming, 'read_write')
        ],
        `if(getStatisticsControl(5u)==0u||scales[scalesOffset+index]==0.0){return;}let targetStart=getChunkParameter(36u);let targetEnd=targetStart+getChunkParameter(37u);let n=getStatisticsControl(0u);var low=0u;var high=n;
while(low<high){let middle=low+(high-low)/2u;if(validRows[validRowsOffset+middle]<targetStart){low=middle+1u;}else{high=middle;}}
var sum=0.0;for(var position=low;position<n;position++){let row=validRows[validRowsOffset+position];if(row>=targetEnd){break;}if(row==index){continue;}var begin=offsets[offsetsOffset+row-targetStart];let end=offsets[offsetsOffset+row-targetStart+1u];var finish=end;
while(begin<finish){let middle=begin+(finish-begin)/2u;if(neighbors[neighborsOffset+middle]<index){begin=middle+1u;}else{finish=middle;}}if(begin<end&&neighbors[neighborsOffset+begin]==index){sum+=delta[deltaOffset+begin];}}
incoming[incomingOffset+index]+=sum;`,
        [dispatch.rowStart, dispatch.rowCount]
      )
    );
  }
  nodes.push(
    node(
      'degrees',
      rows,
      [
        bind('outTerms', outTerms),
        bind('incoming', incoming),
        bind('s2Terms', s2Terms, 'read_write')
      ],
      `let degree=outTerms[outTermsOffset+index]+incoming[incomingOffset+index];s2Terms[s2TermsOffset+index]=degree*degree;`
    )
  );
  for (const [index, terms] of [outTerms, s1Terms, s2Terms, quadTerms, observedTerms].entries())
    nodes.push(
      ...reduce(
        `sum-${index}`,
        terms,
        graph.createDataView(totals.buffer, {format: 'float32', byteOffset: index * 4, length: 1}),
        'sum'
      )
    );
  nodes.push(
    node(
      'finalize',
      1,
      [bind('totals', totals), bind('results', props.summary, 'read_write')],
      `if(getStatisticsControl(5u)==0u){return;}let n=f32(getStatisticsControl(0u));let d0=totals[totalsOffset];let r0=d0;let r1=totals[totalsOffset+1u];let r2=totals[totalsOffset+2u];let kappa=n*(controlFloat(7u)/controlFloat(6u))/controlFloat(6u);
let normal=((2.0*r1+r2)*(n-1.0)-4.0*r0*r0)/(2.0*(n+1.0));
let random=((n-1.0)*r1*(n*n-3.0*n+3.0-(n-1.0)*kappa)-0.25*(n-1.0)*r2*(n*n+3.0*n-6.0-(n*n-n+2.0)*kappa)+r0*r0*(n*n-3.0-(n-1.0)*(n-1.0)*kappa))/(n*(n-2.0)*(n-3.0));
let difference=(n-1.0)/(2.0*n)*totals[totalsOffset+3u]-d0;var zn=getQuietNaN(0u);var zr=getQuietNaN(0u);if(normal>0.0){zn=difference/sqrt(normal);}if(random>0.0){zr=difference/sqrt(random);}
let observed=(n-1.0)/(2.0*n)*totals[totalsOffset+4u];
${['select(observed,1.0+difference,abs(difference)<0.5)', '1.0', 'normal', 'zn', 'getPValue(zn)', 'random', 'zr', 'getPValue(zr)'].map((value, index) => `results[resultsOffset+${16 + index}u]=${value};`).join('\n')}`,
      [],
      props.pValueDeclarations
    )
  );
  return nodes;
}
