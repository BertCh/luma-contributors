# Output ports, row domains, tails and sentinels

## Summary

**A caller cannot safely write a generic reader using `output.status.count` and `output.validity`.** Of 498 public producers, exactly one declares both paths. On that producer, `GPUTrajectoryThinning`, the count indexes the compact output and the validity indexes the original input. The obvious reader silently drops a correct result.

Three issues merit fixes before alpha. `GPUContiguityWeights` advertises a distance buffer that it never writes, and a downstream transpose accepts and reads it. Several public decoders accept the bytes returned by `Buffer.readAsync()` as if each byte were a 32-bit word. `GPUShapeGenerator` tells callers to reuse vertex offsets as polygon offsets, which made the second of two valid polygons disappear in a CPU probe.

The most valuable change is to make each output describe **which rows it addresses, how many are readable, and which producer writes it**. The existing names and scalar formats alone cannot answer those questions. This can be checked against constructor types and declared graph resources before running a shader.

## Census

`S/` means `modules/analysis/src/`, `T/` means `modules/analysis/test/`, and `D/` means `docs/api/`. Paths marked **KI** were read in `luma-contributors-ki` because they occur in `.typecheck/r19/staging-owned.txt`. Other citations refer to `luma-contributors-r19`.

**COUNT.** I resolved the 11 source barrels corresponding to `modules/analysis/package.json#exports` with the installed TypeScript API, deduplicated class symbols, selected classes with an instance `getCommandNodes`, and traversed their constructor types. The compiler host read staging-owned files from KI, including new imported files. I expanded inherited fields, generic constraints, unions and arrays. This matters for chunked transpose, Gaussian scores, voxel merge and the four inherited raster indices. I ran no compiler emit or typecheck.

| Public surface measured | Classes |
|---|---:|
| Public command-node producers | 498 |
| Named map at `props.output` | 469 |
| A view directly at `props.output` | 18 |
| An array directly at `props.output` | 1 |
| No `props.output` | 10 |
| Map with at least one optional top-level member | 333 |
| Map with every top-level member optional in the structural type | 81 |
| An additional or alternative `*.output` below a constructor prop | 15 |

The 81 are a type count. Some constructors require at least one output or a particular combination. A generic allocator cannot infer legal combinations from `?`.

| Source directory of declaration | Producers | With `props.output` | View paths under `props.output` | Optional or conditional paths |
|---|---:|---:|---:|---:|
| `analysis-core` | 6 | 6 | 35 | 17 |
| `computation-record` | 2 | 0 | 0 | 0 |
| `dataframe-analysis` | 34 | 34 | 176 | 119 |
| `geospatial-analysis` | 2 | 1 | 3 | 1 |
| `gpu-network` | 41 | 39 | 310 | 208 |
| `gpu-spatial-analysis` | 270 | 269 | 2,001 | 1,191 |
| `gpu-terrain` | 46 | 46 | 208 | 155 |
| `pointcloud-analysis` | 11 | 11 | 42 | 4 |
| `raster-analysis` | 86 | 82 | 384 | 210 |
| **Total** | **498** | **488** | **3,159** | **1,905** |

The unit in the last two columns is a producer, qualified member path and format, deduplicated across type branches. Optional includes an optional ancestor or a union branch without the member. These paths have **1,392 distinct spellings**. The most frequent formats are scalar `uint32` **2,029**, scalar `float32` **889**, `float32x2` **105**, `float32x4` **41**, `uint32x2` **35**, `float32x3` **14**, `uint32x4` **13**, scalar `sint32` **6**, and `uint32x3` **4**. The remaining **23** have texture, union or generic formats. The 15 classes with separately nested output destinations add **24 named `*.output` paths**. They are listed below.

| Reader-visible variant | Classes | Interpretation |
|---|---:|---|
| Top-level `output.status` is a view | 94 | Scalar, per-row codes and packed words all occur |
| Top-level `output.status` is a map | 200 | The map can contain batch words or row-aligned views |
| Top-level `output.status.count` | 58 | 14 optional or conditional |
| `status.count` anywhere under `output` | 93 | Includes separate datasets and inherited declarations |
| Top-level `status.count` and `status.overflow` | 54 | 48 also declare `status.requiredCount` |
| Top-level `counts` | 47 | 18 optional or conditional; four have vector element formats |
| Top-level `count` / `rowCount` / `resultCount` | 3 / 1 / 0 | The three `count` fields are not a library-wide compact boundary |
| Top-level `validity` / `outputValidity` | 34 / 19 | 15 / 17 optional or conditional |
| Top-level `mask` / `outputMask` | 4 / 6 | 2 / 6 optional or conditional |
| `status.count` plus `validity`, both immediately under their usual parent | **1** | `GPUTrajectoryThinning`, with different row domains |
| A view named `ids` or ending in `Ids` / `rows` or ending in `Rows` | 39 / 37 | Names alone cannot establish an application-ID or source-row domain |
| A view named `offsets` or ending in `Offsets` | 64 | Includes physical offsets such as `signedOffsets`; 53 have a matching field comment mentioning a terminal or `+ 1` |

I also counted what a reader can recover **from the resolved output members’ TSDoc**. Comments on at least one member mention rows, length, entries, slots, capacity or “per” on **436** producers. They mention row order, major order or sorting on **104**, `NaN` on **167**, `0xffffffff` on **140**, and `-1` on **11**. A tail, padding, unused rows, unspecified rows or unwritten rows are mentioned on **66**. These overlap. A missing word is an unclassified contract, not evidence that the shader zero-fills. The differences verified below include zero-filled tails, sentinel-filled tails, stale tails, repeated terminal offsets, and a finite float that looks like data.

For reproduction, the class-selection and port-enumeration core is below. Run as the body of `node -e` from the checkout. It prints the producer, path, format, optionality and available row/layout commentary, so an implementer can recover the names behind the counts without copying a 3,159-row table from this report.

```js
const ts = require('typescript'), fs = require('fs'), path = require('path');
const owned = new Set(fs.readFileSync('.typecheck/r19/staging-owned.txt', 'utf8').split('\n'));
const ki = path.resolve('../luma-contributors-ki');
function source(file) {
  const relative = path.relative(process.cwd(), path.resolve(file));
  const staged = path.join(ki, relative);
  return owned.has(relative) && fs.existsSync(staged) ? staged : file;
}
const options = {
  strict: true, noEmit: true, target: ts.ScriptTarget.ESNext,
  module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler
};
const host = ts.createCompilerHost(options);
host.fileExists = f => ts.sys.fileExists(source(f));
host.readFile = f => ts.sys.readFile(source(f));
host.directoryExists = d => ts.sys.directoryExists(d) ||
  ts.sys.directoryExists(path.join(ki, path.relative(process.cwd(), d)));
const entries = Object.keys(require('./modules/analysis/package.json').exports)
  .map(e => path.resolve('modules/analysis/src', e, 'index.ts'));
const program = ts.createProgram(entries, options, host), checker = program.getTypeChecker();
const seen = new Set();
for (const file of entries) {
  const mod = checker.getSymbolAtLocation(program.getSourceFile(file));
  for (let symbol of checker.getExportsOfModule(mod)) {
    if (symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
    if (seen.has(symbol)) continue;
    seen.add(symbol);
    const decl = symbol.valueDeclaration;
    if (!decl || !ts.isClassDeclaration(decl) ||
        !checker.getDeclaredTypeOfSymbol(symbol).getProperty('getCommandNodes')) continue;
    const ctor = checker.getTypeOfSymbolAtLocation(symbol, decl).getConstructSignatures()[0];
    const props = checker.getTypeOfSymbolAtLocation(ctor.parameters[0], decl);
    const output = props.getProperty('output'), ports = new Map();
    function walk(type, key, optional, doc, depth = 0) {
      type = checker.getNonNullableType(type);
      if (type.flags & ts.TypeFlags.TypeParameter) type = checker.getBaseConstraintOfType(type) || type;
      if (depth > 6 || type.flags & (ts.TypeFlags.StringLike | ts.TypeFlags.NumberLike |
          ts.TypeFlags.BooleanLike)) return;
      const format = checker.getTypeOfPropertyOfType(type, 'format');
      if ((format && type.getProperty('length')) ||
          (type.getProperty('texture') && type.getProperty('dimension'))) {
        const name = key || '(self)', fmt = format ? checker.typeToString(format) : 'texture';
        ports.set(name + ' ' + fmt, [name, fmt, optional, doc]);
        return;
      }
      const element = checker.getIndexTypeOfType(type, ts.IndexKind.Number);
      if (element && type.getProperty('slice') && !type.getProperty('charAt')) {
        walk(element, key + '[]', optional, doc, depth + 1);
        return;
      }
      if (type.getCallSignatures().length || type.getProperty('byteLength') ||
          type.getProperty('usage') || /(^|\.)(spatialContext|information)$/.test(key)) return;
      const variants = type.isUnion() ? type.types : [type];
      const names = new Set(variants.flatMap(v => v.getProperties().map(p => p.name)));
      for (const name of names) {
        const conditional = variants.some(v => !v.getProperty(name) ||
          Boolean(v.getProperty(name).flags & ts.SymbolFlags.Optional));
        for (const variant of variants) {
          const member = variant.getProperty(name);
          if (member) walk(checker.getTypeOfPropertyOfType(variant, name),
            key ? key + '.' + name : name, optional || conditional,
            ts.displayPartsToString(member.getDocumentationComment(checker)), depth + 1);
        }
      }
    }
    if (output) walk(checker.getTypeOfSymbolAtLocation(output, decl), '',
      Boolean(output.flags & ts.SymbolFlags.Optional),
      ts.displayPartsToString(output.getDocumentationComment(checker)));
    console.log(JSON.stringify([symbol.name, decl.getSourceFile().fileName, [...ports.values()]]));
  }
}
```

I used a second traversal over analysis-owned constructor members to locate nested properties literally named `output`, including array elements. Host reproductions used `esbuild --bundle --platform=node --format=esm --loader=ts` on stdin, piped to `node --input-type=module -`. Resource probes used a null device and stopped after `getCommandNodes`.

## Proposed convention

For each writable port, declare its **element format, row domain and order, readable extent, tail policy, and missing-value or validity rule**. `status.count` bounds the readable compact prefix, `…Rows` indexes a named source, `…Ids` carries application identities, and an unqualified `validity` applies to the primary result in its own row domain. Check declarations against public types, then bind a distinct destination to every advertised optional port and require a writer, an explicitly declared pass-through, or a constructor rejection.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| OUT-01 | W | ORDINARY | ADDITIVE | S | Contiguity → transpose | A graph consumes a distance column with no producer. |
| OUT-02 | W | ORDINARY | ADDITIVE | S | Packed readback decoders | Valid `readAsync()` bytes become plausible wrong counts, medians and angles. |
| OUT-03 | W | ORDINARY | ADDITIVE | S | Shape generator | Following the offset type comment loses the second polygon. |
| OUT-04 | W | ORDINARY | ADDITIVE | M | Spatial weights | Three producers accept and advertise seven status destinations they never bind. |
| OUT-05 | W | ORDINARY | ADDITIVE | S | Network statistics | Omitting a repeated histogram dimension silently moves two histogram boundaries. |
| OUT-06 | W | UNUSUAL | ADDITIVE | S | Spherical, subgraph and horizon readers | A short read can become `inside`, `valid`, or terrain nodata. |
| OUT-07 | W | EXTREME | ADDITIVE | S | Point-cloud neighborhoods | A real neighbor can compare worse than an empty slot. |
| OUT-08 | C | ORDINARY | ADDITIVE | S | Trajectory thinning | Applying `validity[i]` to compact row `i` drops an endpoint. |
| OUT-09 | C | ORDINARY | ADDITIVE | S | Streamlines | Even with zero overflow, the full offsets view can descend. |
| OUT-10 | C | ORDINARY | ADDITIVE | S | Path extraction | A found path can have no readable node rows; an all-ones edge ID can be real. |
| OUT-11 | C | ORDINARY | BREAKING | S | Pair gather | The consumer of a counted pair table cannot publish the same pair-table shape. |
| OUT-12 | C | ORDINARY | ADDITIVE | S | Accessibility, Huff | `validity` can be zero for a usable primary result. |
| OUT-13 | C | ORDINARY | ADDITIVE | S | Polygon point sampling | `slotOffsets` contains starts without a terminal, before final capacity clamping. |
| OUT-14 | E | ORDINARY | ADDITIVE | S | Alternative routes | Two top-k route outputs use the same five columns but require different completion readers. |

### OUT-01: A contiguity result advertises an unwritten distance column

**Where.** **KI** `S/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights.ts:131,265,297`, `spatial-weights.ts:328`, `spatial-weights-transpose-pipeline.ts:418`.

**EXECUTED.** I gave queen contiguity two polygon features, eight vertices, two CSR slots and a distinct `float32` destination for distances. I passed `contiguity.outputWeights` directly to `GPUSpatialWeightsTranspose`. Uses of that exact destination were `[]` in the producer and `[['spatial-weights-transpose-scatter','storage-read']]` in the consumer. `contiguity.outputWeights.distances === suppliedDistances` was `true`. The producer’s own comment says distances are unwritten. This is an observable host wiring error.

**Now.** `new GPUSpatialWeightsTranspose({weights: contiguity.outputWeights, output})` can publish caller-initialized bytes as distances. **Wanted.** Reject `output.distances` for ordinary contiguity, omit it from the returned weights description, and say that binary adjacency supplies no metric distance. Fuzzy contiguity and lattice weights genuinely write the field and should retain it.

**Callers, value and cost.** No valid distance-producing call changes. Generic allocators supplying a spare distance destination must omit it here. A topology user gains a reliable presence check. **S.**

### OUT-02: Byte readbacks pass public word-decoder signatures

**Where.** `S/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters.ts:107`, `network-coarsening/gpu-network-coarsening.ts:112`, `network-statistics/gpu-network-statistics.ts:148`; `S/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.ts:96`, `global-coordinates/spherical-status.ts:51,56`; `S/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:152`. A working byte-oriented precedent is **KI** `S/gpu-spatial-analysis/region-statistics/region-statistics-readback.ts:22`.

**Now.** The subgraph guide says to decode `counts` with the named function. The following is type-correct:

```ts
const counts = decodeGPUNetworkSubgraphFilterCounts(await countsBuffer.readAsync());
```

**EXECUTED.** For words `[257,2,3,4]`, the decoder returned `[257,2,3,4]` from `Uint32Array` and `[1,1,0,0]` from a `Uint8Array` over the identical storage. Network statistics likewise decoded `liveNodeCount: 257` as `1`. The contiguity reader decoded a float median of `2.5` as `0` and still returned `exact: true`. The horizon unpacker decoded word `0xffffffff` as `[90,90]` from words and `[-89.30234687337871,NaN]` from bytes. I also reproduced byte/word disagreement in both spherical count readers and the coarsening reader. The network statistics example explicitly constructs a word view; that safeguard is confined to the example. **KI** `D/gpu-network/operators.md:421`.

**Wanted.** Add explicit `…FromBytes` entry points or `{bytes}` overloads that honor `byteOffset`, `byteLength`, scalar format and endianness. Show the one-line safe read on each relevant page. Preserve ordinary numeric-array fixtures. Avoid silently changing the interpretation of existing callers that deliberately pass an array of numeric words.

**Callers, value and cost.** None must migrate for the additive entry point. First-time users can decode the actual object returned by the GPU API. **S** for a shared internal conversion and these readers.

### OUT-03: Shape-generator offsets index vertices, not rings

**Where.** `S/gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:125,567`; **KI** `S/analysis-core/contracts/ports.ts:120`; **KI** `D/gpu-spatial-analysis/operators-geometry.md:1093`.

**EXECUTED.** Two four-segment closed rings have vertex offsets `[0,5,10]`. Following the exported comment that these are “both the ring and the polygon offsets”, I ran:

```ts
computeLabelPointOnCPU({
  positions, featureOffsets: Uint32Array.of(0, 1, 2),
  polygonOffsets: offsets, ringOffsets: offsets
});
```

For two unit diamonds centered at `(0,0)` and `(10,0)`, the result was `points=[0,0,NaN,NaN]`, `degenerate=[0,1]`. With `polygonOffsets=[0,1,2]`, the same consumer returned `points=[0,0,10,0]`, `degenerate=[0,0]`. The offsets and the label-point calculation are independently checkable on the CPU.

**Wanted.** Replace the type comment and add the construction to the guide: “Use `output.offsets` as `ringOffsets`. For one ring and one polygon per feature, both `polygonOffsets` and `featureOffsets` are `[0,1,…,featureCount]`.” A two-feature example catches the error better than one feature.

**Callers, value and cost.** None. GIS callers can feed generated rings to another operator without losing features. **S.**

### OUT-04: Writable weights inherit status fields that have no writer

**Where.** **KI** `S/analysis-core/contracts/ports.ts:230`, `S/gpu-spatial-analysis/spatial-weights/spatial-weights.ts:251,304`, `gpu-lattice-weights.ts:237`, `gpu-contiguity-weights.ts:283`; R19 `gpu-fuzzy-contiguity.ts:184`. The packed transpose takes the opposite approach at **KI** `spatial-weights-transpose-pipeline.ts:87`.

**EXECUTED.** I supplied ten distinct one-word destinations in `output.status`. Lattice, ordinary contiguity and fuzzy contiguity each bound `overflow`, `requiredCount`, and `candidateOverflow`. Across **10, 23 and 50 declared nodes** respectively, none bound `count`, `converged`, `iterationLimitReached`, `iterationCount`, `approximation`, `invalidCount`, or `uncertainCount`. For comparison, packed transpose rejected a destination status with `output.status is unsupported for packed transpose because destination diagnostics are not written`.

**Now.** `const usedSlots = (await read(weights.outputWeights.status.count))[0]` can read an initializer. The output type acquires those fields by intersecting a readable weights description with a writable status type. An intersection preserves the other optional members.

**Wanted.** Define the writable destination from `Omit<GPUSpatialWeights, 'status'>` and add the producer-specific status map. For these three builders, support optional `status.count` by publishing the terminal CSR offset. Reject unsupported diagnostics with their qualified names. Keep input and inherited status descriptions separate from destination capabilities. `GPUIslandAttachment` already provides a real optional count writer at **KI** `gpu-island-attachment.ts:763`.

**Callers, value and cost.** Valid existing status uses continue to work. Reused buffers no longer turn old diagnostics into current results. **M** to cover all builders and their operation forms. This adds a destination-writer test beyond the referenced-member vocabulary gap in `KNOWN(ERG-104)`.

### OUT-05: A network summary can decode with the wrong histogram stride

**Where.** `S/gpu-network/network-statistics/gpu-network-statistics.ts:77,148,312,345`.

**EXECUTED.** I allocated the exact output for 40 bins, `16 + 3*40 = 136` words, and filled the histogram block with `100,101,…,219`. `decodeGPUNetworkStatistics(words).inDegreeHistogram[0]` returned **132**. Supplying `{degreeBinCount:40}` returned **140**, the first in-degree bin. The default decoder accepted the longer output and read the last eight out-degree bins as in-degree bins. Separately, `getGPUNetworkStatisticsLength(-1)` returned `13`, and a decoder call with `degreeBinCount: NaN` returned empty histograms. The producer accepts only integer bin counts in `[1,4096]`.

**Wanted.** Decode a precisely bounded summary view, infer omitted bin count from `(wordLength - 16) / 3`, validate the same range as the constructor, and require an explicit dimension to agree with the view. For a larger backing buffer, take the subview before decoding. The constructor already requires an exact-size output.

**Callers, value and cost.** Correctly sized summaries decoded with their actual dimensions need no change. Callers get correct graph-degree distributions after changing a producer option. **S.**

### OUT-06: Missing metadata can become a positive spatial assertion

**Where.** `S/gpu-spatial-analysis/global-coordinates/spherical-status.ts:5,14,51,56`; `S/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters.ts:107`; `S/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:152`.

**EXECUTED.**

| Input | Observed | Expected |
|---|---|---|
| `decodeGPUSphericalArcOverlayCounts(Uint32Array.of(3,1,3))` | `referenceInside: true` | Reject missing fourth word |
| `decodeGPUSphericalConvexIntersectionCounts(Uint32Array.of(3))` | `requiredVertexCount: undefined` | Reject missing second word |
| Empty `status[0]` passed to either spherical containment status decoder | `valid: true` | Reject missing status |
| `decodeGPUNetworkSubgraphFilterCounts(Uint32Array.of(12))` | Three `undefined` fields typed as numbers | Reject a short count record |
| `unpackGPUTerrainHorizonUnorm16(Uint32Array.of(0xffffffff),3)` | `[90,90,NaN]` | Reject missing second packed word |

**Now.** `counts[3] !== 0` evaluates to `true` for an absent word. Bitwise coercion turns an absent containment status into success and a missing horizon word into the legitimate nodata code.

**Wanted.** Validate the minimum record length before indexing. Require `ceil(elementCount/2)` horizon words and a nonnegative integer element count. Validate a scalar status before bit decoding. Error messages should name the decoder and expected word count. Coarsening already has a useful eight-word guard.

**Callers, value and cost.** None with valid readbacks. Short transfers become visible failures instead of claims about the geometry or terrain. **S.**

### OUT-07: The empty neighbor distance sorts ahead of a representable neighbor

**Where.** **KI** `S/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods.ts:73,434,446,457,519`; **KI** `D/pointcloud-analysis.md:494`.

**EXECUTED host arithmetic.** Let `x` be float32 bits `0x5f7fffff`. The public parameter packer accepted `radius=x` and returned a squared radius with bits `0x7f7ffffe`. The empty-slot literal `Math.fround(3.402823e38)` has bits `0x7f7ffffd`. The documented maximum finite float32 has bits `0x7f7fffff`.

**SOURCE.** The shader increments support before attempting insertion. A candidate at distance squared `0x7f7ffffe` is inside the accepted radius but sorts after the empty slot. A row can therefore claim one neighbor while publishing `indices[0] === 0xffffffff`.

**Smallest GPU probe.** Two positions `(0,0,0)` and `(x,0,0)`, one query for row zero, `capacity=1`, `radius=x`, serial strategy, no grid. Read `counts`, `indices`, `squaredDistances`, `status`. Expect `[1]`, `[1]`, `[float32(x*x)]`, `[0]`. The same input should pass the cooperative route.

**Wanted.** Initialize empty distances from exact float32 bits `0x7f7fffff`, retaining the existing ID tie-break that orders real rows before the missing ID. Pin the empty-output bits as well as the inclusive-boundary case.

**Callers, value and cost.** None relying on the documented value. This fixes an extreme but legal false neighbor report with a two-point test. **S.**

### OUT-08: The only `count` plus `validity` reader loses a real endpoint

**Where.** `S/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning.ts:101,107`, `cpu-trajectory-thinning-operation.ts:114,155`; **KI** `D/gpu-spatial-analysis/movement-semantics.md:325`.

**EXECUTED.** One track with x-coordinates `[0,1,10]`, times `[0,1,2]`, minimum distance 5 and capacity 3 returned `rowIds=[0,2,4294967295]`, `count=[2]`, `validity=[1,0,1]`. Combining the documented compact-count and positive-validity patterns gives:

```ts
Array.from(rowIds.subarray(0, count[0])).filter((_, i) => validity[i])
// [0]; the retained source rows are [0, 2]
```

Here `validity` describes source selection **before capacity clamping**. The compact IDs are already the retained rows. This is separate from the thinning search-budget question in `KNOWN(TRJ-102)`.

**Wanted.** Add the explicit names `sourceRows` and `sourceSelectionMask` as aliases in the contributor and operation forms. Document “read `sourceRows[0..count)`; the selection mask indexes original positions and may include selected rows beyond output capacity.” Show this three-row fixture.

**Callers, value and cost.** None for aliases. Generic readers can associate each array with its correct domain. **S.**

### OUT-09: A successful streamline result can have a descending offsets tail

**Where.** `S/raster-analysis/flow-texture/cpu-streamlines-operation.ts:292,316,321`, `gpu-streamlines.ts:60,872,903`; contrast the stated repeated-terminal contract at **KI** `S/gpu-spatial-analysis/line-merge/gpu-line-merge.ts:66,618`. **KI** `S/analysis-core/contracts/ports.ts:70` describes padded geometry with repeated terminal offsets.

**EXECUTED.** I ran the constant horizontal field from `T/gpu-raster/flow-texture/cpu-streamlines-operation.node.spec.ts:45`: velocity `[3,0]`, extent `[0,0,10,10]`, four seeds, two steps per direction, line capacity 4, point capacity 20, seed 7. Result: `count=[1]`, `positionCount=[5]`, `overflow=[0]`, **`lineOffsets=[0,5,0,0,0]`**. A reader slicing the first two offsets is correct. A reader binding the supplied full view as line geometry gets a descending interval.

**SOURCE.** The GPU contract leaves the later offsets unwritten. A minimal reuse probe is the same fixture with all five offset words prefilled to 99; inspect the whole view after submission.

**Wanted.** Publish `positionCount` into unused **offset** entries, yielding `[0,5,5,5,5]` in both forms. Keep the position tail outside the result and retain `count` for the distinction between active and empty lines. This makes the capacity-sized offsets safe for a GPU consumer without readback or per-frame reconstruction.

**Callers, value and cost.** None consuming the defined active prefix. This adds a guarantee specifically for offset padding. It is independent of `KNOWN(RT-124)` and `KNOWN(RT-125)`. **S.**

### OUT-10: Found paths, readable paths and resolved edge IDs are different facts

**Where.** `S/gpu-network/network-analysis/gpu-network-path-extraction.ts:57,65,77`; `cpu-network-path-extraction-operation.ts:180,205,215`; **KI** `D/gpu-network/operators.md:300`.

**EXECUTED.** For predecessor chain `0→1`, targets `[1,0]` and node capacity 1, the CPU twin returned `pathFound=[1,1]`, `lineOffsets=[0,2,3]`, `count=[1]`, `ids=[0]`, `overflow=[1]`. Neither complete path can be read from that node output. JavaScript `subarray(0,2)` silently returns `[0]`, which looks like a one-node route.

In a separate check on the edge column, a real edge with application ID `0xffffffff` produced `edges.ids=[4294967295]`, `edges.edgeRows=[0]`. With the link absent from the supplied CSR, the same decoder inputs had `edges.ids=[4294967295]`, `edges.edgeRows=[4294967295]`. Both had `edges.count=[1]`, `edges.overflow=[0]`.

**Wanted.** Add a path readback helper returning `found`, `fullyStored`, node rows or IDs, and edge-resolution validity. A found path is fully stored only if its entire required interval is within the written count. Test missing edges using `edgeRows`, never an arbitrary application ID. Document how to request that witness on the GPU. Preserve required offsets for sizing.

**Callers, value and cost.** None. This builds on the existing identity/row split and `KNOWN(NET-116)`’s concern about composing offsets. The two-target partial route and all-ones identity collision are the additional evidence. **S.**

### OUT-11: Pair gather requires a different count path from the pair table it consumes

**Where.** `S/gpu-spatial-analysis/pair-gather/gpu-pair-gather.ts:78,193,312,849,874`; **KI** `S/analysis-core/contracts/ports.ts:224`; **KI** `D/gpu-spatial-analysis/operators-joins.md:947`.

**EXECUTED.** I constructed separate valid input and output pair tables, each `{leftRows,rightRows,status:{count,overflow}}`, and passed them to `new GPUPairGather({pairs:input,output})`. The constructor threw **`Cannot read properties of undefined (reading 'byteOffset')`**. Moving the same output count view to `output.rowCount` declared two nodes. This is the only public producer with a top-level output view named `rowCount`.

**Wanted.** Use `output.status.count`. Preserve `matchedCount` as an explicitly input-domain diagnostic: it can exceed the number of matched rows that fit in the output. Emit `pair-gather output.status.count is required` for a missing destination. A shared pair-table allocator would then work on both sides.

**In-repository callers that change.** `rg 'new GPUPairGather\(' modules examples website -g '*.{ts,tsx,js}'` gave **41 matches: 40 executable constructor sites and one website code string**. The executable sites are in analysis tests, the pair-gather lens and the website run. Several share a props builder, so one edit can repair multiple sites. Update the displayed code and regenerate captured examples. I recommend the breaking rename before alpha. **S.**

### OUT-12: Accessibility validity describes the demand input alone

**Where.** `S/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts:66,145,447`, `gpu-huff-trade-areas.ts:96,218,522`; **KI** `D/gpu-spatial-analysis/operators-accessibility.md:21,88`.

**SOURCE.** Both writers test finiteness of the demand value. `GPUHuffTradeAreas` even requires optional input `demand` to request `validity`. The trade area and probabilities can be usable when demand is missing. Conversely, a finite demand word does not certify complete spatial weights. The guide explains the distinction, but the output name loses it as soon as the buffer is passed to another contributor.

**Smallest GPU probe.** One demand location, one reachable facility of positive attraction, complete forward weights, `demand=[NaN]`. Request `tradeArea` and `validity`. The guide predicts facility row `0` and validity `0`. A map that applies `if (validity[i]) draw(tradeArea[i])` hides the usable assignment.

**Wanted.** Expose `demandValidity` on both producers, make it the documented spelling and describe which aggregate excludes that demand. Reserve a plain result-validity field for the primary result. The same rule explains why the one-word batch `validity` of `GPUAnalysisContiguityStatistics` belongs with its whole-profile result.

**Callers, value and cost.** None for an additive alias. GIS renderers keep usable assignments and apply missing-demand filtering only where appropriate. **S.**

### OUT-13: Dot-sampling offsets are pre-clamp starts without a terminal

**Where.** `GPUDotDensity` and `GPURandomPointsInPolygon`, shared `S/gpu-spatial-analysis/dot-density/dot-density-sampling.ts:157,279,801,817`; **KI** `D/gpu-spatial-analysis/operators-density.md:653`.

**EXECUTED reference; SOURCE producer.** I ran the repository’s independent `generateDotCountsOnCPU` on two positive-area triangles requesting `[2,2]` points with capacity 3. It returned `slotOffsets=[0,2]`, `slotCounts=[2,2]`, `requiredCount=4`, `count=3`, and published feature rows `[0,0,1]`. In `positions.subarray(slotOffsets[i],slotOffsets[i+1])`, the last end is absent. Appending `requiredCount` instead produces an end beyond the published result. The producer scans into a slot-count-length view and computes the final total separately. The same two triangles are the smallest GPU confirmation.

**Wanted.** Document `slotOffsets` as `slotStarts`, with an additive alias. Give the exact stored interval in **dot rows**:

```ts
const start = Math.min(slotStarts[i], writtenCount);
const end = Math.min(slotStarts[i] + slotCounts[i], writtenCount);
```

For a flat position readback, multiply both boundaries by two. Label `signedOffsets` in `GPULinearReferencing` and `matchedOffsets` in `GPUMapMatching` as physical distances in any inventory; they do not delimit rows at all.

**Callers, value and cost.** None. A caller can extract the last category or feature correctly even on overflow. **S.**

### OUT-14: Interchanging top-k route producers requires a second decoder

**Where.** `S/gpu-network/alternative-routes/gpu-alternative-routes.ts:50,136`; **KI** `alternative-routes/cpu-alternative-routes-operation.ts:68`; `k-shortest-paths/k-shortest-paths-output.ts:25,45,94,140`; **KI** `D/gpu-network/alternative-routes.md:22`; `D/gpu-network/k-shortest-paths.md:65,91`.

**EXECUTED.** For `0→1`, queries `0→1` and `1→0`, and two requested routes, the alternative-routes CPU twin returned `counts=[1,0]`, `lengths=[2,0,0,0]`, `complete=[1,1]`. The empty second query is certified. The five data columns use query, route and then vertex-slot order. Alternative routes uses `nodeCount` as its path stride; K-shortest paths uses `maximumPathLength`. Its completion flags apply to the batch. The existing K decoder requires that batch status map.

**Wanted.** Add `getGPUAlternativeRoutesOutputLayout` and `decodeGPUAlternativeRoutes` using the same internal five-column route slicer. Return the per-query completion flag for alternative routes and retain the batch-level interruption flags for K-shortest paths. Show the zero-route query in both examples.

**Callers, value and cost.** None. A routing application can share route presentation code without manufacturing a per-query certificate from a batch flag. This is a concrete bounded application of `KNOWN(DIR-103)`. **S.**

## Departures by name

The following are the additional names needed to interpret the census and apply the convention. Prefix each abbreviated class name with `GPU`.

- **No top-level `props.output`** — `SnapshotTap`, `TransferMask`, `SpatialJoinPrepared`, `NetworkAccessibility`, `NetworkAnalyticsColumns`, `RasterTexture`, `RasterStreamingAccumulator`, `RasterStandardizedIndexCalibration`, `RasterClimateRunAccumulator`, `PointBuffer`. Several are handles or accumulators. Two expose nested output destinations.
- **A direct view** — `H3CellIdentities`, `ClassAssignment`, `LocalFrameProjection`, `SpatialWeightsTransform`, `SpatialLag`, `TrajectoryTrackRelativeTimes`, `CanonicalPolygonEquality`, `CanonicalGeometryEquality`, `NetworkStatistics`, `WideNumericScan`, `CubeLayoutTranspose`, `TimeWordsRelative`, `VariableStandardization`, `RasterNDWI`, `RasterNBR`, `RasterEVI`, `RasterSAVI`, `RasterRollingMean`. `ChunkedSpatialWeightsTransform` takes an array. Direct views are reasonable for single-column results; the packed summaries among them need a decoder.
- **Separate nested output destinations** — `CellSetOutline` (`rings`), `PairGather` (`leftColumns[]`, `rightColumns[]`, `slotColumns[]`), `SpatialTwoStageLeastSquares` (`robust`), `GeneralizedGeographicallyWeightedRegression` (`bandwidthSelection`), `MultiscaleGeographicallyWeightedRegression` (`inference`), `SeeminglyUnrelatedRegression` (`spatial`), `ProbitRegression` (`spatialLag`, `spatialDiagnostics`), `SpatialCrossValidatedOLS` (`folds`), `NetworkAccessibility` (`catchment`), `NetworkPathExtraction` (`edges`), `NetworkAnalyticsColumns` (`degree`, `inDegree`, `components`), `NetworkIsochrones` (`raster`, `raster.rings`, `cellOutline`, `cellOutline.rings`), `TerrainTiledFlow` (`tiles[]`), `GroupStatistics` (`columns[]`), `KeyJoin` (`gather[]`, `aggregates[]`). Each listed path ends in `.output`. A checker restricted to `props.output` misses all 24.
- **Other spellings of counts** — `PairGather.rowCount`; `EmergingHotSpots.count`, `EmpiricalBayesRates.count`, `MultivariateLocalGeary.count`. Top-level `counts` is vector-valued on `S2CapCovering`, `S2PolygonCovering`, `SphericalConvexIntersection` (`uint32x2`) and `SegmentSnap` (`uint32x4`). For example, SegmentSnap packs written vertices, required vertices, overflow and nonfinite input per line.
- **Positive masks with different row domains** — `BoundsFilter`, `LineSimplification`, `CoverageSimplification`, `BufferSelection`, `ColumnQuantiles`, `TimeWindowFilter` spell the output `outputMask`. `RegionStatistics`, `RegionMask` and `PickRegionMask` use `mask` over source rows. `CellTableCompare.mask` marks occupied **output** rows and clears the tail. All ten documented polarities are positive. Uniform spelling alone would hide the consequential difference.
- **`outputValidity`** — `Geomorphons`, `SimpleLocalRelief`, `MultiScaleRelief`, `LocalDominance`, `ReliefBlend`, `TerrainDerivatives`, `TerrainCurvature`, `TerrainRGBDecode`, `TerrainSpikeRepair`, `TerrainHorizon`, `TerrainCastShadow`, `SolarIrradiance`, `ReliefShading`, `TextureShading`, `TerrainRuggedness`, `TerrainVectorRuggedness`, `TerrainTopographicPosition`, `TerrainWeissLandforms`, `RasterSampling`. This confirms the earlier vocabulary inventory. The new reason to qualify these ports is the observed mix of source-selection, field-specific and batch validity elsewhere.
- **Different no-result and tail policies** — `TrajectoryThinning` pads row IDs with all ones; `AlternativeRoutes` and `KShortestPaths` combine all-ones key padding, zero lengths and infinite costs; `Streamlines` leaves later GPU offsets unwritten; `LineMerge` repeats the terminal; `SpatialWeightsTranspose` explicitly clears unused slots; `PointCloudStatisticalFilter` uses `-1` for an unsupported mean distance; `PointCloudNeighborhoods` uses a finite squared-distance fill plus an ID sentinel. Testing only for `Number.isFinite(value)` does not distinguish these cases.

## Quick wins

- [ ] Remove the unwritten contiguity distance column from the advertised result and test the actual contiguity-to-transpose connection.
- [ ] Fix the shape-generator polygon example using two features and all three offset levels.
- [ ] Add byte-oriented decode examples and short-record guards. Test a nonzero byte offset, the word `257`, and a missing final flag.
- [ ] Derive the network histogram stride from the bounded view and share the producer’s dimension validation.
- [ ] Publish a repeated terminal across unused streamline offsets and compare the **whole** offsets array on reuse.
- [ ] Add the three-fix thinning example, the two-category dot interval example, and the all-ones application edge ID example.
- [ ] Fix the neighborhood empty-slot literal and run the two-point inclusive-boundary GPU probe.
- [ ] Add the alternative-routes layout and decoder using the already-tested K-shortest route slicing logic.

## Larger proposals

1. **Make writable status types producer-specific.** Start with the three reproduced weights cases. An audit can give each optional field its own buffer and check `node.resources` by identity. A correctly declared pass-through should identify the supplying input. This would have caught both the untouched words and the unwritten distances. **M.**
2. **Rename `GPUPairGather.output.rowCount` to `output.status.count` before alpha.** The 40 executable uses are local and the same pair shape then works on both sides. An additive alias is cheaper immediately, but leaves two published spellings for the primary pair count. I recommend the breaking rename. **S.**
3. **Put row-domain metadata beside the output declaration.** Start with compact pairs, fixed-stride neighbor or route tables, generated geometry, and packed summaries. For each, record its source or output index domain, stride or terminal rule, completeness witness, and which portion of capacity is defined. Generate the relevant API table from it and check it with a nonempty frame followed by a smaller frame. This supplies the output-specific evidence envisaged by `KNOWN(DIR-109)` without introducing a new graph runner. **M** for these four shapes; **L** for library-wide behavioral qualification.

## Not read

I enumerated all public producer constructor outputs and the separately nested output maps. I read the shared status contract, spherical count/status decoders, region-statistics readback, and K-shortest output layout and decoder in full. I traced selected writable paths through their constructors, publishers, CPU twins and downstream consumers. I chose those paths where the census found conflicting row domains, inherited optional destinations, different offset termination, or byte-versus-word ambiguity.

I sampled first-run and family documentation, relevant fixtures, and in-repository consumers. I checked candidates against the synthesis, area reports, API/docs and direction reports, and the decision list. I did not read every shader or every mode implementation. The census of types is exhaustive for the described traversal; the order, sentinel and tail counts measure explicit member documentation, not exhaustive device behavior. No file was written. No GPU, browser, build, test suite or network was used.