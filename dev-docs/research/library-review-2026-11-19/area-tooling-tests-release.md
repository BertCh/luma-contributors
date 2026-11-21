# Tooling, the test suite and release readiness (2026-11-19)

**Hold the analysis alpha.** TypeScript’s NodeNext resolver failed on **22 relative type references in 12 emitted declarations**, including public parameter schemas. The packed-consumer gate exercises a different schema and misses these failures.  
**The sampled fixes held up.** All 30 sampled public behavior changes have a current assertion that distinguishes the stated old behavior. A behavior-specific pre-fix failure account was located for 26. Four lack that level of historical evidence.  
**Several numerical regressions can still pass with invalid output.** An executed probe replaced network modularity `0.47402005941484665` with NaN and the reference comparison accepted it. Five other numerical assertion sites also accept nonfinite results.  
**The November 18 records cover different trees and execution modes.** The saved bounded run covered 1,409 headless files. Alpha.16 added one. Its recorded Node run used four workers. The two subsequent classification fixes changed source and assertions again.  
**The release machinery now checks substantially more than it did on November 17.** It stages clean outputs, checks export names, extracts the analysis tarball, runs a CPU computation and protects promotion with rollback. Its release record identifies the tarball by path alone.  
**The measured local workload was about 91 minutes**, reconstructed from separate Mac/Metal runs. Headless execution took 59.5 minutes and one-worker Node execution took 25.7. Those two steps dominate the developer-loop decision.  
**Area findings:** 1 W, 2 S, 5 C, 11 T and 2 M. Fourteen have ORDINARY reach and seven UNUSUAL reach. The W concerns a build-helper transformation. No current wrong GPU numerical result was established.

## Method

The area lead (`gpt-6.2-astra`) read all six slice reports and their coverage statements. The six `gpt-6.1-sol` reviewers covered verification/CI, compiler/build/promotion, packed consumers/licences, policy/kernel gates, assertions/oracles, and a fixed sample of 30 behavior changes. The comparison base was `bdc715ea`. The committed review snapshot was `bd8279a4`. The last analysis change in that snapshot was `73e8f638`; the alpha.16 integration was `34179d4c`.

The lead read the shared-file changes and traced four boundaries independently: package source to compiler record, compiler output to packed consumer, verify inventory to runtime report, and staged checks to retained release records. In particular, the lead followed declaration references *past* the packed consumer’s successful line-density example and used the installed TypeScript resolver on the actual emitted files. Both slice W/S reproductions were repeated, as were the network-modularity mutation and the failed-recording allocation probe. Other executed controls attributed to a slice below were inspected as evidence, without representing them as independent lead reruns.

Read in full across the team: the principal verify, heap, build, build-record, release, export, packed-consumer, typecheck, measurement, watcher, benchmark and oracle scripts; Vitest, Biome, workflow and package configurations; the three gate/release documents; the named assertion helpers; and the focused structural fixtures. Sampled: the contributor checker’s shader interpreter, the 95-file kernel-audit directory, the larger gate specs, generator implementations, declaration consumers, fixture contents and the changed implementations around the 30 regressions. Full numerical contributor bodies, all 2,498 spec bodies, all third-party libraries, all lens scenarios and application implementations were outside that reading. Website evidence was limited to committed references.

Executed work consisted of Git and filesystem reads, counting scripts, TypeScript syntax parsing and module resolution, in-memory helper/CPU probes, and package file selection through the installed `npm-packlist` library. The policy reviewer ran the two permitted read-only policy checkers, including contributor-contract stale-entry inspection. No Vitest, typecheck, build, pack, promotion, oracle generation, browser, GPU, network lookup or publication ran.

**Snapshot boundary.** The main census and the 30-change audit describe the committed review population. During the lead’s second pass, other work had added two untracked Node specs and changed analysis source. A fresh hash comparison then found **13 of 1,777 recorded analysis input hashes differed and zero of 5,316 recorded output hashes differed**. The release reviewer had observed zero differences earlier. These are compatible observations of a changing checkout. The later observation is direct evidence against using the existing `dist` as a release of the live sources. It does not indicate failure of the build-record checker.

Citation prefixes in the tables are repository-relative: `S/` = `modules/analysis/src/`, `T/` = `modules/analysis/test/`, `D/` = `modules/analysis/dist/`, and `R17/` = `dev-docs/research/library-review-2026-11-17/`. Historical local evidence `E/` = `.typecheck/library-review-2026-11-17-evidence/`. A path under `E/` or `D/` may be absent from a fresh clone. The tracked implementation account is `R17/implementation.md:60`.

## Census

### Test population and executable selection

“Browser” includes ordinary-name browser specs before opt-in exclusion and excludes benchmark suffixes. “Calls” counts literal `it(` and `test(` sites with optional whitespace. Registration loops and parameterized cases make runtime case counts larger. “Zero” counts files with no lexical `expect(`, `expect.` or `expect<`.

| Directory under `T/` | Node | Browser | Benchmark | Literal calls | Zero |
|---|---:|---:|---:|---:|---:|
| `analysis-core` | 135 | 26 | 0 | 703 | 47 |
| `computation-record` | 9 | 2 | 0 | 54 | 0 |
| `dataframe-analysis` | 2 | 1 | 0 | 14 | 0 |
| `geospatial-analysis` | 2 | 0 | 0 | 9 | 0 |
| `geospatial-expansion` | 14 | 16 | 3 | 138 | 1 |
| `gpu-core` | 1 | 2 | 1 | 9 | 1 |
| `gpu-dataframe` | 76 | 77 | 1 | 851 | 19 |
| `gpu-geometry` | 1 | 2 | 0 | 12 | 0 |
| `gpu-graph` | 1 | 7 | 0 | 8 | 7 |
| `gpu-network` | 110 | 102 | 6 | 1,211 | 12 |
| `gpu-pointcloud` | 13 | 19 | 2 | 190 | 0 |
| `gpu-project` | 5 | 2 | 0 | 45 | 0 |
| `gpu-raster` | 94 | 139 | 3 | 1,057 | 44 |
| `gpu-spatial-analysis` | 654 | 695 | 16 | 7,352 | 77 |
| `gpu-terrain` | 86 | 100 | 6 | 904 | 25 |
| `oracles` | 3 | 0 | 0 | 8 | 0 |
| `utils` | 35 | 29 | 0 | 211 | 0 |
| **Total** | **1,241** | **1,219** | **38** | **12,776** | **233** |

There were **4,613 tracked analysis test files**, including 3,301 TypeScript files. The 2,498 specs contained 669,845 lines. `scripts/` had 37 files and 11,910 lines including data and documentation. The root `test/` had four files. The source inventory contained 1,772 TypeScript files and 565,068 lines, with 633 exported `GPU*` class declarations, 128 exported `CPU*Operation` class declarations and 133 exported `compute*OnCPU` declarations. These declaration counts include workspace-only surfaces.

| Default policy inventory at slice census | Analysis | Lenses | Kepler | Total |
|---|---:|---:|---:|---:|
| Node | 1,241 | 141 | 474 | 1,856 |
| Headless | 1,216 | 30 | 164 | 1,410 |
| Benchmarks | 38 | 1 | 0 | 39 |
| Saved bounded headless inventory | 1,215 | 30 | 164 | 1,409 |

The sole headless addition was `T/analysis-core/grid-coordinate-bump.spec.ts`. The lead’s later live inventory returned 1,243 analysis Node files and 1,858 workspace Node files. The extra paths were `T/gpu-spatial-analysis/neighborhood-summary/neighborhood-summary-weights-loss.node.spec.ts` and `T/gpu-spatial-analysis/recipes/urban-landscape-recipe-preflight.node.spec.ts`. They are outside the committed census.

Reproduce the inventories from the repository root. The first command fixes the population to the review revision. The policy call deliberately measures the current disk tree.

```sh
git ls-tree -r --name-only bd8279a4 modules/analysis/test | rg '\.spec\.(ts|js)$' | wc -l
node --input-type=module -e '
import {listTestFiles} from "./scripts/verification-policy.mjs";
for (const kind of ["node","headless","benchmarks"]) {
  const files = listTestFiles(process.cwd(),kind,{});
  console.log(kind, files.length, ["analysis","lenses","kepler"].map(
    name => files.filter(file => file.startsWith(`modules/${name}/`)).length));
}'
rg --files scripts test
wc -l scripts/*.{mjs,mts,json} vitest.config.ts biome.jsonc .github/workflows/test.yml
```

For the family table, group spec paths by component four under `modules/analysis/test`. Classify `.node.spec.` first, then `/-(bench|benchmark)\.spec\./`, then browser. Count `/\b(?:it|test)\s*\(/g` in each file. For the complete lexical-zero name list, run `rg --files-without-match '\bexpect\s*(\(|\.|<)' modules/analysis/test -g '*.spec.ts'`. Current working-tree output can differ from the committed counts above.

### What the contract checkers count

The contract checker considered **1,746 of 1,772** source `.ts` files, 559,921 lines. `testing` and `computation-record` are excluded from ordinary contract traversal. Eight family roots are derived from package exports. The checker resolved 5,372 public declaration origins and 555 registered identities. The docs structural fixture inventories 116 pages, 61 with 490 required headings and 55 narrative. These denominators measure different things. They do not give a percentage of classes with correct `output.status`, mask polarity, row identity, capacity limits or chunk semantics. Those are family-review questions. `scripts/check-contributor-contract.mjs:27`, `scripts/contributor-contract-public-surface.mjs:145`, `T/analysis-core/analysis-layering.node.spec.ts:598`.

The policy reviewer observed zero new findings and no stale entries under `node scripts/check-contributor-contract.mjs --check-stale`. `node scripts/check-upstream-first.mjs` reported no violations. The baseline had 166 sites with nonempty reasons.

| Rule | Files | Units | Exceptions |
|---|---:|---:|---|
| Internal export | 2 | 3 sites | `GPUVariogram` study-window `packed` and `slabs`; `GPUContributorOperation.createContributor` |
| Multiple classes | 7 | 15 classes | `gpu-concave-hull`, `gpu-precision-reduction`, `gpu-additional-spatial-statistics`, `gpu-local-variability`, `gpu-raster-fractional-coverage`, `gpu-raster-reconstruction`, `gpu-raster-skeleton` |
| Root file | 21 | 21 files | Network: `gpu-network-inhomogeneous-k`, `gpu-network-kernel-density`, `network-spatial-context`, `network-view-utils`, `network-workgroup-counters`. Raster: `gpu-raster-fill-nodata`, `gpu-raster-fractional-categorical-statistics`, `gpu-raster-fractional-coverage`, `gpu-raster-fractional-zonal-distribution`, `gpu-raster-rank-filter`, `gpu-raster-reconstruction`, `gpu-raster-reprojection`, `gpu-raster-skeleton`, `gpu-raster-spectral-indices`, `gpu-raster-tile-reprojection`, `gpu-raster-walking-cost`, `gpu-raster-watershed-segmentation`, `raster-band-utils`, `raster-expansion-utils`, `raster-medial-axis`, `raster-watershed-forest` |
| Single-invocation data loop | 88 | 127 kernel sites | Dataframe 26, network 9, spatial 86, terrain 1, raster 5. Only 16 sites have a structured `owner`; 112 reasons still say “pending data-range” or “separate contributor fix”. |

The following command prints **every named site**, including the 127 too numerous to reproduce in the table:

```sh
node -e 'for(const [rule,files] of Object.entries(require("./scripts/contributor-contract-baseline.json"))) if(!rule.startsWith("_")) for(const [path,sites] of Object.entries(files)) for(const site of sites) console.log(rule,path,site.id,site.reason)'
```

### Gate inventory and cost

Times and sampled process-tree RSS below are historical Mac/Metal observations. The first eight verify steps come from `E/verify-landed/verify-landed-5.log`. The completed Node and headless steps come from `verify-landed-4.log`. The combined time is **90.8 minutes**, including 333.608 seconds through the packed check. No uninterrupted run of that duration was observed.

| Gate or mode | What earns a pass | Remaining ways to overread the result | Recorded cost or evidence |
|---|---|---|---|
| Full `yarn verify` | All ten selected steps exit successfully, with report validation for headless. A failed child, failed monitor or inventory mismatch terminates the sequence. | The wrapper compares headless filenames, not source bytes across the entire run. Its summary carries no immutable release identity. | `scripts/verify.mjs:14`, `:127`, `:208`. The later fresh run was interrupted during its repeated Node step. |
| `verify --from` | Selected suffix of steps succeeds. | Earlier steps require separate evidence. `--from lint` selects all ten and therefore receives a full summary. | `scripts/verify.mjs:53`, `:216`; `T/analysis-core/verify-command.node.spec.ts:232`. |
| `verify --only` | One named step succeeds and receives a partial summary. | Dependencies such as fresh build outputs are still caller responsibilities. | `scripts/verify.mjs:48`. |
| Lint | Biome checks its selected files. The tooling and root Vitest config are now included. | Configured lint coverage does not establish type safety, workflow execution or coverage-provider availability. | 3.660 s, 0.43 GiB. `biome.jsonc:10`, `:73`. |
| Upstream-first | Scanned imports use allowed public package keys; production `engine`, `_internal`, loose root source and direct upstream re-exports are checked. | A local import followed by a separate local export escapes the direct re-export check. Traversal covers the configured module and deck-example roots. | 3.022 s, 0.38 GiB. `scripts/check-upstream-first.mjs:85`, `:117`. |
| Contributor contract | No new unallowlisted finding across its eleven heuristic rules. | Default permits expired allowances. Alternate syntax and incomplete type resolution can evade rules. | 4.661 s, 0.79 GiB. `scripts/check-contributor-contract.mjs:1936`, `:2028`. |
| Baseline staleness | `--check-stale` also rejects expired sites. `--all` expands reporting. | Neither a reason string nor a stable site ID establishes that a stated numerical bound is correct. The wrapper calls the default command. | Included in the permitted slice inspection. `scripts/check-contributor-contract.mjs:2068`. |
| Source typecheck | Workspace source roots and their dependencies typecheck under the source config. | `skipLibCheck`, source aliases and `checkJs: false` limit what this says about installed declarations and JavaScript. | 14.176 s, 2.15 GiB. `package.json:18`, `tsconfig.check-src.json:1`. |
| Ordinary test typecheck | 238 sequential family/shared configurations, 3,678 distinct roots at census, own caches, first failing compiler stops the runner. | Imported source, barrels and JSON can recur in many programs. The root count does not measure compiler memory. | Whole test typecheck 286.267 s, 3.52 GiB warm; earlier 789.027 s, 3.97 GiB. `scripts/typecheck-tests.mjs:133`, `:258`. |
| Example-backed typecheck | Separate project admits 382 of 440 classified files. | 58 excluded roots have recorded reasons. Their diagnostics were not requalified. Standalone example projects have additional roots. | Included above. `scripts/typecheck-tests.mjs:201`; `test/kepler-compute-example-specs.json:1`. |
| Website-backed typecheck | Separate project admits 41 classified files and their imports. | It applies selected options and roots. The complete website project has its own workflow check. | Included above; 240 sequential stages in total. `scripts/typecheck-tests.mjs:217`. |
| Watch/status | Live owner and compiler PIDs, matching token, fresh requested heartbeat, completed cycle and matching source/config fingerprint. Filtered output reports the total diagnostic count. | A transitive declaration outside the inventoried roots can change without changing this fingerprint. | No timing established. `scripts/typecheck-status.mjs:43`, `:170`; `scripts/typescript-project-inputs.mjs:49`. |
| Typecheck measurement | Isolated cold/warm caches, guarded inputs, positive samples for RSS-qualified results. | Timing-only records have no measured memory; unchanged warm references may perform no compile work. | Source 12.516/13.995 s cold/warm; declaration references 19.495/0.124 s; package pipeline 33.558/5.329 s. Experimental RSS absent. `dev-docs/release-package.md:179`. |
| Incremental build | Composite compilation, removed-output pruning, bounded CommonJS conversion and successful record creation. | A failed developer build can leave partial outputs. Equal export names do not establish equal values. | 10.427 s, 1.44 GiB in the warm sequence. `scripts/build.mjs:91`, `:102`, `:132`. |
| Build record | Precompile input hashes are revalidated after all three CJS passes; expected emitted contents are hashed. Missing/corrupt outputs trigger repair. | Captured inputs omit lockfile, compiler/plugin bytes and transformer content. CJS freshness also uses transformer mtime. Hashes certify correspondence to an inventory. | Included in build. `scripts/build-record.mjs:12`, `:52`, `:97`; `scripts/build.mjs:113`. |
| Workspace export check | Expected outputs exist for compiler roots, recorded bytes match, wildcard targets expand, explicit targets exist and sorted ESM/CJS names agree. | Workspace dependencies are available. Existence of a `.d.ts` does not resolve all its type references. Legacy unrecorded output has a weaker timestamp path. | 3.703 s, 0.56 GiB. Historically 55 expanded workspace entries. `scripts/verify-package-exports.mjs:65`, `:81`, `:143`. |
| Packed analysis consumer | One offline tarball extracted outside checkout; all 11 entries loaded with ESM and CJS; dependency ranges checked; limited `.mts` consumer, README, notices and CPU variogram checked. | Dependencies are symlinks into an installed tree. Absolute links receive a syntax check. The declaration-body check misses TOOL-102; `.cts` is uncovered. | 7.692 s, 0.58 GiB. `scripts/check-packed-consumer.mjs:164`, `:217`, `:304`, `:332`, `:383`. |
| Clean staged release | Snapshot package/config bytes, compile with installed tools, verify snapshot exports and packed analysis consumer, retain tarball, recheck sources and outputs. | Third-party tools/dependencies come from parent checkout. Stage success excludes Node/GPU qualification. Ordinary untracked package files enter the snapshot. | Dedicated clean-release time unavailable. `scripts/build-release.mjs:30`, `:178`, `:193`, `:207`, `:234`. |
| Promotion | Back up three output trees and caches, install staged trees, verify final digests, publish records within the rollback window. | Replacement spans multiple renames. Abrupt termination can require inspection of actual paths and journal. The output-tree digest excludes the retained archive itself. | `scripts/build-release.mjs:91`, `:138`, `:246`, `:261`; failure fixtures `T/analysis-core/release-build.node.spec.ts:189`, `:208`, `:225`. |
| Node correctness | Node project selects `.node.spec.ts/js`; default one thread worker; verify explicitly passes `--maxWorkers=1`. | Module state persists. Local focused tests are allowed. The wrapper consumes the exit code without reconciling a Node report to disk. | 1,543.909 s, 10.83 GiB. `vitest.config.ts:72`, `scripts/verify.mjs:172`. |
| Browser/headless correctness | Non-Node, nonbenchmark eligible specs; serial files; required-device helper throws for absent browser device. | Raw `yarn test-headless` lacks the wrapper inventory check. Some individual tests still return before their assertions. | Wrapped chunks total 3,568.686 s; largest sampled peak 5.38 GiB. `vitest.config.ts:51`, `S/testing/require-test-device.ts:22`. |
| Headless inventory and chunk admission | Inventory before execution; remove old report; require exact normalized suite membership, successful command and a passed case in every file; compare final disk membership. | A passing case can contain zero assertions. A same-name edit during the run preserves membership. | 212 saved chunks, 1,409 files, every file had a passed case. `scripts/verify.mjs:140`, `:180`, `:193`, `:208`. |
| Benchmarks | Suffix-selected browser project; runner activates recognized boolean entry gates and prints inventory. | Size and alternate routes can skip; no required count of completed measurements. Timing code in ordinary-name files remains ordinary work. | 39 workspace files, 26 with recognized boolean gates, 23 distinct activation flags. `scripts/test-benchmarks.mjs:15`. |
| Oracle regeneration | Configured generator must recreate all declared outputs inside staging. Missing outputs fail before acceptance; unequal bytes fail comparison. | Unexpected extra writes are outside the output comparison. Regeneration is separate from verify; most fixture files have no configured producer. | No generator run. `scripts/run-oracle-fixture.mjs:315`, `:402`, `:439`, `:467`. |
| Numeric comparison | Shared helper checks length, missing entries, finite tolerances, finite/nonfinite agreement and signed infinities. | Adoption is partial. Hand-written checks can accept NaN. | `S/testing/expect-numeric-array-close.ts:24`; examples below. |
| API documentation/layering | Discovered page inventory, required headings, runtime value exports and family ordering have assertions. | Family-order scanner misses double-quoted relative imports and `require`. Page membership does not certify a section for every public class. | `T/analysis-core/analysis-layering.node.spec.ts:273`, `:598`. |
| Runtime boundaries | Starting at seven domain entries, traverse runtime edges, require nontrivial traversal and reject forbidden imports. | Computed imports are outside literal resolution. Type-only imports intentionally have different rules. | `T/analysis-core/analysis-runtime-boundaries.node.spec.ts:103`, `:204`. |
| Registry/public surface | Parse selected ontology collections and compare qualified public origins; runtime catalog spec separately checks registered exports. | A recognized collection name in dead local code can satisfy the static checker. Some indexed public types escape inspection. | `scripts/check-contributor-contract.mjs:272`, `:446`; `T/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts` can be located by basename. |
| Kernel inventory and memory | Audit cases cover discovered export **names** or explicit skips; shared estimator uses declaration layout and pipeline constants, rejects unresolved layouts. | Distinct entries can export the same name. An alias can bypass a guard against dropping pipeline configurations. Estimates describe WGSL declarations, not driver allocation. | `T/analysis-core/kernel-audit.node.spec.ts:340`; `S/testing/gpu-contributor-test-utils.ts:158`, `:195`. |
| Resource lifecycle | Compare active buffers/textures after warmup and teardown. Three fitting contributors repeat three times; a real texture checks the texture counter separately. | Untested lifecycles include failure during recording. Same-kind counter cancellation is possible. | `T/analysis-core/gpu-resource-counts.spec.ts:26`, `:50`; `T/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.spec.ts:153`. |
| CI and app jobs | Workflow configures immutable install, package build before app checks, standalone example and website typechecks, catalog/build, Node and Chromium headless. | No hosted execution evidence was available. CI omits the packed consumer and wrapper’s resource and inventory semantics. Linux selects SwiftShader. | 60-minute main timeout, 30-minute website timeout; actual hosted cost unknown. `.github/workflows/test.yml:17`, `:44`, `:59`, `:84`; `vitest.config.ts:45`. |

**Resource contract.** The verify disk preflight examines available blocks on the checkout filesystem. Eight GiB is the minimum; a caller can raise it. It runs before each heavy step and each headless chunk. The separate temporary filesystem must have actual room for packing/extraction, but has no independent free-space preflight. RSS monitoring samples the verify process and descendants immediately and once per second, fails on inspection error, and kills the active detached process group above 12 GiB. SIGINT/SIGTERM also fail the command. Between-sample peaks and driver memory are outside that measurement. The 8,192 MiB default is a **per-process V8 heap setting**. The standalone build preserves its caller environment. `AGENTS.md:41`, `scripts/verify.mjs:64`, `:77`, `:94`, `:115`, `scripts/verification-policy.mjs:9`, `scripts/check-packed-consumer.mjs:291`.

## Follow-through on the 2026-11-17 findings

`FIXED` means the cited implementation and a test distinguishing the reported defect were inspected. It does not imply that test was rerun. For compound findings the verdict covers the entire claim.

| ID | 2026-11-17 claim | Verdict | Fix, pin and counterfactual |
|---|---|---|---|
| TOOL-01 | Accessibility and U5 neighbor comparisons accept NaN | **FIXED** at both named sites | `T/gpu-network/network-accessibility/gpu-network-accessibility.spec.ts:52` and `T/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search-u5.spec.ts:40` use the finite-aware helper. `T/utils/expect-numeric-array-close.node.spec.ts:61` rejects actual NaN, missing and sparse rows; restoring the permissive inequality fails this fixture. Other numerical comparisons remain open under TOOL-109. |
| TOOL-02 | Export gate can certify a stale/incomplete release | **PARTLY** | `scripts/verify-package-exports.mjs:65` checks current inputs/outputs and `:143` compares names. `T/analysis-core/gate-scripts.node.spec.ts:129`, `:147`, `:153`, `:196` distinguish missing current output, same-count wrong exports, omitted wildcard and missing type target. The installed declaration graph still has unresolved edges. |
| TOOL-03 | Five lens lazy imports and native JSON loading fail | **FIXED** for reported paths | The Gaussian, forest, scalar-ridges, embedding-quantize and contrast-equalization loaders now target `computation-pack/pack`. `modules/lenses/src/computation-lenses/embedded-dataset-loaders.js:6` handles Node/browser JSON. `modules/lenses/test/computation-docs/lens-package.node.spec.ts:187`, `:284`, `:374` distinguish nonexistent directory targets, invalid emitted JSON and removed registration side effects. The existing-dist smoke remains opt-in. |
| TOOL-04 | Test runner overrides caller heaps and roots all sources | **FIXED** for those defaults | `scripts/typecheck-tests.mjs:133`, `:243`; `T/analysis-core/typecheck-family-stages.node.spec.ts:35` checks exact roots/order/caches and `verify-command.node.spec.ts:83` checks lower environment heaps, other options and stop-on-failure. Transitive fixture costs remain. Direct CLI heap forwarding is narrower than wrapper forwarding. |
| TOOL-05 | Dead watcher’s clean log counts as current | **FIXED** for the reported failure | `scripts/typecheck-status.mjs:43`, `:163`. `T/analysis-core/gate-scripts.node.spec.ts:237`, `:316`, `:393`, `:421` distinguish a dead log, duplicate owner, edit during wait and changed imported JSON. See TOOL-108 for dependencies outside its fingerprint. |
| TOOL-06 | Oracle runner exits zero after reproducing nothing | **FIXED** | Staged outputs are removed and complete regeneration is mandatory. `T/analysis-core/gate-scripts.node.spec.ts:207`, `:562`, `:581`, `:596`, `:610` distinguish zero/partial writes, fresh identical bytes, inventory-only programs and missing declared outputs. Visibility numerical regeneration remains LR-02. |
| TOOL-07 | MIT-only metadata understates shipped source licences | **FIXED** for textual inventory | `modules/analysis/package.json:8`, `:75` and `modules/analysis/THIRD-PARTY-NOTICES.md:11` agree for all 19 non-MIT files. `T/analysis-core/packed-consumer-checker.node.spec.ts:312`, `:318`, `:408` reject omitted expressions, notices and licence files. |
| TOOL-08 | Optional Arrow peer is required by native imports | **FIXED** | `modules/analysis/package.json:95` makes it required. `T/analysis-core/release-package.node.spec.ts:82` checks the dependency and absence of an optional override. `packed-consumer-checker.node.spec.ts:400` checks the missing-peer failure. |
| TOOL-09 | Headless cases pass without a device | **PARTLY** | Original four sites/seven cases now require the device; optional second-profile absence explicitly skips. `T/utils/require-test-device.node.spec.ts:17` pins the named files. Nine other ordinary-selected cases have an early-success path, including one timing case. |
| TOOL-10 | Benchmark selection and activation disagree | **FIXED** for the suffix/flag defect | `scripts/verification-policy.mjs:32`, `scripts/test-benchmarks.mjs:15`. `T/analysis-core/verify-command.node.spec.ts:367` adds suffix-matching files and new boolean gates while retaining a size option. Timing embedded in an ordinary file is a separate issue. |
| TOOL-11 | Normal build always starts cold and transforms every module concurrently | **FIXED** | `scripts/build.mjs:91`, `:132`, `scripts/verification-policy.mjs:168`. `T/analysis-core/verify-command.node.spec.ts:396`, `:411` distinguish bounded vs unbounded callbacks, warm-cache retention, unchanged CJS reuse and removed-output pruning. |
| TOOL-12 | Architecture/registry checkers have named blind spots | **FIXED** for the enumerated examples | `scripts/check-upstream-first.mjs:56`, `:85`; export-derived roots and parsed catalog collections in `scripts/check-contributor-contract.mjs:28`, `:272`. `T/analysis-core/upstream-first-guard.node.spec.ts:80` and `contributor-contract-checker.node.spec.ts:468` distinguish forbidden production imports and comment-only registrations. Related syntactic gaps remain. |
| TOOL-13 / CORE-10 | Replacing a baselined loop with a different loop passes | **FIXED** for the count-based defect | `scripts/check-contributor-contract.mjs:1698`, `:1936`. `T/analysis-core/contributor-contract-checker.node.spec.ts:273`, `:662`, `:1516` distinguish changed IDs/constants at equal counts, reasonless additions, expired sites and harmless line shifts. Stale checking is opt-in. |
| TOOL-14 | No complete local command; lint omits tooling; coverage unavailable | **PARTLY** | Sequential wrapper and expanded Biome scope landed. `T/analysis-core/verify-command.node.spec.ts:232`, `:338`, `:358` check partial runs, failure/resource handling and missing headless work. `vitest.config.ts:85` still selects Istanbul without a declared/installed compatible coverage provider or root coverage command. |
| TOOL-15 | Dependency/release docs disagree | **PARTLY** | `dev-docs/release-package.md:10`, `:91` separates target, compatibility and publication; exclusion counts derive from the manifest. `T/analysis-core/release-package.node.spec.ts:42` distinguishes the local unreleased alpha from published upstream packages. URL, tag and a precise publication handoff remain unresolved. |
| CORE-11a | Empty documented-entry map made layering vacuous | **FIXED** | `T/analysis-core/analysis-layering.node.spec.ts:111`, `:598` check stripped required headings, nonempty runtime exports and checked inventory. |
| CORE-11b | Runtime-boundary traversal never asserted forbidden edges | **FIXED** | `T/analysis-core/analysis-runtime-boundaries.node.spec.ts:103`, `:204` accumulate violations, require traversal and assert zero forbidden edges. Deliberate forbidden edges exercise the check. |
| CORE-12 | Named-size WGSL array counted as four bytes | **FIXED** for demonstrated undercount | `S/testing/gpu-contributor-test-utils.ts:195`; `T/utils/shader-memory.node.spec.ts:96` pins `array<u32,8192>` to 32,768 bytes. Named/numeric overrides, unresolved lengths and rounded workgroup allocations have separate fixtures. |

Of the ten false-green rows in `R17/README.md:147`, all ten original mechanisms received changes. The table distinguishes the fixes to the two structural specs from one another and identifies the three broader surfaces where a green result still overstates coverage: native declarations, unrelated local numerical assertions and device-conditional cases.

## Findings

| ID | Kind | Reach | Evidence | Where | What goes wrong | Proposed change |
|---|---|---|---|---|---|---|
| TOOL-101 | W | UNUSUAL | EXECUTED-CPU | `scripts/build-commonjs.mjs:8` | CommonJS rewriting silently changes an ordinary string value containing import-like text. | Rewrite parsed module specifiers. Include a real import and an identical string/template in the regression. |
| **TOOL-102** | **S** | **ORDINARY** | **EXECUTED-CPU, COUNT** | `D/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.d.ts:5`; `scripts/check-packed-consumer.mjs:175` | Native NodeNext ESM resolution fails for 22 relative type references in 12 selected declarations. Public parameter schemas are affected. | Repair emitted type references and resolve the relative edges of **all** selected declarations. Add a typed consumer of a previously broken schema. `KNOWN(LR-04)` |
| TOOL-103 | S | UNUSUAL | EXECUTED-CPU | `scripts/verification-policy.mjs:98`, `:124`; `AGENTS.md:58` | Verify/release reject a valid lower CLI heap when the overridden environment value is over the ceiling. | Validate the effective absolute value and forward it; test the conflicting inputs explicitly. |
| TOOL-104 | C | ORDINARY | COUNT, SOURCE | `modules/analysis/package.json:75`; `scripts/build-release.mjs:30`; `scripts/check-packed-consumer.mjs:138` | Two untracked files, `S/gpu-spatial-analysis/1500` and `1500:`, enter the proposed package. Each contains six bytes, `files\n`. | Review the actual selected input list before staging. Have the owner remove or exclude the two files. Preserve intentional source notices. |
| TOOL-105 | C | ORDINARY | SOURCE, COUNT | `scripts/build-release.mjs:234`, `:246`; `scripts/verify.mjs:185`; `modules/analysis/package.json:4` | Release records contain source/output digests and an archive path. They cannot authenticate the archive at handoff or associate a full test run and exact dependencies with it. Publication config supplies access only. | Persist archive SHA-256 and size, source/lock identity, dependency versions and qualification receipt. Publish that archive with the chosen explicit tag. `KNOWN(LR-04, LR-22)` |
| TOOL-106 | C | ORDINARY | SOURCE, COUNT | `dev-docs/script-gates.md:202`; `S/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm.ts:360` | The guide says three regression merges have no enforced tile cap and serial bounds/owners were reconciled. The constructors enforce 4,096 tiles; most serial reasons remain generic. | Correct the guide to the enforced limits and name the allowances still awaiting review. `KNOWN(LR-26)` |
| TOOL-107 | C | ORDINARY | COUNT | `T/oracles/GENERATORS.md:1091`; `scripts/oracle-fixtures.json:1` | The prose census says 748 fixtures/242 scripts. Current counts are 790/290; only 337 fixture paths have configured producers. | Generate separate counts for disk, configured outputs, provenance and completed regeneration. `KNOWN(PLAT-06)` |
| TOOL-108 | C | UNUSUAL | EXECUTED-CPU, SOURCE | `scripts/typescript-project-inputs.mjs:49`; `scripts/typecheck-status.mjs:175` | A virtual edit to the resolved `@luma.gl/graph` declaration caused zero reads of that file and no fingerprint change. Status can accept an earlier cycle before the compiler processes a dependency edit. | Include transitive declaration/install identity in freshness, or require a new supervised cycle after dependency changes. |
| **TOOL-109** | **T** | **ORDINARY** | **EXECUTED-CPU** | `T/gpu-network/network-statistics/network-statistics-reference.ts:99`; five further sites below | NaN passes finite numerical expectations for modularity, final bearings, headings, H3 coordinates, eigenpairs and Lee L permutation values. | Mutation-test the actual assertions with NaN and missing elements; retain their existing numerical budgets. |
| TOOL-110 | T | ORDINARY | SOURCE | `T/gpu-terrain/terrain-analysis/gpu-terrain-viewshed.spec.ts:155`; `scripts/verify.mjs:149` | Nine cases in ordinary selection can pass by returning on device/capability absence. One is a timing measurement. A passed case satisfies file-level report admission. | Require the device and explicitly skip unavailable optional capabilities with the reason. Move the timing case to its own workload. |
| TOOL-111 | T | ORDINARY | SOURCE | `vitest.config.ts:72`; `scripts/verify.mjs:172` | Local correctness configuration inherits Vitest’s non-CI `allowOnly` default. A focused test can suppress sibling cases while the Node step exits successfully; the wrapper has no Node report reconciliation. | Set `allowOnly: false` for correctness; add Node inventory/report checks with explicit skips. No committed focused case was identified. |
| TOOL-112 | T | ORDINARY | EXECUTED-CPU, SOURCE | `T/analysis-core/kernel-audit/analysis-contributors.ts:44`; `T/analysis-core/kernel-audit.node.spec.ts:340` | Both spatial and network export `addVisibilityGraphRecipe`. The inventory keeps the first name; both registered audit cases use spatial. Network CSR centrality kernels lose their distinct audit obligation. | Key cases/skips by entry and export; add a network CSR metrics case. |
| TOOL-113 | T | ORDINARY | SOURCE | `dev-docs/roadmap.md` LR discriminating-test rule; `R17/implementation.md:17`; `CHANGELOG.md:17` | Four of 30 sampled behaviors lack a located historical failure record sufficient to identify the specified counterfactual. | Retain parent SHA, exact test identity, command and failure excerpt; label later counterfactual runs with their actual dates. |
| TOOL-114 | T | ORDINARY | SOURCE | `T/gpu-dataframe/column-classification/gpu-bivariate-classification.spec.ts:399` | The new no-colors case requests class IDs along with counts. It distinguishes the original defect but leaves counts alone uncovered. | Request only counts for a hand-classified 2×2 scene; poison outputs, submit again all-masked and require zero counts. |
| TOOL-115 | T | ORDINARY | SOURCE | `scripts/check-packed-consumer.mjs:164`; `scripts/packed-consumer.mts:4` | Runtime `require` succeeds independently of the TypeScript check. The types fixture uses `.mts`, namespace mentions for most entries, one exhaustive property check and `skipLibCheck`. | Add a `.cts` typed consumer and representative typed operations from different entries. Keep declaration-edge validation independent of these examples. |
| TOOL-116 | T | ORDINARY | EXECUTED-CPU | `scripts/check-upstream-first.mjs:117` | `import {Device} from '@luma.gl/core'; export {Device};` passes the checker. Direct re-export of the same binding is rejected. | Resolve local export origins and pin both spellings. No current forbidden export was established. |
| TOOL-117 | T | UNUSUAL | EXECUTED-CPU, SOURCE | `scripts/check-contributor-contract.mjs:272`, `:446`, `:1647`; `T/analysis-core/analysis-layering.node.spec.ts:273` | Dead local catalog arrays, indexed-access public types, indirect internal exports, aliased `assert` and quote-dependent DAG edges can evade individual checks. | Use shared AST import/export resolution and restrict catalog extraction to real construction dependencies. Retain the independent runtime catalog check. |
| TOOL-118 | T | UNUSUAL | EXECUTED-CPU | `T/utils/kernel-resource-configurations.node.spec.ts:41`; `S/testing/gpu-contributor-test-utils.ts:158` | Aliasing the parsed-source map bypasses the lossy-loop source guard. Iterating configurations can still omit their overrides when calling the estimator. | Pass a complete recorded configuration to the budget checker and pin same-source/different-override cases. |
| TOOL-119 | T | UNUSUAL | SOURCE | `modules/lenses/test/computation-docs/lens-package.node.spec.ts:199` | Existing-dist lazy-link/public-role smoke needs `LENS_VERIFY_BUILT=1`; the ordinary root gate does not set it. | Make this an explicit post-build qualification before any lens release. Keep private-lens work outside the analysis publication gate. |
| TOOL-120 | M | ORDINARY | SOURCE, COUNT | `T/gpu-network/flow-aggregation/gpu-flow-aggregation.spec.ts:1023` | Ordinary correctness selection still times million-row atomic vs sorted submits over repeated runs. | Retain a small correctness fixture and move timed repetitions to a benchmark file. The containing chunk took 8.729 s. |
| TOOL-121 | M | UNUSUAL | EXECUTED-CPU | `S/testing/gpu-contributor-test-utils.ts:102` | A recording node that allocates before throwing has no returned compiled object to destroy. The null-device recorder leaves one active buffer. | Reuse the kernel-audit context’s allocation tracking for the recording failure path. |

### TOOL-101 — CommonJS changes a string value

The lead called the exported `transformCommonJSModule` directly and evaluated its returned CommonJS in memory.

```js
const value = "import './sibling.js'";
const source = 'export const value = ' + JSON.stringify(value) + ';';
const {code} = await transformCommonJSModule(source, 'probe.js');
const result = {exports: {}};
new Function('module', 'exports', code)(result, result.exports);
console.log(result.exports.value);
```

Input and expected value were `import './sibling.js'`; output was **`import './sibling.cjs'`**. The replacement at `scripts/build-commonjs.mjs:8` runs on the entire source before transformation. The same regex predates this review. A syntax scan of the 1,579 analysis ESM files containing a potential regex match found zero matching ordinary literal/template nodes. This establishes a build-helper defect with a discriminating input; there is no identified published analysis constant currently changed by it.

### TOOL-102 — The type sentinel misses unresolved public declarations

`D/gpu-spatial-analysis/index.d.ts:34` exports line segmentization. Its barrel exports `GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA`. The declaration at `D/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.d.ts:5` gives the type as `import("../../analysis-core").GPUSpatialParameterSchema<"float32">`. Its defining source uses an inferred return type at `S/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.ts:16`. The line-density sentinel has an explicit type imported through a `.js` path at `S/gpu-spatial-analysis/line-density/gpu-line-density.ts:9`, `:70`.

The lead parsed each selected non-testing `.d.ts`, visited relative `ImportTypeNode` references and used TypeScript’s own implied file format and usage resolution mode. All 22 had `ESNext` mode. For the segmentize input the actual result was **`undefined`**. Substituting `../../analysis-core/index.js` resolved **`D/analysis-core/index.d.ts`**. All 22 resolved with their corresponding explicit file specifiers. A minimal reproduction for the representative edge is:

```js
const ts = require('typescript');
const path = require('node:path');
const file = path.resolve('modules/analysis/dist/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.d.ts');
const options = {module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext};
for (const name of ['../../analysis-core', '../../analysis-core/index.js']) {
  console.log(name, ts.resolveModuleName(name, file, options, ts.sys, undefined, undefined,
    ts.ModuleKind.ESNext).resolvedModule?.resolvedFileName);
}
```

| Directory under `D/` | Affected `.d.ts` file and first affected line | References |
|---|---|---:|
| `gpu-network` | `recipes/track-quality-recipe:66`; `visibility-graph/visibility-graph:94` | 2 |
| `gpu-spatial-analysis` parameter schemas | `geometry-predicates/geometry-predicates-parameters:4`; `line-segmentize/line-segmentize-parameters:5`; `neighbor-search/neighbor-search-parameters:19`; `spatial-autocorrelation/spatial-autocorrelation-parameters:8`; `trajectory-analysis/trajectory-metrics-parameters:4` | 6 |
| `gpu-spatial-analysis/ontology` | `coverage-expansion-catalog:5` | 8 |
| `gpu-spatial-analysis/recipes` | `dasymetric-redistribution-recipe:92`; `raster-summary-recipes:31` | 4 |
| Other `gpu-spatial-analysis` | `spatial-regression/mgwr-bandwidth-selection:25`; `urban-morphology/building-adjacency:51` | 2 |

The fix acceptance condition is zero unresolved relative edges in the selected declaration graph and successful positive/negative consumer assertions for a previously affected export. A full downstream compiler diagnostic was not generated in this read-only review. The confirmed failure is native declaration resolution in the existing output. Runtime module loading and workspace source aliases do not exercise it.

### TOOL-103 — Effective heap precedence

The lead called the actual policy function with environment `--max-old-space-size=16384` and CLI `['--max-old-space-size=256']`. It threw **`verify allows a V8 heap of at most 8192 MiB; use an absolute heap limit in NODE_OPTIONS`**. Native Node with the same inputs reported a total V8 heap limit of **304 MiB**, consistent with the 256 MiB old-space limit. As a positive control, an environment setting of 1,024 and CLI setting of 256 passed the helper. The parser checks every token against the ceiling before selecting the effective one. `AGENTS.md:48` says larger explicit limits are rejected, while `AGENTS.md:58` promises CLI precedence. For an effective-limit policy the expected result is to forward 256. No ceiling bypass was observed.

## Refuted or downgraded

- **A pass elsewhere in a chunk hides a wholly skipped file:** refuted. `scripts/verify.mjs:144` checks each suite. One no-op pass and one substantive skip *in the same file* are admissible.
- **Old numbered headless reports can satisfy a new run:** refuted for the normal path. The wrapper removes each report before starting its chunk.
- **A local `.only` proves most workspace files were skipped:** unsupported. The reproduced configuration permits focus and the installed runner computes a file’s focus at collection. Suppression of sibling cases is sufficient for TOOL-111.
- **All 81 local comparator files accept NaN, or all 233 lexical-zero specs are empty:** refuted. Both populations contain effective checks. The six demonstrated false-green numerical sites are named below.
- **All missing-device-looking guards are silent passes:** refuted. Fourteen of 29 syntactic matches already follow a device requirement. Nine ordinary-selected cases and six benchmark/measurement paths require separate interpretation.
- **The checkout needs 16 GiB heaps for test typechecking:** refuted as a default. The isolated project-reference experiment lacked RSS data; the landed wrapper separately recorded 3.52 GiB for test typechecking.
- **An output digest returned by the export checker identifies `analysis.tgz`:** refuted. The release wrapper hashes output trees before the export command, rechecks them inside the rollback window and records the archive’s path.
- **Promotion accepts a packed-gate mutation of installed outputs:** refuted for the inspected path. Final tree digest equality precedes record commitment inside the rollback window.
- **The packed gate necessarily tests a stock first installation:** refuted. It checks extracted package bytes against linked, locally installed dependencies and explicitly documents that boundary.
- **A missing `.cts` case proves CJS types fail:** unverified. TOOL-115 is a coverage gap. TOOL-102 independently demonstrates an ESM declaration-resolution failure.
- **A hash-bearing fixture without a directly resolvable generator is stale:** unsupported. All 17 directly resolvable generator hashes matched. The remaining 151 need attribution work.
- **The shader estimator proves physical GPU memory use:** refuted. It computes declared layouts. Array parameters without local copies and caller/callee aggregate totals illustrate the boundary.
- **All four missing historical records imply missing current tests:** refuted. Each of the four has a discriminating current assertion.
- **The later working-tree hash mismatches disprove the November 18 passes:** refuted. They establish that the current input bytes differ. The reported earlier passes retain their historical scope.

## Tests

### What the November fixes actually pin

The following sample was fixed before inspection. Its denominator is **30 caller-visible behaviors**, selected across families and late fix rounds. “Historical account” includes a stream status that names the old failure. It does not imply a retained raw run on an identified Git parent or a GPU run. The recorded status files are under `E/tranche-1/state/` and `E/tranche-2/state/`; the bump evidence is under `E/tranche-3/`. `CHANGELOG.md:357` indexes the original fixes.

| # | Behavior | Current discriminating fixture under `T/` | Old result separated by the assertion | Historical account |
|---:|---|---|---|---|
| 1 | CORE-01 parent slice | `analysis-core/core-buffers.node.spec.ts:56` | Two-row parent at byte 8 in a 32-byte handle. `(2,1)` must throw even though byte 16 fits the handle. | `core-buffers.status.md:2` |
| 2 | CORE-02 exact Decimal | `analysis-core/arrow-decimal-normalization.node.spec.ts:25` | Coefficient `9007199254740993` at scale zero must be rejected rather than accepted as `9007199254740992`. | `core-arrow-decimal.status.md:2` |
| 3 | ERG-01 usable mask | `gpu-spatial-analysis/urban-ecology/cpu-population-support-alignment-operation.node.spec.ts:49` | Mixed matched/unmatched values `[10,99]` must produce median 10 through the downstream quantile consumer. | `align-validity.status.md:2` |
| 4 | ERG-02 discarded rebuild | `analysis-core/prepared-rebuild-discard.spec.ts:19` | Translate a square by 10, discard the prepared encoding, then submit. Require translated bounds or one matching row. | `core-latch-props.status.md:4`, recorded Metal result |
| 5 | DGGS-01 signed difference | `gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare.spec.ts:385` | Opposite signed-64 extremes require correct signs and full-width differences; explicit constants also catch a wrong twin. | `dggs.status.md:2`, CPU/oracle |
| 6 | DPC-02 calendar boundaries | `gpu-dataframe/period-index/gpu-period-index.spec.ts:57` | Valid periods `2147483647/2147483648` formerly produced invalid sentinel `4294967294`. | `dpc-time-origin.status.md:3` |
| 7 | DPC-03 nonfinite origin | `gpu-pointcloud/pointcloud-local-origin.node.spec.ts:24` | Mixed finite/nonfinite triples produce `[4,5,6]`; parent yielded `[NaN,Infinity,451.5]`. | `dpc-time-origin.status.md:4` |
| 8 | DPC-10 pre-epoch | `gpu-dataframe/temporal-reduction/timestamp-format.node.spec.ts:49` | `−1` nanosecond belongs to millisecond `−1`; truncation yields epoch 0. | `dpc-time-origin.status.md:2` |
| 9 | GEO-02 ID vs row | `gpu-spatial-analysis/pair-gather/nearest-gather-rows.node.spec.ts:20` | Permuted and out-of-range IDs must still gather attributes `[10,20,NONE]` via rows `[0,1,NONE]`. | `geo-join-rows.status.md:2` |
| 10 | GEO-05 scaled circumcentre | `gpu-spatial-analysis/delaunay-tessellation/voronoi-clip.spec.ts:212` | Right triangles at `2^45` and `2^-52`, both windings: three segments, finite analytic bisectors and exact site pairs. | `geo-triangulation.status.md:2`, interpreted WGSL |
| 11 | NET-01 incomplete table | `gpu-network/map-matching/map-matching-table-completeness.spec.ts:211`, `:359` | Short-budget chain falls back without false breaks; clearing completeness on a longer chain exposes the A* budget. | `net-matching.status.md:2`, structural guard |
| 12 | NET-02 reset | `gpu-network/network-centrality/network-centrality-continuation.node.spec.ts:105` | Forward actual planner words `1,0,0`; numerical zero must accumulate. GPU companion checks a shared vertex and an earlier failure. | `net-centrality.status.md:2` |
| 13 | NET-04 percentile | `gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation.node.spec.ts:57` | Immediately above 50% of two durations selects 90; old epsilon selected 5. | `net-centrality.status.md:3` |
| 14 | NET-05 CPU CSR | `gpu-network/network-optimization/cpu-network-flow-operation.node.spec.ts:30` | Wrong same-length CSR neighbors require invalid status and zero flow; valid control gives `[1,1]`. | `net-twins.status.md:2` |
| 15 | PTS-01 Sinkhorn | `gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation.node.spec.ts:90` | Cross plan settles in two iterations while a self plan has not. Both directions require convergence 0 against a dense primal oracle. | `pts-sinkhorn.status.md:2` |
| 16 | PTS-04 optional count | `gpu-spatial-analysis/space-time-statistics/cross-correlation-optional-count.node.spec.ts:22` | `testedCount` alone, prefilled 99, must become 1 and then 0 after input becomes constant. | `pts-huff.status.md:4` |
| 17 | SSR-02 conformal rank | `gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks.node.spec.ts:8` | At 8,388,607 rows, alpha `.25`, require ranks `2097152,6291456`; scalar fixture needs no large allocation. | `stats-spectrum-conformal.status.md:5` |
| 18 | SSR-06 MGWR budget | `gpu-spatial-analysis/spatial-regression/stats-gwr-review.node.spec.ts:150` | Budget one: current sweep count 1 and status `[3,3,3,1]`; parent count 2 and `[0,0,0,1]`. Coefficients alone agreed. | `stats-gwr.status.md:7` |
| 19 | SSR-08 padded tables | `gpu-spatial-analysis/spatial-regression/regression-padding.node.spec.ts:24` | Trailing `[99,98,97,96]` must not become a predictor or displace the named special coefficient. | `stats-heldout.status.md:3` |
| 20 | TRJ-01 antimeridian | `gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation.node.spec.ts:39` | Both segment orientations have endpoint-bound distance 100; former projection returned 180. | `trj.status.md:3` |
| 21 | RT-01 thin geographic cell | `gpu-raster/raster-zonal-statistics/zonal-geographic-window.node.spec.ts:54` | A one-row `10^-6` degree grid at latitude 60 has finite nonzero spherical area in either row direction and matching window framing. | `rt-zonal.status.md:2` |
| 22 | RT-03 cropped ray | `gpu-terrain/terrain-illumination/terrain-horizon-halo.node.spec.ts:130` | Offset crop retains the whole-grid blocker angle `83.822396°`; old crop yielded zero. | `rt-horizon.status.md:3` |
| 23 | RT-04 reprojection | `gpu-raster/raster-reprojection-affine.node.spec.ts:94` | Sheared identity strip must sample columns 4 and 524288; old packer/mirror selected 7 and 524289. | `rt-reprojection.status.md:2` |
| 24 | RT-08 round budget | `geospatial-expansion/cpu-raster-reconstruction-operation.node.spec.ts:196` | `[1,0,0,0]` dilates to `[1,1,0,0]` in one budgeted round. Diagnostic verification must not publish an extra propagation. | `rt-terrain-morph.status.md:3` |
| 25 | RCP-04 rejected declaration | `gpu-spatial-analysis/recipes/recipe-registry-rollback.node.spec.ts:744` | Inject later-stage failures; assert pre-existing graph only, no leaked registrations/uploads, and successful same-ID retry. | Tranche 2 `recipe-rollback.status.md:6` |
| 26 | Final-flow decomposition | `gpu-network/network-optimization/network-flow-paths.node.spec.ts:121` | A canceled augmentation edge has final flow zero. Paths `[0,3,7]` and `[1,4,6]` exclude it. | **Missing for this counterfactual.** Initial reds concerned absent paths. |
| 27 | Noding candidate loss | `gpu-spatial-analysis/geometry-topology/snap-rounded-noding-executed.node.spec.ts:375` | One vs eight crossing leaves distinguish candidate loss with final overflow zero. | Tranche 2 `shader-memory-audit.status.md:4`; old nested-port rejection |
| 28 | Alpha.16 grid agreement | `analysis-core/grid-coordinate-bump.spec.ts:31` | Five readers × six inputs. At `−177` in `[-180,180]` with 360 cells, old reader selects 2 and updated builder selects 3. Checks occupied cell, writer completion and actual reader result. | **Partial.** `grid-before.log` contains old source/CPU failures; no old device run of the current case located. |
| 29 | Color-scale counts | `gpu-dataframe/column-classification/gpu-color-scale.spec.ts:672` | Counts-only vs full run plus exact total 2,000. Two all-zero outputs cannot pass. | **Commit account only.** `73e8f638` says it failed without the fix. |
| 30 | Bivariate joint counts | `gpu-dataframe/column-classification/gpu-bivariate-classification.spec.ts:399` | Colors omitted: match IDs and joint counts to full run, whose total must be positive. | **Commit account only.** No parent command/failure output located. |

**Strict current discrimination: 30/30 by inspection, with selected CPU reproductions. Historical failure accounts: 26/30 under the stated counting rule.** The latter include structural and interpreter failures. The number of exact, independently replayed parent GPU runs in this review is zero. The four evidence misses are final-flow rerouting, old grid-reader/device agreement, color-scale counts and bivariate counts. No sample member was substituted.

### Assertions, skips and weak numerical evidence

AST inventory found imported Vitest `expect` in 2,265 specs with 54,247 call sites; `expectTypeOf` in 25 with 286; Node assertions in eight with 31. No analysis spec imported the root Tape adapter. `compareParity` appeared in 99 specs with 298 calls. Eleven specs called `expectNumericArrayClose` at 23 sites. The historical named-local pattern still covers 81 files, 82 declarations and 822 calls. Eight declarations in seven files delegate to the shared helper. These counts overlap.

Of the 233 lexical-zero specs, 47 register kernel audits, 154 delegate parity, four contain type assertions, nine register grid/status checks, eight delegate other correctness checks and eleven are benchmark entry points. No empty correctness file was established by tracing those categories. For concrete delegated assertions see `T/analysis-core/kernel-audit/register-kernel-audit.ts:16` and `T/gpu-spatial-analysis/spatial-weights/weights7-chunk-harness.ts:157`. Whole/chunk equality checks framing consistency; it can preserve a shared algorithmic error.

There were zero literal direct `it/test/describe.skip`, zero `.todo` and zero `.disabled.` filenames. There were 30 `it.skipIf` sites in 30 files, one `describe.skipIf`, three `it.runIf` sites in two files and 18 direct context skips in eight files. Context skips cover shader-compiler fast-math/table-fill/table-shape diagnostics, large primitives, isotropic Ripley benchmarking, projection precision, utilization-volume capacity and core-device profiles. Opt-in jobs in the shared selection policy are `delaunay-ab`, `overlay-ab` (`A5_AB=1`), `vecchia-float32-gpu-study` (nonempty `VITE_VECCHIA_FLOAT32_STUDY`) and lens `capture-records` (`VITE_CAPTURE_LENS_RECORDS=1`). Only 23 distinct flags in the benchmark census are recognized boolean activation flags. A count of every `VITE_` string would include workload options.

Complete name inventories are reproducible with:

```sh
rg -n 'function (assert|expect|compare)(Close|Near|Approx|Float)' modules/analysis/test -g '*.spec.ts'
rg -l 'expectNumericArrayClose|compareParity' modules/analysis/test -g '*.spec.ts'
rg -n '(describe|test|it)\.(skipIf|runIf|skip|todo)|\b(context|ctx|t)\.skip\(' modules/analysis/test -g '*.spec.ts'
rg -n 'VITE_|A5_AB' modules/analysis/test -g '*.spec.ts'
```

**Five weakest named numerical regression surfaces in the inspected sample:**

| Contributor | Input the current assertion accepts | Required strengthening |
|---|---|---|
| `GPUNetworkStatistics` | Mutate modularity alone to NaN in the undirected NetworkX scene. The real comparison returns `undefined` for both the valid result and the mutation. | Keep the exact topology/histogram comparisons and require finite modularity near `0.47402005941484665`. `T/gpu-network/network-statistics/network-statistics-reference.ts:99` |
| `GPUGeodesicDestination` | A due-east result with NaN final bearing produces an empty failure list. | Require a finite angle before the wrapped 90-degree error bound. `T/gpu-spatial-analysis/geometry-measures/geodesic-direct-final-bearing.spec.ts:112` |
| `GPUTrajectoryMetrics` | For `[[0,0],[-0,1]]`, a NaN northward step heading passes the signed-zero guard. | Require the finite expected `π/2` for the second row. `T/gpu-spatial-analysis/trajectory-analysis/trajectory-heading-signed-zero.spec.ts:83` |
| `GPUCellGeometry` / H3 boundary helper | For vector `[0,1,0]`, `[NaN,NaN]` passes the longitude/latitude mismatch loop. | Assert finite coordinates near `[90,0]`. `T/gpu-spatial-analysis/cell-indexing/h3-boundary-wgsl.spec.ts:321` |
| `GPUGlobalPermutationTest` | One Lee L reference-distribution value changed to NaN passes against finite `0.5` while observed-statistic and p-value checks can still pass. | Compare the full simulated array with finite-aware tolerances. `T/gpu-spatial-analysis/permutation-inference/new-statistics.spec.ts:331` |

A sixth surface is shared across contributors: the symmetric eigen check accepts status OK, `[NaN,NaN]` eigenvalues and all-NaN vectors for matrix `diag(2,3)`. Residual and orthogonality guards also use `>` and add no mismatch. The slice exercised the extracted comparison code with a supplied result; no solver defect is inferred. `T/utils/symmetric-eigen-wgsl.spec.ts:215`.

The nine early-success cases occur in `T/gpu-dataframe/time-window-filter/gpu-time-window-filter.spec.ts:444`, `T/gpu-raster/raster-zonal-statistics/gpu-raster-zonal-statistics.spec.ts:487`, `T/gpu-spatial-analysis/region-statistics/gpu-region-mask.spec.ts:100`, `T/gpu-terrain/terrain-analysis/gpu-terrain-viewshed.spec.ts:155`, `:201`, `:503`, `T/gpu-terrain/terrain-features/gpu-terrain-peak-snap.spec.ts:586`, `gpu-terrain-summits.spec.ts:452` in the same directory, and `T/gpu-network/edge-bundling/gpu-edge-bundling.spec.ts:608`. The last measures timing. Some intentionally require native hardware or storage-texture support. An explicit capability skip preserves that distinction in the report.

**Order dependence.** The Node config shares modules within each worker. A concrete shared object is the scenario registry `Map` in `modules/lenses/src/computation-lenses/registry-implementation.ts:18`. Disposal now checks descriptor identity before deletion. `modules/lenses/test/computation-lenses/lens-registry.node.spec.ts` can be located by basename for the duplicate/dispose/replacement fixture. The affected static-twin and example-mock specs use fixture-specific IDs, disposal or module resets; the scenario-declaration harness restores spies and destroys resources in `finally`. A single one-worker pass establishes one order. The implementation records shuffled reruns, but the sampled isolation log does not identify a seed or complete ordering. `R17/implementation.md:139`.

### Fixtures, provenance and regeneration

The oracle directory contains 290 top-level generator programs and 790 JSON fixtures totaling 245,614,031 bytes. `scripts/oracle-fixtures.json` declares 188 fixture producers, 101 unconfigured programs and one inventory program. The configured producers own **337 distinct existing outputs**; 453 JSON fixtures have no configured producer. Python-pinned accounts for 168 producers/312 outputs, legacy Python 15/16, visibility 1/4, pyxccd 1/2 and R 3/3. Across the test tree, 745 directly resolved JSON import targets total **212,014,090 bytes**. The import census found 267 direct spec consumers of oracle fixtures and 477 specs reachable through literal local imports.

Among 790 fixture JSONs, 744 have top-level `versions`, 495 `generator`, 168 `generatorSha256`, 386 `reference` and 218 `source`. Field presence does not establish independent provenance. Seventeen hash-bearing fixtures named a directly resolvable generator and all seventeen hashes matched. The remaining 151 did not meet that direct mapping rule.

Only three configured producer/output pairs had all inspected offline prerequisites available under the default profiles: `dataframe-time-zones.py`, `losh-backlog-inference.py`, `rts02-whittaker-sparse-exact.py`. **Zero were regenerated or compared in this review.** System Python 3.9 had NumPy/SciPy, but the configured profiles require other interpreter/package combinations. The inspected 3.14 and 3.12 environments lacked NumPy. `Rscript`, GRASS, GDAL tooling/bindings, `psql` and the configured CGAL adapter were unavailable in the relevant profiles. Private environments were not exhaustively searched. `T/oracles/requirements.txt`, `scripts/run-oracle-fixture.mjs:347`, `:386`.

The 30 largest selected data assets are listed below. `F/` means `T/oracles/fixtures/`. Both halves are in descending order, ranks 1–15 on the left and 16–30 on the right.

| Bytes | Asset | Bytes | Asset |
|---:|---|---:|---|
| 3,370,800 | `T/gpu-spatial-analysis/computation-pack/fixtures/local-permutation-kernels.json` | 1,302,909 | `F/parity-wp4-cell-statistics-real-0.json` |
| 1,463,889 | `F/parity-wp6-dataframe-osm-batch-3.json` | 1,298,473 | `F/parity-wp4-cell-statistics-real-4.json` |
| 1,459,159 | `F/parity-wp6-dataframe-osm-batch-5.json` | 1,289,889 | `F/parity-wp4-cell-statistics-real-2.json` |
| 1,426,587 | `F/parity-wp6-dataframe-osm-batch-7.json` | 1,282,485 | `F/parity-wp4-cell-statistics-real-1.json` |
| 1,415,662 | `F/parity-wp3-snapping-chicago-weights.json` | 1,237,160 | `F/parity-wp6-earthquake-input.json` |
| 1,415,171 | `F/parity-wp6-dataframe-osm-batch-8.json` | 1,226,203 | `F/parity-wp3-network-analysis-chicago-input.json` |
| 1,414,658 | `F/parity-wp6-dataframe-osm-batch-4.json` | 1,187,833 | `F/parity-wp6-round8-composite-pca-selectors.json` |
| 1,414,515 | `F/parity-wp6-dataframe-osm-batch-9.json` | 1,187,824 | `F/parity-wp6-round8-composite-z-score.json` |
| 1,398,112 | `F/parity-wp3-network-accessibility-chicago-power.json` | 1,172,344 | `F/parity-wp6-round8-composite-rank.json` |
| 1,388,211 | `F/parity-wp6-dataframe-osm-batch-10.json` | 1,170,345 | `F/parity-wp6-round8-composite-min-max.json` |
| 1,370,787 | `F/parity-wp6-dataframe-osm-batch-6.json` | 1,135,819 | `F/parity-wp6-round8-indices.json` |
| 1,324,285 | `F/parity-wp4-cell-statistics-real-5.json` | 1,129,655 | `F/parity-wp6-point-patterns-distances.json` |
| 1,317,585 | `F/parity-wp4-cell-statistics-real-3.json` | 1,129,640 | `F/parity-wp6-point-patterns-positions.json` |
| 1,316,349 | `F/parity-wp4-cell-statistics-real-6.json` | 1,109,065 | `F/parity-wp3-network-accessibility-chicago-exponential.json` |
| 1,312,461 | `F/parity-wp4-cell-statistics-real-7.json` | 1,095,689 | `T/gpu-spatial-analysis/global-coordinates/fixtures/s2-polygon-covering.json` |

To reproduce, recursively enumerate `modules/analysis/test`, exclude code/document extensions `ts,js,mjs,cjs,py,R,cpp,h,hpp,md`, sort `fs.statSync(path).size` descending, take 30. The largest asset is emitted kernel text. Several others are inputs. Their size is a developer-loop cost, not a count of independent numerical answers.

## Maintainability

**One definition per gate guarantee.** `scripts/build.mjs:152` and `scripts/verify-package-exports.mjs:81` independently plan expected output suffixes. `scripts/typescript-project-inputs.mjs:25` and `scripts/measure-typecheck.mjs:71` maintain different input traversals. Build output planning needs configured roots; watcher/measurement freshness needs dependencies. Share the underlying resolution and keep both inventories explicit. The 617-line measurement script is the natural source for the more complete traversal.

**Policy parsing.** The 2,074-line `scripts/check-contributor-contract.mjs` mixes export resolution, public types, registry interpretation, WGSL/control-flow analysis and baseline updates. A separate family DAG regex at `T/analysis-core/analysis-layering.node.spec.ts:273` disagrees with the runtime-boundary parser over common import spellings. Extract import/export handling before adding more special cases. A configuration-aware memory checker would also permit removal of the source-spelling guard in `T/utils/kernel-resource-configurations.node.spec.ts:41`.

**Numeric comparisons.** `T/oracles/parity-wp1-compare.ts`, `parity-wp4-compare.ts` and `parity-wp6-compare.ts` repeat length, nonfinite and error aggregation logic. Share the finite/length checks; keep family-specific tolerance functions and independent oracles. The workspace-only `S/testing/expect-numeric-array-close.ts:24` is an appropriate boundary. Publishing it would add a consumer API unrelated to the problem.

**Large files and assets.** The test-tree census found 26 code files over 1,500 lines. Examples include `shapely-predicates-fixture.ts` (7,942), `coverage-topology-scenes.ts` (5,695), `kernel-audit/cases-geometry.ts` (4,509), `h3-grid-path-fixtures.ts` (3,772) and `shapely-clearance-fixture.ts` (3,155). Several are data literals with provenance worth retaining. The checker regression spec itself is 1,792 lines. The 53 files under `T/arrow/` totaled 77,403 bytes with no identified direct asset consumer. Check computed-path access before removing anything. No dead export was established in the five `T/reference/` helpers.

**Release-package contract audit.**

| Statement in `dev-docs/release-package.md` | Evidence and edit needed |
|---|---|
| Lines 3–29, eligibility, dependency target and peers | Analysis is eligible for public access at `0.1.0-alpha.0`; registry history is unknown. Root resolutions pin luma core/graph/shaders to alpha.16. The range accepts alpha.17, stable 10.0.0 and stable 10.0.x; rejects alpha.15, `10.0.1-alpha.1`, 10.1.0 and 11.0.0. The installed-library `semver.satisfies` probe confirms this. Arrow `>=17.0.0` is required and four entries load it eagerly: analysis-core, dataframe, network and spatial. |
| Lines 33–60, artifact and native consumers | Offline pack/extract/link sequence, explicit exports, ESM/CJS names, `.mts` sentinel and documented CPU example match the scripts. State that the existing sentinel misses 22 unresolved import-type references. The source CPU walkthrough returned exponential status 0 and parameters `[0.2000000774860382,1.5,2.200000286102295,2.7552409681032286e-13]` for the documented `[0.2,1.5,2.2,0]` model. |
| Lines 62–72, README and changelog | Markdown token extraction handles relative links, HTML and code-block exclusion. Absolute destinations are parsed as URLs. A separate first-run spec maps recognized repository links to local files/anchors. Public access to the named repository remains undecided. |
| Lines 74–80, licence and contents | Manifest `sideEffects: false`, 11 explicit entries, root exports computation-record, and local licence texts match the package. Selected source had 1,747 MIT entries including a Markdown file, 15 `MIT AND Apache-2.0`, three BSD-3-Clause and one `MIT AND BSD-3-Clause`. All 19 non-MIT sources have exact path/expression notices; checked `.ts` sources had copyright headers. This is a textual inventory. |
| Lines 82–102, retention/publication | Success with `--keep-artifact` yields a tarball path. Neither `.typecheck/release-build-latest.json` nor a retained stage was present at inspection. Resolve canonical repository/branch, version availability and prerelease tag before making the final archive. |
| Lines 105–249, measurement | The retained experiment had zero RSS samples, denied process inspection and uncontrolled idleness. Same-snapshot reference cold checking was 55.8% slower than source checking; unchanged warm references were faster. The experiment does not meet the declared adoption criterion of 20% lower sampled RSS and at most 10% cold/warm regression. No edited-file warm trial was found. |

The package reviewer’s `npm-packlist` selection contained **7,072 files / 72,818,669 uncompressed bytes**: 1,769 source files (23,762,739 bytes), 1,765 ESM files (21,093,227), 1,765 CJS files (20,178,861), 1,765 declarations (6,739,596), a **1,011,575-byte** `dist/.build-record.json`, five licence/notice files (22,856) and README/manifest (9,815). The lead repeated selection after concurrent edits and got **7,072 files / 72,835,448 bytes**. Both included the two scratch files. Neither measure is a tarball size. No testing entry, test fixture, map or lens dataset appeared. Maps would enter through broad `dist` inclusion if present. Source is an intentional current package choice; its size and developer-only build record deserve an explicit publication decision.

The maintainer supplied the upstream publication on November 18 and dist-tags `latest: 10.0.0-alpha.5`, `beta: 10.0.0-alpha.17`. No registry lookup was made. Tags do not pin a consumer’s resolved peer or distribute workspace resolutions. Qualification evidence here identifies alpha.16 and Arrow 17.0.0. It supplies no qualification of alpha.17, later stable patches or future Arrow majors accepted by the manifests.

## Roadmap rows

Verdicts address the tooling, test or release part of each row. Domain results remain with their area owners. Row IDs identify the records across concurrent roadmap edits.

| Row ID | Verdict | Evidence | Replacement text if NARROW |
|---|---|---|---|
| LR-02 | **KEEP** | Runner fails closed on missing declared outputs; visibility’s four configured outputs still require unavailable numerical tools. `scripts/oracle-fixtures.json`; `R17/implementation.md:181`. | — |
| LR-04 | **NARROW, retain H** | Required Arrow, notices, workspace export checks and isolated packed checks exist. TOOL-102, 104, 105 and live hash drift prevent present qualification. | “Repair native declaration resolution. Confirm canonical repository, version and tag. Review package inputs, qualify one final tree, retain the checked archive with SHA-256 and publish those identified bytes.” |
| LR-19 | **NARROW** | `dev-docs/release-package.md:179`, `:208`; landed source/test peaks 2.15/3.52 GiB. Experimental comparison had zero RSS samples. | “Keep current source configs and family test projects. Decide on reference restructuring after same-heap, idle-machine RSS, cold and edited-warm measurements meet the stated thresholds.” |
| UF-01 | **REPRIORITISE to H for publication** | `modules/analysis/package.json:10` and README use `visgl/luma-contributors`; checkout remote and committed website header use BertCh. | Maintainer decides canonical public identity before final packaging. |
| UF-03 | **NARROW** | Existing benchmark suites and historical measurements exist; no current alpha.16 primitive-regression baseline was qualified. | “Record exact revision, dependencies, device and workload. Compare current published primitives with the historical compaction, sort, join, projection and residency cases.” |
| UF-10 | **NARROW** | Contributor contract and package guide exist; manifest public eligibility differs from registry publication. | “Finish the checked-tarball publish command and tag. Verify version availability. Keep lenses and kepler private.” |
| UF-12 | **NARROW** | Local example-backed test project exists. CI configures full checks for three examples after build. `examples/deck/spatial-analysis-showcase/scripts/check-sprint-scenes.mjs:12` references missing `collect-page-diagnostics.mjs`; the launch-options helper exists. | “Local verify checks selected example imports. Standalone app typechecks are configured in CI; hosted completion and visual acceptance remain open. Restore the missing diagnostics helper and decide the explorer check.” |
| INT-08 | **KEEP** | `.github/workflows/test.yml:17` configures CI and a website build. No hosted run or Pages deployment-policy decision was established. | — |
| PLAT-06 | **NARROW** | 337/790 fixtures have configured producers. GeoConformal, CAST, ETAS and GpGp lack the named external-execution comparisons. Phenofit/GRASS, bfast and xeofs have definition-based references. pyxccd, track-similarity and Markov have identifiable external generators and consumers. | “Inventory by generator, version, fixture and consuming output. Distinguish external execution, reimplementation and regeneration. Current configured coverage is 188 producers/337 outputs; record the remaining 453 fixture owners.” |
| PLAT-07 | **NARROW** | `T/gpu-spatial-analysis/spatial-join/spatialbench-scale-benchmark.spec.ts:33`. At SF100 the helper gives 600M points: 4.8 GB for two float32 coordinates; 9.6 GB for two JS numeric coordinates before overhead. JS tuples and flattening remain. | “Synthetic square-zone join, default SF0.1/SF1. Use typed arrays before higher SF. Add Overture, aggregation-only and KNN workloads. Link raw evidence for the historical 216.3 ms SF5 result.” |
| LUM-30 | **NARROW** | `T/analysis-core/gpu-resource-counts.spec.ts:26`, `:50`; spacing checks caller ownership and repeated execution. TOOL-121 demonstrates a failure path. | “Spacing and three fitting contributors have repeated teardown checks; active texture counting has a separate real-texture control. Extend to contributor-owned textures and allocation-then-failure cleanup.” |
| LR-17 | **KEEP** | `scripts/contributor-contract-baseline.json` still names five network and sixteen raster roots. | — |
| LR-22 / KI-04 | **NARROW** | `E/tranche-3/final-node-and-history.txt:1` used four workers. The bump’s 30 grid cases have a focused recorded pass. `T/gpu-raster/cost-distance/gpu-cost-distance-continuation.spec.ts:171` checks costs and iteration count separately. Historical round-count mismatch occurred twice in ten, followed by twenty passes. | “Grid agreement has specific device evidence. Decide the continuation round-count guarantee using diagnostics. Run the bounded full gate on the release tree. Track segmented sums/PCG separately.” |
| LR-20 | **KEEP** for release restriction | `modules/lenses/package.json:4`, `:47`; embedded datasets and private dependency direction persist. The loader corrections do not clear dataset rights. | — |
| KE-TESTS / KE-PORT-6 | **NARROW** | One runtime and 58 typecheck exclusion paths exist; 382 classified example-backed files are admitted. The old 43-package-only tier is not a current census. | “Requalify each recorded exclusion by manifest and keep runtime membership distinct from typecheck roots. Cold-cache DuckDB qualification remains separate.” |
| INT-04 | **CLOSE obsolete rationale** | Current `test/kepler-compute-example-specs.json:1` records contemporary diagnostics. The alpha.10/physical-transient description no longer describes this manifest. Keep remaining qualification under KE-TESTS. | — |
| KE-PORT-0 / KE-EX | **NARROW** for tooling | Root pins and example workspace registration exist; the workspace export check includes kepler; the isolated packed consumer covers analysis. | “Keep adapter/launch checks with the example owner. Package pins, workspace registration and local example-backed test typechecking are implemented.” |
| STD-12 | **NARROW** for test layout | `T/spatial-test-utils.ts` remains at the root. Legacy `gpu-core`, `gpu-dataframe`, `gpu-raster`, `geospatial-expansion` and newer source-family directory names coexist. | “Document the test-to-source family mapping before moving fixtures. Preserve provenance and discovery membership during moves.” |
| LR-26 / KI-07 | **KEEP** for landed behavior | Current three constructors enforce 4,096 tiles; the guide’s no-cap statement is stale. Work in a separate staging copy does not change the inspected package. | — |
| KI-01 / KI-03 and LR discriminating-test rule | **KEEP** | The 30-case ledger supports targeted pins; TOOL-109, 112–115 show missing evidence outside them. `73e8f638` and `R17/implementation.md:195` identify late-fix scope. | Preserve per-output oracle categories and per-fix historical evidence. Do not close the complete cross-stream review from this sample. |
| PLAT-04 / PLAT-05, maintainer decisions | **NARROW** for repository guidance | Compensated sums and the pinned-copy policy already have homes in the contributor contract/upstream-first plan. This review established neither a generic device iteration primitive nor a portable buffer-view implementation. | “Link existing contributor rules from AGENTS; keep upstream filings and device/compiler qualification as distinct actions.” |
| SH-DATA / WEB-INT-16 font decision | **KEEP**, outside analysis tarball | `THIRD-PARTY-DATA.md:48` and package selection separate website font/example assets from analysis. Lenses’ noncommercial/share-alike data remain a private-package consideration. | — |
| dataframe-10 / known Space-time K–Diggle | **KEEP** | `T/oracles/space-time-diggle-splancs-port.py:5` identifies a source-reading port, source archive and excluded behavior. It does not establish an executed R comparison or decide licence acceptability. | — |
| Known COLD / Sky view | **KEEP** as bounded comparisons | `T/gpu-raster/raster-time-series/rts04-cold-pyxccd.spec.ts:461` checks synthetic series with explicit mismatch bounds. `T/gpu-terrain/terrain-illumination/gpu-terrain-horizon-rvt.spec.ts:378` compares where samples coincide. | — |
| Known Lag ML / panel random effects / probit | **KEEP** as numerical-fixture limits | `T/gpu-spatial-analysis/spatial-regression/math-review-lag-conditioning.spec.ts:504`, `math-review-panel-lag-conditioning.spec.ts:362`, `math-review-probit-bound.spec.ts:239`. The named fixtures exercise specific near-boundary statuses; gate success cannot generalize those guarantees. | — |
| Known GPU triangulation / OPTICS xi | **KEEP** | `T/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation-search.spec.ts:259` compares with Earcut; `T/gpu-spatial-analysis/spatial-clustering/optics-xi-dependencies.spec.ts:155` exercises 65,536 rows. No expanded general limit was qualified here. | — |
| Known urban tessellation / visibility graph | **KEEP** and qualify inventory | `T/gpu-spatial-analysis/urban-ecology/building-tessellation-improvements.spec.ts:174` pins a 2,500-building case. `visibility-graph-metrics.spec.ts:520` tests insufficient path budget. Algorithm-specific tests do not close the separate missing qualified kernel-audit identity. | — |
| IN01 | **KEEP** as historical | `dev-docs/research/evidence/interpolation-advanced-2026-10-23.json:2105`, `:2159`. CPU readback/fit/upload medians 1.3/3.2 ms vs resident GPU 4.3/51.3 ms for 20/256 bins. `T/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md:42` still gives a removed `modules/graph/test` command. | Preserve date/device and correct the command. Remeasure before making a current performance claim. |
| Known Published fp64 | **NARROW** | `T/analysis-core/shader-fp64-module.spec.ts:119` checks eight positive operand pairs and a 100,000-term sum, against the uploaded double-single values. | “Operation-specific accuracy is pinned for these positive fixtures on the test adapter. Float64 exponent range, all domains and cross-vendor accuracy remain unqualified.” |

## Next steps

### Before the alpha

| Rank | Action and acceptance condition | Owner and beneficiary | Cost |
|---:|---|---|---|
| **1** | **Finish the native package surface.** Resolve all 22 declaration edges, pin a broken schema through its public entry, add `.cts` coverage and preserve ordinary strings in the CJS transform. Acceptance requires successful resolution of all selected relative declarations and equal literal values in the two module formats. | Package/build owner; TypeScript and CommonJS consumers | **M** |
| **2** | **Make a passing regression meaningful.** Fix the six demonstrated nonfinite false greens, replace the nine silent returns with requirements or explicit skips, reject focused correctness cases, and pin bivariate counts alone. Add the missing network visibility audit identity. | Test and family owners; contributors relying on local green results | **S–M** |
| **3** | **Qualify identifiable bytes at iteration end.** Settle release identity, review selected files, stage the final package, run the full one-worker/resource-monitored gate on its corresponding tree, retain the exact archive with digest and record actual test inventory, device, dependencies, options and completion. | Release maintainer; first installers and subsequent support | **M** |

**Publication checklist**

| Ownership | Decision or work | Completion evidence |
|---|---|---|
| Maintainer | Select public repository URL/branch, verify whether `0.1.0-alpha.0` is available, and choose the alpha tag. | Package URL and installed README destinations agree; release procedure names the explicit tag. |
| Maintainer | Decide the supported compatibility statement for alpha.16, allowed alpha.17/stable patches and Arrow `>=17`; decide whether source and the developer build record should ship. | Documented tested matrix and reviewed package file list. An accepted range alone supplies no qualification. |
| Mechanical | Fix TOOL-102; add native consumer checks; obtain owner cleanup of the two untracked source files. | Declaration-edge census is clean; selected package contains the approved source, licence texts and notices. |
| Mechanical | At the end of the iteration, run the complete local gate and clean release staging without changing inputs between qualifications. Resolve any continuation mismatch against its documented contract. | Uninterrupted full summary, exact inventory, stable source/toolchain identity and retained stage. |
| Mechanical after maintainer authorization | Verify retained tarball SHA-256 immediately before publishing that tarball with the selected tag/access. | Published archive identity matches the receipt. Lenses and kepler remain private. |

**Developer tiers.** These are proposed budgets for an already installed, warm checkout. The historical timings are evidence for sizing them, not promises.

| Tier | What the contributor runs | What the result qualifies |
|---|---|---|
| **5 minutes** | Lint, both policy checks, source typecheck, warm build/export/packed checks, then the changed Node case and smallest relevant GPU regression. Historical non-test work excluding test typecheck totaled about 47 s. | Fast feedback on the edited path. Select the domain’s data/representation boundary explicitly. |
| **30 minutes** | Above, plus the full one-worker Node suite and selected seam/GPU cases within the remaining budget. Node alone consumed 25.7 min. A change to test types can substitute the family typecheck and targeted Node set. | Broad host integration or affected type coverage. Label which workload fit the budget. |
| **Full gate** | Existing ten sequential steps, exact headless inventory and durable receipt. Budget roughly 91 warm Mac minutes and allow for cold caches. | Release-tree qualification on the recorded adapter. Benchmarks, optional studies, lens capture, full app projects and visual acceptance have separate results. |

### After

- **Consolidate compiler and policy helpers — M.** Build and checker maintainers need shared output planning, dependency-aware freshness and AST import/export resolution. Retire redundant spelling-based checks after behavioral negative controls cover their intended policy.
- **Connect fixture provenance to consumer outputs — M.** Algorithm owners need to know which public result has a closed form, external execution, source-reading reimplementation, twin-only comparison or no independent oracle. Start with optional outputs and the 453 fixtures without configured producers. Freeze new fixture formats until generator identity, expected outputs and readers are explicit.
- **Scale testing from measured costs — M.** Preserve per-directory timings and actual worker counts. The slowest recorded GPU chunks were spatial join 241.187 s, kepler root 159.791 s, spatial clustering 137.892 s and spatial regression 117.247 s. Move timing repetitions out of correctness and decide which bounded subset CI owns. Keep SwiftShader and native Metal evidence distinguishable.
- **Freeze compiler-project restructuring pending LR-19.** Run the existing experiment on an idle machine with usable RSS sampling and representative edited-file warm loops. Keep current source configs unless the agreed threshold is met.
- **Remove low-value baggage after checking consumers — S.** Decide the shipped build-record policy, confirm use before removing Arrow sample assets or the apparently unused direct `probe.gl` dependency, and remove or implement the uninstalled Istanbul coverage configuration. These changes have lower release value than repairing public types.

## Limits of this report

The numerical assurance result is a purposive sample of 30 changes. It does not cover every public result, contributor or unreviewed fix round. The four historical evidence gaps are stated explicitly; old structural failures, interpreted WGSL, host reproductions and real device failures remain distinguishable. The second tranche, stage-dependency fixes, state cleanup and bump were sampled at their tooling seams. Arbitrary execution-order independence remains unverified.

The report establishes a public declaration-resolution failure, a build-helper semantic failure and a resource-policy rejection through executed CPU probes. It makes no claim of a newly observed wrong GPU result. Existing exports, fixtures, recorded timings and pack selection were inspected without producing fresh artifacts. No physical GPU memory allocation, hosted CI outcome, current registry state, public URL availability or legal clearance was independently verified. The full release gate belongs at the end of the iteration on the exact tree intended for publication.