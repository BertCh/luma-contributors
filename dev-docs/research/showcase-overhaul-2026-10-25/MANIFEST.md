# Manifest of files left behind

These files exist on disk in the luma.gl `map-contributors` worktree at `dev-docs/research/showcase-overhaul-2026-10-25/<path>` and were deliberately not copied into this repository. Large JSON snapshots and tarballs should become a release asset (for example `evidence-2026-10`) with these checksums; scripts and bytecode are reproducible from sources elsewhere. Sizes are bytes; sha256 from `shasum -a 256`.

## Large JSON and tarballs (1 MB or more) - candidates for a release asset

| Path | Size | sha256 |
| --- | ---: | --- |
| atlas-wave21-sweep.json | 1540189 | 3f4dbb3cae6b022b7c74f96911d67acba3426155cbbc1d0b97124be3d08de12f |
| functionality-after.json | 12028939 | ed925adbbccaf43dee0f3ef5dba8810d3d4135cc4b31fa52303127a2cebee895 |
| functionality-baseline.json | 3148466 | a823bb9d3f3f7ab1f7e27707988b948762d191aeaa2fafcd12c3e35b821c1b9d |
| functionality-concurrent-decisions.json | 9522686 | 2e9aaabd6686d4527909b6048c3bb08429872bd5bb7d302b46ef0544d97e32d4 |
| functionality-source-final-raw.json | 12028939 | ed925adbbccaf43dee0f3ef5dba8810d3d4135cc4b31fa52303127a2cebee895 |
| functionality-source-final.json | 9525272 | faa4711f3afa25bb9673c985bddd55c6dbaf8630dcf0b05a88ab046b894c3a2d |
| functionality-source-wave16.json | 9526390 | 2562719a67bfb61c99a832735db63974bffc94d2964118136ad432f961a7bb93 |
| functionality-source-wave17.json | 9525086 | ffe5197bd69f1e470dc7c4f58f0e7746a20815b5b0a16a11546b42be34273f47 |
| functionality-source-wave19.json | 9525161 | bd73086acd7c71a29e9894d1c0e3dd2000ee8f87e5e11fb74154b6e1d1c40944 |
| functionality-source-wave20.json | 9525161 | 58eb9e347dfb73f6174722d68cc79c6dc57a78c58f65a5b21bba093e4f65a942 |
| functionality-source-wave21.json | 9525161 | 50b007661696ae0d455e3906156eb1631ff6b4dcfc9b0400ec05b78b0c3af528 |
| functionality-source-wave22.json | 9525161 | 314c66c0893caec8d8ceb54d33eae7a70c1b5ff29001687e5cc7f0c476fe2c25 |
| source-audit-wave17.tar.gz | 1603094 | 355d6ce5293f75f9fe23a9d1be6a4e0afa547141f06469d5383189ba93da3baf |
| source-audit-wave19.tar.gz | 1603027 | 65ab2d4e2590d74bbde70c0c48ed32d0716cfc392182694fd93fcbcab2662c79 |
| source-audit-wave20.tar.gz | 1603020 | 1171362994355d08f89df0944419af12b7a94a84b30c48a10b2d6b3b5d928f0f |
| source-audit-wave21.tar.gz | 1603009 | c790eb554ab5823e1da16e7f401bd34e5c12d15662bfd11078ca251656437f9d |

## Reference scripts, bytecode (not copied per migration policy)

The `*-reference.json` oracles that the showcase scripts consume were copied; the Python and Node scripts that generated or check them were not. Do not copy `.pyc`.

| Path | Size | sha256 |
| --- | ---: | --- |
| __pycache__/fire-policy-reference.cpython-315.pyc | 15122 | 5d22b24a9ac356df0f3559f21a2299426286a25d3a90baa8b62c20ce7c3beccd |
| __pycache__/movement-network-reference.cpython-315.pyc | 13338 | 25ff60f5420b017cef8087936d41888968c7218cc14920ace390c5c1dc966513 |
| __pycache__/rail-support-reference.cpython-315.pyc | 11259 | a4d8e4c929240eeded815f8032475af37fbff280183a29868c1618357077af17 |
| __pycache__/reconcile-functionality.cpython-315.pyc | 38154 | f687188f6f9bb7c78272c4d5326e16426c2364e3f9a681443e801d4465e9cc6e |
| __pycache__/reference-rainfall-fit-loo.cpython-315.pyc | 25719 | f74290d4b043c87a325906490f391bfdaad94a2300648d3839f33764f33209e4 |
| ais-encounter-reference.py | 8673 | 7035bda925f79ce17b422b9b1d52bd2eca47c764256162f5bc816e3f10cf74e8 |
| aviation-capacity-flow-reference.mjs | 10657 | 5146720303b2d2021c267d2525c8f05c791de412df5dcb66f58d54117f17bcd7 |
| aviation-network-reference.py | 4256 | 834ce378b46229caad009eb8b87a1fac459eeb3e6fc37b34bf3919b9bde63a59 |
| bixi-models-browser-harness.mjs | 17070 | 52b694052e940dbe27dbfca2539f2706020cdc8caf858f74a9f112476fd96356 |
| check-chicago-population-numerical.mjs | 6571 | 5d621408939466a4a56360e00ee76782015850fb33013a500a699cd4b78834ce |
| check-chicago-population-support.mjs | 8051 | e2021384ef372579579a56098443a130dff1f8fcc81bc9598aed612931355016 |
| check-county-classification.mjs | 7125 | 051949ff923b78fcb30d9303e50bade66c8c7cb25fa3053a35ddc409a7d1bfa0 |
| check-county-dorling.mjs | 6037 | 738c0efbe952c590e7bcabef94228c3f02220d4ceb7f4b7bdb3f26078a2ea3f8 |
| check-county-iv-covariance.mjs | 2404 | efb2e2f15810e987346f049a06c84b09cdc4a2fa3472337b13c135753a99d31d |
| check-county-models.mjs | 8627 | e143734df9f567add7c84578938c304264753355da8c60a6265fafbbb199ec17 |
| check-county-pairs.mjs | 6309 | ee90962e71ba7e0cec5c58edbded7f5b566c0e58f01229f484f78c4214701cae |
| check-health-validation.mjs | 6950 | 2e6719fdde7aa3f3e2fa07a554551a76b1bf573e934bd98112cbff100e882dac |
| check-hotspot-projection-domain.mjs | 2162 | fc12e4b1464341419de2bd65dc2d9fc4ee2950caee418b8933e4f6d9b83ba17d |
| check-hotspot-runtime-smoke.mjs | 582 | 2662bb5056406cfd00d4a9a6bdacaa4d037766b143784550cc5a5c6ea73510ca |
| check-massachusetts-weights.mjs | 6669 | 444a65f3eeb3e5b6cb3cae78c1e495639706434280e745ded15b274c36efc400 |
| check-meuse-labs.mjs | 8988 | 03890cfb1b9d4133d2370ea18b4e4edf191d5ed3295a16c3a0bbbbdf99f3fbd1 |
| check-meuse-model-form.mjs | 9034 | 557f2c90b1feaf3e4426aa8212d0df90e1c5d153a6e7d848fbf6ebe73803aafb |
| check-meuse-neighborhood.mjs | 6445 | 599341d62021ebb972a4e0d744781bea6e22ced2b9e5d2bb9148be6817ae43db |
| check-meuse-quantile.mjs | 2591 | c8dd386cc9896cdf65aa3fd2ecce9c31203ee4c85c403311272c96e5296e2c45 |
| check-meuse-regimes.mjs | 4654 | 506067ebd4ccfa32acf1378400a0592db3897d2089db2f449228a0f51bf0ec1c |
| check-meuse-validation.mjs | 7212 | 08e305ff1e8320cc4b61d0f9b22b99400a5f81877f730514ff23e1e772fb142f |
| check-nature-kernel-density.mjs | 5104 | aeeb89962dd4277bd06876eb0b938f051490e8cafdf856e50c3a2d75d4e71b15 |
| check-point-pattern-global-evidence.mjs | 1193 | ffecd28e8550385171e57ea89fd7545b1b7dc0e1f8420d2203f8a2a6731e7bd5 |
| check-pointwise-csr.mjs | 4474 | 11282509cdc71f3a71be41aef93b7465f4330c2df71510d851175dcb9f093c15 |
| check-rainfall-fit-loo.mjs | 16474 | deb0db0e152cb2d047c633047f7eacba007e6727b2fb586478e66992904416ec |
| check-rate-recipes.mjs | 2801 | f45832d71317fb646b49635be1dab8d324cc59920920a0ddc71a7f90e3435260 |
| check-statistics-readout-contracts.mjs | 2837 | ca131ce4d6e7d845a1232c0c91efd7eda7482b4a593c581f46d934e4f455de16 |
| chicago-restriction-reference.mjs | 9765 | df7e41de9b24f091b076986cc650dea352524cb2bcf90f54bca1156799912a74 |
| chicago-route-reference.mjs | 7952 | ba348d628da24964d30ee8fbde499b24100985c9ed5e86799eb30b97926327de |
| county-expression-reference.py | 2204 | 5566b68010b41990df4eb1fdf55b01b48fb8428cd80a1d7e1687674a9eb276ce |
| county-nan-sort-reference.py | 4286 | e50a9a127eb46fbc492dd6020591f91c7200c10b7925d73fbf3122ccc4289f16 |
| county-query-reference.py | 3971 | 0c718b8bcf7de9cd29af97d6adf351e845707328499b12b6772cb5a164e7fd1b |
| earthquake-standardization-reference.py | 3487 | 40bc348eb313c64e94365f3581dc2e8344660e8d00631d2f498408af4884daff |
| fire-policy-reference.py | 8833 | f0dff6ef24056f3b6aec9de6bff2c787822b6d974c55120221813f27bfe94326 |
| fire-topology-reference.py | 8852 | d7c857ce4ffb469990b21ced714372e5acc83f2cfc4c97e51147bc98e066f35e |
| habitat-circuit-accumulation-review.py | 3049 | d8747df08a41d6021aeba9f9ad894c3dc19331a214383f4bb4be3db678b42cdf |
| habitat-circuit-budget-matrix-review.py | 2144 | 6f49a8f3b8f46025afd721707edc6ba1ca405f97344b61411634f0e3b6308e38 |
| habitat-circuit-float32-review.py | 4682 | da069437469df852343e454245858518f9b445625aba75ef150409b4e42981f3 |
| habitat-circuit-fused-update-review.py | 3008 | 2ce8a7c30f875aee9bc3310b47be11bea0ca7964c88ffabcbfddf17357cf5fe1 |
| habitat-circuit-wave18-physical-audit.py | 5124 | eb52dabb57cf651074eef77d21deb63eda2748855abff0b9a8e574f147bfd728 |
| harbor-polygon-support-reference.mjs | 8103 | be8c8d75cad212dd1674f2de8b42aff6fbab7430ed4db36d449a100a056bfd23 |
| movement-network-reference.py | 7809 | 009095c364daf3262bf215776864b6a0c34b532a735c466ec3fa22100fcd3ff3 |
| osm-group-reference.py | 3030 | 58014efcd894f67a1b69738559ebd1febd5a799bdcf217505ec28b1039b41efb |
| osm-temporal-reference.py | 2203 | f5dbedd6e6abafcf635f93fcab58d1649a193f57b6a62f1bfbb4bd42d431fbd2 |
| projection-contract-reference.py | 3421 | 599043addf25585454fe76245c8c54cb2f751e5adb11d90900ba5ae9988574df |
| rail-support-reference.py | 5865 | c5ab6863a64ab887b93250d929b928075d14a4bf9334840437f8c70d342278dc |
| reconcile-functionality.py | 31899 | 326f34eda81be88f6873fb3d61901e238a4a0680406e9ba0c472551d765855f7 |
| reference-chicago-population-support.py | 5016 | e5c48d02d9c1542968582eea16b0a3b78e635969a7e4a6de88684abb36d181c4 |
| reference-county-classification.py | 11455 | cb5f091212ab2d6b716c3841363565e3c7acecc94c820700cd6f5791b1881360 |
| reference-county-iv-covariance.py | 6518 | d5ea2b92ec7dda5620a89483457e3e625daf6dd0d643d92e5b3f337843a27bd5 |
| reference-county-pairs.py | 6323 | f3ae51a818f2b069b8cada5fee7fe1927fc57ec65b6514fd8e40dab64c83cbb3 |
| reference-county-panel.py | 5957 | 247bb8eb1ddfa4211beeb72ddc8d618fc202b3040e27845c972a173814f1aab7 |
| reference-massachusetts-weights.py | 4631 | 0914a997afa082aad4effdb8068f60749d034d45208a5916e8111730aacfde0a |
| reference-meuse-conditional.py | 5299 | a0746f3bf67ae118e8eede3225a70017d91f76e3f47b2a6af4d61c57e11653d8 |
| reference-meuse-model-form.py | 10450 | bee3f94bbf03b514c2087f5f5963c121d56142ea975e1058db03adc9e73acffa |
| reference-meuse-natural-neighbor.py | 5524 | d0b80eddd36ed5ff8460e9bd4807736006e43dba1f4393345c49151c900b3a2d |
| reference-meuse-neighborhood.py | 3881 | acbf603e8a38ea6ab74a02d64a4183df62d9aed3779991f9953c74f8ca544976 |
| reference-meuse-nested.py | 6930 | ce60b7cc8239a38f9a0051b50130e43da386bd903aaa979917181f4766f54aa5 |
| reference-meuse-quantile.py | 2285 | 6be6a3e60407f963da868934339832414665bdb01d69d3156969bbafbaa93eac |
| reference-meuse-regimes.py | 4077 | 955ac79b082b8bc492a4de5a133ab475a21121ef64ddde035bb6d21db6f98596 |
| reference-nature-kernel-density.py | 5784 | 26efce63b7493055fe90be54db0d8ead2c0693b3ccd085003324660d91958cae |
| reference-rainfall-fit-loo.py | 12894 | 01a49c76716e65624f63bee79402d27e7d9c1895f18bc4dec81ee555d9891f3e |
| reference-rate-recipes.py | 3311 | 18aa6cb0572ee07c6d482527d2eaef50cb32dd66dcb5aaed2f6208f41fc741b3 |
| retail-network-intensity-reference.mjs | 12539 | 258b90b916c310be8af82df20d5d22b758d61dd136950a9c86c6667b54099c0e |
| statistics-browser-check.mjs | 4760 | 6d513fc41f351256873c848273a970dd890e81641c82ca3b24ae9f66b336e43d |
| verify-spherical-observation.py | 2120 | c2718646bb17fcf1ec8cdc52e0166bb9ef6e2278f88de3e260f51b7012214de5 |
| write-source-manifest.py | 3110 | 8bd641bfdd24665ba26074550423aa773f99e7c8b8b194f8735a379ebd993f86 |
