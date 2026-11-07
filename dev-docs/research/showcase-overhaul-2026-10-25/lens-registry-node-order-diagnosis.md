# Shared-worker scenario fixture collision

The full Node run failed while S3 imported the production scenario registry. A fixed-order two-suite
reproduction shows that `static-twin.node.spec.ts` first registers its synthetic guard fixture under
the production ID `gpu-core.scan`; S3 then correctly rejects the genuine production descriptor as
a duplicate. No module reset or import alias is needed to reproduce this failure.

The test fixture now uses `static-twin-test.scan`, a matching entry, and a local recorded envelope
with that ID. Guard/rung assertions remain intact. Production registration and duplicate rejection,
module-reset tests and shared-worker configuration are unchanged.

Before the fix, six static-twin tests passed and the following S3 import failed. After the fix, the
ordered static-twin, duplicate-rejection registry and S3 suites pass all 13 tests in three files.
Scoped Biome passes. The four-suite reset control passed 63 tests before the fix; no reset change
was justified. Root owns the full repository rerun.

[The durable JSON record](lens-registry-node-order-diagnosis.json) contains exact commands, the
ordered temporary configuration, before/after logs, SHA256 hashes and limits. The configuration
forces static-twin first; an ordinary cache-ordered run may load S3 first and conceal the defect.
No production source or GPU artifact changed, so the wave 21 source fingerprint remains valid.
