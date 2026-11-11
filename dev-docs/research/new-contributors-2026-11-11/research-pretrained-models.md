# Pretrained geospatial models and embedding products for the browser

Research for new contributors, 2026-11-11. Question: which pretrained geospatial neural models and embedding
products could run, or be consumed, entirely in a browser on WebGPU, inside the constraints of this
repository (float32 WGSL, one device, 128 MB default storage binding, weights ideally under 10 MB and never
over 50 MB without a user fetch, about one second per run on an Apple M-class integrated GPU, no neural layer
library).

Every factual claim carries a source. Numbers marked **estimate** are my arithmetic, with the working shown.
"Not verified" means I looked and did not find a primary source.

## Summary

- The cheapest win is not a neural network. Precomputed embeddings (AlphaEarth, TESSERA, OlmoEarth v1.3,
  Earth Index, Clay) need only vector operations, and the eleven existing `RasterEmbedding*` classes cover
  most of what people do with them. The gaps are a tree-ensemble predictor (Google's own regression
  tutorial uses a 50-tree random forest), a regression output on the linear probe, more quantization
  formats, more than 128 dimensions, multi-year trajectories, and anomaly scores.
- The best first neural contributor is a **location encoder** (SatCLIP): coordinates in, a 256-dimensional
  vector out. It needs only a spherical-harmonic basis, three dense layers and a sine, all of which map onto
  the published `GPUMatMul`. MIT licence, about 1 M parameters.
- The best first image encoder is **Galileo-Nano** (0.8 M parameters, MIT): small enough to download, and
  cheap enough for 1024² in well under a second by the estimate below. It still needs a transformer layer
  kit (LayerNorm, attention, GELU) written in WGSL, which is the real cost.
- Promptable segmentation (MobileSAM), cloud masks (OmniCloudMask) and land cover (Dynamic World) are all
  technically runnable but carry a large layer inventory, unclear weight licences or both. Skip them as
  contributors for now.

## 1. Precomputed embedding products

| Product | Dims | Storage | Resolution | Format | Licence | Access without account |
|---|---|---|---|---|---|---|
| AlphaEarth Satellite Embedding V1 annual | 64 | int8, `sign(q)(q/127.5)²`, -128 NoData | 10 m, 2017 to 2025 | COG, 8192² tiles per UTM zone, internal overviews; GeoParquet / GPKG / CSV tile index | CC BY 4.0 | Yes: Source Cooperative mirror, and the GCS bucket is "provider pays" since July 2026 |
| TESSERA v1 | 128 | int8 with per-pixel float scale | 10 m, annual | GeoTIFF or npy through the `geotessera` CLI | Weights CC0 1.0; embedding data CC0 per CNG survey | Yes, through a university API |
| OlmoEarth v1.3 | 128 (first 64 usable alone) | not stated | 10 m; 2023 to 2025 global | Source Cooperative tiles; one secondary report says Zarr (not verified) | OlmoEarth Artifact License | Yes (Source Cooperative) |
| OlmoEarth Studio exports | 128 / 192 / 768 by model size | int8, -128 NoData | 10 to 80 m | COG | not stated | No, Studio users |
| Earth Index (Earth Genome) | 384 | float32 | one vector per 320 m patch | GeoParquet, Major TOM grid | CC BY 4.0 (per CNG) | Yes (Source Cooperative, S3) |
| Clay v1.5 (Sentinel-2 / NAIP) | 1024 | float32 | 2.56 km / 256 m patches | Parquet (geometry, embeddings) | CC BY 4.0 | Yes |
| LGND Clay v1.5 monthly | 256 | not stated | not stated | GeoParquet 1.1 | CC BY 4.0 | Yes |
| Major TOM embeddings | 2048 (per CNG) | float32 | about 3 km | GeoParquet | CC BY-SA 4.0 | Yes (Hugging Face, CDSE S3) |
| Presto embeddings | 128 | uint16 | 10 m | GeoTIFF | CC BY 4.0 | Hugging Face |
| ESD (Chen et al., ESSD 2026) | 6 latents x 12 months, FSQ codes | uint16 code index, 65,536-entry codebook | 30 m, 2000 to 2024 | [12, 3600, 3600] tiles on MGRS | not stated for data | No: credentials listed in the paper |

Sources: AlphaEarth GCS readme <https://developers.google.com/earth-engine/guides/aef_on_gcs_readme>,
Source Cooperative mirror <https://source.coop/tge-labs/aef>, AWS registry ("No AWS account required")
<https://registry.opendata.aws/aef-source/>; TESSERA paper <https://arxiv.org/abs/2506.20380> and model card
<https://huggingface.co/geotessera/TESSERA-V-1.1>; the CNG survey of nine products (dims, dtype, resolution,
licence, storage for Africa) <https://cloudnativegeo.org/blog/2026/02/the-technical-debt-of-earth-embedding-products/>;
OlmoEarth v1.3 <https://alphasignal.ai/news/ai2-s-olmoearth-v1-3-turns-a-year-of-satellite-imagery-into-128-numbers>;
OlmoEarth Studio embeddings <https://allenai.org/blog/olmoearth-embeddings>; Earth Index
<https://www.earthgenome.org/blog/embeddings-for-all>; Clay on Source Cooperative
<https://source.coop/clay/clay-v1-5-sentinel2>, <https://source.coop/clay/lgnd-embeddings/monthly-aggregated>;
Major TOM <https://dataspace.copernicus.eu/node/2794>,
<https://huggingface.co/datasets/Major-TOM/Core-S2L1C-SSL4EO/raw/ff6801cb6ebfd947d7e974aaa227a3b4571728bd/README.md>;
ESD <https://essd.copernicus.org/articles/18/5375/2026/>.

Notes that matter for this library:

- **AlphaEarth** is the format the existing classes were built for, and it is now the easiest to reach: COG
  range requests from Source Cooperative, no account. The readme confirms the overview rule the
  `RasterEmbeddingPyramidLevel` implements ("dequantize, sum, renormalize").
- **TESSERA** dequantizes as `int8 * scale` with a separate per-pixel scale file, according to its model
  card. `RasterEmbeddingDequantize` does not support that (its source says "the TESSERA int8 scale is not
  verified"). The CNG survey says TESSERA's API returns raw arrays with "no CRS, no bounds, no metadata".
- **OlmoEarth** int8 uses the same -127..127 / -128 NoData convention as AlphaEarth, but the decode formula
  lives in `olmoearth_pretrain.dequantize_embeddings`; I did not verify whether it is linear or the
  AlphaEarth square law. Its licence restricts military, defence and extractive-industry uses
  (<https://voxel51.com/blog/olmoearth-fiftyone-satellite-embeddings>); it is not an OSI licence.
- **Major TOM** is share-alike (CC BY-SA 4.0), which is the problem flagged in `THIRD-PARTY-DATA.md`.
- **ESD** is unusual: an FSQ code per pixel-month that a small Conv1D decoder (ESDNet) expands. Decoding
  in the browser would need that decoder and its weights; access needs credentials. Not pursued.
- Patch-level products (Earth Index, Clay, Major TOM) are tables of vectors, not rasters. The published
  `GPUIVFFlatIndex` and `GPUSimilaritySearch` already serve them, but every one exceeds the 128-dimension
  limit of the raster-embedding classes (384, 1024, 2048), as does OlmoEarth Tiny (192).

### What people do with embeddings

Google's tutorial series (<https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction>)
covers clustering, supervised classification, regression and similarity search:

- Supervised classification uses `ee.Classifier.smileKNN()` with about ten labelled points
  (<https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-03-supervised-classification>).
  Covered by `RasterEmbeddingKNearestClassifier`.
- Regression uses `smileRandomForest(50)` in regression mode to predict GEDI above-ground biomass from
  1,000 samples, then aggregates to 100 m
  (<https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-04-regression>).
  **Not covered**: there is no tree-ensemble predictor in the census.
- Similarity search takes the dot product with each reference, **averages** the similarities, thresholds at
  0.90 and vectorizes the matches
  (<https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search>).
  `RasterEmbeddingSimilarity` gives the maximum and per-query values; the mean is not an option.
- Ai2 shows cosine similarity, a 60-pixel logistic-regression probe, change by cosine distance between
  years, and PCA to RGB (<https://allenai.org/blog/olmoearth-embeddings>). All covered.
- Earth Genome lists content search and change over time (<https://www.earthgenome.org/blog/embeddings-for-all>).

### Embedding operations missing from the eleven classes

1. **Tree-ensemble prediction** (random forest, gradient-boosted trees) per pixel, from a host-exported
   model. Needed for Google's own regression recipe; also runs s2cloudless-style pixel classifiers.
2. **Regression output** on the linear probe (identity link, `W x + b`, several targets). The current probe
   is softmax only.
3. **More decoders**: TESSERA per-pixel scale, OlmoEarth int8 (formula to verify), Presto uint16, float16;
   and the inverse int8 encoder that the docs already say is missing.
4. **More than 128 dimensions** (192, 256, 384, 1024) where the binding budget allows.
5. **Multi-year trajectory**: nine AlphaEarth years per pixel; cosine to a base year, year of largest
   step, cumulative angle. `RasterEmbeddingChange` handles two years.
6. **Novelty / anomaly score**: Mahalanobis distance from the `RasterEmbeddingCovariance` mean and inverse
   covariance, or distance to the nearest training vector, to flag pixels unlike any label.
7. **Mean-of-references similarity** aggregation, matching the Google tutorial.
8. **Embedding edge strength / local heterogeneity**: mean cosine to the 8 neighbours, a boundary map that
   feeds the existing watershed and sieve contributors.

## 2. Small encoders

| Model | Params | Weights (fp32) | Layers | Input | Licence | Export | Benchmark |
|---|---|---|---|---|---|---|---|
| Presto | 0.40 M encoder (0.81 M with decoder) | ~1.6 MB **estimate** (0.40 M x 4 B) | 2 transformer blocks, width 128, 8 heads, MLP ratio 4; channel-group tokens, month encoding | per-pixel series, 1 to 24 months; S1, S2, ERA5, NDVI, DEM, Dynamic World, location | Code MIT; weights licence not stated separately | PyTorch `.pt`; no ONNX found | CropHarvest F1 0.835; EuroSAT MS 0.953 fine-tuned |
| Galileo-Nano | 0.8 M | ~3.2 MB **estimate** | ViT, 4 blocks, dim 128, 8 heads; variable patch 1 to 8 | space-time, space, time and static tokens; S1, S2, DEM, ERA5 and others | MIT | PyTorch on HF; no ONNX found | m-EuroSat kNN 89.7 %; Sen1Floods11 linear probe 78.6 mIoU |
| Galileo-Tiny | 5.3 M | ~21 MB **estimate** | ViT, 12 blocks, dim 192, 3 heads | as above | MIT | as above | m-EuroSat kNN 90.1 %; CropHarvest Brazil 97.2 |
| OlmoEarth v1 Nano / Tiny | 1.4 M / 6.2 M encoder | not stated | ViT-Nano / ViT-Tiny, 128 / 192 dims | S1, S2, Landsat image time series | OlmoEarth Artifact License (use restrictions) | not stated | not checked |
| TerraMind 1.0 tiny | 5 M | encoder memory 23 MB | ViT, 224² input, example shows 196 tokens | S2 L1C/L2A, S1, DEM, RGB | Apache-2.0 | PyTorch via TerraTorch | 15 % drop vs large (PANGAEA, frozen) |
| Prithvi-EO-2.0 tiny TL | 5 M | encoder memory 22 MB | ViT with 3D patch embedding, temporal and location embeddings | 6 HLS bands, T frames | Apache-2.0 | `Prithvi_EO_V2_tiny.pt` | 4 % drop vs large (GEO-Bench, fine-tuned) |
| TESSERA v1 | ~46 M encoder | too large | two 8-block transformers, attention pooling | 40 S1 + S2 time steps | CC0 | PyTorch | not checked |
| Clay v1.0 / v1.5 | 201 M (v1.0) | 768 MB (v1.0); v1.5 checkpoint 4.8 GB | ViT MAE | S2, NAIP, others | OpenRAIL-M | compiled v1 encoder files on HF | not checked |
| DOFA | ViT-B and ViT-L only found | too large | wavelength-conditioned hypernetwork + ViT | any band set | CC BY 4.0 (HF card) | PyTorch / TorchGeo | not checked |
| SatCLIP location encoder | ~1 M | ~4.8 MB **estimate** (L = 40) | spherical harmonics (L² basis functions) then SIREN: 2 hidden layers of 512, output 256 | latitude, longitude | MIT | `.ckpt` per model (includes image encoder); no ONNX found | see paper |
| GeoCLIP location encoder | not found | not found | random Fourier features at several scales, then MLP | latitude, longitude | MIT | PyTorch | not checked |

Sources: Presto paper <https://arxiv.org/html/2304.14065v4> and repo <https://github.com/nasaharvest/presto>;
Galileo paper (Tables 3 to 6, 12, 14) <https://arxiv.org/html/2502.09356>, repo
<https://github.com/nasaharvest/galileo>, weights <https://huggingface.co/nasaharvest/galileo>; OlmoEarth
<https://huggingface.co/allenai/OlmoEarth-v1-Nano>, <https://pypi.org/project/olmoearth-pretrain/>;
TerraMind tiny <https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny>; Prithvi tiny
<https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL>; IBM size table (10 Oct 2025)
<https://research.ibm.com/blog/terramind-prithvi-tiny-small-models-geospatial>; TESSERA parameter split
<https://arxiv.org/pdf/2506.20380>; Clay <https://clay-foundation.github.io/model/release-notes/specification.html>,
<https://huggingface.co/made-with-clay/Clay>; DOFA <https://arxiv.org/abs/2403.15356>,
<https://huggingface.co/earthflow/DOFA/blob/main/README.md>; SatCLIP <https://arxiv.org/html/2311.17179>,
<https://github.com/microsoft/satclip>, <https://huggingface.co/microsoft/SatCLIP-ViT16-L40>; GeoCLIP
<https://github.com/VicenteVivan/geo-clip>, <https://pypi.org/project/geoclip/>.

Observations:

- **SatCLIP** is the odd one out and the best fit: no convolution, no attention. The model card restricts
  intended use to research, rules out defence and surveillance uses, says it is "not applicable for
  fine-grained geospatial problems" and not intended for production
  (<https://huggingface.co/microsoft/SatCLIP-ViT16-L40>). The licence itself is MIT. The weights would need a
  one-off offline extraction of the location-encoder tensors from the `.ckpt`.
- **Galileo-Nano** is the smallest image encoder with an OSI licence and published per-input MAC counts,
  and it also handles pixel time series (its average rank on time series is 3.5 for Nano, 2.3 for Tiny,
  better than on images). The TerraMind 1.0 tiny card's example output shape is (B, 196, 768); whether
  that is the tiny model's width is not verified.
- **OlmoEarth** Nano is attractive technically but its licence restrictions make it a poor dependency for a
  published package.

## 3. Small task models

| Model | What | Size | Layers | Licence | Notes |
|---|---|---|---|---|---|
| s2cloudless | pixel cloud probability, Sentinel-2 | not found | LightGBM trees on 10 bands | CC BY-SA 4.0 (repo) | Share-alike. A tree-ensemble contributor could run it if a user fetches the model. <https://github.com/sentinel-hub/sentinel2-cloud-detector> |
| OmniCloudMask V4 | cloud, thin cloud, shadow, clear; R, G, NIR at 10 to 50 m | not found | ensemble of two U-Nets, encoders `regnety_004` and `edgenext_small` | code MIT; weights not stated; trained on CloudSEN12 (CC BY-NC data) | Sentinel-2 accuracy 92.2 % clear, 91.2 % cloud, 80.5 % shadow. <https://omnicloudmask.readthedocs.io/en/latest/model-changelog.html>, <https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html>, <https://pypi.org/project/omnicloudmask/> |
| CloudSEN12 UNetMobV2 | cloud segmentation, 13 S2 bands | not found | U-Net with MobileNetV2 encoder | code LGPL-3.0; weights not stated | <https://github.com/IPL-UV/cloudsen12_models> |
| Dynamic World | 9-class land cover, S2 10 m | not found | fully convolutional, receptive field about 21 px | repo Apache-2.0 (archived); TF SavedModel | <https://github.com/google/dynamicworld>, <https://docs.google.com/presentation/d/141OgchhM12UE3mWBZH_sCdPpfIzeHOAYCg8g5zuQ2as/htmlpresent> |
| Lang et al. canopy height | canopy top height from S2 | not found | ensemble of 5 Xception-style CNNs | code MIT; weights not stated | <https://github.com/langnico/global-canopy-height-model> |
| SEN2SR Lite | S2 super-resolution 10 m to 2.5 m (RGBN) and 20 m to 10 m (SWIR) | not found | CNN (full version Mamba / Swin2SR) | CC0 1.0 | <https://huggingface.co/WEO-SAS/sen2sr/blob/main/README.md> |
| Neural relief shading (Jenny et al.) | hand-drawn style shading from a DEM | no public weights found | U-Net | n/a | <https://arxiv.org/abs/2010.01256> |
| MobileSAM | promptable segmentation | 9.66 M params; 40.7 MB file | Tiny-ViT (MBConv, window attention) + SAM decoder (two-way transformer, transposed conv) | not verified (mirrors say Apache-2.0 or MIT) | encoder 8 ms, decoder 4 ms on a GPU. <https://docs.ultralytics.com/models/mobile-sam> |
| EfficientSAM-Ti | promptable segmentation | 10 M, 40 MB | ViT-Tiny + SAM decoder | not checked | <https://pypi.org/pypi/osam> |
| SAM 2 tiny | promptable segmentation | 38.9 M | Hiera + memory attention | not checked | WebGPU `.ort` export exists. <https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/blob/main/README.md> |
| geobase building footprint | building segmentation | not found | U-Net-style CNN, quantized ONNX | not stated | <https://huggingface.co/geobase/building-footprint-segmentation/blob/main/README.md?code=true> |

Meta's 2024 canopy height maps and models were announced as free for commercial use
(<https://sustainability.atmeta.com/?p=5573>), but the successor CHMv2 sits on DINOv3 under a custom,
gated licence (<https://github.com/facebookresearch/dinov3/issues/31>). I found no Sentinel-2 building,
road or water model that is both small and clearly licensed.

## 4. Cost estimates for the most promising models

Assumptions, all **estimates**: Apple rates the M1 8-core GPU at 2.6 TFLOPS and the M2 10-core at
3.6 TFLOPS fp32 (<https://www.anandtech.com/show/17431>). Hand-written WGSL matrix kernels rarely reach
peak; I assume **0.5 to 1 TFLOPS effective** on an M1-class device. That range is a guess, not a
measurement. One multiply-accumulate (MAC) = 2 FLOP.

**SatCLIP location encoder, one million locations.** L² basis functions in, two hidden layers of 512,
256 out (paper). MACs per location:

- L = 10: 100 x 512 + 512 x 512 + 512 x 256 = 51,200 + 262,144 + 131,072 = 444,416.
- L = 40: 1,600 x 512 + 262,144 + 131,072 = 1,212,416. This weight count (about 1.21 M) is consistent
  with the paper's "roughly 1 million parameters".
- 10⁶ locations at L = 40: 1.21 x 10¹² MAC = 2.4 TFLOP, so 2.4 to 4.8 s. At L = 10: 0.9 to 1.8 s.
- Location embeddings are smooth at 10 m imagery scale (model card), so a 256² grid (65,536 locations)
  plus bilinear interpolation costs 0.16 to 0.32 s at L = 40. The spherical-harmonic basis adds about
  L² recurrence steps per point, small next to the dense layers. Weights: 1.21 M x 4 B ≈ 4.8 MB fp32,
  2.4 MB fp16.

**Galileo-Nano and Tiny, images.** The paper gives 0.25 GMAC (Nano) and 1.71 GMAC (Tiny) per m-EuroSat
input at patch 8. I take that input as 64 x 64 pixels (the GEO-Bench m-EuroSat size; not re-verified here).

- Nano, 1024², one time step: 1,048,576 / 4,096 = 256 inputs x 0.25 GMAC = 64 GMAC = 128 GFLOP, so 0.13
  to 0.26 s. At patch 4, tokens are four times as many, so at least four times the MACs: about 0.5 to 1 s.
- Nano, 4096²: sixteen times 1024², so 2 to 4 s at patch 8. Needs tiling.
- Tiny, 1024², patch 8: 256 x 1.71 = 438 GMAC = 876 GFLOP, so 0.9 to 1.8 s.
- Each extra time step multiplies the token count, so T months cost at least T times as much.
- Weights: 0.8 M x 4 B ≈ 3.2 MB (Nano), 5.3 M x 4 B ≈ 21 MB (Tiny), half in fp16.

**Presto, pixel time series.** The paper reports 88.94 M operations per pixel for 12 monthly steps
(Table 8) and 2.37 M for one multispectral step (Table 9), counted with `thop`, which counts MACs. As a
check, 0.40 M parameters x about 200 tokens ≈ 80 M MAC, so 88.94 M is plausibly MACs.

- 10⁶ pixels x 88.94 M = 8.9 x 10¹³ MAC = 178 TFLOP: **3 to 6 minutes**. Not interactive.
- Interactive budget: about 0.5 to 1 TFLOP per second buys 2,800 to 5,600 pixels per second. Presto suits
  sample points (training labels, clicked parcels), not whole rasters.
- Activations: about 200 tokens x 128 dims x 4 B ≈ 100 KB per pixel, so a 128 MB binding holds about
  1,300 pixels at once (**estimate**).

**MobileSAM encoder, one 1024² image.** TinyViT-5M is 1.3 GMAC at 224² (<https://github.com/wkcn/tinyvit>).
Scaling by token count, (1024 / 224)² = 20.9, gives about 27 GMAC = 54 GFLOP, so 0.05 to 0.11 s. The
compute is fine; the 40.7 MB of weights and the layer inventory (MBConv, depthwise conv, window attention
with relative bias, two-way cross attention, transposed convolution, Fourier positional encoding) are not.

**TerraMind / Prithvi tiny, one 1024² image.** About 5 M parameters x 196 tokens per 224² tile ≈ 1 GMAC
per tile, plus attention, so about 21 tiles x 1 GMAC ≈ 21 GMAC for 1024² (**estimate**; depth and width
not verified). Compute is cheap because patches are 16 x 16 pixels; the output is one vector per 160 m
patch, not per pixel, and weights are about 22 MB.

## 5. In-browser geospatial ML, 2024 to 2026

- **geoai.js** (Decision Labs / Geobase, MIT): object, building, car, ship, solar panel and oil-tank
  detection, land cover, wetland and building segmentation, mask generation, zero-shot detection and
  segmentation, and DINOv3 feature extraction, on Transformers.js and ONNX with WebGPU
  (<https://cdn.jsdelivr.net/npm/geoai@1.0.7/README.md>, <https://dev.to/sabman/detect-oil-tanks-in-satellite-imagery-with-geoaijs-webgpu-hl>).
  I did not find model sizes.
- **SAM 2 in the browser** with ONNX Runtime Web (Labelbox, December 2024)
  (<https://labelbox.com/blog/bringing-ai-to-the-browser-sam2-for-interactive-image-segmentation/>) and
  WebSAM (<https://github.com/Xevion/WebSAM>). Not geospatial.
- **AlphaEarth similarity demos** all use a server: Similar Earth reads COGs but runs a Go sidecar against
  Earth Engine (<https://dev.to/riospablo/a-search-engine-for-places-that-look-alike-2fhb>); Earth
  Embeddings Sandbox uses Milvus and FastAPI (<https://earth-embedding-sandbox.vercel.app>). I found no
  fully client-side embedding analysis tool, which is the gap this library already fills.
- Context: Chrome reports WebGPU over 30 times faster than CPU for a text-embedding model on an M1 Max, and
  `shader-f16` three times faster than f32 on the same machine
  (<https://developer.chrome.com/blog/io24-webassembly-webgpu-2>).

## Could not verify

- Weight file sizes for every model except MobileSAM and the IBM encoder memory figures; sizes above are
  parameters x 4 bytes.
- Weight licences for Presto (code MIT), OmniCloudMask, CloudSEN12 models, Dynamic World model files, Lang
  canopy height, MobileSAM (mirrors disagree) and GeoCLIP sizes.
- OlmoEarth int8 decode formula, OlmoEarth v1.3 storage format, Earth Index quantization.
- TerraMind tiny width and depth; Prithvi tiny patch sizes.
- Any ONNX export of Presto, Galileo or SatCLIP. None found.
- Real WGSL throughput on an M-class GPU: the 0.5 to 1 TFLOPS figure is an assumption.

## Ranked candidate contributors

| Rank | Name | Computes | Depends on | Weights | Licence | Layers needed | Effort | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `RasterEmbeddingTreeEnsemble` | per-pixel random forest or gradient-boosted prediction (class or value) from a host-exported model | any embedding or band raster; Google's AlphaEarth regression recipe | user's own model, KB to MB | user's | tree traversal, no neural layers | M | **Do**: the main missing step of the published AlphaEarth workflows |
| 2 | `RasterEmbeddingDequantize` formats + int8 encoder | TESSERA per-pixel scale, Presto uint16, float16, OlmoEarth int8 once verified; inverse AlphaEarth encoder | TESSERA (CC0), Presto, AlphaEarth | none | n/a | elementwise | S | **Do**: unlocks two more products cheaply; verify formulas first |
| 3 | `LocationEncoder` (SatCLIP) | 256-d embedding per coordinate, for regression and spatial-statistics covariates | SatCLIP L10 / L40 | ~1.8 to 4.8 MB fp32 (estimate) | MIT; card says research use, no defence | spherical harmonics, dense layers (`GPUMatMul`), sine | S to M | **Do**: smallest real neural contributor; no conv or attention |
| 4 | `RasterEmbeddingAnomaly` | Mahalanobis distance from the covariance contributor, or distance to the nearest label | any embedding | none | n/a | matrix-vector | S | **Do**: cheap, reuses `RasterEmbeddingCovariance` |
| 5 | `RasterEmbeddingTrajectory` | per-pixel multi-year cosine to a base year, year of largest step, cumulative angle | AlphaEarth 2017 to 2025 | none | CC BY 4.0 data | elementwise, reductions | S | **Do** |
| 6 | Linear probe regression mode; mean-of-references similarity; >128 dimensions | identity-link output; tutorial-compatible similarity; OlmoEarth Tiny, Clay, Earth Index widths | existing classes | none | n/a | none new | S | **Do** as extensions of existing classes |
| 7 | `RasterEmbeddingEdgeStrength` | mean cosine to neighbours as a boundary map for watershed / sieve | any embedding | none | n/a | focal stencil | S | **Maybe**: useful for object delineation; check overlap with `RasterTexture` |
| 8 | `RasterGalileoEncoder` | per-pixel or per-patch embeddings from S1/S2 on the client | Galileo-Nano | ~3.2 MB (estimate) | MIT | linear patch embed, LayerNorm, multi-head attention, GELU MLP, month and position encodings | L | **Maybe**: the first image encoder worth the transformer kit; fits the 1 s budget at 1024² by estimate |
| 9 | `PixelTimeSeriesEncoder` (Presto) | 128-d embedding per pixel time series | Presto | ~1.6 MB (estimate) | code MIT, weights not stated | same transformer kit as Galileo | M once the kit exists | **Maybe**: only for thousands of sample pixels, not rasters |
| 10 | TerraMind / Prithvi tiny encoder | patch embeddings (16 px) | TerraMind tiny or Prithvi tiny | 22 to 23 MB | Apache-2.0 | ViT kit plus 3D patch embedding (Prithvi) | L | **Maybe later**: over the 10 MB target; reuse the Galileo kit |
| 11 | Dynamic World land cover | 9-class probabilities | Dynamic World FCNN | not found | Apache-2.0 repo | convolutions, conversion from TF | L | **Skip** for now: size and layers unknown |
| 12 | OmniCloudMask | cloud and shadow classes | two U-Nets (RegNetY, EdgeNeXt) | not found | weights unclear | grouped and depthwise conv, SE, LayerNorm, transposed attention, upsampling | L | **Skip**: large layer set, weights licence unclear |
| 13 | MobileSAM / EfficientSAM | promptable masks | Tiny-ViT + SAM decoder | ~40 MB | not verified | the largest layer set of all | L | **Skip**: too many layer types; ONNX Runtime Web already does this outside the library |
| 14 | OlmoEarth Nano / Tiny | embeddings | OlmoEarth | not stated | use-restricted | ViT kit | L | **Skip**: licence |
| 15 | Neural relief shading, canopy height, SEN2SR | presentation and derived rasters | various | not found | unknown or CC0 (SEN2SR) | U-Net / CNN kits | L | **Skip**: no public weights (relief), unknown licence (canopy), size unknown (SEN2SR) |
