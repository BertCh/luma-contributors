#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent CPU reference for the bounded surveyed-frame cylindrical comparison."""

import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "public/data/ahk-rock-glacier"


def read_vectors(name):
    return np.fromfile(DATA / f"{name}.bin", dtype="<f4").reshape(-1, 3).astype(np.float64)


def validate_local_fit(reference, comparison):
    reference = reference[np.floor(np.arange(1024) * len(reference) / 1024).astype(int)]
    comparison = comparison[np.floor(np.arange(1024) * len(comparison) / 1024).astype(int)]
    cases = []
    for method in ["none", "huber", "tukey"]:
        transformed = comparison.copy()
        history = []
        for iteration in range(12):
            squared_distances = ((transformed[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            indices = squared_distances.argmin(axis=1)
            squared = squared_distances[np.arange(len(indices)), indices]
            weights = (squared <= 4).astype(float)
            if method == "huber":
                weights *= np.minimum(1, 0.5 / np.maximum(np.sqrt(squared), 1e-30))
            elif method == "tukey":
                weights *= np.maximum(0, 1 - squared / 0.25) ** 2
            accepted = weights > 0
            source = transformed[accepted]
            target = reference[indices[accepted]]
            weight = weights[accepted, None]
            center_source = (source * weight).sum(axis=0) / weight.sum()
            center_target = (target * weight).sum(axis=0) / weight.sum()
            left, _, right = np.linalg.svd(
                (source - center_source).T @ (weight * (target - center_target))
            )
            rotation = right.T @ left.T
            if np.linalg.det(rotation) < 0:
                right[-1, :] *= -1
                rotation = right.T @ left.T
            translation = center_target - rotation @ center_source
            transformed = transformed @ rotation.T + translation
            history.append({
                "iteration": iteration + 1, "inliers": int(accepted.sum()),
                "weightedRmseMeters": float(np.sqrt((squared[accepted] * weights[accepted]).sum() / weights[accepted].sum())),
                "translationStepMeters": float(np.linalg.norm(translation))
            })
        cases.append({"robustMethod": method, "history": history})
    return {
        "method": "Independent float64 SVD rigid fit on 1024 systematic rows per epoch; maximum distance 2 m; robust scale 0.5 m; 12 iterations; residuals before each update. GPU Horn solver may differ numerically.",
        "cases": cases
    }


def validate_plane_fit(reference, comparison):
    reference = reference[np.floor(np.arange(1024) * len(reference) / 1024).astype(int)]
    comparison = comparison[np.floor(np.arange(1024) * len(comparison) / 1024).astype(int)]
    squared = ((reference[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(squared, np.inf)
    normals = np.zeros_like(reference)
    support_counts = (squared <= 9).sum(axis=1)
    for index in range(len(reference)):
        neighbors = np.argsort(squared[index], kind="stable")[:64]
        neighbors = neighbors[squared[index, neighbors] <= 9]
        if len(neighbors) < 3:
            continue
        offsets = reference[neighbors] - reference[neighbors].mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(offsets.T @ offsets / len(neighbors))
        if eigenvalues[1] <= eigenvalues[2] * 1e-6:
            continue
        normal = eigenvectors[:, 0]
        normals[index] = normal if normal[2] >= 0 else -normal
    cases = []
    for method in ["none", "huber", "tukey"]:
        transformed = comparison.copy()
        history = []
        converged = False
        for iteration in range(12):
            distances = ((transformed[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            indices = distances.argmin(axis=1)
            squared_distances = distances[np.arange(len(indices)), indices]
            weights = (squared_distances <= 4).astype(float)
            if method == "huber":
                weights *= np.minimum(1, 0.5 / np.maximum(np.sqrt(squared_distances), 1e-30))
            elif method == "tukey":
                weights *= np.maximum(0, 1 - squared_distances / 0.25) ** 2
            accepted = (weights > 0) & (np.linalg.norm(normals[indices], axis=1) > 0.5)
            source = transformed[accepted]
            target = reference[indices[accepted]]
            normal = normals[indices[accepted]]
            weight = weights[accepted]
            residual = ((target - source) * normal).sum(axis=1)
            jacobian = np.column_stack([np.cross(source, normal), normal])
            update = np.linalg.lstsq(jacobian * np.sqrt(weight[:, None]), residual * np.sqrt(weight), rcond=None)[0]
            angular, translation = update[:3], update[3:]
            angle = np.linalg.norm(angular)
            axis = angular / max(angle, 1e-30)
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(3) * np.cos(angle) + np.outer(axis, axis) * (1 - np.cos(angle)) + skew * np.sin(angle)
            step = 0 if converged else float(np.linalg.norm(translation))
            history.append({
                "iteration": iteration + 1, "inliers": int(accepted.sum()),
                "weightedPlaneRmseMeters": float(np.sqrt((weight * residual ** 2).sum() / weight.sum())),
                "translationStepMeters": step
            })
            if not converged:
                transformed = transformed @ rotation.T + translation
                converged = step <= 1e-5 and np.linalg.norm(rotation - np.eye(3), axis=0).max() <= 1e-5
        cases.append({"robustMethod": method, "history": history})
    return {
        "method": "Independent NumPy float64 eigenvector target normals from the nearest 64 rows within 3 m, then weighted six-parameter least-squares point-to-plane Gauss-Newton with exact Rodrigues updates; 1024 systematic rows; 2 m match bound; 0.5 m robust scale; 12 iterations. GPU Jacobi normals and normal-equation solver may differ numerically.",
        "validTargetNormals": int((np.linalg.norm(normals, axis=1) > 0.5).sum()),
        "unsupportedTargetNormals": int((support_counts < 3).sum()),
        "truncatedTargetNormals": int((support_counts > 64).sum()), "cases": cases
    }


def main():
    reference = read_vectors("referenceXYZ")
    comparison = read_vectors("comparisonXYZ")
    cores = read_vectors("coresXYZ")
    normals = read_vectors("normal")
    finite = np.isfinite(normals).all(axis=1)
    lengths = np.linalg.norm(normals[finite], axis=1)
    report = {
        "referenceCount": len(reference), "comparisonCount": len(comparison),
        "coreCount": len(cores), "invalidNormalCount": int((~finite).sum()),
        "invalidNormalRows": np.flatnonzero(~finite).tolist(),
        "finiteNormalLengthRange": [float(lengths.min()), float(lengths.max())],
        "referenceMethod": "Independent NumPy float64 cylindrical comparison over bundled float32 native coordinates; normalized supplied normals; halfDepth 3 m; minimum support 3; registration error 0.03 m; linear uncertainty; confidence multiplier 1.96",
        "cases": []
    }
    for radius in [0.8, 1.5, 2.0]:
        differences = []
        detection_limits = []
        unsupported = 0
        for core, normal, valid in zip(cores, normals, finite):
            if not valid:
                continue
            normal = normal / np.linalg.norm(normal)
            projections = []
            for cloud in [reference, comparison]:
                offsets = cloud - core
                distances = offsets @ normal
                radial = offsets - distances[:, None] * normal
                selected = (np.abs(distances) <= 3) & ((radial * radial).sum(axis=1) <= radius * radius)
                projections.append(distances[selected])
            if any(len(values) < 3 for values in projections):
                unsupported += 1
                continue
            difference = projections[1].mean() - projections[0].mean()
            sampling_variance = sum(values.var(ddof=1) / len(values) for values in projections)
            detection = 1.96 * (np.sqrt(sampling_variance) + 0.03)
            differences.append(difference)
            detection_limits.append(detection)
        differences = np.array(differences)
        detection_limits = np.array(detection_limits)
        report["cases"].append({
            "radiusMeters": radius, "validCores": len(differences),
            "insufficientSupportCores": unsupported,
            "invalidNormalCores": int((~finite).sum()),
            "detectableCores": int((np.abs(differences) > detection_limits).sum()),
            "medianDifferenceMeters": float(np.median(differences)),
            "minimumDifferenceMeters": float(differences.min()),
            "maximumDifferenceMeters": float(differences.max()),
            "medianDetectionLimitMeters": float(np.median(detection_limits))
        })
    report["localIcpReference"] = validate_local_fit(reference, comparison)
    report["planeIcpReference"] = validate_plane_fit(reference, comparison)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
