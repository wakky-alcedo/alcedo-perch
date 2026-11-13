"""Belted Kingfisher scan: merged surface, and helpers."""
import pathlib, numpy as np, trimesh
ROOT = pathlib.Path(__file__).resolve().parents[2]


def bk_mesh():
    s = trimesh.load(ROOT / 'ref' / 'models' / 'belted_kingfisher.glb')
    return trimesh.util.concatenate(list(s.dump()))
