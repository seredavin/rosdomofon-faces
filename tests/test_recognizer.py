"""Тесты сопоставления вектора лица с базой образцов."""

import numpy as np

from app.recognizer import match_gallery
from app.store import Gallery


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _gallery(rows: list[tuple[int, np.ndarray]]) -> Gallery:
    return Gallery(
        version=1,
        person_ids=np.array([p for p, _ in rows], dtype=np.int64),
        matrix=np.stack([v for _, v in rows]) if rows else np.empty((0, 128), np.float32),
        names={p: f"person{p}" for p, _ in rows},
    )


rng = np.random.default_rng(42)
A = _unit(rng.normal(size=128))
B = _unit(rng.normal(size=128))


def test_empty_gallery_is_unknown():
    assert match_gallery(_gallery([]), A, 0.363) == (None, 0.0)


def test_close_vector_matches_person():
    probe = _unit(A + 0.3 * _unit(rng.normal(size=128)))
    person, similarity = match_gallery(_gallery([(1, A), (2, B)]), probe, 0.363)
    assert person == 1
    assert similarity > 0.363


def test_below_threshold_is_unknown():
    person, similarity = match_gallery(_gallery([(1, A)]), B, 0.363)
    assert person is None
    assert similarity < 0.363


def test_best_sample_wins_across_people():
    # У человека 2 есть образец, очень близкий к пробе, — он и выигрывает
    probe = _unit(B + 0.05 * A)
    gallery = _gallery([(1, A), (1, _unit(A + B)), (2, B)])
    assert match_gallery(gallery, probe, 0.363)[0] == 2
