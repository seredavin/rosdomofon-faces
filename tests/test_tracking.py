"""Тесты фильтра качества, трекера и голосования."""

import cv2
import numpy as np

from app.detector import Face
from app.quality import geometry_reason, is_sharp, pose_metrics, quality_score, sharpness
from app.tracker import KNOWN, PENDING, UNKNOWN, Track, Tracker


def face(x=100, y=100, size=80, score=0.95, nose_dx=0.0, tilt_dy=0.0) -> Face:
    """Лицо с правдоподобными опорными точками; nose_dx/tilt_dy — в долях размера."""
    eye_y = y + size * 0.4
    left = (x + size * 0.3, eye_y)
    right = (x + size * 0.7, eye_y + size * tilt_dy)
    nose = (x + size * (0.5 + nose_dx), y + size * 0.6)
    mouth = ((x + size * 0.35, y + size * 0.8), (x + size * 0.65, y + size * 0.8))
    points = (left, right, nose, *mouth)
    return Face(x, y, size, size, score, tuple((int(px), int(py)) for px, py in points))


# --- Качество ---

def test_frontal_face_passes_geometry():
    assert geometry_reason(face()) is None


def test_small_face_rejected():
    assert geometry_reason(face(size=30)) == "мелкое"


def test_turned_face_rejected():
    offset, _ = pose_metrics(face(nose_dx=0.25))
    assert offset > 0.35
    assert geometry_reason(face(nose_dx=0.25)) == "повёрнуто"


def test_tilted_face_rejected():
    assert geometry_reason(face(tilt_dy=0.3)) == "наклонено"


def test_sharpness_distinguishes_blur():
    rng = np.random.default_rng(0)
    sharp_img = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp_img, (15, 15), 5)
    assert is_sharp(sharpness(sharp_img))
    assert not is_sharp(sharpness(blurred))


def test_quality_prefers_bigger_frontal_face():
    assert quality_score(face(size=110), 500) > quality_score(face(size=50), 500)
    assert quality_score(face(), 500) > quality_score(face(nose_dx=0.2), 500)


# --- Трекер ---

def test_same_face_keeps_track_and_new_face_opens_one():
    tracker = Tracker()
    [t1], _ = tracker.update([face(x=100)], now=0.0, wall=1000.0)
    [t1b, t2], _ = tracker.update([face(x=105), face(x=400)], now=0.1, wall=1000.1)
    assert t1b is t1
    assert t2.id != t1.id
    assert t1.started_at == 1000.0 and t1.ended_at == 1000.1


def test_track_finishes_after_timeout():
    tracker = Tracker()
    [track], _ = tracker.update([face()], now=0.0, wall=0.0)
    _, finished = tracker.update([], now=1.0, wall=1.0)
    assert finished == []
    _, finished = tracker.update([], now=2.5, wall=2.5)
    assert finished == [track]
    # После завершения то же место — новое появление
    [again], _ = tracker.update([face()], now=3.0, wall=3.0)
    assert again.id != track.id


def test_flush_returns_active_tracks():
    tracker = Tracker()
    tracker.update([face(x=0), face(x=300)], now=0, wall=0)
    assert len(tracker.flush()) == 2
    assert tracker.tracks == []


# --- Голосование ---

def _track() -> Track:
    return Track(id=1, face=face(), started_at=0, last_seen=0, ended_at=0)


def test_pending_until_min_votes():
    t = _track()
    t.add_vote(7, now=0)
    t.add_vote(7, now=1)
    assert t.status == PENDING
    t.add_vote(7, now=2)
    assert (t.status, t.person_id) == (KNOWN, 7)


def test_single_miss_does_not_change_name():
    t = _track()
    for i, vote in enumerate([7, 7, 7, None, 7]):
        t.add_vote(vote, now=i)
    assert (t.status, t.person_id) == (KNOWN, 7)


def test_unknown_majority():
    t = _track()
    for i, vote in enumerate([None, None, 7]):
        t.add_vote(vote, now=i)
    assert (t.status, t.person_id) == (UNKNOWN, None)


def test_opinion_changes_with_stable_new_majority():
    t = _track()
    for i, vote in enumerate([None, None, None, 7, 7, 7, 7]):
        t.add_vote(vote, now=i)
    assert (t.status, t.person_id) == (KNOWN, 7)


def test_tie_keeps_previous_decision():
    t = _track()
    for i, vote in enumerate([7, 7, 7, 3, 3, 3]):
        t.add_vote(vote, now=i)
    assert t.person_id == 7


def test_recognition_rate_limited():
    t = _track()
    t.add_vote(None, now=10.0)
    assert not t.needs_recognition(10.1)
    assert t.needs_recognition(10.4)


def test_offer_best_keeps_highest_score():
    t = _track()
    emb = np.zeros(128, np.float32)
    t.offer_best(0.5, b"a", emb)
    t.offer_best(0.3, b"b", emb)
    t.offer_best(0.8, b"c", emb)
    assert t.best_image == b"c" and t.had_usable_face


def test_final_person_when_votes_not_enough():
    t = _track()
    assert t.final_person() is None
    t.add_vote(7, now=0)
    assert t.final_person() == 7
    t.add_vote(None, now=1)
    assert t.final_person() is None  # 1:1 — неоднозначно


def test_final_person_follows_decision():
    t = _track()
    for i, vote in enumerate([7, 7, 7]):
        t.add_vote(vote, now=i)
    assert t.final_person() == 7


def test_confirmed_person_survives_unknown_votes():
    """Узнали, открыли дверь, человек отвернулся — имя остаётся (баг из проверки 6.3)."""
    t = _track()
    votes = [7, 7, 7] + [None] * 30
    for i, vote in enumerate(votes):
        t.add_vote(vote, now=i)
    assert (t.status, t.person_id) == (KNOWN, 7)
    assert t.final_person() == 7


def test_confirmed_person_can_switch_to_other_person():
    t = _track()
    for i, vote in enumerate([7, 7, 7, 3, 3, 3, 3, 3]):
        t.add_vote(vote, now=i)
    assert t.person_id == 3
