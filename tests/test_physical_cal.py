import numpy as np
import cv2
from src.physical_cal import PhysicalCalSession, PatternRenderer


class TestPhysicalCalSession:
    def test_initial_phase_is_focus(self):
        s = PhysicalCalSession()
        assert s.phase == "focus"

    def test_phase_index_starts_at_zero(self):
        s = PhysicalCalSession()
        assert s.phase_index == 0

    def test_total_phases_is_four(self):
        s = PhysicalCalSession()
        assert s.total_phases == 4

    def test_next_advances_through_all_phases(self):
        s = PhysicalCalSession()
        assert s.next_phase() is False and s.phase == "scale"
        assert s.next_phase() is False and s.phase == "horizontal"
        assert s.next_phase() is False and s.phase == "rotation"

    def test_next_on_last_phase_returns_true_and_stays(self):
        s = PhysicalCalSession()
        for _ in range(3):
            s.next_phase()
        assert s.next_phase() is True
        assert s.phase == "rotation"

    def test_prev_goes_back(self):
        s = PhysicalCalSession()
        s.next_phase()
        s.prev_phase()
        assert s.phase == "focus"

    def test_prev_on_first_phase_stays(self):
        s = PhysicalCalSession()
        s.prev_phase()
        assert s.phase == "focus"
        assert s.phase_index == 0


class TestSharpness:
    def test_sharp_image_scores_higher_than_blurred(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for y in range(0, 400, 20):
            for x in range(0, 400, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    img[y:y + 20, x:x + 20] = 255
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        assert PhysicalCalSession.sharpness(img) > PhysicalCalSession.sharpness(blurred)


class TestPatternRenderer:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_render_focus_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_focus(img, sharpness=750.0)
        assert img.sum() > 0

    def test_render_scale_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_scale(img)
        assert img.sum() > 0

    def test_render_horizontal_with_dy_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_horizontal(img, dy=3.5)
        assert img.sum() > 0

    def test_render_horizontal_with_none_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_horizontal(img, dy=None)
        assert img.sum() > 0

    def test_render_rotation_with_angle_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_rotation(img, dtheta_deg=1.2)
        assert img.sum() > 0

    def test_render_rotation_with_none_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_rotation(img, dtheta_deg=None)
        assert img.sum() > 0
