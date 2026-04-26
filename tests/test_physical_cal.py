import numpy as np
import cv2
import pytest
from src.physical_cal import GridEyeMetrics, GridPairMetrics, PhysicalCalSession, PatternRenderer
from src.physical_grid_calibration import generate_chessboard_page


class TestPhysicalCalSession:
    def test_initial_phase_is_brightness(self):
        s = PhysicalCalSession()
        assert s.phase == "brightness"

    def test_phase_index_starts_at_zero(self):
        s = PhysicalCalSession()
        assert s.phase_index == 0

    def test_total_phases_is_five(self):
        s = PhysicalCalSession()
        assert s.total_phases == 5

    def test_next_advances_through_all_phases(self):
        s = PhysicalCalSession()
        assert s.next_phase() is False and s.phase == "focus"
        assert s.next_phase() is False and s.phase == "scale"
        assert s.next_phase() is False and s.phase == "horizontal"
        assert s.next_phase() is False and s.phase == "rotation"

    def test_next_on_last_phase_returns_true_and_stays(self):
        s = PhysicalCalSession()
        for _ in range(4):
            s.next_phase()
        assert s.next_phase() is True
        assert s.phase == "rotation"

    def test_prev_goes_back(self):
        s = PhysicalCalSession()
        s.next_phase()
        s.prev_phase()
        assert s.phase == "brightness"

    def test_prev_on_first_phase_stays(self):
        s = PhysicalCalSession()
        s.prev_phase()
        assert s.phase == "brightness"
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


class TestGridMetrics:
    def test_grid_metrics_detects_brightness_focus_and_square_size(self, tmp_path):
        path = tmp_path / "grid.png"
        generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)
        img = cv2.imread(str(path))

        metrics = PhysicalCalSession.grid_metrics(img)

        assert metrics.detected
        assert metrics.brightness is not None
        assert metrics.saturation_pct is not None
        assert metrics.sharpness is not None
        assert metrics.square_px is not None

    def test_grid_pair_metrics_reports_zoom_ratio(self, tmp_path):
        path = tmp_path / "grid.png"
        generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)
        img = cv2.imread(str(path))

        metrics = PhysicalCalSession.grid_pair_metrics(img, img)

        assert metrics.zoom_ratio == 1.0

    def test_partial_grid_reports_square_size_when_full_board_is_cropped(self, tmp_path):
        path = tmp_path / "grid.png"
        generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        cropped = img[h // 3:h // 3 + 320, w // 3:w // 3 + 320]

        metrics = PhysicalCalSession.grid_metrics(cropped)

        assert not metrics.detected
        assert metrics.partial
        assert metrics.square_px is not None

    def test_partial_grid_pair_can_report_zoom_ratio(self, tmp_path):
        path = tmp_path / "grid.png"
        generate_chessboard_page(path, inner_cols=9, inner_rows=6, square_mm=20, dpi=120)
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        cropped = img[h // 3:h // 3 + 320, w // 3:w // 3 + 320]

        metrics = PhysicalCalSession.grid_pair_metrics(cropped, cropped)

        assert metrics.zoom_ratio is not None
        assert abs(metrics.zoom_ratio - 1.0) < 0.05

    def test_grid_pair_quality_checks(self):
        left = GridEyeMetrics(
            detected=True,
            brightness=120.0,
            saturation_pct=0.5,
            square_px=100.0,
            center=(50.0, 50.0),
            row_angle_deg=0.0,
        )
        right = GridEyeMetrics(
            detected=True,
            brightness=126.0,
            saturation_pct=1.0,
            square_px=101.0,
            center=(60.0, 53.0),
            row_angle_deg=0.2,
        )
        metrics = GridPairMetrics(left, right)

        assert metrics.brightness_ok()
        assert metrics.zoom_ok()
        assert metrics.zoom_status() == "MECHANICAL ZOOM MATCHED"
        assert metrics.zoom_error_pct == pytest.approx(1.0)
        assert metrics.vertical_ok()
        assert metrics.rotation_ok()

    def test_grid_pair_quality_checks_fail_outside_thresholds(self):
        left = GridEyeMetrics(
            detected=True,
            brightness=120.0,
            saturation_pct=0.5,
            square_px=100.0,
            center=(50.0, 50.0),
            row_angle_deg=0.0,
        )
        right = GridEyeMetrics(
            detected=True,
            brightness=140.0,
            saturation_pct=3.0,
            square_px=105.0,
            center=(60.0, 60.0),
            row_angle_deg=1.0,
        )
        metrics = GridPairMetrics(left, right)

        assert not metrics.brightness_ok()
        assert not metrics.zoom_ok()
        assert metrics.zoom_status() == "RIGHT ZOOM LARGER"
        assert metrics.zoom_error_pct == pytest.approx(5.0)
        assert not metrics.vertical_ok()
        assert not metrics.rotation_ok()

    def test_focus_peak_status_tracks_best_observed_focus(self):
        s = PhysicalCalSession()

        ok, best_l, best_r = s.update_focus_peak(100.0, 100.0)
        assert ok
        assert best_l == 100.0
        assert best_r == 100.0

        ok, best_l, best_r = s.update_focus_peak(80.0, 95.0)
        assert not ok
        assert best_l == 100.0
        assert best_r == 100.0


class TestPatternRenderer:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_render_focus_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        r.render_focus(img, sharpness=750.0)
        assert img.sum() > 0

    def test_render_brightness_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        metrics = PhysicalCalSession.grid_metrics(img)
        r.render_brightness(img, metrics)
        assert img.sum() > 0

    def test_render_grid_focus_modifies_image(self):
        r = PatternRenderer()
        img = self._blank()
        metrics = PhysicalCalSession.grid_metrics(img)
        r.render_grid_focus(img, metrics)
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

    def test_render_horizontal_dy_text_differs_from_none(self):
        r = PatternRenderer()
        img_with = self._blank()
        img_none = self._blank()
        r.render_horizontal(img_with, dy=3.5)
        r.render_horizontal(img_none, dy=None)
        assert not np.array_equal(img_with, img_none)

    def test_render_rotation_dot_absent_when_none(self):
        r = PatternRenderer()
        img_with = self._blank()
        img_none = self._blank()
        r.render_rotation(img_with, dtheta_deg=0.0)
        r.render_rotation(img_none, dtheta_deg=None)
        assert not np.array_equal(img_with, img_none)

    def test_render_focus_is_inplace(self):
        r = PatternRenderer()
        img = self._blank()
        result = r.render_focus(img, sharpness=100.0)
        assert result is None
        assert img.sum() > 0
