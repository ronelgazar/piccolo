from src.input_handler import InputHandler, Action
from src.config import ControlsCfg


def _make_handler():
    return InputHandler(aligner=None, cfg=ControlsCfg())


class TestPedalToggleLogic:
    def test_no_mode_at_startup(self):
        h = _make_handler()
        assert h.pedal_mode is None

    def test_press_a_enters_zoom_mode(self):
        h = _make_handler()
        h.on_key_down("a")
        assert h.pedal_mode == "a"

    def test_press_a_twice_toggles_off(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("a")
        assert h.pedal_mode is None

    def test_press_b_while_zoom_mode_is_adjust(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        actions = h.poll_actions()
        assert Action.ZOOM_IN in actions

    def test_hold_b_in_zoom_mode_keeps_firing(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        h.poll_actions()  # drain one-shot
        # b still held → continuous
        assert Action.ZOOM_IN in h.poll_actions()

    def test_release_b_stops_continuous(self):
        h = _make_handler()
        h.on_key_down("a")
        h.on_key_up("a")
        h.on_key_down("b")
        h.on_key_up("b")
        h.poll_actions()  # drain one-shot
        assert Action.ZOOM_IN not in h.poll_actions()

    def test_side_mode_maps_ac_to_left_right(self):
        h = _make_handler()
        h.on_key_down("b")
        h.on_key_up("b")
        h.on_key_down("a")
        assert Action.PEDAL_CENTER_LEFT in h.poll_actions()
        h.on_key_up("a")
        h.on_key_down("c")
        assert Action.PEDAL_CENTER_RIGHT in h.poll_actions()
