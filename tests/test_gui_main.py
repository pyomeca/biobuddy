from biobuddy.gui import __main__ as gui_main


def test_gui_main_launches_model_editor_by_default(monkeypatch):
    launched = []
    monkeypatch.setattr(gui_main, "launch_model_editor", lambda: launched.append("model"))
    monkeypatch.setattr(gui_main, "launch_yeadon_measurement_editor", lambda: launched.append("yeadon"))

    gui_main.main([])

    assert launched == ["model"]


def test_gui_main_launches_yeadon_measurement_editor(monkeypatch):
    launched = []
    monkeypatch.setattr(gui_main, "launch_model_editor", lambda: launched.append("model"))
    monkeypatch.setattr(gui_main, "launch_yeadon_measurement_editor", lambda: launched.append("yeadon"))

    gui_main.main(["--yeadon-measurements"])

    assert launched == ["yeadon"]
