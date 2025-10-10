# model_path = "kuka_lwr.biomod"
model_path = "Rizon10s.bioMod"
viewer = "pyorerun"

if viewer == "bioviz":
    import bioviz

    viz = bioviz.Viz(model_path)
    viz.exec()

elif viewer == "pyorerun":
    from pyorerun import LiveModelAnimation

    animation = LiveModelAnimation(model_path, with_q_charts=False)
    animation.rerun()
