"""Neuron audit of Timestral: labeled corpus, activation capture, pole labeling, scoring, steering.

Modules: `corpus` (series with known generating variables), `capture` (forward hooks on the SwiGLU MLPs),
`labels` (which known variable explains each pole), `score` (one-job curves, held-out check),
`steer` (scaling a pole during the forward pass), `heads` (attention heads), `realdata` (transfer corpus),
`neuronpedia` (export).
"""
