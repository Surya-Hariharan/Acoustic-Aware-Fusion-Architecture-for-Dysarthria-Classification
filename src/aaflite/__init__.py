"""
AAF-Lite: the acoustic-aware three-branch fusion evaluated on frozen
representations.

    learned         frozen wav2vec 2.0, per-layer masked mean + std pools
    segmental       utterance functionals of MFCC+d+dd, F1-F3, HNR
    suprasegmental  utterance functionals of F0, voicing, intensity + timing

Each branch is expressed relative to how the healthy control speakers said
the SAME word (src.aaflite.reference), classified by a linear model, and the
branches are fused late. Every hyperparameter and the fusion weights are
chosen by an inner leave-one-speaker-out loop on the training speakers of each
outer fold (src.aaflite.pipeline), so the held-out speaker is never used for a
decision. Nothing is trained by gradient descent on a GPU, so a full 15-fold
evaluation with every ablation takes minutes and is deterministic.
"""
