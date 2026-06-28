# DEFERRED — not integrated in current system (Bloque 2, TODO-4)
#
# BUG CONOCIDO: add_noise_augmentation agrega ruido solo al mel
# spectrogram. Los escalares rt60 y spectral_centroid quedan limpios
# en las muestras augmentadas. Eso no existe en inferencia real donde
# el ruido afecta toda la cadena de procesamiento.
#
# CORRECCIÓN CORRECTA para v2: augmentar en dominio RIR antes de
# extraer features. Es decir: agregar ruido al RIR, luego llamar
# extract_features() sobre el RIR ruidoso. Así mel, rt60 y centroid
# quedan todos consistentemente afectados.
#
# NOTA: train.py carga dataset.npz por defecto, no dataset_augmented.npz.
# El modelo entrenado en Bloque 1 nunca usó esta función — el bug
# estuvo dormido durante todo el entrenamiento.
"""
Noise augmentation for REBOUND dataset. DEFERRED to v2.
See header above for known bug and correct fix.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_noise_augmentation(
    dataset: dict[str, NDArray],
    snr_range_db: tuple[float, float] = (5.0, 30.0),
    n_augmented: int = 1,
    seed: int = 123,
) -> dict[str, NDArray]:
    """Add noise augmentation to existing dataset.

    For each original sample, generates n_augmented copies with Gaussian
    noise at different SNRs.

    WARNING: ruido aplicado solo al mel spectrogram. Ver header del módulo.
    """
    rng = np.random.default_rng(seed)

    n_orig = len(dataset["mel_spectrograms"])

    aug_mels = []
    aug_mfccs = []
    aug_rt60 = []
    aug_centroid = []
    aug_labels = []
    aug_distances = []

    for i in range(n_orig):
        for _ in range(n_augmented):
            snr_db = rng.uniform(*snr_range_db)
            snr_linear = 10 ** (snr_db / 10)

            mel = dataset["mel_spectrograms"][i].copy()
            signal_power = np.mean(mel ** 2)
            noise_power = signal_power / snr_linear
            noise = rng.standard_normal(mel.shape).astype(np.float32) * np.sqrt(noise_power)
            mel_noisy = mel + noise

            aug_mels.append(mel_noisy)
            aug_mfccs.append(dataset["mfccs"][i])
            aug_rt60.append(dataset["rt60"][i])
            aug_centroid.append(dataset["spectral_centroid"][i])
            aug_labels.append(dataset["labels"][i])
            aug_distances.append(dataset["distances"][i])

    return {
        "mel_spectrograms": np.concatenate([
            dataset["mel_spectrograms"],
            np.array(aug_mels),
        ]),
        "mfccs": np.concatenate([
            dataset["mfccs"],
            np.array(aug_mfccs),
        ]),
        "rt60": np.concatenate([
            dataset["rt60"],
            np.array(aug_rt60),
        ]),
        "spectral_centroid": np.concatenate([
            dataset["spectral_centroid"],
            np.array(aug_centroid),
        ]),
        "labels": np.concatenate([
            dataset["labels"],
            np.array(aug_labels),
        ]),
        "distances": np.concatenate([
            dataset["distances"],
            np.array(aug_distances),
        ]),
    }
