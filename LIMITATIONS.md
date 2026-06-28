# REBOUND — Limitaciones conocidas

Registro empírico. Solo se documentan valores medidos sobre el sistema
real, no estimaciones ni intenciones de diseño.

## Clasificador CNN (5 clases)

- **val_accuracy**: 0.950 sobre split de validación (20% del dataset).
  El valor 0.9836 que resulta de evaluar sobre el dataset completo
  incluye datos de entrenamiento y no es representativo.

- **Sesgo corner/nearby_wall**: evaluando sobre el dataset completo,
  el modelo comete 26 errores corner→nearby_wall contra 5 en sentido
  inverso (relación ~5:1). La asimetría se debe a similitud acústica
  entre ambas clases: reflexiones fuertes y RT60 bajo en los dos casos.

## Detector de escaleras (two-pass periódico)

- **Umbral SNR**: colapso entre noise_level=0.018 (SNR 26.9 dB → OK)
  y noise_level=0.019 (SNR 26.4 dB → FAIL). Umbral empírico: ~27 dB.
  Nota: noise_level es desviación estándar absoluta del ruido gaussiano
  en simulate_capture(), no un parámetro de SNR explícito. El SNR real
  depende de la potencia de la señal convolucionada.

- **RIRs reales deconvolucionados**: comportamiento no evaluado. El
  sistema fue entrenado y testeado exclusivamente sobre RIRs sintéticos
  (pyroomacoustics). Riesgo crítico identificado en red team audit.

## Trabajo futuro diferido

- **EWC (Elastic Weight Consolidation)**: implementado en `future/ewc.py`
  pero no integrado. Requiere DataLoader del modelo base y loop de
  fine-tuning por usuario. Diferido a v2.


## Extracción de features

- **Filtros mel vacíos**: con `n_fft=512` a 44100 Hz la resolución es
  86.1 Hz/bin. Los filtros mel más bajos (cerca de fmin=500 Hz) tienen
  ancho menor a un bin y producen respuesta vacía. Afecta las bandas
  inferiores del mel spectrogram. Las bandas relevantes para el chirp
  CF-FM (8 kHz) no están afectadas. La corrupción es consistente en
  entrenamiento e inferencia — el CNN fue entrenado con este comportamiento.
  Fix: aumentar n_fft a 2048, regenerar dataset y reentrenar. Diferido a v2.

- **Noise augmentation**: `add_noise_augmentation` movida a
  `future/augmentation.py`. Bug: ruido aplicado solo al mel spectrogram,
  dejando rt60 y spectral_centroid limpios — mismatch con inferencia real.
  El modelo de Bloque 1 no fue afectado: train.py usa dataset.npz, no
  dataset_augmented.npz. Fix en v2: augmentar en dominio RIR antes de
  extract_features(). Diferido a v2.

## Split de entrenamiento

- **config_id split sin beneficio actual**: train.py divide train/val
  por config_id para prevenir data leakage. Con un RIR por config
  (dataset actual), es equivalente a split random. El beneficio real
  aparece en v2 cuando augmentation genere múltiples muestras por config.
