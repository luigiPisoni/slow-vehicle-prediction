# slow-vehicle-prediction

Detecta veículos em sequências do UA-DETRAC, acompanha cada um ao longo dos frames e marca quais estão lentos em relação ao fluxo do próprio vídeo.

Os scripts leem `settings.json` na raiz. Rode tudo a partir da raiz do repositório.

## Ambiente

```bash
python -m venv yolov8-env
source yolov8-env/bin/activate
pip install -r requirements.txt
```

O peso `yolov8n.pt` não vai no git. Na primeira execução o Ultralytics baixa o arquivo se ele não estiver na raiz.

As imagens ficam de fora do repositório. Coloque os frames em `DETRAC-Images/<video_id>/*.jpg`.

## Configuração

`settings.json` escolhe o clipe e o peso do YOLO:

```json
{
    "dataset_dir": "./DETRAC-Images/",
    "video_id": "MVI_20011",
    "yolo_model": "yolov8n.pt"
}
```

`config/<video_id>.json` guarda fps, tamanho do frame e as zonas de exclusão. O tracking cria esse arquivo se ele ainda não existir.

## Rodar

1. Desenhe as zonas de exclusão (retângulos). Dois cliques fecham um retângulo, `u` desfaz, `q` salva em `config/<video_id>.json` e grava `output/<video_id>/zones_check.jpg`.

```bash
python scripts/draw_zones.py
```

2. Detecte (carro, moto, ônibus, caminhão) e rastreie com DeepSORT. A saída é `output/<video_id>/tracks.pkl`, `summary.json` e um frame anotado por imagem em `output/<video_id>/frames/`, com id e bounding box.

```bash
python scripts/tracking.py
```

3. Classifique as trajetórias. Um veículo é lento quando a velocidade, normalizada pelo tamanho da caixa, fica abaixo da metade da mediana do vídeo. Tracks com menos de 15 frames ficam como `short`.

```bash
python scripts/slow.py
```

O resultado vai para `output/<video_id>/slow.json` e para o terminal.

`output/`, `*.pkl`, `*.pt`, `DETRAC-Images/` e o ambiente virtual não entram no git.
