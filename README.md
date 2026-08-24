# Eye Blink Analytics

Pipeline de **Visão Computacional + Engenharia de Dados** que mede a frequência de
piscadas em vídeos (OpenCV + MediaPipe Face Mesh), calcula o **EAR (Eye Aspect Ratio)**
quadro a quadro, detecta eventos de piscada e exporta telemetria estruturada em CSV
para consumo no **Power BI**.

---

## 1. Arquitetura

```
eye-blink-analytics/
├── data/
│   ├── raw/                  # Vídeos brutos (.mp4, .mov, .avi)
│   └── processed/            # Telemetria exportada (.csv)
├── dashboards/
│   └── blink_analysis.pbix   # Relatório Power BI
├── src/
│   ├── __init__.py
│   ├── vision_tracker.py     # Ingestão de vídeo + MediaPipe Face Mesh + CLI
│   └── metrics_exporter.py   # EAR, detecção de eventos e exportação de CSV
├── .gitignore
├── README.md
└── requirements.txt
```

Fases do pipeline:

| Fase | Responsável | Descrição |
| --- | --- | --- |
| Ingestão de vídeo | `vision_tracker.VisionTracker.process_video` | Leitura frame a frame com `cv2.VideoCapture` |
| Mapeamento de marcos faciais | `vision_tracker.VisionTracker.extract_eye_points` | MediaPipe Face Mesh isolando os 6 pontos de cada pálpebra |
| Cálculo de EAR | `metrics_exporter.compute_ear` | Distância euclidiana via `scipy.spatial.distance` |
| Detecção de eventos | `metrics_exporter.BlinkDetector` | Máquina de estados com limiar `τ` e `N` quadros consecutivos |
| Exportação | `metrics_exporter.MetricsExporter` | CSV em `data/processed/` com o schema do Power BI |

### Cálculo do EAR

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Os índices do Face Mesh (468 marcos) usados na ordem `p1..p6`:

| Olho | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | --- | --- | --- | --- | --- | --- |
| Esquerdo | 362 | 385 | 387 | 263 | 373 | 380 |
| Direito | 33 | 160 | 158 | 133 | 153 | 144 |

O valor exportado é a **média dos dois olhos**. Quadros sem face detectada recebem
`ear_value = 0.0` e **não alimentam** a máquina de estados de piscadas (evita falsos
positivos por oclusão), preservando a continuidade da série temporal.

### Detecção de piscadas

Uma piscada é contabilizada quando o EAR permanece **abaixo do limiar `τ`**
(padrão `0.21`) por **`N` quadros consecutivos** (padrão `2`). O evento é registrado no
quadro de reabertura do olho (`is_blink = 1`), evitando contagem duplicada em piscadas
longas. Um evento pendente ao final do vídeo é fechado por `BlinkDetector.flush()`.

### Filtro de luz azul (`has_blue_filter`)

Heurística por dominância de canal: o quadro é marcado com `1` quando
`média(B) / ((média(R) + média(G)) / 2) >= blue_filter_threshold` (padrão `1.10`).
Quando o estado do filtro é conhecido a priori, use `--blue-filter` / `--no-blue-filter`
para fixar o valor em toda a sessão.

---

## 2. Instalação

Requer Python 3.9+.

```bash
git clone <url-do-repositorio>
cd eye-blink-analytics

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Dependências: `opencv-python`, `mediapipe`, `pandas`, `numpy`, `scipy`.

---

## 3. Execução

Coloque os vídeos em `data/raw/` e execute:

```bash
python -m src.vision_tracker data/raw/sessao01.mp4 --session-id sessao01
```

O caminho do CSV gerado é impresso no stdout (`data/processed/sessao01.csv`).

### Parâmetros da CLI

| Parâmetro | Padrão | Descrição |
| --- | --- | --- |
| `video` | — | Caminho do vídeo de entrada |
| `--session-id` | nome do arquivo | Identificador da sessão gravado no CSV |
| `--output-dir` | `data/processed` | Diretório de saída |
| `--tau` | `0.21` | Limiar `τ` de EAR para olho fechado |
| `--consec-frames` | `2` | `N` quadros consecutivos para registrar piscada |
| `--blue-filter-threshold` | `1.10` | Razão B/((R+G)/2) da heurística de luz azul |
| `--blue-filter` / `--no-blue-filter` | heurística | Fixa `has_blue_filter` em 1 ou 0 |
| `--max-frames` | — | Limita a quantidade de quadros (testes) |
| `--preview` | desligado | Janela de depuração com marcos e contadores |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Uso programático

```python
from pathlib import Path
from src.vision_tracker import TrackerConfig, run_pipeline

csv_path = run_pipeline(
    video_path=Path("data/raw/sessao01.mp4"),
    session_id="sessao01",
    config=TrackerConfig(tau=0.19, consec_frames=3, force_blue_filter=True),
)
print(csv_path)
```

---

## 4. Schema de exportação

Arquivo: `data/processed/<session_id>.csv`

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `session_id` | string | Identificador da sessão de captura |
| `timestamp` | string `hh:mm:ss.ms` | Posição do quadro no vídeo |
| `frame_index` | integer | Índice sequencial do quadro (base 0) |
| `ear_value` | float | EAR médio dos dois olhos no quadro |
| `is_blink` | integer (0/1) | `1` apenas no quadro que fecha o evento de piscada |
| `accumulated_blinks` | integer | Total acumulado de piscadas na sessão |
| `has_blue_filter` | integer (0/1) | Indica presença de filtro de luz azul |

Exemplo:

```csv
session_id,timestamp,frame_index,ear_value,is_blink,accumulated_blinks,has_blue_filter
sessao01,00:00:00.000,0,0.284531,0,0,0
sessao01,00:00:00.033,1,0.191204,0,0,0
sessao01,00:00:00.066,2,0.302118,1,1,0
```

---

## 5. Power BI

### 5.1 Importação

1. **Obter Dados → Texto/CSV** (ou **Pasta** apontando para `data/processed/` para
   consolidar várias sessões).
2. No Power Query, garanta os tipos: `frame_index`, `is_blink`, `accumulated_blinks`,
   `has_blue_filter` como *Número Inteiro*; `ear_value` como *Número Decimal*;
   `session_id` e `timestamp` como *Texto*.
3. Crie a coluna de tempo em segundos para agregações temporais:

```dax
Segundos =
VAR t = 'blink_telemetry'[timestamp]
VAR h = VALUE ( LEFT ( t, 2 ) )
VAR m = VALUE ( MID ( t, 4, 2 ) )
VAR s = VALUE ( MID ( t, 7, 6 ) )
RETURN h * 3600 + m * 60 + s
```

### 5.2 Medidas DAX recomendadas

```dax
-- Total de piscadas da seleção
Total de Piscadas = SUM ( 'blink_telemetry'[is_blink] )

-- Duração da sessão em minutos
Duracao (min) =
DIVIDE ( MAX ( 'blink_telemetry'[Segundos] ) - MIN ( 'blink_telemetry'[Segundos] ), 60 )

-- Média de Piscadas por Minuto (referência clínica: 15 a 20 bpm)
Media Piscadas por Minuto =
DIVIDE ( [Total de Piscadas], [Duracao (min)], 0 )

-- EAR médio (abertura ocular média)
EAR Medio = AVERAGE ( 'blink_telemetry'[ear_value] )

-- Linha de base do EAR com olho aberto (ignora quadros de oclusão/sem face)
EAR Base Aberto =
CALCULATE ( [EAR Medio], FILTER ( 'blink_telemetry', 'blink_telemetry'[ear_value] > 0.21 ) )

-- Taxa de Queda de Foco: % de quadros com olho fechado/parcialmente fechado
Taxa de Queda de Foco =
VAR QuadrosFechados =
    CALCULATE (
        COUNTROWS ( 'blink_telemetry' ),
        FILTER ( 'blink_telemetry', 'blink_telemetry'[ear_value] < 0.21 && 'blink_telemetry'[ear_value] > 0 )
    )
VAR QuadrosValidos =
    CALCULATE ( COUNTROWS ( 'blink_telemetry' ), 'blink_telemetry'[ear_value] > 0 )
RETURN DIVIDE ( QuadrosFechados, QuadrosValidos, 0 )

-- Comparativo com e sem filtro de luz azul
Piscadas por Minuto (com filtro azul) =
CALCULATE ( [Media Piscadas por Minuto], 'blink_telemetry'[has_blue_filter] = 1 )

Piscadas por Minuto (sem filtro azul) =
CALCULATE ( [Media Piscadas por Minuto], 'blink_telemetry'[has_blue_filter] = 0 )

Delta Filtro Azul (%) =
DIVIDE (
    [Piscadas por Minuto (com filtro azul)] - [Piscadas por Minuto (sem filtro azul)],
    [Piscadas por Minuto (sem filtro azul)],
    0
)

-- Sinalização de fadiga (piscadas abaixo da faixa saudável)
Alerta de Fadiga = IF ( [Media Piscadas por Minuto] < 15, "Atenção", "Normal" )
```

### 5.3 Visuais sugeridos

- **Cartões:** `Total de Piscadas`, `Media Piscadas por Minuto`, `Taxa de Queda de Foco`,
  `Alerta de Fadiga`.
- **Gráfico de linhas:** `ear_value` por `Segundos`, com linha constante em `τ = 0.21`.
- **Gráfico de área:** `accumulated_blinks` por `Segundos` (curva de acúmulo).
- **Colunas agrupadas:** piscadas/minuto por `has_blue_filter` (comparativo A/B).
- **Segmentações:** `session_id` e `has_blue_filter`.

O arquivo `dashboards/blink_analysis.pbix` deve ser criado/aberto no Power BI Desktop
apontando para `data/processed/`; consulte `dashboards/README.md` para o roteiro de
montagem do relatório.

---

## 6. Boas práticas de captura

- Iluminação frontal estável e face totalmente enquadrada.
- Preferir 30 FPS ou mais: `τ` e `N` assumem quadros de ~33 ms.
- Sessões de no mínimo 60 s para que piscadas/minuto seja estatisticamente útil.
- Calibrar `τ` por indivíduo inspecionando a linha de base de `ear_value` com olho aberto.
