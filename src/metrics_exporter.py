"""Calculo de EAR (Eye Aspect Ratio), deteccao de eventos e exportacao de CSV.

Este modulo concentra a camada de "engenharia de dados" do pipeline:

* :func:`compute_ear` implementa a formula do Eye Aspect Ratio usando a
  distancia euclidiana da SciPy.
* :class:`BlinkDetector` transforma a serie temporal de EAR em eventos de
  piscada (maquina de estados com limiar ``tau`` e ``consec_frames``).
* :class:`MetricsExporter` materializa a telemetria em ``data/processed``
  seguindo o schema consumido pelo Power BI.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

#: Ordem das colunas do arquivo CSV exportado (contrato com o Power BI).
CSV_SCHEMA: Final[Tuple[str, ...]] = (
    "session_id",
    "timestamp",
    "frame_index",
    "ear_value",
    "is_blink",
    "accumulated_blinks",
    "has_blue_filter",
)

#: Diretorio padrao de saida (``<repo>/data/processed``).
DEFAULT_OUTPUT_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent / "data" / "processed"
)


class MetricsExporterError(RuntimeError):
    """Erro de dominio na camada de metricas/exportacao."""


# ---------------------------------------------------------------------------
# EAR
# ---------------------------------------------------------------------------
def compute_ear(eye_points: Sequence[Point]) -> float:
    """Calcula o Eye Aspect Ratio de um olho.

    Formula::

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_points: sequencia ordenada ``(p1, p2, p3, p4, p5, p6)`` com as
            coordenadas ``(x, y)`` em pixels dos marcos da palpebra.

    Returns:
        Valor do EAR. Retorna ``0.0`` quando a distancia horizontal e
        degenerada (olho nao visivel / marcos colapsados).

    Raises:
        ValueError: se a sequencia nao possuir exatamente 6 pontos.
    """
    if len(eye_points) != 6:
        raise ValueError(f"EAR exige 6 marcos faciais, recebido: {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = (np.asarray(p, dtype=np.float64) for p in eye_points)

    vertical = dist.euclidean(p2, p6) + dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)

    if horizontal <= 1e-6:
        logger.debug("Distancia horizontal degenerada; EAR definido como 0.0")
        return 0.0

    return float(vertical / (2.0 * horizontal))


def average_ear(left_eye: Sequence[Point], right_eye: Sequence[Point]) -> float:
    """Retorna a media do EAR entre os olhos esquerdo e direito."""
    return (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0


def format_timestamp(seconds: float) -> str:
    """Formata segundos como ``hh:mm:ss.ms`` (milissegundos com 3 digitos)."""
    if seconds < 0:
        raise ValueError("timestamp negativo nao e valido")

    total_ms = int(round(seconds * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ---------------------------------------------------------------------------
# Deteccao de eventos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlinkDetectorConfig:
    """Parametros da maquina de estados de deteccao de piscadas.

    Attributes:
        tau: limiar de EAR abaixo do qual o olho e considerado fechado.
        consec_frames: numero minimo (N) de quadros consecutivos abaixo de
            ``tau`` para registrar uma piscada.
    """

    tau: float = 0.21
    consec_frames: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau deve estar no intervalo (0, 1)")
        if self.consec_frames < 1:
            raise ValueError("consec_frames deve ser >= 1")


@dataclass
class BlinkDetector:
    """Detecta piscadas a partir de uma serie sequencial de valores de EAR.

    A piscada e contabilizada no quadro em que a sequencia de olhos fechados
    termina (reabertura), evitando contagem duplicada em piscadas longas.
    """

    config: BlinkDetectorConfig = field(default_factory=BlinkDetectorConfig)
    accumulated_blinks: int = 0
    _closed_streak: int = 0

    def update(self, ear_value: float) -> Tuple[int, int]:
        """Processa um novo valor de EAR.

        Args:
            ear_value: EAR medio do quadro atual.

        Returns:
            Tupla ``(is_blink, accumulated_blinks)`` onde ``is_blink`` vale 1
            apenas no quadro que fecha o evento de piscada.
        """
        is_blink = 0

        if ear_value < self.config.tau:
            self._closed_streak += 1
        else:
            if self._closed_streak >= self.config.consec_frames:
                self.accumulated_blinks += 1
                is_blink = 1
            self._closed_streak = 0

        return is_blink, self.accumulated_blinks

    def mark_missing_face(self) -> Tuple[int, int]:
        """Registra um quadro sem face detectada.

        A sequencia de olhos fechados e descartada para nao gerar piscadas
        falsas por oclusao ou perda de rastreamento.

        Returns:
            Tupla ``(0, accumulated_blinks)``.
        """
        self._closed_streak = 0
        return 0, self.accumulated_blinks

    def flush(self) -> Tuple[int, int]:
        """Fecha um evento pendente ao fim do video (olho fechado no ultimo quadro)."""
        is_blink = 0
        if self._closed_streak >= self.config.consec_frames:
            self.accumulated_blinks += 1
            is_blink = 1
        self._closed_streak = 0
        return is_blink, self.accumulated_blinks

    def reset(self) -> None:
        """Zera contadores internos."""
        self.accumulated_blinks = 0
        self._closed_streak = 0


# ---------------------------------------------------------------------------
# Registro de telemetria
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameMetrics:
    """Uma linha da telemetria exportada (schema do Power BI)."""

    session_id: str
    timestamp: str
    frame_index: int
    ear_value: float
    is_blink: int
    accumulated_blinks: int
    has_blue_filter: int

    def as_row(self) -> dict[str, object]:
        """Converte o registro em dicionario ordenado conforme :data:`CSV_SCHEMA`."""
        row = asdict(self)
        return {column: row[column] for column in CSV_SCHEMA}


class MetricsExporter:
    """Acumula :class:`FrameMetrics` e exporta para CSV em ``data/processed``."""

    def __init__(
        self,
        session_id: str,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        float_precision: int = 6,
    ) -> None:
        if not session_id:
            raise ValueError("session_id nao pode ser vazio")

        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.float_precision = float_precision
        self._records: List[FrameMetrics] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> Tuple[FrameMetrics, ...]:
        """Registros acumulados (imutavel)."""
        return tuple(self._records)

    def add_frame(
        self,
        frame_index: int,
        timestamp_seconds: float,
        ear_value: float,
        is_blink: int,
        accumulated_blinks: int,
        has_blue_filter: int,
    ) -> FrameMetrics:
        """Registra a telemetria de um quadro e retorna a linha criada."""
        record = FrameMetrics(
            session_id=self.session_id,
            timestamp=format_timestamp(timestamp_seconds),
            frame_index=int(frame_index),
            ear_value=round(float(ear_value), self.float_precision),
            is_blink=int(bool(is_blink)),
            accumulated_blinks=int(accumulated_blinks),
            has_blue_filter=int(bool(has_blue_filter)),
        )
        self._records.append(record)
        return record

    def extend(self, records: Iterable[FrameMetrics]) -> None:
        """Adiciona multiplos registros ja construidos."""
        self._records.extend(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Materializa os registros como ``DataFrame`` com o schema esperado."""
        frame = pd.DataFrame([record.as_row() for record in self._records])
        if frame.empty:
            return pd.DataFrame(columns=list(CSV_SCHEMA))
        return frame.astype(
            {
                "session_id": "string",
                "timestamp": "string",
                "frame_index": "int64",
                "ear_value": "float64",
                "is_blink": "int64",
                "accumulated_blinks": "int64",
                "has_blue_filter": "int64",
            }
        )

    def export(self, filename: str | None = None) -> Path:
        """Grava o CSV de telemetria.

        Args:
            filename: nome do arquivo. Padrao: ``<session_id>.csv``.

        Returns:
            Caminho absoluto do arquivo escrito.

        Raises:
            MetricsExporterError: se nao houver registros ou a escrita falhar.
        """
        if not self._records:
            raise MetricsExporterError(
                "Nenhum registro para exportar; execute o tracker antes."
            )

        target = self.output_dir / (filename or f"{self.session_id}.csv")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.to_dataframe().to_csv(target, index=False, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - depende do filesystem
            raise MetricsExporterError(f"Falha ao escrever {target}: {exc}") from exc

        logger.info("Telemetria exportada: %s (%d linhas)", target, len(self._records))
        return target.resolve()

    def summary(self) -> dict[str, float | int | str]:
        """Resumo agregado da sessao (util para logs e validacao rapida)."""
        if not self._records:
            return {"session_id": self.session_id, "frames": 0, "blinks": 0}

        ears = np.asarray([r.ear_value for r in self._records], dtype=np.float64)
        return {
            "session_id": self.session_id,
            "frames": len(self._records),
            "blinks": self._records[-1].accumulated_blinks,
            "ear_mean": float(ears.mean()),
            "ear_min": float(ears.min()),
            "ear_max": float(ears.max()),
            "duration": self._records[-1].timestamp,
        }
