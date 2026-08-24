"""Ingestao de video (OpenCV) e mapeamento de marcos faciais (MediaPipe Face Mesh).

Fluxo do modulo:

1. Leitura do video quadro a quadro com ``cv2.VideoCapture``.
2. Extracao dos marcos das palpebras via MediaPipe Face Mesh.
3. Calculo do EAR medio e deteccao de piscadas (``src.metrics_exporter``).
4. Deteccao heuristica de filtro de luz azul no quadro.
5. Delegacao da persistencia para :class:`~src.metrics_exporter.MetricsExporter`.

Uso via linha de comando::

    python -m src.vision_tracker data/raw/sessao01.mp4 --session-id sessao01
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Dict, Final, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.metrics_exporter import (
    DEFAULT_OUTPUT_DIR,
    BlinkDetector,
    BlinkDetectorConfig,
    MetricsExporter,
    MetricsExporterError,
    Point,
    average_ear,
)

logger = logging.getLogger(__name__)

#: Indices do Face Mesh (468 marcos) na ordem ``p1..p6`` exigida pelo EAR.
EYE_LANDMARKS: Final[Dict[str, Tuple[int, int, int, int, int, int]]] = {
    "left": (362, 385, 387, 263, 373, 380),
    "right": (33, 160, 158, 133, 153, 144),
}

_DEFAULT_FPS: Final[float] = 30.0


class VisionTrackerError(RuntimeError):
    """Erro na ingestao de video ou no rastreamento facial."""


@dataclass(frozen=True)
class TrackerConfig:
    """Configuracao do rastreador de piscadas.

    Attributes:
        tau: limiar de EAR para olho fechado.
        consec_frames: quadros consecutivos abaixo de ``tau`` para uma piscada.
        min_detection_confidence: confianca minima de deteccao (Face Mesh).
        min_tracking_confidence: confianca minima de rastreamento (Face Mesh).
        refine_landmarks: habilita os marcos refinados de iris/palpebras.
        blue_filter_threshold: razao ``B / ((R + G) / 2)`` acima da qual o quadro
            e considerado sob filtro de luz azul.
        force_blue_filter: sobrescreve a heuristica (``True``/``False``) quando
            o estado do filtro e conhecido a priori.
        preview: exibe uma janela com a anotacao dos marcos (requer display).
    """

    tau: float = 0.21
    consec_frames: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    blue_filter_threshold: float = 1.10
    force_blue_filter: Optional[bool] = None
    preview: bool = False

    def blink_config(self) -> BlinkDetectorConfig:
        """Converte a configuracao para os parametros do detector de piscadas."""
        return BlinkDetectorConfig(tau=self.tau, consec_frames=self.consec_frames)


def detect_blue_filter(frame: np.ndarray, threshold: float = 1.10) -> bool:
    """Heuristica de deteccao de filtro/dominancia de luz azul em um quadro BGR.

    Args:
        frame: quadro no formato BGR (padrao OpenCV).
        threshold: razao minima entre o canal azul e a media de vermelho/verde.

    Returns:
        ``True`` quando o quadro apresenta dominancia do canal azul.
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame deve ser uma imagem BGR de 3 canais")

    blue, green, red = (float(frame[:, :, index].mean()) for index in range(3))
    warm = (red + green) / 2.0
    if warm <= 1e-6:
        return blue > 1e-6
    return (blue / warm) >= threshold


class VisionTracker(AbstractContextManager["VisionTracker"]):
    """Processa um video e produz telemetria de piscadas quadro a quadro."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None

    # -- ciclo de vida ----------------------------------------------------
    def __enter__(self) -> "VisionTracker":
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Libera os recursos do MediaPipe."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    # -- marcos faciais ---------------------------------------------------
    def extract_eye_points(
        self, frame_bgr: np.ndarray
    ) -> Optional[Dict[str, List[Point]]]:
        """Extrai os marcos das palpebras de um quadro.

        Args:
            frame_bgr: quadro BGR lido pelo OpenCV.

        Returns:
            Dicionario ``{"left": [...6 pontos...], "right": [...]}`` em pixels,
            ou ``None`` quando nenhuma face e detectada.

        Raises:
            VisionTrackerError: se o Face Mesh nao foi inicializado.
        """
        if self._face_mesh is None:
            raise VisionTrackerError(
                "Face Mesh nao inicializado; use 'with VisionTracker() as tracker:'"
            )

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        eyes: Dict[str, List[Point]] = {}
        for eye, indices in EYE_LANDMARKS.items():
            eyes[eye] = [
                (landmarks[index].x * width, landmarks[index].y * height)
                for index in indices
            ]
        return eyes

    # -- pipeline ---------------------------------------------------------
    def process_video(
        self,
        video_path: Path | str,
        exporter: MetricsExporter,
        max_frames: Optional[int] = None,
    ) -> MetricsExporter:
        """Executa a ingestao completa de um video, alimentando o ``exporter``.

        Quadros sem face detectada sao registrados com ``ear_value = 0.0`` para
        preservar a continuidade da serie temporal no Power BI.

        Args:
            video_path: caminho do video em ``data/raw``.
            exporter: destino da telemetria.
            max_frames: limite opcional de quadros (util para testes).

        Returns:
            O proprio ``exporter``, ja populado.

        Raises:
            VisionTrackerError: se o video nao existir ou nao puder ser aberto.
        """
        path = Path(video_path)
        if not path.is_file():
            raise VisionTrackerError(f"Video nao encontrado: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VisionTrackerError(f"OpenCV nao conseguiu abrir o video: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or _DEFAULT_FPS
        if fps <= 0 or np.isnan(fps):
            logger.warning("FPS invalido no metadado; assumindo %.1f", _DEFAULT_FPS)
            fps = _DEFAULT_FPS

        detector = BlinkDetector(config=self.config.blink_config())
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                eyes = self.extract_eye_points(frame)
                if eyes is None:
                    logger.debug("Nenhuma face no quadro %d", frame_index)
                    ear_value = 0.0
                    is_blink, accumulated = detector.mark_missing_face()
                else:
                    ear_value = average_ear(eyes["left"], eyes["right"])
                    is_blink, accumulated = detector.update(ear_value)
                has_blue_filter = (
                    self.config.force_blue_filter
                    if self.config.force_blue_filter is not None
                    else detect_blue_filter(frame, self.config.blue_filter_threshold)
                )

                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_seconds = (
                    timestamp_ms / 1000.0
                    if timestamp_ms > 0
                    else frame_index / fps
                )

                exporter.add_frame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    ear_value=ear_value,
                    is_blink=is_blink,
                    accumulated_blinks=accumulated,
                    has_blue_filter=int(has_blue_filter),
                )

                if self.config.preview:
                    self._render_preview(frame, eyes, ear_value, accumulated)

                frame_index += 1
        except KeyboardInterrupt:
            logger.warning("Interrompido pelo usuario no quadro %d", frame_index)
        finally:
            capture.release()
            if self.config.preview:
                cv2.destroyAllWindows()

        pending_blink, accumulated = detector.flush()
        if pending_blink and exporter.records:
            last = exporter.records[-1]
            exporter.add_frame(
                frame_index=last.frame_index,
                timestamp_seconds=frame_index / fps,
                ear_value=last.ear_value,
                is_blink=1,
                accumulated_blinks=accumulated,
                has_blue_filter=last.has_blue_filter,
            )

        logger.info("Processados %d quadros de %s", frame_index, path.name)
        return exporter

    @staticmethod
    def _render_preview(
        frame: np.ndarray,
        eyes: Optional[Dict[str, List[Point]]],
        ear_value: float,
        accumulated_blinks: int,
    ) -> None:
        """Desenha marcos e contadores em uma janela de depuracao."""
        if eyes:
            for points in eyes.values():
                for x, y in points:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"EAR: {ear_value:.3f} | Piscadas: {accumulated_blinks}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Eye Blink Analytics", frame)
        cv2.waitKey(1)


def run_pipeline(
    video_path: Path | str,
    session_id: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    config: TrackerConfig | None = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Orquestra ingestao, deteccao e exportacao de uma sessao.

    Returns:
        Caminho do CSV gerado em ``data/processed``.
    """
    exporter = MetricsExporter(session_id=session_id, output_dir=output_dir)
    with VisionTracker(config=config) as tracker:
        tracker.process_video(video_path, exporter, max_frames=max_frames)
    csv_path = exporter.export()
    logger.info("Resumo da sessao: %s", exporter.summary())
    return csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.vision_tracker",
        description="Eye Blink Analytics - extrai telemetria de piscadas de um video.",
    )
    parser.add_argument("video", type=Path, help="Caminho do video (ex: data/raw/s1.mp4)")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Identificador da sessao (padrao: nome do arquivo de video)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretorio de saida do CSV (padrao: data/processed)",
    )
    parser.add_argument("--tau", type=float, default=0.21, help="Limiar de EAR")
    parser.add_argument(
        "--consec-frames",
        type=int,
        default=2,
        help="Quadros consecutivos abaixo de tau para contar uma piscada",
    )
    parser.add_argument(
        "--blue-filter-threshold",
        type=float,
        default=1.10,
        help="Razao B/((R+G)/2) para considerar filtro de luz azul",
    )
    blue = parser.add_mutually_exclusive_group()
    blue.add_argument(
        "--blue-filter",
        dest="force_blue_filter",
        action="store_true",
        default=None,
        help="Forca has_blue_filter=1 em toda a sessao",
    )
    blue.add_argument(
        "--no-blue-filter",
        dest="force_blue_filter",
        action="store_false",
        help="Forca has_blue_filter=0 em toda a sessao",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limita a quantidade de quadros"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Exibe janela com os marcos faciais"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = TrackerConfig(
        tau=args.tau,
        consec_frames=args.consec_frames,
        blue_filter_threshold=args.blue_filter_threshold,
        force_blue_filter=args.force_blue_filter,
        preview=args.preview,
    )

    try:
        csv_path = run_pipeline(
            video_path=args.video,
            session_id=args.session_id or args.video.stem,
            output_dir=args.output_dir,
            config=config,
            max_frames=args.max_frames,
        )
    except (VisionTrackerError, MetricsExporterError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - salvaguarda de CLI
        logger.exception("Falha inesperada no pipeline: %s", exc)
        return 2

    print(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
