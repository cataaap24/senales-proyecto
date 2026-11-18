"""Fase 4: presentar audios y resultados de la fase 3, sin entrenar el modelo.

Uso: python app/fase4_interfaz.py
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRectF, QPointF, QElapsedTimer, QSize, QSettings
from PySide6.QtGui import (QColor, QPainter, QPen, QBrush, QFont, QLinearGradient, QPainterPath,
                           QIcon, QPixmap, QPalette)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSplitter, QSplitterHandle, QTabWidget, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

import fase3_reconocimiento as reconocimiento
from config import FS, DURACION, N, RAIZ
from espectros_guardados import cargar_espectros

logger = logging.getLogger(__name__)

IMAGENES_ESPECIE = {
    "paloma": RAIZ / "recursos" / "imagenes" / "paloma_torcaza.png",
    "copeton": RAIZ / "recursos" / "imagenes" / "copeton.png",
}

TEMAS = {
    "oscuro": {
        "fondo": "#0F172A", "tarjeta": "#1B2336", "borde": "#2C3852",
        "texto": "#F8FAFC", "apagado": "#94A3B8", "barra_off": "#34425E",
        "acento": "#22C55E", "error": "#F87171",
        "especies": ("#F59E0B", "#38BDF8"),
        "boton": "#272F42", "boton_borde": "#3A4766",
        "boton_hover": "#303B55", "boton_pulsado": "#222A3C",
        "deshabilitado": "#5B6780", "boton_deshabilitado": "#1F273A",
        "primario_texto": "#0F172A", "primario_hover": "#4ADE80",
        "primario_pulsado": "#16A34A", "peligro": "#EF4444",
        "peligro_texto": "#1B0A0A", "peligro_hover": "#F87171",
        "foco": "#86EFAC", "foco_primario": "#F8FAFC",
    },
    "claro": {
        "fondo": "#F4F7FB", "tarjeta": "#FFFFFF", "borde": "#C9D4E2",
        "texto": "#172334", "apagado": "#475569", "barra_off": "#CBD5E1",
        "acento": "#15803D", "error": "#B91C1C",
        "especies": ("#B45309", "#0369A1"),
        "boton": "#EAF0F7", "boton_borde": "#B9C7D7",
        "boton_hover": "#DDE8F3", "boton_pulsado": "#CCDCEC",
        "deshabilitado": "#64748B", "boton_deshabilitado": "#E8EDF3",
        "primario_texto": "#FFFFFF", "primario_hover": "#166534",
        "primario_pulsado": "#14532D", "peligro": "#B91C1C",
        "peligro_texto": "#FFFFFF", "peligro_hover": "#991B1B",
        "foco": "#1D4ED8", "foco_primario": "#172334",
    },
}

FONDO = TARJETA = BORDE = TEXTO = APAGADO = BARRA_OFF = ACENTO = ERROR = QColor()
COLORES_ESPECIES = ()
TONOS = TEMAS["oscuro"]


def seleccionar_colores(modo):
    """Actualiza los colores que leen las tarjetas al repintarse."""
    global FONDO, TARJETA, BORDE, TEXTO, APAGADO, BARRA_OFF, ACENTO, ERROR
    global COLORES_ESPECIES, TONOS
    TONOS = TEMAS[modo]
    FONDO, TARJETA, BORDE, TEXTO = (QColor(TONOS[k]) for k in
                                   ("fondo", "tarjeta", "borde", "texto"))
    APAGADO, BARRA_OFF, ACENTO, ERROR = (QColor(TONOS[k]) for k in
                                       ("apagado", "barra_off", "acento", "error"))
    COLORES_ESPECIES = tuple(QColor(c) for c in TONOS["especies"])


def configurar_paleta(app):
    pal = QPalette()
    for rol, color in ((QPalette.Window, FONDO), (QPalette.WindowText, TEXTO),
                       (QPalette.Base, TARJETA), (QPalette.AlternateBase, FONDO),
                       (QPalette.Text, TEXTO), (QPalette.Button, QColor(TONOS["boton"])),
                       (QPalette.ButtonText, TEXTO), (QPalette.Highlight, ACENTO),
                       (QPalette.HighlightedText, QColor(TONOS["primario_texto"]))):
        pal.setColor(rol, color)
    app.setPalette(pal)
FMAX_INICIAL = 10000.0


def colores_del_modelo(modelo):
    """Asigna colores por posición; los nombres vienen del modelo cargado."""
    return {especie: COLORES_ESPECIES[i % len(COLORES_ESPECIES)]
            for i, especie in enumerate(modelo["especies"])}

SANS = ["Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "DejaVu Sans"]
MONO = ["JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]


def fuente(px, peso=QFont.Normal, mono=False, cursiva=False, espaciado=0.0):
    f = QFont()
    f.setFamilies(MONO if mono else SANS)
    f.setPixelSize(px)
    f.setWeight(peso)
    f.setItalic(cursiva)
    if espaciado:
        f.setLetterSpacing(QFont.AbsoluteSpacing, espaciado)
    return f


def con_alfa(c, a):
    c = QColor(c)
    c.setAlpha(int(max(0, min(255, a))))
    return c


def suave(actual, objetivo, dt, tasa):
    return actual + (objetivo - actual) * (1 - np.exp(-dt * tasa))


NB_ESPECTRO = 220
VENTANA = 4096
HANN = np.hanning(VENTANA)


def perfil_espectro(mag, f, auto, fmax):
    """Reduce un espectro a puntos visibles dentro del rango del modelo."""
    idx = np.searchsorted(f, np.linspace(0, fmax, NB_ESPECTRO + 1))
    fin = int(idx[-1])
    v = np.maximum.reduceat(mag[:fin], idx[:-1])
    db = 20 * np.log10(v / (v.max() if auto else 1.0) + 1e-10)
    piso, techo = (-60.0, 0.0) if auto else (-90.0, -10.0)
    return np.clip((db - piso) / (techo - piso), 0, 1)


def espectro_ventana(seg, fs, fmax):
    """Espectro de una ventana corta (amplitud relativa a plena escala)."""
    if len(seg) < VENTANA:
        seg = np.pad(seg, (0, VENTANA - len(seg)))
    mag = np.abs(np.fft.rfft(seg[:VENTANA] * HANN)) * 2 / HANN.sum()
    return perfil_espectro(mag, np.fft.rfftfreq(VENTANA, 1 / fs), auto=False, fmax=fmax)


def icono(nombre, color, tam=20):
    pm = QPixmap(tam * 2, tam * 2)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    p.setPen(QPen(c, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    if nombre == "mic":
        p.drawRoundedRect(QRectF(7, 2, 6, 10), 3, 3)
        p.drawArc(QRectF(4, 5, 12, 10), 180 * 16, 180 * 16)
        p.drawLine(QPointF(10, 15), QPointF(10, 18))
        p.drawLine(QPointF(7, 18), QPointF(13, 18))
    elif nombre == "carpeta":
        ruta = QPainterPath()
        ruta.moveTo(2.5, 5)
        ruta.lineTo(8, 5)
        ruta.lineTo(10, 7)
        ruta.lineTo(17.5, 7)
        ruta.lineTo(17.5, 16)
        ruta.lineTo(2.5, 16)
        ruta.closeSubpath()
        p.drawPath(ruta)
    elif nombre == "play":
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        ruta = QPainterPath()
        ruta.moveTo(6, 3.5)
        ruta.lineTo(16, 10)
        ruta.lineTo(6, 16.5)
        ruta.closeSubpath()
        p.drawPath(ruta)
    elif nombre == "stop":
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(5, 5, 10, 10), 2, 2)
    elif nombre == "sol":
        p.drawEllipse(QPointF(10, 10), 3.5, 3.5)
        for x1, y1, x2, y2 in (
            (10, 1, 10, 3), (10, 17, 10, 19), (1, 10, 3, 10), (17, 10, 19, 10),
            (3.6, 3.6, 5.1, 5.1), (14.9, 14.9, 16.4, 16.4),
            (3.6, 16.4, 5.1, 14.9), (14.9, 5.1, 16.4, 3.6),
        ):
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    elif nombre == "luna":
        luna = QPainterPath()
        luna.addEllipse(QRectF(2.5, 2.5, 15, 15))
        hueco = QPainterPath()
        hueco.addEllipse(QRectF(8, 0.5, 12, 12))
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPath(luna.subtracted(hueco))
    p.end()
    return QIcon(pm)


class Tarjeta(QWidget):
    def __init__(self, titulo=""):
        super().__init__()
        self.titulo = titulo

    def dibujar_fondo(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(BORDE, 1))
        p.setBrush(TARJETA)
        p.drawRoundedRect(r, 16, 16)
        if self.titulo:
            p.setPen(APAGADO)
            p.setFont(fuente(11, QFont.DemiBold, espaciado=1.2))
            p.drawText(QPointF(20, 29), self.titulo.upper())

    def tick(self, dt):
        return False


class AsaDivisor(QSplitterHandle):
    """Línea de separación discreta con una zona de arrastre generosa."""

    def __init__(self, orientacion, divisor):
        super().__init__(orientacion, divisor)
        self.activa = False
        self.setAttribute(Qt.WA_Hover)

    def enterEvent(self, evento):
        self.activa = True
        self.update()
        super().enterEvent(evento)

    def leaveEvent(self, evento):
        self.activa = False
        self.update()
        super().leaveEvent(evento)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = ACENTO if self.activa else BORDE
        p.setPen(QPen(color, 1.5))
        centro = self.rect().center()
        if self.orientation() == Qt.Horizontal:
            p.drawLine(QPointF(centro.x(), 0), QPointF(centro.x(), self.height()))
        else:
            p.drawLine(QPointF(0, centro.y()), QPointF(self.width(), centro.y()))
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        for desplazamiento in (-8, 0, 8):
            if self.orientation() == Qt.Horizontal:
                punto = QPointF(centro.x(), centro.y() + desplazamiento)
            else:
                punto = QPointF(centro.x() + desplazamiento, centro.y())
            p.drawEllipse(punto, 2, 2)


class Divisor(QSplitter):
    def createHandle(self):
        return AsaDivisor(self.orientation(), self)


class Onda(Tarjeta):
    """Forma de onda como barras; las barras ya recorridas se iluminan."""
    PASO = 6

    def __init__(self):
        super().__init__("Forma de onda")
        self.alturas = np.zeros(0)
        self.objetivo = np.zeros(0)
        self.progreso = 0.0
        self.tiene = False
        self.vivo = False
        self.setMinimumHeight(190)

    def _area(self):
        return QRectF(20, 46, self.width() - 40, self.height() - 46 - 30)

    def _nb(self):
        return max(10, int(self._area().width() // self.PASO))

    def fijar(self, x, vivo=False):
        nb = self._nb()
        n = (len(x) // nb) * nb
        picos = np.abs(x[:n]).reshape(nb, -1).max(axis=1) if n else np.zeros(nb)
        ref = max(float(picos.max()), 0.05)
        self.objetivo = (picos / ref) ** 0.75
        if self.alturas.shape != self.objetivo.shape:
            self.alturas = np.zeros(nb)
        self.tiene, self.vivo = True, vivo

    def vaciar(self):
        self.objetivo = np.zeros_like(self.alturas)
        self.progreso, self.tiene, self.vivo = 0.0, False, False

    def tick(self, dt):
        if self.alturas.shape != self.objetivo.shape:
            return False
        self.alturas = suave(self.alturas, self.objetivo, dt, 22 if self.vivo else 11)
        return bool(np.max(np.abs(self.objetivo - self.alturas)) > 0.002)

    def paintEvent(self, _):
        p = QPainter(self)
        self.dibujar_fondo(p)
        a = self._area()
        cy = a.center().y()
        nb = self._nb()
        p.setPen(Qt.NoPen)
        if self.alturas.shape[0] != nb:
            self.alturas = np.zeros(nb)
            self.objetivo = np.zeros(nb)
        for i in range(nb):
            h = max(1.5, float(self.alturas[i]) * a.height() / 2 * 0.95)
            activa = self.tiene and (i + 0.5) / nb <= self.progreso
            p.setBrush(ACENTO if activa else BARRA_OFF)
            p.drawRoundedRect(QRectF(a.left() + i * self.PASO + 1.5, cy - h, 3, 2 * h), 1.5, 1.5)
        if self.tiene:
            x = a.left() + self.progreso * a.width()
            p.setPen(QPen(con_alfa(TEXTO, 190), 1.5))
            p.drawLine(QPointF(x, a.top() - 4), QPointF(x, a.bottom() + 4))
        p.setPen(APAGADO)
        p.setFont(fuente(11, mono=True))
        p.drawText(QPointF(20, self.height() - 10), "0 s")
        p.drawText(QRectF(70, self.height() - 24, self.width() - 140, 18),
                   Qt.AlignCenter, "Tiempo (s)")
        p.drawText(QRectF(20, 34, self.width() - 40, 16),
                   Qt.AlignRight, "Amplitud relativa")
        p.drawText(
            QRectF(0, self.height() - 24, self.width() - 20, 18),
            Qt.AlignRight | Qt.AlignVCenter, f"{DURACION:.0f} s",
        )
        if self.tiene:
            p.setPen(TEXTO)
            p.drawText(QRectF(0, 14, self.width() - 20, 20), Qt.AlignRight | Qt.AlignVCenter,
                       f"{self.progreso * DURACION:.1f} s")


class Espectro(Tarjeta):
    """Espectro con los cortes que declara el modelo activo."""

    def __init__(self):
        super().__init__("Espectro de frecuencia")
        self.v = np.zeros(NB_ESPECTRO)
        self.t = np.zeros(NB_ESPECTRO)
        self.pico = np.zeros(NB_ESPECTRO)
        self.subtitulo = ""
        self.vivo = False
        self.fase = 0.0
        self.bordes = np.array([0.0, FMAX_INICIAL])
        self.fmax = FMAX_INICIAL
        self.auto = False
        self.setMouseTracking(True)
        self.setMinimumHeight(260)

    def configurar_modelo(self, modelo):
        self.bordes = modelo["bordes_hz"]
        self.fmax = float(self.bordes[-1])
        self.update()

    def fijar(self, perfil, subtitulo="", vivo=False):
        self.t = np.asarray(perfil, float)
        self.subtitulo, self.vivo = subtitulo, vivo
        self.auto = not vivo

    def vaciar(self):
        self.t = np.zeros(NB_ESPECTRO)
        self.subtitulo, self.vivo = "", False

    def _area(self):
        return QRectF(60, 58, max(1, self.width() - 80), max(1, self.height() - 58 - 54))

    def mouseMoveEvent(self, evento):
        area = self._area()
        punto = evento.position()
        if area.contains(punto):
            indice = min(len(self.v) - 1, max(0, round(
                (punto.x() - area.left()) / area.width() * (len(self.v) - 1))))
            valor = float(self.v[indice])
            y = area.bottom() - valor * area.height() * 0.94
            if abs(punto.y() - y) <= 9:
                hz = indice / (len(self.v) - 1) * self.fmax
                db = (-60 + 60 * valor) if self.auto else (-90 + 80 * valor)
                QToolTip.showText(evento.globalPosition().toPoint(),
                                  f"{hz / 1000:.2f} kHz\nNivel: {db:.1f} dB", self)
                return
        QToolTip.hideText()

    def leaveEvent(self, evento):
        QToolTip.hideText()
        super().leaveEvent(evento)

    def tick(self, dt):
        tasa = np.where(self.t > self.v, 24.0, 9.0)
        self.v = self.v + (self.t - self.v) * (1 - np.exp(-dt * tasa))
        self.pico = np.maximum(self.v, self.pico - dt * 0.35)
        self.fase += dt
        return bool(
            np.max(np.abs(self.t - self.v)) > 0.003
            or np.max(self.pico - self.v) > 0.003
            or self.vivo
        )

    def paintEvent(self, _):
        p = QPainter(self)
        self.dibujar_fondo(p)
        a = self._area()
        bordes = self.bordes
        px = lambda f: a.left() + f / self.fmax * a.width()

        for i in range(len(bordes) - 1):
            if i % 2 == 0:
                p.setPen(Qt.NoPen)
                p.setBrush(con_alfa(TEXTO, 7))
                p.drawRect(QRectF(
                    px(bordes[i]), a.top(), px(bordes[i + 1]) - px(bordes[i]), a.height()
                ))
        p.setFont(fuente(11, mono=True))
        piso, techo = (-60, 0) if self.auto else (-90, -10)
        for fraccion in (0.0, 0.5, 1.0):
            y = a.bottom() - fraccion * a.height() * 0.94
            p.setPen(con_alfa(APAGADO, 100))
            p.drawText(QRectF(5, y - 9, 44, 18), Qt.AlignRight | Qt.AlignVCenter,
                       f"{piso + (techo - piso) * fraccion:g}")
        p.setPen(APAGADO)
        p.drawText(QRectF(60, 36, a.width(), 17), Qt.AlignLeft, "Nivel (dB)")
        for i, b in enumerate(bordes):
            x = px(b)
            if 0 < i < len(bordes) - 1:
                p.setPen(QPen(con_alfa(APAGADO, 70), 1, Qt.DashLine))
                p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            p.setPen(APAGADO)
            etiqueta = f"{b / 1000:g}"
            if i == len(bordes) - 1:
                etiqueta += " kHz"
            ancho = 70
            alin = (Qt.AlignLeft if i == 0 else
                    Qt.AlignRight if i == len(bordes) - 1 else Qt.AlignHCenter)
            x0 = x if i == 0 else (x - ancho if i == len(bordes) - 1 else x - ancho / 2)
            p.drawText(QRectF(x0, a.bottom() + 8, ancho, 16), alin | Qt.AlignVCenter, etiqueta)
        p.setPen(QPen(con_alfa(APAGADO, 110), 1))
        p.drawLine(QPointF(a.left(), a.bottom()), QPointF(a.right(), a.bottom()))
        p.setPen(APAGADO)
        p.drawText(QRectF(a.left(), a.bottom() + 26, a.width(), 17),
                   Qt.AlignCenter, "Frecuencia (kHz)")

        def curva(vals):
            k = np.convolve(np.pad(vals, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
            xs = a.left() + np.arange(len(k)) / (len(k) - 1) * a.width()
            ys = a.bottom() - k * a.height() * 0.94
            ruta = QPainterPath(QPointF(xs[0], ys[0]))
            for x, y in zip(xs[1:], ys[1:]):
                ruta.lineTo(x, y)
            return ruta, xs, ys

        fondo, xs, ys = curva(self.pico)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(con_alfa(TEXTO, 70), 1))
        p.drawPath(fondo)

        ruta, xs, ys = curva(self.v)
        area = QPainterPath(ruta)
        area.lineTo(xs[-1], a.bottom())
        area.lineTo(xs[0], a.bottom())
        area.closeSubpath()
        g = QLinearGradient(0, a.top(), 0, a.bottom())
        g.setColorAt(0, con_alfa(ACENTO, 150))
        g.setColorAt(1, con_alfa(ACENTO, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawPath(area)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ACENTO, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(ruta)

        if self.subtitulo:
            p.setFont(fuente(12, QFont.Medium))
            ancho = p.fontMetrics().horizontalAdvance(self.subtitulo)
            p.setPen(TEXTO)
            p.drawText(QRectF(0, 14, self.width() - 20, 20),
                       Qt.AlignRight | Qt.AlignVCenter, self.subtitulo)
            if self.vivo:
                pulso = 0.5 + 0.5 * np.sin(self.fase * 6)
                p.setPen(Qt.NoPen)
                p.setBrush(con_alfa(ERROR, 120 + 135 * pulso))
                p.drawEllipse(QPointF(self.width() - 20 - ancho - 12, 24), 4, 4)


class EspectrosPromedio(Tarjeta):
    """Dibuja las curvas guardadas por especie y los cortes del modelo."""

    def __init__(self):
        super().__init__("Espectros promedio y señal de entrada")
        self.frecuencias = None
        self.series = {}
        self.entrada = None
        self.modelo = None
        self.mensaje = "Los espectros aparecerán cuando la fase 2 exporte sus datos."
        self.setMinimumHeight(380)
        self.setMouseTracking(True)
        self.hover = None
        self.rango = None
        self.controles = False
        self.setAccessibleName("Espectros promedio y señal de entrada")
        self.boton_menos = QToolButton(self)
        self.boton_menos.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.boton_menos.setAutoRaise(True)
        self.boton_menos.setIcon(self._icono_boton("menos"))
        self.boton_menos.setIconSize(QSize(18, 18))
        self.boton_menos.setVisible(False)
        self.boton_menos.clicked.connect(lambda *_: self._ampliar(0.8))
        self.boton_mas = QToolButton(self)
        self.boton_mas.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.boton_mas.setAutoRaise(True)
        self.boton_mas.setIcon(self._icono_boton("mas"))
        self.boton_mas.setIconSize(QSize(18, 18))
        self.boton_mas.setVisible(False)
        self.boton_mas.clicked.connect(lambda *_: self._ampliar(1.25))
        self.boton_reset = QToolButton(self)
        self.boton_reset.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.boton_reset.setAutoRaise(True)
        self.boton_reset.setIcon(self._icono_boton("reset"))
        self.boton_reset.setIconSize(QSize(18, 18))
        self.boton_reset.setVisible(False)
        self.boton_reset.clicked.connect(lambda *_: self._restablecer())

    def fijar(self, modelo, frecuencias, series):
        self.modelo = modelo
        self.frecuencias = frecuencias
        self.series = series
        self.mensaje = ""
        self.hover = None
        self.rango = None
        self.setAccessibleDescription(
            "Curvas de magnitud FFT promedio por especie y de la señal de entrada. "
            "El eje horizontal es k omega cero; las marcas indican el índice k. "
            "Las líneas verticales marcan las subbandas del modelo."
        )
        self.update()

    def vaciar(self, mensaje):
        self.frecuencias = None
        self.series = {}
        self.entrada = None
        self.mensaje = mensaje
        self.hover = None
        self.rango = None
        QToolTip.hideText()
        self.setAccessibleDescription(mensaje)
        self.update()

    def fijar_entrada(self, magnitudes):
        """Agrupa la FFT ya calculada como los espectros guardados."""
        if self.frecuencias is None:
            return
        limites = np.linspace(0, len(magnitudes), len(self.frecuencias) + 1, dtype=int)
        self.entrada = np.add.reduceat(magnitudes, limites[:-1]) / np.diff(limites)
        self.update()

    def vaciar_entrada(self):
        self.entrada = None
        self.update()

    def _rango_completo(self):
        """Rango de frecuencias de la FFT (0 hasta la frecuencia de Nyquist)."""
        puntos_fft = int(self.modelo["fs"] * self.modelo["duracion"])
        return 0.0, (puntos_fft - 1) * self.modelo["fs"] / puntos_fft

    def _escala(self):
        area = QRectF(66, 100, max(1, self.width() - 94), max(1, self.height() - 158))
        minimo, maximo = self.rango if self.rango is not None else self._rango_completo()
        tope = max(max(float(np.max(v)) for v in self.series.values()) * 1.08, 1e-12)
        if self.entrada is not None:
            tope = max(tope, float(np.max(self.entrada)) * 1.08)
        return area, minimo, maximo, tope

    def _icono_boton(self, tipo):
        tamano = 20
        pm = QPixmap(tamano, tamano)
        pm.fill(Qt.transparent)
        pintor = QPainter(pm)
        pintor.setRenderHint(QPainter.Antialiasing)
        centro = QPointF(tamano / 2, tamano / 2)
        radio = 7.0
        color = QColor(148, 163, 184)
        pintor.setPen(QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap))
        if tipo == "mas":
            pintor.drawLine(centro + QPointF(-radio, 0), centro + QPointF(radio, 0))
            pintor.drawLine(centro + QPointF(0, -radio), centro + QPointF(0, radio))
        elif tipo == "menos":
            pintor.drawLine(centro + QPointF(0, -radio), centro + QPointF(0, radio))
        else:
            recto = QRectF(2, 2, tamano - 4, tamano - 4)
            pintor.drawEllipse(recto)
            pintor.drawArc(recto, 90.0, 270.0)
            ruta = QPainterPath(QPointF(10, 3.5))
            ruta.lineTo(QPointF(16.5, 10))
            ruta.lineTo(QPointF(11, 10))
            pintor.setBrush(QBrush(color))
            pintor.fillPath(ruta, QBrush(color))
        pintor.end()
        return QIcon(pm)

    def _ampliar(self, factor):
        if self.frecuencias is None or not self.series or self.modelo is None:
            return
        _, minimo, maximo, _ = self._escala()
        centro = minimo + (maximo - minimo) / 2
        ancho = (maximo - minimo) / factor
        nuevo_minimo = centro - ancho / 2
        nuevo_maximo = centro + ancho / 2
        limite_inferior, limite_superior = self._rango_completo()
        self.rango = (max(limite_inferior, nuevo_minimo), min(limite_superior, nuevo_maximo))
        self.update()

    def _restablecer(self):
        if self.frecuencias is None or not self.series or self.modelo is None:
            return
        self.rango = None
        self.update()

    def _posicionar_controles(self):
        if self.frecuencias is None or not self.series or self.modelo is None:
            return
        area, _, _, _ = self._escala()
        tamano = 30
        separacion = 10
        reset_x = max(area.left(), area.right() - 3 * (tamano + separacion) - tamano)
        self.boton_reset.setGeometry(int(reset_x), int(area.top()), tamano, tamano)
        self.boton_menos.setGeometry(int(reset_x + tamano + separacion), int(area.top()), tamano, tamano)
        self.boton_mas.setGeometry(int(reset_x + 2 * (tamano + separacion)), int(area.top()), tamano, tamano)

    def _mostrar_controles(self):
        for boton in (self.boton_menos, self.boton_mas, self.boton_reset):
            boton.setVisible(True)
        self._posicionar_controles()
        self.update()

    def _ocultar_controles(self):
        for boton in (self.boton_menos, self.boton_mas, self.boton_reset):
            boton.setVisible(False)

    def mouseMoveEvent(self, evento):
        if self.rect().contains(evento.position().toPoint()):
            if not self.controles:
                self.controles = True
                self._mostrar_controles()
        if self.frecuencias is None or not self.series:
            return
        area, minimo, maximo, tope = self._escala()
        punto = evento.position()
        anterior = self.hover
        self.hover = None
        if area.contains(punto):
            hz = minimo + (punto.x() - area.left()) / area.width() * (maximo - minimo)
            if not self.frecuencias[0] <= hz <= self.frecuencias[-1]:
                QToolTip.hideText()
                if anterior is not None:
                    self.update()
                return
            cercano = None
            curvas = list(self.series.items())
            if self.entrada is not None:
                curvas.append(("entrada", self.entrada))
            for especie, valores in curvas:
                magnitud = float(np.interp(hz, self.frecuencias, valores))
                y = area.bottom() - magnitud / tope * area.height()
                distancia = abs(punto.y() - y)
                if distancia <= 9 and (cercano is None or distancia < cercano[0]):
                    cercano = (distancia, especie, magnitud, y)
            if cercano:
                _, especie, magnitud, y = cercano
                self.hover = (especie, hz, magnitud, QPointF(punto.x(), y))
                nombre = ("Señal de entrada" if especie == "entrada" else
                          self.modelo["especies"][especie]["nombre"])
                puntos_fft = int(self.modelo["fs"] * self.modelo["duracion"])
                indice_k = hz * puntos_fft / self.modelo["fs"]
                omega = 2 * np.pi * indice_k / puntos_fft
                QToolTip.showText(evento.globalPosition().toPoint(),
                                  f"{nombre} · k ≈ {indice_k:.0f}\n"
                                  f"kω₀ = {omega:.3f} rad/muestra\n"
                                  f"Magnitud promedio: {magnitud:.3g}", self)
        if self.hover is None:
            QToolTip.hideText()
        if self.hover != anterior:
            self.update()

    def leaveEvent(self, evento):
        self.hover = None
        self.controles = False
        self._ocultar_controles()
        QToolTip.hideText()
        self.update()
        super().leaveEvent(evento)

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self._posicionar_controles()

    def wheelEvent(self, evento):
        """Amplía o reduce el rango de frecuencias visible con la rueda del ratón."""
        if self.frecuencias is None or not self.series or self.modelo is None:
            return
        area, minimo, maximo, _ = self._escala()
        if not area.contains(evento.position()):
            return
        factor = 1.25 ** (-float(evento.angleDelta().y()) / 120.0)
        ancho = (maximo - minimo) / factor
        centro = minimo + (evento.position().x() - area.left()) / area.width() * (maximo - minimo)
        posicion = (evento.position().x() - area.left()) / area.width()
        nuevo_minimo = centro - ancho * posicion
        nuevo_maximo = centro + ancho * (1 - posicion)
        limite_inferior, limite_superior = self._rango_completo()
        self.rango = (max(limite_inferior, nuevo_minimo), min(limite_superior, nuevo_maximo))
        self.update()
        evento.accept()

    def paintEvent(self, _):
        p = QPainter(self)
        self.dibujar_fondo(p)
        if self.frecuencias is None:
            p.setPen(APAGADO)
            p.setFont(fuente(14))
            p.drawText(self.rect().adjusted(36, 60, -36, -36),
                       Qt.AlignCenter | Qt.TextWordWrap, self.mensaje)
            return

        area, minimo, maximo, tope = self._escala()
        bordes = self.modelo["bordes_hz"]
        puntos_fft = int(self.modelo["fs"] * self.modelo["duracion"])

        def x_de(hz):
            return area.left() + (hz - minimo) / (maximo - minimo) * area.width()

        def y_de(magnitud):
            return area.bottom() - magnitud / tope * area.height()

        p.setFont(fuente(11, mono=True))
        marcas = [(0.0, 0)]
        marcas.extend((float(borde), round(float(borde) * puntos_fft / self.modelo["fs"]))
                      for borde in bordes)
        marcas.append((maximo, puntos_fft - 1))
        ultima_etiqueta = -float("inf")
        for i, (hz, indice_k) in enumerate(marcas):
            x = x_de(hz)
            if 0 < i < len(marcas) - 1:
                p.setPen(QPen(con_alfa(APAGADO, 130), 1, Qt.DashLine))
                p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            if x - ultima_etiqueta < 64 and i != len(marcas) - 1:
                continue
            p.setPen(APAGADO)
            ancho = 90
            x0 = x - ancho / 2
            p.drawText(QRectF(x0, area.bottom() + 8, ancho, 18),
                       Qt.AlignHCenter, f"{indice_k:,}".replace(",", " "))
            ultima_etiqueta = x

        for fraccion in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            y = area.bottom() - fraccion * area.height()
            p.setPen(QPen(con_alfa(APAGADO, 70), 1))
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(APAGADO)
            p.drawText(QRectF(4, y - 9, 54, 18), Qt.AlignRight | Qt.AlignVCenter,
                       f"{fraccion * tope:.2g}")

        p.setPen(QPen(APAGADO, 1))
        p.drawLine(QPointF(area.left(), area.top()), QPointF(area.left(), area.bottom()))
        p.drawLine(QPointF(area.left(), area.bottom()), QPointF(area.right(), area.bottom()))
        p.drawText(QRectF(area.left(), area.bottom() + 31, area.width(), 20),
                   Qt.AlignCenter, "kω₀  (ω₀ = 2π/N rad/muestra; marcas en k)")
        p.drawText(QRectF(18, 48, 260, 20), Qt.AlignLeft, "Magnitud ‖X(k)‖")

        colores = colores_del_modelo(self.modelo)
        p.save()
        p.setClipRect(area)
        if self.entrada is not None:
            x_inicial = x_de(float(self.frecuencias[0]))
            y_inicial = y_de(float(self.entrada[0]))
            ruta = QPainterPath(QPointF(x_inicial, y_inicial))
            for hz, magnitud in zip(self.frecuencias[1:], self.entrada[1:]):
                ruta.lineTo(x_de(float(hz)), y_de(float(magnitud)))
            p.setPen(QPen(ACENTO, 2.5, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(ruta)
        for indice, (especie, magnitudes) in reversed(list(enumerate(self.series.items()))):
            x_inicial = x_de(float(self.frecuencias[0]))
            y_inicial = y_de(float(magnitudes[0]))
            ruta = QPainterPath(QPointF(x_inicial, y_inicial))
            for hz, magnitud in zip(self.frecuencias[1:], magnitudes[1:]):
                ruta.lineTo(x_de(float(hz)), y_de(float(magnitud)))
            estilo_linea = Qt.SolidLine if indice == 0 else Qt.DashLine
            p.setPen(QPen(colores[especie], 2.3, estilo_linea, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(ruta)
        p.restore()

        if self.hover:
            especie, _, _, punto = self.hover
            p.setPen(QPen(ACENTO if especie == "entrada" else colores[especie], 2))
            p.setBrush(TARJETA)
            p.drawEllipse(punto, 5, 5)

        p.setFont(fuente(12, QFont.Medium))
        posicion = area.left()
        for indice, especie in enumerate(self.series):
            estilo_linea = Qt.SolidLine if indice == 0 else Qt.DashLine
            p.setPen(QPen(colores[especie], 2.3, estilo_linea))
            p.drawLine(QPointF(posicion, 77), QPointF(posicion + 26, 77))
            p.setPen(TEXTO)
            nombre = self.modelo["especies"][especie]["nombre"]
            p.drawText(QPointF(posicion + 34, 82), nombre)
            posicion += 52 + p.fontMetrics().horizontalAdvance(nombre)
        if self.entrada is not None:
            p.setPen(QPen(ACENTO, 2.5, Qt.DashDotLine))
            p.drawLine(QPointF(posicion, 77), QPointF(posicion + 26, 77))
            p.setPen(TEXTO)
            p.drawText(QPointF(posicion + 34, 82), "Señal de entrada")


class Energias(Tarjeta):
    """Energía por subbanda: el audio frente a los vectores de cada especie."""

    def __init__(self):
        super().__init__("Energía por subbanda")
        self.datos = None
        self.ganadora = None
        self.series = []
        self.colores = {}
        self.bordes = np.array([])
        self.t = 0.0
        self.logs = {}
        self.setMouseTracking(True)
        self.setMinimumHeight(210)

    def fijar(self, res):
        self.series = [("audio", "Tu audio")]
        self.series.extend((clave, datos["nombre"])
                           for clave, datos in res["especies"].items())
        self.colores = {clave: COLORES_ESPECIES[i % len(COLORES_ESPECIES)]
                        for i, clave in enumerate(res["especies"])}
        self.bordes = res["bordes_hz"]
        logs = {"audio": np.log10(res["Ex"] + 1e-12)}
        logs.update({clave: np.log10(vector + 1e-12)
                     for clave, vector in res["vectores"].items()})
        lo = min(v.min() for v in logs.values()) - 0.2
        hi = max(v.max() for v in logs.values()) + 0.15
        self.logs = logs
        self.lo, self.hi = lo, hi
        self.datos = {k: (v - lo) / (hi - lo) for k, v in logs.items()}
        self.ganadora = res["especie"] if res.get("estado", "especie") == "especie" else None
        self.t = 0.0

    def vaciar(self):
        self.datos = None
        QToolTip.hideText()

    def _area(self):
        return QRectF(58, 50, max(1, self.width() - 78), max(1, self.height() - 50 - 84))

    def mouseMoveEvent(self, evento):
        if self.datos is None:
            return
        area = self._area()
        punto = evento.position()
        if not area.contains(punto):
            QToolTip.hideText()
            return
        nb = len(self.bordes) - 1
        banda = min(nb - 1, int((punto.x() - area.left()) / area.width() * nb))
        gw = area.width() / nb
        nseries = len(self.series)
        bw = max(2.0, min(15.0, (gw - 10) / nseries - 2))
        x0 = area.left() + banda * gw + (gw - (nseries * bw + (nseries - 1) * 3)) / 2
        for j, (clave, nombre) in enumerate(self.series):
            x = x0 + j * (bw + 3)
            h = self.datos[clave][banda] * area.height()
            if x - 3 <= punto.x() <= x + bw + 3 and area.bottom() - h - 4 <= punto.y() <= area.bottom():
                QToolTip.showText(evento.globalPosition().toPoint(),
                    f"{nombre} · {self.bordes[banda] / 1000:g}–{self.bordes[banda + 1] / 1000:g} kHz\n"
                    f"Energía: {10 ** self.logs[clave][banda]:.3g}\n"
                    f"log₁₀(energía): {self.logs[clave][banda]:.3f}", self)
                return
        QToolTip.hideText()

    def leaveEvent(self, evento):
        QToolTip.hideText()
        super().leaveEvent(evento)

    def tick(self, dt):
        if self.datos is None or self.t >= 1.6:
            return False
        self.t += dt
        return True

    def paintEvent(self, _):
        p = QPainter(self)
        self.dibujar_fondo(p)
        a = self._area()
        if self.datos is None:
            p.setPen(APAGADO)
            p.setFont(fuente(13))
            p.drawText(self.rect().adjusted(20, 30, -20, 0), Qt.AlignCenter | Qt.TextWordWrap,
                       "Aparece al analizar un audio")
            return
        nb = len(self.bordes) - 1
        nseries = len(self.series)
        gw = a.width() / nb
        bw = max(2.0, min(15.0, (gw - 10) / nseries - 2))
        p.setPen(QPen(con_alfa(APAGADO, 110), 1))
        p.drawLine(QPointF(a.left(), a.bottom()), QPointF(a.right(), a.bottom()))
        p.setFont(fuente(10, mono=True))
        for fraccion in (0.0, 0.5, 1.0):
            y = a.bottom() - fraccion * a.height()
            p.setPen(QPen(con_alfa(APAGADO, 65), 1))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            p.setPen(APAGADO)
            p.drawText(QRectF(4, y - 9, 48, 18), Qt.AlignRight | Qt.AlignVCenter,
                       f"{self.lo + fraccion * (self.hi - self.lo):.1f}")
        p.setFont(fuente(10, QFont.Medium))
        p.drawText(QRectF(58, 34, a.width(), 14), Qt.AlignLeft, "log₁₀(energía)")
        for i in range(nb):
            t = min(1.0, max(0.0, self.t / 0.9 - i * 0.12))
            e = 1 - (1 - t) ** 3
            x0 = a.left() + i * gw + (gw - (nseries * bw + (nseries - 1) * 3)) / 2
            for j, (clave, _) in enumerate(self.series):
                color = TEXTO if clave == "audio" else self.colores[clave]
                if self.ganadora and clave not in ("audio", self.ganadora):
                    color = con_alfa(color, 90)
                h = max(2.0, self.datos[clave][i] * a.height() * e)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(QRectF(x0 + j * (bw + 3), a.bottom() - h, bw, h), 3, 3)
            p.setPen(APAGADO)
            p.setFont(fuente(10, mono=True))
            lo, hi = self.bordes[i] / 1000, self.bordes[i + 1] / 1000
            etiqueta = f"{lo:g}-{hi:g}" if nb <= 5 else f"B{i + 1}"
            p.drawText(QRectF(a.left() + i * gw, a.bottom() + 6, gw, 14),
                       Qt.AlignCenter, etiqueta)
        p.setPen(APAGADO)
        p.setFont(fuente(10, mono=True))
        p.drawText(QRectF(a.left(), a.bottom() + 22, a.width(), 16),
                   Qt.AlignCenter, "Subbanda (kHz)" if nb <= 5 else "Subbanda")
        x = a.left()
        p.setFont(fuente(11))
        for clave, nombre in self.series:
            p.setPen(Qt.NoPen)
            p.setBrush(TEXTO if clave == "audio" else self.colores[clave])
            p.drawEllipse(QPointF(x + 4, self.height() - 14), 4, 4)
            p.setPen(TEXTO)
            ancho = p.fontMetrics().horizontalAdvance(nombre)
            p.drawText(QPointF(x + 14, self.height() - 10), nombre)
            x += ancho + 34


class Resultado(Tarjeta):
    """Muestra el nombre que viene en el modelo o un estado informativo."""

    def __init__(self):
        super().__init__("Resultado")
        self.estado = "vacio"
        self.especie = None
        self.t = 0.0
        self.aparicion = 1.0
        self.especies = {}
        self.colores = {}
        self.imagenes = {}
        self.setMinimumHeight(190)

    def configurar_modelo(self, modelo):
        self.especies = modelo["especies"]
        self.colores = colores_del_modelo(modelo)
        self.imagenes = {}
        for clave in self.especies:
            ruta = IMAGENES_ESPECIE.get(clave)
            if ruta is not None and ruta.exists():
                self.imagenes[clave] = QPixmap(str(ruta))

    def poner(self, estado, especie=None):
        self.estado, self.especie, self.t, self.aparicion = estado, especie, 0.0, 0.0

    def tick(self, dt):
        self.t += dt
        self.aparicion = min(1.0, self.aparicion + dt * 4)
        if self.estado == "analizando":
            return True
        if self.estado == "especie":
            return self.t < 6.0
        return self.aparicion < 1.0

    def paintEvent(self, _):
        p = QPainter(self)
        self.dibujar_fondo(p)
        color = self.colores.get(self.especie, APAGADO)
        c = QPointF(56, 100)
        radio = 34
        if self.estado == "analizando":
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(con_alfa(APAGADO, 60), 3))
            p.drawEllipse(c, radio, radio)
            p.setPen(QPen(ACENTO, 3, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(c.x() - radio, c.y() - radio, 2 * radio, 2 * radio),
                      int(-self.t * 300 * 16), 100 * 16)
        elif self.estado == "especie":
            fase = (self.t % 2.0) / 2.0
            if self.t < 6.0:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(con_alfa(color, 150 * (1 - fase)), 2))
                p.drawEllipse(c, radio + 14 * fase, radio + 14 * fase)
            imagen = self.imagenes.get(self.especie)
            if imagen is not None and not imagen.isNull():
                lado = 2 * radio - 10
                escala = min(lado / imagen.width(), lado / imagen.height())
                ancho_img = imagen.width() * escala
                alto_img = imagen.height() * escala
                p.drawPixmap(QRectF(c.x() - ancho_img / 2, c.y() - alto_img / 2,
                                    ancho_img, alto_img), imagen, imagen.rect())
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(con_alfa(color, 50))
                p.drawEllipse(c, radio, radio)
                p.setBrush(color)
                p.drawEllipse(c, 12, 12)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(con_alfa(APAGADO, 110), 2, Qt.DashLine))
            p.drawEllipse(c, radio, radio)

        if self.estado == "especie":
            datos = self.especies[self.especie]
            nombre = datos["nombre"]
            sub = datos.get("cientifico", "")
            nota = "Especie del banco"
        elif self.estado == "no_reconocido":
            cercana = self.especies.get(self.especie, {}).get("nombre", "")
            nombre, sub, nota = "No reconocido", f"Fuera del intervalo de energías (lo más cercano: {cercana})", ""
        elif self.estado == "sin_canto":
            nombre, sub, nota = "Sin canto", "Se escucha ruido o silencio, no un canto", ""
        elif self.estado == "sin_senal":
            nombre, sub, nota = "Sin señal", "No se detectó audio suficiente", ""
        elif self.estado == "sin_modelo":
            nombre, sub, nota = "Sin modelo", "Ejecuta primero la fase 2", ""
        elif self.estado == "analizando":
            nombre, sub, nota = "Analizando…", "Comparando con el banco de cantos", ""
        else:
            nombre, sub, nota = "Sin análisis", "Carga un audio o escucha con el micrófono", ""
        a = self.aparicion
        x = 110
        ancho = self.width() - x - 20
        p.setPen(con_alfa(TEXTO if self.estado != "vacio" else APAGADO, 255 * a))
        p.setFont(fuente(22, QFont.DemiBold))
        p.drawText(QRectF(x, 62, ancho, 32), Qt.AlignLeft | Qt.AlignVCenter, nombre)
        p.setPen(con_alfa(APAGADO, 255 * a))
        if self.estado == "especie":
            p.setFont(fuente(14, cursiva=True))
        else:
            p.setFont(fuente(13))
        p.drawText(QRectF(x, 96, ancho, 44), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, sub)
        if nota:
            p.setPen(con_alfa(color, 255 * a))
            p.setFont(fuente(12, QFont.DemiBold))
            p.drawText(QRectF(x, 140, ancho, 20), Qt.AlignLeft | Qt.AlignVCenter, nota)


class Trabajo(QThread):
    listo = Signal(object)
    fallo = Signal(str)

    def __init__(self, funcion):
        super().__init__()
        self.funcion = funcion

    def run(self):
        try:
            self.listo.emit(self.funcion())
        except Exception as e:
            self.fallo.emit(str(e))


def estilo():
    c = TONOS
    return f"""
QMainWindow, QWidget#raiz {{ background: {c['fondo']}; }}
QFrame#panel {{ background: {c['tarjeta']}; border: 1px solid {c['borde']}; border-radius: 16px; }}
QTabWidget::pane {{ border: 0; background: transparent; }}
QTabBar::tab {{ background: {c['boton']}; color: {c['apagado']}; border: 1px solid {c['borde']};
    border-radius: 9px; min-height: 38px; min-width: 160px; padding: 0 18px; margin-right: 8px;
    font-size: 13px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {c['tarjeta']}; color: {c['texto']}; border-color: {c['acento']}; }}
QTabBar::tab:focus {{ border: 2px solid {c['foco']}; }}
QLabel {{ background: transparent; color: {c['texto']}; }}
QLabel#sub {{ color: {c['apagado']}; }}
QLabel#aviso {{ color: {c['error']}; }}
QPushButton {{ background: {c['boton']}; color: {c['texto']}; border: 1px solid {c['boton_borde']}; border-radius: 12px;
    min-height: 46px; padding: 0 16px; font-size: 14px; font-weight: 600; text-align: left; }}
QPushButton:hover {{ background: {c['boton_hover']}; }}
QPushButton:pressed {{ background: {c['boton_pulsado']}; }}
QPushButton:focus {{ border: 2px solid {c['foco']}; }}
QPushButton:disabled {{ color: {c['deshabilitado']}; background: {c['boton_deshabilitado']}; border-color: {c['borde']}; }}
QPushButton#primario {{ background: {c['acento']}; color: {c['primario_texto']}; border: 1px solid {c['acento']}; }}
QPushButton#primario:hover {{ background: {c['primario_hover']}; }}
QPushButton#primario:pressed {{ background: {c['primario_pulsado']}; }}
QPushButton#primario:focus {{ border: 2px solid {c['foco_primario']}; }}
QPushButton#peligro {{ background: {c['peligro']}; color: {c['peligro_texto']}; border: 1px solid {c['peligro']}; }}
QPushButton#peligro:hover {{ background: {c['peligro_hover']}; }}
QPushButton#peligro:focus {{ border: 2px solid {c['foco_primario']}; }}
QPushButton#tema {{ min-height: 42px; max-height: 42px; padding: 0; text-align: center; }}
"""


class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocedor de aves")
        self.resize(1240, 900)
        self.setMinimumSize(1060, 840)
        self.ajustes = QSettings("senales-aves", "ReconocedorAves")
        guardado = self.ajustes.value("tema", "oscuro")
        self.tema = guardado if guardado in TEMAS else "oscuro"

        self.modelo = None
        self.x5 = None
        self.resultado = None
        self.estado = "reposo"
        self.token = 0
        self.trabajos = set()
        self.cron = QElapsedTimer()
        self.reloj = QTimer(self)
        self.reloj.setInterval(16)
        self.reloj.timeout.connect(self.cuadro)
        self.stream = None

        raiz = QWidget(objectName="raiz")
        self.setCentralWidget(raiz)
        col = QVBoxLayout(raiz)
        col.setContentsMargins(28, 22, 28, 28)
        col.setSpacing(18)

        cab = QHBoxLayout()
        titulos = QVBoxLayout()
        titulos.setSpacing(2)
        t = QLabel("Reconocedor de aves")
        t.setFont(fuente(24, QFont.DemiBold))
        self.etiqueta_especies = QLabel("Modelo pendiente")
        self.etiqueta_especies.setObjectName("sub")
        self.etiqueta_especies.setFont(fuente(13))
        titulos.addWidget(t)
        titulos.addWidget(self.etiqueta_especies)
        cab.addLayout(titulos)
        cab.addStretch()
        self.b_tema = QPushButton()
        self.b_tema.setObjectName("tema")
        self.b_tema.setFixedSize(44, 44)
        self.b_tema.setIconSize(QSize(20, 20))
        self.b_tema.setCursor(Qt.PointingHandCursor)
        self.b_tema.clicked.connect(self.alternar_tema)
        cab.addWidget(self.b_tema, alignment=Qt.AlignVCenter)
        col.addLayout(cab)

        self.pestanas = QTabWidget()
        self.pestanas.setDocumentMode(True)
        col.addWidget(self.pestanas, 1)
        pagina_reconocimiento = QWidget()
        cuerpo = QVBoxLayout(pagina_reconocimiento)
        cuerpo.setContentsMargins(0, 12, 0, 0)
        self.pestanas.addTab(pagina_reconocimiento, "Reconocimiento")

        self.divisor_columnas = Divisor(Qt.Horizontal)
        self.divisor_columnas.setHandleWidth(12)
        self.divisor_columnas.setChildrenCollapsible(False)
        self.divisor_columnas.setAccessibleName("Ajustar ancho de las gráficas")
        cuerpo.addWidget(self.divisor_columnas)

        self.divisor_graficas = Divisor(Qt.Vertical)
        self.divisor_graficas.setHandleWidth(12)
        self.divisor_graficas.setChildrenCollapsible(False)
        self.divisor_graficas.setAccessibleName("Ajustar altura de forma de onda y espectro")
        self.onda = Onda()
        self.espectro = Espectro()
        self.divisor_graficas.addWidget(self.onda)
        self.divisor_graficas.addWidget(self.espectro)
        self.divisor_graficas.setStretchFactor(0, 1)
        self.divisor_graficas.setStretchFactor(1, 2)
        self.divisor_columnas.addWidget(self.divisor_graficas)

        columna_derecha = QWidget()
        columna_derecha.setMinimumWidth(410)
        der = QVBoxLayout(columna_derecha)
        der.setContentsMargins(0, 0, 0, 0)
        der.setSpacing(18)
        self.divisor_columnas.addWidget(columna_derecha)
        self.divisor_columnas.setStretchFactor(0, 3)
        self.divisor_columnas.setStretchFactor(1, 2)

        panel = QFrame(objectName="panel")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(20, 20, 20, 16)
        pl.setSpacing(10)
        self.b_mic = QPushButton("  Escuchar con micrófono")
        self.b_mic.setObjectName("primario")
        self.b_cargar = QPushButton("  Cargar audio")
        self.b_play = QPushButton("  Reproducir")
        for b in (self.b_mic, self.b_cargar, self.b_play):
            b.setCursor(Qt.PointingHandCursor)
            b.setIconSize(QSize(20, 20))
            pl.addWidget(b)
        self.b_play.setEnabled(False)
        self.info = QLabel("Ningún audio cargado")
        self.info.setObjectName("sub")
        self.info.setFont(fuente(12, mono=True))
        self.aviso = QLabel("")
        self.aviso.setObjectName("aviso")
        self.aviso.setFont(fuente(12))
        self.aviso.setWordWrap(True)
        self.aviso.hide()
        pl.addWidget(self.info)
        pl.addWidget(self.aviso)
        der.addWidget(panel)

        self.tarjeta_res = Resultado()
        self.energias = Energias()
        self.divisor_resultados = Divisor(Qt.Vertical)
        self.divisor_resultados.setHandleWidth(12)
        self.divisor_resultados.setChildrenCollapsible(False)
        self.divisor_resultados.setAccessibleName("Ajustar altura de resultado y energía por subbanda")
        self.divisor_resultados.addWidget(self.tarjeta_res)
        self.divisor_resultados.addWidget(self.energias)
        self.divisor_resultados.setStretchFactor(0, 1)
        self.divisor_resultados.setStretchFactor(1, 2)
        der.addWidget(self.divisor_resultados, 1)
        for clave, divisor, tamanos in (
            ("columnas", self.divisor_columnas, [670, 460]),
            ("graficas", self.divisor_graficas, [240, 400]),
            ("resultados", self.divisor_resultados, [190, 300]),
        ):
            estado = self.ajustes.value(f"divisor/{clave}")
            if estado is None or not divisor.restoreState(estado):
                divisor.setSizes(tamanos)
            divisor.splitterMoved.connect(
                lambda _pos, _indice, nombre=clave, control=divisor:
                self.ajustes.setValue(f"divisor/{nombre}", control.saveState())
            )

        pagina_espectros = QWidget()
        contenido = QVBoxLayout(pagina_espectros)
        contenido.setContentsMargins(0, 12, 0, 0)
        contenido.setSpacing(16)
        cabecera = QHBoxLayout()
        titulo_espectros = QLabel("Comparación de especies")
        titulo_espectros.setFont(fuente(19, QFont.DemiBold))
        cabecera.addWidget(titulo_espectros)
        cabecera.addStretch()
        self.b_recargar_espectros = QPushButton("Actualizar datos")
        self.b_recargar_espectros.setCursor(Qt.PointingHandCursor)
        self.b_recargar_espectros.setAccessibleName("Actualizar espectros promedio")
        cabecera.addWidget(self.b_recargar_espectros)
        contenido.addLayout(cabecera)
        self.espectros_promedio = EspectrosPromedio()
        contenido.addWidget(self.espectros_promedio, 1)
        self.estado_espectros = QLabel("")
        self.estado_espectros.setObjectName("sub")
        self.estado_espectros.setFont(fuente(12))
        self.estado_espectros.setWordWrap(True)
        contenido.addWidget(self.estado_espectros)

        notas = QHBoxLayout()
        notas.setSpacing(16)
        for titulo, descripcion in (
            ("Qué muestran las curvas",
             "Las curvas de referencia unen las magnitudes FFT promedio de cada especie. "
             "Al analizar un audio aparece también su espectro de entrada."),
            ("Energía de referencia por banda", ""),
        ):
            tarjeta = QFrame(objectName="panel")
            caja = QVBoxLayout(tarjeta)
            caja.setContentsMargins(18, 16, 18, 16)
            caja.setSpacing(8)
            encabezado = QLabel(titulo)
            encabezado.setFont(fuente(14, QFont.DemiBold))
            texto = QLabel(descripcion)
            texto.setObjectName("sub")
            texto.setFont(fuente(13))
            texto.setWordWrap(True)
            if not descripcion:
                self.resumen_modelo = texto
            caja.addWidget(encabezado)
            caja.addWidget(texto)
            caja.addStretch()
            notas.addWidget(tarjeta, 1)
        contenido.addLayout(notas)
        self.pestanas.addTab(pagina_espectros, "Espectros promedio")

        self.graficas = [self.onda, self.espectro, self.energias, self.tarjeta_res]
        self.b_mic.clicked.connect(self.alternar_microfono)
        self.b_cargar.clicked.connect(self.cargar)
        self.b_play.clicked.connect(self.alternar_reproduccion)
        self.b_recargar_espectros.clicked.connect(self.actualizar_espectros)
        self.pestanas.currentChanged.connect(self.cambio_de_pestana)
        self.aplicar_tema()

        self.cargar_modelo()

    def aplicar_tema(self):
        """Aplica el tema a Qt y a las gráficas sin alterar el análisis activo."""
        seleccionar_colores(self.tema)
        configurar_paleta(QApplication.instance())
        self.setStyleSheet(estilo())
        self.b_tema.setIcon(icono("sol" if self.tema == "oscuro" else "luna",
                                   TONOS["texto"]))
        descripcion = "Cambiar al modo claro" if self.tema == "oscuro" else "Cambiar al modo oscuro"
        self.b_tema.setToolTip(descripcion)
        self.b_tema.setAccessibleName(descripcion)
        self.refrescar_botones()
        if self.modelo:
            self.tarjeta_res.configurar_modelo(self.modelo)
        especies_visibles = [clave for clave, _ in self.energias.series if clave != "audio"]
        self.energias.colores = {
            clave: COLORES_ESPECIES[i % len(COLORES_ESPECIES)]
            for i, clave in enumerate(especies_visibles)
        }
        self.espectros_promedio.update()
        self.update_all()

    def alternar_tema(self):
        self.tema = "claro" if self.tema == "oscuro" else "oscuro"
        self.ajustes.setValue("tema", self.tema)
        self.aplicar_tema()

    def animar(self):
        if not self.reloj.isActive():
            self.cron.restart()
            self.reloj.start()

    def cuadro(self):
        dt = min(self.cron.restart() / 1000.0, 0.05)
        if self.estado == "reproduciendo":
            pos = time.perf_counter() - self.t0
            if pos >= DURACION:
                self.detener()
            else:
                self.onda.progreso = pos / DURACION
                i = int(pos * FS)
                tramo = self.x5[max(0, i - VENTANA // 2):i + VENTANA // 2]
                self.espectro.fijar(
                    espectro_ventana(tramo, FS, self.espectro.fmax),
                    "Reproduciendo", vivo=True,
                )
        elif self.estado == "grabando":
            idx = self.idx_buf
            self.onda.fijar(self.buf, vivo=True)
            self.onda.progreso = min(1.0, idx / len(self.buf))
            self.espectro.fijar(espectro_ventana(self.buf[max(0, idx - VENTANA): idx], self.sr,
                                                self.espectro.fmax), "Escuchando", vivo=True)
            if idx >= len(self.buf):
                self.terminar_grabacion()
        activos = [g.tick(dt) for g in self.graficas]
        self.update_all()
        if not any(activos) and self.estado == "reposo":
            self.reloj.stop()

    def update_all(self):
        for g in self.graficas:
            g.update()

    def cambio_de_pestana(self, indice):
        if indice == 1:
            self.actualizar_espectros()

    def actualizar_espectros(self):
        """Vuelve a leer los dos archivos existentes; nunca inicia entrenamiento."""
        if not self.cargar_modelo():
            mensaje = "Primero se necesita un modelo válido de la fase 2."
            self.espectros_promedio.vaciar(mensaje)
            self.estado_espectros.clear()
            self.resumen_modelo.setText(mensaje)
            return
        bordes = self.modelo["bordes_hz"]
        bandas = " · ".join(
            f"B{i}: {inferior:g}–{superior:g} Hz"
            for i, (inferior, superior) in enumerate(zip(bordes[:-1], bordes[1:]), 1)
        )
        referencias = [bandas]
        for especie, datos in self.modelo["especies"].items():
            valores = " · ".join(f"{valor:.3g}" for valor in self.modelo["vectores_energia"][especie])
            referencias.append(f"{datos['nombre']}: {valores}")
        self.resumen_modelo.setText("\n".join(referencias))
        try:
            frecuencias, series = cargar_espectros(self.modelo)
        except (OSError, ValueError) as error:
            mensaje = str(error)
            self.espectros_promedio.vaciar(mensaje)
            self.estado_espectros.clear()
            return
        self.espectros_promedio.fijar(self.modelo, frecuencias, series)
        if self.resultado is not None:
            self.espectros_promedio.fijar_entrada(self.resultado["mag"])
        self.estado_espectros.setText(
            "FFT completa · eje kω₀ · escala vertical lineal"
        )

    def refrescar_botones(self):
        grab = self.estado == "grabando"
        rep = self.estado == "reproduciendo"
        self.b_mic.setText("  Detener" if grab else "  Escuchar con micrófono")
        self.b_mic.setObjectName("peligro" if grab else "primario")
        self.b_mic.setIcon(icono("stop" if grab else "mic",
                                  TONOS["peligro_texto" if grab else "primario_texto"]))
        self.b_cargar.setIcon(icono("carpeta", TONOS["texto"]))
        self.b_play.setText("  Detener" if rep else "  Reproducir")
        self.b_cargar.setEnabled(not grab)
        self.b_play.setEnabled(self.x5 is not None and not grab)
        color_play = TONOS["texto" if self.b_play.isEnabled() else "deshabilitado"]
        self.b_play.setIcon(icono("stop" if rep else "play", color_play))
        for b in (self.b_mic,):
            b.style().unpolish(b)
            b.style().polish(b)

    def poner_aviso(self, texto=""):
        self.aviso.setText(texto)
        self.aviso.setVisible(bool(texto))

    def cargar_modelo(self):
        """Lee el JSON actual; nunca inicia el entrenamiento."""
        try:
            self.modelo = reconocimiento.cargar_modelo()
        except (OSError, ValueError) as error:
            self.modelo = None
            logger.warning("Sin modelo: %s", error)
            self.etiqueta_especies.setText("Modelo pendiente")
            self.poner_aviso(str(error))
            return False
        logger.info("Modelo listo")
        self.etiqueta_especies.setText(" · ".join(
            datos["nombre"] for datos in self.modelo["especies"].values()))
        self.espectro.configurar_modelo(self.modelo)
        self.tarjeta_res.configurar_modelo(self.modelo)
        self.poner_aviso("")
        return True

    def lanzar(self, trabajo):
        self.trabajos.add(trabajo)
        trabajo.finished.connect(lambda: self.trabajos.discard(trabajo))
        trabajo.start()

    def parar_todo(self):
        if self.estado == "reproduciendo":
            self.detener()
        elif self.estado == "grabando":
            self.cancelar_grabacion()

    def cargar(self):
        self.parar_todo()
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar audio", "",
            "Audio (*.wav *.flac *.ogg *.mp3 *.aiff);;Todos los archivos (*)",
        )
        if ruta:
            self.abrir_archivo(ruta)

    def abrir_archivo(self, ruta):
        self.poner_aviso("")
        try:
            x = reconocimiento.cargar_audio(ruta)
        except Exception as e:
            self.poner_aviso(f"No se pudo leer el audio: {e}")
            return
        if len(x) < FS // 2:
            self.poner_aviso("El audio es demasiado corto.")
            return
        self.x5 = reconocimiento.recortar(x)
        self.info.setText(f"{Path(ruta).name} · {len(x) / FS:.1f} s")
        self.mostrar_senal()

    def mostrar_senal(self):
        self.onda.fijar(self.x5)
        self.onda.progreso = 1.0
        self.espectro.vaciar()
        self.energias.vaciar()
        self.resultado = None
        self.espectros_promedio.vaciar_entrada()
        self.refrescar_botones()
        if np.max(np.abs(self.x5)) < 1e-4:
            self.token += 1
            self.tarjeta_res.poner("sin_senal")
            self.poner_aviso("No se detectó señal suficiente para reconocer un canto.")
        else:
            self.analizar()
        self.animar()

    def analizar(self):
        self.token += 1
        if not self.cargar_modelo():
            self.tarjeta_res.poner("sin_modelo")
            self.animar()
            return
        token, x = self.token, self.x5
        self.tarjeta_res.poner("analizando")
        modelo = self.modelo
        trabajo = Trabajo(lambda: reconocimiento.reconocer(x, modelo))
        trabajo.listo.connect(lambda r: self.analisis_listo(token, r))
        trabajo.fallo.connect(lambda m: self.poner_aviso(f"Error al analizar: {m}"))
        self.lanzar(trabajo)
        self.animar()

    def analisis_listo(self, token, res):
        if token != self.token:
            return
        self.resultado = res
        self.espectros_promedio.fijar_entrada(res["mag"])
        if self.estado != "reproduciendo":
            self.espectro.fijar(perfil_espectro(res["mag"], res["frecuencias"], auto=True,
                                                  fmax=self.espectro.fmax), "Audio de 5 s")
        self.energias.fijar(res)
        self.tarjeta_res.poner(res.get("estado", "especie"), res["especie"])
        self.animar()

    def alternar_reproduccion(self):
        if self.estado == "reproduciendo":
            self.detener()
            return
        self.poner_aviso("")
        try:
            import sounddevice as sd
            sd.play(0.9 * self.x5 / (np.max(np.abs(self.x5)) + 1e-12), FS)
        except Exception:
            self.poner_aviso(
                "No hay salida de audio disponible: se muestra la animación sin sonido."
            )
        self.estado = "reproduciendo"
        self.t0 = time.perf_counter()
        self.onda.progreso = 0.0
        self.refrescar_botones()
        self.animar()

    def detener(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.estado = "reposo"
        self.onda.progreso = 1.0
        if self.resultado:
            perfil = perfil_espectro(
                self.resultado["mag"], self.resultado["frecuencias"],
                auto=True, fmax=self.espectro.fmax,
            )
            self.espectro.fijar(perfil, "Audio de 5 s")
        else:
            self.espectro.vaciar()
        self.refrescar_botones()
        self.animar()

    def alternar_microfono(self):
        if self.estado == "grabando":
            self.cancelar_grabacion()
            return
        self.parar_todo()
        self.poner_aviso("")
        try:
            import sounddevice as sd
        except Exception:
            self.poner_aviso("No se pudo cargar el audio del sistema (sounddevice / PortAudio).")
            return
        self.stream = None
        for sr in (FS, 2 * FS):
            self.sr = sr
            self.buf = np.zeros(int(DURACION * sr), dtype="float32")
            self.idx_buf = 0
            try:
                self.stream = sd.InputStream(
                    samplerate=sr, channels=1, dtype="float32", callback=self._entrada
                )
                self.stream.start()
                break
            except Exception:
                self.stream = None
        if self.stream is None:
            self.poner_aviso(
                "No se pudo abrir el micrófono. Revisa el dispositivo y los permisos."
            )
            return
        self.x5 = None
        self.resultado = None
        self.espectros_promedio.vaciar_entrada()
        self.token += 1
        self.tarjeta_res.poner("vacio")
        self.energias.vaciar()
        self.onda.vaciar()
        self.info.setText("Micrófono · 5.0 s")
        self.estado = "grabando"
        self.refrescar_botones()
        self.animar()

    def _entrada(self, indata, frames, tiempo, estado):
        n = min(frames, len(self.buf) - self.idx_buf)
        if n > 0:
            self.buf[self.idx_buf:self.idx_buf + n] = indata[:n, 0]
            self.idx_buf += n

    def _cerrar_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def cancelar_grabacion(self):
        self._cerrar_stream()
        self.estado = "reposo"
        self.onda.vaciar()
        self.espectro.vaciar()
        self.info.setText("Ningún audio cargado")
        self.refrescar_botones()
        self.animar()

    def terminar_grabacion(self):
        self._cerrar_stream()
        self.estado = "reposo"
        x = self.buf.astype(float)
        if self.sr != FS:
            import captura
            x = captura.diezmar(x, self.sr // FS, self.sr)
        self.x5 = reconocimiento.recortar(x)
        self.mostrar_senal()

    def closeEvent(self, e):
        self._cerrar_stream()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        for t in list(self.trabajos):
            t.wait(2000)
        super().closeEvent(e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    v = Ventana()
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
