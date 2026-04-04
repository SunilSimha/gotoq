"""Interactive desktop viewer for DESI QSO cutouts and spectra.

This module provides a PyQt6-based GUI for browsing and analyzing DESI QSO data.
Features include:
  - Browsing QSO catalog with navigation controls
  - Viewing cutout images and spectra side-by-side
  - Interactive spectral overlays with emission/absorption lines
  - Redshift adjustment with fine slider and text entry
  - Persistent per-object notes storage
  - Keyboard shortcuts for rapid navigation and parameter adjustment
  - Dynamic line overlay synchronization

Main classes:
  - NotesStore: Persistent storage for per-QSO notes
  - QSODataStore: Indexes and loads QSO assets (cutouts, spectra, catalog data)
  - ImageLabel: Displays and scales cutout images
  - CustomViewBox: Scroll-wheel zoom behavior (X-only or Y-only with Ctrl)
  - QSOViewer: Main application window and controller
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def _dot_line_style():
	"""Return a dotted line style compatible with this Qt version.
	
	Handles differences between Qt5 and Qt6 style enums.
	"""
	if hasattr(QtCore.Qt, "PenStyle"):
		return QtCore.Qt.PenStyle.DotLine
	return QtCore.Qt.DotLine


def normalize_desi_id(value: object) -> str:
	"""Normalize DESI_ID strings by stripping trailing .0 (common in float conversion).
	
	Args:
		value: Object to normalize (typically string or float).
		
	Returns:
		Normalized DESI_ID string.
	"""
	text = str(value).strip()
	if text.endswith(".0"):
		text = text[:-2]
	return text


@dataclass
class NotesRecord:
	"""Container for a single QSO note with timestamp.
	
	Attributes:
		note: User-written note text.
		updated_at: ISO-format timestamp of last update.
	"""
	note: str
	updated_at: str


@dataclass
class SpectralLine:
	"""Container for a spectral line definition.
	
	Attributes:
		name: Line name/label (e.g., 'Ly alpha', 'H-beta').
		wavelength: Rest-frame vacuum wavelength in Angstroms.
		major: If True, included in 'major-only' filter mode.
	"""
	name: str
	wavelength: float
	major: bool


class NotesStore:
	"""Manage persistent per-QSO notes storage as CSV.
	
	Notes are stored in a DESI_ID-indexed CSV file with timestamps.
	Changes are written to a temporary file then atomic-swapped to prevent
	data loss on crash.
	"""
	def __init__(self, notes_path: Path):
		"""Initialize notes store.
		
		Args:
			notes_path: Path to CSV file for persisting notes.
		"""
		self.notes_path = notes_path
		self._notes: Dict[str, NotesRecord] = {}
		self._load()

	def _load(self) -> None:
		"""Load notes from CSV file if it exists."""
		if not self.notes_path.exists():
			return

		with self.notes_path.open("r", newline="", encoding="utf-8") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				desi_id = normalize_desi_id(row.get("DESI_ID", ""))
				if not desi_id:
					continue
				self._notes[desi_id] = NotesRecord(
					note=row.get("note", ""),
					updated_at=row.get("updated_at", ""),
				)

	def get_note(self, desi_id: str) -> str:
		"""Retrieve note for a QSO.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Note text or empty string if not found.
		"""
		record = self._notes.get(normalize_desi_id(desi_id))
		return "" if record is None else record.note

	def save_note(self, desi_id: str, note: str) -> None:
		"""Save or update a note for a QSO.
		
		Args:
			desi_id: DESI identifier.
			note: Note text to save.
		"""
		desi_id = normalize_desi_id(desi_id)
		self._notes[desi_id] = NotesRecord(
			note=note,
			updated_at=datetime.utcnow().isoformat(timespec="seconds"),
		)
		self._flush()

	def _flush(self) -> None:
		"""Write all notes to disk atomically using temp file + swap."""
		self.notes_path.parent.mkdir(parents=True, exist_ok=True)
		# Use temporary file to prevent corruption on crash
		temp_path = self.notes_path.with_suffix(".tmp")

		with temp_path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=["DESI_ID", "note", "updated_at"])
			writer.writeheader()
			for desi_id in sorted(self._notes.keys()):
				record = self._notes[desi_id]
				writer.writerow(
					{
						"DESI_ID": desi_id,
						"note": record.note,
						"updated_at": record.updated_at,
					}
				)

		temp_path.replace(self.notes_path)


class QSODataStore:
	"""Centralized index and loader for QSO assets (cutouts, spectra, catalog data).
	
	Globs the data directory for JPEG cutouts and ASCII spectra, indexes them by
	DESI_ID, and loads catalog metadata from CSV. Only indexes DESI_IDs that have
	assets available to keep memory and startup time low.
	"""
	def __init__(self, package_dir: Path):
		"""Initialize data store by scanning and indexing data directory.
		
		Args:
			package_dir: Root of the gotoq package (parent of 'data' directory).
		"""
		data_dir = package_dir / "data"
		self.catalog_path = data_dir / "tables" / "desi_qsos.csv"
		self.notes_store = NotesStore(data_dir / "tables" / "qso_notes.csv")

		candidate_cutout_dirs = [
			data_dir / "desi_qso_cutouts",
			data_dir / "desi_qso_cutout",
		]
		self.cutout_dir = next((d for d in candidate_cutout_dirs if d.exists()), candidate_cutout_dirs[0])
		self.spectra_dir = data_dir / "spectra" / "ascii"

		self.cutouts_by_id: Dict[str, Path] = {}
		self.spectra_by_id: Dict[str, Path] = {}
		self.rows_by_id: Dict[str, Dict[str, str]] = {}
		self.ids: List[str] = []

		self._index_assets()
		self._load_catalog_rows()

	def _index_assets(self) -> None:
		"""Glob cutout images and spectra to build lookup maps."""
		if self.cutout_dir.exists():
			# Index JPEG cutout images by DESI_ID
			for file_path in self.cutout_dir.glob("*_cutout.jpeg"):
				desi_id = normalize_desi_id(file_path.name.replace("_cutout.jpeg", ""))
				self.cutouts_by_id[desi_id] = file_path

		if self.spectra_dir.exists():
			for file_path in self.spectra_dir.glob("*.dat"):
				desi_id = normalize_desi_id(file_path.stem)
				self.spectra_by_id[desi_id] = file_path

	def _load_catalog_rows(self) -> None:
		"""Load QSO catalog CSV and filter to only IDs with available assets."""
		if not self.catalog_path.exists():
			# If no catalog, use union of available cutouts and spectra
			self.ids = sorted(set(self.cutouts_by_id.keys()) | set(self.spectra_by_id.keys()))
			return

		with self.catalog_path.open("r", newline="", encoding="utf-8") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				desi_id = normalize_desi_id(row.get("DESI_ID", row.get("targetid", "")))
				if not desi_id:
					continue

				# Keep rows relevant to available files so startup and navigation remain responsive.
				if desi_id in self.cutouts_by_id or desi_id in self.spectra_by_id:
					self.rows_by_id[desi_id] = row

		combined = set(self.rows_by_id.keys()) | set(self.cutouts_by_id.keys()) | set(self.spectra_by_id.keys())
		self.ids = sorted(combined, key=lambda value: (len(value), value))

	def get_row(self, desi_id: str) -> Dict[str, str]:
		"""Retrieve catalog row for a QSO.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Dict of catalog columns or empty dict if not found.
		"""
		return self.rows_by_id.get(normalize_desi_id(desi_id), {})

	def get_redshift(self, desi_id: str) -> Optional[float]:
		"""Retrieve redshift for a QSO from catalog.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Redshift float or None if not found or invalid.
		"""
		row = self.get_row(desi_id)
		# Try DESI_z column first, fall back to z
		value = row.get("DESI_z", row.get("z", ""))
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def get_cutout_path(self, desi_id: str) -> Optional[Path]:
		"""Look up JPEG cutout file path for a QSO.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Path to cutout or None if not found.
		"""
		return self.cutouts_by_id.get(normalize_desi_id(desi_id))

	def get_spectrum_path(self, desi_id: str) -> Optional[Path]:
		"""Look up spectrum ASCII file path for a QSO.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Path to spectrum or None if not found.
		"""
		return self.spectra_by_id.get(normalize_desi_id(desi_id))

	def load_spectrum(self, desi_id: str) -> Optional[Dict[str, np.ndarray]]:
		"""Load and parse spectrum file for a QSO.
		
		Expects 7+ columns: wavelength, flux, error, ?, ?, flux_clip, error_clip.
		
		Args:
			desi_id: DESI identifier.
			
		Returns:
			Dict with 'wavelength', 'flux', 'error', 'flux_clip', 'error_clip' arrays,
			or None if file not found or invalid.
		"""
		spectrum_path = self.get_spectrum_path(desi_id)
		if spectrum_path is None or not spectrum_path.exists():
			return None

		array = np.loadtxt(spectrum_path, comments="#")
		if array.ndim == 1:
			array = array.reshape(1, -1)

		if array.shape[1] < 7:
			return None

		return {
			"wavelength": array[:, 0],
			"flux": array[:, 1],
			"error": array[:, 2],
			"flux_clip": array[:, 5],
			"error_clip": array[:, 6],
		}


class ImageLabel(QtWidgets.QLabel):
	"""Custom QLabel that displays and scales images to fit available space.
	
	Maintains a source pixmap and recomputes scaled version on resize events
	(respects aspect ratio, smooth transformation).
	"""
	def __init__(self):
		"""Initialize image label with default styling."""
		super().__init__()
		self._source_pixmap: Optional[QtGui.QPixmap] = None
		self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.setMinimumWidth(360)
		self.setStyleSheet("border: 1px solid #C7C7C7; background: #F9F9F9;")

	def set_source_pixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:
		"""Set or update the source image.
		
		Args:
			pixmap: Image to display, or None to clear.
		"""
		self._source_pixmap = pixmap
		self._refresh()

	def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
		"""Recompute scaled image on widget resize."""
		super().resizeEvent(event)
		self._refresh()

	def _refresh(self) -> None:
		"""Refresh display by scaling pixmap to current widget size."""
		if self._source_pixmap is None or self._source_pixmap.isNull():
			self.setText("No cutout available")
			self.setPixmap(QtGui.QPixmap())
			return

		scaled = self._source_pixmap.scaled(
			self.size(),
			QtCore.Qt.AspectRatioMode.KeepAspectRatio,
			QtCore.Qt.TransformationMode.SmoothTransformation,
		)
		self.setText("")
		self.setPixmap(scaled)


class CustomViewBox(pg.ViewBox):
	"""Custom pyqtgraph ViewBox with controlled scroll behavior.
	
	Scroll wheel zooms X-axis only (default).
	Ctrl+scroll zooms Y-axis only.
	This allows independent exploration of wavelength and flux dimensions.
	"""

	def wheelEvent(self, ev, axis=None):
		"""Handle mouse wheel scrolling with axis control via modifiers.
		
		Plain scroll zooms X (wavelength).
		Ctrl+scroll zooms Y (flux).
		"""
		if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
			super().wheelEvent(ev, axis=1)  # Y zoom
		else:
			super().wheelEvent(ev, axis=0)  # X zoom
		ev.accept()


class QSOViewer(QtWidgets.QMainWindow):
	"""Main application window for QSO data exploration.

	Features:
	  - Navigate catalog with keyboard shortcuts or combo box
	  - Display cutout images and spectra with synchronized zoom
	  - Overlay spectral lines (emission/absorption) with redshift adjustment
	  - Toggle line categories and major-only mode
	  - Write and persist per-object notes
	  - Fine and coarse redshift stepping via slider, text entry, or keyboard
	  - Auto-coalesce rapid redshift updates to prevent UI lag

	Keyboard shortcuts:
	  - Left/Right Arrow: Navigate catalog
	  - Alt+Left/Right: Large redshift steps (100x multiplier)
	  - Shift+Alt+Left/Right: Very large redshift steps (1000x)
	  - q: Quit
	  - F1: Show help
	  - z: Reset redshift
	  - e/a/m: Toggle emission/absorption/major lines
	  - c: Focus notes input
	  - Esc: Clear text input focus
	"""
	def __init__(self, store: QSODataStore):
		"""Initialize the QSO viewer.

		Args:
			store: QSODataStore instance with indexed data.
		"""
		super().__init__()
		self.store = store
		self.current_index = 0
		self.current_id: Optional[str] = None

		self._syncing_ranges = False
		self._suspend_sync = False
		self._loading_note = False
		# Overlay rendering: cache stores (marker, rest_wavelength) tuples for in-place updates
		self._overlay_cache: Dict[str, List[Tuple[pg.InfiniteLine, float]]] = {
			"raw": [],
			"clip": [],
		}
		# Default view ranges used by 'Reset View' button
		self._default_x_range: Optional[Tuple[float, float]] = None
		self._default_raw_y_range: Optional[Tuple[float, float]] = None
		self._default_clip_y_range: Optional[Tuple[float, float]] = None
		# Redshift control state
		self.current_redshift: Optional[float] = None
		self.default_redshift: Optional[float] = None
		self.redshift_min: float = -0.001
		self.redshift_max: Optional[float] = None
		self._redshift_controls_updating = False
		self._fine_scale = 100000.0  # Slider tick resolution for fine control
		# Keyboard step multipliers for Alt+arrow stepping
		self._keyboard_redshift_step_multiplier = 100  # Large steps
		self._keyboard_redshift_coarse_multiplier = 1000  # Very large steps
		# Load spectral line definitions from data files
		lines_dir = Path(__file__).resolve().parent / "data" / "lines"
		self.emission_lines = self._load_line_file(lines_dir / "emission_lines.txt")
		self.absorption_lines = self._load_line_file(lines_dir / "absorption_lines.txt")
		# Legend items for spectrum annotation
		self.raw_legend: Optional[pg.LegendItem] = None
		self.clip_legend: Optional[pg.LegendItem] = None
		self._legend_emission_item: Optional[pg.PlotDataItem] = None
		self._legend_absorption_item: Optional[pg.PlotDataItem] = None
		self.raw_flux_item: Optional[pg.PlotDataItem] = None
		self.raw_error_item: Optional[pg.PlotDataItem] = None
		self.clip_flux_item: Optional[pg.PlotDataItem] = None
		self.clip_error_item: Optional[pg.PlotDataItem] = None

		self.note_save_timer = QtCore.QTimer(self)
		self.note_save_timer.setSingleShot(True)
		self.note_save_timer.timeout.connect(self._flush_current_note)
		self._overlay_update_timer = QtCore.QTimer(self)
		self._overlay_update_timer.setSingleShot(True)
		self._overlay_update_timer.setInterval(16)
		self._overlay_update_timer.timeout.connect(self._apply_redshift_overlay_update)
		self._shortcuts: List[QtGui.QShortcut] = []

		self._build_ui()
		self._connect_signals()
		QtWidgets.QApplication.instance().installEventFilter(self)
		self._setup_shortcuts()
		self._populate_ids()

		self.setWindowTitle("GOTOQ QSO Visualizer")
		self.resize(1600, 980)

		if self.store.ids:
			self.set_current_object(self.store.ids[0])
		else:
			self.statusBar().showMessage("No matching DESI objects found in data folders.")

	def _build_ui(self) -> None:
		"""Construct the complete widget hierarchy and layout."""
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)

		root_layout = QtWidgets.QVBoxLayout(central)

		control_layout = QtWidgets.QHBoxLayout()
		self.prev_button = QtWidgets.QPushButton("Previous")
		self.next_button = QtWidgets.QPushButton("Next")
		self.id_combo = QtWidgets.QComboBox()
		self.id_combo.setMinimumWidth(280)
		self.show_emission_button = QtWidgets.QPushButton("Emission")
		self.show_emission_button.setCheckable(True)
		self.show_emission_button.setChecked(True)
		self.show_absorption_button = QtWidgets.QPushButton("Absorption")
		self.show_absorption_button.setCheckable(True)
		self.show_absorption_button.setChecked(True)
		self.show_major_only_button = QtWidgets.QPushButton("Major Only")
		self.show_major_only_button.setCheckable(True)
		self.show_major_only_button.setChecked(False)
		self.reset_view_button = QtWidgets.QPushButton("Reset View")

		control_layout.addWidget(QtWidgets.QLabel("DESI_ID:"))
		control_layout.addWidget(self.id_combo)
		control_layout.addWidget(self.prev_button)
		control_layout.addWidget(self.next_button)
		control_layout.addWidget(self.show_emission_button)
		control_layout.addWidget(self.show_absorption_button)
		control_layout.addWidget(self.show_major_only_button)
		control_layout.addWidget(self.reset_view_button)
		control_layout.addStretch(1)
		root_layout.addLayout(control_layout)

		z_layout = QtWidgets.QGridLayout()
		z_layout.addWidget(QtWidgets.QLabel("z"), 0, 0)
		self.z_fine_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.z_fine_slider.setEnabled(False)
		z_layout.addWidget(self.z_fine_slider, 0, 1)

		self.z_value_edit = QtWidgets.QLineEdit()
		self.z_value_edit.setPlaceholderText("z")
		self.z_value_edit.setFixedWidth(120)
		self.z_value_edit.setEnabled(False)
		z_layout.addWidget(self.z_value_edit, 0, 2)

		self.reset_redshift_button = QtWidgets.QPushButton("Reset z")
		self.reset_redshift_button.setEnabled(False)
		z_layout.addWidget(self.reset_redshift_button, 1, 2)

		root_layout.addLayout(z_layout)

		body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		root_layout.addWidget(body_splitter, 1)

		self.image_label = ImageLabel()
		body_splitter.addWidget(self.image_label)

		right_panel = QtWidgets.QWidget()
		right_layout = QtWidgets.QVBoxLayout(right_panel)
		body_splitter.addWidget(right_panel)

		plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
		right_layout.addWidget(plot_splitter, 4)

		pg.setConfigOptions(antialias=False)
		self.raw_plot = pg.PlotWidget(viewBox=CustomViewBox(), background="w")
		self.clip_plot = pg.PlotWidget(viewBox=CustomViewBox(), background="w")
		plot_splitter.addWidget(self.raw_plot)
		plot_splitter.addWidget(self.clip_plot)

		self.raw_plot.setLabel("bottom", "Wavelength")
		self.raw_plot.setLabel("left", "Flux")
		self.clip_plot.setLabel("bottom", "Wavelength")
		self.clip_plot.setLabel("left", "Flux clip")
		self.raw_plot.showGrid(x=True, y=True, alpha=0.2)
		self.clip_plot.showGrid(x=True, y=True, alpha=0.2)
		self.raw_plot.enableAutoRange(x=False, y=False)
		self.clip_plot.enableAutoRange(x=False, y=False)
		self._init_spectrum_items()
		self._init_legends()

		lower_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		right_layout.addWidget(lower_splitter, 1)

		self.info_box = QtWidgets.QPlainTextEdit()
		self.info_box.setReadOnly(True)
		self.info_box.setPlaceholderText("Object metadata")
		lower_splitter.addWidget(self.info_box)

		notes_container = QtWidgets.QWidget()
		notes_layout = QtWidgets.QVBoxLayout(notes_container)
		notes_layout.setContentsMargins(0, 0, 0, 0)
		notes_layout.addWidget(QtWidgets.QLabel("Notes"))

		self.notes_editor = QtWidgets.QPlainTextEdit()
		self.notes_editor.setPlaceholderText("Write notes for this DESI_ID")
		notes_layout.addWidget(self.notes_editor)
		lower_splitter.addWidget(notes_container)

		body_splitter.setSizes([560, 1040])
		plot_splitter.setSizes([400, 400])
		lower_splitter.setSizes([520, 520])

	def _connect_signals(self) -> None:
		"""Connect UI widget signals to their handler slots."""
		self.prev_button.clicked.connect(self._go_previous)
		self.next_button.clicked.connect(self._go_next)
		self.id_combo.currentTextChanged.connect(self._on_combo_changed)
		self.show_emission_button.toggled.connect(self._refresh_line_overlays)
		self.show_absorption_button.toggled.connect(self._refresh_line_overlays)
		self.show_major_only_button.toggled.connect(self._refresh_line_overlays)
		self.reset_view_button.clicked.connect(self._reset_view)
		self.z_fine_slider.valueChanged.connect(self._on_fine_redshift_changed)
		self.z_value_edit.returnPressed.connect(self._on_redshift_text_submitted)
		self.reset_redshift_button.clicked.connect(self._reset_redshift)
		self.notes_editor.textChanged.connect(self._on_note_text_changed)

		self.raw_plot.getViewBox().sigRangeChanged.connect(self._sync_from_raw)
		self.clip_plot.getViewBox().sigRangeChanged.connect(self._sync_from_clip)

	def _setup_shortcuts(self) -> None:
		"""Register keyboard shortcuts and their handlers.
		
		Supports modifier combinations (Alt, Shift+Alt) for flexible navigation and control.
		All shortcuts check whether a text input is focused to avoid triggering while typing."""
		self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self._on_shortcut_previous)
		self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self._on_shortcut_next)
		self._add_shortcut(QtGui.QKeySequence("q"), self._on_shortcut_quit)
		self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self._on_shortcut_clear_text_focus)
		self._add_shortcut(QtGui.QKeySequence("F1"), self._on_shortcut_show_help)
		self._add_shortcut(QtGui.QKeySequence("z"), self._on_shortcut_reset_redshift)
		self._add_shortcut(QtGui.QKeySequence("e"), self._on_shortcut_toggle_emission)
		self._add_shortcut(QtGui.QKeySequence("a"), self._on_shortcut_toggle_absorption)
		self._add_shortcut(QtGui.QKeySequence("m"), self._on_shortcut_toggle_major)
		self._add_shortcut(QtGui.QKeySequence("c"), self._on_shortcut_focus_notes)
		self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Modifier.ALT | QtCore.Qt.Key.Key_Left), self._on_shortcut_redshift_left)
		self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Modifier.ALT | QtCore.Qt.Key.Key_Right), self._on_shortcut_redshift_right)
		self._add_shortcut(
			QtGui.QKeySequence(QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.ALT | QtCore.Qt.Key.Key_Left),
			self._on_shortcut_redshift_left_coarse,
		)
		self._add_shortcut(
			QtGui.QKeySequence(QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.ALT | QtCore.Qt.Key.Key_Right),
			self._on_shortcut_redshift_right_coarse,
		)

	def _add_shortcut(self, key_sequence: QtGui.QKeySequence, handler) -> None:
		"""Register a keyboard shortcut that triggers a handler callback.
		
		Args:
			key_sequence: QtGui.QKeySequence representing the key combination.
			handler: Callable (slot) to invoke when shortcut is triggered.
		"""
		shortcut = QtGui.QShortcut(key_sequence, self)
		shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
		shortcut.activated.connect(handler)
		self._shortcuts.append(shortcut)

	def _widget_is_or_inside(self, candidate: Optional[QtWidgets.QWidget], target: QtWidgets.QWidget) -> bool:
		"""Check if candidate widget is target or a child of target.
		
		Walks up the widget parent hierarchy to determine containment.
		
		Args:
			candidate: Widget to check (can be None).
			target: Widget to test against.
			
		Returns:
			True if candidate is target or descended from target.
		"""
		widget = candidate
		while widget is not None:
			if widget is target:
				return True
			widget = widget.parentWidget()
		return False

	def _is_text_input_focused(self) -> bool:
		"""Check if a text input widget (notes or redshift) currently has focus.
		
		Used by shortcut handlers to prevent keyboard actions while typing.
		
		Returns:
			True if notes editor or redshift text field has focus.
		"""
		focused = self.focusWidget()
		if focused is None:
			return False
		return (
			self._widget_is_or_inside(focused, self.notes_editor)
			or self._widget_is_or_inside(focused, self.z_value_edit)
		)

	def _clear_text_input_focus(self) -> None:
		"""Remove focus from active text inputs and return it to main window.
		
		Called by Esc key and external click event filter to dismiss text editing mode."""
		if not self._is_text_input_focused():
			return
		focused = self.focusWidget()
		if focused is not None:
			focused.clearFocus()
		self.centralWidget().setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

	def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
		"""Global event filter to clear text focus on external clicks.
		
		If a text input is focused and user clicks outside it, focus is cleared.
		This allows clicks on any region (plots, empty space, buttons) to dismiss text editing.
		
		Args:
			obj: Object that generated the event.
			event: Event to filter.
			
		Returns:
			False to allow event to propagate normally.
		"""
		if event.type() == QtCore.QEvent.Type.MouseButtonPress and self._is_text_input_focused():
			clicked_widget = obj if isinstance(obj, QtWidgets.QWidget) else None
			if clicked_widget is None or (
				not self._widget_is_or_inside(clicked_widget, self.notes_editor)
				and not self._widget_is_or_inside(clicked_widget, self.z_value_edit)
			):
				self._clear_text_input_focus()
		return super().eventFilter(obj, event)

	def _on_shortcut_previous(self) -> None:
		"""Keyboard shortcut handler: navigate to previous QSO (Left arrow)."""
		if self._is_text_input_focused():
			return
		self._go_previous()

	def _on_shortcut_next(self) -> None:
		"""Keyboard shortcut handler: navigate to next QSO (Right arrow)."""
		if self._is_text_input_focused():
			return
		self._go_next()

	def _on_shortcut_quit(self) -> None:
		"""Keyboard shortcut handler: quit application (q key)."""
		if self._is_text_input_focused():
			return
		self.close()

	def _on_shortcut_clear_text_focus(self) -> None:
		"""Keyboard shortcut handler: clear text input focus (Esc key)."""
		self._clear_text_input_focus()

	def _on_shortcut_show_help(self) -> None:
		"""Keyboard shortcut handler: display keyboard shortcuts help dialog (F1)."""
		if self._is_text_input_focused():
			return
		message = (
			"Keyboard shortcuts\n\n"
			"Left Arrow: Previous object\n"
			"Right Arrow: Next object\n"
			"q: Quit\n"
			"F1: Show this help\n"
			"z: Reset redshift\n"
			"e: Toggle emission lines\n"
			"a: Toggle absorption lines\n"
			"m: Toggle major-only lines\n"
			"c: Focus notes input\n"
			"Alt+Left: Decrease redshift (large step)\n"
			"Alt+Right: Increase redshift (large step)\n"
			"Shift+Alt+Left: Decrease redshift (very large step)\n"
			"Shift+Alt+Right: Increase redshift (very large step)\n\n"
			"Shortcuts are disabled while typing in notes or redshift text fields."
		)
		QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", message)

	def _on_shortcut_reset_redshift(self) -> None:
		"""Keyboard shortcut handler: reset redshift to catalog value (z key)."""
		if self._is_text_input_focused():
			return
		self._reset_redshift()

	def _on_shortcut_toggle_emission(self) -> None:
		"""Keyboard shortcut handler: toggle emission line visibility (e key)."""
		if self._is_text_input_focused():
			return
		self.show_emission_button.toggle()

	def _on_shortcut_toggle_absorption(self) -> None:
		"""Keyboard shortcut handler: toggle absorption line visibility (a key)."""
		if self._is_text_input_focused():
			return
		self.show_absorption_button.toggle()

	def _on_shortcut_toggle_major(self) -> None:
		"""Keyboard shortcut handler: toggle major-only line filter (m key)."""
		if self._is_text_input_focused():
			return
		self.show_major_only_button.toggle()

	def _on_shortcut_focus_notes(self) -> None:
		"""Keyboard shortcut handler: focus notes input and move cursor to end (c key)."""
		if self._is_text_input_focused():
			return
		self.notes_editor.setFocus()
		cursor = self.notes_editor.textCursor()
		cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
		self.notes_editor.setTextCursor(cursor)

	def _step_redshift_slider(self, direction: int, multiplier: int = 1) -> None:
		"""Adjust redshift slider by stepping in the specified direction.
		
		Args:
			direction: +1 to increase, -1 to decrease.
			multiplier: Scaling factor for step size (used by keyboard shortcuts).
		"""
		if self._is_text_input_focused() or not self.z_fine_slider.isEnabled():
			return
		step = max(self.z_fine_slider.singleStep() * max(multiplier, 1), 1)
		new_value = self.z_fine_slider.value() + direction * step
		new_value = min(max(new_value, self.z_fine_slider.minimum()), self.z_fine_slider.maximum())
		self.z_fine_slider.setValue(new_value)

	def _on_shortcut_redshift_left(self) -> None:
		"""Keyboard shortcut handler: decrease redshift with large step (Alt+Left)."""
		self._step_redshift_slider(-1, self._keyboard_redshift_step_multiplier)

	def _on_shortcut_redshift_right(self) -> None:
		"""Keyboard shortcut handler: increase redshift with large step (Alt+Right)."""
		self._step_redshift_slider(1, self._keyboard_redshift_step_multiplier)

	def _on_shortcut_redshift_left_coarse(self) -> None:
		"""Keyboard shortcut handler: decrease redshift with very large step (Shift+Alt+Left)."""
		self._step_redshift_slider(-1, self._keyboard_redshift_coarse_multiplier)

	def _on_shortcut_redshift_right_coarse(self) -> None:
		"""Keyboard shortcut handler: increase redshift with very large step (Shift+Alt+Right)."""
		self._step_redshift_slider(1, self._keyboard_redshift_coarse_multiplier)

	def _populate_ids(self) -> None:
		"""Fill the DESI_ID combo box with all available objects from the store."""
		self.id_combo.blockSignals(True)
		self.id_combo.clear()
		self.id_combo.addItems(self.store.ids)
		self.id_combo.blockSignals(False)

	def _on_combo_changed(self, desi_id: str) -> None:
		"""Handle combo box selection change: load selected QSO."""
		if not desi_id or desi_id == self.current_id:
			return
		self.set_current_object(desi_id)

	def _go_previous(self) -> None:
		"""Navigate to the previous QSO in catalog order."""
		if self.current_index <= 0:
			return
		self.set_current_object(self.store.ids[self.current_index - 1])

	def _go_next(self) -> None:
		"""Navigate to the next QSO in catalog order."""
		if self.current_index >= len(self.store.ids) - 1:
			return
		self.set_current_object(self.store.ids[self.current_index + 1])

	def _configure_redshift_controls(self, desi_id: str) -> None:
		"""Set up redshift slider, text field, and limits for a given QSO.
		
		If no redshift is available, disables controls and shows empty text field.
		
		Args:
			desi_id: DESI identifier to configure for.
		"""
		desi_z = self.store.get_redshift(desi_id)
		if desi_z is None:
			self.current_redshift = None
			self.default_redshift = None
			self.redshift_max = None
			self.z_fine_slider.setEnabled(False)
			self.z_value_edit.setEnabled(False)
			self.reset_redshift_button.setEnabled(False)
			self.z_value_edit.setText("")
			return

		self.redshift_max = max(desi_z, self.redshift_min)
		self.default_redshift = min(max(desi_z, self.redshift_min), self.redshift_max)

		fine_min = int(round(self.redshift_min * self._fine_scale))
		fine_max = int(round(self.redshift_max * self._fine_scale))
		if fine_max < fine_min:
			fine_max = fine_min

		self.z_fine_slider.setRange(fine_min, fine_max)
		self.z_fine_slider.setSingleStep(1)
		self.z_fine_slider.setPageStep(max((fine_max - fine_min) // 20, 1))

		self.z_fine_slider.setEnabled(True)
		self.z_value_edit.setEnabled(True)
		self.reset_redshift_button.setEnabled(True)
		self._set_redshift(self.default_redshift, refresh_overlays=False)

	def _clamp_redshift(self, redshift: float) -> float:
		"""Clamp redshift to valid range [redshift_min, redshift_max].
		
		Args:
			redshift: Proposed redshift value.
			
		Returns:
			Clamped redshift within allowed bounds.
		"""
		if self.redshift_max is None:
			return redshift
		return min(max(redshift, self.redshift_min), self.redshift_max)

	def _set_redshift(self, redshift: float, refresh_overlays: bool = True) -> None:
		"""Update current redshift and synchronize slider/text controls.
		
		Clamps redshift to valid range and triggers overlay refresh if requested.
		Prevents circular updates by using an _redshift_controls_updating flag.
		
		Args:
			redshift: New redshift value.
			refresh_overlays: If True, schedule overlay repositioning (default True).
		"""
		z = self._clamp_redshift(redshift)
		self.current_redshift = z

		self._redshift_controls_updating = True
		try:
			z_text = f"{z:.6f}"
			if self.z_value_edit.text() != z_text:
				self.z_value_edit.setText(z_text)
			self.z_fine_slider.setValue(int(round(z * self._fine_scale)))
		finally:
			self._redshift_controls_updating = False

		if refresh_overlays:
			self._schedule_redshift_overlay_update()

	def _schedule_redshift_overlay_update(self) -> None:
		"""Schedule a coalesced overlay update via timer.
		
		Prevents excessive UI redraws during rapid redshift changes (e.g., keyboard stepping).
		Multiple calls within the timer interval (16ms) are collapsed into a single update."""
		# Coalesce rapid slider/key updates into a single UI refresh tick.
		self._overlay_update_timer.start()

	def _apply_redshift_overlay_update(self) -> None:
		"""Apply pending overlay repositioning (called by timer timeout)."""
		if self.current_id is None:
			return
		if self._overlay_cache["raw"] or self._overlay_cache["clip"]:
			# Update line positions in-place without clearing/rebuilding
			self._update_line_overlay_positions()
		else:
			# Fallback for first update after startup/object switch
			self._refresh_line_overlays()

	def _on_fine_redshift_changed(self, value: int) -> None:
		"""Handle redshift slider value change.
		
		Updates current_redshift and synchronizes text field without triggering loop.
		"""
		if self._redshift_controls_updating:
			return
		self._set_redshift(value / self._fine_scale)

	def _on_redshift_text_submitted(self) -> None:
		"""Handle user pressing Enter in redshift text field.
		
		Parses text as float; reverts to current value if invalid.
		"""
		text = self.z_value_edit.text().strip()
		if not text:
			return
		try:
			z = float(text)
		except ValueError:
			# Revert to the currently active value if parsing fails.
			if self.current_redshift is not None:
				self.z_value_edit.setText(f"{self.current_redshift:.6f}")
			return
		self._set_redshift(z)

	def _reset_redshift(self) -> None:
		"""Reset redshift to the catalog default value for current QSO."""
		if self.default_redshift is None:
			return
		self._set_redshift(self.default_redshift)

	def set_current_object(self, desi_id: str) -> None:
		"""Load and display a QSO by DESI_ID.
		
		Updates all UI components (combo box, buttons, redshift controls, cutout, spectrum, notes).
		Saves any previously active note before switching objects.
		
		Args:
			desi_id: DESI identifier to load.
		"""
		if not desi_id:
			return

		if self.current_id is not None:
			self._flush_current_note()

		normalized = normalize_desi_id(desi_id)
		if normalized not in self.store.ids:
			return

		self.current_id = normalized
		self.current_index = self.store.ids.index(normalized)

		self.id_combo.blockSignals(True)
		self.id_combo.setCurrentText(normalized)
		self.id_combo.blockSignals(False)

		self.prev_button.setEnabled(self.current_index > 0)
		self.next_button.setEnabled(self.current_index < len(self.store.ids) - 1)
		self._configure_redshift_controls(normalized)

		self._render_cutout(normalized)
		self._render_metadata(normalized)
		self._render_spectra(normalized)
		self._load_note(normalized)

	def _render_cutout(self, desi_id: str) -> None:
		"""Load and display cutout image for a QSO.
		
		Args:
			desi_id: DESI identifier.
		"""
		cutout_path = self.store.get_cutout_path(desi_id)
		if cutout_path is None:
			self.image_label.set_source_pixmap(None)
			self.statusBar().showMessage(f"Missing cutout image for {desi_id}")
			return

		pixmap = QtGui.QPixmap(str(cutout_path))
		self.image_label.set_source_pixmap(pixmap)

	def _render_metadata(self, desi_id: str) -> None:
		"""Display catalog rows as key:value pairs in info box.
		
		Args:
			desi_id: DESI identifier.
		"""
		row = self.store.get_row(desi_id)
		if not row:
			self.info_box.setPlainText(f"DESI_ID: {desi_id}\nNo catalog row available")
			return

		lines = [f"{key}: {value}" for key, value in row.items()]
		self.info_box.setPlainText("\n".join(lines))

	def _render_spectra(self, desi_id: str) -> None:
		"""Load spectrum and update plot displays with line overlays.
		
		Args:
			desi_id: DESI identifier.
		"""
		spectrum = self.store.load_spectrum(desi_id)
		self._suspend_sync = True
		self._clear_line_overlays()

		if spectrum is None:
			self._clear_spectrum_items()
			self._suspend_sync = False
			self.statusBar().showMessage(f"Missing spectrum data for {desi_id}")
			return

		wavelength = spectrum["wavelength"]
		self._set_spectrum_item_data(self.raw_flux_item, wavelength, spectrum["flux"])
		self._set_spectrum_item_data(self.raw_error_item, wavelength, spectrum["error"])
		self._set_spectrum_item_data(self.clip_flux_item, wavelength, spectrum["flux_clip"])
		self._set_spectrum_item_data(self.clip_error_item, wavelength, self._shift_error_clip(spectrum))
		self._set_initial_plot_ranges(spectrum)

		self._refresh_line_overlays()
		self._suspend_sync = False

	def _set_initial_plot_ranges(self, spectrum: Dict[str, np.ndarray]) -> None:
		"""Compute and apply appropriate X and Y ranges for both plots.
		
		X range spans full wavelength extent. Y ranges computed separately for raw and clip plots.
		Enforces y-axis floor at 0 for initial/reset ranges to prevent negative fluxes dominating view.
		Stores defaults for use by reset-view button.
		
		Args:
			spectrum: Dict with 'wavelength', 'flux', 'error', 'flux_clip', 'error_clip' arrays.
		"""
		wavelength = spectrum["wavelength"]
		finite_wavelength = wavelength[np.isfinite(wavelength)]
		if finite_wavelength.size == 0:
			return

		x_min = float(np.min(finite_wavelength))
		x_max = float(np.max(finite_wavelength))
		x_span = x_max - x_min if x_max != x_min else 1.0
		x_pad = 0.02 * x_span
		x_range = (x_min - x_pad, x_max + x_pad)

		# Raw y: cover flux + error together
		raw_y_values = []
		for key in ("flux", "error"):
			v = spectrum[key]
			finite_v = v[np.isfinite(v)]
			if finite_v.size:
				raw_y_values.append(finite_v)
		if raw_y_values:
			raw_combined = np.concatenate(raw_y_values)
			raw_y_min = float(np.min(raw_combined))
			raw_y_max = float(np.max(raw_combined))
			raw_y_span = raw_y_max - raw_y_min if raw_y_max != raw_y_min else max(abs(raw_y_min) * 0.1, 1.0)
			raw_y_range: Tuple[float, float] = (raw_y_min - 0.05 * raw_y_span, raw_y_max + 0.05 * raw_y_span)
		else:
			raw_y_range = (-1.0, 1.0)
			raw_y_span = 2.0

		# Clip y: median ± 5.5*std so the normalised continuum (~1) is well-centred
		finite_clip = spectrum["flux_clip"][np.isfinite(spectrum["flux_clip"])]
		if finite_clip.size > 1:
			clip_median = float(np.median(finite_clip))
			clip_std = float(np.std(finite_clip))
			clip_half = max(5.5 * clip_std, 0.1)
			clip_y_range: Tuple[float, float] = (clip_median - clip_half, clip_median + clip_half)
			clip_y_span = 2.0 * clip_half
		else:
			clip_y_range = (-0.5, 2.5)
			clip_y_span = 3.0

		# Enforce non-negative y-axis floor for default/reset ranges.
		raw_y_floor = max(0.0, raw_y_range[0])
		raw_y_ceil = max(raw_y_range[1], raw_y_floor + max(0.05 * raw_y_span, 1e-3))
		raw_y_range = (raw_y_floor, raw_y_ceil)

		clip_y_floor = max(0.0, clip_y_range[0])
		clip_y_ceil = max(clip_y_range[1], clip_y_floor + max(0.05 * clip_y_span, 1e-3))
		clip_y_range = (clip_y_floor, clip_y_ceil)

		# Persist defaults for Reset View
		self._default_x_range = x_range
		self._default_raw_y_range = raw_y_range
		self._default_clip_y_range = clip_y_range

		# X limits: max zoom-out bounded exactly by full data extent; pan bounded to data range
		for plot, y_range, y_span in (
			(self.raw_plot, raw_y_range, raw_y_span),
			(self.clip_plot, clip_y_range, clip_y_span),
		):
			plot.enableAutoRange(x=False, y=False)
			y_floor = max(0.0, y_range[0])
			y_ceil = max(y_range[1], y_floor + max(0.05 * y_span, 1e-3))
			plot.getViewBox().setLimits(
				xMin=x_range[0],
				xMax=x_range[1],
				yMin=0.0,
				yMax=max(y_ceil + y_span, y_ceil + 0.1),
				minXRange=max(x_span * 0.01, 10.0),
				maxXRange=x_range[1] - x_range[0],
			)
			plot.setRange(xRange=x_range, yRange=(y_floor, y_ceil), padding=0.0)

	def _init_spectrum_items(self) -> None:
		"""Create and configure persistent spectrum plot items (flux and error traces)."""
		black_pen = pg.mkPen(color=(0, 0, 0), width=2.0)
		red_dotted_pen = pg.mkPen(color=(200, 0, 0), width=1.6, style=_dot_line_style())

		self.raw_flux_item = self._make_trace_item(self.raw_plot, black_pen)
		self.raw_error_item = self._make_trace_item(self.raw_plot, red_dotted_pen)
		self.clip_flux_item = self._make_trace_item(self.clip_plot, black_pen)
		self.clip_error_item = self._make_trace_item(self.clip_plot, red_dotted_pen)

	def _init_legends(self) -> None:
		"""Initialize legends for both plots with flux, error, and line type indicators."""
		self.raw_legend = self.raw_plot.addLegend(offset=(-10, 10))
		self.clip_legend = self.clip_plot.addLegend(offset=(-10, 10))

		for legend, flux_item, error_item in (
			(self.raw_legend, self.raw_flux_item, self.raw_error_item),
			(self.clip_legend, self.clip_flux_item, self.clip_error_item),
		):
			if legend is None:
				continue
			if flux_item is not None:
				legend.addItem(flux_item, "Flux")
			if error_item is not None:
				legend.addItem(error_item, "Error")

		emission_pen = pg.mkPen(color=(0, 100, 220), width=2.5, style=_dot_line_style())
		absorption_pen = pg.mkPen(color=(0, 150, 0), width=2.5, style=_dot_line_style())
		self._legend_emission_item = pg.PlotDataItem([0, 1], [0, 0], pen=emission_pen)
		self._legend_absorption_item = pg.PlotDataItem([0, 1], [0, 0], pen=absorption_pen)

		if self.raw_legend is not None:
			self.raw_legend.addItem(self._legend_emission_item, "Emission Line")
			self.raw_legend.addItem(self._legend_absorption_item, "Absorption Line")
		if self.clip_legend is not None:
			self.clip_legend.addItem(self._legend_emission_item, "Emission Line")
			self.clip_legend.addItem(self._legend_absorption_item, "Absorption Line")

	def _make_trace_item(self, plot: pg.PlotWidget, pen: pg.mkPen) -> pg.PlotDataItem:
		"""Create a persistent plot item for spectral data (flux or error trace).
		
		Configures step mode, clipping, and downsampling for efficient rendering.
		
		Args:
			plot: Target PlotWidget to add item to.
			pen: Pen style for the trace.
			
		Returns:
			Configured PlotDataItem ready for data updates.
		"""
		item = pg.PlotDataItem(pen=pen, antialias=False, stepMode=True)
		item.setClipToView(True)
		item.setDownsampling(auto=True, method="peak")
		if hasattr(item, "setSkipFiniteCheck"):
			item.setSkipFiniteCheck(True)
		plot.addItem(item)
		return item

	def _set_spectrum_item_data(self, item: Optional[pg.PlotDataItem], wavelength: np.ndarray, values: np.ndarray) -> None:
		"""Update spectrum plot item with new wavelength and flux data.
		
		Handles stepMode adjustment (N+1 bin edges for N y-values).
		Filters out non-finite values to prevent NaN/Inf propagation.
		
		Args:
			item: PlotDataItem to update.
			wavelength: Wavelength array.
			values: Flux or error array.
		"""
		if item is None:
			return
		finite = np.isfinite(wavelength) & np.isfinite(values)
		if not np.any(finite):
			item.setData([], [])
			return

		w = np.ascontiguousarray(wavelength[finite], dtype=np.float64)
		y_data = np.ascontiguousarray(values[finite], dtype=np.float32)
		# stepMode=True requires N+1 bin edges for N y values.
		# Compute edges as midpoints between adjacent centres, extrapolating the ends.
		edges = np.empty(len(w) + 1, dtype=np.float64)
		edges[1:-1] = 0.5 * (w[:-1] + w[1:])
		edges[0] = w[0] - 0.5 * (w[1] - w[0]) if len(w) > 1 else w[0] - 0.5
		edges[-1] = w[-1] + 0.5 * (w[-1] - w[-2]) if len(w) > 1 else w[-1] + 0.5
		x_data = np.ascontiguousarray(edges, dtype=np.float32)
		item.setData(x_data, y_data)

	def _shift_error_clip(self, spectrum: Dict[str, np.ndarray]) -> np.ndarray:
		"""Return error_clip shifted to desired vertical position (currently median = 1).
		
		Args:
			spectrum: Dict with spectrum arrays.
			
		Returns:
			Shifted error_clip array.
		"""
		flux_clip = spectrum["flux_clip"]
		error_clip = spectrum["error_clip"]
		finite_flux = flux_clip[np.isfinite(flux_clip)]
		finite_err = error_clip[np.isfinite(error_clip)]
		if finite_flux.size > 1 and finite_err.size > 1:
			target = 1.0 # - 3.0 * float(np.std(finite_flux))
			offset = target - float(np.median(finite_err))
			return error_clip + offset
		return error_clip

	def _clear_spectrum_items(self) -> None:
		"""Clear data from all spectrum plot items (does not remove items from plots)."""
		for item in (self.raw_flux_item, self.raw_error_item, self.clip_flux_item, self.clip_error_item):
			if item is not None:
				item.setData([], [])

	def _load_line_file(self, file_path: Path) -> List[SpectralLine]:
		"""Load spectral line definitions from a CSV-like file.
		
		Expected format: name, ?, wavelength, ?, major_flag (comma-separated).
		Lines starting with # are treated as comments.
		
		Args:
			file_path: Path to line definition file.
			
		Returns:
			List of SpectralLine objects parsed from file.
		"""
		lines: List[SpectralLine] = []
		if not file_path.exists():
			return lines

		with file_path.open("r", encoding="utf-8") as handle:
			for raw_line in handle:
				line = raw_line.strip()
				if not line or line.startswith("#"):
					continue
				parts = [part.strip() for part in line.split(",")]
				if len(parts) < 5:
					continue
				try:
					wavelength = float(parts[2])
				except ValueError:
					continue
				major = parts[4].lower() == "true"
				lines.append(SpectralLine(name=parts[0], wavelength=wavelength, major=major))
		return lines

	def _line_categories_to_plot(self) -> List[Tuple[List[SpectralLine], pg.mkPen, Tuple[int, int, int]]]:
		"""Build list of line categories to plot based on current toggle states and major-only filter.
		
		Returns:
			List of (lines, pen, label_color) tuples ready to plot.
		"""
		major_only = self.show_major_only_button.isChecked()
		categories: List[Tuple[List[SpectralLine], pg.mkPen, Tuple[int, int, int]]] = []
		if self.show_emission_button.isChecked():
			emission = self.emission_lines
			if major_only:
				emission = [line for line in emission if line.major]
			categories.append((
				emission,
				pg.mkPen(color=(0, 100, 220), width=2.5, style=_dot_line_style()),
				(0, 80, 200),
			))
		if self.show_absorption_button.isChecked():
			absorption = self.absorption_lines
			if major_only:
				absorption = [line for line in absorption if line.major]
			categories.append((
				absorption,
				pg.mkPen(color=(0, 150, 0), width=2.5, style=_dot_line_style()),
				(0, 120, 0),
			))
		return categories

	def _refresh_line_overlays(self) -> None:
		"""Rebuild all line overlays based on current redshift and toggle states.
		
		Called when toggling line categories or loading a new spectrum.
		Creates InfiniteLines with labels for each line in each plot.
		"""
		self._clear_line_overlays()
		if self.current_id is None:
			return

		redshift = self.current_redshift
		if redshift is None:
			redshift = self.store.get_redshift(self.current_id)
		if redshift is None:
			self.statusBar().showMessage(f"No DESI_z available for {self.current_id}")
			return

		categories = self._line_categories_to_plot()
		if not categories:
			return

		label_font = QtGui.QFont("", 11, QtGui.QFont.Weight.Bold)

		for lines, pen, label_color in categories:
			for line in lines:
				observed_wave = line.wavelength * (1.0 + redshift)
				for key, plot in (("raw", self.raw_plot), ("clip", self.clip_plot)):
					marker = pg.InfiniteLine(pos=observed_wave, angle=90, movable=False, pen=pen)
					label = pg.InfLineLabel(
						marker,
						text=line.name,
						position=0.95,
						color=label_color,
						fill=(255, 255, 255, 200),
					)
					label.textItem.setFont(label_font)
					plot.addItem(marker, ignoreBounds=True)
					self._overlay_cache[key].append((marker, line.wavelength))

	def _update_line_overlay_positions(self) -> None:
		"""Reposition existing line overlays to current redshift without clearing/rebuilding.
		
		Performance-critical for rapid redshift stepping: avoids expensive marker creation/deletion.
		Simply updates marker positions in-place using z * (1 + z_shift) formula.
		"""
		redshift = self.current_redshift
		if redshift is None and self.current_id is not None:
			redshift = self.store.get_redshift(self.current_id)
		if redshift is None:
			return

		for key in ("raw", "clip"):
			for marker, rest_wavelength in self._overlay_cache[key]:
				marker.setPos(rest_wavelength * (1.0 + redshift))

	def _clear_line_overlays(self) -> None:
		"""Remove all line overlay markers from both plots and clear cache."""
		for key, plot in (("raw", self.raw_plot), ("clip", self.clip_plot)):
			for marker, _rest in self._overlay_cache[key]:
				plot.removeItem(marker)
			self._overlay_cache[key] = []

	def _sync_from_raw(self, _view_box: pg.ViewBox, view_range: List[List[float]]) -> None:
		"""Handle raw plot range change: sync X-axis to clip plot."""
		self._sync_range(source="raw", view_range=view_range)

	def _sync_from_clip(self, _view_box: pg.ViewBox, view_range: List[List[float]]) -> None:
		"""Handle clip plot range change: sync X-axis to raw plot."""
		self._sync_range(source="clip", view_range=view_range)

	def _reset_view(self) -> None:
		"""Reset both plots to their default (initial) view ranges."""
		if self._default_x_range is None:
			return
		self._suspend_sync = True
		self.raw_plot.setRange(xRange=self._default_x_range, yRange=self._default_raw_y_range, padding=0.0)
		self.clip_plot.setRange(xRange=self._default_x_range, yRange=self._default_clip_y_range, padding=0.0)
		self._suspend_sync = False

	def _sync_range(self, source: str, view_range: List[List[float]]) -> None:
		"""Synchronize X-axis range between raw and clip plots.
		
		Called when either plot's view changes. Prevents circular updates using _syncing_ranges flag.
		
		Args:
			source: 'raw' or 'clip' indicating which plot changed.
			view_range: [[xmin, xmax], [ymin, ymax]] from changed plot.
		"""
		if self._syncing_ranges or self._suspend_sync:
			return

		target = self.clip_plot if source == "raw" else self.raw_plot
		x_range = view_range[0]

		self._syncing_ranges = True
		try:
			target.setXRange(x_range[0], x_range[1], padding=0.0)
		finally:
			self._syncing_ranges = False

	def _load_note(self, desi_id: str) -> None:
		"""Load and display saved note for a QSO.
		
		Args:
			desi_id: DESI identifier.
		"""
		self._loading_note = True
		self.notes_editor.setPlainText(self.store.notes_store.get_note(desi_id))
		self._loading_note = False

	def _on_note_text_changed(self) -> None:
		"""Handle note text edit: schedule auto-save via timer.
		
		Uses debounce timer to avoid writing to disk on every keystroke.
		"""
		if self._loading_note or self.current_id is None:
			return
		self.note_save_timer.start(400)

	def _flush_current_note(self) -> None:
		"""Write current QSO note to persistent storage."""
		if self.current_id is None:
			return
		note = self.notes_editor.toPlainText()
		self.store.notes_store.save_note(self.current_id, note)
		self.statusBar().showMessage(f"Saved note for {self.current_id}", 1500)

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		"""Handle window close event: save any pending note before exit."""
		self._flush_current_note()
		super().closeEvent(event)


def main() -> int:
	"""Launch the QSO viewer application.
	
	Initializes data store, creates window, and runs event loop.
	
	Returns:
		Exit code from Qt application (typically 0 for success).
	"""
	app = QtWidgets.QApplication.instance()
	if app is None:
		app = QtWidgets.QApplication([])

	package_dir = Path(__file__).resolve().parent
	store = QSODataStore(package_dir)
	window = QSOViewer(store)
	window.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
