# gotoq

Interactive desktop viewer for DESI QSO cutouts, spectra, and spectral line overlays.

## Features

- **Browse QSO catalog** with keyboard shortcuts and combo box navigation
- **View cutout images and spectra** side-by-side with synchronized zoom
- **Open Legacy Survey from cutout** by clicking the displayed image (uses QSO RA/DEC)
- **Overlay spectral lines** (emission and absorption) with redshift adjustment
- **Toggle line categories** (emission, absorption, major-only) dynamically
- **Persistent per-object notes** auto-saved to CSV
- **Fine and coarse redshift control** via slider, text entry, and keyboard shortcuts
- **Performance optimized** for rapid redshift stepping (coalesced overlay updates)

## Installation

### Prerequisites

- **Python 3.8+**
- **PyQt6** (or PyQt5 via pyqtgraph Qt wrapper)
- **numpy**, **pyqtgraph**

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/SunilSimha/gotoq.git
   cd gotoq
   ```

2. Install in development mode with dependencies:
   ```bash
   pip install -e .
   ```

3. (Optional) For development, install test/linting tools:
   ```bash
   pip install -e ".[dev]"
   ```

### Data Setup

The viewer expects data in the following directory structure:

```
gotoq/data/
├── desi_qso_cutouts/       # JPEG cutout images (name: <DESI_ID>_cutout.jpeg)
├── spectra/
│   └── ascii/              # ASCII spectrum files (name: <DESI_ID>.dat)
└── tables/
    └── desi_qsos.csv       # Catalog file with DESI_ID, z, and metadata
```
All of these files can be downloaded from [this GDrive link](https://drive.google.com/drive/folders/1u9fv2Z_Hleu7qLNKJQhPDndl6oAPJ4ph?usp=drive_link).
**Spectrum file format:** 7+ columns (space or tab-separated)
- Col 0: wavelength (Å)
- Col 1: flux
- Col 2: error
- Col 3-4: reserved
- Col 5: flux_clip (normalized continuum)
- Col 6: error_clip

**Catalog format:** CSV with headers, e.g., `DESI_ID,DESI_z,RA,DEC,...`

## Usage

### Launching the Viewer

After installation, run:

```bash
gotoq-qso-viewer
```

Or programmatically:

```python
from gotoq.visualizer_tool import main
main()
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Navigation** |
| `Left Arrow` | Previous QSO in catalog |
| `Right Arrow` | Next QSO in catalog |
| **Redshift Control** |
| `Alt+Left` | Decrease redshift (large steps) |
| `Alt+Right` | Increase redshift (large steps) |
| `Shift+Alt+Left` | Decrease redshift (very large steps) |
| `Shift+Alt+Right` | Increase redshift (very large steps) |
| `z` | Reset redshift to catalog value |
| **Line Overlays** |
| `e` | Toggle emission lines |
| `a` | Toggle absorption lines |
| `m` | Toggle major-only filter |
| **Editing & Help** |
| `c` | Focus notes input |
| `Esc` | Clear text focus (dismiss notes/redshift editing) |
| `q` | Quit application |
| `F1` | Show keyboard shortcuts help |

### GUI Elements

#### Control Panel (Top)

- **DESI_ID combo box**: Select object from catalog
- **Previous/Next buttons**: Navigate catalog
- **Emission/Absorption/Major Only buttons**: Toggle line overlays
- **Reset View button**: Return plots to initial view range
- **Help hint**: "Press F1 for help" reminder is shown in the top bar

#### Redshift Controls

- **Fine slider**: Precise redshift adjustment (0.00001 resolution)
- **Text entry**: Type exact redshift value (press Enter to apply)
- **Reset z button**: Revert to catalog redshift

#### Plots (Right Panel)

- **Raw plot (top)**: Full spectrum with flux and error traces
- **Clip plot (bottom)**: Normalized continuum with error offset
- **Scroll wheel**: Zoom X-axis (wavelength)
- **Ctrl+Scroll**: Zoom Y-axis (flux)
- **Click/Drag**: Pan across spectrum

#### Cutout Panel (Left)

- **Clickable cutout image**: Opens Legacy Survey viewer in browser for the current object
- **URL format used**: `https://www.legacysurvey.org/viewer?ra=<RA>&dec=<DEC>&layer=ls-dr10&zoom=16`
- **Coordinates source**: `RA` and `DEC` columns from `data/tables/desi_qsos.csv`

#### Info & Notes (Bottom)

- **Info box (left)**: Displays catalog metadata for current QSO
- **Notes editor (right)**: Write and save per-object notes (auto-saves 400ms after typing)

### Workflow Example

1. **Launch**: `gotoq-qso-viewer`
2. **Browse**: Use Left/Right Arrow keys or combo box to find QSO
3. **Inspect**: View cutout and spectrum side-by-side; click the cutout to open Legacy Survey
4. **Adjust redshift**: Fine-tune with Alt+Left/Right keyboard or slider
5. **Overlay lines**: Toggle `e`/`a`/`m` to show/hide spectral features
6. **Add notes**: Press `c` to focus notes, type observations, press `Esc` to save
7. **Reset view**: Press `Reset View` button to return to full spectrum extent

### Spectral Line Data

Emission and absorption lines are read from CSV files at runtime:

- **`gotoq/data/lines/emission_lines.txt`**: Emission line definitions
- **`gotoq/data/lines/absorption_lines.txt`**: Absorption line definitions

**Format:** `name, ?, wavelength_angstrom, ?, major_flag`

Example:
```
Lyman alpha, , 1215.67, , true
H-alpha, , 6562.85, , true
Na D 5893, , 5893.0, , false
```

Lines are plotted using rest-frame vacuum wavelengths. Observed wavelength is computed as `λ_obs = λ_rest × (1 + z)`.

## Architecture

### Main Classes

- **`QSOViewer`**: Main PyQt6 window with controls, plots, and event handling
- **`QSODataStore`**: Indexes and loads QSO assets (cutouts, spectra, catalog)
- **`NotesStore`**: Persistent CSV-based note storage
- **`CustomViewBox`**: pyqtgraph ViewBox with controlled scroll behavior
- **`ImageLabel`**: Responsive image display widget

### Performance Features

- **Overlay caching**: Stores marker references for in-place position updates during redshift changes
- **Coalesced updates**: Rapid redshift changes are debounced into a single overlay refresh (16ms interval)
- **Dynamic downsampling**: Spectra are downsampled automatically for efficient rendering
- **Clip-to-view**: Plot items only render visible data

## Development Notes

- **Redshift slider resolution**: 100,000 ticks for fine-precision control
- **Keyboard stepping**: Alt+arrow uses 100× multiplier, Shift+Alt uses 1000× multiplier
- **Text focus guard**: Keyboard shortcuts are disabled while typing in notes or redshift text fields
- **External click dismiss**: Clicking anywhere outside text inputs clears focus
- **View synchronization**: X-axis zoom is synchronized between raw and clip plots; Y-axis is independent
