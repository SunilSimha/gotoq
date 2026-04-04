from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def package_files(relative_dir):
	base_dir = ROOT / relative_dir
	if not base_dir.exists():
		return []

	return [
		str(path.relative_to(ROOT / "gotoq"))
		for path in base_dir.rglob("*")
		if path.is_file()
	]


def bin_scripts():
	bin_dir = ROOT / "bin"
	if not bin_dir.exists():
		return []

	return sorted(path.relative_to(ROOT).as_posix() for path in bin_dir.iterdir() if path.is_file())


README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
	name="gotoq",
	version="0.1.0",
	description="A systematic search for Galaxies on Top of Quasars (GOTOQs) in DESI DR1",
	long_description=README,
	long_description_content_type="text/markdown",
	packages=find_packages(),
	install_requires=[
		"numpy",
		"pyqtgraph",
		"PyQt6",
	],
	include_package_data=True,
	package_data={"gotoq": package_files("gotoq/data")},
	scripts=bin_scripts(),
)
