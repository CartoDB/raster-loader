VENV=env
DIST=dist
BUILD=build
BIN=$(VENV)/bin

.PHONY: docs

init:
	test `command -v python3` || echo Please install python3
	[ -d $(VENV) ] || python3 -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pre-commit install
	$(BIN)/pip install -e .

lint:
	$(BIN)/black raster_loader setup.py
	$(BIN)/flake8 raster_loader setup.py

test:
	$(BIN)/pytest raster_loader --cov=raster_loader --verbose

docs:
	cd docs; make clean html

test-docs:
	$(BIN)/sphinx-build -a -W --keep-going docs/source/ docs/build/

publish-pypi:
	rm -rf $(DIST) $(BUILD) *.egg-info
	$(BIN)/python setup.py sdist bdist_wheel
	$(BIN)/twine upload $(DIST)/*

publish-test-pypi:
	rm -rf $(DIST) $(BUILD) *.egg-info
	$(BIN)/python setup.py sdist bdist_wheel
	$(BIN)/twine upload --repository-url https://test.pypi.org/legacy/ $(DIST)/* --verbose

clean:
	rm -rf $(VENV) $(DIST) $(BUILD) *.egg-info
