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
	$(BIN)/pip install -e .[snowflake,bigquery]

lint:
	$(BIN)/black raster_loader setup.py
	$(BIN)/flake8 raster_loader setup.py

test:
	$(BIN)/pytest raster_loader --cov=raster_loader --verbose

test-integration:
	$(BIN)/pytest raster_loader --cov=raster_loader --verbose --runintegration

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

ENTER_CONTAINER:=docker-compose exec raster_loader

.PHONY: docker-build
docker-build: ## Build necessary stuff.
	docker-compose build

.PHONY: docker-start
docker-start: ## Start containers with docker-compose and attach to logs.
	docker-compose up --no-build

.PHONY: docker-test
docker-test: ## Enter the running backend container and run tests.
	$(ENTER_CONTAINER) sh -c 'cd raster_loader && pytest $(PYTEST_FLAGS)'

.PHONY: docker-enter
docker-enter: ## Enter the backend container.
	$(ENTER_CONTAINER) bash

.PHONY: docker-stop
docker-stop: ## Stop all running containers.
	docker-compose stop

.PHONY: docker-remove
docker-remove: ## Remove all containers / volumes
	docker-compose down --volumes
