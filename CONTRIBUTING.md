# Contributor guide

Thank you for your interest in contributing to Raster Loader!

## Feature requests and bug reports

Reporting bugs and submitting ideas for new features are great ways to help make Raster
Loader better.

To report a bug or request a feature, please
[create a new issue in the GitHub repository](https://github.com/CartoDB/raster-loader/issues/new/choose).
The issue tracker gives you the option to choose between a bug report and a feature
request. It also provides templates with more information on how to file your bug report
or feature request.

## Contributing code and documentation

### Prerequisites

You will need to sign a Contributor License Agreement (CLA) before making a submission.
[Learn more here](https://carto.com/contributions/).

Raster Loader uses GitHub and git for version control. If you are new to git, you can
learn more about it [here](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control).

Raster Loader uses a Makefile to automate many aspects of the development process.
Using the Makefile requires that you have [GNU Make](https://www.gnu.org/software/make/) installed.

### Setting up your environment

Create a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
of [the Raster Loader repository](https://github.com/CartoDB/raster-loader).
Use `git clone` to clone the repo to your local machine.

Once the repository is cloned, you can use the Makefile to set up your development
environment:

```bash
make init
```

This will create a virtual environment in the `env` directory and install all
necessary dependencies, including a development version of Raster Loader and the
Raster Loader CLI.

If you don't have `make` available, you can open the file `Makefile` and run the
commands manually to set up a virtual environment and install the dependencies.

After creating your environment, you can enter the virtual environment with
``source env/bin/activate`` on Linux and macOS or ``env\bin\Activate.ps1`` on Windows
(PowerShell).


### Setting up your environment (Docker / Docker-Compose)

As an alternative to setting up a virtual environment, you can also set up a
development environment using Docker:

1. Install Docker and Docker Compose on your system by following the instructions for your operating system from the official Docker website.
2. Use `git clone` to clone [the Raster Loader repository](https://github.com/CartoDB/raster-loader)
to your local machine.
3. Navigate to the root directory of the repository in your terminal.
4. Run `make docker-build` command to build the docker image
5. Run `make docker-start` to start the development environment. Keep this process running.
6. Begin your development in a new terminal.
7. Run `make docker-test` to run the test suite.
8. Run a targeted test using pytest flags: `make docker-test PYTEST_FLAGS='-s -k array'`
9. Run `git checkout -b my-new-feature` to start a new feature branch
10. Consider writing a test in `raster_loader/tests/` to guide your implementation
11. Drop into `pdb` when a test fails: `make docker-test PYTEST_FLAGS='-s --pdb'`
12. Run `make docker-enter` to open a terminal inside of the docker container
13. Run `make docker-stop` to stop the development environment
14. Run `make docker-remove` to remove docker raster_loader Container/Network/Volume from your system

*Note: If you want to make changes to library dependencies (i.e. requirements.txt or requirements-dev.txt) while the container is running, you'll need to rebuild the image using the make docker-build command and restart the container."*

### Tests and linting

Before submitting a pull request, you need to make sure your updates pass tests and
linting.

#### Running linting

To run linting, use the following command:

```bash
make lint
```

This runs [flake8](https://flake8.pycqa.org/en/latest/) and
[black](https://black.readthedocs.io/en/stable/). You can also run these tools
individually using the ``flake8`` or ``black`` command.

#### Running tests

Raster Loader uses [pytest](https://docs.pytest.org/en/stable/) for testing. You can
run the tests with the following command:

```bash
make test
```

This runs all tests in the `tests` directory. You can also run all tests with the
``pytest`` command.

The test suite includes optional integration tests that require credentials for a
BigQuery account. To run these tests, you need to set the `GOOGLE_APPLICATION_CREDENTIALS`
environment variable to the path of a JSON file containing your BigQuery credentials
(see the [GCP documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key)
for more information).

You must also copy the `/test/.env.sample` to `/test/.env` and edit it to set a
test project and dataset in which the used credentials have permissions to create tables.

If you're working on Windows, please set manually the env variables or the .env file in your terminal.

After setting up your credentials and .env, you can enable the integration
test with the following command:

```bash
pytest --runintegration
```

#### Updating tests

All new code needs to be covered by tests. The tests for Raster Loader are located in
the `raster_loader/tests` directory. Each Python module in the package should have its
own test module in this directory.

The `raster_loader/tests` directory also contains tests for the CLI. To learn more about
writing tests for the CLI, see the
[Click documentation](https://click.palletsprojects.com/en/8.1.x/testing/).

To only run a specific test file, use the following command:

```bash
pytest tests/[test_file_name]
```

To only run a specific test, use the following command:

```bash
pytest -k "[test_name]"
```

### Updating documentation

All new features and updates to features need to be documented.

Raster Loader uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate
documentation.

The documentation is located in the `docs` directory. You can build the documentation
with the following command:

```bash
make docs
```

This will generate the documentation in the `docs/build` directory.

The documentation follows the
[Google developer documentation style guide](https://developers.google.com/style).

The documentation also includes a module API reference. This reference is generated
automatically from the docstrings in the code. Please use
[NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) for all your
docstrings.

Also included in the documentation is a reference of all available CLI commands.
This reference is generated automatically from the docstrings in the CLI code. See
the [documentation for sphinx-click](https://sphinx-click.readthedocs.io/en/latest/)
for more information on how to document the CLI.

### Making pull requests

All contributions to Raster Loader are made through pull requests to the
[the Raster Loader repository](https://github.com/CartoDB/raster-loader).

See the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
for more information on how to use pull requests.

All pull requests must reference an issue in the Raster Loader repository (a bug report
or feature request, for example). If you can't find an issue to reference, make
sure to create a new issue before submitting your pull request.

Pull requests to the Raster Loader repository must pass all automated tests and linting
before they are considered for merging. You can use the ["WIP" label](https://github.com/CartoDB/raster-loader/labels/WIP)
or [mark your pull request as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests)
to indicate that it is not ready for review.

Before merging a pull request, the Raster Loader maintainers will review your code and
might request changes. You can make changes to your pull request by pushing new commits.
