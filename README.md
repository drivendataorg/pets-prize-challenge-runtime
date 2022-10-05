# PETs Prize Challenge Runtime

**If you haven't already done so, please start by reading the documentation on the challenge websites: [Track A: Financial Crime](https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/590/), [Track B: Pandemic Forecasting](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/page/583/)**

Welcome to the runtime repository for the PETs Prize Challenge! This repository contains the source code for the runtime container image used to run code submissions in Phase 2 of the challenge.

This repository has three primary uses for competitors:

1. 💡 **Example solutions**: You can find simple examples that will help you develop your own solution. **Coming soon.**
    <!-- - [Financial Crime Federated Quickstart](./examples_src/fincrime/solution_federated.py)
    - [Financial Crime Centralized Quickstart](./examples_src/fincrime/solution_centralized.py)
    - [Pandemic Forecasting Federated Quickstart](./examples_src/pandemic/solution_federated.py)
    - [Pandemic Forecasting Centralized Quickstart](./examples_src/pandemic/solution_centralized.py) -->
2. 🔧 **Test your submission**: Test your submission with a locally running version of the container to discover errors before submitting to the competition site.
3. 📦 **Request new packages in the official runtime**: Since the Docker container will not have network access, all dependencies must be pre-installed. If you want to use a package that is not in the runtime environment, make a pull request to this repository.

----

### [Prerequisites](#prerequisites)
### [Data](#data)
### [Examples](#examples)
### [Developing your own submission](#developing-your-own-submission)
 - [Steps](#steps)
 - [Logging](#logging)
### [Additional information](#additional-information)
 - [Runtime network access](#runtime-network-access)
 - [CPU and GPU](#cpu-and-gpu)
 - [Make commands](#make-commands)
 - [Updating runtime packages](#updating-runtime-packages)


----

## Prerequisites

First, make sure you have the prerequisites prepared.

 - A clone of this repository
 - At least 4 GB of free disk space for the container image
 - [Docker](https://docs.docker.com/get-docker/) installed
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running commands in the Makefile)

## Data

This repository contains a `data/` directory. When running commands to test your solution locally, contents of this directory will be mounted to the launched Docker container. This allows you to do local evaluation using the challenge's development data.

The `data/` directory has been prepopulated with some example directory scaffolding to copy the data into. It should look like this:

```text
data
├── fincrime/
│   ├── centralized/
│   │   ├── test/
│   │   │   └── data.json
│   │   └── train/
│   │       └── data.json
│   ├── scenario01/
│   │   ├── test/
│   │   │   ├── bank01/
│   │   │   ├── partitions.json
│   │   │   └── swift/
│   │   └── train/
│   │       ├── bank01/
│   │       ├── partitions.json
│   │       └── swift/
│   └── scenarios.txt
└── pandemic/
    ├── centralized/
    │   ├── test/
    │   │   └── data.json
    │   └── train/
    │       └── data.json
    ├── scenario01/
    │   ├── test/
    │   │   ├── client01/
    │   │   ├── client02/
    │   │   └── partitions.json
    │   └── train/
    │       ├── client01/
    │       ├── client02/
    │       └── partitions.json
    └── scenarios.txt
```

Here is an explanation to help you understand this directory structure:

- Data for the two tracks are split between subdirectories `data/fincrime/` and `data/pandemic/`.
- **Federated**:
    - Each track's subdirectory has a `scenarios.txt` file. This is a newline-delimited file that lists partioning scenarios. The evaluation runner will loop through the scenarios present here. In the real evaluation runtime, there will be three scenarios defined. In the example provided here, there is one partitioning scenario named `scenario01` for each track.
    - Each scenario has a corresponding subdirectory (e.g., `data/fincrime/scenario01/`).
    - Inside the scenario directory, you will see `train/` and `test/` subdirectories. These will contain data for the respective stages.
    - Inside the `train/` or `test/` subdirectory, you will see a few things:
        - `partitions.json` is a JSON configuration file that lists each client in the scenario and paths to that client's data partition files. The top level key is the partition/client ID (`cid` in the simulation code). The inner JSON object lists the data filenames that will be provided to your client factory function. You will notice that the inner object's keys should match the argument names in the client factory signature. (Docs for [Track A](https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/587/#training); [Track B](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/page/581/#training))
        - Subdirectories for each data partition/client. The directory names should match the client IDs found in `partitions.json`. The simulation code will expect to find data files in each of these subdirectories matching the filenames in `partitions.json`. (You will need to copy your development data into here.)
- **Centralized**:
    - Each track's subdirectory also contains a `centralized/` subdirectory (e.g., `data/fincrime/centralized/`). This will contain data for centralized evaluation.
    - Like with the federated scenarios, the centralized directory contains `train/` and `test/` subdirectories.
    - Inside the train or test subdirectory, you will see a `data.json`. This is a JSON configuration file that lists the data files that the training/test code will have access to. The keys should match the argument names of the data paths provided to your `fit` or `predict` functions (Docs for [Track A](https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/588/#training); [Track B](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/page/585/#training)).
    - The evaluation code will expect to find data files alongside `data.json` that match the filenames in `data.json`. (You will need to copy your development data into here.)

In order to run evaluation locally, you will need to copy the development dataset into this directory structure. First, download the development datasets from the challenge [download page](https://www.drivendata.org/competitions/98/nist-federated-learning-1/data/). Then, you will need to copy data files into either the client subdirectories for federated data matching the filenames in `partitions.json`, or into the `train/` or `test/` subdirectories matching the filenames in `data.json`. For the federated data, it is up to you to partition the development data before copying it into the data directory.

If you have additional questions, please ask on the [challenge forum](https://community.drivendata.org/c/pets-prize-federated-learning/88).

## Example Solutions

We provide examples of simplistic models to demonstrate what a working submission looks like.

**Coming soon.**

----

## Developing your own submission

### Steps

This section provides instructions on how to develop and run your code submission locally using the Docker container. To make things simpler, key processes are already defined in the `Makefile`. Commands from the `Makefile` are then run with `make {command_name}`. The basic steps will look like:

```bash
make pull  # Not yet available
SUBMISSION_TRACK=fincrime make pack-submission
SUBMISSION_TRACK=fincrime SUBMISSION_TYPE=federated make test-submission
```

Note that many of the commands use the `SUBMISSION_TRACK` and `SUBMISSION_TYPE` variables. This repository is used for both tracks and both federated and centralized solutions. You will need to set these variables as appropriate to what you are trying to test. You may find it useful to set one or both as shell or environment variables to avoid needing to repeat it. You may also choose to set `SUBMISSION_TRACK` at the top of your copy of the `Makefile` if your team is only working on one track.

- `SUBMISSION_TRACK` has valid values of `fincrime` or `pandemic`
- `SUBMISSION_TYPE` has valid values of `federated` or `centralized`

Let's walk through what you'll need to do, step-by-step.

1. **[Set up the prerequisites](#prerequisites)**

2. **[Set up development data for testing](#data)**

3. **Download the official challenge Docker image:**

    **Not yet available.**

    ```bash
    make pull
    ```

4. ⚙️ **Save all of your submission files, including the required `solution_federated.py` module or `solution_centralized.py` module, in the `submission_src/fincrime` or `submission_src/pandemic` directory of this repository for whichever track you are working in.**
    * You are free to modify the templates we've provided or copy in your own. Keep in mind that you
    * Splitting your code up among additional modules is fine, as long as you have a `solution_federated.py` or `solution_centralized.py` that follows the requirements ([Track A](); [Track B]()).
    * Keep in mind that dependencies need to be in the present in the built image. A number of packages are already included—see [`environment-cpu.yml`](./runtime/environment-cpu.yml) and [`environment-gpu.yml`](./runtime/environment-gpu.yml) for what is present. If there are other packages you'd like added, see the section below on [updating runtime packages](#updating-runtime-packages).
   * Finally, make sure any model weights or other files you need are also saved in `submission_src`.

5. **Create a `submission/submission.zip` file containing your code:**

    ```bash
    SUBMISSION_TRACK=fincrime make pack-submission
    # or
    SUBMISSION_TRACK=pandemic make pack-submission
    ```

6. **Test your submission by launching an instance of the challenge Docker image, simulating the same evaluation process as official code execution runtime.**

    ```
    # One of:
    SUBMISSION_TRACK=fincrime SUBMISSION_TYPE=federated make test-submission
    SUBMISSION_TRACK=fincrime SUBMISSION_TYPE=centralized make test-submission
    SUBMISSION_TRACK=pandemic SUBMISSION_TYPE=federated make test-submission
    SUBMISSION_TRACK=pandemic SUBMISSION_TYPE=centralized make test-submission
    ```

    This will mount the requisite host directories on the Docker container, unzip `submission/submission.zip` into the container, and then execute the evaluation for the specified track and solution type.

> ⚠️ **Remember** that for local testing purposes, the `code_execution/data` directory is just a mounted version of what you have saved locally in this project's `data` directory. You should use development data files for local testing. For the official code execution when submitted through the drivendata.org platform, `code_execution/data` will contain the sequestered evaluation data that no participants have access to.

### Logging

When you run `make test-submission`, the logs will both be printed to the terminal and written out to `submission/log.txt`. If you run into errors, use the `log.txt` to determine what changes you need to make for your code to execute successfully. You are also encouraged to add logger statements to your own code—we recommend the [loguru](https://github.com/Delgan/loguru) package, which is available in the runtime environment.

---
## Additional information

### Runtime network access

In the real competition runtime, all internet access is blocked. The local test runtime does not impose the same network restrictions. It's up to you to make sure that your code doesn't make requests to any web resources.

You can test your submission _without_ internet access by running `BLOCK_INTERNET=true make test-submission`.

### CPU and GPU

The `make` commands will try to select the CPU or GPU image automatically by setting the `CPU_OR_GPU` variable based on whether `make` detects `nvidia-smi`.

You can explicitly set the `CPU_OR_GPU` variable by prefixing the command with:
```bash
CPU_OR_GPU=cpu <command>
```

**If you have `nvidia-smi` and a CUDA version other than 11**, you will need to explicitly set `make test-submission` to run on CPU rather than GPU. `make` will automatically select the GPU image because you have access to GPU, but it will fail because `make test-submission` requires CUDA version 11.
```bash
CPU_OR_GPU=cpu make pull
CPU_OR_GPU=cpu make test-submission
```

If you want to try using the GPU image on your machine but you don't have a GPU device that can be recognized, you can use `SKIP_GPU=true`. This will invoke `docker` without the `--gpus all` argument.

### Updating runtime packages

If you want to use a package that is not in the environment, you are welcome to make a pull request to this repository. If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://docs.github.com/en/get-started/quickstart/contributing-to-projects). The runtime manages dependencies using [conda](https://docs.conda.io/en/latest/) environments. [Here is a good general guide](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to conda environments. The official runtime uses **Python 3.9.13** environments.

**Note: Since package installations need to be approved, be sure to submit any PRs requesting installation by January 4, 2023 to ensure they are incorporated in time for you to make a successful submission.**

To submit a pull request for a new package:

1. Fork this repository.

2. Edit the [conda](https://docs.conda.io/en/latest/) environment YAML files, `runtime/environment-cpu.yml` and `runtime/environment-gpu.yml`. There are two ways to add a requirement:
    - Add an entry to the `dependencies` section. This installs from a conda channel using `conda install`. Conda performs robust dependency resolution with other packages in the `dependencies` section, so we can avoid package version conflicts.
    - Add an entry to the `pip` section. This installs from PyPI using `pip`, and is an option for packages that are not available in a conda channel.

    If a package is available through both managers, conda is preferred. For both methods be sure to include a version, e.g., `numpy==1.20.3`. This ensures that all environments will be the same.

3. Locally test that the Docker image builds successfully for CPU and GPU images:

    ```sh
    CPU_OR_GPU=cpu make build
    CPU_OR_GPU=gpu make build
    ```

4. Commit the changes to your forked repository.

5. Open a pull request from your branch to the `main` branch of this repository. Navigate to the [Pull requests](https://github.com/drivendataorg/pets-prize-challenge-runtime) tab in this repository, and click the "New pull request" button. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

6. Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and running the tests in `runtime/tests`. These tests can take up to 30 minutes, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.

7. You may be asked to submit revisions to your pull request if the tests fail or if a DrivenData team member has feedback. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.


### Make commands

Running `make` at the terminal will tell you all the commands available in the repository:

```
Settings based on your machine:
SUBMISSION_IMAGE=f6961d910a89   # ID of the image that will be used when running test-submission

Available competition images:
drivendata/belugas-competition:cpu-local (f6961d910a89); drivendata/pets-prize:gpu-local (916b2fbc2308);

Available commands:

build               Builds the container locally
clean               Delete temporary Python cache and bytecode files
interact-container  Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
pack-example        Creates a submission/submission.zip file for one of the examples from the source code in examples_src
pack-submission     Creates a submission/submission.zip file from the source code in submission_src
pull                Pulls the official container from Azure Container Registry
test-container      Ensures that your locally built container can import all the Python packages successfully when it runs
test-submission     Runs container using code from `submission/submission.zip` and data from `data/`
```
---

## Good luck and have fun!

Enjoy the competition, and [visit the forums](https://community.drivendata.org/c/pets-prize-federated-learning/88) if you have any questions!
