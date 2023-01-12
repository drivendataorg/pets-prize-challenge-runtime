# Changelog

## 2022-01-11

- Added ability to run optional `install.sh` script before evaluation.
    - This is an optional script that will be run if it exists. It can be used for environment setup that you need to run, such as installing dependencies bundled with your submission, or setting environment variables. **You should NOT be executing any code that represents a meaningful part your solution's execution.** Use of the install.sh script should be explained in your code guide and justified. Abuse of this script for unintended purposes is grounds for disqualification. If you have any questions, please ask on the [community forum](https://community.drivendata.org/c/pets-prize-federated-learning/88).
- Added ability to run optional `train_setup` and `test_setup` functions at the beginning of federated training and test, respectively.
    - This is an optional addition to the evaluation API to give more flexibility to teams' solutions. Please closely review the documentation for appropriate use ([U.S. Track A](https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/587/#optional-setup-functions); [U.S. Track B](https://www.drivendata.org/competitions/103/nist-federated-learning-2-pandemic-forecasting-federated/page/581/#optional-setup-functions); [U.K. Track A](https://www.drivendata.org/competitions/140/uk-federated-learning-2-financial-crime-federated/page/638/#optional-setup-functions); [U.K. Track B](https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/643/#optional-setup-functions)). Inappropriate use of these functions is grounds for disqualification. If you have any questions, please ask on the [community forum](https://community.drivendata.org/c/pets-prize-federated-learning/88).

## 2022-01-06

- Updated README with information about the `predictions_format.csv` files.

## 2022-01-04

- Fixed CUDA issues in the GPU-based image.
- Updated `make pack-example` and `make pack-submission` to create the `submission/` directory if it does not already exist.
- Updated the financial crime example to handle an edge case when a bank partition has no receiving accounts. ([`examples_src/fincrime`](./examples_src/fincrime/))

## 2022-12-31

- Added post-run steps to the evaluation that collate predictions and some key runtime metrics.
- Updated the pandemic forecasting example to compute more efficiently. ([`examples_src/pandemic`](./examples_src/pandemic/))

## 2022-12-02

- Bumped the version of Flower in runtime image to 1.1.0. This includes improvements such as better error logging when running simulated federated learning. See [Flower changelog](https://flower.dev/docs/changelog.html#v1-1-0-2022-10-31) for full details.
- Added example submission code. These are meant to be simple demonstrations of how to write code that works with evaluation harness. They do not meaningfully model the machine learning task nor do they implement any privacy techniques or technologies.
    - Added simple example submission for Track A: Financial Crime Prevention ([`examples_src/fincrime`](./examples_src/fincrime/)). This is a simplistic, dummy model that illustrates a heterogeneous sequence of client–server communications using the federated training and test simulations.
    - Added simple example submission for Track B: Pandemic Forecasting ([`examples_src/pandemic`](./examples_src/pandemic/)). This is a simplistic, naive model inspired by the SIR epidemiological model.
- Removed limitations on use the `fit` method on clients during the test inference simluation. This allows the use of the `fit` method, whose API better supports sending NumPy arrays to the server than the `evaluate` method, as part of test inference workflows. Note that use of the `fit` method is only meant to facilitate client–server communications. Solutions should _not_ be fitting models on the test data during the test inference.
