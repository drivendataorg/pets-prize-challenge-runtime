#!/bin/bash

set -euxo pipefail

main () {
    submission_type=$1

    if [ $submission_type = centralized ]; then
	expected_filename=solution_centralized.py
    elif [ $submission_type = federated ]; then
	expected_filename=solution_federated.py
    else
	echo "Must provide a single argument with value centralized of federated."
	exit 1
    fi

    cd /code_execution

    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -q ${expected_filename}<<<$submission_files; then
	echo "Submission zip archive must include $expected_filename"
	return 1
    fi

    echo Installed packages
    echo "######################################"
    conda list -n condaenv
    echo "######################################"

    echo Unpacking submission
    unzip ./submission/submission.zip -d ./src

    tree ./src

    if [[ $submission_type = centralized ]]; then
        echo "================ START CENTRALIZED TRAIN ================"
        conda run --no-capture-output -n condaenv python main_centralized_train.py
        echo "================ END CENTRALIZED TRAIN ================"
        echo "================ START CENTRALIZED TEST ================"
        conda run --no-capture-output -n condaenv python main_centralized_test.py
        echo "================ END CENTRALIZED TEST ================"

    elif [ $submission_type = federated ]; then

        while read scenario; do
            echo "================ START FEDERATED TRAIN FOR $scenario ================"
            conda run --no-capture-output -n condaenv python main_federated_train.py /code_execution/data/$scenario/train/partitions.json
            echo "================ END FEDERATED TRAIN FOR $scenario ================"
            echo "================ START FEDERATED TEST FOR $scenario ================"
            conda run --no-capture-output -n condaenv python main_federated_test.py /code_execution/data/$scenario/test/partitions.json
            echo "================ END FEDERATED TEST FOR $scenario ================"
        done </code_execution/data/scenarios.txt

    fi

    echo "================ END ================"
}

main $1 |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

cp /code_execution/submission/log.txt /tmp/log

exit $exit_code
