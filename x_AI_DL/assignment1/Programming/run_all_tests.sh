# Adds the tests folder to path to make sure extract_logic_module is usable
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/code/tests

project_path="./code"

# Check if the project path is already in the PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$project_path:"* ]]; then
  # Append the project path to the PYTHONPATH
  export PYTHONPATH="$PYTHONPATH:$project_path"
  echo "Project path added to PYTHONPATH."
else
  echo "Project path already in PYTHONPATH."
fi

cd "$project_path"

echo
echo "Starting Assignment 1 Tests..."
printf -- '=%.0s' {1..70}
echo

total_score=0
bonus_score=0

test_files=(
    "./tests/test_numpy_basics.py"
    "./tests/test_numpy_applied.py"
    "./tests/test_torch_tensors_basics.py"
    "./tests/test_derivatives_and_graphs.py"
    "./tests/test_linear_regression.py"
    "./tests/test_logistic_regression.py"
)

for test_file in "${test_files[@]}"; do
    test_file_name=$(basename "$test_file")
    notebook="./${test_file_name/test_/}.ipynb"

    echo
    echo
    echo "RUNNING $(basename "$test_file")..."
    echo

    # Run the Python test file and capture output scores
    output=$(python -m unittest "$test_file")
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "There was an error running the tests in $test_file_name. Check the test script."
        printf -- '-%.0s' {1..70}
        continue
    fi

    notebook_score=0
    notebook_bonus_score=0

    while read -r line; do
        if [[ $line == Running* ]]; then
            echo "$line"
        elif [[ $line == Test*failed ]]; then
            echo "$line"
        elif [[ $line == *"Score achieved:"* ]] && [[ ! $line == *"("* ]]; then
          score=$(echo "$line" | grep -o -E 'Score achieved: \[[0-9]+(\.[0-9]+)?' | grep -o -E '[0-9]+(\.[0-9]+)?')
          notebook_score=$(echo "$notebook_score + $score" | bc)
        elif [[ $line == *"Bonus score achieved:"* ]] && [[ ! $line == *"("* ]]; then
          bonus=$(echo "$line" | grep -o -E 'Bonus score achieved: \[[0-9]+(\.[0-9]+)?' | grep -o -E '[0-9]+(\.[0-9]+)?')
          notebook_bonus_score=$(echo "$notebook_bonus_score + $bonus" | bc)
        fi
    done <<< "$output"

    echo
    echo "TOTAL SCORE FOR $(basename "$notebook"): [ $notebook_score ] "
    echo "BONUS SCORE: [ $notebook_bonus_score ]"
    printf -- '-%.0s' {1..70}
    echo

    total_score=$(echo "$total_score + $notebook_score" | bc)
    bonus_score=$(echo "$bonus_score + $notebook_bonus_score" | bc)
done

echo
echo
echo
printf -- '=%.0s' {1..70}
echo
echo "TOTAL SCORE FOR **Assignment 1 Coding part**: [ $total_score / 60.0 ]"
echo "BONUS SCORE: [ $bonus_score / 5.0 ]"
printf -- '=%.0s' {1..70}
echo