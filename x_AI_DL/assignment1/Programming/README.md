# Assignment 1: NumPy, tensors, derivatives and regression

The programming part of assignment 1 of the xAILab Bamberg Deep Learning exercise.

## Layout

```
code/           the six notebooks you work in, plus tests/
data/           the cats-and-dogs images used by the logistic regression part
checkpoints/    written by the notebooks
img/            images the notebooks load at run time (their other figures are embedded)
```

Work through the notebooks in `code/` in this order:

1. `numpy_basics.ipynb`
2. `numpy_applied.ipynb`
3. `torch_tensors_basics.ipynb`
4. `derivatives_and_graphs.ipynb`
5. `linear_regression.ipynb`
6. `logistic_regression.ipynb`

## Environment

```bash
conda env create -f environment.yaml
conda activate xai-dl-a1
```

### PyTorch

`environment.yaml` deliberately does not pin PyTorch, because the right build depends on
whether you run on CPU or GPU and on your CUDA version. Install it yourself from
https://pytorch.org/get-started/locally/ after activating the environment. CPU-only:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```


## What to implement

Open the notebooks listed below and fill in the blocks marked

```
########################################################
#                 START OF YOUR CODE                   #
########################################################
```

Write only between those markers. Do not rename files or functions, do not change
function signatures, and do not add imports. The automated tests rely on all of it,
and a change there scores 0 for that task.

## Checking your work

Each notebook has test cells directly beneath the code you write, so you can check one
task at a time. To score the whole assignment the way we do:

```bash
bash run_all_tests.sh        # every test, with a score per part
```

Passing every local test does not guarantee full marks: we run additional tests that
are not in your copy.

## Submitting

```bash
bash collect_submission.sh            # macOS / Linux
collect_submission.bat                # Windows
```

That writes `assignment1_submission.zip`. Upload it to VC together with your theory
PDF. Check the zip contains your notebooks before you upload it.

The task descriptions, the point breakdown and the submission rules are in
`../assignment1.pdf`. The theory part and its template are in `../Theory/`.
