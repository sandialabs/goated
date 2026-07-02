# Goal-Oriented Tensor Decompositions (GOATED)

## ** Warning:  Under Construction **

GOATED is a pure-python implementation of the goal-oriented Canonical Polyadic (CP) and Tucker decompositions described in [this paper](https://arxiv.org/abs/2508.11139) (currently under review for publication).  It leverages [pyttb](https://github.com/sandialabs/pyttb) to define the CP/Tucker low-rank model and [pyrol](https://pypi.org/project/rol-python) to solve the goal-oriented optimization problem.

## Installation

After downloading the code, GOATED can be easily installed via `pip install ./goated` which will install all needed dependencies.  Note, pyrol only supplies precompiled python wheels for a limited number of architectures, so pip is likely to build a wheel from source.  The code can then be tested via `pytest goated/goated/tests`.

## Usage

Examples for goal-oriented CP and Tucker are provided in the Jupyter notebooks in the examples directory that apply these methods to a smaller version of the Tearing Sheet plasma physics data described in the above publication.
