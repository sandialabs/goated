# goal-oriented-tensor-decompositions

Need Eric's input on:

 0. What `scale` means in _tangent_reconstructed_tensor for concrete base classes.
    (CPObjective, TuckerObjective).

 1. What StdScaler is supposed to do.
    * Why does goated/goals/abstract.py only ever use unscale_tensor?
      It seems like there should be a combination of unscale_tensor and scale_tensor.

 2. Are the classes in goated/abstractobj.py reasonable?
    * Do their function names make sense?
    * Are the implemented functions have unambiguously correct (if undocumented)?

 3. Looking at goated/goals/abstract.py::TimeSeparableGoal, what are the semantics
    of "jac" as returned by self.computeVector(U, True)?

