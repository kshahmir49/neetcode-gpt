import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res = []
        mx = max(z)
        exp = sum(np.exp(z-mx))
        for i in z:
            res.append(np.exp(i-mx)/exp)
        return np.around(res,4)
