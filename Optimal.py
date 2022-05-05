from LSM import AmericanOptionsLSMC
import numpy as np

class Optimal(AmericanOptionsLSMC):
    """ Class for optimal action.
        Inheritance of LSM method - very similar properties.
    """

    @property # override
    def value_matrix(self):
        """ Returns the expected value at each time step and trajectory
            Output:
                value_matrix [matrix of size M x N, float]: expected value
                decision_matrix [matrix of size M x N, bool]: decision
        """
        value_matrix = np.zeros_like(self.MCpayoff)
        decision_matrix = np.zeros_like(self.MCpayoff)

        # Terminal T
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        decision_matrix[-1, :] = self.MCpayoff[-1, :] > 0

        # Recusion from T-1 to 0 
        for t in range(self.M - 1, -1, -1):
            continuation = value_matrix[t + 1, :] * self.discount
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation,
                                          self.MCpayoff[t, :],
                                          continuation)
            decision_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation, 1, 0)

        return value_matrix, decision_matrix