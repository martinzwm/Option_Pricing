from transition import BrownianMotion
from LSM import AmericanOptionsLSMC
import matplotlib.pyplot as plt

# Test baseline - American Put LSM method
AmericanPUT = AmericanOptionsLSMC(option_type='put', S0=36, strike=40, T=1, M=50,
                                  r=0.06, div=0, sigma=0.2, N=10000, transition=BrownianMotion)
print('Price: ', AmericanPUT.price)

# Generate trajectories
traj_gen = BrownianMotion(S0=36, r=0.06, sigma=0.2, T=1, M=50, N=10000)
traj1 = traj_gen.simulate()[:,10] # take a look at the 10th trajectory
# plt.plot(traj1)
# plt.savefig("test.png")