from transition import BrownianMotion
from LSM import AmericanOptionsLSMC

AmericanPUT = AmericanOptionsLSMC('put', 36., 40., 1., 50, 0.06, 0, 0.2, 10000, BrownianMotion)
print('Price: ', AmericanPUT.price)