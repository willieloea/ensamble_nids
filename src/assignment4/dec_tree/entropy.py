# file: entropy.py
import math

def entropy(ps, base=2):
    return -sum(p * math.log(p, base) for p in ps)

if __name__ == '__main__':
    ps = [0.3,0.2,0.5]
    print(entropy(ps, 2))
    ps = [0.25,0.25,0.25,0.25]
    print(entropy(ps, 2))