# file: gini.py

def gini_index(ps):
    return 1.0 - sum((p**2) for p in ps)


if __name__ == '__main__':
    ps = [0.3,0.2,0.5]
    print(gini_index(ps))
    ps = [0.25,0.25,0.25,0.25]
    print(gini_index(ps))