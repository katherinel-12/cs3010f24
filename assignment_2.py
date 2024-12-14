import numpy as np
import sys

def fromFile(fileInput):
    with open(fileInput, 'r') as file:
        n = int(file.readline().strip()) # First line of input file
        listCoeff = []
        listConst = []
        
        for i in range(n): # Read n lines for listCoeff
            line = file.readline().strip()
            coeffNums = list(map(float, line.split()))
            listCoeff.append(coeffNums)

        for line in file: # Read last line for listConst
            constNums = list(map(float, line.split()))
            listConst.extend(constNums)

        return listCoeff, listConst
        print(file.close)

def checkCoeff(eqCoeffs):
    if len(eqCoeffs) < 1:
        print("Incorrect coeff matrix!")
        exit(1)
        
# Naive Gaussian Elimination
def FwdEliminationDouble(eqCoeffs, eqConsts): # 2d array and list
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    for k in range (0, n):
        for i in range (k + 1, n):
            mult = eqCoeffs[i][k] / eqCoeffs[k][k]
            # print(mult)
            for j in range (k, n):
                eqCoeffs[i][j] = eqCoeffs[i][j] - mult * eqCoeffs[k][j]
            eqConsts[i] = eqConsts[i] - mult * eqConsts[k]
            
def BackSubstDouble(eqCoeffs, eqConsts, sol):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0]) - 1
    sol[n] = eqConsts[n] / eqCoeffs[n][n]
    for i in range (n-1, -1, -1):
        sum1 = eqConsts[i]
        for j in range (i + 1, n + 1):
            sum1 = sum1 - eqCoeffs[i][j] * sol[j]
        sol[i] = sum1 / eqCoeffs[i][i]
    outputFile = "outputNaiveDouble.sol"
    with open(outputFile, 'w') as file:
        file.write(str(sol))
    print("Naive Double ", sol)

def NaiveGaussianDouble(eqCoeffs, eqConsts):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    sol = [0.0] * n
    FwdEliminationDouble(eqCoeffs,eqConsts)
    BackSubstDouble(eqCoeffs, eqConsts, sol)

# Gaussian Elimination with Scaled Partial Pivoting
def SPPFwdEliminationDouble(eqCoeffs, eqConsts, ind):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    scaling = [0.0] * n
    for i in range (n):
        smax = 0
        for j in range (n):
            smax = max(smax,abs(eqCoeffs[i][j]))
        scaling[i] = smax
    for k in range (n - 1):
        rmax = 0
        maxInd = k
        for i in range (k, n):
            r = abs(eqCoeffs[ind[i]][k] / scaling[ind[i]])
            if (r > rmax):
                rmax = r
                maxInd = i
        # swap(ind[maxInd], ind[k])
        tempV = ind[maxInd]
        ind[maxInd] = ind[k]
        ind[k] = tempV
        
        for i in range (k + 1, n):
            mult = eqCoeffs[ind[i]][k] / eqCoeffs[ind[k]][k]
            for j in range (k + 1, n):
                    eqCoeffs[ind[i]][j] = eqCoeffs[ind[i]][j] - mult * eqCoeffs[ind[k]][j]
            eqConsts[ind[i]] = eqConsts[ind[i]] - mult * eqConsts[ind[k]]

def SPPBackSubstDouble(eqCoeffs, eqConsts, sol, ind):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0]) -1
    sol[n] = eqConsts[ind[n]] / eqCoeffs[ind[n]][n]
    for i in range (n - 1, -1, -1):
        sum1 = eqConsts[ind[i]]
        for j in range (i + 1, n+1):
            sum1 = sum1 - eqCoeffs[ind[i]][j] * sol[j]
        sol[i] = sum1 / eqCoeffs[ind[i]][i]
    outputFile = "outputSPPDouble.sol"
    with open(outputFile, 'w') as file:
        file.write(str(sol))
    print("SPP Double ", sol)

def SPPGaussianDouble(eqCoeffs, eqConsts):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    sol = [0.0] * n
    ind = [0] * n
    for i in range (n):
        ind[i] = i
    SPPFwdEliminationDouble(eqCoeffs,eqConsts,ind)
    SPPBackSubstDouble(eqCoeffs,eqConsts,sol,ind)

# Naive Gaussian Elimination - Single Precision
def FwdEliminationSingle(eqCoeffs, eqConsts): # 2d array and list
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    for k in range (0, n):
        for i in range (k + 1, n):
            mult = eqCoeffs[i][k] / eqCoeffs[k][k]
            # print(mult)
            for j in range (k, n):
                eqCoeffs[i][j] = eqCoeffs[i][j] - mult * eqCoeffs[k][j]
            eqConsts[i] = eqConsts[i] - mult * eqConsts[k]
            
def BackSubstSingle(eqCoeffs, eqConsts, sol):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0]) - 1
    sol[n] = eqConsts[n] / eqCoeffs[n][n]
    for i in range (n-1, -1, -1):
        sum1 = eqConsts[i]
        for j in range (i + 1, n + 1):
            sum1 = sum1 - eqCoeffs[i][j] * sol[j]
        sol[i] = sum1 / eqCoeffs[i][i]
    solSingle = np.array(sol, dtype=np.float32)

    outputFile = "outputNaiveSingle.sol"
    with open(outputFile, 'w') as file:
        file.write(str(solSingle))
    print("Naive Single ", solSingle)

def NaiveGaussianSingle(eqCoeffs, eqConsts):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    sol = [0.0] * n
    FwdEliminationSingle(eqCoeffs,eqConsts)
    BackSubstSingle(eqCoeffs, eqConsts, sol)

# Gaussian Elimination with Scaled Partial Pivoting - Single Precision
def SPPFwdEliminationSingle(eqCoeffs, eqConsts, ind):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    scaling = [0.0] * n
    for i in range (n):
        smax = 0
        for j in range (n):
            smax = max(smax,abs(eqCoeffs[i][j]))
        scaling[i] = smax
    for k in range (n - 1):
        rmax = 0
        maxInd = k
        for i in range (k, n):
            r = abs(eqCoeffs[ind[i]][k] / scaling[ind[i]])
            if (r > rmax):
                rmax = r
                maxInd = i
        # swap(ind[maxInd], ind[k])
        tempV = ind[maxInd]
        ind[maxInd] = ind[k]
        ind[k] = tempV
        
        for i in range (k + 1, n):
            mult = eqCoeffs[ind[i]][k] / eqCoeffs[ind[k]][k]
            for j in range (k + 1, n):
                    eqCoeffs[ind[i]][j] = eqCoeffs[ind[i]][j] - mult * eqCoeffs[ind[k]][j]
            eqConsts[ind[i]] = eqConsts[ind[i]] - mult * eqConsts[ind[k]]

def SPPBackSubstSingle(eqCoeffs, eqConsts, sol, ind):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0]) -1
    sol[n] = eqConsts[ind[n]] / eqCoeffs[ind[n]][n]
    for i in range (n - 1, -1, -1):
        sum1 = eqConsts[ind[i]]
        for j in range (i + 1, n+1):
            sum1 = sum1 - eqCoeffs[ind[i]][j] * sol[j]
        sol[i] = sum1 / eqCoeffs[ind[i]][i]
    solSingle = np.array(sol, dtype=np.float32)
    
    outputFile = "outputSPPSingle.sol"
    with open(outputFile, 'w') as file:
        file.write(str(solSingle))
    print("SPP Single ", solSingle)

def SPPGaussianSingle(eqCoeffs, eqConsts):
    checkCoeff(eqCoeffs)
    n = len(eqCoeffs[0])
    sol = [0.0] * n
    ind = [0] * n
    for i in range (n):
        ind[i] = i
    SPPFwdEliminationSingle(eqCoeffs,eqConsts,ind)
    SPPBackSubstSingle(eqCoeffs,eqConsts,sol,ind)

fileInput = sys.argv[1]
# options = sys.argv[2:]
    
testCoeff1, testConst1 = fromFile(fileInput)
NaiveGaussianSingle(testCoeff1, testConst1)
SPPGaussianSingle(testCoeff1, testConst1)
NaiveGaussianDouble(testCoeff1, testConst1)
SPPGaussianDouble(testCoeff1, testConst1)

# if '--double' in options and '--spp' in options:
#     SPPGaussianDouble(testCoeff1, testConst1)
# elif '--spp'in options:
#     SPPGaussianSingle(testCoeff1, testConst1)
# elif '--double' in options:
#     NaiveGaussianDouble(testCoeff1, testConst1)
# else:
#     NaiveGaussianSingle(testCoeff1, testConst1)

