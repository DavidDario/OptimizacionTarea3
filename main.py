# space import libraries
from problema1 import GradDesArmijo
from problema2 import RosenbrockAltaDim
from problema3 import MDSaIris
import numpy as np


def Ejercicio1() -> None:
    """
    in this function you can customize the variables impleented in the problem1
    """
    try:
        # Call the library where the functions live
        sol1: object = GradDesArmijo.GradDesArmijo()
        # Set an example function
        def f(X) -> list[float]: return np.sum(X**2)
        # initial value of graddient descent
        x_init: list[float] = np.array([10, 2, 5], dtype=float)
        # Conditions of backtraking
        Conditions: list[float] = np.array([1, 10E-4, 0.5])
        # number of epoch in the implementation
        epoch: int = 10000
        # tolerance, the code has one but is possible select a different
        tol: float = 10E-6

        getval = sol1.GradientDesc(f, x_init, Conditions, epoch, tol)
    except OverflowError as e:
        ValueError(f"overflow {e}")
    except Exception as e:
        ValueError(f"Unepected error {e}")
    print(getval)


def initCondEjer2a():
    # Example of parameter function
    # Call the class of second problem
    sol1 = GradDesArmijo.GradDesArmijo()
    sol2 = RosenbrockAltaDim.RosenbrockAltaDim()
    # set the intial values
    # even elements in the array
    xeve: float = -1.2
    # odd elements in the array
    xodd: float = 1.0
    # half size of array in Rosenbrock function
    n: int = 64
    # conditions to implement the gradient descendent
    Conditions: list[float] = np.array([1, 10E-4, 0.5], dtype=float)
    # Maximum number of epoch
    epoch: int = 30
    # tolerance
    tol: float = 10E-6

    sol2.Ejercicio2a(x_even=xeve, x_odd=xodd, N=n,
                     ArmijoCond=Conditions, epoch=epoch, tol=tol, Nfile="test1")

    # multiexecution function
    # If can run this please uncomment  the loopfor

    # json of intial conditions
    init = [
        {
            "x_even": -1.2,
            "x_odd": 1.0,
            "n": 64,
            "conditions": np.array([1, 10E-4, 0.5], dtype=float),
            "epoch": 30,
            "tol": 10E-6
        },
        {
            "x_even": -2.2,
            "x_odd": 1.3,
            "n": 80,
            "conditions": np.array([1, 10E-4, 0.5], dtype=float),
            "epoch": 30,
            "tol": 10E-6
        },
        {
            "x_even": -2.2,
            "x_odd": 1.3,
            "n": 80,
            "conditions": np.array([1, 10E-4, 0.5], dtype=float),
            "epoch": 30,
            "tol": 10E-6
        },
    ]
    finalresult = []
    header = []
    k = 0
    for val in init:
        filename = f"Exp_class_exec_{k}"
        result = sol2.Ejercicio2a(x_even=val["x_even"], x_odd=val["x_odd"], N=val["n"],
                                  ArmijoCond=val["conditions"], epoch=val["epoch"], tol=val["tol"], Nfile=filename)
        finalresult.append(result)
        for elem in result:
            header.append(elem)

        k += 1
    header = [k for k in finalresult[0]]
    print(header)
    sol1.SaveData("ResultofExecutionClassicalInit",
                  "/OptimizacionTarea3/DataAnalysis/ModClassicalExec/", "w", "csv", header, finalresult)


def initCondEjer2b():
    sol1 = GradDesArmijo.GradDesArmijo()
    sol2 = RosenbrockAltaDim.RosenbrockAltaDim()
    seed = 80825
    # conditions to implement the gradient descendent
    Conditions: list[float] = np.array([1, 10E-4, 0.5], dtype=float)
    # Maximum number of epoch
    epoch: int = 30
    # tolerance
    tol: float = 10E-6
    lse = [
        {
            "seed": 12352343,
            "Conditions": np.array([1, 10E-4, 0.5], dtype=float)
        },
        {
            "seed": 4235645,
            "Conditions": np.array([1, 10E-4, 0.5], dtype=float)
        },
        {
            "seed": 678563,
            "Conditions": np.array([1, 10E-4, 0.5], dtype=float)
        }
    ]
    finalresult = []
    header = []
    for elem in lse:
        result = sol2.Ejercicio2b(elem["seed"], Conditions, epoch, tol)
        finalresult.append(result)

    header = [k for k in finalresult[0]]
    sol1.SaveData("ResultofExecutionRandomInit",
                  "/OptimizacionTarea3/DataAnalysis/ModRandomExec/", "w", "csv", header, finalresult)


def initCondEje3() -> None:
    sol1 = GradDesArmijo.GradDesArmijo()
    sol3 = MDSaIris.MDSAIris()
    t = 0
    finalresult = []
    header = []
    for seed in [4123, 1234, 678, 4564]:
        result = sol3.IrisMDS(seed, 2, [1, 10E-4, 0.5], 5, 10e-5,
                              f"{1223}", f"experiment_with_seed_{4121}", "irisMDSataAnalysis")
        finalresult.append(result)
        for elem in result:
            header.append(elem)

        t += 1
    header = [k for k in finalresult[0]]
    sol1.SaveData("ResultofExecutionMDSExecution",
                  "/OptimizacionTarea3/DataAnalysis/MDSLastExecution/", "w", "csv", header, finalresult)


def main() -> None:
    """
    Funcion principal, su proposito es unicamente ejecutar las funciones 
    """
    initCondEjer2a()
    initCondEjer2b()
    initCondEje3()


main()
