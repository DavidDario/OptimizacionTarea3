import numpy as np
import os
from problema1 import GradDesArmijo
import matplotlib.pyplot as plt


class RosenbrockAltaDim:
    # Como python no tiene como diferenciar

    def Rosenbrock(self, X: list[float]) -> float:
        """
        Esta función toma los elementos en de un arreglo
        y los suma para formarla funcion de _Rosenbrock

        ** args**:
            - X: Vector formado por n variables.
        """

        return np.sum(100.0*(X[1:]-X[:-1]**2)**2+(1.0-X[:-1])**2)

    def ClassicalInit(self, x_2i1: float, x_2i: float, n: int, Conditions: list[float], epoch: int, tol: float, exec, NameFile: str):
        """
        Esta funcion utiliza la inicialiación clasica
        **args**:
            - x_2i1: asgina valores a terminos impares en un arreglo
            - x_2i: asigna valores pares en un arreglo
            - n: amount of elements in initial array
            - Conditions: Proposed conditions by GradientDesc and the armijo.
            - epoch:  max number of iterations in the problem
            - tol: Tolerance for problem convergence
        **returns**:
            - tuple that contain in the following order
            - dict: data of last iteration
            - array: list of number of iterations
            - array: list of function evaluations
            - array: list of gradient evalautions
        """
        # vector o zeros
        try:
            # intialize a 0 vector
            R: list[float] = np.zeros(2*n, dtype=float)
            # call the gradient descendent function with armijo Conditions
            sol1 = GradDesArmijo.GradDesArmijo()
            # abreviation to reference functions into the class
            # assign the initial values to array
            for i in range(n):
                R[2*i+1] = x_2i1
                R[2*i] = x_2i
            f = self.Rosenbrock
            # eturn the solution of gradient descent
            return sol1.GradientDesc(objec_fun=f, points=R,
                                     BaCond=Conditions, epoch=epoch, tol=tol, init_exec=exec, filename=NameFile, dirname="DataAnalysis")

        except Exception as e:
            ValueError(f"[ERROR]: An unexpected error happen {e} \n")

    def randomInit(self, seed: int, Conditions: list[float], epoch: int, tol: float) -> tuple[dict, list, list, list]:
        """
        This funcion generates an random initialization to rosenblock function.
        Using the normal distribution
        **args**
            - seed: this is an integer number for fix the random values
            - Conditions: Proposed conditions by GradientDesc and the armijo.
            - epoch:  max number of iterations in the problem
            - tol: Tolerance for problem convergence
        **returns**:
            - tuple that contain in the following order
            - dict: data of last iteration
            - array: list of number of iterations
            - array: list of function evaluations
            - array: list of gradient evalautions
        """
        try:
            rds = np.random.default_rng(seed)
            # Random intializtion variables using gaussian distribution
            x_2i1 = rds.normal(loc=0, scale=7)
            x_2i = rds.normal(loc=0, scale=7)

            n = np.abs(int(rds.normal(loc=85, scale=25)))
            # intialize a 0 vector
            R: list[float] = np.zeros(2*n, dtype=float)
            # call the gradient descendent function with armijo Conditions
            sol1 = GradDesArmijo.GradDesArmijo()
            # abreviation to reference functions into the class
            # assign the initial values to array
            for i in range(n):
                R[2*i+1] = x_2i1
                R[2*i] = x_2i
            f = self.Rosenbrock
            # eturn the solution of gradient descent
            return sol1.GradientDesc(objec_fun=f, points=R,
                                     BaCond=Conditions, epoch=epoch,
                                     tol=tol, init_exec=f"{seed}",
                                     filename=f"Executionfile_{seed}", dirname="RandomExecutions")
        except Exception as e:
            raise TypeError(
                f"[ERROR]: Error in execution of random init {e} \n")

    def Graph(self, x: list, y: list, title: str, xt: str, yt: str, name: str, dirname: str) -> None:
        """
        This function make graph in 2d of data is introduced

        **args**:
            - x: list of floting or integer numbers
            - y: list of floating or integer numbers
            - title: title of the graph
            - xt set the x label
            - yt set the y label
        **returns**
            - None
        """

        plt.figure(figsize=(10, 8))
        plt.plot(x, y)
        plt.xlabel(xt)
        plt.ylabel(yt)
        plt.title(title)
        plt.grid(True)

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filepath = os.path.join(dirname, name)

        plt.savefig(filepath)
        plt.close()

        # plt.show()

    def Ejercicio2a(self, x_even: float, x_odd: float, N: int, ArmijoCond: list[float], epoch: int, tol: float, Nfile: str) -> None:
        """
        This  function makes an evaluation for each intial values
            **args**:
                - x_even: set the even values
                - x_odd : set the odd vaulues
                - N : set the size of intial array
                - ArmijoCond : armijo conditions
                - epoch : maximum number of iterations
                - tol: tolerance in the method
                - Nfile : Name of file to export in csv
        """

        try:
            sol1 = GradDesArmijo.GradDesArmijo()
            # evaluate the function
            getVal = self.ClassicalInit(
                x_odd, x_even, N, ArmijoCond, epoch, tol, "Classical", Nfile)
            lastPrint = ""
            # This header would be used in print files
            header = []
            for elem in getVal[0]:
                lastPrint += f" {elem} = {getVal[0][elem]} "
                header.append(elem)
            print("Last ejecution")
            print(lastPrint, end="\n\n")

            # name = "lastExecution"
            # dir = "ModClassicalExec"
            # fileT = "a"
            # FileExt = "csv"
            # This funciton generate a directory and save the last execution adding these at end each execution
            # graph of variable

            iter = getVal[1]
            func = getVal[2]
            gradf = getVal[3]
            alphas = getVal[4]

            # sol1.SaveData(name, dir, fileT, FileExt, header, getVal[0])

            dirphotos = "../OptimizacionTarea3/graphsa/"
            # Graphics using original resultf for the function
            self.Graph(
                iter, func, r"Variaciones de $f_k$ a lo largo de las iteraciones",
                "k", r"$f(x_k)$", f"f_k_graph_{Nfile}.png", dirphotos)
            self.Graph(np.array([np.log10(x) for x in iter], dtype=float), func,
                       r"Evaluacion de $f_k$ pero utilizando logaritmo",
                       r"$\log_{10}{k}$", r"$f(x_k)$", f"f_k_graph_log_{Nfile}.png", dirphotos)

            # Variacione en alpha
            self.Graph(iter, alphas, r"Variacion de las $\alpha_k$ a lo largo de las iteraciones",
                       r"$k$", r"$\alpha_k$", f"alpha_graph_{Nfile}.png", dirphotos)

            self.Graph(np.array([np.log10(x) for x in iter]), alphas,
                       r"Variaciones de $\alpha_k$ a lo largo de las iteraciones utilizando logaritmo",
                       r"$\log_{10}{k}$", r"$\alpha_k$", f"alpha_graph_log_{Nfile}.png", dirphotos)

            # Graphics using the log in the axis

            # Graph using the original data to check the gradient decrease
            self.Graph(iter, gradf, r"Variación del gradiene a lo largo de las iteraciones",
                       r"k", r"$\nabla f(x_k)$", f"gradient_f_graph_{Nfile}.png", dirphotos)
            # Graph us the log in axis to check the behavior of gradient
            self.Graph(np.array(list(map(lambda x: np.log10(x), iter)), dtype=float), gradf,
                       r"Variaciones de $\nabla f_k$ a lo largo de las iteracione utilizando log",
                       r"$\log_{10}{k}$", r"$\nabla f(x_k)$", f"gradient_f_graph_log_{Nfile}.png", dirphotos)
            return getVal[0]
        except Exception as e:
            raise TypeError(f"An error happen in execution {e}")

    def Ejercicio2b(self, seed: float, ArmijoCond: list[float], epoch: int, tol: float) -> None:
        try:
            sol1 = GradDesArmijo.GradDesArmijo()
            # evaluate the function

            getVal = self.randomInit(seed, ArmijoCond, epoch, tol)
            lastPrint = ""
            header = []
            for em in getVal[0]:
                lastPrint += f"{em}={getVal[0][em]} "
                header.append(em)
            print(lastPrint, end="\n\n")
            # graph of variable
            iter = getVal[1]
            func = getVal[2]
            gradf = getVal[3]
            alphas = getVal[4]
            # name = "lastExecution"
            # dir = "ModRandomExec"
            # fileT = "a"
            dirphotos = "../OptimizacionTarea3/graphsb/"
            # FileExt = "csv"
            # This funciton generate a directory and save the last execution adding these at end each execution
            # sol1.SaveData(name, dir, fileT, FileExt, header, getVal[0])

            # Graphics using original resultf for the function
            self.Graph(
                iter, func, r"Variaciones de $f_k$ a lo largo de las iteraciones", "k", r"$f(x_k)$", f"f_graph_{seed}.png", dirphotos)
            self.Graph(np.array([np.log10(x) for x in iter], dtype=float), func, r"Evaluacion de $f_k$ pero utilizando logaritmo",
                       r"$\log_{10}{k}$", r"$f(x_k)$", f"log_f_graph_{seed}.png", dirphotos)

            # Variacione en alpha
            self.Graph(iter, alphas, r"Variacion de las $\alpha_k$ a lo largo de las iteraciones",
                       r"$k$", r"$\alpha_k$", f"alpha_graph_{seed}.png", dirphotos)

            self.Graph(np.array([np.log10(x) for x in iter]), alphas,
                       r"Variaciones de $\alpha_k$ a lo largo de las iteraciones utilizando logaritmo", r"$\log_{10}{k}$", r"$\alpha_k$", f"log_alpha_graph_{seed}.png", dirphotos)

            # Graphics using the log in the axis

            # Graph using the original data to check the gradient decrease
            self.Graph(iter, gradf, r"Variación del gradiene a lo largo de las iteraciones",
                       r"k", r"$\nabla f(x_k)$", f"gradient_f_graph_{seed}.png", dirphotos)
            # Graph us the log in axis to check the behavior of gradient
            self.Graph(np.array(list(map(lambda x: np.log10(x), iter)), dtype=float), gradf,
                       r"Variaciones de $\nabla f_k$ a lo largo de las iteracione utilizando log",
                       r"$\log_{10}{k}$", r"$\nabla f(x_k)$", f"log_gradient_f_graph_{seed}.png", dirphotos)
            return getVal[0]
        except Exception as e:
            raise TypeError(f"An error happen in execution {e}")
