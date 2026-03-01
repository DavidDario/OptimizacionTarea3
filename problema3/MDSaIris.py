# imported libaries
import pandas as pd
import numpy as np
import time
import os
import csv
# Own libraries
from problema1 import GradDesArmijo
from problema2 import RosenbrockAltaDim


class MDSAIris:
    def Irisgraph(self):
        irispath = '../OptimizacionTarea3/problema3/iris.csv'
        X = pd.read_csv(irispath).iloc[:, 0:4].values
        Species = pd.read_csv(irispath).iloc[:, 4].values

    def Stress(self, D: list[list[float]], x_init: list[float],
               m_element: int, m_component) -> float:
        r"""
        This calcualte the stress function $\frac{1}{2}\sum_{j>i}{{d_{ij} - \letf\| z_i-z_j \right\|}$
        **args**:
            - D: Matrix of distance
            - x_init: intial points linealized
            - m_element: number of elements in the x_init
            - n_component: number of component in the inital
        **returns**:
            - the evaluation of stress function
        """
        # reshape the intial vector
        z = np.array(x_init).reshape(m_element, m_component)
        sum = 0
        for i in range(m_element):
            for j in range(i+1, m_element):
                # sum the partial norms
                sum += (D[i][j]-np.linalg.norm(z[i]-z[j]))**2
        return 0.5*sum

    def gradStres(self, D: list[list], x0: list[float],
                  m_element: int, m_component) -> list[float]:
        r"""
        Calculate the gradient function of stress fucntion $\nabla f = \sum_{j>i}{({d_{ij} - \letf\|z_i-z_j \right\|) \letf\|z_i-z_j \right\|}$
        $\varepsilon= 1\times 10^{-6}$ thid prevent the 0 divison
        """
        try:

            z = np.array(x0).reshape(m_element, m_component)
            g = np.zeros_like(z)
            for i in range(m_element):
                for j in range(i+1, m_element):
                    diff = z[i]-z[j]
                    zij = np.linalg.norm(diff)
                    if zij < 10e-13:
                        continue
                    # sum the partial norms
                    factor = (D[i][j]-zij) / zij
                    g[i] -= factor*diff
                    g[j] += factor*diff
            return g.ravel()
        except ZeroDivisionError as e:
            raise TypeError(f"[ERROR]: error in divison gradient {e}")
        except Exception as e:
            raise TypeError(f"[ERROR]: Unexpected error {e}")

    def BackTrackingA(self, f: object, a: float,
                      c1: float, rho: float,
                      x_init: list[float], grad: list[float],
                      D: list[list[float]], n_element: int,
                      n_component: int
                      ) -> tuple[float, int]:
        """
        This function calculate the aplha value using backtracing
        **args**:
            - a: intial value of backtracking
            - rho: a number betwen 0,1
            - c1: another number between 0,1
            - x: array of values
            - g: gradient
        **returns**
            - alpha: value of backtraking check the wolfe condition
            - n: number of backtrakings
        """
        MAXITER = 55
        # number of backtraing and check if the conditions never converge
        j: int = 0
        # set tvalue x in initial value
        x: list[float] = x_init
        # Set the variable p of gradient
        p: list[float] = -np.array(grad)
        # print(grad)
        # print(x+a*p)
        # number of backtracking
        if a < 0:
            print("[WARN]: 'a' must be a positive value \n")
            return
        alpha: float = a
        # print(f"function {f(x+alpha*p)} cond {f(x) +
        #                                      c1*alpha*(grad.T@p)} c1gp {c1*grad.T@p}")
        # print("frist func", f(D, x + alpha*p, n_element))
        # print("second funcsi", f(D, x, n_element) + c1*alpha*(grad.T@p))
        # print(grad.T@grad)
        while (f(D, x + alpha*p, n_element, n_component)
               > f(D, x, n_element, n_component) + c1*alpha*(grad.T@p)):

            # print(f"function {f(x+alpha*p)} cond {f(x) +
            #      c1*alpha*(grad.T@p)} alpha {alpha} ")
            alpha = rho*alpha

            j += 1
            if j > MAXITER:
                print(
                    "[WARN]: Maximum number of iterations were reachead in Backtracking alpha is 0 \n")
                return alpha, j

        return alpha, j

    def GradientDesc(self, objec_fun: object, grad: list[float], points: list[float],
                     BaCond: list[float],
                     D: list[list[float]],
                     n_element: int,
                     n_component: int,
                     epoch: float, tol: float,
                     init_exec: str,
                     filename: str, dirname: str,) -> tuple[dict, list, list]:
        """
        This function calculate the gradient descent of specific function
        with intial points
        **args**
            - objec_fun: Function to be evaluated for the descent
            - points: initial array of values of objec_fun
            - BaCond: contias the intial conditions for the implentation of backtracking
            - epoch: total of iterations
            - init_exec: type of execution
            - tol: maximum tolerance in method
            - filename: name of file to save data
            - dirname: place where the data would be save
        **returns**:
            - dictionary with ifno of last iteration
            - array: number of iterations.
            - array: function evaluated in x_k
            - array: gradient function
        """

        try:
            # global timer
            total_start_time: time = time.perf_counter()
            # intialize the gradient calc using the the approx_fprime
            xi: list[float] = np.array(points)
            # save last information
            rLastExec: dict = {}
            # number of iterations
            iterations:     list[int] = np.array([], dtype=int)
            functionarr:    list[float] = np.array([], dtype=float)
            gradientarr:    list[float] = np.array([], dtype=float)
            alphaarr:       list[float] = np.array([], dtype=float)
            backtrakingarr: list[float] = np.array([], dtype=float)
            allData:        list = np.array([])

            for i in range(1, epoch+1):
                resumen: dict = {
                    "init_type": init_exec,
                    "iters": 0,
                    "time_total": 0.0,
                    "f(x_k)": 0.0,
                    "grad_final": 0.0,
                    "alpha": 0.0,
                    "backtraking": 0,
                    "stop_reason": ""
                }
                # start looping time
                loop_start: time = time.perf_counter()
                g: list[float] = grad(D, xi, n_element, n_component)
                # direction of maximum descent
                p: list[float] = -g
                # Check the tolerance
                normg: float = np.linalg.norm(g)
                if normg < tol:
                    print(
                        "value of norm is under of tolerance. Gradient Descendent end \n")
                    rLastExec["stop_reason"] = "grad_tolerance"
                    resumen["stop_reason"] = "grad_tolerance"
                    break
                # calculate the alpha using the backtraking with armijo
                alpha, n_backtracks = self.BackTrackingA(f=objec_fun,
                                                         a=BaCond[0],
                                                         c1=BaCond[1],
                                                         rho=BaCond[2],
                                                         x_init=xi,
                                                         grad=g,
                                                         D=D,
                                                         n_element=n_element,
                                                         n_component=n_component)
                # Check if the function no return any value
                if alpha < 10e-15:
                    rLastExec["stop_reason"] = "alpha is 0"
                    resumen["stop_reason"] = "alpha is 0"
                    allData = np.append(allData, resumen)
                    print(
                        "[ERROR]: the loop reach the maximum iter or alpha not has a real value\n")
                    break
                elif np.isnan(objec_fun(D, xi, n_element, n_component)):  # prevent overflow
                    raise TypeError("[ERROR]: function overflow value")
                # make a array and save all values of alpha
                alphaarr = np.append(alphaarr, alpha)
                # save al number of backtracking
                backtrakingarr = np.append(backtrakingarr, n_backtracks)
                # save iterations
                iterations = np.append(iterations, i)
                # save function evalaution
                functionarr = np.append(
                    functionarr, objec_fun(D, xi, n_element, n_component))
                # save gradient evaluation
                gradientarr = np.append(gradientarr, normg)
                # print(f"xi {xi}  p {p} alpha {alpha} alpha*p {alpha*p}")
                # update the value xi using the conditons in gradient descent
                xi = xi + alpha*p
                # start a contunter to end time in each epoch
                loop_end = time.perf_counter()
                # epoch duration
                loop_duration = loop_end - loop_start
                # maximum time during the execution
                cumulative_time = loop_end - total_start_time
                # get make a itarated table

                # writer = csv.DictWriter(f, fieldnames=fieldnames)
                # writer.writeheader()
                # writer.writerow({"init_type":"classical","k": i, "f_k": objec_fun(xi), "grad_norm_k": normg,
                #                "alpha_k": alpha, "n_backtracks_k": n_backtracks, "time_sec_k": loop_duration})
                resumen["init_type"] = init_exec
                resumen["iters"] = i
                resumen["time_total"] = cumulative_time
                resumen["f(x_k)"] = objec_fun(D, xi, n_element, n_component)
                resumen["grad_final"] = normg
                resumen["alpha"] = alpha
                resumen["backtraking"] = n_backtracks
                resumen["stop_reason"] = ""
                # LastExecution
                rLastExec["init_type"] = init_exec
                rLastExec["iters"] = i
                rLastExec["time_total"] = cumulative_time
                rLastExec["f_final"] = objec_fun(D, xi, n_element, n_component)
                rLastExec["grad_final"] = normg
                rLastExec["alpha"] = alpha
                rLastExec["backtraking"] = n_backtracks
                rLastExec["stop_reason"] = ""

                allData = np.append(allData, resumen)

                print(f"k={i} f_k={objec_fun(D,
                                             xi, n_element, n_component):.010f} grad_norm_k={normg:10e} alpha_k={alpha:.010f} n_backtracks_k={n_backtracks} time_sec_k={loop_duration:.10f}")
            rLastExec["stop_reason"] = "max_iter"
        except ValueError as e:
            ValueError(f"[ERROR]: You introduce a non numerical Value {e}")

        except OverflowError as e:
            # this is in case a calculate overflow time or calculus
            ValueError(f"[ERROR]: Overflow  in evalution function {e} \n")
            ValueError("The calculation is to large to be executed \n")

        except ZeroDivisionError as e:
            ValueError(f"[ERROR]: You cannot divide by zero {e}")

        except Exception as e:
            # Any other kind of error cannot revised in the exception
            print(f"[ERROR]: Miscelaneous error {e} \n")
        finally:
            # print(allData)
            file = filename + ".csv"
            dirs = f"../OptimizacionTarea3/{dirname}"
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            complete_f = os.path.join(dirs, file)
            header = [name for name in allData[0]]
            file = open(complete_f, "w")
            writer = csv.DictWriter(file, fieldnames=header)  # header of row
            writer.writeheader()
            writer.writerows(allData)  # write all data
            return rLastExec, iterations, functionarr, gradientarr, alphaarr, backtrakingarr

    def IrisMDS(self, seed: int, n_component: int, Conditions: list[float], epoch:
                int, tol, type_exec: str, Nfile: str, dirname: str) -> tuple:
        """
        This function calculate the MDS for irisi using the gradient descent with armijo
        **args**:
            - seed: Initial seed.
            - n_component: size of vector
            - Condition: Conditions form armijo
            - epoch: maximum number of iterations
        **returns**:
        tuple with element to be calculated and doing graph
        """
        try:
            sol1 = GradDesArmijo.GradDesArmijo()
            sol2 = RosenbrockAltaDim.RosenbrockAltaDim()
            # set the default seed for random init
            rds = np.random.default_rng(seed)
            # path to csv info of iris
            irispath = '../OptimizacionTarea3/problema3/iris.csv'
            # get data of iris
            X: list[float] = pd.read_csv(irispath).iloc[:, 0:4].values
            # get the size of matrix iris
            l: int = np.array(X).shape[0]

            # get the specieis
            Species: list[str] = pd.read_csv(irispath).iloc[:, 4].values
            # start a zero matrix
            delta: list[list[float]] = np.zeros((l, l), dtype=float)

            # Random intializtion variables using gaussian distribution
            # intialize a vector of size n_component
            x_inital: list[float] = rds.normal(
                loc=80, scale=15, size=(l, n_component)).flatten()
            # call the gradient descendent function with armijo Conditions

            for i in range(l):
                for j in range(l):
                    delta[i][j] = np.linalg.norm(X[i]-X[j])

            # abreviation to reference functions into the class
            # assign the initial values to array
            getVal = self.GradientDesc(self.Stress, self.gradStres, x_inital,
                                       Conditions,
                                       delta, l, n_component, epoch,
                                       tol, type_exec, Nfile, dirname)

            header = []
            lastPrint = ""
            for elem in getVal[0]:
                lastPrint += f" {elem} = {getVal[0][elem]} "
                header.append(elem)
            print("The las iteration")
            print(lastPrint, end="\n\n")

            # name = "lastExecution"
            # dir = "IrisLastExec"
            # fileT = "a"
            # ileExt = "csv"
            # This funciton generate a directory and save the last execution adding these at end each execution
            # graph of variable

            iter = getVal[1]
            func = getVal[2]
            gradf = getVal[3]
            alphas = getVal[4]
            # print(getVal[0])
            # sol1.SaveData(name, dir, fileT, FileExt, header, getVal[0])

            dirphotos = "../OptimizacionTarea3/graphsIrisMDS/"
            # Graphics using original resultf for the function
            sol2.Graph(
                iter, func, r"Variaciones de $MDS$ a lo largo de las iteraciones",
                "k", r"$f(x_k)$", f"f_k_graph_{Nfile}.png", dirphotos)
            sol2.Graph(np.array([np.log10(x) for x in iter], dtype=float), func,
                       r"Evaluacion de $MDS$ pero utilizando logaritmo",
                       r"$\log_{10}{k}$", r"$f(x_k)$", f"f_k_graph_log_{Nfile}.png", dirphotos)

            # Variacione en alpha
            sol2.Graph(iter, alphas, r"Variacion de las $\alpha_k$ a lo largo de las iteraciones",
                       r"$k$", r"$\alpha_k$", f"alpha_graph_{Nfile}.png", dirphotos)

            sol2.Graph(np.array([np.log10(x) for x in iter]), alphas,
                       r"Variaciones de $\alpha_k$ a lo largo de las iteraciones utilizando logaritmo",
                       r"$\log_{10}{k}$", r"$\alpha_k$", f"alpha_graph_log_{Nfile}.png", dirphotos)
            # Graphics using the log in the axis

            # Graph using the original data to check the gradient decrease
            sol2.Graph(iter, gradf, r"Variación del gradiene de $MDS$ a lo largo de las iteraciones",
                       r"k", r"$\nabla f(x_k)$", f"gradient_f_graph_{Nfile}.png", dirphotos)
            # Graph us the log in axis to check the behavior of gradient
            sol2.Graph(np.array(list(map(lambda x: np.log10(x), iter)), dtype=float), gradf,
                       r"Variaciones del gradiente $MDS$ a lo largo de las iteracione utilizando log",
                       r"$\log_{10}{k}$", r"$\nabla f(x_k)$", f"gradient_f_graph_log_{Nfile}.png", dirphotos)

            return getVal[0]
        except Exception as e:
            raise TypeError(f"[ERROR]: Unepected error happend {e}")
