import numpy as np
from scipy.optimize import approx_fprime
import time
import csv
import os


class GradDesArmijo:
    # Como python no tiene como diferenciar

    def BackTrackingA(self, f: object, a: float, c1: float, rho: float, x_init: list[float], grad: list[float]) -> tuple[float, int]:
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
        MAXITER = 45
        # number of backtraing and check if the conditions never converge
        j: int = 0
        # Set the variable p of gradient
        p: list[float] = -grad
        # set tvalue x in initial value
        x: list[float] = x_init
        # number of backtracking
        if a < 0:
            print("[WARN]: 'a' must be a positive value \n")
            return
        alpha: float = a
        # print(f"function {f(x+alpha*p)} cond {f(x) +
        #                                      c1*alpha*(grad.T@p)} c1gp {c1*grad.T@p}")
        while f(x + alpha*p) > f(x) + c1*alpha*(grad.T@p):

            # print(f"function {f(x+alpha*p)} cond {f(x) +
            #      c1*alpha*(grad.T@p)} alpha {alpha} ")
            alpha = rho*alpha

            j += 1
            if j > MAXITER:
                print(
                    "[WARN]: Maximum number of iterations were reachead, Backtracking don't get a well alpha \n")
                return
        return alpha, j

    def SaveData(self, fname: str, dirname: str, arch: str, ext: str, header: str, data: dict) -> None:
        """
        This function helps to save data in a specific format in selected directory 
        **args**:
            - fname: name of file 
            - dirname: name of directory 
            - arch: 
            - type: type of file csv,txt,dat 
            - data: Data to be saved in csv format 
        """

        try:

            directory: str = dirname
            filename = fname
            filename += f".{ext.lower()}"
            filepath = os.path.join(directory, filename)

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(filepath, arch) as file:
                if ext.lower() == "csv":
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(data)
                elif ext.lower() == "txt":
                    file.write(data)
                elif ext.lower() == "dat":
                    file.write(data)
        except Exception as e:
            raise TypeError(f"[ERROR]: Unexpected error happend {e}")

    def GradientDesc(self, objec_fun: object, points: list[float], BaCond: list[float], epoch: float, tol: float, init_exec: str, filename: str, dirname: str,) -> tuple[dict, list, list]:
        """
        This function calculate the gradient descent of specific function 
        with intial points
        **args**
            - objec_fun: Function to be evaluated for the descent
            - points: initial array of values of objec_fun 
            - BaCond: contias the intial conditions for the implentation of backtracking
            - epoch: total of iterations 
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
            xi: list = np.array(points)
            # open file
            # f = open("excutionDataP1.csv", "w")
            # fieldnames
            # fieldnames = ["init_type","k", "f_k", "grad_norm_k",
            #              "alpha_k", "n_backtracks_k", "time_sec_k"]
            # open a file csv
            rLastExec: dict = {}
            # number of iterations
            iterations = np.array([], dtype=int)
            functionarr = np.array([], dtype=float)
            gradientarr = np.array([], dtype=float)
            alphaarr = np.array([], dtype=float)
            backtrakingarr = np.array([], dtype=float)
            allData = np.array([])
            for i in range(1, epoch+1):
                resumen: dict = {}
                # start looping time
                loop_start = time.perf_counter()
                g: list = approx_fprime(xi, objec_fun, 10E-4)
                # direction of maximum descent
                p: list = -g
                # Check the tolerance
                normg: float = np.linalg.norm(g)

                if normg < tol:
                    print(
                        "value of norm is under of tolerance. Gradient Descendent end \n")
                    resumen["stop_reason"] = "grad_tolerance"
                    break
                # calculate the alpha using the backtraking with armijo

                alpha, n_backtracks = self.BackTrackingA(objec_fun, a=BaCond[0],
                                                         c1=BaCond[1],
                                                         rho=BaCond[2],
                                                         x_init=xi,
                                                         grad=g)

                # Check if the function no return any value
                if alpha is None:
                    rLastExec["stop_reason"] = "alpha in armijo is None value"
                    raise TypeError(
                        "[ERROR]: the loop reach the maximum iter or alpha not has a real value\n")

                elif np.isnan(objec_fun(xi)):  # prevent overflow
                    raise TypeError("[ERROR]: function overflow value")
                # make a array and save all values of alpha
                alphaarr = np.append(alphaarr, alpha)
                # save al number of backtracking
                backtrakingarr = np.append(backtrakingarr, n_backtracks)
                # save iterations
                iterations = np.append(iterations, i)
                # save function evalaution
                functionarr = np.append(functionarr, objec_fun(xi))
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
                resumen["f_final"] = objec_fun(xi)
                resumen["grad_final"] = normg
                resumen["alpha"] = alpha
                resumen["backtraking"] = n_backtracks
                # LastExecution
                rLastExec["init_type"] = init_exec
                rLastExec["iters"] = i
                rLastExec["time_total"] = cumulative_time
                rLastExec["f_final"] = objec_fun(xi)
                rLastExec["grad_final"] = normg
                rLastExec["alpha"] = alpha
                rLastExec["backtraking"] = n_backtracks
                # resumen["stop_reason"] = "max_iter"

                allData = np.append(allData, resumen)
                if i % 100 == 0:
                    print(f"k={i} f_k={objec_fun(
                        xi):.010f} grad_norm_k={normg:10e} alpha_k={alpha:.010f} n_backtracks_k={n_backtracks} time_sec_k={loop_duration:.10f}")

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
