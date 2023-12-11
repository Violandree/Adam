#include <iostream>
#include "Point.h"
#include "Monomial.h"
#include "FunctionRn.h"
#include "Adam.h"
#include <random>
# include "Utility.h"
#include <iomanip>


int main() {

    //#include "Test1.h" // (beta0 + beta1 x - y)^2 -> beta0 = 1.03, beta1 = 1.99
    //#include "Test2.h" // (beta0 + beta1 x^2 + beta2 x - y)^2 -> beta0 = -2, beta1 = -3, beta2 = 4
    //#include "Test2B.h" // (beta0 + beta1 x^2 + beta2 x - y)^2 -> should not converge
    #include "Test3.h" // (beta0 + beta1 x + beta2 y + beta3 x y - z)^2 -> beta0 = 0.999, beta1 = 3, beta2 = 2, beta3 = -4

    Adam adam(f, observations, 5, tolerance, max_iterations, inf_limits, sup_limits);
    Point P = adam.solve(initial_parameters_);
    std::cout << "Best parameters found: ";
    P.print();

    double distance = P.distance(exact_solution);
    std:: cout << "Distance from exact solution: " << std::setprecision(3) << distance;

    if (distance < 0.05)
        std::cout << " --> CONVERGED to the exact solution";
    else
        std::cout << " --> NOT CONVERGED to the exact solution";

    return 0;
}
