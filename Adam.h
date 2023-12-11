//
// Created by Roberto Sala on 12/09/23.
//

#ifndef CHALLENGE1_APC_2023_ADAM_H
#define CHALLENGE1_APC_2023_ADAM_H

#include <vector>
#include <random>
#include "FunctionRn.h"
#include "Point.h"


class Adam {

private:
    FunctionRn f;
    std::vector<Point> observations;

    unsigned int dim_batch;

    double tolerance;
    double alpha = 0.001;
    double gamma1 = 0.9;
    double gamma2 = 0.999;
    double epsilon = 1e-8;
    unsigned int max_iterations = 0;
    std::vector<double> inf_limits;
    std::vector<double> sup_limits;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;

    // Create the batch of points used in the algorithm. The function returns the indices of the observations to be considered
    std::vector<int> create_batch ();

    // Evaluate the stochastic function in a batch of points
    double evaluate_batch (const Point &parameters, const std::vector<int> & batch) const;

    // Evaluate the partial derivative of the stochastic function wrt the j-th parameter in a batch of points
    double evaluate_partial_derivative_batch (std::size_t j, const Point &parameters, const std::vector<int> & batch) const;

public:
    // Constructors
    Adam () = default;
    Adam (FunctionRn & f_, std::vector<Point> & observations_,
          unsigned int dim_batch_, double tolerance_,
          unsigned int max_iterations_, std::vector<double> & inf_limits_, std::vector<double> & sup_limits_)
            : f(f_), observations(observations_), dim_batch(dim_batch_), tolerance(tolerance_),
              max_iterations(max_iterations_), inf_limits(inf_limits_), sup_limits(sup_limits_) {

        generator = std::default_random_engine(30042023);
        distribution = std::uniform_int_distribution<int>(0, observations.size()-1);

    };

    // Getter and setters
    void set_f (const FunctionRn & f_);
    void set_observations (const std::vector<Point> & observations_);
    void set_dim_batch (unsigned int dim_batch_);
    void set_tolerance (double tolerance_);
    void set_max_iterations (unsigned int max_iterations_);
    void set_inf_limits (const std::vector<double> & inf_limits_);
    void set_sup_limits (const std::vector<double> & sup_limits_);
    const FunctionRn & get_f () const;
    const std::vector<Point> & get_observations () const;
    unsigned int get_dim_batch () const;
    double get_tolerance () const;
    unsigned int get_max_iterations () const;
    const std::vector<double> & get_inf_limits () const;
    const std::vector<double> & get_sup_limits () const;

    // Apply the Adam method
    Point solve (const Point & initial_parameters);

};


#endif //CHALLENGE1_APC_2023_ADAM_H
