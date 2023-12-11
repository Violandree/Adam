//
// Created by Roberto Sala on 12/09/23.
//

#include "Adam.h"
#include <random>
#include <cmath>


std::vector<int> Adam::create_batch() {

    std::vector<int> indices;

    for (std::size_t i = 0; i < dim_batch; ++i)
        indices.push_back(distribution(generator));

    return indices;

}


double Adam::evaluate_batch(const Point &parameters, const std::vector<int> & batch) const {

    double value = 0.0;

    for (const auto i : batch)
        value += f.evaluate(observations[i], parameters);

    return value / batch.size();

}


double Adam::evaluate_partial_derivative_batch(std::size_t j, const Point &parameters, const std::vector<int> & batch) const {

    double value = 0.0;

    for(const auto i : batch) {
        value += f.evaluate_partial_derivative(j, observations[i], parameters);
    }

    return value / batch.size();

}


void Adam::set_f(const FunctionRn &f_) {

    f = f_;

}


void Adam::set_observations(const std::vector<Point> &observations_) {

    observations = observations_;

}


void Adam::set_dim_batch(unsigned int dim_batch_) {

    dim_batch = dim_batch_;

}


void Adam::set_tolerance(double tolerance_) {

    tolerance = tolerance_;

}


void Adam::set_max_iterations(unsigned int max_iterations_) {

    max_iterations = max_iterations_;

}


void Adam::set_inf_limits(const std::vector<double> &inf_limits_) {

    inf_limits = inf_limits_;

}


void Adam::set_sup_limits(const std::vector<double> &sup_limits_) {

    sup_limits = sup_limits_;

}


const FunctionRn & Adam::get_f() const {

    return f;

}


const std::vector<Point> & Adam::get_observations() const {

    return observations;

}


unsigned int Adam::get_dim_batch() const {

    return dim_batch;

}


double Adam::get_tolerance() const {

    return tolerance;

}


unsigned int Adam::get_max_iterations() const {

    return max_iterations;

}


const std::vector<double> & Adam::get_inf_limits() const {

    return inf_limits;

}


const std::vector<double> & Adam::get_sup_limits() const {

    return sup_limits;

}


Point Adam::solve(const Point &initial_parameters) {

    /// Your code goes HERE ///
    const std::vector<double> zero(initial_parameters.get_dimension(),0.0);

    std::vector<Point> theta = {initial_parameters};
    Point m;
    m.set_coordinates(zero);
    Point v;
    v.set_coordinates(zero);
    Point mHat;
    Point vHat;
    Point g;

    unsigned dim = initial_parameters.get_dimension();
    unsigned t = 0;
    std::vector<int> batch;
    bool converged = false;

    while (! converged){
        ++t;
        batch = create_batch();
        theta.push_back(theta[t-1]);
        for (size_t j = 0; j < dim; ++j){
            g.set_coordinate(j, evaluate_partial_derivative_batch(j,theta[t-1],batch));
            m.set_coordinate(j, gamma1 * m.get_coordinate(j) + (1 - gamma1) * g.get_coordinate(j));
            mHat.set_coordinate(j, m.get_coordinate(j) / (1 - pow(gamma1,t)));
            v.set_coordinate(j, gamma2 * v.get_coordinate(j) + (1 - gamma2) * pow(g.get_coordinate(j),2));
            vHat.set_coordinate(j, v.get_coordinate(j) / (1 - pow(gamma2,t)));
            theta[t].set_coordinate(j,theta[t-1].get_coordinate(j) - alpha * m.get_coordinate(j) / (sqrt(v.get_coordinate(j)) + epsilon));
        }

        if (theta[t].distance(theta[t-1]) < tolerance ||
            t >= max_iterations ||
            std::abs(evaluate_batch(theta[t-1],batch) - evaluate_batch(theta[t],batch)) < tolerance )
            converged = true;
    }

    return theta[t];
}
