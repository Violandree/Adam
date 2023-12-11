//
// Created by Roberto Sala on 12/09/23.
//

#include "FunctionRn.h"
#include <iostream>

double FunctionRn::evaluate(const Point &observation, const Point &parameters) const {

    double value = 0.0;

    for (const auto & monomial: monomials)
        value += monomial.evaluate(observation, parameters);

    return value;

}


void FunctionRn::addMonomial(const Monomial &monomial) {

    monomials.push_back(monomial);

}


double FunctionRn::evaluate_partial_derivative(std::size_t j, const Point &observation, const Point &parameters) const {

    Point P1 (parameters.get_coordinates());
    Point P2 (parameters.get_coordinates());

    P1.set_coordinate(j, P1.get_coordinate(j) + h);
    P2.set_coordinate(j, P2.get_coordinate(j) - h);

    return (evaluate(observation, P1) - evaluate(observation, P2)) / (2*h);

}


void FunctionRn::set_monomials(const std::vector<Monomial> &monomials_) {

    for (const auto & monomial_: monomials_)
        addMonomial(monomial_);

}


const std::vector<Monomial> & FunctionRn::get_monomials() const {

    return monomials;

}


void FunctionRn::print(const std::vector<std::string> &variables_name,
                       const std::vector<std::string> &parameters_name) const {

    for (const auto & monomial_: monomials) {
        monomial_.print(variables_name, parameters_name);
        std::cout << " ";
    }

    std::cout << "+ eps" << std::endl;

}