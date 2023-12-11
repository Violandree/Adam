//
// Created by Roberto Sala on 12/09/23.
//

#include "Monomial.h"
#include <iostream>


double Monomial::evaluate(const Point & observation, const Point & parameters) const {

    double value = coefficient;

    for (std::size_t i = 0; i < powers_parameters.size(); ++i)
        value *= pow(parameters.get_coordinate(i), powers_parameters[i]);

    for (std::size_t j = 0; j < powers_observations.size(); ++j)
        value *= pow(observation.get_coordinate(j), powers_observations[j]);

    return value;

}


void Monomial::print(const std::vector<std::string> &variables_name, const std::vector<std::string> &parameters_name) const {

    if (coefficient != 0.0){

        if (coefficient != 1.0){
            if (coefficient > 0)
                std::cout << "+" << coefficient << " ";
            else
                std::cout << coefficient << " ";
        }
        else
            std::cout << "+" << " ";

        for (std::size_t i = 0; i < powers_parameters.size(); ++i){
            if (powers_parameters[i] != 0.0){
                if (powers_parameters[i] != 1.0)
                    std::cout << parameters_name[i] << "^" << powers_parameters[i] << " ";
                else
                    std::cout << parameters_name[i] << " ";
            }
        }

        for  (std::size_t j = 0; j < powers_observations.size(); ++j){
            if (powers_observations[j] != 0.0){
                if (powers_observations[j] != 1.0)
                    std::cout << variables_name[j] << "^" << powers_observations[j] << " ";
                else
                    std::cout << variables_name[j] << " ";
            }
        }

    }

}


double Monomial::get_power_observation(std::size_t i) const {

    return powers_observations[i];

}


double Monomial::get_power_parameter(std::size_t i) const {

    return powers_parameters[i];

}


void Monomial::set_coefficient(double coefficient_) {

    coefficient = coefficient_;

}


void Monomial::set_powers_observations(const std::vector<double> &powers_observations_) {

    powers_observations = powers_observations_;

}


void Monomial::set_powers_parameters(const std::vector<double> &powers_parameters_) {

    powers_parameters = powers_parameters_;

}


double Monomial::get_coefficient() const {

    return coefficient;

}


const std::vector<double> & Monomial::get_powers_observations() const {

    return powers_observations;

}


const std::vector<double> & Monomial::get_powers_parameters() const {

    return powers_parameters;

}
