//
// Created by Roberto Sala on 12/09/23.
//

#ifndef CHALLENGE1_APC_2023_MONOMIAL_H
#define CHALLENGE1_APC_2023_MONOMIAL_H

#include <vector>
#include <cmath>
#include "Point.h"

class Monomial {

private:
    double coefficient;
    std::vector<double> powers_observations;
    std::vector<double> powers_parameters;

public:

    // Constructors
    Monomial () = default;
    Monomial (double coefficient_, std::vector<double> const & powers_observations_, std::vector<double> const & powers_parameters_)
            : coefficient (coefficient_), powers_observations (powers_observations_), powers_parameters (powers_parameters_) {};

    // Evaluate the monomial given the values of the observation (x,y) and the parameters (beta_0, beta_1)
    double evaluate (const Point & observation, const Point & parameters) const;

    // Print the monomial in a readable way
    void print (const std::vector<std::string> & variables_name, const std::vector<std::string> & parameters_name) const;

    // Get the power of the i-th variable
    double get_power_observation (std::size_t i) const;

    // Get the power of the i-th parameter
    double get_power_parameter (std::size_t i) const;

    // Getter and setters
    void set_coefficient (double coefficient_);
    void set_powers_observations (const std::vector<double> & powers_observations_);
    void set_powers_parameters (const std::vector<double> & powers_parameters_);
    double get_coefficient () const;
    const std::vector<double> & get_powers_observations () const;
    const std::vector<double> & get_powers_parameters () const;

};


#endif //CHALLENGE1_APC_2023_MONOMIAL_H
