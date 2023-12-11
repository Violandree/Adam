//
// Created by Roberto Sala on 12/09/23.
//

#ifndef CHALLENGE1_APC_2023_FUNCTIONRN_H
#define CHALLENGE1_APC_2023_FUNCTIONRN_H

#include "Monomial.h"
#include "Point.h"


class FunctionRn {

private:
    std::vector<Monomial> monomials;
    static constexpr double h = 0.00001;

public:
    FunctionRn () = default;
    FunctionRn (std::vector<Monomial> & monomials_) : monomials(monomials_) {};

    // Evaluate the function given the values of the observation (x,y) and the parameters (beta_0, beta_1)
    double evaluate (const Point & observation, const Point & parameters) const;

    // Add a monomial to the function
    void addMonomial (const Monomial & monomial);

    // Evaluate the partial derivative wrt variable j in the observation point (x,y), given the parameters (beta_0, beta_1)
    double evaluate_partial_derivative (std::size_t j, const Point & observation, const Point & parameters) const;

    // Getters and setters
    void set_monomials (const std::vector<Monomial> & monomials_);
    const std::vector<Monomial> & get_monomials () const;

    // Print the function in a readable way
    void print (const std::vector<std::string> & variables_name, const std::vector<std::string> & parameters_name) const;

};


#endif //CHALLENGE1_APC_2023_FUNCTIONRN_H
