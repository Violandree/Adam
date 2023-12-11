unsigned int dim_samples = 50;

std::vector<double> noises = generate_noise(dim_samples, 2212, 0.02);

std::default_random_engine generator(2212);
std::uniform_real_distribution<double> distribution(-5.0,5.0);

std::vector<double> x_, y_;
double x, y;

for (std::size_t i = 0; i < dim_samples; ++i){

x = distribution(generator);
x_.push_back(x);

y = -3*x*x + 4*x -2 + noises[i];
y_.push_back(y);

}

std::vector<Monomial> monomials; // (beta0 + beta1 x^2 + beta2 x - y)^2
monomials.push_back (Monomial (1, {0, 0}, {2, 0, 0}));
monomials.push_back (Monomial (1, {4, 0}, {0, 2, 0}));
monomials.push_back (Monomial (1, {2, 0}, {0, 0, 2}));
monomials.push_back (Monomial (1, {0, 2}, {0, 0, 0}));
monomials.push_back (Monomial (2, {2, 0}, {1, 1, 0}));
monomials.push_back (Monomial (2, {1, 0}, {1, 0, 1}));
monomials.push_back (Monomial (-2, {0, 1}, {1, 0, 0}));
monomials.push_back (Monomial (2, {3, 0}, {0, 1, 1}));
monomials.push_back (Monomial (-2, {2, 1}, {0, 1, 0}));
monomials.push_back (Monomial (-2, {1, 1}, {0, 0, 1}));


FunctionRn f(monomials);

std::vector<Point> observations;

for (std::size_t i = 0; i < dim_samples; ++i)
observations.push_back(Point({x_[i], y_[i]}));

std::vector<double> inf_limits = {-8.0, -7.0, 0.0};
std::vector<double> sup_limits = {2.0, 4.0, 10.0};
std::vector<double> initial_parameters_ = {0.0, 0.0, 5.0};

Point exact_solution({-2, -3, 4});

const double tolerance = 1e-10;
const unsigned int max_iterations = 1e6;