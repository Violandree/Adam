unsigned int dim_samples = 100;

std::vector<double> noises = generate_noise(dim_samples, 9999, 0.01);

std::default_random_engine generator(9999);
std::uniform_real_distribution<double> distribution1(-2.0,6.0);
std::uniform_real_distribution<double> distribution2(1.0,4.0);

std::vector<double> x_, y_, z_;
double x, y, z;

for (std::size_t i = 0; i < dim_samples; ++i){

x = distribution1(generator);
x_.push_back(x);

y = distribution2(generator);
y_.push_back(y);

z = 1 + 3*x + 2*y -4*x*y + noises[i];
z_.push_back(z);

}

std::vector<Monomial> monomials; // (beta0 + beta1 x + beta2 y + beta3 x y - z)^2
monomials.push_back (Monomial (1, {0, 0, 0}, {2, 0, 0, 0}));
monomials.push_back (Monomial (1, {2, 0, 0}, {0, 2, 0, 0}));
monomials.push_back (Monomial (1, {0, 2, 0}, {0, 0, 2, 0}));
monomials.push_back (Monomial (1, {2, 2, 0}, {0, 0, 0, 2}));
monomials.push_back (Monomial (1, {0, 0, 2}, {0, 0, 0, 0}));
monomials.push_back (Monomial (2, {1, 0, 0}, {1, 1, 0, 0}));
monomials.push_back (Monomial (2, {0, 1, 0}, {1, 0, 1, 0}));
monomials.push_back (Monomial (2, {1, 1, 0}, {1, 0, 0, 1}));
monomials.push_back (Monomial (-2, {0, 0, 1}, {1, 0, 0, 0}));
monomials.push_back (Monomial (2, {1, 1, 0}, {0, 1, 1, 0}));
monomials.push_back (Monomial (2, {2, 1, 0}, {0, 1, 0, 1}));
monomials.push_back (Monomial (-2, {1, 0, 1}, {0, 1, 0, 0}));
monomials.push_back (Monomial (2, {1, 2, 0}, {0, 0, 1, 1}));
monomials.push_back (Monomial (-2, {0, 1, 1}, {0, 0, 1, 0}));
monomials.push_back (Monomial (-2, {1, 1, 1}, {0, 0, 0, 1}));


FunctionRn f(monomials);

std::vector<Point> observations;

for (std::size_t i = 0; i < dim_samples; ++i)
observations.push_back(Point({x_[i], y_[i], z_[i]}));

std::vector<double> inf_limits = {-1.0, -1.0, 1.0, -8.0};
std::vector<double> sup_limits = {5.0, 4.0, 4.0, -2.0};
std::vector<double> initial_parameters_ = {0.0, 0.0, 3.0, -7.0};

Point exact_solution({1, 3, 2, -4});

const double tolerance = 1e-10;
const unsigned int max_iterations = 1e6;