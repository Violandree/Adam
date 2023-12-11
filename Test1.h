unsigned int dim_samples = 30;

std::vector<double> noises = generate_noise(dim_samples, 1234, 0.05);

std::default_random_engine generator(1234);
std::uniform_real_distribution<double> distribution(0.0,3.0);

std::vector<double> x_, y_;
double x, y;

for (std::size_t i = 0; i < dim_samples; ++i){

x = distribution(generator);
x_.push_back(x);

y = 2*x + 1 + noises[i];
y_.push_back(y);

}

std::vector<Monomial> monomials; // (beta0 + beta1 x - y)^2
monomials.push_back (Monomial (1, {0, 0}, {2, 0})); // beta0^2
monomials.push_back (Monomial (1, {2, 0}, {0, 2})); // beta1^2 x^2
monomials.push_back (Monomial (1, {0, 2}, {0, 0})); // y^2
monomials.push_back (Monomial (2, {1, 0}, {1, 1})); // 2 beta0 beta1 x
monomials.push_back (Monomial (-2, {0, 1}, {1, 0})); // -2 beta0 y
monomials.push_back (Monomial (-2, {1, 1}, {0, 1})); // -2 beta1 x y

FunctionRn f(monomials);

std::vector<Point> observations;

for (std::size_t i = 0; i < dim_samples; ++i)
observations.push_back(Point({x_[i], y_[i]}));

std::vector<double> inf_limits = {-1.0, -1.0};
std::vector<double> sup_limits = {3.0, 5.0};
std::vector<double> initial_parameters_ = {0.0, 0.0};

Point exact_solution({1, 2});

const double tolerance = 1e-10;
const unsigned int max_iterations = 1e6;