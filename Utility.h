std::vector<double> generate_noise (const unsigned int n_samples, const unsigned int seed, const double noise){

    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,noise);

    std::vector<double> noises;

    for (std::size_t i = 0; i < 100; ++i)
        noises.push_back(distribution(generator));

    return noises;

}