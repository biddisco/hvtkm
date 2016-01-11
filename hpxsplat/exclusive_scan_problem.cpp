//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/parallel/algorithms/sort_by_key.hpp>
#include <hpx/parallel/algorithms/prefix_scan.hpp>
#include <hpx/parallel/algorithms/reduce_by_key.hpp>
#include <iostream>

// use smaller array sizes for debug tests
#if defined(HPX_DEBUG)
#define HPX_SORT_TEST_SIZE          512
#define HPX_SORT_TEST_SIZE_STRINGS  50000
#endif

#include "sort_tests.hpp"

//
#define DEBUG_OUTPUT
//
namespace debug {
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
#endif
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << std::distance(begin,end) << "} : ";
        std::copy(begin, end,
                  std::ostream_iterator<typename std::iterator_traits<Iter>::value_type>(std::cout, ", "));
        std::cout << "\n";
#endif
    }
};

struct tester {
    double A;
    int    B;
    tester(double a=0.0, int b=0) : A(a), B(b) {}
/*
    tester(const tester &x)  : A(x.A), B(x.B) {}
    tester (tester && ) = default;
    tester& operator = (const tester &x) {
        A = x.A;
        B = x.B;
        return *this;
    }
*/
};

std::ostream& operator << (std::ostream& os, const tester &t) {
    os << "{" << t.A << "," << t.B << "}";
    return os;
}
////////////////////////////////////////////////////////////////////////////////
// call reduce_by_key with no comparison operator
template<typename ExPolicy>
void test_scan(ExPolicy && policy)
{
    static_assert(
            hpx::parallel::is_execution_policy<ExPolicy>::value,
            "hpx::parallel::is_execution_policy<ExPolicy>::value");

    tester dummy{3.4, 5};

    // Fill vector with random values
    std::vector<tester> values(HPX_SORT_TEST_SIZE);
    std::fill(values.begin(), values.end(), tester( (double)std::rand(), std::rand() ));

    // output
    debug::output<tester>("\nvalues", values);

    boost::uint64_t t = hpx::util::high_resolution_clock::now();

    // reduce_by_key, blocking when seq, par, par_vec
    hpx::parallel::inclusive_scan(
            std::forward<ExPolicy>(policy),
            values.begin(),
            values.end(),
            values.begin(),
            //dummy,
            tester(3.4, 5),
            [](tester x, tester y) {
                return tester(x.A + y.A, x.B + y.B);
            });

/*
    // reduce_by_key, blocking when seq, par, par_vec
    hpx::parallel::inclusive_scan(
            std::forward<ExPolicy>(policy),
            values.begin(),
            values.end(),
            values.begin(), tester(3.4, 5), [](const tester &x, const tester &y){
                return tester(x.A+y.A, x.B+y.B);
            });
*/

    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    // output
    debug::output<tester>("\nscan values", values);

    HPX_TEST(true);
}

////////////////////////////////////////////////////////////////////////////////
void test_scan()
{
    using namespace hpx::parallel;

    test_scan(par);
//    test_scan(par_vec, int());
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_scan();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
