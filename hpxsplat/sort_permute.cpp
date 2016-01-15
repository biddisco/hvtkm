//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iterator>
#include <chrono>
//
#include <boost/random/uniform_int_distribution.hpp>
//
#define HPX_REDUCE_BY_KEY_TEST_SIZE (1 << 18)
//

#undef msg
#define msg(a,b,c,d) \
        std::cout \
        << std::setw(60) << a << std::setw(12) <<  b \
        << std::setw(40) << c << std::setw(30) \
        << std::setw(8)  << #d << "\t";


class Timer {
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::milliseconds milliseconds;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
public:
    explicit Timer(bool run = false)
    {
        if (run)
            Reset();
    }
    void Reset()
    {
        _start = high_resolution_clock::now();
    }
    fsec Elapsed() const
    {
        return std::chrono::duration_cast<fsec>(high_resolution_clock::now() - _start);
    }
    template <typename T, typename Traits>
    friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
    {
        return out << timer.Elapsed().count();
    }
private:
    high_resolution_clock::time_point _start;
};

// rearrange vector using a separate permutation array
template <typename IndexType, typename DataType>
void rearrange(std::vector<DataType> &data, const std::vector<IndexType> &perms)
{
    std::vector<bool> done(perms.size(), false);
    for (IndexType i = 0; i < static_cast<IndexType>(perms.size()); i++) {
        if (!done[i]) {
            DataType t = data[i];
            for (IndexType j = i;;) {
                done[j] = true;
                if (perms[j] != i) {
                    //std::cout << "swapping " << j << " with perms["<<j<<"] " << perms[j] << "\n";
                    data[j] = data[perms[j]];
                    j = perms[j];
                } else if (i!=j) {
                    //std::cout << "swapping " << j << " with " << i << "\n";
                    data[j] = t;
                    break;
                } else {
                    break;
                }
            }
        }
    }
}

// rearrange vector using a separate permutation array
template <typename IndexType, typename DataType>
void rearrange_2(std::vector<DataType> &data, const std::vector<IndexType> &perms)
{
    std::vector<bool> done(perms.size(), false);
    for (IndexType i = 0; i < static_cast<IndexType>(perms.size()); i++) {
        if (!done[i]) {
            DataType t = data[i];
            for (IndexType j = i;;) {
                done[j] = true;
                if (perms[j] != i) {
                    //std::cout << "swapping " << j << " with perms["<<j<<"] " << perms[j] << "\n";
                    data[j] = data[perms[j]];
                    j = perms[j];
                } else {
                    //std::cout << "swapping " << j << " with " << i << "\n";
                    data[j] = t;
                    break;
                }
            }
        }
    }
}

namespace debug {
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
        std::cout << name.c_str() << "\t : {" << std::distance(begin,end) << "} : ";
        std::copy(begin, end,
                  std::ostream_iterator<typename std::iterator_traits<Iter>::value_type>(std::cout, ", "));
        std::cout << "\n";
    }
};


int main(int , char **)
{
    const int test_size = 1000000000;
    std::vector<int> vals(test_size);
    std::vector<int> keys(test_size);

    //debug::output("b original", vals);
    //debug::output("b permute ", keys);
    std::cout << "\n";

    const int avgs = 10;
    std::chrono::duration<double> total1, total2, time1, time2;
    for (int i=0; i<avgs; i++) {
        std::iota(vals.begin(), vals.end(), 0);
        std::iota(keys.begin(), keys.end(), 0);
        std::random_shuffle( keys.begin(), keys.end() );
        Timer timer(true);
        rearrange(vals, keys);
        time1 = timer.Elapsed();
        total1 += time1;
        std::cout << "Equal " << std::equal(keys.begin(), keys.end(), vals.begin()) << " skip " << time1.count() << " \n";

        std::iota(vals.begin(), vals.end(), 0);
        std::iota(keys.begin(), keys.end(), 0);
        std::random_shuffle( keys.begin(), keys.end() );
        Timer timer2(true);
        rearrange_2(vals, keys);
        time2 = timer2.Elapsed();
        total2 += time1;
        std::cout << "Equal " << std::equal(keys.begin(), keys.end(), vals.begin()) << " none " << time2.count() << " \n";
    }

    double t1 = (double)(total1.count())/avgs;
    std::cout << "Elapsed time: " << std::fixed << t1 << "s\n";
    double t2 = (double)(total2.count())/avgs;
    std::cout << "Elapsed time: " << std::fixed << t2 << "s\n";

    //debug::output("a rearranged", vals);

    //
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
