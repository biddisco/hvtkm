//  Copyright (c) 2014-2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef __SIMPLE_PROFILER_HPP
#define __SIMPLE_PROFILER_HPP

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>
#include <hpx/util/spinlock.hpp>
//
#include <boost/thread/locks.hpp>
#include <boost/format.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <tuple>
#include <cstdio>

#if defined(HPX_HAVE_APEX)
# include <hpx/util/apex.hpp>sdf
sdsd
f
ssdf
#endif

namespace hpx {
    namespace util {

//----------------------------------------------------------------------------
// an experimental profiling class which times sections of code and collects
// the timing from a tree of profiler objects to break down the time spend
// in nested sections
//
        class simple_profiler {
        public:
            // time, level, count
            typedef std::tuple<double, int, int> valtype;

            simple_profiler(const char *title) {
                _title = title;
                _done = false;
                _iterations = 1;
            }

            simple_profiler(std::shared_ptr<hpx::util::simple_profiler> parent,
                            const char *title) {
                _parent = parent;
                _title = title;
                _done = false;
                _iterations = 1;
            }

            ~simple_profiler() {
                if (!_done)
                    done();
            }

            void set_iterations(int64_t i) {
                _iterations = i;
            }

            void done() {
                double elapsed = _timer.elapsed();
                if (_parent) {
                    _parent->addProfile(_title, std::make_tuple(elapsed, 0, 1));
                    std::for_each(_profiles.begin(), _profiles.end(),
                                  [=](std::map<const char *, valtype>::value_type &p) {
                                      _parent->addProfile(p.first, p.second);
                                  }
                    );
                }
                else {
                    // get the max depth of the profile tree so we can prepare string lengths
                    int maxlevel = 0;
                    std::for_each(_profiles.begin(), _profiles.end(),
                                  [&](std::map<const char *, valtype>::value_type &p) {
                                      maxlevel = (std::max)(maxlevel,
                                                            std::get<1>(p.second));
                                  }
                    );
                    // prepare format string for output
                    char const *fmt0 = "%20s : %2s %5s %-9s\n";
                    char const *fmt1 = "%20s : %2i %5.3g %12g %s%7.3f";
                    std::string spre = std::string(44, ' ');
                    std::string fmt2 = " %7.3f%%";
                    std::string line = std::string(52 + maxlevel * 8, '-') + "\n";
                    std::string last_lev_string;

                    // add this level to top of list
                    _profiles[_title] = std::make_tuple(elapsed, 0, 1);
                    // print each of the sub nodes
                    std::vector<double> level_totals(maxlevel + 1, 0);
                    int last_level = 0;
                    hpx::cout
                    << line
                    << (boost::format(fmt0) % "Func" % "lv" % "count" % "time")
                    << line;
                    for (auto p = _profiles.begin(); p != _profiles.end();) {
                        int &level = std::get<1>(p->second);
                        level_totals[level] += std::get<0>(p->second);
                        // if closing a level : print total from sub-level
                        if (level < last_level) {
                            hpx::cout
                            << spre << last_lev_string << " -------\n"
                            << spre << last_lev_string
                            << (boost::format(fmt2)
                                % (100.0 * level_totals[last_level] / elapsed)) << "\n";
                            last_level = level;
                            last_lev_string = std::string(last_level * 8, ' ');
                        }
                            //
                        else if (level > last_level) {
                            last_level = level;
                            last_lev_string = std::string(last_level * 8, ' ');
                        }
                        hpx::cout << (boost::format(fmt1)
                                      % p->first
                                      % level
                                      % static_cast<float>(std::get<2>(p->second) /
                                                           _iterations)
                                      % static_cast<float>(std::get<0>(p->second) /
                                                           _iterations)
                                      % std::string(level * 8, ' ')
                                      % (100.0 * std::get<0>(p->second) / elapsed)) <<
                        "\n";
                        if ((++p) == _profiles.end()) {
                            hpx::cout
                            << spre << last_lev_string << " -------\n"
                            << spre << last_lev_string
                            << (boost::format(fmt2)
                                % (100.0 * level_totals[last_level] / elapsed)) << "\n";
                            last_level = level;
                            last_lev_string = std::string(last_level * 8, ' ');
                        }
                    }
                    hpx::cout << line;
                }
                _done = true;
            }

            void addProfile(const char *title, valtype value) {
                boost::lock_guard<hpx::util::spinlock> l(_profiler_mtx);
                //
                if (_profiles.find(title) == _profiles.end()) {
                    std::get<1>(value) += 1;                 // level
                    _profiles[title] = value;
                }
                else {
                    valtype &val = _profiles[title];
                    std::get<0>(val) += std::get<0>(value); // time
                    std::get<1>(val) = (std::max)(std::get<1>(value),
                                        std::get<1>(val)); // level
                    std::get<2>(val) += std::get<2>(value); // count
                }
            }

            //
            std::shared_ptr<hpx::util::simple_profiler> _parent;
            hpx::util::high_resolution_timer _timer;
            const char *_title;
            int64_t _iterations;
            std::map<const char *, valtype> _profiles;
            bool _done;
            hpx::util::spinlock _profiler_mtx;
        };
    }
} // namespace hpx:::util

typedef std::shared_ptr<hpx::util::simple_profiler> simple_profiler_ptr;

namespace hpx {
    namespace util {
        simple_profiler_ptr make_profiler(const char *title) {
            return std::make_shared<simple_profiler>(title);

        }

        simple_profiler_ptr make_profiler(simple_profiler_ptr p, const char *title) {
            return std::make_shared<simple_profiler>(p, title);

        }
    }
}
#endif
