//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define _USE_MATH_DEFINES 
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/zip_iterator.hpp>
#include <hpx/util/transform_iterator.hpp>
//
#include <hpx/parallel/algorithms/sort_by_key.hpp>
#include <hpx/parallel/algorithms/reduce_by_key.hpp>
//
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
//
#include "sort_tests.hpp"
#include <vector>
#include <array>
#include <random>
#include <tuple>
#include <functional>
#include <iterator>
//
#include "splatkernels/Gaussian.h"
#include "simple_profiler.hpp"

#ifdef HPX_HAVE_VTK
# include "vtkPVConfig.h"
# include <vtkImageData.h>
# include <vtkSmartPointer.h>
# include <vtkImageImport.h>
# include <vtkMarchingCubes.h>
# include <vtkPolyData.h>
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleSwitch.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#include "vtkMPI.h"
#include "vtkMPIController.h"
#include "vtkMPICommunicator.h"
//
#endif
//
#define DEBUG_OUTPUT
//
namespace debug {
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v)
    {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
#endif
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end)
    {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << std::distance(begin, end) << "} : ";
        std::copy(begin, end,
            std::ostream_iterator<typename std::iterator_traits<Iter>::value_type>(
                std::cout, ", "));
        std::cout << "\n";
#endif
    }
};

// ----------------------------------------------------------------------------
struct options
{
    unsigned int seed{12345};
    unsigned int chains{1};
    unsigned int fchunk{1000};
    unsigned int points{10};
    bool         render{0};

    const int volume_dimension = 16;
    const int kernel_radius = 1;
    const int kernel_scale = 25.0;
#ifdef HPX_HAVE_VTK
    vtkSmartPointer<vtkMPIController> controller;
#endif
};

options global_options;
// ----------------------------------------------------------------------------
template<typename T> using CoordType = std::array<T, 3>;
using PointType = CoordType<double>;
//
#ifndef HPX_HAVE_VTK
# define vtkIdType int64_t 
#endif

typedef std::array<vtkIdType, 3> Id3Type;
typedef std::tuple<Id3Type, Id3Type> MinMaxTuple;
//
template<typename T> using vit = typename std::vector<T>::iterator;

//
// ----------------------------------------------------------------------------
template<typename kernel>
class kernel_footprint
{
private:
    kernel kernel_;
    std::array<double, 3> origin_;
    std::array<double, 3> spacing_;
    Id3Type dimension_;

public:
    kernel_footprint(const kernel &k, const std::array<double, 3> &o,
        const std::array<double, 3> &s, const Id3Type &dim) : kernel_(k), origin_(o),
                                                              spacing_(s),
                                                              dimension_(dim) { }

    template<typename T, typename T2>
    void operator()(const CoordType<T> &worldcoord, const T2 &h, PointType &splatPoint,
        MinMaxTuple &minmaxtuple, vtkIdType &footprintSize) const
    {
        footprintSize = 1;
        double cutoff = kernel_.maxDistance(h);
        for (int i = 0; i < 3; i++) {
            splatPoint[i] = (worldcoord[i] - this->origin_[i]) / this->spacing_[i];
            std::get<0>(minmaxtuple)[i] = (std::max)(
                static_cast<vtkIdType>(ceil(splatPoint[i] - cutoff)), vtkIdType(0));
            std::get<1>(minmaxtuple)[i] = (std::min)(
                static_cast<vtkIdType>(floor(splatPoint[i] + cutoff)),
                this->dimension_[i] - 1);
            footprintSize = footprintSize * (1 + std::get<1>(minmaxtuple)[i] -
                                             std::get<0>(minmaxtuple)[i]);
        }
    }
};

// ----------------------------------------------------------------------------
static std::atomic<int> compute_index{1};

//
template<typename kernel>
struct kernel_compute
{
    kernel kernel_;
    std::array<double, 3> origin_;
    std::array<double, 3> spacing_;
    Id3Type dimension_;

    double Radius2;
    double ExponentFactor;
    double ScalingFactor;
    // Kernel kernel;

    kernel_compute(const kernel &k, const std::array<double, 3> &orig,
        const std::array<double, 3> &s, const Id3Type &dim)
    // const Kernel &k)
        : kernel_(k), spacing_(s), origin_(orig), dimension_(dim) { } // , kernel(k) { }

    template<typename T2, typename P>
    void operator()(const CoordType<P> &splatPoint, MinMaxTuple &minmaxtuple,
        const T2 &kernel_H, const T2 &scale, const vtkIdType localNeighborId,
        vtkIdType &neighborVoxelId, float &splatValue) const
    {
        vtkIdType yRange = 1 + std::get<1>(minmaxtuple)[1] - std::get<0>(minmaxtuple)[1];
        vtkIdType xRange = 1 + std::get<1>(minmaxtuple)[0] - std::get<0>(minmaxtuple)[0];
        vtkIdType divisor = yRange * xRange;
        vtkIdType i = localNeighborId / divisor;
        vtkIdType remainder = localNeighborId % divisor;
        vtkIdType j = remainder / xRange;
        vtkIdType k = remainder % xRange;
        // note the order of k,j,i
        Id3Type voxel = {std::get<0>(minmaxtuple)[0] + k, std::get<0>(minmaxtuple)[1] + j,
                         std::get<0>(minmaxtuple)[2] + i};
        PointType dist = {(splatPoint[0] - voxel[0]) * spacing_[0],
                          (splatPoint[1] - voxel[1]) * spacing_[0],
                          (splatPoint[2] - voxel[2]) * spacing_[0]};
        double dist2 = std::inner_product(std::begin(dist), std::end(dist),
            std::begin(dist), 0.0);
        // Compute splat value using the kernel distance_squared function
        splatValue = kernel_.w2(kernel_H, dist2);
        //
        neighborVoxelId =
            (voxel[2] * dimension_[0] * dimension_[1]) + (voxel[1] * dimension_[0]) +
            voxel[0];
        if (neighborVoxelId < 0) neighborVoxelId = -1; else if (neighborVoxelId >=
                                                                dimension_[0] *
                                                                dimension_[1] *
                                                                dimension_[2])
            neighborVoxelId = dimension_[0] * dimension_[1] * dimension_[2] - 1;
    }
};

// ----------------------------------------------------------------------------
// rearrange vector using a separate permutation array
template<typename T1, typename T2>
void rearrange(std::vector<T1> &data, const std::vector<T2> &perms)
{
    std::vector<bool> done(perms.size(), false);
    for (int i = 0; i < perms.size(); i++) {
        if (!done[i]) {
            T1 t = data[i];
            for (int j = i; ;) {
                done[j] = true;

                if (perms[j] != i) {
                    data[j] = data[perms[j]];
                    j = perms[j];
                } else {
                    data[j] = t;
                    break;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// rearrange vector using a separate permutation array
template<typename T2, typename RanIter1>
void rearrange(RanIter1 listbegin, const std::vector<T2> &perms)
{
    typedef typename std::iterator_traits<RanIter1>::value_type T1;
    std::vector<bool> done(perms.size(), false);
    for (int i = 0; i < perms.size(); i++) {
        if (!done[i]) {
            RanIter1 data_from = std::next(listbegin, i);
            T1 t = *data_from;
            for (int j = i; ;) {
                done[j] = true;
                RanIter1 data_to = std::next(listbegin, j);
                if (perms[j] != i) {
                    *data_to = *std::next(listbegin, perms[j]);
                    j = perms[j];
                } else {
                    *data_to = t;
                    break;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
template<typename ExPolicy, typename T1, typename T2>
void sortByPerm(ExPolicy policy, std::vector<T1> &list1, std::vector<T2> &list2,
    simple_profiler_ptr profiler)
{
    simple_profiler_ptr sort_time = std::make_shared<hpx::util::simple_profiler>(profiler,
        "parallel_sort");
    const auto len = list1.size();
    if (!len || len != list2.size()) throw;

    // create permutation vector
    std::vector<size_t> perms;
    perms.reserve(len);
    for (size_t i = 0; i < len; i++) {
        perms.push_back(i);
    }
    hpx::parallel::sort(policy, perms.begin(), perms.end(), [&](T1 a, T1 b)
    {
        return list1[a] < list1[b];
    });

    {
        simple_profiler_ptr sort = std::make_shared<hpx::util::simple_profiler>(sort_time,
            "permute");
        auto lists_begin = hpx::util::make_zip_iterator(list1.begin(), list2.begin());
        rearrange(lists_begin, perms);
    }
}

// ----------------------------------------------------------------------------
template<typename ExPolicy>
std::pair<std::vector<vtkIdType>, std::vector<float>> ProcessPoints(ExPolicy &&policy,
    vit<PointType> pbegin, vit<PointType> pend, const PointType &origin,
    const PointType &spacing, const Id3Type &dimensions, simple_profiler_ptr prof)
{
    auto policypar = hpx::parallel::par;
    auto policyseq = hpx::parallel::seq;

    vtkIdType N = std::distance(pbegin, pend);
    // array of points that are the transformed coordinates in voxel space
    std::vector<PointType> vox_pts;
    vox_pts.assign(N, PointType{});

    // create a footprint functor
    using gaussian = vtkm::worklet::splatkernels::Gaussian<3>;
    gaussian gaussiansplat{1.0};
    kernel_footprint<gaussian> footprint(gaussiansplat, origin, spacing, dimensions);

    // create an array of min/max elements which will be written to
    std::vector<MinMaxTuple> min_max_list(N, MinMaxTuple());
    // and an array of sizes
    std::vector<vtkIdType> neighbour_size(N, vtkIdType());

    typedef hpx::util::zip_iterator<
        vit<PointType>, vit<PointType>, vit<MinMaxTuple>, vit<vtkIdType>
    > zip_iterator;
    typedef typename zip_iterator::reference zref_1;

    {
        simple_profiler_ptr footprinttime = hpx::util::make_profiler(prof, "Footprint");
        // -----------
        // compute the Footprint information for each input point
        // -----------
        hpx::parallel::for_each(policypar,
            // begin
            hpx::util::make_zip_iterator(pbegin, std::begin(vox_pts),
                std::begin(min_max_list), std::begin(neighbour_size)),
            // end
            hpx::util::make_zip_iterator(pend, std::end(vox_pts), std::end(min_max_list),
                std::end(neighbour_size)),
            // function
            [=, &footprint](zref_1 ref)
            {
                using hpx::util::get;
                footprint(get<0>(ref), global_options.kernel_radius, get<1>(ref),
                    get<2>(ref), get<3>(ref));
            });
        // output
        //debug::output<vtkIdType>("Neighbour sizes", neighbour_size);
    }

    // -----------
    // we want to know how many voxels are affected so we can build a list of contributions
    // -----------
    vtkIdType total_voxpts = neighbour_size.empty() ? 0 : neighbour_size.back();
    std::vector<vtkIdType> exclusive_sum(N);
    {
        simple_profiler_ptr scan = hpx::util::make_profiler(prof, "ScanExclusive");
        exclusive_sum.reserve(N + 1);
        hpx::parallel::exclusive_scan(policypar, std::begin(neighbour_size),
            std::end(neighbour_size), std::begin(exclusive_sum), 0);
        total_voxpts += *std::prev(exclusive_sum.end());
        exclusive_sum.push_back(total_voxpts);
    }
    // output
    //debug::output<vtkIdType>("Exclusive scan", exclusive_sum);
    //std::cout << "total_voxpts is : " << exclusive_sum.back() << "\n";

    // -----------
    // also generate the inclusive scan from the exclusive scan
    // NB. not needed as we use exclusive_scan offset by one
    // -----------
    // output
    //debug::output<vit<vtkIdType>>("inclusive_scan", exclusive_sum.begin()+1, exclusive_sum.end());

    // -----------
    // for each output value, find the point index it comes from
    // -----------
    std::vector<vtkIdType> neighbour_to_id(total_voxpts, 0);
    // we need the zip iterator type
    typedef hpx::util::zip_iterator<
        boost::counting_iterator<vtkIdType>, vit<vtkIdType>> zip_iterator2;
    typedef typename zip_iterator2::reference zref_2;

    {
        simple_profiler_ptr neighbour_time = hpx::util::make_profiler(prof,
            "neighbour_to_id");
        hpx::parallel::for_each(policypar,
            // begin
            hpx::util::make_zip_iterator(boost::counting_iterator<vtkIdType>(0),
                std::begin(neighbour_to_id)),
            // end
            hpx::util::make_zip_iterator(
                boost::counting_iterator<vtkIdType>(total_voxpts),
                std::end(neighbour_to_id)),
            // function
            [&exclusive_sum](zref_2 ref)
            {
                using hpx::util::get;
                get<1>(ref) = std::upper_bound(std::begin(exclusive_sum) + 1,
                    std::end(exclusive_sum), get<0>(ref)) - std::begin(exclusive_sum) - 1;
            });
        // output
        //debug::output<vtkIdType>("neighbour_to_id", neighbour_to_id);
    }

    // -----------
    // create a permutation iterator which gives N for each splat out
    // -----------
    auto p1_begin = boost::make_permutation_iterator(std::begin(neighbour_size),
        std::begin(neighbour_to_id));
    auto p1_end = boost::make_permutation_iterator(std::end(neighbour_size),
        std::end(neighbour_to_id));
    // output
    //debug::output<decltype(p1_begin)>("neighbours permutation", p1_begin, p1_end);

    // -----------
    // create a permutation iterator which gives offsets for each splat
    // -----------
    auto p2_begin = boost::make_permutation_iterator(std::begin(exclusive_sum),
        std::begin(neighbour_to_id));
    auto p2_end = boost::make_permutation_iterator(std::end(exclusive_sum),
        std::end(neighbour_to_id));
    // output
    //debug::output<decltype(p2_begin)>("offsets permutation", p2_begin, p2_end);

    // -----------
    // create a permutation iterator which gives the input voxel point
    // -----------
    auto p3_begin = boost::make_permutation_iterator(std::begin(vox_pts),
        std::begin(neighbour_to_id));
    auto p3_end = boost::make_permutation_iterator(std::end(vox_pts),
        std::end(neighbour_to_id));
    // output
    //debug::output<decltype(p3_begin)>("voxpts permutation", p3_begin, p3_end);

    // -----------
    // create a permutation iterator which gives the input voxel point
    // -----------
    auto p4_begin = boost::make_permutation_iterator(std::begin(min_max_list),
        std::begin(neighbour_to_id));
    auto p4_end = boost::make_permutation_iterator(std::end(min_max_list),
        std::end(neighbour_to_id));
    // output
    //debug::output<decltype(p4_begin)>("minmax tuple permutation", p4_begin, p4_end);

    // -----------
    // Make a transform iterator to compute the local index
    // -----------
    typedef hpx::util::zip_iterator<
        decltype(p1_begin), decltype(p2_begin),
        boost::counting_iterator<vtkIdType>> zip_iterator3;
    typedef typename zip_iterator3::reference zref_3;
    //
    auto t1_begin = hpx::util::make_zip_iterator(p1_begin, p2_begin,
        boost::counting_iterator<vtkIdType>(0));
    //
    auto t2_begin = hpx::util::make_transform_iterator(t1_begin, [](zip_iterator3 it)
    {
        using hpx::util::get;
        return (get<2>(*it) - get<1>(*it)) % get<0>(*it);
    });
    // output
    //debug::output<decltype(t2_begin)>("Array offsets", t2_begin, t2_begin+neighbour_to_id.size());

    // -----------
    // Compute the kernel splat values
    // -----------
    // create a compute functor
    kernel_compute<gaussian> compute(gaussiansplat, origin, spacing, dimensions);

    std::vector<vtkIdType> voxel_ids(total_voxpts);
    std::vector<float> splat_values(total_voxpts);

    typedef hpx::util::zip_iterator<
        vit<PointType>, vit<MinMaxTuple>, decltype(t2_begin), vit<vtkIdType>,
        vit<float>> zip_iterator4;
    typedef typename zip_iterator4::reference zref_4;

    {
        simple_profiler_ptr kernel_compute = hpx::util::make_profiler(prof,
            "kernel_compute");

        hpx::parallel::for_each(policypar,
            // begin
            hpx::util::make_zip_iterator(p3_begin, p4_begin, t2_begin,
                std::begin(voxel_ids), std::begin(splat_values)),
            // end
            hpx::util::make_zip_iterator(p3_end, p4_end, t2_begin + total_voxpts,
                std::end(voxel_ids), std::end(splat_values)), [=, &compute](zref_4 ref)
            {
                using hpx::util::get;
                compute(get<0>(ref), get<1>(ref), global_options.kernel_radius,
                    global_options.kernel_scale, get<2>(ref), get<3>(ref), get<4>(ref));
            });

        // output
        //debug::output<vtkIdType>("voxel_ids ", voxel_ids);
        //debug::output<float>("splat_values ", splat_values);
    }

    {
        simple_profiler_ptr sortbykey = hpx::util::make_profiler(prof, "sort_by_key");
        hpx::parallel::sort_by_key(policypar, std::begin(voxel_ids), std::end(voxel_ids),
            std::begin(splat_values));
    }


//    debug::output<vtkIdType>("sorted voxel_ids ", voxel_ids);
//    debug::output<float>("sorted splat_values ", splat_values);

    {
        simple_profiler_ptr inclusive_scan = hpx::util::make_profiler(prof,
            "inclusive_scan");

        std::vector<float> reduced_splat_values = splat_values;
        hpx::parallel::inclusive_scan(policypar, std::begin(reduced_splat_values),
            std::end(reduced_splat_values), std::begin(reduced_splat_values), 0);
        //debug::output<float>("reduced_splat_values ", reduced_splat_values);
    }

//    std::vector<vtkIdType> outkeys;
//    std::vector<float> outvalues;
//    outkeys = voxel_ids;
//    outvalues = splat_values;
    float max_val;
    std::pair<vit<vtkIdType>, vit<float>> rbk_result;
    {
        simple_profiler_ptr reducebykey = hpx::util::make_profiler(prof, "reduce_by_key");
        rbk_result = hpx::parallel::reduce_by_key(policypar, std::begin(voxel_ids),
            std::end(voxel_ids), std::begin(splat_values), std::begin(voxel_ids),
            std::begin(splat_values));
        //debug::output<vtkIdType>("reduced outkeys ", outkeys);
        //debug::output<float>("reduced outvalues ", outvalues);
        //max_val = *std::max_element(std::begin(outvalues), rbk_result.second);
        //std::cout << "max val is " << max_val << "\n";
    }
    // we don't need the rest of the vector beyond the 'new end'
    voxel_ids.resize(std::distance(std::begin(splat_values), rbk_result.second));
    splat_values.resize(std::distance(std::begin(splat_values), rbk_result.second));
    return std::make_pair(std::move(voxel_ids), std::move(splat_values));
}

// ----------------------------------------------------------------------------
void test_splat()
{
#ifdef HPX_HAVE_VTK
    MPI_Init(0, NULL);
    global_options.controller = vtkSmartPointer<vtkMPIController>::New();
    global_options.controller->Initialize(0, NULL, 1);
#endif

    // size of final volume (cubed)
    const int volume_size = global_options.volume_dimension;
    const int half_size = volume_size / 2;
    // number of points we will create splats for
    const int N = global_options.points;
    // point params

    // create a field array to store the data in
    std::vector<double> fieldArray(volume_size * volume_size * volume_size, 0);
    // origin and spacing of volume
    PointType origin = {0.0, 0.0, 0.0};
    PointType spacing = {1.0, 1.0, 1.0};
    Id3Type dimensions = {volume_size, volume_size, volume_size};

    // random generator between 0 and volume size
    std::default_random_engine gen(global_options.seed);
    std::uniform_real_distribution<> Dx(0, spacing[0] * (volume_size - 1));
    std::uniform_real_distribution<> Dy(0, spacing[1] * (volume_size - 1));
    std::uniform_real_distribution<> Dz(0, spacing[2] * (volume_size - 1));

    hpx::util::high_resolution_timer main_timer;

    simple_profiler_ptr main_loop = hpx::util::make_profiler("Splat");
    // parallel policy, we can use one per algorithm, or share one
    // depending on needs
    auto policypar = hpx::parallel::par;
    auto policyseq = hpx::parallel::seq;
//        .with(hpx::parallel::static_chunk_size(best_chunk_size));
//    auto policy = hpx::parallel::seq;

    // array of {x,y,z} points we will use to splat
    std::vector<PointType> ptsArray;
    {
        simple_profiler_ptr generate = hpx::util::make_profiler(main_loop, "Generate");
        ptsArray.reserve(N);
        std::back_insert_iterator<std::vector<PointType>> back_ins(ptsArray);
        hpx::parallel::generate_n(policypar, back_ins, N, [&]()
        {
            return PointType {Dx(gen), Dy(gen), Dz(gen)};
        });
    }

    hpx::parallel::sort(hpx::parallel::par, ptsArray.begin(), ptsArray.end(),
        [](const PointType &ptA, const PointType &ptB){
        return ptA[2] < ptB[2];
    });

    typedef hpx::util::zip_iterator<vit<vtkIdType>, vit<float>> zip_it;
    typedef zip_it::reference zip_ref;
    //
    vit<PointType> pbegin = std::begin(ptsArray);
    vit<PointType> pend = std::end(ptsArray);
    //
    // create a dummy future we will use as our start result.
    hpx::future<zip_it> final_future = hpx::make_ready_future<zip_it>(zip_it());
    //
    const int numsteps = global_options.chains;
    const int stepsize = ptsArray.size() / numsteps;
    for (int i=0; i<numsteps; i++) {
        vit<PointType> p0 = pbegin + (i * stepsize);
        vit<PointType> p1 = (i == numsteps - 1) ? pend : pbegin + (i + 1) * stepsize;
        //
        typedef std::pair<std::vector<vtkIdType>, std::vector<float>> result_type;
        result_type result;
        {
            simple_profiler_ptr process = hpx::util::make_profiler(main_loop, "Process");
            // run the main processor to generate voxel points/values
            result = std::move(ProcessPoints(policyseq,
                p0, p1, origin,
                spacing, dimensions, process
            ));
        }
        //
        final_future = final_future.then(
            std::bind([&main_loop, &fieldArray](
                    std::vector<vtkIdType> voxel_ids,
                    std::vector<float> outvalues,
                    hpx::future<zip_it> &&f)
                {
                    simple_profiler_ptr field_copy = hpx::util::make_profiler(
                        main_loop, "field_copy"
                    );
                    //
                    f.get();
                    //
                    zip_it zbegin = hpx::util::make_zip_iterator(
                        std::begin(voxel_ids),
                        std::begin(outvalues));
                    zip_it zend = hpx::util::make_zip_iterator(
                        std::end(voxel_ids),
                        std::end(outvalues));
                    //
                    return hpx::parallel::for_each(
                        hpx::parallel::par(hpx::parallel::task).with(hpx::parallel::static_chunk_size(global_options.fchunk)),
                        zbegin, zend, [&fieldArray](zip_ref ref)
                        {
                            vtkIdType id = hpx::util::get<0>(ref);
                            float value = hpx::util::get<1>(ref);
                            fieldArray[id] += value;
                        }
                    );
                },
                std::move(result.first),
                std::move(result.second),
                std::placeholders::_1
            )
        );
    }
    //fieldArray[half_size + (half_size * volume_size) + (half_size * volume_size * volume_size)] = 10.0;

    // wait for the final stage to complete
    final_future.get();
    main_loop->done();

    std::cout << "time " << main_timer.elapsed() << " seconds" << std::endl;

#ifdef HPX_HAVE_VTK
    if (global_options.render) {
        // Convert the c-style image to a vtkImageData
        vtkSmartPointer<vtkImageImport> imageImport = vtkSmartPointer<
            vtkImageImport
        >::New();
        imageImport->SetDataSpacing(spacing.data());
        imageImport->SetDataOrigin(origin.data());
        imageImport
            ->SetWholeExtent(0, volume_size - 1, 0, volume_size - 1, 0, volume_size - 1);
        imageImport->SetDataExtentToWholeExtent();
        imageImport->SetDataScalarTypeToDouble();
        imageImport->SetNumberOfScalarComponents(1);
        imageImport->SetImportVoidPointer(fieldArray.data(), true);
        imageImport->Update();

        vtkSmartPointer<vtkMarchingCubes> isoSurface = vtkSmartPointer<
            vtkMarchingCubes
        >::New();
        isoSurface->SetValue(0, 0.000025);
        isoSurface->SetInputConnection(imageImport->GetOutputPort());
        isoSurface->Update();

        // Get a reference to one of the main threads
        hpx::threads::executors::main_pool_executor scheduler;
        // run an async function on the main thread to start the Qt application
        hpx::future<void> render = hpx::async(scheduler, [&isoSurface]()
        {
            //
            vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
            vtkSmartPointer<vtkRenderWindow> renWindow = vtkSmartPointer<
                vtkRenderWindow
            >::New();
            vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<
                vtkRenderWindowInteractor
            >::New();
            vtkSmartPointer<vtkInteractorStyleSwitch> style = vtkSmartPointer<
                vtkInteractorStyleSwitch
            >::New();
            iren->SetRenderWindow(renWindow);
            iren->SetInteractorStyle(style);
            style->SetCurrentStyleToTrackballCamera();
            ren->SetBackground(0.1, 0.1, 0.1);
            renWindow->SetSize(400, 400);
            renWindow->AddRenderer(ren);

            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<
                vtkPolyDataMapper
            >::New();
            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            mapper->SetInputData(
                vtkPolyData::SafeDownCast(isoSurface->GetOutputDataObject(0)));
            mapper->SetImmediateModeRendering(1);
            mapper->SetScalarModeToUsePointFieldData();
            mapper->SelectColorArray("");
            mapper->SetUseLookupTableScalarRange(0);
            mapper->SetScalarRange(0, 1);
            mapper->SetInterpolateScalarsBeforeMapping(0);
            actor->SetMapper(mapper);
            ren->AddActor(actor);
            ren->ResetCameraClippingRange();
            ren->ResetCamera();
            renWindow->Render();
            iren->Start();
        }
        );

        render.wait();
    }
#endif
}

// ----------------------------------------------------------------------------
int hpx_main(boost::program_options::variables_map &vm)
{
    global_options.seed = (unsigned int) std::time(0);
    if (vm.count("seed"))
        global_options.seed = vm["seed"].as < unsigned
    int > ();
    std::cout << "using seed: " << global_options.seed << std::endl;
    std::srand(global_options.seed);

    global_options.points = 10;
    if (vm.count("points"))
        global_options.points = vm["points"].as < unsigned
    int > ();
    std::cout << "using points: " << global_options.points << std::endl;

    global_options.chains = 1;
    if (vm.count("chains"))
        global_options.chains = vm["chains"].as < unsigned
    int > ();
    std::cout << "using chains: " << global_options.chains << std::endl;

    global_options.fchunk = 0;
    if (vm.count("fchunk"))
        global_options.fchunk = vm["fchunk"].as < unsigned
    int > ();
    std::cout << "using fchunk: " << global_options.fchunk << std::endl;

    global_options.render = 0;
    if (vm.count("render"))
        global_options.render = vm["render"].as < bool > ();
    std::cout << "using render: " << global_options.render << std::endl;

    test_splat();

    return hpx::finalize();
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    desc_commandline.add_options()("chains,c", value<unsigned int>(),
        "the number of chained futures to use in the final field copy step");

    desc_commandline.add_options()("fchunk,f", value<unsigned int>(),
        "the chunk size used for field copy");

    desc_commandline.add_options()("points,p", value<unsigned int>(),
        "the number of points to render");

    desc_commandline.add_options()("render,r", value<bool>(),
        "enable/disable rendering");

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
                  boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
