#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <DGtal/base/Common.h>
#include <DGtal/kernel/SpaceND.h>
#include <DGtal/kernel/domains/HyperRectDomain.h>
#include <DGtal/images/ImageSelector.h>
#include <DGtal/geometry/volumes/distance/DistanceTransformation.h>
#include <DGtal/images/SimpleThresholdForegroundPredicate.h>
#include <DGtal/helpers/StdDefs.h>
#include "DGtal/base/BasicFunctors.h"
#include "DGtal/kernel/BasicPointPredicates.h"
#include "DGtal/images/IntervalForegroundPredicate.h"

#include "DGtal/helpers/StdDefs.h"
#include "DGtal/kernel/sets/DigitalSetInserter.h"
#include "DGtal/images/ImageContainerBySTLVector.h"
#include "DGtal/images/ImageHelper.h"
#include "DGtal/geometry/volumes/distance/DistanceTransformation.h"
#include "DGtal/io/boards/Board2D.h"
#include "DGtal/io/colormaps/HueShadeColorMap.h"
#include "DGtal/io/colormaps/GrayscaleColorMap.h"

namespace py = pybind11;

using namespace DGtal;

typedef ImageSelector<Z3i::Domain, unsigned char>::Type Image;

double computeMaxDistance(py::array_t<int32_t> npArray, py::array_t<int32_t> npPoints) {
    // Convert numpy array to DGtal image
    py::buffer_info buf_info = npArray.request();
    auto ptr = static_cast<int32_t*>(buf_info.ptr);

    // Assuming the numpy array is 3D and the dimensions are known
    Z3i::Domain domain(Z3i::Point(0, 0, 0), Z3i::Point(buf_info.shape[0], buf_info.shape[1], buf_info.shape[2]));
    Image image(domain);

    for (size_t x = 0; x < buf_info.shape[0]; ++x) {
        for (size_t y = 0; y < buf_info.shape[1]; ++y) {
            for (size_t z = 0; z < buf_info.shape[2]; ++z) {
                image.setValue(Z3i::Point(x, y, z), ptr[z + y * buf_info.shape[2] + x * buf_info.shape[1] * buf_info.shape[2]]);
            }
        }
    }

    // Convert numpy array of points to vector of Z3i::Point
    py::buffer_info buf_points_info = npPoints.request();
    auto points_ptr = static_cast<int32_t*>(buf_points_info.ptr);
    std::vector<Z3i::Point> points;

    if (buf_points_info.ndim != 2 || buf_points_info.shape[1] != 3) {
        throw std::runtime_error("Invalid points array shape.");
    }

    for (size_t i = 0; i < buf_points_info.shape[0]; ++i) {
        Z3i::Point p(points_ptr[i], points_ptr[i + buf_points_info.shape[0]], points_ptr[i + 2 * buf_points_info.shape[0]]);
        
        points.push_back(p);
    }

    // print the first 3 points
    for (size_t i = 0; i < 3; ++i) {
        std::cout << points[i] << std::endl;
    }

    // print the last 3 points
    for (size_t i = points.size() - 3; i < points.size(); ++i) {
        std::cout << points[i] << std::endl;
    }
    
    typedef functors::SimpleThresholdForegroundPredicate<Image> Predicate;
    Predicate aPredicate(image, 0);
    typedef  DistanceTransformation<Z3i::Space,Predicate, Z3i::L2Metric> DTL2;
    DTL2 dtL2(&domain, &aPredicate, &Z3i::l2Metric);

    double max = 0;

    for (auto &point : points) {
        double valDist = dtL2(point);
        if (valDist > max) {
            max = valDist;
        }
    }

    return max;
}

PYBIND11_MODULE(my_module, m) {
    m.def("compute_max_distance", &computeMaxDistance, "Compute the maximum distance in the distance transformation");
}