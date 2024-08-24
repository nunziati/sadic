#include <iostream>
#include <DGtal/base/Common.h>
#include <DGtal/kernel/SpaceND.h>
#include <DGtal/kernel/domains/HyperRectDomain.h>
#include <DGtal/images/ImageSelector.h>
#include <DGtal/io/readers/VolReader.h>
#include <DGtal/geometry/volumes/distance/DistanceTransformation.h>
#include <DGtal/images/SimpleThresholdForegroundPredicate.h>
#include <DGtal/helpers/StdDefs.h>

using namespace std;
using namespace DGtal;

template<typename Image>
void randomSeeds(Image &image, const unsigned int nb, const int value)
{
  typename Image::Point p, low = image.domain().lowerBound();
  typename Image::Vector ext;
  srand ( time(NULL) );

  ext = image.extent();

  for (unsigned int k = 0 ; k < nb; k++)
    {
      for (unsigned int dim = 0; dim < Image::dimension; dim++)
        p[dim] = rand() % (ext[dim]) +  low[dim];

      image.setValue(p, value);
    }
}

int main( int argc, char** argv )
{
  std::string inputFilename = "Al.100.vol";

  // Load the 3D image
  typedef ImageSelector<Z3i::Domain, unsigned char>::Type Image;
  Image image = VolReader<Image>::importVol(inputFilename);
  Z3i::Domain domain = image.domain();

  // Initialize a seed image
  Image imageSeeds(domain);
  for (Image::Iterator it = imageSeeds.begin(), itend = imageSeeds.end(); it != itend; ++it)
    (*it) = 1;

  // Set random seeds
  randomSeeds(imageSeeds, 70, 0);

  // Compute the distance transformation
  typedef functors::SimpleThresholdForegroundPredicate<Image> Predicate;
  Predicate aPredicate(imageSeeds, 0);

  typedef DistanceTransformation<Z3i::Space, Predicate, Z3i::L2Metric> DTL2;
  DTL2 dtL2(&domain, &aPredicate, &Z3i::l2Metric);

  // Find the maximum distance using an iterator over points
  unsigned int max = 0;
  for (Z3i::Domain::ConstIterator pointIt = domain.begin(), pointEnd = domain.end(); pointIt != pointEnd; ++pointIt)
  {
    double distanceValue = dtL2(*pointIt);
    if (distanceValue > max)
      max = distanceValue;
  }

  cout << "Maximum distance value: " << max << endl;

  return 0;
}
