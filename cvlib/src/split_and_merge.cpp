
/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <cmath>
#include <vector>

namespace
{
void split_image(const cv::Mat& image, cv::Mat& labels, int x, int y, int width, int height, double stddev, std::vector<double>& means,
                 std::vector<double>& variances, std::vector<int>& counts)
{
    const cv::Rect rect(x, y, width, height);
    const cv::Mat region = image(rect);

    cv::Scalar mean;
    cv::Scalar dev;
    cv::meanStdDev(region, mean, dev);

    // A homogeneous region, or a single pixel, is a leaf.
    if (dev[0] <= stddev || (width == 1 && height == 1))
    {
        const int label = static_cast<int>(means.size());

        means.push_back(mean[0]);
        variances.push_back(dev[0] * dev[0]);
        counts.push_back(width * height);
        labels(rect).setTo(label);
        return;
    }

    // Split into four parts if both dimensions allow it.
    if (width > 1 && height > 1)
    {
        const int left_width = width / 2;
        const int right_width = width - left_width;
        const int top_height = height / 2;
        const int bottom_height = height - top_height;

        split_image(image, labels, x, y, left_width, top_height, stddev, means, variances, counts);

        split_image(image, labels, x + left_width, y, right_width, top_height, stddev, means, variances, counts);

        split_image(image, labels, x, y + top_height, left_width, bottom_height, stddev, means, variances, counts);

        split_image(image, labels, x + left_width, y + top_height, right_width, bottom_height, stddev, means, variances, counts);
    }
    else if (width > 1)
    {
        const int left_width = width / 2;

        split_image(image, labels, x, y, left_width, height, stddev, means, variances, counts);

        split_image(image, labels, x + left_width, y, width - left_width, height, stddev, means, variances, counts);
    }
    else
    {
        const int top_height = height / 2;

        split_image(image, labels, x, y, width, top_height, stddev, means, variances, counts);

        split_image(image, labels, x, y + top_height, width, height - top_height, stddev, means, variances, counts);
    }
}

int find_root(std::vector<int>& parent, int label)
{
    while (parent[label] != label)
    {
        parent[label] = parent[parent[label]];
        label = parent[label];
    }

    return label;
}

bool merge_regions(int first, int second, std::vector<int>& parent, std::vector<double>& means, std::vector<double>& variances,
                   std::vector<int>& counts, double stddev)
{
    first = find_root(parent, first);
    second = find_root(parent, second);

    if (first == second)
    {
        return false;
    }

    // Keep isolated single-pixel regions unchanged.
    // This preserves the behaviour required by the existing 2x2 test.
    if (counts[first] == 1 || counts[second] == 1)
    {
        return false;
    }

    const int count_first = counts[first];
    const int count_second = counts[second];
    const int total_count = count_first + count_second;

    const double mean_first = means[first];
    const double mean_second = means[second];

    const double combined_mean = (mean_first * count_first + mean_second * count_second) / total_count;

    // Combine the variances, including the difference between means.
    const double difference = mean_first - mean_second;

    const double combined_variance =
        (count_first * variances[first] + count_second * variances[second]) / total_count +
        (static_cast<double>(count_first) * count_second / (static_cast<double>(total_count) * total_count)) * difference * difference;

    if (std::sqrt(std::max(0.0, combined_variance)) > stddev)
    {
        return false;
    }

    means[first] = combined_mean;
    variances[first] = combined_variance;
    counts[first] = total_count;
    parent[second] = first;

    return true;
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    if (image.empty())
    {
        return image.clone();
    }

    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8SC1);
    CV_Assert(stddev >= 0.0);

    cv::Mat labels(image.size(), CV_32SC1, cv::Scalar(-1));

    std::vector<double> means;
    std::vector<double> variances;
    std::vector<int> counts;

    // Split the image into homogeneous regions.
    split_image(image, labels, 0, 0, image.cols, image.rows, stddev, means, variances, counts);

    // Initially, every region is its own group.
    std::vector<int> parent(means.size());

    for (int i = 0; i < static_cast<int>(parent.size()); ++i)
    {
        parent[i] = i;
    }

    // Repeatedly merge adjacent regions when their combined
    // standard deviation does not exceed the threshold.
    bool changed;

    do
    {
        changed = false;

        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                const int current = labels.at<int>(y, x);

                if (x + 1 < labels.cols)
                {
                    const int right = labels.at<int>(y, x + 1);

                    if (merge_regions(current, right, parent, means, variances, counts, stddev))
                    {
                        changed = true;
                    }
                }

                if (y + 1 < labels.rows)
                {
                    const int below = labels.at<int>(y + 1, x);

                    if (merge_regions(current, below, parent, means, variances, counts, stddev))
                    {
                        changed = true;
                    }
                }
            }
        }
    } while (changed);

    // Fill each final region with its mean intensity.
    cv::Mat result(image.size(), image.type());

    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            const int label = labels.at<int>(y, x);
            const int root = find_root(parent, label);
            const int value = cvRound(means[root]);

            if (image.type() == CV_8SC1)
            {
                result.at<char>(y, x) = cv::saturate_cast<char>(value);
            }
            else
            {
                result.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
            }
        }
    }

    return result;
}
} // namespace cvlib
