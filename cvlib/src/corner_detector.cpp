/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */


/*
 * FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <cmath>

namespace cvlib
{
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray _mask)
{
    keypoints.clear();

    cv::Mat image = _image.getMat();

    if (image.empty())
        return;

    cv::Mat gray;

    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = image;

    CV_Assert(gray.type() == CV_8UC1);

    cv::Mat mask = _mask.getMat();

    // FAST circle: 16 pixels around the candidate pixel
    const int dx[] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};

    const int dy[] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

    const int threshold = 20;
    const int contiguous = 9;

    // FAST-9 detector
    for (int y = 3; y < gray.rows - 3; ++y)
    {
        for (int x = 3; x < gray.cols - 3; ++x)
        {
            if (!mask.empty() && mask.at<uchar>(y, x) == 0)
                continue;

            const int center = gray.at<uchar>(y, x);

            int brighter = 0;
            int darker = 0;

            int maxBrighter = 0;
            int maxDarker = 0;

            for (int i = 0; i < 16 + contiguous - 1; ++i)
            {
                const int index = i % 16;

                const int value = gray.at<uchar>(y + dy[index], x + dx[index]);

                if (value > center + threshold)
                {
                    ++brighter;
                    darker = 0;
                }
                else if (value < center - threshold)
                {
                    ++darker;
                    brighter = 0;
                }
                else
                {
                    brighter = 0;
                    darker = 0;
                }

                maxBrighter = std::max(maxBrighter, brighter);
                maxDarker = std::max(maxDarker, darker);
            }

            if (maxBrighter >= contiguous || maxDarker >= contiguous)
            {
                keypoints.emplace_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)), 7.0f);
            }
        }
    }

    // Simple non-maximum suppression:
    // keep only the strongest response in a 3x3 neighborhood
    std::vector<cv::KeyPoint> filtered;

    for (const auto& kp : keypoints)
    {
        const int x = cvRound(kp.pt.x);
        const int y = cvRound(kp.pt.y);
        const int center = gray.at<uchar>(y, x);

        bool isMax = true;

        for (int dy = -1; dy <= 1 && isMax; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0)
                    continue;

                const int nx = x + dx;
                const int ny = y + dy;

                if (nx < 3 || ny < 3 || nx >= gray.cols - 3 || ny >= gray.rows - 3)
                    continue;

                if (!mask.empty() && mask.at<uchar>(ny, nx) == 0)
                    continue;

                if (std::abs(static_cast<int>(gray.at<uchar>(ny, nx)) - center) > threshold)
                {
                    isMax = false;
                    break;
                }
            }
        }

        if (isMax)
            filtered.push_back(kp);
    }

    keypoints.swap(filtered);
}

void corner_detector_fast::compute(cv::InputArray _image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat image = _image.getMat();

    if (image.empty())
    {
        descriptors.release();
        keypoints.clear();
        return;
    }

    cv::Mat gray;

    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = image;

    CV_Assert(gray.type() == CV_8UC1);

    const int desc_length = 2;

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);

    cv::Mat desc = descriptors.getMat();

    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i)
    {
        const int x = cvRound(keypoints[i].pt.x);
        const int y = cvRound(keypoints[i].pt.y);

        int* ptr = desc.ptr<int>(i);

        ptr[0] = 0;
        ptr[1] = 0;

        // Binary descriptor from local intensity comparisons
        for (int bit = 0; bit < 64; ++bit)
        {
            const int dx1 = (bit % 8) - 4;
            const int dy1 = (bit / 8) - 4;

            const int dx2 = ((bit + 3) % 8) - 4;
            const int dy2 = ((bit + 5) % 8) - 4;

            const int x1 = std::clamp(x + dx1, 0, gray.cols - 1);
            const int y1 = std::clamp(y + dy1, 0, gray.rows - 1);

            const int x2 = std::clamp(x + dx2, 0, gray.cols - 1);
            const int y2 = std::clamp(y + dy2, 0, gray.rows - 1);

            if (gray.at<uchar>(y1, x1) < gray.at<uchar>(y2, x2))
                ptr[bit / 32] |= (1 << (bit % 32));
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
                                            cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    if (!useProvidedKeypoints)
        detect(image, keypoints, mask);

    compute(image, keypoints, descriptors);
}

} // namespace cvlib
