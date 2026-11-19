
#include "cvlib.hpp"
#include <algorithm>

namespace cvlib
{
motion_segmentation::motion_segmentation()
{
    bg_model_.release();
    threshold_ = 25.0f;
}

void motion_segmentation::setThreshold(float threshold)
{
    threshold_ = std::max(1.0f, threshold);
}

void motion_segmentation::getBackgroundImage(cv::OutputArray backgroundImage) const
{
    if (bg_model_.empty())
    {
        backgroundImage.release();
        return;
    }

    cv::Mat background;
    bg_model_.convertTo(background, CV_8UC3);

    backgroundImage.assign(background);
}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate)
{
    cv::Mat image = _image.getMat();

    if (image.empty())
    {
        _fgmask.release();
        return;
    }

    CV_Assert(image.type() == CV_8UC3);

    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC3);


    if (bg_model_.empty() || bg_model_.size() != image.size())
    {
        imageFloat.copyTo(bg_model_);

        _fgmask.create(image.size(), CV_8UC1);
        _fgmask.getMat().setTo(0);

        return;
    }


    double alpha = learningRate;

    if (alpha < 0.0 || alpha > 1.0)
        alpha = 0.01;

    const float a = static_cast<float>(alpha);

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // Сравниваем текущий кадр со средним фоном
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const cv::Vec3f current = imageFloat.at<cv::Vec3f>(y, x);

            const cv::Vec3f mean = bg_model_.at<cv::Vec3f>(y, x);

            const float db = current[0] - mean[0];
            const float dg = current[1] - mean[1];
            const float dr = current[2] - mean[2];

            const float distanceSquared = db * db + dg * dg + dr * dr;

            if (distanceSquared > threshold_ * threshold_)
            {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }


    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& mean = bg_model_.at<cv::Vec3f>(y, x);

            const cv::Vec3f current = imageFloat.at<cv::Vec3f>(y, x);

            mean = (1.0f - a) * mean + a * current;
        }
    }

    mask.copyTo(_fgmask);
}

} // namespace cvlib
