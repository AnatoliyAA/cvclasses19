
/* Texture segmentation based on Gabor filters.
 * @file
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;

    double norm_l2(const descriptor& other) const
    {
        CV_Assert(size() == other.size());

        double sum = 0.0;

        for (size_t i = 0; i < size(); ++i)
        {
            const double difference = (*this)[i] - other[i];
            sum += difference * difference;
        }

        return std::sqrt(sum);
    }
};

struct integral_feature
{
    cv::Mat sum;
    cv::Mat squared_sum;
};

// Сумма и сумма квадратов значений в прямоугольнике.
double rectangle_sum(const cv::Mat& integral, int x, int y, int width, int height)
{
    const int x2 = x + width;
    const int y2 = y + height;

    return integral.at<double>(y2, x2) - integral.at<double>(y, x2) - integral.at<double>(y2, x) + integral.at<double>(y, x);
}

// Строим интегральные изображения для каждого фильтра Габора.
void calculateFeatures(const cv::Mat& image, int kernel_size, std::vector<integral_feature>& features)
{
    features.clear();

    const double gamma = 0.5;
    const double psi = 0.0;

    const double sigmas[] = {2.0, 4.0};
    const double orientations[] = {0.0, CV_PI / 4.0, CV_PI / 2.0, 3.0 * CV_PI / 4.0};

    for (const double sigma : sigmas)
    {
        for (const double theta : orientations)
        {
            const double lambda = sigma * 4.0;

            cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);

            cv::Mat response;
            cv::filter2D(image, response, CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);

            cv::Mat squared_response;
            cv::multiply(response, response, squared_response);

            integral_feature feature;
            cv::integral(response, feature.sum, CV_64F);
            cv::integral(squared_response, feature.squared_sum, CV_64F);

            features.emplace_back(std::move(feature));
        }
    }
}

// Дескриптор окна по заранее рассчитанным интегральным изображениям.
descriptor calculateDescriptor(const std::vector<integral_feature>& features, const cv::Rect& roi)
{
    descriptor result;
    const double area = static_cast<double>(roi.area());

    for (const auto& feature : features)
    {
        const double sum = rectangle_sum(feature.sum, roi.x, roi.y, roi.width, roi.height);

        const double squared_sum = rectangle_sum(feature.squared_sum, roi.x, roi.y, roi.width, roi.height);

        const double mean = sum / area;
        const double variance = std::max(squared_sum / area - mean * mean, 0.0);

        result.emplace_back(mean);
        result.emplace_back(std::sqrt(variance));
    }

    return result;
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(roi.width > 0 && roi.height > 0);
    CV_Assert(roi.x >= 0 && roi.y >= 0);
    CV_Assert(roi.x + roi.width <= image.cols);
    CV_Assert(roi.y + roi.height <= image.rows);
    CV_Assert(eps >= 0.0);

    int kernel_size = std::min(roi.width, roi.height) / 2;
    kernel_size = std::max(kernel_size, 3);

    if (kernel_size % 2 == 0)
    {
        ++kernel_size;
    }

    // Фильтры применяются один раз ко всему изображению.
    std::vector<integral_feature> features;
    calculateFeatures(image, kernel_size, features);

    // Дескриптор выбранного образца.
    const descriptor reference = calculateDescriptor(features, roi);

    // Разностный массив позволяет объединить все найденные окна
    // без многократного закрашивания перекрывающихся прямоугольников.
    const int rows = image.rows;
    const int cols = image.cols;
    const int stride = cols + 1;

    std::vector<int> difference(static_cast<size_t>(rows + 1) * stride, 0);

    // Перемещаем окно с шагом в один пиксель.
    for (int y = 0; y <= rows - roi.height; ++y)
    {
        for (int x = 0; x <= cols - roi.width; ++x)
        {
            const cv::Rect current_roi(x, y, roi.width, roi.height);

            const descriptor test = calculateDescriptor(features, current_roi);

            if (test.norm_l2(reference) <= eps)
            {
                const int x2 = x + roi.width;
                const int y2 = y + roi.height;

                ++difference[y * stride + x];
                --difference[y * stride + x2];
                --difference[y2 * stride + x];
                ++difference[y2 * stride + x2];
            }
        }
    }

    // Префиксная сумма превращает отметки окон в бинарную маску.
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        uchar* output = result.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x)
        {
            const size_t index = static_cast<size_t>(y) * stride + x;

            int value = difference[index];

            if (y > 0)
            {
                value += difference[index - stride];
            }

            if (x > 0)
            {
                value += difference[index - 1];
            }

            if (x > 0 && y > 0)
            {
                value -= difference[index - stride - 1];
            }

            difference[index] = value;
            output[x] = value > 0 ? 255 : 0;
        }
    }

    return result;
}
} // namespace cvlib
