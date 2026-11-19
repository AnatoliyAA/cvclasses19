/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */




#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        return -1;

   
    auto mseg = cv::makePtr<cvlib::motion_segmentation>();

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 25;

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::createTrackbar("Sensitivity", demo_wnd, &threshold, 100);

    cv::Mat frame;
    cv::Mat frame_mseg;

    while (cv::waitKey(30) != 27)
    {
        cap >> frame;

        if (frame.empty())
            break;

        cv::imshow(main_wnd, frame);
        mseg->setThreshold(static_cast<float>(std::max(1, threshold)));
        mseg->apply(frame, frame_mseg);

        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cap.release();

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
