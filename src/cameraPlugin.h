#pragma once
#include <Kinect.h>
#include <memory>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace wow {
	using namespace std;
	class CameraPlugin {
		// kinect v2 w/h
		static const int depth_width = 512;
		static const int depth_height = 424;

	private:
		int m_color_width;
		int m_color_height;
		int m_depth_width;
		int m_depth_height;

		cv::Mat bufferMat;
		//cv::Mat colorImg;

		int m_bufferSize;

		IKinectSensor* pSensor;
		IColorFrameSource* pColorSource;
		IColorFrameReader* pColorReader;
		IDepthFrameSource* p_depth_source;
		IDepthFrameReader* p_depth_reader;
	public:
		CameraPlugin() : pSensor(nullptr), pColorReader(nullptr) {};
		bool initCamera();
		inline bool resizeCamera(int width, int height) { m_color_width = width; m_color_height = height; };
		bool render_to_screen(const std::string& name, cv::Mat& colorImg, bool offscreen = false);
		bool render(cv::Mat& colorImg);

		bool CameraPlugin::query_current_frame_color2depth(std::vector<int> color_points, std::vector<int>& result);
	};
}