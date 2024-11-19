#include "cameraPlugin.h"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
namespace wow {
	bool CameraPlugin::initCamera()
	{
		//	HRESULT hr = S_OK;
		//
		//	GetDefaultKinectSensor(&pSensor);
		//
		//	if (FAILED(hr)) {
		//		cerr << "Error : GetDefaultKinectSensor" << endl;
		//		return -1;
		//	}
		//
		//	hr = pSensor->Open();
		//	if (FAILED(hr)) {
		//		std::cerr << "Error : IKinectSensor::Open()" << std::endl;
		//		return -1;
		//	}
		//
		//
		//;/*
		//
		//	hr = this->pSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color, &this->pMultiReader);
		//	if (FAILED(hr)) {
		//		std::cerr << "Error : OpenMultiSourceFrameReader" << std::endl;
		//		return -1;
		//	}*/
		//	IColorFrameSource* pColorSource = NULL;
		//
		//	hr = this->pSensor->get_ColorFrameSource(&pColorSource);
		//	if (FAILED(hr)) {
		//		std::cerr << "Error : IKinectSensor :: get_Color_frameSource ()" << std::endl;
		//		return false;
		//	}
		//
		//
		//	hr = pColorSource->OpenReader(&this->pColorReader);
		//	if (FAILED(hr)) {
		//		std::cerr << "Error:IColorFrameSource :: openReader ()" << std::endl;
		//		return false;
		//	}
		//	pColorSource->Release();
		//	
		//	
		//	this->m_color_width = 1920;
		//	this->m_color_height = 1080;
		//
		//	while (true) {
		//		IColorFrame* pColorFrame = nullptr;
		//		std::cout << this->pColorReader << std::endl;
		//		HRESULT hr = this->pColorReader->AcquireLatestFrame(&pColorFrame);
		//		std::cout << "pcolor frame" << hr <<std::endl;
		//		std::cout << "pcolor frame" << pColorFrame<<std::endl;
		//		if(pColorFrame)
		//			pColorFrame->Release();
		//	}
		//	this->bufferMat = cv::Mat(this->m_color_height, this->m_color_width, CV_8UC4);
		//	this->m_bufferSize = this->m_color_width * this->m_color_height * 4 * sizeof(unsigned char);
		//	/*IColorFrame* pColorFrame = nullptr;
		//	std::cout <<"pcolor reader" << pColorReader << std::endl;
		//	std::cout << SUCCEEDED(pColorReader->AcquireLatestFrame(&pColorFrame))<< std::endl;
		//	std::cout << pColorFrame << std::endl;
		//	if (SUCCEEDED(hr)) {
		//		hr = pColorFrame->CopyConvertedFrameDataToArray(m_bufferSize, reinterpret_cast<BYTE*>(bufferMat.data), ColorImageFormat_Bgra);
		//		if (SUCCEEDED(hr)) {
		//			cv::resize(bufferMat, colorImg, cv::Size(), 0.5, 0.5);
		//		}
		//	}
		//
		//	if (pColorFrame) {
		//		pColorFrame->Release();
		//		return true;
		//	}
		//*/

		HRESULT hr = S_OK;
		IMultiSourceFrameReader* pMultiReader;




		GetDefaultKinectSensor(&pSensor);


		if (FAILED(hr)) {
			std::cerr << "Error : GetDefaultKinectSensor" << std::endl;
			return -1;
		}
		hr = pSensor->Open();
		if (FAILED(hr)) {
			std::cerr << "Error : IKinectSensor::Open()" << std::endl;
			return -1;
		}

		hr = pSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color, &pMultiReader);
		if (FAILED(hr)) {
			std::cerr << "Error : OpenMultiSourceFrameReader" << std::endl;
			return -1;
		}


		IColorFrame* colorframe = nullptr;
		IColorFrameReference* frameref = nullptr;

		int KINECT_V2_COLOR_HEIGHT = 1080;
		int KINECT_V2_COLOR_WIDTH = 1920;
		m_color_width = KINECT_V2_COLOR_WIDTH;
		m_color_height = KINECT_V2_COLOR_HEIGHT;
		bufferMat = cv::Mat(KINECT_V2_COLOR_HEIGHT, KINECT_V2_COLOR_WIDTH, CV_8UC4);
		cv::Mat colorImg;
		colorImg = cv::Mat(KINECT_V2_COLOR_HEIGHT / 2, KINECT_V2_COLOR_WIDTH / 2, CV_8UC4);


		hr = pSensor->get_ColorFrameSource(&pColorSource);
		if (FAILED(hr)) {
			std::cerr << "Error : IKinectSensor :: get_Color_frameSource ()" << std::endl;
			return -1;
		}


		hr = pColorSource->OpenReader(&pColorReader);
		if (FAILED(hr)) {
			std::cerr << "Error:IColorFrameSource :: openReader ()" << std::endl;
			return -1;
		}

		m_bufferSize = KINECT_V2_COLOR_WIDTH * KINECT_V2_COLOR_HEIGHT * 4 * sizeof(unsigned char);


		//
		//
		//
		//depth
		hr = pSensor->get_DepthFrameSource(&p_depth_source);
		if (FAILED(hr)) {
			std::cerr << "Error : IKinectSensor :: get_DepthFrameSource ()" << std::endl;
			return -1;
		}
		p_depth_source->OpenReader(&p_depth_reader);
		if (FAILED(hr)) {
			std::cerr << "Error:IDepthFrameSource :: openReader ()" << std::endl;
			return -1;
		}








		return true;
	}
	bool CameraPlugin::render(cv::Mat& colorImg)
	{

		const auto size = colorImg.size();
		if (size.width != (this->m_color_width ) || size.height != (this->m_color_height )) {
			colorImg.create(this->m_color_height, this->m_color_width, CV_8UC4);
		}

		HRESULT hr;
		IColorFrame* pColorFrame = nullptr;
		hr = pColorReader->AcquireLatestFrame(&pColorFrame);
		if (SUCCEEDED(hr)) {
			hr = pColorFrame->CopyConvertedFrameDataToArray(m_bufferSize, reinterpret_cast<BYTE*>(bufferMat.data), ColorImageFormat_Bgra);
			if (SUCCEEDED(hr)) {
				//cv::resize(bufferMat, colorImg, cv::Size(), 0.5, 0.5);
				//cv::Mat tmp;
				cv::cvtColor(bufferMat, colorImg, cv::COLOR_BGRA2BGR);
				//colorImg = std::move(tmp);
			}
		}

		if (pColorFrame) {
			pColorFrame->Release();
			return true;
		}
		return false;
	}
	bool CameraPlugin::render_to_screen(const std::string& name, cv::Mat& colorImg, bool offscreen)
	{

		const auto size = colorImg.size();
		if (size.width != (this->m_color_width / 2) || size.height != (this->m_color_height / 2)) {
			colorImg.create(this->m_color_height / 2, this->m_color_width / 2, CV_8UC4);
		}

		HRESULT hr;
		IColorFrame* pColorFrame = nullptr;
		hr = pColorReader->AcquireLatestFrame(&pColorFrame);
		if (SUCCEEDED(hr)) {
			hr = pColorFrame->CopyConvertedFrameDataToArray(m_bufferSize, reinterpret_cast<BYTE*>(bufferMat.data), ColorImageFormat_Bgra);
			if (SUCCEEDED(hr)) {
				cv::resize(bufferMat, colorImg, cv::Size(), 0.5, 0.5);
				cv::Mat tmp;
				cv::cvtColor(colorImg, tmp, cv::COLOR_BGRA2BGR);
				colorImg = std::move(tmp);
			}
		}

		if (pColorFrame) {
			pColorFrame->Release();
			if (!offscreen) {
				cv::imshow(name, colorImg);
				cv::waitKey(1);
			}
			// show window
			return true;
		}
		return false;
	}

	bool CameraPlugin::query_current_frame_color2depth(std::vector<int> color_points, std::vector<int>& result)
	{

		const UINT point_size = color_points.size();
		std::vector<UINT16> depth_frame_data(color_points.begin(), color_points.end());
		DepthSpacePoint* p_depth_space_point = nullptr;
		const UINT16* p_color_data = depth_frame_data.data();
		ICoordinateMapper* coord_mapper;
		pSensor->get_CoordinateMapper(&coord_mapper);
		coord_mapper->MapColorFrameToDepthSpace(
			point_size,
			p_color_data,
			point_size,
			p_depth_space_point);


		IDepthFrame* p_depth_frame;
		HRESULT hr;


		constexpr int buffer_size = depth_width * depth_height;
		UINT16* depth_data = new UINT16[buffer_size];
		hr = p_depth_reader->AcquireLatestFrame(&p_depth_frame);
		if (SUCCEEDED(hr)) {
			hr = p_depth_frame->CopyFrameDataToArray(buffer_size, depth_data);
			if (SUCCEEDED(hr)) {
				result.reserve(point_size);
				for (int i = 0; i < point_size; i++) {
					UINT16 target_depth = *(depth_data + static_cast<int>(p_depth_space_point[i].X) * depth_width + static_cast<int>(p_depth_space_point[i].Y));

					result.push_back(target_depth);
				}

				if (p_depth_frame) {
					p_depth_frame->Release();
				}

				delete[] depth_data;
				return true;
			}


		}


		if (p_depth_frame) {
			p_depth_frame->Release();
		}

		return false;
	}



}



















//
//#include <Kinect.h>
//
//#include <iostream>
//#include <sstream>
//#include <string>
//#include <ctime>
//#include <cstdio>
//
//#include <opencv2/core.hpp>
//#include <opencv2/core/utility.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//
//
//
//int main() {
//
//
//
//
//	HRESULT hr = S_OK;
//	IKinectSensor* pSensor;
//	IMultiSourceFrameReader* pMultiReader;
//
//
//
//
//	GetDefaultKinectSensor(&pSensor);
//
//
//	if (FAILED(hr)) {
//		std::cerr << "Error : GetDefaultKinectSensor" << std::endl;
//		return -1;
//	}
//
//	hr = pSensor->Open();
//	if (FAILED(hr)) {
//		std::cerr << "Error : IKinectSensor::Open()" << std::endl;
//		return -1;
//	}
//
//	hr = pSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color, &pMultiReader);
//	if (FAILED(hr)) {
//		std::cerr << "Error : OpenMultiSourceFrameReader" << std::endl;
//		return -1;
//	}
//
//
//	IColorFrame* colorframe = nullptr;
//	IColorFrameReference* frameref = nullptr;
//
//	int KINECT_V2_COLOR_HEIGHT = 1080;
//	int KINECT_V2_COLOR_WIDTH = 1920;
//	cv::Mat bufferMat;
//	bufferMat = cv::Mat(KINECT_V2_COLOR_HEIGHT, KINECT_V2_COLOR_WIDTH, CV_8UC4);
//	cv::Mat colorImg;
//	colorImg = cv::Mat(KINECT_V2_COLOR_HEIGHT / 2, KINECT_V2_COLOR_WIDTH / 2, CV_8UC4);
//
//
//	IColorFrameSource* pColorSource;
//	hr = pSensor->get_ColorFrameSource(&pColorSource);
//	if (FAILED(hr)) {
//		std::cerr << "Error : IKinectSensor :: get_Color_frameSource ()" << std::endl;
//		return -1;
//	}
//
//
//	IColorFrameReader* pColorReader;
//	hr = pColorSource->OpenReader(&pColorReader);
//	if (FAILED(hr)) {
//		std::cerr << "Error:IColorFrameSource :: openReader ()" << std::endl;
//		return -1;
//	}
//
//	unsigned int bufferSize = KINECT_V2_COLOR_WIDTH * KINECT_V2_COLOR_HEIGHT * 4 * sizeof(unsigned char);
//	while (true) {
//		IColorFrame* pColorFrame = nullptr;
//		hr = pColorReader->AcquireLatestFrame(&pColorFrame);
//
//		if (SUCCEEDED(hr)) {
//			hr = pColorFrame->CopyConvertedFrameDataToArray(bufferSize, reinterpret_cast<BYTE*>(bufferMat.data), ColorImageFormat_Bgra);
//			if (SUCCEEDED(hr)) {
//				cv::resize(bufferMat, colorImg, cv::Size(), 0.5, 0.5);
//			}
//		}
//
//		if (pColorFrame) {
//			pColorFrame->Release();
//			// show window
//			cv::imshow("Color", colorImg);
//			if (cv::waitKey(30) == VK_ESCAPE) {
//				break;
//			}
//		}
//	}
//
//
//	return true;
//
//
//}