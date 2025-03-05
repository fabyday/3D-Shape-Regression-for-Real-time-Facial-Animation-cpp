#include <iostream>
#include "cascade_regressor.h"
#include "dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <LBFGS.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "constants.h"
#include "math.h"
#include "cameraPlugin.h"
#include <utility>


#include "network.h"

int main() {

	CascadeRegressor cascade_reg;
	struct Blendshapes bl;
	std::vector<int> full_lmk_idx = { 1278, 1272, 12, 1834, 243, 781, 2199, 1447, 966, 3661, 4390, 3022, 2484, 4036, 2253, 3490, 3496, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966 };
	std::vector<int> lmk_idx;
	//load_blendshapes("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\train_dataset\\1.5", bl); // load ref mesh
	load_blendshapes("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\train_dataset\\data1", bl); // load ref mesh
	get_inner_contour("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\predefined_face_weight\\regression_contour.npy", lmk_idx);

	Dataset m;
	m.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\train_dataset\\data1");
	//m.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\train_dataset\\1.5");

	//cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5");
	//cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5_clahe"); // image preporcessing 
	//cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5_clahe_without_kap"); // image preporcessing 
	//cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5_clahe10_kap"); // image preporcessing 
	cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5_data1"); // image preporcessing 
	//cascade_reg.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\pretrain_model\\1.5_without_kap"); // image preporcessing 

	cascade_reg.add_dataset(&m);
	cascade_reg.add_blendshapes(&bl);
	cascade_reg.load_dlib_detector(LANDMARK_PATH);
	cascade_reg.set_68landmark_index(full_lmk_idx);
	cascade_reg.set_autogen_lmk_index(lmk_idx);


	BLendshapes_GMM gmm_model;
	gmm_model.load("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\precompute_blendshapes_gmm\\win5");
	FicalTrackingParams ficial_params(bl, lmk_idx, &gmm_model);


	networkCtx Clientctx;

	CProcessInfo viewerCtx;
	create_child_viewer(viewerCtx);
	create_client_socket(Clientctx);
	std::cout << "create viwer" << std::endl;
	std::cout << "create send socket" << std::endl;


	
	/// camera
	wow::CameraPlugin cam;
	cam.initCamera();





	cv::Mat frame, raw_gray, gray;
	cv::VideoCapture cap("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\video\\face9.mp4");
	//int deviceID = 0; // 0 = open default camera
	//int apiID = cv::CAP_ANY; // 0 = autodetect default API
	// open selected camera using selected API
	//cap.open(deviceID, apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	cv::namedWindow("live");
	int i = 0;
	Eigen::MatrixXd pts;
	pts.conservativeResize(lmk_idx.size(), 3);
	bool init_flag = true;
	auto Q = cascade_reg.get_Q();
	int iter = 0;
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3, 4);
	std::string name = "test";
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
	cv::Mat frame2, frame_tmp;
	Eigen::VectorXd result_bl;

	while (true) {
	
		cam.render( frame);
		if (frame.empty()) {
			return 0;
		}

		//cv::imshow("live", frame);
		//cv::waitKey(0);

		//cv::cvtColor(frame, raw_gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		// TODO if remove process clahe image... commnet below 
		//clahe->apply(raw_gray,  gray);
		
		cv::Mat frame2 = frame.clone();
		//cv::hconcat(frame_tmp, gray, frame2);
		//std::cout << "iter :" << iter++ << std::endl;
		bool detected = false;
		if (init_flag) {
			detected = cascade_reg.predict(gray, nullptr, pts);
			if (!detected)
				continue;
			init_flag = false;
		}
		else {
			detected = cascade_reg.predict(gray, &pts, pts);
		}



		Eigen::MatrixXd pts2d;
		if (detected) {

			add_to_pts(Q, identity, pts, pts2d);
			cvrt_camera_uv_coord(pts2d, frame2.rows, pts2d);
			cv::Mat t;
			for (int i = 0; i < pts2d.rows(); i++)
				cv::circle(frame2, cv::Point(pts2d(i, 0), pts2d(i, 1)), 2, cv::Scalar(0, 255, 0));

			FaceTracking(pts, ficial_params, result_bl);
			send_blendshapes(result_bl, Clientctx);
			std::cout << result_bl << std::endl <<std::endl;

		}
		cv::imshow("live", frame2);
		int k = cv::waitKey(1);
		if (k == 'r') {
			init_flag = true;
		}
	}


}