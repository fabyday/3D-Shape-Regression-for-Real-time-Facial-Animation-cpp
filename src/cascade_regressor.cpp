#include "cascade_regressor.h"
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <utility>
#include "string_function.h"
#include "cnpy.h"

//#include <opencv2/core/eigen.hpp>
#include "math.h"
#include <limits>
#include <dlib/opencv.h>

void CascadeRegressor::load(const std::string& root_pth)
{


		
	std::filesystem::path root = root_pth;
	std::map<std::string, std::string> mm;

	std::filesystem::path meta_path = "";
	for (auto const& dir_entry : std::filesystem::directory_iterator{ root }) {
		if (dir_entry.path().filename() == "meta.txt")
			meta_path = dir_entry.path();
	}
	if (meta_path == "")
		throw std::exception("meta.txt not found.");

	
	std::ifstream  in(meta_path.string());
	std::string buf;
	/*while (true) {
		if (in.is_open()) {
			in >> buf;
			std::cout << buf << std::endl;
			int pos = buf.find(":", 0);
			std::string key = buf.substr(0, pos);
			std::string value = buf.substr(pos+1, buf.size());
			trim(key);
			trim(value);
			mm.insert(std::pair(key, value));
		}
		else {
			break;
		}

	}*/
	while (std::getline(in, buf)) {
			int pos = buf.find(":", 0);
			std::string key = buf.substr(0, pos);
			std::string value = buf.substr(pos+1, buf.size());
			trim(key);
			trim(value);
			mm.insert(std::pair(key, value));
	}



	m_T = std::stoi(mm["T"]);
	m_K = std::stoi(mm["K"]);
	m_F = std::stoi(mm["F"]);
	m_P = std::stoi(mm["P"]);
	m_beta = std::stoi(mm["beta"]);
	m_lmk_size = std::stoi(mm["lmk_size"]);
	
	// Q init 
	//std::cout << (root_pth / std::filesystem::path(mm["Q"] )) << std::endl;
	
	std::string Q_path((root_pth / std::filesystem::path(mm["Q"])).string() );
	cnpy::NpyArray npy_data = cnpy::npy_load( Q_path );
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> mapped_npy(npy_data.data<double>());
	m_Q = mapped_npy;
	//std::cout << m_Q << std::endl;

	std::string S_path = (root_pth / std::filesystem::path(mm["S"])).string();
	//std::cout << " S_path" << S_path << std::endl;
	cnpy::NpyArray S_list_npy = cnpy::npy_load(S_path);
	int num = S_list_npy.shape[0];
	int r = S_list_npy.shape[1];
	int c = S_list_npy.shape[2];
	//std::cout<<num <<std::endl << r << std::endl << c << std::endl;

	// S Init
	for (int i = 0; i < num; i++) {
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >> data(S_list_npy.data<double>() + r*c*i , r, c);
		//std::cout << data << std::endl;
		m_S_list.emplace_back(data);
		//std::cout << m_S_list[0]<< std::endl;
	}

	// regressor init
	for (int i = 0; i < m_T; i++) {
		m_weak_regs.emplace_back(m_Q, m_K, m_F, m_beta, m_P, std::to_string(i));
		m_weak_regs[i].load(root_pth);
	}
	

	




}


/// <summary>
///  img is grayscaled img
/// </summary>
/// <param name="img"></param>
/// <param name="prev_shapes"></param>
/// <param name="result_shape"></param>
/// <param name="init_num"></param>
/// 
/// 

// this is visualize purpose header. cvshow::
#include <opencv2/opencv.hpp>
bool CascadeRegressor::predict(const cv::Mat& img, const Eigen::MatrixXd* prev_shapes, Eigen::MatrixXd& result_shape, int init_num )
{


	//Eigen::Map<Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> > eig_img(img.data, img.rows, img.cols);
	//Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> eig_img = Eigen::Map<Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> >(img.data, img.rows, img.cols);
	Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> eig_img;
	wow::cv2eigen(img, eig_img);
	Eigen::MatrixXd* prev3d;
	
	bool del_flag = false;
	


	if (prev_shapes == nullptr) {
		prev3d = new Eigen::MatrixXd();
		Eigen::MatrixXd prev2d;
		//Eigen::MatrixXd prev2d_test(68,2);
	/*	prev2d_test << 807., 443.,
			808., 472.,
			811., 500.,
			818., 528.,
			830., 552.,
			850., 572.,
			876., 587.,
			905., 596.,
			936., 599.,
			961., 596.,
			978., 583.,
			993., 564.,
			1004., 543.,
			1010., 519.,
			1014., 495.,
			1016., 471.,
			1012., 447.,
			859., 411.,
			876., 397.,
			898., 392.,
			920., 395.,
			942., 402.,
			958., 404.,
			973., 400.,
			987., 399.,
			1000., 403.,
			1006., 416.,
			951., 424.,
			954., 439.,
			957., 453.,
			960., 469.,
			934., 490.,
			944., 491.,
			954., 493.,
			962., 492.,
			969., 490.,
			883., 427.,
			896., 421.,
			907., 420.,
			917., 429.,
			906., 429.,
			895., 428.,
			968., 432.,
			979., 426.,
			988., 427.,
			994., 434.,
			988., 435.,
			978., 434.,
			916., 533.,
			931., 521.,
			946., 514.,
			954., 518.,
			961., 516.,
			969., 523.,
			974., 535.,
			967., 542.,
			959., 545.,
			951., 546.,
			942., 544.,
			929., 540.,
			922., 532.,
			944., 528.,
			953., 529.,
			960., 529.,
			970., 534.,
			960., 530.,
			953., 530.,
			944., 529.;*/
		bool detected = this->get_landmark_pos(img, prev2d);
		if (!detected) {
			return false;
		}
		//std::cout << prev2d_test - prev2d << std::endl;
		//prev2d = prev2d_test;

		/*cv::Mat t;
		cv::cvtColor(img, t, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < prev2d.rows(); i++)
			cv::circle(t, cv::Point(prev2d(i, 0), prev2d(i, 1)), 1, cv::Scalar(0, 255, 0));*/
		cv::Mat t;
		//wow::eigen2cv(eig_img, t);
		////cv::imshow("test_img", t);
		//cv::waitKey(0);
		//shapefit_2d_lmk_to_mesh(*m_blendshapes, prev2d, this->m_lmk_index, m_Q, *prev3d, img.rows);
		cv::Mat c;
		img.copyTo(c);
		Eigen::MatrixXd prev2dh;
		//cvrt_camera_uv_coord(prev2d, img.rows,prev2dh);

		shapefit_2d_lmk_to_mesh(*m_blendshapes, prev2d, this->m_lmk_index, this->m_auotogen_lmk_index, m_Q, *prev3d, c);
		del_flag = true;
		//Eigen::MatrixXd prev3d_test(76, 3);
		//prev3d_test << 7.16371120e-01, 1.80101499e+00, -6.77762759e+01,
		//prev3d->conservativeResize(76, 3);
	/*	*prev3d << 7.16371120e-01, 1.80101499e+00, -6.77762759e+01,
			3.99758165e+00, -7.20341603e-01, -6.62126598e+01,
			6.86647868e+00, -4.41761244e+00, -6.88134363e+01,
			6.90367101e+00, -5.61191186e+00, -6.85624999e+01,
			5.72912124e+00, -6.20985497e+00, -6.79476664e+01,
			5.13389819e+00, 2.66194120e+00, -6.68930178e+01,
			2.50473253e+00, -2.11695229e+00, -6.77303773e+01,
			2.92365950e+00, -7.31559709e-01, -6.70022582e+01,
			2.30809763e+00, 2.51656278e-01, -6.70603383e+01,
			9.59609632e-01, 6.24637612e-01, -6.75845935e+01,
			2.04655605e+00, -5.84535807e+00, -6.74795755e+01,
			5.89192344e+00, -4.03510708e+00, -6.85938023e+01,
			4.99497896e+00, -4.03744794e+00, -6.82492355e+01,
			3.79732660e+00, -4.06314939e+00, -6.87537556e+01,
			3.38433093e+00, -3.99087796e+00, -6.91275046e+01,
			3.93605219e+00, -3.33843555e+00, -6.88515199e+01,
			5.15322189e+00, -3.22622253e+00, -6.83719546e+01,
			4.47218686e+00, -6.44412856e+00, -6.76086208e+01,
			4.28977010e+00, -5.96521714e+00, -6.74816094e+01,
			6.22949003e+00, -7.84984640e+00, -6.83682658e+01,
			3.73096170e+00, -9.21090314e+00, -6.78850723e+01,
			3.01296409e+00, -5.76513436e+00, -6.74109843e+01,
			3.11828277e+00, -6.25522240e+00, -6.73904128e+01,
			1.60155842e+00, 4.52544953e-01, -6.72805922e+01,
			1.37159421e+00, 1.75834156e+00, -6.73274959e+01,
			6.80048679e+00, -3.00487144e+00, -6.81048162e+01,
			6.66709543e+00, -8.13535877e-01, -6.67008652e+01,
			-1.77538591e+00, -6.23012239e-02, -6.92615105e+01,
			-2.94894693e+00, -4.20978705e+00, -7.18064910e+01,
			-3.13715855e+00, -5.47563564e+00, -7.16097075e+01,
			-2.53256143e+00, -6.19239733e+00, -7.04424715e+01,
			-1.37400139e+00, 2.83188083e+00, -7.04815157e+01,
			9.09022868e-02, -1.88413454e+00, -6.86862114e+01,
			-5.40681400e-01, -3.68809670e-01, -6.90716874e+01,
			-1.86734312e-01, 4.05346387e-01, -6.86171890e+01,
			3.14331225e-01, -5.93861161e+00, -6.80064327e+01,
			-2.31043352e+00, -3.89224688e+00, -7.12100423e+01,
			-1.85869206e+00, -3.85422143e+00, -7.04780431e+01,
			-6.11272143e-01, -3.88089549e+00, -7.00232406e+01,
			3.24931521e-02, -3.93454694e+00, -7.00365083e+01,
			-7.19775985e-01, -3.21991937e+00, -7.01369000e+01,
			-1.89406315e+00, -3.10953628e+00, -7.06087614e+01,
			-1.65872362e+00, -6.46649280e+00, -6.94998638e+01,
			-1.58824339e+00, -6.00234856e+00, -6.92785941e+01,
			-2.73894367e+00, -7.79688456e+00, -7.10678349e+01,
			-9.41291265e-01, -9.21793186e+00, -6.93196315e+01,
			-5.71987421e-01, -5.90168750e+00, -6.84736427e+01,
			-6.77874671e-01, -6.36675217e+00, -6.85180826e+01,
			3.19395689e-01, 5.04438732e-01, -6.79874667e+01,
			8.18550778e-02, 1.85830831e+00, -6.82408915e+01,
			-3.29729147e+00, -2.48940469e+00, -7.14131402e+01,
			-3.55227817e+00, -1.72700397e-01, -7.07707467e+01,
			2.37919269e+00, -3.26612203e+00, -6.79875223e+01,
			3.81081465e-01, -3.16483955e+00, -6.86320700e+01,
			2.39824659e+00, -2.75561417e+00, -6.77445230e+01,
			2.14607233e-01, -2.60340538e+00, -6.84989919e+01,
			2.62604009e+00, -1.12148008e+00, -6.70255105e+01,
			6.82821168e-01, -6.57422217e-01, -6.62266465e+01,
			-3.23721647e-01, -8.47506026e-01, -6.83852233e+01,
			8.68915316e-01, 1.82720912e+00, -6.61424638e+01,
			9.03654039e-01, 2.47772169e+00, -6.66187196e+01,
			1.03019183e+00, 2.19736680e+00, -6.85482682e+01,
			2.36482617e+00, 1.88783040e+00, -6.68989911e+01,
			3.50738407e+00, 1.96460725e+00, -6.70579888e+01,
			3.76288997e+00, 2.13158607e+00, -6.68342756e+01,
			2.05335734e+00, 2.20652498e+00, -6.78612840e+01,
			2.82262622e+00, 2.19627759e+00, -6.58139230e+01,
			1.92077475e+00, 2.43187647e+00, -6.60654086e+01,
			2.01678687e+00, 1.81989010e+00, -6.56211986e+01,
			-6.35350302e-01, 2.18507971e+00, -6.92752615e+01,
			-7.57018310e-01, 2.05674355e+00, -7.02424438e+01,
			-1.09949596e+00, 2.24996444e+00, -7.03853137e+01,
			7.31132234e-02, 2.34407344e+00, -6.92685911e+01,
			-8.05758816e-01, 2.07222405e+00, -6.83934966e+01,
			-7.15058324e-02, 2.33550287e+00, -6.73636796e+01,
			-2.98330540e-01, 1.78858855e+00, -6.70876684e+01;*/

		//std::cout << prev3d_test - *prev3d << std::endl;

	}
	else {
		prev3d = const_cast<Eigen::MatrixXd*>(prev_shapes);
	}
	
	int size = m_dataset->m_S_original_list.size();
	double loss = std::numeric_limits<double>::infinity();
	int ind;
	std::vector<double> res;
	for (int i = 0; i < size; i++) {
		double s_loss = distance_between_2object(m_dataset->m_S_original_list[i], *prev3d);
		res.push_back(s_loss);
		if (s_loss < loss) {
			loss = s_loss;
			ind = i;
		}
	}
	//std::cout << "test" << std::endl;
	//for (auto a : res)
		//std::cout << a << std::endl;

	//std::cout << "test" << ind << std::endl;
	Eigen::MatrixXd S_r = m_dataset->m_S_original_list[ind];
	Eigen::Matrix3d R;
	Eigen::MatrixXd combineRt;
	


	
	similarity_transform(*prev3d, S_r, R, combineRt);


	/* {
	//	Eigen::MatrixXd pts2ds;

	//	add_to_pts(m_Q, Eigen::MatrixXd::Identity(3, 4), *prev3d, pts2ds);
	//	cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
	//	

	//	cv::Mat t;
	//	img.copyTo(t);
	//	cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
	//	for (int i = 0; i < pts2ds.rows(); i++)
	//		cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(0, 255, 0));

	//	cv::imshow("fu t", t);
	//	cv::waitKey(100);
	//}*/
	Eigen::MatrixXd prev_frame_S_prime; 
	add_to_pts(combineRt, *prev3d, prev_frame_S_prime);
	std::vector<int> candidate_idx;
	find_most_similar_to_target_pose(m_dataset->m_Ss_list, prev_frame_S_prime, candidate_idx);
	Eigen::MatrixXd invRt;
	InverseRt(combineRt, invRt);
	/* {
		Eigen::MatrixXd pts2ds;

		//add_to_pts(m_Q, Eigen::MatrixXd::Identity(3, 4),  prev_frame_S_prime, pts2ds);
		//cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		add_to_pts(m_Q, invRt, S_r, pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		cv::Mat t;
		img.copyTo(t);
		cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(0, 255, 0));
		add_to_pts(m_Q, Eigen::MatrixXd::Identity(3, 4), S_r, pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(255, 0, 0));


		//cv::imshow("fu t", t);
		//cv::waitKey(100);
	}
	{
		Eigen::MatrixXd pts2ds;

		add_to_pts(m_Q, invRt, prev_frame_S_prime, pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);

		cv::Mat t;
		img.copyTo(t);
		cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(0, 255, 0));
		add_to_pts(m_Q, Eigen::MatrixXd::Identity(3, 4), S_r, pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(255, 0, 0));
		//cv::imshow("fu t", t);
		//cv::waitKey(100);
	}*/


	/*for (int i = 0; i < init_num; i++) {
		Eigen::MatrixXd pts2ds;

		add_to_pts(m_Q, invRt, m_dataset->m_Ss_list[candidate_idx[i]], pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		cv::Mat t;
		img.copyTo(t);
		cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(0, 255, 0));
		add_to_pts(m_Q, Eigen::MatrixXd::Identity(3,4), m_dataset->m_Ss_list[candidate_idx[i]], pts2ds);
		cvrt_camera_uv_coord(pts2ds, img.rows, pts2ds);
		for (int i = 0; i < pts2ds.rows(); i++)
			cv::circle(t, cv::Point(pts2ds(i, 0), pts2ds(i, 1)), 2, cv::Scalar(255, 0, 0));


		//cv::imshow("fu t", t);
		//cv::waitKey(100);
	}*/

  	std::vector<Eigen::MatrixXd> result_queue;
	result_queue.reserve(init_num);
 	for (int i = 0; i < init_num; i++) {

		Eigen::MatrixXd curpose = m_dataset->m_Ss_list[candidate_idx[i]];
		for (int j = 0; j < m_weak_regs.size(); j++) {
			m_weak_regs[j].predict(eig_img, m_meanshape, invRt, curpose);
		}
		result_queue.push_back(curpose);

		//Eigen::MatrixXd tmps;
		//add_to_pts(m_Q, invRt, curpose, tmps);
		//cvrt_camera_uv_coord(tmps, img.rows, tmps);

		//cv::Mat f;


		//cv::Mat t; 
		//img.copyTo(t);
		//cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		//for (int i = 0; i < tmps.rows(); i++)
		//	cv::circle(t, cv::Point(tmps(i, 0) , tmps(i, 1)), 2, cv::Scalar(255, 0, 0));

		//cv::imshow("fu t", t);
		//cv::waitKey(100);
		
	}

	
	int target_idx = int(ceil(result_queue.size() / 2));
	int row_size = result_queue[0].rows();
	std::vector<double> xx;
	std::vector<double> xy;
	std::vector<double> xz;
	xx.resize(result_queue.size());
	xy.resize(result_queue.size());
	xz.resize(result_queue.size());
	result_shape.conservativeResize(row_size, 3);
	for (int i = 0; i < row_size; i++) {
		for (int j= 0; j< result_queue.size(); j++) {
			xx[j] = result_queue[j](i, 0);
			xy[j] = result_queue[j](i, 1);
			xz[j] = result_queue[j](i, 2);
		}
		std::sort(xx.begin(), xx.end());
		std::sort(xy.begin(), xy.end());
		std::sort(xz.begin(), xz.end());
		result_shape.row(i) << xx[target_idx], xy[target_idx], xz[target_idx];
	
	}
	add_to_pts(invRt, result_shape, result_shape);
	if (del_flag) {
		delete prev3d;
	}


	return true;
}

bool CascadeRegressor::get_landmark_pos(const cv::Mat& img, Eigen::MatrixXd& out)
{
	//dlib::cv_image<dlib::bgr_pixel> img_wrapper(img);
	dlib::cv_image<unsigned char> img_wrapper(img);

	std::vector<dlib::full_object_detection> shapes;

	//dlib::array2d<dlib::bgr_pixel> img;

	//assign_image(img, img_wrapper);
	//dlib::pyramid_up(img_wrapper);

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces in the image.
	std::vector<dlib::rectangle> dets = m_detector(img_wrapper);
	//std::cout << "Number of faces detected: " << dets.size() << std::endl;


	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		dlib::full_object_detection shape = m_shape_predictor(img_wrapper, dets[j]);
		//std::cout << "number of parts: " << shape.num_parts() << std::endl;
		//std::cout << "pixel position of first part:  " << shape.part(0) << std::endl;
		//std::cout << "pixel position of second part: " << shape.part(1) << std::endl;
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
	}
	if (shapes.size() == 0)
		return false;


	out.conservativeResize(shapes[0].num_parts(), 2);
	for (int i = 0; i < shapes[0].num_parts(); i++) {
		out(i, 0) = shapes[0].part(i).x();
		out(i, 1) = shapes[0].part(i).y();
	}

	return true;


}
