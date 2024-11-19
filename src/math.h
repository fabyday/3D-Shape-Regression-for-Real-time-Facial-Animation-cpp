#pragma once 
#include <Eigen/Core>
#include <Eigen/SVD>

#include <vector>
#include <algorithm>
#include <Eigen/Dense>




inline void similarity_transform(Eigen::MatrixXd src, Eigen::MatrixXd dest, Eigen::Matrix3d& R, Eigen::MatrixXd& combined_Rt) {

	Eigen::Vector3d src_mean = src.colwise().mean();
	Eigen::Vector3d dest_mean = dest.colwise().mean();

	Eigen::MatrixXd centered_src = src.rowwise() - src_mean.transpose();
	Eigen::MatrixXd centered_dest = dest.rowwise() - dest_mean.transpose();

	//
	Eigen::MatrixXd tmp = centered_src.transpose() * centered_dest;
	//std::cout << tmp << std::endl;

	//Eigen::JacobiSVD<Eigen::MatrixXd> svd(tmp, Eigen::ComputeThinU | Eigen::ComputeThinV);
  	Eigen::JacobiSVD<Eigen::MatrixXd> svd(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::VectorXd S = svd.singularValues();
	//Eigen::MatrixXd Vt = svd.matrixV();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd s = V * U.transpose();
	//double det = s.determinant();
	double det = (V * U.transpose()).determinant();
	//double det = 10;
	Eigen::Matrix3d sig = Eigen::Matrix3d::Identity();
	sig(2, 2) = det;
	//std::cout << V << std::endl;
	//std::cout << U.transpose() << std::endl;

	//std::cout << det << std::endl;
	//std::cout << sig << std::endl;

	Eigen::Matrix3d rot = V * sig * U.transpose();
	R = rot;
	


	Eigen::MatrixXd Rt1 = Eigen::MatrixXd::Identity(4,4);
	Eigen::MatrixXd Rt2 = Eigen::MatrixXd::Identity(4, 4);
	Eigen::MatrixXd Rt3 = Eigen::MatrixXd::Identity(4, 4);
	Rt1.block(0, 3, 3, 1) = dest_mean;
	Rt2.block(0, 0, 3, 3) = R;
	Rt3.block(0, 3, 3, 1) = -src_mean;
	//std::cout << Rt1 << std::endl;
	//std::cout << Rt2 << std::endl;
	//std::cout << Rt3 << std::endl;
	//std::cout << (Rt1 * Rt2 * Rt3).block(0, 0, 3, 4) << std::endl;
	combined_Rt = (Rt1* Rt2* Rt3).block(0,0,3,4);





}





inline void cvrt_camera_uv_coord(const Eigen::MatrixXd& coord, int height, Eigen::MatrixXd& out) {
	 //coord is N,2
	
	if (coord.rows() != out.rows() || coord.cols() != out.cols()) {
		out.conservativeResizeLike(coord);
		out = coord;
	}


	out.col(1) = height  - coord.col(1).array()- 1;


}
inline void InverseRt(const Eigen::MatrixXd& Rt, Eigen::MatrixXd& out) {
	// Rt is 3,4 matrix
	Eigen::Matrix3d Rinv = Rt.block(0, 0, 3, 3).transpose();
	Eigen::Vector3d invt = -Rinv * Rt.block(0, 3, 3, 1);
	/*std::cout << "Rt " << std::endl;
	std::cout << Rt  << std::endl;

	std::cout << "R " << std::endl;
	std::cout << Rt.block(0, 0, 3, 3) << std::endl;
	std::cout << "Rinv " << std::endl;
	std::cout << Rinv << std::endl;
	std::cout << "vec" << std::endl;
	std::cout << Rt.block(0, 3, 3, 1) << std::endl;
	std::cout << "invrt" << std::endl;
	std::cout << invt << std::endl;*/

	out.conservativeResize(3, 4);
	out.block(0, 0, 3, 3) = Rinv;
	out.block(0, 3, 3, 1) = invt;
	//std::cout << out << std::endl;

}


inline void add_to_pts(const Eigen::Matrix3d& Q, const Eigen::MatrixXd& Rt, const Eigen::MatrixXd coord, Eigen::MatrixXd& out) {
	//std::cout << Rt << std::endl << std::endl;
	//std::cout << Q << std::endl << std::endl;
	//std::cout << Eigen::VectorXd(Rt.block(0, 3, 3, 1)) << std::endl << std::endl;

	//std::cout  <<coord.block(0, 0, 3, 3) << std::endl;
	//Eigen::MatrixXd tms = (Rt.block(0, 0, 3, 3) * coord.transpose());
	//std::cout << tms.transpose().block(0, 0, 3, 3) << std::endl << std::endl;
	//Eigen::MatrixXd sk = tms.colwise() + Eigen::VectorXd(Rt.block(0, 3, 3, 1));
	//std::cout << sk.transpose().block(0, 0, 3, 3) << std::endl << std::endl;
	//std::cout << (Q * sk).transpose().block(0, 0, 3, 3) << std::endl << std::endl;
	Eigen::MatrixXd tm = (Rt.block(0, 0, 3, 3) * coord.transpose()).colwise() + Eigen::VectorXd(Rt.block(0, 3, 3, 1));
	out = (Q * tm).transpose();

	//Eigen::MatrixXd test(3,3);
	//test << 1,2,2,
	//		6,8,3,
	//		4,8,4;
	//test.array().colwise() /= test.col(2).array();
	//std::cout << test << std::endl;
	//test.conservativeResize(3, 2);
	//std::cout << test << std::endl;

	out.array().colwise() /= out.col(2).array();
	//std::cout << out << std::endl;
	out.conservativeResize(out.rows(), 2);
   	//std::cout << " " << std::endl;
	//tmp.array() /= tmp.block(0, 2, tmp.rows(), 1);
	//out = tmp.block(0, 0, tmp.rows(), 2);

}


inline void add_to_pts(const Eigen::MatrixXd& Rt, const Eigen::MatrixXd coord, Eigen::MatrixXd& out) {
	if (Rt.cols() == 4) {
		out = ((Rt.block(0, 0, 3, 3) * coord.transpose()).colwise() + Eigen::VectorXd(Rt.block(0, 3, 3, 1))).transpose();
	}
	else { // 3 
		out = (Rt * coord.transpose()).transpose();

	}
}


inline double distance_between_2object(const Eigen::MatrixXd& coord1, const Eigen::MatrixXd& coord2) {
	Eigen::Vector3d src_mean = coord1.colwise().mean();
	Eigen::Vector3d dest_mean = coord2.colwise().mean();

	Eigen::MatrixXd centered_src = coord1.rowwise() - src_mean.transpose();
	Eigen::MatrixXd centered_dest = coord2.rowwise() - dest_mean.transpose();
	/*std::cout << "dist" << std::endl;
	std::cout << coord1.block(0, 0, 4, 3) << std::endl <<std::endl;
	std::cout << src_mean.transpose() << std::endl <<std::endl;
	std::cout << centered_src.block(0, 0, 4, 3) << std::endl <<std::endl;
	
	std::cout << "dest" << std::endl;
	std::cout << coord2.block(0, 0, 4, 3) << std::endl <<std::endl;
	std::cout << dest_mean.transpose() << std::endl <<std::endl;
	std::cout << (centered_dest).block(0, 0, 4, 3) << std::endl <<std::endl;

	std::cout << "end" << std::endl;
	std::cout << (centered_src - centered_dest).block(0, 0, 4, 3) << std::endl <<std::endl;
	std::cout << (centered_src - centered_dest).array().pow(2.0).block(0, 0, 4, 3) << std::endl <<std::endl;*/

	return (centered_src - centered_dest).array().pow(2.0).sum();
}



// rx ry rz tx ty tz
inline void compose_Rt(const Eigen::VectorXd& trnaslate_rot, Eigen::MatrixXd& Rt_out) {

	Eigen::Matrix3d Rx = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d Ry = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d Rz = Eigen::Matrix3d::Identity();
	double theta_x = trnaslate_rot(0);
	double theta_y = trnaslate_rot(1);
	double theta_z = trnaslate_rot(2);

	Rx(1, 1) = cos(theta_x); Rx(1, 2) = -sin(theta_x);
	Rx(2, 1) = sin(theta_x); Rx(2, 2) = cos(theta_x);

	Ry(0, 0) = cos(theta_y); Ry(0, 2)= sin(theta_y);
	Ry(2, 0) = -sin(theta_y); Ry(2, 2) = cos(theta_y);

	Rz(0, 0) = cos(theta_z); Rz(0, 1) = -sin(theta_z);
	Rz(1, 0) = sin(theta_z); Rz(1, 1) = cos(theta_z);


	Rt_out.conservativeResize(3, 4);
	Rt_out.block(0, 0, 3, 3) = Rz * Ry * Rx;
	//std::cout << "g" << trnaslate_rot.segment(3, 3) << std::endl;
	//std::cout << Rt_out << std::endl;
	Rt_out.block(0, 3, 3, 1) = trnaslate_rot.segment(3,3);
	//std::cout << Rt_out << std::endl;
	

}
#include <math.h>
// rx ry rz tx ty tz
inline void decompose_Rt(const Eigen::MatrixXd& Rt, Eigen::Ref<Eigen::VectorXd> trans_rot, int insert_index = 0) {

	double y_angle = atan2(-1 * Rt(2, 0), sqrt(Rt(0, 0) * Rt(0, 0) + Rt(1, 0) * Rt(1, 0)));
	double x_angle = atan2(Rt(2, 1) / cos(y_angle), Rt(2, 2) / cos(y_angle));
	double z_angle = atan2(Rt(1, 0) / cos(y_angle), Rt(0, 0) / cos(y_angle));
	//double y_angle = atan2(sqrt(Rt(0, 0) * Rt(0, 0) + Rt(1, 0) * Rt(1, 0)) , -1 * Rt(2, 0));
	//double x_angle = atan2(Rt(2, 2) / cos(y_angle), Rt(2, 1) / cos(y_angle));
	//double z_angle = atan2(Rt(0, 0) / cos(y_angle), Rt(1, 0) / cos(y_angle));
	trans_rot.segment(insert_index , 3) << x_angle, y_angle, z_angle;
}


#include <LBFGS.h>
#include "blendshapes.h"


class CalibFunctor{
	const struct Blendshapes* m_blendshapes;
	Eigen::Matrix3d Q;

	int m_image_h;
	int selected_index;
	std::vector<int> lmk;
	Eigen::MatrixXd target;
	Eigen::MatrixXd Rt;
	Eigen::VectorXd weights;
	Eigen::VectorXd full_x;
public:
		void set_blendshapes(const struct Blendshapes* blendshapes) {
		m_blendshapes = blendshapes;
	};


		void set_target(const Eigen::MatrixXd& target) {
			this->target = target;
	}

	void select_optimize_variable_index(int i) {
		selected_index = i;
	};
	void set_Q(const Eigen::Matrix3d& Q) {
		this->Q = Q;
	}


	void set_weights(const Eigen::VectorXd& w) {
		this->weights = w;
	}

	void set_Rt(const Eigen::MatrixXd& Rt)
	{
		this->Rt = Rt;
	}

	void set_image_h(const int image_h) {
		this->m_image_h = image_h;
	}
	void set_landmark(const std::vector<int>& lmk) {
		this->lmk = lmk;
	}



	double cost_function(const Eigen::VectorXd& x) {

		//Eigen::MatrixXd Rt;
		//compose_Rt(x.tail(6), Rt);
		Eigen::MatrixXd blend;
		//m_blendshapes->blend(x.segment(0, x.size() - 6), blend, lmk );
		m_blendshapes->blend(x, blend, lmk );
		Eigen::MatrixXd res;
		add_to_pts(this->Q, Rt, blend, res);
		//cvrt_camera_uv_coord(res, m_image_h, res);
		
		Eigen::MatrixXd z = (target - res);

		Eigen::Map<Eigen::VectorXd> flat_z(z.data(), z.size());

		return flat_z.transpose().dot(flat_z);
		
	}

	void set_x_values(const Eigen::VectorXd& x) {
		full_x = x;
	}


	double gradient(int selected_index, const Eigen::VectorXd& xd) {
		double eps = 10e-6;
		weights(selected_index) = xd(0);
		Eigen::VectorXd x_eps_ph = weights;
		Eigen::VectorXd x_eps_mh = weights;
		x_eps_ph(selected_index) += eps;
		x_eps_mh(selected_index) -= eps;

		double x_h1 = cost_function(x_eps_ph);
		double x_h2 = cost_function(x_eps_mh);
		double res = (x_h1 - x_h2) / (2 * eps);
		return res;
	}

	double operator() (const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
		
		//full_x(selected_index) = x(0);
		//Eigen::VectorXd x_eps_ph = full_x;
		//Eigen::VectorXd x_eps_mh = full_x;
		//double eps = 10e-6;
		//x_eps_ph(selected_index) += eps;
		//x_eps_mh(selected_index) -= eps;
		
		
		//double x_h1 = cost_function(x_eps_ph);
		//double x_h2 = cost_function(x_eps_mh);
		//grad.setZero();
		//grad(selected_index) = (x_h1 - x_h2) / (2 * eps);
		//grad(0) = (x_h1 - x_h2) / (2 * eps);
		//std::cout << x << std::endl;
		//double val = cost_function(x);
		
		grad(0) = gradient(selected_index, x);
		//for (int i = 0; i < x.size(); i++)
			//grad(i) = gradient(i, x);
		//std::cout  << grad << std::endl;
		
		
		Eigen::VectorXd tt = weights;
		tt(selected_index) = x(0);
		
		double val = cost_function(tt);
		// 
		//std::cout << "en : " <<  val << "grad : " << (x_h1 - x_h2) / (2 * eps) << std::endl;
		return val;
	}
};
#include <opencv2/calib3d.hpp>
#include "cv_compatible_helper.h"
#include <opencv2/opencv.hpp>
//inline void find_camera(const Eigen::MatrixXd& pts2d, const Eigen::MatrixXd pts3d, const Eigen::Matrix3d& guessed_Q,  Eigen::Ref<Eigen::VectorXd> out, int iterative_n = 5) {
inline void find_camera(const Eigen::MatrixXd& pts2d, const Eigen::MatrixXd pts3d, const Eigen::Matrix3d& guessed_Q,  Eigen::Ref<Eigen::VectorXd> out, cv::Mat& img, int iterative_n = 5) {
	cv::Mat pts2d_cv;
	cv::Mat pts3d_cv;
	cv::Mat guessed_Q_cv;

	bool is_zero = out.isZero(0);

	Eigen::MatrixXd pts2d_copy = pts2d;
	//pts2d_copy.col(1).array() = img.rows - pts2d_copy.col(1).array() - 1;


	wow::eigen2cv(pts2d_copy, pts2d_cv);
	wow::eigen2cv(pts3d, pts3d_cv);
	wow::eigen2cv(guessed_Q, guessed_Q_cv);
	//std::cout << pts3d_cv << std::endl;
	//std::cout << pts2d_cv << std::endl;
	//pts2d_cv = pts2d_cv.reshape(1, { int(pts2d.rows()), int(pts2d.cols()) });
	//pts3d_cv = pts3d_cv.reshape(1, { int(pts3d.rows()), int(pts3d.cols()) });
	//std::cout << pts3d_cv << std::endl;
	//std::cout << pts2d_cv << std::endl;
	double d[] = { 0,0,0,0};	// k1,k2: radial distortion, p1,p2: tangential distortion
	cv::Mat distCoeffs(4, 1, CV_64FC1, d);
	cv::Mat rvec(3,1, CV_64FC1), tvec(3, 1, CV_64FC1);
	rvec.zeros(cv::Size(3, 1), CV_64FC1);
	tvec.zeros(cv::Size(3, 1), CV_64FC1);
	//std::cout << guessed_Q << std::endl;
	//std::cout << guessed_Q_cv << std::endl;
	cv::solvePnP(pts3d_cv, pts2d_cv, guessed_Q_cv, distCoeffs, rvec, tvec);
	//std::cout << rvec << std::endl;
	for (int i = 0; i < iterative_n - 1; i++) {
		cv::solvePnP(pts3d_cv, pts2d_cv, guessed_Q_cv, distCoeffs, rvec, tvec, true);
		
		
		
		//std::cout << rvec << std::endl;
		/*cv::Mat t;
		img.copyTo(t);
		for (int i = 0; i < pts2d.rows(); i++) {
			auto pts_i = cv::Point(pts2d(i, 0), pts2d(i, 1));
			cv::circle(t, pts_i , 1, cv::Scalar(0, 255, 0));
			cv::putText(t, std::to_string(i), pts_i, cv::FONT_HERSHEY_TRIPLEX, 0.1, cv::Scalar(0, 0, 255));

		}

		cv::imshow("test_img", t);
		
		Eigen::MatrixXd RR;

		cv::Mat R;
		cv::Rodrigues(rvec, R);
		wow::cv2eigen(R, RR);
		img.copyTo(t);
		Eigen::MatrixXd Rt(3,4);
		Rt.block(0, 0, 3, 3) = RR;
		Rt.block(0, 3, 3, 1) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
		Eigen::MatrixXd g;
		add_to_pts(guessed_Q, Rt, pts3d, g);

		for (int i = 0; i < g.rows(); i++) {
			auto pts_i = cv::Point( g(i, 0), t.rows - g(i, 1) - 1);
			cv::circle(t, pts_i, 1, cv::Scalar(0, 255, 0));
			cv::putText(t, std::to_string(i), pts_i, cv::FONT_HERSHEY_TRIPLEX, 0.1, cv::Scalar(0, 0, 255));

		}

		cv::imshow("test_img2", t);
		cv::waitKey(0);*/


	}


	cv::Mat R;
	cv::Rodrigues(rvec, R);
	Eigen::MatrixXd RR;
	wow::cv2eigen(R, RR);
	
	//out.conservativeResize(6);
	decompose_Rt(RR, out);
	//std::cout << out << std::endl;
	out.segment(3, 3) << tvec.at<double>(0) , tvec.at<double>(1), tvec.at<double>(2);
	//std::cout << out << std::endl;
	//Eigen::MatrixXd Rt,g;
	//compose_Rt(out, Rt);
	//add_to_pts(guessed_Q, Rt, pts3d, g);
	//Rt.block(0, 0, 3, 3) = RR;
	//Rt.block(0, 3, 3, 1) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
	//add_to_pts(guessed_Q, Rt, pts3d, g);

	//std::cout <<"seg" << out << std::endl;



}

#include <LBFGSB.h>  // Note the different header file

//inline void shapefit_2d_lmk_to_mesh(const struct Blendshapes& blendshapes, const Eigen::MatrixXd& lmk2d,const std::vector<int>& lmk_index68, const Eigen::Matrix3d& Q, Eigen::MatrixXd& out, int img_h,int iter_num = 10) {
inline void shapefit_2d_lmk_to_mesh(const struct Blendshapes& blendshapes, const Eigen::MatrixXd& lmk2d, const std::vector<int>& lmk_index68, const std::vector<int>& dest_lmk,const Eigen::Matrix3d& Q, Eigen::MatrixXd& out,  cv::Mat& img ,int iter_num = 10) {
	
	LBFGSpp::LBFGSBParam param;
	param.epsilon = 10e-6;
	param.max_iterations = 100;
	//Eigen::VectorXd lb = Eigen::VectorXd::Constant(blendshapes.get_blendshape_weight_size(), 0.0);
	//Eigen::VectorXd ub = Eigen::VectorXd::Constant(blendshapes.get_blendshape_weight_size(), 1.0);
	Eigen::VectorXd lb = Eigen::VectorXd::Constant(1, 0.0);
	Eigen::VectorXd ub = Eigen::VectorXd::Constant(1, 1.0);
	
	cv::Mat at;
	//this is test.
	//cv::cvtColor(img, at, cv::COLOR_GRAY2BGR);
	
	LBFGSpp::LBFGSBSolver<double> solver(param);
	Eigen::VectorXd t = ::Eigen::VectorXd::Zero(blendshapes.get_blendshape_weight_size() + 6);
	//t.head(blendshapes.get_blendshape_weight_size()).array() = 1.0 * 0.5;
	t.head(blendshapes.get_blendshape_weight_size()).array() = 1.0 * 0.4;

	double outf = 0;
	double prev_out = std::numeric_limits<double>::infinity(); 
	int camera_iteraton = 5;
	
	CalibFunctor s;
	s.set_Q(Q);

	Eigen::MatrixXd lmk2d_o; // this is for camera calib
	cvrt_camera_uv_coord(lmk2d, img.rows, lmk2d_o);
	//s.set_target(lmk2d);
	s.set_target(lmk2d_o);

	//s.set_image_h(img_h);
	s.set_image_h(img.rows);
	s.set_blendshapes(&blendshapes);
	s.set_landmark(lmk_index68);
	Eigen::MatrixXd b_shapes;
 	for (int i = 0; i < iter_num; i++) {
		
		blendshapes.blend(t.head(blendshapes.get_blendshape_weight_size()), b_shapes, lmk_index68);
		//std::cout << b_shapes.block(0, 0, 4, 3) << std::endl;
		//std::cout << "===" << std::endl;
		//std::cout << lmk2d_o << std::endl;
		//std::cout << "===" << std::endl;
		//std::cout << lmk2d << std::endl;
		//find_camera(lmk2d, b_shapes, Q, t.tail(6), at);
	/*	b_shapes <<
			-7.56983308e+00  , 3.23150106e+00, 2.63366166e+00,
			- 7.40222145e+00  , 9.37581619e-01, 2.89023912e+00,
			- 7.24908643e+00 , - 1.37313585e+00, 3.25062921e+00,
			- 6.67497119e+00 , - 3.74939994e+00, 4.15136622e+00,
			- 5.62362456e+00 , - 5.21320636e+00, 6.09887053e+00,
			- 4.48166420e+00 , - 5.87652173e+00, 7.66544955e+00,
			- 3.19844337e+00 , - 6.83325129e+00, 9.13143087e+00,
			- 1.66071251e+00 , - 7.46012627e+00, 1.01952157e+01,
			1.60269666e-02 , - 8.07705427e+00, 1.04844037e+01,
			1.71545036e+00 , - 7.44326379e+00, 1.01669255e+01,
			3.22358329e+00 , - 6.79813148e+00, 9.07150540e+00,
			4.44287657e+00 , - 5.90840668e+00, 7.61210973e+00,
			5.49512415e+00 , - 5.25405196e+00, 6.03739259e+00,
			6.45621441e+00 , - 3.78737471e+00, 4.09864849e+00,
			6.98074089e+00 , - 1.48511426e+00, 3.19781685e+00,
			7.17182999e+00  , 8.30703348e-01, 2.93619934e+00,
			7.43895204e+00  , 3.13868295e+00, 2.78545258e+00,
			- 5.57336198e+00  , 5.23481807e+00, 8.02650092e+00,
			- 4.65166353e+00  , 5.94790326e+00, 9.03713480e+00,
			- 3.54892975e+00  , 6.17455438e+00, 9.88741490e+00,
			- 2.25735933e+00  , 6.07738092e+00, 1.05154370e+01,
			- 1.10575689e+00  , 5.75419914e+00, 1.08110452e+01,
			1.05771172e+00  , 5.78201666e+00, 1.07923192e+01,
			2.21326715e+00  , 6.12464592e+00, 1.05039762e+01,
			3.49412259e+00  , 6.22459317e+00, 9.84167286e+00,
			4.60644905e+00  , 6.04262229e+00, 9.02233336e+00,
			5.51850625e+00  , 5.28609079e+00, 7.97161279e+00,
			- 2.19338924e-02  , 3.80380700e+00, 1.08305349e+01,
			- 2.23236806e-03  , 2.79055881e+00, 1.15797169e+01,
			1.23463763e-02  , 1.82354428e+00, 1.23391627e+01,
			3.58078958e-02  , 6.12378556e-01, 1.30220014e+01,
			- 1.36266429e+00 , - 4.92550488e-01, 1.11037931e+01,
			- 6.52826940e-01 , - 6.26322908e-01, 1.14011634e+01,
			6.01339653e-02 , - 7.50804484e-01, 1.15176155e+01,
			7.64660836e-01 , - 6.03434073e-01, 1.13780035e+01,
			1.44383951e+00 , - 4.23510783e-01, 1.10628792e+01,
			- 4.43145547e+00  , 3.47100367e+00, 8.72633355e+00,
			- 3.84752952e+00  , 3.55265286e+00, 9.29284946e+00,
			- 2.66692818e+00  , 3.57243179e+00, 9.35903633e+00,
			- 1.75608252e+00  , 3.24662752e+00, 9.19216950e+00,
			- 2.50660503e+00  , 3.38756861e+00, 9.46912953e+00,
			- 3.74140029e+00  , 3.37989881e+00, 9.45023351e+00,
			1.79043264e+00  , 3.28168008e+00, 9.21758055e+00,
			2.71697051e+00  , 3.59418285e+00, 9.38347992e+00,
			3.85435917e+00  , 3.54056693e+00, 9.24905359e+00,
			4.41958453e+00  , 3.49081815e+00, 8.72040340e+00,
			3.78744748e+00  , 3.40757537e+00, 9.40229961e+00,
			2.51229141e+00  , 3.38280245e+00, 9.47412228e+00,
			- 2.37092198e+00 , - 2.45247615e+00, 1.09590669e+01,
			- 1.70277264e+00 , - 1.83763726e+00, 1.14965684e+01,
			- 7.32020116e-01 , - 1.42323553e+00, 1.17246712e+01,
			5.68136871e-02 , - 1.40730495e+00, 1.17352495e+01,
			8.59000932e-01 , - 1.39473366e+00, 1.16897669e+01,
			1.82012947e+00 , - 1.78377709e+00, 1.14180348e+01,
			2.41162593e+00 , - 2.35695399e+00, 1.08399202e+01,
			1.85610421e+00 , - 3.09320420e+00, 1.17167668e+01,
			1.09229019e+00 , - 3.64581815e+00, 1.18520800e+01,
			7.44410381e-02 , - 3.92586340e+00, 1.19146771e+01,
			- 9.69698011e-01 , - 3.67236808e+00, 1.19241866e+01,
			- 1.76641672e+00 , - 3.14499773e+00, 1.18338577e+01,
			- 2.07948241e+00 , - 2.47114623e+00, 1.07841150e+01,
			- 1.00319602e+00 , - 2.34525551e+00, 1.12512165e+01,
			5.90794548e-02 , - 2.21254786e+00, 1.12889063e+01,
			1.10590147e+00 , - 2.31033915e+00, 1.11784014e+01,
			2.12579379e+00 , - 2.40471321e+00, 1.06844947e+01,
			1.19822905e+00 , - 2.83193480e+00, 1.19478919e+01,
			5.42260133e-02 , - 3.07504549e+00, 1.19595851e+01,
			-1.09139066e+00 , - 2.86873248e+00, 1.20046700e+01;*/


		find_camera(lmk2d_o, b_shapes, Q, t.tail(6), at);
		Eigen::MatrixXd rt;
		compose_Rt(t.tail(6), rt);
		s.set_Rt(rt);
	
		
		
		//{
		
		//Eigen::MatrixXd out;
		//add_to_pts(Q, rt, b_shapes, out);
		////std::cout << out << std::endl;
		//cvrt_camera_uv_coord(out, img.rows, out);


		//	

		//	
		//	

		//	cv::Mat t;
		//	img.copyTo(t);
		//	cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		//	for (int i = 0; i < out.rows(); i++)
		//		cv::circle(t, cv::Point(out(i, 0), out(i, 1)), 2, cv::Scalar(0, 255, 0));

		//	cv::imshow("fu t", t);
		//	cv::waitKey(0);
		//}

		//Eigen::VectorXd xx = (t.head(blendshapes.get_blendshape_weight_size()));
		Eigen::VectorXd xx = Eigen::VectorXd(1);
		//solver.minimize(s, xx, outf, lb, ub);
		//t.head(blendshapes.get_blendshape_weight_size()) = xx;
		//xx = t.head(blendshapes.get_blendshape_weight_size());
		
		for (int j = 0; j < t.size() - 6; j++) {


			s.set_x_values(t.head(blendshapes.get_blendshape_weight_size()));
			xx(0) = t(j);
			s.select_optimize_variable_index(j);
			s.set_weights(t.head(blendshapes.get_blendshape_weight_size()));
			try {
				solver.minimize(s, xx, outf, lb, ub);
				t(j) = xx(0);
			}
			catch (std::exception exception) {
				//;
				//goto bb;
				//continue;
			}




		//{ // visualize
		//		Eigen::MatrixXd tmpo, tmp2d;
		//		blendshapes.blend(t, tmpo, lmk_index68);
		//		add_to_pts(Q, rt, tmpo, tmp2d);

		//		//std::cout << "==========" << outf << std::endl;

		//		//std::cout << j << "oiut : " << outf << std::endl;
		//		//std::cout << "------proj bshapes---------" << std::endl;
		//		//std::cout << tmp2d.block(0, 0, 4, 2) << std::endl;
		//		//std::cout << "-------lmk_2d--------" << std::endl;
		//	//std::cout << lmk2d_o.block(0,0,4,2) << std::endl;
		//		//std::cout << "------lmk2d_o---------"  << std::endl;
		//	//std::cout << lmk2d.block(0, 0, 4, 2) << std::endl;
		//		//std::cout << "==========" << outf << std::endl;


		//		cvrt_camera_uv_coord(tmp2d, img.rows, tmp2d);
		//			

		//		cv::Mat t;
		//		img.copyTo(t);
		//		cv::cvtColor(t, t, cv::COLOR_GRAY2BGR);
		//		// green is pred
		//		// blue is ground truth
		//		for (int i = 0; i < tmp2d.rows(); i++)
		//			cv::circle(t, cv::Point(tmp2d(i, 0), tmp2d(i, 1)), 2, cv::Scalar(0, 255, 0));
		//		for (int i = 0; i < tmp2d.rows(); i++)
		//			cv::circle(t, cv::Point(lmk2d(i, 0), lmk2d(i, 1)), 2, cv::Scalar(255, 0, 0));

		//		cv::imshow("fu t", t);
		//		cv::waitKey(0);
		//}

		}

		if (abs(prev_out - outf) < 10e-6) {
			goto bb;
		}
		else {
			prev_out = outf;
		}

	}

bb:
	blendshapes.blend(t.head(blendshapes.get_blendshape_weight_size()), b_shapes, dest_lmk);
	Eigen::MatrixXd Rt;
	compose_Rt(t.tail(6), Rt);
	add_to_pts(Rt, b_shapes, out);
	return;
}


#include <algorithm>
inline void find_most_similar_to_target_pose(const std::vector<Eigen::MatrixXd>& pose_list, const Eigen::MatrixXd& target, std::vector<int>& index_list, int candidate_num = 15) {

	auto f = [&pose_list, &target](int i1, int i2)->bool {
		return (pose_list[i1] - target).array().pow(2.0).sum() < (pose_list[i2] - target).array().pow(2.0).sum();
		};

	std::vector<int> sel;
	sel.resize(pose_list.size());
	for (int i = 0; i < pose_list.size(); i++)
		sel[i] = i;

	std::sort(sel.begin(), sel.end(), f);
	index_list.assign(sel.begin(), sel.begin()+candidate_num);



}