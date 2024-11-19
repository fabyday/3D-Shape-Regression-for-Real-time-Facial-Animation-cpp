#include "weak_regressor.h"
#include <string>
#include "cnpy.h"
#include "math.h"
#include <filesystem>
void WeakRegressor::load(const std::string& root_pth)
{
	std::string directory_name = "weak_regressor_" + m_name;
	std::filesystem::path save_path(root_pth);
	save_path /= directory_name;

	std::filesystem::path V_path = save_path / "V.npy";
	std::filesystem::path disp_path = save_path / "disp.npy";
	std::filesystem::path nearest_path = save_path / "nearest_index.npy";
	//cnpy::NpyArray v_data = cnpy::npy_load(V_path.string());
	//v_data.shape[0]
	
	
	cnpy::NpyArray disp_data = cnpy::npy_load(disp_path.string());
	int i = disp_data.shape[0];
	int j = disp_data.shape[1];
	m_disp = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>>(disp_data.data<double>(), i, j);
		
	cnpy::NpyArray nearest_data = cnpy::npy_load(nearest_path.string());
	//std::cout << nearest_data.shape[0] << "," << nearest_data.shape[1] << std::endl;
	auto mmm  = Eigen::Map<Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(nearest_data.data< long long>(), nearest_data.shape[0], 1);
	//std::cout << mmm << std::endl;
	m_nearest_index = mmm.cast<int>();
	//std::cout << m_nearest_index << std::endl;




	
	for (int i = 0; i < m_K; i++) {
		m_ferns.emplace_back(m_F, m_beta, m_P, std::to_string(i));
		m_ferns[i].load_model(save_path.string());

	}




	
	

}

void WeakRegressor::calc_app_vector(const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& img, const Eigen::MatrixXd& meanshapes, const Eigen::MatrixXd& Rt, const Eigen::MatrixXd& cur_shape, Eigen::VectorXd& out_app_vec)
{
		
		Eigen::Matrix3d R;
		Eigen::MatrixXd T;
		similarity_transform(meanshapes, cur_shape, R, T);
		//std::cout << R << std::endl;

		Eigen::MatrixXd transformed_disp;
		add_to_pts(R, m_disp, transformed_disp);
		//std::cout << m_disp.block(0,0,4,3) << std::endl <<std::endl;
		//std::cout << transformed_disp.block(0,0,4,3) << std::endl <<std::endl;

		
		Eigen::MatrixXd position(m_nearest_index.size(), 3);
		for (int i = 0; i < m_nearest_index.rows(); i++) {
			 position.row(i) = cur_shape.row(m_nearest_index(i, 0)) + transformed_disp.row(i);
		}
		//std::cout << position.block(0,0,4,3) << std::endl <<std::endl;

		Eigen::MatrixXd loc;
		add_to_pts(m_Q, Rt, position, loc);
	

		//std::cout << loc<< std::endl << std::endl;

		cvrt_camera_uv_coord(loc, img.rows(), loc);
		Eigen::MatrixXi loc_i = loc.cast<int>();
		loc_i.col(0) = loc_i.col(0).cwiseMin(img.cols() - 1).cwiseMax(0);
		loc_i.col(1) = loc_i.col(1).cwiseMin(img.rows() - 1).cwiseMax(0);

		//loc.col(0) = loc.col(0).cwiseMin(img.cols() - 1).cwiseMax(0);
		//loc.col(1) = loc.col(1).cwiseMin(img.rows() - 1).cwiseMax(0);
		//cvrt_camera_uv_coord(loc, img.rows(), loc);
		//Eigen::MatrixXi loc_i = loc.cast<int>();


		//std::cout << position.block(0,0,4,2) << std::endl <<std::endl;
		/*Eigen::MatrixXd q(4, 3);q << 100, 200, 300,
			99, 200, 301,
			100, 200, 300,
			99, 200, 301;
		q.cwiseMax(100).cwiseMin(300);
		std::cout << q << std::endl;
		q = q.cwiseMax(100).cwiseMin(300);
		std::cout << q << std::endl;*/

		//Eigen::MatrixXd tmps;
		//add_to_pts(m_Q, Rt, cur_shape, tmps);
		//cvrt_camera_uv_coord(tmps, img.rows(), tmps);

		//Eigen::MatrixXi tmpi;
		//tmpi = tmps.cast<int>();
		//cv::Mat f;
		//
		//
		//wow::eigen2cv(img, f);
		//std::cout << f.rows << ", " << f.cols << std::endl;
		//cv::Mat t;
		//cv::cvtColor(f, t, cv::COLOR_GRAY2BGR);
		//for (int i = 0; i < tmps.rows(); i++)
		//	cv::circle(t, cv::Point(tmps(i, 0) , tmps(i, 1)), 2, cv::Scalar(255, 0, 0));

		//cv::imshow("fu t", t);
		//cv::waitKey(0);


		

		out_app_vec.conservativeResize(m_nearest_index.size());
		for (int i = 0; i < loc_i.rows(); i++) {
			int x = loc_i(i, 0);
			int y = loc_i(i,1);
			out_app_vec(i) = img(y,x);
			//out_app_vec(i) = img(y,x);
		}

		



		



}


void WeakRegressor::predict(const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& img, const Eigen::MatrixXd& meanshapes, const Eigen::MatrixXd& Rt,  Eigen::MatrixXd& cur_pose)
{
		Eigen::VectorXd app_vec;
		calc_app_vector(img, meanshapes, Rt, cur_pose, app_vec);



		for (int i = 0; i < m_ferns.size(); i++) {
			m_ferns[i].predict(app_vec, cur_pose);


		/*Eigen::MatrixXd tmps;
		add_to_pts(m_Q, Rt, cur_pose, tmps);
		cvrt_camera_uv_coord(tmps, img.rows(), tmps);

		cv::Mat f;
		
		
		wow::eigen2cv(img, f);
		cv::Mat t;
		cv::cvtColor(f, t, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < tmps.rows(); i++)
			cv::circle(t, cv::Point(tmps(i, 0) , tmps(i, 1)), 2, cv::Scalar(255, 0, 0));

		cv::imshow("fu t", t);
		cv::waitKey(10);*/
		}
}
