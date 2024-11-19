#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace wow {


	void cv2eigen(const cv::Mat& cv_mat, Eigen::MatrixXd& mat) {
		cv::cv2eigen(cv_mat, mat);
	}
	void cv2eigen(const cv::Mat& cv_mat, Eigen::Matrix<uchar, Eigen::Dynamic,Eigen::Dynamic>& mat) {
		cv::cv2eigen(cv_mat, mat);
	}



	void eigen2cv(Eigen::MatrixXd& mat, cv::Mat& cv_mat) {
		cv::eigen2cv(mat, cv_mat);
	}
	//void cv2eigen(cv::Mat& cv_mat, Eigen::MatrixXd& mat) {
	//	cv::cv2eigen(cv_mat, mat);
	//}

	void eigen2cv(const Eigen::MatrixXd& mat, cv::Mat& cv_mat) {
		cv::eigen2cv(mat, cv_mat);
	}
	void eigen2cv(const Eigen::Matrix3d& mat, cv::Mat& cv_mat) {
		cv::eigen2cv(mat, cv_mat);
	}

	void eigen2cv(const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& mat, cv::Mat& cv_mat) {
		cv::eigen2cv(mat, cv_mat);
	}



	void cv2eigen(cv::Mat& cv_mat, Eigen::Vector3d& mat) {
		cv::cv2eigen(cv_mat, mat);
	}

	void eigen2cv(Eigen::Vector3d& mat, cv::Mat& cv_mat) {
		cv::eigen2cv(mat, cv_mat);
	}
}