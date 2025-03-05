#pragma once 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include <string>


#define PI 3.1415926535897931
//
// this class only predict prb, not learning model.
class BLendshapes_GMM {


private : 
	int m_window_size = 0;
	int m_gaussian_dist_num; // same m_weights size
	std::vector<Eigen::MatrixXd> m_inv_covariances;
	std::vector<Eigen::VectorXd> m_means;
	std::vector<Eigen::MatrixXd> m_covariances;
	Eigen::VectorXd m_log_pi;
	Eigen::VectorXd m_sigma_squared;
	std::vector<double> m_det_covariances;
	std::vector<Eigen::MatrixXd> m_linear_transform;
	Eigen::MatrixXd m_pca_component;
	Eigen::VectorXd m_pca_mean;

	Eigen::VectorXd m_weights;
	
	int m_blendshapes_coeff_size;


	// generated origianl space from latent space N(0, I)
	std::vector<Eigen::VectorXd> m_synthetic_mean;
	std::vector<Eigen::MatrixXd> m_synthetic_cov;
	std::vector<Eigen::MatrixXd> m_synthetic_inv_cov;
	Eigen::VectorXd m_synthetic_log_det_cov;
	// sliced upper data to fit window prediction/
	std::vector<Eigen::VectorXd> m_synthetic_mean_for_target_win;
	std::vector<Eigen::MatrixXd> m_synthetic_cov_for_target_win;
	std::vector<Eigen::MatrixXd> m_synthetic_inv_cov_for_target_win;
	Eigen::VectorXd m_synthetic_det_cov_for_target_win;
	

public :
	
	void load(const std::string& pth);
	void preprop();

	inline int get_winsize() { return m_window_size; };
	inline int coeff_size() { return m_blendshapes_coeff_size; };
	double penalty_term(const Eigen::VectorXd& x);
	void penalty_gradient(const Eigen::VectorXd& x, Eigen::VectorXd& out);
};