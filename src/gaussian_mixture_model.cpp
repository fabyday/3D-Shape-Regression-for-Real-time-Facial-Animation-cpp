#include "gaussian_mixture_model.h"
#include "cnpy.h"
#include <iostream>
#include <fstream>
#include "string_function.h"


void psuedo_determinant_and_inv(const Eigen::MatrixXd& i_cov, Eigen::MatrixXd& o_inv_matrix, double& o_log_determinant) {

	//auto new_cov = i_cov + 0.01 * Eigen::MatrixXd::Identity(i_cov.rows(), i_cov.cols());
	//auto new_cov = i_cov + 0.01 * Eigen::MatrixXd::Identity(i_cov.rows(), i_cov.cols());
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(i_cov);
	auto eig_v = eig.eigenvalues();
	auto eigen_vec_real = eig.eigenvectors();

	auto diag_mat = eig_v.asDiagonal();

	//std::cout << (eigen_vec_real.transpose() * diag_mat * eigen_vec_real).isApprox(i_cov) << std::endl;
	//std::cout << (eigen_vec_real.transpose() * diag_mat * eigen_vec_real).block(0,0,10,10) << std::endl;
	//std::cout << std::endl<< i_cov.block(0, 0, 10, 10) << std::endl;



	

	Eigen::VectorXd sqrt_reciprocal_sig = 1 / eig_v.array();
	/*for (int i = 0 < i < eig_v.size(); i++) {
		if(eig_v[i] < 1e-6)
	}*/
	auto diagonal_sig = sqrt_reciprocal_sig.asDiagonal();
	
	//o_inv_matrix = V * diagonal_sqrt * diagonal_sqrt * U.transpose();

	o_inv_matrix = eigen_vec_real * diagonal_sig * eigen_vec_real.transpose();


	//std::cout << (o_inv_matrix * i_cov).block(0,0,5,5) << std::endl;
	//std::cout << (i_cov*o_inv_matrix).block(0,0,5,5) << std::endl;
	//std::cout << (o_inv_matrix * i_cov).isApprox(Eigen::MatrixXd::Identity(i_cov.rows(), i_cov.cols())) << std::endl;
	/*
	auto X = V * diagonal_sqrt * diagonal_sqrt * U.transpose();
	auto I = Eigen::MatrixXd::Identity(new_cov.rows(), new_cov.cols());
	std::cout << (X* new_cov).isApprox(I) << std::endl;
	std::cout << (new_cov*X).isApprox(I) << std::endl;
	*/
	
	// log sum
	o_log_determinant = eig_v.array().log().sum();
	
	//det = std::accumulate(sig.begin(), sig.end(), 0.0, [](double x, double y) {
	//	return x + std::log(y); // Accumulate the logarithms of the elements
	//	});

	//std::cout << s << std::endl;
	//std::cout << det << std::endl;
	//std::cout << "assddet" << std::endl;

}

void BLendshapes_GMM::load(const std::string& pth)
{
	// D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\precompute_blendshapes_gmm

	std::ifstream in(pth + "/meta.txt");
	std::string buf;

	while (std::getline(in, buf)) {
		int pos = buf.find(":", 0);
		std::string key = buf.substr(0, pos);
		std::string value = buf.substr(pos + 1, buf.size());
		trim(key);
		trim(value);
		if (key == "window_size") {
			m_window_size = std::stoi(value);
		}

	}


	std::string w_path = pth + "/weights.npy";
	std::string cov_path = pth + "/covariances.npy";
	std::string means_path = pth + "/means.npy";
	std::string log_pi_path = pth + "/log_pi.npy";
	std::string linear_trans_path = pth + "/linear_transform.npy";
	std::string sigma_squared = pth + "/sigma_squared.npy";
	std::string pca_mean = pth + "/pca_mean.npy";
	std::string pca_component= pth + "/pca_component.npy";
	cnpy::NpyArray weights_npy = cnpy::npy_load(w_path);
	cnpy::NpyArray covs_npy = cnpy::npy_load(cov_path);
	cnpy::NpyArray  means_npy = cnpy::npy_load(means_path);
	cnpy::NpyArray  log_pi_npy = cnpy::npy_load(log_pi_path);
	cnpy::NpyArray  linear_trans_npy = cnpy::npy_load(linear_trans_path);
	cnpy::NpyArray  sigma_squared_npy = cnpy::npy_load(sigma_squared);
	cnpy::NpyArray  pca_mean_npy = cnpy::npy_load(pca_mean);
	cnpy::NpyArray  pca_component_npy = cnpy::npy_load(pca_component);


	using MatrixR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


	// N := GMM model component sizes
	m_weights = Eigen::Map<Eigen::VectorXd>(weights_npy.data<double>(), weights_npy.shape[0], 1); // N 
	m_log_pi = Eigen::Map<Eigen::VectorXd>(log_pi_npy.data<double>(), log_pi_npy.shape[0]); // N 
	m_sigma_squared = Eigen::Map<Eigen::VectorXd>(sigma_squared_npy.data<double>(), sigma_squared_npy.shape[0]); // N 


	int cov_size = covs_npy.shape[0];
	int cov_row = covs_npy.shape[1];
	int cov_col = covs_npy.shape[2];

	m_blendshapes_coeff_size = int(cov_row / m_window_size);
	m_gaussian_dist_num = cov_size;
	m_covariances.resize(cov_size, Eigen::MatrixXd(cov_row, cov_col));
	m_inv_covariances.resize(cov_size, Eigen::MatrixXd(cov_row, cov_col));
	m_det_covariances.resize(cov_size);
	for (int i = 0; i < cov_size; i++) {
		int offset = cov_row * cov_col * i;
		Eigen::MatrixXd cov = Eigen::Map<MatrixR>(covs_npy.data<double>() + offset, covs_npy.shape[1], covs_npy.shape[2]);// N X (blendshapes_coeff_sizes*window_size) X (blendshapes_coeff_sizes*window_size)
		m_covariances[i] = cov;
		//psuedo_determinant_and_inv(cov, m_inv_covariances[i], m_det_covariances[i]);
		//m_det_covariances[i] = cov.determinant();
		//m_inv_covariances[i] = cov.inverse();

	}


	int num_mean = means_npy.shape[0];
	int mean_size = means_npy.shape[1];
	m_means.resize(num_mean, Eigen::VectorXd());
	for (int i = 0; i < num_mean; i++) {
		int offset = mean_size * i ;
		m_means[i] = Eigen::Map<Eigen::VectorXd>(means_npy.data<double>() + offset, mean_size); // N X (blendshapes_coeff_size * window_size)
	}


	m_pca_component = Eigen::Map<MatrixR>(pca_component_npy.data<double>(), pca_component_npy.shape[0], pca_component_npy.shape[1]);// N X (blendshapes_coeff_sizes*window_size) X (blendshapes_coeff_sizes*window_size)
	std::cout << m_pca_component.block(0, 0, 10, 10) << std::endl;
	m_pca_mean = Eigen::Map<Eigen::VectorXd>(pca_mean_npy.data<double>(), pca_mean_npy.shape[0]);// N X (blendshapes_coeff_sizes*window_size) X (blendshapes_coeff_sizes*window_size)
	std::cout << m_pca_mean.transpose() << std::endl;



	int num_lin_trans = linear_trans_npy.shape[0];
	int lin_trans_row = linear_trans_npy.shape[1];
	int lin_trans_col = linear_trans_npy.shape[2];
	m_linear_transform.resize(num_lin_trans);
	for (int i = 0; i < num_lin_trans; i++) {
		int offset = lin_trans_row * lin_trans_col * i;
		m_linear_transform[i] = Eigen::Map<MatrixR>(linear_trans_npy.data<double>() + offset, lin_trans_row, lin_trans_col); // N X (blendshapes_coeff_size * window_size)
	}

	this->preprop();
	std::cout << "test" << std::endl;

	


}

void BLendshapes_GMM::preprop() {
	int model_sizes = this->m_covariances.size();

	Eigen::VectorXd latent_mean = Eigen::VectorXd::Zero(m_linear_transform[0].cols());
	Eigen::MatrixXd latent_cov = Eigen::MatrixXd::Identity(m_linear_transform[0].cols(), m_linear_transform[0].cols());
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(m_means[0].size(), m_means[0].size());
		
	m_synthetic_mean.resize(model_sizes);
	m_synthetic_cov.resize(model_sizes);
	m_synthetic_log_det_cov.conservativeResize(model_sizes);
	m_synthetic_inv_cov.resize(model_sizes);
	m_synthetic_mean_for_target_win.resize(model_sizes);
	m_synthetic_cov_for_target_win.resize(model_sizes);
	m_synthetic_det_cov_for_target_win.conservativeResize(model_sizes);
	m_synthetic_inv_cov_for_target_win.resize(model_sizes);


	double min_limit = std::numeric_limits<double>::epsilon();
	double cond = 1e6;
	double eps = cond * min_limit;


	for (int i = 0; i < model_sizes; i++) {
		//m_synthetic_mean[i]  = (m_linear_transform[i] * latent_mean + m_means[i])* m_pca_component + m_pca_mean;
		m_synthetic_mean[i] = (m_linear_transform[i] * latent_mean + m_means[i]);
		m_synthetic_mean[i] = m_means[i].transpose() * m_pca_component + m_pca_mean;
		//m_synthetic_cov[i] = (m_sigma_squared[i] * identity + m_linear_transform[i] * latent_cov * m_linear_transform[i].transpose());
		//m_synthetic_cov[i] = m_sigma_squared[i] * identity + m_linear_transform[i].transpose() * latent_cov * m_linear_transform[i];
		m_synthetic_cov[i] = m_pca_component.transpose() * (m_sigma_squared[i] * identity + m_linear_transform[i] * latent_cov * m_linear_transform[i].transpose())*m_pca_component;
		psuedo_determinant_and_inv(m_synthetic_cov[i], m_synthetic_inv_cov[i], m_synthetic_log_det_cov[i]);
		std::cout << m_synthetic_cov[i].determinant() << std::endl;
	}


	std::cout << m_sigma_squared << std::endl;
	for (int i = 0; i < m_det_covariances.size(); i++) {
		std::cout << m_synthetic_cov[i].block(0, 0, 6,6) << std::endl << std::endl;
	}
	std::cout << std::endl;
	
	for (int i = 0; i < m_det_covariances.size(); i++) {
		std::cout << m_synthetic_log_det_cov[i] << std::endl;

	}
	
	std::cout << std::endl;



	/// slicing
	int crop_size = m_blendshapes_coeff_size * (m_window_size - 1);
	for (int i = 0; i < model_sizes; i++) {
		m_synthetic_cov_for_target_win[i] = m_synthetic_cov[i].topLeftCorner(crop_size, crop_size);
		m_synthetic_inv_cov_for_target_win[i] = m_synthetic_cov_for_target_win[i].inverse();
		m_synthetic_det_cov_for_target_win[i] = m_synthetic_cov_for_target_win[i].determinant();
		m_synthetic_mean_for_target_win[i] = m_synthetic_mean[i].head(crop_size);
	}


}

//
//inline double sum_logs(Eigen::VectorXd& logs) {
//
//	double max_coeff = logs.maxCoeff();
//	auto max_centered = logs.array() - max_coeff;
//	max_centered.
//
//}


#define PI 3.141592653589793

inline double  logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, double log_determ, Eigen::MatrixXd& inv_cov) {
	auto dev = x - mean;
	auto maha = (dev.transpose() * inv_cov).dot(dev);
	//auto maha = dev.transpose().dot(dev);
	//std::cout << maha << std::endl;
	return -0.5 * ( x.size()*log(2 * PI) + maha + log_determ );
}



inline double  gausian_pdf(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mean, double log_determ, Eigen::MatrixXd& inv_cov) {


	return exp(logpdf(x, mean, log_determ, inv_cov));
}




/*	
* w
*/
double BLendshapes_GMM::penalty_term(const Eigen::VectorXd& w)
{
	//w := win_size*exprs
	double denom = 0.0f;
	double nom = 0.0f;
	int model_size = m_covariances.size();
	Eigen::VectorXd logsums(model_size);
	Eigen::VectorXd logpdf_val(model_size);
	for (int i = 0; i < model_size; i++) {
		//logpdf_val[i] =logpdf(w, m_synthetic_mean[i], m_synthetic_log_det_cov[i], m_synthetic_inv_cov[i]);
		logpdf_val[i] = logpdf(w, m_synthetic_mean[i], m_synthetic_log_det_cov[i], m_synthetic_inv_cov[i]);
		logsums[i] = m_log_pi[i] + logpdf_val[i];
	}
	//std::cout << "logpi" << m_log_pi.transpose()<< std::endl;
	//std::cout << "pdf" << logpdf_val.transpose() << std::endl;
	//std::cout << "sum" << logsums.transpose() << std::endl;
	double max_c = logsums.maxCoeff();
	double sumlogs = log((logsums.array() - max_c).exp().sum()) + max_c;
	//std::cout << sumlogs << std::endl;
	Eigen::VectorXd RR(logpdf_val.size());
	for (int i = 0; i < logpdf_val.size(); i++) {
		//RR[i] = logpdf_val[i] + m_log_pi[i] - sumlogs;
		RR[i] = logpdf_val[i] + m_log_pi[i];
		//std::cout << RR[i] << std::endl;
	}
	
	double res = RR.array().exp().sum();
	//std::cout << res << std::endl;
	return res;


	return nom / denom;
}

/*
* x : window_size X expression_size
*/
void BLendshapes_GMM::penalty_gradient(const Eigen::VectorXd& x, Eigen::VectorXd& out)
{

	double denom = 0.0f;
	Eigen::MatrixXd nom;
	nom.resizeLike(x);
	nom.setZero();
	for (int i = 0; i < m_gaussian_dist_num; i++) {
		
		double pi = exp(m_log_pi[i]);
		double g = logpdf(x, m_synthetic_mean[i], m_synthetic_log_det_cov[i], m_synthetic_inv_cov[i]);
		double pdf_val = pi * g;
		/*std::cout << "G : " << g << std::endl;
		std::cout << "log p" << m_log_pi[i] << std::endl;
		std::cout << "p : " << pi << std::endl;
		std::cout <<"pi*g" << pdf_val << std::endl;*/
		denom += pdf_val;
		
		auto x_centered_x = x - m_means[i];
		auto blocked_cov_inv = (m_inv_covariances[i].topLeftCorner(m_inv_covariances[i].rows(), x.size()));
		auto coeff = (x_centered_x.transpose() * blocked_cov_inv);
		nom = coeff * pdf_val;
		//std::cout << "vec + pdf*pi" << nom << std::endl;
	}

	out.conservativeResizeLike(nom);
	out = nom.array() / denom;
	//std::cout << nom << "/ " << denom << std::endl;
	//std::cout << out << std::endl;

}
