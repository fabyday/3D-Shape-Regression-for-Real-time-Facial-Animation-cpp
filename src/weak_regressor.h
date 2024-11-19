#include <Eigen/Core>
#include "Fern.h"
#include <vector>
#include <opencv2/core.hpp>
class WeakRegressor {


public:
	WeakRegressor(Eigen::Matrix3d Q, int K, int F, int beta, int P, const std::string& name) :m_Q(Q), m_K(K), m_F(F), m_beta(beta), m_P(P), m_name(name) {};
	void load(const std::string& pth);
	void calc_app_vector(const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& img, const Eigen::MatrixXd& meanshapes, const Eigen::MatrixXd& Rt, const Eigen::MatrixXd& cur_shape, Eigen::VectorXd& out_app_vec);
	void predict(const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& img, const Eigen::MatrixXd& meanshapes, const Eigen::MatrixXd& Rt, Eigen::MatrixXd& cur_pose);
private:
	std::vector<Fern> m_ferns;




	Eigen::MatrixXd m_V;
	Eigen::MatrixXd m_disp;
	Eigen::MatrixXi m_nearest_index;
	Eigen::Matrix3d m_Q;


	Eigen::MatrixXd m_mean_shape;
	int m_K;
	int m_F;
	int m_beta;
	int m_P;
	std::string m_name;


};